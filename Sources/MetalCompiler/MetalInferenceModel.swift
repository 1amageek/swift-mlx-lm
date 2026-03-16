import Metal
import LMIR

public struct MetalInferenceModel: @unchecked Sendable {

    public let plan: MetalDispatchPlan
    public var prefillPlan: MetalPrefillPlan?
    public let commandQueue: MTLCommandQueue
    public var position: Int = 0

    private var pendingCommandBuffer: MTLCommandBuffer?
    private var hasPendingResult: Bool = false

    public init(plan: MetalDispatchPlan, device: MTLDevice) throws {
        self.plan = plan
        self.prefillPlan = nil
        guard let queue = device.makeCommandQueue() else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create command queue")
        }
        self.commandQueue = queue
    }

    // MARK: - Decode

    private func encodeSteps(_ enc: MTLComputeCommandEncoder) {
        for step in plan.steps {
            if step.sync == .bufferBarrier { enc.memoryBarrier(scope: .buffers) }
            enc.setComputePipelineState(step.pipeline)
            for (index, buffer, offset) in step.bufferBindings {
                enc.setBuffer(buffer, offset: offset, index: index)
            }
            for (index, value) in step.bytesBindings {
                value.withUnsafeBufferPointer { enc.setBytes($0.baseAddress!, length: $0.count, index: index) }
            }
            if step.threadgroupMemoryLength > 0 {
                enc.setThreadgroupMemoryLength(step.threadgroupMemoryLength, index: 0)
            }
            enc.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
        }
    }

    public mutating func decode(tokenID: Int32) -> Int32 {
        let b = plan.buffers
        var result: Int32 = -1
        if hasPendingResult {
            pendingCommandBuffer?.waitUntilCompleted()
            result = b.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
        }
        b.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(position)
        b.tokenIn.contents().bindMemory(to: Int32.self, capacity: 1).pointee = tokenID
        guard let cb = commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return result }
        encodeSteps(enc)
        enc.endEncoding()
        cb.commit()
        pendingCommandBuffer = cb
        hasPendingResult = true
        position += 1
        return result
    }

    public mutating func decodeSync(tokenID: Int32) -> Int32 {
        let b = plan.buffers
        b.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(position)
        b.tokenIn.contents().bindMemory(to: Int32.self, capacity: 1).pointee = tokenID
        guard let cb = commandQueue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return -1 }
        encodeSteps(enc)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { print("[MetalInference] GPU error: \(error)") }
        position += 1
        return b.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
    }

    // MARK: - Prefill

    public mutating func prefill(tokens: [Int32]) {
        guard let prefill = prefillPlan else {
            for token in tokens { let _ = decodeSync(tokenID: token) }
            return
        }
        guard !tokens.isEmpty else { return }
        guard tokens.count <= prefill.maximumSequenceLength else { return }

        let seqLen = tokens.count
        let prefillStart = CFAbsoluteTimeGetCurrent()

        // Fill tokenIDs and positions
        let tokenPtr = prefill.buffers.tokenIDs.contents().bindMemory(to: Int32.self, capacity: seqLen)
        let posPtr = prefill.buffers.positions.contents().bindMemory(to: UInt32.self, capacity: seqLen)
        for i in 0..<seqLen {
            tokenPtr[i] = tokens[i]
            posPtr[i] = UInt32(position + i)
        }

        // Execute sequence graph
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        var dispatchCount = 0
        for step in prefill.steps {
            switch step.mode {
            case .batch:
                encodeBatchStep(encoder, step: step, sequenceLength: UInt32(seqLen))
                dispatchCount += 1
            case .lastToken:
                if step.sync == .bufferBarrier { encoder.memoryBarrier(scope: .buffers) }
                encoder.setComputePipelineState(step.pipeline)
                let lastPos = seqLen - 1
                for (index, buffer, baseOffset) in step.bufferBindings {
                    encoder.setBuffer(buffer, offset: baseOffset + lastPos * (step.perPositionStrides[index] ?? 0), index: index)
                }
                for (index, value) in step.bytesBindings {
                    value.withUnsafeBufferPointer { encoder.setBytes($0.baseAddress!, length: $0.count, index: index) }
                }
                encoder.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
                dispatchCount += 1
            case .perPosition:
                for pos in 0..<seqLen {
                    if step.sync == .bufferBarrier { encoder.memoryBarrier(scope: .buffers) }
                    encoder.setComputePipelineState(step.pipeline)
                    for (index, buffer, baseOffset) in step.bufferBindings {
                        encoder.setBuffer(buffer, offset: baseOffset + pos * (step.perPositionStrides[index] ?? 0), index: index)
                    }
                    for (index, value) in step.bytesBindings {
                        value.withUnsafeBufferPointer { encoder.setBytes($0.baseAddress!, length: $0.count, index: index) }
                    }
                    if let posIndex = step.positionBufferIndex {
                        var posValue = UInt32(position + pos)
                        withUnsafeBytes(of: &posValue) { encoder.setBytes($0.baseAddress!, length: $0.count, index: posIndex) }
                    }
                    encoder.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
                    dispatchCount += 1
                }
            }
        }

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            print("[MetalInference] PREFILL FAILED: \(error.localizedDescription)")
            return
        }

        // Transfer state: prefill float32 hidden → decode float16 hidden
        let decodeHiddenSize = self.plan.buffers.hidden.length / MemoryLayout<Float16>.size
        let prefillHiddenStride = decodeHiddenSize * MemoryLayout<Float>.size
        let lastTokenOffset = (seqLen - 1) * prefillHiddenStride
        if lastTokenOffset + prefillHiddenStride <= prefill.buffers.hidden.length {
            let src = (prefill.buffers.hidden.contents() + lastTokenOffset).bindMemory(to: Float.self, capacity: decodeHiddenSize)
            let dst = self.plan.buffers.hidden.contents().bindMemory(to: Float16.self, capacity: decodeHiddenSize)
            for i in 0..<decodeHiddenSize { dst[i] = Float16(src[i]) }
        }

        // Copy KV cache
        if let prefillKV = prefill.buffers.kvCache,
           let decodeKV = self.plan.buffers.kvCache,
           prefillKV.keys !== decodeKV.keys {
            memcpy(decodeKV.keys.contents(), prefillKV.keys.contents(),
                   min(decodeKV.keys.length, prefillKV.keys.length))
            memcpy(decodeKV.values.contents(), prefillKV.values.contents(),
                   min(decodeKV.values.length, prefillKV.values.length))
        }

        // Warm up conv_state by replaying last kernelSize tokens through decode
        if let convState = self.plan.buffers.convState, self.plan.buffers.convStateKernelSize > 0 {
            let ks = self.plan.buffers.convStateKernelSize
            let warmupCount = min(ks, seqLen)
            memset(convState.contents(), 0, convState.length)

            for i in 0..<warmupCount {
                let replayPosition = seqLen - warmupCount + i
                let b = self.plan.buffers
                b.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(replayPosition)
                b.tokenIn.contents().bindMemory(to: Int32.self, capacity: 1).pointee = tokens[replayPosition]
                guard let cb = commandQueue.makeCommandBuffer(),
                      let enc = cb.makeComputeCommandEncoder() else { break }
                encodeSteps(enc)
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
            }

            // Restore hidden from prefill (warm-up decode overwrote it)
            if lastTokenOffset + prefillHiddenStride <= prefill.buffers.hidden.length {
                let src = (prefill.buffers.hidden.contents() + lastTokenOffset).bindMemory(to: Float.self, capacity: decodeHiddenSize)
                let dst = self.plan.buffers.hidden.contents().bindMemory(to: Float16.self, capacity: decodeHiddenSize)
                for i in 0..<decodeHiddenSize { dst[i] = Float16(src[i]) }
            }
        }

        let totalTime = CFAbsoluteTimeGetCurrent() - prefillStart
        print("[MetalInference] prefill: \(seqLen) tokens, \(dispatchCount) dispatches [\(String(format: "%.3f", totalTime))s] \(String(format: "%.0f", Double(seqLen) / totalTime)) tok/s")
        position += seqLen
    }

    private func encodeBatchStep(_ encoder: MTLComputeCommandEncoder, step: MetalPrefillStep, sequenceLength: UInt32) {
        if step.sync == .bufferBarrier { encoder.memoryBarrier(scope: .buffers) }
        encoder.setComputePipelineState(step.pipeline)
        for (index, buffer, offset) in step.bufferBindings { encoder.setBuffer(buffer, offset: offset, index: index) }
        for (index, value) in step.bytesBindings {
            value.withUnsafeBufferPointer { encoder.setBytes($0.baseAddress!, length: $0.count, index: index) }
        }
        if let seqLenIndex = step.sequenceLengthBindingIndex {
            var seqLen = sequenceLength
            withUnsafeBytes(of: &seqLen) { encoder.setBytes($0.baseAddress!, length: $0.count, index: seqLenIndex) }
        }
        var grid = step.gridSize
        if step.sequenceLengthBindingIndex != nil && grid.height > 1 {
            grid = MTLSize(width: grid.width, height: Int(sequenceLength), depth: grid.depth)
        }
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: step.threadgroupSize)
    }

    // MARK: - Lifecycle

    public mutating func flush() -> Int32 {
        guard hasPendingResult else { return -1 }
        pendingCommandBuffer?.waitUntilCompleted()
        hasPendingResult = false
        pendingCommandBuffer = nil
        return plan.buffers.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
    }

    public mutating func resetCaches() {
        pendingCommandBuffer?.waitUntilCompleted()
        pendingCommandBuffer = nil
        hasPendingResult = false
        position = 0
        if let convState = plan.buffers.convState {
            memset(convState.contents(), 0, convState.length)
        }
    }
}
