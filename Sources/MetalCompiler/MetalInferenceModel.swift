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

    /// Prefill the KV cache with prompt tokens and return the first predicted token.
    ///
    /// Returns the argmax of the prefill logits (the model's first generated token).
    /// The caller should output this token and feed it to the first decode step.
    @discardableResult
    public mutating func prefill(tokens: [Int32]) -> Int32 {
        guard let prefill = prefillPlan else {
            var lastOutput: Int32 = -1
            for token in tokens { lastOutput = decodeSync(tokenID: token) }
            return lastOutput
        }
        guard !tokens.isEmpty else { return -1 }
        guard tokens.count <= prefill.maximumSequenceLength else { return -1 }

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
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return -1 }

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
            return -1
        }

        // Diagnostic: check prefill output before transfer
        do {
            let hiddenDim = self.plan.buffers.hidden.length / MemoryLayout<Float32>.size
            let prefillHiddenStride = hiddenDim * MemoryLayout<Float32>.size
            let lastOff = (seqLen - 1) * prefillHiddenStride
            let pHidden = (prefill.buffers.hidden.contents() + lastOff).bindMemory(to: Float32.self, capacity: hiddenDim)
            var pNorm: Float = 0; var pNan = 0; var pZero = 0
            for i in 0..<hiddenDim {
                let v = Float(pHidden[i])
                if v.isNaN { pNan += 1 }
                if v == 0 { pZero += 1 }
                pNorm += v * v
            }
            pNorm = sqrtf(pNorm)
            print("[Prefill→Decode] prefill hidden (last token): [\(Float(pHidden[0])), \(Float(pHidden[1])), \(Float(pHidden[2])), \(Float(pHidden[3]))] norm=\(pNorm) nan=\(pNan) zero=\(pZero)/\(hiddenDim)")

            // Check prefill logits
            let logitsDim = prefill.buffers.logits.length / MemoryLayout<Float32>.size
            let pLogits = prefill.buffers.logits.contents().bindMemory(to: Float32.self, capacity: logitsDim)
            var maxVal: Float = -.infinity; var maxIdx = 0; var lNan = 0
            for i in 0..<logitsDim {
                let v = Float(pLogits[i])
                if v.isNaN { lNan += 1 }
                if v > maxVal { maxVal = v; maxIdx = i }
            }
            print("[Prefill→Decode] prefill logits[\(logitsDim)]: max=\(maxVal)@\(maxIdx) nan=\(lNan)")

            // Check conv_state
            if let cs = prefill.buffers.convState {
                let csDim = cs.length / MemoryLayout<Float32>.size
                let csPtr = cs.contents().bindMemory(to: Float32.self, capacity: csDim)
                var csNan = 0; var csZero = 0
                for i in 0..<csDim {
                    let v = Float(csPtr[i])
                    if v.isNaN { csNan += 1 }
                    if v == 0 { csZero += 1 }
                }
                print("[Prefill→Decode] conv_state[\(csDim)]: nan=\(csNan) zero=\(csZero) first=[\(Float(csPtr[0])), \(Float(csPtr[1]))]")
            }
        }

        // Transfer hidden: convert last token from prefill F32 to decode F16
        let decodeHiddenSize = self.plan.buffers.hidden.length / MemoryLayout<Float16>.size
        let prefillHiddenStride = decodeHiddenSize * MemoryLayout<Float32>.size
        let lastTokenOffset = (seqLen - 1) * prefillHiddenStride
        if lastTokenOffset + prefillHiddenStride <= prefill.buffers.hidden.length {
            let src = (prefill.buffers.hidden.contents() + lastTokenOffset)
                .bindMemory(to: Float32.self, capacity: decodeHiddenSize)
            let dst = self.plan.buffers.hidden.contents()
                .bindMemory(to: Float16.self, capacity: decodeHiddenSize)
            for i in 0..<decodeHiddenSize {
                dst[i] = Float16(src[i])
            }
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

        // Copy conv_state (already in Float16 from extract_conv_state kernel)
        if let prefillConvState = prefill.buffers.convState,
           let decodeConvState = self.plan.buffers.convState {
            memcpy(decodeConvState.contents(), prefillConvState.contents(),
                   min(decodeConvState.length, prefillConvState.length))
        }

        let totalTime = CFAbsoluteTimeGetCurrent() - prefillStart
        print("[MetalInference] prefill: \(seqLen) tokens, \(dispatchCount) dispatches [\(String(format: "%.3f", totalTime))s] \(String(format: "%.0f", Double(seqLen) / totalTime)) tok/s")
        position += seqLen

        // Return the first predicted token (argmax of prefill logits)
        return prefill.buffers.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
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
