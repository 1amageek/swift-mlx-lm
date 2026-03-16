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
        for (stepIdx, step) in plan.steps.enumerated() {
            if step.sync == .bufferBarrier { enc.memoryBarrier(scope: .buffers) }
            enc.setComputePipelineState(step.pipeline)
            for (index, buffer, offset) in step.bufferBindings {
                if offset >= buffer.length {
                    print("[Metal] ERROR: step \(stepIdx) bind[\(index)] offset \(offset) >= len \(buffer.length)")
                    continue
                }
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

    // MARK: - Prefill (step-by-step for diagnostic)

    public mutating func prefill(tokens: [Int32]) {
        guard let prefill = prefillPlan else {
            print("[MetalInference] prefill: fallback to sequential decode (\(tokens.count) tokens)")
            for token in tokens { let _ = decodeSync(tokenID: token) }
            return
        }

        guard !tokens.isEmpty else { return }
        let seqLen = tokens.count
        let prefillStart = CFAbsoluteTimeGetCurrent()

        print("[MetalInference] prefill: seqLen=\(seqLen) maxSeqLen=\(prefill.maximumSequenceLength)")
        print("[MetalInference] buffers: hidden=\(prefill.buffers.hidden.length) scratch=\(prefill.buffers.scratch.length) logits=\(prefill.buffers.logits.length)")

        // Fill tokenIDs and positions
        let tokenPtr = prefill.buffers.tokenIDs.contents().bindMemory(to: Int32.self, capacity: seqLen)
        let posPtr = prefill.buffers.positions.contents().bindMemory(to: UInt32.self, capacity: seqLen)
        for i in 0..<seqLen {
            tokenPtr[i] = tokens[i]
            posPtr[i] = UInt32(position + i)
        }

        let batchCount = prefill.steps.filter { $0.mode == .batch }.count
        let perPosCount = prefill.steps.filter { $0.mode == .perPosition }.count
        let lastTokenCount = prefill.steps.filter { $0.mode == .lastToken }.count
        print("[MetalInference] prefill: \(prefill.stepCount) steps (\(batchCount) batch + \(perPosCount) perPos + \(lastTokenCount) lastToken)")

        // Run step-by-step: each step gets its own command buffer.
        // This isolates NaN to the exact step.
        let elementSize = MemoryLayout<Float16>.size
        let hiddenSize = self.plan.buffers.hidden.length / elementSize
        var dispatchCount = 0
        var nanStep = -1

        for (stepIndex, step) in prefill.steps.enumerated() {
            guard let cb = commandQueue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else {
                print("[MetalInference] ERROR: cannot create encoder at step \(stepIndex)")
                return
            }
            enc.memoryBarrier(scope: .buffers)

            switch step.mode {
            case .batch:
                enc.setComputePipelineState(step.pipeline)
                for (index, buffer, offset) in step.bufferBindings { enc.setBuffer(buffer, offset: offset, index: index) }
                for (index, value) in step.bytesBindings { value.withUnsafeBufferPointer { enc.setBytes($0.baseAddress!, length: $0.count, index: index) } }
                if let si = step.sequenceLengthBindingIndex {
                    var sl = UInt32(seqLen)
                    withUnsafeBytes(of: &sl) { enc.setBytes($0.baseAddress!, length: $0.count, index: si) }
                }
                var grid = step.gridSize
                if step.sequenceLengthBindingIndex != nil && grid.height > 1 {
                    grid = MTLSize(width: grid.width, height: seqLen, depth: grid.depth)
                }
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: step.threadgroupSize)
                dispatchCount += 1

            case .perPosition:
                for pos in 0..<seqLen {
                    enc.setComputePipelineState(step.pipeline)
                    for (index, buffer, base) in step.bufferBindings {
                        enc.setBuffer(buffer, offset: base + pos * (step.perPositionStrides[index] ?? 0), index: index)
                    }
                    for (index, value) in step.bytesBindings { value.withUnsafeBufferPointer { enc.setBytes($0.baseAddress!, length: $0.count, index: index) } }
                    if let pi = step.positionBufferIndex {
                        var pv = UInt32(position + pos)
                        withUnsafeBytes(of: &pv) { enc.setBytes($0.baseAddress!, length: $0.count, index: pi) }
                    }
                    enc.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
                    dispatchCount += 1
                }

            case .lastToken:
                enc.setComputePipelineState(step.pipeline)
                for (index, buffer, base) in step.bufferBindings {
                    enc.setBuffer(buffer, offset: base + (seqLen - 1) * (step.perPositionStrides[index] ?? 0), index: index)
                }
                for (index, value) in step.bytesBindings { value.withUnsafeBufferPointer { enc.setBytes($0.baseAddress!, length: $0.count, index: index) } }
                enc.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
                dispatchCount += 1
            }

            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()

            if let error = cb.error {
                print("[Prefill step \(stepIndex)] GPU ERROR: \(error)")
                return
            }

            // Check for NaN after each step
            let hp = prefill.buffers.hidden.contents().bindMemory(to: Float.self, capacity: hiddenSize)
            let hasNaN = (0..<hiddenSize).contains { hp[$0].isNaN }
            let sp = prefill.buffers.scratch.contents().bindMemory(to: Float.self, capacity: hiddenSize)
            let scratchNaN = (0..<hiddenSize).contains { sp[$0].isNaN }

            let modeStr = step.mode == .batch ? "B" : step.mode == .perPosition ? "P" : "L"
            let kernelLabel = step.pipeline.label ?? "?"

            // Log every step
            if stepIndex < 20 || hasNaN || scratchNaN || stepIndex % 50 == 0 {
                let hs = (0..<min(4, hiddenSize)).map { hp[$0] }
                let ss = (0..<min(4, hiddenSize)).map { sp[$0] }
                print("[Prefill step \(stepIndex)] (\(modeStr)) [\(kernelLabel)] hidden=\(hs) hNaN=\(hasNaN) scratch=\(ss) sNaN=\(scratchNaN)")
            }

            if (hasNaN || scratchNaN) && nanStep < 0 {
                nanStep = stepIndex

                // === Hypothesis 1: scratch has NaN/inf somewhere in the full 12288×seqLen region ===
                let scratchF32Count = prefill.buffers.scratch.length / MemoryLayout<Float>.size
                let scratchAll = prefill.buffers.scratch.contents().bindMemory(to: Float.self, capacity: scratchF32Count)
                var scratchNaNCount = 0
                var scratchInfCount = 0
                var scratchMaxAbs: Float = 0
                var firstNaNIdx = -1
                var firstInfIdx = -1
                for j in 0..<min(scratchF32Count, 12288 * seqLen) {
                    let v = scratchAll[j]
                    if v.isNaN { scratchNaNCount += 1; if firstNaNIdx < 0 { firstNaNIdx = j } }
                    if v.isInfinite { scratchInfCount += 1; if firstInfIdx < 0 { firstInfIdx = j } }
                    let a = abs(v)
                    if a.isFinite && a > scratchMaxAbs { scratchMaxAbs = a }
                }
                print("[Diag H1] scratch full scan: NaN=\(scratchNaNCount) inf=\(scratchInfCount) maxAbs=\(scratchMaxAbs) firstNaN=\(firstNaNIdx) firstInf=\(firstInfIdx)")

                // === Hypothesis 2: weight data at this step's offset ===
                for (index, buffer, offset) in step.bufferBindings {
                    if buffer.length > 100_000_000 {  // weight buffer
                        let wCount = min((buffer.length - offset) / 2, 100)
                        let wp = (buffer.contents() + offset).bindMemory(to: UInt16.self, capacity: wCount)
                        var wNaN = 0
                        for j in 0..<wCount {
                            let f = Float(Float16(bitPattern: wp[j]))
                            if f.isNaN { wNaN += 1 }
                        }
                        let wSample = (0..<min(4, wCount)).map { String(format: "0x%04x", wp[$0]) }
                        print("[Diag H2] bind[\(index)] weight: first4=\(wSample) NaN_in_100=\(wNaN)")
                    }
                }

                // === Hypothesis 3: buffer identity (aliasing) ===
                let scratchPtr = prefill.buffers.hidden.contents()
                let hiddenPtr = prefill.buffers.scratch.contents()
                print("[Diag H3] hidden.ptr=\(scratchPtr) scratch.ptr=\(hiddenPtr) same=\(scratchPtr == hiddenPtr)")

                // === Hypothesis 4: bytes bindings (inputDim, outputDim, seqLen) ===
                for (index, value) in step.bytesBindings {
                    if value.count == 4 {
                        let v = value.withUnsafeBufferPointer { $0.baseAddress!.withMemoryRebound(to: UInt32.self, capacity: 1) { $0.pointee } }
                        print("[Diag H4] bytes[\(index)]=\(v)")
                    }
                }

                // === Hypothesis 5: hidden buffer capacity check ===
                let outputDimFromBytes = step.bytesBindings.first(where: { $0.0 == 4 }).flatMap { val -> UInt32? in
                    val.1.withUnsafeBufferPointer { $0.baseAddress?.withMemoryRebound(to: UInt32.self, capacity: 1) { $0.pointee } }
                } ?? 0
                let requiredHiddenBytes = seqLen * Int(outputDimFromBytes) * MemoryLayout<Float>.size
                let actualHiddenBytes = prefill.buffers.hidden.length
                print("[Diag H5] hidden capacity: required=\(requiredHiddenBytes) actual=\(actualHiddenBytes) sufficient=\(requiredHiddenBytes <= actualHiddenBytes)")

                // === Hypothesis 6: Metal library cache ===
                print("[Diag H6] kernel label=\(kernelLabel) pipeline.maxThreads=\(step.pipeline.maxTotalThreadsPerThreadgroup)")

                // Buffer binding details
                for (index, buffer, offset) in step.bufferBindings {
                    print("[Diag] bind[\(index)]: offset=\(offset) len=\(buffer.length) ptr=\(buffer.contents())")
                }
            }
            if nanStep >= 0 && stepIndex >= nanStep + 3 { break }
        }

        if nanStep >= 0 {
            print("[MetalInference] PREFILL FAILED: NaN at step \(nanStep). State NOT advanced.")
            return
        }

        // Transfer state: prefill hidden (float32) → decode hidden (float16)
        let decodeHiddenSize = self.plan.buffers.hidden.length / MemoryLayout<Float16>.size
        let prefillHiddenStride = decodeHiddenSize * MemoryLayout<Float>.size
        let lastTokenOffset = (seqLen - 1) * prefillHiddenStride
        if lastTokenOffset + prefillHiddenStride <= prefill.buffers.hidden.length {
            let src = (prefill.buffers.hidden.contents() + lastTokenOffset).bindMemory(to: Float.self, capacity: decodeHiddenSize)
            let dst = self.plan.buffers.hidden.contents().bindMemory(to: Float16.self, capacity: decodeHiddenSize)
            for i in 0..<decodeHiddenSize {
                dst[i] = Float16(src[i])
            }
        }
        if let prefillKV = prefill.buffers.kvCache,
           let decodeKV = self.plan.buffers.kvCache,
           prefillKV.keys !== decodeKV.keys {
            memcpy(decodeKV.keys.contents(), prefillKV.keys.contents(),
                   min(decodeKV.keys.length, prefillKV.keys.length))
            memcpy(decodeKV.values.contents(), prefillKV.values.contents(),
                   min(decodeKV.values.length, prefillKV.values.length))
        }

        let totalTime = CFAbsoluteTimeGetCurrent() - prefillStart
        let tokPerSec = totalTime > 0 ? Double(seqLen) / totalTime : 0
        print("[MetalInference] prefill done: \(seqLen) tokens, \(dispatchCount) dispatches, position \(position)→\(position + seqLen) [\(String(format: "%.3f", totalTime))s] \(String(format: "%.0f", tokPerSec)) tok/s")
        position += seqLen
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
    }
}
