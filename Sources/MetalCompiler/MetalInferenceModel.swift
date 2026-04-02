import Metal
import LMIR

public struct MetalInferenceModel: @unchecked Sendable {

    public let plan: MetalDispatchPlan
    public var prefillPlan: MetalPrefillPlan?
    public let commandQueue: MTLCommandQueue
    public var position: Int = 0

    private let submission: MetalSubmissionContext
    private var pendingCommandBuffer: MTLCommandBuffer?
    private var hasPendingResult: Bool = false

    private struct PostPrefillTransferPlan {
        let hiddenSourceOffset: Int
        let hiddenCopySize: Int
        let kvCopySize: Int
        let valueCopySize: Int
        let convCopySize: Int
        let requiresCPUHiddenConversion: Bool
        let usesSharedDecodeHidden: Bool

        var needsBlit: Bool {
            hiddenCopySize > 0 || kvCopySize > 0 || valueCopySize > 0 || convCopySize > 0
        }

        var canEncodeInSameTransaction: Bool {
            !requiresCPUHiddenConversion && needsBlit
        }

        var shouldStageHiddenOnCPU: Bool {
            requiresCPUHiddenConversion && hiddenCopySize > 0
        }

        var needsStandaloneBlit: Bool {
            needsBlit && !canEncodeInSameTransaction
        }

        func afterInlineBlit() -> Self {
            Self(
                hiddenSourceOffset: hiddenSourceOffset,
                hiddenCopySize: 0,
                kvCopySize: 0,
                valueCopySize: 0,
                convCopySize: 0,
                requiresCPUHiddenConversion: requiresCPUHiddenConversion,
                usesSharedDecodeHidden: usesSharedDecodeHidden
            )
        }

        func withoutHiddenCopy() -> Self {
            Self(
                hiddenSourceOffset: hiddenSourceOffset,
                hiddenCopySize: 0,
                kvCopySize: kvCopySize,
                valueCopySize: valueCopySize,
                convCopySize: convCopySize,
                requiresCPUHiddenConversion: false,
                usesSharedDecodeHidden: false
            )
        }
    }

    public init(plan: MetalDispatchPlan, device: MTLDevice) throws {
        self.plan = plan
        self.prefillPlan = nil
        guard let queue = device.makeCommandQueue() else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create command queue")
        }
        self.commandQueue = queue
        self.submission = MetalSubmissionContext(commandQueue: queue)
        try Self.zeroStateBuffers(plan.buffers, submission: submission)
    }

    private static func zeroStateBuffers(_ buffers: MetalBufferSet, submission: MetalSubmissionContext) throws {
        _ = try submission.withTransaction(label: "state.zero") { transaction in
            try transaction.withBlitEncoder { blit in
                blit.fill(buffer: buffers.hidden, range: 0..<buffers.hidden.length, value: 0)
                blit.fill(buffer: buffers.residual, range: 0..<buffers.residual.length, value: 0)
                blit.fill(buffer: buffers.scratch, range: 0..<buffers.scratch.length, value: 0)
                blit.fill(buffer: buffers.logits, range: 0..<buffers.logits.length, value: 0)
                if let kv = buffers.kvCache {
                    blit.fill(buffer: kv.keys, range: 0..<kv.keys.length, value: 0)
                    blit.fill(buffer: kv.values, range: 0..<kv.values.length, value: 0)
                }
                if let convState = buffers.convState {
                    blit.fill(buffer: convState, range: 0..<convState.length, value: 0)
                }
            }
        }
    }

    private static func makePrivateBuffer(length: Int, device: MTLDevice) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: length, options: .storageModePrivate) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate prompt state buffer")
        }
        return buffer
    }

    // MARK: - Decode

    private func encodeSteps(_ enc: MTLComputeCommandEncoder) {
        for step in plan.steps {
            step.bindings.bind(to: enc)
            step.descriptor.encode(on: enc)
        }
    }

    private mutating func consumePendingDecodeResult() -> Int32 {
        guard let pendingCommandBuffer else {
            return -1
        }
        do {
            try submission.waitUntilCompleted(pendingCommandBuffer)
            return plan.buffers.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
        } catch {
            print("[MetalInference] Pending decode failed: \(error)")
            return -1
        }
    }

    public mutating func decode(tokenID: Int32) -> Int32 {
        let b = plan.buffers
        var result: Int32 = -1
        if hasPendingResult {
            result = consumePendingDecodeResult()
        }
        b.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(position)
        b.tokenIn.contents().bindMemory(to: Int32.self, capacity: 1).pointee = tokenID
        do {
            let cb = try submission.withCompute(label: "decode", waitUntilCompleted: false) { enc in
                encodeSteps(enc)
            }
            pendingCommandBuffer = cb
            hasPendingResult = true
            position += 1
        } catch {
            print("[MetalInference] Failed to submit decode: \(error)")
        }
        return result
    }

    public mutating func decodeSync(tokenID: Int32) -> Int32 {
        let b = plan.buffers
        b.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(position)
        b.tokenIn.contents().bindMemory(to: Int32.self, capacity: 1).pointee = tokenID
        do {
            _ = try submission.withCompute(label: "decode.sync") { enc in
                encodeSteps(enc)
            }
            position += 1
            return b.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
        } catch {
            print("[MetalInference] GPU error: \(error)")
            return -1
        }
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

        // Use sequence-parallel prefill for short prompts (verified correct for ≤8 tokens).
        // Longer prompts fall back to sequential decode to avoid a known
        // cross-threadgroup KV cache visibility issue in perPosition flash attention.
        // TODO: Replace with a proper batch flash attention kernel.
        if tokens.count > 8 {
            var lastOutput: Int32 = -1
            for token in tokens { lastOutput = decodeSync(tokenID: token) }
            return lastOutput
        }
        guard !tokens.isEmpty else { return -1 }
        guard tokens.count <= prefill.maximumSequenceLength else { return -1 }

        let seqLen = tokens.count

        // Fill tokenIDs and positions
        let tokenPtr = prefill.buffers.tokenIDs.contents().bindMemory(to: Int32.self, capacity: seqLen)
        let posPtr = prefill.buffers.positions.contents().bindMemory(to: UInt32.self, capacity: seqLen)
        for i in 0..<seqLen {
            tokenPtr[i] = tokens[i]
            posPtr[i] = UInt32(position + i)
        }

        var transferPlan = makePostPrefillTransferPlan(prefill: prefill, sequenceLength: seqLen)

        // Execute sequence graph
        do {
            if transferPlan.canEncodeInSameTransaction {
                _ = try submission.withTransaction(label: "prefill+postprocess") { transaction in
                    try transaction.withComputeEncoder { compute in
                        encodePrefillSteps(compute, prefill: prefill, sequenceLength: seqLen)
                    }
                    try transaction.withBlitEncoder { blit in
                        encodePostPrefillCopies(blit, prefill: prefill, plan: transferPlan)
                    }
                }
                transferPlan = transferPlan.afterInlineBlit()
            } else {
                _ = try submission.withCompute(label: "prefill") { encoder in
                    encodePrefillSteps(encoder, prefill: prefill, sequenceLength: seqLen)
                }
            }
        } catch {
            print("[MetalInference] PREFILL FAILED: \(error)")
            return -1
        }

        if transferPlan.shouldStageHiddenOnCPU || transferPlan.usesSharedDecodeHidden {
            let decodeElementSize = self.plan.buffers.bufferPrecision.byteSize
            let decodeHiddenSize = self.plan.buffers.hidden.length / decodeElementSize
            let src = (prefill.buffers.hidden.contents() + transferPlan.hiddenSourceOffset)
                    .bindMemory(to: Float32.self, capacity: decodeHiddenSize)
            if transferPlan.shouldStageHiddenOnCPU {
                switch self.plan.buffers.bufferPrecision {
                case .float16:
                    let staging = prefill.buffers.scratch.contents()
                        .bindMemory(to: Float16.self, capacity: decodeHiddenSize)
                    for i in 0..<decodeHiddenSize {
                        staging[i] = Float16(src[i])
                    }
                case .bfloat16:
                    let staging = prefill.buffers.scratch.contents()
                        .bindMemory(to: BFloat16.self, capacity: decodeHiddenSize)
                    for i in 0..<decodeHiddenSize {
                        staging[i] = BFloat16(src[i])
                    }
                case .float32:
                    let staging = prefill.buffers.scratch.contents()
                        .bindMemory(to: Float32.self, capacity: decodeHiddenSize)
                    for i in 0..<decodeHiddenSize {
                        staging[i] = src[i]
                    }
                }
            } else if transferPlan.usesSharedDecodeHidden {
                switch self.plan.buffers.bufferPrecision {
                case .float16:
                    let dst = self.plan.buffers.hidden.contents()
                        .bindMemory(to: Float16.self, capacity: decodeHiddenSize)
                    for i in 0..<decodeHiddenSize {
                        dst[i] = Float16(src[i])
                    }
                case .bfloat16:
                    let dst = self.plan.buffers.hidden.contents()
                        .bindMemory(to: BFloat16.self, capacity: decodeHiddenSize)
                    for i in 0..<decodeHiddenSize {
                        dst[i] = BFloat16(src[i])
                    }
                case .float32:
                    let dst = self.plan.buffers.hidden.contents()
                        .bindMemory(to: Float32.self, capacity: decodeHiddenSize)
                    for i in 0..<decodeHiddenSize {
                        dst[i] = src[i]
                    }
                }
            }
        }

        if transferPlan.needsStandaloneBlit {
            do {
                try submission.withBlit(label: "prefill.postprocess") { blit in
                    if transferPlan.shouldStageHiddenOnCPU {
                        blit.copy(from: prefill.buffers.scratch, sourceOffset: 0,
                                  to: self.plan.buffers.hidden, destinationOffset: 0,
                                  size: transferPlan.hiddenCopySize)
                    }
                    encodePostPrefillCopies(
                        blit,
                        prefill: prefill,
                        plan: transferPlan.shouldStageHiddenOnCPU ? transferPlan.withoutHiddenCopy() : transferPlan)
                }
            } catch {
                print("[MetalInference] Failed to copy post-prefill state: \(error)")
                return -1
            }
        }

        position += seqLen

        // Return the first predicted token (argmax of prefill logits)
        return prefill.buffers.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
    }

    private func encodeBatchStep(_ encoder: MTLComputeCommandEncoder, step: MetalPrefillStep, sequenceLength: UInt32) {
        step.bindings.bind(to: encoder)
        step.bindRuntimeArguments(encoder: encoder, sequenceLength: sequenceLength)
        let grid = step.resolvedGridSize(sequenceLength: Int(sequenceLength))
        step.descriptor.encode(on: encoder, gridSize: grid)
    }

    private func encodePrefillSteps(
        _ encoder: MTLComputeCommandEncoder,
        prefill: MetalPrefillPlan,
        sequenceLength: Int
    ) {
        for step in prefill.steps {
            switch step.mode {
            case .batch:
                encodeBatchStep(encoder, step: step, sequenceLength: UInt32(sequenceLength))
            case .lastToken:
                if step.sync == .bufferBarrier { encoder.memoryBarrier(scope: .buffers) }
                encoder.setComputePipelineState(step.pipeline)
                let lastPos = sequenceLength - 1
                step.bindStaticArguments(encoder: encoder, position: lastPos)
                encoder.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
            case .perPosition:
                for pos in 0..<sequenceLength {
                    if step.sync == .bufferBarrier { encoder.memoryBarrier(scope: .buffers) }
                    encoder.setComputePipelineState(step.pipeline)
                    step.bindStaticArguments(encoder: encoder, position: pos)
                    if let posIndex = step.positionBufferIndex {
                        var posValue = UInt32(position + pos)
                        withUnsafeBytes(of: &posValue) { encoder.setBytes($0.baseAddress!, length: $0.count, index: posIndex) }
                    }
                    encoder.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
                }
            }
        }
    }

    private func encodePostPrefillCopies(
        _ blit: MTLBlitCommandEncoder,
        prefill: MetalPrefillPlan,
        plan: PostPrefillTransferPlan
    ) {
        if plan.hiddenCopySize > 0 {
            blit.copy(from: prefill.buffers.hidden, sourceOffset: plan.hiddenSourceOffset,
                      to: self.plan.buffers.hidden, destinationOffset: 0,
                      size: plan.hiddenCopySize)
        }
        if plan.kvCopySize > 0,
           let prefillKV = prefill.buffers.kvCache,
           let decodeKV = self.plan.buffers.kvCache {
            blit.copy(from: prefillKV.keys, sourceOffset: 0,
                      to: decodeKV.keys, destinationOffset: 0,
                      size: plan.kvCopySize)
        }
        if plan.valueCopySize > 0,
           let prefillKV = prefill.buffers.kvCache,
           let decodeKV = self.plan.buffers.kvCache {
            blit.copy(from: prefillKV.values, sourceOffset: 0,
                      to: decodeKV.values, destinationOffset: 0,
                      size: plan.valueCopySize)
        }
        if plan.convCopySize > 0,
           let prefillConvState = prefill.buffers.convState,
           let decodeConvState = self.plan.buffers.convState {
            blit.copy(from: prefillConvState, sourceOffset: 0,
                      to: decodeConvState, destinationOffset: 0,
                      size: plan.convCopySize)
        }
    }

    private func makePostPrefillTransferPlan(
        prefill: MetalPrefillPlan,
        sequenceLength: Int
    ) -> PostPrefillTransferPlan {
        let decodeElementSize = self.plan.buffers.bufferPrecision.byteSize
        let decodeHiddenSize = self.plan.buffers.hidden.length / decodeElementSize
        let prefillHiddenStride = decodeHiddenSize * MemoryLayout<Float32>.size
        let hiddenSourceOffset = (sequenceLength - 1) * prefillHiddenStride
        let requiresCPUHiddenConversion =
            self.plan.buffers.hidden.storageMode == .private &&
            self.plan.buffers.bufferPrecision != .float32
        let usesSharedDecodeHidden = self.plan.buffers.hidden.storageMode != .private

        let hiddenCopySize: Int
        if hiddenSourceOffset + prefillHiddenStride <= prefill.buffers.hidden.length,
           self.plan.buffers.hidden.storageMode == .private {
            hiddenCopySize = decodeHiddenSize * decodeElementSize
        } else {
            hiddenCopySize = 0
        }

        let kvCopySize: Int
        if let prefillKV = prefill.buffers.kvCache,
           let decodeKV = self.plan.buffers.kvCache,
           prefillKV.keys !== decodeKV.keys {
            kvCopySize = min(decodeKV.keys.length, prefillKV.keys.length)
        } else {
            kvCopySize = 0
        }

        let valueCopySize: Int
        if let prefillKV = prefill.buffers.kvCache,
           let decodeKV = self.plan.buffers.kvCache,
           prefillKV.values !== decodeKV.values {
            valueCopySize = min(decodeKV.values.length, prefillKV.values.length)
        } else {
            valueCopySize = 0
        }

        let convCopySize: Int
        if let prefillConvState = prefill.buffers.convState,
           let decodeConvState = self.plan.buffers.convState,
           prefillConvState !== decodeConvState {
            convCopySize = min(decodeConvState.length, prefillConvState.length)
        } else {
            convCopySize = 0
        }

        return PostPrefillTransferPlan(
            hiddenSourceOffset: hiddenSourceOffset,
            hiddenCopySize: hiddenCopySize,
            kvCopySize: kvCopySize,
            valueCopySize: valueCopySize,
            convCopySize: convCopySize,
            requiresCPUHiddenConversion: requiresCPUHiddenConversion,
            usesSharedDecodeHidden: usesSharedDecodeHidden
        )
    }

    // MARK: - Lifecycle

    public mutating func flush() -> Int32 {
        guard hasPendingResult else { return -1 }
        let result = consumePendingDecodeResult()
        hasPendingResult = false
        pendingCommandBuffer = nil
        return result
    }

    public func makePromptState(firstToken: Int32) throws -> MetalPromptState {
        let device = submission.device
        let snapshotKVKeys = try plan.buffers.kvCache.map {
            try Self.makePrivateBuffer(length: $0.keys.length, device: device)
        }
        let snapshotKVValues = try plan.buffers.kvCache.map {
            try Self.makePrivateBuffer(length: $0.values.length, device: device)
        }
        let snapshotConvState = try plan.buffers.convState.map {
            try Self.makePrivateBuffer(length: $0.length, device: device)
        }

        _ = try submission.withTransaction(label: "prompt.snapshot") { transaction in
            try transaction.withBlitEncoder { blit in
                if let liveKV = plan.buffers.kvCache,
                   let snapshotKVKeys,
                   let snapshotKVValues {
                    blit.copy(from: liveKV.keys, sourceOffset: 0, to: snapshotKVKeys, destinationOffset: 0, size: liveKV.keys.length)
                    blit.copy(from: liveKV.values, sourceOffset: 0, to: snapshotKVValues, destinationOffset: 0, size: liveKV.values.length)
                }
                if let liveConvState = plan.buffers.convState,
                   let snapshotConvState {
                    blit.copy(from: liveConvState, sourceOffset: 0, to: snapshotConvState, destinationOffset: 0, size: liveConvState.length)
                }
            }
        }

        return MetalPromptState(
            position: position,
            firstToken: firstToken,
            kvKeys: snapshotKVKeys,
            kvValues: snapshotKVValues,
            convState: snapshotConvState
        )
    }

    public mutating func restore(promptState: MetalPromptState) throws {
        pendingCommandBuffer = nil
        hasPendingResult = false

        _ = try submission.withTransaction(label: "prompt.restore") { transaction in
            try transaction.withBlitEncoder { blit in
                blit.fill(buffer: plan.buffers.hidden, range: 0..<plan.buffers.hidden.length, value: 0)
                blit.fill(buffer: plan.buffers.residual, range: 0..<plan.buffers.residual.length, value: 0)
                blit.fill(buffer: plan.buffers.scratch, range: 0..<plan.buffers.scratch.length, value: 0)
                blit.fill(buffer: plan.buffers.logits, range: 0..<plan.buffers.logits.length, value: 0)

                if let liveKV = plan.buffers.kvCache,
                   let snapshotKVKeys = promptState.kvKeys,
                   let snapshotKVValues = promptState.kvValues {
                    blit.copy(from: snapshotKVKeys, sourceOffset: 0, to: liveKV.keys, destinationOffset: 0, size: liveKV.keys.length)
                    blit.copy(from: snapshotKVValues, sourceOffset: 0, to: liveKV.values, destinationOffset: 0, size: liveKV.values.length)
                }
                if let liveConvState = plan.buffers.convState,
                   let snapshotConvState = promptState.convState {
                    blit.copy(from: snapshotConvState, sourceOffset: 0, to: liveConvState, destinationOffset: 0, size: liveConvState.length)
                }
            }
        }

        position = promptState.position
    }

    public mutating func resetCaches() {
        if let pendingCommandBuffer {
            do {
                try submission.waitUntilCompleted(pendingCommandBuffer)
            } catch {
                print("[MetalInference] Pending decode failed during reset: \(error)")
            }
        }
        pendingCommandBuffer = nil
        hasPendingResult = false
        position = 0
        do {
            try Self.zeroStateBuffers(plan.buffers, submission: submission)
        } catch {
            print("[MetalInference] Failed to reset GPU state: \(error)")
        }
    }
}
