import Metal
import Darwin
import LMIR

/// Mutable runtime for token-by-token inference using a compiled Metal model.
public struct MetalInferenceModel: @unchecked Sendable {

    public private(set) var compiledModel: MetalCompiledModel
    public let commandQueue: MTLCommandQueue
    public var position: Int = 0

    private let submission: MetalSubmissionContext
    private let decodeExecutor = MetalDecodeExecutor()
    private let prefillExecutor = MetalPrefillExecutor()
    private let promptStateStore = MetalPromptStateStore()
    private var pendingCommandBuffer: MTLCommandBuffer?
    private var hasPendingResult: Bool = false
    private var hiddenOverrideStagingWorkspace: MetalHiddenOverrideWorkspace?

    /// Backward-compatible decode plan view.
    ///
    /// Prefer ``decodePlan`` or ``compiledModel`` for new call sites.
    public var plan: MetalDispatchPlan { compiledModel.decodePlan }

    /// Decode-time dispatch plan extracted from the compiled model.
    public var decodePlan: MetalDispatchPlan { compiledModel.decodePlan }

    /// Optional sequence-oriented prefill plan paired with the decode plan.
    public var prefillPlan: MetalPrefillPlan? {
        get { compiledModel.prefillPlan }
        set { compiledModel = compiledModel.withPrefillPlan(newValue) }
    }

    /// Shared runtime buffers used by decode execution.
    public var buffers: MetalBufferSet { decodePlan.buffers }

    public init(plan: MetalDispatchPlan, device: MTLDevice) throws {
        self.compiledModel = MetalCompiledModel(decodePlan: plan)
        guard let queue = device.makeCommandQueue() else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create command queue")
        }
        self.commandQueue = queue
        self.submission = MetalSubmissionContext(commandQueue: queue)
        try Self.zeroStateBuffers(plan.buffers, submission: submission)
    }

    public init(compiledModel: MetalCompiledModel, device: MTLDevice) throws {
        self.compiledModel = compiledModel
        guard let queue = device.makeCommandQueue() else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create command queue")
        }
        self.commandQueue = queue
        self.submission = MetalSubmissionContext(commandQueue: queue)
        try Self.zeroStateBuffers(compiledModel.decodePlan.buffers, submission: submission)
    }

    public init(plan: MetalCompiledModel, device: MTLDevice) throws {
        try self.init(compiledModel: plan, device: device)
    }

    private static func zeroStateBuffers(_ buffers: MetalBufferSet, submission: MetalSubmissionContext) throws {
        _ = try submission.withTransaction(label: "state.zero") { transaction in
            try transaction.withBlitEncoder { blit in
                blit.fill(buffer: buffers.hidden, range: 0..<buffers.hidden.length, value: 0)
                blit.fill(buffer: buffers.residual, range: 0..<buffers.residual.length, value: 0)
                blit.fill(buffer: buffers.scratch, range: 0..<buffers.scratch.length, value: 0)
                blit.fill(buffer: buffers.logits, range: 0..<buffers.logits.length, value: 0)
                blit.fill(buffer: buffers.ropePositionAxes, range: 0..<buffers.ropePositionAxes.length, value: 0)
                if let kv = buffers.kvCache {
                    blit.fill(buffer: kv.keys, range: 0..<kv.keys.length, value: 0)
                    blit.fill(buffer: kv.values, range: 0..<kv.values.length, value: 0)
                }
                if let convState = buffers.convState {
                    blit.fill(buffer: convState, range: 0..<convState.length, value: 0)
                }
                if let recurrentState = buffers.recurrentState {
                    blit.fill(buffer: recurrentState, range: 0..<recurrentState.length, value: 0)
                }
                if let perLayerInputs = buffers.perLayerInputs {
                    blit.fill(buffer: perLayerInputs, range: 0..<perLayerInputs.length, value: 0)
                }
            }
        }
    }

    public mutating func decode(
        tokenID: Int32,
        ropePositionAxes: (UInt32, UInt32, UInt32)? = nil
    ) -> Int32 {
        decodeExecutor.decode(
            plan: decodePlan,
            submission: submission,
            position: &position,
            pendingCommandBuffer: &pendingCommandBuffer,
            hasPendingResult: &hasPendingResult,
            tokenID: tokenID,
            ropePositionAxes: ropePositionAxes
        )
    }

    public mutating func decodeSync(
        tokenID: Int32,
        ropePositionAxes: (UInt32, UInt32, UInt32)? = nil
    ) -> Int32 {
        decodeExecutor.decodeSync(
            plan: decodePlan,
            submission: submission,
            position: &position,
            tokenID: tokenID,
            ropePositionAxes: ropePositionAxes
        )
    }

    public mutating func decodeSync(
        hiddenState: [Float],
        ropePositionAxes: (UInt32, UInt32, UInt32)? = nil,
        deepstackFeatures: [Int: [Float]] = [:]
    ) throws -> Int32 {
        let workspace = try resolveHiddenOverrideWorkspace(hiddenElementCount: hiddenState.count)
        let hiddenStagingBuffer = try workspace.writeHiddenState(hiddenState)
        let deepstackBuffers = try workspace.writeDeepstackFeatures(deepstackFeatures)
        return try decodeSync(
            hiddenStagingBuffer: hiddenStagingBuffer,
            hiddenElementCount: hiddenState.count,
            ropePositionAxes: ropePositionAxes,
            deepstackBuffers: deepstackBuffers
        )
    }

    public mutating func decodeSync(
        hiddenState: [Float],
        ropePositionAxes: (UInt32, UInt32, UInt32)? = nil,
        deepstackFeaturesByLayer: [Int: [[Float]]],
        tokenIndex: Int
    ) throws -> Int32 {
        let workspace = try resolveHiddenOverrideWorkspace(hiddenElementCount: hiddenState.count)
        let hiddenStagingBuffer = try workspace.writeHiddenState(hiddenState)
        let deepstackBuffers = try workspace.writeDeepstackFeatures(
            byLayer: deepstackFeaturesByLayer,
            tokenIndex: tokenIndex
        )
        return try decodeSync(
            hiddenStagingBuffer: hiddenStagingBuffer,
            hiddenElementCount: hiddenState.count,
            ropePositionAxes: ropePositionAxes,
            deepstackBuffers: deepstackBuffers
        )
    }

    public mutating func writeDecodePerLayerInputs(_ valuesByLayer: [[Float]]) throws {
        guard let buffer = decodePlan.buffers.perLayerInputs else {
            if valuesByLayer.isEmpty { return }
            throw MetalCompilerError.deviceSetupFailed("Decode plan does not allocate per-layer input storage")
        }
        let layerCount = decodePlan.buffers.perLayerInputLayerCount
        let dimension = decodePlan.buffers.perLayerInputDimension
        guard valuesByLayer.count == layerCount else {
            throw MetalCompilerError.deviceSetupFailed("Per-layer input layer count mismatch")
        }
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: layerCount * dimension)
        memset(buffer.contents(), 0, buffer.length)
        for layerIndex in 0..<layerCount {
            let values = valuesByLayer[layerIndex]
            guard values.count == dimension else {
                throw MetalCompilerError.deviceSetupFailed("Per-layer input dimension mismatch at layer \(layerIndex)")
            }
            let base = layerIndex * dimension
            values.withUnsafeBufferPointer { source in
                pointer.advanced(by: base).update(from: source.baseAddress!, count: dimension)
            }
        }
    }

    public mutating func writePrefillPerLayerInputs(_ valuesByLayer: [[[Float]]]) throws {
        guard let prefillPlan else {
            if valuesByLayer.isEmpty { return }
            throw MetalCompilerError.deviceSetupFailed("Prefill plan is required for sequence per-layer inputs")
        }
        guard let buffer = prefillPlan.buffers.perLayerInputs else {
            if valuesByLayer.isEmpty { return }
            throw MetalCompilerError.deviceSetupFailed("Prefill plan does not allocate per-layer input storage")
        }
        let layerCount = prefillPlan.buffers.perLayerInputLayerCount
        let dimension = prefillPlan.buffers.perLayerInputDimension
        let maximumSequenceLength = prefillPlan.maximumSequenceLength
        guard valuesByLayer.count == layerCount else {
            throw MetalCompilerError.deviceSetupFailed("Per-layer prefill input layer count mismatch")
        }
        let pointer = buffer.contents().bindMemory(
            to: Float.self,
            capacity: layerCount * maximumSequenceLength * dimension
        )
        memset(buffer.contents(), 0, buffer.length)
        for layerIndex in 0..<layerCount {
            let tokens = valuesByLayer[layerIndex]
            guard tokens.count <= maximumSequenceLength else {
                throw MetalCompilerError.deviceSetupFailed("Per-layer prefill input exceeds maximum sequence length")
            }
            let layerBase = layerIndex * maximumSequenceLength * dimension
            if tokens.isEmpty {
                continue
            }
            for tokenIndex in 0..<tokens.count {
                let values = tokens[tokenIndex]
                guard values.count == dimension else {
                    throw MetalCompilerError.deviceSetupFailed(
                        "Per-layer prefill input dimension mismatch at layer \(layerIndex), token \(tokenIndex)"
                    )
                }
                let base = layerBase + tokenIndex * dimension
                values.withUnsafeBufferPointer { source in
                    pointer.advanced(by: base).update(from: source.baseAddress!, count: dimension)
                }
            }
        }
    }

    // MARK: - Prefill

    /// Prefill the KV cache with prompt tokens and return the first predicted token.
    ///
    /// Returns the argmax of the prefill logits (the model's first generated token).
    /// The caller should output this token and feed it to the first decode step.
    @discardableResult
    public mutating func prefill(tokens: [Int32]) -> Int32 {
        guard let prefillPlan else {
            var lastOutput: Int32 = -1
            for token in tokens {
                lastOutput = decodeSync(tokenID: token)
            }
            return lastOutput
        }

        return prefillExecutor.prefill(
            prefillPlan: prefillPlan,
            decodePlan: decodePlan,
            submission: submission,
            position: &position,
            tokens: tokens
        )
    }

    @discardableResult
    public mutating func prefillEmbeddings(
        _ hiddenStates: [[Float]],
        ropePositionAxes: [(UInt32, UInt32, UInt32)],
        deepstackFeaturesByLayer: [Int: [[Float]]] = [:]
    ) throws -> Int32 {
        guard hiddenStates.count == ropePositionAxes.count else {
            throw MetalCompilerError.deviceSetupFailed(
                "Visual prefill embeddings and RoPE axes count mismatch"
            )
        }

        for (layerIndex, featuresByToken) in deepstackFeaturesByLayer {
            guard featuresByToken.count == hiddenStates.count else {
                throw MetalCompilerError.deviceSetupFailed(
                    "Deepstack feature count mismatch at layer \(layerIndex)"
                )
            }
        }

        guard let prefillPlan else {
            var lastOutput: Int32 = -1
            for tokenIndex in hiddenStates.indices {
                lastOutput = try decodeSync(
                    hiddenState: hiddenStates[tokenIndex],
                    ropePositionAxes: ropePositionAxes[tokenIndex],
                    deepstackFeaturesByLayer: deepstackFeaturesByLayer,
                    tokenIndex: tokenIndex
                )
            }
            return lastOutput
        }

        var hiddenOverridesByTokenIndex: [Int: [Float]] = [:]
        hiddenOverridesByTokenIndex.reserveCapacity(hiddenStates.count)
        for (index, hiddenState) in hiddenStates.enumerated() {
            hiddenOverridesByTokenIndex[index] = hiddenState
        }

        var deepstackFeaturesByLayerAndTokenIndex: [Int: [Int: [Float]]] = [:]
        for (layerIndex, featuresByToken) in deepstackFeaturesByLayer {
            var featuresByIndex: [Int: [Float]] = [:]
            featuresByIndex.reserveCapacity(featuresByToken.count)
            for (index, values) in featuresByToken.enumerated() {
                featuresByIndex[index] = values
            }
            deepstackFeaturesByLayerAndTokenIndex[layerIndex] = featuresByIndex
        }

        return try prefillExecutor.prefill(
            prefillPlan: prefillPlan,
            decodePlan: decodePlan,
            submission: submission,
            position: &position,
            tokens: [Int32](repeating: 0, count: hiddenStates.count),
            ropePositionAxesByTokenIndex: ropePositionAxes,
            hiddenOverridesByTokenIndex: hiddenOverridesByTokenIndex,
            deepstackFeaturesByLayerAndTokenIndex: deepstackFeaturesByLayerAndTokenIndex
        )
    }

    @discardableResult
    public mutating func prefill(
        tokens: [Int32],
        ropePositionAxesByTokenIndex: [(UInt32, UInt32, UInt32)],
        hiddenOverridesByTokenIndex: [Int: [Float]],
        deepstackFeaturesByLayerAndTokenIndex: [Int: [Int: [Float]]]
    ) throws -> Int32 {
        guard let prefillPlan else {
            throw MetalCompilerError.deviceSetupFailed(
                "Multimodal prefill requires a sequence prefill plan"
            )
        }
        let chunkSize = prefillPlan.maximumSequenceLength
        if tokens.count > chunkSize {
            var lastOutput: Int32 = -1
            var startIndex = 0
            while startIndex < tokens.count {
                let endIndex = min(startIndex + chunkSize, tokens.count)
                var localHiddenOverrides: [Int: [Float]] = [:]
                for (index, values) in hiddenOverridesByTokenIndex where index >= startIndex && index < endIndex {
                    localHiddenOverrides[index - startIndex] = values
                }
                var localDeepstack: [Int: [Int: [Float]]] = [:]
                for (layerIndex, featuresByTokenIndex) in deepstackFeaturesByLayerAndTokenIndex {
                    var filtered: [Int: [Float]] = [:]
                    for (index, values) in featuresByTokenIndex where index >= startIndex && index < endIndex {
                        filtered[index - startIndex] = values
                    }
                    localDeepstack[layerIndex] = filtered
                }
                lastOutput = try prefillExecutor.prefill(
                    prefillPlan: prefillPlan,
                    decodePlan: decodePlan,
                    submission: submission,
                    position: &position,
                    tokens: Array(tokens[startIndex..<endIndex]),
                    ropePositionAxesByTokenIndex: Array(ropePositionAxesByTokenIndex[startIndex..<endIndex]),
                    hiddenOverridesByTokenIndex: localHiddenOverrides,
                    deepstackFeaturesByLayerAndTokenIndex: localDeepstack
                )
                startIndex = endIndex
            }
            return lastOutput
        }
        return try prefillExecutor.prefill(
            prefillPlan: prefillPlan,
            decodePlan: decodePlan,
            submission: submission,
            position: &position,
            tokens: tokens,
            ropePositionAxesByTokenIndex: ropePositionAxesByTokenIndex,
            hiddenOverridesByTokenIndex: hiddenOverridesByTokenIndex,
            deepstackFeaturesByLayerAndTokenIndex: deepstackFeaturesByLayerAndTokenIndex
        )
    }


    // MARK: - Lifecycle

    public mutating func flush() -> Int32 {
        decodeExecutor.flush(
            plan: decodePlan,
            submission: submission,
            pendingCommandBuffer: &pendingCommandBuffer,
            hasPendingResult: &hasPendingResult
        )
    }

    public func makePromptState(firstToken: Int32) throws -> MetalPromptState {
        try promptStateStore.makePromptState(
            plan: decodePlan,
            submission: submission,
            position: position,
            firstToken: firstToken
        )
    }

    public mutating func restore(promptState: MetalPromptState) throws {
        pendingCommandBuffer = nil
        hasPendingResult = false
        try promptStateStore.restore(plan: decodePlan, submission: submission, promptState: promptState)
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
            try Self.zeroStateBuffers(decodePlan.buffers, submission: submission)
        } catch {
            print("[MetalInference] Failed to reset GPU state: \(error)")
        }
    }

    private mutating func resolveHiddenOverrideWorkspace(
        hiddenElementCount: Int
    ) throws -> MetalHiddenOverrideWorkspace {
        if let workspace = hiddenOverrideStagingWorkspace,
           workspace.hiddenElementCount == hiddenElementCount {
            return workspace
        }
        let workspace = try MetalHiddenOverrideWorkspace(
            device: commandQueue.device,
            hiddenElementCount: hiddenElementCount
        )
        hiddenOverrideStagingWorkspace = workspace
        return workspace
    }

    private mutating func decodeSync(
        hiddenStagingBuffer: MTLBuffer,
        hiddenElementCount: Int,
        ropePositionAxes: (UInt32, UInt32, UInt32)?,
        deepstackBuffers: [Int: MTLBuffer]
    ) throws -> Int32 {
        guard hiddenElementCount == decodePlan.buffers.hidden.length / decodePlan.buffers.bufferPrecision.byteSize else {
            throw MetalCompilerError.deviceSetupFailed("Hidden state override dimension mismatch")
        }

        let buffers = decodePlan.buffers
        buffers.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(position)
        let resolvedAxes = ropePositionAxes ?? (UInt32(position), UInt32(position), UInt32(position))
        let ropeAxesPointer = buffers.ropePositionAxes.contents().bindMemory(to: UInt32.self, capacity: 3)
        ropeAxesPointer[0] = resolvedAxes.0
        ropeAxesPointer[1] = resolvedAxes.1
        ropeAxesPointer[2] = resolvedAxes.2

        let copyKernelName = helperKernelName(base: "hidden_copy_from_float")
        let addKernelName = helperKernelName(base: "hidden_add_from_float")
        guard
            let copyPipeline = compiledModel.auxiliaryPipelines[copyKernelName],
            let addPipeline = compiledModel.auxiliaryPipelines[addKernelName]
        else {
            throw MetalCompilerError.kernelNotFound("\(copyKernelName) / \(addKernelName)")
        }

        _ = try submission.withTransaction(label: "decode.hidden.sync") { transaction in
            try transaction.withBlitEncoder { blit in
                blit.fill(buffer: buffers.residual, range: 0..<buffers.residual.length, value: 0)
                blit.fill(buffer: buffers.scratch, range: 0..<buffers.scratch.length, value: 0)
                blit.fill(buffer: buffers.logits, range: 0..<buffers.logits.length, value: 0)
            }
            try transaction.withComputeEncoder { encoder in
                decodeExecutor.logDebugStep("[MetalInference][debug] hidden decode copy: \(copyKernelName)")
                encodeHiddenOverride(
                    encoder: encoder,
                    pipeline: copyPipeline,
                    hidden: buffers.hidden,
                    staging: hiddenStagingBuffer,
                    count: hiddenElementCount
                )

                var injectedLayers = Set<Int>()
                for step in decodePlan.steps {
                    if step.metadata.kernelName?.hasPrefix("embedding_lookup") == true {
                        continue
                    }
                    if let layerIndex = step.metadata.layerIndex,
                       let deepstackBuffer = deepstackBuffers[layerIndex],
                       !injectedLayers.contains(layerIndex) {
                        decodeExecutor.logDebugStep("[MetalInference][debug] hidden decode add: \(addKernelName) layer=\(layerIndex)")
                        encodeHiddenOverride(
                            encoder: encoder,
                            pipeline: addPipeline,
                            hidden: buffers.hidden,
                            staging: deepstackBuffer,
                            count: hiddenElementCount
                        )
                        injectedLayers.insert(layerIndex)
                    }
                    decodeExecutor.logDebugStep(
                        "[MetalInference][debug] hidden decode step \(step.metadata.layerIndex.map(String.init) ?? "-"): \(step.metadata.kernelName ?? "(unknown)")"
                    )
                    if step.metadata.kernelName == "ssm_recurrence" || step.metadata.kernelName == "ssm_recurrence_f32" {
                        let bindingSummary = step.bufferBindings
                            .map { "\($0.index)=\($0.offset)" }
                            .joined(separator: ",")
                        decodeExecutor.logDebugStep(
                            "[MetalInference][debug] hidden decode buffers \(step.metadata.layerIndex.map(String.init) ?? "-"): \(bindingSummary)"
                        )
                    }
                    step.bindings.bind(to: encoder)
                    step.descriptor.encode(on: encoder)
                }
            }
        }

        position += 1
        return buffers.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
    }

    private func helperKernelName(base: String) -> String {
        switch decodePlan.buffers.bufferPrecision {
        case .float16:
            return base
        case .bfloat16:
            return base + "_bf16"
        case .float32:
            return base + "_f32"
        }
    }

    private func encodeHiddenOverride(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        hidden: MTLBuffer,
        staging: MTLBuffer,
        count: Int
    ) {
        var countValue = UInt32(count)
        let threadCount = min(max(1, count), pipeline.maxTotalThreadsPerThreadgroup)
        let gridSize = MTLSize(width: (count + threadCount - 1) / threadCount, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: threadCount, height: 1, depth: 1)
        encoder.memoryBarrier(scope: .buffers)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(hidden, offset: 0, index: 0)
        encoder.setBuffer(staging, offset: 0, index: 1)
        withUnsafeBytes(of: &countValue) { bytes in
            encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 2)
        }
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.memoryBarrier(scope: .buffers)
    }
}
