import Metal
import Darwin
import LMIR

/// Mutable runtime for token-by-token inference using a compiled Metal model.
public struct MetalInferenceModel: @unchecked Sendable {

    public private(set) var compiledModel: MetalCompiledModel
    public var position: Int = 0

    private var submission: MetalSubmissionContext
    private var stableResidencyRegistry: MetalStableResidencyRegistry
    private let decodeExecutor = MetalDecodeExecutor()
    private let prefillExecutor: MetalPrefillExecutor
    private let promptStateStore = MetalPromptStateStore()
    private var hiddenOverrideStagingWorkspace: MetalHiddenOverrideWorkspace?

    /// Runtime constant buffer for hidden override decode path.
    /// Layout: [hiddenElementCount: UInt32] at offset 0.
    private let hiddenOverrideConstantBuffer: MTLBuffer

    /// Decode-time dispatch plan extracted from the compiled model.
    public var decodePlan: MetalDispatchPlan { compiledModel.decodePlan }

    /// Optional sequence-oriented prefill plan paired with the decode plan.
    public var prefillPlan: MetalPrefillPlan? {
        get { compiledModel.prefillPlan }
        set {
            compiledModel = compiledModel.withPrefillPlan(newValue)
            do {
                try rebuildStableResidencyRegistry()
                if let prefillPlan = newValue {
                    try Self.zeroStateBuffers(prefillPlan.buffers, submission: &submission)
                }
            } catch {
                preconditionFailure("Failed to rebuild Metal stable residency: \(error)")
            }
        }
    }

    /// Shared runtime buffers used by decode execution.
    public var buffers: MetalBufferSet { decodePlan.buffers }

    /// The Metal device used for all allocations.
    public var device: MTLDevice { submission.device }

    /// The Metal 4 command queue used for all submissions.
    public var queue: MTL4CommandQueue { submission.queue }

#if ENABLE_METAL_PROBES
    /// Compile-time gated GPU binding probes used by focused diagnostics.
    /// Enable with `-DENABLE_METAL_PROBES` when you need to inspect live
    /// prefill bindings without turning on every DEBUG-only code path.
    public enum DebugPrefillProbePhase: Sendable {
        case beforeStep
        case afterStep
    }

    public enum DebugDecodeProbePhase: Sendable {
        case beforeStep
        case afterStep
    }

    public struct DebugPrefillBindingProbe: Sendable {
        public let label: String
        public let stepIndex: Int?
        public let bindingIndex: Int
        public let phase: DebugPrefillProbePhase
        public let rowIndex: Int?
        public let elementOffset: Int
        public let rowStride: Int
        public let count: Int
        public let precision: BufferPrecision

        public init(
            label: String,
            stepIndex: Int? = nil,
            bindingIndex: Int,
            phase: DebugPrefillProbePhase,
            rowIndex: Int? = nil,
            elementOffset: Int = 0,
            rowStride: Int,
            count: Int,
            precision: BufferPrecision = .float32
        ) {
            self.label = label
            self.stepIndex = stepIndex
            self.bindingIndex = bindingIndex
            self.phase = phase
            self.rowIndex = rowIndex
            self.elementOffset = elementOffset
            self.rowStride = rowStride
            self.count = count
            self.precision = precision
        }
    }

    public struct DebugDecodeBindingProbe: Sendable {
        public let label: String
        public let stepIndex: Int
        public let bindingIndex: Int
        public let phase: DebugDecodeProbePhase
        public let elementOffset: Int
        public let count: Int
        public let precision: BufferPrecision

        public init(
            label: String,
            stepIndex: Int,
            bindingIndex: Int,
            phase: DebugDecodeProbePhase,
            elementOffset: Int = 0,
            count: Int,
            precision: BufferPrecision
        ) {
            self.label = label
            self.stepIndex = stepIndex
            self.bindingIndex = bindingIndex
            self.phase = phase
            self.elementOffset = elementOffset
            self.count = count
            self.precision = precision
        }
    }
#endif

    public init(plan: MetalDispatchPlan, device: MTLDevice) throws {
        self.compiledModel = MetalCompiledModel(decodePlan: plan)
        self.submission = try MetalSubmissionContext(device: device)
        self.prefillExecutor = MetalPrefillExecutor()
        self.hiddenOverrideConstantBuffer = try Self.makeHiddenOverrideConstantBuffer(device: device)
        self.stableResidencyRegistry = try MetalStableResidencyRegistry(
            device: device,
            compiledModel: self.compiledModel,
            hiddenOverrideConstantBuffer: self.hiddenOverrideConstantBuffer
        )
        self.stableResidencyRegistry.register(on: self.submission.queue)
        try Self.zeroStateBuffers(plan.buffers, submission: &self.submission)
        if let prefillPlan = self.compiledModel.prefillPlan {
            try Self.zeroStateBuffers(prefillPlan.buffers, submission: &self.submission)
        }
    }

    public init(compiledModel: MetalCompiledModel, device: MTLDevice) throws {
        self.compiledModel = compiledModel
        self.submission = try MetalSubmissionContext(device: device)
        self.prefillExecutor = MetalPrefillExecutor(
            hiddenConversionPipeline: Self.resolveHiddenConversionPipeline(compiledModel),
            kvTransferPipeline: Self.resolveKVTransferPipeline(compiledModel),
            kvTransferConstantBuffer: try Self.makeKVTransferConstantBuffer(device: device)
        )
        self.hiddenOverrideConstantBuffer = try Self.makeHiddenOverrideConstantBuffer(device: device)
        self.stableResidencyRegistry = try MetalStableResidencyRegistry(
            device: device,
            compiledModel: self.compiledModel,
            hiddenOverrideConstantBuffer: self.hiddenOverrideConstantBuffer
        )
        self.stableResidencyRegistry.register(on: self.submission.queue)
        try Self.zeroStateBuffers(compiledModel.decodePlan.buffers, submission: &self.submission)
        if let prefillPlan = compiledModel.prefillPlan {
            try Self.zeroStateBuffers(prefillPlan.buffers, submission: &self.submission)
        }
    }

    public init(plan: MetalCompiledModel, device: MTLDevice) throws {
        try self.init(compiledModel: plan, device: device)
    }

    private static func makeHiddenOverrideConstantBuffer(device: MTLDevice) throws -> MTLBuffer {
        // Single UInt32 slot for element count
        guard let buffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: [.storageModeShared]) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate hidden override constant buffer")
        }
        return buffer
    }

    private static func makeKVTransferConstantBuffer(device: MTLDevice) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(
            length: MemoryLayout<KVTransferRuntimeConstants>.stride,
            options: [.storageModeShared]
        ) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate KV transfer constant buffer")
        }
        return buffer
    }

    private static func zeroStateBuffers(_ buffers: MetalBufferSet, submission: inout MetalSubmissionContext) throws {
        var fills: [(buffer: MTLBuffer, value: UInt8)] = [
            (buffers.hidden, 0),
            (buffers.residual, 0),
            (buffers.scratch, 0),
            (buffers.logits, 0),
            (buffers.position, 0),
            (buffers.ropePositionAxes, 0),
            (buffers.tokenIn, 0),
            (buffers.tokenOut, 0),
        ]
        if let kv = buffers.kvCache {
            fills.append((kv.keys, 0))
            fills.append((kv.values, 0))
            if let qjlResidualK = kv.qjlResidualK {
                fills.append((qjlResidualK, 0))
            }
        }
        if let convState = buffers.convState {
            fills.append((convState, 0))
        }
        if let recurrentState = buffers.recurrentState {
            fills.append((recurrentState, 0))
        }
        if let perLayerInputs = buffers.perLayerInputs {
            fills.append((perLayerInputs, 0))
        }
        try submission.fillBuffers(fills)
    }

    private static func zeroStateBuffers(_ buffers: PrefillBufferSet, submission: inout MetalSubmissionContext) throws {
        var fills: [(buffer: MTLBuffer, value: UInt8)] = [
            (buffers.hidden, 0),
            (buffers.residual, 0),
            (buffers.scratch, 0),
            (buffers.logits, 0),
            (buffers.tokenIDs, 0),
            (buffers.positions, 0),
            (buffers.ropePositionAxes, 0),
            (buffers.tokenOut, 0),
            (buffers.runtimeConstantBuffer, 0),
        ]
        if let kv = buffers.kvCache {
            fills.append((kv.keys, 0))
            fills.append((kv.values, 0))
            if let qjlResidualK = kv.qjlResidualK {
                fills.append((qjlResidualK, 0))
            }
        }
        if let convState = buffers.convState {
            fills.append((convState, 0))
        }
        if let recurrentState = buffers.recurrentState {
            fills.append((recurrentState, 0))
        }
        if let perLayerInputs = buffers.perLayerInputs {
            fills.append((perLayerInputs, 0))
        }
        try submission.fillBuffers(fills)
    }

    public mutating func decode(
        tokenID: Int32,
        ropePositionAxes: (UInt32, UInt32, UInt32)? = nil
    ) -> Int32 {
        decodeExecutor.decodeSync(
            plan: decodePlan,
            submission: &submission,
            position: &position,
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
            submission: &submission,
            position: &position,
            tokenID: tokenID,
            ropePositionAxes: ropePositionAxes
        )
    }

    /// Decode with GPU timing feedback for profiling.
    public mutating func decodeSyncTimed(
        tokenID: Int32,
        ropePositionAxes: (UInt32, UInt32, UInt32)? = nil
    ) -> (token: Int32, gpuStartTime: CFTimeInterval, gpuEndTime: CFTimeInterval) {
        decodeExecutor.decodeSyncTimed(
            plan: decodePlan,
            submission: &submission,
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
            deepstackBuffers: deepstackBuffers,
            ephemeralResidency: workspace.residencyLease
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
            deepstackBuffers: deepstackBuffers,
            ephemeralResidency: workspace.residencyLease
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
            submission: &submission,
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
            submission: &submission,
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
                    submission: &submission,
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
            submission: &submission,
            position: &position,
            tokens: tokens,
            ropePositionAxesByTokenIndex: ropePositionAxesByTokenIndex,
            hiddenOverridesByTokenIndex: hiddenOverridesByTokenIndex,
            deepstackFeaturesByLayerAndTokenIndex: deepstackFeaturesByLayerAndTokenIndex
        )
    }

    public mutating func debugPrefillLastTokenHiddenSnapshots(
        tokens: [Int32],
        stepIndices: Set<Int>,
        prefillPerLayerInputs: [[[Float]]]? = nil
    ) throws -> [Int: [Float]] {
        guard let prefillPlan else { return [:] }
        guard !tokens.isEmpty else { return [:] }
        guard tokens.count <= prefillPlan.maximumSequenceLength else {
            throw MetalCompilerError.deviceSetupFailed("Debug prefill token count exceeds maximum sequence length")
        }
        resetState()
        if let prefillPerLayerInputs {
            try writePrefillPerLayerInputs(prefillPerLayerInputs)
        }
        return try prefillExecutor.captureLastTokenHiddenSnapshots(
            prefillPlan: prefillPlan,
            submission: &submission,
            position: position,
            tokens: tokens,
            stepIndices: stepIndices
        )
    }

    public mutating func debugPrefillLastTokenFinalHidden(
        tokens: [Int32]
    ) throws -> [Float] {
        guard let prefillPlan else { return [] }
        guard !tokens.isEmpty else { return [] }
        guard tokens.count <= prefillPlan.maximumSequenceLength else {
            throw MetalCompilerError.deviceSetupFailed("Debug prefill token count exceeds maximum sequence length")
        }

        resetState()
        return try prefillExecutor.captureLastTokenFinalHidden(
            prefillPlan: prefillPlan,
            submission: &submission,
            position: position,
            tokens: tokens
        )
    }

    public mutating func debugPrefillLastTokenResidualSnapshots(
        tokens: [Int32],
        stepIndices: Set<Int>
    ) throws -> [Int: [Float]] {
        guard let prefillPlan else { return [:] }
        guard !tokens.isEmpty else { return [:] }
        guard tokens.count <= prefillPlan.maximumSequenceLength else {
            throw MetalCompilerError.deviceSetupFailed("Debug prefill token count exceeds maximum sequence length")
        }
        resetState()
        return try prefillExecutor.captureLastTokenResidualSnapshots(
            prefillPlan: prefillPlan,
            submission: &submission,
            position: position,
            tokens: tokens,
            stepIndices: stepIndices
        )
    }

    public mutating func debugPrefillLastTokenResidualSnapshots(
        hiddenStates: [[Float]],
        ropePositionAxes: [(UInt32, UInt32, UInt32)],
        stepIndices: Set<Int>,
        prefillPerLayerInputs: [[[Float]]]? = nil
    ) throws -> [Int: [Float]] {
        guard let prefillPlan else { return [:] }
        guard !hiddenStates.isEmpty else { return [:] }
        guard hiddenStates.count == ropePositionAxes.count else {
            throw MetalCompilerError.deviceSetupFailed(
                "Debug prefill embeddings and RoPE axes count mismatch"
            )
        }
        guard hiddenStates.count <= prefillPlan.maximumSequenceLength else {
            throw MetalCompilerError.deviceSetupFailed(
                "Debug prefill embedding count exceeds maximum sequence length"
            )
        }

        var hiddenOverridesByTokenIndex: [Int: [Float]] = [:]
        hiddenOverridesByTokenIndex.reserveCapacity(hiddenStates.count)
        for (index, hiddenState) in hiddenStates.enumerated() {
            hiddenOverridesByTokenIndex[index] = hiddenState
        }

        resetState()
        if let prefillPerLayerInputs {
            try writePrefillPerLayerInputs(prefillPerLayerInputs)
        }
        return try prefillExecutor.captureLastTokenResidualSnapshots(
            prefillPlan: prefillPlan,
            submission: &submission,
            position: position,
            tokens: [Int32](repeating: 0, count: hiddenStates.count),
            ropePositionAxesByTokenIndex: ropePositionAxes,
            hiddenOverridesByTokenIndex: hiddenOverridesByTokenIndex,
            stepIndices: stepIndices
        )
    }

    public mutating func debugPrefillLastTokenHiddenSnapshots(
        hiddenStates: [[Float]],
        ropePositionAxes: [(UInt32, UInt32, UInt32)],
        stepIndices: Set<Int>,
        prefillPerLayerInputs: [[[Float]]]? = nil
    ) throws -> [Int: [Float]] {
        guard let prefillPlan else { return [:] }
        guard !hiddenStates.isEmpty else { return [:] }
        guard hiddenStates.count == ropePositionAxes.count else {
            throw MetalCompilerError.deviceSetupFailed(
                "Debug prefill embeddings and RoPE axes count mismatch"
            )
        }
        guard hiddenStates.count <= prefillPlan.maximumSequenceLength else {
            throw MetalCompilerError.deviceSetupFailed(
                "Debug prefill embedding count exceeds maximum sequence length"
            )
        }
        var hiddenOverridesByTokenIndex: [Int: [Float]] = [:]
        hiddenOverridesByTokenIndex.reserveCapacity(hiddenStates.count)
        for (index, hiddenState) in hiddenStates.enumerated() {
            hiddenOverridesByTokenIndex[index] = hiddenState
        }
        resetState()
        if let prefillPerLayerInputs {
            try writePrefillPerLayerInputs(prefillPerLayerInputs)
        }
        return try prefillExecutor.captureLastTokenHiddenSnapshots(
            prefillPlan: prefillPlan,
            submission: &submission,
            position: position,
            tokens: [Int32](repeating: 0, count: hiddenStates.count),
            ropePositionAxesByTokenIndex: ropePositionAxes,
            hiddenOverridesByTokenIndex: hiddenOverridesByTokenIndex,
            stepIndices: stepIndices
        )
    }

    public mutating func debugPrefillLastTokenScratchSnapshots(
        tokens: [Int32],
        stepIndices: Set<Int>,
        slotIndex: Int,
        rowStride: Int,
        count: Int,
        prefillPerLayerInputs: [[[Float]]]? = nil
    ) throws -> [Int: [Float]] {
        guard let prefillPlan else { return [:] }
        guard !tokens.isEmpty else { return [:] }
        guard tokens.count <= prefillPlan.maximumSequenceLength else {
            throw MetalCompilerError.deviceSetupFailed("Debug prefill token count exceeds maximum sequence length")
        }
        guard count > 0 else { return [:] }
        guard rowStride > 0 else {
            throw MetalCompilerError.deviceSetupFailed("Debug scratch row stride must be positive")
        }
        resetState()
        if let prefillPerLayerInputs {
            try writePrefillPerLayerInputs(prefillPerLayerInputs)
        }
        return try prefillExecutor.captureLastTokenScratchSnapshots(
            prefillPlan: prefillPlan,
            submission: &submission,
            position: position,
            tokens: tokens,
            stepIndices: stepIndices,
            slotIndex: slotIndex,
            rowStride: rowStride,
            count: count
        )
    }

    public mutating func debugPrefillLastTokenScratchSnapshots(
        hiddenStates: [[Float]],
        ropePositionAxes: [(UInt32, UInt32, UInt32)],
        stepIndices: Set<Int>,
        slotIndex: Int,
        rowStride: Int,
        count: Int,
        prefillPerLayerInputs: [[[Float]]]? = nil
    ) throws -> [Int: [Float]] {
        guard let prefillPlan else { return [:] }
        guard !hiddenStates.isEmpty else { return [:] }
        guard hiddenStates.count == ropePositionAxes.count else {
            throw MetalCompilerError.deviceSetupFailed(
                "Debug prefill embeddings and RoPE axes count mismatch"
            )
        }
        guard hiddenStates.count <= prefillPlan.maximumSequenceLength else {
            throw MetalCompilerError.deviceSetupFailed(
                "Debug prefill embedding count exceeds maximum sequence length"
            )
        }
        guard count > 0 else { return [:] }
        guard rowStride > 0 else {
            throw MetalCompilerError.deviceSetupFailed("Debug scratch row stride must be positive")
        }

        var hiddenOverridesByTokenIndex: [Int: [Float]] = [:]
        hiddenOverridesByTokenIndex.reserveCapacity(hiddenStates.count)
        for (index, hiddenState) in hiddenStates.enumerated() {
            hiddenOverridesByTokenIndex[index] = hiddenState
        }

        resetState()
        if let prefillPerLayerInputs {
            try writePrefillPerLayerInputs(prefillPerLayerInputs)
        }
        return try prefillExecutor.captureLastTokenScratchSnapshots(
            prefillPlan: prefillPlan,
            submission: &submission,
            position: position,
            tokens: [Int32](repeating: 0, count: hiddenStates.count),
            ropePositionAxesByTokenIndex: ropePositionAxes,
            hiddenOverridesByTokenIndex: hiddenOverridesByTokenIndex,
            stepIndices: stepIndices,
            slotIndex: slotIndex,
            rowStride: rowStride,
            count: count
        )
    }

    public mutating func debugPrefillScratchRows(
        tokens: [Int32],
        stepIndex: Int,
        slotIndex: Int,
        rowStride: Int,
        rowIndices: [Int],
        count: Int
    ) throws -> [Int: [Float]] {
        guard let prefillPlan else { return [:] }
        guard !tokens.isEmpty else { return [:] }
        guard tokens.count <= prefillPlan.maximumSequenceLength else {
            throw MetalCompilerError.deviceSetupFailed("Debug prefill token count exceeds maximum sequence length")
        }
        guard stepIndex >= 0, stepIndex < prefillPlan.steps.count else {
            throw MetalCompilerError.deviceSetupFailed("Debug step index out of range")
        }
        guard count > 0 else { return [:] }
        guard rowStride > 0 else {
            throw MetalCompilerError.deviceSetupFailed("Debug scratch row stride must be positive")
        }

        let slotDimension = prefillPlan.slotDimension
        guard slotIndex >= 0, slotIndex * slotDimension < prefillPlan.buffers.scratch.length / MemoryLayout<Float>.stride else {
            throw MetalCompilerError.deviceSetupFailed("Debug scratch slot index out of range")
        }

        let clampedCount = min(count, rowStride)
        let stagingLength = clampedCount * MemoryLayout<Float>.stride
        guard let stagingBuffer = submission.device.makeBuffer(length: stagingLength, options: [.storageModeShared]) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate debug scratch staging buffer")
        }

        resetState()
        Self.populatePrefillInputs(
            prefillPlan: prefillPlan,
            position: position,
            tokens: tokens,
            ropePositionAxesByTokenIndex: nil
        )
        Self.writePrefillRuntimeConstants(
            prefillPlan: prefillPlan,
            basePosition: position,
            sequenceLength: tokens.count,
            hiddenConversionElementCount: 0
        )
        try submission.withCompute { encoder, argumentTable in
            for currentStepIndex in 0...stepIndex {
                Self.encodeDebugPrefillStep(
                    prefillPlan.steps[currentStepIndex],
                    encoder: encoder,
                    argumentTable: argumentTable,
                    runtimeConstantBuffer: prefillPlan.buffers.runtimeConstantBuffer,
                    sequenceLength: tokens.count
                )
            }
        }

        var snapshots: [Int: [Float]] = [:]
        snapshots.reserveCapacity(rowIndices.count)
        for rowIndex in rowIndices {
            guard rowIndex >= 0, rowIndex < tokens.count else { continue }
            let sourceOffset = Self.prefillScratchRowOffset(
                prefillPlan: prefillPlan,
                rowIndex: rowIndex,
                slotIndex: slotIndex,
                rowStride: rowStride
            )
            try submission.copyBuffers([
                (
                    from: prefillPlan.buffers.scratch,
                    sourceOffset: sourceOffset,
                    to: stagingBuffer,
                    destinationOffset: 0,
                    size: stagingLength
                )
            ])
            snapshots[rowIndex] = Self.readSharedFloatBuffer(stagingBuffer, count: clampedCount)
        }
        return snapshots
    }

    public mutating func debugPrefillLastTokenBufferSnapshot(
        tokens: [Int32],
        stepIndex: Int,
        buffer: MTLBuffer,
        baseOffset: Int,
        rowStride: Int,
        count: Int,
        precision: BufferPrecision = .float32
    ) throws -> [Float] {
        guard let prefillPlan else { return [] }
        guard !tokens.isEmpty else { return [] }
        guard tokens.count <= prefillPlan.maximumSequenceLength else {
            throw MetalCompilerError.deviceSetupFailed("Debug prefill token count exceeds maximum sequence length")
        }
        guard stepIndex >= 0, stepIndex < prefillPlan.steps.count else {
            throw MetalCompilerError.deviceSetupFailed("Debug step index out of range")
        }
        guard rowStride > 0 else {
            throw MetalCompilerError.deviceSetupFailed("Debug row stride must be positive")
        }
        guard count > 0 else { return [] }

        let clampedCount = min(count, rowStride)
        let elementSize = max(precision.byteSize, 1)
        let stagingLength = clampedCount * elementSize
        guard let stagingBuffer = submission.device.makeBuffer(length: stagingLength, options: [.storageModeShared]) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate debug staging buffer")
        }

        resetState()
        Self.populatePrefillInputs(
            prefillPlan: prefillPlan,
            position: position,
            tokens: tokens,
            ropePositionAxesByTokenIndex: nil
        )
        Self.writePrefillRuntimeConstants(
            prefillPlan: prefillPlan,
            basePosition: position,
            sequenceLength: tokens.count,
            hiddenConversionElementCount: 0
        )
        for passRange in prefillPassRanges(for: prefillPlan.steps, within: 0..<(stepIndex + 1)) {
            try submission.withCompute { encoder, argumentTable in
                for currentStepIndex in passRange {
                    Self.encodeDebugPrefillStep(
                        prefillPlan.steps[currentStepIndex],
                        encoder: encoder,
                        argumentTable: argumentTable,
                        runtimeConstantBuffer: prefillPlan.buffers.runtimeConstantBuffer,
                        sequenceLength: tokens.count
                    )
                }
            }
        }

        let sourceOffset = baseOffset + (tokens.count - 1) * rowStride * elementSize
        try submission.copyBuffers([
            (
                from: buffer,
                sourceOffset: sourceOffset,
                to: stagingBuffer,
                destinationOffset: 0,
                size: stagingLength
            )
        ])
        return Self.readBuffer(stagingBuffer, precision: precision, count: clampedCount)
    }

#if ENABLE_METAL_PROBES
    mutating func debugPrefillLastTokenBufferSnapshotChunked(
        tokens: [Int32],
        stepIndex: Int,
        buffer: MTLBuffer,
        baseOffset: Int,
        rowStride: Int,
        count: Int,
        chunkElementCount: Int = 128,
        precision: BufferPrecision = .float32,
        visibilityOptions: MTL4VisibilityOptions = []
    ) throws -> [Float] {
        guard let prefillPlan else { return [] }
        guard !tokens.isEmpty else { return [] }
        guard tokens.count <= prefillPlan.maximumSequenceLength else {
            throw MetalCompilerError.deviceSetupFailed("Debug prefill token count exceeds maximum sequence length")
        }
        guard stepIndex >= 0, stepIndex < prefillPlan.steps.count else {
            throw MetalCompilerError.deviceSetupFailed("Debug step index out of range")
        }
        guard rowStride > 0 else {
            throw MetalCompilerError.deviceSetupFailed("Debug row stride must be positive")
        }
        guard count > 0 else { return [] }
        guard chunkElementCount > 0 else {
            throw MetalCompilerError.deviceSetupFailed("Debug chunk element count must be positive")
        }

        let clampedCount = min(count, rowStride)
        let elementSize = max(precision.byteSize, 1)
        let stagingLength = clampedCount * elementSize
        guard let stagingBuffer = submission.device.makeBuffer(length: stagingLength, options: [.storageModeShared]) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate chunked debug staging buffer")
        }
        stagingBuffer.contents().initializeMemory(
            as: UInt8.self,
            repeating: 0,
            count: stagingLength
        )

        resetState()
        Self.populatePrefillInputs(
            prefillPlan: prefillPlan,
            position: position,
            tokens: tokens,
            ropePositionAxesByTokenIndex: nil
        )
        Self.writePrefillRuntimeConstants(
            prefillPlan: prefillPlan,
            basePosition: position,
            sequenceLength: tokens.count,
            hiddenConversionElementCount: 0
        )

        let sourceBaseOffset = baseOffset + (tokens.count - 1) * rowStride * elementSize
        try submission.withCompute { encoder, argumentTable in
            for currentStepIndex in 0...stepIndex {
                Self.encodeDebugPrefillStep(
                    prefillPlan.steps[currentStepIndex],
                    encoder: encoder,
                    argumentTable: argumentTable,
                    runtimeConstantBuffer: prefillPlan.buffers.runtimeConstantBuffer,
                    sequenceLength: tokens.count,
                    visibilityOptions: visibilityOptions
                )
            }
            encoder.barrier(
                afterEncoderStages: .dispatch,
                beforeEncoderStages: .blit,
                visibilityOptions: resolvedMetalVisibilityOptions(visibilityOptions)
            )
            for elementOffset in stride(from: 0, to: clampedCount, by: chunkElementCount) {
                let chunkCount = min(chunkElementCount, clampedCount - elementOffset)
                let byteOffset = elementOffset * elementSize
                encoder.copy(
                    sourceBuffer: buffer,
                    sourceOffset: sourceBaseOffset + byteOffset,
                    destinationBuffer: stagingBuffer,
                    destinationOffset: byteOffset,
                    size: chunkCount * elementSize
                )
            }
            encoder.barrier(
                afterEncoderStages: .blit,
                beforeEncoderStages: .dispatch,
                visibilityOptions: resolvedMetalVisibilityOptions(visibilityOptions)
            )
        }

        return Self.readBuffer(stagingBuffer, precision: precision, count: clampedCount)
    }

    mutating func debugPrefillLastTokenBufferSnapshotSplitPass(
        tokens: [Int32],
        prefixThroughStepIndex: Int,
        isolatedStepIndex: Int,
        buffer: MTLBuffer,
        baseOffset: Int,
        rowStride: Int,
        count: Int,
        precision: BufferPrecision = .float32,
        visibilityOptions: MTL4VisibilityOptions = []
    ) throws -> [Float] {
        guard let prefillPlan else { return [] }
        guard !tokens.isEmpty else { return [] }
        guard tokens.count <= prefillPlan.maximumSequenceLength else {
            throw MetalCompilerError.deviceSetupFailed("Debug prefill token count exceeds maximum sequence length")
        }
        guard isolatedStepIndex >= 0, isolatedStepIndex < prefillPlan.steps.count else {
            throw MetalCompilerError.deviceSetupFailed("Debug isolated step index out of range")
        }
        guard prefixThroughStepIndex < isolatedStepIndex else {
            throw MetalCompilerError.deviceSetupFailed("Debug prefix must end before isolated step")
        }
        guard rowStride > 0 else {
            throw MetalCompilerError.deviceSetupFailed("Debug row stride must be positive")
        }
        guard count > 0 else { return [] }

        let clampedCount = min(count, rowStride)
        let elementSize = max(precision.byteSize, 1)
        let stagingLength = clampedCount * elementSize
        guard let stagingBuffer = submission.device.makeBuffer(length: stagingLength, options: [.storageModeShared]) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate split-pass debug staging buffer")
        }

        resetState()
        Self.populatePrefillInputs(
            prefillPlan: prefillPlan,
            position: position,
            tokens: tokens,
            ropePositionAxesByTokenIndex: nil
        )
        Self.writePrefillRuntimeConstants(
            prefillPlan: prefillPlan,
            basePosition: position,
            sequenceLength: tokens.count,
            hiddenConversionElementCount: 0
        )

        if prefixThroughStepIndex >= 0 {
            try submission.withCompute { encoder, argumentTable in
                for currentStepIndex in 0...prefixThroughStepIndex {
                    Self.encodeDebugPrefillStep(
                        prefillPlan.steps[currentStepIndex],
                        encoder: encoder,
                        argumentTable: argumentTable,
                        runtimeConstantBuffer: prefillPlan.buffers.runtimeConstantBuffer,
                        sequenceLength: tokens.count,
                        visibilityOptions: visibilityOptions
                    )
                }
            }
        }

        try submission.withCompute { encoder, argumentTable in
            Self.encodeDebugPrefillStep(
                prefillPlan.steps[isolatedStepIndex],
                encoder: encoder,
                argumentTable: argumentTable,
                runtimeConstantBuffer: prefillPlan.buffers.runtimeConstantBuffer,
                sequenceLength: tokens.count,
                visibilityOptions: visibilityOptions
            )
        }

        let sourceOffset = baseOffset + (tokens.count - 1) * rowStride * elementSize
        try submission.copyBuffers([
            (
                from: buffer,
                sourceOffset: sourceOffset,
                to: stagingBuffer,
                destinationOffset: 0,
                size: stagingLength
            )
        ])
        return Self.readBuffer(stagingBuffer, precision: precision, count: clampedCount)
    }

    mutating func debugPrefillLastTokenBufferSnapshotManualDispatch(
        tokens: [Int32],
        prefixThroughStepIndex: Int,
        pipeline: MTLComputePipelineState,
        gridSize: MTLSize,
        threadgroupSize: MTLSize,
        threadgroupMemoryLength: Int,
        bufferBindings: [(index: Int, buffer: MTLBuffer, offset: Int)],
        bytesBindings: [(index: Int, value: [UInt8])],
        runtimeSequenceLengthBindingIndex: Int?,
        outputBuffer: MTLBuffer,
        outputBaseOffset: Int,
        outputRowStride: Int,
        count: Int,
        precision: BufferPrecision = .float32,
        visibilityOptions: MTL4VisibilityOptions = []
    ) throws -> [Float] {
        guard let prefillPlan else { return [] }
        guard !tokens.isEmpty else { return [] }
        guard tokens.count <= prefillPlan.maximumSequenceLength else {
            throw MetalCompilerError.deviceSetupFailed("Debug prefill token count exceeds maximum sequence length")
        }
        guard outputRowStride > 0 else {
            throw MetalCompilerError.deviceSetupFailed("Debug output row stride must be positive")
        }
        guard count > 0 else { return [] }

        let clampedCount = min(count, outputRowStride)
        let elementSize = max(precision.byteSize, 1)
        let stagingLength = clampedCount * elementSize
        guard let stagingBuffer = submission.device.makeBuffer(length: stagingLength, options: [.storageModeShared]) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate manual-dispatch debug staging buffer")
        }

        let constantAllocator = MetalConstantBindingAllocator(device: submission.device)
        let bindingTable = try constantAllocator.makeBindingTable(
            bufferBindings: bufferBindings,
            bytesBindings: bytesBindings,
            argumentPolicy: .inlineBindings
        )
        let ephemeralResidency = try MetalResidencyLease.required(
            device: submission.device,
            label: "swift-lm.debug.manual-prefill-dispatch",
            buffers: bindingTable.ownedResidencyBuffers
        )

        resetState()
        Self.populatePrefillInputs(
            prefillPlan: prefillPlan,
            position: position,
            tokens: tokens,
            ropePositionAxesByTokenIndex: nil
        )
        Self.writePrefillRuntimeConstants(
            prefillPlan: prefillPlan,
            basePosition: position,
            sequenceLength: tokens.count,
            hiddenConversionElementCount: 0
        )

        if prefixThroughStepIndex >= 0 {
            try submission.withCompute { encoder, argumentTable in
                for currentStepIndex in 0...prefixThroughStepIndex {
                    Self.encodeDebugPrefillStep(
                        prefillPlan.steps[currentStepIndex],
                        encoder: encoder,
                        argumentTable: argumentTable,
                        runtimeConstantBuffer: prefillPlan.buffers.runtimeConstantBuffer,
                        sequenceLength: tokens.count,
                        visibilityOptions: visibilityOptions
                    )
                }
            }
        }

        try submission.withCompute(ephemeralResidency: ephemeralResidency) { encoder, argumentTable in
            bindingTable.bind(to: argumentTable)
            if let runtimeSequenceLengthBindingIndex {
                argumentTable.setAddress(
                    prefillPlan.buffers.runtimeConstantBuffer.gpuAddress
                        + UInt64(PrefillBufferSet.sequenceLengthOffset),
                    index: runtimeSequenceLengthBindingIndex
                )
            }
            encoder.setArgumentTable(argumentTable)
            encoder.setComputePipelineState(pipeline)
            if threadgroupMemoryLength > 0 {
                encoder.setThreadgroupMemoryLength(threadgroupMemoryLength, index: 0)
            }
            encoder.dispatchThreadgroups(
                threadgroupsPerGrid: gridSize,
                threadsPerThreadgroup: threadgroupSize
            )
        }

        let sourceOffset = outputBaseOffset + (tokens.count - 1) * outputRowStride * elementSize
        try submission.copyBuffers([
            (
                from: outputBuffer,
                sourceOffset: sourceOffset,
                to: stagingBuffer,
                destinationOffset: 0,
                size: stagingLength
            )
        ])
        return Self.readBuffer(stagingBuffer, precision: precision, count: clampedCount)
    }

    public mutating func debugPrefillBindingProbes(
        tokens: [Int32],
        stepIndex: Int,
        probes: [DebugPrefillBindingProbe],
        isolatedSubmission: Bool = true,
        visibilityOptions: MTL4VisibilityOptions = [],
        stepVisibilityOptions: MTL4VisibilityOptions? = nil,
        probeVisibilityOptions: MTL4VisibilityOptions? = nil
    ) throws -> [String: [Float]] {
        if isolatedSubmission {
            var replaySubmission = try submission.makeReplayContext()
            return try debugPrefillBindingProbes(
                using: &replaySubmission,
                tokens: tokens,
                stepIndex: stepIndex,
                probes: probes,
                visibilityOptions: visibilityOptions,
                stepVisibilityOptions: stepVisibilityOptions,
                probeVisibilityOptions: probeVisibilityOptions
            )
        }
        var sharedSubmission = submission
        let snapshots = try debugPrefillBindingProbes(
            using: &sharedSubmission,
            tokens: tokens,
            stepIndex: stepIndex,
            probes: probes,
            visibilityOptions: visibilityOptions,
            stepVisibilityOptions: stepVisibilityOptions,
            probeVisibilityOptions: probeVisibilityOptions
        )
        submission = sharedSubmission
        return snapshots
    }

    public mutating func debugPrefillBindingProbes(
        hiddenStates: [[Float]],
        ropePositionAxes: [(UInt32, UInt32, UInt32)],
        stepIndex: Int,
        probes: [DebugPrefillBindingProbe],
        prefillPerLayerInputs: [[[Float]]]? = nil,
        isolatedSubmission: Bool = true,
        visibilityOptions: MTL4VisibilityOptions = [],
        stepVisibilityOptions: MTL4VisibilityOptions? = nil,
        probeVisibilityOptions: MTL4VisibilityOptions? = nil
    ) throws -> [String: [Float]] {
        guard let prefillPlan else { return [:] }
        guard !hiddenStates.isEmpty else { return [:] }
        guard hiddenStates.count == ropePositionAxes.count else {
            throw MetalCompilerError.deviceSetupFailed(
                "Debug prefill embeddings and RoPE axes count mismatch"
            )
        }
        guard hiddenStates.count <= prefillPlan.maximumSequenceLength else {
            throw MetalCompilerError.deviceSetupFailed(
                "Debug prefill embedding count exceeds maximum sequence length"
            )
        }

        var hiddenOverridesByTokenIndex: [Int: [Float]] = [:]
        hiddenOverridesByTokenIndex.reserveCapacity(hiddenStates.count)
        for (index, hiddenState) in hiddenStates.enumerated() {
            hiddenOverridesByTokenIndex[index] = hiddenState
        }
        let tokens = [Int32](repeating: 0, count: hiddenStates.count)

        if isolatedSubmission {
            var replaySubmission = try submission.makeReplayContext()
            return try debugPrefillBindingProbes(
                using: &replaySubmission,
                tokens: tokens,
                ropePositionAxesByTokenIndex: ropePositionAxes,
                hiddenOverridesByTokenIndex: hiddenOverridesByTokenIndex,
                prefillPerLayerInputs: prefillPerLayerInputs,
                stepIndex: stepIndex,
                probes: probes,
                visibilityOptions: visibilityOptions,
                stepVisibilityOptions: stepVisibilityOptions,
                probeVisibilityOptions: probeVisibilityOptions
            )
        }

        var sharedSubmission = submission
        let snapshots = try debugPrefillBindingProbes(
            using: &sharedSubmission,
            tokens: tokens,
            ropePositionAxesByTokenIndex: ropePositionAxes,
            hiddenOverridesByTokenIndex: hiddenOverridesByTokenIndex,
            prefillPerLayerInputs: prefillPerLayerInputs,
            stepIndex: stepIndex,
            probes: probes,
            visibilityOptions: visibilityOptions,
            stepVisibilityOptions: stepVisibilityOptions,
            probeVisibilityOptions: probeVisibilityOptions
        )
        submission = sharedSubmission
        return snapshots
    }

    public mutating func debugActualPrefillBindingProbes(
        hiddenStates: [[Float]],
        ropePositionAxes: [(UInt32, UInt32, UInt32)],
        stepIndex: Int,
        probes: [DebugPrefillBindingProbe],
        prefillPerLayerInputs: [[[Float]]]? = nil,
        visibilityOptions: MTL4VisibilityOptions = []
    ) throws -> [String: [Float]] {
        guard let prefillPlan else { return [:] }
        guard hiddenStates.count == ropePositionAxes.count else {
            throw MetalCompilerError.deviceSetupFailed(
                "Debug prefill embeddings and RoPE axes count mismatch"
            )
        }
        var hiddenOverridesByTokenIndex: [Int: [Float]] = [:]
        hiddenOverridesByTokenIndex.reserveCapacity(hiddenStates.count)
        for (index, hiddenState) in hiddenStates.enumerated() {
            hiddenOverridesByTokenIndex[index] = hiddenState
        }

        var replaySubmission = try submission.makeReplayContext()
        resetState(using: &replaySubmission)
        if let prefillPerLayerInputs {
            try writePrefillPerLayerInputs(prefillPerLayerInputs)
        }
        return try prefillExecutor.captureBindingProbes(
            prefillPlan: prefillPlan,
            submission: &replaySubmission,
            position: position,
            tokens: [Int32](repeating: 0, count: hiddenStates.count),
            ropePositionAxesByTokenIndex: ropePositionAxes,
            hiddenOverridesByTokenIndex: hiddenOverridesByTokenIndex,
            probes: probes,
            visibilityOptions: visibilityOptions
        )
    }

    private mutating func debugPrefillBindingProbes(
        using probeSubmission: inout MetalSubmissionContext,
        tokens: [Int32],
        ropePositionAxesByTokenIndex: [(UInt32, UInt32, UInt32)]? = nil,
        hiddenOverridesByTokenIndex: [Int: [Float]] = [:],
        prefillPerLayerInputs: [[[Float]]]? = nil,
        stepIndex: Int,
        probes: [DebugPrefillBindingProbe],
        visibilityOptions: MTL4VisibilityOptions = [],
        stepVisibilityOptions: MTL4VisibilityOptions? = nil,
        probeVisibilityOptions: MTL4VisibilityOptions? = nil
    ) throws -> [String: [Float]] {
        guard let prefillPlan else { return [:] }
        guard !tokens.isEmpty else { return [:] }
        guard tokens.count <= prefillPlan.maximumSequenceLength else {
            throw MetalCompilerError.deviceSetupFailed("Debug prefill token count exceeds maximum sequence length")
        }
        guard stepIndex >= 0, stepIndex < prefillPlan.steps.count else {
            throw MetalCompilerError.deviceSetupFailed("Debug step index out of range")
        }
        guard !probes.isEmpty else { return [:] }
        var stagingBuffers: [String: (buffer: MTLBuffer, precision: BufferPrecision, count: Int)] = [:]
        stagingBuffers.reserveCapacity(probes.count)
        let effectiveStepIndices = probes.map { $0.stepIndex ?? stepIndex }
        guard let maximumStepIndex = effectiveStepIndices.max() else {
            return [:]
        }
        guard effectiveStepIndices.allSatisfy({ $0 >= 0 && $0 < prefillPlan.steps.count }) else {
            throw MetalCompilerError.deviceSetupFailed("Debug probe step index out of range")
        }
        for probe in probes {
            guard probe.rowStride > 0 else {
                throw MetalCompilerError.deviceSetupFailed("Debug probe row stride must be positive")
            }
            guard probe.count > 0 else {
                throw MetalCompilerError.deviceSetupFailed("Debug probe count must be positive")
            }
            let clampedCount = min(probe.count, probe.rowStride)
            let stagingLength = clampedCount * probe.precision.byteSize
            guard let stagingBuffer = probeSubmission.device.makeBuffer(length: stagingLength, options: [.storageModeShared]) else {
                throw MetalCompilerError.deviceSetupFailed("Cannot allocate debug probe staging buffer")
            }
            stagingBuffers[probe.label] = (stagingBuffer, probe.precision, clampedCount)
        }
        let resolvedStepVisibilityOptions = stepVisibilityOptions ?? visibilityOptions
        let resolvedProbeVisibilityOptions = probeVisibilityOptions ?? visibilityOptions

        resetState(using: &probeSubmission)
        if let prefillPerLayerInputs {
            try writePrefillPerLayerInputs(prefillPerLayerInputs)
        }
        Self.populatePrefillInputs(
            prefillPlan: prefillPlan,
            position: position,
            tokens: tokens,
            ropePositionAxesByTokenIndex: ropePositionAxesByTokenIndex
        )
        Self.writePrefillRuntimeConstants(
            prefillPlan: prefillPlan,
            basePosition: position,
            sequenceLength: tokens.count,
            hiddenConversionElementCount: 0
        )

        let lastTokenIndex = tokens.count - 1
        let hiddenOverrideReplayStart = prefillHiddenOverrideReplayStartStepIndex(in: prefillPlan.steps)
        let allTokensOverridden = hiddenOverridesByTokenIndex.count >= tokens.count
        var replayStartStepIndex = 0

        if hiddenOverrideReplayStart > 0 && !allTokensOverridden {
            let embeddingReplayEnd = min(hiddenOverrideReplayStart - 1, maximumStepIndex)
            if embeddingReplayEnd >= 0 {
                try probeSubmission.withCompute { encoder, argumentTable in
                    for currentStepIndex in 0...embeddingReplayEnd {
                        let currentStep = prefillPlan.steps[currentStepIndex]
                        let currentProbes = probes.filter { ($0.stepIndex ?? stepIndex) == currentStepIndex }
                        if !currentProbes.isEmpty {
                            try Self.encodePrefillProbes(
                                currentProbes.filter { $0.phase == .beforeStep },
                                step: currentStep,
                                lastTokenIndex: lastTokenIndex,
                                stagingBuffers: stagingBuffers,
                                encoder: encoder,
                                visibilityOptions: resolvedProbeVisibilityOptions
                            )
                        }
                        Self.encodeDebugPrefillStep(
                            currentStep,
                            encoder: encoder,
                            argumentTable: argumentTable,
                            runtimeConstantBuffer: prefillPlan.buffers.runtimeConstantBuffer,
                            sequenceLength: tokens.count,
                            visibilityOptions: resolvedStepVisibilityOptions
                        )
                        if !currentProbes.isEmpty {
                            try Self.encodePrefillProbes(
                                currentProbes.filter { $0.phase == .afterStep },
                                step: currentStep,
                                lastTokenIndex: lastTokenIndex,
                                stagingBuffers: stagingBuffers,
                                encoder: encoder,
                                visibilityOptions: resolvedProbeVisibilityOptions
                            )
                        }
                    }
                }
            }
            replayStartStepIndex = min(hiddenOverrideReplayStart, maximumStepIndex + 1)
        } else if allTokensOverridden {
            replayStartStepIndex = min(hiddenOverrideReplayStart, maximumStepIndex + 1)
        }

        if !hiddenOverridesByTokenIndex.isEmpty {
            try Self.overwritePrefillHiddenRows(
                overridesByTokenIndex: hiddenOverridesByTokenIndex,
                prefillPlan: prefillPlan,
                sequenceLength: tokens.count
            )
        }

        guard replayStartStepIndex <= maximumStepIndex else {
            var snapshots: [String: [Float]] = [:]
            snapshots.reserveCapacity(stagingBuffers.count)
            for (label, staging) in stagingBuffers {
                snapshots[label] = Self.readBuffer(
                    staging.buffer,
                    precision: staging.precision,
                    count: staging.count
                )
            }
            return snapshots
        }

        try probeSubmission.withCompute { encoder, argumentTable in
            for currentStepIndex in replayStartStepIndex...maximumStepIndex {
                let currentStep = prefillPlan.steps[currentStepIndex]
                let currentProbes = probes.filter { ($0.stepIndex ?? stepIndex) == currentStepIndex }
                if !currentProbes.isEmpty {
                    try Self.encodePrefillProbes(
                        currentProbes.filter { $0.phase == .beforeStep },
                        step: currentStep,
                        lastTokenIndex: lastTokenIndex,
                        stagingBuffers: stagingBuffers,
                        encoder: encoder,
                        visibilityOptions: resolvedProbeVisibilityOptions
                    )
                }

                Self.encodeDebugPrefillStep(
                    prefillPlan.steps[currentStepIndex],
                    encoder: encoder,
                    argumentTable: argumentTable,
                    runtimeConstantBuffer: prefillPlan.buffers.runtimeConstantBuffer,
                    sequenceLength: tokens.count,
                    visibilityOptions: resolvedStepVisibilityOptions
                )

                if !currentProbes.isEmpty {
                    try Self.encodePrefillProbes(
                        currentProbes.filter { $0.phase == .afterStep },
                        step: currentStep,
                        lastTokenIndex: lastTokenIndex,
                        stagingBuffers: stagingBuffers,
                        encoder: encoder,
                        visibilityOptions: resolvedProbeVisibilityOptions
                    )
                }
            }
        }

        var snapshots: [String: [Float]] = [:]
        snapshots.reserveCapacity(stagingBuffers.count)
        for (label, staging) in stagingBuffers {
            snapshots[label] = Self.readBuffer(
                staging.buffer,
                precision: staging.precision,
                count: staging.count
            )
        }
        return snapshots
    }

    private static func overwritePrefillHiddenRows(
        overridesByTokenIndex: [Int: [Float]],
        prefillPlan: MetalPrefillPlan,
        sequenceLength: Int
    ) throws {
        guard !overridesByTokenIndex.isEmpty else { return }
        let hiddenStride = prefillPlan.buffers.hidden.length
            / MemoryLayout<Float>.stride
            / prefillPlan.maximumSequenceLength
        let hiddenPointer = prefillPlan.buffers.hidden.contents().bindMemory(
            to: Float.self,
            capacity: hiddenStride * sequenceLength
        )
        for (tokenIndex, values) in overridesByTokenIndex {
            guard tokenIndex >= 0, tokenIndex < sequenceLength else {
                throw MetalCompilerError.deviceSetupFailed("Hidden override token index out of range")
            }
            guard values.count == hiddenStride else {
                throw MetalCompilerError.deviceSetupFailed("Hidden override dimension mismatch")
            }
            (hiddenPointer + tokenIndex * hiddenStride).update(from: values, count: hiddenStride)
        }
    }

    public mutating func debugDecodeBindingProbes(
        promptTokens: [Int32],
        tokenID: Int32,
        ropePositionAxes: (UInt32, UInt32, UInt32)? = nil,
        probes: [DebugDecodeBindingProbe],
        visibilityOptions: MTL4VisibilityOptions = []
    ) throws -> [String: [Float]] {
        guard !promptTokens.isEmpty else { return [:] }
        guard !probes.isEmpty else { return [:] }
        guard let maximumStepIndex = probes.map(\.stepIndex).max() else {
            return [:]
        }
        guard maximumStepIndex >= 0, maximumStepIndex < decodePlan.steps.count else {
            throw MetalCompilerError.deviceSetupFailed("Debug decode probe step index out of range")
        }

        var stagingBuffers: [String: (buffer: MTLBuffer, precision: BufferPrecision, count: Int)] = [:]
        stagingBuffers.reserveCapacity(probes.count)
        for probe in probes {
            guard probe.count > 0 else {
                throw MetalCompilerError.deviceSetupFailed("Debug decode probe count must be positive")
            }
            let stagingLength = probe.count * probe.precision.byteSize
            guard let stagingBuffer = submission.device.makeBuffer(length: stagingLength, options: [.storageModeShared]) else {
                throw MetalCompilerError.deviceSetupFailed("Cannot allocate debug decode staging buffer")
            }
            stagingBuffers[probe.label] = (stagingBuffer, probe.precision, probe.count)
        }

        resetState()
        _ = prefill(tokens: promptTokens)

        let buffers = decodePlan.buffers
        buffers.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(position)
        let resolvedAxes = ropePositionAxes ?? (UInt32(position), UInt32(position), UInt32(position))
        let ropeAxesPointer = buffers.ropePositionAxes.contents().bindMemory(to: UInt32.self, capacity: 3)
        ropeAxesPointer[0] = resolvedAxes.0
        ropeAxesPointer[1] = resolvedAxes.1
        ropeAxesPointer[2] = resolvedAxes.2
        buffers.tokenIn.contents().bindMemory(to: Int32.self, capacity: 1).pointee = tokenID
        let steps = decodePlan.steps

        try submission.withCompute { encoder, argumentTable in
            for currentStepIndex in 0...maximumStepIndex {
                let step = steps[currentStepIndex]
                let currentProbes = probes.filter { $0.stepIndex == currentStepIndex }
                if !currentProbes.isEmpty {
                    try Self.encodeDecodeProbes(
                        currentProbes.filter { $0.phase == .beforeStep },
                        step: step,
                        stagingBuffers: stagingBuffers,
                        encoder: encoder,
                        visibilityOptions: visibilityOptions
                    )
                }

                MetalDecodeEncoder.encodeStep(
                    step: step,
                    encoder: encoder,
                    argumentTable: argumentTable,
                    visibilityOptions: visibilityOptions
                )

                if !currentProbes.isEmpty {
                    try Self.encodeDecodeProbes(
                        currentProbes.filter { $0.phase == .afterStep },
                        step: step,
                        stagingBuffers: stagingBuffers,
                        encoder: encoder,
                        visibilityOptions: visibilityOptions
                    )
                }
            }
        }

        var snapshots: [String: [Float]] = [:]
        snapshots.reserveCapacity(stagingBuffers.count)
        for (label, staging) in stagingBuffers {
            snapshots[label] = Self.readBuffer(
                staging.buffer,
                precision: staging.precision,
                count: staging.count
            )
        }
        return snapshots
    }

    private static func encodeDecodeProbes(
        _ probes: [DebugDecodeBindingProbe],
        step: MetalDispatchStep,
        stagingBuffers: [String: (buffer: MTLBuffer, precision: BufferPrecision, count: Int)],
        encoder: MTL4ComputeCommandEncoder,
        visibilityOptions: MTL4VisibilityOptions = []
    ) throws {
        guard !probes.isEmpty else { return }
        encoder.barrier(
            afterEncoderStages: .dispatch,
            beforeEncoderStages: .blit,
            visibilityOptions: resolvedMetalVisibilityOptions(visibilityOptions)
        )
        for probe in probes {
            guard let binding = step.bindings.buffers.first(where: { $0.index == probe.bindingIndex }) else {
                throw MetalCompilerError.deviceSetupFailed(
                    "Debug decode probe missing binding[\(probe.bindingIndex)] for step \(probe.stepIndex)"
                )
            }
            guard let staging = stagingBuffers[probe.label] else {
                throw MetalCompilerError.deviceSetupFailed("Debug decode staging buffer missing for \(probe.label)")
            }
            let maxReadableCount = max(
                (binding.buffer.length - binding.offset) / probe.precision.byteSize,
                0
            )
            guard probe.elementOffset >= 0, probe.elementOffset + staging.count <= maxReadableCount else {
                throw MetalCompilerError.deviceSetupFailed(
                    "Debug decode probe element offset out of range for \(probe.label)"
                )
            }
            encoder.copy(
                sourceBuffer: binding.buffer,
                sourceOffset: binding.offset + probe.elementOffset * probe.precision.byteSize,
                destinationBuffer: staging.buffer,
                destinationOffset: 0,
                size: staging.count * probe.precision.byteSize
            )
        }
        encoder.barrier(
            afterEncoderStages: .blit,
            beforeEncoderStages: .dispatch,
            visibilityOptions: resolvedMetalVisibilityOptions(visibilityOptions)
        )
    }
#endif


    // MARK: - Lifecycle

    public mutating func makePromptSnapshot(firstToken: Int32) throws -> MetalPromptState {
        try promptStateStore.makePromptSnapshot(
            plan: decodePlan,
            submission: &submission,
            position: position,
            firstToken: firstToken
        )
    }

    public mutating func restore(promptState: MetalPromptState) throws {
        try promptStateStore.restore(plan: decodePlan, submission: &submission, promptState: promptState)
        position = promptState.position
    }

    public mutating func resetState() {
        var sharedSubmission = submission
        resetState(using: &sharedSubmission)
        submission = sharedSubmission
    }

    private mutating func resetState(using submission: inout MetalSubmissionContext) {
        position = 0
        submission.resetReuseState()
        do {
            try Self.zeroStateBuffers(decodePlan.buffers, submission: &submission)
            if let prefillPlan {
                try Self.zeroStateBuffers(prefillPlan.buffers, submission: &submission)
            }
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
            device: submission.device,
            hiddenElementCount: hiddenElementCount
        )
        hiddenOverrideStagingWorkspace = workspace
        return workspace
    }

    private mutating func decodeSync(
        hiddenStagingBuffer: MTLBuffer,
        hiddenElementCount: Int,
        ropePositionAxes: (UInt32, UInt32, UInt32)?,
        deepstackBuffers: [Int: MTLBuffer],
        ephemeralResidency: MetalResidencyLease
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

        // Write element count to constant buffer
        let constantBuffer = hiddenOverrideConstantBuffer
        constantBuffer.contents().storeBytes(
            of: UInt32(hiddenElementCount),
            as: UInt32.self
        )

        // Capture values before closure to avoid exclusivity violation with submission
        let steps = decodePlan.steps
        let hiddenBuffer = buffers.hidden

        try submission.withCompute(ephemeralResidency: ephemeralResidency) { encoder, argumentTable in
            // Zero residual, scratch, logits
            encoder.fill(buffer: buffers.residual, range: 0..<buffers.residual.length, value: 0)
            encoder.fill(buffer: buffers.scratch, range: 0..<buffers.scratch.length, value: 0)
            encoder.fill(buffer: buffers.logits, range: 0..<buffers.logits.length, value: 0)

            encoder.barrier(
                afterEncoderStages: .dispatch,
                beforeEncoderStages: .dispatch,
                visibilityOptions: .device
            )

            // Copy hidden state from staging buffer
            Self.encodeHiddenOverride(
                encoder: encoder,
                argumentTable: argumentTable,
                pipeline: copyPipeline,
                hidden: hiddenBuffer,
                staging: hiddenStagingBuffer,
                count: hiddenElementCount,
                constantBuffer: constantBuffer
            )

            var injectedLayers = Set<Int>()
            for step in steps {
                if step.metadata.kernelName?.hasPrefix("embedding_lookup") == true {
                    continue
                }
                if let layerIndex = step.metadata.layerIndex,
                   let deepstackBuffer = deepstackBuffers[layerIndex],
                   !injectedLayers.contains(layerIndex) {
                    Self.encodeHiddenOverride(
                        encoder: encoder,
                        argumentTable: argumentTable,
                        pipeline: addPipeline,
                        hidden: hiddenBuffer,
                        staging: deepstackBuffer,
                        count: hiddenElementCount,
                        constantBuffer: constantBuffer
                    )
                    injectedLayers.insert(layerIndex)
                }
                MetalDecodeEncoder.encodeStep(
                    step: step,
                    encoder: encoder,
                    argumentTable: argumentTable,
                    visibilityOptions: .device
                )
            }
        }

        position += 1
        return buffers.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
    }

    private mutating func rebuildStableResidencyRegistry() throws {
        stableResidencyRegistry.remove(from: submission.queue)
        let registry = try MetalStableResidencyRegistry(
            device: submission.device,
            compiledModel: compiledModel,
            hiddenOverrideConstantBuffer: hiddenOverrideConstantBuffer
        )
        registry.register(on: submission.queue)
        stableResidencyRegistry = registry
    }

    private static func resolveHiddenConversionPipeline(
        _ compiledModel: MetalCompiledModel
    ) -> MTLComputePipelineState? {
        let base = "hidden_copy_from_float"
        let name: String
        switch compiledModel.decodePlan.buffers.bufferPrecision {
        case .float16: name = base
        case .bfloat16: name = base + "_bf16"
        case .float32: return nil  // F32→F32 uses blit copy, no conversion needed
        }
        return compiledModel.auxiliaryPipelines[name]
    }

    private static func resolveKVTransferPipeline(
        _ compiledModel: MetalCompiledModel
    ) -> MTLComputePipelineState? {
        compiledModel.auxiliaryPipelines["kv_cache_transfer"]
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

    private static func populatePrefillInputs(
        prefillPlan: MetalPrefillPlan,
        position: Int,
        tokens: [Int32],
        ropePositionAxesByTokenIndex: [(UInt32, UInt32, UInt32)]?
    ) {
        let sequenceLength = tokens.count
        let tokenPointer = prefillPlan.buffers.tokenIDs.contents().bindMemory(
            to: Int32.self,
            capacity: sequenceLength
        )
        let positionPointer = prefillPlan.buffers.positions.contents().bindMemory(
            to: UInt32.self,
            capacity: sequenceLength
        )
        let ropeAxesPointer = prefillPlan.buffers.ropePositionAxes.contents().bindMemory(
            to: UInt32.self,
            capacity: sequenceLength * 3
        )
        for index in 0..<sequenceLength {
            tokenPointer[index] = tokens[index]
            let absolutePosition = UInt32(position + index)
            positionPointer[index] = absolutePosition
            let axes = ropePositionAxesByTokenIndex?[index]
                ?? (absolutePosition, absolutePosition, absolutePosition)
            ropeAxesPointer[index * 3] = axes.0
            ropeAxesPointer[index * 3 + 1] = axes.1
            ropeAxesPointer[index * 3 + 2] = axes.2
        }
    }

    private static func writePrefillRuntimeConstants(
        prefillPlan: MetalPrefillPlan,
        basePosition: Int,
        sequenceLength: Int,
        hiddenConversionElementCount: Int
    ) {
        let pointer = prefillPlan.buffers.runtimeConstantBuffer.contents()
        pointer.storeBytes(
            of: UInt32(sequenceLength),
            toByteOffset: PrefillBufferSet.sequenceLengthOffset,
            as: UInt32.self
        )
        pointer.storeBytes(
            of: UInt32(hiddenConversionElementCount),
            toByteOffset: PrefillBufferSet.hiddenConversionCountOffset,
            as: UInt32.self
        )
        for positionOffset in 0..<sequenceLength {
            pointer.storeBytes(
                of: UInt32(basePosition + positionOffset),
                toByteOffset: PrefillBufferSet.positionOffset(at: positionOffset),
                as: UInt32.self
            )
        }
    }

    private static func encodeDebugPrefillStep(
        _ step: MetalPrefillStep,
        encoder: MTL4ComputeCommandEncoder,
        argumentTable: MTL4ArgumentTable,
        runtimeConstantBuffer: MTLBuffer,
        sequenceLength: Int,
        visibilityOptions: MTL4VisibilityOptions = []
    ) {
        switch step.mode {
        case .batch:
            step.bindings.bind(to: argumentTable)
            step.bindRuntimeArguments(
                argumentTable: argumentTable,
                runtimeConstantBuffer: runtimeConstantBuffer,
                sequenceLengthOffset: PrefillBufferSet.sequenceLengthOffset
            )
            let gridSize = step.resolvedGridSize(sequenceLength: sequenceLength)
            step.descriptor.encode(
                on: encoder,
                argumentTable: argumentTable,
                visibilityOptions: visibilityOptions,
                gridSize: gridSize
            )
        case .lastToken:
            step.bindStaticArguments(argumentTable: argumentTable, position: sequenceLength - 1)
            step.descriptor.encode(
                on: encoder,
                argumentTable: argumentTable,
                visibilityOptions: visibilityOptions
            )
        case .perPosition:
            for positionOffset in 0..<sequenceLength {
                step.bindStaticArguments(argumentTable: argumentTable, position: positionOffset)
                if let positionBufferIndex = step.positionBufferIndex {
                    argumentTable.setAddress(
                        runtimeConstantBuffer.gpuAddress
                            + UInt64(PrefillBufferSet.positionOffset(at: positionOffset)),
                        index: positionBufferIndex
                    )
                }
                step.descriptor.encode(
                    on: encoder,
                    argumentTable: argumentTable,
                    visibilityOptions: visibilityOptions
                )
            }
        }
    }

#if ENABLE_METAL_PROBES
    static func encodePrefillProbes(
        _ probes: [DebugPrefillBindingProbe],
        step: MetalPrefillStep,
        lastTokenIndex: Int,
        stagingBuffers: [String: (buffer: MTLBuffer, precision: BufferPrecision, count: Int)],
        encoder: MTL4ComputeCommandEncoder,
        visibilityOptions: MTL4VisibilityOptions = []
    ) throws {
        guard !probes.isEmpty else { return }
        encoder.barrier(
            afterEncoderStages: .dispatch,
            beforeEncoderStages: .blit,
            visibilityOptions: resolvedMetalVisibilityOptions(visibilityOptions)
        )
        for probe in probes {
            guard let binding = step.bindings.buffers.first(where: { $0.index == probe.bindingIndex }) else {
                throw MetalCompilerError.deviceSetupFailed("Debug probe missing binding[\(probe.bindingIndex)] for step")
            }
            guard let staging = stagingBuffers[probe.label] else {
                throw MetalCompilerError.deviceSetupFailed("Debug probe staging buffer missing for \(probe.label)")
            }
            let rowIndex = probe.rowIndex ?? lastTokenIndex
            guard rowIndex >= 0, rowIndex <= lastTokenIndex else {
                throw MetalCompilerError.deviceSetupFailed("Debug probe row index out of range for \(probe.label)")
            }
            guard probe.elementOffset >= 0, probe.elementOffset + staging.count <= probe.rowStride else {
                throw MetalCompilerError.deviceSetupFailed("Debug probe element offset out of range for \(probe.label)")
            }
            let sourceOffset = binding.offset
                + (rowIndex * probe.rowStride + probe.elementOffset) * probe.precision.byteSize
            let size = staging.count * probe.precision.byteSize
            encoder.copy(
                sourceBuffer: binding.buffer,
                sourceOffset: sourceOffset,
                destinationBuffer: staging.buffer,
                destinationOffset: 0,
                size: size
            )
        }
        encoder.barrier(
            afterEncoderStages: .blit,
            beforeEncoderStages: .dispatch,
            visibilityOptions: resolvedMetalVisibilityOptions(visibilityOptions)
        )
    }
#endif

    private static func readPrefillLastTokenHidden(
        prefillPlan: MetalPrefillPlan,
        sequenceLength: Int
    ) -> [Float] {
        let hiddenSize = prefillPlan.buffers.hidden.length
            / max(prefillPlan.maximumSequenceLength, 1)
            / max(prefillPlan.buffers.bufferPrecision.byteSize, 1)
        let source = prefillPlan.finalHiddenSource(sequenceLength: sequenceLength)
        switch prefillPlan.buffers.bufferPrecision {
        case .float32:
            let pointer = source.buffer.contents().bindMemory(
                to: Float.self,
                capacity: source.buffer.length / MemoryLayout<Float>.stride
            )
            let elementOffset = source.offset / MemoryLayout<Float>.stride
            return Array(UnsafeBufferPointer(start: pointer + elementOffset, count: hiddenSize))
        case .float16:
            let pointer = source.buffer.contents().bindMemory(
                to: Float16.self,
                capacity: source.buffer.length / MemoryLayout<Float16>.stride
            )
            let elementOffset = source.offset / MemoryLayout<Float16>.stride
            return (0..<hiddenSize).map { Float(pointer[elementOffset + $0]) }
        case .bfloat16:
            let pointer = source.buffer.contents().bindMemory(
                to: UInt16.self,
                capacity: source.buffer.length / MemoryLayout<UInt16>.stride
            )
            let elementOffset = source.offset / MemoryLayout<UInt16>.stride
            return (0..<hiddenSize).map { index in
                Float(bitPattern: UInt32(pointer[elementOffset + index]) << 16)
            }
        }
    }

    private static func prefillScratchRowOffset(
        prefillPlan: MetalPrefillPlan,
        rowIndex: Int,
        slotIndex: Int,
        rowStride: Int
    ) -> Int {
        let slotBase = slotIndex
            * prefillPlan.slotDimension
            * prefillPlan.maximumSequenceLength
            * MemoryLayout<Float>.stride
        return slotBase + rowIndex * rowStride * MemoryLayout<Float>.stride
    }

    private static func prefillLastTokenScratchOffset(
        prefillPlan: MetalPrefillPlan,
        sequenceLength: Int,
        slotIndex: Int,
        rowStride: Int
    ) -> Int {
        let slotDimension = prefillPlan.slotDimension
        guard slotDimension > 0 else { return 0 }
        let slotBase = slotIndex * slotDimension * prefillPlan.maximumSequenceLength
        let lastTokenBase = slotBase + (sequenceLength - 1) * rowStride
        return lastTokenBase * MemoryLayout<Float>.stride
    }

    private static func readSharedFloatBuffer(
        _ buffer: MTLBuffer,
        count: Int
    ) -> [Float] {
        guard count > 0 else { return [] }
        let pointer = buffer.contents().bindMemory(
            to: Float.self,
            capacity: buffer.length / MemoryLayout<Float>.stride
        )
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private static func readBuffer(
        _ buffer: MTLBuffer,
        precision: BufferPrecision,
        count: Int
    ) -> [Float] {
        guard count > 0 else { return [] }
        switch precision {
        case .float32:
            let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: pointer, count: count))
        case .float16:
            let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
            return (0..<count).map { Float(pointer[$0]) }
        case .bfloat16:
            let pointer = buffer.contents().bindMemory(to: UInt16.self, capacity: count)
            return (0..<count).map { Float(bitPattern: UInt32(pointer[$0]) << 16) }
        }
    }

    private static func encodeHiddenOverride(
        encoder: MTL4ComputeCommandEncoder,
        argumentTable: MTL4ArgumentTable,
        pipeline: MTLComputePipelineState,
        hidden: MTLBuffer,
        staging: MTLBuffer,
        count: Int,
        constantBuffer: MTLBuffer
    ) {
        let threadCount = min(max(1, count), pipeline.maxTotalThreadsPerThreadgroup)
        let gridSize = MTLSize(width: (count + threadCount - 1) / threadCount, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: threadCount, height: 1, depth: 1)

        encoder.barrier(
            afterEncoderStages: .dispatch,
            beforeEncoderStages: .dispatch,
            visibilityOptions: .device
        )
        argumentTable.setAddress(hidden.gpuAddress, index: 0)
        argumentTable.setAddress(staging.gpuAddress, index: 1)
        argumentTable.setAddress(constantBuffer.gpuAddress, index: 2)
        encoder.setArgumentTable(argumentTable)
        encoder.setComputePipelineState(pipeline)
        encoder.dispatchThreadgroups(
            threadgroupsPerGrid: gridSize,
            threadsPerThreadgroup: threadgroupSize
        )
    }
}
