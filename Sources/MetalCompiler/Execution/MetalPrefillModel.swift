import Metal

/// Mutable runtime for sequence prefill without a decode plan.
///
/// This is used by embedding-style workloads that need final hidden states from
/// a backbone graph but do not own a language-model output head or decode loop.
public struct MetalPrefillModel: @unchecked Sendable {
    public let prefillPlan: MetalPrefillPlan

    private var submission: MetalSubmissionContext
    private let prefillExecutor = MetalPrefillExecutor()
    private let runtimeLease: MetalResidencyLease
    private let weightLease: MetalResidencyLease
    private let supplementalLease: MetalResidencyLease
    private let stableResidency: MetalResidencyLease

    /// Cached residency lease for an embedding workspace. Keyed by the identity of
    /// the workspace's output buffer (buffers are stable for the workspace's lifetime).
    /// Avoids recreating `MTLResidencySet` on every `captureEmbeddingVector` call.
    private var cachedEmbeddingWorkspaceKey: ObjectIdentifier?
    private var cachedEmbeddingCombinedResidency: MetalResidencyLease?

    public var device: MTLDevice { submission.device }
    public var queue: MTL4CommandQueue { submission.queue }

    public init(plan: MetalPrefillPlan, device: MTLDevice) throws {
        self.prefillPlan = plan
        self.submission = try MetalSubmissionContext(device: device)
        self.runtimeLease = try MetalResidencyLease.required(
            device: device,
            label: "swift-lm.prefill.runtime",
            buffers: plan.buffers.runtimeResidencyBuffers
        )
        self.weightLease = try MetalResidencyLease.required(
            device: device,
            label: "swift-lm.prefill.weights",
            buffers: plan.buffers.weightResidencyBuffers
        )
        self.supplementalLease = try MetalResidencyLease.required(
            device: device,
            label: "swift-lm.prefill.supplemental",
            buffers: plan.supplementalResidencyBuffers
        )
        self.stableResidency = MetalResidencyLease.combined(
            label: "swift-lm.prefill.stable",
            leases: [runtimeLease, weightLease, supplementalLease]
        )
        self.runtimeLease.add(to: self.submission.queue)
        self.weightLease.add(to: self.submission.queue)
        self.supplementalLease.add(to: self.submission.queue)
        try Self.zeroStateBuffers(
            plan.buffers,
            submission: &self.submission,
            residency: self.stableResidency
        )
    }

    public mutating func resetState() throws {
        try Self.zeroStateBuffers(
            prefillPlan.buffers,
            submission: &submission,
            residency: stableResidency
        )
    }

    /// Zero only the buffers that carry state across embedding invocations.
    ///
    /// For pure self-attention backbones (e.g. EmbeddingGemma) nothing carries
    /// over — every consumed buffer is overwritten within the current sequence
    /// range — so this is a no-op and no command buffer is submitted. For SSM /
    /// conv-bearing backbones the stateful buffers must be cleared because
    /// their kernels read memory from before `position 0`.
    private mutating func resetStatefulEmbeddingBuffers() throws {
        var fills: [(buffer: MTLBuffer, value: UInt8)] = []
        if let convState = prefillPlan.buffers.convState {
            fills.append((convState, 0))
        }
        if let recurrentState = prefillPlan.buffers.recurrentState {
            fills.append((recurrentState, 0))
        }
        guard !fills.isEmpty else { return }
        try submission.fillBuffers(fills, ephemeralResidency: stableResidency)
    }

    /// Run prefill and GPU post-processing, returning the final embedding vector.
    ///
    /// Prefill and post-processing (pooling + dense GEMV + L2 normalize) run in a
    /// single command buffer. The GPU reads final hidden states directly from the
    /// prefill output buffer (which may be `.storageModePrivate` scratch), avoiding
    /// any CPU-side memcpy that would fail on compressed private storage.
    public mutating func captureEmbeddingVector(
        tokens: [Int32],
        workspace: MetalEmbeddingWorkspace,
        promptTokenCount: Int
    ) throws -> [Float] {
        guard !tokens.isEmpty else { return [] }
        guard tokens.count <= prefillPlan.maximumSequenceLength else {
            throw MetalCompilerError.deviceSetupFailed(
                "Prefill token count exceeds maximum sequence length"
            )
        }

        // Skip the full `resetState()` on the embedding hot path. The prefill pipeline
        // overwrites every buffer it reads within the current sequence range:
        // `populatePrefillInputs` / `writeRuntimeConstants` overwrite tokens, positions,
        // and runtime constants; prefill dispatches overwrite hidden/residual/scratch
        // within [0..sequenceLength); KV-cache slots beyond the current sequence length
        // are never read by flash-attn (it clamps to the per-position sequence prefix).
        //
        // The only buffers that must be cleared between embeddings are stateful
        // SSM carry-over buffers (conv_state / recurrent_state): their conv1d /
        // recurrence kernels read memory from positions *before* the current sequence,
        // so leftover state from a previous embedding would contaminate the next one.
        // For EmbeddingGemma neither buffer exists and this is a no-op.
        try resetStatefulEmbeddingBuffers()
        let sequenceLength = tokens.count
        prefillExecutor.preparePrefillInputsForEmbedding(
            prefillPlan: prefillPlan,
            position: 0,
            tokens: tokens
        )

        let hiddenBuffer = prefillPlan.finalHiddenBuffer
        let hiddenBaseOffset = prefillPlan.finalHiddenBaseOffset
        let hiddenRowStride = prefillPlan.finalHiddenRowStride
        let elementSize = max(prefillPlan.buffers.bufferPrecision.byteSize, 1)
        let hiddenDimension = prefillPlan.buffers.hidden.length
            / max(prefillPlan.maximumSequenceLength, 1)
            / elementSize

        // Combine all residency: prefill buffers + workspace + post-processor weights.
        // The weight buffers MUST be resident — without them, GPU page table faults
        // silently zero all kernel outputs in the command buffer.
        //
        // Cache the residency lease across repeated calls on the same workspace.
        // Creating MTLResidencySet and attaching it to the queue every embedding is
        // pure overhead: the workspace's buffer identities are stable for its lifetime.
        let combinedResidency: MetalResidencyLease
        let workspaceKey = workspace.identityKey
        if cachedEmbeddingWorkspaceKey == workspaceKey,
           let cached = cachedEmbeddingCombinedResidency {
            combinedResidency = cached
        } else {
            let embeddingResidency = try MetalResidencyLease.required(
                device: submission.device,
                label: "swift-lm.embedding.workspace",
                buffers: workspace.allResidencyBuffers
            )
            embeddingResidency.add(to: submission.queue)
            combinedResidency = MetalResidencyLease.combined(
                label: "swift-lm.embedding.combined",
                leases: [stableResidency, embeddingResidency]
            )
            self.cachedEmbeddingWorkspaceKey = workspaceKey
            self.cachedEmbeddingCombinedResidency = combinedResidency
        }

        // Single command buffer: prefill produces hidden states, then post-processing
        // reads them directly on GPU. The barrier inside workspace.encode ensures
        // prefill dispatches complete before post-processing reads the hidden buffer.
        try submission.withCompute(ephemeralResidency: combinedResidency) { encoder, argumentTable in
            prefillExecutor.encodePrefillStepsForEmbedding(
                encoder: encoder,
                argumentTable: argumentTable,
                prefillPlan: prefillPlan,
                basePosition: 0,
                sequenceLength: sequenceLength
            )

            workspace.encode(
                encoder: encoder,
                argumentTable: argumentTable,
                hiddenBuffer: hiddenBuffer,
                hiddenBaseOffset: hiddenBaseOffset,
                hiddenRowStride: hiddenRowStride,
                hiddenDimension: hiddenDimension,
                sequenceLength: sequenceLength,
                promptTokenCount: promptTokenCount
            )
        }

        return workspace.readResult(hiddenDimension: hiddenDimension)
    }

    public mutating func finalHiddenStates(tokens: [Int32]) throws -> [[Float]] {
        guard !tokens.isEmpty else { return [] }
        guard tokens.count <= prefillPlan.maximumSequenceLength else {
            throw MetalCompilerError.deviceSetupFailed(
                "Prefill token count exceeds maximum sequence length"
            )
        }

        try resetState()
        return try prefillExecutor.captureFinalHiddenRows(
            prefillPlan: prefillPlan,
            submission: &submission,
            position: 0,
            tokens: tokens,
            ephemeralResidency: stableResidency
        )
    }

    private static func zeroStateBuffers(
        _ buffers: PrefillBufferSet,
        submission: inout MetalSubmissionContext,
        residency: MetalResidencyLease = .empty
    ) throws {
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
        try submission.fillBuffers(fills, ephemeralResidency: residency)
    }
}
