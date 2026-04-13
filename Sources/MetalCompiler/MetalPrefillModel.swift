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

        try resetState()

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
        let embeddingResidency = try MetalResidencyLease.required(
            device: submission.device,
            label: "swift-lm.embedding.workspace",
            buffers: workspace.allResidencyBuffers
        )
        embeddingResidency.add(to: submission.queue)
        let combinedResidency = MetalResidencyLease.combined(
            label: "swift-lm.embedding.combined",
            leases: [stableResidency, embeddingResidency]
        )

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
