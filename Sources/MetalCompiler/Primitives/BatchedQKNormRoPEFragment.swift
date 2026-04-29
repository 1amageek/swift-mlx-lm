import Metal
import LMIR

/// Fuses batched Q+K per-head RMS norm with RoPE into a single prefill dispatch.
///
/// Replaces the sequence `BatchedFragment([QKNorm_Q, QKNorm_K])` + `rope_seq_f32`
/// — two separate dispatches with a barrier between them — with a single fused
/// dispatch `batched_qk_rms_norm_rope_seq_f32`. Per layer this removes one
/// dispatch and one barrier (`~30 µs`), amortised across every prefill step.
///
/// Decode path is **unchanged**: the fragment delegates to the existing
/// `batched_qk_rms_norm_2` kernel because RoPE is applied inline inside
/// `rope_flash_attn_decode`. The caller must set
/// `FlashAttentionFragment.suppressPrefillRoPE = true` when emitting this
/// fragment so the downstream attention stage does not re-apply RoPE in
/// prefill.
public struct BatchedQKNormRoPEFragment: PrimitiveMetalKernelFragment {
    public let qNorm: QKNormFragment
    public let kNorm: QKNormFragment
    public let ropeDimension: Int
    public let ropeBase: Float
    public let ropeScaling: RoPEScaling?
    public let mropeAxes: MRoPEAxes?

    public init(
        qNorm: QKNormFragment,
        kNorm: QKNormFragment,
        ropeDimension: Int,
        ropeBase: Float,
        ropeScaling: RoPEScaling? = nil,
        mropeAxes: MRoPEAxes? = nil
    ) {
        precondition(qNorm.headDimension == kNorm.headDimension,
                     "Q and K must share the same head dimension")
        precondition(qNorm.epsilon == kNorm.epsilon,
                     "Q and K must share the same epsilon")
        precondition(qNorm.weightBias == kNorm.weightBias,
                     "Q and K must share the same weight bias")
        self.qNorm = qNorm
        self.kNorm = kNorm
        self.ropeDimension = ropeDimension
        self.ropeBase = ropeBase
        self.ropeScaling = ropeScaling
        self.mropeAxes = mropeAxes
    }

    // MARK: - PrimitiveMetalKernelFragment

    public var isFusable: Bool { false }

    public var dispatchDimension: MetalDispatchDimension {
        .perHead(headCount: qNorm.headCount + kNorm.headCount)
    }

    public var weightSlots: [MetalWeightSlot] {
        qNorm.weightSlots + kNorm.weightSlots
    }

    public func requiredFallbackBufferSize(for role: String, bytesPerScalar: Int) -> Int {
        let q = qNorm.requiredFallbackBufferSize(for: role, bytesPerScalar: bytesPerScalar)
        if q > 0 { return q }
        return kNorm.requiredFallbackBufferSize(for: role, bytesPerScalar: bytesPerScalar)
    }

    // MARK: - Decode

    /// Decode reuses the existing `batched_qk_rms_norm_2` kernel. RoPE is
    /// applied inline inside `rope_flash_attn_decode`, so the fused norm+RoPE
    /// kernel is unnecessary here.
    public func kernelName(context: KernelContext) -> String {
        context.weightFormat.isBFloat16
            ? "batched_qk_rms_norm_bf16_2"
            : "batched_qk_rms_norm_2"
    }

    public func kernelSource(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        MetalSourceGenerator.generateBatchedPerHead2(
            name: name,
            bufferPrecision: bufferPrecision,
            weightFormat: weightFormat
        )
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        // Delegate to the equivalent BatchedFragment to keep decode bindings
        // byte-for-byte identical with the existing path.
        let equivalent = BatchedFragment(
            fragments: [qNorm, kNorm],
            dispatchDimension: dispatchDimension
        )
        return equivalent.decodeBindings(context: context)
    }

    // MARK: - Prefill (fused)

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let kernelName = context.kernelContext.weightFormat.isBFloat16
            ? "batched_qk_rms_norm_rope_seq_bf16_f32"
            : "batched_qk_rms_norm_rope_seq_f32"
        let pipeline = try context.getPipeline(kernelName)

        let scratchSlotSize = context.slotDimension
            * context.scratchElementSize
            * context.maximumSequenceLength

        let (qWeightBuffer, qWeightOffset) = context.resolveWeight(qNorm.weightRole)
        let (kWeightBuffer, kWeightOffset) = context.resolveWeight(kNorm.weightRole)

        let totalHeads = qNorm.headCount + kNorm.headCount
        let threads = min(256, pipeline.maxTotalThreadsPerThreadgroup)

        let qTotalDimension = qNorm.headCount * qNorm.headDimension
        let kTotalDimension = kNorm.headCount * kNorm.headDimension

        return FragmentPrefillSteps(
            steps: [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: totalHeads, height: context.maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
                bufferBindings: [
                    (0, context.buffers.scratch, qNorm.scratchSlotIndex * scratchSlotSize),
                    (1, context.buffers.scratch, kNorm.scratchSlotIndex * scratchSlotSize),
                    (2, qWeightBuffer, qWeightOffset),
                    (3, kWeightBuffer, kWeightOffset),
                    (4, context.buffers.ropePositionAxes, 0),
                ],
                bytesBindings: [
                    uint32Binding(5, UInt32(qNorm.headCount)),
                    uint32Binding(6, UInt32(kNorm.headCount)),
                    uint32Binding(7, UInt32(qNorm.headDimension)),
                    uint32Binding(8, UInt32(ropeDimension)),
                    floatBinding(9, qNorm.epsilon),
                    floatBinding(10, qNorm.weightBias),
                    floatBinding(11, ropeBase),
                    uint32Binding(12, UInt32(sectionCount(at: 0))),
                    uint32Binding(13, UInt32(sectionCount(at: 1))),
                    uint32Binding(14, UInt32(sectionCount(at: 2))),
                    uint32Binding(15, UInt32(mropeAxes?.interleaved == true ? 1 : 0)),
                    uint32Binding(16, UInt32(context.maximumSequenceLength)),
                    uint32Binding(17, UInt32(qTotalDimension)),
                    uint32Binding(18, UInt32(kTotalDimension)),
                    uint32Binding(19, UInt32(usesProportionalRoPE ? 1 : 0)),
                    uint32Binding(20, UInt32(context.slotDimension)),
                    uint32Binding(21, UInt32(context.slotDimension)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 16),
                positionBufferIndex: nil,
                perPositionStrides: [:],
                metadata: .init(
                    kernelName: kernelName,
                    bufferAccessPattern: .init(reads: [0, 1, 2, 3, 4], writes: [0, 1])
                )
            )],
            outputIsHidden: false
        )
    }

    private func sectionCount(at index: Int) -> Int {
        guard let mropeAxes, index < mropeAxes.sections.count else { return 0 }
        return mropeAxes.sections[index]
    }

    private var usesProportionalRoPE: Bool {
        ropeScaling?.kind == .custom("proportional")
    }
}
