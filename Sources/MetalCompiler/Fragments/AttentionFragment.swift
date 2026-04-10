import LMIR

extension AttentionAttributes: MetalKernelFragment, _FragmentBodyAccessor {
    @MetalKernelFragmentBuilder
    public func fragment(context: KernelContext) -> some MetalKernelFragment {
        let attentionDimension = headCount * headDimension
        let qProjectionDimension = outputGate == .sigmoidPackedInQProj
            ? attentionDimension * 2
            : attentionDimension
        let qkWeightBias: Float = qkNorm == .rmsNormUnitOffset ? 1 : 0
        let queryScratchSlotIndex = outputGate == .sigmoidPackedInQProj ? 4 : 1
        let usesSharedKV = sharedKeyValueSourceLayerIndex != nil

        LinearFragment(field: "q_proj", inputDimension: hiddenSize, outputDimension: qProjectionDimension)
        if !usesSharedKV {
            LinearFragment(field: "k_proj", inputDimension: hiddenSize, outputDimension: kvHeadCount * headDimension)
            LinearFragment(
                field: valueProjectionSource == .keyProjection ? "k_proj" : "v_proj",
                inputDimension: hiddenSize,
                outputDimension: kvHeadCount * headDimension
            )
        }
        if outputGate == .sigmoidPackedInQProj {
            PackedQueryExtractFragment(
                headCount: headCount,
                headDimension: headDimension,
                packedSourceSlotIndex: 1,
                outputSlotIndex: queryScratchSlotIndex
            )
        }
        if let qkNorm = qkNorm, qkNorm != .none {
            QKNormFragment(
                headCount: headCount,
                headDimension: headDimension,
                epsilon: 1e-6,
                weightRole: "q_layernorm",
                weightBias: qkWeightBias,
                scratchSlotIndex: queryScratchSlotIndex
            )
            if !usesSharedKV {
                QKNormFragment(
                    headCount: kvHeadCount,
                    headDimension: headDimension,
                    epsilon: 1e-6,
                    weightRole: "k_layernorm",
                    weightBias: qkWeightBias
                )
            }
        }
        if !usesSharedKV, valueNorm == .rmsNormNoScale {
            PerHeadRMSNormFragment(
                headCount: kvHeadCount,
                headDimension: headDimension,
                epsilon: 1e-6,
                scratchSlotIndex: 3
            )
        }
        // RoPE is inlined into FlashAttentionFragment for decode (saves 1 dispatch + 1 barrier per layer).
        // For prefill, FlashAttentionFragment emits a separate rope_seq step internally.
        FlashAttentionFragment(
            headCount: headCount, kvHeadCount: kvHeadCount,
            headDimension: headDimension,
            attentionScale: attentionScale,
            ropeDimension: rope?.dimension ?? 0,
            ropeBase: rope?.base ?? 0,
            ropeScaling: rope?.scaling,
            mropeAxes: rope?.mropeAxes,
            querySlotIndex: queryScratchSlotIndex,
            windowLeft: window?.left,
            sharedKVSourceLayerIndex: sharedKeyValueSourceLayerIndex)
        if outputGate == .sigmoidPackedInQProj {
            PackedSigmoidGateFragment(
                dimension: attentionDimension,
                headDimension: headDimension,
                packedSourceSlotIndex: 1,
                packedHeadStride: headDimension * 2,
                gateHeadOffset: headDimension
            )
        }
        LinearFragment(field: "o_proj", inputDimension: attentionDimension, outputDimension: hiddenSize)
    }
    public var isFusable: Bool { false }
    public func _visitBody(context: KernelContext, _ visitor: (any MetalKernelFragment) -> Void) { visitor(fragment(context: context)) }
}
