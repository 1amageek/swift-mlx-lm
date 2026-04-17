import LMIR

extension AttentionAttributes: MetalCompilable {

    /// Fragment expansion for Attention: batched Q/K/V projections, optional QKNorm/RoPE,
    /// scaled dot-product attention, optional output gate, and O projection.
    @MetalKernelFragmentBuilder
    package func fragment(context: KernelContext) -> some MetalKernelFragment {
        let attentionDimension = headCount * headDimension
        let qProjectionDimension = outputGate == .sigmoidPackedInQProj
            ? attentionDimension * 2
            : attentionDimension
        let qkWeightBias: Float = qkNorm == .rmsNormUnitOffset ? 1 : 0
        let queryScratchSlotIndex = outputGate == .sigmoidPackedInQProj ? 4 : 1
        let usesSharedKV = sharedKeyValueSourceLayerIndex != nil

        // Batched Q/K/V projections (component-internal optimization)
        if usesSharedKV {
            LinearFragment(field: "q_proj", inputDimension: hiddenSize, outputDimension: qProjectionDimension)
        } else {
            BatchedProjection(projections: [
                .init(field: "q_proj", inputDimension: hiddenSize, outputDimension: qProjectionDimension),
                .init(field: "k_proj", inputDimension: hiddenSize, outputDimension: kvHeadCount * headDimension),
                .init(
                    field: valueProjectionSource == .keyProjection ? "k_proj" : "v_proj",
                    inputDimension: hiddenSize,
                    outputDimension: kvHeadCount * headDimension
                ),
            ])
        }
        if outputGate == .sigmoidPackedInQProj {
            PackedQueryExtractFragment(
                headCount: headCount,
                headDimension: headDimension,
                packedSourceSlotIndex: 1,
                outputSlotIndex: queryScratchSlotIndex
            )
        }
        // Decide whether Q/K RMS norm can be fused with RoPE in prefill.
        // Requires: QK norm is present, RoPE is present, K/V are not shared,
        // and `v_norm` is absent (v norm would need to sit between QK norm and
        // RoPE on K, breaking the fused kernel's assumptions).
        let fuseQKNormWithRoPE: Bool = {
            guard let qkNorm, qkNorm != .none else { return false }
            guard !usesSharedKV else { return false }
            guard rope != nil else { return false }
            guard valueNorm == nil else { return false }
            return true
        }()
        if let qkNorm = qkNorm, qkNorm != .none {
            if usesSharedKV {
                QKNormFragment(
                    headCount: headCount,
                    headDimension: headDimension,
                    epsilon: 1e-6,
                    weightRole: "q_layernorm",
                    weightBias: qkWeightBias,
                    scratchSlotIndex: queryScratchSlotIndex
                )
            } else if fuseQKNormWithRoPE, let ropeParams = rope {
                // Fused batched Q+K RMS norm + RoPE (single prefill dispatch).
                // Decode path is unaffected — RoPE runs inline in flash_attn_decode.
                BatchedQKNormRoPEFragment(
                    qNorm: QKNormFragment(
                        headCount: headCount,
                        headDimension: headDimension,
                        epsilon: 1e-6,
                        weightRole: "q_layernorm",
                        weightBias: qkWeightBias,
                        scratchSlotIndex: queryScratchSlotIndex
                    ),
                    kNorm: QKNormFragment(
                        headCount: kvHeadCount,
                        headDimension: headDimension,
                        epsilon: 1e-6,
                        weightRole: "k_layernorm",
                        weightBias: qkWeightBias
                    ),
                    ropeDimension: ropeParams.dimension,
                    ropeBase: ropeParams.base,
                    ropeScaling: ropeParams.scaling,
                    mropeAxes: ropeParams.mropeAxes
                )
            } else {
                // Batched QK norm (component-internal optimization)
                BatchedFragment(
                    fragments: [
                        QKNormFragment(
                            headCount: headCount,
                            headDimension: headDimension,
                            epsilon: 1e-6,
                            weightRole: "q_layernorm",
                            weightBias: qkWeightBias,
                            scratchSlotIndex: queryScratchSlotIndex
                        ),
                        QKNormFragment(
                            headCount: kvHeadCount,
                            headDimension: headDimension,
                            epsilon: 1e-6,
                            weightRole: "k_layernorm",
                            weightBias: qkWeightBias
                        ),
                    ],
                    dispatchDimension: .perHead(headCount: headCount + kvHeadCount)
                )
            }
        }
        if !usesSharedKV, let valueNorm {
            switch valueNorm {
            case .rmsNormNoScale:
                PerHeadRMSNormFragment(
                    headCount: kvHeadCount,
                    headDimension: headDimension,
                    epsilon: 1e-6,
                    scratchSlotIndex: 3
                )
            case .rmsNormUnitOffset:
                QKNormFragment(
                    headCount: kvHeadCount,
                    headDimension: headDimension,
                    epsilon: 1e-6,
                    weightRole: "v_layernorm",
                    weightBias: 1,
                    scratchSlotIndex: 3
                )
            }
        }
        FlashAttentionFragment(
            headCount: headCount, kvHeadCount: kvHeadCount,
            headDimension: headDimension,
            attentionScale: attentionScale,
            ropeDimension: rope?.dimension ?? 0,
            ropeBase: rope?.base ?? 0,
            ropeScaling: rope?.scaling,
            mropeAxes: rope?.mropeAxes,
            querySlotIndex: queryScratchSlotIndex,
            causal: causal,
            windowLeft: window?.left,
            windowRight: window?.right,
            sharedKVSourceLayerIndex: sharedKeyValueSourceLayerIndex,
            suppressPrefillRoPE: fuseQKNormWithRoPE)
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
}
