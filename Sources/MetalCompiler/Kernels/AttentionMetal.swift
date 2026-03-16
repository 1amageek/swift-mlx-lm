import LMIR

extension AttentionAttributes: MetalComponent {

    public var dispatchDeclarations: [MetalDispatchDeclaration] {
        let queryDimension = headCount * headDimension
        let kvDimension = kvHeadCount * headDimension
        var declarations: [MetalDispatchDeclaration] = [
            .projection(MetalProjection(field: "q_proj", inputDimension: hiddenSize, outputDimension: queryDimension)),
            .projection(MetalProjection(field: "k_proj", inputDimension: hiddenSize, outputDimension: kvDimension)),
            .projection(MetalProjection(field: "v_proj", inputDimension: hiddenSize, outputDimension: kvDimension)),
        ]
        // QK normalization: normalize Q and K per-head before RoPE
        if let normKind = qkNorm, normKind == .rmsNorm || normKind == .layerNorm {
            let normEpsilon: Float = 1e-6
            declarations.append(.compute(QKNormOperation(
                headCount: headCount, headDimension: headDimension,
                epsilon: normEpsilon, weightRole: "q_layernorm")))
            declarations.append(.compute(QKNormOperation(
                headCount: kvHeadCount, headDimension: headDimension,
                epsilon: normEpsilon, weightRole: "k_layernorm")))
        }
        if let ropeAttributes = rope {
            declarations.append(.compute(RoPEOperation(
                headCount: headCount, kvHeadCount: kvHeadCount,
                headDimension: headDimension,
                ropeDimension: ropeAttributes.dimension, base: ropeAttributes.base)))
        }
        declarations.append(.compute(FlashAttentionDecodeOperation(
            headCount: headCount, kvHeadCount: kvHeadCount, headDimension: headDimension,
            ropeDimension: rope?.dimension ?? 0, ropeBase: rope?.base ?? 500000.0)))
        declarations.append(.projection(MetalProjection(
            field: "o_proj", inputDimension: queryDimension, outputDimension: hiddenSize)))
        if case .sigmoidPackedInQProj = outputGate {
            declarations.append(.compute(SigmoidGateOperation(dimension: hiddenSize)))
        }
        return declarations
    }

    public var weightSlots: [MetalWeightSlot] {
        var slots = [
            MetalWeightSlot(field: "q_proj", role: .weight),
            MetalWeightSlot(field: "k_proj", role: .weight),
            MetalWeightSlot(field: "v_proj", role: .weight),
            MetalWeightSlot(field: "o_proj", role: .weight),
        ]
        if let normKind = qkNorm, normKind == .rmsNorm || normKind == .layerNorm {
            slots.append(MetalWeightSlot(field: "q_layernorm", role: .scale))
            slots.append(MetalWeightSlot(field: "k_layernorm", role: .scale))
        }
        return slots
    }

    public var cacheSlots: [MetalCacheSlot] {
        [MetalCacheSlot(name: "k_cache", kind: .kv),
         MetalCacheSlot(name: "v_cache", kind: .kv)]
    }
}
