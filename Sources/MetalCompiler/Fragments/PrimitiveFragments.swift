/// Primitive Metal kernel fragments — leaf nodes of the fragment tree.
///
/// Each primitive fragment is a single dispatch unit. The compiler:
/// 1. Reads fragment parameters (dimension, epsilon, etc.)
/// 2. Determines buffer precision (F16 decode / F32 prefill) and weight format (from STAF)
/// 3. Calls MetalSourceGenerator to produce MSL on-demand
///
/// No hardcoded kernel names. The compiler derives names from fragment type + context.

// MARK: - Primitive Protocol

/// Leaf fragment that the compiler translates into a single Metal dispatch.
public protocol PrimitiveMetalKernelFragment: MetalKernelFragment where Fragment == Never {
    /// GPU dispatch pattern (determines grid/threadgroup sizing).
    var dispatchDimension: MetalDispatchDimension { get }
    /// Weight tensors this fragment reads from STAF.
    var weightSlots: [MetalWeightSlot] { get }
    /// Persistent cache slots (KV cache, conv state).
    var cacheSlots: [MetalCacheSlot] { get }
}

extension PrimitiveMetalKernelFragment {
    public var fragment: Never { fatalError() }
    public var weightSlots: [MetalWeightSlot] { [] }
    public var cacheSlots: [MetalCacheSlot] { [] }
}

// MARK: - Reduction

/// Reduction: all threads cooperate to reduce across a dimension.
/// Used by: RMSNorm, LayerNorm.
public struct Reduction: PrimitiveMetalKernelFragment {
    public let dimension: Int
    public let epsilon: Float

    public init(dimension: Int, epsilon: Float = 0) {
        self.dimension = dimension
        self.epsilon = epsilon
    }

    public var isFusable: Bool { true }
    public var dispatchDimension: MetalDispatchDimension { .reduction(dimension: dimension) }
    public var weightSlots: [MetalWeightSlot] { [MetalWeightSlot(field: nil, role: .weight)] }
}

// MARK: - Elementwise

/// Elementwise: one thread per element, trivially parallel.
/// Used by: SwiGLU, SigmoidGate.
public struct ElementwiseFragment: PrimitiveMetalKernelFragment {
    public let count: Int
    public let kind: ElementwiseKind

    public enum ElementwiseKind: Sendable {
        case swiglu
        case sigmoidGate
    }

    public init(count: Int, kind: ElementwiseKind = .swiglu) {
        self.count = count
        self.kind = kind
    }

    public var isFusable: Bool { true }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: count) }
}

// MARK: - Linear Projection

/// Matrix-vector (decode) or matrix-matrix (prefill) multiply.
public struct LinearFragment: PrimitiveMetalKernelFragment {
    public let field: String
    public let inputDimension: Int
    public let outputDimension: Int

    public init(field: String, inputDimension: Int, outputDimension: Int) {
        self.field = field
        self.inputDimension = inputDimension
        self.outputDimension = outputDimension
    }

    public var isFusable: Bool { false }
    public var dispatchDimension: MetalDispatchDimension {
        .gemv(outputDimension: outputDimension, inputDimension: inputDimension)
    }
    public var weightSlots: [MetalWeightSlot] { [MetalWeightSlot(field: field, role: .weight)] }
}

// MARK: - Gather (Embedding Lookup)

/// Token ID → embedding vector lookup.
public struct GatherFragment: PrimitiveMetalKernelFragment {
    public let vocabularySize: Int
    public let embeddingDimension: Int

    public init(vocabularySize: Int, embeddingDimension: Int) {
        self.vocabularySize = vocabularySize
        self.embeddingDimension = embeddingDimension
    }

    public var isFusable: Bool { false }
    public var dispatchDimension: MetalDispatchDimension { .gather(count: embeddingDimension) }
    public var weightSlots: [MetalWeightSlot] { [MetalWeightSlot(field: nil, role: .weight)] }
}

// MARK: - Argmax

/// Argmax over vocabulary: logits → token ID.
public struct ArgmaxFragment: PrimitiveMetalKernelFragment {
    public let vocabularySize: Int

    public init(vocabularySize: Int) {
        self.vocabularySize = vocabularySize
    }

    public var isFusable: Bool { false }
    public var dispatchDimension: MetalDispatchDimension { .reduction(dimension: vocabularySize) }
}

// MARK: - Flash Attention

/// Single-token attention against KV cache.
public struct FlashAttentionFragment: PrimitiveMetalKernelFragment {
    public let headCount: Int
    public let kvHeadCount: Int
    public let headDimension: Int
    public let ropeDimension: Int
    public let ropeBase: Float

    public init(headCount: Int, kvHeadCount: Int, headDimension: Int,
                ropeDimension: Int = 0, ropeBase: Float = 0) {
        self.headCount = headCount
        self.kvHeadCount = kvHeadCount
        self.headDimension = headDimension
        self.ropeDimension = ropeDimension
        self.ropeBase = ropeBase
    }

    public var isFusable: Bool { false }
    public var dispatchDimension: MetalDispatchDimension {
        .perHead(headCount: headCount)
    }
    public var cacheSlots: [MetalCacheSlot] { [MetalCacheSlot(name: "kv_cache", kind: .kv)] }
}

// MARK: - RoPE

/// Rotary position embedding (in-place on Q and K).
public struct RoPEFragment: PrimitiveMetalKernelFragment {
    public let headCount: Int
    public let kvHeadCount: Int
    public let headDimension: Int
    public let ropeDimension: Int
    public let base: Float

    public init(headCount: Int, kvHeadCount: Int, headDimension: Int,
                ropeDimension: Int, base: Float) {
        self.headCount = headCount
        self.kvHeadCount = kvHeadCount
        self.headDimension = headDimension
        self.ropeDimension = ropeDimension
        self.base = base
    }

    public var isFusable: Bool { true }
    public var dispatchDimension: MetalDispatchDimension {
        .perHead(headCount: max(headCount, kvHeadCount))
    }
}

// MARK: - QK Norm

/// Per-head RMS normalization for Q or K projections.
public struct QKNormFragment: PrimitiveMetalKernelFragment {
    public let headCount: Int
    public let headDimension: Int
    public let epsilon: Float
    public let weightRole: String  // "q_layernorm" or "k_layernorm"

    public init(headCount: Int, headDimension: Int, epsilon: Float, weightRole: String) {
        self.headCount = headCount
        self.headDimension = headDimension
        self.epsilon = epsilon
        self.weightRole = weightRole
    }

    public var isFusable: Bool { false }
    public var dispatchDimension: MetalDispatchDimension { .perHead(headCount: headCount) }
    public var weightSlots: [MetalWeightSlot] { [MetalWeightSlot(field: weightRole, role: .weight)] }
}

// MARK: - Conv1d

/// Depthwise temporal convolution with double gating (decode: state update).
public struct Conv1dFragment: PrimitiveMetalKernelFragment {
    public let dimension: Int
    public let kernelSize: Int

    public init(dimension: Int, kernelSize: Int) {
        self.dimension = dimension
        self.kernelSize = kernelSize
    }

    public var isFusable: Bool { false }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: dimension) }
    public var weightSlots: [MetalWeightSlot] { [MetalWeightSlot(field: "conv_weight", role: .weight)] }
    public var cacheSlots: [MetalCacheSlot] { [MetalCacheSlot(name: "conv_cache", kind: .conv)] }
}

// MARK: - SSM Recurrence

/// DeltaNet/Mamba state-space model recurrence step.
public struct SSMRecurrenceFragment: PrimitiveMetalKernelFragment {
    public let headCount: Int
    public let keyHeadDimension: Int
    public let valueHeadDimension: Int

    public init(headCount: Int, keyHeadDimension: Int, valueHeadDimension: Int) {
        self.headCount = headCount
        self.keyHeadDimension = keyHeadDimension
        self.valueHeadDimension = valueHeadDimension
    }

    public var isFusable: Bool { false }
    public var dispatchDimension: MetalDispatchDimension { .perHead(headCount: headCount) }
}

// MARK: - Sigmoid Gate

/// Sigmoid-gated element-wise operation.
public struct SigmoidGateFragment: PrimitiveMetalKernelFragment {
    public let dimension: Int

    public init(dimension: Int) {
        self.dimension = dimension
    }

    public var isFusable: Bool { true }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: dimension) }
}

// MARK: - KV Quantization

/// Runtime quantization of KV cache entries.
public struct QuantizeKVFragment: PrimitiveMetalKernelFragment {
    public let totalElements: Int
    public let groupSize: Int
    public let bytesPerBlock: Int

    public init(totalElements: Int, groupSize: Int, bytesPerBlock: Int) {
        self.totalElements = totalElements
        self.groupSize = groupSize
        self.bytesPerBlock = bytesPerBlock
    }

    public var isFusable: Bool { false }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: totalElements / groupSize) }
}
