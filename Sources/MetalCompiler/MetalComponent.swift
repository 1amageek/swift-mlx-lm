import LMIR

// MARK: - MetalComputeOperation Protocol

/// Declares a non-GEMV compute operation.
///
/// Each concrete struct represents one kind of GPU computation.
/// The compiler reads protocol properties to determine dispatch configuration.
/// The fusion pass uses concrete type casting (as? RMSNormOperation) for pattern matching.
///
/// Adding a new compute operation = adding a new struct. No enum changes needed.
public protocol MetalComputeOperation: Sendable {
    /// Kernel function name in the compiler's MSL source.
    var kernelName: String { get }
    /// Whether the fusion pass should attempt to merge this with adjacent operations.
    var isFusable: Bool { get }
    /// Dispatch dimension for grid/threadgroup calculation.
    var dispatchDimension: MetalDispatchDimension { get }
}

/// GPU dispatch dimension categories. Fixed set — determined by GPU hardware constraints.
public enum MetalDispatchDimension: Sendable, Equatable {
    /// Single threadgroup reduction (norm, argmax).
    case reduction(dimension: Int)
    /// Trivially parallel element-wise (activation, add, copy).
    case elementwise(count: Int)
    /// One threadgroup per head (flash attention, SSM recurrence).
    case perHead(headCount: Int)
    /// Embedding gather.
    case gather(count: Int)
    /// Matrix-vector multiply.
    case gemv(outputDimension: Int, inputDimension: Int)
}

// MARK: - Concrete Compute Operations

public struct RMSNormOperation: MetalComputeOperation {
    public let dimension: Int
    public let epsilon: Float
    public var kernelName: String { "rms_norm" }
    public var isFusable: Bool { true }
    public var dispatchDimension: MetalDispatchDimension { .reduction(dimension: dimension) }

    public init(dimension: Int, epsilon: Float) {
        self.dimension = dimension
        self.epsilon = epsilon
    }
}

public struct LayerNormOperation: MetalComputeOperation {
    public let dimension: Int
    public let epsilon: Float
    public let affine: Bool
    public var kernelName: String { "layer_norm" }
    public var isFusable: Bool { true }
    public var dispatchDimension: MetalDispatchDimension { .reduction(dimension: dimension) }

    public init(dimension: Int, epsilon: Float, affine: Bool) {
        self.dimension = dimension
        self.epsilon = epsilon
        self.affine = affine
    }
}

public struct SwiGLUOperation: MetalComputeOperation {
    public let dimension: Int
    public var kernelName: String { "swiglu" }
    public var isFusable: Bool { true }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: dimension) }

    public init(dimension: Int) {
        self.dimension = dimension
    }
}

public struct FlashAttentionDecodeOperation: MetalComputeOperation {
    public let headCount: Int
    public let kvHeadCount: Int
    public let headDimension: Int
    public let ropeDimension: Int
    public let ropeBase: Float
    public var kernelName: String { "flash_attn_decode" }
    public var isFusable: Bool { false }
    public var dispatchDimension: MetalDispatchDimension { .perHead(headCount: headCount) }

    public init(headCount: Int, kvHeadCount: Int, headDimension: Int, ropeDimension: Int = 0, ropeBase: Float = 500000.0) {
        self.headCount = headCount
        self.kvHeadCount = kvHeadCount
        self.headDimension = headDimension
        self.ropeDimension = ropeDimension
        self.ropeBase = ropeBase
    }
}

public struct RoPEOperation: MetalComputeOperation {
    public let headCount: Int
    public let kvHeadCount: Int
    public let headDimension: Int
    public let ropeDimension: Int
    public let base: Float
    public var kernelName: String { "rope" }
    public var isFusable: Bool { true }
    public var dispatchDimension: MetalDispatchDimension {
        .perHead(headCount: max(headCount, kvHeadCount))
    }

    public init(headCount: Int, kvHeadCount: Int, headDimension: Int, ropeDimension: Int, base: Float) {
        self.headCount = headCount
        self.kvHeadCount = kvHeadCount
        self.headDimension = headDimension
        self.ropeDimension = ropeDimension
        self.base = base
    }
}

public struct EmbeddingLookupOperation: MetalComputeOperation {
    public let vocabularySize: Int
    public let embeddingDimension: Int
    public var kernelName: String { "embedding_lookup" }
    public var isFusable: Bool { false }
    public var dispatchDimension: MetalDispatchDimension { .gather(count: embeddingDimension) }

    public init(vocabularySize: Int, embeddingDimension: Int) {
        self.vocabularySize = vocabularySize
        self.embeddingDimension = embeddingDimension
    }
}

public struct ArgmaxOperation: MetalComputeOperation {
    public let vocabularySize: Int
    public var kernelName: String { "argmax" }
    public var isFusable: Bool { false }
    public var dispatchDimension: MetalDispatchDimension { .reduction(dimension: vocabularySize) }

    public init(vocabularySize: Int) {
        self.vocabularySize = vocabularySize
    }
}

public struct Conv1dOperation: MetalComputeOperation {
    public let dimension: Int
    public let kernelSize: Int
    public var kernelName: String { "conv1d_gated" }
    public var isFusable: Bool { false }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: dimension) }

    public init(dimension: Int, kernelSize: Int) {
        self.dimension = dimension
        self.kernelSize = kernelSize
    }
}

public struct SSMRecurrenceOperation: MetalComputeOperation {
    public let headCount: Int
    public let keyHeadDimension: Int
    public let valueHeadDimension: Int
    public var kernelName: String { "ssm_recurrence" }
    public var isFusable: Bool { false }
    public var dispatchDimension: MetalDispatchDimension { .perHead(headCount: headCount) }

    public init(headCount: Int, keyHeadDimension: Int, valueHeadDimension: Int) {
        self.headCount = headCount
        self.keyHeadDimension = keyHeadDimension
        self.valueHeadDimension = valueHeadDimension
    }
}

public struct SigmoidGateOperation: MetalComputeOperation {
    public let dimension: Int
    public var kernelName: String { "sigmoid_gate" }
    public var isFusable: Bool { true }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: dimension) }

    public init(dimension: Int) {
        self.dimension = dimension
    }
}

/// Quantize FP16 K/V values to quantized format before writing to KV cache.
///
/// Emitted by the compiler when KVCacheSpecification uses a quantized scheme.
/// Placed before flash_attn_decode so the cache stores quantized blocks.
public struct QuantizeKVOperation: MetalComputeOperation {
    public let totalElements: Int
    public let groupSize: Int
    public let bytesPerBlock: Int
    public var kernelName: String { "quantize_kv_q8" }
    public var isFusable: Bool { false }
    public var dispatchDimension: MetalDispatchDimension {
        .elementwise(count: (totalElements + groupSize - 1) / groupSize)
    }

    public init(totalElements: Int, groupSize: Int, bytesPerBlock: Int) {
        self.totalElements = totalElements
        self.groupSize = groupSize
        self.bytesPerBlock = bytesPerBlock
    }
}

/// Per-head RMS normalization for QK normalization in attention.
///
/// Applied to Q and K vectors independently after projection and before RoPE.
/// Each head's headDimension elements are normalized independently.
/// One dispatch normalizes either all Q heads or all K heads.
public struct QKNormOperation: MetalComputeOperation {
    public let headCount: Int
    public let headDimension: Int
    public let epsilon: Float
    /// Weight role used to resolve the norm weight from parameter bindings.
    /// "q_layernorm" for Q norm, "k_layernorm" for K norm.
    public let weightRole: String
    public var kernelName: String { "qk_rms_norm" }
    public var isFusable: Bool { false }
    public var dispatchDimension: MetalDispatchDimension { .perHead(headCount: headCount) }

    public init(headCount: Int, headDimension: Int, epsilon: Float, weightRole: String) {
        self.headCount = headCount
        self.headDimension = headDimension
        self.epsilon = epsilon
        self.weightRole = weightRole
    }
}

// MARK: - Fused Compute Operations

/// A fused kernel combining multiple operations into a single dispatch.
/// Conforms to MetalComputeOperation so the compiler treats it uniformly.
public protocol FusedComputeOperation: MetalComputeOperation {
    /// The source operations that were fused (for debugging/verification).
    var sourceOperations: [any MetalComputeOperation] { get }
}

public struct ResidualAddCopyRMSNormOperation: FusedComputeOperation {
    public let dimension: Int
    public let epsilon: Float
    public var kernelName: String { "fused_residual_add_copy_rms_norm" }
    public var isFusable: Bool { false }
    public var dispatchDimension: MetalDispatchDimension { .reduction(dimension: dimension) }
    public var sourceOperations: [any MetalComputeOperation] { [] }

    public init(dimension: Int, epsilon: Float) {
        self.dimension = dimension
        self.epsilon = epsilon
    }
}

public struct CopyRMSNormOperation: FusedComputeOperation {
    public let dimension: Int
    public let epsilon: Float
    public var kernelName: String { "fused_copy_rms_norm" }
    public var isFusable: Bool { false }
    public var dispatchDimension: MetalDispatchDimension { .reduction(dimension: dimension) }
    public var sourceOperations: [any MetalComputeOperation] { [] }

    public init(dimension: Int, epsilon: Float) {
        self.dimension = dimension
        self.epsilon = epsilon
    }
}

// MARK: - Dispatch Declaration

/// A single step in the dispatch sequence declared by a MetalComponent.
/// Either a GEMV projection or a non-GEMV compute operation.
public enum MetalDispatchDeclaration: Sendable {
    /// A linear projection (GEMV).
    case projection(MetalProjection)
    /// A non-GEMV compute operation.
    case compute(any MetalComputeOperation)
}

// MARK: - MetalComponent Protocol

/// Protocol for IR operations that declare their Metal compute requirements.
///
/// MetalComponent is a **declaration-only** protocol. It tells the compiler
/// WHAT the operation needs, not HOW to compute it.
///
/// The **compiler** owns:
///   - All kernel source code (MSL)
///   - Dispatch order and sequencing
///   - Fusion decisions (which adjacent ops to merge)
///   - Buffer routing and scratch allocation
///   - Pipeline selection (fused vs unfused variants)
public protocol MetalComponent: Sendable {

    /// Ordered sequence of dispatches this operation requires.
    /// Each step is either a projection (GEMV) or a compute operation.
    /// The compiler processes these in order to build the dispatch list.
    var dispatchDeclarations: [MetalDispatchDeclaration] { get }

    /// Weight slot declarations for the compiler to resolve from safetensors.
    var weightSlots: [MetalWeightSlot] { get }

    /// Cache slot declarations (KV cache, conv cache, recurrent state).
    var cacheSlots: [MetalCacheSlot] { get }
}

// MARK: - Derived Properties

extension MetalComponent {
    /// All projections extracted from dispatchDeclarations.
    public var projections: [MetalProjection] {
        dispatchDeclarations.compactMap {
            if case .projection(let projection) = $0 { return projection }
            return nil
        }
    }
}

// MARK: - Projection Declaration

/// A linear projection (GEMV) that this operation needs.
public struct MetalProjection: Sendable {
    /// Field name in the IR path (e.g., "q_proj", "gate_proj").
    public let field: String
    /// Input dimension.
    public let inputDimension: Int
    /// Output dimension.
    public let outputDimension: Int

    public init(field: String, inputDimension: Int, outputDimension: Int) {
        self.field = field
        self.inputDimension = inputDimension
        self.outputDimension = outputDimension
    }
}

// MARK: - Weight / Cache Slots

public struct MetalWeightSlot: Sendable {
    public let field: String?
    public let role: MetalWeightRole

    public init(field: String? = nil, role: MetalWeightRole) {
        self.field = field
        self.role = role
    }
}

public struct MetalCacheSlot: Sendable {
    public let name: String
    public let kind: MetalCacheKind

    public init(name: String, kind: MetalCacheKind = .kv) {
        self.name = name
        self.kind = kind
    }
}

public enum MetalCacheKind: Sendable {
    case kv
    case conv
    case recurrent
}

public enum MetalWeightRole: Sendable {
    case weight
    case scale
    case embeddingTable
}

public enum SynchronizationKind: Sendable {
    case none
    case bufferBarrier
}

// MARK: - Defaults

extension MetalComponent {
    public var cacheSlots: [MetalCacheSlot] { [] }
}
