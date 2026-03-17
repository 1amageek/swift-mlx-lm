/// Metal dispatch types used by the compiler.
///
/// Shared types: MetalDispatchDimension, MetalProjection, MetalWeightSlot,
/// MetalCacheSlot, fused operation structs, SynchronizationKind.

// MARK: - Dispatch Dimension

/// GPU dispatch pattern for grid/threadgroup sizing.
public enum MetalDispatchDimension: Sendable, Equatable {
    /// Single threadgroup reduces across the dimension (RMSNorm, Argmax).
    case reduction(dimension: Int)
    /// One thread per element (SwiGLU, SigmoidGate, Copy, Add).
    case elementwise(count: Int)
    /// One threadgroup per head (FlashAttention, RoPE, QKNorm, SSM).
    case perHead(headCount: Int)
    /// Token → embedding gather.
    case gather(count: Int)
    /// GEMV/GEMM projection.
    case gemv(outputDimension: Int, inputDimension: Int)
}

// MARK: - Fused Operations

/// Fused operation: residualAdd + copy + RMSNorm → single dispatch.
public struct FusedResidualAddCopyNorm: Sendable {
    public let dimension: Int
    public let epsilon: Float
    public init(dimension: Int, epsilon: Float) {
        self.dimension = dimension
        self.epsilon = epsilon
    }
}

/// Fused operation: copy + RMSNorm → single dispatch.
public struct FusedCopyNorm: Sendable {
    public let dimension: Int
    public let epsilon: Float
    public init(dimension: Int, epsilon: Float) {
        self.dimension = dimension
        self.epsilon = epsilon
    }
}

// MARK: - Projection

public struct MetalProjection: Sendable {
    public let field: String
    public let inputDimension: Int
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
