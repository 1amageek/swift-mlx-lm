import Metal

/// Metal dispatch types used by the compiler.
///
/// Shared types: MetalDispatchDimension, MetalProjection, MetalWeightSlot,
/// MetalCacheSlot, fused operation structs, SynchronizationKind,
/// BufferPrecision, WeightFormat.

// MARK: - Buffer Precision

/// Buffer precision for intermediate values (hidden, scratch, residual).
public enum BufferPrecision: Sendable {
    /// Float16 — used in decode (single token, no accumulation).
    case float16
    /// Float32 — used in prefill (multi-token, prevents accumulation error).
    case float32

    public var metalType: String {
        switch self {
        case .float16: return "half"
        case .float32: return "float"
        }
    }

    public var byteSize: Int {
        switch self {
        case .float16: return 2
        case .float32: return 4
        }
    }
}

// MARK: - Weight Format

/// Weight data format determines how weight bytes are read and converted to float.
public enum WeightFormat: Sendable, Equatable {
    /// Float16 — direct read as half, convert to float.
    case float16
    /// BFloat16 — read as uint16_t, shift left 16 to get float32.
    case bfloat16
    /// Quantized 4-bit with group size.
    case quantized4Bit(groupSize: Int)
    /// Quantized 8-bit with group size.
    case quantized8Bit(groupSize: Int)

    /// MSL type for the weight buffer parameter.
    public var bufferType: String {
        switch self {
        case .float16: return "half"
        case .bfloat16: return "uint16_t"
        case .quantized4Bit, .quantized8Bit: return "uchar"
        }
    }

    /// MSL expression to convert a weight value to float.
    public func readExpression(_ expr: String) -> String {
        switch self {
        case .float16: return "float(\(expr))"
        case .bfloat16: return "bf16_to_float(\(expr))"
        case .quantized4Bit, .quantized8Bit:
            return "dequantize(\(expr))"
        }
    }
}

// MARK: - Kernel Context

/// Context passed to fragment tree traversal for kernel name resolution.
///
/// Carries the buffer precision (F16 decode / F32 prefill) and weight format
/// (from STAF) so that fragments can resolve context-dependent kernel names
/// without hardcoding variants.
public struct KernelContext: Sendable {
    public let bufferPrecision: BufferPrecision
    public let weightFormat: WeightFormat

    public init(bufferPrecision: BufferPrecision, weightFormat: WeightFormat) {
        self.bufferPrecision = bufferPrecision
        self.weightFormat = weightFormat
    }
}

// MARK: - Buffer Binding Context

/// Context provided by the compiler for fragment buffer binding resolution.
///
/// Fragments use this to declare their decode-path buffer layout
/// without knowing the concrete MTLBuffer allocation details.
public struct BufferBindingContext: @unchecked Sendable {
    public let bufferSet: MetalBufferSet
    public let slotDimension: Int
    public let elementSize: Int
    public let kvCacheIndex: Int
    public let convLayerIndex: Int
    public let resolveWeight: (String) -> (buffer: MTLBuffer, offset: Int)

    public init(bufferSet: MetalBufferSet, slotDimension: Int, elementSize: Int,
                kvCacheIndex: Int, convLayerIndex: Int,
                resolveWeight: @escaping (String) -> (buffer: MTLBuffer, offset: Int)) {
        self.bufferSet = bufferSet
        self.slotDimension = slotDimension
        self.elementSize = elementSize
        self.kvCacheIndex = kvCacheIndex
        self.convLayerIndex = convLayerIndex
        self.resolveWeight = resolveWeight
    }
}

/// Buffer bindings declared by a fragment for decode dispatch.
public struct FragmentBindings: @unchecked Sendable {
    public let buffers: [(index: Int, buffer: MTLBuffer, offset: Int)]
    public let bytes: [(index: Int, value: [UInt8])]
    /// Whether this fragment's output goes to hidden (true) or scratch (false).
    public let outputIsHidden: Bool
    /// Whether to reset projection index after this fragment.
    public let resetsProjectionIndex: Bool
    /// Whether this fragment consumes a KV cache layer slot.
    public let consumesKVCacheLayer: Bool
    /// Whether this fragment consumes a conv state layer slot.
    public let consumesConvLayer: Bool

    public init(buffers: [(index: Int, buffer: MTLBuffer, offset: Int)],
                bytes: [(index: Int, value: [UInt8])],
                outputIsHidden: Bool,
                resetsProjectionIndex: Bool = false,
                consumesKVCacheLayer: Bool = false,
                consumesConvLayer: Bool = false) {
        self.buffers = buffers
        self.bytes = bytes
        self.outputIsHidden = outputIsHidden
        self.resetsProjectionIndex = resetsProjectionIndex
        self.consumesKVCacheLayer = consumesKVCacheLayer
        self.consumesConvLayer = consumesConvLayer
    }
}

// MARK: - Binding Helpers

/// Create a bytes binding for a UInt32 constant.
public func uint32Binding(_ index: Int, _ value: UInt32) -> (index: Int, value: [UInt8]) {
    withUnsafeBytes(of: value) { (index: index, value: Array($0)) }
}

/// Create a bytes binding for a Float constant.
public func floatBinding(_ index: Int, _ value: Float) -> (index: Int, value: [UInt8]) {
    withUnsafeBytes(of: value) { (index: index, value: Array($0)) }
}

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

/// Fused operation: residualAdd + RMSNorm → single dispatch (no copy).
/// Used at model end where no next residual is needed.
public struct FusedResidualAddNorm: Sendable {
    public let dimension: Int
    public let epsilon: Float
    public init(dimension: Int, epsilon: Float) {
        self.dimension = dimension
        self.epsilon = epsilon
    }
}

/// Batched projection: multiple GEMV projections in a single dispatch.
/// All projections share the same input but have different weights and outputs.
public struct BatchedProjection: Sendable {
    public struct Entry: Sendable {
        public let field: String
        public let inputDimension: Int
        public let outputDimension: Int
        public init(field: String, inputDimension: Int, outputDimension: Int) {
            self.field = field
            self.inputDimension = inputDimension
            self.outputDimension = outputDimension
        }
    }
    public let projections: [Entry]

    public var totalOutputDimension: Int {
        projections.reduce(0) { $0 + $1.outputDimension }
    }

    public var inputDimension: Int {
        projections[0].inputDimension
    }

    public init(projections: [Entry]) {
        self.projections = projections
    }
}

/// Batched in-place fragments with the same dispatch dimension.
/// The compiler dispatches all instances in a single kernel, routing
/// threadgroups to the correct data/weight buffers.
public struct BatchedFragment: @unchecked Sendable {
    public let fragments: [any PrimitiveMetalKernelFragment]
    public let dispatchDimension: MetalDispatchDimension

    public init(fragments: [any PrimitiveMetalKernelFragment], dispatchDimension: MetalDispatchDimension) {
        self.fragments = fragments
        self.dispatchDimension = dispatchDimension
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
    /// Temporal window size for conv cache (kernelSize), or 0 for non-conv caches.
    public let temporalSize: Int

    public init(name: String, kind: MetalCacheKind = .kv, temporalSize: Int = 0) {
        self.name = name
        self.kind = kind
        self.temporalSize = temporalSize
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
