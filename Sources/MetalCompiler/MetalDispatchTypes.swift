import LMIR
import Metal

/// Metal dispatch types used by the compiler.
///
/// Shared types: MetalDispatchDimension, MetalWeightSlot,
/// MetalCacheSlot, fused operation structs, SynchronizationKind,
/// BufferPrecision, WeightFormat.

// MARK: - Buffer Precision

/// Buffer precision for intermediate values (hidden, scratch, residual).
public enum BufferPrecision: Sendable {
    /// Float16 — used in decode (single token, no accumulation).
    case float16
    /// BFloat16 — used in decode for BF16-native models.
    case bfloat16
    /// Float32 — used in prefill (multi-token, prevents accumulation error).
    case float32

    public var metalType: String {
        switch self {
        case .float16: return "half"
        case .bfloat16: return "bfloat"
        case .float32: return "float"
        }
    }

    public var byteSize: Int {
        switch self {
        case .float16: return 2
        case .bfloat16: return 2
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
    /// Float32 — direct read as float.
    case float32
    /// Quantized 2-bit (aligned) with group size.
    case quantized2Bit(groupSize: Int)
    /// Quantized 3-bit (non-aligned: 8 weights per 3 bytes) with group size.
    case quantized3Bit(groupSize: Int)
    /// Quantized 4-bit (aligned) with group size.
    case quantized4Bit(groupSize: Int)
    /// Quantized 5-bit (non-aligned: 8 weights per 5 bytes) with group size.
    case quantized5Bit(groupSize: Int)
    /// Quantized 6-bit (non-aligned: 4 weights per 3 bytes) with group size.
    case quantized6Bit(groupSize: Int)
    /// Quantized 8-bit (aligned) with group size.
    case quantized8Bit(groupSize: Int)

    /// MSL type for the weight buffer parameter.
    public var bufferType: String {
        switch self {
        case .float16: return "half"
        case .bfloat16: return "uint16_t"
        case .float32: return "float"
        case .quantized2Bit, .quantized3Bit, .quantized4Bit,
             .quantized5Bit, .quantized6Bit, .quantized8Bit:
            return "uchar"
        }
    }

    /// MSL expression to convert a weight value to float.
    ///
    /// Only valid for dense formats (float16 / bfloat16 / float32). Block-packed
    /// quantized formats cannot produce a single-element read expression because
    /// they require per-block scale/zero lookup; they must be routed to dedicated
    /// quantized kernels (`gemv_q*`), not a dense GEMV/GEMM template.
    public func readExpression(_ expr: String) -> String {
        switch self {
        case .float16: return "float(\(expr))"
        case .bfloat16: return "bf16_to_float(\(expr))"
        case .float32: return "(\(expr))"
        case .quantized2Bit, .quantized3Bit, .quantized4Bit,
             .quantized5Bit, .quantized6Bit, .quantized8Bit:
            fatalError("WeightFormat.readExpression called with quantized format \(self); quantized weights must be routed to a dedicated quantized kernel, not a dense GEMV/GEMM template. The caller should pass `effectiveWeightFormat` (post dequant→BF16 substitution) rather than the original quantized `weightFormat`.")
        }
    }

    var isQuantized: Bool {
        switch self {
        case .quantized2Bit, .quantized3Bit, .quantized4Bit,
             .quantized5Bit, .quantized6Bit, .quantized8Bit:
            return true
        case .float16, .bfloat16, .float32:
            return false
        }
    }

    var storageByteSize: Int {
        switch self {
        case .float16, .bfloat16:
            return MemoryLayout<UInt16>.stride
        case .float32:
            return MemoryLayout<Float>.stride
        case .quantized2Bit, .quantized3Bit, .quantized4Bit,
             .quantized5Bit, .quantized6Bit, .quantized8Bit:
            return MemoryLayout<UInt8>.stride
        }
    }

    /// Bit width for quantized formats (nil for dense).
    var quantizationBits: Int? {
        switch self {
        case .quantized2Bit: return 2
        case .quantized3Bit: return 3
        case .quantized4Bit: return 4
        case .quantized5Bit: return 5
        case .quantized6Bit: return 6
        case .quantized8Bit: return 8
        case .float16, .bfloat16, .float32: return nil
        }
    }

    /// Group size for quantized formats (nil for dense).
    var quantizationGroupSize: Int? {
        switch self {
        case .quantized2Bit(let g), .quantized3Bit(let g),
             .quantized4Bit(let g), .quantized5Bit(let g),
             .quantized6Bit(let g), .quantized8Bit(let g):
            return g
        case .float16, .bfloat16, .float32: return nil
        }
    }

    /// Resolve the protocol-driven `QuantizationFormat` instance for this enum case.
    ///
    /// Returns nil for dense formats and for quantized (bits, groupSize) pairs that
    /// the registry does not recognise. Used by the catalog and prefill builder to
    /// select kernels through `QuantizationFormat` properties instead of switching
    /// on enum cases — the long-term Phase 5 migration path.
    var quantizationFormat: (any QuantizationFormat)? {
        switch self {
        case .quantized2Bit(let g):
            return QuantizationFormatRegistry.formatForMLXQuantization(bits: 2, groupSize: g)
        case .quantized3Bit(let g):
            return QuantizationFormatRegistry.formatForMLXQuantization(bits: 3, groupSize: g)
        case .quantized4Bit(let g):
            return QuantizationFormatRegistry.formatForMLXQuantization(bits: 4, groupSize: g)
        case .quantized5Bit(let g):
            return QuantizationFormatRegistry.formatForMLXQuantization(bits: 5, groupSize: g)
        case .quantized6Bit(let g):
            return QuantizationFormatRegistry.formatForMLXQuantization(bits: 6, groupSize: g)
        case .quantized8Bit(let g):
            return QuantizationFormatRegistry.formatForMLXQuantization(bits: 8, groupSize: g)
        case .float16, .bfloat16, .float32:
            return nil
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
    public let currentInputBuffer: MTLBuffer
    public let currentInputOffset: Int
    public let layerIndex: Int?
    public let kvCacheIndex: Int
    public let convLayerIndex: Int
    public let recurrentLayerIndex: Int
    /// Current projection index for scratch slot allocation.
    /// Projection outputs use scratch slot `projectionIndex + 1`.
    public let projectionIndex: Int
    public let resolveWeight: (String) -> (buffer: MTLBuffer, offset: Int)

    public init(bufferSet: MetalBufferSet, slotDimension: Int, elementSize: Int,
                currentInputBuffer: MTLBuffer, currentInputOffset: Int,
                layerIndex: Int?, kvCacheIndex: Int, convLayerIndex: Int, recurrentLayerIndex: Int,
                projectionIndex: Int = 0,
                resolveWeight: @escaping (String) -> (buffer: MTLBuffer, offset: Int)) {
        self.bufferSet = bufferSet
        self.slotDimension = slotDimension
        self.elementSize = elementSize
        self.currentInputBuffer = currentInputBuffer
        self.currentInputOffset = currentInputOffset
        self.layerIndex = layerIndex
        self.kvCacheIndex = kvCacheIndex
        self.convLayerIndex = convLayerIndex
        self.recurrentLayerIndex = recurrentLayerIndex
        self.projectionIndex = projectionIndex
        self.resolveWeight = resolveWeight
    }
}

/// Buffer bindings declared by a fragment for decode dispatch.
public struct FragmentBindings: @unchecked Sendable {
    // @unchecked: contains MTLBuffer (Metal protocol, not Sendable)
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
    /// Whether this fragment consumes a recurrent state layer slot.
    public let consumesRecurrentLayer: Bool
    /// Buffer access pattern for barrier optimization.
    /// Indices into the `buffers` array that are written by this fragment.
    /// Buffers not listed here are treated as read-only.
    /// When nil, the barrier optimizer falls back to conservative (all read+write).
    public let writeBufferIndices: Set<Int>?

    /// Number of projection scratch slots consumed by this fragment.
    /// The routing planner advances projectionIndex by this amount.
    /// Default 0 for non-projection fragments.
    public let projectionSlotsConsumed: Int

    public init(buffers: [(index: Int, buffer: MTLBuffer, offset: Int)],
                bytes: [(index: Int, value: [UInt8])],
                outputIsHidden: Bool,
                resetsProjectionIndex: Bool = false,
                consumesKVCacheLayer: Bool = false,
                consumesConvLayer: Bool = false,
                consumesRecurrentLayer: Bool = false,
                writeBufferIndices: Set<Int>? = nil,
                projectionSlotsConsumed: Int = 0) {
        self.buffers = buffers
        self.bytes = bytes
        self.outputIsHidden = outputIsHidden
        self.resetsProjectionIndex = resetsProjectionIndex
        self.consumesKVCacheLayer = consumesKVCacheLayer
        self.consumesConvLayer = consumesConvLayer
        self.consumesRecurrentLayer = consumesRecurrentLayer
        self.writeBufferIndices = writeBufferIndices
        self.projectionSlotsConsumed = projectionSlotsConsumed
    }
}

// MARK: - Prefill Binding Context

/// Context provided by the compiler for fragment prefill step generation.
public struct PrefillBindingContext: @unchecked Sendable {
    // @unchecked: contains MTLBuffer (via PrefillBufferSet) and MTLComputePipelineState
    public let buffers: PrefillBufferSet
    public let slotDimension: Int
    public let scratchElementSize: Int
    public let maximumSequenceLength: Int
    public let currentInputBuffer: MTLBuffer
    public let currentInputOffset: Int
    public let layerIndex: Int?
    public let kvCacheIndex: Int
    public let convLayerIndex: Int
    public let recurrentLayerIndex: Int
    /// Current projection index for scratch slot allocation.
    public let projectionIndex: Int
    public let kernelContext: KernelContext
    public let resolveWeight: (String) -> (buffer: MTLBuffer, offset: Int)
    public let getPipeline: (String) throws -> MTLComputePipelineState

    public init(buffers: PrefillBufferSet, slotDimension: Int, scratchElementSize: Int,
                maximumSequenceLength: Int, currentInputBuffer: MTLBuffer, currentInputOffset: Int,
                layerIndex: Int?, kvCacheIndex: Int, convLayerIndex: Int, recurrentLayerIndex: Int,
                projectionIndex: Int = 0,
                kernelContext: KernelContext,
                resolveWeight: @escaping (String) -> (buffer: MTLBuffer, offset: Int),
                getPipeline: @escaping (String) throws -> MTLComputePipelineState) {
        self.buffers = buffers
        self.slotDimension = slotDimension
        self.scratchElementSize = scratchElementSize
        self.maximumSequenceLength = maximumSequenceLength
        self.currentInputBuffer = currentInputBuffer
        self.currentInputOffset = currentInputOffset
        self.layerIndex = layerIndex
        self.kvCacheIndex = kvCacheIndex
        self.convLayerIndex = convLayerIndex
        self.recurrentLayerIndex = recurrentLayerIndex
        self.projectionIndex = projectionIndex
        self.kernelContext = kernelContext
        self.resolveWeight = resolveWeight
        self.getPipeline = getPipeline
    }
}

/// Prefill steps declared by a fragment.
public struct FragmentPrefillSteps: @unchecked Sendable {
    // @unchecked: contains [MetalPrefillStep] which has MTLBuffer/MTLComputePipelineState
    public let steps: [MetalPrefillStep]
    public let outputIsHidden: Bool
    public let resetsProjectionIndex: Bool
    public let consumesKVCacheLayer: Bool
    public let consumesConvLayer: Bool
    public let consumesRecurrentLayer: Bool

    /// Number of projection scratch slots consumed by this fragment.
    /// The prefill routing planner advances projectionIndex by this amount.
    public let projectionSlotsConsumed: Int

    public init(steps: [MetalPrefillStep], outputIsHidden: Bool,
                resetsProjectionIndex: Bool = false,
                consumesKVCacheLayer: Bool = false,
                consumesConvLayer: Bool = false,
                consumesRecurrentLayer: Bool = false,
                projectionSlotsConsumed: Int = 0) {
        self.steps = steps
        self.outputIsHidden = outputIsHidden
        self.resetsProjectionIndex = resetsProjectionIndex
        self.consumesKVCacheLayer = consumesKVCacheLayer
        self.consumesConvLayer = consumesConvLayer
        self.consumesRecurrentLayer = consumesRecurrentLayer
        self.projectionSlotsConsumed = projectionSlotsConsumed
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
    /// Multiple independent threadgroups, each reducing over its own partition
    /// (SSM recurrence: one threadgroup per key-group, disjoint conv channels
    /// and recurrent state slices). `threadsPerPartition` is clamped to the
    /// pipeline's maxTotalThreadsPerThreadgroup at dispatch time.
    case partitionedReduction(partitionCount: Int, threadsPerPartition: Int)
}

// MARK: - Batched Operations

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
public struct BatchedFragment: Sendable {
    public let fragments: [any PrimitiveMetalKernelFragment]
    public let dispatchDimension: MetalDispatchDimension

    public init(fragments: [any PrimitiveMetalKernelFragment], dispatchDimension: MetalDispatchDimension) {
        self.fragments = fragments
        self.dispatchDimension = dispatchDimension
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
    /// KV head count (for .kv caches).
    public let kvHeadCount: Int
    /// Head dimension (for .kv caches).
    public let headDimension: Int

    public init(name: String, kind: MetalCacheKind = .kv, temporalSize: Int = 0,
                kvHeadCount: Int = 0, headDimension: Int = 0) {
        self.name = name
        self.kind = kind
        self.temporalSize = temporalSize
        self.kvHeadCount = kvHeadCount
        self.headDimension = headDimension
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
