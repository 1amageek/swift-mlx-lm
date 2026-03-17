/// Primitive Metal kernel fragments — leaf nodes of the fragment tree.
///
/// Each primitive fragment is a single dispatch unit. The compiler:
/// 1. Reads fragment parameters (dimension, epsilon, etc.)
/// 2. Determines buffer precision (F16 decode / F32 prefill) and weight format (from STAF)
/// 3. Calls MetalSourceGenerator to produce MSL on-demand
///
/// No hardcoded kernel names. The compiler derives names from fragment type + context.
///
/// Individual fragment types are in the Primitives/ subdirectory (1 file per type).

// MARK: - Primitive Protocol

/// Leaf fragment that the compiler translates into a single Metal dispatch.
///
/// The compiler uses these properties for generic graph optimization:
/// - `dispatchDimension`: kernel scaffold and batching strategy
/// - `isFusable`: whether this fragment participates in compiler optimizations
/// - `isInPlace`: whether this fragment modifies its primary buffer in-place
/// - `kernelName(context:)`: context-aware kernel name resolution
/// - `kernelBody()`: composable MSL computation body for kernel composition
///
/// The optimizer batches fragments based on properties alone — no type checks.
/// Within a composite fragment, consecutive in-place fragments with the same
/// dispatchDimension type are independent and batchable.
public protocol PrimitiveMetalKernelFragment: MetalKernelFragment where Fragment == Never {
    /// GPU dispatch pattern (determines grid/threadgroup sizing).
    var dispatchDimension: MetalDispatchDimension { get }
    /// Weight tensors this fragment reads from STAF.
    var weightSlots: [MetalWeightSlot] { get }
    /// Persistent cache slots (KV cache, conv state).
    var cacheSlots: [MetalCacheSlot] { get }

    /// Whether this fragment modifies its primary data buffer in-place.
    ///
    /// In-place fragments within the same composite fragment that have different
    /// data sources (different preceding projection outputs) are independent.
    /// The optimizer uses this to determine batchability.
    var isInPlace: Bool { get }

    /// Epsilon value for normalization-type fragments.
    /// nil for fragments that don't perform normalization.
    var normEpsilon: Float? { get }

    /// Kernel name for this fragment, resolved using the kernel context.
    ///
    /// Returns the correct kernel name for both decode (F16) and prefill (F32)
    /// based on `context.bufferPrecision` and `context.weightFormat`.
    func kernelName(context: KernelContext) -> String

    /// Declare buffer bindings for decode dispatch.
    ///
    /// The compiler provides a context with buffer set, slot dimensions,
    /// and weight resolution. The fragment returns its concrete bindings
    /// and routing state updates.
    func decodeBindings(context: BufferBindingContext) -> FragmentBindings

    /// Build prefill steps for this fragment.
    ///
    /// The compiler provides a context with prefill buffers, slot dimensions,
    /// pipeline cache, and kernel context. The fragment returns its prefill
    /// steps (batch, perPosition, or lastToken mode).
    func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps

    /// Generate the composable MSL computation body for this fragment.
    ///
    /// Fusable fragments return a body using standardized variable names
    /// based on `dispatchDimension`. The compiler wraps the body in a kernel
    /// scaffold (dispatch routing, batching, structural prefix).
    ///
    /// Standard variables by dispatch dimension:
    /// - `.reduction`: data (R), weight (R), output (W), dimension, epsilon,
    ///                 tid, threadgroupSize, shared (threadgroup float[32])
    /// - `.perHead`:   data (R/W), weight (R), headDimension, epsilon,
    ///                 offset (= headIndex * headDimension), tid, threadgroupSize, shared
    /// - `.elementwise`: input0, input1, output, count, gid
    ///
    /// Returns nil for non-fusable (opaque) fragments.
    /// The compiler calls `kernelSource()` instead.
    func kernelBody(
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String?

    /// Generate the complete MSL kernel for non-fusable (opaque) fragments.
    ///
    /// Called only when `kernelBody()` returns nil. The compiler uses the
    /// returned source as-is, without composition or wrapping.
    func kernelSource(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String
}

extension PrimitiveMetalKernelFragment {
    public func fragment(context: KernelContext) -> Never { fatalError() }
    public var weightSlots: [MetalWeightSlot] { [] }
    public var cacheSlots: [MetalCacheSlot] { [] }
    public var isInPlace: Bool { false }
    public var normEpsilon: Float? { nil }
    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        fatalError("Fragment \(type(of: self)) must implement decodeBindings(context:)")
    }
    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        fatalError("Fragment \(type(of: self)) must implement prefillSteps(context:)")
    }
    public func kernelBody(bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String? { nil }
    public func kernelSource(name: String, bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String {
        fatalError("Fragment \(type(of: self)) must implement either kernelBody() or kernelSource()")
    }
}
