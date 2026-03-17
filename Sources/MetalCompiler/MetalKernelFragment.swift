/// A declarative protocol for describing Metal kernel computations.
///
/// `MetalKernelFragment` mirrors `ModelComponent` — both form a compositional tree.
/// - `ModelComponent` tree → IR (structure validation, serialization)
/// - `MetalKernelFragment` tree → Metal dispatch plan (execution)
///
/// Every component conforms to both protocols:
/// ```swift
/// struct RMSNorm: ModelComponent, MetalKernelFragment {
///     var body: some ModelComponent { ... }      // IR structure
///     func fragment(context: KernelContext) -> some MetalKernelFragment { ... } // Metal execution
/// }
/// ```
///
/// The Metal compiler walks the fragment tree to:
/// 1. Generate MSL kernel source (with correct dtype/precision)
/// 2. Determine buffer routing (from structural fragments)
/// 3. Fuse adjacent fragments (eliminating intermediate device memory R/W)
public protocol MetalKernelFragment: Sendable {

    /// The type of the fragment body.
    associatedtype Fragment: MetalKernelFragment

    /// The fragment body describing how this operation executes on Metal.
    @MetalKernelFragmentBuilder
    func fragment(context: KernelContext) -> Fragment

    /// Whether this fragment can be fused with adjacent fragments.
    ///
    /// The compiler checks `isFusable` at every node in the fragment tree
    /// to decide if intermediate device memory reads/writes can be eliminated
    /// by keeping values in registers.
    var isFusable: Bool { get }
}

// MARK: - Default Implementations

extension MetalKernelFragment {
    public var isFusable: Bool { false }
}

extension Never: MetalKernelFragment {
    public func fragment(context: KernelContext) -> Never { fatalError() }
}
