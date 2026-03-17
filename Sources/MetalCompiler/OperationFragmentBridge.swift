import LMIR

/// Bridge from IR Operation to MetalKernelFragment.
///
/// The IR holds `any OperationAttributes`. MetalCompiler casts
/// to MetalKernelFragment to walk the fragment tree.
/// Structural operations return nil — compiler handles them directly.
extension Operation {

    /// MetalKernelFragment for fragment-driven compilation.
    public var kernelFragment: (any MetalKernelFragment)? {
        guard case .primitive(let attrs) = kind else { return nil }
        return attrs as? (any MetalKernelFragment)
    }
}
