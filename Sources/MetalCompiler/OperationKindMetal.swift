import LMIR

/// Resolve MetalComponent from an Operation's opaque attributes.
///
/// The IR holds `any OperationAttributes`. MetalCompiler casts
/// to `MetalComponent` to get Metal-specific declarations.
/// Structural operations return nil — compiler handles them directly.
extension Operation {

    public var metalComponent: (any MetalComponent)? {
        guard case .primitive(let attrs) = kind else { return nil }
        return attrs as? (any MetalComponent)
    }
}
