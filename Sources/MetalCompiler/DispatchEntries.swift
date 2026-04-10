import LMIR

// MARK: - Dispatch Entry

/// A single kernel dispatch in the plan (IR → fusion → steps).
public struct DispatchEntry: Sendable {
    public let index: Int
    public let kind: DispatchKind
    public let parameterBindings: [ParameterBinding]
    public let layerIndex: Int?
    public let compositeID: Int?

    public init(
        index: Int,
        kind: DispatchKind,
        parameterBindings: [ParameterBinding] = [],
        layerIndex: Int? = nil,
        compositeID: Int? = nil
    ) {
        self.index = index
        self.kind = kind
        self.parameterBindings = parameterBindings
        self.layerIndex = layerIndex
        self.compositeID = compositeID
    }
}

/// Kind of dispatch: projection, fragment, fused, batched, or structural.
public enum DispatchKind: Sendable {
    case projection(MetalProjection, isOutput: Bool = false)
    case fragment(any PrimitiveMetalKernelFragment)
    case fusedCopyNorm(FusedCopyNorm)
    case fusedResidualAddCopyNorm(FusedResidualAddCopyNorm)
    case fusedResidualAddNorm(FusedResidualAddNorm)
    case fusedSwiGLUProjection(FusedSwiGLUProjection)
    case batchedProjection(BatchedProjection)
    case batchedFragment(BatchedFragment)
    case structuralCopy(dimension: Int)
    case structuralAdd(dimension: Int)
}
