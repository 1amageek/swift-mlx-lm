import LMArchitecture

/// Learned scalar multiply applied to the current hidden state.
public struct LayerScale: ModelComponent {

    public typealias Attributes = LayerScaleAttributes

    public let dimension: Int

    public init(dimension: Int) {
        precondition(dimension > 0, "dimension must be positive")
        self.dimension = dimension
    }
}

extension LayerScale {

    public var attributes: LayerScaleAttributes {
        LayerScaleAttributes(dimension: dimension)
    }
}
