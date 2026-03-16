/// Parameter tensors bound to semantic slots defined by the model graph.
///
/// Produced by `WeightBinder` after mapping raw checkpoint tensor names
/// to parameter slots. Ready for consumption by the compiler.
public struct BoundWeights: Sendable {

    /// Tensors keyed by their semantic parameter slot.
    public let tensors: [ParameterSlot: TensorData]

    public init(tensors: [ParameterSlot: TensorData]) {
        self.tensors = tensors
    }
}
