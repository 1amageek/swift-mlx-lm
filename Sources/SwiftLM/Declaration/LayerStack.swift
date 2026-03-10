/// Heterogeneous layer stack component.
///
/// Produces a `.layerStack(layers: [Region])` IR node where each layer
/// can have a different structure. Used for models like Qwen 3.5 where
/// layers alternate between different attention/state-space mechanisms.
///
/// ```swift
/// LayerStack(0..<24) { layerIndex in
///     if isFullAttentionLayer(layerIndex) {
///         Residual { RMSNorm(...); Attention(...) }
///         Residual { RMSNorm(...); MLP(...) }
///     } else {
///         Residual { RMSNorm(...); StateSpace(...) }
///         Residual { RMSNorm(...); MLP(...) }
///     }
/// }
/// ```
public struct LayerStack<Content: ModelComponent>: ModelComponent {

    public typealias Body = Never

    package let children: [Content]

    public init<Data: Collection>(
        _ data: Data,
        @ModelComponentBuilder content: (Data.Element) -> Content
    ) {
        self.children = data.map { content($0) }
    }
}
