/// Heterogeneous layer stack.
///
/// Defines a sequence of decoder layers where each layer can have
/// a different structure. Each layer is built independently via
/// the closure, which receives the layer descriptor as input.
///
/// In the IR, each layer becomes a `repeating(count: 1)` node to
/// preserve layer index tracking for weight path resolution.
///
/// ```swift
/// LayerStack(layerDescriptors) { descriptor in
///     if descriptor.isConvolution {
///         ConvDecoderLayer(config: config)
///     } else {
///         AttnDecoderLayer(config: config)
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
