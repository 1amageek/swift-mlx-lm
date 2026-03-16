/// Data-driven iteration component, analogous to SwiftUI's `ForEach`.
///
/// Iterates over a collection and produces a typed component for each element.
/// The concrete component type from the builder closure is preserved — no
/// type erasure occurs.
///
/// ```swift
/// ForEach(config.sections) { section in
///     Repeat(count: section.layerCount) {
///         DecoderLayer(hiddenSize: section.hiddenSize)
///     }
/// }
/// ```
public struct ForEach<Content: ModelComponent>: ModelComponent {

    public typealias Body = Never

    package let children: [Content]

    public init<Data: Collection>(
        _ data: Data,
        @ModelComponentBuilder content: (Data.Element) -> Content
    ) {
        self.children = data.map { content($0) }
    }
}
