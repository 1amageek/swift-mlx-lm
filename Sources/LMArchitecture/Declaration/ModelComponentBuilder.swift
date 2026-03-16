/// Result builder for composing `ModelComponent` values with type safety.
///
/// Follows the SwiftUI `@ViewBuilder` pattern: the builder returns
/// concrete generic types (`TupleComponent`, `OptionalComponent`,
/// `ConditionalComponent`) instead of type-erased arrays.
///
/// ```swift
/// @ModelComponentBuilder
/// var body: some ModelComponent {
///     TokenEmbedding(vocabSize: 32000, embeddingSize: 4096)
///     Repeat(count: 32) {
///         TransformerBlock(...)
///     }
///     RMSNorm(dimension: 4096)
///     OutputHead(inputSize: 4096, vocabSize: 32000)
/// }
/// ```
@resultBuilder
public enum ModelComponentBuilder {

    // MARK: - Single Component (identity)

    public static func buildBlock<C: ModelComponent>(_ component: C) -> C {
        component
    }

    // MARK: - Multiple Components (tuple)

    public static func buildBlock<each C: ModelComponent>(
        _ components: repeat each C
    ) -> TupleComponent<repeat each C> {
        TupleComponent(repeat each components)
    }

    // MARK: - Optional (if without else)

    public static func buildOptional<C: ModelComponent>(
        _ component: C?
    ) -> OptionalComponent<C> {
        OptionalComponent(content: component)
    }

    // MARK: - Conditional (if/else)

    public static func buildEither<First: ModelComponent, Second: ModelComponent>(
        first component: First
    ) -> ConditionalComponent<First, Second> {
        ConditionalComponent(storage: .first(component))
    }

    public static func buildEither<First: ModelComponent, Second: ModelComponent>(
        second component: Second
    ) -> ConditionalComponent<First, Second> {
        ConditionalComponent(storage: .second(component))
    }
}
