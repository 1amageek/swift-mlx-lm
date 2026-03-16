/// Residual connection component.
///
/// Wraps child components with a skip connection: `output = input + f(input)`.
///
/// ```swift
/// Residual {
///     RMSNorm(dimension: 4096)
///     Attention(hiddenSize: 4096, headCount: 32, kvHeadCount: 8)
/// }
/// ```
public struct Residual<Content: ModelComponent>: ModelComponent {

    public typealias Body = Never

    public let strategy: ResidualStrategy
    public let content: Content

    public init(
        strategy: ResidualStrategy = .add,
        @ModelComponentBuilder content: () -> Content
    ) {
        self.strategy = strategy
        self.content = content()
    }
}

/// Parallel execution component.
///
/// Executes multiple branches on the same input and merges results.
///
/// **Branch semantics:** Each direct child expression in the builder block
/// becomes an independent branch. Branches are NOT wired sequentially —
/// they all receive the same upstream input and their outputs are merged
/// by the given strategy.
///
/// ```swift
/// Parallel(merge: .add) {
///     Attention(hiddenSize: 4096, headCount: 32, kvHeadCount: 8)
///     MLP(inputSize: 4096, intermediateSize: 11008)
/// }
/// // -> 2 branches: one Attention, one MLP
/// ```
public struct Parallel<Content: ModelComponent>: ModelComponent {

    public typealias Body = Never

    public let merge: ParallelMergeStrategy
    public let content: Content

    public init(
        merge: ParallelMergeStrategy = .add,
        @ModelComponentBuilder content: () -> Content
    ) {
        self.merge = merge
        self.content = content()
    }
}

/// Repeat component.
///
/// Repeats a block of components a fixed number of times.
/// Used for stacking identical transformer layers.
///
/// ```swift
/// Repeat(count: 32) {
///     TransformerBlock(hiddenSize: 4096, headCount: 32, kvHeadCount: 8)
/// }
/// ```
public struct Repeat<Content: ModelComponent>: ModelComponent {

    public typealias Body = Never

    public let count: Int
    public let label: String?
    public let content: Content

    public init(
        count: Int,
        label: String? = nil,
        @ModelComponentBuilder content: () -> Content
    ) {
        precondition(count > 0, "repeat count must be positive")
        self.count = count
        self.label = label
        self.content = content()
    }
}
