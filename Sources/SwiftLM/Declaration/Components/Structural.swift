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
public struct Residual: ModelComponent {

    public let strategy: ResidualStrategy
    public let body: ModelDeclaration

    public init(
        strategy: ResidualStrategy = .add,
        @ModelComponentBuilder content: () -> ModelDeclaration
    ) {
        self.strategy = strategy
        self.body = content()
    }

    public func makeDeclaration() -> ModelDeclaration {
        .residual(strategy: strategy, body: body)
    }
}

/// Result builder for `Parallel` branches.
///
/// Each expression in the builder block becomes an independent branch.
/// Unlike `ModelComponentBuilder`, this builder does NOT compose expressions
/// sequentially — each expression produces a separate `ModelDeclaration`
/// branch.
@resultBuilder
public enum ParallelBranchBuilder {

    public static func buildExpression(_ component: any ModelComponent) -> [ModelDeclaration] {
        [component.makeDeclaration()]
    }

    public static func buildBlock(_ branches: [ModelDeclaration]...) -> [ModelDeclaration] {
        branches.flatMap { $0 }
    }

    public static func buildOptional(_ branches: [ModelDeclaration]?) -> [ModelDeclaration] {
        branches ?? []
    }

    public static func buildEither(first branches: [ModelDeclaration]) -> [ModelDeclaration] {
        branches
    }

    public static func buildEither(second branches: [ModelDeclaration]) -> [ModelDeclaration] {
        branches
    }

    public static func buildArray(_ branches: [[ModelDeclaration]]) -> [ModelDeclaration] {
        branches.flatMap { $0 }
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
public struct Parallel: ModelComponent {

    public let merge: ParallelMergeStrategy
    public let branches: [ModelDeclaration]

    public init(
        merge: ParallelMergeStrategy = .add,
        @ParallelBranchBuilder content: () -> [ModelDeclaration]
    ) {
        self.merge = merge
        self.branches = content()
    }

    public func makeDeclaration() -> ModelDeclaration {
        .parallel(merge: merge, branches: branches)
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
public struct Repeat: ModelComponent {

    public let count: Int
    public let label: String?
    public let body: ModelDeclaration

    public init(
        count: Int,
        label: String? = nil,
        @ModelComponentBuilder content: () -> ModelDeclaration
    ) {
        precondition(count > 0, "repeat count must be positive")
        self.count = count
        self.label = label
        self.body = content()
    }

    public func makeDeclaration() -> ModelDeclaration {
        .repeating(count: count, label: label, body: body)
    }
}
