/// Result builder for composing `ModelComponent` values into a declaration tree.
///
/// The builder converts each component expression into a `ModelDeclaration`
/// via `makeDeclaration()`, then combines them into a `.sequence(...)`.
/// No graph is built — only a pure declaration value is produced.
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

    public static func buildExpression(_ component: any ModelComponent) -> ModelDeclaration {
        component.makeDeclaration()
    }

    public static func buildBlock(_ declarations: ModelDeclaration...) -> ModelDeclaration {
        if declarations.count == 1 { return declarations[0] }
        return .sequence(declarations)
    }

    public static func buildOptional(_ declaration: ModelDeclaration?) -> ModelDeclaration {
        declaration ?? .sequence([])
    }

    public static func buildEither(first declaration: ModelDeclaration) -> ModelDeclaration {
        declaration
    }

    public static func buildEither(second declaration: ModelDeclaration) -> ModelDeclaration {
        declaration
    }

    public static func buildArray(_ declarations: [ModelDeclaration]) -> ModelDeclaration {
        .sequence(declarations)
    }
}
