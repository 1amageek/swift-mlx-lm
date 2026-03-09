/// Result builder for composing `WeightsSpec` values into a declaration tree.
///
/// The builder converts each weight spec expression into a `WeightsDeclaration`
/// via `makeDeclaration()`, then combines them via `.merge(...)`.
///
/// ```swift
/// @WeightsBuilder
/// var weights: some WeightsSpec {
///     WeightsDeclaration.gguf(location: "base.gguf")
///     WeightsDeclaration.override(
///         base: .gguf(location: "base.gguf"),
///         with: .safetensors(directory: "adapter/", indexFile: nil)
///     )
/// }
/// ```
@resultBuilder
public enum WeightsBuilder {

    public static func buildExpression(_ spec: any WeightsSpec) -> WeightsDeclaration {
        spec.makeDeclaration()
    }

    public static func buildBlock(_ declarations: WeightsDeclaration...) -> WeightsDeclaration {
        if declarations.count == 1 { return declarations[0] }
        return .merge(declarations)
    }

    public static func buildOptional(_ declaration: WeightsDeclaration?) -> WeightsDeclaration {
        declaration ?? .empty
    }

    public static func buildEither(first declaration: WeightsDeclaration) -> WeightsDeclaration {
        declaration
    }

    public static func buildEither(second declaration: WeightsDeclaration) -> WeightsDeclaration {
        declaration
    }

    public static func buildArray(_ declarations: [WeightsDeclaration]) -> WeightsDeclaration {
        .merge(declarations)
    }
}
