/// A declarative specification of weight origin and composition.
///
/// `WeightsSpec` describes where weights come from and how they are composed.
/// It is NOT an eager loader — resolution into actual tensors happens later
/// via `WeightsResolver`.
///
/// Like `ModelComponent`, `WeightsSpec` produces a pure declaration value
/// (`WeightsDeclaration`) that can be inspected, compared, cached, and
/// resolved independently.
///
/// ```swift
/// var weights: some WeightsSpec {
///     GGUFWeightsSpec(file: modelFile)
/// }
///
/// var weights: some WeightsSpec {
///     Merge {
///         GGUFWeightsSpec(file: baseModel)
///         LoRAWeightsSpec(file: adapter)
///     }
/// }
/// ```
public protocol WeightsSpec: Sendable {

    /// Produce a pure declaration value describing this weight specification.
    func makeDeclaration() -> WeightsDeclaration
}

/// Closed, canonical weight declaration IR.
///
/// `WeightsDeclaration` is the weights-side counterpart to `ModelDeclaration`.
/// All weight front-ends (GGUF spec, safetensors spec, random init, LoRA overlay)
/// produce `WeightsDeclaration` values. The `WeightsResolver` then converts them
/// into concrete `RawWeights`.
///
/// ```text
/// WeightsSpec --makeDeclaration()--> WeightsDeclaration --resolve()--> RawWeights
/// ```
public indirect enum WeightsDeclaration: WeightsSpec, Codable, Equatable, Sendable {

    /// No weights (identity element for merge).
    case empty

    /// Weights from a GGUF file.
    case gguf(location: String)

    /// Weights from a safetensors directory.
    case safetensors(directory: String, indexFile: String?)

    /// Random initialization with a given seed and scheme.
    case random(seed: UInt64, scheme: InitializationScheme)

    /// Merge multiple weight sources (later sources override earlier ones).
    case merge([WeightsDeclaration])

    /// Override specific weights from a base source.
    case override(base: WeightsDeclaration, with: WeightsDeclaration)

    /// Apply a transformation to weights from a base source.
    case map(base: WeightsDeclaration, transform: WeightTransform)

    public func makeDeclaration() -> WeightsDeclaration { self }
}

/// Canonicalize a `WeightsDeclaration` for equivalence comparison.
///
/// Normalizes structural patterns:
/// - `.merge([])` → `.empty`
/// - `.merge([single])` → the single element (recursively canonicalized)
/// - `.override(base, with: .empty)` → canonicalized base
/// - Nested merges are flattened and `.empty` elements removed.
public func canonicalize(_ declaration: WeightsDeclaration) -> WeightsDeclaration {
    switch declaration {
    case .empty:
        return .empty

    case .merge(let children):
        var flat: [WeightsDeclaration] = []
        for child in children.map({ canonicalize($0) }) {
            switch child {
            case .empty:
                break
            case .merge(let nested):
                flat.append(contentsOf: nested)
            default:
                flat.append(child)
            }
        }
        if flat.isEmpty { return .empty }
        if flat.count == 1 { return flat[0] }
        return .merge(flat)

    case .override(let base, let overlay):
        let canonOverlay = canonicalize(overlay)
        if canonOverlay == .empty { return canonicalize(base) }
        return .override(base: canonicalize(base), with: canonOverlay)

    case .map(let base, let transform):
        return .map(base: canonicalize(base), transform: transform)

    case .gguf, .safetensors, .random:
        return declaration
    }
}

/// Initialization scheme for random weight generation.
public enum InitializationScheme: Codable, Equatable, Sendable {
    case xavier
    case kaiming
    case uniform(low: Float, high: Float)
    case normal(mean: Float, stddev: Float)
    case zeros
    case ones
}

/// Transformation applied to resolved weights.
public enum WeightTransform: Codable, Equatable, Sendable {
    case quantize(bits: Int, groupSize: Int)
    case cast(DTypeHint)
    case custom(String)
}
