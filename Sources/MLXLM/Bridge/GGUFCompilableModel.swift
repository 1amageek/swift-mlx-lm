import GGUFParser
import SwiftLM
import MLXCompiler

/// Extended capability for GGUF models that support the compiled inference path.
///
/// Models conforming to this protocol provide a `ModelComponent` declaration
/// that can be compiled via `MLXInferenceCompiler` for compile-time kernel
/// selection and optimized inference.
///
/// Conforming types must also conform to `GGUFLoadableModel` for the
/// standard MLXNN loading path. This protocol adds the compiled path
/// as an alternative.
package protocol GGUFCompilableModel: GGUFLoadableModel {

    /// Build a ModelComponent declaration from GGUF metadata.
    ///
    /// The returned component represents the full model architecture and
    /// can be normalized into a `ModelGraph` via `makeModelGraph()`.
    static func makeModelDeclaration(from file: GGUFFile) throws -> any ModelComponent

    /// Apply model-specific weight transforms before compilation.
    ///
    /// Compiled path equivalent of `LanguageModel.sanitize(weights:)`.
    /// Operates on `[String: TensorData]` keyed by MLX weight paths.
    ///
    /// Default implementation filters out `rotary_emb.inv_freq` keys.
    static func sanitizeCompiledWeights(_ weights: [String: TensorData]) -> [String: TensorData]
}

extension GGUFCompilableModel {
    package static func sanitizeCompiledWeights(_ weights: [String: TensorData]) -> [String: TensorData] {
        weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
    }
}
