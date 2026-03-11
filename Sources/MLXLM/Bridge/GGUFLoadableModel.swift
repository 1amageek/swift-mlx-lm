import Foundation
import GGUFParser
import GGUFTokenizer

/// GGUF self-loading capability for language models.
///
/// Models conforming to this protocol can inspect a GGUF file's structure
/// (tensors and metadata) to determine compatibility and construct themselves.
/// This is a package-internal protocol — consumers interact only with
/// `LanguageModel` for runtime inference.
///
/// ## `canLoad` contract
///
/// Each implementation's `canLoad` must be a **sufficient condition** —
/// if it returns `true`, the model is guaranteed to be the correct type
/// for this GGUF file. Checks must be **mutually exclusive** across all
/// model types (except the universal fallback) so that correctness does
/// not depend on evaluation order.
///
/// Achieve this by checking for a **combination** of structural features
/// that uniquely identifies the architecture, using both positive and
/// negative conditions when necessary.
package protocol GGUFLoadableModel: LanguageModel {

    /// Check whether this model type can handle the given GGUF file.
    ///
    /// Implementations must inspect structural features (tensor names,
    /// metadata presence) rather than matching architecture strings.
    /// The check must be a **sufficient condition**: mutually exclusive
    /// with all other model types' checks (except the universal fallback).
    static func canLoad(from file: GGUFFile, context: GGUFLoadContext) -> Bool

    /// Construct the model from GGUF metadata and return everything needed
    /// for the loading pipeline: model instance, tensor name mapper, and
    /// processor factory.
    static func load(
        from file: GGUFFile,
        context: GGUFLoadContext
    ) throws -> GGUFLoadResult
}

/// Context provided to `GGUFLoadableModel.load(from:context:)`.
package struct GGUFLoadContext: Sendable {

    /// Tokenizer created from the GGUF file's metadata.
    package let tokenizer: any Tokenizer

    /// Optional mmproj GGUF URL for VLM vision encoder loading.
    package let mmprojURL: URL?

    package init(tokenizer: any Tokenizer, mmprojURL: URL?) {
        self.tokenizer = tokenizer
        self.mmprojURL = mmprojURL
    }
}

/// Result of `GGUFLoadableModel.load(from:context:)`.
package struct GGUFLoadResult {

    /// The constructed (but not yet weight-loaded) model.
    package let model: any LanguageModel

    /// Tensor name mapper for GGUF → MLX weight path translation.
    package let mapper: any GGUFTensorNameMapper

    /// Optional closure to load vision encoder weights from mmproj GGUF.
    /// Called only when `mmprojURL` is provided.
    package let visionLoader: ((URL) throws -> Void)?

    /// Factory for creating the appropriate `UserInputProcessor`.
    package let makeProcessor: @Sendable (
        _ tokenizer: any Tokenizer,
        _ chatTemplate: String?,
        _ bosToken: String?,
        _ eosToken: String?,
        _ addBosToken: Bool
    ) -> any UserInputProcessor

    package init(
        model: any LanguageModel,
        mapper: any GGUFTensorNameMapper,
        visionLoader: ((URL) throws -> Void)? = nil,
        makeProcessor: @Sendable @escaping (
            _ tokenizer: any Tokenizer,
            _ chatTemplate: String?,
            _ bosToken: String?,
            _ eosToken: String?,
            _ addBosToken: Bool
        ) -> any UserInputProcessor
    ) {
        self.model = model
        self.mapper = mapper
        self.visionLoader = visionLoader
        self.makeProcessor = makeProcessor
    }
}

/// Errors specific to GGUF model loading.
enum GGUFLoadError: Error {
    case missingMetadata(String)
    case unsupportedArchitecture(String)
    case unsupportedQuantization(UInt32)
    case tensorNotFound(String)
    case dimensionMismatch(String)
    case invalidData(String)
}
