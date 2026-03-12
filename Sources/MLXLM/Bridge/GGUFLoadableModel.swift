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

extension GGUFLoadError: CustomStringConvertible, LocalizedError {

    var description: String {
        switch self {
        case .missingMetadata(let key):
            return "Missing required GGUF metadata: \(key)"
        case .unsupportedArchitecture(let architecture):
            return "Unsupported GGUF architecture: \(architecture)"
        case .unsupportedQuantization(let rawValue):
            return "Unsupported GGUF quantization type: \(rawValue)"
        case .tensorNotFound(let name):
            return "Required GGUF tensor not found: \(name)"
        case .dimensionMismatch(let message):
            return "GGUF tensor dimension mismatch: \(message)"
        case .invalidData(let message):
            return "Invalid GGUF data: \(message)"
        }
    }

    var errorDescription: String? { description }
}

package enum GGUFMetadataDiagnostics {

    package static func missingMetadataMessage(_ key: String, in file: GGUFFile) -> String {
        if let detail = diagnosticDetail(for: key, in: file) {
            return "\(key) (\(detail))"
        }
        if let architecture = file.architecture {
            return "\(key) (expected key: \(architecture).\(key))"
        }
        return key
    }

    private static func diagnosticDetail(for key: String, in file: GGUFFile) -> String? {
        switch key {
        case "rope.partial_rotary_factor":
            return partialRotaryFactorDetail(in: file)
        default:
            return nil
        }
    }

    private static func partialRotaryFactorDetail(in file: GGUFFile) -> String? {
        guard let architecture = file.architecture else { return nil }

        let expectedKey = "\(architecture).rope.partial_rotary_factor"
        guard let ropeDimension = file.ropeDimensionCount else {
            return "expected key: \(expectedKey)"
        }

        let ropeDimensionKey = "\(architecture).rope.dimension_count"
        let attentionKeyLength = file.attentionKeyLength ?? file.headDimension

        guard let attentionKeyLength, attentionKeyLength > 0 else {
            return "expected key: \(expectedKey); found \(ropeDimensionKey)=\(ropeDimension)"
        }

        let attentionKeyLengthKey = "\(architecture).attention.key_length"
        let inferred = Float(ropeDimension) / Float(attentionKeyLength)
        let inferredString = String(format: "%.6g", inferred)
        return
            "expected key: \(expectedKey); found \(ropeDimensionKey)=\(ropeDimension) and \(attentionKeyLengthKey)=\(attentionKeyLength); inferred factor would be \(inferredString), but strict loading requires explicit metadata"
    }
}
