import Foundation

public enum GGUFToolingError: Error, Sendable, LocalizedError {
    case unsupportedVersion(UInt32)
    case inPlaceRewriteNotAllowed
    case unsupportedEmptyArray(key: String)
    case mixedArrayTypes(key: String)
    case outputWouldTruncateTensorData
    case invalidAlignment(Int)

    public var errorDescription: String? {
        switch self {
        case .unsupportedVersion(let version):
            return "Unsupported GGUF version for rewrite: \(version)"
        case .inPlaceRewriteNotAllowed:
            return "Metadata repair must write a new GGUF file; in-place overwrite is not allowed."
        case .unsupportedEmptyArray(let key):
            return "Cannot rewrite empty GGUF metadata array for key '\(key)' because the element type is unknown."
        case .mixedArrayTypes(let key):
            return "Cannot rewrite GGUF metadata array for key '\(key)' because it contains mixed element types."
        case .outputWouldTruncateTensorData:
            return "Tensor payload range could not be preserved during GGUF rewrite."
        case .invalidAlignment(let alignment):
            return "GGUF alignment must be a positive power of two, got \(alignment)."
        }
    }
}
