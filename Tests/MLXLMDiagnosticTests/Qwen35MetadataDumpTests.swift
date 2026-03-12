import Foundation
import GGUFParser
import Testing
import TestHeartbeat
@testable import MLXLM

@Suite("Qwen3.5 GGUF Metadata Dump", .tags(.diagnostic), .heartbeat)
struct Qwen35MetadataDumpTests {

    private static let repo = "unsloth/Qwen3.5-0.8B-GGUF"
    private static let filename = "Qwen3.5-0.8B-Q4_K_M.gguf"

    private func downloadModel() async throws -> URL {
        let downloader = HuggingFaceDownloader()
        return try await downloader.download(
            repo: Self.repo,
            filename: Self.filename
        )
    }

    @Test("Dump all GGUF metadata and tensor info")
    func dumpMetadataAndTensors() async throws {
        let url = try await downloadModel()
        let file = try GGUFFile.parse(url: url)

        // -- Metadata --
        let sortedKeys = file.metadata.keys.sorted()
        print("========== GGUF METADATA (\(sortedKeys.count) keys) ==========")
        for key in sortedKeys {
            let value = file.metadata[key]!
            print("  \(key) = \(formatMetadataValue(value))")
        }

        // -- Tensors --
        print("")
        print("========== GGUF TENSORS (\(file.tensors.count) tensors) ==========")
        for tensor in file.tensors {
            let dims = tensor.dimensions.map(String.init).joined(separator: ", ")
            print("  \(tensor.name)  dims=[\(dims)]  qtype=\(tensor.quantizationType)  bytes=\(tensor.dataSize)")
        }

        // -- Summary --
        print("")
        print("========== SUMMARY ==========")
        print("  GGUF version: \(file.version)")
        print("  Metadata keys: \(sortedKeys.count)")
        print("  Tensors: \(file.tensors.count)")
        print("  Tensor data offset: \(file.tensorDataOffset)")

        var qtypeCounts: [GGUFQuantizationType: Int] = [:]
        for tensor in file.tensors {
            qtypeCounts[tensor.quantizationType, default: 0] += 1
        }
        print("  Quantization type distribution:")
        for (qtype, count) in qtypeCounts.sorted(by: { $0.key.rawValue < $1.key.rawValue }) {
            print("    \(qtype): \(count) tensors")
        }
    }

    // MARK: - Formatting

    private func formatMetadataValue(_ value: GGUFMetadataValue) -> String {
        switch value {
        case .uint8(let v):   return "\(v) (uint8)"
        case .int8(let v):    return "\(v) (int8)"
        case .uint16(let v):  return "\(v) (uint16)"
        case .int16(let v):   return "\(v) (int16)"
        case .uint32(let v):  return "\(v) (uint32)"
        case .int32(let v):   return "\(v) (int32)"
        case .float32(let v): return "\(v) (float32)"
        case .bool(let v):    return "\(v) (bool)"
        case .string(let v):  return "\"\(truncate(v, maxLength: 200))\" (string)"
        case .uint64(let v):  return "\(v) (uint64)"
        case .int64(let v):   return "\(v) (int64)"
        case .float64(let v): return "\(v) (float64)"
        case .array(let arr):
            if arr.count <= 10 {
                let elements = arr.map { formatMetadataValue($0) }
                return "[\(elements.joined(separator: ", "))] (array, \(arr.count) elements)"
            } else {
                let first5 = arr.prefix(5).map { formatMetadataValue($0) }
                let last2 = arr.suffix(2).map { formatMetadataValue($0) }
                return "[\(first5.joined(separator: ", ")), ... , \(last2.joined(separator: ", "))] (array, \(arr.count) elements)"
            }
        }
    }

    private func truncate(_ string: String, maxLength: Int) -> String {
        if string.count <= maxLength { return string }
        let prefix = string.prefix(maxLength)
        return "\(prefix)... (\(string.count) chars total)"
    }
}
