import Foundation
import GGUFParser
import GGUFToolingCore
import Testing

@Suite("GGUFToolingCore", .tags(.unit))
struct GGUFToolingCoreTests {
    @Test("materialized metadata includes deferred arrays")
    func materializedMetadataIncludesDeferredArrays() throws {
        let file = try makeGGUFFile()
        let metadata = try file.materializedMetadata()

        #expect(metadata["tokenizer.ggml.tokens"]?.arrayValue?.count == 4)
        #expect(metadata["tokenizer.ggml.scores"]?.arrayValue?.count == 4)
    }

    @Test("metadata patch adds an architecture key")
    func metadataPatchAddsKey() throws {
        let fileURL = try writeTemporaryGGUF()
        let outputURL = fileURL.deletingPathExtension().appendingPathExtension("patched.gguf")
        let patch = GGUFMetadataPatch(values: [
            "qwen35.rope.partial_rotary_factor": .float32(0.25)
        ])

        try GGUFFileRewriter().applying(patch, to: fileURL, outputURL: outputURL)
        let rewritten = try GGUFFile.parse(url: outputURL)

        #expect(rewritten.metadata["qwen35.rope.partial_rotary_factor"] == .float32(0.25))
    }

    @Test("metadata patch replaces an existing key")
    func metadataPatchReplacesKey() throws {
        let fileURL = try writeTemporaryGGUF()
        let outputURL = fileURL.deletingPathExtension().appendingPathExtension("replaced.gguf")
        let patch = GGUFMetadataPatch(values: [
            "general.name": .string("patched")
        ])

        try GGUFFileRewriter().applying(patch, to: fileURL, outputURL: outputURL)
        let rewritten = try GGUFFile.parse(url: outputURL)

        #expect(rewritten.metadata["general.name"] == .string("patched"))
    }

    @Test("rewritten GGUF reparses and preserves tensor payload")
    func rewrittenGGUFPreservesTensorPayload() throws {
        let fileURL = try writeTemporaryGGUF()
        let outputURL = fileURL.deletingPathExtension().appendingPathExtension("repaired.gguf")
        let patch = GGUFMetadataPatch(values: [
            "qwen35.rope.partial_rotary_factor": .float32(0.25),
            "general.name": .string("rewritten")
        ])

        let original = try GGUFFile.parse(url: fileURL)
        try GGUFFileRewriter().applying(patch, to: fileURL, outputURL: outputURL)
        let rewritten = try GGUFFile.parse(url: outputURL)

        #expect(rewritten.tensors.count == original.tensors.count)
        #expect(rewritten.tensors.map(\.name) == original.tensors.map(\.name))
        #expect(rewritten.tensors.map(\.quantizationType) == original.tensors.map(\.quantizationType))
        #expect(rewritten.tensors.map(\.offset) == original.tensors.map(\.offset))
        #expect(rewritten.tensors.map(\.dataSize) == original.tensors.map(\.dataSize))

        for (originalTensor, rewrittenTensor) in zip(original.tensors, rewritten.tensors) {
            let originalData = try original.tensorData(for: originalTensor)
            let rewrittenData = try rewritten.tensorData(for: rewrittenTensor)
            #expect(originalData == rewrittenData)
        }
    }

    private func writeTemporaryGGUF() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("gguf")
        try makeGGUFFile().data.write(to: url)
        return url
    }

    private func makeGGUFFile() throws -> GGUFFile {
        var builder = TestGGUFBuilder()
        builder.addMetadata("general.architecture", value: .string("qwen35"))
        builder.addMetadata("general.name", value: .string("fixture"))
        builder.addMetadata("general.alignment", value: .uint32(32))
        builder.addMetadata("qwen35.embedding_length", value: .uint32(1024))
        builder.addMetadata("qwen35.block_count", value: .uint32(24))
        builder.addMetadata("qwen35.attention.head_count", value: .uint32(8))
        builder.addMetadata("qwen35.attention.key_length", value: .uint32(256))
        builder.addMetadata("qwen35.feed_forward_length", value: .uint32(3584))
        builder.addMetadata("qwen35.ssm.group_count", value: .uint32(8))
        builder.addMetadata("qwen35.ssm.state_size", value: .uint32(128))
        builder.addMetadata("qwen35.ssm.conv_kernel", value: .uint32(4))
        builder.addMetadata("qwen35.full_attention_interval", value: .uint32(4))
        builder.addMetadata("qwen35.rope.dimension_count", value: .uint32(64))
        builder.addMetadata(
            "tokenizer.ggml.tokens",
            value: .array(["<bos>", "a", "b", "<eos>"].map(GGUFMetadataValue.string))
        )
        builder.addMetadata(
            "tokenizer.ggml.scores",
            value: .array([0.0, 1.0, 2.0, 3.0].map(GGUFMetadataValue.float32))
        )
        builder.addTensor(
            name: "blk.0.ssm_beta.weight",
            shape: [256, 1],
            type: .f32,
            bytes: Data((0..<1024).map { UInt8($0 % 255) })
        )
        return try GGUFFile.parse(data: builder.build())
    }
}

private struct TestGGUFBuilder {
    private var metadata: [(String, GGUFMetadataValue)] = []
    private var tensors: [(name: String, shape: [UInt64], type: GGUFQuantizationType, bytes: Data)] = []

    mutating func addMetadata(_ key: String, value: GGUFMetadataValue) {
        metadata.append((key, value))
    }

    mutating func addTensor(name: String, shape: [UInt64], type: GGUFQuantizationType, bytes: Data) {
        tensors.append((name, shape, type, bytes))
    }

    func build() -> Data {
        var data = Data()
        append(UInt32(GGUFFile.magic), to: &data)
        append(UInt32(3), to: &data)
        append(UInt64(tensors.count), to: &data)
        append(UInt64(metadata.count), to: &data)

        for (key, value) in metadata {
            appendString(key, to: &data)
            appendMetadataValue(value, to: &data)
        }

        var tensorOffset = 0
        for tensor in tensors {
            appendString(tensor.name, to: &data)
            append(UInt32(tensor.shape.count), to: &data)
            for dimension in tensor.shape {
                append(dimension, to: &data)
            }
            append(tensor.type.rawValue, to: &data)
            append(UInt64(tensorOffset), to: &data)
            tensorOffset += tensor.bytes.count
        }

        let padding = (32 - (data.count % 32)) % 32
        if padding > 0 {
            data.append(Data(repeating: 0, count: padding))
        }
        for tensor in tensors {
            data.append(tensor.bytes)
        }
        return data
    }

    private func appendMetadataValue(_ value: GGUFMetadataValue, to data: inout Data) {
        append(value.ggufType.rawValue, to: &data)
        switch value {
        case .uint32(let rawValue):
            append(rawValue, to: &data)
        case .float32(let rawValue):
            append(rawValue.bitPattern, to: &data)
        case .string(let rawValue):
            appendString(rawValue, to: &data)
        case .array(let values):
            let elementType = values.first?.ggufType ?? .string
            append(elementType.rawValue, to: &data)
            append(UInt64(values.count), to: &data)
            for element in values {
                switch element {
                case .string(let rawValue):
                    appendString(rawValue, to: &data)
                case .float32(let rawValue):
                    append(rawValue.bitPattern, to: &data)
                case .uint32(let rawValue):
                    append(rawValue, to: &data)
                default:
                    break
                }
            }
        default:
            break
        }
    }

    private func appendString(_ string: String, to data: inout Data) {
        let bytes = Data(string.utf8)
        append(UInt64(bytes.count), to: &data)
        data.append(bytes)
    }

    private func append<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { bytes in
            data.append(contentsOf: bytes)
        }
    }
}
