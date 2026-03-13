import Foundation
import GGUFParser
import GGUFToolingCore
import GGUFValidation
import MLXLM
import Testing

@Suite("GGUFValidation", .tags(.unit))
struct GGUFValidationTests {
    @Test("Qwen35-like file reports missing partial rotary factor")
    func missingPartialRotaryFactorProducesIssue() throws {
        let file = try makeQwen35Fixture(
            includePartialRotaryFactor: false,
            includeAttentionKeyLength: true,
            includeTimeStepRank: true
        )
        let report = GGUFValidationRegistry.default.validate(file: file)

        #expect(report.family == "hybridDeltaNetAttention")
        #expect(report.issues.count == 1)
        #expect(report.issues.first?.metadataKey == "qwen35.rope.partial_rotary_factor")
        #expect(report.issues.first?.suggestedValue == .float32(0.25))
    }

    @Test("repair plan adds inferred partial rotary factor")
    func repairPlanAddsInferredPartialRotaryFactor() throws {
        let file = try makeQwen35Fixture(
            includePartialRotaryFactor: false,
            includeAttentionKeyLength: true,
            includeTimeStepRank: true
        )
        let plan = GGUFValidationRegistry.default.makeRepairPlan(
            file: file,
            mode: .includeInferredRepairs
        )

        #expect(plan.actions.count == 1)
        #expect(
            plan.actions == [
                .addMetadata(
                    key: "qwen35.rope.partial_rotary_factor",
                    value: .float32(0.25),
                    rationale: "Inferred from qwen35.rope.dimension_count / qwen35.attention.key_length = 64 / 256."
                )
            ]
        )
    }

    @Test("complete file yields no repair actions")
    func completeFileHasNoRepairActions() throws {
        let file = try makeQwen35Fixture(
            includePartialRotaryFactor: true,
            includeAttentionKeyLength: true,
            includeTimeStepRank: true
        )
        let report = GGUFValidationRegistry.default.validate(file: file)
        let plan = GGUFValidationRegistry.default.makeRepairPlan(
            file: file,
            mode: .includeInferredRepairs
        )

        #expect(report.issues.isEmpty)
        #expect(plan.actions.isEmpty)
    }

    @Test("missing inference evidence reports issue without suggested value")
    func missingInferenceEvidenceHasNoSuggestedValue() throws {
        let file = try makeQwen35Fixture(
            includePartialRotaryFactor: false,
            includeAttentionKeyLength: false,
            includeTimeStepRank: true
        )
        let report = GGUFValidationRegistry.default.validate(file: file)

        #expect(report.issues.count == 2)
        let partialRotaryIssue = report.issues.first { $0.metadataKey == "qwen35.rope.partial_rotary_factor" }
        let attentionKeyLengthIssue = report.issues.first { $0.metadataKey == "qwen35.attention.key_length" }
        #expect(partialRotaryIssue?.suggestedValue == nil)
        #expect(attentionKeyLengthIssue?.suggestedValue == nil)
    }

    @Test("missing ssm.time_step_rank is detected and inferred from inner_size / state_size")
    func missingTimeStepRankIsDetectedAndInferred() throws {
        let file = try makeQwen35Fixture(
            includePartialRotaryFactor: true,
            includeAttentionKeyLength: true,
            includeTimeStepRank: false
        )
        let report = GGUFValidationRegistry.default.validate(file: file)

        #expect(report.issues.count == 1)
        let issue = report.issues.first { $0.metadataKey == "qwen35.ssm.time_step_rank" }
        #expect(issue != nil)
        #expect(issue?.severity == .error)
        // inner_size=1024, state_size=128 → inferred time_step_rank = 8
        #expect(issue?.suggestedValue == .uint32(8))

        let plan = GGUFValidationRegistry.default.makeRepairPlan(
            file: file,
            mode: .includeInferredRepairs
        )
        #expect(plan.actions.count == 1)
        #expect(plan.actions.first?.key == "qwen35.ssm.time_step_rank")
    }

    @Test("repaired file contains patched metadata value")
    func repairedFileContainsPatchedMetadata() throws {
        let originalURL = try writeTemporaryFixture(
            includePartialRotaryFactor: false,
            includeAttentionKeyLength: true,
            includeTimeStepRank: true
        )
        let repairedURL = originalURL.deletingPathExtension().appendingPathExtension("repaired.gguf")
        let original = try GGUFFile.parse(url: originalURL)
        let plan = GGUFValidationRegistry.default.makeRepairPlan(
            file: original,
            mode: .includeInferredRepairs
        )

        try GGUFFileRewriter().applying(
            GGUFMetadataPatch(actions: plan.actions),
            to: originalURL,
            outputURL: repairedURL
        )

        let repaired = try GGUFFile.parse(url: repairedURL)
        let value = repaired.metadata.first { $0.key == "qwen35.rope.partial_rotary_factor" }?.value
        #expect(value == .float32(0.25))
    }

    private func makeQwen35Fixture(
        includePartialRotaryFactor: Bool,
        includeAttentionKeyLength: Bool,
        includeTimeStepRank: Bool
    ) throws -> GGUFFile {
        var builder = TestGGUFBuilder()
        builder.addMetadata("general.architecture", value: .string("qwen35"))
        builder.addMetadata("qwen35.embedding_length", value: .uint32(1024))
        builder.addMetadata("qwen35.block_count", value: .uint32(24))
        builder.addMetadata("qwen35.attention.head_count", value: .uint32(8))
        builder.addMetadata("qwen35.feed_forward_length", value: .uint32(3584))
        builder.addMetadata("qwen35.ssm.group_count", value: .uint32(8))
        builder.addMetadata("qwen35.ssm.state_size", value: .uint32(128))
        builder.addMetadata("qwen35.ssm.inner_size", value: .uint32(1024))
        builder.addMetadata("qwen35.ssm.conv_kernel", value: .uint32(4))
        builder.addMetadata("qwen35.full_attention_interval", value: .uint32(4))
        builder.addMetadata("qwen35.rope.dimension_count", value: .uint32(64))
        builder.addMetadata(
            "tokenizer.ggml.tokens",
            value: .array(["a", "b", "c", "d"].map(GGUFMetadataValue.string))
        )
        if includeAttentionKeyLength {
            builder.addMetadata("qwen35.attention.key_length", value: .uint32(256))
        }
        if includePartialRotaryFactor {
            builder.addMetadata("qwen35.rope.partial_rotary_factor", value: .float32(0.25))
        }
        if includeTimeStepRank {
            builder.addMetadata("qwen35.ssm.time_step_rank", value: .uint32(8))
        }
        builder.addTensor(
            name: "blk.0.ssm_beta.weight",
            shape: [256, 1],
            type: .f32,
            bytes: Data(repeating: 7, count: 1024)
        )
        return try GGUFFile.parse(data: builder.build())
    }

    private func writeTemporaryFixture(
        includePartialRotaryFactor: Bool,
        includeAttentionKeyLength: Bool,
        includeTimeStepRank: Bool
    ) throws -> URL {
        let file = try makeQwen35Fixture(
            includePartialRotaryFactor: includePartialRotaryFactor,
            includeAttentionKeyLength: includeAttentionKeyLength,
            includeTimeStepRank: includeTimeStepRank
        )
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("gguf")
        try file.data.write(to: url)
        return url
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
