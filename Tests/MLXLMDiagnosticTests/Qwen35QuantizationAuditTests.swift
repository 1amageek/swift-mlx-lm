import Foundation
import GGUFParser
import MLX
import MLXNN
import Testing
import TestHeartbeat
@testable import MLXLM

@Suite("Qwen3.5 Quantization Audit", .tags(.diagnostic), .heartbeat)
struct Qwen35QuantizationAuditTests {

    private static let repo = "unsloth/Qwen3.5-0.8B-GGUF"
    private static let filename = "Qwen3.5-0.8B-Q4_K_M.gguf"
    private static let losslessDirectTypes: Set<GGUFQuantizationType> = [
        .q4_0, .q4_1, .q4_K, .q5_0, .q5_1, .q5_K, .q6_K, .q8_0, .q8_1, .q8_K, .q2_K, .q3_K, .tq2_0,
    ]

    private func downloadModel() async throws -> URL {
        let downloader = HuggingFaceDownloader()
        return try await downloader.download(
            repo: Self.repo,
            filename: Self.filename
        )
    }

    @Test("Real q5_K and q6_K tensors direct-pack match dense fallback")
    func realModelMixedQuantizationRoundTrip() async throws {
        let url = try await downloadModel()
        let file = try GGUFFile.parse(url: url)
        let bridge = GGUFTensorBridge()

        try verifyDirectPackedLinearTensor(
            file: file,
            bridge: bridge,
            tensorName: "blk.0.ssm_out.weight",
            tolerance: 0.25
        )
        try verifyDirectPackedLinearTensor(
            file: file,
            bridge: bridge,
            tensorName: "blk.3.attn_v.weight",
            tolerance: 0.15
        )
    }

    @Test("Real Qwen3.5 GGUF keeps 2D quantized weights packed without lossy requantization")
    func realModelQuantizedTensorAudit() async throws {
        let url = try await downloadModel()
        let file = try GGUFFile.parse(url: url)
        let bridge = GGUFTensorBridge()

        let quantized2DWeights = file.tensors.filter {
            !$0.quantizationType.isUnquantized
                && $0.dimensions.count >= 2
                && $0.name.hasSuffix(".weight")
        }
        #expect(!quantized2DWeights.isEmpty)

        let groupedByType = Dictionary(grouping: quantized2DWeights, by: \.quantizationType)
        let sortedSummary = groupedByType
            .sorted { lhs, rhs in String(describing: lhs.key) < String(describing: rhs.key) }
            .map { "\($0.key)=\($0.value.count)" }
            .joined(separator: ", ")
        print("[qwen35][quant-audit] quantized2DWeights=\(quantized2DWeights.count)")
        print("[qwen35][quant-audit] qtypes=\(sortedSummary)")

        for (qtype, tensors) in groupedByType.sorted(by: { String(describing: $0.key) < String(describing: $1.key) }) {
            let tensorNames = tensors.map { $0.name }
            #expect(
                Self.losslessDirectTypes.contains(qtype),
                Comment(
                    rawValue: "Expected this Qwen3.5 GGUF to use only lossless direct-pack types, found \(qtype) in \(tensorNames)"
                )
            )

            let tensor = try #require(tensors.first)
            let data = try file.tensorData(for: tensor)
            let converted = try bridge.convertDirect(tensor: tensor, data: data)
            guard case .quantized(_, _, _, let groupSize, let bits) = converted else {
                Issue.record(
                    "Expected representative tensor \(tensor.name) (\(qtype)) to stay quantized, but it fell back to F16"
                )
                continue
            }

            if let expectedLayout = expectedLayout(for: qtype) {
                #expect(
                    bits == expectedLayout.bits,
                    Comment(
                        rawValue: "\(tensor.name) expected \(expectedLayout.bits)-bit packing for \(qtype), got \(bits)"
                    )
                )
                #expect(
                    groupSize == expectedLayout.groupSize,
                    Comment(
                        rawValue: "\(tensor.name) expected groupSize \(expectedLayout.groupSize) for \(qtype), got \(groupSize)"
                    )
                )
            }
        }
    }

    private func expectedLayout(
        for qtype: GGUFQuantizationType
    ) -> (bits: Int, groupSize: Int)? {
        switch qtype {
        case .q2_K:
            return (2, 32)
        case .q3_K:
            return (3, 32)
        case .q4_0, .q4_1, .q4_K, .iq4_NL, .iq4_XS:
            return (4, 32)
        case .q5_0, .q5_1, .q5_K:
            return (5, 32)
        case .q6_K:
            return (6, 32)
        case .q8_0, .q8_1, .q8_K:
            return (8, 32)
        case .tq1_0:
            return (2, 32)
        case .tq2_0:
            return (2, 32)
        case .iq2_XXS, .iq2_XS, .iq2_S, .iq3_XXS, .iq3_S, .iq1_S, .iq1_M:
            return (4, 32)
        default:
            return nil
        }
    }

    private func verifyDirectPackedLinearTensor(
        file: GGUFFile,
        bridge: GGUFTensorBridge,
        tensorName: String,
        tolerance: Float
    ) throws {
        let tensor = try #require(file.tensors.first { $0.name == tensorName })
        let data = try file.tensorData(for: tensor)

        let denseWeight = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected direct quantized tensor for \(tensorName)")
            return
        }

        let shape = tensor.dimensions.reversed().map { Int($0) }
        let outputDim = shape[0]
        let inputDim = shape[1]

        print(
            "[qwen35][tensor-check] name=\(tensorName) qtype=\(tensor.quantizationType) shape=\(shape) bits=\(bits) groupSize=\(groupSize)"
        )

        let qLinear: Linear
        if groupSize >= 32 {
            qLinear = QuantizedLinear(
                weight: weight,
                bias: nil,
                scales: scales,
                biases: biases,
                groupSize: groupSize,
                bits: bits,
                mode: .affine
            )
        } else {
            qLinear = DirectQuantizedLinear(
                weight: weight,
                bias: nil,
                scales: scales,
                biases: biases,
                groupSize: groupSize,
                bits: bits,
                mode: .affine
            )
        }
        let denseLinear = Linear(weight: denseWeight, bias: nil)

        MLXRandom.seed(7)
        let input = MLXRandom.normal([1, inputDim]).asType(.float16)
        let qOutput = qLinear(input)
        let denseOutput = denseLinear(input)
        eval(qOutput, denseOutput)

        let qValues = qOutput.asType(.float32)
        let dValues = denseOutput.asType(.float32)
        let diff = MLX.abs(qValues - dValues)
        let maxMagnitude = MLX.abs(dValues).max()
        eval(diff, maxMagnitude)

        let maxDiff: Float = diff.max().item()
        let magnitude: Float = maxMagnitude.item()
        let effectiveTolerance = max(tolerance, magnitude * 0.002)

        print(
            "[qwen35][tensor-check] name=\(tensorName) maxDiff=\(maxDiff) tolerance=\(effectiveTolerance) magnitude=\(magnitude) outputDim=\(outputDim)"
        )
        #expect(
            maxDiff < effectiveTolerance,
            Comment(rawValue: "\(tensorName) direct pack diverged: diff=\(maxDiff) tolerance=\(effectiveTolerance)")
        )
    }
}
