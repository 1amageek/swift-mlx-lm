import Foundation

enum MetalPrefillProjectionAccelerationAvailability: Sendable, Equatable {
    case enabled
    case disabledByEnvironment
    case unavailable

    var isEnabled: Bool {
        if case .enabled = self {
            return true
        }
        return false
    }
}

struct MetalQuantizationCapabilities: Sendable, Equatable {
    let prefillProjectionAcceleration: MetalPrefillProjectionAccelerationAvailability

    static let none = MetalQuantizationCapabilities(prefillProjectionAcceleration: .unavailable)
}

enum MetalQuantizationExecutionPath: String, Sendable, Equatable {
    case prefillProjection
    case decodeProjection
    case embeddingLookup
    case kvCache
}

enum MetalQuantizationKernelFamily: Sendable, Equatable {
    case mppGEMM
    case naiveGEMM
    case genericGEMV
    case input2048GEMV
    case input8192TiledGEMV
    case vocabGEMV
    case denseEmbeddingLookup
    case bf16EmbeddingLookup
    case fp32EmbeddingLookup
    case q4G64EmbeddingLookup
    case q4G128EmbeddingLookup
    case q8G32EmbeddingLookup
    case q8G64EmbeddingLookup
    case q4G64GEMM
    case q4G128GEMM
    case q4G64GEMV
    case q4G128GEMV
    case q8G32GEMV
    case q8G64GEMV
    case other(String)

    var description: String {
        switch self {
        case .mppGEMM:
            return "mppGEMM"
        case .naiveGEMM:
            return "naiveGEMM"
        case .genericGEMV:
            return "genericGEMV"
        case .input2048GEMV:
            return "input2048GEMV"
        case .input8192TiledGEMV:
            return "input8192TiledGEMV"
        case .vocabGEMV:
            return "vocabGEMV"
        case .denseEmbeddingLookup:
            return "denseEmbeddingLookup"
        case .bf16EmbeddingLookup:
            return "bf16EmbeddingLookup"
        case .fp32EmbeddingLookup:
            return "fp32EmbeddingLookup"
        case .q4G64EmbeddingLookup:
            return "q4G64EmbeddingLookup"
        case .q4G128EmbeddingLookup:
            return "q4G128EmbeddingLookup"
        case .q8G32EmbeddingLookup:
            return "q8G32EmbeddingLookup"
        case .q8G64EmbeddingLookup:
            return "q8G64EmbeddingLookup"
        case .q4G64GEMM:
            return "q4G64GEMM"
        case .q4G128GEMM:
            return "q4G128GEMM"
        case .q4G64GEMV:
            return "q4G64GEMV"
        case .q4G128GEMV:
            return "q4G128GEMV"
        case .q8G32GEMV:
            return "q8G32GEMV"
        case .q8G64GEMV:
            return "q8G64GEMV"
        case .other(let name):
            return name
        }
    }

    static func classify(
        kernelName: String,
        usesMPP: Bool
    ) -> MetalQuantizationKernelFamily {
        let normalizedName = kernelName.replacingOccurrences(of: "naive::", with: "")
        if usesMPP {
            return .mppGEMM
        }
        if normalizedName.hasPrefix("embedding_lookup_seq_q4_g64")
            || normalizedName.hasPrefix("embedding_lookup_q4_g64")
        {
            return .q4G64EmbeddingLookup
        }
        if normalizedName.hasPrefix("embedding_lookup_seq_q4_g128")
            || normalizedName.hasPrefix("embedding_lookup_q4_g128")
        {
            return .q4G128EmbeddingLookup
        }
        if normalizedName.hasPrefix("embedding_lookup_seq_q8_g32")
            || normalizedName.hasPrefix("embedding_lookup_q8_g32")
        {
            return .q8G32EmbeddingLookup
        }
        if normalizedName.hasPrefix("embedding_lookup_seq_q8_g64")
            || normalizedName.hasPrefix("embedding_lookup_q8_g64")
        {
            return .q8G64EmbeddingLookup
        }
        if normalizedName.hasPrefix("embedding_lookup_seq_bf16")
            || normalizedName.hasPrefix("embedding_lookup_bf16")
        {
            return .bf16EmbeddingLookup
        }
        if normalizedName.hasPrefix("embedding_lookup_seq_fp32")
            || normalizedName.hasPrefix("embedding_lookup_fp32")
        {
            return .fp32EmbeddingLookup
        }
        if normalizedName.hasPrefix("embedding_lookup_seq")
            || normalizedName.hasPrefix("embedding_lookup")
        {
            return .denseEmbeddingLookup
        }
        switch normalizedName {
        case "gemm_q4_g64_f32s":
            return .q4G64GEMM
        case "gemm_q4_g128_f32s":
            return .q4G128GEMM
        case "gemv_q4_g64":
            return .q4G64GEMV
        case "gemv_q4_g128":
            return .q4G128GEMV
        case "gemv_q8_g32":
            return .q8G32GEMV
        case "gemv_q8_g64":
            return .q8G64GEMV
        case "gemm_f32s", "gemm_bf16_f32s":
            return .naiveGEMM
        case "gemv", "gemv_bf16", "gemv_f32s", "gemv_bf16_f32s":
            return .genericGEMV
        case _ where normalizedName.contains("vocab"):
            return .vocabGEMV
        case _ where normalizedName.contains("input8192"):
            return .input8192TiledGEMV
        case _ where normalizedName.contains("input2048"):
            return .input2048GEMV
        default:
            return .other(normalizedName)
        }
    }
}

enum MetalQuantizationFallbackReason: String, Sendable, Equatable {
    case missingTensorBinding
    case missingWeightStore
    case missingTensorMetadata
    case inputStrideMismatch
    case lastTokenProjectionUsesDecodeKernel
    case disabledByEnvironment
    case unavailableAcceleration
}

struct MetalQuantizationPlanEntry: Sendable, Equatable {
    let entryIndex: Int?
    let layerIndex: Int?
    let tensorName: String?
    let path: MetalQuantizationExecutionPath
    let schemeIdentifier: QuantizationSchemeIdentifier
    let layout: STAFWeightLayout
    let kernelFamily: MetalQuantizationKernelFamily
    let usedFallback: Bool
    let fallbackReason: MetalQuantizationFallbackReason?

    var isQuantizedWeight: Bool {
        schemeIdentifier.isWeightQuantized
    }
}

struct MetalQuantizationPlan: Sendable, Equatable {
    let capabilities: MetalQuantizationCapabilities
    let entries: [MetalQuantizationPlanEntry]

    static let empty = MetalQuantizationPlan(capabilities: .none, entries: [])

    var fallbackEntries: [MetalQuantizationPlanEntry] {
        entries.filter(\.usedFallback)
    }

    func summarizedLines(limit: Int = 8) -> [String] {
        guard !entries.isEmpty else { return [] }
        var lines: [String] = []
        let quantizedCount = entries.filter(\.isQuantizedWeight).count
        lines.append(
            "  quantization: entries=\(entries.count), quantizedWeights=\(quantizedCount), fallbacks=\(fallbackEntries.count), prefillAccel=\(capabilities.prefillProjectionAcceleration)"
        )
        for entry in entries.prefix(limit) {
            var line = "    path=\(entry.path.rawValue) kernel=\(entry.kernelFamily.description) scheme=\(entry.schemeIdentifier)"
            if let tensorName = entry.tensorName {
                line += " tensor=\(tensorName)"
            }
            if entry.layout != .rowMajor {
                line += " layout=\(entry.layout)"
            }
            if let fallbackReason = entry.fallbackReason {
                line += " fallback=\(fallbackReason.rawValue)"
            }
            lines.append(line)
        }
        if entries.count > limit {
            lines.append("    ... \(entries.count - limit) more")
        }
        return lines
    }
}

extension QuantizationSchemeIdentifier {
    var isWeightQuantized: Bool {
        switch baseScheme {
        case .q8Group32ScaleF16, .q8Group64ScaleF16, .q8Group128ScaleF16,
             .q6Group16ScaleF16, .q6Group32ScaleF16,
             .q5Group32ScaleF16, .q5Group64ScaleF16,
             .q4Group64ScaleF16, .q4Group128ScaleF16, .q4Group128ScaleF16Zero,
             .q3Group16ScaleF16, .q3Group32ScaleF16,
             .q2Group16ScaleF16, .q2Group32ScaleF16,
             .rotorQ8Group32ScaleF16, .rotorQ4Group64ScaleF16:
            return true
        case .fp16RowMajor, .bf16RowMajor, .fp32RowMajor, .passthrough:
            return false
        }
    }
}

extension WeightFormat {
    var quantizationSchemeIdentifier: QuantizationSchemeIdentifier {
        switch self {
        case .float16:
            return .fp16RowMajor
        case .bfloat16:
            return .bf16RowMajor
        case .float32:
            return .fp32RowMajor
        case .quantized2Bit(let groupSize):
            switch groupSize {
            case 16:
                return .q2Group16ScaleF16
            case 32:
                return .q2Group32ScaleF16
            default:
                return .passthrough
            }
        case .quantized3Bit(let groupSize):
            switch groupSize {
            case 16:
                return .q3Group16ScaleF16
            case 32:
                return .q3Group32ScaleF16
            default:
                return .passthrough
            }
        case .quantized4Bit(let groupSize):
            switch groupSize {
            case 64:
                return .q4Group64ScaleF16
            case 128:
                return .q4Group128ScaleF16
            default:
                return .passthrough
            }
        case .quantized5Bit(let groupSize):
            switch groupSize {
            case 32:
                return .q5Group32ScaleF16
            case 64:
                return .q5Group64ScaleF16
            default:
                return .passthrough
            }
        case .quantized6Bit(let groupSize):
            switch groupSize {
            case 16:
                return .q6Group16ScaleF16
            case 32:
                return .q6Group32ScaleF16
            default:
                return .passthrough
            }
        case .quantized8Bit(let groupSize):
            switch groupSize {
            case 32:
                return .q8Group32ScaleF16
            case 64:
                return .q8Group64ScaleF16
            case 128:
                return .q8Group128ScaleF16
            default:
                return .passthrough
            }
        }
    }
}
