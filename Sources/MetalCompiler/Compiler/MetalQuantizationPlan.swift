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
    case q3G16EmbeddingLookup
    case q3G32EmbeddingLookup
    case q3G64EmbeddingLookup
    case q4G64EmbeddingLookup
    case q4G128EmbeddingLookup
    case q5G32EmbeddingLookup
    case q5G64EmbeddingLookup
    case q6G16EmbeddingLookup
    case q6G32EmbeddingLookup
    case q8G32EmbeddingLookup
    case q8G64EmbeddingLookup
    case q3G16GEMM
    case q3G32GEMM
    case q3G64GEMM
    case q4G64GEMM
    case q4G128GEMM
    case q5G32GEMM
    case q5G64GEMM
    case q6G16GEMM
    case q6G32GEMM
    case q3G16GEMV
    case q3G32GEMV
    case q3G64GEMV
    case q4G64GEMV
    case q4G128GEMV
    case q5G32GEMV
    case q5G64GEMV
    case q6G16GEMV
    case q6G32GEMV
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
        case .q3G16EmbeddingLookup:
            return "q3G16EmbeddingLookup"
        case .q3G32EmbeddingLookup:
            return "q3G32EmbeddingLookup"
        case .q3G64EmbeddingLookup:
            return "q3G64EmbeddingLookup"
        case .q4G64EmbeddingLookup:
            return "q4G64EmbeddingLookup"
        case .q4G128EmbeddingLookup:
            return "q4G128EmbeddingLookup"
        case .q5G32EmbeddingLookup:
            return "q5G32EmbeddingLookup"
        case .q5G64EmbeddingLookup:
            return "q5G64EmbeddingLookup"
        case .q6G16EmbeddingLookup:
            return "q6G16EmbeddingLookup"
        case .q6G32EmbeddingLookup:
            return "q6G32EmbeddingLookup"
        case .q8G32EmbeddingLookup:
            return "q8G32EmbeddingLookup"
        case .q8G64EmbeddingLookup:
            return "q8G64EmbeddingLookup"
        case .q3G16GEMM:
            return "q3G16GEMM"
        case .q3G32GEMM:
            return "q3G32GEMM"
        case .q3G64GEMM:
            return "q3G64GEMM"
        case .q4G64GEMM:
            return "q4G64GEMM"
        case .q4G128GEMM:
            return "q4G128GEMM"
        case .q5G32GEMM:
            return "q5G32GEMM"
        case .q5G64GEMM:
            return "q5G64GEMM"
        case .q6G16GEMM:
            return "q6G16GEMM"
        case .q6G32GEMM:
            return "q6G32GEMM"
        case .q3G16GEMV:
            return "q3G16GEMV"
        case .q3G32GEMV:
            return "q3G32GEMV"
        case .q3G64GEMV:
            return "q3G64GEMV"
        case .q4G64GEMV:
            return "q4G64GEMV"
        case .q4G128GEMV:
            return "q4G128GEMV"
        case .q5G32GEMV:
            return "q5G32GEMV"
        case .q5G64GEMV:
            return "q5G64GEMV"
        case .q6G16GEMV:
            return "q6G16GEMV"
        case .q6G32GEMV:
            return "q6G32GEMV"
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
        if normalizedName.hasPrefix("embedding_lookup_seq_q3_g16")
            || normalizedName.hasPrefix("embedding_lookup_q3_g16")
        {
            return .q3G16EmbeddingLookup
        }
        if normalizedName.hasPrefix("embedding_lookup_seq_q3_g32")
            || normalizedName.hasPrefix("embedding_lookup_q3_g32")
        {
            return .q3G32EmbeddingLookup
        }
        if normalizedName.hasPrefix("embedding_lookup_seq_q3_g64")
            || normalizedName.hasPrefix("embedding_lookup_q3_g64")
        {
            return .q3G64EmbeddingLookup
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
        if normalizedName.hasPrefix("embedding_lookup_seq_q5_g32")
            || normalizedName.hasPrefix("embedding_lookup_q5_g32")
        {
            return .q5G32EmbeddingLookup
        }
        if normalizedName.hasPrefix("embedding_lookup_seq_q5_g64")
            || normalizedName.hasPrefix("embedding_lookup_q5_g64")
        {
            return .q5G64EmbeddingLookup
        }
        if normalizedName.hasPrefix("embedding_lookup_seq_q6_g16")
            || normalizedName.hasPrefix("embedding_lookup_q6_g16")
        {
            return .q6G16EmbeddingLookup
        }
        if normalizedName.hasPrefix("embedding_lookup_seq_q6_g32")
            || normalizedName.hasPrefix("embedding_lookup_q6_g32")
        {
            return .q6G32EmbeddingLookup
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
        case "gemm_q3_g16_f32s":
            return .q3G16GEMM
        case "gemm_q3_g32_f32s":
            return .q3G32GEMM
        case "gemm_q3_g64_f32s":
            return .q3G64GEMM
        case "gemm_q4_g64_f32s":
            return .q4G64GEMM
        case "gemm_q4_g128_f32s":
            return .q4G128GEMM
        case "gemm_q5_g32_f32s":
            return .q5G32GEMM
        case "gemm_q5_g64_f32s":
            return .q5G64GEMM
        case "gemm_q6_g16_f32s":
            return .q6G16GEMM
        case "gemm_q6_g32_f32s":
            return .q6G32GEMM
        case "gemv_q3_g16":
            return .q3G16GEMV
        case "gemv_q3_g32":
            return .q3G32GEMV
        case "gemv_q3_g64":
            return .q3G64GEMV
        case "gemv_q4_g64":
            return .q4G64GEMV
        case "gemv_q4_g128":
            return .q4G128GEMV
        case "gemv_q5_g32":
            return .q5G32GEMV
        case "gemv_q5_g64":
            return .q5G64GEMV
        case "gemv_q6_g16":
            return .q6G16GEMV
        case "gemv_q6_g32":
            return .q6G32GEMV
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
             .q3Group16ScaleF16, .q3Group32ScaleF16, .q3Group64ScaleF16,
             .q2Group16ScaleF16, .q2Group32ScaleF16,
             .rotorQ8Group32ScaleF16, .rotorQ4Group64ScaleF16:
            return true
        case .fp16RowMajor, .bf16RowMajor, .fp32RowMajor, .passthrough:
            return false
        }
    }
}

// Note: `WeightFormat.quantizationSchemeIdentifier` was previously derived by
// switching over enum cases. Now that `WeightFormat == any QuantizationFormat`,
// callers should use `format.schemeIdentifier` directly.
