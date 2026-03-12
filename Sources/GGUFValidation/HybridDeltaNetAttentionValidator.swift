import GGUFParser
import GGUFToolingCore
import MLXLM

public struct HybridDeltaNetAttentionValidator: GGUFFamilyValidator {
    public let family: DetectedArchitecture = .hybridDeltaNetAttention

    public init() {}

    public func validate(context: ValidationContext) -> [GGUFValidationIssue] {
        var issues: [GGUFValidationIssue] = []
        let file = context.file

        for key in [
            "ssm.group_count",
            "ssm.state_size",
            "ssm.conv_kernel",
            "full_attention_interval",
        ] {
            if file.architectureMetadata(key) == nil {
                issues.append(
                    GGUFValidationIssue(
                        severity: .error,
                        kind: .missingMetadata,
                        metadataKey: fullyQualifiedKey(key, architecture: context.architecture),
                        message: "Missing required GGUF metadata: \(key)",
                        evidence: ["expected key: \(fullyQualifiedKey(key, architecture: context.architecture))"]
                    )
                )
            }
        }

        if file.partialRotaryFactor == nil {
            let detail = partialRotaryFactorEvidence(file: file, architecture: context.architecture)
            issues.append(
                GGUFValidationIssue(
                    severity: .error,
                    kind: .missingMetadata,
                    metadataKey: fullyQualifiedKey("rope.partial_rotary_factor", architecture: context.architecture),
                    message: detail.message,
                    evidence: detail.evidence,
                    suggestedValue: detail.suggestedValue
                )
            )
        }

        return issues
    }

    public func repairActions(
        context: ValidationContext,
        mode: RepairPlanningMode
    ) -> [GGUFRepairAction] {
        guard mode == .includeInferredRepairs else {
            return []
        }
        let file = context.file
        guard file.partialRotaryFactor == nil else {
            return []
        }
        let detail = partialRotaryFactorEvidence(file: file, architecture: context.architecture)
        guard let suggestedValue = detail.suggestedValue else {
            return []
        }
        return [
            .addMetadata(
                key: fullyQualifiedKey("rope.partial_rotary_factor", architecture: context.architecture),
                value: suggestedValue,
                rationale: detail.rationale
            )
        ]
    }

    private func partialRotaryFactorEvidence(
        file: GGUFFile,
        architecture: String?
    ) -> (message: String, evidence: [String], suggestedValue: GGUFMetadataValue?, rationale: String) {
        let key = fullyQualifiedKey("rope.partial_rotary_factor", architecture: architecture)
        var evidence = ["expected key: \(key)"]

        guard let ropeDimension = file.ropeDimensionCount else {
            return (
                "Missing required GGUF metadata: rope.partial_rotary_factor",
                evidence,
                nil,
                "No deterministic inference rule was available."
            )
        }
        evidence.append("\(fullyQualifiedKey("rope.dimension_count", architecture: architecture))=\(ropeDimension)")

        guard let attentionKeyLength = file.attentionKeyLength, attentionKeyLength > 0 else {
            return (
                "Missing required GGUF metadata: rope.partial_rotary_factor",
                evidence,
                nil,
                "No deterministic inference rule was available."
            )
        }
        evidence.append("\(fullyQualifiedKey("attention.key_length", architecture: architecture))=\(attentionKeyLength)")

        let inferred = Float(ropeDimension) / Float(attentionKeyLength)
        let inferredValue = GGUFMetadataValue.float32(inferred)
        let inferredString = inferredValue.displayString
        return (
            "Missing required GGUF metadata: rope.partial_rotary_factor (inferred factor would be \(inferredString), but strict loading requires explicit metadata)",
            evidence,
            inferredValue,
            "Inferred from \(fullyQualifiedKey("rope.dimension_count", architecture: architecture)) / \(fullyQualifiedKey("attention.key_length", architecture: architecture)) = \(ropeDimension) / \(attentionKeyLength)."
        )
    }

    private func fullyQualifiedKey(_ key: String, architecture: String?) -> String {
        guard let architecture, !architecture.isEmpty else {
            return key
        }
        return "\(architecture).\(key)"
    }
}
