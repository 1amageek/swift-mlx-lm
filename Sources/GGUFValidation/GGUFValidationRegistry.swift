import Foundation
import GGUFParser
import GGUFToolingCore
import MLXLM

public struct GGUFValidationRegistry: Sendable {
    public static let `default` = GGUFValidationRegistry(validators: [
        HybridDeltaNetAttentionValidator()
    ])

    private let detector: GGUFArchitectureDetector
    private let validatorsByFamily: [DetectedArchitecture: any GGUFFamilyValidator]

    public init(
        detector: GGUFArchitectureDetector = GGUFArchitectureDetector(),
        validators: [any GGUFFamilyValidator]
    ) {
        self.detector = detector
        self.validatorsByFamily = Dictionary(
            uniqueKeysWithValues: validators.map { ($0.family, $0) }
        )
    }

    public func validate(file: GGUFFile) -> GGUFValidationReport {
        let family = detector.detect(file: file)
        let context = ValidationContext(file: file, family: family)
        guard let validator = validatorsByFamily[family] else {
            let familyName = family.displayName
            let issue = GGUFValidationIssue(
                severity: .warning,
                kind: .unsupportedFamily,
                metadataKey: "",
                message: "No GGUF validator is registered for family '\(familyName)'.",
                evidence: ["detected family: \(familyName)"]
            )
            return GGUFValidationReport(
                architecture: file.architecture,
                family: familyName,
                issues: [issue]
            )
        }

        return GGUFValidationReport(
            architecture: file.architecture,
            family: family.displayName,
            issues: validator.validate(context: context)
        )
    }

    public func makeRepairPlan(
        file: GGUFFile,
        mode: RepairPlanningMode,
        sourceURL: URL? = nil
    ) -> GGUFRepairPlan {
        let family = detector.detect(file: file)
        let context = ValidationContext(file: file, family: family)
        guard let validator = validatorsByFamily[family] else {
            return GGUFRepairPlan(
                sourceURL: sourceURL,
                architecture: file.architecture,
                actions: [],
                warnings: ["No GGUF repair rules are registered for family '\(family.displayName)'."]
            )
        }

        return GGUFRepairPlan(
            sourceURL: sourceURL,
            architecture: file.architecture,
            actions: validator.repairActions(context: context, mode: mode)
        )
    }
}

private extension DetectedArchitecture {
    var displayName: String {
        switch self {
        case .transformer:
            return "transformer"
        case .parallelAttentionMLP:
            return "parallelAttentionMLP"
        case .moe:
            return "moe"
        case .hybridDeltaNetAttention:
            return "hybridDeltaNetAttention"
        }
    }
}
