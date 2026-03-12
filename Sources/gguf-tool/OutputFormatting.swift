import Foundation
import GGUFParser
import GGUFToolingCore

enum TextOutput {
    static func render(report: GGUFValidationReport) -> String {
        var lines: [String] = []
        lines.append("GGUF validation report")
        if let architecture = report.architecture {
            lines.append("architecture: \(architecture)")
        }
        if let family = report.family {
            lines.append("family: \(family)")
        }
        lines.append("issues: \(report.issues.count)")

        for (index, issue) in report.issues.enumerated() {
            lines.append("")
            lines.append("[\(index + 1)] \(issue.severity.rawValue.uppercased()) \(issue.kind.rawValue)")
            if !issue.metadataKey.isEmpty {
                lines.append("key: \(issue.metadataKey)")
            }
            lines.append(issue.message)
            for evidence in issue.evidence {
                lines.append("evidence: \(evidence)")
            }
            if let suggestedValue = issue.suggestedValue {
                lines.append("suggested: \(suggestedValue.displayString)")
            }
        }

        for warning in report.warnings {
            lines.append("")
            lines.append("warning: \(warning)")
        }

        return lines.joined(separator: "\n")
    }

    static func render(plan: GGUFRepairPlan) -> String {
        var lines: [String] = []
        lines.append("GGUF repair plan")
        if let sourceURL = plan.sourceURL {
            lines.append("source: \(sourceURL.path)")
        }
        if let architecture = plan.architecture {
            lines.append("architecture: \(architecture)")
        }
        lines.append("actions: \(plan.actions.count)")

        for (index, action) in plan.actions.enumerated() {
            lines.append("")
            lines.append("[\(index + 1)] \(actionSummary(action))")
        }

        for warning in plan.warnings {
            lines.append("")
            lines.append("warning: \(warning)")
        }
        return lines.joined(separator: "\n")
    }

    static func render(repairPlan plan: GGUFRepairPlan, outputURL: URL) -> String {
        let lines = [render(plan: plan), "", "wrote repaired GGUF: \(outputURL.path)"]
        return lines.joined(separator: "\n")
    }

    private static func actionSummary(_ action: GGUFRepairAction) -> String {
        switch action {
        case .addMetadata(let key, let value, let rationale):
            return "add \(key)=\(value.displayString) (\(rationale))"
        case .replaceMetadata(let key, let oldValue, let newValue, let rationale):
            return "replace \(key): \(oldValue.displayString) -> \(newValue.displayString) (\(rationale))"
        }
    }
}

enum JSONOutput {
    static func render(report: GGUFValidationReport) throws -> String {
        let payload: [String: Any] = [
            "architecture": optionalValue(report.architecture),
            "family": optionalValue(report.family),
            "hasErrors": report.hasErrors,
            "issues": report.issues.map(issueDictionary),
            "warnings": report.warnings,
        ]
        return try serialize(payload)
    }

    static func render(plan: GGUFRepairPlan) throws -> String {
        let payload: [String: Any] = [
            "sourceURL": optionalValue(plan.sourceURL?.path),
            "architecture": optionalValue(plan.architecture),
            "actions": plan.actions.map(actionDictionary),
            "warnings": plan.warnings,
        ]
        return try serialize(payload)
    }

    private static func issueDictionary(_ issue: GGUFValidationIssue) -> [String: Any] {
        [
            "severity": issue.severity.rawValue,
            "kind": issue.kind.rawValue,
            "metadataKey": issue.metadataKey,
            "message": issue.message,
            "evidence": issue.evidence,
            "suggestedValue": optionalValue(issue.suggestedValue.map(valueDictionary)),
        ]
    }

    private static func actionDictionary(_ action: GGUFRepairAction) -> [String: Any] {
        switch action {
        case .addMetadata(let key, let value, let rationale):
            return [
                "kind": "addMetadata",
                "key": key,
                "value": valueDictionary(value),
                "rationale": rationale,
            ]
        case .replaceMetadata(let key, let oldValue, let newValue, let rationale):
            return [
                "kind": "replaceMetadata",
                "key": key,
                "oldValue": valueDictionary(oldValue),
                "newValue": valueDictionary(newValue),
                "rationale": rationale,
            ]
        }
    }

    private static func valueDictionary(_ value: GGUFMetadataValue) -> [String: Any] {
        [
            "type": String(describing: value.ggufType),
            "displayValue": value.displayString,
        ]
    }

    private static func serialize(_ payload: [String: Any]) throws -> String {
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        guard let string = String(data: data, encoding: .utf8) else {
            throw CocoaError(.fileWriteInapplicableStringEncoding)
        }
        return string
    }

    private static func optionalValue(_ value: Any?) -> Any {
        value ?? NSNull()
    }
}
