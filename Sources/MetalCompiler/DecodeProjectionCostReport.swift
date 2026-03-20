import Foundation

public struct DecodeProjectionCostReport: Sendable {
    public struct FamilyEstimate: Sendable {
        public let kernelName: String
        public let inputDimension: Int
        public let outputDimension: Int
        public let stepCount: Int
        public let weightTensorCount: Int
        public let layouts: [STAFWeightLayout]
        public let formatIdentifiers: [QuantizationSchemeIdentifier]
        public let inputBytesPerStep: Int64
        public let weightBytesPerStep: Int64
        public let outputBytesPerStep: Int64
        public let estimatedFLOPsPerStep: Int64

        public var totalBytesPerStep: Int64 {
            inputBytesPerStep + weightBytesPerStep + outputBytesPerStep
        }

        public var totalEstimatedBytes: Int64 {
            totalBytesPerStep * Int64(stepCount)
        }

        public var totalEstimatedFLOPs: Int64 {
            estimatedFLOPsPerStep * Int64(stepCount)
        }

        public var arithmeticIntensity: Double {
            guard totalBytesPerStep > 0 else { return 0 }
            return Double(estimatedFLOPsPerStep) / Double(totalBytesPerStep)
        }
    }

    public let totalProjectionSteps: Int
    public let families: [FamilyEstimate]

    public func formatted(limit: Int? = nil) -> String {
        var lines: [String] = []
        lines.append("Decode Projection Cost Report (\(totalProjectionSteps) projection steps)")
        let header = "Kernel".padding(toLength: 34, withPad: " ", startingAt: 0)
            + "Steps  In→Out  Wt/step  Total   AI  Layouts"
        lines.append(header)
        lines.append(String(repeating: "-", count: 96))

        let selectedFamilies: ArraySlice<FamilyEstimate>
        if let limit {
            selectedFamilies = families.prefix(limit)
        } else {
            selectedFamilies = ArraySlice(families)
        }

        for family in selectedFamilies {
            let kernel = family.kernelName.padding(toLength: 34, withPad: " ", startingAt: 0)
            let dimensions = "\(family.inputDimension)→\(family.outputDimension)".padding(toLength: 10, withPad: " ", startingAt: 0)
            let layouts = family.layouts.map(Self.describe(layout:)).joined(separator: ",")
            let weightMegabytes = Double(family.weightBytesPerStep) / 1_048_576
            let totalMegabytes = Double(family.totalEstimatedBytes) / 1_048_576
            lines.append(
                "\(kernel)\(String(format: "%5d  %@", family.stepCount, dimensions))"
                + "\(String(format: "%7.1f", weightMegabytes))MB "
                + "\(String(format: "%6.1f", totalMegabytes))MB "
                + "\(String(format: "%4.2f", family.arithmeticIntensity)) "
                + layouts
            )
        }

        return lines.joined(separator: "\n")
    }

    private static func describe(layout: STAFWeightLayout) -> String {
        switch layout {
        case .rowMajor:
            return "rowMajor"
        case .blockedRows4Tiles128:
            return "blocked4x128"
        case .blockedRows8Tiles128:
            return "blocked8x128"
        }
    }
}
