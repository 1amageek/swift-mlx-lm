import Foundation

struct Input2048GEMVSourcePolicy: Sendable {
    let fixedOutputDimension: Int
    let fixedRowsPerThreadgroup: Int?
    let stagesInputAsFloat: Bool
    let unrollFactor: Int
    let usesPairwiseBF16ArgumentRead: Bool

    static func square(weightFormat: WeightFormat) -> Self {
        Self(
            fixedOutputDimension: 2_048,
            fixedRowsPerThreadgroup: 8,
            stagesInputAsFloat: true,
            unrollFactor: 8,
            usesPairwiseBF16ArgumentRead: weightFormat == .bfloat16
        )
    }

    static func expanded6144(weightFormat: WeightFormat) -> Self {
        Self(
            fixedOutputDimension: 6_144,
            fixedRowsPerThreadgroup: 8,
            stagesInputAsFloat: false,
            unrollFactor: 4,
            usesPairwiseBF16ArgumentRead: weightFormat == .bfloat16
        )
    }

    static func expanded8192(weightFormat: WeightFormat) -> Self {
        Self(
            fixedOutputDimension: 8_192,
            fixedRowsPerThreadgroup: nil,
            stagesInputAsFloat: true,
            unrollFactor: 4,
            usesPairwiseBF16ArgumentRead: false
        )
    }
}
