import Foundation

public enum Input2048BF16ArgumentReadPolicy: Sendable {
    case scalar
    case pairwise
    case pairwisePointerInput
    case pairwisePointerFloatInput
    case packed4PointerInput
    case packed4FixedPointerInput
    case packed4ThreadgroupFixedPointerInput
}

public enum Input2048WeightLayoutPolicy: Sendable {
    case rowMajor
    case blockedRows4Tiles128
    case blockedRows8Tiles128
}

extension Input2048WeightLayoutPolicy {
    var stafWeightLayout: STAFWeightLayout {
        switch self {
        case .rowMajor:
            return .rowMajor
        case .blockedRows4Tiles128:
            return .blockedRows4Tiles128
        case .blockedRows8Tiles128:
            return .blockedRows8Tiles128
        }
    }

    var kernelNameSuffix: String {
        switch self {
        case .rowMajor:
            return ""
        case .blockedRows4Tiles128:
            return "_blocked4x128"
        case .blockedRows8Tiles128:
            return "_blocked8x128"
        }
    }

    init(stafWeightLayout: STAFWeightLayout) {
        switch stafWeightLayout {
        case .rowMajor:
            self = .rowMajor
        case .blockedRows4Tiles128:
            self = .blockedRows4Tiles128
        case .blockedRows8Tiles128:
            self = .blockedRows8Tiles128
        }
    }
}

struct Input2048GEMVSourcePolicy: Sendable {
    let fixedOutputDimension: Int
    let fixedRowsPerThreadgroup: Int?
    let fixedSimdgroups: Int?
    let stagesInputAsFloat: Bool
    let unrollFactor: Int
    let bf16ArgumentReadPolicy: Input2048BF16ArgumentReadPolicy
    let weightLayoutPolicy: Input2048WeightLayoutPolicy

    static func square(weightFormat: WeightFormat) -> Self {
        Self(
            fixedOutputDimension: 2_048,
            fixedRowsPerThreadgroup: 8,
            fixedSimdgroups: nil,
            stagesInputAsFloat: true,
            unrollFactor: 8,
            bf16ArgumentReadPolicy: weightFormat == .bfloat16 ? .pairwisePointerFloatInput : .scalar,
            weightLayoutPolicy: .rowMajor
        )
    }

    static func expanded6144(weightFormat: WeightFormat) -> Self {
        Self(
            fixedOutputDimension: 6_144,
            fixedRowsPerThreadgroup: 4,
            fixedSimdgroups: 4,
            stagesInputAsFloat: false,
            unrollFactor: 4,
            bf16ArgumentReadPolicy: weightFormat == .bfloat16 ? .packed4ThreadgroupFixedPointerInput : .scalar,
            weightLayoutPolicy: .rowMajor
        )
    }

    static func expanded8192(weightFormat: WeightFormat) -> Self {
        Self(
            fixedOutputDimension: 8_192,
            fixedRowsPerThreadgroup: nil,
            fixedSimdgroups: nil,
            stagesInputAsFloat: true,
            unrollFactor: 4,
            bf16ArgumentReadPolicy: .scalar,
            weightLayoutPolicy: .rowMajor
        )
    }

    func with(weightLayoutPolicy: Input2048WeightLayoutPolicy) -> Self {
        Self(
            fixedOutputDimension: fixedOutputDimension,
            fixedRowsPerThreadgroup: fixedRowsPerThreadgroup,
            fixedSimdgroups: fixedSimdgroups,
            stagesInputAsFloat: stagesInputAsFloat,
            unrollFactor: unrollFactor,
            bf16ArgumentReadPolicy: bf16ArgumentReadPolicy,
            weightLayoutPolicy: weightLayoutPolicy
        )
    }
}
