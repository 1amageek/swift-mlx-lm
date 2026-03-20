import Metal

public enum STAFWeightLayout: Sendable {
    case rowMajor
    case blockedRows4Tiles128
    case blockedRows8Tiles128
}

public enum STAFWeightExecutionPhase: Sendable {
    case decode
    case prefill
}

public enum STAFWeightLayoutPreference: Sendable {
    case canonicalRowMajor
    case optimized(STAFWeightLayout)

    var requestedLayout: STAFWeightLayout {
        switch self {
        case .canonicalRowMajor:
            return .rowMajor
        case .optimized(let layout):
            return layout
        }
    }
}

public struct STAFWeightAccessRequest: Sendable {
    public let tensorName: String
    public let executionPhase: STAFWeightExecutionPhase
    public let layoutPreference: STAFWeightLayoutPreference

    public init(
        tensorName: String,
        executionPhase: STAFWeightExecutionPhase = .decode,
        layoutPreference: STAFWeightLayoutPreference = .canonicalRowMajor
    ) {
        self.tensorName = tensorName
        self.executionPhase = executionPhase
        self.layoutPreference = layoutPreference
    }

    public var preferredLayout: STAFWeightLayout {
        switch executionPhase {
        case .decode:
            return layoutPreference.requestedLayout
        case .prefill:
            return .rowMajor
        }
    }
}

public struct STAFWeightBufferAccess: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let offset: Int
    public let size: Int
    public let layout: STAFWeightLayout

    public init(
        buffer: MTLBuffer,
        offset: Int,
        size: Int,
        layout: STAFWeightLayout
    ) {
        self.buffer = buffer
        self.offset = offset
        self.size = size
        self.layout = layout
    }
}

struct STAFSpecializedWeightKey: Hashable {
    let tensorName: String
    let layout: STAFWeightLayout
}
