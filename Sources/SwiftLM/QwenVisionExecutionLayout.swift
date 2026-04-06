struct QwenVisionExecutionLayout: Sendable, Equatable {
    let layout: PromptLayout
    let imageGridTHW: [[Int]]
    let videoGridTHW: [[Int]]

    var mmTokenTypeIDs: [Int] {
        layout.tokenTypeIDs
    }

    var ropePositionIDs: QwenVisionRopePositionIDs {
        layout.ropePositionIDs
    }

    var mropePositionDelta: Int {
        layout.ropePositionDelta
    }
}

public struct QwenVisionRopePositionIDs: Sendable, Equatable {
    public let temporal: [Int]
    public let height: [Int]
    public let width: [Int]

    public init(temporal: [Int], height: [Int], width: [Int]) {
        self.temporal = temporal
        self.height = height
        self.width = width
    }

    var count: Int { temporal.count }

    func axes(at index: Int) -> (UInt32, UInt32, UInt32) {
        (
            UInt32(temporal[index]),
            UInt32(height[index]),
            UInt32(width[index])
        )
    }
}
