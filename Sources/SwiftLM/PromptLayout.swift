public struct PromptLayout: Sendable, Equatable {
    public var tokenTypeIDs: [Int]
    public var segments: [PromptSegment]
    public var ropePositionIDs: QwenVisionRopePositionIDs
    public var ropePositionDelta: Int

    public init(
        tokenTypeIDs: [Int],
        segments: [PromptSegment],
        ropePositionIDs: QwenVisionRopePositionIDs,
        ropePositionDelta: Int
    ) {
        self.tokenTypeIDs = tokenTypeIDs
        self.segments = segments
        self.ropePositionIDs = ropePositionIDs
        self.ropePositionDelta = ropePositionDelta
    }

    public var mmTokenTypeIDs: [Int] {
        tokenTypeIDs
    }

    public var mropePositionDelta: Int {
        ropePositionDelta
    }
}

public struct PromptSegment: Sendable, Equatable {
    public var modality: Int
    public var tokenRange: Range<Int>

    public init(modality: Int, tokenRange: Range<Int>) {
        self.modality = modality
        self.tokenRange = tokenRange
    }

    public var tokenCount: Int {
        tokenRange.count
    }
}
