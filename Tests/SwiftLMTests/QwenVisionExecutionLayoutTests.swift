import Testing
@testable import SwiftLM

@Suite("Qwen Vision Execution Layout")
struct QwenVisionExecutionLayoutTests {
    @Test("Build official-style rope indices for text-image-text prompts")
    func buildRoPEIndices() throws {
        let prepared = PreparedInput(
            renderedText: "ignored",
            tokenIDs: Array(repeating: 0, count: 68),
            multimodalMetadata: PreparedInput.Multimodal(
                mmTokenTypeIDs: [0, 0] + Array(repeating: 1, count: 64) + [0, 0],
                images: [
                    .init(
                        gridTHW: [1, 16, 16],
                        placeholderTokenCount: 64
                    )
                ]
            )
        )

        let layout = try QwenVisionExecutionLayoutBuilder().makeLayout(for: prepared)
        #expect(layout.mmTokenTypeIDs.count == 68)
        #expect(layout.ropePositionIDs.count == 68)
        #expect(layout.ropePositionIDs.axes(at: 0) == (0, 0, 0))
        #expect(layout.ropePositionIDs.axes(at: 1) == (1, 1, 1))
        #expect(layout.ropePositionIDs.axes(at: 2) == (2, 2, 2))
        #expect(layout.ropePositionIDs.axes(at: 65) == (2, 9, 9))
        #expect(layout.ropePositionIDs.axes(at: 66) == (10, 10, 10))
        #expect(layout.ropePositionIDs.axes(at: 67) == (11, 11, 11))
        #expect(layout.mropePositionDelta == -56)
    }

    @Test("Reject mismatched image placeholder runs")
    func rejectMismatchedImagePlaceholderRuns() throws {
        let prepared = PreparedInput(
            renderedText: "ignored",
            tokenIDs: Array(repeating: 0, count: 5),
            multimodalMetadata: PreparedInput.Multimodal(
                mmTokenTypeIDs: [0, 1, 1, 1, 0],
                images: [
                    .init(
                        gridTHW: [1, 16, 16],
                        placeholderTokenCount: 64
                    )
                ]
            )
        )

        #expect(throws: ModelContainerError.self) {
            _ = try QwenVisionExecutionLayoutBuilder().makeLayout(for: prepared)
        }
    }

    @Test("Split Qwen video runs per frame when timestamps separate placeholders")
    func buildVideoRoPEIndices() throws {
        let prepared = PreparedInput(
            renderedText: "ignored",
            tokenIDs: Array(repeating: 0, count: 35),
            multimodalMetadata: PreparedInput.Multimodal(
                mmTokenTypeIDs: [0]
                    + Array(repeating: 2, count: 16)
                    + [0]
                    + Array(repeating: 2, count: 16)
                    + [0],
                videos: [
                    .init(
                        gridTHW: [2, 8, 8],
                        placeholderTokenCount: 32,
                        frameTimestamps: [0.0, 1.0]
                    )
                ]
            )
        )

        let layout = try QwenVisionExecutionLayoutBuilder().makeLayout(for: prepared)
        #expect(layout.mmTokenTypeIDs.count == 35)
        #expect(layout.ropePositionIDs.count == 35)
        #expect(layout.ropePositionIDs.axes(at: 0) == (0, 0, 0))
        #expect(layout.ropePositionIDs.axes(at: 1) == (1, 1, 1))
        #expect(layout.ropePositionIDs.axes(at: 16) == (1, 4, 4))
        #expect(layout.ropePositionIDs.axes(at: 17) == (5, 5, 5))
        #expect(layout.ropePositionIDs.axes(at: 18) == (6, 6, 6))
        #expect(layout.ropePositionIDs.axes(at: 33) == (6, 9, 9))
        #expect(layout.ropePositionIDs.axes(at: 34) == (10, 10, 10))
        #expect(layout.mropePositionDelta == -24)
    }

    @Test("Build official-style rope indices for text-video-text prompts")
    func buildTextVideoTextRoPEIndices() throws {
        let prepared = PreparedInput(
            renderedText: "ignored",
            tokenIDs: Array(repeating: 0, count: 10),
            multimodalMetadata: PreparedInput.Multimodal(
                mmTokenTypeIDs: [0, 0, 2, 2, 2, 2, 0, 0, 0, 0],
                videos: [
                    .init(
                        gridTHW: [1, 4, 4],
                        placeholderTokenCount: 4,
                        frameTimestamps: [0.5]
                    )
                ]
            )
        )

        let layout = try QwenVisionExecutionLayoutBuilder().makeLayout(for: prepared)
        #expect(layout.ropePositionIDs.axes(at: 0) == (0, 0, 0))
        #expect(layout.ropePositionIDs.axes(at: 1) == (1, 1, 1))
        #expect(layout.ropePositionIDs.axes(at: 2) == (2, 2, 2))
        #expect(layout.ropePositionIDs.axes(at: 5) == (2, 3, 3))
        #expect(layout.ropePositionIDs.axes(at: 6) == (4, 4, 4))
        #expect(layout.ropePositionIDs.axes(at: 9) == (7, 7, 7))
        #expect(layout.mropePositionDelta == -2)
    }

    @Test("Build rope indices for mixed image and video prompts")
    func buildMixedImageAndVideoRoPEIndices() throws {
        let prepared = PreparedInput(
            renderedText: "ignored",
            tokenIDs: Array(repeating: 0, count: 9),
            multimodalMetadata: PreparedInput.Multimodal(
                mmTokenTypeIDs: [0, 1, 0, 2, 0, 2, 0, 1, 0],
                images: [
                    .init(gridTHW: [1, 2, 2], placeholderTokenCount: 1),
                    .init(gridTHW: [1, 2, 2], placeholderTokenCount: 1),
                ],
                videos: [
                    .init(gridTHW: [2, 2, 2], placeholderTokenCount: 2, frameTimestamps: [0.0, 1.0]),
                ]
            )
        )

        let layout = try QwenVisionExecutionLayoutBuilder().makeLayout(for: prepared)
        #expect(layout.imageGridTHW == [[1, 2, 2], [1, 2, 2]])
        #expect(layout.videoGridTHW == [[2, 2, 2]])
        #expect(layout.ropePositionIDs.axes(at: 0) == (0, 0, 0))
        #expect(layout.ropePositionIDs.axes(at: 1) == (1, 1, 1))
        #expect(layout.ropePositionIDs.axes(at: 2) == (2, 2, 2))
        #expect(layout.ropePositionIDs.axes(at: 3) == (3, 3, 3))
        #expect(layout.ropePositionIDs.axes(at: 5) == (5, 5, 5))
        #expect(layout.ropePositionIDs.axes(at: 7) == (7, 7, 7))
        #expect(layout.ropePositionIDs.axes(at: 8) == (8, 8, 8))
        #expect(layout.mropePositionDelta == 0)
    }

    @Test("Parity fixture matches official-style video layout positions")
    func layoutParityFixture() throws {
        let parity = try QwenVisionTestSupport.parityFixture()
        let prepared = PreparedInput(
            renderedText: "ignored",
            tokenIDs: Array(repeating: 0, count: 35),
            multimodalMetadata: PreparedInput.Multimodal(
                mmTokenTypeIDs: [0]
                    + Array(repeating: 2, count: 16)
                    + [0]
                    + Array(repeating: 2, count: 16)
                    + [0],
                videos: [
                    .init(
                        gridTHW: parity.videoPrompt.gridTHW,
                        placeholderTokenCount: parity.videoPrompt.placeholderTokenCount,
                        frameTimestamps: [0.0, 1.0]
                    )
                ]
            )
        )

        let layout = try QwenVisionExecutionLayoutBuilder().makeLayout(for: prepared)
        #expect(layout.mropePositionDelta == parity.videoLayout.mropePositionDelta)
        for (indexString, axes) in parity.videoLayout.axesByIndex {
            let index = try #require(Int(indexString))
            let actual = layout.ropePositionIDs.axes(at: index)
            #expect(Int(actual.0) == axes[0])
            #expect(Int(actual.1) == axes[1])
            #expect(Int(actual.2) == axes[2])
        }
    }
}
