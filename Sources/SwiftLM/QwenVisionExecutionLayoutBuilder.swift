import Foundation

struct QwenVisionExecutionLayoutBuilder {
    func makeLayout(for prepared: PreparedPrompt) throws -> QwenVisionExecutionLayout {
        guard let multimodal = prepared.multimodalMetadata else {
            return QwenVisionExecutionLayout(
                layout: PromptLayout(
                    tokenTypeIDs: [],
                    segments: [],
                    ropePositionIDs: QwenVisionRopePositionIDs(temporal: [], height: [], width: []),
                    ropePositionDelta: 0
                ),
                imageGridTHW: [],
                videoGridTHW: []
            )
        }

        let mmTokenTypeIDs = multimodal.mmTokenTypeIDs
        guard mmTokenTypeIDs.count == prepared.tokenIDs.count else {
            throw LanguageModelContextError.multimodalInputNotSupported(
                "Multimodal token type IDs must align with the tokenized prompt."
            )
        }

        let segments = contiguousSegments(in: mmTokenTypeIDs)

        let imageGridTHW = multimodal.images.map(\.gridTHW)
        let videoGridTHW = multimodal.videos.map(\.gridTHW)
        let imagePlaceholderCounts = multimodal.images.map(\.placeholderTokenCount)
        let videoPlaceholderCounts = multimodal.videos.map(\.placeholderTokenCount)
        let expandedVideoGrids = try expandedVideoGridTHW(
            videoGridTHW: videoGridTHW,
            videoPlaceholderCounts: videoPlaceholderCounts
        )
        let positionIDs = try makeRoPEPositionIDs(
            mmTokenTypeIDs: mmTokenTypeIDs,
            imageGridTHW: imageGridTHW,
            imagePlaceholderCounts: imagePlaceholderCounts,
            videoGridTHW: expandedVideoGrids.grids,
            videoPlaceholderCounts: expandedVideoGrids.placeholderCounts
        )
        return QwenVisionExecutionLayout(
            layout: PromptLayout(
                tokenTypeIDs: mmTokenTypeIDs,
                segments: segments,
                ropePositionIDs: positionIDs.ids,
                ropePositionDelta: positionIDs.delta
            ),
            imageGridTHW: imageGridTHW,
            videoGridTHW: videoGridTHW
        )
    }

    private func expandedVideoGridTHW(
        videoGridTHW: [[Int]],
        videoPlaceholderCounts: [Int]
    ) throws -> (grids: [[Int]], placeholderCounts: [Int]) {
        var expandedGrids: [[Int]] = []
        var expandedPlaceholderCounts: [Int] = []
        for (grid, placeholderCount) in zip(videoGridTHW, videoPlaceholderCounts) {
            guard grid.count == 3, grid[0] > 0 else {
                throw LanguageModelContextError.multimodalInputNotSupported(
                    "Video grid metadata is invalid for Qwen multimodal execution."
                )
            }
            guard placeholderCount % grid[0] == 0 else {
                throw LanguageModelContextError.multimodalInputNotSupported(
                    "Video placeholder count does not divide evenly across temporal frames."
                )
            }
            let framePlaceholderCount = placeholderCount / grid[0]
            for _ in 0..<grid[0] {
                expandedGrids.append([1, grid[1], grid[2]])
                expandedPlaceholderCounts.append(framePlaceholderCount)
            }
        }
        return (expandedGrids, expandedPlaceholderCounts)
    }

    private func makeRoPEPositionIDs(
        mmTokenTypeIDs: [Int],
        imageGridTHW: [[Int]],
        imagePlaceholderCounts: [Int],
        videoGridTHW: [[Int]],
        videoPlaceholderCounts: [Int]
    ) throws -> (ids: QwenVisionRopePositionIDs, delta: Int) {
        var temporal: [Int] = []
        var height: [Int] = []
        var width: [Int] = []
        temporal.reserveCapacity(mmTokenTypeIDs.count)
        height.reserveCapacity(mmTokenTypeIDs.count)
        width.reserveCapacity(mmTokenTypeIDs.count)

        let groupedRuns = contiguousRuns(in: mmTokenTypeIDs)
        var currentPosition = 0
        var nextImageIndex = 0
        var nextVideoIndex = 0

        for run in groupedRuns {
            switch run.modality {
            case 0:
                for offset in 0..<run.length {
                    let value = currentPosition + offset
                    temporal.append(value)
                    height.append(value)
                    width.append(value)
                }
                currentPosition += run.length
            case 1:
                guard nextImageIndex < imageGridTHW.count else {
                    throw LanguageModelContextError.multimodalInputNotSupported(
                        "Multimodal layout is missing image grid metadata."
                    )
                }
                let grid = imageGridTHW[nextImageIndex]
                let expectedCount = imagePlaceholderCounts[nextImageIndex]
                guard run.length == expectedCount else {
                    throw LanguageModelContextError.multimodalInputNotSupported(
                        "Image placeholder count does not match the Qwen processor grid."
                    )
                }
                let spatialMergeSize = resolveSpatialMergeSize(
                    gridTHW: grid,
                    placeholderTokenCount: expectedCount
                )
                appendVisionPositionIDs(
                    startPosition: currentPosition,
                    gridTHW: grid,
                    spatialMergeSize: spatialMergeSize,
                    temporal: &temporal,
                    height: &height,
                    width: &width
                )
                currentPosition += max(grid[1], grid[2]) / spatialMergeSize
                nextImageIndex += 1
            case 2:
                guard nextVideoIndex < videoGridTHW.count else {
                    throw LanguageModelContextError.multimodalInputNotSupported(
                        "Multimodal layout is missing video grid metadata."
                    )
                }
                let grid = videoGridTHW[nextVideoIndex]
                let expectedCount = videoPlaceholderCounts[nextVideoIndex]
                guard run.length == expectedCount else {
                    throw LanguageModelContextError.multimodalInputNotSupported(
                        "Video placeholder count does not match the Qwen processor grid."
                    )
                }
                let spatialMergeSize = resolveSpatialMergeSize(
                    gridTHW: grid,
                    placeholderTokenCount: expectedCount
                )
                appendVisionPositionIDs(
                    startPosition: currentPosition,
                    gridTHW: grid,
                    spatialMergeSize: spatialMergeSize,
                    temporal: &temporal,
                    height: &height,
                    width: &width
                )
                currentPosition += max(grid[1], grid[2]) / spatialMergeSize
                nextVideoIndex += 1
            default:
                throw LanguageModelContextError.multimodalInputNotSupported(
                    "Unsupported multimodal token type ID: \(run.modality)"
                )
            }
        }

        guard temporal.count == mmTokenTypeIDs.count else {
            throw LanguageModelContextError.multimodalInputNotSupported(
                "Failed to build Qwen multimodal RoPE indices."
            )
        }

        let maxPosition = max(
            temporal.max() ?? 0,
            height.max() ?? 0,
            width.max() ?? 0
        )
        let delta = maxPosition + 1 - mmTokenTypeIDs.count
        return (
            ids: QwenVisionRopePositionIDs(
                temporal: temporal,
                height: height,
                width: width
            ),
            delta: delta
        )
    }

    private func appendVisionPositionIDs(
        startPosition: Int,
        gridTHW: [Int],
        spatialMergeSize: Int,
        temporal: inout [Int],
        height: inout [Int],
        width: inout [Int]
    ) {
        let temporalCount = gridTHW[0]
        let heightCount = gridTHW[1] / spatialMergeSize
        let widthCount = gridTHW[2] / spatialMergeSize
        let imageSequenceLength = temporalCount * heightCount * widthCount
        temporal.reserveCapacity(temporal.count + imageSequenceLength)
        height.reserveCapacity(height.count + imageSequenceLength)
        width.reserveCapacity(width.count + imageSequenceLength)

        for row in 0..<heightCount {
            for column in 0..<widthCount {
                for _ in 0..<temporalCount {
                    temporal.append(startPosition)
                    height.append(startPosition + row)
                    width.append(startPosition + column)
                }
            }
        }
    }

    private func contiguousRuns(in tokenTypes: [Int]) -> [(modality: Int, length: Int)] {
        guard let first = tokenTypes.first else { return [] }
        var runs: [(modality: Int, length: Int)] = []
        var currentModality = first
        var currentLength = 1

        for tokenType in tokenTypes.dropFirst() {
            if tokenType == currentModality {
                currentLength += 1
            } else {
                runs.append((modality: currentModality, length: currentLength))
                currentModality = tokenType
                currentLength = 1
            }
        }
        runs.append((modality: currentModality, length: currentLength))
        return runs
    }

    private func contiguousSegments(in tokenTypes: [Int]) -> [PromptSegment] {
        guard let first = tokenTypes.first else { return [] }
        var segments: [PromptSegment] = []
        var currentModality = first
        var startIndex = 0

        for (index, tokenType) in tokenTypes.enumerated().dropFirst() {
            if tokenType == currentModality {
                continue
            }
            segments.append(
                PromptSegment(
                    modality: currentModality,
                    tokenRange: startIndex..<index
                )
            )
            currentModality = tokenType
            startIndex = index
        }

        segments.append(
            PromptSegment(
                modality: currentModality,
                tokenRange: startIndex..<tokenTypes.count
            )
        )
        return segments
    }

    private func resolveSpatialMergeSize(
        gridTHW: [Int],
        placeholderTokenCount: Int
    ) -> Int {
        guard placeholderTokenCount > 0 else {
            return 1
        }
        let tokenRatio = max(1, (gridTHW[0] * gridTHW[1] * gridTHW[2]) / placeholderTokenCount)
        let merge = Int(Double(tokenRatio).squareRoot())
        return max(1, merge)
    }
}
