import Foundation

struct QwenVisionEncodedOutputs: Sendable {
    let visualTokenEmbeddings: [[Float]]
    let deepstackFeaturesByLayer: [Int: [[Float]]]
}

final class QwenVisionEncoder {
    private let configuration: ModelVisionConfiguration
    private let weights: QwenVisionWeightStore

    private let hiddenSize: Int
    private let intermediateSize: Int
    private let outHiddenSize: Int
    private let headCount: Int
    private let inChannels: Int
    private let depth: Int
    private let patchSize: Int
    private let temporalPatchSize: Int
    private let spatialMergeSize: Int
    private let numPositionEmbeddings: Int
    private let ropeTheta: Float
    private let hiddenAct: String

    init(configuration: ModelVisionConfiguration, weights: QwenVisionWeightStore) throws {
        guard let hiddenSize = configuration.hiddenSize else {
            throw ModelBundleLoaderError.invalidConfig("vision_config.hidden_size is required")
        }
        guard let intermediateSize = configuration.intermediateSize else {
            throw ModelBundleLoaderError.invalidConfig("vision_config.intermediate_size is required")
        }
        guard let outHiddenSize = configuration.outHiddenSize else {
            throw ModelBundleLoaderError.invalidConfig("vision_config.out_hidden_size is required")
        }
        guard let headCount = configuration.headCount else {
            throw ModelBundleLoaderError.invalidConfig("vision_config.num_heads is required")
        }
        guard let inChannels = configuration.inChannels else {
            throw ModelBundleLoaderError.invalidConfig("vision_config.in_channels is required")
        }
        guard let depth = configuration.depth else {
            throw ModelBundleLoaderError.invalidConfig("vision_config.depth is required")
        }
        guard let patchSize = configuration.patchSize else {
            throw ModelBundleLoaderError.invalidConfig("vision_config.patch_size is required")
        }
        guard let temporalPatchSize = configuration.temporalPatchSize else {
            throw ModelBundleLoaderError.invalidConfig("vision_config.temporal_patch_size is required")
        }
        guard let spatialMergeSize = configuration.spatialMergeSize ?? configuration.mergeSize else {
            throw ModelBundleLoaderError.invalidConfig("vision_config.spatial_merge_size is required")
        }
        guard let numPositionEmbeddings = configuration.numPositionEmbeddings else {
            throw ModelBundleLoaderError.invalidConfig(
                "vision_config.num_position_embeddings is required"
            )
        }

        self.configuration = configuration
        self.weights = weights
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.outHiddenSize = outHiddenSize
        self.headCount = headCount
        self.inChannels = inChannels
        self.depth = depth
        self.patchSize = patchSize
        self.temporalPatchSize = temporalPatchSize
        self.spatialMergeSize = spatialMergeSize
        self.numPositionEmbeddings = numPositionEmbeddings
        self.ropeTheta = configuration.ropeTheta ?? 10_000.0
        self.hiddenAct = configuration.hiddenAct ?? "gelu"
    }

    func encode(images: [PreparedPrompt.Multimodal.Image]) throws -> QwenVisionEncodedOutputs {
        try encode(samples: images.map(VisionSample.init))
    }

    func encode(videos: [PreparedPrompt.Multimodal.Video]) throws -> QwenVisionEncodedOutputs {
        try encode(samples: videos.map(VisionSample.init))
    }

    private func encode(samples: [VisionSample]) throws -> QwenVisionEncodedOutputs {
        guard !samples.isEmpty else {
            return QwenVisionEncodedOutputs(visualTokenEmbeddings: [], deepstackFeaturesByLayer: [:])
        }

        let gridTHW = samples.map(\.gridTHW)
        let chunkLengths = gridTHW.map { $0[0] * $0[1] * $0[2] }
        var hiddenStates = try patchEmbed(samples: samples)
        let positionalEmbeddings = try interpolatePositionEmbeddings(for: gridTHW)
        hiddenStates = QwenVisionMath.add(hiddenStates, positionalEmbeddings)

        let rotaryEmbeddings = makeRotaryEmbeddings(for: gridTHW)
        var deepstackMerged: [(layerIndex: Int, values: [Float])] = []

        for layerIndex in 0..<depth {
            hiddenStates = try blockForward(
                hiddenStates: hiddenStates,
                layerIndex: layerIndex,
                chunkLengths: chunkLengths,
                rotaryEmbeddings: rotaryEmbeddings
            )
            if configuration.deepstackVisualIndexes.contains(layerIndex) {
                let mergerIndex = try deepstackMergerIndex(for: layerIndex)
                let merged = try patchMerge(
                    hiddenStates: hiddenStates,
                    prefix: "model.visual.deepstack_merger_list.\(mergerIndex)",
                    usePostshuffleNorm: true
                )
                deepstackMerged.append((layerIndex: layerIndex, values: merged))
            }
        }

        let pooled = try patchMerge(
            hiddenStates: hiddenStates,
            prefix: "model.visual.merger",
            usePostshuffleNorm: false
        )

        let placeholderCounts = samples.map(\.placeholderTokenCount)
        let visualTokenEmbeddings = splitRows(
            pooled,
            rowDimension: outHiddenSize,
            counts: placeholderCounts
        )

        var deepstackByLayer: [Int: [[Float]]] = [:]
        for (layerIndex, values) in deepstackMerged {
            deepstackByLayer[layerIndex] = splitRows(
                values,
                rowDimension: outHiddenSize,
                counts: placeholderCounts
            )
        }

        return QwenVisionEncodedOutputs(
            visualTokenEmbeddings: visualTokenEmbeddings,
            deepstackFeaturesByLayer: deepstackByLayer
        )
    }

    private func patchEmbed(samples: [VisionSample]) throws -> [Float] {
        let inputDimension = inChannels * temporalPatchSize * patchSize * patchSize
        let rowCount = samples.reduce(0) { partialResult, sample in
            partialResult + (sample.pixelValuesShape.first ?? 0)
        }
        let pixels = samples.flatMap(\.pixelValues)
        let weight = try weights.floatTensor(named: "model.visual.patch_embed.proj.weight")
        let bias = try weights.floatTensor(named: "model.visual.patch_embed.proj.bias")
        return QwenVisionMath.linear(
            input: pixels,
            rowCount: rowCount,
            inputDimension: inputDimension,
            weight: weight,
            outputDimension: hiddenSize,
            bias: bias
        )
    }

    private func interpolatePositionEmbeddings(for gridTHW: [[Int]]) throws -> [Float] {
        let weight = try weights.floatTensor(named: "model.visual.pos_embed.weight")
        let gridSide = Int(Double(numPositionEmbeddings).squareRoot())
        guard gridSide * gridSide == numPositionEmbeddings else {
            throw ModelBundleLoaderError.invalidConfig(
                "vision_config.num_position_embeddings must be a square"
            )
        }

        var output: [Float] = []
        output.reserveCapacity(gridTHW.reduce(0) { $0 + ($1[0] * $1[1] * $1[2] * hiddenSize) })
        for grid in gridTHW {
            let temporal = grid[0]
            let height = grid[1]
            let width = grid[2]
            let yCoordinates = linspace(count: height, upperBound: gridSide - 1)
            let xCoordinates = linspace(count: width, upperBound: gridSide - 1)

            let mergedHeight = height / spatialMergeSize
            let mergedWidth = width / spatialMergeSize
            for _ in 0..<temporal {
                for blockRow in 0..<mergedHeight {
                    for blockColumn in 0..<mergedWidth {
                        for intraRow in 0..<spatialMergeSize {
                            for intraColumn in 0..<spatialMergeSize {
                                let row = blockRow * spatialMergeSize + intraRow
                                let column = blockColumn * spatialMergeSize + intraColumn
                                output.append(
                                    contentsOf: bilinearLookup(
                                        weight: weight,
                                        gridSide: gridSide,
                                        row: yCoordinates[row],
                                        column: xCoordinates[column]
                                    )
                                )
                            }
                        }
                    }
                }
            }
        }
        return output
    }

    private func makeRotaryEmbeddings(for gridTHW: [[Int]]) -> (cos: [Float], sin: [Float]) {
        let headDimension = hiddenSize / headCount
        let freqCount = headDimension / 4
        let invFreq = (0..<freqCount).map { [ropeTheta] index -> Float in
            let exponent = Float(index * 2) / Float(headDimension / 2)
            return 1.0 / powf(ropeTheta, exponent)
        }

        var cos: [Float] = []
        var sin: [Float] = []
        cos.reserveCapacity(gridTHW.reduce(0) { $0 + ($1[0] * $1[1] * $1[2] * headDimension) })
        sin.reserveCapacity(cos.capacity)

        for grid in gridTHW {
            let temporal = grid[0]
            let height = grid[1]
            let width = grid[2]
            let mergedHeight = height / spatialMergeSize
            let mergedWidth = width / spatialMergeSize
            for _ in 0..<temporal {
                for blockRow in 0..<mergedHeight {
                    for blockColumn in 0..<mergedWidth {
                        for intraRow in 0..<spatialMergeSize {
                            for intraColumn in 0..<spatialMergeSize {
                                let row = Float(blockRow * spatialMergeSize + intraRow)
                                let column = Float(blockColumn * spatialMergeSize + intraColumn)
                                var halfEmbedding = [Float]()
                                halfEmbedding.reserveCapacity(headDimension / 2)
                                for frequency in invFreq {
                                    halfEmbedding.append(row * frequency)
                                }
                                for frequency in invFreq {
                                    halfEmbedding.append(column * frequency)
                                }
                                let fullEmbedding = halfEmbedding + halfEmbedding
                                cos.append(contentsOf: fullEmbedding.map(cosf))
                                sin.append(contentsOf: fullEmbedding.map(sinf))
                            }
                        }
                    }
                }
            }
        }

        return (cos: cos, sin: sin)
    }

    private func blockForward(
        hiddenStates: [Float],
        layerIndex: Int,
        chunkLengths: [Int],
        rotaryEmbeddings: (cos: [Float], sin: [Float])
    ) throws -> [Float] {
        let rowCount = hiddenStates.count / hiddenSize
        let prefix = "model.visual.blocks.\(layerIndex)"

        let norm1 = QwenVisionMath.layerNorm(
            input: hiddenStates,
            rowCount: rowCount,
            dimension: hiddenSize,
            weight: try weights.floatTensor(named: "\(prefix).norm1.weight"),
            bias: try weights.floatTensor(named: "\(prefix).norm1.bias")
        )
        let attention = try visionAttention(
            hiddenStates: norm1,
            prefix: "\(prefix).attn",
            chunkLengths: chunkLengths,
            rotaryEmbeddings: rotaryEmbeddings
        )
        let afterAttention = QwenVisionMath.add(hiddenStates, attention)

        let norm2 = QwenVisionMath.layerNorm(
            input: afterAttention,
            rowCount: rowCount,
            dimension: hiddenSize,
            weight: try weights.floatTensor(named: "\(prefix).norm2.weight"),
            bias: try weights.floatTensor(named: "\(prefix).norm2.bias")
        )
        let mlp = try visionMLP(hiddenStates: norm2, prefix: "\(prefix).mlp")
        return QwenVisionMath.add(afterAttention, mlp)
    }

    private func visionAttention(
        hiddenStates: [Float],
        prefix: String,
        chunkLengths: [Int],
        rotaryEmbeddings: (cos: [Float], sin: [Float])
    ) throws -> [Float] {
        let rowCount = hiddenStates.count / hiddenSize
        let headDimension = hiddenSize / headCount
        let qkv = QwenVisionMath.linear(
            input: hiddenStates,
            rowCount: rowCount,
            inputDimension: hiddenSize,
            weight: try weights.floatTensor(named: "\(prefix).qkv.weight"),
            outputDimension: hiddenSize * 3,
            bias: try weights.floatTensor(named: "\(prefix).qkv.bias")
        )

        var query = [Float](repeating: 0, count: rowCount * hiddenSize)
        var key = [Float](repeating: 0, count: rowCount * hiddenSize)
        var value = [Float](repeating: 0, count: rowCount * hiddenSize)
        for tokenIndex in 0..<rowCount {
            let sourceBase = tokenIndex * hiddenSize * 3
            let targetBase = tokenIndex * hiddenSize
            query.replaceSubrange(
                targetBase..<(targetBase + hiddenSize),
                with: qkv[sourceBase..<(sourceBase + hiddenSize)]
            )
            key.replaceSubrange(
                targetBase..<(targetBase + hiddenSize),
                with: qkv[(sourceBase + hiddenSize)..<(sourceBase + hiddenSize * 2)]
            )
            value.replaceSubrange(
                targetBase..<(targetBase + hiddenSize),
                with: qkv[(sourceBase + hiddenSize * 2)..<(sourceBase + hiddenSize * 3)]
            )
        }

        applyRotary(
            query: &query,
            key: &key,
            rowCount: rowCount,
            headCount: headCount,
            headDimension: headDimension,
            cos: rotaryEmbeddings.cos,
            sin: rotaryEmbeddings.sin
        )

        let scale = 1.0 / sqrt(Float(headDimension))
        var attentionOutput = [Float](repeating: 0, count: rowCount * hiddenSize)
        var chunkStart = 0
        for chunkLength in chunkLengths {
            for headIndex in 0..<headCount {
                let queryChunk = sliceHead(
                    query,
                    chunkStart: chunkStart,
                    chunkLength: chunkLength,
                    headIndex: headIndex,
                    headCount: headCount,
                    headDimension: headDimension
                )
                let keyChunk = sliceHead(
                    key,
                    chunkStart: chunkStart,
                    chunkLength: chunkLength,
                    headIndex: headIndex,
                    headCount: headCount,
                    headDimension: headDimension
                )
                var scores = QwenVisionMath.linear(
                    input: queryChunk,
                    rowCount: chunkLength,
                    inputDimension: headDimension,
                    weight: keyChunk,
                    outputDimension: chunkLength
                )
                for index in scores.indices {
                    scores[index] *= scale
                }
                let probabilities = QwenVisionMath.softmaxRows(
                    scores,
                    rowCount: chunkLength,
                    columnCount: chunkLength
                )
                let valueChunk = sliceHead(
                    value,
                    chunkStart: chunkStart,
                    chunkLength: chunkLength,
                    headIndex: headIndex,
                    headCount: headCount,
                    headDimension: headDimension
                )
                let context = QwenVisionMath.linear(
                    input: probabilities,
                    rowCount: chunkLength,
                    inputDimension: chunkLength,
                    weight: transposeRows(
                        valueChunk,
                        rowCount: chunkLength,
                        columnCount: headDimension
                    ),
                    outputDimension: headDimension
                )
                writeHead(
                    context,
                    destination: &attentionOutput,
                    chunkStart: chunkStart,
                    chunkLength: chunkLength,
                    headIndex: headIndex,
                    headCount: headCount,
                    headDimension: headDimension
                )
            }
            chunkStart += chunkLength
        }

        return QwenVisionMath.linear(
            input: attentionOutput,
            rowCount: rowCount,
            inputDimension: hiddenSize,
            weight: try weights.floatTensor(named: "\(prefix).proj.weight"),
            outputDimension: hiddenSize,
            bias: try weights.floatTensor(named: "\(prefix).proj.bias")
        )
    }

    private func visionMLP(hiddenStates: [Float], prefix: String) throws -> [Float] {
        let rowCount = hiddenStates.count / hiddenSize
        let fc1 = QwenVisionMath.linear(
            input: hiddenStates,
            rowCount: rowCount,
            inputDimension: hiddenSize,
            weight: try weights.floatTensor(named: "\(prefix).linear_fc1.weight"),
            outputDimension: intermediateSize,
            bias: try weights.floatTensor(named: "\(prefix).linear_fc1.bias")
        )
        let activated = QwenVisionMath.gelu(fc1, kind: hiddenAct)
        return QwenVisionMath.linear(
            input: activated,
            rowCount: rowCount,
            inputDimension: intermediateSize,
            weight: try weights.floatTensor(named: "\(prefix).linear_fc2.weight"),
            outputDimension: hiddenSize,
            bias: try weights.floatTensor(named: "\(prefix).linear_fc2.bias")
        )
    }

    private func patchMerge(
        hiddenStates: [Float],
        prefix: String,
        usePostshuffleNorm: Bool
    ) throws -> [Float] {
        let mergeUnit = spatialMergeSize * spatialMergeSize
        let tokenCount = hiddenStates.count / hiddenSize
        let mergedDimension = hiddenSize * mergeUnit
        precondition(tokenCount % mergeUnit == 0)

        let normalized: [Float]
        if usePostshuffleNorm {
            normalized = groupTokens(hiddenStates, mergeUnit: mergeUnit, hiddenDimension: hiddenSize)
            let normWeight = try weights.floatTensor(named: "\(prefix).norm.weight")
            let normBias = try weights.floatTensor(named: "\(prefix).norm.bias")
            let groupedCount = normalized.count / mergedDimension
            let postshuffle = QwenVisionMath.layerNorm(
                input: normalized,
                rowCount: groupedCount,
                dimension: mergedDimension,
                weight: normWeight,
                bias: normBias
            )
            return try patchMergeMLP(
                groupedInput: postshuffle,
                prefix: prefix,
                rowCount: groupedCount,
                inputDimension: mergedDimension
            )
        }

        let normWeight = try weights.floatTensor(named: "\(prefix).norm.weight")
        let normBias = try weights.floatTensor(named: "\(prefix).norm.bias")
        let tokenNormalized = QwenVisionMath.layerNorm(
            input: hiddenStates,
            rowCount: tokenCount,
            dimension: hiddenSize,
            weight: normWeight,
            bias: normBias
        )
        normalized = groupTokens(tokenNormalized, mergeUnit: mergeUnit, hiddenDimension: hiddenSize)
        return try patchMergeMLP(
            groupedInput: normalized,
            prefix: prefix,
            rowCount: tokenCount / mergeUnit,
            inputDimension: mergedDimension
        )
    }

    private func patchMergeMLP(
        groupedInput: [Float],
        prefix: String,
        rowCount: Int,
        inputDimension: Int
    ) throws -> [Float] {
        let fc1 = QwenVisionMath.linear(
            input: groupedInput,
            rowCount: rowCount,
            inputDimension: inputDimension,
            weight: try weights.floatTensor(named: "\(prefix).linear_fc1.weight"),
            outputDimension: inputDimension,
            bias: try weights.floatTensor(named: "\(prefix).linear_fc1.bias")
        )
        let activated = QwenVisionMath.gelu(fc1, kind: "gelu")
        return QwenVisionMath.linear(
            input: activated,
            rowCount: rowCount,
            inputDimension: inputDimension,
            weight: try weights.floatTensor(named: "\(prefix).linear_fc2.weight"),
            outputDimension: outHiddenSize,
            bias: try weights.floatTensor(named: "\(prefix).linear_fc2.bias")
        )
    }

    private func deepstackMergerIndex(for layerIndex: Int) throws -> Int {
        guard let index = configuration.deepstackVisualIndexes.firstIndex(of: layerIndex) else {
            throw ModelBundleLoaderError.invalidConfig(
                "Deepstack merger index missing for visual layer \(layerIndex)"
            )
        }
        return index
    }

    private func linspace(count: Int, upperBound: Int) -> [Float] {
        guard count > 1 else { return [0] }
        let step = Float(upperBound) / Float(count - 1)
        return (0..<count).map { Float($0) * step }
    }

    private func bilinearLookup(
        weight: [Float],
        gridSide: Int,
        row: Float,
        column: Float
    ) -> [Float] {
        let rowFloor = max(0, min(gridSide - 1, Int(floor(row))))
        let rowCeil = max(0, min(gridSide - 1, Int(ceil(row))))
        let columnFloor = max(0, min(gridSide - 1, Int(floor(column))))
        let columnCeil = max(0, min(gridSide - 1, Int(ceil(column))))
        let rowDelta = row - Float(rowFloor)
        let columnDelta = column - Float(columnFloor)

        let topLeft = embeddingRow(weight, gridSide: gridSide, row: rowFloor, column: columnFloor)
        let topRight = embeddingRow(weight, gridSide: gridSide, row: rowFloor, column: columnCeil)
        let bottomLeft = embeddingRow(weight, gridSide: gridSide, row: rowCeil, column: columnFloor)
        let bottomRight = embeddingRow(weight, gridSide: gridSide, row: rowCeil, column: columnCeil)

        var output = [Float](repeating: 0, count: hiddenSize)
        for index in 0..<hiddenSize {
            let top = topLeft[index] * (1 - columnDelta) + topRight[index] * columnDelta
            let bottom = bottomLeft[index] * (1 - columnDelta) + bottomRight[index] * columnDelta
            output[index] = top * (1 - rowDelta) + bottom * rowDelta
        }
        return output
    }

    private func embeddingRow(
        _ weight: [Float],
        gridSide: Int,
        row: Int,
        column: Int
    ) -> [Float] {
        let index = (row * gridSide + column) * hiddenSize
        return Array(weight[index..<(index + hiddenSize)])
    }

    private func applyRotary(
        query: inout [Float],
        key: inout [Float],
        rowCount: Int,
        headCount: Int,
        headDimension: Int,
        cos: [Float],
        sin: [Float]
    ) {
        for tokenIndex in 0..<rowCount {
            let positionBase = tokenIndex * headDimension
            for headIndex in 0..<headCount {
                let base = tokenIndex * headCount * headDimension + headIndex * headDimension
                let half = headDimension / 2
                for index in 0..<half {
                    let cosValue = cos[positionBase + index]
                    let sinValue = sin[positionBase + index]
                    let qFirst = query[base + index]
                    let qSecond = query[base + half + index]
                    query[base + index] = qFirst * cosValue - qSecond * sinValue
                    query[base + half + index] = qSecond * cosValue + qFirst * sinValue

                    let kFirst = key[base + index]
                    let kSecond = key[base + half + index]
                    key[base + index] = kFirst * cosValue - kSecond * sinValue
                    key[base + half + index] = kSecond * cosValue + kFirst * sinValue
                }
            }
        }
    }

    private func sliceHead(
        _ values: [Float],
        chunkStart: Int,
        chunkLength: Int,
        headIndex: Int,
        headCount: Int,
        headDimension: Int
    ) -> [Float] {
        var output = [Float](repeating: 0, count: chunkLength * headDimension)
        for tokenOffset in 0..<chunkLength {
            let sourceBase = (chunkStart + tokenOffset) * headCount * headDimension + headIndex * headDimension
            let destinationBase = tokenOffset * headDimension
            output.replaceSubrange(
                destinationBase..<(destinationBase + headDimension),
                with: values[sourceBase..<(sourceBase + headDimension)]
            )
        }
        return output
    }

    private func writeHead(
        _ values: [Float],
        destination: inout [Float],
        chunkStart: Int,
        chunkLength: Int,
        headIndex: Int,
        headCount: Int,
        headDimension: Int
    ) {
        for tokenOffset in 0..<chunkLength {
            let destinationBase = (chunkStart + tokenOffset) * headCount * headDimension + headIndex * headDimension
            let sourceBase = tokenOffset * headDimension
            destination.replaceSubrange(
                destinationBase..<(destinationBase + headDimension),
                with: values[sourceBase..<(sourceBase + headDimension)]
            )
        }
    }

    private func transposeRows(
        _ values: [Float],
        rowCount: Int,
        columnCount: Int
    ) -> [Float] {
        var transposed = [Float](repeating: 0, count: values.count)
        for row in 0..<rowCount {
            for column in 0..<columnCount {
                transposed[column * rowCount + row] = values[row * columnCount + column]
            }
        }
        return transposed
    }

    private func groupTokens(
        _ values: [Float],
        mergeUnit: Int,
        hiddenDimension: Int
    ) -> [Float] {
        let tokenCount = values.count / hiddenDimension
        let groupedCount = tokenCount / mergeUnit
        let groupedDimension = hiddenDimension * mergeUnit
        var grouped = [Float](repeating: 0, count: groupedCount * groupedDimension)
        for groupIndex in 0..<groupedCount {
            let sourceBase = groupIndex * groupedDimension
            grouped.replaceSubrange(
                sourceBase..<(sourceBase + groupedDimension),
                with: values[sourceBase..<(sourceBase + groupedDimension)]
            )
        }
        return grouped
    }

    private func splitRows(
        _ values: [Float],
        rowDimension: Int,
        counts: [Int]
    ) -> [[Float]] {
        var rows: [[Float]] = []
        rows.reserveCapacity(counts.reduce(0, +))
        var cursor = 0
        for count in counts {
            for _ in 0..<count {
                rows.append(Array(values[cursor..<(cursor + rowDimension)]))
                cursor += rowDimension
            }
        }
        return rows
    }
}

private struct VisionSample {
    let gridTHW: [Int]
    let placeholderTokenCount: Int
    let pixelValuesShape: [Int]
    let pixelValues: [Float]

    init(_ image: PreparedPrompt.Multimodal.Image) {
        self.gridTHW = image.gridTHW
        self.placeholderTokenCount = image.placeholderTokenCount
        self.pixelValuesShape = image.pixelValuesShape
        self.pixelValues = image.pixelValues
    }

    init(_ video: PreparedPrompt.Multimodal.Video) {
        self.gridTHW = video.gridTHW
        self.placeholderTokenCount = video.placeholderTokenCount
        self.pixelValuesShape = video.pixelValuesShape
        self.pixelValues = video.pixelValues
    }
}
