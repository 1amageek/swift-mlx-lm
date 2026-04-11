import Foundation

final class Gemma4VisionEncoder {
    private let configuration: ModelVisionConfiguration
    private let weights: Gemma4WeightStore
    private let textHiddenSize: Int
    private let hiddenSize: Int
    private let intermediateSize: Int
    private let outputDimension: Int
    private let headCount: Int
    private let layerCount: Int
    private let patchSize: Int
    private let poolingKernelSize: Int
    private let positionEmbeddingSize: Int
    private let hiddenAct: String

    init(
        configuration: ModelVisionConfiguration,
        textHiddenSize: Int,
        weights: Gemma4WeightStore
    ) throws {
        guard let hiddenSize = configuration.hiddenSize else {
            throw ModelBundleLoaderError.invalidConfig("Gemma4 vision hidden_size is required")
        }
        guard let intermediateSize = configuration.intermediateSize else {
            throw ModelBundleLoaderError.invalidConfig("Gemma4 vision intermediate_size is required")
        }
        guard let outputDimension = configuration.outHiddenSize else {
            throw ModelBundleLoaderError.invalidConfig("Gemma4 vision output_proj_dims is required")
        }
        guard let headCount = configuration.headCount else {
            throw ModelBundleLoaderError.invalidConfig("Gemma4 vision num_attention_heads is required")
        }
        guard let layerCount = configuration.depth else {
            throw ModelBundleLoaderError.invalidConfig("Gemma4 vision num_hidden_layers is required")
        }
        guard let patchSize = configuration.patchSize else {
            throw ModelBundleLoaderError.invalidConfig("Gemma4 vision patch_size is required")
        }
        guard let poolingKernelSize = configuration.poolingKernelSize else {
            throw ModelBundleLoaderError.invalidConfig("Gemma4 vision pooling_kernel_size is required")
        }
        guard let positionEmbeddingSize = configuration.positionEmbeddingSize else {
            throw ModelBundleLoaderError.invalidConfig("Gemma4 vision position_embedding_size is required")
        }

        self.configuration = configuration
        self.weights = weights
        self.textHiddenSize = textHiddenSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.outputDimension = outputDimension
        self.headCount = headCount
        self.layerCount = layerCount
        self.patchSize = patchSize
        self.poolingKernelSize = poolingKernelSize
        self.positionEmbeddingSize = positionEmbeddingSize
        self.hiddenAct = configuration.hiddenAct ?? "gelu_pytorch_tanh"
    }

    func encode(images: [PreparedPrompt.Multimodal.Image]) throws -> [[Float]] {
        var allEmbeddings: [[Float]] = []
        for image in images {
            allEmbeddings.append(contentsOf: try encode(image: image))
        }
        return allEmbeddings
    }

    private func encode(image: PreparedPrompt.Multimodal.Image) throws -> [[Float]] {
        guard image.gridTHW.count == 3, image.gridTHW[0] == 1 else {
            throw LanguageModelContextError.multimodalInputNotSupported(
                "Gemma4 image execution expects a single-frame image grid."
            )
        }
        let patchCount = image.gridTHW[1] * image.gridTHW[2]
        guard image.pixelValuesShape.count == 2, image.pixelValuesShape[0] == patchCount else {
            throw LanguageModelContextError.multimodalInputNotSupported(
                "Gemma4 image patch metadata does not match the prepared pixel values."
            )
        }

        var hiddenStates = try patchEmbed(image: image)
        let positionEmbeddings = try makePositionEmbeddings(gridTHW: image.gridTHW)
        hiddenStates = QwenVisionMath.add(hiddenStates, positionEmbeddings)

        for layerIndex in 0..<layerCount {
            hiddenStates = try forwardVisionLayer(
                hiddenStates: hiddenStates,
                gridTHW: image.gridTHW,
                layerIndex: layerIndex
            )
        }

        hiddenStates = try pool(hiddenStates: hiddenStates, gridTHW: image.gridTHW)
        if configuration.standardize == true {
            hiddenStates = try standardize(hiddenStates)
        }
        return try projectToLanguageSpace(hiddenStates)
    }

    private func patchEmbed(image: PreparedPrompt.Multimodal.Image) throws -> [Float] {
        let input = image.pixelValues.map { 2 * ($0 - 0.5) }
        return QwenVisionMath.linear(
            input: input,
            rowCount: image.pixelValuesShape[0],
            inputDimension: image.pixelValuesShape[1],
            weight: try weights.floatTensor(named: "model.vision_tower.patch_embedder.input_proj.weight"),
            outputDimension: hiddenSize
        )
    }

    private func makePositionEmbeddings(gridTHW: [Int]) throws -> [Float] {
        let gridH = gridTHW[1]
        let gridW = gridTHW[2]
        let table = try weights.floatTensor(
            named: "model.vision_tower.patch_embedder.position_embedding_table"
        )
        let expectedCount = 2 * positionEmbeddingSize * hiddenSize
        guard table.count == expectedCount else {
            throw ModelBundleLoaderError.invalidConfig(
                "Gemma4 vision position_embedding_table shape mismatch"
            )
        }
        let xTableBase = 0
        let yTableBase = positionEmbeddingSize * hiddenSize
        var output = [Float](repeating: 0, count: gridH * gridW * hiddenSize)
        for row in 0..<gridH {
            for column in 0..<gridW {
                let targetBase = (row * gridW + column) * hiddenSize
                let xBase = xTableBase + column * hiddenSize
                let yBase = yTableBase + row * hiddenSize
                for index in 0..<hiddenSize {
                    output[targetBase + index] = table[xBase + index] + table[yBase + index]
                }
            }
        }
        return output
    }

    private func forwardVisionLayer(
        hiddenStates: [Float],
        gridTHW: [Int],
        layerIndex: Int
    ) throws -> [Float] {
        let rowCount = hiddenStates.count / hiddenSize
        let prefix = "model.vision_tower.encoder.layers.\(layerIndex)"

        let norm1 = try rmsNormRows(
            hiddenStates,
            rowCount: rowCount,
            dimension: hiddenSize,
            weight: weights.optionalFloatTensor(named: "\(prefix).input_layernorm.weight")
        )
        let attention = try attention(
            hiddenStates: norm1,
            gridTHW: gridTHW,
            prefix: "\(prefix).self_attn"
        )
        let afterAttention = QwenVisionMath.add(hiddenStates, try rmsNormRows(
            attention,
            rowCount: rowCount,
            dimension: hiddenSize,
            weight: weights.optionalFloatTensor(named: "\(prefix).post_attention_layernorm.weight")
        ))

        let norm2 = try rmsNormRows(
            afterAttention,
            rowCount: rowCount,
            dimension: hiddenSize,
            weight: weights.optionalFloatTensor(named: "\(prefix).pre_feedforward_layernorm.weight")
        )
        let mlp = try mlp(hiddenStates: norm2, prefix: "\(prefix).mlp")
        let postFF = try rmsNormRows(
            mlp,
            rowCount: rowCount,
            dimension: hiddenSize,
            weight: weights.optionalFloatTensor(named: "\(prefix).post_feedforward_layernorm.weight")
        )
        return QwenVisionMath.add(afterAttention, postFF)
    }

    private func attention(
        hiddenStates: [Float],
        gridTHW: [Int],
        prefix: String
    ) throws -> [Float] {
        let tokenCount = hiddenStates.count / hiddenSize
        let headDimension = hiddenSize / headCount
        let query = try projectHeads(hiddenStates: hiddenStates, prefix: "\(prefix).q_proj.linear.weight")
        let key = try projectHeads(hiddenStates: hiddenStates, prefix: "\(prefix).k_proj.linear.weight")
        let value = try projectHeads(hiddenStates: hiddenStates, prefix: "\(prefix).v_proj.linear.weight")
        let qNormWeight = try weights.optionalFloatTensor(named: "\(prefix).q_norm.weight")
        let kNormWeight = try weights.optionalFloatTensor(named: "\(prefix).k_norm.weight")

        var queryHeads = reshapeToHeads(query, tokenCount: tokenCount, headDimension: headDimension)
        var keyHeads = reshapeToHeads(key, tokenCount: tokenCount, headDimension: headDimension)
        var valueHeads = reshapeToHeads(value, tokenCount: tokenCount, headDimension: headDimension)
        let coordinates = patchCoordinates(gridTHW: gridTHW)

        for tokenIndex in 0..<tokenCount {
            for headIndex in 0..<headCount {
                queryHeads[tokenIndex][headIndex] = rmsNormVector(
                    queryHeads[tokenIndex][headIndex],
                    weight: qNormWeight
                )
                keyHeads[tokenIndex][headIndex] = rmsNormVector(
                    keyHeads[tokenIndex][headIndex],
                    weight: kNormWeight
                )
                valueHeads[tokenIndex][headIndex] = rmsNormVector(valueHeads[tokenIndex][headIndex], weight: nil)
                queryHeads[tokenIndex][headIndex] = apply2DRotary(
                    queryHeads[tokenIndex][headIndex],
                    coordinate: coordinates[tokenIndex]
                )
                keyHeads[tokenIndex][headIndex] = apply2DRotary(
                    keyHeads[tokenIndex][headIndex],
                    coordinate: coordinates[tokenIndex]
                )
            }
        }

        var output = [Float](repeating: 0, count: tokenCount * hiddenSize)
        let scale = 1.0 / sqrtf(Float(headDimension))
        for headIndex in 0..<headCount {
            for queryIndex in 0..<tokenCount {
                var scores = [Float](repeating: 0, count: tokenCount)
                for keyIndex in 0..<tokenCount {
                    scores[keyIndex] = dot(
                        queryHeads[queryIndex][headIndex],
                        keyHeads[keyIndex][headIndex]
                    ) * scale
                }
                let weights = QwenVisionMath.softmaxRows(
                    scores,
                    rowCount: 1,
                    columnCount: tokenCount
                )
                var combined = [Float](repeating: 0, count: headDimension)
                for keyIndex in 0..<tokenCount {
                    let score = weights[keyIndex]
                    for dim in 0..<headDimension {
                        combined[dim] += valueHeads[keyIndex][headIndex][dim] * score
                    }
                }
                let targetBase = queryIndex * hiddenSize + headIndex * headDimension
                for dim in 0..<headDimension {
                    output[targetBase + dim] = combined[dim]
                }
            }
        }

        return QwenVisionMath.linear(
            input: output,
            rowCount: tokenCount,
            inputDimension: hiddenSize,
            weight: try weights.floatTensor(named: "\(prefix).o_proj.linear.weight"),
            outputDimension: hiddenSize
        )
    }

    private func mlp(hiddenStates: [Float], prefix: String) throws -> [Float] {
        let rowCount = hiddenStates.count / hiddenSize
        let gate = QwenVisionMath.linear(
            input: hiddenStates,
            rowCount: rowCount,
            inputDimension: hiddenSize,
            weight: try weights.floatTensor(named: "\(prefix).gate_proj.linear.weight"),
            outputDimension: intermediateSize
        )
        let up = QwenVisionMath.linear(
            input: hiddenStates,
            rowCount: rowCount,
            inputDimension: hiddenSize,
            weight: try weights.floatTensor(named: "\(prefix).up_proj.linear.weight"),
            outputDimension: intermediateSize
        )
        let activated = QwenVisionMath.gelu(gate, kind: hiddenAct)
        var fused = [Float](repeating: 0, count: activated.count)
        for index in activated.indices {
            fused[index] = activated[index] * up[index]
        }
        return QwenVisionMath.linear(
            input: fused,
            rowCount: rowCount,
            inputDimension: intermediateSize,
            weight: try weights.floatTensor(named: "\(prefix).down_proj.linear.weight"),
            outputDimension: hiddenSize
        )
    }

    private func pool(hiddenStates: [Float], gridTHW: [Int]) throws -> [Float] {
        let gridH = gridTHW[1]
        let gridW = gridTHW[2]
        let pooledH = gridH / poolingKernelSize
        let pooledW = gridW / poolingKernelSize
        var output = [Float](repeating: 0, count: pooledH * pooledW * hiddenSize)
        let scale = Float(hiddenSize).squareRoot()
        let kernelArea = Float(poolingKernelSize * poolingKernelSize)
        for pooledRow in 0..<pooledH {
            for pooledColumn in 0..<pooledW {
                let targetBase = (pooledRow * pooledW + pooledColumn) * hiddenSize
                for rowOffset in 0..<poolingKernelSize {
                    for columnOffset in 0..<poolingKernelSize {
                        let row = pooledRow * poolingKernelSize + rowOffset
                        let column = pooledColumn * poolingKernelSize + columnOffset
                        let sourceBase = (row * gridW + column) * hiddenSize
                        for index in 0..<hiddenSize {
                            output[targetBase + index] += hiddenStates[sourceBase + index] / kernelArea
                        }
                    }
                }
                for index in 0..<hiddenSize {
                    output[targetBase + index] *= scale
                }
            }
        }
        return output
    }

    private func standardize(_ hiddenStates: [Float]) throws -> [Float] {
        guard let bias = try weights.optionalFloatTensor(named: "model.vision_tower.std_bias"),
              let scale = try weights.optionalFloatTensor(named: "model.vision_tower.std_scale") else {
            return hiddenStates
        }
        var output = hiddenStates
        let rowCount = hiddenStates.count / hiddenSize
        for row in 0..<rowCount {
            let base = row * hiddenSize
            for index in 0..<hiddenSize {
                output[base + index] = (hiddenStates[base + index] - bias[index]) * scale[index]
            }
        }
        return output
    }

    private func projectToLanguageSpace(_ hiddenStates: [Float]) throws -> [[Float]] {
        let rowCount = hiddenStates.count / hiddenSize
        var normalized = [Float](repeating: 0, count: hiddenStates.count)
        for row in 0..<rowCount {
            let start = row * hiddenSize
            let end = start + hiddenSize
            let normed = rmsNormVector(Array(hiddenStates[start..<end]), weight: nil)
            normalized.replaceSubrange(start..<end, with: normed)
        }
        let projected = QwenVisionMath.linear(
            input: normalized,
            rowCount: rowCount,
            inputDimension: hiddenSize,
            weight: try weights.floatTensor(named: "model.embed_vision.embedding_projection.weight"),
            outputDimension: textHiddenSize
        )
        return splitRows(projected, rowCount: rowCount, rowWidth: textHiddenSize)
    }

    private func projectHeads(hiddenStates: [Float], prefix: String) throws -> [Float] {
        let tokenCount = hiddenStates.count / hiddenSize
        return QwenVisionMath.linear(
            input: hiddenStates,
            rowCount: tokenCount,
            inputDimension: hiddenSize,
            weight: try weights.floatTensor(named: prefix),
            outputDimension: hiddenSize
        )
    }

    private func patchCoordinates(gridTHW: [Int]) -> [(x: Int, y: Int)] {
        let gridH = gridTHW[1]
        let gridW = gridTHW[2]
        var result: [(x: Int, y: Int)] = []
        result.reserveCapacity(gridH * gridW)
        for row in 0..<gridH {
            for column in 0..<gridW {
                result.append((x: column, y: row))
            }
        }
        return result
    }

    private func reshapeToHeads(
        _ flattened: [Float],
        tokenCount: Int,
        headDimension: Int
    ) -> [[[Float]]] {
        var result = Array(
            repeating: Array(
                repeating: [Float](repeating: 0, count: headDimension),
                count: headCount
            ),
            count: tokenCount
        )
        for tokenIndex in 0..<tokenCount {
            for headIndex in 0..<headCount {
                let base = tokenIndex * hiddenSize + headIndex * headDimension
                result[tokenIndex][headIndex] = Array(flattened[base..<(base + headDimension)])
            }
        }
        return result
    }

    private func splitRows(_ flattened: [Float], rowCount: Int, rowWidth: Int) -> [[Float]] {
        (0..<rowCount).map { index in
            let start = index * rowWidth
            return Array(flattened[start..<(start + rowWidth)])
        }
    }

    private func apply2DRotary(_ vector: [Float], coordinate: (x: Int, y: Int)) -> [Float] {
        let headDimension = vector.count
        let halfDimension = headDimension / 2
        let quarterDimension = max(1, halfDimension / 2)
        let frequencies = (0..<quarterDimension).map { index -> Float in
            let exponent = Float(index * 2) / Float(max(halfDimension, 1))
            return 1.0 / powf(10_000.0, exponent)
        }

        var positionEmbedding = [Float]()
        positionEmbedding.reserveCapacity(headDimension)
        for frequency in frequencies {
            positionEmbedding.append(Float(coordinate.y) * frequency)
        }
        for frequency in frequencies {
            positionEmbedding.append(Float(coordinate.x) * frequency)
        }
        positionEmbedding += positionEmbedding

        var output = vector
        let pairCount = headDimension / 2
        for pairIndex in 0..<pairCount {
            let angle = positionEmbedding[pairIndex]
            let cosValue = cosf(angle)
            let sinValue = sinf(angle)
            let evenIndex = pairIndex * 2
            let oddIndex = evenIndex + 1
            if oddIndex >= output.count {
                break
            }
            let even = vector[evenIndex]
            let odd = vector[oddIndex]
            output[evenIndex] = even * cosValue - odd * sinValue
            output[oddIndex] = odd * cosValue + even * sinValue
        }
        return output
    }

    private func rmsNormRows(
        _ input: [Float],
        rowCount: Int,
        dimension: Int,
        weight: [Float]?
    ) throws -> [Float] {
        guard input.count == rowCount * dimension else {
            throw ModelBundleLoaderError.invalidConfig("Gemma4 vision RMSNorm shape mismatch")
        }
        var output = [Float](repeating: 0, count: input.count)
        for row in 0..<rowCount {
            let base = row * dimension
            let normalized = rmsNormVector(
                Array(input[base..<(base + dimension)]),
                weight: weight
            )
            output.replaceSubrange(base..<(base + dimension), with: normalized)
        }
        return output
    }

    private func rmsNormVector(_ input: [Float], weight: [Float]?) -> [Float] {
        var meanSquare: Float = 0
        for value in input {
            meanSquare += value * value
        }
        meanSquare /= Float(max(input.count, 1))
        let scale = 1 / sqrtf(meanSquare + 1e-6)
        var output = [Float](repeating: 0, count: input.count)
        if let weight {
            for index in input.indices {
                output[index] = input[index] * scale * weight[index]
            }
        } else {
            for index in input.indices {
                output[index] = input[index] * scale
            }
        }
        return output
    }

    private func dot(_ lhs: [Float], _ rhs: [Float]) -> Float {
        var value: Float = 0
        for index in lhs.indices {
            value += lhs[index] * rhs[index]
        }
        return value
    }
}
