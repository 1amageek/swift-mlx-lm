import Darwin
import LMArchitecture
import ModelDeclarations

final class Gemma4TextRuntime {
    private let config: ModelConfig
    private let weights: Gemma4WeightStore

    init(config: ModelConfig, weights: Gemma4WeightStore) throws {
        try Gemma4.validate(config)
        self.config = config
        self.weights = weights
    }

    func tokenEmbeddings(tokenIDs: [Int]) throws -> [[Float]] {
        let hiddenSize = config.hiddenSize
        let table = try weights.floatTensor(named: "model.language_model.embed_tokens.weight")
        return try gatherRows(
            table: table,
            rowCount: config.vocabSize,
            rowWidth: hiddenSize,
            indices: tokenIDs
        )
    }

    func buildPrefillPerLayerInputs(
        tokenIDs: [Int],
        promptEmbeddings: [[Float]]
    ) throws -> [[[Float]]] {
        let layerCount = config.layerCount
        let perLayerSize = try requirePerLayerInputSize()
        let hiddenSize = config.hiddenSize
        guard promptEmbeddings.count == tokenIDs.count else {
            throw ModelBundleLoaderError.invalidConfig("Gemma4 prompt embedding count mismatch")
        }

        let perLayerEmbeddingTable = try weights.floatTensor(
            named: "model.language_model.embed_tokens_per_layer.weight"
        )
        let gatheredPerLayer = try gatherRows(
            table: perLayerEmbeddingTable,
            rowCount: config.vocabSizePerLayerInput ?? config.vocabSize,
            rowWidth: layerCount * perLayerSize,
            indices: tokenIDs
        )

        let flattenedPromptEmbeddings = promptEmbeddings.flatMap { $0 }
        let perLayerProjectionWeight = try weights.floatTensor(
            named: "model.language_model.per_layer_model_projection.weight"
        )
        var projected = QwenVisionMath.linear(
            input: flattenedPromptEmbeddings,
            rowCount: promptEmbeddings.count,
            inputDimension: hiddenSize,
            weight: perLayerProjectionWeight,
            outputDimension: layerCount * perLayerSize
        )
        let projectionScale = 1.0 / Float(hiddenSize).squareRoot()
        projected = projected.map { $0 * projectionScale }

        let projectionNormWeight = try weights.floatTensor(
            named: "model.language_model.per_layer_projection_norm.weight"
        )
        var byLayer = Array(
            repeating: Array(repeating: [Float](repeating: 0, count: perLayerSize), count: tokenIDs.count),
            count: layerCount
        )
        let inputScale = powf(2.0, -0.5)

        for tokenIndex in tokenIDs.indices {
            let embedRow = gatheredPerLayer[tokenIndex]
            let projectedRow = Array(
                projected[(tokenIndex * layerCount * perLayerSize)..<((tokenIndex + 1) * layerCount * perLayerSize)]
            )
            for layerIndex in 0..<layerCount {
                let start = layerIndex * perLayerSize
                let end = start + perLayerSize
                let projectedLayer = Array(projectedRow[start..<end])
                let normalizedProjection = rmsNorm(
                    projectedLayer,
                    weight: projectionNormWeight,
                    epsilon: config.normEps
                )
                let embedLayer = Array(embedRow[start..<end])
                var combined = [Float](repeating: 0, count: perLayerSize)
                for index in 0..<perLayerSize {
                    combined[index] = (embedLayer[index] + normalizedProjection[index]) * inputScale
                }
                byLayer[layerIndex][tokenIndex] = combined
            }
        }

        return byLayer
    }

    func buildDecodePerLayerInputs(tokenID: Int) throws -> [[Float]] {
        let embeddings = try tokenEmbeddings(tokenIDs: [tokenID])
        let perLayer = try buildPrefillPerLayerInputs(tokenIDs: [tokenID], promptEmbeddings: embeddings)
        return perLayer.map { layer in
            layer[0]
        }
    }

    private func requirePerLayerInputSize() throws -> Int {
        guard let value = config.hiddenSizePerLayerInput else {
            throw ModelBundleLoaderError.invalidConfig("Gemma4 hidden_size_per_layer_input is required")
        }
        return value
    }

    private func gatherRows(
        table: [Float],
        rowCount: Int,
        rowWidth: Int,
        indices: [Int]
    ) throws -> [[Float]] {
        guard table.count == rowCount * rowWidth else {
            throw ModelBundleLoaderError.invalidConfig("Gemma4 tensor shape does not match expected row-major layout")
        }
        return try indices.map { index in
            guard index >= 0, index < rowCount else {
                throw ModelBundleLoaderError.invalidConfig("Gemma4 token ID \(index) is out of range")
            }
            let start = index * rowWidth
            return Array(table[start..<(start + rowWidth)])
        }
    }

    private func rmsNorm(
        _ input: [Float],
        weight: [Float],
        epsilon: Float
    ) -> [Float] {
        precondition(input.count == weight.count)
        var meanSquare: Float = 0
        for value in input {
            meanSquare += value * value
        }
        meanSquare /= Float(input.count)
        let scale = 1 / sqrtf(meanSquare + epsilon)
        var output = [Float](repeating: 0, count: input.count)
        for index in input.indices {
            output[index] = input[index] * scale * weight[index]
        }
        return output
    }
}
