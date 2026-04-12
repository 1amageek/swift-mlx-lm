import Foundation
import Darwin
import MetalCompiler
import Tokenizers

struct SentenceTransformerTextEmbeddingRuntime: Sendable {
    struct PreparedInput: Sendable {
        let renderedText: String
        let tokenIDs: [Int]
        let promptTokenCount: Int
    }

    struct DenseLayer: Sendable {
        let weights: [Float]
        let outputDimension: Int
        let inputDimension: Int
        let bias: [Float]?
        let activation: SentenceTransformerMetadata.DenseActivation
    }

    let availablePromptNames: [String]
    let defaultPromptName: String?

    private let prompts: [String: String]
    private let pooling: SentenceTransformerMetadata.Pooling
    private let denseLayers: [DenseLayer]
    private let postprocessors: [SentenceEmbeddingPostprocessor]

    init(resources: ModelBundleResources, weights: STAFWeightStore) throws {
        let metadata = try SentenceTransformerMetadata.load(from: resources)
        let weightStore = CPUWeightStore(weights: weights)
        try self.init(metadata: metadata, weightStore: weightStore)
    }

    init(
        metadata: SentenceTransformerMetadata,
        weightStore: CPUWeightStore
    ) throws {
        self.prompts = metadata.prompts
        self.availablePromptNames = metadata.availablePromptNames
        self.defaultPromptName = metadata.defaultPromptName
        self.pooling = metadata.pooling
        self.postprocessors = metadata.postprocessors
        self.denseLayers = try metadata.denseLayers.map { module in
            let weights = try weightStore.floatTensor(named: module.weightName)
            let shape = try weightStore.shape(named: module.weightName)
            guard shape.count == 2 else {
                throw ModelBundleLoaderError.invalidConfig(
                    "Dense tensor must be rank-2: \(module.weightName)"
                )
            }
            let outputDimension = shape[0]
            let inputDimension = shape[1]
            if let declaredInput = module.inputDimension,
               declaredInput != inputDimension {
                throw ModelBundleLoaderError.invalidConfig(
                    "Dense tensor input size mismatch: \(module.weightName)"
                )
            }
            if let declaredOutput = module.outputDimension,
               declaredOutput != outputDimension {
                throw ModelBundleLoaderError.invalidConfig(
                    "Dense tensor output size mismatch: \(module.weightName)"
                )
            }

            let bias = try weightStore.optionalFloatTensor(named: module.biasName)
            if let bias, bias.count != outputDimension {
                throw ModelBundleLoaderError.invalidConfig(
                    "Dense bias size mismatch: \(module.biasName)"
                )
            }

            return DenseLayer(
                weights: weights,
                outputDimension: outputDimension,
                inputDimension: inputDimension,
                bias: bias,
                activation: module.activation
            )
        }
    }

    func prepare(
        text: String,
        promptName: String?,
        tokenizer: any Tokenizer
    ) throws -> PreparedInput {
        let resolvedPromptName = promptName ?? defaultPromptName
        let prefix: String
        if let resolvedPromptName {
            guard let promptPrefix = prompts[resolvedPromptName] else {
                throw ModelBundleLoaderError.invalidConfig(
                    "Unknown embedding prompt name: \(resolvedPromptName)"
                )
            }
            prefix = promptPrefix
        } else {
            prefix = ""
        }

        let renderedText = prefix + text
        let tokenIDs = tokenizer.encode(text: renderedText, addSpecialTokens: true)
        let promptTokenCount: Int
        if prefix.isEmpty || pooling.includePrompt {
            promptTokenCount = 0
        } else {
            promptTokenCount = tokenizer.encode(text: prefix, addSpecialTokens: true).count
        }
        return PreparedInput(
            renderedText: renderedText,
            tokenIDs: tokenIDs,
            promptTokenCount: promptTokenCount
        )
    }

    func embed(
        hiddenStates: [[Float]],
        promptTokenCount: Int
    ) throws -> [Float] {
        var embedding = try pool(hiddenStates: hiddenStates, promptTokenCount: promptTokenCount)
        for layer in denseLayers {
            embedding = try applyDense(layer, to: embedding)
        }
        return applyPostprocessors(to: embedding)
    }

    private func pool(
        hiddenStates: [[Float]],
        promptTokenCount: Int
    ) throws -> [Float] {
        guard hiddenStates.isEmpty == false else { return [] }
        let startIndex = pooling.includePrompt ? 0 : promptTokenCount
        guard startIndex < hiddenStates.count else {
            throw ModelBundleLoaderError.invalidConfig(
                "Prompt tokens consumed the full embedding sequence"
            )
        }

        let selectedStates = Array(hiddenStates[startIndex...])
        guard let first = selectedStates.first else { return [] }
        let dimension = first.count
        for state in selectedStates where state.count != dimension {
            throw ModelBundleLoaderError.invalidConfig("Embedding hidden state width mismatch")
        }

        switch pooling.strategy {
        case .mean:
            var sums = [Float](repeating: 0, count: dimension)
            for state in selectedStates {
                for index in 0..<dimension {
                    sums[index] += state[index]
                }
            }
            let count = Float(selectedStates.count)
            return sums.map { $0 / count }
        case .cls:
            return first
        case .max:
            var values = first
            for state in selectedStates.dropFirst() {
                for index in 0..<dimension {
                    values[index] = max(values[index], state[index])
                }
            }
            return values
        case .lastToken:
            guard let last = selectedStates.last else { return [] }
            return last
        }
    }

    private func applyDense(_ layer: DenseLayer, to input: [Float]) throws -> [Float] {
        guard input.count == layer.inputDimension else {
            throw ModelBundleLoaderError.invalidConfig(
                "Dense input dimension mismatch: expected \(layer.inputDimension), got \(input.count)"
            )
        }

        var output = [Float](repeating: 0, count: layer.outputDimension)
        for outputIndex in 0..<layer.outputDimension {
            var value = layer.bias?[outputIndex] ?? 0
            let rowOffset = outputIndex * layer.inputDimension
            for inputIndex in 0..<layer.inputDimension {
                value += layer.weights[rowOffset + inputIndex] * input[inputIndex]
            }
            output[outputIndex] = apply(layer.activation, to: value)
        }
        return output
    }

    private func apply(
        _ activation: SentenceTransformerMetadata.DenseActivation,
        to value: Float
    ) -> Float {
        switch activation {
        case .identity:
            return value
        case .tanh:
            return Float(Darwin.tanh(Double(value)))
        case .relu:
            return max(0, value)
        case .gelu:
            let coefficient: Float = 0.79788456
            let cubic: Float = 0.044715
            let inner = coefficient * (value + cubic * value * value * value)
            return 0.5 * value * (1 + Float(Darwin.tanh(Double(inner))))
        }
    }

    private func l2Normalize(_ values: [Float]) -> [Float] {
        let squaredNorm = values.reduce(into: Float(0)) { partial, value in
            partial += value * value
        }
        guard squaredNorm > 0 else { return values }
        let scale = 1 / squaredNorm.squareRoot()
        return values.map { $0 * scale }
    }

    private func applyPostprocessors(to values: [Float]) -> [Float] {
        postprocessors.reduce(values) { partial, postprocessor in
            switch postprocessor {
            case .l2Normalize:
                return l2Normalize(partial)
            }
        }
    }
}
