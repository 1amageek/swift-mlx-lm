import Foundation
import Testing
@testable import SwiftLM

@Suite("Gemma4 Runtime", .serialized)
struct Gemma4RuntimeTests {
    @Test("Gemma4 token embeddings apply model embedding scale", .timeLimit(.minutes(2)))
    func tokenEmbeddingsApplyModelScale() throws {
        let config = Gemma4TestSupport.syntheticConfig()
        let weights = Gemma4WeightStore(
            denseTensors: Gemma4TestSupport.syntheticDenseTensors(hiddenSize: config.hiddenSize)
        )
        let runtime = try Gemma4TextRuntime(config: config, weights: weights)

        let embeddings = try runtime.tokenEmbeddings(tokenIDs: [0])
        let embeddingRow = try #require(embeddings.first)
        let rawTable = try weights.floatTensor(named: "model.language_model.embed_tokens.weight")
        let expectedScale = Float(config.hiddenSize).squareRoot()
        let expected = Array(rawTable.prefix(config.hiddenSize)).map { $0 * expectedScale }

        #expect(embeddingRow.count == config.hiddenSize)
        for index in 0..<config.hiddenSize {
            #expect(abs(embeddingRow[index] - expected[index]) < 1e-6)
        }
    }

    @Test("Gemma4 per-layer projection norm uses checkpoint scales directly", .timeLimit(.minutes(2)))
    func perLayerProjectionNormUsesCheckpointScalesDirectly() throws {
        let config = Gemma4TestSupport.syntheticConfig()
        let weights = Gemma4WeightStore(
            denseTensors: Gemma4TestSupport.syntheticDenseTensors(hiddenSize: config.hiddenSize)
        )
        let runtime = try Gemma4TextRuntime(config: config, weights: weights)

        let tokenIDs = [0]
        let promptEmbeddings = try runtime.tokenEmbeddings(tokenIDs: tokenIDs)
        let actual = try runtime.buildPrefillPerLayerInputs(
            tokenIDs: tokenIDs,
            promptEmbeddings: promptEmbeddings
        )

        let perLayerSize = try #require(config.hiddenSizePerLayerInput)
        let layerCount = config.layerCount
        let hiddenSize = config.hiddenSize
        let promptEmbedding = try #require(promptEmbeddings.first)
        let perLayerEmbeddingScale = Float(perLayerSize).squareRoot()
        let perLayerEmbedding = try #require(
            try weights.gatherRows(
                named: "model.language_model.embed_tokens_per_layer.weight",
                rowCount: config.vocabSizePerLayerInput ?? config.vocabSize,
                rowWidth: layerCount * perLayerSize,
                indices: tokenIDs
            ).first
        ).map { $0 * perLayerEmbeddingScale }
        let projectionWeight = try weights.floatTensor(
            named: "model.language_model.per_layer_model_projection.weight"
        )
        let projected = QwenVisionMath.linear(
            input: promptEmbedding,
            rowCount: 1,
            inputDimension: hiddenSize,
            weight: projectionWeight,
            outputDimension: layerCount * perLayerSize
        ).map { $0 * (1.0 / Float(hiddenSize).squareRoot()) }
        let projectionNormWeight = try weights.floatTensor(
            named: "model.language_model.per_layer_projection_norm.weight"
        )
        let projectedLayer = Array(projected[..<perLayerSize])
        let normalizedProjection = rmsNorm(
            projectedLayer,
            weight: projectionNormWeight,
            epsilon: config.normEps
        )
        let embeddedLayer = Array(perLayerEmbedding[..<perLayerSize])
        let expected = zip(embeddedLayer, normalizedProjection).map { pair in
            (pair.0 + pair.1) * powf(2, -0.5)
        }

        let actualFirstLayer = actual[0][0]
        #expect(actualFirstLayer.count == expected.count)
        for index in actualFirstLayer.indices {
            #expect(abs(actualFirstLayer[index] - expected[index]) < 1e-5)
        }
    }

    @Test("Text-only prepared prompt becomes executable Gemma4 prompt", .timeLimit(.minutes(2)))
    func textOnlyPreparedPromptBecomesExecutable() async throws {
        guard let container = try await Gemma4TestSupport.syntheticGemma4Container() else {
            print("[Skip] No Metal device available for Gemma4 runtime tests")
            return
        }
        let prepared = try await container.prepare( ModelInput(prompt: "hello gemma4"))
        let executable = try container.makeExecutablePrompt(from: prepared)

        #expect(executable.visualContext == nil)
        #expect(executable.tokenIDs == prepared.tokenIDs)
    }

    @Test("Image prepared prompt becomes executable Gemma4 prompt", .timeLimit(.minutes(2)))
    func imagePreparedPromptBecomesExecutable() async throws {
        guard let container = try await Gemma4TestSupport.syntheticGemma4Container() else {
            print("[Skip] No Metal device available for Gemma4 runtime tests")
            return
        }

        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let prepared = try await container.prepare( ModelInput(chat: [
                .user([
                    .text("Describe"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                ])
            ])
        )
        let executable = try container.makeExecutablePrompt(from: prepared)

        #expect(executable.visualContext == nil)
        #expect(executable.tokenIDs == prepared.tokenIDs)
    }

    @Test("Gemma4 text generation runs end-to-end", .timeLimit(.minutes(2)))
    func textGeneration() async throws {
        guard let container = try await Gemma4TestSupport.syntheticGemma4Container() else {
            print("[Skip] No Metal device available for Gemma4 runtime tests")
            return
        }
        let stream = try await container.generate( ModelInput(prompt: "hello gemma4"),
            parameters: GenerationParameters(maxTokens: 2, streamChunkTokenCount: 1)
        )
        let result = await QwenVisionTestSupport.collectGeneration(from: stream)

        #expect(!result.chunks.isEmpty)
        #expect(try #require(result.completion).tokenCount > 0)
    }

    @Test("Gemma4 image generation runs end-to-end", .timeLimit(.minutes(2)))
    func imageGeneration() async throws {
        guard let container = try await Gemma4TestSupport.syntheticGemma4Container() else {
            print("[Skip] No Metal device available for Gemma4 runtime tests")
            return
        }

        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let stream = try await container.generate( ModelInput(chat: [
                .user([
                    .text("Describe"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                    .text("briefly"),
                ])
            ]),
            parameters: GenerationParameters(maxTokens: 2, streamChunkTokenCount: 1)
        )
        let result = await QwenVisionTestSupport.collectGeneration(from: stream)

        #expect(!result.chunks.isEmpty)
        #expect(try #require(result.completion).tokenCount > 0)
    }

    @Test("Gemma4 multimodal prompt state can be reused", .timeLimit(.minutes(2)))
    func promptStateReuse() async throws {
        guard let container = try await Gemma4TestSupport.syntheticGemma4Container() else {
            print("[Skip] No Metal device available for Gemma4 runtime tests")
            return
        }

        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let prepared = try await container.prepare( ModelInput(chat: [
                .user([
                    .text("Reuse"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                    .text("prompt"),
                ])
            ])
        )
        let prompt = try container.makeExecutablePrompt(from: prepared)

        let direct = await QwenVisionTestSupport.collectGeneration(
            from: try container.generate(from: prompt,
                parameters: GenerationParameters(
                    maxTokens: 2,
                    streamChunkTokenCount: 1,
                    temperature: 0.6,
                    topK: 20
                )
            )
        )

        container.resetState()
        let promptState = try container.makePromptSnapshot(from: prompt)
        let restored = await QwenVisionTestSupport.collectGeneration(
            from: try container.generate(
                from: promptState,
                parameters: GenerationParameters(
                    maxTokens: 2,
                    streamChunkTokenCount: 1,
                    temperature: 0.6,
                    topK: 20
                )
            )
        )

        #expect(direct.chunks == restored.chunks)
        #expect(direct.completion?.tokenCount == restored.completion?.tokenCount)
    }

    @Test("Gemma4 output-head diagnostics expose transfer layout", .timeLimit(.minutes(2)))
    func outputHeadDiagnosticsExposeTransferLayout() async throws {
        guard let container = try await Gemma4TestSupport.syntheticGemma4Container() else {
            print("[Skip] No Metal device available for Gemma4 runtime tests")
            return
        }

        let prepared = try await container.prepare( ModelInput(prompt: "hello gemma4"))
        let prompt = try container.makeExecutablePrompt(from: prepared)
        let diagnostics = try container.debugPrefillOutputHeadDiagnostics(prompt: prompt, topK: 5)

        #expect(diagnostics.inputLayout.hiddenCount == 64)
        #expect(diagnostics.transferLayout.transferCount == 64)
        #expect(diagnostics.transferSource.count == 64)
        #expect(diagnostics.transferDestination.count == 64)
    }

    private func rmsNorm(
        _ input: [Float],
        weight: [Float],
        epsilon: Float
    ) -> [Float] {
        let meanSquare = input.reduce(Float.zero) { partial, value in
            partial + value * value
        } / Float(input.count)
        let scale = 1 / sqrt(meanSquare + epsilon)
        return input.enumerated().map { index, value in
            value * scale * weight[index]
        }
    }
}
