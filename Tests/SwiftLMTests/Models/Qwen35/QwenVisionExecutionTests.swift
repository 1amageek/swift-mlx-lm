import Foundation
import Testing
@testable import SwiftLM

@Suite("Qwen Vision Runtime", .serialized)
struct QwenVisionExecutionTests {
    @Test("Prepared text-only input becomes executable text-only input", .timeLimit(.minutes(2)))
    func preparedTextOnlyBecomesExecutable() async throws {
        guard let container = try await QwenVisionTestSupport.syntheticMultimodalContainer() else {
            print("[Skip] No local text bundle available for synthetic multimodal runtime tests")
            return
        }
        container.resetState()

        let prepared = try await container.prepare( ModelInput(prompt: "Hello multimodal runtime"))
        let executable = try ExecutablePrompt(preparedPrompt: prepared, using: container)

        #expect(executable.visualContext == nil)
        #expect(executable.tokenIDs == prepared.tokenIDs)
    }

    @Test("Prepared image input becomes executable multimodal input", .timeLimit(.minutes(2)))
    func preparedImageBecomesExecutableMultimodal() async throws {
        guard let container = try await QwenVisionTestSupport.syntheticMultimodalContainer() else {
            print("[Skip] No local text bundle available for synthetic multimodal runtime tests")
            return
        }
        container.resetState()

        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let prepared = try await container.prepare( ModelInput(chat: [
                .user([
                    .text("Describe"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                ])
            ])
        )
        let executable = try ExecutablePrompt(preparedPrompt: prepared, using: container)

        let multimodal = try #require(executable.visualContext)
        #expect(multimodal.layout.mmTokenTypeIDs.contains(1))
        #expect(!multimodal.imageTokenEmbeddings.isEmpty)
        #expect(multimodal.imageTokenEmbeddings.count == multimodal.layout.mmTokenTypeIDs.filter { $0 == 1 }.count)
    }

    @Test("Prepared video input becomes executable multimodal input", .timeLimit(.minutes(2)))
    func preparedVideoBecomesExecutableMultimodal() async throws {
        guard let container = try await QwenVisionTestSupport.syntheticMultimodalContainer() else {
            print("[Skip] No local text bundle available for synthetic multimodal runtime tests")
            return
        }
        container.resetState()

        let videoData = try await TestVideoFixtures.makeMP4Data()
        let prepared = try await container.prepare( ModelInput(chat: [
                .user([
                    .text("Describe"),
                    .video(InputVideo(data: videoData, mimeType: "video/mp4")),
                ])
            ])
        )
        let executable = try ExecutablePrompt(preparedPrompt: prepared, using: container)

        let multimodal = try #require(executable.visualContext)
        #expect(multimodal.layout.mmTokenTypeIDs.contains(2))
        #expect(!multimodal.videoTokenEmbeddings.isEmpty)
        #expect(multimodal.videoTokenEmbeddings.count == multimodal.layout.mmTokenTypeIDs.filter { $0 == 2 }.count)
    }

    @Test("Synthetic multimodal model input runs through generation", .timeLimit(.minutes(2)))
    func syntheticModelInputGenerates() async throws {
        guard let container = try await QwenVisionTestSupport.syntheticMultimodalContainer() else {
            print("[Skip] No local text bundle available for synthetic multimodal runtime tests")
            return
        }
        container.resetState()

        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let stream = try await container.generate( ModelInput(chat: [
                .user([
                    .text("Hello"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                    .text("world"),
                ])
            ]),
            parameters: GenerationParameters(maxTokens: 2, streamChunkTokenCount: 1)
        )
        let result = await QwenVisionTestSupport.collectGeneration(from: stream)

        #expect(!result.chunks.isEmpty)
        #expect(try #require(result.completion).tokenCount > 0)
    }

    @Test("Synthetic video model input can build a prompt state", .timeLimit(.minutes(2)))
    func syntheticVideoModelInputBuildsPromptState() async throws {
        guard let container = try await QwenVisionTestSupport.syntheticMultimodalContainer() else {
            print("[Skip] No local text bundle available for synthetic multimodal runtime tests")
            return
        }
        container.resetState()

        let videoData = try await TestVideoFixtures.makeMP4Data()
        let promptState = try await PromptSnapshot(from: ModelInput(chat: [
                .user([
                    .text("Watch"),
                    .video(InputVideo(data: videoData, mimeType: "video/mp4")),
                    .text("carefully"),
                ])
            ]), using: container)

        #expect(promptState.promptTokenCount > 0)
    }

    @Test("Prompt state reuse preserves multimodal execution flow", .timeLimit(.minutes(2)))
    func promptStateReuseForMultimodalInput() async throws {
        guard let container = try await QwenVisionTestSupport.syntheticMultimodalContainer() else {
            print("[Skip] No local text bundle available for synthetic multimodal runtime tests")
            return
        }
        container.resetState()

        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let prepared = try await container.prepare( ModelInput(chat: [
                .user([
                    .text("Reuse"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                    .text("prompt"),
                ])
            ])
        )
        let executable = try ExecutablePrompt(preparedPrompt: prepared, using: container)
        let directStream = try container.generate(from: executable,
            parameters: GenerationParameters(maxTokens: 2, streamChunkTokenCount: 1)
        )
        let direct = await QwenVisionTestSupport.collectGeneration(from: directStream)

        container.resetState()
        let promptState = try PromptSnapshot(from: executable, using: container)
        let restoredStream = try container.generate(
            from: promptState,
            parameters: GenerationParameters(maxTokens: 2, streamChunkTokenCount: 1)
        )
        let restored = await QwenVisionTestSupport.collectGeneration(from: restoredStream)

        #expect(direct.chunks == restored.chunks)
        #expect(direct.completion?.tokenCount == restored.completion?.tokenCount)
    }

    @Test("Prompt state restore reproduces multimodal prefill logits and tokenOut", .timeLimit(.minutes(2)))
    func promptStateRestoreReproducesMultimodalPrefillState() async throws {
        guard let container = try await QwenVisionTestSupport.syntheticMultimodalContainer() else {
            print("[Skip] No local text bundle available for synthetic multimodal runtime tests")
            return
        }
        container.resetState()

        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let prepared = try await container.prepare( ModelInput(chat: [
                .user([
                    .text("Reuse"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                    .text("prompt"),
                ])
            ])
        )
        let executable = try ExecutablePrompt(preparedPrompt: prepared, using: container)
        let diagnostics = try container.debugPromptStateRestoreDiagnostics(prompt: executable, topK: 5)

        #expect(diagnostics.directTokenOut == diagnostics.promptStateFirstToken)
        #expect(diagnostics.directTokenOut == diagnostics.restoredTokenOut)
        #expect(diagnostics.directTopLogits.map(\.tokenID) == diagnostics.restoredTopLogits.map(\.tokenID))
    }

    @Test("Prompt state reuse preserves first sampled multimodal token", .timeLimit(.minutes(2)))
    func promptStateReusePreservesFirstSampledMultimodalToken() async throws {
        guard let container = try await QwenVisionTestSupport.syntheticMultimodalContainer() else {
            print("[Skip] No local text bundle available for synthetic multimodal runtime tests")
            return
        }
        container.resetState()

        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let prepared = try await container.prepare( ModelInput(chat: [
                .user([
                    .text("Reuse"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                    .text("prompt"),
                ])
            ])
        )
        let executable = try ExecutablePrompt(preparedPrompt: prepared, using: container)
        let parameters = GenerationParameters(maxTokens: 1, streamChunkTokenCount: 1)

        let direct = await QwenVisionTestSupport.collectGeneration(
            from: try container.generate(from: executable, parameters: parameters)
        )

        container.resetState()
        let promptState = try PromptSnapshot(from: executable, using: container)
        let restored = await QwenVisionTestSupport.collectGeneration(
            from: try container.generate(from: promptState, parameters: parameters)
        )

        #expect(direct.chunks == restored.chunks)
    }

    @Test("Reset caches restores multimodal prefill determinism", .timeLimit(.minutes(2)))
    func resetCachesRestoresMultimodalPrefillDeterminism() async throws {
        guard let container = try await QwenVisionTestSupport.syntheticMultimodalContainer() else {
            print("[Skip] No local text bundle available for synthetic multimodal runtime tests")
            return
        }
        container.resetState()

        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let prepared = try await container.prepare( ModelInput(chat: [
                .user([
                    .text("Reuse"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                    .text("prompt"),
                ])
            ])
        )
        let executable = try ExecutablePrompt(preparedPrompt: prepared, using: container)
        let initial = try container.debugPrefillTopLogits(prompt: executable, topK: 5)

        _ = await QwenVisionTestSupport.collectGeneration(
            from: try container.generate(from: executable,
                parameters: GenerationParameters(maxTokens: 1, streamChunkTokenCount: 1)
            )
        )

        container.resetState()
        let replayed = try container.debugPrefillTopLogits(prompt: executable, topK: 5)

        #expect(initial.map(\.tokenID) == replayed.map(\.tokenID))
    }

    @Test("Prompt state sampling state matches direct multimodal prefill", .timeLimit(.minutes(2)))
    func promptStateSamplingStateMatchesDirectMultimodalPrefill() async throws {
        guard let container = try await QwenVisionTestSupport.syntheticMultimodalContainer() else {
            print("[Skip] No local text bundle available for synthetic multimodal runtime tests")
            return
        }
        container.resetState()

        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let prepared = try await container.prepare( ModelInput(chat: [
                .user([
                    .text("Reuse"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                    .text("prompt"),
                ])
            ])
        )
        let executable = try ExecutablePrompt(preparedPrompt: prepared, using: container)
        let sampled = try container.debugPromptStateSampledFirstTokens(
            prompt: executable,
            parameters: GenerationParameters(maxTokens: 1, streamChunkTokenCount: 1)
        )

        #expect(sampled.directRecentTokenIDs == sampled.restoredRecentTokenIDs)
        #expect(sampled.directTopLogits.map(\.tokenID) == sampled.restoredTopLogits.map(\.tokenID))
        #expect(sampled.directTopLogits.map(\.logit) == sampled.restoredTopLogits.map(\.logit))
        #expect(sampled.direct == sampled.restored)
    }

    @Test("Repeated multimodal prefills preserve sampled first token", .timeLimit(.minutes(2)))
    func repeatedMultimodalPrefillsPreserveSampledFirstToken() async throws {
        guard let container = try await QwenVisionTestSupport.syntheticMultimodalContainer() else {
            print("[Skip] No local text bundle available for synthetic multimodal runtime tests")
            return
        }
        container.resetState()

        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let prepared = try await container.prepare( ModelInput(chat: [
                .user([
                    .text("Reuse"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                    .text("prompt"),
                ])
            ])
        )
        let executable = try ExecutablePrompt(preparedPrompt: prepared, using: container)
        let repeated = try container.debugRepeatedPrefillSampledFirstTokens(
            prompt: executable,
            parameters: GenerationParameters(maxTokens: 1, streamChunkTokenCount: 1)
        )
        print(
            "[Qwen repeated prefill] nanCounts=(\(repeated.firstNaNCount), \(repeated.secondNaNCount)) "
                + "diffEntries=\(repeated.firstDifferingEntries)"
        )

        #expect(repeated.firstTopLogits.map(\.tokenID) == repeated.secondTopLogits.map(\.tokenID))
        #expect(repeated.firstTopLogits.map(\.logit) == repeated.secondTopLogits.map(\.logit))
        #expect(repeated.firstLogitFingerprint == repeated.secondLogitFingerprint)
        #expect(repeated.maxAbsDiff == 0)
        #expect(repeated.differingCount == 0)
        #expect(repeated.first == repeated.second)
    }

    @Test("Repeated multimodal prefills preserve final hidden before output head", .timeLimit(.minutes(2)))
    func repeatedMultimodalPrefillsPreserveFinalHidden() async throws {
        guard let container = try await QwenVisionTestSupport.syntheticMultimodalContainer() else {
            print("[Skip] No local text bundle available for synthetic multimodal runtime tests")
            return
        }
        container.resetState()

        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let prepared = try await container.prepare( ModelInput(chat: [
                .user([
                    .text("Reuse"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                    .text("prompt"),
                ])
            ])
        )
        let executable = try ExecutablePrompt(preparedPrompt: prepared, using: container)
        let diagnostics = try container.debugRepeatedPrefillFinalHiddenDiagnostics(prompt: executable)
        print(
            "[Qwen repeated hidden] nanCounts=(\(diagnostics.firstNaNCount), \(diagnostics.secondNaNCount)) "
                + "fingerprints=(\(diagnostics.firstFingerprint), \(diagnostics.secondFingerprint)) "
                + "differingCount=\(diagnostics.differingCount)"
        )

        #expect(diagnostics.firstNaNCount == diagnostics.secondNaNCount)
        #expect(diagnostics.differingCount == 0)
        #expect(diagnostics.firstFingerprint == diagnostics.secondFingerprint)
    }

}
