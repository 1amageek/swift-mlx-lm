import Testing
import TestHeartbeat
import Foundation
import Metal
import Tokenizers
@testable import SwiftLM
@testable import MetalCompiler
@testable import ModelDeclarations
@testable import LMArchitecture

@Suite("Performance: Generation Throughput", .tags(.performance), .serialized, .heartbeat)
struct GenerationThroughputBenchmarkTests {

    @Test("LanguageModelContext generate throughput", .timeLimit(.minutes(2)))
    func languageModelContextGenerateThroughput() async throws {
        let promptTokens = [1, 1, 6, 6423, 708]
        let generateCount = 50
        var resources = try await GenerationPipelineBenchmarkSupport.makeResources(
            maximumSequenceLength: promptTokens.count
        )
        defer { resources.release() }

        let syncResult = try GenerationPipelineBenchmarkSupport.measureMedian(
            name: "sync decode",
            iterations: 5,
            warmup: 1
        ) {
            try GenerationPipelineBenchmarkSupport.runSynchronousLoop(
                model: &resources.syncModel,
                tokenizer: resources.tokenizer,
                promptTokens: promptTokens,
                generateCount: generateCount
            )
        }

        let pipelinedResult = try GenerationPipelineBenchmarkSupport.measureMedian(
            name: "pipelined decode",
            iterations: 5,
            warmup: 1
        ) {
            try GenerationPipelineBenchmarkSupport.runPipelinedLoop(
                model: &resources.pipelinedModel,
                tokenizer: resources.tokenizer,
                promptTokens: promptTokens,
                generateCount: generateCount
            )
        }

        let generateResult = try await GenerationPipelineBenchmarkSupport.measureMedianAsync(
            name: "LanguageModelContext.generate(greedy)",
            iterations: 5,
            warmup: 1
        ) {
            try await GenerationPipelineBenchmarkSupport.runContainerGenerate(
                container: resources.container,
                promptTokens: promptTokens,
                parameters: GenerationParameters(maxTokens: generateCount, temperature: 0)
            )
        }

        print("")
        print("=== GenerationEvent Pipeline Benchmark: LFM2.5-1.2B ===")
        print("Mode                    tok/s   ms/tok  generated")
        print("--------------------------------------------------")
        print(GenerationPipelineBenchmarkSupport.format(syncResult))
        print(GenerationPipelineBenchmarkSupport.format(pipelinedResult))
        print(GenerationPipelineBenchmarkSupport.format(generateResult))

        let pipelinedGain = ((pipelinedResult.tokensPerSecond / syncResult.tokensPerSecond) - 1.0) * 100.0
        let generateGain = ((generateResult.tokensPerSecond / syncResult.tokensPerSecond) - 1.0) * 100.0
        print(String(format: "[Benchmark] pipelined vs sync: %+0.1f%%", pipelinedGain))
        print(String(format: "[Benchmark] generate vs sync: %+0.1f%%", generateGain))

        #expect(pipelinedResult.generatedTokenCount == generateCount)
        #expect(generateResult.generatedTokenCount == generateCount)
    }

    @Test("GenerationEvent host overhead breakdown", .timeLimit(.minutes(2)))
    func generationHostOverheadBreakdown() async throws {
        let promptTokens = [1, 1, 6, 6423, 708]
        let generateCount = 50
        var resources = try await GenerationPipelineBenchmarkSupport.makeResources(
            maximumSequenceLength: promptTokens.count
        )
        defer { resources.release() }

        let rawTokens = try GenerationPipelineBenchmarkSupport.collectGeneratedTokens(
            model: &resources.syncModel,
            promptTokens: promptTokens,
            generateCount: generateCount
        )

        let rawDecode = try GenerationPipelineBenchmarkSupport.measureMedian(
            name: "raw GPU decode",
            iterations: 5,
            warmup: 1
        ) {
            try GenerationPipelineBenchmarkSupport.runSynchronousLoopNoTokenizer(
                model: &resources.syncModel,
                promptTokens: promptTokens,
                generateCount: generateCount
            )
        }

        let tokenDecode = GenerationPipelineBenchmarkSupport.measureTokenizerMedian(
            name: "tokenizer.decode",
            iterations: 10,
            warmup: 2,
            tokenizer: resources.tokenizer,
            tokens: rawTokens
        )

        let syncWithTokenizer = try GenerationPipelineBenchmarkSupport.measureMedian(
            name: "GPU+tokenizer",
            iterations: 5,
            warmup: 1
        ) {
            try GenerationPipelineBenchmarkSupport.runSynchronousLoop(
                model: &resources.pipelinedModel,
                tokenizer: resources.tokenizer,
                promptTokens: promptTokens,
                generateCount: generateCount
            )
        }

        print("")
        print("=== Host Overhead Breakdown: LFM2.5-1.2B ===")
        print("Mode                    tok/s   ms/tok  generated")
        print("--------------------------------------------------")
        print(GenerationPipelineBenchmarkSupport.format(rawDecode))
        print(GenerationPipelineBenchmarkSupport.format(tokenDecode))
        print(GenerationPipelineBenchmarkSupport.format(syncWithTokenizer))

        let tokenizerShare = tokenDecode.elapsed / syncWithTokenizer.elapsed * 100.0
        print(String(format: "[Benchmark] tokenizer share of GPU+tokenizer path: %.1f%%", tokenizerShare))

        #expect(rawTokens.count == generateCount)
    }

}

@Suite("Performance: Generation Scaling", .tags(.performance), .serialized, .heartbeat)
struct GenerationScalingBenchmarkTests {

    @Test("Request-level scaling (50)", .timeLimit(.minutes(2)))
    func requestLevelScaling50() async throws {
        try await Self.runScalingCase(generateCount: 50)
    }

    @Test("Request-level scaling (128)", .timeLimit(.minutes(2)))
    func requestLevelScaling128() async throws {
        try await Self.runScalingCase(generateCount: 128)
    }

    @Test("Request-level scaling (256)", .timeLimit(.minutes(2)))
    func requestLevelScaling256() async throws {
        try await Self.runScalingCase(generateCount: 256)
    }

    static func runScalingCase(generateCount: Int) async throws {
        let promptTokens = [1, 1, 6, 6423, 708]
        let iterations = generateCount >= 512 ? 2 : 3
        let warmup = generateCount >= 512 ? 0 : 1
        var resources = try await GenerationPipelineBenchmarkSupport.makeResources(
            maximumSequenceLength: promptTokens.count
        )
        defer { resources.release() }

        let rawResult = try GenerationPipelineBenchmarkSupport.measureMedian(
            name: "raw-\(generateCount)",
            iterations: iterations,
            warmup: warmup
        ) {
            try GenerationPipelineBenchmarkSupport.runSynchronousLoopNoTokenizer(
                model: &resources.syncModel,
                promptTokens: promptTokens,
                generateCount: generateCount
            )
        }

        let apiResult = try await GenerationPipelineBenchmarkSupport.measureMedianAsync(
            name: "api-\(generateCount)",
            iterations: iterations,
            warmup: warmup
        ) {
            try await GenerationPipelineBenchmarkSupport.runContainerGenerate(
                container: resources.container,
                promptTokens: promptTokens,
                parameters: GenerationParameters(maxTokens: generateCount, temperature: 0)
            )
        }

        print("")
        print("=== Request-Level Scaling: LFM2.5-1.2B ===")
        print("Generated    raw tok/s   raw ms/tok   API tok/s   API ms/tok")
        print("-------------------------------------------------------------")
        let generated = String(format: "%9d", generateCount)
        let rawTok = String(format: "%10.1f", rawResult.tokensPerSecond)
        let rawMs = String(format: "%12.2f", rawResult.millisecondsPerToken)
        let apiTok = String(format: "%10.1f", apiResult.tokensPerSecond)
        let apiMs = String(format: "%12.2f", apiResult.millisecondsPerToken)
        print("\(generated)\(rawTok)\(rawMs)\(apiTok)\(apiMs)")

        #expect(rawResult.generatedTokenCount == generateCount)
        #expect(apiResult.generatedTokenCount == generateCount)
    }
}

@Suite("Performance: Generation Scaling (Long)", .tags(.performance), .serialized, .heartbeat)
struct GenerationScalingLongBenchmarkTests {
    @Test("Request-level scaling (512)", .timeLimit(.minutes(2)))
    func requestLevelScaling512() async throws {
        try await GenerationScalingBenchmarkTests.runScalingCase(generateCount: 512)
    }
}

@Suite("Performance: Generation Streaming", .tags(.performance), .serialized, .heartbeat)
struct GenerationStreamingBenchmarkTests {

    @Test("Stream chunk size comparison", .timeLimit(.minutes(3)))
    func streamChunkSizeComparison() async throws {
        let promptTokens = [1, 1, 6, 6423, 708]
        let generateCount = 128
        let chunkSizes = [1, 8, 16]

        print("")
        print("=== Stream Chunk Size Comparison: LFM2.5-1.2B ===")
        print("Chunk size   tok/s   ms/tok   first chunk ms   chunks")
        print("-----------------------------------------------------")

        for chunkSize in chunkSizes {
            let result: GenerationPipelineBenchmarkSupport.StreamResult = try await {
                var resources = try await GenerationPipelineBenchmarkSupport.makeResources(
                    maximumSequenceLength: promptTokens.count
                )
                defer { resources.release() }
                return try await GenerationPipelineBenchmarkSupport.measureStreamMedian(
                    iterations: 1,
                    warmup: 0
                ) {
                    try await GenerationPipelineBenchmarkSupport.runContainerGenerateMeasured(
                        container: resources.container,
                        promptTokens: promptTokens,
                        generateCount: generateCount,
                        chunkTokenCount: chunkSize,
                        temperature: 0
                    )
                }
            }()

            let sizeText = String(format: "%10d", chunkSize)
            let tokText = String(format: "%7.1f", result.tokensPerSecond)
            let msText = String(format: "%8.2f", result.millisecondsPerToken)
            let firstText = String(format: "%16.2f", result.firstChunkMilliseconds)
            let chunkCountText = String(format: "%8d", result.chunkCount)
            print("\(sizeText)\(tokText)\(msText)\(firstText)\(chunkCountText)")

            #expect(result.generatedTokenCount == generateCount)
        }
    }

    @Test("Prompt state reuse comparison", .timeLimit(.minutes(3)))
    func promptStateReuseComparison() async throws {
        let promptTokens = [Int](repeating: 1, count: 256)
        let generateCount = 50
        var resources = try await GenerationPipelineBenchmarkSupport.makeResources(
            maximumSequenceLength: promptTokens.count
        )
        defer { resources.release() }
        let promptState = try PromptSnapshot(
            from: ExecutablePrompt(tokenIDs: promptTokens),
            using: resources.container
        )

        let baseline = try await GenerationPipelineBenchmarkSupport.measureStreamMedian(
            iterations: 3,
            warmup: 1
        ) {
            try await GenerationPipelineBenchmarkSupport.runContainerGenerateMeasured(
                container: resources.container,
                promptTokens: promptTokens,
                generateCount: generateCount,
                chunkTokenCount: 8,
                temperature: 0
            )
        }

        let reused = try await GenerationPipelineBenchmarkSupport.measureStreamMedian(
            iterations: 3,
            warmup: 1
        ) {
            try await GenerationPipelineBenchmarkSupport.runContainerGenerateMeasured(
                container: resources.container,
                promptState: promptState,
                generateCount: generateCount,
                chunkTokenCount: 8,
                temperature: 0
            )
        }

        print("")
        print("=== Prompt State Reuse Comparison: LFM2.5-1.2B ===")
        print("Mode                    tok/s   ms/tok   first chunk ms")
        print("-------------------------------------------------------")
        print(GenerationPipelineBenchmarkSupport.format(stream: baseline, name: "baseline"))
        print(GenerationPipelineBenchmarkSupport.format(stream: reused, name: "prompt-state"))

        let reuseGain = ((reused.tokensPerSecond / baseline.tokensPerSecond) - 1.0) * 100.0
        let ttftGain = ((baseline.firstChunkMilliseconds / reused.firstChunkMilliseconds) - 1.0) * 100.0
        print(String(format: "[Benchmark] prompt-state reuse vs baseline: %+0.1f%%", reuseGain))
        print(String(format: "[Benchmark] TTFT reduction vs baseline: %+0.1f%%", ttftGain))

        #expect(reused.generatedTokenCount == generateCount)
    }
}

private enum GenerationPipelineBenchmarkSupport {
    static func makeResources(
        maximumSequenceLength: Int = 256
    ) async throws -> BenchmarkResources {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw BenchmarkError.noDevice
        }

        let directory = try findModelDirectory()
        let bundleResources = try ModelBundleInspector().inspect(directory: directory)
        async let tokenizerTask = AutoTokenizer.from(modelFolder: directory)
        async let weightStoreTask = STAFCacheLoader().load(resources: bundleResources, device: device)

        let tokenizer = try await tokenizerTask
        let store = try await weightStoreTask

        let graphResolver = ModelGraphResolver()
        let graph = try graphResolver.resolveModelGraph(
            modelType: bundleResources.modelType,
            config: bundleResources.config
        )
        let convention = graphResolver.namingConvention(for: bundleResources.modelType)
        let resolved = ParameterResolver().resolve(graph: graph, convention: convention)
        let decodePolicy = resolveDecodePolicy(
            maximumSequenceLength: maximumSequenceLength,
            graph: resolved
        )
        let prefillPolicy = resolvePrefillPolicy(for: decodePolicy)

        let compiler = MetalInferenceCompiler()
        let decodePlan = try compiler.compile(
            graph: resolved,
            hiddenSize: bundleResources.config.hiddenSize,
            intermediateSize: bundleResources.config.intermediateSize,
            vocabSize: bundleResources.config.vocabSize,
            inferencePolicy: decodePolicy,
            stafWeightStore: store,
            device: device
        )
        let prefillPlan = try compiler.compilePrefill(
            graph: resolved,
            hiddenSize: bundleResources.config.hiddenSize,
            intermediateSize: bundleResources.config.intermediateSize,
            vocabSize: bundleResources.config.vocabSize,
            inferencePolicy: prefillPolicy,
            stafWeightStore: store,
            sharedKVCache: shouldShareKVCache(
                decodePolicy: decodePolicy,
                prefillPolicy: prefillPolicy
            ) ? decodePlan.buffers.kvCache : nil,
            sharedConvState: decodePlan.buffers.convState,
            sharedConvStateDimension: decodePlan.buffers.convStateDimension,
            sharedConvStateKernelSize: decodePlan.buffers.convStateKernelSize,
            sharedRecurrentState: decodePlan.buffers.recurrentState,
            sharedRecurrentStateBytesPerLayer: decodePlan.buffers.recurrentStateBytesPerLayer,
            device: device
        )

        var syncModel = try MetalInferenceModel(plan: decodePlan, device: device)
        syncModel.prefillPlan = prefillPlan

        var pipelinedModel = try MetalInferenceModel(plan: decodePlan, device: device)
        pipelinedModel.prefillPlan = prefillPlan

        var containerModel = try MetalInferenceModel(plan: decodePlan, device: device)
        containerModel.prefillPlan = prefillPlan

        var configuration = ModelConfiguration(
            name: bundleResources.modelType,
            inputCapabilities: bundleResources.inputCapabilities,
            executionCapabilities: ModelExecutionCapabilities(
                supportsTextGeneration: true,
                supportsPromptStateReuse: true
            ),
            vision: bundleResources.visionConfiguration
        )
        if let eosId = tokenizer.eosTokenId {
            configuration.eosTokenIds.insert(eosId)
        }

        let container = LanguageModelContext(
            inferenceModel: containerModel,
            tokenizer: tokenizer,
            configuration: configuration,
            chatTemplate: bundleResources.chatTemplate,
            chatTemplateSource: bundleResources.chatTemplateSource,
            vocabularySize: bundleResources.config.vocabSize,
            finalLogitSoftcapping: bundleResources.config.finalLogitSoftcapping
        )

        return BenchmarkResources(
            tokenizer: tokenizer,
            syncModel: syncModel,
            pipelinedModel: pipelinedModel,
            container: container
        )
    }

    static func runSynchronousLoop(
        model: inout MetalInferenceModel,
        tokenizer: any Tokenizer,
        promptTokens: [Int],
        generateCount: Int
    ) throws -> Int {
        model.resetState()
        let prompt = promptTokens.map(Int32.init)
        var token = model.prefill(tokens: prompt)
        guard token >= 0 else { throw BenchmarkError.invalidToken }

        _ = tokenizer.decode(tokens: [Int(token)])
        var generated = 1

        while generated < generateCount {
            token = model.decodeSync(tokenID: token)
            guard token >= 0 else { throw BenchmarkError.invalidToken }
            _ = tokenizer.decode(tokens: [Int(token)])
            generated += 1
        }

        return generated
    }

    static func runSynchronousLoopNoTokenizer(
        model: inout MetalInferenceModel,
        promptTokens: [Int],
        generateCount: Int
    ) throws -> Int {
        model.resetState()
        let prompt = promptTokens.map(Int32.init)
        var token = model.prefill(tokens: prompt)
        guard token >= 0 else { throw BenchmarkError.invalidToken }

        var generated = 1
        while generated < generateCount {
            token = model.decodeSync(tokenID: token)
            guard token >= 0 else { throw BenchmarkError.invalidToken }
            generated += 1
        }
        return generated
    }

    static func runPipelinedLoop(
        model: inout MetalInferenceModel,
        tokenizer: any Tokenizer,
        promptTokens: [Int],
        generateCount: Int
    ) throws -> Int {
        model.resetState()
        let prompt = promptTokens.map(Int32.init)
        let firstToken = model.prefill(tokens: prompt)
        guard firstToken >= 0 else { throw BenchmarkError.invalidToken }

        _ = tokenizer.decode(tokens: [Int(firstToken)])
        var generated = 1
        var nextToken = firstToken

        while generated < generateCount {
            let token = model.decode(tokenID: nextToken)
            guard token >= 0 else { throw BenchmarkError.invalidToken }

            _ = tokenizer.decode(tokens: [Int(token)])
            generated += 1
            nextToken = token
        }

        return generated
    }

    static func runContainerGenerate(
        container: LanguageModelContext,
        promptTokens: [Int],
        parameters: GenerationParameters
    ) async throws -> Int {
        container.resetState()
        let stream = try container.generate(
            from: ExecutablePrompt(tokenIDs: promptTokens),
            parameters: parameters
        )

        var completed = false
        for await item in stream {
            if case .completed(let info) = item {
                completed = info.totalTime >= 0
            }
        }
        return completed ? (parameters.maxTokens ?? 0) : 0
    }

    static func runContainerGenerate(
        container: LanguageModelContext,
        promptState: PromptSnapshot,
        parameters: GenerationParameters
    ) async throws -> Int {
        let stream = try container.generate(
            from: promptState,
            parameters: parameters
        )

        var completed = false
        for await item in stream {
            if case .completed(let info) = item {
                completed = info.totalTime >= 0
            }
        }
        return completed ? (parameters.maxTokens ?? 0) : 0
    }

    static func runContainerGenerateMeasured(
        container: LanguageModelContext,
        promptTokens: [Int],
        generateCount: Int,
        chunkTokenCount: Int,
        temperature: Float
    ) async throws -> StreamResult {
        container.resetState()
        let start = CFAbsoluteTimeGetCurrent()
        let stream = try container.generate(
            from: ExecutablePrompt(tokenIDs: promptTokens),
            parameters: GenerationParameters(
                maxTokens: generateCount,
                streamChunkTokenCount: chunkTokenCount,
                temperature: temperature
            )
        )

        var completed = false
        var chunkCount = 0
        var firstChunkElapsed: Double?

        for await item in stream {
            switch item {
            case .text:
                chunkCount += 1
                if firstChunkElapsed == nil {
                    firstChunkElapsed = CFAbsoluteTimeGetCurrent() - start
                }
            case .reasoning:
                break
            case .completed(let info):
                completed = info.totalTime >= 0
            }
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        return StreamResult(
            generatedTokenCount: completed ? generateCount : 0,
            elapsed: elapsed,
            firstChunkElapsed: firstChunkElapsed ?? elapsed,
            chunkCount: chunkCount
        )
    }

    static func runContainerGenerateMeasured(
        container: LanguageModelContext,
        promptState: PromptSnapshot,
        generateCount: Int,
        chunkTokenCount: Int,
        temperature: Float
    ) async throws -> StreamResult {
        let start = CFAbsoluteTimeGetCurrent()
        let stream = try container.generate(
            from: promptState,
            parameters: GenerationParameters(
                maxTokens: generateCount,
                streamChunkTokenCount: chunkTokenCount,
                temperature: temperature
            )
        )

        var completed = false
        var chunkCount = 0
        var firstChunkElapsed: Double?

        for await item in stream {
            switch item {
            case .text:
                chunkCount += 1
                if firstChunkElapsed == nil {
                    firstChunkElapsed = CFAbsoluteTimeGetCurrent() - start
                }
            case .reasoning:
                break
            case .completed(let info):
                completed = info.totalTime >= 0
            }
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        return StreamResult(
            generatedTokenCount: completed ? generateCount : 0,
            elapsed: elapsed,
            firstChunkElapsed: firstChunkElapsed ?? elapsed,
            chunkCount: chunkCount
        )
    }

    static func collectGeneratedTokens(
        model: inout MetalInferenceModel,
        promptTokens: [Int],
        generateCount: Int
    ) throws -> [Int] {
        model.resetState()
        let prompt = promptTokens.map(Int32.init)
        var token = model.prefill(tokens: prompt)
        guard token >= 0 else { throw BenchmarkError.invalidToken }

        var generated: [Int] = [Int(token)]
        while generated.count < generateCount {
            token = model.decodeSync(tokenID: token)
            guard token >= 0 else { throw BenchmarkError.invalidToken }
            generated.append(Int(token))
        }
        return generated
    }

    static func measureMedian(
        name: String,
        iterations: Int,
        warmup: Int,
        block: () throws -> Int
    ) throws -> ThroughputResult {
        for _ in 0..<warmup {
            _ = try block()
        }

        var samples: [ThroughputResult] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let generated = try block()
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            samples.append(ThroughputResult(name: name, generatedTokenCount: generated, elapsed: elapsed))
        }
        samples.sort { $0.elapsed < $1.elapsed }
        return samples[samples.count / 2]
    }

    static func measureMedianAsync(
        name: String,
        iterations: Int,
        warmup: Int,
        block: () async throws -> Int
    ) async throws -> ThroughputResult {
        for _ in 0..<warmup {
            _ = try await block()
        }

        var samples: [ThroughputResult] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let generated = try await block()
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            samples.append(ThroughputResult(name: name, generatedTokenCount: generated, elapsed: elapsed))
        }
        samples.sort { $0.elapsed < $1.elapsed }
        return samples[samples.count / 2]
    }

    static func measureTokenizerMedian(
        name: String,
        iterations: Int,
        warmup: Int,
        tokenizer: any Tokenizer,
        tokens: [Int]
    ) -> ThroughputResult {
        for _ in 0..<warmup {
            for token in tokens {
                _ = tokenizer.decode(tokens: [token])
            }
        }

        var samples: [ThroughputResult] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            for token in tokens {
                _ = tokenizer.decode(tokens: [token])
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            samples.append(ThroughputResult(name: name, generatedTokenCount: tokens.count, elapsed: elapsed))
        }
        samples.sort { $0.elapsed < $1.elapsed }
        return samples[samples.count / 2]
    }

    static func measureStreamMedian(
        iterations: Int,
        warmup: Int,
        block: () async throws -> StreamResult
    ) async throws -> StreamResult {
        for _ in 0..<warmup {
            _ = try await block()
        }

        var samples: [StreamResult] = []
        for _ in 0..<iterations {
            samples.append(try await block())
        }
        samples.sort { $0.elapsed < $1.elapsed }
        return samples[samples.count / 2]
    }

    static func format(_ result: ThroughputResult) -> String {
        let paddedName = result.name.padding(toLength: 22, withPad: " ", startingAt: 0)
        return "\(paddedName) \(String(format: "%7.1f", result.tokensPerSecond)) \(String(format: "%8.2f", result.millisecondsPerToken)) \(String(format: "%10d", result.generatedTokenCount))"
    }

    static func format(stream result: StreamResult, name: String) -> String {
        let paddedName = name.padding(toLength: 22, withPad: " ", startingAt: 0)
        return "\(paddedName) \(String(format: "%7.1f", result.tokensPerSecond)) \(String(format: "%8.2f", result.millisecondsPerToken)) \(String(format: "%16.2f", result.firstChunkMilliseconds))"
    }

    static func findModelDirectory() throws -> URL {
        // Resolve from HF cache only — model bundles live under ~/.cache/huggingface/hub/.
        let candidates = [
            "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct",
            "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B",
            "~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Thinking",
        ]

        for candidate in candidates {
            let expanded = NSString(string: candidate).expandingTildeInPath
            let baseURL = URL(fileURLWithPath: expanded)
            let snapshotsURL = baseURL.appendingPathComponent("snapshots")
            let snapshots: [URL]
            do {
                snapshots = try FileManager.default.contentsOfDirectory(
                    at: snapshotsURL,
                    includingPropertiesForKeys: nil
                )
            } catch {
                continue
            }

            for snapshot in snapshots {
                let configPath = snapshot.appendingPathComponent("config.json")
                guard FileManager.default.fileExists(atPath: configPath.path) else { continue }
                let tokenizerPath = snapshot.appendingPathComponent("tokenizer.json")
                guard FileManager.default.fileExists(atPath: tokenizerPath.path) else { continue }
                return snapshot
            }
        }

        throw BenchmarkError.noModelDirectory
    }

    static func resolveDecodePolicy(
        maximumSequenceLength: Int,
        graph: ModelGraph
    ) -> InferencePolicy {
        let requestedPolicy = InferencePolicy(maximumSequenceLength: max(1, maximumSequenceLength))
        return ModelBundleLoader.resolveInferencePolicy(requestedPolicy, for: graph)
    }

    static func resolvePrefillPolicy(for decodePolicy: InferencePolicy) -> InferencePolicy {
        guard decodePolicy.kvCache.usesRotorQuant else {
            return decodePolicy
        }

        return InferencePolicy(
            maximumSequenceLength: decodePolicy.maximumSequenceLength,
            kvCache: KVCachePolicy(
                keyScheme: .automatic,
                valueScheme: .automatic,
                layoutMode: decodePolicy.kvCache.layoutMode,
                qjlDimension: 0
            )
        )
    }

    static func shouldShareKVCache(
        decodePolicy: InferencePolicy,
        prefillPolicy: InferencePolicy
    ) -> Bool {
        decodePolicy.kvCache == prefillPolicy.kvCache
    }

    struct BenchmarkResources {
        let tokenizer: any Tokenizer
        var syncModel: MetalInferenceModel
        var pipelinedModel: MetalInferenceModel
        let container: LanguageModelContext

        mutating func release() {
            syncModel.resetState()
            pipelinedModel.resetState()
            container.resetState()
        }
    }

    struct ThroughputResult {
        let name: String
        let generatedTokenCount: Int
        let elapsed: Double

        var tokensPerSecond: Double {
            Double(generatedTokenCount) / elapsed
        }

        var millisecondsPerToken: Double {
            elapsed / Double(generatedTokenCount) * 1000.0
        }
    }

    struct StreamResult {
        let generatedTokenCount: Int
        let elapsed: Double
        let firstChunkElapsed: Double
        let chunkCount: Int

        var tokensPerSecond: Double {
            Double(generatedTokenCount) / elapsed
        }

        var millisecondsPerToken: Double {
            elapsed / Double(generatedTokenCount) * 1000.0
        }

        var firstChunkMilliseconds: Double {
            firstChunkElapsed * 1000.0
        }
    }

    enum BenchmarkError: Error {
        case noDevice
        case noModelDirectory
        case invalidToken
    }
}
