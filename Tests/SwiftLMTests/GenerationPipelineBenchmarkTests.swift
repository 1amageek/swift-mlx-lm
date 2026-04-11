import Testing
import TestHeartbeat
import Foundation
import Metal
import Tokenizers
@testable import SwiftLM
@testable import MetalCompiler
@testable import ModelDeclarations
@testable import LMArchitecture

@Suite("Performance: GenerationEvent Pipeline", .tags(.performance), .serialized, .heartbeat)
struct GenerationPipelineBenchmarkTests {

    private static let stafPath = "/Users/1amageek/Desktop/swift-lm/TestData/LFM2.5-1.2B-Thinking/model.staf"

    @Test("InferenceSession generate throughput", .timeLimit(.minutes(2)))
    func modelContainerGenerateThroughput() async throws {
        let promptTokens = [1, 1, 6, 6423, 708]
        let generateCount = 50
        var resources = try await makeResources(maximumSequenceLength: promptTokens.count)
        defer { resources.release() }

        let syncResult = try measureMedian(name: "sync decode", iterations: 5, warmup: 1) {
            try runSynchronousLoop(
                model: &resources.syncModel,
                tokenizer: resources.tokenizer,
                promptTokens: promptTokens,
                generateCount: generateCount)
        }

        let pipelinedResult = try measureMedian(name: "pipelined decode", iterations: 5, warmup: 1) {
            try runPipelinedLoop(
                model: &resources.pipelinedModel,
                tokenizer: resources.tokenizer,
                promptTokens: promptTokens,
                generateCount: generateCount)
        }

        let generateResult = try await measureMedianAsync(name: "InferenceSession.generate(greedy)", iterations: 5, warmup: 1) {
            try await runContainerGenerate(
                container: resources.container,
                promptTokens: promptTokens,
                parameters: GenerationParameters(maxTokens: generateCount, temperature: 0)
            )
        }

        print("")
        print("=== GenerationEvent Pipeline Benchmark: LFM2.5-1.2B ===")
        print("Mode                    tok/s   ms/tok  generated")
        print("--------------------------------------------------")
        print(format(syncResult))
        print(format(pipelinedResult))
        print(format(generateResult))

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
        var resources = try await makeResources(maximumSequenceLength: promptTokens.count)
        defer { resources.release() }

        let rawTokens = try collectGeneratedTokens(
            model: &resources.syncModel,
            promptTokens: promptTokens,
            generateCount: generateCount
        )

        let rawDecode = try measureMedian(name: "raw GPU decode", iterations: 5, warmup: 1) {
            try runSynchronousLoopNoTokenizer(
                model: &resources.syncModel,
                promptTokens: promptTokens,
                generateCount: generateCount
            )
        }

        let tokenDecode = measureTokenizerMedian(
            name: "tokenizer.decode",
            iterations: 10,
            warmup: 2,
            tokenizer: resources.tokenizer,
            tokens: rawTokens
        )

        let syncWithTokenizer = try measureMedian(name: "GPU+tokenizer", iterations: 5, warmup: 1) {
            try runSynchronousLoop(
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
        print(format(rawDecode))
        print(format(tokenDecode))
        print(format(syncWithTokenizer))

        let tokenizerShare = tokenDecode.elapsed / syncWithTokenizer.elapsed * 100.0
        print(String(format: "[Benchmark] tokenizer share of GPU+tokenizer path: %.1f%%", tokenizerShare))

        #expect(rawTokens.count == generateCount)
    }

    @Test("Request-level optimizer comparison", .timeLimit(.minutes(2)))
    func requestLevelOptimizerComparison() async throws {
        let promptTokens = [1, 1, 6, 6423, 708]
        let generateCount = 50

        let standardResult: ThroughputResult = try await {
            var resources = try await makeResources(
                optimizer: StandardOptimizer(),
                maximumSequenceLength: promptTokens.count
            )
            defer { resources.release() }
            return try measureMedian(name: "standard", iterations: 5, warmup: 1) {
                try runSynchronousLoopNoTokenizer(
                    model: &resources.syncModel,
                    promptTokens: promptTokens,
                    generateCount: generateCount
                )
            }
        }()

        let aggressiveResult: ThroughputResult = try await {
            var resources = try await makeResources(
                optimizer: AggressiveOptimizer(),
                maximumSequenceLength: promptTokens.count
            )
            defer { resources.release() }
            return try measureMedian(name: "aggressive", iterations: 5, warmup: 1) {
                try runSynchronousLoopNoTokenizer(
                    model: &resources.syncModel,
                    promptTokens: promptTokens,
                    generateCount: generateCount
                )
            }
        }()

        print("")
        print("=== Request-Level Optimizer Comparison: LFM2.5-1.2B ===")
        print("Optimizer               tok/s   ms/tok  generated")
        print("------------------------------------------------------")
        print(format(standardResult))
        print(format(aggressiveResult))

        let aggressiveGain = ((aggressiveResult.tokensPerSecond / standardResult.tokensPerSecond) - 1.0) * 100.0
        print(String(format: "[Benchmark] aggressive vs standard: %+0.1f%%", aggressiveGain))

        #expect(standardResult.generatedTokenCount == generateCount)
        #expect(aggressiveResult.generatedTokenCount == generateCount)
    }

    @Test("Request-level scaling", .timeLimit(.minutes(3)))
    func requestLevelScaling() async throws {
        let promptTokens = [1, 1, 6, 6423, 708]
        let generateCounts = [50, 128, 256, 512]
        var resources = try await makeResources(
            optimizer: AggressiveOptimizer(),
            maximumSequenceLength: promptTokens.count
        )
        defer { resources.release() }

        print("")
        print("=== Request-Level Scaling: LFM2.5-1.2B ===")
        print("Generated    raw tok/s   raw ms/tok   API tok/s   API ms/tok")
        print("-------------------------------------------------------------")

        var previousRawTokensPerSecond: Double?

        for generateCount in generateCounts {
            let rawResult = try measureMedian(
                name: "raw-\(generateCount)",
                iterations: 3,
                warmup: 1
            ) {
                try runSynchronousLoopNoTokenizer(
                    model: &resources.syncModel,
                    promptTokens: promptTokens,
                    generateCount: generateCount
                )
            }

            let apiResult = try await measureMedianAsync(
                name: "api-\(generateCount)",
                iterations: 3,
                warmup: 1
            ) {
                try await runContainerGenerate(
                    container: resources.container,
                    promptTokens: promptTokens,
                    parameters: GenerationParameters(maxTokens: generateCount, temperature: 0)
                )
            }

            let generated = String(format: "%9d", generateCount)
            let rawTok = String(format: "%10.1f", rawResult.tokensPerSecond)
            let rawMs = String(format: "%12.2f", rawResult.millisecondsPerToken)
            let apiTok = String(format: "%10.1f", apiResult.tokensPerSecond)
            let apiMs = String(format: "%12.2f", apiResult.millisecondsPerToken)
            print("\(generated)\(rawTok)\(rawMs)\(apiTok)\(apiMs)")

            if let previous = previousRawTokensPerSecond, generateCount > 50 {
                let gain = ((rawResult.tokensPerSecond / previous) - 1.0) * 100.0
                print(String(format: "[Benchmark] raw throughput gain vs previous length: %+0.1f%%", gain))
            }
            previousRawTokensPerSecond = rawResult.tokensPerSecond
        }
    }

    @Test("Stream chunk size comparison", .timeLimit(.minutes(3)))
    func streamChunkSizeComparison() async throws {
        let promptTokens = [1, 1, 6, 6423, 708]
        let generateCount = 256
        let chunkSizes = [1, 4, 8, 16]

        print("")
        print("=== Stream Chunk Size Comparison: LFM2.5-1.2B ===")
        print("Chunk size   tok/s   ms/tok   first chunk ms   chunks")
        print("-----------------------------------------------------")

        for chunkSize in chunkSizes {
            let result: StreamResult = try await {
                var resources = try await makeResources(
                    optimizer: AggressiveOptimizer(),
                    maximumSequenceLength: promptTokens.count
                )
                defer { resources.release() }
                return try await measureStreamMedian(
                    iterations: 3,
                    warmup: 1
                ) {
                    try await runContainerGenerateMeasured(
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
        var resources = try await makeResources(
            optimizer: AggressiveOptimizer(),
            maximumSequenceLength: promptTokens.count
        )
        defer { resources.release() }
        let promptState = try resources.container.makePromptSnapshot(from: ExecutablePrompt(tokenIDs: promptTokens)
        )

        let baseline = try await measureStreamMedian(iterations: 3, warmup: 1) {
            try await runContainerGenerateMeasured(
                container: resources.container,
                promptTokens: promptTokens,
                generateCount: generateCount,
                chunkTokenCount: 8,
                temperature: 0
            )
        }

        let reused = try await measureStreamMedian(iterations: 3, warmup: 1) {
            try await runContainerGenerateMeasured(
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
        print(format(stream: baseline, name: "baseline"))
        print(format(stream: reused, name: "prompt-state"))

        let reuseGain = ((reused.tokensPerSecond / baseline.tokensPerSecond) - 1.0) * 100.0
        let ttftGain = ((baseline.firstChunkMilliseconds / reused.firstChunkMilliseconds) - 1.0) * 100.0
        print(String(format: "[Benchmark] prompt-state reuse vs baseline: %+0.1f%%", reuseGain))
        print(String(format: "[Benchmark] TTFT reduction vs baseline: %+0.1f%%", ttftGain))

        #expect(reused.generatedTokenCount == generateCount)
    }

    private func makeResources(
        optimizer: any DispatchOptimizer = AggressiveOptimizer(),
        maximumSequenceLength: Int = 256
    ) async throws -> BenchmarkResources {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw BenchmarkError.noDevice
        }

        let directory = try findModelDirectory()
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let store = try STAFLoader().load(at: URL(fileURLWithPath: Self.stafPath), device: device)

        let config = ModelConfig(
            hiddenSize: 2048, layerCount: 16, intermediateSize: 8192,
            vocabSize: 65536, attentionHeads: 32, kvHeads: 8, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 1000000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: true,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: nil, convLCache: 3,
            partialRotaryFactor: nil, slidingWindow: nil,
            layerTypes: ["conv", "conv", "full_attention", "conv", "conv", "full_attention",
                         "conv", "conv", "full_attention", "conv", "full_attention", "conv",
                         "full_attention", "conv", "full_attention", "conv"]
        )
        let graph = try LFM2(config: config).makeModelGraph()
        let resolved = ParameterResolver().resolve(graph: graph, convention: .lfm2Family)

        let compiler = MetalInferenceCompiler(optimizer: optimizer)
        let decodePlan = try compiler.compile(
            graph: resolved,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            stafWeightStore: store,
            device: device)
        let prefillPlan = try compiler.compilePrefill(
            graph: resolved,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: max(1, maximumSequenceLength)),
            stafWeightStore: store,
            sharedKVCache: decodePlan.buffers.kvCache,
            sharedConvState: decodePlan.buffers.convState,
            sharedConvStateDimension: decodePlan.buffers.convStateDimension,
            sharedConvStateKernelSize: decodePlan.buffers.convStateKernelSize,
            device: device)

        var syncModel = try MetalInferenceModel(plan: decodePlan, device: device)
        syncModel.prefillPlan = prefillPlan

        var pipelinedModel = try MetalInferenceModel(plan: decodePlan, device: device)
        pipelinedModel.prefillPlan = prefillPlan

        var containerModel = try MetalInferenceModel(plan: decodePlan, device: device)
        containerModel.prefillPlan = prefillPlan

        var configuration = ModelConfiguration(name: "lfm2")
        if let eosId = tokenizer.eosTokenId {
            configuration.eosTokenIds.insert(eosId)
        }

        let container = InferenceSession(
            inferenceModel: containerModel,
            tokenizer: tokenizer,
            configuration: configuration,
            vocabularySize: config.vocabSize
        )

        return BenchmarkResources(
            tokenizer: tokenizer,
            syncModel: syncModel,
            pipelinedModel: pipelinedModel,
            container: container
        )
    }

    private func runSynchronousLoop(
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

    private func runSynchronousLoopNoTokenizer(
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

    private func runPipelinedLoop(
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

    private func runContainerGenerate(
        container: InferenceSession,
        promptTokens: [Int],
        parameters: GenerationParameters
    ) async throws -> Int {
        container.resetState()
        let stream = try container.generate(from: ExecutablePrompt(tokenIDs: promptTokens),
            parameters: parameters
        )

        var generated = 0
        for await item in stream {
            if case .completed(let info) = item {
                generated = info.tokenCount
            }
        }
        return generated
    }

    private func runContainerGenerate(
        container: InferenceSession,
        promptState: PromptSnapshot,
        parameters: GenerationParameters
    ) async throws -> Int {
        let stream = try container.generate(
            from: promptState,
            parameters: parameters
        )

        var generated = 0
        for await item in stream {
            if case .completed(let info) = item {
                generated = info.tokenCount
            }
        }
        return generated
    }

    private func runContainerGenerateMeasured(
        container: InferenceSession,
        promptTokens: [Int],
        generateCount: Int,
        chunkTokenCount: Int,
        temperature: Float
    ) async throws -> StreamResult {
        container.resetState()
        let start = CFAbsoluteTimeGetCurrent()
        let stream = try container.generate(from: ExecutablePrompt(tokenIDs: promptTokens),
            parameters: GenerationParameters(
                maxTokens: generateCount,
                streamChunkTokenCount: chunkTokenCount,
                temperature: temperature
            )
        )

        var generated = 0
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
                generated = info.tokenCount
            }
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        return StreamResult(
            generatedTokenCount: generated,
            elapsed: elapsed,
            firstChunkElapsed: firstChunkElapsed ?? elapsed,
            chunkCount: chunkCount
        )
    }

    private func runContainerGenerateMeasured(
        container: InferenceSession,
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

        var generated = 0
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
                generated = info.tokenCount
            }
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        return StreamResult(
            generatedTokenCount: generated,
            elapsed: elapsed,
            firstChunkElapsed: firstChunkElapsed ?? elapsed,
            chunkCount: chunkCount
        )
    }

    private func collectGeneratedTokens(
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

    private func measureMedian(
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

    private func measureMedianAsync(
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

    private func measureTokenizerMedian(
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

    private func measureStreamMedian(
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

    private func format(_ result: ThroughputResult) -> String {
        let paddedName = result.name.padding(toLength: 22, withPad: " ", startingAt: 0)
        return "\(paddedName) \(String(format: "%7.1f", result.tokensPerSecond)) \(String(format: "%8.2f", result.millisecondsPerToken)) \(String(format: "%10d", result.generatedTokenCount))"
    }

    private func format(stream result: StreamResult, name: String) -> String {
        let paddedName = name.padding(toLength: 22, withPad: " ", startingAt: 0)
        return "\(paddedName) \(String(format: "%7.1f", result.tokensPerSecond)) \(String(format: "%8.2f", result.millisecondsPerToken)) \(String(format: "%16.2f", result.firstChunkMilliseconds))"
    }

    private func findModelDirectory() throws -> URL {
        let candidates = [
            "~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Thinking",
            "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct",
            "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B",
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

    private struct BenchmarkResources {
        let tokenizer: any Tokenizer
        var syncModel: MetalInferenceModel
        var pipelinedModel: MetalInferenceModel
        let container: InferenceSession

        mutating func release() {
            syncModel.resetState()
            pipelinedModel.resetState()
            container.resetState()
        }
    }

    private struct ThroughputResult {
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

    private struct StreamResult {
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

    private enum BenchmarkError: Error {
        case noDevice
        case noModelDirectory
        case invalidToken
    }
}
