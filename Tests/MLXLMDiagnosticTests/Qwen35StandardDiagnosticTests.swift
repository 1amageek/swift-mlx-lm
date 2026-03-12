import Foundation
import GGUFParser
import GGUFTokenizer
import MLX
import MLXNN
import Testing
import TestHeartbeat
@testable import MLXLM

@Suite("Qwen3.5 Standard Diagnostics", .tags(.diagnostic), .heartbeat)
struct Qwen35StandardDiagnosticTests {

    private static let repo = "unsloth/Qwen3.5-0.8B-GGUF"
    private static let filename = "Qwen3.5-0.8B-Q4_K_M.gguf"
    private func downloadModel() async throws -> URL {
        let downloader = HuggingFaceDownloader()
        return try await downloader.download(
            repo: Self.repo,
            filename: Self.filename
        )
    }

    @Test("Standard MLXNN path keeps mixed GGUF weights quantized and emits first-step trace")
    func standardPathMixedQuantizationTrace() async throws {
        let url = try await downloadModel()
        let loader = GGUFModelLoader()
        let context = try loader.loadContext(url: url)

        if let report = context.loadReport {
            print("[qwen35][mlxnn][load-report]")
            print(report.summary)
        }

        let model = try #require(context.model as? Qwen35Model)
        let delta0 = try #require(model.model.layers[0].deltaNet)
        let attn3 = try #require(model.model.layers[3].fullAttn)

        print("[qwen35][mlxnn][module] embed=\(type(of: model.model.embedTokens))")
        print("[qwen35][mlxnn][module] delta0.inProjQKV=\(type(of: delta0.inProjQKV))")
        print("[qwen35][mlxnn][module] delta0.outProj=\(type(of: delta0.outProj))")
        print("[qwen35][mlxnn][module] attn3.wv=\(type(of: attn3.wv))")

        #expect(model.model.embedTokens is any Quantized)
        #expect(delta0.inProjQKV is any Quantized)
        #expect(delta0.outProj is any Quantized)
        #expect(attn3.wv is any Quantized)

        let lmInput = try await context.processor.prepare(input: UserInput(prompt: "hi"))
        let promptTokens = tokenIDs(from: lmInput.text.tokens)
        let promptTail = Array(promptTokens.suffix(24))
        let promptTailPieces = promptTail.map { context.tokenizer.tokenToString($0) ?? "<?>" }
        let promptTailJoined = promptTailPieces.joined()
        print("[qwen35][mlxnn][prompt-tail] ids=\(promptTail)")
        print("[qwen35][mlxnn][prompt-tail] pieces=\(promptTailPieces)")
        print("[qwen35][mlxnn][prompt-tail] joined=\(quoted(promptTailJoined))")
        let promptTokenCount = lmInput.text.tokens.dim(1)
        let cache = context.model.newCache(parameters: nil)
        let prefillOutput = try runPrefill(
            model: context.model,
            input: lmInput,
            cache: cache,
            windowSize: 1024
        )

        let prefillOffset = cache.first?.offset ?? -1
        #expect(prefillOffset == promptTokenCount)

        let logits = nextTokenLogits(from: prefillOutput)
        printTopCandidates(
            label: "qwen35/mlxnn/prefill",
            logits: logits,
            tokenizer: context.tokenizer,
            limit: 10
        )

        let sampler = ArgMaxSampler()
        let sampledTokenArray = sampler.sample(logits: logits)
        let sampledTokenID = Int(sampledTokenArray.item(Int32.self))
        let sampledPiece = context.tokenizer.tokenToString(sampledTokenID) ?? "<??>"
        let sampledDecoded = context.tokenizer.decode(tokens: [sampledTokenID])
        print(
            "[qwen35][mlxnn][sample] id=\(sampledTokenID) piece=\(quoted(sampledPiece)) decoded=\(quoted(sampledDecoded))"
        )

        let greedyTokens = greedyDecodeTrace(
            model: context.model,
            cache: cache,
            promptTokenCount: promptTokenCount,
            startTokenID: sampledTokenID,
            steps: 8
        )
        let greedyText = context.tokenizer.decode(tokens: greedyTokens)
        print("[qwen35][mlxnn][greedy-trace] tokens=\(greedyTokens)")
        print("[qwen35][mlxnn][greedy-trace] text=\(quoted(greedyText))")
        assertNotDegenerateGreedyTrace(
            label: "qwen35/mlxnn/greedy",
            tokens: greedyTokens,
            text: greedyText
        )
    }

    @Test("Direct-quantized Qwen3.5 stays aligned with dense fallback for first decode steps")
    func quantizedVsDenseFirstStepAlignment() async throws {
        let url = try await downloadModel()
        let loader = GGUFModelLoader()
        let quantizedContext = try loader.loadContext(url: url)
        let denseContext = try loadDenseContext(url: url)

        let input = UserInput(prompt: "hi")
        let quantizedInput = try await quantizedContext.processor.prepare(input: input)
        let denseInput = try await denseContext.processor.prepare(input: input)

        let quantizedCache = quantizedContext.model.newCache(parameters: nil)
        let denseCache = denseContext.model.newCache(parameters: nil)

        let quantizedOutput = try runPrefill(
            model: quantizedContext.model,
            input: quantizedInput,
            cache: quantizedCache,
            windowSize: 1024
        )
        let denseOutput = try runPrefill(
            model: denseContext.model,
            input: denseInput,
            cache: denseCache,
            windowSize: 1024
        )

        let quantizedLogits = nextTokenLogits(from: quantizedOutput)
        let denseLogits = nextTokenLogits(from: denseOutput)
        let quantizedTop = topTokenIDs(logits: quantizedLogits, limit: 10)
        let denseTop = topTokenIDs(logits: denseLogits, limit: 10)
        let commonTop = Set(quantizedTop).intersection(denseTop)

        print("[qwen35][quant-vs-dense] quantTop=\(quantizedTop)")
        print("[qwen35][quant-vs-dense] denseTop=\(denseTop)")
        print("[qwen35][quant-vs-dense] commonTop=\(Array(commonTop).sorted()) count=\(commonTop.count)")

        let quantizedGreedy = greedyDecodeTrace(
            model: quantizedContext.model,
            cache: quantizedCache,
            promptTokenCount: quantizedInput.text.tokens.dim(1),
            startTokenID: quantizedTop[0],
            steps: 4
        )
        let denseGreedy = greedyDecodeTrace(
            model: denseContext.model,
            cache: denseCache,
            promptTokenCount: denseInput.text.tokens.dim(1),
            startTokenID: denseTop[0],
            steps: 4
        )

        print("[qwen35][quant-vs-dense] quantGreedy=\(quantizedGreedy)")
        print("[qwen35][quant-vs-dense] denseGreedy=\(denseGreedy)")
        print("[qwen35][quant-vs-dense] quantText=\(quoted(quantizedContext.tokenizer.decode(tokens: quantizedGreedy)))")
        print("[qwen35][quant-vs-dense] denseText=\(quoted(denseContext.tokenizer.decode(tokens: denseGreedy)))")

        assertNotDegenerateGreedyTrace(
            label: "qwen35/dense/greedy",
            tokens: denseGreedy,
            text: denseContext.tokenizer.decode(tokens: denseGreedy)
        )
        #expect(quantizedTop[0] == denseTop[0])
        #expect(commonTop.count >= 5)
        #expect(quantizedGreedy == denseGreedy)
    }

    private func runPrefill(
        model: any LanguageModel,
        input: LMInput,
        cache: [KVCache],
        windowSize: Int
    ) throws -> LMOutput {
        var currentInput = input

        while true {
            let result = try model.prepare(currentInput, cache: cache, windowSize: windowSize)
            switch result {
            case .tokens(let remaining):
                currentInput = LMInput(
                    text: remaining,
                    image: currentInput.image,
                    video: currentInput.video
                )
            case .logits(let output):
                return output
            }
        }
    }

    private func nextTokenLogits(from output: LMOutput) -> MLXArray {
        output.logits[0..., (-1)..., 0...].squeezed(axis: 0)
    }

    private func tokenIDs(from tokens: MLXArray) -> [Int] {
        tokens.flattened().asArray(Int32.self).map(Int.init)
    }

    private func topTokenIDs(logits: MLXArray, limit: Int) -> [Int] {
        let flatLogits = logits.flattened().asType(.float32)
        let sorted = MLX.argSort(flatLogits, axis: -1)
        eval(sorted)
        return Array(sorted.asArray(Int32.self).suffix(limit).reversed()).map(Int.init)
    }

    private func greedyDecodeTrace(
        model: any LanguageModel,
        cache: [KVCache],
        promptTokenCount: Int,
        startTokenID: Int,
        steps: Int
    ) -> [Int] {
        let sampler = ArgMaxSampler()
        var generated: [Int] = []
        var tokenID = startTokenID

        for step in 0..<steps {
            generated.append(tokenID)
            let decodeInput = LMInput.Text(
                tokens: MLXArray([Int32(tokenID)]).reshaped([1, 1])
            )
            let decodeOutput = model.callAsFunction(decodeInput, cache: cache, state: nil)
            let offset = cache.first?.offset ?? -1
            #expect(offset == promptTokenCount + step + 1)

            let nextLogits = nextTokenLogits(from: decodeOutput)
            tokenID = Int(sampler.sample(logits: nextLogits).item(Int32.self))
        }

        return generated
    }

    private func loadDenseContext(url: URL) throws -> ModelContext {
        try GGUFModelLoader().loadContext(url: url, quantization: .disabled)
    }

    private func assertNotDegenerateGreedyTrace(
        label: String,
        tokens: [Int],
        text: String
    ) {
        let isDegenerate = hasRepeatingCycle(tokens) || Set(tokens).count <= 2
        #expect(
            !isDegenerate,
            Comment(rawValue: "\(label) looks stuck in a loop: tokens=\(tokens) text=\(quoted(text))")
        )
    }

    private func hasRepeatingCycle(_ tokens: [Int]) -> Bool {
        guard tokens.count >= 6 else { return false }
        for cycleLength in 1...(tokens.count / 2) {
            let fullCycles = tokens.count / cycleLength
            guard fullCycles >= 2 else { continue }
            let cycle = Array(tokens.prefix(cycleLength))
            var matches = true
            for index in tokens.indices {
                if tokens[index] != cycle[index % cycleLength] {
                    matches = false
                    break
                }
            }
            if matches {
                return true
            }
        }
        return false
    }

    private func printTopCandidates(
        label: String,
        logits: MLXArray,
        tokenizer: any Tokenizer,
        limit: Int
    ) {
        let flatLogits = logits.flattened().asType(.float32)
        let sorted = MLX.argSort(flatLogits, axis: -1)
        eval(flatLogits, sorted)

        let values = flatLogits.asArray(Float.self)
        let topIDs = Array(sorted.asArray(Int32.self).suffix(limit).reversed()).map(Int.init)

        print("[\(label)][topk] count=\(topIDs.count)")
        for (rank, tokenID) in topIDs.enumerated() {
            let piece = tokenizer.tokenToString(tokenID) ?? "<??>"
            let decoded = tokenizer.decode(tokens: [tokenID])
            let value = values[tokenID]
            print(
                "[\(label)][topk] rank=\(rank + 1) id=\(tokenID) logit=\(value) piece=\(quoted(piece)) decoded=\(quoted(decoded))"
            )
        }
    }

    private func quoted(_ string: String) -> String {
        var escaped = "\""
        for scalar in string.unicodeScalars {
            switch scalar.value {
            case 0x0A:
                escaped += "\\n"
            case 0x0D:
                escaped += "\\r"
            case 0x09:
                escaped += "\\t"
            case 0x22:
                escaped += "\\\""
            case 0x5C:
                escaped += "\\\\"
            default:
                escaped.unicodeScalars.append(scalar)
            }
        }
        escaped += "\""
        return escaped
    }
}
