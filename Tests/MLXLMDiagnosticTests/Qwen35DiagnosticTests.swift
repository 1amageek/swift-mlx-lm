import Foundation
import MLX
import MLXNN
import GGUFParser
import GGUFTokenizer
import Testing
import TestHeartbeat
@testable import MLXLM

@Suite("Qwen3.5 Diagnostics", .tags(.diagnostic), .heartbeat)
struct Qwen35DiagnosticTests {

    private static let repo = "unsloth/Qwen3.5-0.8B-GGUF"
    private static let filename = "Qwen3.5-0.8B-Q4_K_M.gguf"

    private func downloadModel() async throws -> URL {
        let downloader = HuggingFaceDownloader()
        return try await downloader.download(
            repo: Self.repo,
            filename: Self.filename
        )
    }

    @Test("Real Qwen3.5 GGUF prompt keeps assistant suffix and special tokens")
    func promptPipelineFromRealModel() async throws {
        let url = try await downloadModel()
        let file = try GGUFFile.parse(url: url)
        let tokenizer = try GGUFTokenizerFactory.create(from: file)

        let bosToken = tokenizer.bosTokenID.flatMap { tokenizer.tokenToString($0) }
        let eosToken = tokenizer.eosTokenID.flatMap { tokenizer.tokenToString($0) }
        let processor = GGUFUserInputProcessor(
            tokenizer: tokenizer,
            chatTemplate: file.chatTemplate,
            bosToken: bosToken,
            eosToken: eosToken,
            addBosToken: file.addBosToken ?? false
        )

        let lmInput = try await processor.prepare(input: UserInput(prompt: "hi"))
        let promptTokens = tokenIDs(from: lmInput.text.tokens)
        let imStart = try #require(tokenizer.tokenID(for: "<|im_start|>"))
        let imEnd = try #require(tokenizer.tokenID(for: "<|im_end|>"))

        #expect(promptTokens.count > 0)
        #expect(promptTokens.filter { $0 == imStart }.count >= 2)
        #expect(promptTokens.filter { $0 == imEnd }.count >= 1)

        let tail = Array(promptTokens.suffix(24))
        let tailPieces = tail.map { tokenizer.tokenToString($0) ?? "<?>" }
        let joinedTail = tailPieces.joined()
        print("[qwen35][prompt-tail] ids=\(tail)")
        print("[qwen35][prompt-tail] pieces=\(tailPieces)")

        #expect(joinedTail.contains("<|im_start|>assistant"))
        #expect(joinedTail.contains("<|im_end|>"))
    }

    @Test("Standard Qwen3.5 MLXNN path keeps mixed GGUF weights quantized and emits first-step trace")
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
    }

    @Test("Real Qwen3.5 q5_K and q6_K tensors direct-pack match dense fallback")
    func realModelMixedQuantizationRoundTrip() async throws {
        let url = try await downloadModel()
        let file = try GGUFFile.parse(url: url)
        let bridge = GGUFTensorBridge()

        try verifyDirectPackedLinearTensor(
            file: file,
            bridge: bridge,
            tensorName: "blk.0.ssm_out.weight",
            tolerance: 0.25
        )
        try verifyDirectPackedLinearTensor(
            file: file,
            bridge: bridge,
            tensorName: "blk.3.attn_v.weight",
            tolerance: 0.15
        )
    }

    @Test("Compiled Qwen3.5 prefill and decode advance cache state")
    func compiledPrefillAndDecodeState() async throws {
        let url = try await downloadModel()
        let loader = GGUFModelLoader()
        let context = try loader.loadCompiledContext(url: url)

        let lmInput = try await context.processor.prepare(input: UserInput(prompt: "hi"))
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

        let firstStepLogits = nextTokenLogits(from: prefillOutput)
        printTopCandidates(
            label: "qwen35/compiled/prefill",
            logits: firstStepLogits,
            tokenizer: context.tokenizer,
            limit: 10
        )

        let sampler = ArgMaxSampler()
        let sampledTokenArray = sampler.sample(logits: firstStepLogits)
        let sampledTokenID = Int(sampledTokenArray.item(Int32.self))
        let sampledPiece = context.tokenizer.tokenToString(sampledTokenID) ?? "<??>"
        let sampledDecoded = context.tokenizer.decode(tokens: [sampledTokenID])
        print(
            "[qwen35][compiled][sample] id=\(sampledTokenID) piece=\(quoted(sampledPiece)) decoded=\(quoted(sampledDecoded))"
        )

        let decodeInput = LMInput.Text(
            tokens: MLXArray([Int32(sampledTokenID)]).reshaped([1, 1])
        )
        let decodeOutput = context.model.callAsFunction(decodeInput, cache: cache, state: nil)
        let decodeOffset = cache.first?.offset ?? -1
        #expect(decodeOffset == promptTokenCount + 1)

        let secondStepLogits = nextTokenLogits(from: decodeOutput)
        printTopCandidates(
            label: "qwen35/compiled/decode",
            logits: secondStepLogits,
            tokenizer: context.tokenizer,
            limit: 10
        )
    }

    @Test("MLXNN vs Compiled path logit comparison")
    func mlxnnVsCompiledComparison() async throws {
        let url = try await downloadModel()
        let loader = GGUFModelLoader()

        // Load via both paths
        let compiledCtx = try loader.loadCompiledContext(url: url)
        let mlxnnCtx = try loader.loadContext(url: url)

        // Same input
        let lmInput = try await compiledCtx.processor.prepare(input: UserInput(prompt: "hi"))

        // Run compiled path
        let compiledCache = compiledCtx.model.newCache(parameters: nil)
        let compiledOutput = try runPrefill(
            model: compiledCtx.model, input: lmInput,
            cache: compiledCache, windowSize: 1024
        )
        let compiledLogits = nextTokenLogits(from: compiledOutput)

        printTopCandidates(
            label: "compare/compiled",
            logits: compiledLogits,
            tokenizer: compiledCtx.tokenizer,
            limit: 5
        )

        // Run MLXNN path
        let mlxnnCache = mlxnnCtx.model.newCache(parameters: nil)
        let mlxnnOutput = try runPrefill(
            model: mlxnnCtx.model, input: lmInput,
            cache: mlxnnCache, windowSize: 1024
        )
        let mlxnnLogits = nextTokenLogits(from: mlxnnOutput)

        printTopCandidates(
            label: "compare/mlxnn",
            logits: mlxnnLogits,
            tokenizer: mlxnnCtx.tokenizer,
            limit: 5
        )

        // Compare: top-1 should match
        let compiledTop = Int(MLX.argMax(compiledLogits).item(Int32.self))
        let mlxnnTop = Int(MLX.argMax(mlxnnLogits).item(Int32.self))
        print("[compare] compiledTop=\(compiledTop) mlxnnTop=\(mlxnnTop)")
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

    private func verifyDirectPackedLinearTensor(
        file: GGUFFile,
        bridge: GGUFTensorBridge,
        tensorName: String,
        tolerance: Float
    ) throws {
        let tensor = try #require(file.tensors.first { $0.name == tensorName })
        let data = try file.tensorData(for: tensor)

        let denseWeight = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected direct quantized tensor for \(tensorName)")
            return
        }

        let shape = tensor.dimensions.reversed().map { Int($0) }
        let outputDim = shape[0]
        let inputDim = shape[1]

        print(
            "[qwen35][tensor-check] name=\(tensorName) qtype=\(tensor.quantizationType) shape=\(shape) bits=\(bits) groupSize=\(groupSize)"
        )

        let qLinear: Linear
        if groupSize >= 32 {
            qLinear = QuantizedLinear(
                weight: weight,
                bias: nil,
                scales: scales,
                biases: biases,
                groupSize: groupSize,
                bits: bits,
                mode: .affine
            )
        } else {
            qLinear = DirectQuantizedLinear(
                weight: weight,
                bias: nil,
                scales: scales,
                biases: biases,
                groupSize: groupSize,
                bits: bits,
                mode: .affine
            )
        }
        let denseLinear = Linear(weight: denseWeight, bias: nil)

        MLXRandom.seed(7)
        let input = MLXRandom.normal([1, inputDim]).asType(.float16)
        let qOutput = qLinear(input)
        let denseOutput = denseLinear(input)
        eval(qOutput, denseOutput)

        let qValues = qOutput.asType(.float32)
        let dValues = denseOutput.asType(.float32)
        let diff = MLX.abs(qValues - dValues)
        let maxMagnitude = MLX.abs(dValues).max()
        eval(diff, maxMagnitude)

        let maxDiff: Float = diff.max().item()
        let magnitude: Float = maxMagnitude.item()
        let effectiveTolerance = max(tolerance, magnitude * 0.002)

        print(
            "[qwen35][tensor-check] name=\(tensorName) maxDiff=\(maxDiff) tolerance=\(effectiveTolerance) magnitude=\(magnitude) outputDim=\(outputDim)"
        )
        #expect(
            maxDiff < effectiveTolerance,
            Comment(rawValue: "\(tensorName) direct pack diverged: diff=\(maxDiff) tolerance=\(effectiveTolerance)")
        )
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
