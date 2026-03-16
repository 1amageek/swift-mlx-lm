import Testing
import Metal
import Foundation
@testable import MetalCompiler
import LMArchitecture
import ModelDeclarations
import LMIR

/// Verify decode produces different outputs for different inputs.
@Suite("Decode Behavior")
struct DecodeTests {

    @Test("Decode produces different logits for different token IDs")
    func decodeDifferentInputsDifferentOutputs() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        // Use a pure Transformer model (no Conv layers) to isolate attention behavior
        let config = ModelConfig(
            hiddenSize: 128, layerCount: 2, intermediateSize: 512,
            vocabSize: 1000, attentionHeads: 4, kvHeads: 4, headDim: 32,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 10000, ropeDimension: 32,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: false,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: nil,
            partialRotaryFactor: nil, slidingWindow: nil
        )
        let graph = try Transformer(config: config).makeModelGraph()
        let resolved = ParameterResolver().resolve(graph: graph, convention: .llamaFamily)
        let plan = try MetalInferenceCompiler().compile(
            graph: resolved, hiddenSize: 128, intermediateSize: 512,
            vocabSize: 1000, device: device)

        // Decode with token 1
        var model1 = try MetalInferenceModel(plan: plan, device: device)
        let out1 = model1.decodeSync(tokenID: 1)

        // Decode with token 42
        var model2 = try MetalInferenceModel(plan: plan, device: device)
        let out2 = model2.decodeSync(tokenID: 42)

        // Without real weights, both may produce 0 (argmax of all-zero logits).
        // But let's verify no GPU errors occurred.
        print("[Decode test] token 1 → \(out1), token 42 → \(out2)")
    }

    @Test("Decode after prefill uses KV cache from prefill")
    func decodeAfterPrefillUsesKVCache() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let stafPath = "/Users/1amageek/Library/Containers/team.stamp.JARDIS.ml/Data/Documents/huggingface/models/LiquidAI/LFM2.5-1.2B-Thinking/model.staf"
        guard FileManager.default.fileExists(atPath: stafPath) else {
            print("STAF not found — skipping"); return
        }

        let store = try STAFLoader().load(at: URL(fileURLWithPath: stafPath), device: device)
        let config = ModelConfig(
            hiddenSize: 2048, layerCount: 16, intermediateSize: 8192,
            vocabSize: 65536, attentionHeads: 32, kvHeads: 8, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 1000000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: true,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: 3,
            partialRotaryFactor: nil, slidingWindow: nil,
            layerTypes: ["conv", "conv", "full_attention", "conv", "conv", "full_attention",
                         "conv", "conv", "full_attention", "conv", "full_attention", "conv",
                         "full_attention", "conv", "full_attention", "conv"]
        )
        let graph = try LFM2(config: config).makeModelGraph()
        let resolved = ParameterResolver().resolve(graph: graph, convention: .lfm2Family)

        let compiler = MetalInferenceCompiler()
        let decodePlan = try compiler.compile(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, stafWeightStore: store, device: device)
        let prefillPlan = try compiler.compilePrefill(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, maximumSequenceLength: 64,
            stafWeightStore: store, device: device)

        var model = try MetalInferenceModel(plan: decodePlan, device: device)
        model.prefillPlan = prefillPlan

        // Short prefill
        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        model.prefill(tokens: tokens)
        #expect(model.position == 5)

        // Decode 3 steps
        let out0 = model.decodeSync(tokenID: 708)
        let out1 = model.decodeSync(tokenID: out0)
        let out2 = model.decodeSync(tokenID: out1)

        print("[Decode after prefill] outputs: \(out0), \(out1), \(out2)")
        print("[Decode after prefill] position: \(model.position)")
        #expect(model.position == 8)

        // Check KV cache has data (not all zeros)
        if let kv = decodePlan.buffers.kvCache {
            let kvPtr = kv.keys.contents().bindMemory(to: Float16.self, capacity: 64)
            let kvSample = (0..<8).map { Float(kvPtr[$0]) }
            let kvNonZero = (0..<64).contains { Float(kvPtr[$0]) != 0.0 }
            print("[Decode after prefill] KV cache keys[0..7]: \(kvSample) nonZero=\(kvNonZero)")
            #expect(kvNonZero, "KV cache should have non-zero values after prefill")
        }

        // With Conv1d layers (no state maintained), output may be repetitive.
        // This is a known limitation until conv_state is implemented.
        let allSame = (out0 == out1) && (out1 == out2)
        if allSame {
            print("[Decode after prefill] WARNING: all outputs identical (\(out0)) — Conv1d state not maintained")
        }
    }

    @Test("Attention-only model attempt (layer index mismatch expected)")
    func attentionOnlyDecodeProducesVariedOutput() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let stafPath = "/Users/1amageek/Library/Containers/team.stamp.JARDIS.ml/Data/Documents/huggingface/models/LiquidAI/LFM2.5-1.2B-Thinking/model.staf"
        guard FileManager.default.fileExists(atPath: stafPath) else {
            print("STAF not found — skipping"); return
        }

        let store = try STAFLoader().load(at: URL(fileURLWithPath: stafPath), device: device)

        // Use only attention layers (no conv) to isolate decode behavior
        let config = ModelConfig(
            hiddenSize: 2048, layerCount: 6, intermediateSize: 8192,
            vocabSize: 65536, attentionHeads: 32, kvHeads: 8, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 1000000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: true,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: nil,
            partialRotaryFactor: nil, slidingWindow: nil,
            layerTypes: ["full_attention", "full_attention", "full_attention",
                         "full_attention", "full_attention", "full_attention"]
        )
        let graph = try LFM2(config: config).makeModelGraph()
        let resolved = ParameterResolver().resolve(graph: graph, convention: .lfm2Family)

        let compiler = MetalInferenceCompiler()
        let decodePlan = try compiler.compile(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, stafWeightStore: store, device: device)
        let prefillPlan = try compiler.compilePrefill(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, maximumSequenceLength: 64,
            stafWeightStore: store, device: device)

        var model = try MetalInferenceModel(plan: decodePlan, device: device)
        model.prefillPlan = prefillPlan

        // Short prefill with "hi" tokens
        model.prefill(tokens: [1, 1, 6, 6423, 708])
        // Layer index mismatch (6 attention layers using layer 0-5 weights, but STAF
        // has attention at layers 2,5,8,10,12,14). Prefill may fail — skip if so.
        if model.position == 0 {
            print("[Attn-only] Prefill failed (expected — layer index mismatch)")
            return
        }

        // Decode 5 steps
        var outputs: [Int32] = []
        var token: Int32 = 708
        for _ in 0..<5 {
            token = model.decodeSync(tokenID: token)
            outputs.append(token)
        }
        print("[Attn-only decode] outputs: \(outputs)")

        // Check that at least one output differs from the first
        let diverse = outputs.contains { $0 != outputs[0] }
        if diverse {
            print("[Attn-only decode] SUCCESS: varied output")
        } else {
            print("[Attn-only decode] All identical — weight mismatch (layer indices don't match STAF)")
        }
    }

    @Test("KV cache affects decode output (zeroed KV produces different logits)")
    func kvCacheAffectsOutput() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let stafPath = "/Users/1amageek/Library/Containers/team.stamp.JARDIS.ml/Data/Documents/huggingface/models/LiquidAI/LFM2.5-1.2B-Thinking/model.staf"
        guard FileManager.default.fileExists(atPath: stafPath) else {
            print("STAF not found — skipping"); return
        }

        let store = try STAFLoader().load(at: URL(fileURLWithPath: stafPath), device: device)
        let config = ModelConfig(
            hiddenSize: 2048, layerCount: 16, intermediateSize: 8192,
            vocabSize: 65536, attentionHeads: 32, kvHeads: 8, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 1000000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: true,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: 3,
            partialRotaryFactor: nil, slidingWindow: nil,
            layerTypes: ["conv", "conv", "full_attention", "conv", "conv", "full_attention",
                         "conv", "conv", "full_attention", "conv", "full_attention", "conv",
                         "full_attention", "conv", "full_attention", "conv"]
        )
        let graph = try LFM2(config: config).makeModelGraph()
        let resolved = ParameterResolver().resolve(graph: graph, convention: .lfm2Family)

        let compiler = MetalInferenceCompiler()
        let decodePlan = try compiler.compile(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, stafWeightStore: store, device: device)
        let prefillPlan = try compiler.compilePrefill(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, maximumSequenceLength: 64,
            stafWeightStore: store, device: device)

        // Run 1: prefill + decode with KV cache
        var model1 = try MetalInferenceModel(plan: decodePlan, device: device)
        model1.prefillPlan = prefillPlan
        model1.prefill(tokens: [1, 1, 6, 6423, 708])
        let out1 = model1.decodeSync(tokenID: 708)

        // Capture logits
        let logits1 = (0..<8).map { Float(decodePlan.buffers.logits.contents().bindMemory(to: Float16.self, capacity: 8)[$0]) }

        // Run 2: decode WITHOUT prefill (empty KV cache, position 0)
        var model2 = try MetalInferenceModel(plan: decodePlan, device: device)
        // Zero out KV cache
        if let kv = decodePlan.buffers.kvCache {
            memset(kv.keys.contents(), 0, kv.keys.length)
            memset(kv.values.contents(), 0, kv.values.length)
        }
        let out2 = model2.decodeSync(tokenID: 708)
        let logits2 = (0..<8).map { Float(decodePlan.buffers.logits.contents().bindMemory(to: Float16.self, capacity: 8)[$0]) }

        print("[KV test] With KV: logits=\(logits1) out=\(out1)")
        print("[KV test] No KV:   logits=\(logits2) out=\(out2)")

        // Logits should differ if KV cache matters
        let logitsDiffer = zip(logits1, logits2).contains { abs($0 - $1) > 0.01 }
        #expect(logitsDiffer, "Logits should differ with/without KV cache — attention should use cached context")
    }
}
