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

    @Test("Decode after prefill uses KV cache from prefill")
    func decodeAfterPrefillUsesKVCache() throws {
        guard let resources = try RealModelTestSupport.loadOrSkip(skipMessage: "STAF not found — skipping") else {
            return
        }
        defer { resources.release() }

        let device = resources.device
        let store = resources.store
        let config = ModelConfig(
            hiddenSize: 2048, layerCount: 16, intermediateSize: 8192,
            vocabSize: 65536, attentionHeads: 32, kvHeads: 8, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 1000000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: true,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: 3, convLCache: 3,
            partialRotaryFactor: nil, slidingWindow: nil,
            layerTypes: ["conv", "conv", "full_attention", "conv", "conv", "full_attention",
                         "conv", "conv", "full_attention", "conv", "full_attention", "conv",
                         "full_attention", "conv", "full_attention", "conv"]
        )
        let graph = try ModelGraph(LFM2(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .lfm2Family)

        let compiler = MetalInferenceCompiler()
        let decodePlan = try compiler.compile(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, stafWeightStore: store, device: device)
        let prefillPlan = try compiler.compilePrefill(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store, device: device)

        var model = try MetalInferenceModel(plan: decodePlan, device: device)
        model.prefillPlan = prefillPlan

        // Short prefill — use returned first token
        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        let firstToken = model.prefill(tokens: tokens)
        #expect(model.position == 5)
        print("[Decode after prefill] first predicted token from prefill: \(firstToken)")

        // Decode 3 steps — first input is the prefill's predicted token
        let out0 = model.decodeSync(tokenID: firstToken)
        let out1 = model.decodeSync(tokenID: out0)
        let out2 = model.decodeSync(tokenID: out1)

        print("[Decode after prefill] outputs: \(out0), \(out1), \(out2)")
        print("[Decode after prefill] position: \(model.position)")
        #expect(model.position == 8)

        // Check KV cache has data (not all zeros)
        if let kv = decodePlan.buffers.kvCache {
            let kvValues = try readBuffer(kv.keys, precision: decodePlan.buffers.bufferPrecision, count: 64)
            let kvSample = Array(kvValues.prefix(8))
            let kvNonZero = kvValues.contains { $0 != 0.0 }
            print("[Decode after prefill] KV cache keys[0..7]: \(kvSample) nonZero=\(kvNonZero)")
            #expect(kvNonZero, "KV cache should have non-zero values after prefill")
        }

    }

    @Test("KV cache affects decode output (zeroed KV produces different logits)")
    func kvCacheAffectsOutput() throws {
        guard let resources = try RealModelTestSupport.loadOrSkip(skipMessage: "STAF not found — skipping") else {
            return
        }
        defer { resources.release() }

        let device = resources.device
        let store = resources.store
        let config = ModelConfig(
            hiddenSize: 2048, layerCount: 16, intermediateSize: 8192,
            vocabSize: 65536, attentionHeads: 32, kvHeads: 8, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 1000000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: true,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: 3, convLCache: 3,
            partialRotaryFactor: nil, slidingWindow: nil,
            layerTypes: ["conv", "conv", "full_attention", "conv", "conv", "full_attention",
                         "conv", "conv", "full_attention", "conv", "full_attention", "conv",
                         "full_attention", "conv", "full_attention", "conv"]
        )
        let graph = try ModelGraph(LFM2(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .lfm2Family)

        let compiler = MetalInferenceCompiler()
        let decodePlan = try compiler.compile(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, stafWeightStore: store, device: device)
        let prefillPlan = try compiler.compilePrefill(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store, device: device)

        // Run 1: prefill + decode with KV cache
        var model1 = try MetalInferenceModel(plan: decodePlan, device: device)
        model1.prefillPlan = prefillPlan
        model1.prefill(tokens: [1, 1, 6, 6423, 708])
        let out1 = model1.decodeSync(tokenID: 708)

        // Capture logits
        let logits1 = try readBuffer(decodePlan.buffers.logits, precision: decodePlan.buffers.bufferPrecision, count: 8)

        // Run 2: decode WITHOUT prefill (empty KV cache, position 0)
        var model2 = try MetalInferenceModel(plan: decodePlan, device: device)
        // Zero out KV cache
        if let kv = decodePlan.buffers.kvCache {
            try zeroBuffer(kv.keys)
            try zeroBuffer(kv.values)
        }
        let out2 = model2.decodeSync(tokenID: 708)
        let logits2 = try readBuffer(decodePlan.buffers.logits, precision: decodePlan.buffers.bufferPrecision, count: 8)

        print("[KV test] With KV: logits=\(logits1) out=\(out1)")
        print("[KV test] No KV:   logits=\(logits2) out=\(out2)")

        // Logits should differ if KV cache matters
        let logitsDiffer = zip(logits1, logits2).contains { abs($0 - $1) > 0.01 }
        #expect(logitsDiffer, "Logits should differ with/without KV cache — attention should use cached context")
    }

    @Test("Different input tokens produce different hidden states and logits")
    func differentTokensDifferentHiddenStates() throws {
        guard let resources = try RealModelTestSupport.loadOrSkip(skipMessage: "STAF not found — skipping") else {
            return
        }
        defer { resources.release() }

        let device = resources.device
        let store = resources.store
        let config = ModelConfig(
            hiddenSize: 2048, layerCount: 16, intermediateSize: 8192,
            vocabSize: 65536, attentionHeads: 32, kvHeads: 8, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 1000000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: true,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: 3, convLCache: 3,
            partialRotaryFactor: nil, slidingWindow: nil,
            layerTypes: ["conv", "conv", "full_attention", "conv", "conv", "full_attention",
                         "conv", "conv", "full_attention", "conv", "full_attention", "conv",
                         "full_attention", "conv", "full_attention", "conv"]
        )
        let graph = try ModelGraph(LFM2(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .lfm2Family)

        let compiler = MetalInferenceCompiler()
        let plan = try compiler.compile(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, stafWeightStore: store, device: device)

        // Decode token 708 (newline)
        var model1 = try MetalInferenceModel(plan: plan, device: device)
        let out1 = model1.decodeSync(tokenID: 708)
        let hidden1 = try readBuffer(plan.buffers.hidden, precision: plan.buffers.bufferPrecision, count: 16)
        let logits1 = try readBuffer(plan.buffers.logits, precision: plan.buffers.bufferPrecision)
        let logitsDim = logits1.count
        var maxVal1: Float = -.infinity; var maxIdx1 = 0
        for i in 0..<logitsDim { let v = logits1[i]; if v > maxVal1 { maxVal1 = v; maxIdx1 = i } }

        // Decode token 6423 ("hi" in this tokenizer)
        var model2 = try MetalInferenceModel(plan: plan, device: device)
        let out2 = model2.decodeSync(tokenID: 6423)
        let hidden2 = try readBuffer(plan.buffers.hidden, precision: plan.buffers.bufferPrecision, count: 16)
        let logits2 = try readBuffer(plan.buffers.logits, precision: plan.buffers.bufferPrecision)
        var maxVal2: Float = -.infinity; var maxIdx2 = 0
        for i in 0..<logitsDim { let v = logits2[i]; if v > maxVal2 { maxVal2 = v; maxIdx2 = i } }

        print("[Token test] token=708: hidden[0..3]=\(Array(hidden1[0..<4])) logits max=\(maxVal1)@\(maxIdx1) out=\(out1)")
        print("[Token test] token=6423: hidden[0..3]=\(Array(hidden2[0..<4])) logits max=\(maxVal2)@\(maxIdx2) out=\(out2)")

        // Hidden states MUST differ for different input tokens
        let hiddenDiffer = zip(hidden1, hidden2).contains { abs($0 - $1) > 1e-3 }
        #expect(hiddenDiffer, "Hidden states must differ for different input tokens — embedding or forward pass is broken")

        // Logits should differ
        let logitsDiffer = (maxIdx1 != maxIdx2) || (abs(maxVal1 - maxVal2) > 0.1)
        #expect(logitsDiffer, "Logits should differ for different tokens — got max@\(maxIdx1) and max@\(maxIdx2)")

        // Decode 5 steps from prefill — dump top-5 logits per step
        let prefillPlan = try compiler.compilePrefill(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store, device: device)
        var model3 = try MetalInferenceModel(plan: plan, device: device)
        model3.prefillPlan = prefillPlan
        model3.prefill(tokens: [1, 1, 6, 6423, 708])

        // Dump prefill logits top-5 from the PREFILL plan's logits buffer (not decode plan)
        do {
            let pLogits = try readBuffer(prefillPlan.buffers.logits, precision: prefillPlan.buffers.bufferPrecision)
            var top5: [(Int, Float)] = []
            for i in 0..<pLogits.count {
                let v = pLogits[i]
                if top5.count < 5 { top5.append((i, v)); top5.sort { $0.1 > $1.1 } }
                else if v > top5.last!.1 { top5[4] = (i, v); top5.sort { $0.1 > $1.1 } }
            }
            let firstToken = model3.buffers.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
            print("[Token test] prefill first token (from tokenOut): \(firstToken)")
            print("[Token test] prefill logits top5 (PREFILL buffer): \(top5.map { "id=\($0.0) val=\(String(format: "%.2f", $0.1))" }.joined(separator: ", "))")
        }

        var outputs: [Int32] = []
        var token: Int32 = 708
        for step in 0..<5 {
            let hiddenPre = try readBuffer(plan.buffers.hidden, precision: plan.buffers.bufferPrecision, count: 4)
            token = model3.decodeSync(tokenID: token)
            outputs.append(token)

            // Top-5 logits
            let logits = try readBuffer(plan.buffers.logits, precision: plan.buffers.bufferPrecision)
            var top5: [(Int, Float)] = []
            for i in 0..<logitsDim {
                let v = logits[i]
                if top5.count < 5 { top5.append((i, v)); top5.sort { $0.1 > $1.1 } }
                else if v > top5.last!.1 { top5[4] = (i, v); top5.sort { $0.1 > $1.1 } }
            }
            let hiddenPost = try readBuffer(plan.buffers.hidden, precision: plan.buffers.bufferPrecision, count: 4)
            print("[Token test] step \(step): in=\(token == outputs.last! ? token : outputs[step]) → out=\(token) hidden_pre=\(hiddenPre) hidden_post=\(hiddenPost)")
            print("[Token test] step \(step) top5: \(top5.map { "id=\($0.0) val=\(String(format: "%.2f", $0.1))" }.joined(separator: ", "))")
        }

        print("[Token test] 5-step outputs: \(outputs)")

        // Also test with a prompt NOT ending in 708
        var model4 = try MetalInferenceModel(plan: plan, device: device)
        model4.prefillPlan = prefillPlan
        let firstToken2 = model4.prefill(tokens: [1, 1, 6, 6423])  // ends with "hi", not newline
        print("[Token test] prompt=[1,1,6,6423] → first token: \(firstToken2)")
        var outputs2: [Int32] = []
        var token2 = firstToken2
        for step in 0..<5 {
            token2 = model4.decodeSync(tokenID: token2)
            outputs2.append(token2)
            // Top-3 logits
            let logits = try readBuffer(plan.buffers.logits, precision: plan.buffers.bufferPrecision)
            var top3: [(Int, Float)] = []
            for i in 0..<logitsDim {
                let v = logits[i]
                if top3.count < 3 { top3.append((i, v)); top3.sort { $0.1 > $1.1 } }
                else if v > top3.last!.1 { top3[2] = (i, v); top3.sort { $0.1 > $1.1 } }
            }
            print("[Token test B] step \(step): in=\(step == 0 ? firstToken2 : outputs2[step-1]) → out=\(token2) top3: \(top3.map { "id=\($0.0) val=\(String(format: "%.2f", $0.1))" }.joined(separator: ", "))")
        }
        print("[Token test B] 5-step outputs: \(outputs2)")

        // At least one of the two test paths should produce varied output
        let uniqueTokens = Set(outputs)
        let uniqueTokens2 = Set(outputs2)
        let anyVaried = uniqueTokens.count > 1 || uniqueTokens2.count > 1
        #expect(anyVaried, "Decode must produce varied output — path A: \(uniqueTokens), path B: \(uniqueTokens2)")
    }

    private func readBuffer(_ buffer: MTLBuffer, precision: BufferPrecision, count: Int? = nil) throws -> [Float] {
        let readableBuffer = try makeReadableBuffer(buffer)
        let readableCount = readableBuffer.length / precision.byteSize
        let resolvedCount = min(count ?? readableCount, readableCount)

        switch precision {
        case .float16:
            let pointer = readableBuffer.contents().bindMemory(to: Float16.self, capacity: resolvedCount)
            return (0..<resolvedCount).map { Float(pointer[$0]) }
        case .bfloat16:
            let pointer = readableBuffer.contents().bindMemory(to: BFloat16.self, capacity: resolvedCount)
            return (0..<resolvedCount).map { Float(pointer[$0]) }
        case .float32:
            let pointer = readableBuffer.contents().bindMemory(to: Float32.self, capacity: resolvedCount)
            return (0..<resolvedCount).map { pointer[$0] }
        }
    }

    private func zeroBuffer(_ buffer: MTLBuffer) throws {
        if buffer.storageMode != .private {
            memset(buffer.contents(), 0, buffer.length)
            return
        }

        guard let queue = buffer.device.makeCommandQueue(),
              let commandBuffer = queue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw DecodeTestError.unavailableCommandSubmission
        }

        blit.fill(buffer: buffer, range: 0..<buffer.length, value: 0)
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    private func makeReadableBuffer(_ buffer: MTLBuffer) throws -> MTLBuffer {
        if buffer.storageMode != .private {
            return buffer
        }

        guard let staging = buffer.device.makeBuffer(length: buffer.length, options: .storageModeShared),
              let queue = buffer.device.makeCommandQueue(),
              let commandBuffer = queue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw DecodeTestError.unavailableCommandSubmission
        }

        blit.copy(from: buffer, sourceOffset: 0, to: staging, destinationOffset: 0, size: buffer.length)
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return staging
    }
}

private enum DecodeTestError: Error {
    case unavailableCommandSubmission
}
