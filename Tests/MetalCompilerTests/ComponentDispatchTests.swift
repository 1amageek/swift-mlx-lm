import Testing
import Metal
import Foundation
@testable import MetalCompiler
import LMArchitecture
import ModelDeclarations
import LMIR

/// Verify every ModelComponent's dispatch declarations match actual weight shapes.
/// Prevents the intermediate_size mismatch bug: GEMM outputDimension must match weight row count.
@Suite("Component Dispatch Consistency")
struct ComponentDispatchTests {

    // MARK: - MLP: intermediateSize must match weight shape

    @Test("MLP gate_proj outputDimension matches weight rows")
    func mlpGateProjDimension() {
        let mlp = MLPAttributes(
            inputSize: 2048, outputSize: 2048, intermediateSize: 8192,
            activation: .silu, gating: .swiglu, bias: false)
        let decls = mlp.dispatchDeclarations
        // gate_proj is the first projection
        guard case .projection(let proj) = decls[0] else {
            Issue.record("Expected projection at index 0")
            return
        }
        #expect(proj.field == "gate_proj")
        #expect(proj.inputDimension == 2048)
        #expect(proj.outputDimension == 8192, "gate_proj output must be intermediateSize, not some scaled value")
    }

    @Test("MLP up_proj outputDimension matches intermediateSize")
    func mlpUpProjDimension() {
        let mlp = MLPAttributes(
            inputSize: 2048, outputSize: 2048, intermediateSize: 8192,
            activation: .silu, gating: .swiglu, bias: false)
        let decls = mlp.dispatchDeclarations
        guard case .projection(let proj) = decls[1] else {
            Issue.record("Expected projection at index 1")
            return
        }
        #expect(proj.field == "up_proj")
        #expect(proj.outputDimension == 8192)
    }

    @Test("MLP down_proj inputDimension matches intermediateSize")
    func mlpDownProjDimension() {
        let mlp = MLPAttributes(
            inputSize: 2048, outputSize: 2048, intermediateSize: 8192,
            activation: .silu, gating: .swiglu, bias: false)
        let decls = mlp.dispatchDeclarations
        // down_proj is the last projection
        let lastProj = decls.last { if case .projection = $0 { return true }; return false }
        guard case .projection(let proj) = lastProj else {
            Issue.record("Expected projection as last declaration")
            return
        }
        #expect(proj.field == "down_proj")
        #expect(proj.inputDimension == 8192, "down_proj input must be intermediateSize")
        #expect(proj.outputDimension == 2048)
    }

    // MARK: - Attention: projection dimensions

    @Test("Attention Q/K/V/O projection dimensions are consistent")
    func attentionProjectionDimensions() {
        let attn = AttentionAttributes(
            hiddenSize: 2048, headCount: 32, kvHeadCount: 8,
            headDimension: 64, bias: false, causal: true,
            rope: RoPEAttributes(dimension: 64, base: 10000.0),
            qkNorm: .rmsNorm)
        let decls = attn.dispatchDeclarations
        let projections = decls.compactMap { decl -> MetalProjection? in
            if case .projection(let p) = decl { return p }
            return nil
        }

        let qProj = projections.first { $0.field == "q_proj" }!
        let kProj = projections.first { $0.field == "k_proj" }!
        let vProj = projections.first { $0.field == "v_proj" }!
        let oProj = projections.first { $0.field == "o_proj" }!

        #expect(qProj.inputDimension == 2048)
        #expect(qProj.outputDimension == 32 * 64, "Q: headCount * headDim")
        #expect(kProj.outputDimension == 8 * 64, "K: kvHeadCount * headDim")
        #expect(vProj.outputDimension == 8 * 64, "V: kvHeadCount * headDim")
        #expect(oProj.inputDimension == 32 * 64, "O input: headCount * headDim")
        #expect(oProj.outputDimension == 2048, "O output: hiddenSize")
    }

    @Test("Attention with qkNorm generates QKNormOperation dispatches")
    func attentionQKNormDispatches() {
        let attn = AttentionAttributes(
            hiddenSize: 2048, headCount: 32, kvHeadCount: 8,
            headDimension: 64, bias: false, causal: true,
            rope: RoPEAttributes(dimension: 64, base: 10000.0),
            qkNorm: .rmsNorm)
        let computes = attn.dispatchDeclarations.compactMap { decl -> (any MetalComputeOperation)? in
            if case .compute(let op) = decl { return op }
            return nil
        }
        let qkNorms = computes.compactMap { $0 as? QKNormOperation }
        #expect(qkNorms.count == 2, "Should have Q norm and K norm")
        #expect(qkNorms[0].weightRole == "q_layernorm")
        #expect(qkNorms[1].weightRole == "k_layernorm")
        #expect(qkNorms[0].headCount == 32)
        #expect(qkNorms[1].headCount == 8, "K norm uses kvHeadCount")
    }

    // MARK: - ShortConv: in_proj outputDimension = 3 * hiddenSize

    @Test("ShortConv in_proj outputs 3x hiddenSize")
    func shortConvInProjDimension() {
        let conv = ShortConvAttributes(hiddenSize: 2048, kernelSize: 3)
        let decls = conv.dispatchDeclarations
        guard case .projection(let inProj) = decls[0] else {
            Issue.record("Expected projection at index 0")
            return
        }
        #expect(inProj.field == "in_proj")
        #expect(inProj.inputDimension == 2048)
        #expect(inProj.outputDimension == 2048 * 3, "in_proj output = 3 * hiddenSize for conv1d")
    }

    @Test("ShortConv out_proj dimensions match hiddenSize")
    func shortConvOutProjDimension() {
        let conv = ShortConvAttributes(hiddenSize: 2048, kernelSize: 3)
        let decls = conv.dispatchDeclarations
        guard case .projection(let outProj) = decls[2] else {
            Issue.record("Expected projection at index 2")
            return
        }
        #expect(outProj.field == "out_proj")
        #expect(outProj.inputDimension == 2048)
        #expect(outProj.outputDimension == 2048)
    }

    // MARK: - TokenEmbedding

    @Test("TokenEmbedding embeddingDimension matches")
    func tokenEmbeddingDimension() {
        let emb = TokenEmbeddingAttributes(vocabSize: 65536, embeddingSize: 2048)
        guard case .compute(let op) = emb.dispatchDeclarations[0] else {
            Issue.record("Expected compute declaration")
            return
        }
        let embOp = op as! EmbeddingLookupOperation
        #expect(embOp.vocabularySize == 65536)
        #expect(embOp.embeddingDimension == 2048)
    }

    // MARK: - OutputHead

    @Test("OutputHead projection dimension matches vocabSize")
    func outputHeadDimension() {
        let head = OutputHeadAttributes(inputSize: 2048, vocabSize: 65536, tiedToEmbedding: true, bias: false)
        let decls = head.dispatchDeclarations
        guard case .projection(let proj) = decls[0] else {
            Issue.record("Expected projection")
            return
        }
        #expect(proj.inputDimension == 2048)
        #expect(proj.outputDimension == 65536)
    }

    // MARK: - Full Model: all projections have matching STAF weights

    @Test("LFM2 model compiles without GPU error using real STAF weights")
    func lfm2CompileAndRunWithSTAF() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let stafPath = "/Users/1amageek/Library/Containers/team.stamp.JARDIS.ml/Data/Documents/huggingface/models/LiquidAI/LFM2.5-1.2B-Thinking/model.staf"
        guard FileManager.default.fileExists(atPath: stafPath) else {
            print("STAF not found — skipping")
            return
        }

        let store = try STAFLoader().load(at: URL(fileURLWithPath: stafPath), device: device)

        // Use the CORRECT intermediateSize (8192 = actual weight dim, not config's 12288)
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

        // Compile and run a short prefill — if dimensions mismatch, this will produce NaN
        let prefillPlan = try MetalInferenceCompiler().compilePrefill(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, maximumSequenceLength: 32,
            stafWeightStore: store, device: device)

        let seqLen = 8
        let tokenPtr = prefillPlan.buffers.tokenIDs.contents().bindMemory(to: Int32.self, capacity: seqLen)
        let posPtr = prefillPlan.buffers.positions.contents().bindMemory(to: UInt32.self, capacity: seqLen)
        for i in 0..<seqLen { tokenPtr[i] = Int32(i + 1); posPtr[i] = UInt32(i) }

        // Run each step and verify no NaN
        for (stepIndex, step) in prefillPlan.steps.enumerated() {
            guard let queue = device.makeCommandQueue(),
                  let cb = queue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else { break }
            enc.memoryBarrier(scope: .buffers)

            switch step.mode {
            case .batch:
                enc.setComputePipelineState(step.pipeline)
                for (i, buf, off) in step.bufferBindings { enc.setBuffer(buf, offset: off, index: i) }
                for (i, val) in step.bytesBindings { val.withUnsafeBufferPointer { enc.setBytes($0.baseAddress!, length: $0.count, index: i) } }
                if let si = step.sequenceLengthBindingIndex { var sl = UInt32(seqLen); withUnsafeBytes(of: &sl) { enc.setBytes($0.baseAddress!, length: $0.count, index: si) } }
                var g = step.gridSize; if step.sequenceLengthBindingIndex != nil && g.height > 1 { g = MTLSize(width: g.width, height: seqLen, depth: g.depth) }
                enc.dispatchThreadgroups(g, threadsPerThreadgroup: step.threadgroupSize)
            case .perPosition:
                for pos in 0..<seqLen {
                    enc.setComputePipelineState(step.pipeline)
                    for (i, buf, base) in step.bufferBindings { enc.setBuffer(buf, offset: base + pos * (step.perPositionStrides[i] ?? 0), index: i) }
                    for (i, val) in step.bytesBindings { val.withUnsafeBufferPointer { enc.setBytes($0.baseAddress!, length: $0.count, index: i) } }
                    if let pi = step.positionBufferIndex { var pv = UInt32(pos); withUnsafeBytes(of: &pv) { enc.setBytes($0.baseAddress!, length: $0.count, index: pi) } }
                    enc.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
                }
            case .lastToken:
                enc.setComputePipelineState(step.pipeline)
                for (i, buf, base) in step.bufferBindings { enc.setBuffer(buf, offset: base + (seqLen-1) * (step.perPositionStrides[i] ?? 0), index: i) }
                for (i, val) in step.bytesBindings { val.withUnsafeBufferPointer { enc.setBytes($0.baseAddress!, length: $0.count, index: i) } }
                enc.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
            }
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()

            if let error = cb.error {
                Issue.record("GPU error at step \(stepIndex): \(error)")
                return
            }

            let hp = prefillPlan.buffers.hidden.contents().bindMemory(to: Float.self, capacity: 2048)
            let hasNaN = (0..<2048).contains { hp[$0].isNaN }
            if hasNaN {
                Issue.record("NaN at step \(stepIndex) (label: \(step.pipeline.label ?? "?"))")
                return
            }
        }
    }

    // MARK: - SwiGLU dimension consistency

    @Test("SwiGLU dimension equals intermediateSize")
    func swigluDimensionConsistency() {
        let mlp = MLPAttributes(
            inputSize: 2048, outputSize: 2048, intermediateSize: 8192,
            activation: .silu, gating: .swiglu, bias: false)
        let swiglu = mlp.dispatchDeclarations.compactMap { decl -> SwiGLUOperation? in
            if case .compute(let op) = decl { return op as? SwiGLUOperation }
            return nil
        }.first!
        #expect(swiglu.dimension == 8192, "SwiGLU dimension must equal intermediateSize")
    }

    // MARK: - Norm dimensions

    @Test("RMSNorm dimension matches hiddenSize")
    func rmsNormDimension() {
        let norm = RMSNormAttributes(dimension: 2048, epsilon: 1e-5)
        guard case .compute(let op) = norm.dispatchDeclarations[0] else {
            Issue.record("Expected compute")
            return
        }
        let normOp = op as! RMSNormOperation
        #expect(normOp.dimension == 2048)
        #expect(normOp.epsilon == 1e-5)
    }

    // MARK: - Conv1d dimension

    @Test("Conv1d dimension matches hiddenSize")
    func conv1dDimension() {
        let conv = ShortConvAttributes(hiddenSize: 2048, kernelSize: 3)
        let conv1d = conv.dispatchDeclarations.compactMap { decl -> Conv1dOperation? in
            if case .compute(let op) = decl { return op as? Conv1dOperation }
            return nil
        }.first!
        #expect(conv1d.dimension == 2048)
        #expect(conv1d.kernelSize == 3)
    }

    // MARK: - Regression: SwiGLU intermediate_size must be actual weight dimension

    @Test("SwiGLU intermediate_size × 2/3 matches safetensors weight shape for LFM2")
    func swigluIntermediateSizeRegression() throws {
        // config.json has intermediate_size=12288, but actual weight shape is [8192, 2048]
        // because SwiGLU uses 2/3 of the config value.
        let configIntermediateSize = 12288
        let actualWeightRows = 8192
        let swiGLUAdjusted = configIntermediateSize * 2 / 3
        #expect(swiGLUAdjusted == actualWeightRows,
                "SwiGLU intermediate = config * 2/3 = \(swiGLUAdjusted), weight rows = \(actualWeightRows)")

        // MLP with adjusted size should produce matching projections
        let mlp = MLPAttributes(
            inputSize: 2048, outputSize: 2048, intermediateSize: swiGLUAdjusted,
            activation: .silu, gating: .swiglu, bias: false)
        let projs = mlp.dispatchDeclarations.compactMap { decl -> MetalProjection? in
            if case .projection(let p) = decl { return p }
            return nil
        }
        let gate = projs.first { $0.field == "gate_proj" }!
        #expect(gate.outputDimension == actualWeightRows,
                "gate_proj output \(gate.outputDimension) must match weight rows \(actualWeightRows)")
    }

    // MARK: - Regression: Float32 intermediate buffers prevent overflow

    @Test("Prefill scratch buffer is Float32 (4 bytes per element)")
    func prefillScratchIsFloat32() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = ModelConfig(
            hiddenSize: 64, layerCount: 1, intermediateSize: 128,
            vocabSize: 100, attentionHeads: 4, kvHeads: 4, headDim: 16,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 10000, ropeDimension: 16,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: false,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: nil,
            partialRotaryFactor: nil, slidingWindow: nil
        )
        let graph = try Transformer(config: config).makeModelGraph()
        let resolved = ParameterResolver().resolve(graph: graph, convention: .llamaFamily)

        let maxSeqLen = 32
        let plan = try MetalInferenceCompiler().compilePrefill(
            graph: resolved, hiddenSize: 64, intermediateSize: 128,
            vocabSize: 100, maximumSequenceLength: maxSeqLen, device: device)

        // Scratch should be Float32 (4 bytes per element, not 2)
        // Verify by checking element size: scratch.length / (slots * slotDim * maxSeqLen) should be 4
        let scratchBytesPerElement = plan.buffers.scratch.length / (maxSeqLen * 1024)
        #expect(scratchBytesPerElement == MemoryLayout<Float>.size,
                "Scratch element size should be \(MemoryLayout<Float>.size) (Float32), got \(scratchBytesPerElement)")

        // Hidden should also be Float32
        let expectedHiddenF32 = maxSeqLen * 64 * MemoryLayout<Float>.size
        #expect(plan.buffers.hidden.length == expectedHiddenF32,
                "Hidden should be Float32: expected \(expectedHiddenF32), got \(plan.buffers.hidden.length)")
    }

    // MARK: - Regression: flash_attn_decode_f32 kernel exists and is used

    @Test("Prefill attention uses flash_attn_decode_f32 (not half-precision variant)")
    func prefillAttentionUsesFloat32Kernel() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = ModelConfig(
            hiddenSize: 64, layerCount: 1, intermediateSize: 128,
            vocabSize: 100, attentionHeads: 4, kvHeads: 4, headDim: 16,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 10000, ropeDimension: 16,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: false,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: nil,
            partialRotaryFactor: nil, slidingWindow: nil
        )
        let graph = try Transformer(config: config).makeModelGraph()
        let resolved = ParameterResolver().resolve(graph: graph, convention: .llamaFamily)

        let plan = try MetalInferenceCompiler().compilePrefill(
            graph: resolved, hiddenSize: 64, intermediateSize: 128,
            vocabSize: 100, maximumSequenceLength: 32, device: device)

        // Find perPosition steps (attention) and verify they use the f32 variant
        let attnSteps = plan.steps.filter { $0.mode == .perPosition }
        #expect(!attnSteps.isEmpty, "Should have perPosition attention steps")

        for step in attnSteps {
            let label = step.pipeline.label ?? ""
            #expect(label.contains("f32"),
                    "Attention step should use float32 kernel, got '\(label)'")
        }
    }

    // MARK: - Regression: GEMM projection dimensions vs STAF weight size

    @Test("Every GEMM projection outputDimension × inputDimension fits STAF weight payload")
    func gemmDimensionsMatchSTAFWeights() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let stafPath = "/Users/1amageek/Library/Containers/team.stamp.JARDIS.ml/Data/Documents/huggingface/models/LiquidAI/LFM2.5-1.2B-Thinking/model.staf"
        guard FileManager.default.fileExists(atPath: stafPath) else {
            print("STAF not found — skipping"); return
        }

        let store = try STAFLoader().load(at: URL(fileURLWithPath: stafPath), device: device)

        // Check every weight tensor: outputDim × inputDim × 2 (BF16) must equal payloadSize
        let projectionWeights = [
            // Layer 0 conv
            ("model.layers.0.conv.in_proj.weight", 6144, 2048),
            ("model.layers.0.conv.out_proj.weight", 2048, 2048),
            // Layer 0 MLP
            ("model.layers.0.feed_forward.w1.weight", 8192, 2048),  // gate_proj
            ("model.layers.0.feed_forward.w3.weight", 8192, 2048),  // up_proj
            ("model.layers.0.feed_forward.w2.weight", 2048, 8192),  // down_proj
            // Layer 2 attention
            ("model.layers.2.self_attn.q_proj.weight", 2048, 2048),
            ("model.layers.2.self_attn.k_proj.weight", 512, 2048),
            ("model.layers.2.self_attn.v_proj.weight", 512, 2048),
            ("model.layers.2.self_attn.out_proj.weight", 2048, 2048),
        ]

        var mismatches: [String] = []
        for (name, expectedRows, expectedCols) in projectionWeights {
            guard let entry = store.entries[name] else {
                mismatches.append("\(name): NOT FOUND in STAF")
                continue
            }
            let expectedBytes = expectedRows * expectedCols * 2  // BF16
            if entry.payloadSize != expectedBytes {
                let actualRows = entry.payloadSize / (expectedCols * 2)
                mismatches.append("\(name): expected \(expectedRows)×\(expectedCols) (\(expectedBytes)B), STAF has \(entry.payloadSize)B (\(actualRows) rows)")
            }
        }

        if !mismatches.isEmpty {
            for m in mismatches { print("[STAF mismatch] \(m)") }
            Issue.record("\(mismatches.count) weight dimension mismatches")
        }
    }

    // MARK: - Prefill→Decode transfer: float32 hidden → float16 decode hidden

    @Test("Prefill to decode hidden transfer preserves finite values")
    func prefillToDecodeTransfer() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        // Create a float32 buffer simulating prefill hidden
        let hiddenSize = 64
        let seqLen = 4
        let f32Buffer = device.makeBuffer(length: seqLen * hiddenSize * MemoryLayout<Float>.size, options: .storageModeShared)!
        let f16Buffer = device.makeBuffer(length: hiddenSize * MemoryLayout<Float16>.size, options: .storageModeShared)!

        // Fill with known values — including large values that would overflow Float16
        let f32Ptr = f32Buffer.contents().bindMemory(to: Float.self, capacity: seqLen * hiddenSize)
        for i in 0..<(seqLen * hiddenSize) {
            f32Ptr[i] = Float(i) * 0.01 - 1.0  // range [-1.0, ~1.56]
        }
        // Set last position to have a large value
        f32Ptr[(seqLen - 1) * hiddenSize] = 50000.0  // within Float16 range (65504)

        // Transfer last position: float32 → float16
        let lastOffset = (seqLen - 1) * hiddenSize
        let src = f32Buffer.contents().advanced(by: lastOffset * MemoryLayout<Float>.size).bindMemory(to: Float.self, capacity: hiddenSize)
        let dst = f16Buffer.contents().bindMemory(to: Float16.self, capacity: hiddenSize)
        for i in 0..<hiddenSize {
            dst[i] = Float16(src[i])
        }

        // Verify transfer
        #expect(dst[0] == Float16(50000.0), "First element should be 50000")
        #expect(!dst[0].isNaN, "Transfer should not produce NaN")
        #expect(!dst[0].isInfinite, "50000 should be finite in Float16")
        #expect(dst[1] == Float16(src[1]), "Other elements should transfer correctly")
    }
}
