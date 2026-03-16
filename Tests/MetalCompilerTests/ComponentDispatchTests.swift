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
}
