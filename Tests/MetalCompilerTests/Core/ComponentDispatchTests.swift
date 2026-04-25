import Testing
import Metal
import Foundation
@testable import MetalCompiler
import LMArchitecture
import ModelDeclarations
import LMIR

/// Verify every ModelComponent's fragment tree produces correct projection dimensions.
/// Prevents dimension mismatch bugs: GEMM outputDimension must match weight row count.
@Suite("Component Dispatch Consistency")
struct ComponentDispatchTests {

    private let defaultContext = KernelContext(bufferPrecision: .float16, weightFormat: .float16)

    // MARK: - Fragment Tree Helpers

    /// Get fragment tree from MetalCompilable attributes via capability query.
    private func fragmentTree(_ attributes: any OperationAttributes) -> any MetalKernelFragment {
        guard let compilable = attributes as? any MetalCompilable else {
            fatalError("\(type(of: attributes)) does not conform to MetalCompilable")
        }
        return compilable.fragment(context: defaultContext)
    }

    /// Collect all LinearFragment instances from a fragment tree.
    private func collectLinearFragments(_ fragment: any MetalKernelFragment) -> [LinearFragment] {
        var result: [LinearFragment] = []
        visitFragmentTree(fragment) { primitive in
            if let linear = primitive as? LinearFragment {
                result.append(linear)
            }
        }
        return result
    }

    /// Collect all primitive fragments from a fragment tree.
    private func collectPrimitives(_ fragment: any MetalKernelFragment) -> [any PrimitiveMetalKernelFragment] {
        var result: [any PrimitiveMetalKernelFragment] = []
        visitFragmentTree(fragment) { result.append($0) }
        return result
    }

    /// Walk a fragment tree and visit all primitive fragments.
    private func visitFragmentTree(_ fragment: any MetalKernelFragment, visitor: (any PrimitiveMetalKernelFragment) -> Void) {
        if let primitive = fragment as? any PrimitiveMetalKernelFragment {
            visitor(primitive)
            return
        }
        if let tuple = fragment as? any _TupleFragmentProtocol {
            tuple._visitChildren { child in visitFragmentTree(child, visitor: visitor) }
            return
        }
        if let opt = fragment as? any _OptionalFragmentProtocol {
            opt._visitContent { child in visitFragmentTree(child, visitor: visitor) }
            return
        }
        if let cond = fragment as? any _ConditionalFragmentProtocol {
            cond._visitActive { child in visitFragmentTree(child, visitor: visitor) }
            return
        }
    }

    // MARK: - MLP

    @Test("MLP gate and up projections are batched with correct dimensions")
    func mlpBatchedProjections() {
        let mlp = MLPAttributes(inputSize: 2048, outputSize: 2048, intermediateSize: 8192,
                                activation: .silu, gating: .swiglu, bias: false)
        let primitives = collectPrimitives(fragmentTree(mlp))
        let batched = primitives.compactMap { $0 as? BatchedProjection }
        #expect(batched.count == 1)

        let entries = batched[0].projections
        #expect(entries.count == 2)
        #expect(entries[0].field == "gate_proj")
        #expect(entries[0].outputDimension == 8192)
        #expect(entries[0].inputDimension == 2048)
        #expect(entries[1].field == "up_proj")
        #expect(entries[1].outputDimension == 8192)
    }

    @Test("MLP down_proj inputDimension matches intermediateSize")
    func mlpDownProjDimension() {
        let mlp = MLPAttributes(inputSize: 2048, outputSize: 2048, intermediateSize: 8192,
                                activation: .silu, gating: .swiglu, bias: false)
        let linears = collectLinearFragments(fragmentTree(mlp))
        #expect(linears.count == 1)
        #expect(linears[0].field == "down_proj")
        #expect(linears[0].inputDimension == 8192)
        #expect(linears[0].outputDimension == 2048)
    }

    // MARK: - Attention

    @Test("Attention Q/K/V/O projection dimensions are consistent")
    func attentionProjectionDimensions() {
        let attn = AttentionAttributes(hiddenSize: 2048, headCount: 32, kvHeadCount: 8,
                                       headDimension: 64, bias: false, causal: true,
                                       rope: RoPEAttributes(dimension: 64, base: 500000.0))
        let primitives = collectPrimitives(fragmentTree(attn))

        // Q/K/V are batched (component-internal optimization)
        let batched = primitives.compactMap { $0 as? BatchedProjection }
        #expect(batched.count == 1)
        let entries = batched[0].projections
        #expect(entries.count == 3)
        #expect(entries[0].field == "q_proj")
        #expect(entries[0].outputDimension == 32 * 64)
        #expect(entries[1].field == "k_proj")
        #expect(entries[1].outputDimension == 8 * 64)

        // O projection is standalone LinearFragment
        let linears = collectLinearFragments(fragmentTree(attn))
        #expect(linears.count == 1)
        #expect(linears[0].field == "o_proj")
        #expect(linears[0].inputDimension == 32 * 64)
    }

    @Test("Attention with qkNorm and RoPE generates fused QKNorm RoPE fragment")
    func attentionQKNormDispatches() throws {
        let attn = AttentionAttributes(hiddenSize: 2048, headCount: 32, kvHeadCount: 8,
                                       headDimension: 64, bias: false, causal: true,
                                       rope: RoPEAttributes(dimension: 64, base: 500000.0),
                                       qkNorm: .rmsNorm)
        let primitives = collectPrimitives(fragmentTree(attn))
        // QKNorm + RoPE is fused for prefill to avoid an extra dispatch.
        let fusedFragments = primitives.compactMap { $0 as? BatchedQKNormRoPEFragment }
        #expect(fusedFragments.count == 1)
        let fragment = try #require(fusedFragments.first)
        #expect(fragment.qNorm.weightRole == "q_layernorm")
        #expect(fragment.kNorm.weightRole == "k_layernorm")
        #expect(fragment.ropeDimension == 64)
        #expect(fragment.ropeBase == 500000.0)
    }

    // MARK: - ShortConv

    @Test("ShortConv in_proj outputs 3x hiddenSize")
    func shortConvInProjDimension() {
        let conv = ShortConvAttributes(hiddenSize: 2048, kernelSize: 3)
        let linears = collectLinearFragments(fragmentTree(conv))
        #expect(linears[0].field == "in_proj")
        #expect(linears[0].outputDimension == 2048 * 3)
    }

    @Test("ShortConv out_proj dimensions match hiddenSize")
    func shortConvOutProjDimension() {
        let conv = ShortConvAttributes(hiddenSize: 2048, kernelSize: 3)
        let linears = collectLinearFragments(fragmentTree(conv))
        #expect(linears[1].field == "out_proj")
        #expect(linears[1].inputDimension == 2048)
        #expect(linears[1].outputDimension == 2048)
    }

    // MARK: - TokenEmbedding / OutputHead

    @Test("TokenEmbedding produces GatherFragment with correct dimension")
    func tokenEmbeddingDimension() {
        let emb = TokenEmbeddingAttributes(vocabSize: 65536, embeddingSize: 2048)
        let frag = emb.fragment(context: KernelContext(bufferPrecision: .float16, weightFormat: .float16))
        #expect(frag.embeddingDimension == 2048)
        #expect(frag.vocabularySize == 65536)
    }

    @Test("OutputHead projection dimension matches vocabSize")
    func outputHeadDimension() {
        let head = OutputHeadAttributes(inputSize: 2048, vocabSize: 65536, tiedToEmbedding: true, bias: false)
        let linears = collectLinearFragments(fragmentTree(head))
        #expect(linears[0].field == "weight")
        #expect(linears[0].outputDimension == 65536)
    }

    // MARK: - SwiGLU

    @Test("SwiGLU dimension consistency with MLP intermediateSize")
    func swigluDimensionConsistency() {
        let mlp = MLPAttributes(inputSize: 2048, outputSize: 2048, intermediateSize: 8192,
                                activation: .silu, gating: .swiglu, bias: false)
        let primitives = collectPrimitives(fragmentTree(mlp))
        let swiglu = primitives.compactMap { $0 as? ElementwiseFragment }
        #expect(swiglu.count == 1)
        #expect(swiglu[0].count == 8192)
    }

    // MARK: - RMSNorm

    @Test("RMSNorm fragment has correct dimension")
    func rmsNormDimension() {
        let norm = RMSNormAttributes(dimension: 2048, epsilon: 1e-5)
        let frag = norm.fragment(context: KernelContext(bufferPrecision: .float16, weightFormat: .float16))
        #expect(frag.dimension == 2048)
        #expect(frag.epsilon == 1e-5)
        #expect(frag.isFusable == true)
    }

    // MARK: - Conv1d

    @Test("Conv1dFragment has correct dimension and kernelSize")
    func conv1dDimension() {
        let conv = ShortConvAttributes(hiddenSize: 2048, kernelSize: 3)
        let primitives = collectPrimitives(fragmentTree(conv))
        let convFrags = primitives.compactMap { $0 as? Conv1dFragment }
        #expect(convFrags.count == 1)
        #expect(convFrags[0].dimension == 2048)
        #expect(convFrags[0].kernelSize == 3)
    }

    // MARK: - Linear

    @Test("Linear produces LinearFragment with correct dimensions")
    func linearDimension() {
        let linear = LinearAttributes(inputSize: 2048, outputSize: 4096)
        let linears = collectLinearFragments(fragmentTree(linear))
        #expect(linears.count == 1)
        #expect(linears[0].field == "weight")
        #expect(linears[0].inputDimension == 2048)
        #expect(linears[0].outputDimension == 4096)
    }

    // MARK: - LayerNorm

    @Test("LayerNorm fragment has correct dimension")
    func layerNormDimension() {
        let norm = LayerNormAttributes(dimension: 2048, epsilon: 1e-5)
        let frag = norm.fragment(context: defaultContext)
        #expect(frag.dimension == 2048)
        #expect(frag.epsilon == 1e-5)
    }

    // MARK: - StateSpace

    @Test("StateSpace projections have correct dimensions")
    func stateSpaceProjections() {
        let ssm = StateSpaceAttributes(
            hiddenSize: 2048, numHeads: 16,
            keyHeadDim: 128, valueHeadDim: 128,
            convKernelSize: 4, variant: "deltanet"
        )
        let primitives = collectPrimitives(fragmentTree(ssm))

        // MetalCompilable returns optimized: BatchedProjection(4-way) + SSMRecurrence + LinearFragment(out_proj)
        let batched = primitives.compactMap { $0 as? BatchedProjection }
        #expect(batched.count == 1)
        let batchedEntries = batched[0].projections
        #expect(batchedEntries.count == 4)

        // in_proj_qkv: 2 * groupCount * keyHeadDim + numHeads * valueHeadDim
        let expectedQKV = 2 * 16 * 128 + 16 * 128  // 6144
        #expect(batchedEntries[0].field == "in_proj_qkv")
        #expect(batchedEntries[0].inputDimension == 2048)
        #expect(batchedEntries[0].outputDimension == expectedQKV)

        // in_proj_z: numHeads * valueHeadDim
        #expect(batchedEntries[1].field == "in_proj_z")
        #expect(batchedEntries[1].outputDimension == 16 * 128)

        // in_proj_b, in_proj_a: numHeads
        #expect(batchedEntries[2].field == "in_proj_b")
        #expect(batchedEntries[2].outputDimension == 16)
        #expect(batchedEntries[3].field == "in_proj_a")
        #expect(batchedEntries[3].outputDimension == 16)

        // out_proj (standalone LinearFragment)
        let linears = collectLinearFragments(fragmentTree(ssm))
        #expect(linears.count == 1)
        #expect(linears[0].field == "out_proj")
        #expect(linears[0].inputDimension == 16 * 128)
        #expect(linears[0].outputDimension == 2048)
    }

    @Test("StateSpace produces SSMRecurrenceFragment")
    func stateSpaceRecurrence() {
        let ssm = StateSpaceAttributes(
            hiddenSize: 2048, numHeads: 16,
            keyHeadDim: 128, valueHeadDim: 128,
            convKernelSize: 4, variant: "deltanet"
        )
        let primitives = collectPrimitives(fragmentTree(ssm))
        let recurrences = primitives.compactMap { $0 as? SSMRecurrenceFragment }
        #expect(recurrences.count == 1)
        #expect(recurrences[0].headCount == 16)
        #expect(recurrences[0].keyHeadDimension == 128)
        #expect(recurrences[0].valueHeadDimension == 128)
        #expect(recurrences[0].convKernelSize == 4)
    }

    // MARK: - MoE

    @Test("MoE produces router projection")
    func moeRouterProjection() {
        let moe = MoEAttributes(
            expertCount: 8,
            expertsPerToken: 2,
            expertMLP: MLPAttributes(
                inputSize: 2048, outputSize: 2048,
                intermediateSize: 8192,
                activation: .silu, gating: .swiglu, bias: false
            )
        )
        let linears = collectLinearFragments(fragmentTree(moe))
        #expect(linears.count == 1)
        #expect(linears[0].field == "router")
        #expect(linears[0].inputDimension == 2048)
        #expect(linears[0].outputDimension == 8)
    }

    // MARK: - PerLayerInput

    @Test("PerLayerInput produces gate, modulation, projection, and norm")
    func perLayerInputFragments() {
        let pli = PerLayerInputAttributes(
            hiddenSize: 2048, perLayerInputSize: 256, vocabSize: 262144
        )
        let primitives = collectPrimitives(fragmentTree(pli))
        let linears = primitives.compactMap { $0 as? LinearFragment }
        let modulations = primitives.compactMap { $0 as? PerLayerInputModulationFragment }
        let norms = primitives.compactMap { $0 as? Reduction }

        #expect(linears.count == 2)
        #expect(linears[0].field == "per_layer_input_gate")
        #expect(linears[0].inputDimension == 2048)
        #expect(linears[0].outputDimension == 256)
        #expect(linears[1].field == "per_layer_projection")
        #expect(linears[1].inputDimension == 256)
        #expect(linears[1].outputDimension == 2048)

        #expect(modulations.count == 1)
        #expect(modulations[0].dimension == 256)

        #expect(norms.count == 1)
        #expect(norms[0].dimension == 2048)
    }

    // MARK: - LayerScale

    @Test("LayerScale produces ScalarMultiplyFragment")
    func layerScaleFragment() {
        let ls = LayerScaleAttributes(dimension: 2048)
        let primitives = collectPrimitives(fragmentTree(ls))
        let scalars = primitives.compactMap { $0 as? ScalarMultiplyFragment }
        #expect(scalars.count == 1)
        #expect(scalars[0].count == 2048)
    }
}
