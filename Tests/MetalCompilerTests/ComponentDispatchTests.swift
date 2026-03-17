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

    // MARK: - Fragment Tree Helpers

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
        if let body = fragment as? any _FragmentBodyAccessor {
            body._visitBody { child in visitFragmentTree(child, visitor: visitor) }
        }
    }

    // MARK: - MLP

    @Test("MLP gate_proj outputDimension matches intermediateSize")
    func mlpGateProjDimension() {
        let mlp = MLPAttributes(inputSize: 2048, outputSize: 2048, intermediateSize: 8192,
                                activation: .silu, gating: .swiglu, bias: false)
        let linears = collectLinearFragments(mlp)
        #expect(linears.count == 3) // gate, up, down (SwiGLU is ElementwiseFragment, not Linear)
        #expect(linears[0].field == "gate_proj")
        #expect(linears[0].outputDimension == 8192)
        #expect(linears[0].inputDimension == 2048)
    }

    @Test("MLP up_proj outputDimension matches intermediateSize")
    func mlpUpProjDimension() {
        let mlp = MLPAttributes(inputSize: 2048, outputSize: 2048, intermediateSize: 8192,
                                activation: .silu, gating: .swiglu, bias: false)
        let linears = collectLinearFragments(mlp)
        #expect(linears[1].field == "up_proj")
        #expect(linears[1].outputDimension == 8192)
    }

    @Test("MLP down_proj inputDimension matches intermediateSize")
    func mlpDownProjDimension() {
        let mlp = MLPAttributes(inputSize: 2048, outputSize: 2048, intermediateSize: 8192,
                                activation: .silu, gating: .swiglu, bias: false)
        let linears = collectLinearFragments(mlp)
        #expect(linears[2].field == "down_proj")
        #expect(linears[2].inputDimension == 8192)
        #expect(linears[2].outputDimension == 2048)
    }

    // MARK: - Attention

    @Test("Attention Q/K/V/O projection dimensions are consistent")
    func attentionProjectionDimensions() {
        let attn = AttentionAttributes(hiddenSize: 2048, headCount: 32, kvHeadCount: 8,
                                       headDimension: 64, bias: false, causal: true,
                                       rope: RoPEAttributes(dimension: 64, base: 500000.0))
        let linears = collectLinearFragments(attn)
        #expect(linears.count == 4) // q, k, v, o
        #expect(linears[0].field == "q_proj")
        #expect(linears[0].outputDimension == 32 * 64)
        #expect(linears[1].field == "k_proj")
        #expect(linears[1].outputDimension == 8 * 64)
        #expect(linears[3].field == "o_proj")
        #expect(linears[3].inputDimension == 32 * 64)
    }

    @Test("Attention with qkNorm generates QKNormFragment")
    func attentionQKNormDispatches() {
        let attn = AttentionAttributes(hiddenSize: 2048, headCount: 32, kvHeadCount: 8,
                                       headDimension: 64, bias: false, causal: true,
                                       rope: RoPEAttributes(dimension: 64, base: 500000.0),
                                       qkNorm: .rmsNorm)
        let primitives = collectPrimitives(attn)
        let qkNorms = primitives.compactMap { $0 as? QKNormFragment }
        #expect(qkNorms.count == 2)
        #expect(qkNorms[0].weightRole == "q_layernorm")
        #expect(qkNorms[1].weightRole == "k_layernorm")
    }

    // MARK: - ShortConv

    @Test("ShortConv in_proj outputs 3x hiddenSize")
    func shortConvInProjDimension() {
        let conv = ShortConvAttributes(hiddenSize: 2048, kernelSize: 3)
        let linears = collectLinearFragments(conv)
        #expect(linears[0].field == "in_proj")
        #expect(linears[0].outputDimension == 2048 * 3)
    }

    @Test("ShortConv out_proj dimensions match hiddenSize")
    func shortConvOutProjDimension() {
        let conv = ShortConvAttributes(hiddenSize: 2048, kernelSize: 3)
        let linears = collectLinearFragments(conv)
        #expect(linears[1].field == "out_proj")
        #expect(linears[1].inputDimension == 2048)
        #expect(linears[1].outputDimension == 2048)
    }

    // MARK: - TokenEmbedding / OutputHead

    @Test("TokenEmbedding produces GatherFragment with correct dimension")
    func tokenEmbeddingDimension() {
        let emb = TokenEmbeddingAttributes(vocabSize: 65536, embeddingSize: 2048)
        let frag = emb.fragment
        #expect(frag.embeddingDimension == 2048)
        #expect(frag.vocabularySize == 65536)
    }

    @Test("OutputHead projection dimension matches vocabSize")
    func outputHeadDimension() {
        let head = OutputHeadAttributes(inputSize: 2048, vocabSize: 65536, tiedToEmbedding: true, bias: false)
        let linears = collectLinearFragments(head)
        #expect(linears[0].field == "weight")
        #expect(linears[0].outputDimension == 65536)
    }

    // MARK: - SwiGLU

    @Test("SwiGLU dimension consistency with MLP intermediateSize")
    func swigluDimensionConsistency() {
        let mlp = MLPAttributes(inputSize: 2048, outputSize: 2048, intermediateSize: 8192,
                                activation: .silu, gating: .swiglu, bias: false)
        let primitives = collectPrimitives(mlp)
        let swiglu = primitives.compactMap { $0 as? ElementwiseFragment }
        #expect(swiglu.count == 1)
        #expect(swiglu[0].count == 8192)
    }

    // MARK: - RMSNorm

    @Test("RMSNorm fragment has correct dimension")
    func rmsNormDimension() {
        let norm = RMSNormAttributes(dimension: 2048, epsilon: 1e-5)
        let frag = norm.fragment
        #expect(frag.dimension == 2048)
        #expect(frag.epsilon == 1e-5)
        #expect(frag.isFusable == true)
    }

    // MARK: - Conv1d

    @Test("Conv1dFragment has correct dimension and kernelSize")
    func conv1dDimension() {
        let conv = ShortConvAttributes(hiddenSize: 2048, kernelSize: 3)
        let primitives = collectPrimitives(conv)
        let convFrags = primitives.compactMap { $0 as? Conv1dFragment }
        #expect(convFrags.count == 1)
        #expect(convFrags[0].dimension == 2048)
        #expect(convFrags[0].kernelSize == 3)
    }
}
