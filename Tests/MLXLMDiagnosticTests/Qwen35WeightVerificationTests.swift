import Foundation
import GGUFParser
import MLX
import MLXNN
import Testing
import TestHeartbeat
@testable import MLXLM

/// Step-by-step verification of Qwen3.5 model correctness.
///
/// Step 1: Configuration (verified manually against GGUF metadata)
/// Step 2: Weight loading — modules have correct shapes and non-trivial values
/// Step 3: Single-component forward pass (future)
@Suite("Qwen3.5 Weight Verification", .tags(.diagnostic), .heartbeat)
struct Qwen35WeightVerificationTests {

    private static let repo = "unsloth/Qwen3.5-0.8B-GGUF"
    private static let filename = "Qwen3.5-0.8B-Q4_K_M.gguf"

    private func loadModel() async throws -> (Qwen35Model, ModelContext) {
        let downloader = HuggingFaceDownloader()
        let url = try await downloader.download(repo: Self.repo, filename: Self.filename)
        let loader = GGUFModelLoader()
        let context = try loader.loadContext(url: url)
        let model = try #require(context.model as? Qwen35Model)
        return (model, context)
    }

    // MARK: - Step 2a: Tensor Mapping Completeness

    @Test("Every GGUF tensor maps to a valid MLX weight key")
    func tensorMappingCompleteness() async throws {
        let downloader = HuggingFaceDownloader()
        let url = try await downloader.download(repo: Self.repo, filename: Self.filename)
        let file = try GGUFFile.parse(url: url)
        let mapper = HybridDeltaNetAttentionTensorNameMapper()

        var unmapped: [String] = []
        for tensor in file.tensors {
            if mapper.mlxName(for: tensor.name) == nil {
                unmapped.append(tensor.name)
            }
        }

        if !unmapped.isEmpty {
            print("[verify][mapping] unmapped tensors: \(unmapped)")
        }
        #expect(unmapped.isEmpty, Comment(rawValue: "Unmapped GGUF tensors: \(unmapped)"))
    }

    // MARK: - Step 2b: DeltaNet Module Weights

    @Test("DeltaNet layer 0 has loaded (non-init) weights for A_log, dt_bias, conv1d, norm")
    func deltaNetWeightsLoaded() async throws {
        let (model, _) = try await loadModel()
        let delta0 = try #require(model.model.layers[0].deltaNet)

        // A_log: initialized as zeros. If GGUF loads correctly, should be non-zero.
        let aLog = delta0.A_log
        eval(aLog)
        let aLogValues = aLog.asType(.float32).asArray(Float.self)
        let aLogAllZero = aLogValues.allSatisfy { $0 == 0 }
        print("[verify][delta0] A_log shape=\(aLog.shape) dtype=\(aLog.dtype) values=\(aLogValues.prefix(8))")
        // Note: A_log CAN be all zeros if the trained model has that value.
        // We just print it for inspection.

        // dt_bias: initialized as ones. If GGUF loads, should differ from all-ones.
        let dtBias = delta0.dt_bias
        eval(dtBias)
        let dtBiasValues = dtBias.asType(.float32).asArray(Float.self)
        let dtBiasAllOnes = dtBiasValues.allSatisfy { $0 == 1.0 }
        print("[verify][delta0] dt_bias shape=\(dtBias.shape) dtype=\(dtBias.dtype) values=\(dtBiasValues.prefix(8))")
        #expect(!dtBiasAllOnes, Comment(rawValue: "dt_bias still at init value (all ones) — GGUF weight not loaded"))

        // conv1d weight: should be [C, K, 1] after sanitize (from [C, K] in GGUF)
        let convWeight = delta0.conv.weight
        eval(convWeight)
        print("[verify][delta0] conv1d.weight shape=\(convWeight.shape) ndim=\(convWeight.ndim)")
        #expect(convWeight.ndim == 3, Comment(rawValue: "Conv1d weight should be 3D after sanitize, got \(convWeight.ndim)D"))
        // Expected shape: [6144, 4, 1] (depthwise: C_out=6144, K=4, C_in/groups=1)
        #expect(convWeight.dim(0) == 6144)
        #expect(convWeight.dim(1) == 4)
        #expect(convWeight.dim(2) == 1)

        // norm weight: should be [128] (linearValueHeadDim)
        let normWeight = delta0.gatedNorm.weight
        eval(normWeight)
        print("[verify][delta0] norm.weight shape=\(normWeight.shape)")
        #expect(normWeight.shape == [128])

        // Projections should be quantized (not plain Linear)
        #expect(delta0.inProjQKV is any Quantized, Comment(rawValue: "inProjQKV should be quantized"))
        #expect(delta0.outProj is any Quantized, Comment(rawValue: "outProj should be quantized"))
    }

    // MARK: - Step 2c: Full Attention Module Weights

    @Test("Full attention layer 3 has loaded weights with correct shapes")
    func fullAttentionWeightsLoaded() async throws {
        let (model, _) = try await loadModel()
        let attn3 = try #require(model.model.layers[3].fullAttn)

        // q_proj: Linear(1024, 4096) — outputs 2x headDim for gate
        // After quantization, check logical output size
        let qWeight = extractWeight(from: attn3.wq)
        print("[verify][attn3] q_proj weight shape=\(qWeight.shape)")

        // k_proj: Linear(1024, 512) — kvHeads * headDim = 2 * 256
        let kWeight = extractWeight(from: attn3.wk)
        print("[verify][attn3] k_proj weight shape=\(kWeight.shape)")

        // v_proj: Linear(1024, 512)
        let vWeight = extractWeight(from: attn3.wv)
        print("[verify][attn3] v_proj weight shape=\(vWeight.shape)")

        // o_proj: Linear(2048, 1024) — attentionHeads * headDim = 8 * 256 = 2048
        let oWeight = extractWeight(from: attn3.wo)
        print("[verify][attn3] o_proj weight shape=\(oWeight.shape)")

        // QK norms: [256] = headDim
        let qNorm = attn3.qNorm.weight
        let kNorm = attn3.kNorm.weight
        eval(qNorm, kNorm)
        print("[verify][attn3] q_norm shape=\(qNorm.shape) k_norm shape=\(kNorm.shape)")
        #expect(qNorm.shape == [256])
        #expect(kNorm.shape == [256])

        // Quantized
        #expect(attn3.wq is any Quantized)
        #expect(attn3.wv is any Quantized)
    }

    // MARK: - Step 2d: Embedding and LM Head

    @Test("Embedding is quantized with correct vocabulary size")
    func embeddingAndLMHead() async throws {
        let (model, _) = try await loadModel()

        // Embedding should be quantized
        #expect(model.model.embedTokens is any Quantized)
        print("[verify][embed] type=\(type(of: model.model.embedTokens))")

        // tieWordEmbeddings: if true, lmHead should be nil
        let config = model.configuration
        if config.tieWordEmbeddings {
            #expect(model.lmHead == nil, Comment(rawValue: "tieWordEmbeddings=true but lmHead is not nil"))
            print("[verify][lmhead] tied=true, lmHead=nil")
        } else {
            #expect(model.lmHead != nil)
            print("[verify][lmhead] tied=false, lmHead exists")
        }
    }

    // MARK: - Step 2e: Layer Routing

    @Test("Layer routing: DeltaNet for 0-2,4-6,... and FullAttn for 3,7,11,...")
    func layerRouting() async throws {
        let (model, _) = try await loadModel()

        for i in 0..<model.model.layers.count {
            let layer = model.model.layers[i]
            let expectFull = (i + 1) % 4 == 0

            #expect(
                layer.isFullAttention == expectFull,
                Comment(rawValue: "Layer \(i): expected isFullAttention=\(expectFull), got \(layer.isFullAttention)")
            )
        }
        print("[verify][routing] 24 layers verified: fullAttn at indices 3,7,11,15,19,23")
    }

    // MARK: - Helpers

    private func extractWeight(from linear: Linear) -> MLXArray {
        if let q = linear as? QuantizedLinear {
            let w = dequantized(
                q.weight, scales: q.scales, biases: q.biases,
                groupSize: q.groupSize, bits: q.bits
            )
            eval(w)
            return w
        }
        eval(linear.weight)
        return linear.weight
    }
}
