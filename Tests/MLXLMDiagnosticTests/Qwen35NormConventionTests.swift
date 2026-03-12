import Foundation
import GGUFParser
import MLX
import MLXNN
import Testing
import TestHeartbeat
@testable import MLXLM

/// Step 3: Verify the gated RMS norm weight convention.
///
/// HuggingFace Qwen3.5 uses: `self.weight = nn.Parameter(torch.ones(dim))`
/// and applies `self.weight * rms_norm(x)`.
///
/// Our code uses: `self.weight = MLXArray.zeros([dim])`
/// and applies `(1 + weight) * rms_norm(x)`.
///
/// If GGUF stores the HF checkpoint values (near 1), then:
/// - HF: weight * rms_norm(x) ≈ 1 * rms_norm(x)
/// - Ours: (1 + weight) * rms_norm(x) ≈ 2 * rms_norm(x) ← WRONG
///
/// This test checks what values the loaded norm weights actually have.
@Suite("Qwen3.5 Norm Convention Verification", .tags(.diagnostic), .heartbeat)
struct Qwen35NormConventionTests {

    private static let repo = "unsloth/Qwen3.5-0.8B-GGUF"
    private static let filename = "Qwen3.5-0.8B-Q4_K_M.gguf"

    @Test("Gated norm weight values reveal convention: near 0 = delta-from-1, near 1 = direct")
    func gatedNormWeightConvention() async throws {
        let downloader = HuggingFaceDownloader()
        let url = try await downloader.download(repo: Self.repo, filename: Self.filename)
        let loader = GGUFModelLoader()
        let context = try loader.loadContext(url: url)
        let model = try #require(context.model as? Qwen35Model)

        // Check gated norm weights across multiple DeltaNet layers
        let deltaNetLayers = [0, 1, 2, 4, 5, 6]
        for layerIdx in deltaNetLayers {
            let delta = try #require(model.model.layers[layerIdx].deltaNet)
            let normWeight = delta.gatedNorm.weight
            eval(normWeight)

            let values = normWeight.asType(.float32).asArray(Float.self)
            let mean = values.reduce(0, +) / Float(values.count)
            let minVal = values.min() ?? 0
            let maxVal = values.max() ?? 0
            let absMax = values.map { abs($0) }.max() ?? 0

            print("[norm-check] layer=\(layerIdx) mean=\(mean) min=\(minVal) max=\(maxVal) absMax=\(absMax)")
            print("[norm-check] layer=\(layerIdx) first8=\(Array(values.prefix(8)))")

            // If values are near 0, the convention is delta-from-1 (our code is correct)
            // If values are near 1, the convention is direct (our code has a 2x bug)
            if absMax > 0.5 {
                print("[norm-check] WARNING: layer=\(layerIdx) norm weights are NOT near zero — possible convention mismatch!")
            }
        }

        // Also check standard RMSNorm (input_layernorm, post_attention_layernorm)
        // These use MLXNN's standard RMSNorm which initializes to ones and applies directly
        let inputNorm = model.model.layers[0].inputLayerNorm
        let inputNormWeight = inputNorm.weight
        eval(inputNormWeight)
        let inValues = inputNormWeight.asType(.float32).asArray(Float.self)
        let inMean = inValues.reduce(0, +) / Float(inValues.count)
        print("[norm-check] input_layernorm.weight mean=\(inMean) first4=\(Array(inValues.prefix(4)))")
    }

    @Test("Compare gated norm output with both conventions")
    func gatedNormConventionComparison() async throws {
        let downloader = HuggingFaceDownloader()
        let url = try await downloader.download(repo: Self.repo, filename: Self.filename)
        let loader = GGUFModelLoader()
        let context = try loader.loadContext(url: url)
        let model = try #require(context.model as? Qwen35Model)
        let delta0 = try #require(model.model.layers[0].deltaNet)

        let normWeight = delta0.gatedNorm.weight
        eval(normWeight)

        // Create synthetic input
        MLXRandom.seed(42)
        let x = MLXRandom.normal([2, 128]).asType(.float32)
        let gate = MLXRandom.normal([2, 128]).asType(.float32)

        // Convention A: (1 + weight) — our current implementation
        let resultA = MLXFast.rmsNorm(x, weight: 1 + normWeight, eps: 1e-6) * silu(gate)

        // Convention B: weight directly — HuggingFace standard
        let resultB = MLXFast.rmsNorm(x, weight: normWeight, eps: 1e-6) * silu(gate)

        eval(resultA, resultB)

        let diffAB = MLX.abs(resultA - resultB)
        eval(diffAB)
        let maxDiffAB: Float = diffAB.max().item()
        let meanA: Float = MLX.abs(resultA).mean().item()
        let meanB: Float = MLX.abs(resultB).mean().item()
        let ratio = meanA / meanB

        print("[norm-compare] maxDiff(A-B)=\(maxDiffAB)")
        print("[norm-compare] mean|A|=\(meanA) mean|B|=\(meanB) ratio(A/B)=\(ratio)")
        print("[norm-compare] If ratio ≈ 2.0, convention A doubles the output (BUG)")
        print("[norm-compare] If ratio ≈ 1.0, conventions are equivalent (OK)")
    }
}
