import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Patch Embedding (Conv2d)

/// Extracts patches from images using 2D convolution.
///
/// Input: `[N, H, W, C]` (NHWC) -> Output: `[totalPatches, hiddenSize]`.
class SpatialVisionPatchEmbedding: Module {

    @ModuleInfo(key: "proj") var proj: Conv2d

    let patchSize: Int
    let hiddenSize: Int

    init(_ config: VisionConfig) {
        self.patchSize = config.patchSize
        self.hiddenSize = config.hiddenSize

        self._proj.wrappedValue = Conv2d(
            inputChannels: config.inChannels,
            outputChannels: config.hiddenSize,
            kernelSize: IntOrPair((config.patchSize, config.patchSize)),
            stride: IntOrPair((config.patchSize, config.patchSize)),
            bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [N, H, W, C] (NHWC)
        let out = proj(x)
        // out: [N, H/ps, W/ps, hidden]
        return out.reshaped(-1, hiddenSize)
    }
}

// MARK: - Vision Attention

/// Multi-head attention for vision encoder with fused QKV and 2D-RoPE.
class FullAttentionVisionAttention: Module {

    let numHeads: Int
    let headDim: Int

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var oProj: Linear

    init(_ config: VisionConfig) {
        self.numHeads = config.numHeads
        self.headDim = config.headDim
        let dim = config.hiddenSize

        self._qkv.wrappedValue = Linear(dim, dim * 3, bias: true)
        self._oProj.wrappedValue = Linear(dim, dim, bias: true)
    }

    func callAsFunction(_ x: MLXArray, rotaryFreqs: MLXArray, mask: MLXArray?) -> MLXArray {
        let seqLen = x.dim(0)

        let qkvOut = qkv(x)
        let split = qkvOut.split(parts: 3, axis: -1)
        var q = split[0].reshaped(seqLen, numHeads, headDim)
        var k = split[1].reshaped(seqLen, numHeads, headDim)
        let v = split[2].reshaped(seqLen, numHeads, headDim)

        // Apply 2D-RoPE (reuse algorithm from Qwen2.5-VL)
        q = VisionRoPE2D.apply(q, freqs: rotaryFreqs)
        k = VisionRoPE2D.apply(k, freqs: rotaryFreqs)

        // Reshape for SDPA: [1, H, S, D]
        let qT = q.expandedDimensions(axis: 0).transposed(0, 2, 1, 3)
        let kT = k.expandedDimensions(axis: 0).transposed(0, 2, 1, 3)
        let vT = v.expandedDimensions(axis: 0).transposed(0, 2, 1, 3)

        let scale = pow(Float(headDim), -0.5)

        let attnOut: MLXArray
        if let mask {
            attnOut = MLXFast.scaledDotProductAttention(
                queries: qT, keys: kT, values: vT,
                scale: scale, mask: mask
            )
        } else {
            attnOut = MLXFast.scaledDotProductAttention(
                queries: qT, keys: kT, values: vT,
                scale: scale, mask: .none
            )
        }

        let out = attnOut.transposed(0, 2, 1, 3).reshaped(seqLen, -1)
        return oProj(out)
    }
}

// MARK: - Vision MLP (GELU)

/// GELU feed-forward network for Qwen 3.5 vision encoder.
///
/// Uses standard fc1 + GELU + fc2 (NOT SwiGLU like Qwen2.5-VL).
class GELUVisionMLP: Module {

    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ config: VisionConfig) {
        let dim = config.hiddenSize
        let ffnDim = config.intermediateSize

        self._fc1.wrappedValue = Linear(dim, ffnDim, bias: true)
        self._fc2.wrappedValue = Linear(ffnDim, dim, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(gelu(fc1(x)))
    }
}

// MARK: - Vision Block

/// A single transformer block in the Qwen 3.5 vision encoder.
class FullAttentionVisionBlock: Module {

    @ModuleInfo var norm1: RMSNorm
    @ModuleInfo var attn: FullAttentionVisionAttention
    @ModuleInfo var norm2: RMSNorm
    @ModuleInfo var mlp: GELUVisionMLP

    init(_ config: VisionConfig) {
        self._norm1.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        self._attn.wrappedValue = FullAttentionVisionAttention(config)
        self._norm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        self._mlp.wrappedValue = GELUVisionMLP(config)
    }

    func callAsFunction(_ x: MLXArray, rotaryFreqs: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x + attn(norm1(x), rotaryFreqs: rotaryFreqs, mask: mask)
        h = h + mlp(norm2(h))
        return h
    }
}

// MARK: - Patch Merger

/// Merges 2x2 adjacent patches and projects to LLM hidden dimension.
class FullAttentionVisionPatchMerger: Module {

    let spatialMergeSize: Int
    let hiddenSize: Int

    @ModuleInfo(key: "ln_q") var lnQ: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: MergerMLP

    init(_ config: VisionConfig) {
        self.spatialMergeSize = config.spatialMergeSize
        self.hiddenSize = config.hiddenSize

        let mergedDim = config.hiddenSize * config.spatialMergeSize * config.spatialMergeSize
        self._lnQ.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        self._mlp.wrappedValue = MergerMLP(
            inputDim: mergedDim, hiddenDim: mergedDim, outputDim: config.outHiddenSize
        )
    }

    func callAsFunction(_ x: MLXArray, gridTHW: [LMInput.THW]) -> MLXArray {
        let normalized = lnQ(x)

        var merged = [MLXArray]()
        var offset = 0

        for thw in gridTHW {
            let t = thw.t
            let h = thw.h / spatialMergeSize
            let w = thw.w / spatialMergeSize
            let patchCount = t * thw.h * thw.w

            let segment = normalized[offset..<(offset + patchCount)]
            let spatial = segment.reshaped(t, thw.h, thw.w, hiddenSize)
            let grouped = spatial.reshaped(t, h, spatialMergeSize, w, spatialMergeSize, hiddenSize)
            let transposed = grouped.transposed(0, 1, 3, 2, 4, 5)
            let flat = transposed.reshaped(
                t * h * w, spatialMergeSize * spatialMergeSize * hiddenSize)

            merged.append(flat)
            offset += patchCount
        }

        let mergedTensor = concatenated(merged, axis: 0)
        return mlp(mergedTensor)
    }

    /// Two-layer MLP with GELU activation for the patch merger.
    class MergerMLP: Module {

        @ModuleInfo(key: "0") var linear1: Linear
        @ModuleInfo(key: "2") var linear2: Linear

        init(inputDim: Int, hiddenDim: Int, outputDim: Int) {
            self._linear1.wrappedValue = Linear(inputDim, hiddenDim, bias: true)
            self._linear2.wrappedValue = Linear(hiddenDim, outputDim, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            linear2(gelu(linear1(x)))
        }
    }
}

// MARK: - Vision Transformer

/// Complete Qwen 3.5 vision encoder.
///
/// Processes image pixels through Conv2d patch embedding, 12 transformer blocks
/// with full attention (no windowing), and a patch merger that reduces token count 4x.
class FullAttentionVisionTransformer: Module, VisionEncoder {

    let config: VisionConfig
    let rotaryEmb: VisionRoPE2D

    @ModuleInfo(key: "patch_embed") var patchEmbed: SpatialVisionPatchEmbedding
    @ModuleInfo(key: "blocks") var blocks: [FullAttentionVisionBlock]
    @ModuleInfo(key: "merger") var merger: FullAttentionVisionPatchMerger

    init(_ config: VisionConfig) {
        self.config = config
        self.rotaryEmb = VisionRoPE2D(dim: config.headDim / 2)

        self._patchEmbed.wrappedValue = SpatialVisionPatchEmbedding(config)
        self._blocks.wrappedValue = (0..<config.depth).map { _ in FullAttentionVisionBlock(config) }
        self._merger.wrappedValue = FullAttentionVisionPatchMerger(config)
    }

    func callAsFunction(_ pixels: MLXArray, gridTHW: [LMInput.THW]) -> MLXArray {
        // 1. Patch embedding
        var hiddenStates = patchEmbed(pixels)

        // 2. Compute 2D rotary embeddings
        let rotaryFreqs = rotaryEmb.frequencies(gridTHW: gridTHW)

        // 3. Compute attention mask for multi-image batching
        let cuSeqlens = computeCuSeqlens(gridTHW: gridTHW)
        let mask = buildAttentionMask(cuSeqlens: cuSeqlens, seqLen: hiddenStates.dim(0))

        // 4. All blocks use full attention (no windowing)
        for block in blocks {
            hiddenStates = block(hiddenStates, rotaryFreqs: rotaryFreqs, mask: mask)
        }

        // 5. Merge patches (4x token reduction)
        return merger(hiddenStates, gridTHW: gridTHW)
    }

    // MARK: - Attention Mask Helpers

    private func computeCuSeqlens(gridTHW: [LMInput.THW]) -> [Int] {
        var cuSeqlens = [0]
        for thw in gridTHW {
            let patchCount = thw.t * thw.h * thw.w
            cuSeqlens.append(cuSeqlens.last! + patchCount)
        }
        return cuSeqlens
    }

    private func buildAttentionMask(cuSeqlens: [Int], seqLen: Int) -> MLXArray? {
        if cuSeqlens.count <= 2 {
            return nil
        }

        var maskArray = [Float](repeating: Float(-1e9), count: seqLen * seqLen)
        for s in 0..<(cuSeqlens.count - 1) {
            let start = cuSeqlens[s]
            let end = cuSeqlens[s + 1]
            for i in start..<end {
                for j in start..<end {
                    maskArray[i * seqLen + j] = 0
                }
            }
        }
        return MLXArray(maskArray, [1, 1, seqLen, seqLen])
    }
}
