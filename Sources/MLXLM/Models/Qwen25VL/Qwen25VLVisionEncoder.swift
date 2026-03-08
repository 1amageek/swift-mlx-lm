import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Patch Embedding

/// Extracts patches from images/video using 3D convolution.
///
/// Input: `[N, T, H, W, C]` (NDHWC) → Output: `[totalPatches, hiddenSize]`.
class Qwen25VLPatchEmbed: Module {

    @ModuleInfo(key: "proj") var proj: Conv3d

    let patchSize: Int
    let temporalPatchSize: Int
    let hiddenSize: Int

    init(_ config: Qwen25VLConfiguration.VisionConfiguration) {
        self.patchSize = config.patchSize
        self.temporalPatchSize = config.temporalPatchSize
        self.hiddenSize = config.hiddenSize

        self._proj.wrappedValue = Conv3d(
            inputChannels: config.inChannels,
            outputChannels: config.hiddenSize,
            kernelSize: IntOrTriple((config.temporalPatchSize, config.patchSize, config.patchSize)),
            stride: IntOrTriple((config.temporalPatchSize, config.patchSize, config.patchSize)),
            bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [N, T, H, W, C] (NDHWC)
        let out = proj(x)
        // out: [N, T/tp, H/ps, W/ps, hidden]
        // Flatten to [totalPatches, hidden]
        return out.reshaped(-1, hiddenSize)
    }
}

// MARK: - 2D Rotary Position Embedding (Vision)

/// Generates 2D rotary position embeddings from spatial grid positions.
struct Qwen25VLVisionRoPE {

    let dim: Int
    let theta: Float

    init(dim: Int, theta: Float = 10000.0) {
        self.dim = dim
        self.theta = theta
    }

    /// Compute rotary embeddings from grid positions.
    ///
    /// - Parameter gridTHW: Per-image/video temporal-height-width dimensions.
    /// - Returns: Frequency tensor of shape `[totalPatches, dim]` for Q/K rotation.
    func frequencies(gridTHW: [LMInput.THW], spatialMergeSize: Int) -> MLXArray {
        // Build position IDs for height and width across all images/videos
        var allFreqs = [MLXArray]()

        for thw in gridTHW {
            let t = thw.t
            let h = thw.h / spatialMergeSize
            let w = thw.w / spatialMergeSize

            // Create height and width position grids
            let hPositions = MLXArray(0..<h)
            let wPositions = MLXArray(0..<w)

            // Inverse frequencies
            let halfDim = dim / 2
            let freqExponents = MLXArray(stride(from: Float(0), to: Float(halfDim), by: 1.0))
            let invFreq = 1.0 / pow(MLXArray(theta), freqExponents / Float(halfDim))

            // Height frequencies: [h, halfDim]
            let hFreqs = hPositions.expandedDimensions(axis: 1) * invFreq.expandedDimensions(axis: 0)
            // Width frequencies: [w, halfDim]
            let wFreqs = wPositions.expandedDimensions(axis: 1) * invFreq.expandedDimensions(axis: 0)

            // Create 2D grid: [h*w, dim] by tiling h over w and w over h
            // hFreqs: repeat each row w times → [h*w, halfDim]
            let hTiled = tiled(hFreqs.expandedDimensions(axis: 1), repetitions: [1, w, 1])
                .reshaped(h * w, halfDim)
            // wFreqs: repeat entire grid h times → [h*w, halfDim]
            let wTiled = tiled(wFreqs.expandedDimensions(axis: 0), repetitions: [h, 1, 1])
                .reshaped(h * w, halfDim)

            // Concatenate height and width: [h*w, dim]
            let spatialFreqs = concatenated([hTiled, wTiled], axis: 1)

            // Tile over temporal dimension: [t*h*w, dim]
            let temporalTiled = tiled(spatialFreqs.expandedDimensions(axis: 0), repetitions: [t, 1, 1])
                .reshaped(t * h * w, dim)

            allFreqs.append(temporalTiled)
        }

        return concatenated(allFreqs, axis: 0)
    }

    /// Apply rotary embedding to queries or keys.
    ///
    /// - Parameters:
    ///   - x: Input tensor of shape `[totalPatches, numHeads, headDim]`.
    ///   - freqs: Frequency tensor of shape `[totalPatches, rotateDim]`.
    /// - Returns: Rotated tensor with same shape as input.
    static func apply(_ x: MLXArray, freqs: MLXArray) -> MLXArray {
        let rotateDim = freqs.dim(-1)
        let cosFreqs = cos(freqs)
        let sinFreqs = sin(freqs)

        // Split x into rotated and passthrough parts
        let xRot = x[0..., 0..., ..<rotateDim]
        let xPass = x[0..., 0..., rotateDim...]

        // Rotate-half: [-x2, x1] pattern
        let half = rotateDim / 2
        let x1 = xRot[0..., 0..., ..<half]
        let x2 = xRot[0..., 0..., half...]
        let rotated = concatenated([-x2, x1], axis: -1)

        let cosExpanded = cosFreqs.expandedDimensions(axis: 1)
        let sinExpanded = sinFreqs.expandedDimensions(axis: 1)

        let xRotated = xRot * cosExpanded + rotated * sinExpanded
        return concatenated([xRotated, xPass], axis: -1)
    }
}

// MARK: - Vision Attention

/// Multi-head attention for vision encoder with fused QKV and 2D-RoPE.
class Qwen25VLVisionAttention: Module {

    let numHeads: Int
    let headDim: Int

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var oProj: Linear

    init(_ config: Qwen25VLConfiguration.VisionConfiguration) {
        self.numHeads = config.numHeads
        self.headDim = config.headDim
        let dim = config.hiddenSize

        self._qkv.wrappedValue = Linear(dim, dim * 3, bias: true)
        self._oProj.wrappedValue = Linear(dim, dim, bias: true)
    }

    /// Forward pass.
    ///
    /// - Parameters:
    ///   - x: Hidden states `[seqLen, dim]`.
    ///   - rotaryFreqs: Precomputed 2D-RoPE frequencies `[seqLen, rotateDim]`.
    ///   - mask: Attention mask `[1, 1, seqLen, seqLen]` or `nil`.
    func callAsFunction(_ x: MLXArray, rotaryFreqs: MLXArray, mask: MLXArray?) -> MLXArray {
        let seqLen = x.dim(0)

        // Fused QKV projection
        let qkvOut = qkv(x)
        let split = qkvOut.split(parts: 3, axis: -1)
        var q = split[0].reshaped(seqLen, numHeads, headDim)
        var k = split[1].reshaped(seqLen, numHeads, headDim)
        let v = split[2].reshaped(seqLen, numHeads, headDim)

        // Apply 2D-RoPE
        q = Qwen25VLVisionRoPE.apply(q, freqs: rotaryFreqs)
        k = Qwen25VLVisionRoPE.apply(k, freqs: rotaryFreqs)

        // Reshape for attention: [1, seqLen, numHeads, headDim] → SDPA expects [B, H, S, D]
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

        // [1, numHeads, seqLen, headDim] → [seqLen, dim]
        let out = attnOut.transposed(0, 2, 1, 3).reshaped(seqLen, -1)
        return oProj(out)
    }
}

// MARK: - Vision MLP (SwiGLU)

/// SwiGLU feed-forward network for vision encoder blocks.
class Qwen25VLVisionMLP: Module {

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: Qwen25VLConfiguration.VisionConfiguration) {
        let dim = config.hiddenSize
        let ffnDim = config.intermediateSize

        self._gateProj.wrappedValue = Linear(dim, ffnDim, bias: true)
        self._upProj.wrappedValue = Linear(dim, ffnDim, bias: true)
        self._downProj.wrappedValue = Linear(ffnDim, dim, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Vision Block

/// A single transformer block in the vision encoder.
class Qwen25VLVisionBlock: Module {

    @ModuleInfo var norm1: RMSNorm
    @ModuleInfo var attn: Qwen25VLVisionAttention
    @ModuleInfo var norm2: RMSNorm
    @ModuleInfo var mlp: Qwen25VLVisionMLP

    init(_ config: Qwen25VLConfiguration.VisionConfiguration) {
        self._norm1.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        self._attn.wrappedValue = Qwen25VLVisionAttention(config)
        self._norm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        self._mlp.wrappedValue = Qwen25VLVisionMLP(config)
    }

    func callAsFunction(_ x: MLXArray, rotaryFreqs: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x + attn(norm1(x), rotaryFreqs: rotaryFreqs, mask: mask)
        h = h + mlp(norm2(h))
        return h
    }
}

// MARK: - Patch Merger

/// Merges 2×2 adjacent patches and projects to LLM hidden dimension.
///
/// Reduces token count by 4× while aligning vision feature dimension
/// with the language model's hidden size.
class Qwen25VLPatchMerger: Module {

    let spatialMergeSize: Int
    let hiddenSize: Int

    @ModuleInfo(key: "ln_q") var lnQ: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: MergerMLP

    init(_ config: Qwen25VLConfiguration.VisionConfiguration) {
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

        // Group 2x2 adjacent patches and concatenate features
        var merged = [MLXArray]()
        var offset = 0

        for thw in gridTHW {
            let t = thw.t
            let h = thw.h / spatialMergeSize
            let w = thw.w / spatialMergeSize
            let patchCount = t * thw.h * thw.w

            // Extract patches for this image/video
            let segment = normalized[offset..<(offset + patchCount)]

            // Reshape to spatial grid: [t, h*merge, w*merge, hidden]
            let spatial = segment.reshaped(t, thw.h, thw.w, hiddenSize)

            // Reshape to merge 2x2: [t, h, merge, w, merge, hidden]
            let grouped = spatial.reshaped(t, h, spatialMergeSize, w, spatialMergeSize, hiddenSize)

            // Transpose and flatten merge dims: [t, h, w, merge*merge*hidden]
            let transposed = grouped.transposed(0, 1, 3, 2, 4, 5)
            let flat = transposed.reshaped(t * h * w, spatialMergeSize * spatialMergeSize * hiddenSize)

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

/// Complete Qwen2.5-VL vision encoder.
///
/// Processes image/video pixels through patch embedding, 32 transformer blocks
/// with window/full attention, and a patch merger that reduces token count 4×.
class Qwen25VLVisionTransformer: Module, VisionEncoder {

    let config: Qwen25VLConfiguration.VisionConfiguration
    let rotaryEmb: Qwen25VLVisionRoPE

    @ModuleInfo(key: "patch_embed") var patchEmbed: Qwen25VLPatchEmbed
    @ModuleInfo(key: "blocks") var blocks: [Qwen25VLVisionBlock]
    @ModuleInfo(key: "merger") var merger: Qwen25VLPatchMerger

    init(_ config: Qwen25VLConfiguration.VisionConfiguration) {
        self.config = config
        self.rotaryEmb = Qwen25VLVisionRoPE(dim: config.headDim / 2)

        self._patchEmbed.wrappedValue = Qwen25VLPatchEmbed(config)
        self._blocks.wrappedValue = (0..<config.depth).map { _ in Qwen25VLVisionBlock(config) }
        self._merger.wrappedValue = Qwen25VLPatchMerger(config)
    }

    func callAsFunction(_ pixels: MLXArray, gridTHW: [LMInput.THW]) -> MLXArray {
        // 1. Patch embedding
        var hiddenStates = patchEmbed(pixels)

        // 2. Compute 2D rotary embeddings
        let rotaryFreqs = rotaryEmb.frequencies(
            gridTHW: gridTHW, spatialMergeSize: config.spatialMergeSize
        )

        // 3. Compute attention masks
        let cuSeqlens = computeCuSeqlens(gridTHW: gridTHW)
        let fullMask = buildAttentionMask(cuSeqlens: cuSeqlens, seqLen: hiddenStates.dim(0))

        // Window attention masks (for non-full-attention layers)
        let windowInfo = computeWindowInfo(gridTHW: gridTHW)

        // 4. Run through transformer blocks
        for (i, block) in blocks.enumerated() {
            let useFullAttention = config.fullAttBlockIndexes.contains(i)
            let mask = useFullAttention ? fullMask : windowInfo.mask
            let freqs = useFullAttention ? rotaryFreqs : windowInfo.rotaryFreqs

            if !useFullAttention, let windowIndex = windowInfo.index {
                // Reorder patches by window
                let reordered = hiddenStates.take(windowIndex, axis: 0)
                let reorderedOut = block(reordered, rotaryFreqs: freqs, mask: mask)
                // Reverse reorder
                let reverseIndex = windowInfo.reverseIndex!
                hiddenStates = reorderedOut.take(reverseIndex, axis: 0)
            } else {
                hiddenStates = block(hiddenStates, rotaryFreqs: freqs, mask: mask)
            }
        }

        // 5. Merge patches (4× token reduction)
        return merger(hiddenStates, gridTHW: gridTHW)
    }

    // MARK: - Attention Mask Helpers

    /// Cumulative sequence lengths for multi-image batching.
    private func computeCuSeqlens(gridTHW: [LMInput.THW]) -> [Int] {
        var cuSeqlens = [0]
        for thw in gridTHW {
            let patchCount = thw.t * thw.h * thw.w
            cuSeqlens.append(cuSeqlens.last! + patchCount)
        }
        return cuSeqlens
    }

    /// Build block-diagonal attention mask from cumulative sequence lengths.
    private func buildAttentionMask(cuSeqlens: [Int], seqLen: Int) -> MLXArray? {
        if cuSeqlens.count <= 2 {
            return nil  // Single sequence, no mask needed
        }

        // Block-diagonal: allow attention only within each sequence
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

    /// Window attention information for non-full-attention layers.
    private struct WindowInfo {
        let mask: MLXArray?
        let rotaryFreqs: MLXArray
        let index: MLXArray?
        let reverseIndex: MLXArray?
    }

    private func computeWindowInfo(gridTHW: [LMInput.THW]) -> WindowInfo {
        let patchesPerWindow = config.windowSize / config.patchSize
        var windowIndices = [Int32]()
        var windowCuSeqlens = [0]
        var allRotaryFreqs = [MLXArray]()

        let spatialMerge = config.spatialMergeSize
        var globalOffset = 0

        for thw in gridTHW {
            let t = thw.t
            let h = thw.h
            let w = thw.w
            let patchCount = t * h * w

            // Compute padded dimensions for windowing
            let hPad = ((h + patchesPerWindow - 1) / patchesPerWindow) * patchesPerWindow
            let wPad = ((w + patchesPerWindow - 1) / patchesPerWindow) * patchesPerWindow

            // Number of windows in each dimension
            let numWinH = hPad / patchesPerWindow
            let numWinW = wPad / patchesPerWindow

            for tIdx in 0..<t {
                for wh in 0..<numWinH {
                    for ww in 0..<numWinW {
                        var windowPatchIndices = [Int32]()
                        for ph in 0..<patchesPerWindow {
                            for pw in 0..<patchesPerWindow {
                                let hi = wh * patchesPerWindow + ph
                                let wi = ww * patchesPerWindow + pw
                                if hi < h && wi < w {
                                    let idx = globalOffset + tIdx * h * w + hi * w + wi
                                    windowPatchIndices.append(Int32(idx))
                                }
                            }
                        }
                        if !windowPatchIndices.isEmpty {
                            windowIndices.append(contentsOf: windowPatchIndices)
                            windowCuSeqlens.append(
                                windowCuSeqlens.last! + windowPatchIndices.count
                            )
                        }
                    }
                }
            }

            // Compute rotary frequencies for this image/video
            let invFreqDim = config.headDim / 4  // half of dim
            let freqExponents = MLXArray(
                stride(from: Float(0), to: Float(invFreqDim), by: 1.0)
            )
            let invFreq = 1.0 / pow(MLXArray(Float(10000)), freqExponents / Float(invFreqDim))

            let hMerged = h / spatialMerge
            let wMerged = w / spatialMerge

            let hPos = MLXArray(0..<hMerged)
            let wPos = MLXArray(0..<wMerged)

            let hFreqs = hPos.expandedDimensions(axis: 1) * invFreq.expandedDimensions(axis: 0)
            let wFreqs = wPos.expandedDimensions(axis: 1) * invFreq.expandedDimensions(axis: 0)

            let hTiled = tiled(hFreqs.expandedDimensions(axis: 1), repetitions: [1, wMerged, 1])
                .reshaped(hMerged * wMerged, invFreqDim)
            let wTiled = tiled(wFreqs.expandedDimensions(axis: 0), repetitions: [hMerged, 1, 1])
                .reshaped(hMerged * wMerged, invFreqDim)

            let spatialFreqs = concatenated([hTiled, wTiled], axis: 1)
            let temporalTiled = tiled(spatialFreqs.expandedDimensions(axis: 0), repetitions: [t, 1, 1])
                .reshaped(patchCount, config.headDim / 2)

            allRotaryFreqs.append(temporalTiled)
            globalOffset += patchCount
        }

        let totalPatches = globalOffset

        // Build window attention mask
        let windowMask = buildAttentionMask(
            cuSeqlens: windowCuSeqlens, seqLen: windowIndices.count
        )

        // Build reorder index
        let indexArray = MLXArray(windowIndices)

        // Build reverse index
        var reverseArray = [Int32](repeating: 0, count: totalPatches)
        for (newIdx, origIdx) in windowIndices.enumerated() {
            reverseArray[Int(origIdx)] = Int32(newIdx)
        }
        let reverseIndexArray = MLXArray(reverseArray)

        // Reorder rotary freqs
        let fullFreqs = concatenated(allRotaryFreqs, axis: 0)
        let reorderedFreqs = fullFreqs.take(indexArray, axis: 0)

        return WindowInfo(
            mask: windowMask,
            rotaryFreqs: reorderedFreqs,
            index: indexArray,
            reverseIndex: reverseIndexArray
        )
    }
}
