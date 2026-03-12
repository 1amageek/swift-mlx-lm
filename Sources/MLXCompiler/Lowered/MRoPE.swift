@preconcurrency import MLX
import MLXFast
import SwiftLM

// MARK: - M-RoPE (Multi-axis Rotary Position Embedding)

/// Apply contiguous M-RoPE (Qwen 2.5-VL pattern).
///
/// Sections are laid out contiguously: the first section covers dims 0..<2*section[0],
/// the second covers the next 2*section[1] dims, etc. Each section uses positions
/// from one of the 3 axes (temporal, height, width).
///
/// - Parameters:
///   - x: Input tensor `[B, H, L, D]`
///   - positionIds: Per-axis positions `[3, B, L]`
///   - ropeBase: Theta for frequency computation
///   - sections: Half-dimension allocation per axis
///   - headDim: Full head dimension
/// - Returns: Rotated tensor `[B, H, L, D]`
func applyContiguousMRoPE(
    _ x: MLXArray, positionIds: MLXArray,
    ropeBase: Float, sections: [Int], headDim: Int
) -> MLXArray {
    let halfDim = headDim / 2
    let freqExponents = MLXArray(stride(from: Float(0), to: Float(halfDim), by: 1.0))
    let invFreq = 1.0 / pow(MLXArray(ropeBase), freqExponents / Float(halfDim))

    var cosPerAxis = [MLXArray]()
    var sinPerAxis = [MLXArray]()

    for axis in 0..<3 {
        let axisPositions = positionIds[axis]
        let freqs = expandedDimensions(axisPositions, axis: -1).asType(DType.float32)
            * invFreq.reshaped(1, 1, halfDim)
        let emb = concatenated([freqs, freqs], axis: -1)
        cosPerAxis.append(cos(emb))
        sinPerAxis.append(sin(emb))
    }

    let doubledSections = sections.map { $0 * 2 }
    var cosChunks = [MLXArray]()
    var sinChunks = [MLXArray]()
    var dimOffset = 0

    for (i, sectionDim) in doubledSections.enumerated() {
        let axisIdx = i % 3
        cosChunks.append(cosPerAxis[axisIdx][0..., 0..., dimOffset..<(dimOffset + sectionDim)])
        sinChunks.append(sinPerAxis[axisIdx][0..., 0..., dimOffset..<(dimOffset + sectionDim)])
        dimOffset += sectionDim
    }

    if dimOffset < headDim {
        let remaining = headDim - dimOffset
        let axisIdx = doubledSections.count % 3
        cosChunks.append(cosPerAxis[axisIdx][0..., 0..., dimOffset..<(dimOffset + remaining)])
        sinChunks.append(sinPerAxis[axisIdx][0..., 0..., dimOffset..<(dimOffset + remaining)])
    }

    var cosEmb = concatenated(cosChunks, axis: -1)
    var sinEmb = concatenated(sinChunks, axis: -1)
    cosEmb = cosEmb.expandedDimensions(axis: 1)
    sinEmb = sinEmb.expandedDimensions(axis: 1)

    let half = headDim / 2
    let x1 = x[0..., 0..., 0..., ..<half]
    let x2 = x[0..., 0..., 0..., half...]
    let rotateHalf = concatenated([-x2, x1], axis: -1)

    return x * cosEmb + rotateHalf * sinEmb
}

/// Apply interleaved M-RoPE (Qwen 3.5-VL pattern).
///
/// Sections are interleaved across axes. Only applies to the first
/// `ropeDim` dimensions; the remainder passes through unchanged.
///
/// - Parameters:
///   - x: Input tensor `[B, H, L, D]`
///   - positionIds: Per-axis positions `[3, B, L]`
///   - ropeDim: Partial rotary dimension (e.g. 64 of 256)
///   - ropeBase: Theta for frequency computation
///   - sections: Half-dimension allocation per axis (e.g. [11, 11, 10])
///   - headDim: Full head dimension
/// - Returns: Rotated tensor `[B, H, L, D]`
func applyInterleavedMRoPE(
    _ x: MLXArray, positionIds: MLXArray,
    ropeDim: Int, ropeBase: Float, sections: [Int], headDim: Int
) -> MLXArray {
    let halfRpd = ropeDim / 2
    let freqExponents = MLXArray(stride(from: Float(0), to: Float(halfRpd), by: 1.0))
    let invFreq = 1.0 / pow(MLXArray(ropeBase), freqExponents / Float(halfRpd))

    var axisFreqs = [MLXArray]()
    for axis in 0..<3 {
        let positions = positionIds[axis].asType(DType.float32)
        let f = expandedDimensions(positions, axis: -1) * invFreq.reshaped(1, 1, halfRpd)
        axisFreqs.append(f)
    }

    var cosSlices = [MLXArray]()
    var sinSlices = [MLXArray]()
    var dimOffset = 0

    for (sectionIdx, sectionSize) in sections.enumerated() {
        let axisIdx = sectionIdx % 3
        let slice = axisFreqs[axisIdx][0..., 0..., dimOffset..<(dimOffset + sectionSize)]
        cosSlices.append(cos(slice))
        sinSlices.append(sin(slice))
        dimOffset += sectionSize
    }

    let cosHalf = concatenated(cosSlices, axis: -1)
    let sinHalf = concatenated(sinSlices, axis: -1)
    let cosEmb = concatenated([cosHalf, cosHalf], axis: -1)
    let sinEmb = concatenated([sinHalf, sinHalf], axis: -1)

    let xRot = x[0..., 0..., 0..., ..<ropeDim]
    let xPass = x[0..., 0..., 0..., ropeDim...]

    let cos4d = cosEmb.expandedDimensions(axis: 1)
    let sin4d = sinEmb.expandedDimensions(axis: 1)

    let half = ropeDim / 2
    let x1 = xRot[0..., 0..., 0..., ..<half]
    let x2 = xRot[0..., 0..., 0..., half...]
    let rotated = concatenated([-x2, x1], axis: -1)

    let xRotated = xRot * cos4d + rotated * sin4d
    return concatenated([xRotated, xPass], axis: -1)
}
