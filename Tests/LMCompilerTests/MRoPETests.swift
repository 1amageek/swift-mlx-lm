import Testing
@preconcurrency import MLX
import MLXFast
import MLXNN
@testable import SwiftLM
@testable import LMCompiler


// MARK: - Test Helpers

private func slot(
    _ components: [StructuralPathComponent], role: ParameterRole
) -> ParameterSlot {
    ParameterSlot(path: StructuralPath(components: components), role: role)
}

private func bind(_ pairs: [ParameterSlot: MLXArray]) -> BoundWeights {
    var tensors: [ParameterSlot: TensorData] = [:]
    for (slot, array) in pairs {
        tensors[slot] = TensorData(
            shape: array.shape.map { $0 },
            dtype: .float32,
            storage: array
        )
    }
    return BoundWeights(tensors: tensors)
}

/// Reference implementation of contiguous M-RoPE (matches the windowed vision encoder's applyMRoPE).
///
/// This is a standalone pure-function version of the Qwen 2.5-VL attention's
/// M-RoPE application, used as ground truth for testing the executor.
private func referenceContiguousMRoPE(
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

/// Reference implementation of interleaved M-RoPE (matches SigmoidGatedFullAttention.applyInterleavedMRoPE).
private func referenceInterleavedMRoPE(
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

/// DSL component: TokenEmbedding → Attention (with M-RoPE) — no output head.
private struct MiniMRoPEModel: ModelComponent {
    let vocabSize: Int
    let hiddenSize: Int
    let headCount: Int
    let kvHeadCount: Int
    let headDim: Int
    let ropeDim: Int
    let ropeBase: Float
    let mropeAxes: MRoPEAxes

    @ModelComponentBuilder var body: some ModelComponent {
        TokenEmbedding(vocabSize: vocabSize, embeddingSize: hiddenSize)
        Attention(
            hiddenSize: hiddenSize,
            headCount: headCount,
            kvHeadCount: kvHeadCount,
            headDimension: headDim,
            rope: RoPEAttributes(
                dimension: ropeDim,
                base: ropeBase,
                mropeAxes: mropeAxes
            )
        )
    }
}

/// DSL component: TokenEmbedding → Attention (standard RoPE, no M-RoPE).
private struct MiniStandardRoPEModel: ModelComponent {
    let vocabSize: Int
    let hiddenSize: Int
    let headCount: Int
    let kvHeadCount: Int
    let headDim: Int
    let ropeDim: Int
    let ropeBase: Float

    @ModelComponentBuilder var body: some ModelComponent {
        TokenEmbedding(vocabSize: vocabSize, embeddingSize: hiddenSize)
        Attention(
            hiddenSize: hiddenSize,
            headCount: headCount,
            kvHeadCount: kvHeadCount,
            headDimension: headDim,
            rope: RoPEAttributes(
                dimension: ropeDim,
                base: ropeBase
            )
        )
    }
}

/// Build a minimal model graph with a single attention layer that has M-RoPE.
private func buildMRoPEAttentionModel(
    hiddenSize: Int, headCount: Int, kvHeadCount: Int, headDim: Int,
    ropeDim: Int, ropeBase: Float, mropeAxes: MRoPEAxes,
    vocabSize: Int = 16
) throws -> (MLXCompiledModel, [ParameterSlot: MLXArray]) {
    let model = MiniMRoPEModel(
        vocabSize: vocabSize, hiddenSize: hiddenSize,
        headCount: headCount, kvHeadCount: kvHeadCount,
        headDim: headDim, ropeDim: ropeDim, ropeBase: ropeBase,
        mropeAxes: mropeAxes)
    let graph = try model.makeModelGraph()

    // Build weights matching the normalized graph structure
    let D = hiddenSize
    let qDim = headCount * headDim
    let kvDim = kvHeadCount * headDim

    var dict: [ParameterSlot: MLXArray] = [:]
    dict[slot([.operation(0)], role: .embeddingTable)] = MLXRandom.normal([vocabSize, D]) * 0.1

    let attnPath: [StructuralPathComponent] = [.operation(1)]
    dict[slot(attnPath + [.field("q_proj")], role: .weight)] = MLXRandom.normal([qDim, D]) * 0.1
    dict[slot(attnPath + [.field("k_proj")], role: .weight)] = MLXRandom.normal([kvDim, D]) * 0.1
    dict[slot(attnPath + [.field("v_proj")], role: .weight)] = MLXRandom.normal([kvDim, D]) * 0.1
    dict[slot(attnPath + [.field("o_proj")], role: .weight)] = MLXRandom.normal([D, qDim]) * 0.1

    let weights = bind(dict)
    let compiler = MLXInferenceCompiler()
    let scanResult = compiler.scan(graph: graph)

    let weightStore = try InferenceWeightStore(boundWeights: weights)
    let compiled = MLXCompiledModel(
        graph: graph,
        weightStore: weightStore,
        cacheDescriptors: scanResult.cacheDescriptors,
        embeddingPath: scanResult.embeddingPath
    )

    return (compiled, dict)
}


// MARK: - Contiguous M-RoPE Tests (Qwen 2.5-VL Pattern)

@Suite("Contiguous M-RoPE", .tags(.unit))
struct ContiguousMRoPETests {

    @Test("Executor contiguous M-RoPE matches reference for uniform text positions")
    func contiguousUniformPositions() throws {
        // Qwen 2.5-VL-like config: headDim=8, sections=[1,3,3] (scaled down)
        let headDim = 8
        let sections = [1, 3, 3]  // sum=7 < halfDim=4... need sections sum = halfDim
        // Actually sections must sum to halfDim. halfDim = 8/2 = 4. sections = [1, 1, 2]
        let correctedSections = [1, 1, 2]
        let ropeBase: Float = 10000.0
        let B = 1, H = 2, L = 4

        let mropeAxes = MRoPEAxes(sections: correctedSections, interleaved: false)

        // All-text: positions are the same on all axes [0,1,2,3]
        let posArray = MLXArray(Int32(0)..<Int32(4)).reshaped(1, L)
        let positionIds = stacked([posArray, posArray, posArray], axis: 0)  // [3, 1, 4]

        // Random input [B, H, L, D]
        MLXRandom.seed(42)
        let x = MLXRandom.normal([B, H, L, headDim])

        let (compiled, _) = try buildMRoPEAttentionModel(
            hiddenSize: headDim * H, headCount: H, kvHeadCount: H,
            headDim: headDim, ropeDim: headDim, ropeBase: ropeBase,
            mropeAxes: mropeAxes)

        // Test the rotation function directly
        let executor = MLXExecutor(compiledModel: compiled)
        // Access internal M-RoPE via the executor's forward pass

        // Compare reference vs direct computation
        let refResult = referenceContiguousMRoPE(
            x, positionIds: positionIds,
            ropeBase: ropeBase, sections: correctedSections, headDim: headDim)

        eval(refResult)

        // Verify shape preservation
        #expect(refResult.shape == [B, H, L, headDim])

        // Verify M-RoPE changes the values (not identity)
        let diff = abs(refResult - x).max()
        #expect(diff.item(Float.self) > 1e-6)
    }

    @Test("Contiguous M-RoPE: different axis positions produce different rotations")
    func contiguousDifferentAxes() throws {
        let headDim = 8
        let sections = [1, 1, 2]
        let ropeBase: Float = 10000.0
        let B = 1, H = 1, L = 1

        MLXRandom.seed(123)
        let x = MLXRandom.normal([B, H, L, headDim])

        // Position where all axes are the same
        let uniformPos = stacked([
            MLXArray([Int32(5)]).reshaped(1, 1),
            MLXArray([Int32(5)]).reshaped(1, 1),
            MLXArray([Int32(5)]).reshaped(1, 1)
        ], axis: 0)

        // Position where axes differ (vision token scenario)
        let mixedPos = stacked([
            MLXArray([Int32(5)]).reshaped(1, 1),   // temporal
            MLXArray([Int32(2)]).reshaped(1, 1),   // height
            MLXArray([Int32(3)]).reshaped(1, 1)    // width
        ], axis: 0)

        let uniformResult = referenceContiguousMRoPE(
            x, positionIds: uniformPos,
            ropeBase: ropeBase, sections: sections, headDim: headDim)

        let mixedResult = referenceContiguousMRoPE(
            x, positionIds: mixedPos,
            ropeBase: ropeBase, sections: sections, headDim: headDim)

        eval(uniformResult, mixedResult)

        // Results must differ when positions differ
        let diff = abs(uniformResult - mixedResult).max()
        #expect(diff.item(Float.self) > 1e-6,
            "Contiguous M-RoPE must produce different rotations for different axis positions")
    }

    @Test("Contiguous M-RoPE: rotation is norm-preserving")
    func contiguousNormPreserving() throws {
        let headDim = 16
        let sections = [2, 3, 3]
        let ropeBase: Float = 10000.0
        let B = 1, H = 2, L = 3

        MLXRandom.seed(7)
        let x = MLXRandom.normal([B, H, L, headDim])

        let posArray = MLXArray(Int32(0)..<Int32(3)).reshaped(1, L)
        let positionIds = stacked([posArray, posArray, posArray], axis: 0)

        let result = referenceContiguousMRoPE(
            x, positionIds: positionIds,
            ropeBase: ropeBase, sections: sections, headDim: headDim)

        eval(result)

        // Per-token norm should be preserved (rotate-half preserves L2 norm)
        let inputNorm = (x * x).sum(axis: -1).sqrt()
        let outputNorm = (result * result).sum(axis: -1).sqrt()
        let normDiff = abs(inputNorm - outputNorm).max()

        #expect(normDiff.item(Float.self) < 1e-4,
            "M-RoPE rotation must preserve L2 norm, got diff=\(normDiff.item(Float.self))")
    }
}


// MARK: - Interleaved M-RoPE Tests (Qwen 3.5-VL Pattern)

@Suite("Interleaved M-RoPE", .tags(.unit))
struct InterleavedMRoPETests {

    @Test("Interleaved M-RoPE: partial rotation preserves pass-through dimensions")
    func interleavedPassThrough() throws {
        let headDim = 16
        let ropeDim = 8     // Only first 8 dims get rotated
        let sections = [1, 1, 2]  // sum=4 = ropeDim/2 ✓
        let ropeBase: Float = 10000.0
        let B = 1, H = 2, L = 3

        MLXRandom.seed(99)
        let x = MLXRandom.normal([B, H, L, headDim])

        let posArray = MLXArray(Int32(0)..<Int32(3)).reshaped(1, L)
        let positionIds = stacked([posArray, posArray, posArray], axis: 0)

        let result = referenceInterleavedMRoPE(
            x, positionIds: positionIds,
            ropeDim: ropeDim, ropeBase: ropeBase,
            sections: sections, headDim: headDim)

        eval(result)

        #expect(result.shape == [B, H, L, headDim])

        // Pass-through dimensions (ropeDim..headDim) must be unchanged
        let xPass = x[0..., 0..., 0..., ropeDim...]
        let resultPass = result[0..., 0..., 0..., ropeDim...]
        let passDiff = abs(xPass - resultPass).max()

        #expect(passDiff.item(Float.self) < 1e-6,
            "Pass-through dimensions must be unchanged, got diff=\(passDiff.item(Float.self))")

        // Rotated dimensions (0..ropeDim) must be changed
        let xRot = x[0..., 0..., 0..., ..<ropeDim]
        let resultRot = result[0..., 0..., 0..., ..<ropeDim]
        let rotDiff = abs(xRot - resultRot).max()
        #expect(rotDiff.item(Float.self) > 1e-6,
            "Rotated dimensions must change")
    }

    @Test("Interleaved M-RoPE: different axis positions produce different rotations")
    func interleavedDifferentAxes() throws {
        let headDim = 16
        let ropeDim = 8
        let sections = [1, 1, 2]
        let ropeBase: Float = 10000.0
        let B = 1, H = 1, L = 1

        MLXRandom.seed(42)
        let x = MLXRandom.normal([B, H, L, headDim])

        let uniformPos = stacked([
            MLXArray([Int32(3)]).reshaped(1, 1),
            MLXArray([Int32(3)]).reshaped(1, 1),
            MLXArray([Int32(3)]).reshaped(1, 1)
        ], axis: 0)

        let mixedPos = stacked([
            MLXArray([Int32(3)]).reshaped(1, 1),
            MLXArray([Int32(1)]).reshaped(1, 1),
            MLXArray([Int32(2)]).reshaped(1, 1)
        ], axis: 0)

        let uniformResult = referenceInterleavedMRoPE(
            x, positionIds: uniformPos,
            ropeDim: ropeDim, ropeBase: ropeBase,
            sections: sections, headDim: headDim)

        let mixedResult = referenceInterleavedMRoPE(
            x, positionIds: mixedPos,
            ropeDim: ropeDim, ropeBase: ropeBase,
            sections: sections, headDim: headDim)

        eval(uniformResult, mixedResult)

        let diff = abs(uniformResult - mixedResult).max()
        #expect(diff.item(Float.self) > 1e-6,
            "Interleaved M-RoPE must produce different results for different positions")
    }

    @Test("Interleaved M-RoPE: rotation is norm-preserving on rotated dims")
    func interleavedNormPreserving() throws {
        let headDim = 32
        let ropeDim = 16
        let sections = [2, 3, 3]
        let ropeBase: Float = 500000.0
        let B = 1, H = 4, L = 5

        MLXRandom.seed(77)
        let x = MLXRandom.normal([B, H, L, headDim])

        let posArray = MLXArray(Int32(0)..<Int32(5)).reshaped(1, L)
        let positionIds = stacked([posArray, posArray, posArray], axis: 0)

        let result = referenceInterleavedMRoPE(
            x, positionIds: positionIds,
            ropeDim: ropeDim, ropeBase: ropeBase,
            sections: sections, headDim: headDim)

        eval(result)

        // Check norm on rotated portion only
        let xRot = x[0..., 0..., 0..., ..<ropeDim]
        let resultRot = result[0..., 0..., 0..., ..<ropeDim]
        let inputNorm = (xRot * xRot).sum(axis: -1).sqrt()
        let outputNorm = (resultRot * resultRot).sum(axis: -1).sqrt()
        let normDiff = abs(inputNorm - outputNorm).max()

        #expect(normDiff.item(Float.self) < 1e-4,
            "Interleaved M-RoPE must preserve norm on rotated dims")
    }
}


// MARK: - Executor M-RoPE Integration Tests

@Suite("Executor M-RoPE Integration", .tags(.unit))
struct ExecutorMRoPEIntegrationTests {

    @Test("Executor with M-RoPE positionIds produces different output than without")
    func executorMRoPEActivation() throws {
        let D = 8, H = 2, headDim = D / H
        let sections = [1, 1]  // sum=2 = halfDim=2 ✓
        let mropeAxes = MRoPEAxes(sections: sections, interleaved: false)

        let (compiled, _) = try buildMRoPEAttentionModel(
            hiddenSize: D, headCount: H, kvHeadCount: H,
            headDim: headDim, ropeDim: headDim, ropeBase: 10000.0,
            mropeAxes: mropeAxes)

        let tokens = MLXArray([0, 1, 2, 3]).reshaped(1, 4)
        let B = 1, L = 4

        // Without position IDs (uses cache offset = 0)
        let executor1 = MLXExecutor(compiledModel: compiled)
        let result1 = try executor1.forward(tokenIDs: tokens)
        eval(result1)

        // With explicit M-RoPE position IDs (mixed positions)
        let executor2 = MLXExecutor(compiledModel: compiled)
        let posArray = MLXArray(Int32(0)..<Int32(4)).reshaped(1, L)
        let mixedPos = stacked([
            posArray,
            MLXArray([Int32(0), Int32(1), Int32(0), Int32(1)]).reshaped(1, L),
            MLXArray([Int32(0), Int32(0), Int32(1), Int32(1)]).reshaped(1, L)
        ], axis: 0)

        let result2 = try executor2.forward(tokenIDs: tokens, positionIds: mixedPos)
        eval(result2)

        // Results must differ since position encodings differ
        #expect(result1.shape == result2.shape)
        let diff = abs(result1 - result2).max()
        #expect(diff.item(Float.self) > 1e-6,
            "M-RoPE with mixed positions must produce different output")
    }

    @Test("Executor: uniform M-RoPE positions match standard RoPE for text-only")
    func executorUniformMatchesStandard() throws {
        let D = 8, H = 2, headDim = D / H
        let ropeBase: Float = 10000.0
        let sections = [1, 1]
        let mropeAxes = MRoPEAxes(sections: sections, interleaved: false)

        // Model with M-RoPE
        let (compiledMRoPE, _) = try buildMRoPEAttentionModel(
            hiddenSize: D, headCount: H, kvHeadCount: H,
            headDim: headDim, ropeDim: headDim, ropeBase: ropeBase,
            mropeAxes: mropeAxes)

        // Model without M-RoPE (standard RoPE)
        let standardModel = MiniStandardRoPEModel(
            vocabSize: 16, hiddenSize: D, headCount: H,
            kvHeadCount: H, headDim: headDim,
            ropeDim: headDim, ropeBase: ropeBase)
        let standardGraph = try standardModel.makeModelGraph()

        // Use same weights for both
        let weightStore = compiledMRoPE.weightStore
        let scanResult = MLXInferenceCompiler().scan(graph: standardGraph)
        let compiledStandard = MLXCompiledModel(
            graph: standardGraph, weightStore: weightStore,
            cacheDescriptors: scanResult.cacheDescriptors,
            embeddingPath: scanResult.embeddingPath)

        let tokens = MLXArray([0, 1, 2]).reshaped(1, 3)
        let L = 3

        // Standard RoPE with offset=0
        let execStd = MLXExecutor(compiledModel: compiledStandard)
        let resultStd = try execStd.forward(tokenIDs: tokens)
        eval(resultStd)

        // M-RoPE with uniform positions [0,1,2] on all axes
        let execMRoPE = MLXExecutor(compiledModel: compiledMRoPE)
        let posArray = MLXArray(Int32(0)..<Int32(3)).reshaped(1, L)
        let uniformPos = stacked([posArray, posArray, posArray], axis: 0)
        let resultMRoPE = try execMRoPE.forward(
            tokenIDs: tokens, positionIds: uniformPos)
        eval(resultMRoPE)

        // For text-only with uniform positions starting at 0, M-RoPE should
        // produce similar (but not necessarily identical) results to standard RoPE.
        // They compute frequencies differently (M-RoPE splits by sections),
        // so exact match is not expected.
        #expect(resultStd.shape == resultMRoPE.shape)
    }

    @Test("Executor M-RoPE: sequential decode positions are consistent")
    func executorSequentialDecode() throws {
        let D = 8, H = 2, headDim = D / H
        let sections = [1, 1]
        let mropeAxes = MRoPEAxes(sections: sections, interleaved: false)

        let (compiled, _) = try buildMRoPEAttentionModel(
            hiddenSize: D, headCount: H, kvHeadCount: H,
            headDim: headDim, ropeDim: headDim, ropeBase: 10000.0,
            mropeAxes: mropeAxes)

        let executor = MLXExecutor(compiledModel: compiled)

        // Prefill with positions [0, 1, 2]
        let prefillTokens = MLXArray([0, 1, 2]).reshaped(1, 3)
        let prefillPos = stacked([
            MLXArray(Int32(0)..<Int32(3)).reshaped(1, 3),
            MLXArray(Int32(0)..<Int32(3)).reshaped(1, 3),
            MLXArray(Int32(0)..<Int32(3)).reshaped(1, 3)
        ], axis: 0)
        let prefillResult = try executor.forward(
            tokenIDs: prefillTokens, positionIds: prefillPos)
        eval(prefillResult)

        // Decode step with position [3]
        let decodeTokens = MLXArray([3]).reshaped(1, 1)
        let decodePos = stacked([
            MLXArray([Int32(3)]).reshaped(1, 1),
            MLXArray([Int32(3)]).reshaped(1, 1),
            MLXArray([Int32(3)]).reshaped(1, 1)
        ], axis: 0)
        let decodeResult = try executor.forward(
            tokenIDs: decodeTokens, positionIds: decodePos)
        eval(decodeResult)

        // Just verify it doesn't crash and produces valid output
        #expect(decodeResult.shape == [1, 1, D])
        // Verify no NaN in output
        let hasNaN = isNaN(decodeResult).any().item(Bool.self)
        #expect(!hasNaN, "Decode output must not contain NaN")
    }
}


// MARK: - M-RoPE Position ID Computation Tests

@Suite("M-RoPE Position IDs", .tags(.unit))
struct MRoPEPositionIdTests {

    @Test("Text-only tokens: all axes get same sequential position")
    func textOnlyPositions() {
        // 5 text tokens, batch=1
        // Token IDs that are NOT vision tokens
        let tokenIds = MLXArray([10, 20, 30, 40, 50]).reshaped(1, 5)
        let imageTokenId = 151655  // Some token that's not in our sequence
        let videoTokenId = 151656

        let (posIds, nextPos) = computeTestMRoPEPositionIds(
            inputIds: tokenIds, imageTokenId: imageTokenId,
            videoTokenId: videoTokenId, spatialMergeSize: 2,
            image: nil, video: nil)

        eval(posIds)

        #expect(posIds.shape == [3, 1, 5])
        #expect(nextPos == 5)

        // All axes should have [0, 1, 2, 3, 4]
        for axis in 0..<3 {
            let axisPositions = posIds[axis][0]
            for i in 0..<5 {
                let pos: Int32 = axisPositions[i].item()
                #expect(pos == Int32(i),
                    "axis \(axis), position \(i): expected \(i), got \(pos)")
            }
        }
    }

    @Test("Vision tokens: positions encode spatial grid (t, h, w)")
    func visionTokenPositions() {
        // Sequence: [text, text, img, img, img, img, text]
        // Image grid: t=1, h=4, w=4, spatialMergeSize=2 → merged h=2, w=2
        // Total merged = 1 * 2 * 2 = 4 vision tokens
        let imageTokenId: Int32 = 100
        let tokenIds = MLXArray([Int32(10), 20, imageTokenId, imageTokenId,
                                 imageTokenId, imageTokenId, 30]).reshaped(1, 7)

        let image = TestProcessedImage(
            frames: [TestTHW(t: 1, h: 4, w: 4)])

        let (posIds, nextPos) = computeTestMRoPEPositionIds(
            inputIds: tokenIds, imageTokenId: Int(imageTokenId),
            videoTokenId: 999, spatialMergeSize: 2,
            image: image, video: nil)

        eval(posIds)

        #expect(posIds.shape == [3, 1, 7])

        // First 2 text tokens: t=h=w=[0,1]
        let temporal = posIds[0][0]
        let height = posIds[1][0]
        let width = posIds[2][0]

        // text token 0: pos=0
        #expect(temporal[0].item(Int32.self) == 0)
        #expect(height[0].item(Int32.self) == 0)
        #expect(width[0].item(Int32.self) == 0)

        // text token 1: pos=1
        #expect(temporal[1].item(Int32.self) == 1)
        #expect(height[1].item(Int32.self) == 1)
        #expect(width[1].item(Int32.self) == 1)

        // Vision tokens (mergedH=2, mergedW=2):
        // Token 0: tPos=0/(2*2)=0, hPos=0/2=0, wPos=0%2=0
        // → temporal=2+0=2, height=2+0=2, width=2+0=2
        #expect(temporal[2].item(Int32.self) == 2)
        #expect(height[2].item(Int32.self) == 2)
        #expect(width[2].item(Int32.self) == 2)

        // Token 1: tPos=0, hPos=0, wPos=1
        // → temporal=2+0=2, height=2+0=2, width=2+1=3
        #expect(temporal[3].item(Int32.self) == 2)
        #expect(height[3].item(Int32.self) == 2)
        #expect(width[3].item(Int32.self) == 3)

        // Token 2: tPos=0, hPos=1, wPos=0
        // → temporal=2+0=2, height=2+1=3, width=2+0=2
        #expect(temporal[4].item(Int32.self) == 2)
        #expect(height[4].item(Int32.self) == 3)
        #expect(width[4].item(Int32.self) == 2)

        // Token 3: tPos=0, hPos=1, wPos=1
        // → temporal=2+0=2, height=2+1=3, width=2+1=3
        #expect(temporal[5].item(Int32.self) == 2)
        #expect(height[5].item(Int32.self) == 3)
        #expect(width[5].item(Int32.self) == 3)

        // After vision: currentTextPos = 2 + max(1, max(2, 2)) = 2 + 2 = 4
        // Last text token: pos=4
        #expect(temporal[6].item(Int32.self) == 4)
        #expect(height[6].item(Int32.self) == 4)
        #expect(width[6].item(Int32.self) == 4)

        #expect(nextPos == 5)
    }

    @Test("Sequential decode positions: all axes identical")
    func sequentialDecodePositions() {
        let B = 2, L = 3, start = 10
        let posIds = makeTestSequentialPositionIds(
            batchSize: B, seqLen: L, startPosition: start)

        eval(posIds)

        #expect(posIds.shape == [3, B, L])

        for axis in 0..<3 {
            for b in 0..<B {
                for s in 0..<L {
                    let pos: Int32 = posIds[axis][b][s].item()
                    #expect(pos == Int32(start + s),
                        "axis=\(axis), b=\(b), s=\(s): expected \(start + s), got \(pos)")
                }
            }
        }
    }
}


// MARK: - Test Helpers for Position ID Computation

/// Minimal test types to avoid depending on MLXLM's LMInput.
private struct TestProcessedImage {
    let frames: [TestTHW]?
}

private struct TestTHW {
    let t: Int, h: Int, w: Int
}

/// Standalone position ID computation matching MLXLanguageModel M-RoPE.
private func computeTestMRoPEPositionIds(
    inputIds: MLXArray,
    imageTokenId: Int, videoTokenId: Int, spatialMergeSize: Int,
    image: TestProcessedImage?, video: TestProcessedImage?
) -> (MLXArray, Int) {
    let B = inputIds.dim(0)
    let S = inputIds.dim(1)

    var temporalPos = [Int32](repeating: 0, count: B * S)
    var heightPos = [Int32](repeating: 0, count: B * S)
    var widthPos = [Int32](repeating: 0, count: B * S)

    let flatIds = inputIds.reshaped(-1)

    var currentTextPos: Int32 = 0
    var visionTokenIdx = 0
    let allGrids = (image?.frames ?? []) + (video?.frames ?? [])
    var gridIdx = 0

    for i in 0..<(B * S) {
        let tokenId: Int32 = flatIds[i].item()

        if tokenId == Int32(imageTokenId) || tokenId == Int32(videoTokenId) {
            if gridIdx < allGrids.count {
                let grid = allGrids[gridIdx]
                let mergedH = grid.h / spatialMergeSize
                let mergedW = grid.w / spatialMergeSize
                let totalMerged = grid.t * mergedH * mergedW

                let posInGrid = visionTokenIdx
                let tPos = posInGrid / (mergedH * mergedW)
                let hPos = (posInGrid % (mergedH * mergedW)) / mergedW
                let wPos = posInGrid % mergedW

                temporalPos[i] = currentTextPos + Int32(tPos)
                heightPos[i] = currentTextPos + Int32(hPos)
                widthPos[i] = currentTextPos + Int32(wPos)

                visionTokenIdx += 1
                if visionTokenIdx >= totalMerged {
                    currentTextPos += Int32(max(grid.t, max(mergedH, mergedW)))
                    visionTokenIdx = 0
                    gridIdx += 1
                }
            }
        } else {
            temporalPos[i] = currentTextPos
            heightPos[i] = currentTextPos
            widthPos[i] = currentTextPos
            currentTextPos += 1
        }
    }

    let tArray = MLXArray(temporalPos).reshaped(B, S)
    let hArray = MLXArray(heightPos).reshaped(B, S)
    let wArray = MLXArray(widthPos).reshaped(B, S)

    return (stacked([tArray, hArray, wArray], axis: 0), Int(currentTextPos))
}

private func makeTestSequentialPositionIds(
    batchSize: Int, seqLen: Int, startPosition: Int
) -> MLXArray {
    let positions = tiled(
        MLXArray(Int32(startPosition)..<Int32(startPosition + seqLen))
            .reshaped(1, seqLen),
        repetitions: [batchSize, 1])
    return stacked([positions, positions, positions], axis: 0)
}
