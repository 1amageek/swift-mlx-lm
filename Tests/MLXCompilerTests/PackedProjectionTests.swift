import Testing
import TestHeartbeat
@preconcurrency import MLX
@testable import SwiftLM
@testable import MLXCompiler

@Suite("PackedProjection Tests", .tags(.unit, .compiled), .heartbeat)
struct PackedProjectionTests {

    // MARK: - Dense Packing

    @Test("Dense: packed QKV matches individual projections")
    func densePackedMatchesIndividual() throws {
        let D = 8
        let qDim = 4
        let kvDim = 4

        let wQ = MLXRandom.normal([qDim, D]) * 0.02
        let wK = MLXRandom.normal([kvDim, D]) * 0.02
        let wV = MLXRandom.normal([kvDim, D]) * 0.02

        let qProj = LoweredProjection(weight: wQ)
        let kProj = LoweredProjection(weight: wK)
        let vProj = LoweredProjection(weight: wV)

        let x = MLXRandom.normal([1, 4, D])

        // Individual
        let qRef = qProj.apply(x)
        let kRef = kProj.apply(x)
        let vRef = vProj.apply(x)

        // Packed
        let packed = PackedProjection.pack([qProj, kProj, vProj])
        #expect(packed != nil)
        let results = packed!.apply(x)
        #expect(results.count == 3)

        // Exact match for dense (no quantization rounding)
        let qDiff = abs(results[0] - qRef).max().item(Float.self)
        let kDiff = abs(results[1] - kRef).max().item(Float.self)
        let vDiff = abs(results[2] - vRef).max().item(Float.self)

        #expect(qDiff < 1e-5, "Q diff: \(qDiff)")
        #expect(kDiff < 1e-5, "K diff: \(kDiff)")
        #expect(vDiff < 1e-5, "V diff: \(vDiff)")
    }

    @Test("Dense: packed gate+up matches individual projections")
    func denseGateUpPackedMatchesIndividual() throws {
        let D = 8
        let inter = 16

        let wGate = MLXRandom.normal([inter, D]) * 0.02
        let wUp = MLXRandom.normal([inter, D]) * 0.02

        let gateProj = LoweredProjection(weight: wGate)
        let upProj = LoweredProjection(weight: wUp)

        let x = MLXRandom.normal([1, 4, D])

        // Individual
        let gateRef = gateProj.apply(x)
        let upRef = upProj.apply(x)

        // Packed
        let packed = PackedProjection.pack([gateProj, upProj])
        #expect(packed != nil)
        let results = packed!.apply(x)
        #expect(results.count == 2)

        let gateDiff = abs(results[0] - gateRef).max().item(Float.self)
        let upDiff = abs(results[1] - upRef).max().item(Float.self)

        #expect(gateDiff < 1e-5, "Gate diff: \(gateDiff)")
        #expect(upDiff < 1e-5, "Up diff: \(upDiff)")
    }

    // MARK: - GQA Dimensions

    @Test("Dense: GQA with headCount != kvHeadCount")
    func denseGQAPacking() throws {
        let D = 8
        let headCount = 4
        let kvHeadCount = 2
        let headDim = 2

        let qDim = headCount * headDim      // 64
        let kvDim = kvHeadCount * headDim    // 16

        let wQ = MLXRandom.normal([qDim, D]) * 0.02
        let wK = MLXRandom.normal([kvDim, D]) * 0.02
        let wV = MLXRandom.normal([kvDim, D]) * 0.02

        let qProj = LoweredProjection(weight: wQ)
        let kProj = LoweredProjection(weight: wK)
        let vProj = LoweredProjection(weight: wV)

        let x = MLXRandom.normal([1, 8, D])

        // Individual
        let qRef = qProj.apply(x)
        let kRef = kProj.apply(x)
        let vRef = vProj.apply(x)

        // Packed
        let packed = PackedProjection.pack([qProj, kProj, vProj])
        #expect(packed != nil)
        let results = packed!.apply(x)

        // Verify shapes
        #expect(results[0].shape == qRef.shape)
        #expect(results[1].shape == kRef.shape)
        #expect(results[2].shape == vRef.shape)

        // Verify split indices produce correct dimensions
        #expect(results[0].dim(-1) == qDim)
        #expect(results[1].dim(-1) == kvDim)
        #expect(results[2].dim(-1) == kvDim)

        // Verify values
        let qDiff = abs(results[0] - qRef).max().item(Float.self)
        let kDiff = abs(results[1] - kRef).max().item(Float.self)
        let vDiff = abs(results[2] - vRef).max().item(Float.self)

        #expect(qDiff < 1e-5)
        #expect(kDiff < 1e-5)
        #expect(vDiff < 1e-5)
    }

    // MARK: - Bias Handling

    @Test("Dense: packed with biases")
    func densePackedWithBias() throws {
        let D = 8
        let outA = 4
        let outB = 4

        let wA = MLXRandom.normal([outA, D]) * 0.02
        let bA = MLXRandom.normal([outA]) * 0.01
        let wB = MLXRandom.normal([outB, D]) * 0.02
        let bB = MLXRandom.normal([outB]) * 0.01

        let projA = LoweredProjection(weight: wA, bias: bA)
        let projB = LoweredProjection(weight: wB, bias: bB)

        let x = MLXRandom.normal([1, 4, D])

        let refA = projA.apply(x)
        let refB = projB.apply(x)

        let packed = PackedProjection.pack([projA, projB])
        #expect(packed != nil)
        let results = packed!.apply(x)

        let diffA = abs(results[0] - refA).max().item(Float.self)
        let diffB = abs(results[1] - refB).max().item(Float.self)

        #expect(diffA < 1e-5, "A diff with bias: \(diffA)")
        #expect(diffB < 1e-5, "B diff with bias: \(diffB)")
    }

    @Test("Dense: packed with mixed bias (some nil)")
    func densePackedWithMixedBias() throws {
        let D = 8
        let outA = 4
        let outB = 4

        let wA = MLXRandom.normal([outA, D]) * 0.02
        let bA = MLXRandom.normal([outA]) * 0.01
        let wB = MLXRandom.normal([outB, D]) * 0.02

        let projA = LoweredProjection(weight: wA, bias: bA)
        let projB = LoweredProjection(weight: wB, bias: nil)

        let x = MLXRandom.normal([1, 4, D])

        let refA = projA.apply(x)
        let refB = projB.apply(x)

        let packed = PackedProjection.pack([projA, projB])
        #expect(packed != nil)
        let results = packed!.apply(x)

        let diffA = abs(results[0] - refA).max().item(Float.self)
        let diffB = abs(results[1] - refB).max().item(Float.self)

        #expect(diffA < 1e-5)
        #expect(diffB < 1e-5)
    }

    // MARK: - Edge Cases

    @Test("Single projection returns nil (requires >= 2)")
    func singleProjectionReturnsNil() throws {
        let w = MLXRandom.normal([4, 8])
        let proj = LoweredProjection(weight: w)
        let packed = PackedProjection.pack([proj])
        #expect(packed == nil)
    }

    @Test("Empty list returns nil")
    func emptyListReturnsNil() throws {
        let packed = PackedProjection.pack([])
        #expect(packed == nil)
    }

    // MARK: - Quantized Packing

    @Test("AffineQuantized: packed projections with same bits/groupSize")
    func affineQuantizedPacking() throws {
        let D = 32
        let outA = 8
        let outB = 4

        let groupSize = 32
        let bits = 4
        let elemsPerInt32 = 32 / bits  // 8

        let pwA = MLXArray.zeros([outA, D / elemsPerInt32]).asType(.uint32)
        let scA = MLXRandom.normal([outA, D / groupSize]).asType(.float16)
        let zbA = MLXRandom.normal([outA, D / groupSize]).asType(.float16)

        let pwB = MLXArray.zeros([outB, D / elemsPerInt32]).asType(.uint32)
        let scB = MLXRandom.normal([outB, D / groupSize]).asType(.float16)
        let zbB = MLXRandom.normal([outB, D / groupSize]).asType(.float16)

        let qtA = AffineQuantizedTensor(
            logicalShape: [outA, D],
            packedWeight: pwA, scales: scA, zeroBiases: zbA,
            groupSize: groupSize, bits: bits, origin: .unknown
        )
        let qtB = AffineQuantizedTensor(
            logicalShape: [outB, D],
            packedWeight: pwB, scales: scB, zeroBiases: zbB,
            groupSize: groupSize, bits: bits, origin: .unknown
        )

        let projA = LoweredProjection(
            storage: .affineQuantized(qtA), bias: nil)
        let projB = LoweredProjection(
            storage: .affineQuantized(qtB), bias: nil)

        let packed = PackedProjection.pack([projA, projB])
        #expect(packed != nil)
        #expect(packed!.count == 2)
        #expect(packed!.splitIndices == [outA])

        // Verify the packed kernel is affineQuantized
        if case .affineQuantized(let q) = packed!.kernel {
            #expect(q.logicalShape == [outA + outB, D])
            #expect(q.bits == bits)
            #expect(q.groupSize == groupSize)
        } else {
            Issue.record("Expected affineQuantized kernel")
        }
    }

    @Test("Mixed kernel variants return nil")
    func mixedKernelReturnsNil() throws {
        let D = 32

        let wDense = MLXRandom.normal([8, D])
        let projDense = LoweredProjection(weight: wDense)

        let groupSize = 32
        let bits = 4
        let elemsPerInt32 = 32 / bits
        let pw = MLXArray.zeros([4, D / elemsPerInt32]).asType(.uint32)
        let sc = MLXRandom.normal([4, D / groupSize]).asType(.float16)
        let zb = MLXRandom.normal([4, D / groupSize]).asType(.float16)

        let qt = AffineQuantizedTensor(
            logicalShape: [4, D],
            packedWeight: pw, scales: sc, zeroBiases: zb,
            groupSize: groupSize, bits: bits, origin: .unknown
        )
        let projQuant = LoweredProjection(
            storage: .affineQuantized(qt), bias: nil)

        let packed = PackedProjection.pack([projDense, projQuant])
        #expect(packed == nil)
    }

    // MARK: - Integration with LoweredAttention

    @Test("LoweredAttention uses packed QKV projection")
    func attentionUsesPackedQKV() throws {
        let D = 8
        let headCount = 2
        let kvHeadCount = 2
        let headDim = 4

        let qDim = headCount * headDim
        let kvDim = kvHeadCount * headDim

        let wQ = MLXRandom.normal([qDim, D]) * 0.02
        let wK = MLXRandom.normal([kvDim, D]) * 0.02
        let wV = MLXRandom.normal([kvDim, D]) * 0.02
        let wO = MLXRandom.normal([D, D]) * 0.02

        let qProj = LoweredProjection(weight: wQ)
        let kProj = LoweredProjection(weight: wK)
        let vProj = LoweredProjection(weight: wV)
        let oProj = LoweredProjection(weight: wO)

        let attrs = AttentionAttributes(
            hiddenSize: D,
            headCount: headCount,
            kvHeadCount: kvHeadCount,
            headDimension: headDim
        )

        // Create packed attention
        let packed = PackedProjection.pack([qProj, kProj, vProj])!

        let packedAttn = LoweredAttention(
            qkvPacked: packed, oProj: oProj,
            attrs: attrs,
            qNormWeight: nil, kNormWeight: nil,
            qNormBias: nil, kNormBias: nil,
            cacheSlotIndex: 0
        )

        // Create individual attention
        let individualAttn = LoweredAttention(
            qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
            attrs: attrs,
            qNormWeight: nil, kNormWeight: nil,
            qNormBias: nil, kNormBias: nil,
            cacheSlotIndex: 0
        )

        let x = MLXRandom.normal([1, 4, D]) * 0.1

        var packedCaches: [LoweredCacheState] = [
            .kv(LoweredKVCache())
        ]
        var individualCaches: [LoweredCacheState] = [
            .kv(LoweredKVCache())
        ]

        let packedOut = packedAttn.apply(x, caches: &packedCaches)
        let individualOut = individualAttn.apply(x, caches: &individualCaches)

        eval(packedOut, individualOut)

        let maxDiff = abs(packedOut - individualOut).max().item(Float.self)
        #expect(maxDiff < 1e-4, "Packed vs individual attention diff: \(maxDiff)")
    }

    // MARK: - Integration with LoweredMLP

    @Test("LoweredMLP uses packed gate+up projection")
    func mlpUsesPackedGateUp() throws {
        let D = 8
        let inter = 16

        let wGate = MLXRandom.normal([inter, D]) * 0.02
        let wUp = MLXRandom.normal([inter, D]) * 0.02
        let wDown = MLXRandom.normal([D, inter]) * 0.02

        let gateProj = LoweredProjection(weight: wGate)
        let upProj = LoweredProjection(weight: wUp)
        let downProj = LoweredProjection(weight: wDown)

        let x = MLXRandom.normal([1, 4, D]) * 0.1

        // Packed MLP
        let packed = PackedProjection.pack([gateProj, upProj])!
        let packedMLP = LoweredMLP(
            gateUpPacked: packed,
            downProj: downProj,
            activation: .silu
        )

        // Individual MLP
        let individualMLP = LoweredMLP(
            gateProj: gateProj, downProj: downProj,
            upProj: upProj, activation: .silu
        )

        let packedOut = packedMLP.apply(x)
        let individualOut = individualMLP.apply(x)

        eval(packedOut, individualOut)

        let maxDiff = abs(packedOut - individualOut).max().item(Float.self)
        #expect(maxDiff < 1e-4, "Packed vs individual MLP diff: \(maxDiff)")
    }
}
