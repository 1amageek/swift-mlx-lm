import Testing
import TestHeartbeat
@preconcurrency import MLX
@testable import SwiftLM
@testable import LMCompiler

@Suite("FusedBlock Tests", .tags(.unit, .compiled), .heartbeat)
struct FusedBlockTests {

    // MARK: - Pattern Detection

    @Test("Fuse: norm + attention residual body")
    func fuseNormAttention() throws {
        let D = 4
        let headCount = 2
        let headDim = 2

        let norm = LoweredNorm.rms(
            weight: MLXRandom.normal([D]), epsilon: 1e-5)

        let wQ = MLXRandom.normal([D, D]) * 0.02
        let wK = MLXRandom.normal([D, D]) * 0.02
        let wV = MLXRandom.normal([D, D]) * 0.02
        let wO = MLXRandom.normal([D, D]) * 0.02

        let qProj = LoweredProjection(weight: wQ)
        let kProj = LoweredProjection(weight: wK)
        let vProj = LoweredProjection(weight: wV)
        let oProj = LoweredProjection(weight: wO)

        let attrs = AttentionAttributes(
            hiddenSize: D, headCount: headCount,
            kvHeadCount: headCount, headDimension: headDim)

        let attn = LoweredAttention(
            qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
            attrs: attrs,
            qNormWeight: nil, kNormWeight: nil,
            qNormBias: nil, kNormBias: nil,
            cacheSlotIndex: 0)

        let body: [LoweredStep] = [
            .op(.norm(norm)),
            .op(.attention(attn)),
        ]

        let fused = tryFuseResidual(body)
        #expect(fused != nil)
        if case .attention = fused! {
            // Expected
        } else {
            Issue.record("Expected .attention fused sub-layer")
        }
    }

    @Test("Fuse: norm + mlp residual body")
    func fuseNormMLP() throws {
        let D = 4
        let inter = 8

        let norm = LoweredNorm.rms(
            weight: MLXRandom.normal([D]), epsilon: 1e-5)

        let gateProj = LoweredProjection(weight: MLXRandom.normal([inter, D]))
        let upProj = LoweredProjection(weight: MLXRandom.normal([inter, D]))
        let downProj = LoweredProjection(weight: MLXRandom.normal([D, inter]))

        let mlp = LoweredMLP(
            gateProj: gateProj, downProj: downProj,
            upProj: upProj, activation: .silu)

        let body: [LoweredStep] = [
            .op(.norm(norm)),
            .op(.mlp(mlp)),
        ]

        let fused = tryFuseResidual(body)
        #expect(fused != nil)
        if case .mlp = fused! {
            // Expected
        } else {
            Issue.record("Expected .mlp fused sub-layer")
        }
    }

    @Test("Fuse: norm + moe residual body")
    func fuseNormMoE() throws {
        let D = 4
        let inter = 8

        let norm = LoweredNorm.rms(
            weight: MLXRandom.normal([D]), epsilon: 1e-5)

        let router = LoweredProjection(weight: MLXRandom.normal([4, D]) * 0.02)
        var experts: [LoweredExpertMLP] = []
        for _ in 0..<4 {
            let gate = LoweredProjection(weight: MLXRandom.normal([inter, D]) * 0.02)
            let up = LoweredProjection(weight: MLXRandom.normal([inter, D]) * 0.02)
            let down = LoweredProjection(weight: MLXRandom.normal([D, inter]) * 0.02)
            experts.append(LoweredExpertMLP(
                gateProj: gate, upProj: up, downProj: down, activation: .silu))
        }

        let moe = LoweredMoE(router: router, experts: experts, expertsPerToken: 2)

        let body: [LoweredStep] = [
            .op(.norm(norm)),
            .op(.moe(moe)),
        ]

        let fused = tryFuseResidual(body)
        #expect(fused != nil)
        if case .moe = fused! {
            // Expected
        } else {
            Issue.record("Expected .moe fused sub-layer")
        }
    }

    @Test("No fuse: single op body")
    func noFuseSingleOp() throws {
        let norm = LoweredNorm.rms(
            weight: MLXRandom.normal([4]), epsilon: 1e-5)
        let body: [LoweredStep] = [.op(.norm(norm))]
        let fused = tryFuseResidual(body)
        #expect(fused == nil)
    }

    @Test("No fuse: three op body")
    func noFuseThreeOps() throws {
        let D = 4
        let norm1 = LoweredNorm.rms(weight: MLXRandom.normal([D]), epsilon: 1e-5)
        let norm2 = LoweredNorm.rms(weight: MLXRandom.normal([D]), epsilon: 1e-5)
        let mlp = LoweredMLP(
            gateProj: LoweredProjection(weight: MLXRandom.normal([8, D])),
            downProj: LoweredProjection(weight: MLXRandom.normal([D, 8])),
            upProj: nil, activation: .silu)

        let body: [LoweredStep] = [
            .op(.norm(norm1)),
            .op(.norm(norm2)),
            .op(.mlp(mlp)),
        ]
        let fused = tryFuseResidual(body)
        #expect(fused == nil)
    }

    @Test("No fuse: norm + norm body")
    func noFuseNormNorm() throws {
        let norm1 = LoweredNorm.rms(weight: MLXRandom.normal([4]), epsilon: 1e-5)
        let norm2 = LoweredNorm.rms(weight: MLXRandom.normal([4]), epsilon: 1e-5)
        let body: [LoweredStep] = [
            .op(.norm(norm1)),
            .op(.norm(norm2)),
        ]
        let fused = tryFuseResidual(body)
        #expect(fused == nil)
    }

    @Test("No fuse: attention + norm (wrong order)")
    func noFuseWrongOrder() throws {
        let D = 4
        let headCount = 2
        let headDim = 2

        let norm = LoweredNorm.rms(
            weight: MLXRandom.normal([D]), epsilon: 1e-5)
        let attn = LoweredAttention(
            qProj: LoweredProjection(weight: MLXRandom.normal([D, D])),
            kProj: LoweredProjection(weight: MLXRandom.normal([D, D])),
            vProj: LoweredProjection(weight: MLXRandom.normal([D, D])),
            oProj: LoweredProjection(weight: MLXRandom.normal([D, D])),
            attrs: AttentionAttributes(
                hiddenSize: D, headCount: headCount,
                kvHeadCount: headCount, headDimension: headDim),
            qNormWeight: nil, kNormWeight: nil,
            qNormBias: nil, kNormBias: nil,
            cacheSlotIndex: 0)

        let body: [LoweredStep] = [
            .op(.attention(attn)),
            .op(.norm(norm)),
        ]
        let fused = tryFuseResidual(body)
        #expect(fused == nil)
    }

    // MARK: - Flattening with Fusion

    @Test("flattenSteps produces fusedSubLayer for [norm, attn] residual")
    func flattenDetectsFusion() throws {
        let D = 4
        let headCount = 2
        let headDim = 2

        let norm = LoweredNorm.rms(
            weight: MLXRandom.normal([D]), epsilon: 1e-5)
        let attn = LoweredAttention(
            qProj: LoweredProjection(weight: MLXRandom.normal([D, D])),
            kProj: LoweredProjection(weight: MLXRandom.normal([D, D])),
            vProj: LoweredProjection(weight: MLXRandom.normal([D, D])),
            oProj: LoweredProjection(weight: MLXRandom.normal([D, D])),
            attrs: AttentionAttributes(
                hiddenSize: D, headCount: headCount,
                kvHeadCount: headCount, headDimension: headDim),
            qNormWeight: nil, kNormWeight: nil,
            qNormBias: nil, kNormBias: nil,
            cacheSlotIndex: 0)

        let steps: [LoweredStep] = [
            .residual(body: [.op(.norm(norm)), .op(.attention(attn))]),
        ]

        let flat = flattenSteps(steps)

        // Should produce a single fusedSubLayer step, not save/add markers
        #expect(flat.count == 1)
        if case .fusedSubLayer(.attention) = flat[0] {
            // Expected
        } else {
            Issue.record("Expected fusedSubLayer(.attention), got \(flat[0])")
        }
    }

    @Test("flattenSteps preserves unfuseable residuals")
    func flattenPreservesUnfuseable() throws {
        let D = 4
        let norm = LoweredNorm.rms(
            weight: MLXRandom.normal([D]), epsilon: 1e-5)

        let steps: [LoweredStep] = [
            .residual(body: [.op(.norm(norm))]),  // Single op, not fuseable
        ]

        let flat = flattenSteps(steps)

        // Should produce saveResidual + op + addResidual
        #expect(flat.count == 3)
        if case .saveResidual = flat[0] {} else {
            Issue.record("Expected saveResidual")
        }
        if case .addResidual = flat[2] {} else {
            Issue.record("Expected addResidual")
        }
    }

    // MARK: - Correctness: Fused vs Individual

    @Test("Fused attention sub-layer matches individual execution")
    func fusedAttentionCorrectness() throws {
        let D = 8
        let headCount = 2
        let headDim = 4

        let normWeight = MLXRandom.normal([D]) * 0.1
        let norm = LoweredNorm.rms(weight: normWeight, epsilon: 1e-5)

        let wQ = MLXRandom.normal([D, D]) * 0.02
        let wK = MLXRandom.normal([D, D]) * 0.02
        let wV = MLXRandom.normal([D, D]) * 0.02
        let wO = MLXRandom.normal([D, D]) * 0.02

        let qProj = LoweredProjection(weight: wQ)
        let kProj = LoweredProjection(weight: wK)
        let vProj = LoweredProjection(weight: wV)
        let oProj = LoweredProjection(weight: wO)

        let attrs = AttentionAttributes(
            hiddenSize: D, headCount: headCount,
            kvHeadCount: headCount, headDimension: headDim)

        let attn = LoweredAttention(
            qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
            attrs: attrs,
            qNormWeight: nil, kNormWeight: nil,
            qNormBias: nil, kNormBias: nil,
            cacheSlotIndex: 0)

        let x = MLXRandom.normal([1, 4, D]) * 0.1

        // Individual path: x + attn(norm(x))
        var individualCaches: [LoweredCacheState] = [.kv(LoweredKVCache())]
        let normed = norm.apply(x)
        let attnOut = attn.apply(normed, caches: &individualCaches)
        let individualResult = x + attnOut

        // Fused path
        let fused = FusedSubLayer.attention(norm: norm, attention: attn)
        var fusedState = InferenceState(
            caches: [.kv(LoweredKVCache())], nextPosition: 0)
        let fusedResult = fused.apply(x, state: &fusedState)

        eval(individualResult, fusedResult)

        let maxDiff = abs(fusedResult - individualResult).max().item(Float.self)
        #expect(maxDiff < 1e-5, "Fused vs individual attention diff: \(maxDiff)")
    }

    @Test("Fused MLP sub-layer matches individual execution")
    func fusedMLPCorrectness() throws {
        let D = 8
        let inter = 16

        let normWeight = MLXRandom.normal([D]) * 0.1
        let norm = LoweredNorm.rms(weight: normWeight, epsilon: 1e-5)

        let gateProj = LoweredProjection(weight: MLXRandom.normal([inter, D]) * 0.02)
        let upProj = LoweredProjection(weight: MLXRandom.normal([inter, D]) * 0.02)
        let downProj = LoweredProjection(weight: MLXRandom.normal([D, inter]) * 0.02)

        let mlp = LoweredMLP(
            gateProj: gateProj, downProj: downProj,
            upProj: upProj, activation: .silu)

        let x = MLXRandom.normal([1, 4, D]) * 0.1

        // Individual path: x + mlp(norm(x))
        let individualResult = x + mlp.apply(norm.apply(x))

        // Fused path
        let fused = FusedSubLayer.mlp(norm: norm, mlp: mlp)
        var fusedState = InferenceState(caches: [], nextPosition: 0)
        let fusedResult = fused.apply(x, state: &fusedState)

        eval(individualResult, fusedResult)

        let maxDiff = abs(fusedResult - individualResult).max().item(Float.self)
        #expect(maxDiff < 1e-5, "Fused vs individual MLP diff: \(maxDiff)")
    }

    // MARK: - End-to-End: Full Decode Step

    @Test("Full transformer block: fused decode matches unfused")
    func fullBlockFusedMatchesUnfused() throws {
        let D = 8
        let headCount = 2
        let headDim = 4
        let inter = 16

        // Attention sub-layer
        let attnNormWeight = MLXRandom.normal([D]) * 0.1
        let attnNorm = LoweredNorm.rms(weight: attnNormWeight, epsilon: 1e-5)

        let attn = LoweredAttention(
            qProj: LoweredProjection(weight: MLXRandom.normal([D, D]) * 0.02),
            kProj: LoweredProjection(weight: MLXRandom.normal([D, D]) * 0.02),
            vProj: LoweredProjection(weight: MLXRandom.normal([D, D]) * 0.02),
            oProj: LoweredProjection(weight: MLXRandom.normal([D, D]) * 0.02),
            attrs: AttentionAttributes(
                hiddenSize: D, headCount: headCount,
                kvHeadCount: headCount, headDimension: headDim),
            qNormWeight: nil, kNormWeight: nil,
            qNormBias: nil, kNormBias: nil,
            cacheSlotIndex: 0)

        // MLP sub-layer
        let mlpNormWeight = MLXRandom.normal([D]) * 0.1
        let mlpNorm = LoweredNorm.rms(weight: mlpNormWeight, epsilon: 1e-5)

        let mlp = LoweredMLP(
            gateProj: LoweredProjection(weight: MLXRandom.normal([inter, D]) * 0.02),
            downProj: LoweredProjection(weight: MLXRandom.normal([D, inter]) * 0.02),
            upProj: LoweredProjection(weight: MLXRandom.normal([inter, D]) * 0.02),
            activation: .silu)

        // Build steps (unfused form)
        let unfusedSteps: [LoweredStep] = [
            .residual(body: [.op(.norm(attnNorm)), .op(.attention(attn))]),
            .residual(body: [.op(.norm(mlpNorm)), .op(.mlp(mlp))]),
        ]

        let x = MLXRandom.normal([1, 4, D]) * 0.1

        // Unfused execution via flattenSteps (which should auto-fuse)
        let fusedFlat = flattenSteps(unfusedSteps)
        #expect(fusedFlat.count == 2, "Expected 2 fused steps, got \(fusedFlat.count)")

        var fusedState = InferenceState(
            caches: [.kv(LoweredKVCache())], nextPosition: 0)
        let fusedResult = executeFlatSteps(fusedFlat, input: x, state: &fusedState)

        // Manual unfused execution
        var unfusedState = InferenceState(
            caches: [.kv(LoweredKVCache())], nextPosition: 0)
        var h = x
        // Attention sub-layer
        let normedAttn = attnNorm.apply(h)
        let attnOut = attn.apply(normedAttn, caches: &unfusedState.caches)
        h = h + attnOut
        // MLP sub-layer
        let normedMLP = mlpNorm.apply(h)
        let mlpOut = mlp.apply(normedMLP)
        h = h + mlpOut

        eval(fusedResult, h)

        let maxDiff = abs(fusedResult - h).max().item(Float.self)
        #expect(maxDiff < 1e-4, "Full block fused vs unfused diff: \(maxDiff)")
    }

    // MARK: - Step Count Verification

    @Test("Fusion reduces step count for N-layer model")
    func fusionReducesStepCount() throws {
        let D = 4
        let headCount = 2
        let headDim = 2
        let inter = 8
        let numLayers = 8

        var steps: [LoweredStep] = []

        // Embedding
        steps.append(.op(.tokenEmbedding(LoweredEmbedding(
            table: MLXRandom.normal([100, D])))))

        // N layers of [attn_sublayer, mlp_sublayer]
        for i in 0..<numLayers {
            let attnNorm = LoweredNorm.rms(
                weight: MLXRandom.normal([D]), epsilon: 1e-5)
            let attn = LoweredAttention(
                qProj: LoweredProjection(weight: MLXRandom.normal([D, D])),
                kProj: LoweredProjection(weight: MLXRandom.normal([D, D])),
                vProj: LoweredProjection(weight: MLXRandom.normal([D, D])),
                oProj: LoweredProjection(weight: MLXRandom.normal([D, D])),
                attrs: AttentionAttributes(
                    hiddenSize: D, headCount: headCount,
                    kvHeadCount: headCount, headDimension: headDim),
                qNormWeight: nil, kNormWeight: nil,
                qNormBias: nil, kNormBias: nil,
                cacheSlotIndex: i)
            steps.append(.residual(body: [.op(.norm(attnNorm)), .op(.attention(attn))]))

            let mlpNorm = LoweredNorm.rms(
                weight: MLXRandom.normal([D]), epsilon: 1e-5)
            let mlp = LoweredMLP(
                gateProj: LoweredProjection(weight: MLXRandom.normal([inter, D])),
                downProj: LoweredProjection(weight: MLXRandom.normal([D, inter])),
                upProj: LoweredProjection(weight: MLXRandom.normal([inter, D])),
                activation: .silu)
            steps.append(.residual(body: [.op(.norm(mlpNorm)), .op(.mlp(mlp))]))
        }

        // Final norm + output head
        steps.append(.op(.norm(LoweredNorm.rms(
            weight: MLXRandom.normal([D]), epsilon: 1e-5))))
        steps.append(.op(.outputHead(LoweredOutputHead(
            projection: LoweredProjection(weight: MLXRandom.normal([100, D])),
            isTied: false))))

        let flat = flattenSteps(steps)

        // Without fusion: 1 (emb) + 8*(4+4) (save/norm/op/add × 2) + 2 (norm+head) = 67
        // With fusion:    1 (emb) + 8*2 (fused × 2) + 2 (norm+head) = 19
        let expectedWithFusion = 1 + numLayers * 2 + 2
        #expect(flat.count == expectedWithFusion,
            "Expected \(expectedWithFusion) steps with fusion, got \(flat.count)")

        // Verify no saveResidual/addResidual markers remain
        let markerCount = flat.filter {
            if case .saveResidual = $0 { return true }
            if case .addResidual = $0 { return true }
            return false
        }.count
        #expect(markerCount == 0, "Expected 0 residual markers, got \(markerCount)")
    }
}
