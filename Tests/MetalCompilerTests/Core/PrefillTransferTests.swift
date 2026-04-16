import Testing
import Metal
@testable import MetalCompiler
import LMArchitecture
import ModelDeclarations
import LMIR

/// Verify prefill-to-decode state transfer correctness.
@Suite("Prefill Transfer")
struct PrefillTransferTests {

    // MARK: - Transfer Plan

    @Test("Transfer plan computes correct hidden source offset for last token")
    func hiddenSourceOffsetPointsToLastToken() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let hiddenSize = 128
        let maxSeqLen = 64
        let f32Size = MemoryLayout<Float32>.size

        _ = try #require(
            device.makeBuffer(length: hiddenSize * MemoryLayout<Float16>.size, options: .storageModeShared))
        _ = try #require(
            device.makeBuffer(length: maxSeqLen * hiddenSize * f32Size, options: .storageModeShared))

        let planner = MetalPrefillTransferPlanner()

        let config = makeTestConfig(hiddenSize: hiddenSize)
        let (decodePlan, prefillPlan) = try compilePlans(config: config, device: device, maxSeqLen: maxSeqLen)

        // Sequence length 10: last token is at index 9
        let transferPlan = planner.makeTransferPlan(
            prefillPlan: prefillPlan,
            decodePlan: decodePlan,
            sequenceLength: 10
        )

        let decodeElementSize = decodePlan.buffers.bufferPrecision.byteSize
        let decodeHiddenSize = decodePlan.buffers.hidden.length / decodeElementSize
        let expectedOffset = 9 * decodeHiddenSize * f32Size

        #expect(transferPlan.hiddenSourceOffset == expectedOffset,
                "Hidden source offset should point to last token (index 9), expected \(expectedOffset) got \(transferPlan.hiddenSourceOffset)")
    }

    @Test("Transfer plan uses GPU conversion for F16 decode precision")
    func f16DecodeUsesGPUConversion() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = makeTestConfig(hiddenSize: 128)
        let (decodePlan, prefillPlan) = try compilePlans(config: config, device: device, maxSeqLen: 64)

        let planner = MetalPrefillTransferPlanner()
        let transferPlan = planner.makeTransferPlan(
            prefillPlan: prefillPlan,
            decodePlan: decodePlan,
            sequenceLength: 10
        )

        if decodePlan.buffers.bufferPrecision != .float32 {
            #expect(transferPlan.hiddenConversionElementCount > 0,
                    "F16/BF16 decode should use GPU conversion (hiddenConversionElementCount > 0)")
            #expect(transferPlan.hiddenBlitCopySize == 0,
                    "F16/BF16 decode should not use blit copy for hidden")
        } else {
            #expect(transferPlan.hiddenConversionElementCount == 0,
                    "F32 decode should not need GPU conversion")
            #expect(transferPlan.hiddenBlitCopySize > 0,
                    "F32 decode should use blit copy for hidden")
        }
    }

    @Test("Transfer plan copies KV cache when prefill and decode use separate buffers")
    func kvCacheCopyWhenSeparateBuffers() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = makeTestConfig(hiddenSize: 128)
        let (decodePlan, prefillPlan) = try compilePlans(config: config, device: device, maxSeqLen: 64)

        let planner = MetalPrefillTransferPlanner()
        let transferPlan = planner.makeTransferPlan(
            prefillPlan: prefillPlan,
            decodePlan: decodePlan,
            sequenceLength: 10
        )

        // Both plans should have KV caches for a Transformer model
        if let prefillKV = prefillPlan.buffers.kvCache,
           let decodeKV = decodePlan.buffers.kvCache {
            if prefillKV.keys !== decodeKV.keys {
                if transferPlan.kvTransformSequenceLength > 0 {
                    #expect(transferPlan.kvCopySize == 0,
                            "Format-converting transfers should use GPU KV transform instead of raw key copies")
                    #expect(transferPlan.valueCopySize == 0,
                            "Format-converting transfers should use GPU KV transform instead of raw value copies")
                } else {
                    #expect(transferPlan.kvCopySize > 0,
                            "Separate same-format KV cache buffers should trigger a raw key copy")
                    #expect(transferPlan.valueCopySize > 0,
                            "Separate same-format KV cache value buffers should trigger a raw value copy")
                }
            } else {
                #expect(transferPlan.kvCopySize == 0,
                        "Shared KV cache should not trigger a copy")
            }
        }
    }

    @Test("Transfer plan uses KV transform for dense prefill into RotorQuant decode cache")
    func rotorDecodeUsesKVTransformAfterDensePrefill() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = makeTestConfig(hiddenSize: 128)
        let rotorDecodePolicy = InferencePolicy(
            maximumSequenceLength: 64,
            kvCache: KVCachePolicy(
                keyScheme: .fixed(.rotorQ8Group32ScaleF16),
                valueScheme: .fixed(.rotorQ4Group64ScaleF16),
                layoutMode: .sequenceMajor,
                qjlDimension: 16
            )
        )
        let (decodePlan, prefillPlan) = try compilePlans(
            config: config,
            device: device,
            maxSeqLen: 64,
            decodePolicy: rotorDecodePolicy,
            prefillPolicy: InferencePolicy(maximumSequenceLength: 64)
        )

        let planner = MetalPrefillTransferPlanner()
        let transferPlan = planner.makeTransferPlan(
            prefillPlan: prefillPlan,
            decodePlan: decodePlan,
            sequenceLength: 8
        )

        let prefillKV = try #require(prefillPlan.buffers.kvCache)
        let decodeKV = try #require(decodePlan.buffers.kvCache)
        #expect(!prefillKV.specification.usesRotorQuant,
                "Production prefill policy for RotorQuant decode should stay dense")
        #expect(decodeKV.specification.usesRotorQuant,
                "Decode policy should request RotorQuant buffers")
        #expect(transferPlan.kvTransformSequenceLength == 8,
                "Dense prefill into RotorQuant decode should use GPU KV transform")
        #expect(transferPlan.kvCopySize == 0,
                "Format-converting transfers should not fall back to raw key copies")
        #expect(transferPlan.valueCopySize == 0,
                "Format-converting transfers should not fall back to raw value copies")
        #expect(transferPlan.qjlResidualCopySize == 0,
                "RotorQuant decode should rebuild QJL residuals during KV transform instead of blitting them")
    }

    @Test("Transfer plan hidden conversion element count matches decode hidden size")
    func hiddenConversionCountMatchesDecodeHidden() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let hiddenSize = 256
        let config = makeTestConfig(hiddenSize: hiddenSize)
        let (decodePlan, prefillPlan) = try compilePlans(config: config, device: device, maxSeqLen: 64)

        let planner = MetalPrefillTransferPlanner()
        let transferPlan = planner.makeTransferPlan(
            prefillPlan: prefillPlan,
            decodePlan: decodePlan,
            sequenceLength: 32
        )

        if decodePlan.buffers.bufferPrecision != .float32 {
            let decodeElementSize = decodePlan.buffers.bufferPrecision.byteSize
            let expectedElements = decodePlan.buffers.hidden.length / decodeElementSize
            #expect(transferPlan.hiddenConversionElementCount == expectedElements,
                    "Conversion element count (\(transferPlan.hiddenConversionElementCount)) should match decode hidden element count (\(expectedElements))")
        }
    }

    // MARK: - Prefill Buffer Allocation

    @Test("Prefill buffers use correct storage modes")
    func prefillBufferStorageModes() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = makeTestConfig(hiddenSize: 128)
        let (_, prefillPlan) = try compilePlans(config: config, device: device, maxSeqLen: 64)

        // Hidden must be shared (vision model CPU access)
        #expect(prefillPlan.buffers.hidden.storageMode == .shared,
                "Prefill hidden must be storageModeShared for CPU access")

        // Residual and scratch should be private (GPU compression)
        #expect(prefillPlan.buffers.residual.storageMode == .private,
                "Prefill residual should be storageModePrivate for GPU compression")
        #expect(prefillPlan.buffers.scratch.storageMode == .private,
                "Prefill scratch should be storageModePrivate for GPU compression")
        #expect(prefillPlan.buffers.logits.storageMode == .shared,
                "Prefill logits must stay CPU-readable for host sampling and prompt-state token resolution")

        // Token I/O must be shared
        #expect(prefillPlan.buffers.tokenIDs.storageMode == .shared,
                "Prefill tokenIDs must be storageModeShared")
        #expect(prefillPlan.buffers.positions.storageMode == .shared,
                "Prefill positions must be storageModeShared")
        #expect(prefillPlan.buffers.tokenOut.storageMode == .shared,
                "Prefill tokenOut must be storageModeShared")
    }

    @Test("Decode buffers use correct storage modes")
    func decodeBufferStorageModes() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = makeTestConfig(hiddenSize: 128)
        let (decodePlan, _) = try compilePlans(config: config, device: device, maxSeqLen: 64)

        // GPU-only buffers should be private with hazard tracking disabled
        #expect(decodePlan.buffers.hidden.storageMode == .private,
                "Decode hidden should be storageModePrivate")
        #expect(decodePlan.buffers.residual.storageMode == .private,
                "Decode residual should be storageModePrivate")
        #expect(decodePlan.buffers.scratch.storageMode == .private,
                "Decode scratch should be storageModePrivate")
        #expect(decodePlan.buffers.logits.storageMode == .shared,
                "Decode logits must stay CPU-readable for host sampling")

        // CPU-accessible buffers
        #expect(decodePlan.buffers.tokenIn.storageMode == .shared,
                "Decode tokenIn must be storageModeShared")
        #expect(decodePlan.buffers.tokenOut.storageMode == .shared,
                "Decode tokenOut must be storageModeShared")
        #expect(decodePlan.buffers.position.storageMode == .shared,
                "Decode position must be storageModeShared")
    }

    // MARK: - Prefill Buffer Sizing

    @Test("Prefill hidden buffer is sized for full sequence at F32 precision")
    func prefillHiddenBufferSizing() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let hiddenSize = 256
        let maxSeqLen = 128
        let config = makeTestConfig(hiddenSize: hiddenSize)
        let (_, prefillPlan) = try compilePlans(config: config, device: device, maxSeqLen: maxSeqLen)

        let expectedSize = maxSeqLen * hiddenSize * MemoryLayout<Float32>.size
        #expect(prefillPlan.buffers.hidden.length == expectedSize,
                "Prefill hidden should be maxSeqLen × hiddenSize × sizeof(F32) = \(expectedSize), got \(prefillPlan.buffers.hidden.length)")
    }

    @Test("Prefill precision is always F32")
    func prefillPrecisionIsFloat32() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = makeTestConfig(hiddenSize: 128)
        let (_, prefillPlan) = try compilePlans(config: config, device: device, maxSeqLen: 64)

        #expect(prefillPlan.buffers.bufferPrecision == .float32,
                "Prefill must operate in F32 to prevent accumulated precision error")
    }

    // MARK: - End-to-End Prefill + Decode

    @Test("Prefill followed by decode produces valid position advancement")
    func prefillThenDecodePositionAdvancement() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = makeTestConfig(hiddenSize: 128)
        let graph = try ModelGraph(Transformer(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .llamaFamily)

        let compiler = MetalInferenceCompiler()
        let decodePlan = try compiler.compile(
            graph: resolved, hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize, device: device)
        let prefillPlan = try compiler.compilePrefill(
            graph: resolved, hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            device: device)

        var model = try MetalInferenceModel(plan: decodePlan, device: device)
        model.prefillPlan = prefillPlan

        // Prefill 5 tokens
        let tokens: [Int32] = [1, 2, 3, 4, 5]
        model.prefill(tokens: tokens)
        #expect(model.position == 5, "Position should be 5 after prefilling 5 tokens")

        // Decode 3 tokens
        var token: Int32 = 42
        for _ in 0..<3 {
            token = model.decodeSync(tokenID: token)
        }
        #expect(model.position == 8, "Position should be 8 after prefill(5) + decode(3)")
    }

    @Test("Multiple prefills accumulate position correctly")
    func multiplePrefillsAccumulatePosition() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = makeTestConfig(hiddenSize: 128)
        let graph = try ModelGraph(Transformer(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .llamaFamily)

        let compiler = MetalInferenceCompiler()
        let decodePlan = try compiler.compile(
            graph: resolved, hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize, device: device)
        let prefillPlan = try compiler.compilePrefill(
            graph: resolved, hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            device: device)

        var model = try MetalInferenceModel(plan: decodePlan, device: device)
        model.prefillPlan = prefillPlan

        // First prefill
        model.prefill(tokens: [1, 2, 3])
        #expect(model.position == 3)

        // Second prefill (continuation)
        model.prefill(tokens: [4, 5])
        #expect(model.position == 5)

        // Decode
        _ = model.decodeSync(tokenID: 42)
        #expect(model.position == 6)
    }

    // MARK: - Helpers

    private func makeTestConfig(hiddenSize: Int) -> ModelConfig {
        ModelConfig(
            hiddenSize: hiddenSize, layerCount: 2,
            intermediateSize: hiddenSize * 4,
            vocabSize: 1000, attentionHeads: 4, kvHeads: 4,
            headDim: hiddenSize / 4,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 10000,
            ropeDimension: hiddenSize / 4,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: false,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: nil,
            partialRotaryFactor: nil, slidingWindow: nil
        )
    }

    private func compilePlans(
        config: ModelConfig,
        device: MTLDevice,
        maxSeqLen: Int,
        decodePolicy: InferencePolicy = .default,
        prefillPolicy: InferencePolicy? = nil
    ) throws -> (MetalDispatchPlan, MetalPrefillPlan) {
        let graph = try ModelGraph(Transformer(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .llamaFamily)
        let compiler = MetalInferenceCompiler()
        let decodePlan = try compiler.compile(
            graph: resolved, hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: decodePolicy,
            device: device)
        let prefillPlan = try compiler.compilePrefill(
            graph: resolved, hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: prefillPolicy ?? InferencePolicy(maximumSequenceLength: maxSeqLen),
            device: device)
        return (decodePlan.decodePlan, prefillPlan)
    }
}
