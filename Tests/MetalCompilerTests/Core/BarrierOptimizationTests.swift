import Testing
import Metal
import Foundation
@testable import MetalCompiler
import LMArchitecture
import ModelDeclarations
import LMIR

/// Verify barrier optimization correctly inserts/elides GPU memory barriers.
@Suite("Barrier Optimization")
struct BarrierOptimizationTests {

    // MARK: - BufferRegion Identity

    @Test("BufferRegion distinguishes same buffer at different offsets")
    func bufferRegionOffsetDistinction() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }
        let buffer = try #require(device.makeBuffer(length: 4096, options: .storageModeShared))

        let regionA = BufferRegion(buffer: buffer, offset: 0)
        let regionB = BufferRegion(buffer: buffer, offset: 1024)
        let regionC = BufferRegion(buffer: buffer, offset: 0)

        #expect(regionA != regionB, "Same buffer at different offsets must be distinct")
        #expect(regionA == regionC, "Same buffer at same offset must be equal")
        #expect(regionA.hashValue == regionC.hashValue)
    }

    @Test("BufferRegion distinguishes different buffers at same offset")
    func bufferRegionBufferDistinction() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }
        let bufferA = try #require(device.makeBuffer(length: 1024, options: .storageModeShared))
        let bufferB = try #require(device.makeBuffer(length: 1024, options: .storageModeShared))

        let regionA = BufferRegion(buffer: bufferA, offset: 0)
        let regionB = BufferRegion(buffer: bufferB, offset: 0)

        #expect(regionA != regionB, "Different buffers at same offset must be distinct")
    }

    // MARK: - MetalBufferAccesses Barrier Decision

    @Test("No barrier needed when reads and writes are disjoint from pending writes")
    func disjointAccessesNoBarrier() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }
        let scratch = try #require(device.makeBuffer(length: 8192, options: .storageModeShared))
        let hidden = try #require(device.makeBuffer(length: 1024, options: .storageModeShared))

        let pendingWrites: Set<BufferRegion> = [
            BufferRegion(buffer: scratch, offset: 0)
        ]

        // Step reads hidden, writes scratch at different offset
        let accesses = MetalBufferAccesses(
            readBuffers: [(buffer: hidden, offset: 0)],
            writeBuffers: [(buffer: scratch, offset: 4096)]
        )

        #expect(!accesses.requiresBarrier(after: pendingWrites),
                "Disjoint regions should not require a barrier")
    }

    @Test("Barrier needed when reading a region with pending write")
    func readAfterWriteRequiresBarrier() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }
        let buffer = try #require(device.makeBuffer(length: 4096, options: .storageModeShared))

        let pendingWrites: Set<BufferRegion> = [
            BufferRegion(buffer: buffer, offset: 0)
        ]

        let accesses = MetalBufferAccesses(
            readBuffers: [(buffer: buffer, offset: 0)],
            writeBuffers: []
        )

        #expect(accesses.requiresBarrier(after: pendingWrites),
                "Read-after-write on same region must require a barrier")
    }

    @Test("Barrier needed when writing to a region with pending write (WAW)")
    func writeAfterWriteRequiresBarrier() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }
        let buffer = try #require(device.makeBuffer(length: 4096, options: .storageModeShared))

        let pendingWrites: Set<BufferRegion> = [
            BufferRegion(buffer: buffer, offset: 0)
        ]

        let accesses = MetalBufferAccesses(
            readBuffers: [],
            writeBuffers: [(buffer: buffer, offset: 0)]
        )

        #expect(accesses.requiresBarrier(after: pendingWrites),
                "Write-after-write on same region must require a barrier")
    }

    @Test("Barrier needed when writing to a region with pending read (WAR)")
    func writeAfterReadRequiresBarrier() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }
        let buffer = try #require(device.makeBuffer(length: 4096, options: .storageModeShared))

        let pendingReads: Set<BufferRegion> = [
            BufferRegion(buffer: buffer, offset: 0)
        ]

        let accesses = MetalBufferAccesses(
            readBuffers: [],
            writeBuffers: [(buffer: buffer, offset: 0)]
        )

        #expect(
            accesses.requiresBarrier(after: pendingReads, pendingWrites: []),
            "Write-after-read on same region must require a barrier"
        )
    }

    @Test("No barrier when pending writes set is empty")
    func emptyPendingWritesNoBarrier() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }
        let buffer = try #require(device.makeBuffer(length: 1024, options: .storageModeShared))

        let accesses = MetalBufferAccesses(
            readBuffers: [(buffer: buffer, offset: 0)],
            writeBuffers: [(buffer: buffer, offset: 0)]
        )

        #expect(!accesses.requiresBarrier(after: []),
                "Empty pending writes means no hazard")
    }

    // MARK: - Scratch Slot Independence

    @Test("Independent scratch slots do not trigger false barriers")
    func scratchSlotIndependence() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }
        let scratch = try #require(device.makeBuffer(length: 16384, options: .storageModeShared))
        let slotStride = 4096

        // Step 1 writes to slot 1
        let slot1Write: Set<BufferRegion> = [
            BufferRegion(buffer: scratch, offset: 1 * slotStride)
        ]

        // Step 2 reads slot 0 (RMSNorm output) and writes slot 2
        let step2Accesses = MetalBufferAccesses(
            readBuffers: [(buffer: scratch, offset: 0 * slotStride)],
            writeBuffers: [(buffer: scratch, offset: 2 * slotStride)]
        )

        #expect(!step2Accesses.requiresBarrier(after: slot1Write),
                "Reading slot 0 after writing slot 1 should not require a barrier")

        // Step 3 reads slot 1 (the written slot) — this DOES need a barrier
        let step3Accesses = MetalBufferAccesses(
            readBuffers: [(buffer: scratch, offset: 1 * slotStride)],
            writeBuffers: []
        )

        #expect(step3Accesses.requiresBarrier(after: slot1Write),
                "Reading slot 1 after writing slot 1 requires a barrier")
    }

    // MARK: - Conservative Fallback

    @Test("Conservative access treats all bindings as read and write")
    func conservativeAccessMarksAllRegions() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }
        let bufferA = try #require(device.makeBuffer(length: 1024, options: .storageModeShared))
        let bufferB = try #require(device.makeBuffer(length: 1024, options: .storageModeShared))

        let bindings = [
            MetalBufferBinding(index: 0, buffer: bufferA, offset: 0),
            MetalBufferBinding(index: 1, buffer: bufferB, offset: 0)
        ]

        let conservative = MetalBufferAccesses.conservative(bindings)

        #expect(conservative.reads.count == 2, "Conservative: all bindings are reads")
        #expect(conservative.writes.count == 2, "Conservative: all bindings are writes")
        #expect(conservative.reads == conservative.writes, "Conservative: reads == writes")
    }

    // MARK: - Compile Plan Barrier Statistics

    #if ENABLE_METAL_PROBES
    @Test("Compiled decode plan has fewer barriers than steps")
    func compiledPlanBarrierReduction() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = ModelConfig(
            hiddenSize: 128, layerCount: 2, intermediateSize: 512,
            vocabSize: 1000, attentionHeads: 4, kvHeads: 4, headDim: 32,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 10000, ropeDimension: 32,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: false,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: nil,
            partialRotaryFactor: nil, slidingWindow: nil
        )
        let graph = try ModelGraph(Transformer(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .llamaFamily)
        let store = try makeWeightStore(for: resolved, device: device)
        let compiled = try MetalInferenceCompiler().compile(
            graph: resolved, hiddenSize: 128, intermediateSize: 512,
            vocabSize: 1000, stafWeightStore: store, device: device)

        let totalSteps = compiled.steps.count
        let barrierSteps = compiled.steps.filter { $0.barrierPolicy.isBarrier }.count
        let noBarrierSteps = totalSteps - barrierSteps

        print("[Barrier test] \(totalSteps) steps: \(barrierSteps) barriers, \(noBarrierSteps) elided")

        #expect(totalSteps > 0, "Plan must have at least one step")
        #expect(noBarrierSteps > 0,
                "Barrier optimization must elide at least one barrier (got \(barrierSteps)/\(totalSteps))")
    }

    @Test("First step never has a barrier")
    func firstStepNoBarrier() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = ModelConfig(
            hiddenSize: 128, layerCount: 2, intermediateSize: 512,
            vocabSize: 1000, attentionHeads: 4, kvHeads: 4, headDim: 32,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 10000, ropeDimension: 32,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: false,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: nil,
            partialRotaryFactor: nil, slidingWindow: nil
        )
        let graph = try ModelGraph(Transformer(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .llamaFamily)
        let store = try makeWeightStore(for: resolved, device: device)
        let compiled = try MetalInferenceCompiler().compile(
            graph: resolved, hiddenSize: 128, intermediateSize: 512,
            vocabSize: 1000, stafWeightStore: store, device: device)

        let firstStep = try #require(compiled.steps.first)
        #expect(firstStep.barrierPolicy == .none,
                "First step should have no barrier (no prior writes to fence)")
    }
    #endif

    private func makeWeightStore(
        for graph: ModelGraph,
        device: MTLDevice
    ) throws -> STAFWeightStore {
        let payloadSize = 4 * 1024 * 1024
        guard let buffer = device.makeBuffer(length: payloadSize, options: .storageModeShared) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate dummy STAF weight buffer")
        }

        var entries: [String: STAFTensorEntry] = [:]
        var names = tensorNames(in: graph.rootRegion)
        names.formUnion([
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight"
        ])
        for layerIndex in 0..<8 {
            let prefix = "model.layers.\(layerIndex)"
            names.formUnion([
                "\(prefix).input_layernorm.weight",
                "\(prefix).self_attn.q_proj.weight",
                "\(prefix).self_attn.k_proj.weight",
                "\(prefix).self_attn.v_proj.weight",
                "\(prefix).self_attn.o_proj.weight",
                "\(prefix).post_attention_layernorm.weight",
                "\(prefix).mlp.gate_proj.weight",
                "\(prefix).mlp.up_proj.weight",
                "\(prefix).mlp.down_proj.weight"
            ])
        }

        for tensorName in names {
            entries[tensorName] = STAFTensorEntry(
                name: tensorName,
                payloadOffset: 0,
                payloadSize: payloadSize,
                schemeIdentifier: .passthrough,
                semanticRole: .unknown,
                shape: [payloadSize / MemoryLayout<UInt16>.stride],
                blockSize: 0,
                groupSize: 0,
                bufferOffset: 0
            )
        }

        return STAFWeightStore(
            buffer: buffer,
            entries: entries,
            metadata: .empty,
            specializedBufferAccesses: [:]
        )
    }

    private func tensorNames(in region: Region) -> Set<String> {
        var names = Set(region.operations.flatMap { operation in
            operation.parameterBindings.map(\.tensorName)
        })

        for operation in region.operations {
            switch operation.kind {
            case .primitive:
                break
            case .residual(_, let body):
                names.formUnion(tensorNames(in: body))
            case .parallel(_, let branches):
                for branch in branches {
                    names.formUnion(tensorNames(in: branch))
                }
            case .repeating(_, let body):
                names.formUnion(tensorNames(in: body))
            case .conditional(_, let thenRegion, let elseRegion):
                names.formUnion(tensorNames(in: thenRegion))
                names.formUnion(tensorNames(in: elseRegion))
            }
        }

        return names
    }
}
