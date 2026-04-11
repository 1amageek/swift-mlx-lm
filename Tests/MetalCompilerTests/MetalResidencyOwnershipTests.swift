import Testing
import Metal
@testable import MetalCompiler

@Suite("Metal Residency Ownership", .serialized)
struct MetalResidencyOwnershipTests {

    @Test("resourceBarrier equality compares resource identity, not just count")
    func resourceBarrierEqualityUsesResourceIdentity() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let a = try #require(device.makeBuffer(length: 256, options: .storageModeShared))
        let b = try #require(device.makeBuffer(length: 256, options: .storageModeShared))
        let c = try #require(device.makeBuffer(length: 256, options: .storageModeShared))

        let lhs = MetalBarrierPolicy.resourceBarrier(resources: [a, b])
        let sameResourcesDifferentOrder = MetalBarrierPolicy.resourceBarrier(resources: [b, a])
        let differentResourcesSameCount = MetalBarrierPolicy.resourceBarrier(resources: [a, c])

        #expect(lhs == sameResourcesDifferentOrder)
        #expect(lhs != differentResourcesSameCount)
    }

    @Test("binding table supplemental residency buffers include resident constants and prepared/encoded argument buffers")
    func bindingTableOwnedResidencyBuffers() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let resource = try #require(device.makeBuffer(length: 256, options: .storageModeShared))
        let prepared = try #require(device.makeBuffer(length: 512, options: .storageModeShared))
        let encoded = try #require(device.makeBuffer(length: 1024, options: .storageModeShared))
        let constantArena = try #require(device.makeBuffer(length: 128, options: .storageModeShared))

        let preparedTable = MetalBindingTable(
            bufferBindings: .argumentTable(MetalArgumentTableBindings(
                layout: MetalArgumentTableLayout(id: 1, indices: [0]),
                bindings: [MetalBufferBinding(index: 0, buffer: resource, offset: 0)],
                encodingState: .prepared(buffer: prepared, index: 30, offset: 0)
            )),
            constantBindings: .resident(MetalResidentConstantBindings(
                buffer: constantArena,
                bindings: [MetalConstantBufferBinding(index: 1, buffer: constantArena, offset: 0, length: 16)]
            ))
        )

        let encodedTable = MetalBindingTable(
            bufferBindings: .argumentTable(MetalArgumentTableBindings(
                layout: MetalArgumentTableLayout(id: 2, indices: [0]),
                bindings: [MetalBufferBinding(index: 0, buffer: resource, offset: 0)],
                encodingState: .encoded(buffer: encoded, index: 30, offset: 0)
            )),
            constantBindings: .inline([])
        )

        #expect(preparedTable.ownedResidencyBuffers.count == 2)
        #expect(encodedTable.ownedResidencyBuffers.count == 1)
        #expect(preparedTable.ownedResidencyBuffers.contains { $0 === prepared })
        #expect(preparedTable.ownedResidencyBuffers.contains { $0 === constantArena })
        #expect(encodedTable.ownedResidencyBuffers.contains { $0 === encoded })
    }

    @Test("prompt state owns snapshot residency for restore")
    func promptStateOwnsSnapshotResidency() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let bufferSet = try makeDecodeBufferSet(device: device)
        let plan = MetalDispatchPlan(
            steps: [],
            buffers: bufferSet,
            unfusedEntryCount: 0,
            fusedEntryCount: 0,
            supplementalResidencyBuffers: []
        )
        var submission = try MetalSubmissionContext(device: device)
        let store = MetalPromptStateStore()

        let promptState = try store.makePromptSnapshot(
            plan: plan,
            submission: &submission,
            position: 7,
            firstToken: 42
        )

        #expect(promptState.residencyLease.setCount == 1)
        #expect(promptState.residencyLease.trackedBufferCount == 8)

        try store.restore(plan: plan, submission: &submission, promptState: promptState)
    }

    @Test("hidden override workspace rebuilds residency when deepstack buffers grow")
    func hiddenOverrideWorkspaceRebuildsResidency() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let workspace = try MetalHiddenOverrideWorkspace(device: device, hiddenElementCount: 16)
        #expect(workspace.residencyLease.trackedBufferCount == 1)

        _ = try workspace.writeDeepstackFeatures([0: Array(repeating: 1.0, count: 8)])
        #expect(workspace.residencyLease.trackedBufferCount == 2)

        _ = try workspace.writeDeepstackFeatures([1: Array(repeating: 1.0, count: 8)])
        #expect(workspace.residencyLease.trackedBufferCount == 3)

        _ = try workspace.writeDeepstackFeatures([1: Array(repeating: 1.0, count: 16)])
        #expect(workspace.residencyLease.trackedBufferCount == 3)
    }

    @Test("stable residency registry groups runtime, weight, and supplemental buffers")
    func stableResidencyRegistryTopology() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let privateWeight = try #require(device.makeBuffer(length: 4096, options: .storageModePrivate))
        let supplemental = try #require(device.makeBuffer(length: 512, options: .storageModePrivate))
        let hiddenOverrideConstantBuffer = try #require(device.makeBuffer(length: 4, options: .storageModeShared))

        let decodePlan = MetalDispatchPlan(
            steps: [],
            buffers: try makeDecodeBufferSet(device: device, weights: [privateWeight]),
            unfusedEntryCount: 0,
            fusedEntryCount: 0,
            supplementalResidencyBuffers: [supplemental]
        )
        let prefillBuffers = try makePrefillBufferSet(device: device, weights: [privateWeight])
        let prefillPlan = MetalPrefillPlan(
            steps: [],
            buffers: prefillBuffers,
            slotDimension: 128,
            maximumSequenceLength: 4,
            stepCount: 0,
            finalHiddenBuffer: prefillBuffers.hidden,
            finalHiddenBaseOffset: 0,
            finalHiddenRowStride: prefillBuffers.hidden.length / 4,
            supplementalResidencyBuffers: [supplemental]
        )
        let compiledModel = MetalCompiledModel(
            decodePlan: decodePlan,
            prefillPlan: prefillPlan
        )

        let registry = try MetalStableResidencyRegistry(
            device: device,
            compiledModel: compiledModel,
            hiddenOverrideConstantBuffer: hiddenOverrideConstantBuffer
        )

        #expect(registry.setCount == 3)
        #expect(registry.weightLease.trackedBufferCount == 1)
        #expect(registry.runtimeLease.trackedBufferCount >= 7)
        #expect(registry.supplementalLease.trackedBufferCount == 1)
    }

    private func makeDecodeBufferSet(
        device: MTLDevice,
        weights: [MTLBuffer] = []
    ) throws -> MetalBufferSet {
        MetalBufferSet(
            bufferPrecision: .float16,
            hidden: try requiredPrivateBuffer(device, length: 256),
            residual: try requiredPrivateBuffer(device, length: 256),
            scratch: try requiredPrivateBuffer(device, length: 512),
            weights: weights,
            kvCache: nil,
            convState: nil,
            recurrentState: nil,
            convStateDimension: 0,
            convStateKernelSize: 0,
            recurrentStateBytesPerLayer: 0,
            perLayerInputs: nil,
            perLayerInputDimension: 0,
            perLayerInputLayerCount: 0,
            logits: try requiredPrivateBuffer(device, length: 1024),
            position: try requiredSharedBuffer(device, length: 4),
            ropePositionAxes: try requiredSharedBuffer(device, length: 12),
            tokenIn: try requiredSharedBuffer(device, length: 4),
            tokenOut: try requiredSharedBuffer(device, length: 4)
        )
    }

    private func makePrefillBufferSet(
        device: MTLDevice,
        weights: [MTLBuffer] = []
    ) throws -> PrefillBufferSet {
        PrefillBufferSet(
            bufferPrecision: .float32,
            hidden: try requiredSharedBuffer(device, length: 1024),
            residual: try requiredPrivateBuffer(device, length: 1024),
            scratch: try requiredPrivateBuffer(device, length: 2048),
            weights: weights,
            kvCache: nil,
            convState: nil,
            recurrentState: nil,
            convStateDimension: 0,
            convStateKernelSize: 0,
            recurrentStateBytesPerLayer: 0,
            perLayerInputs: nil,
            perLayerInputDimension: 0,
            perLayerInputLayerCount: 0,
            logits: try requiredPrivateBuffer(device, length: 1024),
            tokenIDs: try requiredSharedBuffer(device, length: 16),
            positions: try requiredSharedBuffer(device, length: 16),
            ropePositionAxes: try requiredSharedBuffer(device, length: 48),
            tokenOut: try requiredSharedBuffer(device, length: 4),
            runtimeConstantBuffer: try requiredSharedBuffer(device, length: 32)
        )
    }

    private func requiredSharedBuffer(_ device: MTLDevice, length: Int) throws -> MTLBuffer {
        try #require(device.makeBuffer(length: length, options: .storageModeShared))
    }

    private func requiredPrivateBuffer(_ device: MTLDevice, length: Int) throws -> MTLBuffer {
        try #require(device.makeBuffer(length: length, options: .storageModePrivate))
    }
}
