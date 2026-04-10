import Metal

struct MetalResidencyLease: @unchecked Sendable {
    let label: String?
    let sets: [MTLResidencySet]
    let trackedBufferCount: Int

    static let empty = MetalResidencyLease(label: nil, sets: [], trackedBufferCount: 0)

    var isEmpty: Bool { sets.isEmpty }
    var setCount: Int { sets.count }

    static func required(
        device: MTLDevice,
        label: String,
        buffers: [MTLBuffer]
    ) throws -> MetalResidencyLease {
        let uniqueBuffers = deduplicated(buffers)
        guard !uniqueBuffers.isEmpty else { return .empty }

        let descriptor = MTLResidencySetDescriptor()
        descriptor.label = label
        let residencySet = try device.makeResidencySet(descriptor: descriptor)
        for buffer in uniqueBuffers {
            residencySet.addAllocation(buffer)
        }
        residencySet.commit()
        residencySet.requestResidency()
        return MetalResidencyLease(
            label: label,
            sets: [residencySet],
            trackedBufferCount: uniqueBuffers.count
        )
    }

    func use(on commandBuffer: MTL4CommandBuffer) {
        for residencySet in sets {
            commandBuffer.useResidencySet(residencySet)
        }
    }

    func add(to queue: MTL4CommandQueue) {
        for residencySet in sets {
            queue.addResidencySet(residencySet)
        }
    }

    func remove(from queue: MTL4CommandQueue) {
        for residencySet in sets {
            queue.removeResidencySet(residencySet)
        }
    }

    private static func deduplicated(_ buffers: [MTLBuffer]) -> [MTLBuffer] {
        var seen = Set<ObjectIdentifier>()
        var uniqueBuffers: [MTLBuffer] = []
        uniqueBuffers.reserveCapacity(buffers.count)
        for buffer in buffers {
            let identifier = ObjectIdentifier(buffer as AnyObject)
            guard seen.insert(identifier).inserted else { continue }
            uniqueBuffers.append(buffer)
        }
        return uniqueBuffers
    }
}

struct MetalStableResidencyRegistry: @unchecked Sendable {
    let weightLease: MetalResidencyLease
    let runtimeLease: MetalResidencyLease
    let supplementalLease: MetalResidencyLease

    var setCount: Int {
        weightLease.setCount + runtimeLease.setCount + supplementalLease.setCount
    }

    init(
        device: MTLDevice,
        compiledModel: MetalCompiledModel,
        hiddenOverrideConstantBuffer: MTLBuffer
    ) throws {
        let runtimeBuffers = Self.prune(
            compiledModel.stableRuntimeResidencyBuffers(hiddenOverrideConstantBuffer: hiddenOverrideConstantBuffer),
            excluding: [],
            queuePolicy: .allBuffers
        )
        let weightBuffers = Self.prune(
            compiledModel.stableWeightResidencyBuffers,
            excluding: runtimeBuffers,
            queuePolicy: .allBuffers
        )
        let supplementalBuffers = Self.prune(
            compiledModel.stableSupplementalResidencyBuffers,
            excluding: runtimeBuffers + weightBuffers,
            queuePolicy: .allBuffers
        )

        self.runtimeLease = try MetalResidencyLease.required(
            device: device,
            label: "swift-lm.runtime",
            buffers: runtimeBuffers
        )
        self.weightLease = try MetalResidencyLease.required(
            device: device,
            label: "swift-lm.weights",
            buffers: weightBuffers
        )
        self.supplementalLease = try MetalResidencyLease.required(
            device: device,
            label: "swift-lm.supplemental",
            buffers: supplementalBuffers
        )
    }

    func register(on queue: MTL4CommandQueue) {
        runtimeLease.add(to: queue)
        weightLease.add(to: queue)
        supplementalLease.add(to: queue)
    }

    func remove(from queue: MTL4CommandQueue) {
        runtimeLease.remove(from: queue)
        weightLease.remove(from: queue)
        supplementalLease.remove(from: queue)
    }

    private enum QueueResidencyPolicy {
        case allBuffers
        case gpuOwnedOnly
    }

    private static func prune(
        _ buffers: [MTLBuffer],
        excluding excluded: [MTLBuffer],
        queuePolicy: QueueResidencyPolicy
    ) -> [MTLBuffer] {
        let excludedIdentifiers = Set(excluded.map { ObjectIdentifier($0 as AnyObject) })
        var seen = Set<ObjectIdentifier>()
        var result: [MTLBuffer] = []
        result.reserveCapacity(buffers.count)
        for buffer in buffers {
            guard isQueueResidencyEligible(buffer, policy: queuePolicy) else { continue }
            let identifier = ObjectIdentifier(buffer as AnyObject)
            guard !excludedIdentifiers.contains(identifier) else { continue }
            guard seen.insert(identifier).inserted else { continue }
            result.append(buffer)
        }
        return result
    }

    private static func isQueueResidencyEligible(
        _ buffer: MTLBuffer,
        policy: QueueResidencyPolicy
    ) -> Bool {
        switch policy {
        case .allBuffers:
            switch buffer.storageMode {
            case .shared, .private, .managed:
                return true
            case .memoryless:
                return false
            @unknown default:
                return false
            }
        case .gpuOwnedOnly:
            switch buffer.storageMode {
            case .private, .managed:
                return true
            case .shared, .memoryless:
                return false
            @unknown default:
                return false
            }
        }
    }

    static func debugTrackedBufferCounts(
        compiledModel: MetalCompiledModel,
        hiddenOverrideConstantBuffer: MTLBuffer
    ) -> (runtime: Int, weights: Int, supplemental: Int) {
        let runtimeBuffers = prune(
            compiledModel.stableRuntimeResidencyBuffers(hiddenOverrideConstantBuffer: hiddenOverrideConstantBuffer),
            excluding: [],
            queuePolicy: .allBuffers
        )
        let weightBuffers = prune(
            compiledModel.stableWeightResidencyBuffers,
            excluding: runtimeBuffers,
            queuePolicy: .allBuffers
        )
        let supplementalBuffers = prune(
            compiledModel.stableSupplementalResidencyBuffers,
            excluding: runtimeBuffers + weightBuffers,
            queuePolicy: .allBuffers
        )
        return (runtimeBuffers.count, weightBuffers.count, supplementalBuffers.count)
    }
}

extension MetalCompiledModel {
    fileprivate var stableWeightResidencyBuffers: [MTLBuffer] {
        decodePlan.buffers.weightResidencyBuffers
        + (prefillPlan?.buffers.weightResidencyBuffers ?? [])
    }

    fileprivate func stableRuntimeResidencyBuffers(hiddenOverrideConstantBuffer: MTLBuffer) -> [MTLBuffer] {
        decodePlan.buffers.runtimeResidencyBuffers
        + (prefillPlan?.buffers.runtimeResidencyBuffers ?? [])
        + [hiddenOverrideConstantBuffer]
    }

    fileprivate var stableSupplementalResidencyBuffers: [MTLBuffer] {
        decodePlan.supplementalResidencyBuffers
        + (prefillPlan?.supplementalResidencyBuffers ?? [])
    }
}
