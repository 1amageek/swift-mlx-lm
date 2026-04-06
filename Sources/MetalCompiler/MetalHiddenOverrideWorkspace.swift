import Metal

final class MetalHiddenOverrideWorkspace: @unchecked Sendable {
    private let device: MTLDevice
    let hiddenElementCount: Int

    private(set) var hiddenStagingBuffer: MTLBuffer
    private var deepstackBuffersByLayer: [Int: MTLBuffer] = [:]

    init(device: MTLDevice, hiddenElementCount: Int) throws {
        self.device = device
        self.hiddenElementCount = hiddenElementCount
        self.hiddenStagingBuffer = try Self.makeSharedFloatBuffer(
            device: device,
            elementCount: hiddenElementCount
        )
    }

    func writeHiddenState(_ values: [Float]) throws -> MTLBuffer {
        guard values.count == hiddenElementCount else {
            throw MetalCompilerError.deviceSetupFailed("Hidden state override dimension mismatch")
        }
        hiddenStagingBuffer.contents()
            .bindMemory(to: Float.self, capacity: values.count)
            .update(from: values, count: values.count)
        return hiddenStagingBuffer
    }

    func writeDeepstackFeatures(_ features: [Int: [Float]]) throws -> [Int: MTLBuffer] {
        var buffers: [Int: MTLBuffer] = [:]
        buffers.reserveCapacity(features.count)
        for (layerIndex, values) in features {
            let buffer = try deepstackBuffer(for: layerIndex, elementCount: values.count)
            buffer.contents()
                .bindMemory(to: Float.self, capacity: values.count)
                .update(from: values, count: values.count)
            buffers[layerIndex] = buffer
        }
        return buffers
    }

    func writeDeepstackFeatures(
        byLayer featuresByLayer: [Int: [[Float]]],
        tokenIndex: Int
    ) throws -> [Int: MTLBuffer] {
        var buffers: [Int: MTLBuffer] = [:]
        buffers.reserveCapacity(featuresByLayer.count)
        for (layerIndex, tokenFeatures) in featuresByLayer {
            guard tokenIndex < tokenFeatures.count else {
                throw MetalCompilerError.deviceSetupFailed(
                    "Deepstack visual feature count mismatch at layer \(layerIndex)"
                )
            }
            let values = tokenFeatures[tokenIndex]
            let buffer = try deepstackBuffer(for: layerIndex, elementCount: values.count)
            buffer.contents()
                .bindMemory(to: Float.self, capacity: values.count)
                .update(from: values, count: values.count)
            buffers[layerIndex] = buffer
        }
        return buffers
    }

    private func deepstackBuffer(for layerIndex: Int, elementCount: Int) throws -> MTLBuffer {
        if let existing = deepstackBuffersByLayer[layerIndex],
           existing.length == elementCount * MemoryLayout<Float>.stride {
            return existing
        }
        let buffer = try Self.makeSharedFloatBuffer(device: device, elementCount: elementCount)
        deepstackBuffersByLayer[layerIndex] = buffer
        return buffer
    }

    private static func makeSharedFloatBuffer(
        device: MTLDevice,
        elementCount: Int
    ) throws -> MTLBuffer {
        let byteCount = elementCount * MemoryLayout<Float>.stride
        guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate shared staging buffer")
        }
        return buffer
    }
}
