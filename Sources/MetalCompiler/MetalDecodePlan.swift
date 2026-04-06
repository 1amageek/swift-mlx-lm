import Metal

// MARK: - Decode Dispatch Plan

/// A single GPU dispatch step in the decode plan.
public struct MetalDispatchStep: @unchecked Sendable {
    public let descriptor: MetalDispatchDescriptor
    public let bindings: MetalBindingTable
    public let bufferAccesses: MetalBufferAccesses
    public let metadata: MetalDispatchStepMetadata

    public var pipeline: MTLComputePipelineState { descriptor.pipeline }
    public var gridSize: MTLSize { descriptor.gridSize }
    public var threadgroupSize: MTLSize { descriptor.threadgroupSize }
    public var threadgroupMemoryLength: Int { descriptor.threadgroupMemoryLength }
    public var sync: SynchronizationKind { descriptor.sync }
    public var barrierPolicy: MetalBarrierPolicy { descriptor.barrierPolicy }
    public var bufferBindings: [(index: Int, buffer: MTLBuffer, offset: Int)] {
        bindings.buffers.map { (index: $0.index, buffer: $0.buffer, offset: $0.offset) }
    }
    public var bytesBindings: [(index: Int, value: [UInt8])] {
        bindings.constantBindings.inlineBindings.map { (index: $0.index, value: $0.value) }
    }

    public init(
        pipeline: MTLComputePipelineState,
        gridSize: MTLSize,
        threadgroupSize: MTLSize,
        bufferBindings: [(index: Int, buffer: MTLBuffer, offset: Int)],
        bytesBindings: [(index: Int, value: [UInt8])],
        threadgroupMemoryLength: Int,
        sync: SynchronizationKind,
        bufferAccesses: MetalBufferAccesses? = nil,
        metadata: MetalDispatchStepMetadata = .init()
    ) {
        let mappedBindings = bufferBindings.map { MetalBufferBinding(index: $0.index, buffer: $0.buffer, offset: $0.offset) }
        self.descriptor = MetalDispatchDescriptor(
            pipeline: pipeline,
            gridSize: gridSize,
            threadgroupSize: threadgroupSize,
            threadgroupMemoryLength: threadgroupMemoryLength,
            barrierPolicy: MetalBarrierPolicy(sync))
        self.bindings = MetalBindingTable(
            bufferBindings: bufferBindings,
            bytesBindings: bytesBindings)
        self.bufferAccesses = bufferAccesses ?? MetalBufferAccesses.conservative(mappedBindings)
        self.metadata = metadata
    }

    public init(
        descriptor: MetalDispatchDescriptor,
        bindings: MetalBindingTable,
        bufferAccesses: MetalBufferAccesses? = nil,
        metadata: MetalDispatchStepMetadata = .init()
    ) {
        self.descriptor = descriptor
        self.bindings = bindings
        self.bufferAccesses = bufferAccesses ?? MetalBufferAccesses.conservative(bindings.buffers)
        self.metadata = metadata
    }
}

public struct MetalDispatchStepMetadata: Sendable, Equatable {
    public let kernelName: String?
    public let layerIndex: Int?

    public init(kernelName: String? = nil, layerIndex: Int? = nil) {
        self.kernelName = kernelName
        self.layerIndex = layerIndex
    }
}

/// Complete decode dispatch plan: steps + buffers.
public struct MetalDispatchPlan: @unchecked Sendable {
    public let steps: [MetalDispatchStep]
    public let buffers: MetalBufferSet
    public let unfusedEntryCount: Int
    public let fusedEntryCount: Int
}

/// Decode buffer set (single-token layout).
public struct MetalBufferSet: @unchecked Sendable {
    public let bufferPrecision: BufferPrecision
    public let hidden: MTLBuffer
    public let residual: MTLBuffer
    public let scratch: MTLBuffer
    public let weights: [MTLBuffer]
    public let kvCache: MetalKVCache?
    public let convState: MTLBuffer?
    public let recurrentState: MTLBuffer?
    public let convStateDimension: Int
    public let convStateKernelSize: Int
    public let recurrentStateBytesPerLayer: Int
    public let perLayerInputs: MTLBuffer?
    public let perLayerInputDimension: Int
    public let perLayerInputLayerCount: Int
    public let logits: MTLBuffer
    public let position: MTLBuffer
    public let ropePositionAxes: MTLBuffer
    public let tokenIn: MTLBuffer
    public let tokenOut: MTLBuffer
}
