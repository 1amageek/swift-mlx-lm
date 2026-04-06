import Metal

// MARK: - Prefill Dispatch Plan

/// Execution mode for a prefill step.
public enum PrefillStepMode: Sendable {
    case batch
    case perPosition
    case lastToken
}

/// How a prefill step consumes the runtime sequence length.
public enum PrefillSequenceLengthPolicy: Sendable {
    case none
    case bind(index: Int)
    case bindAndAdjustGridHeight(index: Int)

    public var bindingIndex: Int? {
        switch self {
        case .none:
            return nil
        case .bind(let index), .bindAndAdjustGridHeight(let index):
            return index
        }
    }

    public var adjustsGridHeightToSequenceLength: Bool {
        switch self {
        case .bindAndAdjustGridHeight:
            return true
        case .none, .bind:
            return false
        }
    }
}

/// A single step in the prefill sequence graph.
public struct MetalPrefillStep: @unchecked Sendable {
    public let descriptor: MetalDispatchDescriptor
    public let bindings: MetalBindingTable
    public let mode: PrefillStepMode
    public let sequenceLengthPolicy: PrefillSequenceLengthPolicy
    public let positionBufferIndex: Int?
    public let perPositionStrides: [Int: Int]
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
        mode: PrefillStepMode,
        sequenceLengthPolicy: PrefillSequenceLengthPolicy,
        positionBufferIndex: Int?,
        perPositionStrides: [Int : Int],
        metadata: MetalDispatchStepMetadata = .init()
    ) {
        self.descriptor = MetalDispatchDescriptor(
            pipeline: pipeline,
            gridSize: gridSize,
            threadgroupSize: threadgroupSize,
            threadgroupMemoryLength: threadgroupMemoryLength,
            barrierPolicy: MetalBarrierPolicy(sync))
        self.bindings = MetalBindingTable(
            bufferBindings: bufferBindings,
            bytesBindings: bytesBindings)
        self.mode = mode
        self.sequenceLengthPolicy = sequenceLengthPolicy
        self.positionBufferIndex = positionBufferIndex
        self.perPositionStrides = perPositionStrides
        self.metadata = metadata
    }

    public init(
        descriptor: MetalDispatchDescriptor,
        bindings: MetalBindingTable,
        mode: PrefillStepMode,
        sequenceLengthPolicy: PrefillSequenceLengthPolicy,
        positionBufferIndex: Int?,
        perPositionStrides: [Int: Int],
        metadata: MetalDispatchStepMetadata = .init()
    ) {
        self.descriptor = descriptor
        self.bindings = bindings
        self.mode = mode
        self.sequenceLengthPolicy = sequenceLengthPolicy
        self.positionBufferIndex = positionBufferIndex
        self.perPositionStrides = perPositionStrides
        self.metadata = metadata
    }

    public func bindRuntimeArguments(
        encoder: MTLComputeCommandEncoder,
        sequenceLength: UInt32
    ) {
        if let bindingIndex = sequenceLengthPolicy.bindingIndex {
            var runtimeSequenceLength = sequenceLength
            withUnsafeBytes(of: &runtimeSequenceLength) {
                encoder.setBytes($0.baseAddress!, length: $0.count, index: bindingIndex)
            }
        }
    }

    public func bindStaticArguments(
        encoder: MTLComputeCommandEncoder,
        position: Int? = nil
    ) {
        guard let position else {
            bindings.bind(to: encoder)
            return
        }
        var adjustedOffsets: [Int: Int] = [:]
        adjustedOffsets.reserveCapacity(bindings.buffers.count)
        for binding in bindings.buffers {
            adjustedOffsets[binding.index] = binding.offset + position * (perPositionStrides[binding.index] ?? 0)
        }
        bindings.bind(to: encoder, adjustedBufferOffsets: adjustedOffsets)
    }

    public func resolvedGridSize(sequenceLength: Int) -> MTLSize {
        guard sequenceLengthPolicy.adjustsGridHeightToSequenceLength, gridSize.height > 1 else {
            return gridSize
        }
        return MTLSize(width: gridSize.width, height: sequenceLength, depth: gridSize.depth)
    }
}

/// Sequence-aware execution plan for prefill.
public struct MetalPrefillPlan: @unchecked Sendable {
    public let steps: [MetalPrefillStep]
    public let buffers: PrefillBufferSet
    public let maximumSequenceLength: Int
    public let stepCount: Int
}

/// Prefill buffer set ([maxSeqLen × dim] layout).
public struct PrefillBufferSet: @unchecked Sendable {
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
    public let tokenIDs: MTLBuffer
    public let positions: MTLBuffer
    public let ropePositionAxes: MTLBuffer
    public let tokenOut: MTLBuffer
}
