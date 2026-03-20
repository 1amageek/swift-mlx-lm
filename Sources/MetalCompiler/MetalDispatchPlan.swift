import Metal
import LMIR

public enum MetalArgumentBindingPolicy: Sendable, Equatable {
    case inlineBindings
    case argumentTable
}

public enum MetalConstantBindingPolicy: Sendable, Equatable {
    case inlineBytes
    case residentConstantBuffer
}

public enum MetalBarrierPolicy: Sendable, Equatable {
    case none
    case bufferBarrier

    public init(_ synchronizationKind: SynchronizationKind) {
        switch synchronizationKind {
        case .none:
            self = .none
        case .bufferBarrier:
            self = .bufferBarrier
        }
    }

    public var synchronizationKind: SynchronizationKind {
        switch self {
        case .none:
            return .none
        case .bufferBarrier:
            return .bufferBarrier
        }
    }
}

public struct MetalBufferBinding: @unchecked Sendable {
    public let index: Int
    public let buffer: MTLBuffer
    public let offset: Int
}

public enum MetalArgumentTableEncodingState: @unchecked Sendable {
    case planned
    case prepared(buffer: MTLBuffer, index: Int, offset: Int)
    case encoded(buffer: MTLBuffer, index: Int, offset: Int)

    public var isEncoded: Bool {
        switch self {
        case .planned:
            return false
        case .prepared:
            return false
        case .encoded:
            return true
        }
    }
}

public struct MetalArgumentTableBindings: @unchecked Sendable {
    public let layout: MetalArgumentTableLayout
    public let bindings: [MetalBufferBinding]
    public let encodingState: MetalArgumentTableEncodingState

    public init(
        layout: MetalArgumentTableLayout,
        bindings: [MetalBufferBinding],
        encodingState: MetalArgumentTableEncodingState = .planned
    ) {
        self.layout = layout
        self.bindings = bindings
        self.encodingState = encodingState
    }

    public var hasEncodedArgumentBuffer: Bool {
        encodingState.isEncoded
    }
}

public enum MetalBufferBindingSet: @unchecked Sendable {
    case inline([MetalBufferBinding])
    case argumentTable(MetalArgumentTableBindings)

    public var policy: MetalArgumentBindingPolicy {
        switch self {
        case .inline:
            return .inlineBindings
        case .argumentTable:
            return .argumentTable
        }
    }

    public var bindings: [MetalBufferBinding] {
        switch self {
        case .inline(let bindings):
            return bindings
        case .argumentTable(let table):
            return table.bindings
        }
    }
}

public struct MetalBytesBinding: Sendable {
    public let index: Int
    public let value: [UInt8]
}

public struct MetalConstantBufferBinding: @unchecked Sendable {
    public let index: Int
    public let buffer: MTLBuffer
    public let offset: Int
    public let length: Int
}

public struct MetalResidentConstantBindings: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let bindings: [MetalConstantBufferBinding]
}

public enum MetalConstantBinding: @unchecked Sendable {
    case inline(MetalBytesBinding)
    case buffer(MetalConstantBufferBinding)

    public var index: Int {
        switch self {
        case .inline(let binding):
            return binding.index
        case .buffer(let binding):
            return binding.index
        }
    }
}

public enum MetalConstantBindingSet: @unchecked Sendable {
    case inline([MetalBytesBinding])
    case resident(MetalResidentConstantBindings)
    case mixed([MetalConstantBinding])

    public var policy: MetalConstantBindingPolicy {
        switch self {
        case .inline:
            return .inlineBytes
        case .resident:
            return .residentConstantBuffer
        case .mixed(let bindings):
            if bindings.allSatisfy({
                if case .buffer = $0 { return true }
                return false
            }) {
                return .residentConstantBuffer
            }
            return .inlineBytes
        }
    }

    public var bindings: [MetalConstantBinding] {
        switch self {
        case .inline(let bindings):
            return bindings.map(MetalConstantBinding.inline)
        case .resident(let resident):
            return resident.bindings.map(MetalConstantBinding.buffer)
        case .mixed(let bindings):
            return bindings
        }
    }

    public var inlineBindings: [MetalBytesBinding] {
        bindings.compactMap { binding in
            guard case .inline(let bytes) = binding else { return nil }
            return bytes
        }
    }
}

public struct MetalBindingTable: @unchecked Sendable {
    private static let bufferEncoder = MetalBufferBindingEncoder()
    private static let constantEncoder = MetalConstantBindingEncoder()

    public let bufferBindings: MetalBufferBindingSet
    public let constantBindings: MetalConstantBindingSet
    public var argumentPolicy: MetalArgumentBindingPolicy {
        bufferBindings.policy
    }
    public var buffers: [MetalBufferBinding] {
        bufferBindings.bindings
    }
    public var constantPolicy: MetalConstantBindingPolicy {
        constantBindings.policy
    }
    public var constants: [MetalConstantBinding] {
        constantBindings.bindings
    }

    public init(
        buffers: [MetalBufferBinding] = [],
        constants: [MetalConstantBinding] = [],
        argumentPolicy: MetalArgumentBindingPolicy = .inlineBindings,
        constantPolicy: MetalConstantBindingPolicy = .inlineBytes
    ) {
        switch argumentPolicy {
        case .inlineBindings:
            self.bufferBindings = .inline(buffers)
        case .argumentTable:
            self.bufferBindings = .argumentTable(MetalArgumentTableBindings(
                layout: MetalArgumentTableLayout(id: 0, indices: buffers.map(\.index)),
                bindings: buffers))
        }
        switch constantPolicy {
        case .inlineBytes:
            self.constantBindings = .mixed(constants)
        case .residentConstantBuffer:
            self.constantBindings = .mixed(constants)
        }
    }

    public init(
        buffers: [MetalBufferBinding] = [],
        bufferBindings: MetalBufferBindingSet,
        constantBindings: MetalConstantBindingSet,
        argumentPolicy: MetalArgumentBindingPolicy = .inlineBindings,
        constantPolicy: MetalConstantBindingPolicy? = nil
    ) {
        _ = buffers
        self.bufferBindings = bufferBindings
        self.constantBindings = constantBindings
        _ = argumentPolicy
        _ = constantPolicy
    }

    public init(
        bufferBindings: [(index: Int, buffer: MTLBuffer, offset: Int)],
        bytesBindings: [(index: Int, value: [UInt8])],
        argumentPolicy: MetalArgumentBindingPolicy = .inlineBindings,
        constantPolicy: MetalConstantBindingPolicy = .inlineBytes
    ) {
        let mappedBuffers = bufferBindings.map { MetalBufferBinding(index: $0.index, buffer: $0.buffer, offset: $0.offset) }
        switch argumentPolicy {
        case .inlineBindings:
            self.bufferBindings = .inline(mappedBuffers)
        case .argumentTable:
            self.bufferBindings = .argumentTable(MetalArgumentTableBindings(
                layout: MetalArgumentTableLayout(id: 0, indices: mappedBuffers.map(\.index)),
                bindings: mappedBuffers))
        }
        let inlineBindings = bytesBindings.map { MetalBytesBinding(index: $0.index, value: $0.value) }
        switch constantPolicy {
        case .inlineBytes:
            self.constantBindings = .inline(inlineBindings)
        case .residentConstantBuffer:
            self.constantBindings = .mixed(inlineBindings.map(MetalConstantBinding.inline))
        }
    }

    public func bind(to encoder: MTLComputeCommandEncoder) {
        bind(to: encoder, adjustedBufferOffsets: [:])
    }

    public func bind(
        to encoder: MTLComputeCommandEncoder,
        adjustedBufferOffsets: [Int: Int]
    ) {
        Self.bufferEncoder.bind(
            bufferBindings,
            to: encoder,
            adjustedBufferOffsets: adjustedBufferOffsets)
        Self.constantEncoder.bind(
            constantBindings,
            to: encoder)
    }
}

public struct MetalDispatchDescriptor: @unchecked Sendable {
    public let pipeline: MTLComputePipelineState
    public let gridSize: MTLSize
    public let threadgroupSize: MTLSize
    public let threadgroupMemoryLength: Int
    public let barrierPolicy: MetalBarrierPolicy

    public var sync: SynchronizationKind {
        barrierPolicy.synchronizationKind
    }

    public func encode(on encoder: MTLComputeCommandEncoder, gridSize overrideGridSize: MTLSize? = nil) {
        if barrierPolicy == .bufferBarrier {
            encoder.memoryBarrier(scope: .buffers)
        }
        encoder.setComputePipelineState(pipeline)
        if threadgroupMemoryLength > 0 {
            encoder.setThreadgroupMemoryLength(threadgroupMemoryLength, index: 0)
        }
        encoder.dispatchThreadgroups(overrideGridSize ?? gridSize, threadsPerThreadgroup: threadgroupSize)
    }
}

// MARK: - Decode Dispatch Plan

/// A single GPU dispatch step in the decode plan.
public struct MetalDispatchStep: @unchecked Sendable {
    public let descriptor: MetalDispatchDescriptor
    public let bindings: MetalBindingTable
    public let bufferAccesses: MetalBufferAccesses

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
        bufferAccesses: MetalBufferAccesses? = nil
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
    }

    public init(
        descriptor: MetalDispatchDescriptor,
        bindings: MetalBindingTable,
        bufferAccesses: MetalBufferAccesses? = nil
    ) {
        self.descriptor = descriptor
        self.bindings = bindings
        self.bufferAccesses = bufferAccesses ?? MetalBufferAccesses.conservative(bindings.buffers)
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
    public let convStateDimension: Int
    public let convStateKernelSize: Int
    public let logits: MTLBuffer
    public let position: MTLBuffer
    public let tokenIn: MTLBuffer
    public let tokenOut: MTLBuffer
}

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
        perPositionStrides: [Int : Int]
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
    }

    public init(
        descriptor: MetalDispatchDescriptor,
        bindings: MetalBindingTable,
        mode: PrefillStepMode,
        sequenceLengthPolicy: PrefillSequenceLengthPolicy,
        positionBufferIndex: Int?,
        perPositionStrides: [Int: Int]
    ) {
        self.descriptor = descriptor
        self.bindings = bindings
        self.mode = mode
        self.sequenceLengthPolicy = sequenceLengthPolicy
        self.positionBufferIndex = positionBufferIndex
        self.perPositionStrides = perPositionStrides
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
    public let convStateDimension: Int
    public let convStateKernelSize: Int
    public let logits: MTLBuffer
    public let tokenIDs: MTLBuffer
    public let positions: MTLBuffer
    public let tokenOut: MTLBuffer
}

// MARK: - Dispatch Entry

/// A single kernel dispatch in the plan (IR → fusion → steps).
public struct DispatchEntry: Sendable {
    public let index: Int
    public let kind: DispatchKind
    public let parameterBindings: [ParameterBinding]
    public let layerIndex: Int?

    public init(index: Int, kind: DispatchKind, parameterBindings: [ParameterBinding] = [], layerIndex: Int? = nil) {
        self.index = index
        self.kind = kind
        self.parameterBindings = parameterBindings
        self.layerIndex = layerIndex
    }
}

/// Kind of dispatch: projection, fragment, fused, batched, or structural.
public enum DispatchKind: Sendable {
    case projection(MetalProjection, isOutput: Bool = false)
    case fragment(any PrimitiveMetalKernelFragment)
    case fusedCopyNorm(FusedCopyNorm)
    case fusedResidualAddCopyNorm(FusedResidualAddCopyNorm)
    case fusedResidualAddNorm(FusedResidualAddNorm)
    case fusedSwiGLUProjection(FusedSwiGLUProjection)
    case batchedProjection(BatchedProjection)
    case batchedFragment(BatchedFragment)
    case structuralCopy(dimension: Int)
    case structuralAdd(dimension: Int)
}
