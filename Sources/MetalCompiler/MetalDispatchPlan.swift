import Metal
import LMIR

// MARK: - Decode Dispatch Plan

/// A single GPU dispatch step in the decode plan.
public struct MetalDispatchStep: @unchecked Sendable {
    public let pipeline: MTLComputePipelineState
    public let gridSize: MTLSize
    public let threadgroupSize: MTLSize
    public let bufferBindings: [(index: Int, buffer: MTLBuffer, offset: Int)]
    public let bytesBindings: [(index: Int, value: [UInt8])]
    public let threadgroupMemoryLength: Int
    public let sync: SynchronizationKind
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

/// A single step in the prefill sequence graph.
public struct MetalPrefillStep: @unchecked Sendable {
    public let pipeline: MTLComputePipelineState
    public let gridSize: MTLSize
    public let threadgroupSize: MTLSize
    public let bufferBindings: [(index: Int, buffer: MTLBuffer, offset: Int)]
    public let bytesBindings: [(index: Int, value: [UInt8])]
    public let threadgroupMemoryLength: Int
    public let sync: SynchronizationKind
    public let mode: PrefillStepMode
    public let sequenceLengthBindingIndex: Int?
    public let positionBufferIndex: Int?
    public let perPositionStrides: [Int: Int]
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
    case batchedProjection(BatchedProjection)
    case batchedFragment(BatchedFragment)
    case structuralCopy(dimension: Int)
    case structuralAdd(dimension: Int)
}
