import Foundation
import LMIR
import Metal

enum DecodeProjectionShapeFamily {
    case generic
    case largeDense
    case input2048SquareDense
    case input20486144Dense
    case input20488192Dense
    case input2048ExpandedDense
    case input8192Tiled
    case vocabDense

    static func resolve(outputDimension: Int, inputDimension: Int) -> Self {
        if outputDimension >= 65_536 && inputDimension == 2_048 {
            return .vocabDense
        }
        if inputDimension == 2_048 && outputDimension == 2_048 {
            return .input2048SquareDense
        }
        if inputDimension == 2_048 && outputDimension == 6_144 {
            return .input20486144Dense
        }
        if inputDimension == 2_048 && outputDimension == 8_192 {
            return .input20488192Dense
        }
        if inputDimension == 2_048 && outputDimension > 2_048 && outputDimension < 65_536 {
            return .input2048ExpandedDense
        }
        if inputDimension == 8_192 && outputDimension >= 2_048 && outputDimension < 65_536 {
            return .input8192Tiled
        }
        if outputDimension >= 32_768 && inputDimension >= 2_048 {
            return .largeDense
        }
        return .generic
    }

    var preferredSimdgroups: Int {
        switch self {
        case .generic:
            return 4
        case .vocabDense:
            return 16
        case .input20486144Dense:
            return 4
        case .input2048ExpandedDense, .input20488192Dense:
            return 8
        case .input2048SquareDense:
            return 8
        case .largeDense, .input8192Tiled:
            return 8
        }
    }

    var tileElements: Int {
        switch self {
        case .generic:
            return 256
        case .largeDense:
            return 512
        case .input2048SquareDense, .input20486144Dense, .input20488192Dense, .input2048ExpandedDense, .vocabDense:
            return 2_048
        case .input8192Tiled:
            return 1_024
        }
    }

    var kernelBaseName: String {
        switch self {
        case .generic:
            return "gemv"
        case .largeDense:
            return "gemv_large"
        case .input2048SquareDense:
            return "gemv_2048_sq"
        case .input20486144Dense:
            return "gemv_2048_6144"
        case .input20488192Dense:
            return "gemv_2048_8192"
        case .input2048ExpandedDense:
            return "gemv_2048"
        case .input8192Tiled:
            return "gemv_8192_tiled"
        case .vocabDense:
            return "gemv_vocab"
        }
    }
}

enum FusedSwiGLUProjectionFamily {
    case generic
    case input2048Dense

    static func resolve(inputDimension: Int, outputDimension: Int) -> Self {
        if inputDimension == 2_048 && outputDimension > 2_048 && outputDimension < 65_536 {
            return .input2048Dense
        }
        return .generic
    }

    func kernelBaseName(activation: MetalSourceGenerator.GatedActivation = .silu) -> String {
        let prefix = switch activation {
        case .silu: "fused_swiglu_projection"
        case .geluTanh: "fused_geglu_projection"
        }
        switch self {
        case .generic:
            return prefix
        case .input2048Dense:
            return prefix + "_2048"
        }
    }

    /// Backward-compatible property for callers that don't specify activation.
    var kernelBaseName: String {
        kernelBaseName(activation: .silu)
    }
}

struct CompileContext {
    let graph: ModelGraph
    let hiddenSize: Int
    let intermediateSize: Int
    let vocabSize: Int
    let inferencePolicy: InferencePolicy
    let stafWeightStore: STAFWeightStore?
    let device: MTLDevice
    let weightFormat: WeightFormat
    let decodeBufferPrecision: BufferPrecision
    let accessPolicyResolver: ProjectionWeightAccessPolicyResolver

    var maximumSequenceLength: Int { inferencePolicy.maximumSequenceLength }

    var decodeKernelContext: KernelContext {
        KernelContext(
            bufferPrecision: decodeBufferPrecision,
            weightFormat: weightFormat
        )
    }

    var prefillKernelContext: KernelContext {
        KernelContext(
            bufferPrecision: .float32,
            weightFormat: weightFormat
        )
    }

    var resolvedIntermediateSize: Int {
        max(intermediateSize, hiddenSize * 4)
    }

    var resolvedVocabSize: Int {
        max(vocabSize, 1)
    }
}

struct WeightResolver {
    let entry: DispatchEntry
    let stafWeightStore: STAFWeightStore?
    let fallbackBuffer: MTLBuffer
    let logsMisses: Bool
    let executionPhase: STAFWeightExecutionPhase
    let accessPolicyResolver: ProjectionWeightAccessPolicyResolver

    func resolve(role: String) -> (MTLBuffer, Int) {
        if let binding = entry.parameterBindings.first(where: { $0.role == role }),
           let staf = stafWeightStore,
           let access = staf.resolvedBufferAccess(for: accessPolicyResolver.accessRequest(
            for: entry,
            role: role,
            binding: binding,
            executionPhase: executionPhase,
            stafWeightStore: staf
           )) {
            return (access.buffer, access.offset)
        }

        if logsMisses {
            let bindingName = entry.parameterBindings.first(where: { $0.role == role })?.tensorName ?? "(no binding)"
            print("[Compiler] WEIGHT MISS: role='\(role)' tensorName='\(bindingName)' bindings=\(entry.parameterBindings.map(\.role))")
        }

        return (fallbackBuffer, 0)
    }
}

struct PlanBuildContext {
    let compileContext: CompileContext
    let kernelContext: KernelContext
    let pipelineCache: [String: MTLComputePipelineState]
    let dispatchHeuristics: DispatchHeuristics

    var hiddenSize: Int { compileContext.hiddenSize }
    var stafWeightStore: STAFWeightStore? { compileContext.stafWeightStore }
    var device: MTLDevice { compileContext.device }
}

struct DispatchHeuristics {
    func config(
        for dimension: MetalDispatchDimension,
        pipeline: MTLComputePipelineState,
        roundUp: (Int, Int) -> Int
    ) -> (grid: MTLSize, threadgroup: MTLSize, sharedMemoryBytes: Int) {
        let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
        let simdWidth = pipeline.threadExecutionWidth

        switch dimension {
        case .reduction(let dimension):
            let threads = min(roundUp(min(max(dimension, 1), 1024), simdWidth), maxThreads)
            return (
                MTLSize(width: 1, height: 1, depth: 1),
                MTLSize(width: threads, height: 1, depth: 1),
                0
            )

        case .elementwise(let count):
            let clampedCount = max(count, 1)
            let threadgroupSize = min(roundUp(min(clampedCount, 256), simdWidth), maxThreads)
            let groupCount = (clampedCount + threadgroupSize - 1) / threadgroupSize
            return (
                MTLSize(width: groupCount, height: 1, depth: 1),
                MTLSize(width: threadgroupSize, height: 1, depth: 1),
                0
            )

        case .gemv(let outputDimension, let inputDimension):
            let family = DecodeProjectionShapeFamily.resolve(
                outputDimension: outputDimension,
                inputDimension: inputDimension
            )
            let preferredSimdgroups = family.preferredSimdgroups
            let simdgroupCount = max(1, min(preferredSimdgroups, maxThreads / max(simdWidth, 1)))
            let rowsPerThreadgroup = simdgroupCount
            let threads = min(simdgroupCount * simdWidth, maxThreads)
            let groupCount = (outputDimension + rowsPerThreadgroup - 1) / rowsPerThreadgroup
            return (
                MTLSize(width: groupCount, height: 1, depth: 1),
                MTLSize(width: threads, height: 1, depth: 1),
                0
            )

        case .perHead(let headCount):
            let threads = min(256, maxThreads)
            return (
                MTLSize(width: headCount, height: 1, depth: 1),
                MTLSize(width: threads, height: 1, depth: 1),
                0
            )

        case .gather(let count):
            let clampedCount = max(count, 1)
            let threadgroupSize = min(256, maxThreads)
            let groupCount = (clampedCount + threadgroupSize - 1) / threadgroupSize
            return (
                MTLSize(width: groupCount, height: 1, depth: 1),
                MTLSize(width: threadgroupSize, height: 1, depth: 1),
                0
            )
        }
    }
}

struct ConvStateRequirements {
    let layerCount: Int
    let dimension: Int
    let kernelSize: Int
}

struct RecurrentStateRequirements {
    let layerCount: Int
    let bytesPerLayer: Int
}

struct DecodeBufferAllocation {
    let bufferSet: MetalBufferSet
    let slotDimension: Int
}

struct PrefillBufferAllocation {
    let bufferSet: PrefillBufferSet
    let slotDimension: Int
    let resolvedIntermediateSize: Int
    let resolvedVocabSize: Int
    let maximumSequenceLength: Int
}

struct WalkContext {
    var entries: [DispatchEntry] = []
    var cacheSlots: [CacheSlotInfo] = []
    var nextIndex: Int = 0

    mutating func emit(
        _ kind: DispatchKind,
        parameterBindings: [ParameterBinding] = [],
        layerIndex: Int? = nil
    ) {
        entries.append(DispatchEntry(
            index: nextIndex,
            kind: kind,
            parameterBindings: parameterBindings,
            layerIndex: layerIndex
        ))
        nextIndex += 1
    }

    mutating func emitOptimized(_ entry: OptimizedEntry) {
        switch entry {
        case .single(let primitive):
            if case .gemv(let outputDimension, let inputDimension) = primitive.fragment.dispatchDimension {
                let field = primitive.fragment.weightSlots.first?.field ?? "weight"
                let projection = MetalProjection(
                    field: field,
                    inputDimension: inputDimension,
                    outputDimension: outputDimension
                )
                emit(
                    .projection(projection),
                    parameterBindings: primitive.parameterBindings,
                    layerIndex: primitive.layerIndex
                )
            } else {
                emit(
                    .fragment(primitive.fragment),
                    parameterBindings: primitive.parameterBindings,
                    layerIndex: primitive.layerIndex
                )
            }
        case .batchedProjection(let batched, let bindings, let layer):
            emit(.batchedProjection(batched), parameterBindings: bindings, layerIndex: layer)
        case .fusedSwiGLUProjection(let fused, let bindings, let layer):
            emit(.fusedSwiGLUProjection(fused), parameterBindings: bindings, layerIndex: layer)
        case .batchedFragment(let batched, let bindings, let layer):
            emit(.batchedFragment(batched), parameterBindings: bindings, layerIndex: layer)
        }
    }
}

struct CacheSlotInfo {
    let kvHeadCount: Int
    let headDimension: Int
}

struct BufferRoutingState {
    var currentInputOffset: Int = 0
    var projectionIndex: Int = 0
    var lastOutputIsHidden: Bool = true
    var convLayerIndex: Int = 0
    var recurrentLayerIndex: Int = 0
}
