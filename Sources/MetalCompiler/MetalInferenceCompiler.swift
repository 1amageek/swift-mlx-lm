import Metal
import Foundation
import LMIR

// MARK: - Compiler

/// Compiles a ModelGraph into a MetalDispatchPlan.
///
/// ## Phases
///
/// 1. **IR walk**: traverse the graph, read each MetalComponent's dispatchDeclarations
/// 2. **Fusion pass**: detect adjacent fusable operations via pattern matching
/// 3. **Compile**: build one MTLLibrary from compiler-owned kernel sources
/// 4. **Buffer routing**: assign concrete MTLBuffers and offsets to each dispatch
/// 5. **Dispatch plan**: compute grid/threadgroup, build MetalDispatchStep array
public struct MetalInferenceCompiler: Sendable {
    private static let argumentTableBindingIndex = 30

    /// The optimization strategy used for dispatch entry generation.
    public let optimizer: any DispatchOptimizer

    private enum DecodeProjectionShapeFamily {
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
            case .input2048ExpandedDense, .input20486144Dense, .input20488192Dense:
                return 8
            case .largeDense, .input2048SquareDense, .input8192Tiled:
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

    private enum FusedSwiGLUProjectionFamily {
        case generic
        case input2048Dense

        static func resolve(inputDimension: Int, outputDimension: Int) -> Self {
            if inputDimension == 2_048 && outputDimension > 2_048 && outputDimension < 65_536 {
                return .input2048Dense
            }
            return .generic
        }

        var kernelBaseName: String {
            switch self {
            case .generic:
                return "fused_swiglu_projection"
            case .input2048Dense:
                return "fused_swiglu_projection_2048"
            }
        }

    }

    private static func denseDecodeProjectionFamily(
        outputDimension: Int,
        inputDimension: Int,
        schemeIdentifier: QuantizationSchemeIdentifier
    ) -> DecodeProjectionShapeFamily? {
        guard schemeIdentifier == .fp16RowMajor || schemeIdentifier == .bf16RowMajor else {
            return nil
        }
        return DecodeProjectionShapeFamily.resolve(
            outputDimension: outputDimension,
            inputDimension: inputDimension
        )
    }

    /// Immutable inputs shared across one compile invocation.
    private struct CompileContext {
        let graph: ModelGraph
        let hiddenSize: Int
        let intermediateSize: Int
        let vocabSize: Int
        let maximumSequenceLength: Int
        let stafWeightStore: STAFWeightStore?
        let device: MTLDevice
        let weightFormat: WeightFormat
        let decodeBufferPrecision: BufferPrecision

        var decodeKernelContext: KernelContext {
            KernelContext(
                bufferPrecision: decodeBufferPrecision,
                weightFormat: weightFormat)
        }

        var prefillKernelContext: KernelContext {
            KernelContext(
                bufferPrecision: .float32,
                weightFormat: weightFormat)
        }

        var resolvedIntermediateSize: Int {
            max(intermediateSize, hiddenSize * 4)
        }

        var resolvedVocabSize: Int {
            max(vocabSize, 1)
        }
    }

    private struct WeightResolver {
        let entry: DispatchEntry
        let stafWeightStore: STAFWeightStore?
        let fallbackBuffer: MTLBuffer
        let logsMisses: Bool

        func resolve(role: String) -> (MTLBuffer, Int) {
            if let binding = entry.parameterBindings.first(where: { $0.role == role }),
               let staf = stafWeightStore,
               let access = staf.bufferAccess(for: binding.tensorName) {
                return (access.buffer, access.offset)
            }

            if logsMisses {
                let bindingName = entry.parameterBindings.first(where: { $0.role == role })?.tensorName ?? "(no binding)"
                print("[Compiler] WEIGHT MISS: role='\(role)' tensorName='\(bindingName)' bindings=\(entry.parameterBindings.map(\.role))")
            }

            return (fallbackBuffer, 0)
        }
    }

    /// Immutable inputs shared while lowering optimized entries into a concrete plan.
    private struct PlanBuildContext {
        let compileContext: CompileContext
        let kernelContext: KernelContext
        let pipelineCache: [String: MTLComputePipelineState]
        let dispatchHeuristics: DispatchHeuristics

        var hiddenSize: Int { compileContext.hiddenSize }
        var stafWeightStore: STAFWeightStore? { compileContext.stafWeightStore }
        var device: MTLDevice { compileContext.device }
    }

    private struct DispatchHeuristics {
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

    private struct ConvStateRequirements {
        let layerCount: Int
        let dimension: Int
        let kernelSize: Int
    }

    private struct DecodeBufferAllocation {
        let bufferSet: MetalBufferSet
        let slotDimension: Int
    }

    private struct PrefillBufferAllocation {
        let bufferSet: PrefillBufferSet
        let slotDimension: Int
        let resolvedIntermediateSize: Int
        let resolvedVocabSize: Int
        let maximumSequenceLength: Int
    }

    private struct DispatchEntryDiagnosticsFormatter {
        let kernelContext: KernelContext

        func format(entries: [DispatchEntry], unfusedCount: Int) -> String {
            var lines: [String] = []
            lines.append("Dispatch Entries (\(entries.count) total, \(unfusedCount) unfused):")
            for entry in entries {
                lines.append(format(entry))
            }
            return lines.joined(separator: "\n")
        }

        private func format(_ entry: DispatchEntry) -> String {
            let layer = entry.layerIndex.map { "L\($0)" } ?? "--"
            let kind: String
            switch entry.kind {
            case .projection(let p, let isOut):
                kind = "projection(\(p.field), in=\(p.inputDimension), out=\(p.outputDimension), isOutput=\(isOut))"
            case .fragment(let f):
                kind = "fragment(\(type(of: f)), kernel=\(f.kernelName(context: kernelContext)))"
            case .fusedCopyNorm(let f):
                kind = "fusedCopyNorm(dim=\(f.dimension), eps=\(f.epsilon))"
            case .fusedResidualAddCopyNorm(let f):
                kind = "fusedResidualAddCopyNorm(dim=\(f.dimension), eps=\(f.epsilon))"
            case .fusedResidualAddNorm(let f):
                kind = "fusedResidualAddNorm(dim=\(f.dimension), eps=\(f.epsilon))"
            case .fusedSwiGLUProjection(let f):
                kind = "fusedSwiGLUProjection(gate=\(f.gateField), up=\(f.upField), in=\(f.inputDimension), out=\(f.outputDimension))"
            case .batchedProjection(let b):
                kind = "batchedProjection(\(b.projections.map(\.field).joined(separator: ",")))"
            case .batchedFragment(let b):
                kind = "batchedFragment(\(b.fragments.count)x)"
            case .structuralCopy(let d):
                kind = "structuralCopy(dim=\(d))"
            case .structuralAdd(let d):
                kind = "structuralAdd(dim=\(d))"
            }
            return "  [\(String(format: "%3d", entry.index))] \(layer) \(kind)"
        }
    }

    private struct OptimizationReportBuilder {
        let optimizerName: String

        func makeReport(
            unfusedEntries: [DispatchEntry],
            optimizedEntries: [DispatchEntry]
        ) -> OptimizationReport {
            var patterns: [String: (count: Int, saved: Int)] = [:]
            for entry in optimizedEntries {
                let name: String
                switch entry.kind {
                case .fusedCopyNorm:
                    name = "fusedCopyNorm"
                case .fusedResidualAddCopyNorm:
                    name = "fusedResidualAddCopyNorm"
                case .fusedResidualAddNorm:
                    name = "fusedResidualAddNorm"
                case .fusedSwiGLUProjection:
                    name = "fusedSwiGLUProjection"
                case .batchedProjection(let batched):
                    name = "batchedProjection(\(batched.projections.count)-way)"
                case .batchedFragment(let batch):
                    name = "batchedFragment(\(batch.fragments.count)-way)"
                default:
                    continue
                }
                patterns[name, default: (0, 0)].count += 1
            }

            return OptimizationReport(
                optimizerName: optimizerName,
                unfusedCount: unfusedEntries.count,
                optimizedCount: optimizedEntries.count,
                patterns: patterns.map { .init(name: $0.key, count: $0.value.count, savedDispatches: 0) }
            )
        }
    }

    private struct CompiledPlanDiagnosticsFormatter {
        func formatDecodePlan(_ plan: MetalDispatchPlan) -> String {
            var lines: [String] = []
            let argumentTableSteps = plan.steps.filter { $0.bindings.argumentPolicy == .argumentTable }.count
            let preparedArgumentSteps = plan.steps.filter {
                guard case .argumentTable(let table) = $0.bindings.bufferBindings else { return false }
                if case .prepared = table.encodingState { return true }
                return false
            }.count
            let encodedArgumentSteps = plan.steps.filter {
                guard case .argumentTable(let table) = $0.bindings.bufferBindings else { return false }
                if case .encoded = table.encodingState { return true }
                return false
            }.count
            let residentConstantSteps = plan.steps.filter { $0.bindings.constantPolicy == .residentConstantBuffer }.count
            let layoutUsage = summarizeArgumentTableLayouts(in: plan.steps.map(\.bindings))
            lines.append(
                "Compiled Decode Plan (\(plan.steps.count) steps, argTable=\(argumentTableSteps), argPrepared=\(preparedArgumentSteps), argEncoded=\(encodedArgumentSteps), residentConst=\(residentConstantSteps), argLayouts=\(layoutUsage.count))")
            if !layoutUsage.isEmpty {
                lines.append("  topArgLayouts: \(describe(layoutUsage: layoutUsage, limit: 5))")
            }
            for (index, step) in plan.steps.enumerated() {
                lines.append(formatDecodeStep(step, index: index, buffers: plan.buffers))
            }
            return lines.joined(separator: "\n")
        }

        func formatPrefillPlan(
            _ plan: MetalPrefillPlan,
            maximumSequenceLength: Int
        ) -> String {
            var lines: [String] = []
            let argumentTableSteps = plan.steps.filter { $0.bindings.argumentPolicy == .argumentTable }.count
            let preparedArgumentSteps = plan.steps.filter {
                guard case .argumentTable(let table) = $0.bindings.bufferBindings else { return false }
                if case .prepared = table.encodingState { return true }
                return false
            }.count
            let encodedArgumentSteps = plan.steps.filter {
                guard case .argumentTable(let table) = $0.bindings.bufferBindings else { return false }
                if case .encoded = table.encodingState { return true }
                return false
            }.count
            let residentConstantSteps = plan.steps.filter { $0.bindings.constantPolicy == .residentConstantBuffer }.count
            let layoutUsage = summarizeArgumentTableLayouts(in: plan.steps.map(\.bindings))
            lines.append(
                "Compiled Prefill Plan (\(plan.steps.count) steps, maxSeq=\(maximumSequenceLength), argTable=\(argumentTableSteps), argPrepared=\(preparedArgumentSteps), argEncoded=\(encodedArgumentSteps), residentConst=\(residentConstantSteps), argLayouts=\(layoutUsage.count))")
            if !layoutUsage.isEmpty {
                lines.append("  topArgLayouts: \(describe(layoutUsage: layoutUsage, limit: 5))")
            }
            for (index, step) in plan.steps.enumerated() {
                lines.append(formatPrefillStep(step, index: index, buffers: plan.buffers))
            }
            return lines.joined(separator: "\n")
        }

        private func formatDecodeStep(
            _ step: MetalDispatchStep,
            index: Int,
            buffers: MetalBufferSet
        ) -> String {
            let kernel = step.pipeline.label ?? "(unlabeled)"
            let header = "[\(String(format: "%3d", index))] kernel=\(kernel) grid=\(format(step.gridSize)) tg=\(format(step.threadgroupSize)) sync=\(step.sync) tgmem=\(step.threadgroupMemoryLength) arg=\(describe(step.bindings.bufferBindings)) const=\(describe(step.bindings.constantBindings))"
            let bindings = step.bufferBindings
                .map { "b\($0.index)=\(describe(buffer: $0.buffer, offset: $0.offset, decodeBuffers: buffers))" }
                .joined(separator: ", ")
            let bytes = step.bytesBindings
                .map { "c\($0.index)=\(describe(bytes: $0.value))" }
                .joined(separator: ", ")
            return "\(header)\n    buffers: [\(bindings)]\n    bytes: [\(bytes)]"
        }

        private func formatPrefillStep(
            _ step: MetalPrefillStep,
            index: Int,
            buffers: PrefillBufferSet
        ) -> String {
            let kernel = step.pipeline.label ?? "(unlabeled)"
            let header = "[\(String(format: "%3d", index))] kernel=\(kernel) mode=\(step.mode) seqPolicy=\(step.sequenceLengthPolicy) grid=\(format(step.gridSize)) tg=\(format(step.threadgroupSize)) sync=\(step.sync) tgmem=\(step.threadgroupMemoryLength) arg=\(describe(step.bindings.bufferBindings)) const=\(describe(step.bindings.constantBindings))"
            let bindings = step.bufferBindings
                .map { "b\($0.index)=\(describe(buffer: $0.buffer, offset: $0.offset, prefillBuffers: buffers))" }
                .joined(separator: ", ")
            let bytes = step.bytesBindings
                .map { "c\($0.index)=\(describe(bytes: $0.value))" }
                .joined(separator: ", ")
            return "\(header)\n    buffers: [\(bindings)]\n    bytes: [\(bytes)]"
        }

        private func describe(_ bindings: MetalBufferBindingSet) -> String {
            switch bindings {
            case .inline(let inline):
                return "inline[\(inline.count)]"
            case .argumentTable(let table):
                let state: String
                switch table.encodingState {
                case .planned:
                    state = "planned"
                case .prepared:
                    state = "prepared"
                case .encoded:
                    state = "encoded"
                }
                return "argumentTable#\(table.layout.id)[\(table.bindings.count),\(state)]"
            }
        }

        private func describe(_ bindings: MetalConstantBindingSet) -> String {
            switch bindings {
            case .inline(let inline):
                return "inline[\(inline.count)]"
            case .resident(let resident):
                return "resident[\(resident.bindings.count)]"
            case .mixed(let mixed):
                return "mixed[\(mixed.count)]"
            }
        }

        private func summarizeArgumentTableLayouts(
            in tables: [MetalBindingTable]
        ) -> [MetalArgumentTableLayoutUsage] {
            MetalArgumentBindingAllocator().summarizeUsage(in: tables)
        }

        private func describe(
            layoutUsage: [MetalArgumentTableLayoutUsage],
            limit: Int
        ) -> String {
            layoutUsage.prefix(limit)
                .map { usage in
                    "#\(usage.layout.id)x\(usage.useCount){slots=\(usage.bindingCount),indices=\(usage.layout.indices)}"
                }
                .joined(separator: " | ")
        }

        private func format(_ size: MTLSize) -> String {
            "(\(size.width),\(size.height),\(size.depth))"
        }

        private func describe(
            buffer: MTLBuffer,
            offset: Int,
            decodeBuffers: MetalBufferSet
        ) -> String {
            let name: String
            if buffer === decodeBuffers.hidden {
                name = "hidden"
            } else if buffer === decodeBuffers.residual {
                name = "residual"
            } else if buffer === decodeBuffers.scratch {
                name = "scratch"
            } else if buffer === decodeBuffers.logits {
                name = "logits"
            } else if buffer === decodeBuffers.position {
                name = "position"
            } else if buffer === decodeBuffers.tokenIn {
                name = "tokenIn"
            } else if buffer === decodeBuffers.tokenOut {
                name = "tokenOut"
            } else if let keys = decodeBuffers.kvCache?.keys, buffer === keys {
                name = "kv.keys"
            } else if let values = decodeBuffers.kvCache?.values, buffer === values {
                name = "kv.values"
            } else if let convState = decodeBuffers.convState, buffer === convState {
                name = "convState"
            } else if let weightIndex = decodeBuffers.weights.firstIndex(where: { $0 === buffer }) {
                name = "weights[\(weightIndex)]"
            } else {
                name = "buffer@\(ObjectIdentifier(buffer))"
            }
            return "\(name)+\(offset)"
        }

        private func describe(
            buffer: MTLBuffer,
            offset: Int,
            prefillBuffers: PrefillBufferSet
        ) -> String {
            let name: String
            if buffer === prefillBuffers.hidden {
                name = "hidden"
            } else if buffer === prefillBuffers.residual {
                name = "residual"
            } else if buffer === prefillBuffers.scratch {
                name = "scratch"
            } else if buffer === prefillBuffers.logits {
                name = "logits"
            } else if buffer === prefillBuffers.tokenIDs {
                name = "tokenIDs"
            } else if buffer === prefillBuffers.positions {
                name = "positions"
            } else if buffer === prefillBuffers.tokenOut {
                name = "tokenOut"
            } else if let keys = prefillBuffers.kvCache?.keys, buffer === keys {
                name = "kv.keys"
            } else if let values = prefillBuffers.kvCache?.values, buffer === values {
                name = "kv.values"
            } else if let convState = prefillBuffers.convState, buffer === convState {
                name = "convState"
            } else {
                name = "buffer@\(ObjectIdentifier(buffer))"
            }
            return "\(name)+\(offset)"
        }

        private func describe(bytes value: [UInt8]) -> String {
            switch value.count {
            case MemoryLayout<UInt32>.size:
                let uintValue = value.withUnsafeBytes { $0.load(as: UInt32.self) }
                let floatValue = value.withUnsafeBytes { $0.load(as: Float.self) }
                return "u32=\(uintValue) f32=\(String(format: "%.6g", floatValue))"
            case MemoryLayout<Float>.size:
                let floatValue = value.withUnsafeBytes { $0.load(as: Float.self) }
                return "f32=\(String(format: "%.6g", floatValue))"
            default:
                let hex = value.prefix(16).map { String(format: "%02x", $0) }.joined()
                return "bytes[\(value.count)]=\(hex)"
            }
        }
    }

    private struct KernelWeightFormatResolver {
        let stafWeightStore: STAFWeightStore?

        func resolve(role: String, entry: DispatchEntry) -> MetalSourceGenerator.WeightFormat {
            guard let staf = stafWeightStore,
                  let binding = entry.parameterBindings.first(where: { $0.role == role }),
                  let info = staf.tensor(for: binding.tensorName) else { return .float16 }
            return info.format.schemeIdentifier == .bf16RowMajor ? .bfloat16 : .float16
        }

        func resolve(forFragment fragment: any PrimitiveMetalKernelFragment, entry: DispatchEntry) -> MetalSourceGenerator.WeightFormat {
            let roles = fragment.weightSlots.compactMap(\.field) + ["scale", "embedding_table", "conv_weight"]
            for role in roles {
                let format = resolve(role: role, entry: entry)
                if format == .bfloat16 {
                    return .bfloat16
                }
            }
            return .float16
        }
    }

    private struct KernelSourceBuilder {
        let stafWeightStore: STAFWeightStore?
        let modelWeightFormat: WeightFormat
        let bufferPrecision: MetalSourceGenerator.BufferPrecision
        let kernelNameResolver: (DispatchKind, DispatchEntry, STAFWeightStore?, KernelContext) -> String

        func generateSources(entries: [DispatchEntry]) -> GeneratedKernelSources {
            let weightFormatResolver = KernelWeightFormatResolver(stafWeightStore: stafWeightStore)
            var sources: [String] = [MetalSourceGenerator.commonHeader]
            var generatedNames: Set<String> = []
            var mppGEMMNames: Set<String> = []
            var mppSources: [String] = []
            var mppGEMMWeightFormat: MetalSourceGenerator.WeightFormat = .float16
            var needsFlashAttnHelper = false

            for entry in entries {
                let name: String
                switch entry.kind {
                case .projection(let projection, _):
                    let weightFormat = weightFormatResolver.resolve(role: projection.field, entry: entry)
                    let decodeFamily = bufferPrecision == .float32
                        ? nil
                        : DecodeProjectionShapeFamily.resolve(
                            outputDimension: projection.outputDimension,
                            inputDimension: projection.inputDimension
                        )
                    if let decodeFamily {
                        name = weightFormat == .bfloat16
                            ? decodeFamily.kernelBaseName + "_bf16"
                            : decodeFamily.kernelBaseName
                    } else {
                        name = weightFormat == .bfloat16 ? "gemv_bf16" : "gemv"
                    }
                    let isSequenceKernel = bufferPrecision == .float32
                    let emittedName = isSequenceKernel
                        ? name.replacingOccurrences(of: "gemv", with: "gemm") + "_f32s"
                        : name
                    if generatedNames.insert(emittedName).inserted {
                        if isSequenceKernel {
                            let gemmName = weightFormat == .bfloat16 ? "gemm_bf16_f32s" : "gemm_f32s"
                            mppGEMMNames.insert(gemmName)
                            mppGEMMWeightFormat = weightFormat
                            sources.append(MetalSourceGenerator.generateGEMM(
                                name: gemmName,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat))
                        } else if decodeFamily == .vocabDense {
                            sources.append(MetalSourceGenerator.generateVocabGEMV(
                                name: name,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat))
                            let argumentKernelName = MetalInferenceCompiler.argumentTableVariantKernelName(for: name)
                            if generatedNames.insert(argumentKernelName).inserted {
                                sources.append(MetalSourceGenerator.generateVocabGEMVArgumentTableVariant(
                                    name: argumentKernelName,
                                    argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                    bufferPrecision: bufferPrecision,
                                    weightFormat: weightFormat))
                            }
                        } else if decodeFamily == .input8192Tiled {
                            sources.append(MetalSourceGenerator.generateInput8192TiledGEMV(
                                name: name,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat,
                                tileElements: 1_024,
                                unrollFactor: 4))
                            let argumentKernelName = MetalInferenceCompiler.argumentTableVariantKernelName(for: name)
                            if generatedNames.insert(argumentKernelName).inserted {
                                sources.append(MetalSourceGenerator.generateInput8192TiledGEMVArgumentTableVariant(
                                    name: argumentKernelName,
                                    argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                    bufferPrecision: bufferPrecision,
                                    weightFormat: weightFormat,
                                    tileElements: 1_024,
                                    unrollFactor: 4))
                            }
                        } else if decodeFamily == .input2048SquareDense {
                            let sourcePolicy = Input2048GEMVSourcePolicy.square(weightFormat: weightFormat)
                            sources.append(MetalSourceGenerator.generateInput2048GEMV(
                                name: name,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat,
                                fixedOutputDimension: sourcePolicy.fixedOutputDimension,
                                fixedRowsPerThreadgroup: sourcePolicy.fixedRowsPerThreadgroup,
                                stagesInputAsFloat: sourcePolicy.stagesInputAsFloat,
                                unrollFactor: sourcePolicy.unrollFactor))
                            let argumentKernelName = MetalInferenceCompiler.argumentTableVariantKernelName(for: name)
                            if generatedNames.insert(argumentKernelName).inserted {
                                sources.append(MetalSourceGenerator.generateInput2048GEMVArgumentTableVariant(
                                    name: argumentKernelName,
                                    argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                    bufferPrecision: bufferPrecision,
                                    weightFormat: weightFormat,
                                    fixedOutputDimension: sourcePolicy.fixedOutputDimension,
                                    includesDimensionBindings: false,
                                    fixedRowsPerThreadgroup: sourcePolicy.fixedRowsPerThreadgroup,
                                    stagesInputAsFloat: sourcePolicy.stagesInputAsFloat,
                                    usesPairwiseBF16Read: sourcePolicy.usesPairwiseBF16ArgumentRead,
                                    unrollFactor: sourcePolicy.unrollFactor))
                            }
                        } else if decodeFamily == .input20486144Dense {
                            let sourcePolicy = Input2048GEMVSourcePolicy.expanded6144(weightFormat: weightFormat)
                            sources.append(MetalSourceGenerator.generateInput2048GEMV(
                                name: name,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat,
                                fixedOutputDimension: sourcePolicy.fixedOutputDimension,
                                fixedRowsPerThreadgroup: sourcePolicy.fixedRowsPerThreadgroup,
                                stagesInputAsFloat: sourcePolicy.stagesInputAsFloat,
                                unrollFactor: sourcePolicy.unrollFactor))
                            let argumentKernelName = MetalInferenceCompiler.argumentTableVariantKernelName(for: name)
                            if generatedNames.insert(argumentKernelName).inserted {
                                sources.append(MetalSourceGenerator.generateInput2048GEMVArgumentTableVariant(
                                    name: argumentKernelName,
                                    argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                    bufferPrecision: bufferPrecision,
                                    weightFormat: weightFormat,
                                    fixedOutputDimension: sourcePolicy.fixedOutputDimension,
                                    includesDimensionBindings: false,
                                    fixedRowsPerThreadgroup: sourcePolicy.fixedRowsPerThreadgroup,
                                    stagesInputAsFloat: sourcePolicy.stagesInputAsFloat,
                                    usesPairwiseBF16Read: sourcePolicy.usesPairwiseBF16ArgumentRead,
                                    unrollFactor: sourcePolicy.unrollFactor))
                            }
                        } else if decodeFamily == .input20488192Dense {
                            let sourcePolicy = Input2048GEMVSourcePolicy.expanded8192(weightFormat: weightFormat)
                            sources.append(MetalSourceGenerator.generateInput2048GEMV(
                                name: name,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat,
                                fixedOutputDimension: sourcePolicy.fixedOutputDimension,
                                fixedRowsPerThreadgroup: sourcePolicy.fixedRowsPerThreadgroup,
                                stagesInputAsFloat: sourcePolicy.stagesInputAsFloat,
                                unrollFactor: sourcePolicy.unrollFactor))
                        } else if decodeFamily == .input2048ExpandedDense {
                            sources.append(MetalSourceGenerator.generateInput2048GEMV(
                                name: name,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat,
                                unrollFactor: 4))
                        } else {
                            sources.append(MetalSourceGenerator.generateGEMV(
                                name: name,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat,
                                tileElements: decodeFamily?.tileElements ?? 256))
                            let argumentKernelName = MetalInferenceCompiler.argumentTableVariantKernelName(for: name)
                            if generatedNames.insert(argumentKernelName).inserted {
                                sources.append(MetalSourceGenerator.generateGEMVArgumentTableVariant(
                                    name: argumentKernelName,
                                    argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                    bufferPrecision: bufferPrecision,
                                    weightFormat: weightFormat,
                                    tileElements: decodeFamily?.tileElements ?? 256))
                            }
                        }
                    }

                case .fragment(let fragment):
                    let weightFormat = weightFormatResolver.resolve(forFragment: fragment, entry: entry)
                    let fragmentContext = KernelContext(bufferPrecision: bufferPrecision, weightFormat: weightFormat)
                    let kernelName = fragment.kernelName(context: fragmentContext)
                    if generatedNames.insert(kernelName).inserted {
                        let source = fragment.kernelSource(
                            name: kernelName,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat)
                        if fragment.cacheSlots.contains(where: { $0.kind == .kv }) {
                            needsFlashAttnHelper = true
                        }
                        sources.append(source)
                        if bufferPrecision != .float32 {
                            let argumentKernelName = MetalInferenceCompiler.argumentTableVariantKernelName(for: kernelName)
                            if argumentKernelName != kernelName, generatedNames.insert(argumentKernelName).inserted {
                                switch kernelName {
                                case "embedding_lookup", "embedding_lookup_bf16":
                                    sources.append(MetalSourceGenerator.generateEmbeddingLookupArgumentTableVariant(
                                        name: argumentKernelName,
                                        argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                        bufferPrecision: bufferPrecision,
                                        weightFormat: weightFormat))
                                case "argmax":
                                    sources.append(MetalSourceGenerator.generateArgmaxArgumentTableVariant(
                                        name: argumentKernelName,
                                        argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                        bufferPrecision: bufferPrecision))
                                case "rms_norm", "rms_norm_bf16":
                                    sources.append(MetalSourceGenerator.generateReductionArgumentTableVariant(
                                        name: argumentKernelName,
                                        argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                        bufferPrecision: bufferPrecision,
                                        weightFormat: weightFormat))
                                case "qk_rms_norm", "qk_rms_norm_bf16":
                                    sources.append(MetalSourceGenerator.generateQKNormArgumentTableVariant(
                                        name: argumentKernelName,
                                        argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                        bufferPrecision: bufferPrecision,
                                        weightFormat: weightFormat))
                                case "rope":
                                    sources.append(MetalSourceGenerator.generateRoPEArgumentTableVariant(
                                        name: argumentKernelName,
                                        argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                        bufferPrecision: bufferPrecision))
                                case "flash_attn_decode":
                                    sources.append(MetalSourceGenerator.generateFlashAttentionArgumentTableVariant(
                                        name: argumentKernelName,
                                        argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                        bufferPrecision: bufferPrecision))
                                case "conv_state_update_bf16", "conv_state_update":
                                    sources.append(MetalSourceGenerator.generateConvStateUpdateArgumentTableVariant(
                                        name: argumentKernelName,
                                        argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                        bufferPrecision: bufferPrecision,
                                        weightFormat: weightFormat))
                                default:
                                    break
                                }
                            }
                        }
                    }
                    if fragment.cacheSlots.contains(where: { $0.kind == .conv }) && bufferPrecision == .float32 {
                        let extractName = "extract_conv_state_f32"
                        if generatedNames.insert(extractName).inserted {
                            sources.append(MetalSourceGenerator.generateExtractConvState(
                                name: extractName,
                                bufferPrecision: bufferPrecision))
                        }
                    }
                    if fragment.cacheSlots.contains(where: { $0.kind == .kv }) && bufferPrecision == .float32 {
                        for helperName in ["kv_cache_fill_seq_f32", "flash_attn_batch_f32"] {
                            if generatedNames.insert(helperName).inserted {
                                if helperName.contains("kv_cache_fill") {
                                    sources.append(MetalSourceGenerator.generateKVCacheFillSeq(
                                        name: helperName,
                                        bufferPrecision: bufferPrecision))
                                } else {
                                    sources.append(MetalSourceGenerator.generateBatchFlashAttention(
                                        name: helperName,
                                        bufferPrecision: bufferPrecision))
                                }
                            }
                        }
                    }

                case .fusedCopyNorm:
                    let weightFormat = weightFormatResolver.resolve(role: "scale", entry: entry)
                    if bufferPrecision != .float32 {
                        let kernelName = weightFormat == .bfloat16 ? "fused_copy_rms_norm_bf16" : "fused_copy_rms_norm"
                        if generatedNames.insert(kernelName).inserted {
                            sources.append(MetalSourceGenerator.generateFusedCopyRMSNorm(
                                name: kernelName,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat))
                            let argumentKernelName = MetalInferenceCompiler.argumentTableVariantKernelName(for: kernelName)
                            if generatedNames.insert(argumentKernelName).inserted {
                                sources.append(MetalSourceGenerator.generateFusedCopyRMSNormArgumentTableVariant(
                                    name: argumentKernelName,
                                    argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                    bufferPrecision: bufferPrecision,
                                    weightFormat: weightFormat))
                            }
                        }
                    } else {
                        let copyName = "copy_buffer_seq_f32"
                        let normName = weightFormat == .bfloat16 ? "rms_norm_seq_bf16_f32_inplace" : "rms_norm_seq_f32_inplace"
                        if generatedNames.insert(copyName).inserted {
                            sources.append(MetalSourceGenerator.generateCopy(name: copyName, bufferPrecision: bufferPrecision))
                        }
                        if generatedNames.insert(normName).inserted {
                            sources.append(MetalSourceGenerator.generateReduction(
                                name: normName,
                                dimension: 0,
                                epsilon: 0,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat))
                        }
                    }

                case .fusedResidualAddCopyNorm:
                    let weightFormat = weightFormatResolver.resolve(role: "scale", entry: entry)
                    if bufferPrecision != .float32 {
                        let kernelName = weightFormat == .bfloat16 ? "fused_residual_add_copy_rms_norm_bf16" : "fused_residual_add_copy_rms_norm"
                        if generatedNames.insert(kernelName).inserted {
                            sources.append(MetalSourceGenerator.generateFusedResidualAddCopyRMSNorm(
                                name: kernelName,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat))
                            let argumentKernelName = MetalInferenceCompiler.argumentTableVariantKernelName(for: kernelName)
                            if generatedNames.insert(argumentKernelName).inserted {
                                sources.append(MetalSourceGenerator.generateFusedResidualAddCopyRMSNormArgumentTableVariant(
                                    name: argumentKernelName,
                                    argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                    bufferPrecision: bufferPrecision,
                                    weightFormat: weightFormat))
                            }
                        }
                    } else {
                        let addName = "residual_add_seq_f32"
                        let copyName = "copy_buffer_seq_f32"
                        let normName = weightFormat == .bfloat16 ? "rms_norm_seq_bf16_f32_inplace" : "rms_norm_seq_f32_inplace"
                        if generatedNames.insert(addName).inserted {
                            sources.append(MetalSourceGenerator.generateResidualAdd(name: addName, bufferPrecision: bufferPrecision))
                        }
                        if generatedNames.insert(copyName).inserted {
                            sources.append(MetalSourceGenerator.generateCopy(name: copyName, bufferPrecision: bufferPrecision))
                        }
                        if generatedNames.insert(normName).inserted {
                            sources.append(MetalSourceGenerator.generateReduction(
                                name: normName,
                                dimension: 0,
                                epsilon: 0,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat))
                        }
                    }

                case .structuralCopy:
                    let kernelName = bufferPrecision == .float32 ? "copy_buffer_seq_f32" : "copy_buffer"
                    if generatedNames.insert(kernelName).inserted {
                        sources.append(MetalSourceGenerator.generateCopy(
                            name: kernelName,
                            bufferPrecision: bufferPrecision,
                            isSequence: bufferPrecision == .float32))
                    }

                case .structuralAdd:
                    let kernelName = bufferPrecision == .float32 ? "residual_add_seq_f32" : "residual_add"
                    if generatedNames.insert(kernelName).inserted {
                        sources.append(MetalSourceGenerator.generateResidualAdd(
                            name: kernelName,
                            bufferPrecision: bufferPrecision,
                            isSequence: bufferPrecision == .float32))
                        if bufferPrecision != .float32 {
                            let argumentKernelName = MetalInferenceCompiler.argumentTableVariantKernelName(for: kernelName)
                            if generatedNames.insert(argumentKernelName).inserted {
                                sources.append(MetalSourceGenerator.generateResidualAddArgumentTableVariant(
                                    name: argumentKernelName,
                                    argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                    bufferPrecision: bufferPrecision))
                            }
                        }
                    }

                case .fusedResidualAddNorm:
                    let weightFormat = weightFormatResolver.resolve(role: "scale", entry: entry)
                    let kernelName = weightFormat == .bfloat16 ? "fused_residual_add_rms_norm_bf16" : "fused_residual_add_rms_norm"
                    if bufferPrecision != .float32, generatedNames.insert(kernelName).inserted {
                        sources.append(MetalSourceGenerator.generateFusedResidualAddRMSNorm(
                            name: kernelName,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat))
                    }

                case .fusedSwiGLUProjection(let fused):
                    let weightFormat = weightFormatResolver.resolve(role: fused.gateField, entry: entry)
                    if bufferPrecision != .float32 {
                        let family = FusedSwiGLUProjectionFamily.resolve(
                            inputDimension: fused.inputDimension,
                            outputDimension: fused.outputDimension)
                        let kernelName = weightFormat == .bfloat16
                            ? family.kernelBaseName + "_bf16"
                            : family.kernelBaseName
                        if generatedNames.insert(kernelName).inserted {
                            if family == .input2048Dense {
                                sources.append(MetalSourceGenerator.generateInput2048FusedSwiGLUProjection(
                                    name: kernelName,
                                    bufferPrecision: bufferPrecision,
                                    weightFormat: weightFormat,
                                    unrollFactor: 8))
                                let argumentKernelName = MetalInferenceCompiler.argumentTableVariantKernelName(for: kernelName)
                                if generatedNames.insert(argumentKernelName).inserted {
                                    sources.append(MetalSourceGenerator.generateInput2048FusedSwiGLUProjectionArgumentTableVariant(
                                        name: argumentKernelName,
                                        argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                        bufferPrecision: bufferPrecision,
                                        weightFormat: weightFormat,
                                        stagesInputAsFloat: false,
                                        unrollFactor: 8))
                                }
                            } else {
                                sources.append(MetalSourceGenerator.generateFusedSwiGLUProjection(
                                    name: kernelName,
                                    bufferPrecision: bufferPrecision,
                                    weightFormat: weightFormat))
                            }
                        }
                    } else {
                        let gemmName = weightFormat == .bfloat16 ? "gemm_bf16_f32s" : "gemm_f32s"
                        if generatedNames.insert(gemmName).inserted {
                            sources.append(MetalSourceGenerator.generateGEMM(
                                name: gemmName,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat))
                        }
                        if generatedNames.insert("swiglu_seq_f32").inserted {
                            sources.append(MetalSourceGenerator.generateSwiGLU(
                                name: "swiglu_seq_f32",
                                bufferPrecision: bufferPrecision))
                        }
                    }

                case .batchedProjection(let batched):
                    let count = batched.projections.count
                    let weightFormat = weightFormatResolver.resolve(role: batched.projections[0].field, entry: entry)
                    if bufferPrecision != .float32 {
                        let suffix = weightFormat == .bfloat16 ? "_bf16" : ""
                        let kernelName = "batched_gemv\(count)\(suffix)"
                        if generatedNames.insert(kernelName).inserted {
                            if count == 2 {
                                sources.append(MetalSourceGenerator.generateBatchedGEMV2(
                                    name: kernelName,
                                    bufferPrecision: bufferPrecision,
                                    weightFormat: weightFormat))
                            } else {
                                sources.append(MetalSourceGenerator.generateBatchedGEMV3(
                                    name: kernelName,
                                    bufferPrecision: bufferPrecision,
                                    weightFormat: weightFormat))
                            }
                        }
                    } else {
                        let gemmName = weightFormat == .bfloat16 ? "gemm_bf16_f32s" : "gemm_f32s"
                        if generatedNames.insert(gemmName).inserted {
                            sources.append(MetalSourceGenerator.generateGEMM(
                                name: gemmName,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat))
                        }
                    }

                case .batchedFragment(let batch):
                    if bufferPrecision != .float32 {
                        let kernelContext = KernelContext(
                            bufferPrecision: bufferPrecision,
                            weightFormat: modelWeightFormat)
                        let kernelName = kernelNameResolver(entry.kind, entry, stafWeightStore, kernelContext)
                        if generatedNames.insert(kernelName).inserted {
                            let weightFormat = weightFormatResolver.resolve(role: "q_layernorm", entry: entry)
                            if batch.fragments.count == 2, case .perHead = batch.dispatchDimension {
                                sources.append(MetalSourceGenerator.generateBatchedPerHead2(
                                    name: kernelName,
                                    bufferPrecision: bufferPrecision,
                                    weightFormat: weightFormat))
                            }
                        }
                    } else {
                        let weightFormat = weightFormatResolver.resolve(role: "q_layernorm", entry: entry)
                        let normName = "qk_rms_norm_seq_f32"
                        if generatedNames.insert(normName).inserted {
                            sources.append(MetalSourceGenerator.generateQKNormSeq(
                                name: normName,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat))
                        }
                    }
                }
            }

            if needsFlashAttnHelper {
                sources.insert(MetalSourceGenerator.flashAttentionHelperSource, at: 1)
            }

            for name in mppGEMMNames.sorted() {
                mppSources.append(MetalSourceGenerator.generateMPPGEMM(
                    name: name,
                    bufferPrecision: bufferPrecision,
                    weightFormat: mppGEMMWeightFormat))
            }

            return GeneratedKernelSources(
                baseSource: sources.joined(separator: "\n\n"),
                mppSources: mppSources,
                mppKernelNames: mppGEMMNames)
        }

        func format(_ generated: GeneratedKernelSources) -> String {
            var lines: [String] = []
            lines.append("=== BASE LIBRARY ===")
            lines.append(generated.baseSource)
            if !generated.mppSources.isEmpty {
                lines.append("=== MPP LIBRARY ===")
                lines.append(generated.mppSources.joined(separator: "\n\n"))
            }
            return lines.joined(separator: "\n\n")
        }
    }


    public init(optimizer: (any DispatchOptimizer)? = nil) {
        self.optimizer = optimizer ?? StandardOptimizer()
    }

    private enum SharedPipelineCache {
        private static let lock = NSLock()
        nonisolated(unsafe) private static var pipelines: [String: MTLComputePipelineState] = [:]

        static func pipeline(named name: String) -> MTLComputePipelineState? {
            lock.lock()
            defer { lock.unlock() }
            return pipelines[name]
        }

        static func store(_ pipeline: MTLComputePipelineState, named name: String) {
            lock.lock()
            pipelines[name] = pipeline
            lock.unlock()
        }
    }

    private struct PipelineLibraryBuilder {
        let device: MTLDevice

        func compile(_ generated: GeneratedKernelSources) throws -> (pipelines: [String: MTLComputePipelineState], usesMPP: Bool) {
            let baseLibrary = try makeLibrary(source: generated.baseSource, options: baseCompileOptions())
            var pipelineCache = try makeBasePipelineCache(
                from: baseLibrary,
                mppKernelNames: generated.mppKernelNames)

            guard !generated.mppSources.isEmpty else {
                return (pipelineCache, false)
            }

            do {
                let mppLibrary = try makeLibrary(
                    source: generated.mppSources.joined(separator: "\n\n"),
                    options: mppCompileOptions())
                try mergeMPPipelines(from: mppLibrary, into: &pipelineCache)
                return (pipelineCache, true)
            } catch {
                return (pipelineCache, false)
            }
        }

        private func makeLibrary(source: String, options: MTLCompileOptions) throws -> MTLLibrary {
            try device.makeLibrary(source: source, options: options)
        }

        private func baseCompileOptions() -> MTLCompileOptions {
            let options = MTLCompileOptions()
            options.mathMode = .safe
            options.languageVersion = .version4_0
            return options
        }

        private func mppCompileOptions() -> MTLCompileOptions {
            let options = MTLCompileOptions()
            options.languageVersion = .version4_0
            return options
        }

        private func makeBasePipelineCache(
            from library: MTLLibrary,
            mppKernelNames: Set<String>
        ) throws -> [String: MTLComputePipelineState] {
            var pipelineCache: [String: MTLComputePipelineState] = [:]
            for name in library.functionNames {
                if mppKernelNames.contains(name),
                   let cachedMPP = SharedPipelineCache.pipeline(named: "mpp::\(name)") {
                    pipelineCache[name] = cachedMPP
                    continue
                }
                if let cached = SharedPipelineCache.pipeline(named: name) {
                    pipelineCache[name] = cached
                    continue
                }
                guard let function = library.makeFunction(name: name) else {
                    continue
                }
                let pipeline = try makePipeline(function: function, label: name)
                pipelineCache[name] = pipeline
                if !mppKernelNames.contains(name) {
                    SharedPipelineCache.store(pipeline, named: name)
                }
            }
            return pipelineCache
        }

        private func mergeMPPipelines(
            from library: MTLLibrary,
            into pipelineCache: inout [String: MTLComputePipelineState]
        ) throws {
            for name in library.functionNames {
                let cacheKey = "mpp::\(name)"
                if let cached = SharedPipelineCache.pipeline(named: cacheKey) {
                    pipelineCache[name] = cached
                    continue
                }
                guard let function = library.makeFunction(name: name) else {
                    continue
                }
                let pipeline = try makePipeline(function: function, label: name)
                pipelineCache[name] = pipeline
                SharedPipelineCache.store(pipeline, named: cacheKey)
            }
        }

        private func makePipeline(
            function: MTLFunction,
            label: String
        ) throws -> MTLComputePipelineState {
            let descriptor = MTLComputePipelineDescriptor()
            descriptor.computeFunction = function
            descriptor.label = label
            return try device.makeComputePipelineState(
                descriptor: descriptor,
                options: [],
                reflection: nil)
        }
    }

    private func makeCompileContext(
        graph: ModelGraph,
        hiddenSize: Int,
        intermediateSize: Int,
        vocabSize: Int,
        maximumSequenceLength: Int = 4096,
        stafWeightStore: STAFWeightStore?,
        device: MTLDevice
    ) -> CompileContext {
        let weightFormat = resolveModelWeightFormat(stafWeightStore)
        return CompileContext(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            maximumSequenceLength: maximumSequenceLength,
            stafWeightStore: stafWeightStore,
            device: device,
            weightFormat: weightFormat,
            decodeBufferPrecision: preferredDecodeBufferPrecision(for: weightFormat))
    }

    private func optimizedEntries(
        using context: CompileContext,
        kernelContext: KernelContext
    ) -> (walkContext: WalkContext, unfusedCount: Int, fusedEntries: [DispatchEntry]) {
        var walkContext = WalkContext()
        walkRegion(
            context.graph.rootRegion,
            pathComponents: [],
            layerIndex: nil,
            hiddenSize: context.hiddenSize,
            context: &walkContext,
            kernelContext: kernelContext
        )
        let unfusedCount = walkContext.entries.count
        let fusedEntries = optimizer.optimizeGraph(walkContext.entries)
        return (walkContext, unfusedCount, fusedEntries)
    }

    private func makePlanBuildContext(
        compileContext: CompileContext,
        kernelContext: KernelContext,
        pipelineCache: [String: MTLComputePipelineState]
    ) -> PlanBuildContext {
        PlanBuildContext(
            compileContext: compileContext,
            kernelContext: kernelContext,
            pipelineCache: pipelineCache,
            dispatchHeuristics: DispatchHeuristics())
    }

    private func resolvedPipeline(
        for entry: DispatchEntry,
        using context: PlanBuildContext
    ) throws -> (name: String, pipeline: MTLComputePipelineState) {
        let resolvedKernelName = kernelName(
            for: entry.kind,
            entry: entry,
            stafWeightStore: context.stafWeightStore,
            kernelContext: context.kernelContext)
        guard let pipeline = context.pipelineCache[resolvedKernelName] else {
            throw MetalCompilerError.kernelNotFound(resolvedKernelName)
        }
        return (resolvedKernelName, pipeline)
    }

    private func resolvedDispatch(
        for entry: DispatchEntry,
        using context: PlanBuildContext
    ) throws -> (
        name: String,
        pipeline: MTLComputePipelineState,
        config: (grid: MTLSize, threadgroup: MTLSize, sharedMemoryBytes: Int)
    ) {
        let resolved = try resolvedPipeline(for: entry, using: context)
        let dimension = dispatchDimension(for: entry.kind, hiddenSize: context.hiddenSize)
        let config = context.dispatchHeuristics.config(
            for: dimension,
            pipeline: resolved.pipeline,
            roundUp: roundUp(_:to:))
        return (
            resolved.name,
            resolved.pipeline,
            config
        )
    }

    private func maximumScratchProjectionDimension(in entries: [DispatchEntry]) -> Int {
        var maximumOutputDimension = 0
        for entry in entries {
            if case .projection(let projection, let isOutput) = entry.kind, !isOutput {
                maximumOutputDimension = max(maximumOutputDimension, projection.outputDimension)
            }
        }
        return maximumOutputDimension
    }

    private func convStateRequirements(in entries: [DispatchEntry]) -> ConvStateRequirements {
        var layerCount = 0
        var dimension = 0
        var kernelSize = 0
        for entry in entries {
            if case .fragment(let fragment) = entry.kind,
               let convSlot = fragment.cacheSlots.first(where: { $0.kind == .conv }),
               case .elementwise(let fragmentDimension) = fragment.dispatchDimension {
                layerCount += 1
                dimension = max(dimension, fragmentDimension)
                kernelSize = max(kernelSize, convSlot.temporalSize)
            }
        }
        return ConvStateRequirements(
            layerCount: layerCount,
            dimension: dimension,
            kernelSize: kernelSize)
    }

    private func makeDecodeBufferAllocation(
        compileContext context: CompileContext,
        walkContext: WalkContext,
        fusedEntries: [DispatchEntry]
    ) throws -> DecodeBufferAllocation {
        let elementSize = context.decodeBufferPrecision.byteSize
        let resolvedIntermediateSize = context.resolvedIntermediateSize
        let resolvedVocabSize = context.resolvedVocabSize
        let slotDimension = max(
            context.hiddenSize,
            resolvedIntermediateSize,
            maximumScratchProjectionDimension(in: fusedEntries))
        let scratchElementCount = max(slotDimension * 4, resolvedIntermediateSize * 4)

        let gpuOnlyOptions: MTLResourceOptions = [.storageModePrivate, .hazardTrackingModeUntracked]
        let cpuAccessOptions: MTLResourceOptions = [.storageModeShared]

        let hiddenBuffer = context.device.makeBuffer(length: context.hiddenSize * elementSize, options: gpuOnlyOptions)!
        let residualBuffer = context.device.makeBuffer(length: context.hiddenSize * elementSize, options: gpuOnlyOptions)!
        let scratchBuffer = context.device.makeBuffer(length: scratchElementCount * elementSize, options: gpuOnlyOptions)!
        let logitsBuffer = context.device.makeBuffer(length: resolvedVocabSize * elementSize, options: gpuOnlyOptions)!
        let positionBuffer = context.device.makeBuffer(length: 4, options: cpuAccessOptions)!
        let tokenInputBuffer = context.device.makeBuffer(length: 4, options: cpuAccessOptions)!
        let tokenOutputBuffer = context.device.makeBuffer(length: 4, options: cpuAccessOptions)!

        let kvCache: MetalKVCache?
        if let firstSlot = walkContext.cacheSlots.first {
            let kvCacheScheme = preferredKVCacheScheme(for: context.weightFormat)
            kvCache = try MetalKVCache(
                device: context.device,
                specification: KVCacheSpecification(
                    keyQuantizationScheme: kvCacheScheme,
                    valueQuantizationScheme: kvCacheScheme,
                    layerCount: walkContext.cacheSlots.count,
                    kvHeadCount: firstSlot.kvHeadCount,
                    headDimension: firstSlot.headDimension),
                resourceOptions: gpuOnlyOptions)
        } else {
            kvCache = nil
        }

        let convState = convStateRequirements(in: fusedEntries)
        let convStateBuffer: MTLBuffer?
        if convState.layerCount > 0 {
            let byteCount = convState.layerCount * convState.kernelSize * convState.dimension * elementSize
            convStateBuffer = context.device.makeBuffer(length: byteCount, options: gpuOnlyOptions)
        } else {
            convStateBuffer = nil
        }

        let weightBuffers = context.stafWeightStore.map { [$0.buffer] } ?? []
        let bufferSet = MetalBufferSet(
            bufferPrecision: context.decodeBufferPrecision,
            hidden: hiddenBuffer,
            residual: residualBuffer,
            scratch: scratchBuffer,
            weights: weightBuffers,
            kvCache: kvCache,
            convState: convStateBuffer,
            convStateDimension: convState.dimension,
            convStateKernelSize: convState.kernelSize,
            logits: logitsBuffer,
            position: positionBuffer,
            tokenIn: tokenInputBuffer,
            tokenOut: tokenOutputBuffer
        )
        return DecodeBufferAllocation(bufferSet: bufferSet, slotDimension: slotDimension)
    }

    private func makePrefillBufferAllocation(
        compileContext context: CompileContext,
        walkContext: WalkContext,
        fusedEntries: [DispatchEntry],
        sharedKVCache: MetalKVCache?,
        sharedConvState: MTLBuffer?,
        sharedConvStateDimension: Int,
        sharedConvStateKernelSize: Int
    ) throws -> PrefillBufferAllocation {
        let elementSize = MemoryLayout<Float16>.size
        let f32ElementSize = MemoryLayout<Float32>.size
        let resolvedIntermediateSize = context.resolvedIntermediateSize
        let resolvedVocabSize = context.resolvedVocabSize
        let maximumSequenceLength = context.maximumSequenceLength
        let slotDimension = max(
            context.hiddenSize,
            resolvedIntermediateSize,
            maximumScratchProjectionDimension(in: fusedEntries))
        let scratchElementCount = max(slotDimension * 4, resolvedIntermediateSize * 4)
        let gpuOptions: MTLResourceOptions = [.storageModeShared]

        let convStateRequirements = convStateRequirements(in: fusedEntries)
        let prefillConvStateBuffer: MTLBuffer?
        let resolvedConvDimension: Int
        let resolvedConvKernelSize: Int
        if let sharedConvState {
            prefillConvStateBuffer = sharedConvState
            resolvedConvDimension = sharedConvStateDimension
            resolvedConvKernelSize = sharedConvStateKernelSize
        } else if convStateRequirements.layerCount > 0 {
            let byteCount = convStateRequirements.layerCount
                * convStateRequirements.kernelSize
                * convStateRequirements.dimension
                * elementSize
            prefillConvStateBuffer = context.device.makeBuffer(length: byteCount, options: gpuOptions)
            if let prefillConvStateBuffer {
                memset(prefillConvStateBuffer.contents(), 0, prefillConvStateBuffer.length)
            }
            resolvedConvDimension = convStateRequirements.dimension
            resolvedConvKernelSize = convStateRequirements.kernelSize
        } else {
            prefillConvStateBuffer = nil
            resolvedConvDimension = 0
            resolvedConvKernelSize = 0
        }

        let prefillKVCache: MetalKVCache?
        if let sharedKVCache {
            prefillKVCache = sharedKVCache
        } else if let firstSlot = walkContext.cacheSlots.first {
            let kvCacheScheme = preferredKVCacheScheme(for: context.weightFormat)
            prefillKVCache = try MetalKVCache(
                device: context.device,
                specification: KVCacheSpecification(
                    keyQuantizationScheme: kvCacheScheme,
                    valueQuantizationScheme: kvCacheScheme,
                    layerCount: walkContext.cacheSlots.count,
                    kvHeadCount: firstSlot.kvHeadCount,
                    headDimension: firstSlot.headDimension),
                resourceOptions: gpuOptions)
        } else {
            prefillKVCache = nil
        }

        let bufferSet = PrefillBufferSet(
            bufferPrecision: .float32,
            hidden: context.device.makeBuffer(length: maximumSequenceLength * context.hiddenSize * f32ElementSize, options: gpuOptions)!,
            residual: context.device.makeBuffer(length: maximumSequenceLength * context.hiddenSize * f32ElementSize, options: gpuOptions)!,
            scratch: context.device.makeBuffer(length: maximumSequenceLength * scratchElementCount * f32ElementSize, options: gpuOptions)!,
            weights: context.stafWeightStore.map { [$0.buffer] } ?? [],
            kvCache: prefillKVCache,
            convState: prefillConvStateBuffer,
            convStateDimension: resolvedConvDimension,
            convStateKernelSize: resolvedConvKernelSize,
            logits: context.device.makeBuffer(length: resolvedVocabSize * f32ElementSize, options: gpuOptions)!,
            tokenIDs: context.device.makeBuffer(length: maximumSequenceLength * 4, options: [.storageModeShared])!,
            positions: context.device.makeBuffer(length: maximumSequenceLength * 4, options: [.storageModeShared])!,
            tokenOut: context.device.makeBuffer(length: 4, options: [.storageModeShared])!
        )

        return PrefillBufferAllocation(
            bufferSet: bufferSet,
            slotDimension: slotDimension,
            resolvedIntermediateSize: resolvedIntermediateSize,
            resolvedVocabSize: resolvedVocabSize,
            maximumSequenceLength: maximumSequenceLength)
    }

    /// Dump optimized dispatch entries for diagnostic purposes.
    /// Returns a human-readable list of all dispatch entries after optimization.
    public func dumpDispatchEntries(
        graph: ModelGraph,
        hiddenSize: Int,
        stafWeightStore: STAFWeightStore? = nil
    ) -> String {
        let context = makeCompileContext(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: 0,
            vocabSize: 0,
            stafWeightStore: stafWeightStore,
            device: MTLCreateSystemDefaultDevice()!)
        let optimization = optimizedEntries(using: context, kernelContext: context.decodeKernelContext)
        let formatter = DispatchEntryDiagnosticsFormatter(kernelContext: context.decodeKernelContext)
        return formatter.format(entries: optimization.fusedEntries, unfusedCount: optimization.unfusedCount)
    }

    /// Analyze optimization without Metal compilation.
    /// Returns a report comparing unfused vs optimized dispatch counts.
    public func analyzeOptimization(
        graph: ModelGraph,
        hiddenSize: Int,
        stafWeightStore: STAFWeightStore? = nil
    ) -> OptimizationReport {
        let context = makeCompileContext(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: 0,
            vocabSize: 0,
            stafWeightStore: stafWeightStore,
            device: MTLCreateSystemDefaultDevice()!)
        let optimization = optimizedEntries(using: context, kernelContext: context.decodeKernelContext)
        let reportBuilder = OptimizationReportBuilder(optimizerName: optimizer.name)
        return reportBuilder.makeReport(
            unfusedEntries: optimization.walkContext.entries,
            optimizedEntries: optimization.fusedEntries)
    }

    /// Dump the compiled decode plan with concrete kernels, grid sizes, and bindings.
    ///
    /// This is a post-compilation diagnostic. Unlike `dumpDispatchEntries`, it shows
    /// the actual pipeline selected for each step after optimization, lowering,
    /// buffer routing, and kernel name resolution.
    public func dumpCompiledDecodePlan(
        graph: ModelGraph,
        hiddenSize: Int,
        intermediateSize: Int = 0,
        vocabSize: Int = 0,
        stafWeightStore: STAFWeightStore? = nil,
        device: MTLDevice
    ) throws -> String {
        let plan = try compile(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            stafWeightStore: stafWeightStore,
            device: device)
        let formatter = CompiledPlanDiagnosticsFormatter()
        return formatter.formatDecodePlan(plan)
    }

    /// Dump the compiled prefill plan with concrete kernels, launch modes, and bindings.
    public func dumpCompiledPrefillPlan(
        graph: ModelGraph,
        hiddenSize: Int,
        intermediateSize: Int = 0,
        vocabSize: Int = 0,
        maximumSequenceLength: Int,
        stafWeightStore: STAFWeightStore? = nil,
        device: MTLDevice
    ) throws -> String {
        let plan = try compilePrefill(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            maximumSequenceLength: maximumSequenceLength,
            stafWeightStore: stafWeightStore,
            device: device)
        let formatter = CompiledPlanDiagnosticsFormatter()
        return formatter.formatPrefillPlan(plan, maximumSequenceLength: maximumSequenceLength)
    }

    /// Dump the generated decode kernel source for the optimized entry set.
    ///
    /// Unlike `dumpCompiledDecodePlan`, this returns the actual MSL that is fed
    /// into `makeLibrary` for the current model after optimization and kernel
    /// selection.
    public func dumpGeneratedDecodeKernelLibrary(
        graph: ModelGraph,
        hiddenSize: Int,
        stafWeightStore: STAFWeightStore? = nil
    ) -> String {
        let context = makeCompileContext(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: 0,
            vocabSize: 0,
            stafWeightStore: stafWeightStore,
            device: MTLCreateSystemDefaultDevice()!)
        let optimization = optimizedEntries(using: context, kernelContext: context.decodeKernelContext)
        let sourceBuilder = KernelSourceBuilder(
            stafWeightStore: context.stafWeightStore,
            modelWeightFormat: context.weightFormat,
            bufferPrecision: context.decodeBufferPrecision,
            kernelNameResolver: kernelName(for:entry:stafWeightStore:kernelContext:))
        let generated = sourceBuilder.generateSources(entries: optimization.fusedEntries)
        return sourceBuilder.format(generated)
    }

    /// Dump the generated prefill kernel source for the optimized entry set.
    public func dumpGeneratedPrefillKernelLibrary(
        graph: ModelGraph,
        hiddenSize: Int,
        stafWeightStore: STAFWeightStore? = nil
    ) -> String {
        let context = makeCompileContext(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: 0,
            vocabSize: 0,
            stafWeightStore: stafWeightStore,
            device: MTLCreateSystemDefaultDevice()!)
        let optimization = optimizedEntries(using: context, kernelContext: context.prefillKernelContext)
        let sourceBuilder = KernelSourceBuilder(
            stafWeightStore: context.stafWeightStore,
            modelWeightFormat: context.weightFormat,
            bufferPrecision: .float32,
            kernelNameResolver: kernelName(for:entry:stafWeightStore:kernelContext:))
        let generated = sourceBuilder.generateSources(entries: optimization.fusedEntries)
        return sourceBuilder.format(generated)
    }

    public func compile(
        graph: ModelGraph,
        hiddenSize: Int,
        intermediateSize: Int = 0,
        vocabSize: Int = 0,
        stafWeightStore: STAFWeightStore? = nil,
        device: MTLDevice
    ) throws -> MetalDispatchPlan {
        let context = makeCompileContext(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            stafWeightStore: stafWeightStore,
            device: device)
        let optimization = optimizedEntries(using: context, kernelContext: context.decodeKernelContext)
        let walkContext = optimization.walkContext
        let unfusedCount = optimization.unfusedCount
        let fusedEntries = optimization.fusedEntries

        // Phase 3: Compile only the kernels needed by this model's dispatch entries
        // Decode uses F16 buffers (single token, no accumulation)
        let (pipelineCache, _) = try compilePipelineCache(
            entries: fusedEntries, stafWeightStore: context.stafWeightStore,
            bufferPrecision: context.decodeBufferPrecision, device: context.device)
        let planBuildContext = makePlanBuildContext(
            compileContext: context,
            kernelContext: context.decodeKernelContext,
            pipelineCache: pipelineCache)
        let allocation = try makeDecodeBufferAllocation(
            compileContext: context,
            walkContext: walkContext,
            fusedEntries: fusedEntries)
        let bufferSet = allocation.bufferSet
        let decodeSlotDimension = allocation.slotDimension
        let constantAllocator = MetalConstantBindingAllocator(device: context.device)
        let argumentAllocator = MetalArgumentBindingAllocator()
        let preparedArgumentAllocator = MetalPreparedArgumentBufferAllocator(device: context.device)

        print("[Compiler] \(fusedEntries.count) dispatch entries (\(optimizer.name) optimizer)")

        // Phase 5: Build dispatch steps with buffer routing
        var steps: [MetalDispatchStep] = []
        var routingPlanner = DecodeRoutingPlanner(
            bufferSet: bufferSet,
            stafWeightStore: context.stafWeightStore,
            hiddenSize: context.hiddenSize,
            slotDimension: decodeSlotDimension)

        for entry in fusedEntries {
            let resolved = try resolvedDispatch(for: entry, using: planBuildContext)
            let bindings = routingPlanner.bindings(for: entry)
            steps.append(MetalDispatchStep(
                pipeline: resolved.pipeline,
                gridSize: resolved.config.grid,
                threadgroupSize: resolved.config.threadgroup,
                bufferBindings: bindings.buffers,
                bytesBindings: bindings.bytes,
                threadgroupMemoryLength: resolved.config.sharedMemoryBytes,
                sync: .bufferBarrier
            ))
        }

        let residentSteps = try makeResidentConstantSteps(steps, allocator: constantAllocator)
        let argumentTableSteps = makeArgumentTableSteps(residentSteps, allocator: argumentAllocator)
        let preparedArgumentSteps = try makePreparedArgumentTableSteps(
            argumentTableSteps,
            allocator: preparedArgumentAllocator)
        let encodedArgumentSteps = makeEncodedArgumentTableSteps(
            preparedArgumentSteps,
            pipelineCache: planBuildContext.pipelineCache)

        return MetalDispatchPlan(
            steps: encodedArgumentSteps, buffers: bufferSet,
            unfusedEntryCount: unfusedCount, fusedEntryCount: fusedEntries.count)
    }

    // MARK: - Prefill Compilation

    /// Compile a sequence-aware prefill plan.
    ///
    /// The prefill plan is a **sequence graph**: step count is O(layers × ops_per_layer),
    /// NOT O(tokens × layers × ops_per_layer). Each kernel operates on [seqLen × dim]
    /// buffers. The GPU kernel itself iterates over the sequence dimension.
    ///
    /// - Projections: GEMM instead of GEMV ([seqLen × in] × [out × in]^T → [seqLen × out])
    /// - Embedding/Norm/Activation/Structural: batched variants with seqLen grid dimension
    /// - Attention: perPosition mode — runtime loops over positions for KV cache fill
    public func compilePrefill(
        graph: ModelGraph,
        hiddenSize: Int,
        intermediateSize: Int = 0,
        vocabSize: Int = 0,
        maximumSequenceLength: Int = 4096,
        stafWeightStore: STAFWeightStore? = nil,
        sharedKVCache: MetalKVCache? = nil,
        sharedConvState: MTLBuffer? = nil,
        sharedConvStateDimension: Int = 0,
        sharedConvStateKernelSize: Int = 0,
        device: MTLDevice
    ) throws -> MetalPrefillPlan {
        let context = makeCompileContext(
            graph: graph,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            maximumSequenceLength: maximumSequenceLength,
            stafWeightStore: stafWeightStore,
            device: device)
        let optimization = optimizedEntries(using: context, kernelContext: context.prefillKernelContext)
        let walkContext = optimization.walkContext
        let fusedEntries = optimization.fusedEntries

        // Compile only the kernels needed by this model's prefill dispatch entries
        // For prefill (F32), attempts Metal 4 MPP GEMM with fallback to naive GEMM.
        let (pipelineCache, prefillUsesMPP) = try compilePipelineCache(
            entries: fusedEntries, stafWeightStore: context.stafWeightStore,
            bufferPrecision: .float32, device: context.device)
        let planBuildContext = makePlanBuildContext(
            compileContext: context,
            kernelContext: context.prefillKernelContext,
            pipelineCache: pipelineCache)
        let allocation = try makePrefillBufferAllocation(
            compileContext: context,
            walkContext: walkContext,
            fusedEntries: fusedEntries,
            sharedKVCache: sharedKVCache,
            sharedConvState: sharedConvState,
            sharedConvStateDimension: sharedConvStateDimension,
            sharedConvStateKernelSize: sharedConvStateKernelSize)
        let prefillBuffers = allocation.bufferSet
        let slotDimension = allocation.slotDimension
        let maxSeq = allocation.maximumSequenceLength
        let f32ElementSize = MemoryLayout<Float32>.size
        let constantAllocator = MetalConstantBindingAllocator(device: context.device)

        // Build prefill steps — sequence-aware graph
        var steps: [MetalPrefillStep] = []
        var planner = PrefillStepPlanner(
            buffers: prefillBuffers,
            stafWeightStore: context.stafWeightStore,
            hiddenSize: context.hiddenSize,
            slotDimension: slotDimension,
            maximumSequenceLength: maxSeq,
            scratchElementSize: f32ElementSize,
            usesMPP: prefillUsesMPP,
            planBuildContext: planBuildContext,
            resolveDispatch: { try resolvedDispatch(for: $0, using: planBuildContext) })

        for entry in fusedEntries {
            let prefillSteps = try planner.buildSteps(for: entry)
            steps.append(contentsOf: prefillSteps)
        }

        let residentSteps = try makeResidentConstantSteps(steps, allocator: constantAllocator)

        // Prefill plan compiled silently — step count reported by ModelBundleLoader

        return MetalPrefillPlan(
            steps: residentSteps,
            buffers: prefillBuffers,
            maximumSequenceLength: maxSeq,
            stepCount: residentSteps.count
        )
    }

    private func makeResidentConstantSteps(
        _ steps: [MetalDispatchStep],
        allocator: MetalConstantBindingAllocator
    ) throws -> [MetalDispatchStep] {
        let bindingTables = steps.map(\.bindings)
        let residentBindings = try allocator.makeBindingTables(from: bindingTables)
        return zip(steps, residentBindings).map { step, bindings in
            MetalDispatchStep(descriptor: step.descriptor, bindings: bindings)
        }
    }

    private func makeResidentConstantSteps(
        _ steps: [MetalPrefillStep],
        allocator: MetalConstantBindingAllocator
    ) throws -> [MetalPrefillStep] {
        let bindingTables = steps.map(\.bindings)
        let residentBindings = try allocator.makeBindingTables(from: bindingTables)
        return zip(steps, residentBindings).map { step, bindings in
            MetalPrefillStep(
                descriptor: step.descriptor,
                bindings: bindings,
                mode: step.mode,
                sequenceLengthPolicy: step.sequenceLengthPolicy,
                positionBufferIndex: step.positionBufferIndex,
                perPositionStrides: step.perPositionStrides)
        }
    }

    private func makeArgumentTableSteps(
        _ steps: [MetalDispatchStep],
        allocator: MetalArgumentBindingAllocator
    ) -> [MetalDispatchStep] {
        let bindingTables = steps.map(\.bindings)
        let plannedBindings = allocator.makeBindingTables(from: bindingTables)
        return zip(steps, plannedBindings).map { step, bindings in
            MetalDispatchStep(descriptor: step.descriptor, bindings: bindings)
        }
    }

    private func makePreparedArgumentTableSteps(
        _ steps: [MetalDispatchStep],
        allocator: MetalPreparedArgumentBufferAllocator
    ) throws -> [MetalDispatchStep] {
        let bindingTables = steps.map(\.bindings)
        let preparedBindings = try allocator.makeBindingTables(from: bindingTables)
        return zip(steps, preparedBindings).map { step, bindings in
            MetalDispatchStep(descriptor: step.descriptor, bindings: bindings)
        }
    }

    private func makeEncodedArgumentTableSteps(
        _ steps: [MetalDispatchStep],
        pipelineCache: [String: MTLComputePipelineState]
    ) -> [MetalDispatchStep] {
        steps.map { step in
            guard
                let kernelLabel = step.pipeline.label,
                let variantKernelName = Self.encodedArgumentTableKernelName(
                    for: kernelLabel,
                    bindings: step.bindings),
                let variantPipeline = pipelineCache[variantKernelName],
                case .argumentTable(let table) = step.bindings.bufferBindings,
                case .prepared(let buffer, let index, let offset) = table.encodingState
            else {
                return step
            }

            let encodedBindings = MetalBindingTable(
                bufferBindings: .argumentTable(MetalArgumentTableBindings(
                    layout: table.layout,
                    bindings: table.bindings,
                    encodingState: .encoded(buffer: buffer, index: index, offset: offset))),
                constantBindings: Self.constantBindingsForEncodedVariant(
                    step.bindings.constantBindings,
                    variantKernelName: variantKernelName))
            let encodedDescriptor = MetalDispatchDescriptor(
                pipeline: variantPipeline,
                gridSize: step.gridSize,
                threadgroupSize: step.threadgroupSize,
                threadgroupMemoryLength: step.threadgroupMemoryLength,
                barrierPolicy: step.barrierPolicy)
            return MetalDispatchStep(descriptor: encodedDescriptor, bindings: encodedBindings)
        }
    }

    private static func constantBindingsForEncodedVariant(
        _ bindings: MetalConstantBindingSet,
        variantKernelName: String
    ) -> MetalConstantBindingSet {
        switch variantKernelName {
        case "gemv_2048_sq_argbuf", "gemv_2048_sq_bf16_argbuf",
             "gemv_2048_6144_argbuf", "gemv_2048_6144_bf16_argbuf":
            return .inline([])
        default:
            return bindings
        }
    }

    private static func encodedArgumentTableKernelName(
        for kernelName: String,
        bindings: MetalBindingTable
    ) -> String? {
        guard case .argumentTable(let table) = bindings.bufferBindings else {
            return nil
        }
        switch table.layout.indices {
        case [0, 1]:
            switch kernelName {
            case "argmax":
                return argumentTableVariantKernelName(for: kernelName)
            case "rms_norm", "rms_norm_bf16":
                return argumentTableVariantKernelName(for: kernelName)
            case "qk_rms_norm", "qk_rms_norm_bf16":
                return argumentTableVariantKernelName(for: kernelName)
            default:
                return nil
            }
        case [0, 1, 2]:
            switch kernelName {
            case "embedding_lookup", "embedding_lookup_bf16":
                return argumentTableVariantKernelName(for: kernelName)
            case "gemv_2048_sq", "gemv_2048_sq_bf16":
                return argumentTableVariantKernelName(for: kernelName)
            case "gemv_2048_6144", "gemv_2048_6144_bf16":
                return argumentTableVariantKernelName(for: kernelName)
            case "gemv_8192_tiled", "gemv_8192_tiled_bf16":
                return argumentTableVariantKernelName(for: kernelName)
            case "gemv", "gemv_bf16":
                return argumentTableVariantKernelName(for: kernelName)
            case "gemv_vocab", "gemv_vocab_bf16":
                return argumentTableVariantKernelName(for: kernelName)
            case "residual_add":
                return argumentTableVariantKernelName(for: kernelName)
            case "rope":
                return argumentTableVariantKernelName(for: kernelName)
            default:
                return nil
            }
        case [0, 1, 2, 3]:
            switch kernelName {
            case "fused_copy_rms_norm", "fused_copy_rms_norm_bf16":
                return argumentTableVariantKernelName(for: kernelName)
            case "fused_residual_add_copy_rms_norm", "fused_residual_add_copy_rms_norm_bf16":
                return argumentTableVariantKernelName(for: kernelName)
            case "fused_swiglu_projection_2048", "fused_swiglu_projection_2048_bf16":
                return argumentTableVariantKernelName(for: kernelName)
            case "conv_state_update", "conv_state_update_bf16":
                return argumentTableVariantKernelName(for: kernelName)
            default:
                return nil
            }
        default:
            if table.layout.indices == [0, 1, 2, 3, 4, 5, 6] {
                switch kernelName {
                case "flash_attn_decode":
                    return argumentTableVariantKernelName(for: kernelName)
                default:
                    return nil
                }
            }
            return nil
        }
    }

    private static func argumentTableVariantKernelName(for kernelName: String) -> String {
        kernelName + "_argbuf"
    }

    private struct PrefillStepPlanner {
        let buffers: PrefillBufferSet
        let stafWeightStore: STAFWeightStore?
        let hiddenSize: Int
        let slotDimension: Int
        let maximumSequenceLength: Int
        let scratchElementSize: Int  // 4 for float32 scratch
        let usesMPP: Bool
        let planBuildContext: PlanBuildContext
        let resolveDispatch: (DispatchEntry) throws -> (
            name: String,
            pipeline: MTLComputePipelineState,
            config: (grid: MTLSize, threadgroup: MTLSize, sharedMemoryBytes: Int)
        )
        var kvCacheIndex: Int = 0
        var routingState = BufferRoutingState()

        init(
            buffers: PrefillBufferSet,
            stafWeightStore: STAFWeightStore?,
            hiddenSize: Int,
            slotDimension: Int,
            maximumSequenceLength: Int,
            scratchElementSize: Int,
            usesMPP: Bool,
            planBuildContext: PlanBuildContext,
            resolveDispatch: @escaping (DispatchEntry) throws -> (
                name: String,
                pipeline: MTLComputePipelineState,
                config: (grid: MTLSize, threadgroup: MTLSize, sharedMemoryBytes: Int)
            )
        ) {
            self.buffers = buffers
            self.stafWeightStore = stafWeightStore
            self.hiddenSize = hiddenSize
            self.slotDimension = slotDimension
            self.maximumSequenceLength = maximumSequenceLength
            self.scratchElementSize = scratchElementSize
            self.usesMPP = usesMPP
            self.planBuildContext = planBuildContext
            self.resolveDispatch = resolveDispatch
        }

        mutating func buildSteps(for entry: DispatchEntry) throws -> [MetalPrefillStep] {
            let weightResolver = WeightResolver(
                entry: entry,
                stafWeightStore: stafWeightStore,
                fallbackBuffer: buffers.hidden,
                logsMisses: false)

            // Determine the sequence-aware kernel and buffer routing
            switch entry.kind {

        // MARK: Fragment-driven prefill steps (protocol dispatch — no type checks)
            case .fragment(let frag):
                let pipelineCache = planBuildContext.pipelineCache
                let prefillContext = PrefillBindingContext(
                    buffers: buffers, slotDimension: slotDimension,
                    scratchElementSize: scratchElementSize,
                    maximumSequenceLength: maximumSequenceLength,
                    kvCacheIndex: kvCacheIndex,
                    convLayerIndex: routingState.convLayerIndex,
                    kernelContext: planBuildContext.kernelContext,
                    resolveWeight: weightResolver.resolve,
                    getPipeline: { name in
                        guard let pipeline = pipelineCache[name] else {
                            throw MetalCompilerError.kernelNotFound(name)
                        }
                        return pipeline
                    })
                let result = try frag.prefillSteps(context: prefillContext)
                if result.resetsProjectionIndex { routingState.projectionIndex = 0 }
                if result.consumesKVCacheLayer { kvCacheIndex += 1 }
                if result.consumesConvLayer { routingState.convLayerIndex += 1 }
                routingState.lastOutputIsHidden = result.outputIsHidden
                return result.steps

        // MARK: Batched Projection → decompose into individual GEMMs
            case .batchedProjection(let batched):
                var steps: [MetalPrefillStep] = []
                for (i, proj) in batched.projections.enumerated() {
                    let singleProjection = MetalProjection(
                        field: proj.field,
                        inputDimension: proj.inputDimension,
                        outputDimension: proj.outputDimension)
                    let singleEntry = DispatchEntry(
                        index: entry.index + i,
                        kind: .projection(singleProjection, isOutput: false),
                        parameterBindings: entry.parameterBindings,
                        layerIndex: entry.layerIndex)
                    let projSteps = try buildSteps(for: singleEntry)
                    steps.append(contentsOf: projSteps)
                }
                return steps

        // MARK: Batched Fragment → decompose into individual per-head dispatches
            case .batchedFragment(let batch):
                var steps: [MetalPrefillStep] = []
                for (i, frag) in batch.fragments.enumerated() {
                    let singleEntry = DispatchEntry(
                        index: entry.index + i,
                        kind: .fragment(frag),
                        parameterBindings: entry.parameterBindings,
                        layerIndex: entry.layerIndex)
                    let fragSteps = try buildSteps(for: singleEntry)
                    steps.append(contentsOf: fragSteps)
                }
                return steps

        // MARK: Fused Residual Add + Norm → decompose into add + norm
            case .fusedResidualAddNorm(let fusedOp):
                // Decompose into structuralAdd + Reduction for prefill
                let addEntry = DispatchEntry(
                    index: entry.index,
                    kind: .structuralAdd(dimension: fusedOp.dimension),
                    parameterBindings: [],
                    layerIndex: entry.layerIndex)
                let normEntry = DispatchEntry(
                    index: entry.index + 1,
                    kind: .fragment(Reduction(dimension: fusedOp.dimension, epsilon: fusedOp.epsilon)),
                    parameterBindings: entry.parameterBindings,
                    layerIndex: entry.layerIndex)
                var steps: [MetalPrefillStep] = []
                for decomposed in [addEntry, normEntry] {
                    let s = try buildSteps(for: decomposed)
                    steps.append(contentsOf: s)
                }
                return steps

        // MARK: Fused gate_proj + up_proj + SwiGLU → decompose for prefill
            case .fusedSwiGLUProjection(let fusedOp):
                let batchedEntry = DispatchEntry(
                    index: entry.index,
                    kind: .batchedProjection(BatchedProjection(projections: [
                        .init(field: fusedOp.gateField, inputDimension: fusedOp.inputDimension, outputDimension: fusedOp.outputDimension),
                        .init(field: fusedOp.upField, inputDimension: fusedOp.inputDimension, outputDimension: fusedOp.outputDimension),
                    ])),
                    parameterBindings: entry.parameterBindings,
                    layerIndex: entry.layerIndex)
                let swigluEntry = DispatchEntry(
                    index: entry.index + 1,
                    kind: .fragment(ElementwiseFragment(count: fusedOp.outputDimension, kind: .swiglu)),
                    parameterBindings: entry.parameterBindings,
                    layerIndex: entry.layerIndex)

                var steps: [MetalPrefillStep] = []
                for decomposed in [batchedEntry, swigluEntry] {
                    let built = try buildSteps(for: decomposed)
                    steps.append(contentsOf: built)
                }
                return steps

        // MARK: Fused Copy + Norm → copy hidden→residual, then norm hidden→scratch[0]
        //
        // Decode fused kernel: hidden → residual (copy) + hidden → scratch[0] (norm).
        // Prefill: decompose into separate steps with same buffer routing.
        // Norm output goes to scratch[0] (NOT hidden in-place) so that
        // parallel projections (gate_proj, up_proj) can both read from scratch[0].
            case .fusedCopyNorm(let fusedOp):
                var steps: [MetalPrefillStep] = []

                // Step 1: copy hidden → residual
                let copyEntry = DispatchEntry(
                    index: entry.index,
                    kind: .structuralCopy(dimension: fusedOp.dimension),
                    parameterBindings: [],
                    layerIndex: entry.layerIndex)
                let copySteps = try buildSteps(for: copyEntry)
                steps.append(contentsOf: copySteps)

                // Step 2: norm hidden → scratch[0]
                let normSteps = try buildNormToScratchStep(
                    dimension: fusedOp.dimension,
                    epsilon: fusedOp.epsilon,
                    entry: entry)
                steps.append(contentsOf: normSteps)

                routingState.lastOutputIsHidden = false
                routingState.projectionIndex = 0
                return steps

        // MARK: Fused Residual Add + Copy + Norm → add + copy + norm→scratch[0]
            case .fusedResidualAddCopyNorm(let fusedOp):
                var steps: [MetalPrefillStep] = []

                // Step 1: residual add → hidden
                let addEntry = DispatchEntry(
                    index: entry.index,
                    kind: .structuralAdd(dimension: fusedOp.dimension),
                    parameterBindings: [],
                    layerIndex: entry.layerIndex)
                let addSteps = try buildSteps(for: addEntry)
                steps.append(contentsOf: addSteps)

                // Step 2: copy hidden → residual
                let copyEntry = DispatchEntry(
                    index: entry.index + 1,
                    kind: .structuralCopy(dimension: fusedOp.dimension),
                    parameterBindings: [],
                    layerIndex: entry.layerIndex)
                let copySteps = try buildSteps(for: copyEntry)
                steps.append(contentsOf: copySteps)

                // Step 3: norm hidden → scratch[0]
                let normSteps = try buildNormToScratchStep(
                    dimension: fusedOp.dimension,
                    epsilon: fusedOp.epsilon,
                    entry: entry)
                steps.append(contentsOf: normSteps)

                routingState.lastOutputIsHidden = false
                routingState.projectionIndex = 0
                return steps

        // MARK: Projection → GEMM (sequence matrix multiply)
            case .projection(let projection, let isOutput):
                let resolved = try resolveDispatch(entry)

                let (weightBuffer, weightOffset) = weightResolver.resolve(role: projection.field)

                let inputBuffer: MTLBuffer
                let inputOffset: Int
                if routingState.lastOutputIsHidden {
                    inputBuffer = buffers.hidden
                    inputOffset = 0
                } else {
                    inputBuffer = buffers.scratch
                    inputOffset = 0
                }

                let outputBuffer: MTLBuffer
                let outputOffset: Int
                let mode: PrefillStepMode
                let seqLenValue: UInt32

                // Prefill scratch uses slot-major layout:
                // slot N offset = N * slotDimension * scratchElementSize * maxSeqLen
                let scratchSlotSize = slotDimension * scratchElementSize * maximumSequenceLength

                if isOutput && projection.outputDimension > hiddenSize {
                    // OutputHead: logits buffer is [vocabSize], not seq-sized.
                    // Compute only the last position.
                    outputBuffer = buffers.logits
                    outputOffset = 0
                    mode = .lastToken
                    seqLenValue = 1
                    routingState.lastOutputIsHidden = false
                } else if isOutput {
                    outputBuffer = buffers.hidden
                    outputOffset = 0
                    mode = .batch
                    seqLenValue = UInt32(maximumSequenceLength)
                    routingState.lastOutputIsHidden = true
                } else {
                    let scratchSlot = routingState.projectionIndex + 1
                    outputBuffer = buffers.scratch
                    outputOffset = scratchSlot * scratchSlotSize
                    mode = .batch
                    seqLenValue = UInt32(maximumSequenceLength)
                    routingState.lastOutputIsHidden = false
                }
                routingState.projectionIndex += 1

                var perPositionStrides: [Int: Int] = [:]
                if mode == .lastToken {
                    perPositionStrides[0] = projection.inputDimension * scratchElementSize
                }

                // MPP matmul2d uses tile 64(M)×32(N) with 4 simdgroups.
                // Grid: (outputDim/32, seqLen/64, 1). Kernel handles edge tiles internally.
                // Naive GEMM: (outputDim/2, seqLen, 1) with 2 simdgroups.
                let gridSize: MTLSize
                let threadgroupSize: MTLSize
                if usesMPP && mode == .batch {
                    let simdWidth = resolved.pipeline.threadExecutionWidth
                    gridSize = MTLSize(
                        width: (projection.outputDimension + 31) / 32,
                        height: (maximumSequenceLength + 63) / 64,
                        depth: 1)
                    threadgroupSize = MTLSize(width: simdWidth * 4, height: 1, depth: 1)
                } else if mode == .lastToken {
                    gridSize = MTLSize(width: resolved.config.grid.width, height: 1, depth: 1)
                    threadgroupSize = resolved.config.threadgroup
                } else {
                    gridSize = MTLSize(width: resolved.config.grid.width, height: maximumSequenceLength, depth: 1)
                    threadgroupSize = resolved.config.threadgroup
                }

                return [MetalPrefillStep(
                    pipeline: resolved.pipeline,
                    gridSize: gridSize,
                    threadgroupSize: threadgroupSize,
                    bufferBindings: [
                        (0, inputBuffer, inputOffset),
                        (1, weightBuffer, weightOffset),
                        (2, outputBuffer, outputOffset),
                    ],
                    bytesBindings: [
                        uint32Binding(3, UInt32(projection.inputDimension)),
                        uint32Binding(4, UInt32(projection.outputDimension)),
                        uint32Binding(5, seqLenValue),
                    ],
                    threadgroupMemoryLength: (usesMPP && mode == .batch) ? 0 : resolved.config.sharedMemoryBytes,
                    sync: .bufferBarrier,
                    mode: mode,
                    // MPP GEMM still needs the runtime seqLen in buffer(5), but its grid height
                    // is tile-based instead of token-based.
                    sequenceLengthPolicy: mode == .batch
                        ? (usesMPP ? .bind(index: 5) : .bindAndAdjustGridHeight(index: 5))
                        : .none,
                    positionBufferIndex: nil,
                    perPositionStrides: perPositionStrides
                )]

        // MARK: Structural Copy (hidden → residual)
            case .structuralCopy(let dimension):
                let resolved = try resolveDispatch(entry)

                routingState.projectionIndex = 0

                return [MetalPrefillStep(
                    pipeline: resolved.pipeline,
                    gridSize: MTLSize(width: resolved.config.grid.width, height: maximumSequenceLength, depth: 1),
                    threadgroupSize: resolved.config.threadgroup,
                    bufferBindings: [
                        (0, buffers.hidden, 0),
                        (1, buffers.residual, 0),
                    ],
                    bytesBindings: [
                        uint32Binding(2, UInt32(dimension)),
                        uint32Binding(3, UInt32(maximumSequenceLength)),
                    ],
                    threadgroupMemoryLength: 0,
                    sync: .bufferBarrier,
                    mode: .batch,
                    sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 3),
                    positionBufferIndex: nil,
                    perPositionStrides: [:]
                )]

        // MARK: Structural Add (hidden + residual → hidden)
            case .structuralAdd(let dimension):
                let resolved = try resolveDispatch(entry)

                routingState.lastOutputIsHidden = true

                return [MetalPrefillStep(
                    pipeline: resolved.pipeline,
                    gridSize: MTLSize(width: resolved.config.grid.width, height: maximumSequenceLength, depth: 1),
                    threadgroupSize: resolved.config.threadgroup,
                    bufferBindings: [
                        (0, buffers.hidden, 0),
                        (1, buffers.residual, 0),
                        (2, buffers.hidden, 0),
                    ],
                    bytesBindings: [
                        uint32Binding(3, UInt32(dimension)),
                        uint32Binding(4, UInt32(maximumSequenceLength)),
                    ],
                    threadgroupMemoryLength: 0,
                    sync: .bufferBarrier,
                    mode: .batch,
                    sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 4),
                    positionBufferIndex: nil,
                    perPositionStrides: [:]
                )]
            }
        }

        /// Build a RMSNorm step that reads from hidden and writes to scratch[0].
        ///
        /// This matches the decode fusedCopyNorm/fusedResidualAddCopyNorm behavior where
        /// norm output goes to scratch[0], allowing parallel projections (gate+up, q+k+v)
        /// to all read from scratch[0] without interference.
        private func buildNormToScratchStep(
            dimension: Int,
            epsilon: Float,
            entry: DispatchEntry
        ) throws -> [MetalPrefillStep] {
            let weightResolver = WeightResolver(
                entry: entry,
                stafWeightStore: stafWeightStore,
                fallbackBuffer: buffers.hidden,
                logsMisses: false)

            let normKernelName = Reduction(dimension: dimension, epsilon: epsilon)
                .kernelName(context: planBuildContext.kernelContext)
            guard let pipeline = planBuildContext.pipelineCache[normKernelName] else {
                throw MetalCompilerError.kernelNotFound(normKernelName)
            }
            let simdWidth = pipeline.threadExecutionWidth
            let clamped = min(max(dimension, 1), 1024)
            let rounded = ((clamped + simdWidth - 1) / simdWidth) * simdWidth
            let threads = min(rounded, pipeline.maxTotalThreadsPerThreadgroup)

            let (weightBuffer, weightOffset) = weightResolver.resolve(role: "scale")

            return [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: maximumSequenceLength, height: 1, depth: 1),
                threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
                bufferBindings: [
                    (0, buffers.hidden, 0),
                    (1, weightBuffer, weightOffset),
                    (2, buffers.scratch, 0),
                ],
                bytesBindings: [
                    uint32Binding(3, UInt32(dimension)),
                    floatBinding(4, epsilon),
                    uint32Binding(5, UInt32(maximumSequenceLength)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 5),
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )]
        }
    }

    // MARK: - IR Walk

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
                index: nextIndex, kind: kind,
                parameterBindings: parameterBindings, layerIndex: layerIndex))
            nextIndex += 1
        }

        /// Emit an optimizer result entry.
        mutating func emitOptimized(_ entry: OptimizedEntry) {
            switch entry {
            case .single(let p):
                // .gemv dispatch dimension → .projection, others → .fragment
                if case .gemv(let outputDim, let inputDim) = p.fragment.dispatchDimension {
                    let field = p.fragment.weightSlots.first?.field ?? "weight"
                    let projection = MetalProjection(
                        field: field,
                        inputDimension: inputDim,
                        outputDimension: outputDim)
                    emit(.projection(projection),
                         parameterBindings: p.parameterBindings,
                         layerIndex: p.layerIndex)
                } else {
                    emit(.fragment(p.fragment),
                         parameterBindings: p.parameterBindings,
                         layerIndex: p.layerIndex)
                }
            case .batchedProjection(let batched, let bindings, let layer):
                emit(.batchedProjection(batched),
                     parameterBindings: bindings,
                     layerIndex: layer)
            case .fusedSwiGLUProjection(let fused, let bindings, let layer):
                emit(.fusedSwiGLUProjection(fused),
                     parameterBindings: bindings,
                     layerIndex: layer)
            case .batchedFragment(let batched, let bindings, let layer):
                emit(.batchedFragment(batched),
                     parameterBindings: bindings,
                     layerIndex: layer)
            }
        }
    }

    struct CacheSlotInfo {
        let kvHeadCount: Int
        let headDimension: Int
    }

    /// Resolve the primary weight format from the STAF weight store.
    private func resolveModelWeightFormat(_ stafWeightStore: STAFWeightStore?) -> WeightFormat {
        guard let staf = stafWeightStore else { return .float16 }
        for name in staf.entries.keys {
            if let info = staf.tensor(for: name) {
                return info.format.schemeIdentifier == .bf16RowMajor ? .bfloat16 : .float16
            }
        }
        return .float16
    }

    private func preferredKVCacheScheme(for weightFormat: WeightFormat) -> QuantizationSchemeIdentifier {
        if weightFormat == .bfloat16 {
            return .bf16RowMajor
        }
        return .fp16RowMajor
    }

    private func preferredDecodeBufferPrecision(for weightFormat: WeightFormat) -> BufferPrecision {
        _ = weightFormat
        return .float16
    }

    private func walkRegion(
        _ region: Region,
        pathComponents: [StructuralPathComponent],
        layerIndex: Int?,
        hiddenSize: Int,
        context: inout WalkContext,
        kernelContext: KernelContext
    ) {
        for (operationIndex, operation) in region.operations.enumerated() {
            let operationPath = pathComponents + [.operation(operationIndex)]
            let _ = StructuralPath(components: operationPath)

            switch operation.kind {
            case .residual(_, let body):
                context.emit(.structuralCopy(dimension: hiddenSize), layerIndex: layerIndex)
                walkRegion(body, pathComponents: operationPath + [.regionBody],
                           layerIndex: layerIndex, hiddenSize: hiddenSize, context: &context,
                           kernelContext: kernelContext)
                context.emit(.structuralAdd(dimension: hiddenSize), layerIndex: layerIndex)

            case .repeating(let count, let body):
                for iteration in 0..<count {
                    walkRegion(body,
                               pathComponents: operationPath + [.regionBody, .index(iteration)],
                               layerIndex: iteration, hiddenSize: hiddenSize, context: &context,
                               kernelContext: kernelContext)
                }

            case .conditional(let condition, let thenBody, let elseBody):
                if let currentLayer = layerIndex, case .layerIndices(let indices) = condition {
                    let selectedBody = indices.contains(currentLayer) ? thenBody : elseBody
                    walkRegion(selectedBody, pathComponents: operationPath + [.regionBody],
                               layerIndex: currentLayer, hiddenSize: hiddenSize, context: &context,
                               kernelContext: kernelContext)
                } else {
                    walkRegion(thenBody, pathComponents: operationPath + [.regionBody],
                               layerIndex: layerIndex, hiddenSize: hiddenSize, context: &context,
                               kernelContext: kernelContext)
                }

            case .parallel(_, let branches):
                for (branchIndex, branch) in branches.enumerated() {
                    walkRegion(branch,
                               pathComponents: operationPath + [.regionBranch(branchIndex)],
                               layerIndex: layerIndex, hiddenSize: hiddenSize, context: &context,
                               kernelContext: kernelContext)
                }

            case .primitive(let attributes):
                // Resolve layer index in parameterBindings
                let bindings: [ParameterBinding]
                if let currentLayerIndex = layerIndex {
                    bindings = operation.parameterBindings.map { binding in
                        let resolved = binding.tensorName.replacingOccurrences(
                            of: ".layers.0.", with: ".layers.\(currentLayerIndex).")
                        return ParameterBinding(role: binding.role, tensorName: resolved)
                    }
                } else {
                    bindings = operation.parameterBindings
                }

                // Fragment-driven path: collect → optimize → emit
                guard let fragment = attributes as? (any MetalKernelFragment) else { continue }
                var primitives: [CollectedPrimitive] = []
                collectPrimitives(fragment, bindings: bindings, layerIndex: layerIndex,
                                  primitives: &primitives, context: &context,
                                  kernelContext: kernelContext)
                let optimized = optimizer.optimizeFragment(primitives)
                let startIndex = context.entries.count
                for entry in optimized {
                    context.emitOptimized(entry)
                }
                markLastProjectionAsOutput(entries: &context.entries, from: startIndex)
            }
        }
    }

    // MARK: - Fragment Tree Walk

    // MARK: - Primitive Collection (for optimizer)

    /// Collect all primitives from a fragment tree without emitting.
    ///
    /// Similar to `emitFragmentTree()` but appends to an array instead of emitting.
    /// FlashAttentionFragment still registers KV cache slots in the context.
    private func collectPrimitives(
        _ fragment: any MetalKernelFragment,
        bindings: [ParameterBinding],
        layerIndex: Int?,
        primitives: inout [CollectedPrimitive],
        context: inout WalkContext,
        kernelContext: KernelContext
    ) {
        if let primitive = fragment as? any PrimitiveMetalKernelFragment {
            // Register KV cache slot from fragment's cache slot metadata
            for slot in primitive.cacheSlots where slot.kind == .kv {
                context.cacheSlots.append(CacheSlotInfo(
                    kvHeadCount: slot.kvHeadCount,
                    headDimension: slot.headDimension))
            }
            primitives.append(CollectedPrimitive(
                fragment: primitive,
                parameterBindings: bindings,
                layerIndex: layerIndex))
            return
        }

        // Walk composite fragments recursively
        if let tuple = fragment as? any _TupleFragmentProtocol {
            tuple._visitChildren { child in
                collectPrimitives(child, bindings: bindings, layerIndex: layerIndex,
                                  primitives: &primitives, context: &context,
                                  kernelContext: kernelContext)
            }
            return
        }
        if let opt = fragment as? any _OptionalFragmentProtocol {
            opt._visitContent { child in
                collectPrimitives(child, bindings: bindings, layerIndex: layerIndex,
                                  primitives: &primitives, context: &context,
                                  kernelContext: kernelContext)
            }
            return
        }
        if let cond = fragment as? any _ConditionalFragmentProtocol {
            cond._visitActive { child in
                collectPrimitives(child, bindings: bindings, layerIndex: layerIndex,
                                  primitives: &primitives, context: &context,
                                  kernelContext: kernelContext)
            }
            return
        }
        if let bodyAccessor = fragment as? any _FragmentBodyAccessor {
            bodyAccessor._visitBody(context: kernelContext) { child in
                collectPrimitives(child, bindings: bindings, layerIndex: layerIndex,
                                  primitives: &primitives, context: &context,
                                  kernelContext: kernelContext)
            }
        }
    }

    // MARK: - isOutput Resolution

    /// Mark the last projection in a range of entries as isOutput.
    private func markLastProjectionAsOutput(entries: inout [DispatchEntry], from startIndex: Int) {
        // Find the last projection in the range [startIndex..<entries.count]
        for i in stride(from: entries.count - 1, through: startIndex, by: -1) {
            if case .projection(let proj, _) = entries[i].kind {
                entries[i] = DispatchEntry(
                    index: entries[i].index,
                    kind: .projection(proj, isOutput: true),
                    parameterBindings: entries[i].parameterBindings,
                    layerIndex: entries[i].layerIndex)
                break
            }
        }
    }

    // MARK: - Kernel Name Resolution

    /// Map a DispatchKind to the MSL kernel function name.
    /// Map a DispatchKind to the MSL kernel function name.
    ///
    /// Uses KernelContext for weight format resolution — no STAF lookups.
    /// Projection kernels still use STAF for per-tensor quantization format
    /// (e.g., gemv_q4_g64 for quantized models).
    private func kernelName(
        for kind: DispatchKind,
        entry: DispatchEntry,
        stafWeightStore: STAFWeightStore?,
        kernelContext: KernelContext
    ) -> String {
        let weightFormatResolver = KernelWeightFormatResolver(stafWeightStore: stafWeightStore)
        let isBF16 = kernelContext.weightFormat == .bfloat16
        let bf16Suffix = isBF16 ? "_bf16" : ""

        let isPrefill = kernelContext.bufferPrecision == .float32

        switch kind {
        case .projection(let projection, _):
            // Projection uses per-tensor format from STAF (supports mixed quantization)
            if let binding = entry.parameterBindings.first(where: { $0.role == projection.field }),
               let staf = stafWeightStore,
               let tensorInfo = staf.tensor(for: binding.tensorName) {
                if !isPrefill,
                   let family = Self.denseDecodeProjectionFamily(
                    outputDimension: projection.outputDimension,
                    inputDimension: projection.inputDimension,
                    schemeIdentifier: tensorInfo.format.schemeIdentifier) {
                    return tensorInfo.format.schemeIdentifier == .bf16RowMajor
                        ? family.kernelBaseName + "_bf16"
                        : family.kernelBaseName
                }
                return isPrefill
                    ? tensorInfo.format.gemmKernelName(bufferPrecision: kernelContext.bufferPrecision)
                    : tensorInfo.format.gemvKernelName
            }
            if !isPrefill,
               let family = Self.denseDecodeProjectionFamily(
                outputDimension: projection.outputDimension,
                inputDimension: projection.inputDimension,
                schemeIdentifier: isBF16 ? .bf16RowMajor : .fp16RowMajor) {
                return isBF16 ? family.kernelBaseName + "_bf16" : family.kernelBaseName
            }
            return isPrefill ? (isBF16 ? "gemm_bf16_f32s" : "gemm_f32s") : "gemv"
        case .fragment(let frag):
            return frag.kernelName(context: kernelContext)
        case .fusedCopyNorm:
            return "fused_copy_rms_norm" + bf16Suffix
        case .fusedResidualAddCopyNorm:
            return "fused_residual_add_copy_rms_norm" + bf16Suffix
        case .fusedResidualAddNorm:
            return "fused_residual_add_rms_norm" + bf16Suffix
        case .fusedSwiGLUProjection(let fused):
            let weightFormat = weightFormatResolver.resolve(role: fused.gateField, entry: entry)
            let family = FusedSwiGLUProjectionFamily.resolve(
                inputDimension: fused.inputDimension,
                outputDimension: fused.outputDimension)
            return weightFormat == .bfloat16 ? family.kernelBaseName + "_bf16" : family.kernelBaseName
        case .batchedProjection(let batched):
            return "batched_gemv\(batched.projections.count)" + bf16Suffix
        case .batchedFragment(let batch):
            let baseName = batch.fragments[0].kernelName(context: kernelContext)
            return "batched_\(baseName)_\(batch.fragments.count)"
        case .structuralCopy:
            return isPrefill ? "copy_buffer_seq_f32" : "copy_buffer"
        case .structuralAdd:
            return isPrefill ? "residual_add_seq_f32" : "residual_add"
        }
    }

    /// Get the dispatch dimension for grid/threadgroup calculation.
    private func dispatchDimension(for kind: DispatchKind, hiddenSize: Int) -> MetalDispatchDimension {
        switch kind {
        case .projection(let projection, _):
            return .gemv(outputDimension: projection.outputDimension, inputDimension: projection.inputDimension)
        case .fusedCopyNorm(let fused):
            return .reduction(dimension: fused.dimension)
        case .fusedResidualAddCopyNorm(let fused):
            return .reduction(dimension: fused.dimension)
        case .fragment(let frag):
            return frag.dispatchDimension
        case .structuralCopy(let dimension):
            return .elementwise(count: dimension)
        case .structuralAdd(let dimension):
            return .elementwise(count: dimension)
        case .fusedResidualAddNorm(let fused):
            return .reduction(dimension: fused.dimension)
        case .fusedSwiGLUProjection(let fused):
            return .gemv(outputDimension: fused.outputDimension, inputDimension: fused.inputDimension)
        case .batchedProjection(let batched):
            return .gemv(outputDimension: batched.totalOutputDimension, inputDimension: batched.inputDimension)
        case .batchedFragment(let batch):
            return batch.dispatchDimension
        }
    }

    // MARK: - Buffer Routing

    /// Tracks the current data flow state through the dispatch sequence.
    struct BufferRoutingState {
        /// Which scratch sub-region the next GEMV should read from.
        var currentInputOffset: Int = 0
        /// Counter for parallel projections within one operation.
        var projectionIndex: Int = 0
        /// Whether the last dispatch wrote to hidden (vs scratch).
        var lastOutputIsHidden: Bool = true
        /// Counter for conv layers (for conv_state offset calculation).
        var convLayerIndex: Int = 0
    }

    private struct DecodeRoutingPlanner {
        let bufferSet: MetalBufferSet
        let stafWeightStore: STAFWeightStore?
        let hiddenSize: Int
        let slotDimension: Int
        private let elementSize = MemoryLayout<Float16>.size
        private var kvCacheIndex: Int = 0
        private var routingState = BufferRoutingState()

        init(
            bufferSet: MetalBufferSet,
            stafWeightStore: STAFWeightStore?,
            hiddenSize: Int,
            slotDimension: Int
        ) {
            self.bufferSet = bufferSet
            self.stafWeightStore = stafWeightStore
            self.hiddenSize = hiddenSize
            self.slotDimension = slotDimension
        }

        mutating func bindings(
            for entry: DispatchEntry
        ) -> (buffers: [(index: Int, buffer: MTLBuffer, offset: Int)],
              bytes: [(index: Int, value: [UInt8])]) {

            let weightResolver = WeightResolver(
                entry: entry,
                stafWeightStore: stafWeightStore,
                fallbackBuffer: bufferSet.hidden,
                logsMisses: true)

            func fusedNormBindings(dimension: Int, epsilon: Float) -> (
            buffers: [(index: Int, buffer: MTLBuffer, offset: Int)],
            bytes: [(index: Int, value: [UInt8])]
            ) {
                let (weightBuffer, weightOffset) = weightResolver.resolve(role: "scale")
                routingState.lastOutputIsHidden = false
                routingState.currentInputOffset = 0
                routingState.projectionIndex = 0
                return (
                    buffers: [
                        (0, bufferSet.hidden, 0),
                        (1, bufferSet.residual, 0),
                        (2, weightBuffer, weightOffset),
                        (3, bufferSet.scratch, 0),
                    ],
                    bytes: [
                        uint32Binding(4, UInt32(dimension)),
                        floatBinding(5, epsilon),
                    ]
                )
            }

            switch entry.kind {

            // MARK: Embedding Lookup
            // MARK: RMS Norm
            // Standalone RMSNorm (not fused with structuralCopy) writes in-place to hidden.
            // This ensures embedding_norm and final_norm results stay in hidden for
            // the next operation (Residual's structuralCopy or OutputHead projection).
            // In-place is safe: the kernel reads all elements for RMS before any writes.
            // MARK: Fused Copy + RMS Norm
            case .fusedCopyNorm(let fusedOperation):
                return fusedNormBindings(dimension: fusedOperation.dimension, epsilon: fusedOperation.epsilon)

            // MARK: Fused Residual Add + Copy + RMS Norm
            case .fusedResidualAddCopyNorm(let fusedOperation):
                return fusedNormBindings(dimension: fusedOperation.dimension, epsilon: fusedOperation.epsilon)

            // MARK: GEMV Projection
            case .projection(let projection, let isOutput):
                let (weightBuffer, weightOffset) = weightResolver.resolve(role: projection.field)

                // Input: scratch[0] after norm/compute, or hidden
                let inputBuffer: MTLBuffer
                let inputOffset: Int
                if routingState.lastOutputIsHidden {
                    inputBuffer = bufferSet.hidden
                    inputOffset = 0
                } else {
                    inputBuffer = bufferSet.scratch
                    inputOffset = routingState.currentInputOffset
                }

                // Output routing:
                // - OutputHead (vocabSize > hiddenSize): write to logits buffer
                // - Other isOutput projections (o_proj, down_proj): write to hidden
                // - Non-output projections: write to scratch slots
                let outputBuffer: MTLBuffer
                let outputOffset: Int

                if isOutput && projection.outputDimension > hiddenSize {
                    // OutputHead: output is vocabSize, too large for hidden buffer
                    outputBuffer = bufferSet.logits
                    outputOffset = 0
                    routingState.lastOutputIsHidden = false
                } else if isOutput {
                    outputBuffer = bufferSet.hidden
                    outputOffset = 0
                    routingState.lastOutputIsHidden = true
                } else {
                    let scratchSlot = routingState.projectionIndex + 1
                    outputBuffer = bufferSet.scratch
                    outputOffset = scratchSlot * slotDimension * elementSize
                    routingState.lastOutputIsHidden = false
                }

                routingState.projectionIndex += 1

                return (
                    buffers: [
                        (0, inputBuffer, inputOffset),
                        (1, weightBuffer, weightOffset),
                        (2, outputBuffer, outputOffset),
                    ],
                    bytes: [
                        uint32Binding(3, UInt32(projection.inputDimension)),
                        uint32Binding(4, UInt32(projection.outputDimension)),
                    ]
                )

            case .fusedSwiGLUProjection(let fusedOperation):
                let (gateWeightBuffer, gateWeightOffset) = weightResolver.resolve(role: fusedOperation.gateField)
                let (upWeightBuffer, upWeightOffset) = weightResolver.resolve(role: fusedOperation.upField)

                let inputBuffer: MTLBuffer
                let inputOffset: Int
                if routingState.lastOutputIsHidden {
                    inputBuffer = bufferSet.hidden
                    inputOffset = 0
                } else {
                    inputBuffer = bufferSet.scratch
                    inputOffset = routingState.currentInputOffset
                }

                routingState.lastOutputIsHidden = false
                routingState.currentInputOffset = slotDimension * elementSize
                routingState.projectionIndex = 0

                return (
                    buffers: [
                        (0, inputBuffer, inputOffset),
                        (1, gateWeightBuffer, gateWeightOffset),
                        (2, upWeightBuffer, upWeightOffset),
                        (3, bufferSet.scratch, slotDimension * elementSize),
                    ],
                    bytes: [
                        uint32Binding(4, UInt32(fusedOperation.inputDimension)),
                        uint32Binding(5, UInt32(fusedOperation.outputDimension)),
                    ]
                )

            // MARK: Flash Attention Decode
            // MARK: SwiGLU
            // MARK: Argmax
            // MARK: Structural Copy
            case .structuralCopy(let dimension):
                routingState.projectionIndex = 0
                return (
                    buffers: [
                        (0, bufferSet.hidden, 0),
                        (1, bufferSet.residual, 0),
                    ],
                    bytes: [
                        uint32Binding(2, UInt32(dimension)),
                    ]
                )

            // MARK: Structural Add
            case .structuralAdd(let dimension):
                routingState.lastOutputIsHidden = true
                return (
                    buffers: [
                        (0, bufferSet.hidden, 0),
                        (1, bufferSet.residual, 0),
                        (2, bufferSet.hidden, 0),
                    ],
                    bytes: [
                        uint32Binding(3, UInt32(dimension)),
                    ]
                )

            // MARK: RoPE
            // MARK: QK Norm (per-head RMS norm on Q or K in scratch)
            // MARK: Conv1d (double-gated depthwise conv with conv_state)
            //
            // in_proj output [3 × hiddenSize] per token: [B | C | x].
            // conv_state_update kernel: Bx = B*x → shift + append to conv_state → conv → C*convOut.
            // conv_state is required — ShortConv always declares a .conv cache slot.
            // MARK: Fragment-driven buffer routing (protocol dispatch — no type checks)
            case .fragment(let fragment):
                let bindingContext = BufferBindingContext(
                    bufferSet: bufferSet, slotDimension: slotDimension,
                    elementSize: elementSize, kvCacheIndex: kvCacheIndex,
                    convLayerIndex: routingState.convLayerIndex,
                    resolveWeight: weightResolver.resolve)
                let bindings = fragment.decodeBindings(context: bindingContext)
                if bindings.resetsProjectionIndex {
                    routingState.projectionIndex = 0
                    if !bindings.outputIsHidden {
                        routingState.currentInputOffset = 0
                    }
                }
                if bindings.consumesKVCacheLayer { kvCacheIndex += 1 }
                if bindings.consumesConvLayer { routingState.convLayerIndex += 1 }
                routingState.lastOutputIsHidden = bindings.outputIsHidden
                return (buffers: bindings.buffers, bytes: bindings.bytes)

            // MARK: Fused Residual Add + RMS Norm (no copy)
            case .fusedResidualAddNorm(let fusedOperation):
                let (weightBuffer, weightOffset) = weightResolver.resolve(role: "scale")
                routingState.lastOutputIsHidden = false
                routingState.currentInputOffset = 0
                routingState.projectionIndex = 0
                return (
                    buffers: [
                        (0, bufferSet.hidden, 0),
                        (1, bufferSet.residual, 0),
                        (2, weightBuffer, weightOffset),
                        (3, bufferSet.scratch, 0),
                    ],
                    bytes: [
                        uint32Binding(4, UInt32(fusedOperation.dimension)),
                        floatBinding(5, fusedOperation.epsilon),
                    ]
                )

            // MARK: Batched Projection
            case .batchedProjection(let batched):
                let inputBuffer: MTLBuffer
                let inputOffset: Int
                if routingState.lastOutputIsHidden {
                    inputBuffer = bufferSet.hidden
                    inputOffset = 0
                } else {
                    inputBuffer = bufferSet.scratch
                    inputOffset = 0
                }

                let count = batched.projections.count
                var bufferBindings: [(index: Int, buffer: MTLBuffer, offset: Int)] = [
                    (0, inputBuffer, inputOffset),
                ]
                var bytesBindings: [(index: Int, value: [UInt8])] = []

                // Bind weights: indices 1..<1+count
                for (i, proj) in batched.projections.enumerated() {
                    let (weightBuf, weightOff) = weightResolver.resolve(role: proj.field)
                    bufferBindings.append((1 + i, weightBuf, weightOff))
                }

                // Bind outputs: indices 1+count..<1+2*count
                for i in 0..<count {
                    let scratchSlot = routingState.projectionIndex + 1
                    let outputOffset = scratchSlot * slotDimension * elementSize
                    bufferBindings.append((1 + count + i, bufferSet.scratch, outputOffset))
                    routingState.projectionIndex += 1
                }

                // Bytes: inputDimension, then each outputDimension
                let bytesStart = 1 + 2 * count
                bytesBindings.append(uint32Binding(bytesStart, UInt32(batched.inputDimension)))
                for (i, proj) in batched.projections.enumerated() {
                    bytesBindings.append(uint32Binding(bytesStart + 1 + i, UInt32(proj.outputDimension)))
                }

                routingState.lastOutputIsHidden = false
                return (buffers: bufferBindings, bytes: bytesBindings)

            // MARK: Batched Fragment (per-head)
            case .batchedFragment(let batch):
                let slotBytes = slotDimension * elementSize
                var bufferBindings: [(index: Int, buffer: MTLBuffer, offset: Int)] = []
                var bytesBindings: [(index: Int, value: [UInt8])] = []

                // Data slots: in-place fragments operate on consecutive projection outputs.
                // First batched fragment → slot 1 (first projection output),
                // second → slot 2, etc.
                for i in 0..<batch.fragments.count {
                    let scratchSlotIndex = 1 + i
                    bufferBindings.append((i, bufferSet.scratch, scratchSlotIndex * slotBytes))
                }

                // Weight bindings: resolve from each fragment's weightSlots
                for (i, frag) in batch.fragments.enumerated() {
                    if let weightSlot = frag.weightSlots.first {
                        let role = weightSlot.field ?? "weight"
                        let (weightBuffer, weightOffset) = weightResolver.resolve(role: role)
                        bufferBindings.append((batch.fragments.count + i, weightBuffer, weightOffset))
                    }
                }

                // Constants: head count per fragment, then shared headDimension and epsilon.
                // All derived from fragment properties — no type checks.
                let bytesStart = 2 * batch.fragments.count
                for (i, frag) in batch.fragments.enumerated() {
                    if case .perHead(let headCount) = frag.dispatchDimension {
                        bytesBindings.append(uint32Binding(bytesStart + i, UInt32(headCount)))
                    }
                }

                // Extract headDimension from dispatchDimension + weight slot count.
                // For per-head fragments: total elements / headCount = headDimension.
                if case .perHead = batch.dispatchDimension,
                   let firstFrag = batch.fragments.first,
                   case .perHead(let firstHeadCount) = firstFrag.dispatchDimension {
                    // headDimension: infer from hiddenSize / headCount of first fragment.
                    // For QK norm: q_proj output = headCount * headDimension.
                    let headDimension = hiddenSize / firstHeadCount
                    bytesBindings.append(uint32Binding(bytesStart + batch.fragments.count, UInt32(headDimension)))
                    let epsilon = batch.fragments.first?.normEpsilon ?? 1e-6
                    bytesBindings.append(floatBinding(bytesStart + batch.fragments.count + 1, epsilon))
                }

                return (buffers: bufferBindings, bytes: bytesBindings)
            }
        }
    }

    // MARK: - Helpers

    // MARK: - Metal Library Cache


    /// Compile Metal libraries and build a pipeline cache for the given dispatch entries.
    ///
    /// Kernel source is generated on-demand from fragment parameters + STAF weight format.
    /// No hardcoded catalog — only the kernels actually used are compiled.
    /// For prefill (F32), attempts Metal 4 MPP GEMM with fallback to naive GEMM.
    private func compilePipelineCache(
        entries: [DispatchEntry],
        stafWeightStore: STAFWeightStore?,
        bufferPrecision: MetalSourceGenerator.BufferPrecision,
        device: MTLDevice
    ) throws -> (pipelines: [String: MTLComputePipelineState], usesMPP: Bool) {
        let sourceBuilder = KernelSourceBuilder(
            stafWeightStore: stafWeightStore,
            modelWeightFormat: resolveModelWeightFormat(stafWeightStore),
            bufferPrecision: bufferPrecision,
            kernelNameResolver: kernelName(for:entry:stafWeightStore:kernelContext:))
        let generated = sourceBuilder.generateSources(entries: entries)
        let libraryBuilder = PipelineLibraryBuilder(device: device)
        return try libraryBuilder.compile(generated)
    }

    private struct GeneratedKernelSources {
        let baseSource: String
        let mppSources: [String]
        let mppKernelNames: Set<String>
    }


    // MARK: - Helpers

    private func roundUp(_ value: Int, to multiple: Int) -> Int {
        guard multiple > 0 else { return max(value, 1) }
        return ((value + multiple - 1) / multiple) * multiple
    }

}
