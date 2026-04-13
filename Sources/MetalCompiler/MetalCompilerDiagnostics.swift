import Foundation
import Metal

struct DispatchEntryDiagnosticsFormatter {
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
            let preNormTag = f.preNorm != nil ? ", preNorm" : ""
            kind = "fusedResidualAddCopyNorm(dim=\(f.dimension), eps=\(f.epsilon)\(preNormTag))"
        case .fusedResidualAddNorm(let f):
            let preNormTag = f.preNorm != nil ? ", preNorm" : ""
            kind = "fusedResidualAddNorm(dim=\(f.dimension), eps=\(f.epsilon)\(preNormTag))"
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

struct OptimizationReportBuilder {
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

struct DecodeProjectionCostReportBuilder {
    private struct FamilyKey: Hashable {
        let kernelName: String
        let inputDimension: Int
        let outputDimension: Int
        let layoutKey: String
        let formatKey: String
    }

    private struct FamilyAccumulator {
        let kernelName: String
        let inputDimension: Int
        let outputDimension: Int
        let weightTensorCount: Int
        let layouts: [STAFWeightLayout]
        let formatIdentifiers: [QuantizationSchemeIdentifier]
        let inputBytesPerStep: Int64
        let weightBytesPerStep: Int64
        let outputBytesPerStep: Int64
        let estimatedFLOPsPerStep: Int64
        var stepCount: Int
    }

    private struct ProjectionComponent {
        let role: String
        let tensorName: String?
        let inputDimension: Int
        let outputDimension: Int
    }

    let decodeBufferPrecision: BufferPrecision
    let kernelContext: KernelContext
    let stafWeightStore: STAFWeightStore?
    let accessPolicyResolver: ProjectionWeightAccessPolicyResolver
    let kernelNameResolver: MetalKernelNameResolver

    func makeReport(entries: [DispatchEntry]) -> DecodeProjectionCostReport {
        var accumulators: [FamilyKey: FamilyAccumulator] = [:]
        var totalProjectionSteps = 0

        for entry in entries {
            guard let estimate = estimate(for: entry) else {
                continue
            }
            totalProjectionSteps += 1
            let key = FamilyKey(
                kernelName: estimate.kernelName,
                inputDimension: estimate.inputDimension,
                outputDimension: estimate.outputDimension,
                layoutKey: estimate.layouts.map(Self.describe(layout:)).joined(separator: "|"),
                formatKey: estimate.formatIdentifiers.map { String($0.rawValue) }.joined(separator: "|")
            )
            if var existing = accumulators[key] {
                existing.stepCount += 1
                accumulators[key] = existing
            } else {
                accumulators[key] = estimate
            }
        }

        let families = accumulators.values
            .map { accumulator in
                DecodeProjectionCostReport.FamilyEstimate(
                    kernelName: accumulator.kernelName,
                    inputDimension: accumulator.inputDimension,
                    outputDimension: accumulator.outputDimension,
                    stepCount: accumulator.stepCount,
                    weightTensorCount: accumulator.weightTensorCount,
                    layouts: accumulator.layouts,
                    formatIdentifiers: accumulator.formatIdentifiers,
                    inputBytesPerStep: accumulator.inputBytesPerStep,
                    weightBytesPerStep: accumulator.weightBytesPerStep,
                    outputBytesPerStep: accumulator.outputBytesPerStep,
                    estimatedFLOPsPerStep: accumulator.estimatedFLOPsPerStep
                )
            }
            .sorted { lhs, rhs in
                if lhs.totalEstimatedBytes == rhs.totalEstimatedBytes {
                    return lhs.kernelName < rhs.kernelName
                }
                return lhs.totalEstimatedBytes > rhs.totalEstimatedBytes
            }

        return DecodeProjectionCostReport(
            totalProjectionSteps: totalProjectionSteps,
            families: families
        )
    }

    private func estimate(for entry: DispatchEntry) -> FamilyAccumulator? {
        guard let components = projectionComponents(for: entry), !components.isEmpty else {
            return nil
        }

        let kernelName = kernelNameResolver.kernelName(for: entry, kernelContext: kernelContext)

        let decodeElementBytes = Int64(decodeBufferPrecision.byteSize)
        let inputDimension = components[0].inputDimension
        let outputDimension = components.reduce(0) { $0 + $1.outputDimension }
        let inputBytesPerStep = Int64(inputDimension) * decodeElementBytes
        let outputBytesPerStep = Int64(outputDimension) * decodeElementBytes

        var weightBytesPerStep: Int64 = 0
        var estimatedFLOPsPerStep: Int64 = 0
        var layouts: [STAFWeightLayout] = []
        var formatIdentifiers: [QuantizationSchemeIdentifier] = []

        for component in components {
            let resolved = resolveWeightAccess(for: component, entry: entry)
            weightBytesPerStep += logicalWeightBytes(
                inputDimension: component.inputDimension,
                outputDimension: component.outputDimension,
                format: resolved.format
            )
            estimatedFLOPsPerStep += 2 * Int64(component.inputDimension) * Int64(component.outputDimension)
            layouts.append(resolved.layout)
            formatIdentifiers.append(resolved.format.schemeIdentifier)
        }

        let uniqueLayouts = uniquedLayouts(layouts)
        let uniqueFormats = uniquedFormats(formatIdentifiers)

        return FamilyAccumulator(
            kernelName: kernelName,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            weightTensorCount: components.count,
            layouts: uniqueLayouts,
            formatIdentifiers: uniqueFormats,
            inputBytesPerStep: inputBytesPerStep,
            weightBytesPerStep: weightBytesPerStep,
            outputBytesPerStep: outputBytesPerStep,
            estimatedFLOPsPerStep: estimatedFLOPsPerStep,
            stepCount: 1
        )
    }

    private func projectionComponents(for entry: DispatchEntry) -> [ProjectionComponent]? {
        switch entry.kind {
        case .projection(let projection, _):
            return [
                ProjectionComponent(
                    role: projection.field,
                    tensorName: entry.parameterBindings.first(where: { $0.role == projection.field })?.tensorName,
                    inputDimension: projection.inputDimension,
                    outputDimension: projection.outputDimension
                )
            ]
        case .batchedProjection(let batched):
            return batched.projections.map { projection in
                ProjectionComponent(
                    role: projection.field,
                    tensorName: entry.parameterBindings.first(where: { $0.role == projection.field })?.tensorName,
                    inputDimension: projection.inputDimension,
                    outputDimension: projection.outputDimension
                )
            }
        default:
            return nil
        }
    }

    private func resolveWeightAccess(
        for component: ProjectionComponent,
        entry: DispatchEntry
    ) -> (format: any QuantizationFormat, layout: STAFWeightLayout) {
        let fallbackFormat: any QuantizationFormat = kernelContext.weightFormat == .bfloat16
            ? BFloat16Format()
            : Float16Format()
        guard
            let stafWeightStore,
            let tensorName = component.tensorName,
            let binding = entry.parameterBindings.first(where: { $0.role == component.role })
        else {
            return (fallbackFormat, .rowMajor)
        }

        let request = accessPolicyResolver.accessRequest(
            for: entry,
            role: component.role,
            binding: binding,
            executionPhase: .decode,
            stafWeightStore: stafWeightStore
        )
        let resolvedLayout = stafWeightStore
            .resolvedBufferAccess(for: request)?
            .layout ?? request.preferredLayout
        let format = stafWeightStore.tensor(for: tensorName)?.format ?? fallbackFormat
        return (format, resolvedLayout)
    }

    private func logicalWeightBytes(
        inputDimension: Int,
        outputDimension: Int,
        format: any QuantizationFormat
    ) -> Int64 {
        let logicalWeightCount = Int64(inputDimension) * Int64(outputDimension)
        let weightsPerBlock = Int64(format.weightsPerBlock)
        let blockCount = (logicalWeightCount + weightsPerBlock - 1) / weightsPerBlock
        return blockCount * Int64(format.bytesPerBlock)
    }

    private static func compare(layout lhs: STAFWeightLayout, rhs: STAFWeightLayout) -> Bool {
        rank(for: lhs) < rank(for: rhs)
    }

    private func uniquedLayouts(_ layouts: [STAFWeightLayout]) -> [STAFWeightLayout] {
        var unique: [STAFWeightLayout] = []
        for layout in layouts {
            if !unique.contains(where: { $0 == layout }) {
                unique.append(layout)
            }
        }
        return unique.sorted(by: Self.compare(layout:rhs:))
    }

    private func uniquedFormats(
        _ formatIdentifiers: [QuantizationSchemeIdentifier]
    ) -> [QuantizationSchemeIdentifier] {
        var unique: [QuantizationSchemeIdentifier] = []
        for identifier in formatIdentifiers {
            if !unique.contains(where: { $0 == identifier }) {
                unique.append(identifier)
            }
        }
        return unique.sorted { $0.rawValue < $1.rawValue }
    }

    private static func rank(for layout: STAFWeightLayout) -> Int {
        switch layout {
        case .rowMajor:
            return 0
        case .blockedRows4Tiles128:
            return 1
        case .blockedRows8Tiles128:
            return 2
        }
    }

    private static func describe(layout: STAFWeightLayout) -> String {
        switch layout {
        case .rowMajor:
            return "rowMajor"
        case .blockedRows4Tiles128:
            return "blocked4x128"
        case .blockedRows8Tiles128:
            return "blocked8x128"
        }
    }
}

struct CompiledPlanDiagnosticsFormatter {
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
        lines.append(contentsOf: plan.quantizationPlan.summarizedLines())
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
        lines.append(contentsOf: plan.quantizationPlan.summarizedLines())
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
