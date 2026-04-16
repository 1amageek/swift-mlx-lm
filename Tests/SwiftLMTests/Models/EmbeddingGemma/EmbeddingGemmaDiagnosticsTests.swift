import Metal
import Testing
@testable import MetalCompiler
@testable import SwiftLM

@Suite("EmbeddingGemma Diagnostics", .serialized)
struct EmbeddingGemmaDiagnosticsTests {
    @Test("Configured EmbeddingGemma reports the first unstable prefill step", .timeLimit(.minutes(10)))
    func configuredEmbeddingGemmaReportsFirstUnstablePrefillStep() async throws {
        guard let container = try await EmbeddingGemmaTestSupport.realEmbeddingGemmaContainer() else {
            print("[Skip] No local or configured EmbeddingGemma snapshot found")
            return
        }

        let prepared = try container.runtime.prepare(
            text: "swift metal inference",
            promptName: "query",
            tokenizer: container.tokenizer
        )
        let isolatedPlan = try container.prefillPlan.makeRuntimeIsolatedCopy(device: container.device)
        let modelDirectory = try #require(try EmbeddingGemmaTestSupport.optionalRealEmbeddingGemmaDirectory())
        let resources = try ModelBundleInspector().inspect(directory: modelDirectory)
        let weightStore = try STAFCacheLoader().load(resources: resources, device: container.device)
        let residency = try MetalResidencyLease.combined(
            label: "swift-lm.embeddinggemma.diagnostics",
            leases: [
                MetalResidencyLease.required(
                    device: container.device,
                    label: "swift-lm.embeddinggemma.runtime",
                    buffers: isolatedPlan.buffers.runtimeResidencyBuffers
                ),
                MetalResidencyLease.required(
                    device: container.device,
                    label: "swift-lm.embeddinggemma.weights",
                    buffers: isolatedPlan.buffers.weightResidencyBuffers
                ),
                MetalResidencyLease.required(
                    device: container.device,
                    label: "swift-lm.embeddinggemma.supplemental",
                    buffers: isolatedPlan.supplementalResidencyBuffers
                ),
            ]
        )

        print(
            "[EmbeddingGemma.Diag] prefill steps=\(isolatedPlan.steps.count) " +
                "maxSeq=\(isolatedPlan.maximumSequenceLength) " +
                "slotDim=\(isolatedPlan.slotDimension) " +
                "finalHiddenBaseOffset=\(isolatedPlan.finalHiddenBaseOffset) " +
                "finalHiddenRowStride=\(isolatedPlan.finalHiddenRowStride)"
        )
        for (index, step) in isolatedPlan.steps.enumerated() {
            let kernel = step.metadata.kernelName ?? step.pipeline.label ?? "(unlabeled)"
            let weight = step.metadata.weightTensorName ?? "-"
            let bindings = step.bindings.buffers
                .map { "b\($0.index)@+\($0.offset)" }
                .joined(separator: ",")
            let constants = step.bindings.constants
                .map { binding in
                    let value = decodeUInt32(binding).map(String.init) ?? "?"
                    return "c\(binding.index)=\(value)"
                }
                .joined(separator: ",")
            print(
                "[EmbeddingGemma.Diag] plan step=\(index) kernel=\(kernel) " +
                    "weight=\(weight) mode=\(step.mode) barrier=\(step.barrierPolicy) " +
                    "bindings=[\(bindings)] constants=[\(constants)]"
            )
        }

        var submission = try MetalSubmissionContext(device: container.device)
        let stepIndices = Array(isolatedPlan.steps.indices)
        let snapshots = try MetalPrefillExecutor().captureLastTokenHiddenSnapshots(
            prefillPlan: isolatedPlan,
            submission: &submission,
            position: 0,
            tokens: prepared.tokenIDs.map(Int32.init),
            stepIndices: Set(stepIndices),
            ephemeralResidency: residency
        )
        var residualSubmission = try MetalSubmissionContext(device: container.device)
        let residualSnapshots = try MetalPrefillExecutor().captureLastTokenResidualSnapshots(
            prefillPlan: isolatedPlan,
            submission: &residualSubmission,
            position: 0,
            tokens: prepared.tokenIDs.map(Int32.init),
            stepIndices: Set(stepIndices),
            ephemeralResidency: residency
        )

        for index in stepIndices {
            guard let snapshot = snapshots[index] else { continue }
            let kernelName = isolatedPlan.steps[index].metadata.kernelName
                ?? isolatedPlan.steps[index].pipeline.label
                ?? "(unlabeled)"
            let bindings = isolatedPlan.steps[index].bindings.buffers
            let finite = snapshot.allSatisfy { $0.isFinite }
            let norm = l2Norm(snapshot)
            let preview = snapshot.prefix(8).map { String(format: "%.5f", $0) }.joined(separator: ", ")
            let residualPreviewValues = residualSnapshots[index] ?? []
            let residualNorm = l2Norm(residualPreviewValues)
            let residualPreview = residualPreviewValues
                .prefix(8)
                .map { String(format: "%.5f", $0) }
                .joined(separator: ", ")
            print(
                "[EmbeddingGemma.Diag] step=\(index) kernel=\(kernelName) " +
                "finite=\(finite) norm=\(String(format: "%.5f", norm)) values=[\(preview)]"
            )
            if (34...36).contains(index) {
                print(
                    "[EmbeddingGemma.Diag]   residual@step=\(index) " +
                    "norm=\(String(format: "%.5f", residualNorm)) " +
                    "values=[\(residualPreview)]"
                )
            }
            if finite == false || norm.isFinite == false || norm <= 0.001 {
                print(
                    "[EmbeddingGemma.Diag]   residualNorm=\(String(format: "%.5f", residualNorm)) " +
                    "residualValues=[\(residualPreview)]"
                )
                for binding in bindings {
                    let label = binding.buffer.label ?? "(nil)"
                    let bufferKind: String
                    if binding.buffer === isolatedPlan.buffers.hidden {
                        bufferKind = "hidden"
                    } else if binding.buffer === isolatedPlan.buffers.residual {
                        bufferKind = "residual"
                    } else if binding.buffer === isolatedPlan.buffers.scratch {
                        bufferKind = "scratch"
                    } else if binding.buffer === isolatedPlan.buffers.logits {
                        bufferKind = "logits"
                    } else {
                        bufferKind = "other"
                    }
                    print(
                        "[EmbeddingGemma.Diag]   binding index=\(binding.index) " +
                        "offset=\(binding.offset) length=\(binding.buffer.length) " +
                            "label=\(label) kind=\(bufferKind)"
                    )
                }
                let constants = isolatedPlan.steps[index].bindings.constants
                    .map { binding in
                        let value = decodeUInt32(binding).map(String.init) ?? "?"
                        return "\(binding.index)=\(value)"
                    }
                    .joined(separator: ", ")
                print("[EmbeddingGemma.Diag]   constants=[\(constants)]")
                if let input = bindings.first(where: { $0.index == 0 }),
                   let output = bindings.first(where: { $0.index == 2 }) {
                    print(
                        "[EmbeddingGemma.Diag]   inputOutputAlias=" +
                            "\(input.buffer === output.buffer && input.offset == output.offset)"
                    )
                }
                if let weightBinding = bindings.first(where: { $0.index == 1 }) {
                    let matchingEntry = weightStore.entries.first { _, entry in
                        entry.bufferOffset == weightBinding.offset
                    }
                    if let matchingEntry {
                        print(
                            "[EmbeddingGemma.Diag]   weightTensor=\(matchingEntry.key) " +
                                "format=\(matchingEntry.value.schemeIdentifier)"
                        )
                    } else {
                        print("[EmbeddingGemma.Diag]   weightTensor=(unresolved)")
                    }
                    let weightPreview = readFloat16Values(
                        buffer: weightBinding.buffer,
                        offset: weightBinding.offset,
                        count: 8
                    )
                        .map { String(format: "%.5f", $0) }
                        .joined(separator: ", ")
                    print("[EmbeddingGemma.Diag]   weightPreview=[\(weightPreview)]")
                }
#if ENABLE_METAL_PROBES
                for projectionStep in [3, 4, 5, 11] where projectionStep < isolatedPlan.steps.count {
                    if let projectionWeight = isolatedPlan.steps[projectionStep].bindings.buffers.first(where: { $0.index == 1 }) {
                        let matchingEntry = weightStore.entries.first { _, entry in
                            entry.bufferOffset == projectionWeight.offset
                        }
                        let tensorName = matchingEntry?.key ?? "(unresolved)"
                        let preview = readFloat16Values(
                            buffer: projectionWeight.buffer,
                            offset: projectionWeight.offset,
                            count: 8
                        )
                            .map { String(format: "%.5f", $0) }
                            .joined(separator: ", ")
                        print(
                            "[EmbeddingGemma.Diag]   weight step=\(projectionStep) tensor=\(tensorName) " +
                                "values=[\(preview)]"
                        )
                    }
                }
                let probeSnapshots = try MetalPrefillExecutor().captureBindingProbes(
                    prefillPlan: isolatedPlan,
                    submission: &submission,
                    position: 0,
                    tokens: prepared.tokenIDs.map(Int32.init),
                    probes: makePrefillBindingProbes(for: isolatedPlan),
                    visibilityOptions: []
                )
                for label in probeSnapshots.keys.sorted() {
                    let values = probeSnapshots[label, default: []]
                    let probeFinite = values.allSatisfy { $0.isFinite }
                    let probeNorm = l2Norm(values)
                    let probePreview = values
                        .prefix(8)
                        .map { String(format: "%.5f", $0) }
                        .joined(separator: ", ")
                    print(
                        "[EmbeddingGemma.Diag]   probe=\(label) " +
                            "finite=\(probeFinite) norm=\(String(format: "%.5f", probeNorm)) " +
                            "values=[\(probePreview)]"
                    )
                }
                if let normalizedHidden = snapshots[2],
                   let qWeight = isolatedPlan.steps[3].bindings.buffers.first(where: { $0.index == 1 }) {
                    let cpuQ0 = dotRow0(
                        input: normalizedHidden,
                        weightBuffer: qWeight.buffer,
                        weightOffset: qWeight.offset,
                        inputDimension: 768
                    )
                    let gpuQ0 = probeSnapshots["q-out"]?.first ?? .nan
                    print(
                        "[EmbeddingGemma.Diag]   q-row0 cpu=\(String(format: "%.5f", cpuQ0)) " +
                            "gpu=\(String(format: "%.5f", gpuQ0))"
                    )
                }
#else
                print("[EmbeddingGemma.Diag]   binding probes disabled; rebuild with ENABLE_METAL_PROBES=1")
#endif
                break
            }
        }

        #expect(snapshots.isEmpty == false)
    }

    private func l2Norm(_ values: [Float]) -> Float {
        values.reduce(into: Float.zero) { partial, value in
            partial += value * value
        }.squareRoot()
    }

    private func decodeUInt32(_ bytes: [UInt8]) -> UInt32? {
        guard bytes.count >= MemoryLayout<UInt32>.size else { return nil }
        return bytes.prefix(4).enumerated().reduce(UInt32.zero) { partial, element in
            partial | (UInt32(element.element) << (UInt32(element.offset) * 8))
        }
    }

    private func decodeUInt32(_ binding: MetalConstantBinding) -> UInt32? {
        switch binding {
        case .inline(let bytes):
            return decodeUInt32(bytes.value)
        case .buffer(let buffer):
            let elementOffset = buffer.offset / MemoryLayout<UInt32>.stride
            let pointer = buffer.buffer.contents().bindMemory(
                to: UInt32.self,
                capacity: buffer.buffer.length / MemoryLayout<UInt32>.stride
            )
            guard buffer.offset + MemoryLayout<UInt32>.stride <= buffer.buffer.length else {
                return nil
            }
            return pointer[elementOffset]
        }
    }

    private func readFloat16Values(
        buffer: MTLBuffer,
        offset: Int,
        count: Int
    ) -> [Float] {
        guard offset >= 0, offset + count * MemoryLayout<Float16>.stride <= buffer.length else {
            return []
        }
        let pointer = buffer.contents().advanced(by: offset).assumingMemoryBound(to: Float16.self)
        return (0..<count).map { Float(pointer[$0]) }
    }

    private func dotRow0(
        input: [Float],
        weightBuffer: MTLBuffer,
        weightOffset: Int,
        inputDimension: Int
    ) -> Float {
        let weights = readFloat16Values(
            buffer: weightBuffer,
            offset: weightOffset,
            count: inputDimension
        )
        return zip(input.prefix(inputDimension), weights).reduce(Float.zero) { partial, pair in
            partial + pair.0 * pair.1
        }
    }

#if ENABLE_METAL_PROBES
    private func makePrefillBindingProbes(
        for plan: MetalPrefillPlan
    ) -> [MetalInferenceModel.DebugPrefillBindingProbe] {
        var probes: [MetalInferenceModel.DebugPrefillBindingProbe] = []

        func uint32Value(stepIndex: Int, bindingIndex: Int) -> Int {
            guard stepIndex >= 0, stepIndex < plan.steps.count else { return 0 }
            guard let binding = plan.steps[stepIndex].bindings.constants.first(where: { $0.index == bindingIndex }),
                  let value = decodeUInt32(binding) else {
                return 0
            }
            return Int(value)
        }

        func stepIndex(
            matchingWeightSuffix suffix: String
        ) -> Int? {
            plan.steps.firstIndex { step in
                guard let tensorName = step.metadata.weightTensorName else { return false }
                return tensorName.hasSuffix(suffix)
            }
        }

        func stepIndex(
            matchingKernelContains fragment: String
        ) -> Int? {
            plan.steps.firstIndex { step in
                let name = step.metadata.kernelName ?? step.pipeline.label ?? ""
                return name.contains(fragment)
            }
        }

        func addProbe(
            label: String,
            stepIndex: Int,
            bindingIndex: Int,
            phase: MetalInferenceModel.DebugPrefillProbePhase,
            rowStride: Int,
            count: Int
        ) {
            guard stepIndex < plan.steps.count else { return }
            guard rowStride > 0, count > 0 else { return }
            probes.append(
                .init(
                    label: label,
                    stepIndex: stepIndex,
                    bindingIndex: bindingIndex,
                    phase: phase,
                    rowStride: rowStride,
                    count: count,
                    precision: .float32
                )
            )
        }

        if let qStep = stepIndex(matchingWeightSuffix: ".self_attn.q_proj.weight") {
            addProbe(
                label: "q-out",
                stepIndex: qStep,
                bindingIndex: 2,
                phase: .afterStep,
                rowStride: plan.slotDimension,
                count: uint32Value(stepIndex: qStep, bindingIndex: 4)
            )
        }
        if let kStep = stepIndex(matchingWeightSuffix: ".self_attn.k_proj.weight") {
            addProbe(
                label: "k-out",
                stepIndex: kStep,
                bindingIndex: 2,
                phase: .afterStep,
                rowStride: plan.slotDimension,
                count: uint32Value(stepIndex: kStep, bindingIndex: 4)
            )
        }
        if let vStep = stepIndex(matchingWeightSuffix: ".self_attn.v_proj.weight") {
            addProbe(
                label: "v-out",
                stepIndex: vStep,
                bindingIndex: 2,
                phase: .afterStep,
                rowStride: plan.slotDimension,
                count: uint32Value(stepIndex: vStep, bindingIndex: 4)
            )
        }

        if let attnStep = stepIndex(matchingKernelContains: "flash_attn") {
            let headCount = uint32Value(stepIndex: attnStep, bindingIndex: 4)
            let headDimension = uint32Value(stepIndex: attnStep, bindingIndex: 6)
            addProbe(
                label: "attn-out",
                stepIndex: attnStep,
                bindingIndex: 3,
                phase: .afterStep,
                rowStride: plan.slotDimension,
                count: headCount * headDimension
            )
        }

        if let oStep = stepIndex(matchingWeightSuffix: ".self_attn.o_proj.weight") {
            addProbe(
                label: "o-proj-in",
                stepIndex: oStep,
                bindingIndex: 0,
                phase: .beforeStep,
                rowStride: plan.slotDimension,
                count: uint32Value(stepIndex: oStep, bindingIndex: 3)
            )
            addProbe(
                label: "o-proj-out",
                stepIndex: oStep,
                bindingIndex: 2,
                phase: .afterStep,
                rowStride: uint32Value(stepIndex: oStep, bindingIndex: 4),
                count: uint32Value(stepIndex: oStep, bindingIndex: 4)
            )
        }

        return probes
    }
#endif
}
