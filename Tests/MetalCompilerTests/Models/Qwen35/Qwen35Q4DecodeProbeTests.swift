import Foundation
import Metal
import Testing
@testable import MetalCompiler

#if ENABLE_METAL_PROBES
@Suite("Qwen35 Prompt Ingestion", .serialized)
struct Qwen35PromptIngestionTests {

    @Test("BF16 prefill ingestion matches decode-equivalent ingestion")
    func bf16PrefillIngestionMatchesDecodeEquivalentTrace() throws {
        guard let bundlePath = try Self.resolveBundle(repoName: "mlx-community--Qwen3.5-0.8B-MLX-bf16") else {
            Issue.record("BF16 bundle not cached. Expected ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-MLX-bf16")
            return
        }

        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens = Self.japanPromptTokens
        try autoreleasepool {
            let (model, _, _) = try BenchmarkSupport.setupFromBundle(
                bundlePath: bundlePath,
                inferencePolicy: InferencePolicy(maximumSequenceLength: 64)
            )

            let prefillPlan = try #require(model.prefillPlan)
            #expect(!prefillPlan.requiresSequentialPromptIngestion)
            #expect(prefillPlan.sequencePrefillFallbackReason == nil)

            var prefillModel = try Self.makeRuntimeIsolatedModel(from: model)
            prefillModel.resetState()
            let prefillTrace = BenchmarkSupport.decodeTokenTrace(
                model: &prefillModel,
                promptTokens: promptTokens,
                decodeSteps: 12
            )

            var sequentialModel = try Self.makeRuntimeIsolatedModel(from: model)
            sequentialModel.prefillPlan = nil
            sequentialModel.resetState()
            let sequentialTrace = BenchmarkSupport.decodeTokenTrace(
                model: &sequentialModel,
                promptTokens: promptTokens,
                decodeSteps: 12
            )

            let postPrefillDifference = try Self.firstPostPrefillStateDifference(
                model: model,
                promptTokens: promptTokens
            )
            if let postPrefillDifference {
                print("[QwenPrefillState] \(postPrefillDifference)")
                print("[QwenPrefillPlan] \(Self.prefillKernelSummary(prefillPlan))")
            }
            #expect(
                postPrefillDifference == nil,
                "BF16 post-prefill runtime state drifted; see QwenPrefillState logs"
            )

            #expect(
                prefillTrace == sequentialTrace,
                "BF16 prefill transfer must match decode-equivalent prompt ingestion. prefill=\(prefillTrace), sequential=\(sequentialTrace)"
            )
        }
    }

    @Test("Q3 prefill ingestion matches decode-equivalent ingestion")
    func q3PrefillIngestionMatchesDecodeEquivalentTrace() throws {
        guard let bundlePath = try Self.resolveBundle(repoName: "mlx-community--Qwen3.5-0.8B-3bit") else {
            Issue.record("Q3 bundle not cached. Expected ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-3bit")
            return
        }

        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens = Self.japanPromptTokens

        try autoreleasepool {
            let (model, _, _) = try BenchmarkSupport.setupFromBundle(
                bundlePath: bundlePath,
                inferencePolicy: InferencePolicy(maximumSequenceLength: 64)
            )

            let prefillPlan = try #require(model.prefillPlan)
            #expect(prefillPlan.requiresSequentialPromptIngestion)
            #expect(prefillPlan.sequencePrefillFallbackReason == .unsupportedQ3Quantization)

            var prefillModel = try Self.makeRuntimeIsolatedModel(from: model)
            prefillModel.resetState()
            let prefillTrace = BenchmarkSupport.decodeTokenTrace(
                model: &prefillModel,
                promptTokens: promptTokens,
                decodeSteps: 8
            )

            var sequentialModel = try Self.makeRuntimeIsolatedModel(from: model)
            sequentialModel.prefillPlan = nil
            sequentialModel.resetState()
            let sequentialTrace = BenchmarkSupport.decodeTokenTrace(
                model: &sequentialModel,
                promptTokens: promptTokens,
                decodeSteps: 8
            )

            #expect(
                Array(prefillTrace.prefix(9)) == Array(sequentialTrace.prefix(9)),
                "Q3 prompt ingestion must stay decode-equivalent. prefill=\(prefillTrace), sequential=\(sequentialTrace)"
            )
        }
    }

    private static let japanPromptTokens: [Int32] = [
        248045, 846, 198, 3710, 369, 279, 6511, 314, 6124, 30,
        248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
    ]

    private static func resolveBundle(repoName: String) throws -> String? {
        let hubRoot = NSString(string: "~/.cache/huggingface/hub").expandingTildeInPath
        let snapshotsDir = "\(hubRoot)/models--\(repoName)/snapshots"
        guard FileManager.default.fileExists(atPath: snapshotsDir) else { return nil }
        let entries = try FileManager.default.contentsOfDirectory(atPath: snapshotsDir).sorted()
        for entry in entries {
            let candidate = "\(snapshotsDir)/\(entry)"
            if FileManager.default.fileExists(atPath: "\(candidate)/config.json") {
                return candidate
            }
        }
        return nil
    }

    private static func makeRuntimeIsolatedModel(from model: MetalInferenceModel) throws -> MetalInferenceModel {
        let isolated = try model.compiledModel.makeRuntimeIsolatedCopy(device: model.device)
        return try MetalInferenceModel(compiledModel: isolated, device: model.device)
    }

    private static func prefillKernelSummary(_ prefillPlan: MetalPrefillPlan) -> String {
        prefillPlan.steps.enumerated()
            .compactMap { index, step -> String? in
                let name = step.metadata.kernelName ?? step.pipeline.label ?? "(unknown)"
                guard name.contains("gemv")
                    || name.contains("gemm")
                    || name.contains("round")
                    || name.contains("rope")
                    || name.contains("attn")
                    || name.contains("kv_cache")
                    || name.contains("sigmoid")
                    || name.contains("residual")
                    || name.contains("ssm_recurrence")
                else {
                    return nil
                }
                let layer = step.metadata.layerIndex.map { "L\($0)" } ?? "L-"
                let entry = step.metadata.entryIndex.map { "E\($0)" } ?? "E-"
                return "\(index):\(layer):\(entry):\(step.mode):\(step.sequenceLengthPolicy):\(name)"
            }
            .prefix(160)
            .joined(separator: ", ")
    }

    private static func firstPostPrefillStateDifference(
        model: MetalInferenceModel,
        promptTokens: [Int32]
    ) throws -> String? {
        var sequenceDebugModel = try makeRuntimeIsolatedModel(from: model)
        sequenceDebugModel.resetState()
        let sequenceFinalHiddenBeforeTransfer = try sequenceDebugModel.debugPrefillLastTokenFinalHidden(tokens: promptTokens)

        var sequenceModel = try makeRuntimeIsolatedModel(from: model)
        sequenceModel.resetState()
        let sequenceFirstToken = sequenceModel.prefill(tokens: promptTokens)
        let sequenceState = try capturePostPrefillState(model: sequenceModel, tokenCount: promptTokens.count)

        var sequentialModel = try makeRuntimeIsolatedModel(from: model)
        sequentialModel.prefillPlan = nil
        sequentialModel.resetState()
        let sequentialFirstToken = sequentialModel.prefill(tokens: promptTokens)
        let sequentialState = try capturePostPrefillState(model: sequentialModel, tokenCount: promptTokens.count)

        print("[QwenPrefillState] firstToken sequence=\(sequenceFirstToken) sequential=\(sequentialFirstToken)")
        print("[QwenPrefillState] position sequence=\(sequenceState.position) sequential=\(sequentialState.position)")
        if let prefillPlan = sequenceModel.prefillPlan {
            let finalHiddenSource = prefillPlan.finalHiddenSource(sequenceLength: promptTokens.count)
            let outputHeadBinding = sequenceModel.compiledModel.decodePlan.outputHeadInputBinding()
            print(
                "[QwenPrefillState] finalHiddenSource=\(bufferKind(finalHiddenSource.buffer, prefillPlan: prefillPlan, decodePlan: sequenceModel.compiledModel.decodePlan)) offset=\(finalHiddenSource.offset) rowStride=\(prefillPlan.finalHiddenRowStride)"
            )
            print(
                "[QwenPrefillState] outputHeadInputBinding=\(bufferKind(outputHeadBinding.buffer, prefillPlan: prefillPlan, decodePlan: sequenceModel.compiledModel.decodePlan)) offset=\(outputHeadBinding.offset)"
            )
            if let difference = firstFloatDifference(
                name: "finalHiddenBeforeTransfer",
                lhs: sequenceFinalHiddenBeforeTransfer,
                rhs: sequentialState.outputHeadInput,
                tolerance: 0.0001
            ) {
                print("[QwenPrefillState] \(difference)")
            }
        }
        if let prefillPlan = sequenceModel.prefillPlan {
            let projectedRows = try sequenceModel.debugPrefillScratchRows(
                tokens: promptTokens,
                stepIndex: min(6, prefillPlan.steps.count - 1),
                slotIndex: 1,
                rowStride: sequenceState.convStateDimension,
                rowIndices: [max(0, promptTokens.count - sequenceState.convStateKernelSize)],
                count: min(8, sequenceState.convStateDimension)
            )
            for (rowIndex, values) in projectedRows.sorted(by: { $0.key < $1.key }) {
                let bits = values.map { String(format: "0x%04x", BFloat16($0).bitPattern) }
                print("[QwenPrefillState] projectedQKV row=\(rowIndex) firstValues=\(values) bf16Bits=\(bits)")
            }
            if let qkStepIndex = prefillPlan.steps.firstIndex(where: {
                ($0.metadata.kernelName ?? $0.pipeline.label ?? "").contains("batched_qk_rms_norm_rope")
            }) {
                let keyRows = try sequenceModel.debugPrefillScratchRows(
                    tokens: promptTokens,
                    stepIndex: qkStepIndex,
                    slotIndex: 2,
                    rowStride: prefillPlan.slotDimension,
                    rowIndices: [0, max(0, promptTokens.count - 1)],
                    count: 8
                )
                for (rowIndex, values) in keyRows.sorted(by: { $0.key < $1.key }) {
                    print("[QwenPrefillState] qkNormKey row=\(rowIndex) firstValues=\(values)")
                }
            }
            var snapshotModel = try makeRuntimeIsolatedModel(from: model)
            if let keySnapshot = try snapshotModel.debugPrefillKVCacheLayerSnapshot(
                tokens: promptTokens,
                layerIndex: 0,
                kind: .keys
            ) {
                print("[QwenPrefillState] prefillKVLayer0Key0=\(firstSnapshotValue(keySnapshot)) scheme=\(keySnapshot.scheme)")
            }
        }

        if let entryDifference = try firstEntryOutputDifference(
            model: model,
            promptTokens: promptTokens,
            count: 128,
            tolerance: 0.05
        ) {
            print("[QwenPrefillState] \(entryDifference)")
        }
        if sequenceState.position != sequentialState.position {
            return "post-prefill position mismatch: sequence=\(sequenceState.position), sequential=\(sequentialState.position)"
        }
        for layerIndex in 0..<min(sequenceState.keyLayers.count, sequentialState.keyLayers.count) {
            if let difference = firstFloatDifference(
                name: "kvCache.keys[\(layerIndex)]",
                lhs: sequenceState.keyLayers[layerIndex],
                rhs: sequentialState.keyLayers[layerIndex],
                tolerance: 0.015
            ) {
                return difference
            }
            if let difference = firstFloatDifference(
                name: "kvCache.values[\(layerIndex)]",
                lhs: sequenceState.valueLayers[layerIndex],
                rhs: sequentialState.valueLayers[layerIndex],
                tolerance: 0.015
            ) {
                return difference
            }
        }
        if sequenceState.keyLayers.count != sequentialState.keyLayers.count {
            return "kvCache key layer count mismatch: sequence=\(sequenceState.keyLayers.count), sequential=\(sequentialState.keyLayers.count)"
        }
        if sequenceState.valueLayers.count != sequentialState.valueLayers.count {
            return "kvCache value layer count mismatch: sequence=\(sequenceState.valueLayers.count), sequential=\(sequentialState.valueLayers.count)"
        }
        if let difference = firstByteDifference(
            name: "convState",
            lhs: sequenceState.convStateBytes,
            rhs: sequentialState.convStateBytes,
            convStateDimension: sequenceState.convStateDimension,
            convStateKernelSize: sequenceState.convStateKernelSize
        ) {
            return difference
        }
        if let difference = firstFloatDifference(
            name: "recurrentState",
            lhs: sequenceState.recurrentStateFloats,
            rhs: sequentialState.recurrentStateFloats,
            tolerance: 0.0001
        ) {
            return difference
        }
        if let difference = firstFloatDifference(
            name: "outputHeadInput",
            lhs: sequenceState.outputHeadInput,
            rhs: sequentialState.outputHeadInput,
            tolerance: 0.0001
        ) {
            return difference
        }
        if let difference = firstFloatDifference(
            name: "logits",
            lhs: sequenceState.logits,
            rhs: sequentialState.logits,
            tolerance: 0.05
        ) {
            return difference
        }
        if let difference = firstFloatDifference(
            name: "hidden",
            lhs: sequenceState.hidden,
            rhs: sequentialState.hidden,
            tolerance: 0.015
        ) {
            print("[QwenPrefillState] non-authoritative decode hidden differs after prefill: \(difference)")
        }
        if sequenceFirstToken != sequentialFirstToken {
            return "post-prefill first token mismatch: sequence=\(sequenceFirstToken), sequential=\(sequentialFirstToken)"
        }
        return nil
    }

    private static func bufferKind(
        _ buffer: MTLBuffer,
        prefillPlan: MetalPrefillPlan,
        decodePlan: MetalDispatchPlan
    ) -> String {
        if buffer === prefillPlan.buffers.hidden { return "prefill.hidden" }
        if buffer === prefillPlan.buffers.residual { return "prefill.residual" }
        if buffer === prefillPlan.buffers.scratch { return "prefill.scratch" }
        if buffer === prefillPlan.buffers.logits { return "prefill.logits" }
        if buffer === decodePlan.buffers.hidden { return "decode.hidden" }
        if buffer === decodePlan.buffers.residual { return "decode.residual" }
        if buffer === decodePlan.buffers.scratch { return "decode.scratch" }
        if buffer === decodePlan.buffers.logits { return "decode.logits" }
        return "unknown"
    }

    private static func firstSnapshotValue(_ snapshot: MetalInferenceModel.DebugKVCacheLayerSnapshot) -> Float {
        switch snapshot.scheme {
        case .bf16RowMajor:
            let bits = UInt16(snapshot.bytes[0]) | (UInt16(snapshot.bytes[1]) << 8)
            return Float(bitPattern: UInt32(bits) << 16)
        case .fp16RowMajor:
            let bits = UInt16(snapshot.bytes[0]) | (UInt16(snapshot.bytes[1]) << 8)
            return Float(Float16(bitPattern: bits))
        case .fp32RowMajor:
            let bits = UInt32(snapshot.bytes[0])
                | (UInt32(snapshot.bytes[1]) << 8)
                | (UInt32(snapshot.bytes[2]) << 16)
                | (UInt32(snapshot.bytes[3]) << 24)
            return Float(bitPattern: bits)
        default:
            return .nan
        }
    }

    private struct ActivationProbeKey: Hashable, CustomStringConvertible {
        enum BufferKind: String {
            case hidden
            case residual
            case scratch
        }

        let entryIndex: Int
        let layerIndex: Int?
        let bufferKind: BufferKind
        let slotIndex: Int

        var description: String {
            let layer = layerIndex.map { "L\($0)" } ?? "L-"
            return "\(layer):E\(entryIndex):\(bufferKind.rawValue)[\(slotIndex)]"
        }
    }

    private struct ActivationProbeRequest {
        let key: ActivationProbeKey
        let stepIndex: Int
        let bindingIndex: Int
        let rowStride: Int
        let logicalCount: Int
        let precision: BufferPrecision
        let kernelName: String
    }

    private static func firstEntryOutputDifference(
        model: MetalInferenceModel,
        promptTokens: [Int32],
        count: Int,
        tolerance: Float
    ) throws -> String? {
        var sequenceModel = try makeRuntimeIsolatedModel(from: model)
        var sequentialModel = try makeRuntimeIsolatedModel(from: model)
        sequentialModel.prefillPlan = nil

        let prefillPlan = try #require(sequenceModel.prefillPlan)
        let targetRowIndex = 0
        guard promptTokens.indices.contains(targetRowIndex) else {
            return nil
        }
        let finalToken = promptTokens[targetRowIndex]
        print("[QwenPrefillState] decodeBufferPrecision=\(sequentialModel.buffers.bufferPrecision)")
        let prefixTokens = Array(promptTokens.prefix(targetRowIndex))

        let prefillRequests = activationProbeRequests(
            steps: prefillPlan.steps,
            hiddenBuffer: prefillPlan.buffers.hidden,
            residualBuffer: prefillPlan.buffers.residual,
            scratchBuffer: prefillPlan.buffers.scratch,
            hiddenStride: prefillPlan.buffers.hidden.length
                / prefillPlan.maximumSequenceLength
                / MemoryLayout<Float>.stride,
            scratchStride: prefillPlan.slotDimension,
            scratchSlotByteStride: prefillPlan.maximumSequenceLength
                * prefillPlan.slotDimension
                * MemoryLayout<Float>.stride,
            precision: .float32
        )
        let decodeScratchSlotByteStride = Self.scratchSlotByteStride(
            steps: sequentialModel.decodePlan.steps,
            scratchBuffer: sequentialModel.buffers.scratch,
            fallback: prefillPlan.slotDimension * sequentialModel.buffers.bufferPrecision.byteSize
        )
        let decodeScratchStride = decodeScratchSlotByteStride
            / max(sequentialModel.buffers.bufferPrecision.byteSize, 1)
        let decodeRequests = activationProbeRequests(
            steps: sequentialModel.decodePlan.steps,
            hiddenBuffer: sequentialModel.buffers.hidden,
            residualBuffer: sequentialModel.buffers.residual,
            scratchBuffer: sequentialModel.buffers.scratch,
            hiddenStride: sequentialModel.buffers.hidden.length
                / max(sequentialModel.buffers.bufferPrecision.byteSize, 1),
            scratchStride: decodeScratchStride,
            scratchSlotByteStride: decodeScratchSlotByteStride,
            precision: sequentialModel.buffers.bufferPrecision
        )

        let sharedKeys = Set(prefillRequests.map(\.key))
            .intersection(Set(decodeRequests.map(\.key)))
            .filter { $0.entryIndex >= 3 }
            .sorted { lhs, rhs in
                if lhs.entryIndex != rhs.entryIndex { return lhs.entryIndex < rhs.entryIndex }
                if lhs.layerIndex != rhs.layerIndex { return (lhs.layerIndex ?? -1) < (rhs.layerIndex ?? -1) }
                if lhs.bufferKind.rawValue != rhs.bufferKind.rawValue {
                    return lhs.bufferKind.rawValue < rhs.bufferKind.rawValue
                }
                return lhs.slotIndex < rhs.slotIndex
            }
        print(
            "[QwenPrefillState] entryProbe prefill=\(prefillRequests.count) decode=\(decodeRequests.count) shared=\(sharedKeys.count) lastShared=\(sharedKeys.suffix(8).map(\.description))"
        )
        guard !sharedKeys.isEmpty else { return nil }

        let prefillByKey = Dictionary(uniqueKeysWithValues: prefillRequests.map { ($0.key, $0) })
        let decodeByKey = Dictionary(uniqueKeysWithValues: decodeRequests.map { ($0.key, $0) })

        var prefillProbes: [MetalInferenceModel.DebugPrefillBindingProbe] = []
        var decodeProbes: [MetalInferenceModel.DebugDecodeBindingProbe] = []
        var prefillLabelsByKey: [ActivationProbeKey: String] = [:]
        var decodeLabelsByKey: [ActivationProbeKey: String] = [:]
        for (index, key) in sharedKeys.enumerated() {
            guard let prefill = prefillByKey[key],
                  let decode = decodeByKey[key] else {
                continue
            }
            let prefillCount = min(count, prefill.rowStride)
            let decodeCount = min(count, decode.rowStride)
            let probeCount = min(prefillCount, decodeCount, prefill.logicalCount, decode.logicalCount)
            guard probeCount > 0 else { continue }
            let prefillLabel = "prefillEntry\(index)"
            let decodeLabel = "decodeEntry\(index)"
            prefillLabelsByKey[key] = prefillLabel
            decodeLabelsByKey[key] = decodeLabel
            prefillProbes.append(MetalInferenceModel.DebugPrefillBindingProbe(
                label: prefillLabel,
                stepIndex: prefill.stepIndex,
                bindingIndex: prefill.bindingIndex,
                phase: .afterStep,
                rowIndex: targetRowIndex,
                rowStride: prefill.rowStride,
                count: probeCount,
                precision: prefill.precision
            ))
            decodeProbes.append(MetalInferenceModel.DebugDecodeBindingProbe(
                label: decodeLabel,
                stepIndex: decode.stepIndex,
                bindingIndex: decode.bindingIndex,
                phase: .afterStep,
                count: probeCount,
                precision: decode.precision
            ))
        }

        let prefillSnapshots = try sequenceModel.debugPrefillBindingProbes(
            tokens: promptTokens,
            stepIndex: prefillPlan.steps.count - 1,
            probes: prefillProbes
        )
        let decodeSnapshots = try sequentialModel.debugDecodeBindingProbes(
            promptTokens: prefixTokens,
            tokenID: finalToken,
            probes: decodeProbes
        )

        for key in sharedKeys {
            guard let prefillLabel = prefillLabelsByKey[key],
                  let decodeLabel = decodeLabelsByKey[key],
                  let prefill = prefillSnapshots[prefillLabel],
                  let decode = decodeSnapshots[decodeLabel] else {
                continue
            }
            if let difference = firstFloatDifference(
                name: "entryOutput \(key)",
                lhs: prefill,
                rhs: decode,
                tolerance: tolerance
            ) {
                let prefillKernel = prefillByKey[key]?.kernelName ?? "(unknown)"
                let decodeKernel = decodeByKey[key]?.kernelName ?? "(unknown)"
                print("[QwenDecodePlan] \(decodeKernelSummary(sequentialModel.decodePlan, limit: 24))")
                if key.entryIndex == 3 || key.entryIndex == 6 {
                    let prefillStepIndex = prefillByKey[key]?.stepIndex ?? 0
                    let decodeStepIndex = decodeByKey[key]?.stepIndex ?? 0
                    print("[QwenPrefillState] \(bindingSummary(label: "prefill-E\(key.entryIndex)", step: prefillPlan.steps[prefillStepIndex]))")
                    print("[QwenPrefillState] \(bindingSummary(label: "decode-E\(key.entryIndex)", step: sequentialModel.decodePlan.steps[decodeStepIndex]))")
                    if key.entryIndex == 3 {
                        if let inputDifference = try firstEntryInputDifference(
                            sequenceModel: sequenceModel,
                            sequentialModel: sequentialModel,
                            promptTokens: promptTokens,
                            prefixTokens: prefixTokens,
                            finalToken: finalToken,
                            prefillStepIndex: prefillStepIndex,
                            decodeStepIndex: decodeStepIndex,
                            count: 1024,
                            tolerance: tolerance
                        ) {
                            print("[QwenPrefillState] \(inputDifference)")
                        }
                        let replayDifference = try actualE3ReplayDifference(
                            sequenceModel: sequenceModel,
                            sequentialModel: sequentialModel,
                            promptTokens: promptTokens,
                            prefixTokens: prefixTokens,
                            finalToken: finalToken,
                            prefillStepIndex: prefillStepIndex,
                            decodeStepIndex: decodeStepIndex,
                            tolerance: tolerance
                        )
                        if let replayDifference {
                            print("[QwenPrefillState] \(replayDifference)")
                        }
                        if replayDifference == "actualE3Replay matched decode kernel for actual input and weights" {
                            print("[QwenPrefillState] skipping non-reproducible E3 probe mismatch")
                            continue
                        }
                    }
                } else if key.entryIndex == 4 {
                    let prefillStepIndex = prefillByKey[key]?.stepIndex ?? 0
                    let decodeStepIndex = decodeByKey[key]?.stepIndex ?? 0
                    print("[QwenPrefillState] \(bindingSummary(label: "prefill-E\(key.entryIndex)", step: prefillPlan.steps[prefillStepIndex]))")
                    print("[QwenPrefillState] \(bindingSummary(label: "decode-E\(key.entryIndex)", step: sequentialModel.decodePlan.steps[decodeStepIndex]))")
                    if let inputDifference = try firstE4InputDifference(
                        sequenceModel: sequenceModel,
                        sequentialModel: sequentialModel,
                        promptTokens: promptTokens,
                        prefixTokens: prefixTokens,
                        finalToken: finalToken,
                        prefillStepIndex: prefillStepIndex,
                        decodeStepIndex: decodeStepIndex,
                        tolerance: 0.0001
                    ) {
                        print("[QwenPrefillState] \(inputDifference)")
                    }
                    let e3PrefillStepIndex = prefillRequests.first { $0.key.entryIndex == 3 && $0.key.slotIndex == 1 }?.stepIndex
                    let e3DecodeStepIndex = decodeRequests.first { $0.key.entryIndex == 3 && $0.key.slotIndex == 1 }?.stepIndex
                    if let e3PrefillStepIndex, let e3DecodeStepIndex,
                       let traceDifference = try firstE3TraceDifference(
                           sequenceModel: sequenceModel,
                           sequentialModel: sequentialModel,
                           promptTokens: promptTokens,
                           prefillStepIndex: e3PrefillStepIndex,
                           decodeStepIndex: e3DecodeStepIndex,
                           tolerance: 0.0001
                       ) {
                        print("[QwenPrefillState] \(traceDifference)")
                    }
                }
                return "\(difference), prefillKernel=\(prefillKernel), decodeKernel=\(decodeKernel)"
            }
        }
        print("[QwenPrefillState] entryProbe outputs matched for \(sharedKeys.count) shared activation regions")
        return nil
    }

    private static func firstEntryInputDifference(
        sequenceModel: MetalInferenceModel,
        sequentialModel: MetalInferenceModel,
        promptTokens: [Int32],
        prefixTokens: [Int32],
        finalToken: Int32,
        prefillStepIndex: Int,
        decodeStepIndex: Int,
        count: Int,
        tolerance: Float
    ) throws -> String? {
        var sequenceModel = sequenceModel
        var sequentialModel = sequentialModel
        let prefillSnapshots = try sequenceModel.debugPrefillBindingProbes(
            tokens: promptTokens,
            stepIndex: prefillStepIndex,
            probes: [
                MetalInferenceModel.DebugPrefillBindingProbe(
                    label: "prefillInput",
                    stepIndex: prefillStepIndex,
                    bindingIndex: 0,
                    phase: .beforeStep,
                    rowIndex: promptTokens.count - 1,
                    rowStride: 1024,
                    count: count,
                    precision: .float32
                ),
            ]
        )
        let decodeSnapshots = try sequentialModel.debugDecodeBindingProbes(
            promptTokens: prefixTokens,
            tokenID: finalToken,
            probes: [
                MetalInferenceModel.DebugDecodeBindingProbe(
                    label: "decodeInput",
                    stepIndex: decodeStepIndex,
                    bindingIndex: 0,
                    phase: .beforeStep,
                    count: count,
                    precision: sequentialModel.buffers.bufferPrecision
                ),
            ]
        )
        guard let prefill = prefillSnapshots["prefillInput"],
              let decode = decodeSnapshots["decodeInput"] else {
            return "entryInput E3 missing debug snapshots"
        }
        return firstFloatDifference(
            name: "entryInput E3",
            lhs: prefill,
            rhs: decode,
            tolerance: tolerance
        )
    }

    private static func firstE4InputDifference(
        sequenceModel: MetalInferenceModel,
        sequentialModel: MetalInferenceModel,
        promptTokens: [Int32],
        prefixTokens: [Int32],
        finalToken: Int32,
        prefillStepIndex: Int,
        decodeStepIndex: Int,
        tolerance: Float
    ) throws -> String? {
        var sequenceModel = sequenceModel
        var sequentialModel = sequentialModel
        let prefillPlan = try #require(sequenceModel.prefillPlan)
        let dimensions = [6144, 2048, 16, 16]
        var prefillProbes: [MetalInferenceModel.DebugPrefillBindingProbe] = []
        for rowIndex in promptTokens.indices {
            for bindingIndex in dimensions.indices {
                prefillProbes.append(MetalInferenceModel.DebugPrefillBindingProbe(
                    label: "prefill-input-r\(rowIndex)-b\(bindingIndex)",
                    stepIndex: prefillStepIndex,
                    bindingIndex: bindingIndex,
                    phase: .beforeStep,
                    rowIndex: rowIndex,
                    rowStride: prefillPlan.slotDimension,
                    count: dimensions[bindingIndex],
                    precision: .float32
                ))
            }
        }
        let prefillSnapshots = try sequenceModel.debugPrefillBindingProbes(
            tokens: promptTokens,
            stepIndex: prefillStepIndex,
            probes: prefillProbes
        )
        _ = prefixTokens
        _ = finalToken
        for rowIndex in promptTokens.indices {
            let decodeSnapshots = try sequentialModel.debugDecodeBindingProbes(
                promptTokens: Array(promptTokens.prefix(rowIndex)),
                tokenID: promptTokens[rowIndex],
                probes: dimensions.indices.map { bindingIndex in
                    MetalInferenceModel.DebugDecodeBindingProbe(
                        label: "decode-input-\(bindingIndex)",
                        stepIndex: decodeStepIndex,
                        bindingIndex: bindingIndex,
                        phase: .beforeStep,
                        count: dimensions[bindingIndex],
                        precision: sequentialModel.buffers.bufferPrecision
                    )
                }
            )
            for bindingIndex in dimensions.indices {
                guard let prefill = prefillSnapshots["prefill-input-r\(rowIndex)-b\(bindingIndex)"],
                      let decode = decodeSnapshots["decode-input-\(bindingIndex)"] else {
                    return "E4 input missing row=\(rowIndex) binding=\(bindingIndex)"
                }
                if let difference = firstFloatDifference(
                    name: "E4 input row=\(rowIndex) binding=\(bindingIndex)",
                    lhs: prefill,
                    rhs: decode,
                    tolerance: tolerance
                ) {
                    return difference
                }
            }
        }
        return "E4 inputs matched decode"
    }

    private static func firstE3TraceDifference(
        sequenceModel: MetalInferenceModel,
        sequentialModel: MetalInferenceModel,
        promptTokens: [Int32],
        prefillStepIndex: Int,
        decodeStepIndex: Int,
        tolerance: Float
    ) throws -> String? {
        var sequenceModel = sequenceModel
        var sequentialModel = sequentialModel
        let outputDimensions = [6144, 2048, 16, 16]
        let prefillPlan = try #require(sequenceModel.prefillPlan)
        var prefillProbes: [MetalInferenceModel.DebugPrefillBindingProbe] = []
        for rowIndex in promptTokens.indices {
            for projectionIndex in outputDimensions.indices {
                prefillProbes.append(MetalInferenceModel.DebugPrefillBindingProbe(
                    label: "prefill-r\(rowIndex)-p\(projectionIndex)",
                    stepIndex: prefillStepIndex,
                    bindingIndex: 5 + projectionIndex,
                    phase: .afterStep,
                    rowIndex: rowIndex,
                    rowStride: prefillPlan.slotDimension,
                    count: outputDimensions[projectionIndex],
                    precision: .float32
                ))
            }
        }
        let prefillSnapshots = try sequenceModel.debugPrefillBindingProbes(
            tokens: promptTokens,
            stepIndex: prefillStepIndex,
            probes: prefillProbes
        )

        for rowIndex in promptTokens.indices {
            let prefixTokens = Array(promptTokens.prefix(rowIndex))
            let tokenID = promptTokens[rowIndex]
            let decodeSnapshots = try sequentialModel.debugDecodeBindingProbes(
                promptTokens: prefixTokens,
                tokenID: tokenID,
                probes: outputDimensions.indices.map { projectionIndex in
                    MetalInferenceModel.DebugDecodeBindingProbe(
                        label: "decode-p\(projectionIndex)",
                        stepIndex: decodeStepIndex,
                        bindingIndex: 5 + projectionIndex,
                        phase: .afterStep,
                        count: outputDimensions[projectionIndex],
                        precision: sequentialModel.buffers.bufferPrecision
                    )
                }
            )
            for projectionIndex in outputDimensions.indices {
                guard let prefill = prefillSnapshots["prefill-r\(rowIndex)-p\(projectionIndex)"],
                      let decode = decodeSnapshots["decode-p\(projectionIndex)"] else {
                    return "E3 trace missing row=\(rowIndex) projection=\(projectionIndex)"
                }
                if let difference = firstFloatDifference(
                    name: "E3 trace row=\(rowIndex) projection=\(projectionIndex)",
                    lhs: prefill,
                    rhs: decode,
                    tolerance: tolerance
                ) {
                    return difference
                }
            }
        }
        return "E3 trace matched decode for all \(promptTokens.count) rows"
    }

    private static func actualE3ReplayDifference(
        sequenceModel: MetalInferenceModel,
        sequentialModel: MetalInferenceModel,
        promptTokens: [Int32],
        prefixTokens: [Int32],
        finalToken: Int32,
        prefillStepIndex: Int,
        decodeStepIndex: Int,
        tolerance: Float
    ) throws -> String? {
        var sequenceModel = sequenceModel
        let sequentialModel = sequentialModel
        let inputSnapshots = try sequenceModel.debugPrefillBindingProbes(
            tokens: promptTokens,
            stepIndex: prefillStepIndex,
            probes: [
                MetalInferenceModel.DebugPrefillBindingProbe(
                    label: "input",
                    stepIndex: prefillStepIndex,
                    bindingIndex: 0,
                    phase: .beforeStep,
                    rowIndex: promptTokens.count - 1,
                    rowStride: 1024,
                    count: 1024,
                    precision: .float32
                ),
            ]
        )
        guard let input = inputSnapshots["input"] else {
            return "actualE3Replay missing input snapshot"
        }

        let prefillStep = sequenceModel.prefillPlan?.steps[prefillStepIndex]
        let decodeStep = sequentialModel.decodePlan.steps[decodeStepIndex]
        guard let prefillStep else {
            return "actualE3Replay missing prefill step"
        }

        let outputDimensions = [6144, 2048, 16, 16]
        let slotDimension = 6144
        let decodeOutputs = try runActualE3DecodeReplay(
            device: sequenceModel.device,
            step: decodeStep,
            input: input,
            outputDimensions: outputDimensions,
            slotDimension: slotDimension
        )
        let sequenceOutputs = try runActualE3SequenceReplay(
            device: sequenceModel.device,
            step: prefillStep,
            input: input,
            outputDimensions: outputDimensions,
            slotDimension: slotDimension,
            sequenceLength: promptTokens.count,
            rowIndex: promptTokens.count - 1
        )
        for projectionIndex in outputDimensions.indices {
            if let difference = firstFloatDifference(
                name: "actualE3Replay projection\(projectionIndex)",
                lhs: sequenceOutputs[projectionIndex],
                rhs: decodeOutputs[projectionIndex],
                tolerance: tolerance
            ) {
                return difference
            }
        }
        return "actualE3Replay matched decode kernel for actual input and weights"
    }

    private static func runActualE3DecodeReplay(
        device: MTLDevice,
        step: MetalDispatchStep,
        input: [Float],
        outputDimensions: [Int],
        slotDimension: Int
    ) throws -> [[Float]] {
        let queue = try #require(device.makeCommandQueue())
        let inputValues = input.map { BFloat16($0) }
        let scratch = try #require(device.makeBuffer(
            length: 5 * slotDimension * MemoryLayout<BFloat16>.stride,
            options: .storageModeShared
        ))
        memset(scratch.contents(), 0, scratch.length)
        scratch.contents()
            .bindMemory(to: BFloat16.self, capacity: 5 * slotDimension)
            .update(from: inputValues, count: inputValues.count)
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(step.pipeline)
        encoder.setBuffer(scratch, offset: 0, index: 0)
        for bindingIndex in 1...4 {
            let binding = try #require(step.bindings.buffers.first { $0.index == bindingIndex })
            encoder.setBuffer(binding.buffer, offset: binding.offset, index: bindingIndex)
        }
        for projectionIndex in 0..<4 {
            encoder.setBuffer(
                scratch,
                offset: (projectionIndex + 1) * slotDimension * MemoryLayout<BFloat16>.stride,
                index: 5 + projectionIndex
            )
        }
        var inputDimension = UInt32(input.count)
        var outputDim0 = UInt32(outputDimensions[0])
        var outputDim1 = UInt32(outputDimensions[1])
        var outputDim2 = UInt32(outputDimensions[2])
        var outputDim3 = UInt32(outputDimensions[3])
        encoder.setBytes(&inputDimension, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.setBytes(&outputDim0, length: MemoryLayout<UInt32>.stride, index: 10)
        encoder.setBytes(&outputDim1, length: MemoryLayout<UInt32>.stride, index: 11)
        encoder.setBytes(&outputDim2, length: MemoryLayout<UInt32>.stride, index: 12)
        encoder.setBytes(&outputDim3, length: MemoryLayout<UInt32>.stride, index: 13)
        dispatchBatchedProjection(
            encoder: encoder,
            pipeline: step.pipeline,
            totalOutputDimension: outputDimensions.reduce(0, +),
            sequenceLength: 1
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw MetalCompilerError.deviceSetupFailed("actual E3 decode replay failed: \(error.localizedDescription)")
        }
        let values = scratch.contents().bindMemory(to: BFloat16.self, capacity: 5 * slotDimension)
        return outputDimensions.indices.map { projectionIndex in
            let offset = (projectionIndex + 1) * slotDimension
            return (0..<outputDimensions[projectionIndex]).map { Float(values[offset + $0]) }
        }
    }

    private static func runActualE3SequenceReplay(
        device: MTLDevice,
        step: MetalPrefillStep,
        input: [Float],
        outputDimensions: [Int],
        slotDimension: Int,
        sequenceLength: Int,
        rowIndex: Int
    ) throws -> [[Float]] {
        let queue = try #require(device.makeCommandQueue())
        let scratch = try #require(device.makeBuffer(
            length: 5 * sequenceLength * slotDimension * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        memset(scratch.contents(), 0, scratch.length)
        scratch.contents()
            .bindMemory(to: Float.self, capacity: 5 * sequenceLength * slotDimension)
            .advanced(by: rowIndex * slotDimension)
            .update(from: input, count: input.count)
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(step.pipeline)
        encoder.setBuffer(scratch, offset: 0, index: 0)
        for bindingIndex in 1...4 {
            let binding = try #require(step.bindings.buffers.first { $0.index == bindingIndex })
            encoder.setBuffer(binding.buffer, offset: binding.offset, index: bindingIndex)
        }
        for projectionIndex in 0..<4 {
            encoder.setBuffer(
                scratch,
                offset: (projectionIndex + 1) * sequenceLength * slotDimension * MemoryLayout<Float>.stride,
                index: 5 + projectionIndex
            )
        }
        var inputDimension = UInt32(input.count)
        var outputDim0 = UInt32(outputDimensions[0])
        var outputDim1 = UInt32(outputDimensions[1])
        var outputDim2 = UInt32(outputDimensions[2])
        var outputDim3 = UInt32(outputDimensions[3])
        var sequenceLengthValue = UInt32(sequenceLength)
        var inputRowStride = UInt32(input.count)
        var outputRowStride = UInt32(slotDimension)
        encoder.setBytes(&inputDimension, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.setBytes(&outputDim0, length: MemoryLayout<UInt32>.stride, index: 10)
        encoder.setBytes(&outputDim1, length: MemoryLayout<UInt32>.stride, index: 11)
        encoder.setBytes(&outputDim2, length: MemoryLayout<UInt32>.stride, index: 12)
        encoder.setBytes(&outputDim3, length: MemoryLayout<UInt32>.stride, index: 13)
        encoder.setBytes(&sequenceLengthValue, length: MemoryLayout<UInt32>.stride, index: 14)
        encoder.setBytes(&inputRowStride, length: MemoryLayout<UInt32>.stride, index: 15)
        encoder.setBytes(&outputRowStride, length: MemoryLayout<UInt32>.stride, index: 16)
        dispatchBatchedProjection(
            encoder: encoder,
            pipeline: step.pipeline,
            totalOutputDimension: outputDimensions.reduce(0, +),
            sequenceLength: sequenceLength
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw MetalCompilerError.deviceSetupFailed("actual E3 sequence replay failed: \(error.localizedDescription)")
        }
        let values = scratch.contents().bindMemory(to: Float.self, capacity: 5 * sequenceLength * slotDimension)
        return outputDimensions.indices.map { projectionIndex in
            let offset = (projectionIndex + 1) * sequenceLength * slotDimension
                + rowIndex * slotDimension
            return (0..<outputDimensions[projectionIndex]).map { values[offset + $0] }
        }
    }

    private static func dispatchBatchedProjection(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        totalOutputDimension: Int,
        sequenceLength: Int
    ) {
        let simdWidth = 32
        let threads = min(simdWidth * 2, pipeline.maxTotalThreadsPerThreadgroup)
        let rowsPerThreadgroup = max(1, threads / simdWidth)
        let grid = MTLSize(
            width: (totalOutputDimension + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            height: sequenceLength,
            depth: 1
        )
        let threadgroup = MTLSize(width: threads, height: 1, depth: 1)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
    }

    private static func decodeKernelSummary(
        _ decodePlan: MetalDispatchPlan,
        limit: Int
    ) -> String {
        decodePlan.steps.enumerated()
            .prefix(limit)
            .map { index, step in
                let name = step.metadata.kernelName ?? step.pipeline.label ?? "(unknown)"
                let entry = step.metadata.entryIndex.map { "E\($0)" } ?? "E-"
                let writes = step.metadata.bufferAccessPattern?.writeIndices.sorted()
                    .map(String.init)
                    .joined(separator: "/") ?? "-"
                return "\(index):\(entry):w\(writes):\(name)"
            }
            .joined(separator: ", ")
    }

    private static func bindingSummary(label: String, step: MetalPrefillStep) -> String {
        let bindings = step.bindings.buffers
            .map { "b\($0.index)@off\($0.offset)#len\($0.buffer.length)" }
            .joined(separator: ",")
        let constants = constantSummary(step.bindings.constantBindings)
        let name = step.metadata.kernelName ?? step.pipeline.label ?? "(unknown)"
        return "\(label): \(name) mode=\(step.mode) seqPolicy=\(step.sequenceLengthPolicy) grid=\(step.gridSize.width)x\(step.gridSize.height)x\(step.gridSize.depth) strides=\(step.perPositionStrides) bindings=[\(bindings)] constants=[\(constants)]"
    }

    private static func bindingSummary(label: String, step: MetalDispatchStep) -> String {
        let bindings = step.bindings.buffers
            .map { "b\($0.index)@off\($0.offset)#len\($0.buffer.length)" }
            .joined(separator: ",")
        let constants = constantSummary(step.bindings.constantBindings)
        let name = step.metadata.kernelName ?? step.pipeline.label ?? "(unknown)"
        return "\(label): \(name) bindings=[\(bindings)] constants=[\(constants)]"
    }

    private static func constantSummary(_ constants: MetalConstantBindingSet) -> String {
        constants.bindings
            .map { binding -> String in
                switch binding {
                case .inline(let bytes):
                    let value = bytes.value.withUnsafeBytes { rawBuffer in
                        rawBuffer.loadUnaligned(as: UInt32.self)
                    }
                    return "c\(bytes.index)=\(value)"
                case .buffer(let buffer):
                    let value = buffer.buffer.contents()
                        .advanced(by: buffer.offset)
                        .loadUnaligned(as: UInt32.self)
                    return "c\(buffer.index)=\(value)"
                }
            }
            .joined(separator: ",")
    }

    private static func activationProbeRequests(
        steps: [MetalPrefillStep],
        hiddenBuffer: MTLBuffer,
        residualBuffer: MTLBuffer,
        scratchBuffer: MTLBuffer,
        hiddenStride: Int,
        scratchStride: Int,
        scratchSlotByteStride: Int,
        precision: BufferPrecision
    ) -> [ActivationProbeRequest] {
        var requestsByKey: [ActivationProbeKey: ActivationProbeRequest] = [:]
        for (stepIndex, step) in steps.enumerated() {
            guard let entryIndex = step.metadata.entryIndex,
                  let accessPattern = step.metadata.bufferAccessPattern else {
                continue
            }
            for binding in step.bindings.buffers where accessPattern.writeIndices.contains(binding.index) {
                guard let key = activationProbeKey(
                    entryIndex: entryIndex,
                    layerIndex: step.metadata.layerIndex,
                    binding: binding,
                    hiddenBuffer: hiddenBuffer,
                    residualBuffer: residualBuffer,
                    scratchBuffer: scratchBuffer,
                    scratchSlotByteStride: scratchSlotByteStride
                ) else {
                    continue
                }
                let rowStride: Int
                switch key.bufferKind {
                case .hidden, .residual:
                    rowStride = hiddenStride
                case .scratch:
                    rowStride = scratchStride
                }
                let logicalCount = activationLogicalCount(
                    step: step,
                    bindingIndex: binding.index,
                    fallback: requestsByKey[key]?.logicalCount ?? rowStride
                )
                let kernelName = step.metadata.kernelName ?? step.pipeline.label ?? "(unknown)"
                if let existing = requestsByKey[key],
                   !isStorageRoundingKernel(existing.kernelName),
                   isStorageRoundingKernel(kernelName) {
                    continue
                }
                requestsByKey[key] = ActivationProbeRequest(
                    key: key,
                    stepIndex: stepIndex,
                    bindingIndex: binding.index,
                    rowStride: rowStride,
                    logicalCount: logicalCount,
                    precision: precision,
                    kernelName: kernelName
                )
            }
        }
        return Array(requestsByKey.values)
    }

    private static func scratchSlotByteStride(
        steps: [MetalDispatchStep],
        scratchBuffer: MTLBuffer,
        fallback: Int
    ) -> Int {
        let offsets = steps
            .flatMap(\.bindings.buffers)
            .filter { $0.buffer === scratchBuffer && $0.offset > 0 }
            .map(\.offset)
            .sorted()
        guard let firstOffset = offsets.first else { return fallback }
        return firstOffset
    }

    private static func debugPrecision(kernelName: String?, fallback: BufferPrecision) -> BufferPrecision {
        guard let kernelName else { return fallback }
        if kernelName.contains("_f16_") {
            return .float16
        }
        return fallback
    }

    private static func activationProbeRequests(
        steps: [MetalDispatchStep],
        hiddenBuffer: MTLBuffer,
        residualBuffer: MTLBuffer,
        scratchBuffer: MTLBuffer,
        hiddenStride: Int,
        scratchStride: Int,
        scratchSlotByteStride: Int,
        precision: BufferPrecision
    ) -> [ActivationProbeRequest] {
        var requestsByKey: [ActivationProbeKey: ActivationProbeRequest] = [:]
        for (stepIndex, step) in steps.enumerated() {
            guard let entryIndex = step.metadata.entryIndex,
                  let accessPattern = step.metadata.bufferAccessPattern else {
                continue
            }
            for binding in step.bindings.buffers where accessPattern.writeIndices.contains(binding.index) {
                guard let key = activationProbeKey(
                    entryIndex: entryIndex,
                    layerIndex: step.metadata.layerIndex,
                    binding: binding,
                    hiddenBuffer: hiddenBuffer,
                    residualBuffer: residualBuffer,
                    scratchBuffer: scratchBuffer,
                    scratchSlotByteStride: scratchSlotByteStride
                ) else {
                    continue
                }
                let rowStride: Int
                switch key.bufferKind {
                case .hidden, .residual:
                    rowStride = hiddenStride
                case .scratch:
                    rowStride = scratchStride
                }
                let logicalCount = activationLogicalCount(
                    step: step,
                    bindingIndex: binding.index,
                    fallback: requestsByKey[key]?.logicalCount ?? rowStride
                )
                let kernelName = step.metadata.kernelName ?? step.pipeline.label ?? "(unknown)"
                requestsByKey[key] = ActivationProbeRequest(
                    key: key,
                    stepIndex: stepIndex,
                    bindingIndex: binding.index,
                    rowStride: rowStride,
                    logicalCount: logicalCount,
                    precision: debugPrecision(kernelName: kernelName, fallback: precision),
                    kernelName: kernelName
                )
            }
        }
        return Array(requestsByKey.values)
    }

    private static func isStorageRoundingKernel(_ kernelName: String) -> Bool {
        kernelName.hasPrefix("round_f16_seq_") || kernelName.hasPrefix("round_bf16_seq_")
    }

    private static func activationProbeKey(
        entryIndex: Int,
        layerIndex: Int?,
        binding: MetalBufferBinding,
        hiddenBuffer: MTLBuffer,
        residualBuffer: MTLBuffer,
        scratchBuffer: MTLBuffer,
        scratchSlotByteStride: Int
    ) -> ActivationProbeKey? {
        if binding.buffer === hiddenBuffer {
            return ActivationProbeKey(
                entryIndex: entryIndex,
                layerIndex: layerIndex,
                bufferKind: .hidden,
                slotIndex: 0
            )
        }
        if binding.buffer === residualBuffer {
            return ActivationProbeKey(
                entryIndex: entryIndex,
                layerIndex: layerIndex,
                bufferKind: .residual,
                slotIndex: 0
            )
        }
        if binding.buffer === scratchBuffer {
            guard scratchSlotByteStride > 0 else { return nil }
            return ActivationProbeKey(
                entryIndex: entryIndex,
                layerIndex: layerIndex,
                bufferKind: .scratch,
                slotIndex: binding.offset / scratchSlotByteStride
            )
        }
        return nil
    }

    private static func activationLogicalCount(
        step: MetalPrefillStep,
        bindingIndex: Int,
        fallback: Int
    ) -> Int {
        activationLogicalCount(
            kernelName: step.metadata.kernelName ?? step.pipeline.label,
            bindingIndex: bindingIndex,
            constants: step.bindings.constantBindings.inlineBindings,
            fallback: fallback
        )
    }

    private static func activationLogicalCount(
        step: MetalDispatchStep,
        bindingIndex: Int,
        fallback: Int
    ) -> Int {
        activationLogicalCount(
            kernelName: step.metadata.kernelName ?? step.pipeline.label,
            bindingIndex: bindingIndex,
            constants: step.bindings.constantBindings.inlineBindings,
            fallback: fallback
        )
    }

    private static func activationLogicalCount(
        kernelName: String?,
        bindingIndex: Int,
        constants: [MetalBytesBinding],
        fallback: Int
    ) -> Int {
        guard let kernelName else { return fallback }
        if kernelName.contains("batched_gemv"),
           let count = batchedProjectionCount(kernelName: kernelName),
           bindingIndex >= 1 + count,
           bindingIndex < 1 + 2 * count {
            let projectionIndex = bindingIndex - (1 + count)
            let dimBase = 1 + 2 * count
            return uint32Constant(constants, index: dimBase + 1 + projectionIndex) ?? fallback
        }
        if kernelName.contains("gemv"), bindingIndex == 2 {
            return uint32Constant(constants, index: 4) ?? fallback
        }
        if kernelName.contains("ssm_recurrence"),
           bindingIndex == 10,
           let headCount = uint32Constant(constants, index: 11),
           let valueDimension = uint32Constant(constants, index: 14) {
            return headCount * valueDimension
        }
        return fallback
    }

    private static func batchedProjectionCount(kernelName: String) -> Int? {
        for count in 2...4 where kernelName.contains("gemv\(count)") {
            return count
        }
        return nil
    }

    private static func uint32Constant(
        _ constants: [MetalBytesBinding],
        index: Int
    ) -> Int? {
        guard let binding = constants.first(where: { $0.index == index }),
              binding.value.count >= MemoryLayout<UInt32>.stride else {
            return nil
        }
        let value = binding.value.withUnsafeBytes { rawBuffer in
            rawBuffer.loadUnaligned(as: UInt32.self)
        }
        return Int(value)
    }

    private struct PostPrefillState {
        let position: Int
        let hidden: [Float]
        let outputHeadInput: [Float]
        let logits: [Float]
        let convStateBytes: [UInt8]?
        let convStateDimension: Int
        let convStateKernelSize: Int
        let recurrentStateFloats: [Float]?
        let keyLayers: [[Float]]
        let valueLayers: [[Float]]
    }

    private static func capturePostPrefillState(
        model: MetalInferenceModel,
        tokenCount: Int
    ) throws -> PostPrefillState {
        let buffers = model.buffers
        let recurrentStateFloats: [Float]? = if let recurrentState = buffers.recurrentState {
            try readBuffer(recurrentState, precision: .float32)
        } else {
            nil
        }

        var keyLayers: [[Float]] = []
        var valueLayers: [[Float]] = []
        if let cache = buffers.kvCache {
            for layerIndex in 0..<cache.specification.layerCount {
                keyLayers.append(try readKVLayer(cache: cache, layerIndex: layerIndex, tokenCount: tokenCount, kind: .keys))
                valueLayers.append(try readKVLayer(cache: cache, layerIndex: layerIndex, tokenCount: tokenCount, kind: .values))
            }
        }

        return PostPrefillState(
            position: model.position,
            hidden: try readBuffer(buffers.hidden, precision: buffers.bufferPrecision),
            outputHeadInput: try readOutputHeadInput(model),
            logits: try readBuffer(buffers.logits, precision: buffers.bufferPrecision),
            convStateBytes: try buffers.convState.map { try readBytes($0) },
            convStateDimension: buffers.convStateDimension,
            convStateKernelSize: buffers.convStateKernelSize,
            recurrentStateFloats: recurrentStateFloats,
            keyLayers: keyLayers,
            valueLayers: valueLayers
        )
    }

    private static func readOutputHeadInput(_ model: MetalInferenceModel) throws -> [Float] {
        let binding = model.compiledModel.decodePlan.outputHeadInputBinding()
        let precision = model.compiledModel.decodePlan.buffers.bufferPrecision
        let count = model.compiledModel.decodePlan.buffers.hidden.length / precision.byteSize
        return try readBufferSlice(
            binding.buffer,
            offset: binding.offset,
            count: count,
            precision: precision
        )
    }

    private enum KVKind {
        case keys
        case values
    }

    private static func readKVLayer(
        cache: MetalKVCache,
        layerIndex: Int,
        tokenCount: Int,
        kind: KVKind
    ) throws -> [Float] {
        let spec = cache.specification
        let scheme: QuantizationSchemeIdentifier
        let buffer: MTLBuffer
        switch kind {
        case .keys:
            scheme = spec.keyQuantizationScheme
            buffer = cache.keys
        case .values:
            scheme = spec.valueQuantizationScheme
            buffer = cache.values
        }
        let precision = try denseKVPrecision(for: scheme)
        var values: [Float] = []
        values.reserveCapacity(tokenCount * spec.kvHeadCount * spec.headDimension)
        for position in 0..<tokenCount {
            for head in 0..<spec.kvHeadCount {
                let offset = spec.offset(layer: layerIndex, head: head, position: position, scheme: scheme)
                values.append(contentsOf: try readBufferSlice(
                    buffer,
                    offset: offset,
                    count: spec.headDimension,
                    precision: precision
                ))
            }
        }
        return values
    }

    private static func denseKVPrecision(for scheme: QuantizationSchemeIdentifier) throws -> BufferPrecision {
        switch scheme {
        case .fp16RowMajor:
            return .float16
        case .bf16RowMajor:
            return .bfloat16
        case .fp32RowMajor:
            return .float32
        default:
            throw ProbeError.unsupportedKVScheme("\(scheme)")
        }
    }

    private static func readBuffer(_ buffer: MTLBuffer, precision: BufferPrecision) throws -> [Float] {
        let readableBuffer = try makeReadableBuffer(buffer)
        let count = readableBuffer.length / precision.byteSize
        switch precision {
        case .float16:
            let pointer = readableBuffer.contents().bindMemory(to: Float16.self, capacity: count)
            return (0..<count).map { Float(pointer[$0]) }
        case .bfloat16:
            let pointer = readableBuffer.contents().bindMemory(to: BFloat16.self, capacity: count)
            return (0..<count).map { Float(pointer[$0]) }
        case .float32, .float32Decode:
            let pointer = readableBuffer.contents().bindMemory(to: Float32.self, capacity: count)
            return (0..<count).map { pointer[$0] }
        }
    }

    private static func readBufferSlice(
        _ buffer: MTLBuffer,
        offset: Int,
        count: Int,
        precision: BufferPrecision
    ) throws -> [Float] {
        let byteCount = count * precision.byteSize
        let readableBuffer = try makeReadableBuffer(buffer, offset: offset, byteCount: byteCount)
        switch precision {
        case .float16:
            let pointer = readableBuffer.contents().bindMemory(to: Float16.self, capacity: count)
            return (0..<count).map { Float(pointer[$0]) }
        case .bfloat16:
            let pointer = readableBuffer.contents().bindMemory(to: BFloat16.self, capacity: count)
            return (0..<count).map { Float(pointer[$0]) }
        case .float32, .float32Decode:
            let pointer = readableBuffer.contents().bindMemory(to: Float32.self, capacity: count)
            return (0..<count).map { pointer[$0] }
        }
    }

    private static func readBytes(_ buffer: MTLBuffer) throws -> [UInt8] {
        let readableBuffer = try makeReadableBuffer(buffer)
        let pointer = readableBuffer.contents().bindMemory(to: UInt8.self, capacity: readableBuffer.length)
        return Array(UnsafeBufferPointer(start: pointer, count: readableBuffer.length))
    }

    private static func makeReadableBuffer(_ buffer: MTLBuffer) throws -> MTLBuffer {
        try makeReadableBuffer(buffer, offset: 0, byteCount: buffer.length)
    }

    private static func makeReadableBuffer(
        _ buffer: MTLBuffer,
        offset: Int,
        byteCount: Int
    ) throws -> MTLBuffer {
        guard offset >= 0, byteCount >= 0, offset + byteCount <= buffer.length else {
            throw ProbeError.invalidBufferRange
        }
        guard let staging = buffer.device.makeBuffer(length: byteCount, options: .storageModeShared),
              let queue = buffer.device.makeCommandQueue(),
              let commandBuffer = queue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw ProbeError.unavailableCommandSubmission
        }
        blit.copy(from: buffer, sourceOffset: offset, to: staging, destinationOffset: 0, size: byteCount)
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }
        return staging
    }

    private static func firstByteDifference(
        name: String,
        lhs: [UInt8]?,
        rhs: [UInt8]?,
        convStateDimension: Int? = nil,
        convStateKernelSize: Int? = nil
    ) -> String? {
        switch (lhs, rhs) {
        case (.none, .none):
            return nil
        case (.some, .none), (.none, .some):
            return "\(name) presence mismatch"
        case let (.some(lhs), .some(rhs)):
            guard lhs.count == rhs.count else {
                return "\(name) byte count mismatch: sequence=\(lhs.count), sequential=\(rhs.count)"
            }
            for index in lhs.indices where lhs[index] != rhs[index] {
                if let decoded = decodeConvStateByteIndex(
                    index,
                    lhs: lhs,
                    rhs: rhs,
                    convStateDimension: convStateDimension,
                    convStateKernelSize: convStateKernelSize
                ) {
                    return "\(name) byte mismatch at \(index) \(decoded)"
                }
                return "\(name) byte mismatch at \(index): sequence=\(lhs[index]), sequential=\(rhs[index])"
            }
            return nil
        }
    }

    private static func decodeConvStateByteIndex(
        _ byteIndex: Int,
        lhs: [UInt8],
        rhs: [UInt8],
        convStateDimension: Int?,
        convStateKernelSize: Int?
    ) -> String? {
        guard byteIndex % 2 == 0,
              byteIndex + 1 < lhs.count,
              byteIndex + 1 < rhs.count,
              let convStateDimension,
              let convStateKernelSize,
              convStateDimension > 0,
              convStateKernelSize > 0 else {
            return nil
        }
        let elementIndex = byteIndex / 2
        let elementsPerLayer = convStateDimension * convStateKernelSize
        let layer = elementIndex / elementsPerLayer
        let offsetInLayer = elementIndex % elementsPerLayer
        let kernel = offsetInLayer / convStateDimension
        let channel = offsetInLayer % convStateDimension
        let sequenceBits = UInt16(lhs[byteIndex]) | (UInt16(lhs[byteIndex + 1]) << 8)
        let sequentialBits = UInt16(rhs[byteIndex]) | (UInt16(rhs[byteIndex + 1]) << 8)
        let sequenceValue = Float(bitPattern: UInt32(sequenceBits) << 16)
        let sequentialValue = Float(bitPattern: UInt32(sequentialBits) << 16)
        return "element=\(elementIndex), layer=\(layer), kernel=\(kernel), channel=\(channel), sequenceBits=0x\(String(format: "%04x", sequenceBits)), sequentialBits=0x\(String(format: "%04x", sequentialBits)), sequenceValue=\(sequenceValue), sequentialValue=\(sequentialValue)"
    }

    private static func firstFloatDifference(
        name: String,
        lhs: [Float]?,
        rhs: [Float]?,
        tolerance: Float
    ) -> String? {
        switch (lhs, rhs) {
        case (.none, .none):
            return nil
        case (.some, .none), (.none, .some):
            return "\(name) presence mismatch"
        case let (.some(lhs), .some(rhs)):
            guard lhs.count == rhs.count else {
                return "\(name) count mismatch: sequence=\(lhs.count), sequential=\(rhs.count)"
            }
            var maxError: Float = 0
            var maxIndex = -1
            for index in lhs.indices {
                let error = abs(lhs[index] - rhs[index])
                if error > maxError {
                    maxError = error
                    maxIndex = index
                }
                if error > tolerance {
                    return "\(name) mismatch at \(index): sequence=\(lhs[index]), sequential=\(rhs[index]), maxErrorSoFar=\(maxError), maxIndex=\(maxIndex), tolerance=\(tolerance)"
                }
            }
            print("[QwenPrefillState] \(name) maxError=\(maxError) maxIndex=\(maxIndex)")
            return nil
        }
    }

    private enum ProbeError: Error {
        case unavailableCommandSubmission
        case invalidBufferRange
        case unsupportedKVScheme(String)
    }
}
#endif
