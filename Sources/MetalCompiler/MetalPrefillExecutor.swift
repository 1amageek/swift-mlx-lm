import Foundation
import Metal

private let prefillSplitCopyNormPassesEnabled =
    ProcessInfo.processInfo.environment["SWIFTLM_METAL_SPLIT_COPY_NORM_PASSES"] == "1"

func prefillHiddenOverrideReplayStartStepIndex(kernelNames: [String]) -> Int {
    kernelNames.prefix { kernelName in
        kernelName.hasPrefix("embedding_lookup_seq")
            || kernelName == "embedding_lookup"
            || kernelName.hasPrefix("embedding_lookup_")
    }.count
}

func prefillHiddenOverrideReplayStartStepIndex(in steps: [MetalPrefillStep]) -> Int {
    prefillHiddenOverrideReplayStartStepIndex(
        kernelNames: steps.map { $0.metadata.kernelName ?? $0.pipeline.label ?? "" }
    )
}

func prefillPassRanges(
    for steps: [MetalPrefillStep],
    within range: Range<Int>
) -> [Range<Int>] {
    guard !range.isEmpty else { return [] }
    var ranges: [Range<Int>] = []
    var lowerBound = range.lowerBound
    for index in (range.lowerBound + 1)..<range.upperBound {
        if shouldStartNewPrefillPass(before: index, steps: steps) {
            ranges.append(lowerBound..<index)
            lowerBound = index
        }
    }
    ranges.append(lowerBound..<range.upperBound)
    return ranges
}

private func shouldStartNewPrefillPass(
    before index: Int,
    steps: [MetalPrefillStep]
) -> Bool {
    guard prefillSplitCopyNormPassesEnabled else { return false }
    guard index > 0 else { return false }
    let previousStep = steps[index - 1]
    let currentStep = steps[index]
    let previousKernel = previousStep.metadata.kernelName ?? previousStep.pipeline.label
    let currentKernel = currentStep.metadata.kernelName ?? currentStep.pipeline.label
    guard previousKernel == "copy_buffer_seq_f32",
          currentKernel == "rms_norm_seq_bf16_f32_inplace" else {
        return false
    }
    guard let previousInput = previousStep.bindings.buffers.first(where: { $0.index == 0 }),
          let previousOutput = previousStep.bindings.buffers.first(where: { $0.index == 1 }),
          let currentInput = currentStep.bindings.buffers.first(where: { $0.index == 0 }),
          let currentOutput = currentStep.bindings.buffers.first(where: { $0.index == 2 }) else {
        return false
    }
    let previousInputRegion = BufferRegion(buffer: previousInput.buffer, offset: previousInput.offset)
    let previousOutputRegion = BufferRegion(buffer: previousOutput.buffer, offset: previousOutput.offset)
    let currentInputRegion = BufferRegion(buffer: currentInput.buffer, offset: currentInput.offset)
    let currentOutputRegion = BufferRegion(buffer: currentOutput.buffer, offset: currentOutput.offset)
    return previousInputRegion == currentOutputRegion
        && previousOutputRegion == currentInputRegion
}

struct KVTransferRuntimeConstants {
    var layerCount: UInt32 = 0
    var kvHeadCount: UInt32 = 0
    var headDimension: UInt32 = 0
    var maxSequenceLength: UInt32 = 0
    var sequenceLength: UInt32 = 0
    var layoutMode: UInt32 = 0
    var sourceKScheme: UInt32 = 0
    var sourceVScheme: UInt32 = 0
    var destinationKScheme: UInt32 = 0
    var destinationVScheme: UInt32 = 0
    var sourceKHeadSlotBytes: UInt32 = 0
    var sourceVHeadSlotBytes: UInt32 = 0
    var destinationKHeadSlotBytes: UInt32 = 0
    var destinationVHeadSlotBytes: UInt32 = 0
    var numRotorGroups: UInt32 = 0
    var qjlDimension: UInt32 = 0
}

struct MetalPrefillExecutor: @unchecked Sendable {
    private let transferPlanner = MetalPrefillTransferPlanner()

    /// Pipeline for GPU-side F32→F16/BF16 hidden conversion (replaces CPU staging).
    let hiddenConversionPipeline: MTLComputePipelineState?
    /// Pipeline for post-prefill KV format conversion into the decode cache.
    let kvTransferPipeline: MTLComputePipelineState?
    /// Shared runtime constants for the post-prefill KV transfer kernel.
    let kvTransferConstantBuffer: MTLBuffer?

    init(
        hiddenConversionPipeline: MTLComputePipelineState? = nil,
        kvTransferPipeline: MTLComputePipelineState? = nil,
        kvTransferConstantBuffer: MTLBuffer? = nil
    ) {
        self.hiddenConversionPipeline = hiddenConversionPipeline
        self.kvTransferPipeline = kvTransferPipeline
        self.kvTransferConstantBuffer = kvTransferConstantBuffer
    }

#if ENABLE_METAL_PROBES
    func captureBindingProbes(
        prefillPlan: MetalPrefillPlan,
        submission: inout MetalSubmissionContext,
        position: Int,
        tokens: [Int32],
        ropePositionAxesByTokenIndex: [(UInt32, UInt32, UInt32)]? = nil,
        hiddenOverridesByTokenIndex: [Int: [Float]] = [:],
        deepstackFeaturesByLayerAndTokenIndex: [Int: [Int: [Float]]] = [:],
        probes: [MetalInferenceModel.DebugPrefillBindingProbe],
        visibilityOptions: MTL4VisibilityOptions = []
    ) throws -> [String: [Float]] {
        guard !tokens.isEmpty else { return [:] }
        guard tokens.count <= prefillPlan.maximumSequenceLength else { return [:] }
        guard !probes.isEmpty else { return [:] }

        let sequenceLength = tokens.count
        populatePrefillInputs(
            prefillPlan: prefillPlan,
            position: position,
            tokens: tokens,
            ropePositionAxesByTokenIndex: ropePositionAxesByTokenIndex
        )
        writeRuntimeConstants(
            prefillPlan: prefillPlan,
            basePosition: position,
            sequenceLength: sequenceLength,
            hiddenConversionElementCount: 0
        )

        let effectiveStepIndices = probes.map(\.stepIndex).compactMap { $0 }
        guard effectiveStepIndices.allSatisfy({ $0 >= 0 && $0 < prefillPlan.steps.count }) else {
            throw MetalCompilerError.deviceSetupFailed("Debug probe step index out of range")
        }
        let lastTokenIndex = sequenceLength - 1
        var stagingBuffers: [String: (buffer: MTLBuffer, precision: BufferPrecision, count: Int)] = [:]
        stagingBuffers.reserveCapacity(probes.count)
        for probe in probes {
            guard probe.rowStride > 0 else {
                throw MetalCompilerError.deviceSetupFailed("Debug probe row stride must be positive")
            }
            guard probe.count > 0 else {
                throw MetalCompilerError.deviceSetupFailed("Debug probe count must be positive")
            }
            let clampedCount = min(probe.count, probe.rowStride)
            let stagingLength = clampedCount * probe.precision.byteSize
            guard let stagingBuffer = submission.device.makeBuffer(length: stagingLength, options: [.storageModeShared]) else {
                throw MetalCompilerError.deviceSetupFailed("Cannot allocate debug probe staging buffer")
            }
            stagingBuffers[probe.label] = (stagingBuffer, probe.precision, clampedCount)
        }

        let hiddenOverrideReplayStart = prefillHiddenOverrideReplayStartStepIndex(in: prefillPlan.steps)
        let allTokensOverridden = hiddenOverridesByTokenIndex.count >= sequenceLength
        let layerStepIndices = firstStepIndicesByLayer(in: prefillPlan.steps)

        func encodeStepRange(
            _ range: Range<Int>,
            encoder: MTL4ComputeCommandEncoder,
            argumentTable: MTL4ArgumentTable
        ) throws {
            guard !range.isEmpty else { return }
            for currentStepIndex in range {
                let currentStep = prefillPlan.steps[currentStepIndex]
                let currentProbes = probes.filter { ($0.stepIndex ?? currentStepIndex) == currentStepIndex }
                if !currentProbes.isEmpty {
                    try MetalInferenceModel.encodePrefillProbes(
                        currentProbes.filter { $0.phase == .beforeStep },
                        step: currentStep,
                        lastTokenIndex: lastTokenIndex,
                        stagingBuffers: stagingBuffers,
                        encoder: encoder,
                        visibilityOptions: visibilityOptions
                    )
                }
                encodePrefillSteps(
                    encoder: encoder,
                    argumentTable: argumentTable,
                    prefillPlan: prefillPlan,
                    basePosition: position,
                    sequenceLength: sequenceLength,
                    range: currentStepIndex..<(currentStepIndex + 1)
                )
                if !currentProbes.isEmpty {
                    try MetalInferenceModel.encodePrefillProbes(
                        currentProbes.filter { $0.phase == .afterStep },
                        step: currentStep,
                        lastTokenIndex: lastTokenIndex,
                        stagingBuffers: stagingBuffers,
                        encoder: encoder,
                        visibilityOptions: visibilityOptions
                    )
                }
            }
        }

        if hiddenOverrideReplayStart > 0 && !allTokensOverridden {
            try submission.withCompute { encoder, argumentTable in
                try encodeStepRange(0..<hiddenOverrideReplayStart, encoder: encoder, argumentTable: argumentTable)
            }
        }

        if !hiddenOverridesByTokenIndex.isEmpty {
            try overwriteHiddenRows(
                overridesByTokenIndex: hiddenOverridesByTokenIndex,
                prefillPlan: prefillPlan,
                sequenceLength: sequenceLength
            )
        }

        var currentStepIndex = hiddenOverrideReplayStart
        for layerIndex in deepstackFeaturesByLayerAndTokenIndex.keys.sorted() {
            guard let layerStepIndex = layerStepIndices[layerIndex] else {
                throw MetalCompilerError.deviceSetupFailed(
                    "Missing prefill step range for deepstack layer \(layerIndex)"
                )
            }
            if currentStepIndex < layerStepIndex {
                try submission.withCompute { encoder, argumentTable in
                    try encodeStepRange(currentStepIndex..<layerStepIndex, encoder: encoder, argumentTable: argumentTable)
                }
            }
            try addDeepstackRows(
                featuresByTokenIndex: deepstackFeaturesByLayerAndTokenIndex[layerIndex] ?? [:],
                prefillPlan: prefillPlan,
                sequenceLength: sequenceLength
            )
            currentStepIndex = layerStepIndex
        }

        if currentStepIndex < prefillPlan.steps.count {
            try submission.withCompute { encoder, argumentTable in
                try encodeStepRange(currentStepIndex..<prefillPlan.steps.count, encoder: encoder, argumentTable: argumentTable)
            }
        }

        var snapshots: [String: [Float]] = [:]
        snapshots.reserveCapacity(stagingBuffers.count)
        for (label, staging) in stagingBuffers {
            snapshots[label] = Self.readBuffer(
                staging.buffer,
                precision: staging.precision,
                count: staging.count
            )
        }
        return snapshots
    }
#endif

    func captureLastTokenHiddenSnapshots(
        prefillPlan: MetalPrefillPlan,
        submission: inout MetalSubmissionContext,
        position: Int,
        tokens: [Int32],
        ropePositionAxesByTokenIndex: [(UInt32, UInt32, UInt32)]? = nil,
        hiddenOverridesByTokenIndex: [Int: [Float]] = [:],
        stepIndices: Set<Int>
    ) throws -> [Int: [Float]] {
        guard !tokens.isEmpty else { return [:] }
        guard tokens.count <= prefillPlan.maximumSequenceLength else { return [:] }

        let captureSteps = stepIndices.sorted().filter { $0 >= 0 && $0 < prefillPlan.steps.count }
        guard !captureSteps.isEmpty else { return [:] }

        let sequenceLength = tokens.count
        populatePrefillInputs(
            prefillPlan: prefillPlan,
            position: position,
            tokens: tokens,
            ropePositionAxesByTokenIndex: ropePositionAxesByTokenIndex
        )
        writeRuntimeConstants(
            prefillPlan: prefillPlan,
            basePosition: position,
            sequenceLength: sequenceLength,
            hiddenConversionElementCount: 0
        )

        let hiddenOverrideReplayStart = prefillHiddenOverrideReplayStartStepIndex(in: prefillPlan.steps)
        let allTokensOverridden = hiddenOverridesByTokenIndex.count >= sequenceLength

        let elementSize = max(prefillPlan.buffers.bufferPrecision.byteSize, 1)
        let hiddenCount = prefillPlan.buffers.hidden.length
            / max(prefillPlan.maximumSequenceLength, 1)
            / elementSize
        let stagingLength = hiddenCount * elementSize
        let hiddenRowStride = hiddenCount * elementSize
        let hiddenSourceOffset = max(sequenceLength - 1, 0) * hiddenRowStride
        let stagingBuffers = try captureSteps.map { _ in
            guard let buffer = submission.device.makeBuffer(length: stagingLength, options: [.storageModeShared]) else {
                throw MetalCompilerError.deviceSetupFailed("Cannot allocate debug hidden staging buffer")
            }
            return buffer
        }

        var lowerBound = 0
        if !hiddenOverridesByTokenIndex.isEmpty {
            if hiddenOverrideReplayStart > 0, !allTokensOverridden {
                try encodePrefillPasses(
                    submission: &submission,
                    prefillPlan: prefillPlan,
                    basePosition: position,
                    sequenceLength: sequenceLength,
                    range: 0..<hiddenOverrideReplayStart
                )
            }
            try overwriteHiddenRows(
                overridesByTokenIndex: hiddenOverridesByTokenIndex,
                prefillPlan: prefillPlan,
                sequenceLength: sequenceLength
            )
            lowerBound = hiddenOverrideReplayStart
        }
        for (captureIndex, stepIndex) in captureSteps.enumerated() {
            if lowerBound < stepIndex + 1 {
                try encodePrefillPasses(
                    submission: &submission,
                    prefillPlan: prefillPlan,
                    basePosition: position,
                    sequenceLength: sequenceLength,
                    range: lowerBound..<(stepIndex + 1)
                )
            }
            try submission.copyBuffers([
                (
                    from: prefillPlan.buffers.hidden,
                    sourceOffset: hiddenSourceOffset,
                    to: stagingBuffers[captureIndex],
                    destinationOffset: 0,
                    size: stagingLength
                )
            ])
            lowerBound = stepIndex + 1
        }

        var snapshots: [Int: [Float]] = [:]
        snapshots.reserveCapacity(captureSteps.count)
        for (captureIndex, stepIndex) in captureSteps.enumerated() {
            snapshots[stepIndex] = Self.readBuffer(
                stagingBuffers[captureIndex],
                precision: prefillPlan.buffers.bufferPrecision,
                count: hiddenCount
            )
        }
        return snapshots
    }

    func captureLastTokenScratchSnapshots(
        prefillPlan: MetalPrefillPlan,
        submission: inout MetalSubmissionContext,
        position: Int,
        tokens: [Int32],
        ropePositionAxesByTokenIndex: [(UInt32, UInt32, UInt32)]? = nil,
        hiddenOverridesByTokenIndex: [Int: [Float]] = [:],
        stepIndices: Set<Int>,
        slotIndex: Int,
        rowStride: Int,
        count: Int
    ) throws -> [Int: [Float]] {
        guard !tokens.isEmpty else { return [:] }
        guard tokens.count <= prefillPlan.maximumSequenceLength else { return [:] }
        guard count > 0 else { return [:] }
        guard rowStride > 0 else {
            throw MetalCompilerError.deviceSetupFailed("Debug scratch row stride must be positive")
        }

        let slotDimension = prefillPlan.slotDimension
        let scratchElementCapacity = prefillPlan.buffers.scratch.length / MemoryLayout<Float>.stride
        guard slotIndex >= 0, slotIndex * slotDimension < scratchElementCapacity else {
            throw MetalCompilerError.deviceSetupFailed("Debug scratch slot index out of range")
        }

        let captureSteps = stepIndices.sorted().filter { $0 >= 0 && $0 < prefillPlan.steps.count }
        guard !captureSteps.isEmpty else { return [:] }

        let sequenceLength = tokens.count
        populatePrefillInputs(
            prefillPlan: prefillPlan,
            position: position,
            tokens: tokens,
            ropePositionAxesByTokenIndex: ropePositionAxesByTokenIndex
        )
        writeRuntimeConstants(
            prefillPlan: prefillPlan,
            basePosition: position,
            sequenceLength: sequenceLength,
            hiddenConversionElementCount: 0
        )

        let clampedCount = min(count, rowStride)
        let stagingLength = clampedCount * MemoryLayout<Float>.stride
        let stagingBuffers = try captureSteps.map { _ in
            guard let buffer = submission.device.makeBuffer(length: stagingLength, options: [.storageModeShared]) else {
                throw MetalCompilerError.deviceSetupFailed("Cannot allocate debug scratch staging buffer")
            }
            return buffer
        }

        let hiddenOverrideReplayStart = prefillHiddenOverrideReplayStartStepIndex(in: prefillPlan.steps)
        let allTokensOverridden = hiddenOverridesByTokenIndex.count >= sequenceLength
        let sourceOffset = lastTokenScratchOffset(
            prefillPlan: prefillPlan,
            sequenceLength: sequenceLength,
            slotIndex: slotIndex,
            rowStride: rowStride
        )

        var lowerBound = 0
        if !hiddenOverridesByTokenIndex.isEmpty {
            if hiddenOverrideReplayStart > 0, !allTokensOverridden {
                try encodePrefillPasses(
                    submission: &submission,
                    prefillPlan: prefillPlan,
                    basePosition: position,
                    sequenceLength: sequenceLength,
                    range: 0..<hiddenOverrideReplayStart
                )
            }
            try overwriteHiddenRows(
                overridesByTokenIndex: hiddenOverridesByTokenIndex,
                prefillPlan: prefillPlan,
                sequenceLength: sequenceLength
            )
            lowerBound = hiddenOverrideReplayStart
        }
        for (captureIndex, stepIndex) in captureSteps.enumerated() {
            if lowerBound < stepIndex + 1 {
                try encodePrefillPasses(
                    submission: &submission,
                    prefillPlan: prefillPlan,
                    basePosition: position,
                    sequenceLength: sequenceLength,
                    range: lowerBound..<(stepIndex + 1)
                )
            }
            try submission.copyBuffers([
                (
                    from: prefillPlan.buffers.scratch,
                    sourceOffset: sourceOffset,
                    to: stagingBuffers[captureIndex],
                    destinationOffset: 0,
                    size: stagingLength
                )
            ])
            lowerBound = stepIndex + 1
        }

        var snapshots: [Int: [Float]] = [:]
        snapshots.reserveCapacity(captureSteps.count)
        for (captureIndex, stepIndex) in captureSteps.enumerated() {
            snapshots[stepIndex] = Self.readBuffer(
                stagingBuffers[captureIndex],
                precision: .float32,
                count: clampedCount
            )
        }
        return snapshots
    }

    func captureLastTokenResidualSnapshots(
        prefillPlan: MetalPrefillPlan,
        submission: inout MetalSubmissionContext,
        position: Int,
        tokens: [Int32],
        ropePositionAxesByTokenIndex: [(UInt32, UInt32, UInt32)]? = nil,
        hiddenOverridesByTokenIndex: [Int: [Float]] = [:],
        stepIndices: Set<Int>
    ) throws -> [Int: [Float]] {
        guard !tokens.isEmpty else { return [:] }
        guard tokens.count <= prefillPlan.maximumSequenceLength else { return [:] }

        let captureSteps = stepIndices.sorted().filter { $0 >= 0 && $0 < prefillPlan.steps.count }
        guard !captureSteps.isEmpty else { return [:] }

        let sequenceLength = tokens.count
        populatePrefillInputs(
            prefillPlan: prefillPlan,
            position: position,
            tokens: tokens,
            ropePositionAxesByTokenIndex: ropePositionAxesByTokenIndex
        )
        writeRuntimeConstants(
            prefillPlan: prefillPlan,
            basePosition: position,
            sequenceLength: sequenceLength,
            hiddenConversionElementCount: 0
        )

        let elementSize = max(prefillPlan.buffers.bufferPrecision.byteSize, 1)
        let hiddenCount = prefillPlan.buffers.hidden.length
            / max(prefillPlan.maximumSequenceLength, 1)
            / elementSize
        let stagingLength = hiddenCount * elementSize
        let residualRowStride = hiddenCount * elementSize
        let residualSourceOffset = max(sequenceLength - 1, 0) * residualRowStride
        let stagingBuffers = try captureSteps.map { _ in
            guard let buffer = submission.device.makeBuffer(length: stagingLength, options: [.storageModeShared]) else {
                throw MetalCompilerError.deviceSetupFailed("Cannot allocate debug residual staging buffer")
            }
            return buffer
        }

        var lowerBound = 0
        if !hiddenOverridesByTokenIndex.isEmpty {
            let hiddenOverrideReplayStart = prefillHiddenOverrideReplayStartStepIndex(in: prefillPlan.steps)
            let allTokensOverridden = hiddenOverridesByTokenIndex.count >= sequenceLength
            if hiddenOverrideReplayStart > 0, !allTokensOverridden {
                try encodePrefillPasses(
                    submission: &submission,
                    prefillPlan: prefillPlan,
                    basePosition: position,
                    sequenceLength: sequenceLength,
                    range: 0..<hiddenOverrideReplayStart
                )
            }
            try overwriteHiddenRows(
                overridesByTokenIndex: hiddenOverridesByTokenIndex,
                prefillPlan: prefillPlan,
                sequenceLength: sequenceLength
            )
            lowerBound = hiddenOverrideReplayStart
        }
        for (captureIndex, stepIndex) in captureSteps.enumerated() {
            if lowerBound < stepIndex + 1 {
                try encodePrefillPasses(
                    submission: &submission,
                    prefillPlan: prefillPlan,
                    basePosition: position,
                    sequenceLength: sequenceLength,
                    range: lowerBound..<(stepIndex + 1)
                )
            }
            try submission.copyBuffers([
                (
                    from: prefillPlan.buffers.residual,
                    sourceOffset: residualSourceOffset,
                    to: stagingBuffers[captureIndex],
                    destinationOffset: 0,
                    size: stagingLength
                )
            ])
            lowerBound = stepIndex + 1
        }

        var snapshots: [Int: [Float]] = [:]
        snapshots.reserveCapacity(captureSteps.count)
        for (captureIndex, stepIndex) in captureSteps.enumerated() {
            snapshots[stepIndex] = Self.readBuffer(
                stagingBuffers[captureIndex],
                precision: prefillPlan.buffers.bufferPrecision,
                count: hiddenCount
            )
        }
        return snapshots
    }

    func captureLastTokenFinalHidden(
        prefillPlan: MetalPrefillPlan,
        submission: inout MetalSubmissionContext,
        position: Int,
        tokens: [Int32]
    ) throws -> [Float] {
        guard !tokens.isEmpty else { return [] }
        guard tokens.count <= prefillPlan.maximumSequenceLength else { return [] }

        let sequenceLength = tokens.count
        populatePrefillInputs(
            prefillPlan: prefillPlan,
            position: position,
            tokens: tokens,
            ropePositionAxesByTokenIndex: nil
        )
        writeRuntimeConstants(
            prefillPlan: prefillPlan,
            basePosition: position,
            sequenceLength: sequenceLength,
            hiddenConversionElementCount: 0
        )

        let elementSize = max(prefillPlan.buffers.bufferPrecision.byteSize, 1)
        let hiddenCount = prefillPlan.buffers.hidden.length
            / max(prefillPlan.maximumSequenceLength, 1)
            / elementSize
        let stagingLength = hiddenCount * elementSize
        guard let stagingBuffer = submission.device.makeBuffer(length: stagingLength, options: [.storageModeShared]) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate debug final hidden staging buffer")
        }

        try encodePrefillPasses(
            submission: &submission,
            prefillPlan: prefillPlan,
            basePosition: position,
            sequenceLength: sequenceLength
        )
        let source = prefillPlan.finalHiddenSource(sequenceLength: sequenceLength)
        try submission.copyBuffers([
            (
                from: source.buffer,
                sourceOffset: source.offset,
                to: stagingBuffer,
                destinationOffset: 0,
                size: stagingLength
            )
        ])

        return Self.readBuffer(
            stagingBuffer,
            precision: prefillPlan.buffers.bufferPrecision,
            count: hiddenCount
        )
    }

    func prefill(
        prefillPlan: MetalPrefillPlan,
        decodePlan: MetalDispatchPlan,
        submission: inout MetalSubmissionContext,
        position: inout Int,
        tokens: [Int32]
    ) -> Int32 {
        guard !tokens.isEmpty else { return -1 }
        guard tokens.count <= prefillPlan.maximumSequenceLength else { return -1 }

        let sequenceLength = tokens.count
        populatePrefillInputs(
            prefillPlan: prefillPlan,
            position: position,
            tokens: tokens,
            ropePositionAxesByTokenIndex: nil
        )

        let transferPlan = transferPlanner.makeTransferPlan(
            prefillPlan: prefillPlan,
            decodePlan: decodePlan,
            sequenceLength: sequenceLength
        )

        writeRuntimeConstants(
            prefillPlan: prefillPlan,
            basePosition: position,
            sequenceLength: sequenceLength,
            hiddenConversionElementCount: transferPlan.hiddenConversionElementCount
        )

        do {
            try encodePrefillPasses(
                submission: &submission,
                prefillPlan: prefillPlan,
                basePosition: position,
                sequenceLength: sequenceLength
            )
            try submission.withCompute { encoder, argumentTable in
                encodeHiddenConversion(
                    encoder: encoder,
                    argumentTable: argumentTable,
                    transferPlan: transferPlan,
                    prefillPlan: prefillPlan,
                    decodePlan: decodePlan
                )
                encodePostPrefillCopies(
                    encoder: encoder,
                    argumentTable: argumentTable,
                    prefillPlan: prefillPlan,
                    decodePlan: decodePlan,
                    transferPlan: transferPlan
                )
                encodeDecodeOutputHead(
                    encoder: encoder,
                    argumentTable: argumentTable,
                    decodePlan: decodePlan
                )
            }
        } catch {
            print("[MetalInference] PREFILL FAILED: \(error)")
            return -1
        }

        position += sequenceLength
        return decodePlan.buffers.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
    }

    func prefill(
        prefillPlan: MetalPrefillPlan,
        decodePlan: MetalDispatchPlan,
        submission: inout MetalSubmissionContext,
        position: inout Int,
        tokens: [Int32],
        ropePositionAxesByTokenIndex: [(UInt32, UInt32, UInt32)],
        hiddenOverridesByTokenIndex: [Int: [Float]],
        deepstackFeaturesByLayerAndTokenIndex: [Int: [Int: [Float]]]
    ) throws -> Int32 {
        guard !tokens.isEmpty else { return -1 }
        guard tokens.count <= prefillPlan.maximumSequenceLength else { return -1 }

        let sequenceLength = tokens.count
        populatePrefillInputs(
            prefillPlan: prefillPlan,
            position: position,
            tokens: tokens,
            ropePositionAxesByTokenIndex: ropePositionAxesByTokenIndex
        )

        let layerStepIndices = firstStepIndicesByLayer(in: prefillPlan.steps)
        let hiddenOverrideReplayStart = prefillHiddenOverrideReplayStartStepIndex(in: prefillPlan.steps)

        writeRuntimeConstants(
            prefillPlan: prefillPlan,
            basePosition: position,
            sequenceLength: sequenceLength,
            hiddenConversionElementCount: 0
        )

        // Hidden override can replace the embedding lookup, but not the bootstrap copies that seed
        // residual/scratch state before the first layer. Those steps must still run after overwrite.
        let allTokensOverridden = hiddenOverridesByTokenIndex.count >= sequenceLength
        if hiddenOverrideReplayStart > 0 && !allTokensOverridden {
            try encodePrefillPasses(
                submission: &submission,
                prefillPlan: prefillPlan,
                basePosition: position,
                sequenceLength: sequenceLength,
                range: 0..<hiddenOverrideReplayStart
            )
        }

        try overwriteHiddenRows(
            overridesByTokenIndex: hiddenOverridesByTokenIndex,
            prefillPlan: prefillPlan,
            sequenceLength: sequenceLength
        )

        var currentStepIndex = hiddenOverrideReplayStart
        for layerIndex in deepstackFeaturesByLayerAndTokenIndex.keys.sorted() {
            guard let layerStepIndex = layerStepIndices[layerIndex] else {
                throw MetalCompilerError.deviceSetupFailed(
                    "Missing prefill step range for deepstack layer \(layerIndex)"
                )
            }
            if currentStepIndex < layerStepIndex {
                try encodePrefillPasses(
                    submission: &submission,
                    prefillPlan: prefillPlan,
                    basePosition: position,
                    sequenceLength: sequenceLength,
                    range: currentStepIndex..<layerStepIndex
                )
            }
            try addDeepstackRows(
                featuresByTokenIndex: deepstackFeaturesByLayerAndTokenIndex[layerIndex] ?? [:],
                prefillPlan: prefillPlan,
                sequenceLength: sequenceLength
            )
            currentStepIndex = layerStepIndex
        }

        if currentStepIndex < prefillPlan.steps.count {
            try encodePrefillPasses(
                submission: &submission,
                prefillPlan: prefillPlan,
                basePosition: position,
                sequenceLength: sequenceLength,
                range: currentStepIndex..<prefillPlan.steps.count
            )
        }

        let transferPlan = transferPlanner.makeTransferPlan(
            prefillPlan: prefillPlan,
            decodePlan: decodePlan,
            sequenceLength: sequenceLength
        )

        writeRuntimeConstants(
            prefillPlan: prefillPlan,
            basePosition: position,
            sequenceLength: sequenceLength,
            hiddenConversionElementCount: transferPlan.hiddenConversionElementCount
        )

        try submission.withCompute { encoder, argumentTable in
            encodeHiddenConversion(
                encoder: encoder,
                argumentTable: argumentTable,
                transferPlan: transferPlan,
                prefillPlan: prefillPlan,
                decodePlan: decodePlan
            )
            encodePostPrefillCopies(
                encoder: encoder,
                argumentTable: argumentTable,
                prefillPlan: prefillPlan,
                decodePlan: decodePlan,
                transferPlan: transferPlan
            )
            encodeDecodeOutputHead(
                encoder: encoder,
                argumentTable: argumentTable,
                decodePlan: decodePlan
            )
        }

        position += sequenceLength
        return decodePlan.buffers.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
    }

    // MARK: - Runtime Constant Buffer

    /// Write runtime constant values (sequenceLength, positions, hiddenConversionCount)
    /// to the shared constant buffer before GPU submission.
    private func writeRuntimeConstants(
        prefillPlan: MetalPrefillPlan,
        basePosition: Int,
        sequenceLength: Int,
        hiddenConversionElementCount: Int
    ) {
        let buffer = prefillPlan.buffers.runtimeConstantBuffer
        let pointer = buffer.contents()

        // Sequence length at offset 0
        pointer.storeBytes(of: UInt32(sequenceLength), toByteOffset: PrefillBufferSet.sequenceLengthOffset, as: UInt32.self)

        // Hidden conversion count at offset 4
        pointer.storeBytes(of: UInt32(hiddenConversionElementCount), toByteOffset: PrefillBufferSet.hiddenConversionCountOffset, as: UInt32.self)

        // Per-position absolute positions
        for positionOffset in 0..<sequenceLength {
            let absolutePosition = UInt32(basePosition + positionOffset)
            pointer.storeBytes(
                of: absolutePosition,
                toByteOffset: PrefillBufferSet.positionOffset(at: positionOffset),
                as: UInt32.self
            )
        }
    }

    // MARK: - Prefill Step Encoding (Metal 4)

    private func encodePrefillSteps(
        encoder: MTL4ComputeCommandEncoder,
        argumentTable: MTL4ArgumentTable,
        prefillPlan: MetalPrefillPlan,
        basePosition: Int,
        sequenceLength: Int,
        range: Range<Int>? = nil
    ) {
        let runtimeConstantBuffer = prefillPlan.buffers.runtimeConstantBuffer
        let steps: ArraySlice<MetalPrefillStep>
        if let range {
            steps = prefillPlan.steps[range]
        } else {
            steps = prefillPlan.steps[...]
        }
        for step in steps {
            switch step.mode {
            case .batch:
                step.bindings.bind(to: argumentTable)
                step.bindRuntimeArguments(
                    argumentTable: argumentTable,
                    runtimeConstantBuffer: runtimeConstantBuffer,
                    sequenceLengthOffset: PrefillBufferSet.sequenceLengthOffset
                )
                let gridSize = step.resolvedGridSize(sequenceLength: sequenceLength)
                step.descriptor.encode(on: encoder, argumentTable: argumentTable, gridSize: gridSize)
            case .lastToken:
                let lastPosition = sequenceLength - 1
                step.bindStaticArguments(argumentTable: argumentTable, position: lastPosition)
                step.descriptor.encode(on: encoder, argumentTable: argumentTable)
            case .perPosition:
                for positionOffset in 0..<sequenceLength {
                    step.bindStaticArguments(argumentTable: argumentTable, position: positionOffset)
                    if let positionBufferIndex = step.positionBufferIndex {
                        argumentTable.setAddress(
                            runtimeConstantBuffer.gpuAddress + UInt64(PrefillBufferSet.positionOffset(at: positionOffset)),
                            index: positionBufferIndex
                        )
                    }
                    step.descriptor.encode(on: encoder, argumentTable: argumentTable)
                }
            }
        }
    }

    // MARK: - Input Population

    private func populatePrefillInputs(
        prefillPlan: MetalPrefillPlan,
        position: Int,
        tokens: [Int32],
        ropePositionAxesByTokenIndex: [(UInt32, UInt32, UInt32)]?
    ) {
        let sequenceLength = tokens.count
        let tokenPointer = prefillPlan.buffers.tokenIDs.contents().bindMemory(to: Int32.self, capacity: sequenceLength)
        let positionPointer = prefillPlan.buffers.positions.contents().bindMemory(to: UInt32.self, capacity: sequenceLength)
        let ropeAxesPointer = prefillPlan.buffers.ropePositionAxes.contents()
            .bindMemory(to: UInt32.self, capacity: sequenceLength * 3)
        for index in 0..<sequenceLength {
            tokenPointer[index] = tokens[index]
            let absolutePosition = UInt32(position + index)
            positionPointer[index] = absolutePosition
            let axes = ropePositionAxesByTokenIndex?[index] ?? (absolutePosition, absolutePosition, absolutePosition)
            ropeAxesPointer[index * 3] = axes.0
            ropeAxesPointer[index * 3 + 1] = axes.1
            ropeAxesPointer[index * 3 + 2] = axes.2
        }
    }

    private func lastTokenScratchOffset(
        prefillPlan: MetalPrefillPlan,
        sequenceLength: Int,
        slotIndex: Int,
        rowStride: Int
    ) -> Int {
        let _ = rowStride
        let slotBase = slotIndex
            * prefillPlan.slotDimension
            * prefillPlan.maximumSequenceLength
        let lastTokenBase = slotBase + (sequenceLength - 1) * prefillPlan.slotDimension
        return lastTokenBase * MemoryLayout<Float>.stride
    }

    private func firstStepIndicesByLayer(in steps: [MetalPrefillStep]) -> [Int: Int] {
        var indices: [Int: Int] = [:]
        for (index, step) in steps.enumerated() {
            if let layerIndex = step.metadata.layerIndex, indices[layerIndex] == nil {
                indices[layerIndex] = index
            }
        }
        return indices
    }

    private func encodePrefillPasses(
        submission: inout MetalSubmissionContext,
        prefillPlan: MetalPrefillPlan,
        basePosition: Int,
        sequenceLength: Int,
        range: Range<Int>? = nil
    ) throws {
        let targetRange = range ?? (0..<prefillPlan.steps.count)
        for passRange in prefillPassRanges(for: prefillPlan.steps, within: targetRange) {
            try submission.withCompute { encoder, argumentTable in
                encodePrefillSteps(
                    encoder: encoder,
                    argumentTable: argumentTable,
                    prefillPlan: prefillPlan,
                    basePosition: basePosition,
                    sequenceLength: sequenceLength,
                    range: passRange
                )
            }
        }
    }

    // MARK: - Hidden Override (CPU)

    private func overwriteHiddenRows(
        overridesByTokenIndex: [Int: [Float]],
        prefillPlan: MetalPrefillPlan,
        sequenceLength: Int
    ) throws {
        guard !overridesByTokenIndex.isEmpty else { return }
        let hiddenStride = prefillPlan.buffers.hidden.length
            / MemoryLayout<Float>.stride
            / prefillPlan.maximumSequenceLength
        let hiddenPointer = prefillPlan.buffers.hidden.contents().bindMemory(to: Float.self, capacity: hiddenStride * sequenceLength)
        for (tokenIndex, values) in overridesByTokenIndex {
            guard tokenIndex >= 0, tokenIndex < sequenceLength else {
                throw MetalCompilerError.deviceSetupFailed("Hidden override token index out of range")
            }
            guard values.count == hiddenStride else {
                throw MetalCompilerError.deviceSetupFailed("Hidden override dimension mismatch")
            }
            (hiddenPointer + tokenIndex * hiddenStride).update(from: values, count: hiddenStride)
        }
    }

    private func addDeepstackRows(
        featuresByTokenIndex: [Int: [Float]],
        prefillPlan: MetalPrefillPlan,
        sequenceLength: Int
    ) throws {
        guard !featuresByTokenIndex.isEmpty else { return }
        let hiddenStride = prefillPlan.buffers.hidden.length
            / MemoryLayout<Float>.stride
            / prefillPlan.maximumSequenceLength
        let hiddenPointer = prefillPlan.buffers.hidden.contents().bindMemory(to: Float.self, capacity: hiddenStride * sequenceLength)
        for (tokenIndex, values) in featuresByTokenIndex {
            guard tokenIndex >= 0, tokenIndex < sequenceLength else {
                throw MetalCompilerError.deviceSetupFailed("Deepstack token index out of range")
            }
            guard values.count == hiddenStride else {
                throw MetalCompilerError.deviceSetupFailed("Deepstack feature dimension mismatch")
            }
            let rowPointer = hiddenPointer + tokenIndex * hiddenStride
            for elementIndex in 0..<hiddenStride {
                rowPointer[elementIndex] += values[elementIndex]
            }
        }
    }

    // MARK: - Hidden Conversion (Metal 4)

    /// Encode GPU-side F32→F16/BF16 hidden conversion from prefill to decode buffer.
    private func encodeHiddenConversion(
        encoder: MTL4ComputeCommandEncoder,
        argumentTable: MTL4ArgumentTable,
        transferPlan: PostPrefillTransferPlan,
        prefillPlan: MetalPrefillPlan,
        decodePlan: MetalDispatchPlan
    ) {
        guard transferPlan.hiddenConversionElementCount > 0,
              let pipeline = hiddenConversionPipeline else { return }

        let count = transferPlan.hiddenConversionElementCount
        let runtimeConstantBuffer = prefillPlan.buffers.runtimeConstantBuffer

        encoder.barrier(
            afterEncoderStages: .dispatch,
            beforeEncoderStages: .dispatch,
            visibilityOptions: .device
        )
        argumentTable.setAddress(
            transferPlan.hiddenDestinationBuffer.gpuAddress + UInt64(transferPlan.hiddenDestinationOffset),
            index: 0
        )
        argumentTable.setAddress(
            transferPlan.hiddenSourceBuffer.gpuAddress + UInt64(transferPlan.hiddenSourceOffset),
            index: 1
        )
        argumentTable.setAddress(
            runtimeConstantBuffer.gpuAddress + UInt64(PrefillBufferSet.hiddenConversionCountOffset),
            index: 2
        )
        encoder.setArgumentTable(argumentTable)
        encoder.setComputePipelineState(pipeline)
        let threadCount = min(max(1, count), pipeline.maxTotalThreadsPerThreadgroup)
        let gridSize = MTLSize(width: (count + threadCount - 1) / threadCount, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: threadCount, height: 1, depth: 1)
        encoder.dispatchThreadgroups(
            threadgroupsPerGrid: gridSize,
            threadsPerThreadgroup: threadgroupSize
        )
    }

    // MARK: - Post-Prefill Copies (Metal 4 Unified Encoder)

    private func encodePostPrefillCopies(
        encoder: MTL4ComputeCommandEncoder,
        argumentTable: MTL4ArgumentTable,
        prefillPlan: MetalPrefillPlan,
        decodePlan: MetalDispatchPlan,
        transferPlan: PostPrefillTransferPlan
    ) {
        let hasCopies =
            transferPlan.hiddenBlitCopySize > 0
            || transferPlan.kvTransformSequenceLength > 0
            || transferPlan.kvCopySize > 0
            || transferPlan.valueCopySize > 0
            || transferPlan.qjlResidualCopySize > 0
            || transferPlan.convCopySize > 0
            || transferPlan.recurrentCopySize > 0
        if hasCopies {
            encoder.barrier(
                afterEncoderStages: .dispatch,
                beforeEncoderStages: .dispatch,
                visibilityOptions: .device
            )
        }

        // F32→F32 hidden copy (when decode precision is F32, no conversion needed)
        if transferPlan.hiddenBlitCopySize > 0 {
            encoder.copy(
                sourceBuffer: transferPlan.hiddenSourceBuffer,
                sourceOffset: transferPlan.hiddenSourceOffset,
                destinationBuffer: transferPlan.hiddenDestinationBuffer,
                destinationOffset: transferPlan.hiddenDestinationOffset,
                size: transferPlan.hiddenBlitCopySize
            )
        }
        if transferPlan.kvTransformSequenceLength > 0,
           let prefillKV = prefillPlan.buffers.kvCache,
           let decodeKV = decodePlan.buffers.kvCache,
           let pipeline = kvTransferPipeline,
           let kvTransferConstantBuffer {
            argumentTable.setAddress(prefillKV.keys.gpuAddress, index: 0)
            argumentTable.setAddress(prefillKV.values.gpuAddress, index: 1)
            argumentTable.setAddress(decodeKV.keys.gpuAddress, index: 2)
            argumentTable.setAddress(decodeKV.values.gpuAddress, index: 3)
            var constants = KVTransferRuntimeConstants(
                layerCount: UInt32(decodeKV.specification.layerCount),
                kvHeadCount: UInt32(decodeKV.specification.kvHeadCount),
                headDimension: UInt32(decodeKV.specification.headDimension),
                maxSequenceLength: UInt32(decodeKV.specification.maximumSequenceLength),
                sequenceLength: UInt32(transferPlan.kvTransformSequenceLength),
                layoutMode: UInt32(decodeKV.specification.layoutMode.rawValue),
                sourceKScheme: UInt32(prefillKV.specification.keyQuantizationScheme.rawValue),
                sourceVScheme: UInt32(prefillKV.specification.valueQuantizationScheme.rawValue),
                destinationKScheme: UInt32(decodeKV.specification.keyQuantizationScheme.rawValue),
                destinationVScheme: UInt32(decodeKV.specification.valueQuantizationScheme.rawValue),
                sourceKHeadSlotBytes: UInt32(prefillKV.specification.bytesPerHeadSlot(scheme: prefillKV.specification.keyQuantizationScheme)),
                sourceVHeadSlotBytes: UInt32(prefillKV.specification.bytesPerHeadSlot(scheme: prefillKV.specification.valueQuantizationScheme)),
                destinationKHeadSlotBytes: UInt32(decodeKV.specification.bytesPerHeadSlot(scheme: decodeKV.specification.keyQuantizationScheme)),
                destinationVHeadSlotBytes: UInt32(decodeKV.specification.bytesPerHeadSlot(scheme: decodeKV.specification.valueQuantizationScheme)),
                numRotorGroups: UInt32(decodeKV.numRotorGroups),
                qjlDimension: UInt32(decodeKV.qjlDimension)
            )
            withUnsafeBytes(of: &constants) { bytes in
                kvTransferConstantBuffer.contents().copyMemory(
                    from: bytes.baseAddress!,
                    byteCount: bytes.count
                )
            }
            argumentTable.setAddress(kvTransferConstantBuffer.gpuAddress, index: 4)
            argumentTable.setAddress((decodeKV.rotorParameters ?? decodeKV.keys).gpuAddress, index: 5)
            argumentTable.setAddress((decodeKV.qjlMatrix ?? decodeKV.keys).gpuAddress, index: 6)
            argumentTable.setAddress((decodeKV.qjlResidualK ?? decodeKV.keys).gpuAddress, index: 7)
            encoder.setArgumentTable(argumentTable)
            encoder.setComputePipelineState(pipeline)
            let threadCount = min(
                max(1, decodeKV.specification.headDimension),
                pipeline.maxTotalThreadsPerThreadgroup
            )
            let gridSize = MTLSize(
                width: max(1, decodeKV.specification.layerCount * transferPlan.kvTransformSequenceLength),
                height: 1,
                depth: 1
            )
            let threadgroupSize = MTLSize(width: threadCount, height: 1, depth: 1)
            encoder.dispatchThreadgroups(
                threadgroupsPerGrid: gridSize,
                threadsPerThreadgroup: threadgroupSize
            )
        } else if transferPlan.kvCopySize > 0,
           let prefillKV = prefillPlan.buffers.kvCache,
           let decodeKV = decodePlan.buffers.kvCache {
            encoder.copy(
                sourceBuffer: prefillKV.keys, sourceOffset: 0,
                destinationBuffer: decodeKV.keys, destinationOffset: 0,
                size: transferPlan.kvCopySize
            )
        }
        if transferPlan.valueCopySize > 0,
           let prefillKV = prefillPlan.buffers.kvCache,
           let decodeKV = decodePlan.buffers.kvCache {
            encoder.copy(
                sourceBuffer: prefillKV.values, sourceOffset: 0,
                destinationBuffer: decodeKV.values, destinationOffset: 0,
                size: transferPlan.valueCopySize
            )
        }
        if transferPlan.qjlResidualCopySize > 0,
           let prefillKV = prefillPlan.buffers.kvCache,
           let decodeKV = decodePlan.buffers.kvCache,
           let prefillQJLResidual = prefillKV.qjlResidualK,
           let decodeQJLResidual = decodeKV.qjlResidualK {
            encoder.copy(
                sourceBuffer: prefillQJLResidual, sourceOffset: 0,
                destinationBuffer: decodeQJLResidual, destinationOffset: 0,
                size: transferPlan.qjlResidualCopySize
            )
        }
        if transferPlan.convCopySize > 0,
           let prefillConvState = prefillPlan.buffers.convState,
           let decodeConvState = decodePlan.buffers.convState {
            encoder.copy(
                sourceBuffer: prefillConvState, sourceOffset: 0,
                destinationBuffer: decodeConvState, destinationOffset: 0,
                size: transferPlan.convCopySize
            )
        }
        if transferPlan.recurrentCopySize > 0,
           let prefillRecurrentState = prefillPlan.buffers.recurrentState,
           let decodeRecurrentState = decodePlan.buffers.recurrentState {
            encoder.copy(
                sourceBuffer: prefillRecurrentState, sourceOffset: 0,
                destinationBuffer: decodeRecurrentState, destinationOffset: 0,
                size: transferPlan.recurrentCopySize
            )
        }
    }

    private func encodeDecodeOutputHead(
        encoder: MTL4ComputeCommandEncoder,
        argumentTable: MTL4ArgumentTable,
        decodePlan: MetalDispatchPlan
    ) {
        encoder.barrier(
            afterEncoderStages: [.dispatch, .blit],
            beforeEncoderStages: .dispatch,
            visibilityOptions: .device
        )
        for step in decodePlan.outputHeadSteps {
            MetalDecodeEncoder.encodeStep(
                step: step,
                encoder: encoder,
                argumentTable: argumentTable,
                visibilityOptions: .device
            )
        }
    }

    private static func readBuffer(
        _ buffer: MTLBuffer,
        precision: BufferPrecision,
        count: Int
    ) -> [Float] {
        guard count > 0 else { return [] }
        switch precision {
        case .float32:
            let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: pointer, count: count))
        case .float16:
            let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
            return (0..<count).map { Float(pointer[$0]) }
        case .bfloat16:
            let pointer = buffer.contents().bindMemory(to: UInt16.self, capacity: count)
            return (0..<count).map { Float(bitPattern: UInt32(pointer[$0]) << 16) }
        }
    }
}
