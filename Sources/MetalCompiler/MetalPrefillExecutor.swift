import Metal

struct MetalPrefillExecutor: Sendable {
    private let transferPlanner = MetalPrefillTransferPlanner()

    func prefill(
        prefillPlan: MetalPrefillPlan,
        decodePlan: MetalDispatchPlan,
        submission: MetalSubmissionContext,
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

        var transferPlan = transferPlanner.makeTransferPlan(
            prefillPlan: prefillPlan,
            decodePlan: decodePlan,
            sequenceLength: sequenceLength
        )

        do {
            if transferPlan.canEncodeInSameTransaction {
                _ = try submission.withTransaction(label: "prefill+postprocess") { transaction in
                    try transaction.withComputeEncoder { compute in
                        encodePrefillSteps(
                            encoder: compute,
                            prefillPlan: prefillPlan,
                            basePosition: position,
                            sequenceLength: sequenceLength
                        )
                    }
                    try transaction.withBlitEncoder { blit in
                        encodePostPrefillCopies(
                            blit: blit,
                            prefillPlan: prefillPlan,
                            decodePlan: decodePlan,
                            transferPlan: transferPlan
                        )
                    }
                }
                transferPlan = transferPlan.afterInlineBlit()
            } else {
                _ = try submission.withCompute(label: "prefill") { encoder in
                    encodePrefillSteps(
                        encoder: encoder,
                        prefillPlan: prefillPlan,
                        basePosition: position,
                        sequenceLength: sequenceLength
                    )
                }
            }
        } catch {
            print("[MetalInference] PREFILL FAILED: \(error)")
            return -1
        }

        stageHiddenIfNeeded(
            transferPlan: transferPlan,
            prefillPlan: prefillPlan,
            decodePlan: decodePlan
        )

        if transferPlan.needsStandaloneBlit {
            do {
                try submission.withBlit(label: "prefill.postprocess") { blit in
                    if transferPlan.shouldStageHiddenOnCPU {
                        blit.copy(
                            from: prefillPlan.buffers.scratch,
                            sourceOffset: 0,
                            to: decodePlan.buffers.hidden,
                            destinationOffset: 0,
                            size: transferPlan.hiddenCopySize
                        )
                    }
                    encodePostPrefillCopies(
                        blit: blit,
                        prefillPlan: prefillPlan,
                        decodePlan: decodePlan,
                        transferPlan: transferPlan.shouldStageHiddenOnCPU
                            ? transferPlan.withoutHiddenCopy()
                            : transferPlan
                    )
                }
            } catch {
                print("[MetalInference] Failed to copy post-prefill state: \(error)")
                return -1
            }
        }

        position += sequenceLength
        return prefillPlan.buffers.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
    }

    func prefill(
        prefillPlan: MetalPrefillPlan,
        decodePlan: MetalDispatchPlan,
        submission: MetalSubmissionContext,
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
        let embeddingPrefixEnd = firstLayerStepIndex(in: prefillPlan.steps) ?? prefillPlan.steps.count

        if embeddingPrefixEnd > 0 {
            _ = try submission.withCompute(label: "prefill.embedding") { encoder in
                encodePrefillSteps(
                    encoder: encoder,
                    prefillPlan: prefillPlan,
                    basePosition: position,
                    sequenceLength: sequenceLength,
                    range: 0..<embeddingPrefixEnd
                )
            }
        }

        try overwriteHiddenRows(
            overridesByTokenIndex: hiddenOverridesByTokenIndex,
            prefillPlan: prefillPlan,
            sequenceLength: sequenceLength
        )

        var currentStepIndex = embeddingPrefixEnd
        for layerIndex in deepstackFeaturesByLayerAndTokenIndex.keys.sorted() {
            guard let layerStepIndex = layerStepIndices[layerIndex] else {
                throw MetalCompilerError.deviceSetupFailed(
                    "Missing prefill step range for deepstack layer \(layerIndex)"
                )
            }
            if currentStepIndex < layerStepIndex {
                _ = try submission.withCompute(label: "prefill.range.\(currentStepIndex)") { encoder in
                    encodePrefillSteps(
                        encoder: encoder,
                        prefillPlan: prefillPlan,
                        basePosition: position,
                        sequenceLength: sequenceLength,
                        range: currentStepIndex..<layerStepIndex
                    )
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
            _ = try submission.withCompute(label: "prefill.tail") { encoder in
                encodePrefillSteps(
                    encoder: encoder,
                    prefillPlan: prefillPlan,
                    basePosition: position,
                    sequenceLength: sequenceLength,
                    range: currentStepIndex..<prefillPlan.steps.count
                )
            }
        }

        let transferPlan = transferPlanner.makeTransferPlan(
            prefillPlan: prefillPlan,
            decodePlan: decodePlan,
            sequenceLength: sequenceLength
        )

        stageHiddenIfNeeded(
            transferPlan: transferPlan,
            prefillPlan: prefillPlan,
            decodePlan: decodePlan
        )

        if transferPlan.needsStandaloneBlit {
            try submission.withBlit(label: "prefill.postprocess") { blit in
                if transferPlan.shouldStageHiddenOnCPU {
                    blit.copy(
                        from: prefillPlan.buffers.scratch,
                        sourceOffset: 0,
                        to: decodePlan.buffers.hidden,
                        destinationOffset: 0,
                        size: transferPlan.hiddenCopySize
                    )
                }
                encodePostPrefillCopies(
                    blit: blit,
                    prefillPlan: prefillPlan,
                    decodePlan: decodePlan,
                    transferPlan: transferPlan.shouldStageHiddenOnCPU
                        ? transferPlan.withoutHiddenCopy()
                        : transferPlan
                )
            }
        }

        position += sequenceLength
        return prefillPlan.buffers.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
    }

    private func stageHiddenIfNeeded(
        transferPlan: PostPrefillTransferPlan,
        prefillPlan: MetalPrefillPlan,
        decodePlan: MetalDispatchPlan
    ) {
        guard transferPlan.shouldStageHiddenOnCPU || transferPlan.usesSharedDecodeHidden else {
            return
        }

        let decodeElementSize = decodePlan.buffers.bufferPrecision.byteSize
        let decodeHiddenSize = decodePlan.buffers.hidden.length / decodeElementSize
        let source = (prefillPlan.buffers.hidden.contents() + transferPlan.hiddenSourceOffset)
            .bindMemory(to: Float32.self, capacity: decodeHiddenSize)

        if transferPlan.shouldStageHiddenOnCPU {
            switch decodePlan.buffers.bufferPrecision {
            case .float16:
                let staging = prefillPlan.buffers.scratch.contents()
                    .bindMemory(to: Float16.self, capacity: decodeHiddenSize)
                for index in 0..<decodeHiddenSize {
                    staging[index] = Float16(source[index])
                }
            case .bfloat16:
                let staging = prefillPlan.buffers.scratch.contents()
                    .bindMemory(to: BFloat16.self, capacity: decodeHiddenSize)
                for index in 0..<decodeHiddenSize {
                    staging[index] = BFloat16(source[index])
                }
            case .float32:
                let staging = prefillPlan.buffers.scratch.contents()
                    .bindMemory(to: Float32.self, capacity: decodeHiddenSize)
                for index in 0..<decodeHiddenSize {
                    staging[index] = source[index]
                }
            }
            return
        }

        switch decodePlan.buffers.bufferPrecision {
        case .float16:
            let destination = decodePlan.buffers.hidden.contents()
                .bindMemory(to: Float16.self, capacity: decodeHiddenSize)
            for index in 0..<decodeHiddenSize {
                destination[index] = Float16(source[index])
            }
        case .bfloat16:
            let destination = decodePlan.buffers.hidden.contents()
                .bindMemory(to: BFloat16.self, capacity: decodeHiddenSize)
            for index in 0..<decodeHiddenSize {
                destination[index] = BFloat16(source[index])
            }
        case .float32:
            let destination = decodePlan.buffers.hidden.contents()
                .bindMemory(to: Float32.self, capacity: decodeHiddenSize)
            for index in 0..<decodeHiddenSize {
                destination[index] = source[index]
            }
        }
    }

    private func encodePrefillSteps(
        encoder: MTLComputeCommandEncoder,
        prefillPlan: MetalPrefillPlan,
        basePosition: Int,
        sequenceLength: Int,
        range: Range<Int>? = nil
    ) {
        let steps: ArraySlice<MetalPrefillStep>
        if let range {
            steps = prefillPlan.steps[range]
        } else {
            steps = prefillPlan.steps[...]
        }
        for step in steps {
            switch step.mode {
            case .batch:
                encodeBatchStep(
                    encoder: encoder,
                    step: step,
                    sequenceLength: UInt32(sequenceLength)
                )
            case .lastToken:
                if step.sync == .bufferBarrier { encoder.memoryBarrier(scope: .buffers) }
                encoder.setComputePipelineState(step.pipeline)
                let lastPosition = sequenceLength - 1
                step.bindStaticArguments(encoder: encoder, position: lastPosition)
                encoder.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
            case .perPosition:
                for positionOffset in 0..<sequenceLength {
                    if step.sync == .bufferBarrier { encoder.memoryBarrier(scope: .buffers) }
                    encoder.setComputePipelineState(step.pipeline)
                    step.bindStaticArguments(encoder: encoder, position: positionOffset)
                    if let positionBufferIndex = step.positionBufferIndex {
                        var positionValue = UInt32(basePosition + positionOffset)
                        withUnsafeBytes(of: &positionValue) { bytes in
                            encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: positionBufferIndex)
                        }
                    }
                    encoder.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
                }
            }
        }
    }

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

    private func firstLayerStepIndex(in steps: [MetalPrefillStep]) -> Int? {
        steps.firstIndex { $0.metadata.layerIndex != nil }
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

    private func encodeBatchStep(
        encoder: MTLComputeCommandEncoder,
        step: MetalPrefillStep,
        sequenceLength: UInt32
    ) {
        step.bindings.bind(to: encoder)
        step.bindRuntimeArguments(encoder: encoder, sequenceLength: sequenceLength)
        let gridSize = step.resolvedGridSize(sequenceLength: Int(sequenceLength))
        step.descriptor.encode(on: encoder, gridSize: gridSize)
    }

    private func encodePostPrefillCopies(
        blit: MTLBlitCommandEncoder,
        prefillPlan: MetalPrefillPlan,
        decodePlan: MetalDispatchPlan,
        transferPlan: PostPrefillTransferPlan
    ) {
        if transferPlan.hiddenCopySize > 0 {
            blit.copy(
                from: prefillPlan.buffers.hidden,
                sourceOffset: transferPlan.hiddenSourceOffset,
                to: decodePlan.buffers.hidden,
                destinationOffset: 0,
                size: transferPlan.hiddenCopySize
            )
        }
        if transferPlan.kvCopySize > 0,
           let prefillKV = prefillPlan.buffers.kvCache,
           let decodeKV = decodePlan.buffers.kvCache {
            blit.copy(from: prefillKV.keys, sourceOffset: 0, to: decodeKV.keys, destinationOffset: 0, size: transferPlan.kvCopySize)
        }
        if transferPlan.valueCopySize > 0,
           let prefillKV = prefillPlan.buffers.kvCache,
           let decodeKV = decodePlan.buffers.kvCache {
            blit.copy(from: prefillKV.values, sourceOffset: 0, to: decodeKV.values, destinationOffset: 0, size: transferPlan.valueCopySize)
        }
        if transferPlan.convCopySize > 0,
           let prefillConvState = prefillPlan.buffers.convState,
           let decodeConvState = decodePlan.buffers.convState {
            blit.copy(from: prefillConvState, sourceOffset: 0, to: decodeConvState, destinationOffset: 0, size: transferPlan.convCopySize)
        }
        if transferPlan.recurrentCopySize > 0,
           let prefillRecurrentState = prefillPlan.buffers.recurrentState,
           let decodeRecurrentState = decodePlan.buffers.recurrentState {
            blit.copy(
                from: prefillRecurrentState,
                sourceOffset: 0,
                to: decodeRecurrentState,
                destinationOffset: 0,
                size: transferPlan.recurrentCopySize
            )
        }
    }
}
