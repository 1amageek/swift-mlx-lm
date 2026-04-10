import Metal

struct MetalBufferAllocator {
    func makeDecodeBufferAllocation(
        compileContext context: CompileContext,
        walkContext: WalkContext,
        fusedEntries: [DispatchEntry]
    ) throws -> DecodeBufferAllocation {
        let elementSize = context.decodeBufferPrecision.byteSize
        let resolvedIntermediateSize = context.resolvedIntermediateSize
        let resolvedVocabSize = context.resolvedVocabSize
        let sizingEntries = walkContext.entries
        let slotDimension = max(
            context.hiddenSize,
            resolvedIntermediateSize,
            maximumScratchProjectionDimension(in: sizingEntries)
        )
        let scratchElementCount = max(slotDimension * 5, resolvedIntermediateSize * 5)

        let gpuOnlyOptions: MTLResourceOptions = [.storageModePrivate, .hazardTrackingModeUntracked]
        let cpuAccessOptions: MTLResourceOptions = [.storageModeShared]

        let hiddenBuffer = context.device.makeBuffer(length: context.hiddenSize * elementSize, options: gpuOnlyOptions)!
        let residualBuffer = context.device.makeBuffer(length: context.hiddenSize * elementSize, options: gpuOnlyOptions)!
        let scratchBuffer = context.device.makeBuffer(length: scratchElementCount * elementSize, options: gpuOnlyOptions)!
        let logitsBuffer = context.device.makeBuffer(length: resolvedVocabSize * elementSize, options: cpuAccessOptions)!
        let positionBuffer = context.device.makeBuffer(length: 4, options: cpuAccessOptions)!
        let ropePositionAxesBuffer = context.device.makeBuffer(length: 3 * 4, options: cpuAccessOptions)!
        let tokenInputBuffer = context.device.makeBuffer(length: 4, options: cpuAccessOptions)!
        let tokenOutputBuffer = context.device.makeBuffer(length: 4, options: cpuAccessOptions)!

        let kvCache: MetalKVCache?
        if let firstSlot = walkContext.cacheSlots.first {
            let kvCachePolicy = context.inferencePolicy.kvCache
            let keyScheme = resolveKVCacheScheme(selection: kvCachePolicy.keyScheme, weightFormat: context.weightFormat)
            let valueScheme = resolveKVCacheScheme(selection: kvCachePolicy.valueScheme, weightFormat: context.weightFormat)
            kvCache = try MetalKVCache(
                device: context.device,
                specification: KVCacheSpecification(
                    keyQuantizationScheme: keyScheme,
                    valueQuantizationScheme: valueScheme,
                    layoutMode: kvCachePolicy.layoutMode,
                    layerCount: walkContext.cacheSlots.count,
                    kvHeadCount: firstSlot.kvHeadCount,
                    headDimension: firstSlot.headDimension,
                    maximumSequenceLength: context.maximumSequenceLength
                ),
                qjlDimension: kvCachePolicy.qjlDimension,
                resourceOptions: gpuOnlyOptions
            )
        } else {
            kvCache = nil
        }

        let convState = convStateRequirements(in: sizingEntries)
        let convStateElementSize = MemoryLayout<Float16>.size
        let perLayerInput = perLayerInputRequirements(in: fusedEntries)
        let statefulGPUOnlyOptions: MTLResourceOptions = [.storageModePrivate]
        let sharedStateOptions: MTLResourceOptions = [.storageModeShared]
        let convStateBuffer: MTLBuffer?
        if convState.layerCount > 0 {
            let byteCount = convState.layerCount
                * convState.kernelSize
                * convState.dimension
                * convStateElementSize
            convStateBuffer = context.device.makeBuffer(length: byteCount, options: statefulGPUOnlyOptions)
        } else {
            convStateBuffer = nil
        }

        let perLayerInputBuffer: MTLBuffer?
        if perLayerInput.layerCount > 0, perLayerInput.dimension > 0 {
            perLayerInputBuffer = context.device.makeBuffer(
                length: perLayerInput.layerCount * perLayerInput.dimension * MemoryLayout<Float>.size,
                options: sharedStateOptions
            )
            if let perLayerInputBuffer {
                memset(perLayerInputBuffer.contents(), 0, perLayerInputBuffer.length)
            }
        } else {
            perLayerInputBuffer = nil
        }

        let recurrentState = recurrentStateRequirements(in: sizingEntries)
        let recurrentStateBuffer: MTLBuffer?
        if recurrentState.layerCount > 0 {
            recurrentStateBuffer = context.device.makeBuffer(
                length: recurrentState.layerCount * recurrentState.bytesPerLayer,
                options: statefulGPUOnlyOptions
            )
        } else {
            recurrentStateBuffer = nil
        }

        let weightBuffers = context.stafWeightStore?.residencyCandidateBuffers ?? []
        return DecodeBufferAllocation(
            bufferSet: MetalBufferSet(
                bufferPrecision: context.decodeBufferPrecision,
                hidden: hiddenBuffer,
                residual: residualBuffer,
                scratch: scratchBuffer,
                weights: weightBuffers,
                kvCache: kvCache,
                convState: convStateBuffer,
                recurrentState: recurrentStateBuffer,
                convStateDimension: convState.dimension,
                convStateKernelSize: convState.kernelSize,
                recurrentStateBytesPerLayer: recurrentState.bytesPerLayer,
                perLayerInputs: perLayerInputBuffer,
                perLayerInputDimension: perLayerInput.dimension,
                perLayerInputLayerCount: perLayerInput.layerCount,
                logits: logitsBuffer,
                position: positionBuffer,
                ropePositionAxes: ropePositionAxesBuffer,
                tokenIn: tokenInputBuffer,
                tokenOut: tokenOutputBuffer
            ),
            slotDimension: slotDimension
        )
    }

    func makePrefillBufferAllocation(
        compileContext context: CompileContext,
        walkContext: WalkContext,
        fusedEntries: [DispatchEntry],
        sharedKVCache: MetalKVCache?,
        sharedConvState: MTLBuffer?,
        sharedConvStateDimension: Int,
        sharedConvStateKernelSize: Int,
        sharedRecurrentState: MTLBuffer?,
        sharedRecurrentStateBytesPerLayer: Int
    ) throws -> PrefillBufferAllocation {
        let elementSize = MemoryLayout<Float16>.size
        let f32ElementSize = MemoryLayout<Float32>.size
        let resolvedIntermediateSize = context.resolvedIntermediateSize
        let resolvedVocabSize = context.resolvedVocabSize
        let maximumSequenceLength = context.maximumSequenceLength
        let sizingEntries = walkContext.entries
        let slotDimension = max(
            context.hiddenSize,
            resolvedIntermediateSize,
            maximumScratchProjectionDimension(in: sizingEntries)
        )
        let scratchElementCount = max(slotDimension * 5, resolvedIntermediateSize * 5)
        // Hidden stays shared for vision model CPU access (overwriteHiddenRows, addDeepstackRows).
        // Other GPU-only buffers use private for GPU compression on Apple Silicon.
        let cpuGpuOptions: MTLResourceOptions = [.storageModeShared]
        let gpuOnlyOptions: MTLResourceOptions = [.storageModePrivate]

        let convStateRequirements = convStateRequirements(in: sizingEntries)
        let perLayerInputRequirements = perLayerInputRequirements(in: sizingEntries)
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
            prefillConvStateBuffer = context.device.makeBuffer(length: byteCount, options: cpuGpuOptions)
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
            let kvCachePolicy = context.inferencePolicy.kvCache
            let keyScheme = resolveKVCacheScheme(selection: kvCachePolicy.keyScheme, weightFormat: context.weightFormat)
            let valueScheme = resolveKVCacheScheme(selection: kvCachePolicy.valueScheme, weightFormat: context.weightFormat)
            prefillKVCache = try MetalKVCache(
                device: context.device,
                specification: KVCacheSpecification(
                    keyQuantizationScheme: keyScheme,
                    valueQuantizationScheme: valueScheme,
                    layoutMode: kvCachePolicy.layoutMode,
                    layerCount: walkContext.cacheSlots.count,
                    kvHeadCount: firstSlot.kvHeadCount,
                    headDimension: firstSlot.headDimension,
                    maximumSequenceLength: context.maximumSequenceLength
                ),
                qjlDimension: kvCachePolicy.qjlDimension,
                resourceOptions: gpuOnlyOptions
            )
        } else {
            prefillKVCache = nil
        }

        let recurrentStateRequirements = recurrentStateRequirements(in: sizingEntries)
        let prefillRecurrentStateBuffer: MTLBuffer?
        let resolvedRecurrentBytesPerLayer: Int
        if let sharedRecurrentState {
            prefillRecurrentStateBuffer = sharedRecurrentState
            resolvedRecurrentBytesPerLayer = sharedRecurrentStateBytesPerLayer
        } else if recurrentStateRequirements.layerCount > 0 {
            prefillRecurrentStateBuffer = context.device.makeBuffer(
                length: recurrentStateRequirements.layerCount * recurrentStateRequirements.bytesPerLayer,
                options: cpuGpuOptions
            )
            if let prefillRecurrentStateBuffer {
                memset(prefillRecurrentStateBuffer.contents(), 0, prefillRecurrentStateBuffer.length)
            }
            resolvedRecurrentBytesPerLayer = recurrentStateRequirements.bytesPerLayer
        } else {
            prefillRecurrentStateBuffer = nil
            resolvedRecurrentBytesPerLayer = 0
        }

        let bufferSet = PrefillBufferSet(
            bufferPrecision: .float32,
            hidden: context.device.makeBuffer(length: maximumSequenceLength * context.hiddenSize * f32ElementSize, options: cpuGpuOptions)!,
            residual: context.device.makeBuffer(length: maximumSequenceLength * context.hiddenSize * f32ElementSize, options: gpuOnlyOptions)!,
            scratch: context.device.makeBuffer(length: maximumSequenceLength * scratchElementCount * f32ElementSize, options: gpuOnlyOptions)!,
            weights: context.stafWeightStore?.residencyCandidateBuffers ?? [],
            kvCache: prefillKVCache,
            convState: prefillConvStateBuffer,
            recurrentState: prefillRecurrentStateBuffer,
            convStateDimension: resolvedConvDimension,
            convStateKernelSize: resolvedConvKernelSize,
            recurrentStateBytesPerLayer: resolvedRecurrentBytesPerLayer,
            perLayerInputs: {
                guard perLayerInputRequirements.layerCount > 0, perLayerInputRequirements.dimension > 0 else {
                    return nil
                }
                let buffer = context.device.makeBuffer(
                    length: perLayerInputRequirements.layerCount
                        * maximumSequenceLength
                        * perLayerInputRequirements.dimension
                        * MemoryLayout<Float>.size,
                    options: cpuGpuOptions
                )
                if let buffer {
                    memset(buffer.contents(), 0, buffer.length)
                }
                return buffer
            }(),
            perLayerInputDimension: perLayerInputRequirements.dimension,
            perLayerInputLayerCount: perLayerInputRequirements.layerCount,
            logits: context.device.makeBuffer(length: resolvedVocabSize * f32ElementSize, options: [.storageModeShared])!,
            tokenIDs: context.device.makeBuffer(length: maximumSequenceLength * 4, options: [.storageModeShared])!,
            positions: context.device.makeBuffer(length: maximumSequenceLength * 4, options: [.storageModeShared])!,
            ropePositionAxes: context.device.makeBuffer(length: maximumSequenceLength * 3 * 4, options: [.storageModeShared])!,
            tokenOut: context.device.makeBuffer(length: 4, options: [.storageModeShared])!,
            runtimeConstantBuffer: context.device.makeBuffer(
                length: PrefillBufferSet.runtimeConstantBufferSize(maximumSequenceLength: maximumSequenceLength),
                options: [.storageModeShared]
            )!
        )

        return PrefillBufferAllocation(
            bufferSet: bufferSet,
            slotDimension: slotDimension,
            resolvedIntermediateSize: resolvedIntermediateSize,
            resolvedVocabSize: resolvedVocabSize,
            maximumSequenceLength: maximumSequenceLength
        )
    }

    private func maximumScratchProjectionDimension(in entries: [DispatchEntry]) -> Int {
        entries.reduce(into: 0) { maximumOutputDimension, entry in
            maximumOutputDimension = max(
                maximumOutputDimension,
                maximumScratchProjectionDimension(in: entry.kind)
            )
        }
    }

    private func maximumScratchProjectionDimension(in kind: DispatchKind) -> Int {
        switch kind {
        case .projection(let projection, let isOutput):
            return isOutput ? 0 : projection.outputDimension
        case .batchedProjection(let batched):
            return batched.projections.map(\.outputDimension).max() ?? 0
        case .fusedSwiGLUProjection(let fused):
            return fused.outputDimension
        case .fragment,
             .batchedFragment,
             .fusedCopyNorm,
             .fusedResidualAddCopyNorm,
             .fusedResidualAddNorm,
             .structuralCopy,
             .structuralAdd:
            return 0
        }
    }

    private func convStateRequirements(in entries: [DispatchEntry]) -> ConvStateRequirements {
        var layerCount = 0
        var dimension = 0
        var kernelSize = 0
        for entry in entries {
            if case .fragment(let fragment) = entry.kind,
               let convSlot = fragment.cacheSlots.first(where: { $0.kind == .conv }) {
                layerCount += 1
                kernelSize = max(kernelSize, convSlot.temporalSize)
                if let conv = fragment as? Conv1dFragment {
                    dimension = max(dimension, conv.dimension)
                } else if let recurrence = fragment as? SSMRecurrenceFragment {
                    dimension = max(
                        dimension,
                        2 * recurrence.groupCount * recurrence.keyHeadDimension
                            + recurrence.headCount * recurrence.valueHeadDimension
                    )
                } else if case .elementwise(let fragmentDimension) = fragment.dispatchDimension {
                    dimension = max(dimension, fragmentDimension)
                }
            }
        }
        return ConvStateRequirements(
            layerCount: layerCount,
            dimension: dimension,
            kernelSize: kernelSize
        )
    }

    private func recurrentStateRequirements(in entries: [DispatchEntry]) -> RecurrentStateRequirements {
        var layerCount = 0
        var bytesPerLayer = 0
        for entry in entries {
            if case .fragment(let fragment) = entry.kind,
               let recurrence = fragment as? SSMRecurrenceFragment {
                layerCount += 1
                bytesPerLayer = max(
                    bytesPerLayer,
                    recurrence.headCount
                        * recurrence.keyHeadDimension
                        * recurrence.valueHeadDimension
                        * MemoryLayout<Float>.size
                )
            }
        }
        return RecurrentStateRequirements(layerCount: layerCount, bytesPerLayer: bytesPerLayer)
    }

    private func perLayerInputRequirements(in entries: [DispatchEntry]) -> PerLayerInputRequirements {
        var layerCount = 0
        var dimension = 0
        for entry in entries {
            if case .fragment(let fragment) = entry.kind,
               let modulation = fragment as? PerLayerInputModulationFragment {
                layerCount = max(layerCount, (entry.layerIndex ?? -1) + 1)
                dimension = max(dimension, modulation.dimension)
            }
        }
        return PerLayerInputRequirements(layerCount: layerCount, dimension: dimension)
    }

    private func resolveKVCacheScheme(
        selection: SchemeSelection,
        weightFormat: WeightFormat
    ) -> QuantizationSchemeIdentifier {
        switch selection {
        case .automatic:
            return weightFormat == .bfloat16 ? .bf16RowMajor : .fp16RowMajor
        case .fixed(let scheme):
            return scheme
        }
    }
}

private struct PerLayerInputRequirements {
    let layerCount: Int
    let dimension: Int
}
