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
        if !walkContext.cacheSlots.isEmpty {
            // Mixed-attention models (e.g. Gemma4 sliding_attention head_dim=256 +
            // full_attention head_dim=512) have per-layer head_dim variation. The cache
            // slot stride must accommodate the LARGEST head_dim across all layers,
            // otherwise kernels writing head_dim=512 into a slot sized for 256 overflow
            // into adjacent layers/positions and cause non-determinism.
            let maxHeadDimension = walkContext.cacheSlots.map(\.headDimension).max()!
            let maxKVHeadCount = walkContext.cacheSlots.map(\.kvHeadCount).max()!
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
                    kvHeadCount: maxKVHeadCount,
                    headDimension: maxHeadDimension,
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

        // Private storage buffers cannot be zeroed from CPU — use GPU blit fill.
        // Without this, SSM recurrence accumulates uninitialized memory and produces
        // non-deterministic per-head output.
        try zeroPrivateBuffers([convStateBuffer].compactMap { $0 }, device: context.device)

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

        try zeroPrivateBuffers([recurrentStateBuffer].compactMap { $0 }, device: context.device)

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
        } else if !walkContext.cacheSlots.isEmpty {
            // Per-layer head_dim / kv_head_count variation (e.g. Gemma4 mixed
            // sliding/full attention) requires the cache stride to match the maximum.
            // See makeDecodeBufferAllocation for the detailed rationale.
            let maxHeadDimension = walkContext.cacheSlots.map(\.headDimension).max()!
            let maxKVHeadCount = walkContext.cacheSlots.map(\.kvHeadCount).max()!
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
                    kvHeadCount: maxKVHeadCount,
                    headDimension: maxHeadDimension,
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

        let dequantScratchBuffer: MTLBuffer?
        if hasQuantizedProjectionWeights(in: sizingEntries, stafWeightStore: context.stafWeightStore) {
            let maxWeightElements = maximumProjectionWeightElementCount(in: sizingEntries)
            if maxWeightElements > 0 {
                let bf16Size = MemoryLayout<UInt16>.stride
                dequantScratchBuffer = context.device.makeBuffer(
                    length: maxWeightElements * bf16Size,
                    options: gpuOnlyOptions
                )
            } else {
                dequantScratchBuffer = nil
            }
        } else {
            dequantScratchBuffer = nil
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
            dequantScratch: dequantScratchBuffer,
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
        entries.reduce(into: 0) { maxDim, entry in
            guard let projection = entry.fragment as? ProjectionDescribing,
                  !projection.isOutputProjection else { return }
            for field in projection.projectionFields {
                maxDim = max(maxDim, field.outputDimension)
            }
        }
    }

    /// Check if any projection weight in the dispatch entries uses quantized format.
    /// Queries STAF weight store directly since `CompileContext.weightFormat` only reflects
    /// the dense format (BF16/FP16), not quantized variants.
    private func hasQuantizedProjectionWeights(
        in entries: [DispatchEntry],
        stafWeightStore: STAFWeightStore?
    ) -> Bool {
        guard let staf = stafWeightStore else { return false }
        for entry in entries {
            guard let projection = entry.fragment as? ProjectionDescribing else { continue }
            let roles = projection.projectionFields.map(\.field)
            for role in roles {
                if let binding = entry.parameterBindings.first(where: { $0.role == role }),
                   let info = staf.tensor(for: binding.tensorName),
                   info.format.schemeIdentifier.isWeightQuantized {
                    return true
                }
            }
        }
        return false
    }

    /// Maximum weight element count (outputDim × inputDim) across all projections.
    /// Used to size the dequant scratch buffer for Q4→BF16 unpacking.
    private func maximumProjectionWeightElementCount(in entries: [DispatchEntry]) -> Int {
        entries.reduce(into: 0) { maxElements, entry in
            guard let projection = entry.fragment as? ProjectionDescribing else { return }
            for field in projection.projectionFields {
                maxElements = max(maxElements, field.inputDimension * field.outputDimension)
            }
        }
    }

    private func convStateRequirements(in entries: [DispatchEntry]) -> ConvStateRequirements {
        var layerCount = 0
        var dimension = 0
        var kernelSize = 0
        for entry in entries {
            let fragment = entry.fragment
            if let convSlot = fragment.cacheSlots.first(where: { $0.kind == .conv }) {
                layerCount += 1
                kernelSize = max(kernelSize, convSlot.temporalSize)
                if let convReq = fragment as? ConvStateRequiring {
                    dimension = max(dimension, convReq.convStateDimension)
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
            if let recReq = entry.fragment as? RecurrentStateRequiring {
                layerCount += 1
                bytesPerLayer = max(bytesPerLayer, recReq.recurrentStateBytesPerLayer)
            }
        }
        return RecurrentStateRequirements(layerCount: layerCount, bytesPerLayer: bytesPerLayer)
    }

    private func perLayerInputRequirements(in entries: [DispatchEntry]) -> PerLayerInputRequirements {
        var layerCount = 0
        var dimension = 0
        for entry in entries {
            if let pli = entry.fragment as? PerLayerInputCapable {
                layerCount = max(layerCount, (entry.layerIndex ?? -1) + 1)
                dimension = max(dimension, pli.perLayerInputDimension)
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
            return weightFormat.isBFloat16 ? .bf16RowMajor : .fp16RowMajor
        case .fixed(let scheme):
            return scheme
        }
    }

    /// Zero-fill the given private-storage buffers via a synchronous blit.
    ///
    /// Private buffers are GPU-only and cannot be memset from CPU. Stateful recurrence
    /// buffers (SSM recurrent state, conv state) must start at zero; uninitialized
    /// GPU memory accumulates garbage into the state and produces non-deterministic
    /// per-head output.
    private func zeroPrivateBuffers(_ buffers: [MTLBuffer], device: MTLDevice) throws {
        guard !buffers.isEmpty else { return }
        guard let queue = device.makeCommandQueue(),
              let commandBuffer = queue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw MetalBufferAllocatorError.commandEncoderCreationFailed
        }
        for buffer in buffers {
            blit.fill(buffer: buffer, range: 0..<buffer.length, value: 0)
        }
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}

enum MetalBufferAllocatorError: Error {
    case commandEncoderCreationFailed
}

private struct PerLayerInputRequirements {
    let layerCount: Int
    let dimension: Int
}
