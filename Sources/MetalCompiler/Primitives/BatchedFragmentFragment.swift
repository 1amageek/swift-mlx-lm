import Metal

// MARK: - PrimitiveMetalKernelFragment conformance for BatchedFragment

extension BatchedFragment: PrimitiveMetalKernelFragment {

    // dispatchDimension is already a stored property — protocol is satisfied.

    public var weightSlots: [MetalWeightSlot] {
        fragments.flatMap(\.weightSlots)
    }

    public func kernelName(context: KernelContext) -> String {
        let baseName = fragments[0].kernelName(context: context)
        return "batched_\(baseName)_\(fragments.count)"
    }

    public func kernelSource(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        guard fragments.count == 2, case .perHead = dispatchDimension else {
            fatalError("BatchedFragment only supports 2-way perHead batching")
        }
        return MetalSourceGenerator.generateBatchedPerHead2(
            name: name,
            bufferPrecision: bufferPrecision,
            weightFormat: weightFormat
        )
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        var bufferBindings: [(index: Int, buffer: MTLBuffer, offset: Int)] = []
        var bytesBindings: [(index: Int, value: [UInt8])] = []
        var mappedWriteIndices = Set<Int>()
        var consumedKVLayers = 0
        var consumedConvLayers = 0
        var consumedRecurrentLayers = 0
        var resetsProjectionIndex = false
        var outputIsHidden = false
        var expectedHeadDimension: Int?
        var expectedEpsilon: Float?
        var expectedWeightBias: Float?

        let fragmentCount = fragments.count
        bufferBindings.reserveCapacity(fragmentCount * 2)

        for (fragmentIndex, fragment) in fragments.enumerated() {
            let resolvedKVCacheIndex = fragment.kvCacheIndexOverride ?? (context.kvCacheIndex + consumedKVLayers)
            let fragmentContext = BufferBindingContext(
                bufferSet: context.bufferSet,
                slotDimension: context.slotDimension,
                elementSize: context.elementSize,
                currentInputBuffer: context.currentInputBuffer,
                currentInputOffset: context.currentInputOffset,
                layerIndex: context.layerIndex,
                kvCacheIndex: resolvedKVCacheIndex,
                convLayerIndex: context.convLayerIndex + consumedConvLayers,
                recurrentLayerIndex: context.recurrentLayerIndex + consumedRecurrentLayers,
                projectionIndex: context.projectionIndex,
                resolveWeight: context.resolveWeight
            )
            let fragmentBindings = fragment.decodeBindings(context: fragmentContext)

            guard let dataBinding = fragmentBindings.buffers.first(where: { $0.index == 0 }) else {
                preconditionFailure("Batched fragment \(type(of: fragment)) missing data binding[0]")
            }
            guard let weightBinding = fragmentBindings.buffers.first(where: { $0.index == 1 }) else {
                preconditionFailure("Batched fragment \(type(of: fragment)) missing weight binding[1]")
            }
            guard case .perHead(let headCount) = fragment.dispatchDimension else {
                preconditionFailure("Batched fragment \(type(of: fragment)) must use .perHead dispatch")
            }

            let headDimension: Int
            if let perHeadDimension = fragment.perHeadDimension {
                headDimension = perHeadDimension
            } else {
                preconditionFailure("Batched fragment \(type(of: fragment)) must declare perHeadDimension")
            }

            if let expectedHeadDimension, expectedHeadDimension != headDimension {
                preconditionFailure("Batched fragments must share the same head dimension")
            }
            expectedHeadDimension = headDimension

            let epsilon = fragment.normEpsilon ?? 1e-6
            if let expectedEpsilon, expectedEpsilon != epsilon {
                preconditionFailure("Batched fragments must share the same epsilon")
            }
            expectedEpsilon = epsilon

            let weightBias = fragment.normWeightBias ?? 0
            if let expectedWeightBias, expectedWeightBias != weightBias {
                preconditionFailure("Batched fragments must share the same weight bias")
            }
            expectedWeightBias = weightBias

            bufferBindings.append((fragmentIndex, dataBinding.buffer, dataBinding.offset))
            bufferBindings.append((fragmentCount + fragmentIndex, weightBinding.buffer, weightBinding.offset))
            bytesBindings.append(uint32Binding(2 * fragmentCount + fragmentIndex, UInt32(headCount)))

            if fragmentBindings.writeBufferIndices?.contains(dataBinding.index) == true {
                mappedWriteIndices.insert(fragmentIndex)
            }

            resetsProjectionIndex = resetsProjectionIndex || fragmentBindings.resetsProjectionIndex
            outputIsHidden = fragmentBindings.outputIsHidden
            if fragmentBindings.consumesKVCacheLayer { consumedKVLayers += 1 }
            if fragmentBindings.consumesConvLayer { consumedConvLayers += 1 }
            if fragmentBindings.consumesRecurrentLayer { consumedRecurrentLayers += 1 }
        }

        if let expectedHeadDimension {
            bytesBindings.append(uint32Binding(3 * fragmentCount, UInt32(expectedHeadDimension)))
        }
        if let expectedEpsilon {
            bytesBindings.append(floatBinding(3 * fragmentCount + 1, expectedEpsilon))
        }
        bytesBindings.append(floatBinding(3 * fragmentCount + 2, expectedWeightBias ?? 0))

        return FragmentBindings(
            buffers: bufferBindings,
            bytes: bytesBindings,
            outputIsHidden: outputIsHidden,
            resetsProjectionIndex: resetsProjectionIndex,
            consumesKVCacheLayer: consumedKVLayers > 0,
            consumesConvLayer: consumedConvLayers > 0,
            consumesRecurrentLayer: consumedRecurrentLayers > 0,
            writeBufferIndices: mappedWriteIndices
        )
    }

    public func requiredFallbackBufferSize(for role: String, bytesPerScalar: Int) -> Int {
        for child in fragments {
            let size = child.requiredFallbackBufferSize(for: role, bytesPerScalar: bytesPerScalar)
            if size > 0 { return size }
        }
        return 0
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        // Prefill decomposes into individual fragments or batched QK norm.
        // The caller (PrefillStepPlanner) handles this decomposition
        // by checking for BatchedFragment and expanding.
        fatalError("BatchedFragment.prefillSteps should not be called directly; planner decomposes to individual steps")
    }
}
