import Metal

// MARK: - PrimitiveMetalKernelFragment conformance for BatchedProjection

extension BatchedProjection: PrimitiveMetalKernelFragment {

    public var dispatchDimension: MetalDispatchDimension {
        .gemv(outputDimension: totalOutputDimension, inputDimension: inputDimension)
    }

    public var weightSlots: [MetalWeightSlot] {
        projections.map { MetalWeightSlot(field: $0.field, role: .weight) }
    }

    public func kernelName(context: KernelContext) -> String {
        let count = projections.count
        let isPrefill = context.bufferPrecision == .float32

        // Prefill paths:
        //   Q4 packed weights → block-level batched GEMM kernel.
        //   Dense weights (BF16/FP16/FP32) → MPP matmul2d-based batched kernel.
        if isPrefill {
            switch context.weightFormat {
            case .quantized4Bit(let groupSize):
                return "batched_gemm_q4_g\(groupSize)_\(count)"
            case .bfloat16:
                return "batched_gemm_bf16_f32s_\(count)"
            case .float16:
                return "batched_gemm_f16_f32s_\(count)"
            case .float32:
                return "batched_gemm_f32_f32s_\(count)"
            case .quantized2Bit, .quantized3Bit, .quantized5Bit, .quantized6Bit, .quantized8Bit:
                // No batched prefill kernel for these formats yet — fragment is
                // decomposed into per-projection dispatches upstream.
                break
            }
        }

        // Decode / dense weights: batched GEMV
        let suffix = context.weightFormat == .bfloat16 ? "_bf16" : ""
        return "batched_gemv\(count)\(suffix)"
    }

    public func kernelSource(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let count = projections.count
        switch count {
        case 2:
            return MetalSourceGenerator.generateBatchedGEMV2(
                name: name,
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat
            )
        case 3:
            return MetalSourceGenerator.generateBatchedGEMV3(
                name: name,
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat
            )
        default:
            return MetalSourceGenerator.generateBatchedGEMV4(
                name: name,
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat
            )
        }
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let inputBuffer = context.currentInputBuffer
        let inputOffset = context.currentInputOffset
        let count = projections.count

        var bufferBindings: [(index: Int, buffer: MTLBuffer, offset: Int)] = [
            (0, inputBuffer, inputOffset),
        ]
        var bytesBindings: [(index: Int, value: [UInt8])] = []
        var lastOutputOffset = context.currentInputOffset

        for (i, proj) in projections.enumerated() {
            let weight = context.resolveWeight(proj.field)
            bufferBindings.append((1 + i, weight.buffer, weight.offset))
        }

        for i in 0..<count {
            let scratchSlot = context.projectionIndex + 1 + i
            let outputOffset = scratchSlot * context.slotDimension * context.elementSize
            lastOutputOffset = outputOffset
            bufferBindings.append((1 + count + i, context.bufferSet.scratch, outputOffset))
        }

        let bytesStart = 1 + 2 * count
        bytesBindings.append(uint32Binding(bytesStart, UInt32(inputDimension)))
        for (i, proj) in projections.enumerated() {
            bytesBindings.append(uint32Binding(bytesStart + 1 + i, UInt32(proj.outputDimension)))
        }

        let writeIndices = Set((1 + count)..<(1 + 2 * count))
        return FragmentBindings(
            buffers: bufferBindings,
            bytes: bytesBindings,
            outputIsHidden: false,
            writeBufferIndices: writeIndices,
            projectionSlotsConsumed: count
        )
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        // Prefill decomposes into individual projections.
        // The caller (PrefillStepPlanner) handles this decomposition
        // by checking for BatchedProjection and expanding.
        fatalError("BatchedProjection.prefillSteps should not be called directly; planner decomposes to individual steps")
    }

    public func requiredFallbackBufferSize(for role: String, bytesPerScalar: Int) -> Int {
        guard let projection = projections.first(where: { $0.field == role }) else { return 0 }
        return projection.inputDimension * projection.outputDimension * bytesPerScalar
    }
}

extension BatchedProjection: ProjectionDescribing {
    public var projectionFields: [ProjectionFieldDescriptor] {
        projections.map {
            ProjectionFieldDescriptor(field: $0.field, inputDimension: $0.inputDimension, outputDimension: $0.outputDimension)
        }
    }
    public var isOutputProjection: Bool { false }
    public func withOutputProjectionEnabled() -> any PrimitiveMetalKernelFragment { self }
}
