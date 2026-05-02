import Metal

/// Add the residual buffer to the current input and write to hidden.
///
/// Emitted at the end of each residual block: `output = input + residual → hidden`.
/// The input comes from the routing state (hidden or scratch depending on
/// the last fragment's output location).
///
/// Fusion: when followed by CopyFragment + Reduction, the compiler can
/// fuse the entire residual boundary (add + copy + norm) into a single dispatch.
public struct ResidualAddFragment: PrimitiveMetalKernelFragment {
    public let dimension: Int

    public init(dimension: Int) {
        self.dimension = dimension
    }

    public var isFusable: Bool { true }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: dimension) }

    public func kernelName(context: KernelContext) -> String {
        context.bufferPrecision.isPrefillSequencePrecision
            ? "residual_add_seq_f32"
            : "residual_add\(context.bufferPrecision.decodeKernelNameSuffix)"
    }

    // MARK: - Fusion Contract

    public var fusionContract: FusionContract? {
        FusionContract(
            ports: [
                FusionPort(name: "data", direction: .input, role: .buffer, accessPattern: .singlePass),
                FusionPort(name: "residual", direction: .input, role: .buffer, bufferIntent: .residual),
                FusionPort(name: "output", direction: .output, role: .buffer),
            ],
            parallelism: .perElement(count: dimension)
        )
    }

    public func kernelBody(bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String? {
        let bt = bufferPrecision.metalType
        let value = "float(data[idx]) + float(residual[idx])"
        let stored = bufferPrecision.isPrefillSequencePrecision
            ? MetalSourceGenerator.sequenceStorageValue(value, weightFormat: weightFormat)
            : value
        return """
        output[idx] = \(bt)(\(stored));
        """
    }

    // MARK: - Standalone Kernel

    public func kernelSource(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        MetalSourceGenerator.generateResidualAdd(
            name: name,
            bufferPrecision: bufferPrecision,
            isSequence: bufferPrecision.isPrefillSequencePrecision
        )
    }

    // MARK: - Decode Bindings

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        FragmentBindings(
            buffers: [
                (0, context.currentInputBuffer, context.currentInputOffset),
                (1, context.bufferSet.residual, 0),
                (2, context.bufferSet.hidden, 0),
            ],
            bytes: [
                uint32Binding(3, UInt32(dimension)),
            ],
            outputIsHidden: true,
            resetsProjectionIndex: false,
            writeBufferIndices: Set([2])
        )
    }

    // MARK: - Prefill Steps

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let kernelName = kernelName(context: context.kernelContext)

        // In-place optimization: when input is already hidden, use residual_add_inplace
        // to avoid allocating a separate output buffer.
        if context.currentInputBuffer === context.buffers.hidden,
           context.currentInputOffset == 0 {
            let inplaceKernelName = context.kernelContext.bufferPrecision.isPrefillSequencePrecision
                ? "residual_add_inplace_seq_f32"
                : "residual_add_inplace"
            let pipeline = try context.getPipeline(inplaceKernelName)
            let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
            let gridX = (dimension + tgSize - 1) / tgSize
            return FragmentPrefillSteps(
                steps: [MetalPrefillStep(
                    pipeline: pipeline,
                    gridSize: MTLSize(width: gridX, height: context.maximumSequenceLength, depth: 1),
                    threadgroupSize: MTLSize(width: tgSize, height: 1, depth: 1),
                    bufferBindings: [
                        (0, context.buffers.hidden, 0),
                        (1, context.buffers.residual, 0),
                    ],
                    bytesBindings: [
                        uint32Binding(2, UInt32(dimension)),
                        uint32Binding(3, UInt32(context.maximumSequenceLength)),
                    ],
                    threadgroupMemoryLength: 0,
                    sync: .bufferBarrier,
                    mode: .batch,
                    sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 3),
                    positionBufferIndex: nil,
                    perPositionStrides: [:],
                    metadata: .init(
                        kernelName: inplaceKernelName,
                        bufferAccessPattern: .init(reads: [0, 1], writes: [0])
                    )
                )],
                outputIsHidden: true,
                resetsProjectionIndex: false
            )
        }

        // Out-of-place: input + residual → hidden
        let pipeline = try context.getPipeline(kernelName)
        let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let gridX = (dimension + tgSize - 1) / tgSize
        return FragmentPrefillSteps(
            steps: [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: gridX, height: context.maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: tgSize, height: 1, depth: 1),
                bufferBindings: [
                    (0, context.currentInputBuffer, context.currentInputOffset),
                    (1, context.buffers.residual, 0),
                    (2, context.buffers.hidden, 0),
                ],
                bytesBindings: [
                    uint32Binding(3, UInt32(dimension)),
                    uint32Binding(4, UInt32(context.maximumSequenceLength)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 4),
                positionBufferIndex: nil,
                perPositionStrides: [:],
                metadata: .init(
                    kernelName: kernelName,
                    bufferAccessPattern: .init(reads: [0, 1], writes: [2])
                )
            )],
            outputIsHidden: true,
            resetsProjectionIndex: false
        )
    }
}
