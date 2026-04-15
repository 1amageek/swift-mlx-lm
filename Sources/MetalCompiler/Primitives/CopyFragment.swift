import Metal

/// Copy the hidden buffer to the residual buffer.
///
/// Emitted at the start of each residual block: `hidden → residual`.
/// The hidden buffer is not modified — the next fragment still reads from hidden.
///
/// Fusion: provides a pass-through `output` port so that the compiler can
/// fuse CopyFragment + Reduction into a single dispatch (read hidden once,
/// copy to residual AND compute RMS norm in the same kernel).
public struct CopyFragment: PrimitiveMetalKernelFragment {
    public let dimension: Int

    public init(dimension: Int) {
        self.dimension = dimension
    }

    public var isFusable: Bool { true }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: dimension) }

    public func kernelName(context: KernelContext) -> String {
        context.bufferPrecision == .float32 ? "copy_buffer_seq_f32" : "copy_buffer"
    }

    // MARK: - Fusion Contract

    public var fusionContract: FusionContract? {
        FusionContract(
            ports: [
                FusionPort(name: "data", direction: .input, role: .buffer, accessPattern: .singlePass),
                FusionPort(name: "output", direction: .output, role: .buffer),
                FusionPort(name: "residual", direction: .output, role: .buffer, bufferIntent: .residual),
            ],
            parallelism: .perElement(count: dimension)
        )
    }

    public func kernelBody(bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String? {
        let bt = bufferPrecision.metalType
        return """
        residual[idx] = \(bt)(data[idx]);
        output[idx] = \(bt)(data[idx]);
        """
    }

    // MARK: - Standalone Kernel

    public func kernelSource(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        MetalSourceGenerator.generateCopy(
            name: name,
            bufferPrecision: bufferPrecision,
            isSequence: bufferPrecision == .float32
        )
    }

    // MARK: - Decode Bindings

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        FragmentBindings(
            buffers: [
                (0, context.bufferSet.hidden, 0),
                (1, context.bufferSet.residual, 0),
            ],
            bytes: [
                uint32Binding(2, UInt32(dimension)),
            ],
            outputIsHidden: true,
            resetsProjectionIndex: true,
            writeBufferIndices: Set([1])
        )
    }

    // MARK: - Prefill Steps

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let kernelName = kernelName(context: context.kernelContext)
        let pipeline = try context.getPipeline(kernelName)
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
                perPositionStrides: [:]
            )],
            outputIsHidden: true,
            resetsProjectionIndex: true
        )
    }
}
