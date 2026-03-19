import Metal

/// Argmax over vocabulary: logits → token ID.
public struct ArgmaxFragment: PrimitiveMetalKernelFragment {
    public let vocabularySize: Int

    public init(vocabularySize: Int) {
        self.vocabularySize = vocabularySize
    }

    public var isFusable: Bool { false }
    public func kernelName(context: KernelContext) -> String {
        context.bufferPrecision == .float32 ? "argmax_f32" : "argmax"
    }
    public var dispatchDimension: MetalDispatchDimension { .reduction(dimension: vocabularySize) }

    public func kernelSource(name: String, bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String {
        MetalSourceGenerator.generateArgmax(name: name, bufferPrecision: bufferPrecision)
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.logits, 0),
                (1, context.bufferSet.tokenOut, 0),
            ],
            bytes: [
                uint32Binding(2, UInt32(vocabularySize)),
            ],
            outputIsHidden: false
        )
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let kernelName = kernelName(context: context.kernelContext)
        let pipeline = try context.getPipeline(kernelName)
        let simdWidth = pipeline.threadExecutionWidth
        let clamped = min(max(vocabularySize, 1), 1024)
        let rounded = ((clamped + simdWidth - 1) / simdWidth) * simdWidth
        let threads = min(rounded, pipeline.maxTotalThreadsPerThreadgroup)
        return FragmentPrefillSteps(
            steps: [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: 1, height: 1, depth: 1),
                threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
                bufferBindings: [
                    (0, context.buffers.logits, 0),
                    (1, context.buffers.tokenOut, 0),
                ],
                bytesBindings: [
                    uint32Binding(2, UInt32(vocabularySize)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .lastToken,
                sequenceLengthPolicy: .none,
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )],
            outputIsHidden: false
        )
    }
}
