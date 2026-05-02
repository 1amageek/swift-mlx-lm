import Metal

/// Spatial average pooling fragment.
///
/// Reduces sequence length by averaging non-overlapping kernel-sized windows.
/// Each output position is the average of `kernelSize * kernelSize` input
/// positions, optionally rescaled.
///
/// Sequence length reduction: outputLength = inputLength / (kernelSize²).
public struct PoolingFragment: PrimitiveMetalKernelFragment {
    public let kernelSize: Int
    public let hiddenSize: Int
    public let rescale: Float?

    public init(kernelSize: Int, hiddenSize: Int, rescale: Float? = nil) {
        self.kernelSize = kernelSize
        self.hiddenSize = hiddenSize
        self.rescale = rescale
    }

    public var isFusable: Bool { false }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: hiddenSize) }

    public func kernelName(context: KernelContext) -> String {
        let scaled = rescale != nil ? "_scaled" : ""
        return context.bufferPrecision == .float32
            ? "avg_pool_seq_f32\(scaled)"
            : "avg_pool_f16\(scaled)"
    }

    public func kernelSource(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let precision = bufferPrecision == .float32 ? "float" : "half"
        let kernelArea = kernelSize * kernelSize
        var source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void \(name)(
            device const \(precision)* input [[buffer(0)]],
            device \(precision)* output [[buffer(1)]],
            constant uint& inputLength [[buffer(2)]],
            constant uint& dimension [[buffer(3)]],
        """
        if rescale != nil {
            source += """
                constant float& rescaleValue [[buffer(4)]],

            """
        }
        source += """
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint outPos = gid.y;
            uint dim = gid.x;
            if (dim >= dimension) return;

            uint outputLength = inputLength / \(kernelArea);
            if (outPos >= outputLength) return;

            float sum = 0.0;
            uint basePos = outPos * \(kernelArea);
            for (uint k = 0; k < \(kernelArea); k++) {
                uint srcPos = basePos + k;
                if (srcPos < inputLength) {
                    sum += float(input[srcPos * dimension + dim]);
                }
            }
            float avg = sum / float(\(kernelArea));
        """
        if rescale != nil {
            source += """

                avg *= rescaleValue;
            """
        }
        source += """

            output[outPos * dimension + dim] = \(precision)(avg);
        }
        """
        return source
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        var bytes: [(Int, [UInt8])] = [
            uint32Binding(2, UInt32(1)),
            uint32Binding(3, UInt32(hiddenSize)),
        ]
        if let rescale {
            bytes.append(floatBinding(4, rescale))
        }
        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.hidden, 0),
                (1, context.bufferSet.hidden, 0),
            ],
            bytes: bytes,
            outputIsHidden: true,
            writeBufferIndices: Set<Int>([1])
        )
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let kernelName = kernelName(context: context.kernelContext)
        let pipeline = try context.getPipeline(kernelName)
        let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let gridX = (hiddenSize + tgSize - 1) / tgSize
        var bytes: [(Int, [UInt8])] = [
            uint32Binding(3, UInt32(hiddenSize)),
        ]
        if let rescale {
            bytes.append(floatBinding(4, rescale))
        }
        return FragmentPrefillSteps(
            steps: [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: gridX, height: context.maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: tgSize, height: 1, depth: 1),
                bufferBindings: [
                    (0, context.buffers.hidden, 0),
                    (1, context.buffers.hidden, 0),
                ],
                bytesBindings: bytes,
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 2),
                positionBufferIndex: nil,
                perPositionStrides: [:],
                metadata: .init(
                    kernelName: kernelName,
                    bufferAccessPattern: .init(reads: [0], writes: [1])
                )
            )],
            outputIsHidden: true
        )
    }
}
