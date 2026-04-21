import Metal

/// Per-element affine standardization: `(x - bias) * scale`.
///
/// Used by Gemma4 vision encoder's post-pooling standardization step.
/// Two weight buffers: `std_bias` and `std_scale`.
public struct StandardizeFragment: PrimitiveMetalKernelFragment {
    public let dimension: Int

    public init(dimension: Int) {
        self.dimension = dimension
    }

    public var isFusable: Bool { false }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: dimension) }
    public var weightSlots: [MetalWeightSlot] {
        [
            MetalWeightSlot(field: "std_bias", role: .weight),
            MetalWeightSlot(field: "std_scale", role: .weight),
        ]
    }

    public var fusionContract: FusionContract? {
        FusionContract(
            ports: [
                FusionPort(name: "data", direction: .input, role: .buffer, accessPattern: .singlePass),
                FusionPort(name: "std_bias", direction: .input, role: .weight(field: "std_bias")),
                FusionPort(name: "std_scale", direction: .input, role: .weight(field: "std_scale")),
                FusionPort(name: "output", direction: .output, role: .buffer),
            ],
            parallelism: .perElement(count: dimension)
        )
    }

    public func kernelBody(bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String? {
        let readBias = weightFormat.readExpression("std_bias[idx]")
        let readScale = weightFormat.readExpression("std_scale[idx]")
        let bt = bufferPrecision.metalType
        return """
        output[idx] = \(bt)((float(data[idx]) - \(readBias)) * \(readScale));
        """
    }

    public func kernelName(context: KernelContext) -> String {
        let weightSuffix: String = switch context.weightFormat {
        case .float16:
            "f16"
        case .bfloat16:
            "bf16"
        case .float32:
            "f32"
        case .quantized2Bit, .quantized3Bit, .quantized4Bit, .quantized5Bit, .quantized6Bit, .quantized8Bit:
            fatalError("[Compiler] StandardizeFragment does not support quantized weights")
        }
        if context.bufferPrecision == .float32 {
            return "standardize_seq_\(weightSuffix)"
        }
        return "standardize_\(weightSuffix)"
    }

    public func kernelSource(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        guard let contract = fusionContract,
              let body = kernelBody(bufferPrecision: bufferPrecision, weightFormat: weightFormat) else {
            fatalError("StandardizeFragment must provide fusionContract and kernelBody")
        }
        return KernelScaffold.generate(
            name: name,
            body: body,
            contract: contract,
            bufferPrecision: bufferPrecision,
            weightFormats: ["std_bias": weightFormat, "std_scale": weightFormat],
            isSequence: bufferPrecision == .float32
        )
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let (biasBuffer, biasOffset) = context.resolveWeight("std_bias")
        let (scaleBuffer, scaleOffset) = context.resolveWeight("std_scale")
        return FragmentBindings(
            buffers: [
                (0, context.currentInputBuffer, context.currentInputOffset),
                (1, biasBuffer, biasOffset),
                (2, scaleBuffer, scaleOffset),
                (3, context.bufferSet.hidden, 0),
            ],
            bytes: [
                uint32Binding(4, UInt32(dimension)),
            ],
            outputIsHidden: true,
            resetsProjectionIndex: true,
            writeBufferIndices: Set<Int>([3])
        )
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let (biasBuffer, biasOffset) = context.resolveWeight("std_bias")
        let (scaleBuffer, scaleOffset) = context.resolveWeight("std_scale")
        let pipeline = try context.getPipeline(kernelName(context: context.kernelContext))
        let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let gridX = (dimension + tgSize - 1) / tgSize
        return FragmentPrefillSteps(
            steps: [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: gridX, height: context.maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: tgSize, height: 1, depth: 1),
                bufferBindings: [
                    (0, context.currentInputBuffer, context.currentInputOffset),
                    (1, biasBuffer, biasOffset),
                    (2, scaleBuffer, scaleOffset),
                    (3, context.buffers.hidden, 0),
                ],
                bytesBindings: [
                    uint32Binding(4, UInt32(dimension)),
                    uint32Binding(5, UInt32(context.maximumSequenceLength)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 5),
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )],
            outputIsHidden: true,
            resetsProjectionIndex: true
        )
    }
}
