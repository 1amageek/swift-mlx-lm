import Metal

/// Multiply the current hidden state by a learned scalar.
public struct ScalarMultiplyFragment: PrimitiveMetalKernelFragment {
    public let count: Int
    public let weightRole: String

    public init(count: Int, weightRole: String) {
        self.count = count
        self.weightRole = weightRole
    }

    public var isFusable: Bool { true }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: count) }
    public var weightSlots: [MetalWeightSlot] { [MetalWeightSlot(field: weightRole, role: .weight)] }

    public var fusionContract: FusionContract? {
        FusionContract(
            ports: [
                FusionPort(name: "data", direction: .input, role: .buffer, accessPattern: .singlePass),
                FusionPort(name: "weight", direction: .input, role: .weight(field: weightRole)),
                FusionPort(name: "output", direction: .output, role: .buffer),
            ],
            parallelism: .perElement(count: count)
        )
    }

    public func kernelBody(bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String? {
        let readWeight = weightFormat.readExpression("weight[0]")
        let bt = bufferPrecision.metalType
        return """
        float scale = \(readWeight);
        output[idx] = \(bt)(float(data[idx]) * scale);
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
            fatalError("[Compiler] ScalarMultiplyFragment does not support quantized weights")
        }
        if context.bufferPrecision == .float32 {
            return "scalar_multiply_seq_\(weightSuffix)"
        }
        let outputPrefix = context.bufferPrecision == .bfloat16 ? "scalar_multiply_bf16" : "scalar_multiply"
        return "\(outputPrefix)_\(weightSuffix)"
    }

    public func kernelSource(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        guard let contract = fusionContract,
              let body = kernelBody(bufferPrecision: bufferPrecision, weightFormat: weightFormat) else {
            fatalError("ScalarMultiplyFragment must provide fusionContract and kernelBody")
        }
        return KernelScaffold.generate(
            name: name,
            body: body,
            contract: contract,
            bufferPrecision: bufferPrecision,
            weightFormats: [weightRole: weightFormat],
            isSequence: bufferPrecision == .float32
        )
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let (weightBuffer, weightOffset) = context.resolveWeight(weightRole)
        return FragmentBindings(
            buffers: [
                (0, context.currentInputBuffer, context.currentInputOffset),
                (1, weightBuffer, weightOffset),
                (2, context.bufferSet.hidden, 0),
            ],
            bytes: [
                uint32Binding(3, UInt32(count)),
            ],
            outputIsHidden: true,
            resetsProjectionIndex: true,
            writeBufferIndices: Set<Int>([2])
        )
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let (weightBuffer, weightOffset) = context.resolveWeight(weightRole)
        let pipeline = try context.getPipeline(kernelName(context: context.kernelContext))
        let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let gridX = (count + tgSize - 1) / tgSize
        return FragmentPrefillSteps(
            steps: [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: gridX, height: context.maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: tgSize, height: 1, depth: 1),
                bufferBindings: [
                    (0, context.currentInputBuffer, context.currentInputOffset),
                    (1, weightBuffer, weightOffset),
                    (2, context.buffers.hidden, 0),
                ],
                bytesBindings: [
                    uint32Binding(3, UInt32(count)),
                    uint32Binding(4, UInt32(context.maximumSequenceLength)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 4),
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )],
            outputIsHidden: true,
            resetsProjectionIndex: true
        )
    }
}
