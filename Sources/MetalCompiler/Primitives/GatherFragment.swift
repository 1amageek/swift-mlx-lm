import Metal

/// Token ID → embedding vector lookup.
public struct GatherFragment: PrimitiveMetalKernelFragment {
    public let vocabularySize: Int
    public let embeddingDimension: Int
    public let embeddingScale: Float?

    public init(vocabularySize: Int, embeddingDimension: Int, embeddingScale: Float? = nil) {
        self.vocabularySize = vocabularySize
        self.embeddingDimension = embeddingDimension
        self.embeddingScale = embeddingScale
    }

    public var isFusable: Bool { false }
    public func kernelName(context: KernelContext) -> String {
        let scaled = embeddingScale != nil ? "_scaled" : ""
        switch (context.bufferPrecision, context.weightFormat) {
        case (.float32, .bfloat16):
            return "embedding_lookup_seq_bf16_f32\(scaled)"
        case (.float32, .float32):
            return "embedding_lookup_seq_fp32_f32\(scaled)"
        case (.float32, .quantized4Bit(let groupSize)):
            return "embedding_lookup_seq_q4_g\(groupSize)_f32\(scaled)"
        case (.float32, .quantized8Bit(let groupSize)):
            return "embedding_lookup_seq_q8_g\(groupSize)_f32\(scaled)"
        case (_, .bfloat16):
            return "embedding_lookup_bf16\(scaled)"
        case (_, .float32):
            return "embedding_lookup_fp32\(scaled)"
        case (_, .quantized4Bit(let groupSize)):
            return "embedding_lookup_q4_g\(groupSize)\(scaled)"
        case (_, .quantized8Bit(let groupSize)):
            return "embedding_lookup_q8_g\(groupSize)\(scaled)"
        case (_, .float16):
            return context.bufferPrecision == .float32
                ? "embedding_lookup_seq_f32\(scaled)"
                : "embedding_lookup\(scaled)"
        case (_, .quantized2Bit(let groupSize)):
            return context.bufferPrecision == .float32
                ? "embedding_lookup_seq_q2_g\(groupSize)_f32\(scaled)"
                : "embedding_lookup_q2_g\(groupSize)\(scaled)"
        case (_, .quantized3Bit(let groupSize)):
            return context.bufferPrecision == .float32
                ? "embedding_lookup_seq_q3_g\(groupSize)_f32\(scaled)"
                : "embedding_lookup_q3_g\(groupSize)\(scaled)"
        case (_, .quantized5Bit(let groupSize)):
            return context.bufferPrecision == .float32
                ? "embedding_lookup_seq_q5_g\(groupSize)_f32\(scaled)"
                : "embedding_lookup_q5_g\(groupSize)\(scaled)"
        case (_, .quantized6Bit(let groupSize)):
            return context.bufferPrecision == .float32
                ? "embedding_lookup_seq_q6_g\(groupSize)_f32\(scaled)"
                : "embedding_lookup_q6_g\(groupSize)\(scaled)"
        }
    }
    public var dispatchDimension: MetalDispatchDimension { .gather(count: embeddingDimension) }
    public var weightSlots: [MetalWeightSlot] { [MetalWeightSlot(field: nil, role: .weight)] }

    public func kernelSource(name: String, bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String {
        switch weightFormat {
        case .quantized4Bit(let groupSize):
            return MetalSourceGenerator.generateQuantizedEmbeddingLookupQ4(
                name: name,
                bufferPrecision: bufferPrecision,
                groupSize: groupSize,
                isSequence: bufferPrecision == .float32,
                embeddingScale: embeddingScale
            )
        case .quantized8Bit(let groupSize):
            return MetalSourceGenerator.generateQuantizedEmbeddingLookupQ8(
                name: name,
                bufferPrecision: bufferPrecision,
                groupSize: groupSize,
                isSequence: bufferPrecision == .float32,
                embeddingScale: embeddingScale
            )
        case .quantized2Bit, .quantized3Bit, .quantized5Bit, .quantized6Bit:
            guard let format = weightFormat.quantizationFormat else {
                fatalError("GatherFragment.kernelSource: registry missing format for \(weightFormat)")
            }
            return MetalSourceGenerator.generateUnifiedQuantizedEmbeddingLookup(
                name: name,
                format: format,
                bufferPrecision: bufferPrecision,
                isSequence: bufferPrecision == .float32,
                embeddingScale: embeddingScale
            )
        default:
            return MetalSourceGenerator.generateEmbeddingLookup(
                name: name,
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat,
                isSequence: bufferPrecision == .float32,
                embeddingScale: embeddingScale
            )
        }
    }

    public func requiredFallbackBufferSize(for role: String, bytesPerScalar: Int) -> Int {
        vocabularySize * embeddingDimension * bytesPerScalar
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let (weightBuffer, weightOffset) = context.resolveWeight("embedding_table")
        var bytes: [(Int, [UInt8])] = [
            uint32Binding(3, UInt32(embeddingDimension)),
        ]
        if let scale = embeddingScale {
            bytes.append(floatBinding(4, scale))
        }
        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.tokenIn, 0),
                (1, weightBuffer, weightOffset),
                (2, context.bufferSet.hidden, 0),
            ],
            bytes: bytes,
            outputIsHidden: true,
            writeBufferIndices: Set<Int>([2])
        )
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let (weightBuffer, weightOffset) = context.resolveWeight("embedding_table")
        let kernelName = kernelName(context: context.kernelContext)
        let pipeline = try context.getPipeline(kernelName)
        let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let gridX = (embeddingDimension + tgSize - 1) / tgSize
        var bytes: [(Int, [UInt8])] = [
            uint32Binding(3, UInt32(embeddingDimension)),
            uint32Binding(4, UInt32(context.maximumSequenceLength)),
        ]
        if let scale = embeddingScale {
            bytes.append(floatBinding(5, scale))
        }
        return FragmentPrefillSteps(
            steps: [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: gridX, height: context.maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: tgSize, height: 1, depth: 1),
                bufferBindings: [
                    (0, context.buffers.tokenIDs, 0),
                    (1, weightBuffer, weightOffset),
                    (2, context.buffers.hidden, 0),
                ],
                bytesBindings: bytes,
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 4),
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )],
            outputIsHidden: true
        )
    }
}
