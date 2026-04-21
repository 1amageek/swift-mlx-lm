import Metal

/// Adds 2D separable position embeddings to the hidden state.
///
/// For each grid position (row, col):
///   hidden[pos] += table_x[col * hidden + idx] + table_y[row * hidden + idx]
///
/// Table layout: [x_table: tableSize × hiddenSize, y_table: tableSize × hiddenSize]
public struct PositionEmbeddingFragment: PrimitiveMetalKernelFragment {
    public let hiddenSize: Int
    public let tableSize: Int
    public let gridWidth: Int

    public init(hiddenSize: Int, tableSize: Int, gridWidth: Int) {
        self.hiddenSize = hiddenSize
        self.tableSize = tableSize
        self.gridWidth = gridWidth
    }

    public var isFusable: Bool { false }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: hiddenSize) }
    public var weightSlots: [MetalWeightSlot] {
        [MetalWeightSlot(field: "position_embedding_table", role: .weight)]
    }

    public func kernelName(context: KernelContext) -> String {
        let weightSuffix: String = switch context.weightFormat {
        case .float16: "f16"
        case .bfloat16: "bf16"
        case .float32: "f32"
        case .quantized2Bit, .quantized3Bit, .quantized4Bit, .quantized5Bit, .quantized6Bit, .quantized8Bit:
            fatalError("[Compiler] PositionEmbeddingFragment does not support quantized weights")
        }
        if context.bufferPrecision == .float32 {
            return "position_embedding_2d_seq_\(weightSuffix)"
        }
        return "position_embedding_2d_\(weightSuffix)"
    }

    public func kernelSource(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let isSequence = bufferPrecision == .float32

        if isSequence {
            return """
            #include <metal_stdlib>
            using namespace metal;

            kernel void \(name)(
                device \(bt)* data [[buffer(0)]],
                device const \(wt)* table [[buffer(1)]],
                constant uint& hiddenSize [[buffer(2)]],
                constant uint& tableSize [[buffer(3)]],
                constant uint& gridW [[buffer(4)]],
                constant uint& seqLen [[buffer(5)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                uint idx = gid.x;
                uint pos = gid.y;
                if (idx >= hiddenSize) return;
                if (pos >= seqLen) return;
                uint col = pos % gridW;
                uint row = pos / gridW;
                uint xOffset = col * hiddenSize + idx;
                uint yOffset = tableSize * hiddenSize + row * hiddenSize + idx;
                float posEmb = \(readWeight("table[xOffset]")) + \(readWeight("table[yOffset]"));
                uint dataIdx = pos * hiddenSize + idx;
                data[dataIdx] = \(bt)(float(data[dataIdx]) + posEmb);
            }
            """
        } else {
            return """
            #include <metal_stdlib>
            using namespace metal;

            kernel void \(name)(
                device \(bt)* data [[buffer(0)]],
                device const \(wt)* table [[buffer(1)]],
                constant uint& hiddenSize [[buffer(2)]],
                constant uint& tableSize [[buffer(3)]],
                constant uint& gridW [[buffer(4)]],
                uint idx [[thread_position_in_grid]]
            ) {
                if (idx >= hiddenSize) return;
                uint col = 0;
                uint row = 0;
                uint xOffset = col * hiddenSize + idx;
                uint yOffset = tableSize * hiddenSize + row * hiddenSize + idx;
                float posEmb = \(readWeight("table[xOffset]")) + \(readWeight("table[yOffset]"));
                data[idx] = \(bt)(float(data[idx]) + posEmb);
            }
            """
        }
    }

    public func requiredFallbackBufferSize(for role: String, bytesPerScalar: Int) -> Int {
        2 * tableSize * hiddenSize * bytesPerScalar
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let (tableBuffer, tableOffset) = context.resolveWeight("position_embedding_table")
        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.hidden, 0),
                (1, tableBuffer, tableOffset),
            ],
            bytes: [
                uint32Binding(2, UInt32(hiddenSize)),
                uint32Binding(3, UInt32(tableSize)),
                uint32Binding(4, UInt32(gridWidth)),
            ],
            outputIsHidden: true,
            resetsProjectionIndex: true,
            writeBufferIndices: Set<Int>([0])
        )
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let (tableBuffer, tableOffset) = context.resolveWeight("position_embedding_table")
        let pipeline = try context.getPipeline(kernelName(context: context.kernelContext))
        let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let gridX = (hiddenSize + tgSize - 1) / tgSize
        return FragmentPrefillSteps(
            steps: [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: gridX, height: context.maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: tgSize, height: 1, depth: 1),
                bufferBindings: [
                    (0, context.buffers.hidden, 0),
                    (1, tableBuffer, tableOffset),
                ],
                bytesBindings: [
                    uint32Binding(2, UInt32(hiddenSize)),
                    uint32Binding(3, UInt32(tableSize)),
                    uint32Binding(4, UInt32(gridWidth)),
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
