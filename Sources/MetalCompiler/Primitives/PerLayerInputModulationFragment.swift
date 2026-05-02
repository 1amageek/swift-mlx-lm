import Metal
import LMIR

/// Gemma 4 per-layer input modulation.
///
/// This consumes the gate projection output and multiplies it with the
/// precomputed per-layer input vector for the current layer.
public struct PerLayerInputModulationFragment: PrimitiveMetalKernelFragment {
    public let dimension: Int
    public let activation: ActivationKind

    public init(dimension: Int, activation: ActivationKind) {
        self.dimension = dimension
        self.activation = activation
    }

    public var isFusable: Bool { false }
    public func kernelName(context: KernelContext) -> String {
        context.bufferPrecision.isPrefillSequencePrecision
            ? "per_layer_input_modulation_seq_f32"
            : "per_layer_input_modulation\(context.bufferPrecision.decodeKernelNameSuffix)"
    }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: dimension) }

    public func kernelSource(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        MetalSourceGenerator.generatePerLayerInputModulation(
            name: name,
            bufferPrecision: bufferPrecision,
            activation: activation,
            isSequence: bufferPrecision.isPrefillSequencePrecision
        )
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        guard let layerIndex = context.layerIndex else {
            fatalError("[Compiler] PerLayerInputModulationFragment requires layer index")
        }
        guard let perLayerInputs = context.bufferSet.perLayerInputs else {
            fatalError("[Compiler] PerLayerInputModulationFragment requires per-layer input buffer")
        }
        let slotBytes = context.slotDimension * context.elementSize
        let perLayerOffset = layerIndex * context.bufferSet.perLayerInputDimension * MemoryLayout<Float>.size
        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.scratch, slotBytes),
                (1, perLayerInputs, perLayerOffset),
                (2, context.bufferSet.scratch, 0),
            ],
            bytes: [
                uint32Binding(3, UInt32(dimension)),
            ],
            outputIsHidden: false,
            resetsProjectionIndex: true,
            writeBufferIndices: Set<Int>([2])
        )
    }

    public func requiredFallbackBufferSize(for role: String, bytesPerScalar: Int) -> Int {
        dimension * bytesPerScalar
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        guard let layerIndex = context.layerIndex else {
            fatalError("[Compiler] PerLayerInputModulationFragment requires layer index")
        }
        guard let perLayerInputs = context.buffers.perLayerInputs else {
            fatalError("[Compiler] PerLayerInputModulationFragment requires per-layer input buffer")
        }
        let kernelName = kernelName(context: context.kernelContext)
        let pipeline = try context.getPipeline(kernelName)
        let scratchSlotSize = context.slotDimension * context.scratchElementSize * context.maximumSequenceLength
        let perLayerOffset = layerIndex
            * context.maximumSequenceLength
            * context.buffers.perLayerInputDimension
            * MemoryLayout<Float>.size
        let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let gridX = (dimension + tgSize - 1) / tgSize
        return FragmentPrefillSteps(
            steps: [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: gridX, height: context.maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: tgSize, height: 1, depth: 1),
                bufferBindings: [
                    (0, context.buffers.scratch, scratchSlotSize),
                    (1, perLayerInputs, perLayerOffset),
                    (2, context.buffers.scratch, 0),
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
            outputIsHidden: false,
            resetsProjectionIndex: true
        )
    }
}

extension PerLayerInputModulationFragment: PerLayerInputCapable {
    public var perLayerInputDimension: Int { dimension }
}
