import Metal

/// Sigmoid-gated element-wise operation.
public struct SigmoidGateFragment: PrimitiveMetalKernelFragment {
    public let dimension: Int

    public init(dimension: Int) {
        self.dimension = dimension
    }

    public var isFusable: Bool { true }

    // MARK: - Fusion Contract

    /// Fusion contract with two buffer inputs.
    ///
    /// Note: SynthesizedFragment.decodeBindings binds all `.dataFlow` input
    /// ports to `currentInputBuffer`. With two buffer inputs, the non-primary
    /// input would be bound incorrectly if this fragment were fused as a
    /// consumer. Currently unused in production (PackedSigmoidGateFragment
    /// is used instead).
    public var fusionContract: FusionContract? {
        FusionContract(
            ports: [
                FusionPort(name: "input", direction: .input, role: .buffer, accessPattern: .singlePass),
                FusionPort(name: "gate", direction: .input, role: .buffer, accessPattern: .singlePass),
                FusionPort(name: "output", direction: .output, role: .buffer),
            ],
            parallelism: .perElement(count: dimension)
        )
    }

    public func kernelBody(bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String? {
        let bt = bufferPrecision.metalType
        let value = "float(input[idx]) * (1.0f / (1.0f + exp(-g)))"
        let stored = bufferPrecision.isPrefillSequencePrecision
            ? MetalSourceGenerator.sequenceStorageValue(value, weightFormat: weightFormat)
            : value
        return """
        float g = float(gate[idx]);
        output[idx] = \(bt)(\(stored));
        """
    }

    public func kernelName(context: KernelContext) -> String { "sigmoid_gate" }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: dimension) }

    public func kernelSource(name: String, bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String {
        MetalSourceGenerator.generateSigmoidGate(name: name, bufferPrecision: bufferPrecision)
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let slotBytes = context.slotDimension * context.elementSize
        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.scratch, 0),
                (1, context.bufferSet.scratch, 1 * slotBytes),
                (2, context.bufferSet.scratch, 0),
            ],
            bytes: [
                uint32Binding(3, UInt32(dimension)),
            ],
            outputIsHidden: false,
            writeBufferIndices: Set<Int>([2])
        )
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        fatalError("[Compiler] SigmoidGateFragment prefill steps not used in prefill currently")
    }
}
