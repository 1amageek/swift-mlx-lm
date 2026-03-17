/// Matrix-vector (decode) or matrix-matrix (prefill) multiply.
public struct LinearFragment: PrimitiveMetalKernelFragment {
    public let field: String
    public let inputDimension: Int
    public let outputDimension: Int

    public init(field: String, inputDimension: Int, outputDimension: Int) {
        self.field = field
        self.inputDimension = inputDimension
        self.outputDimension = outputDimension
    }

    public var isFusable: Bool { false }
    public func kernelName(context: KernelContext) -> String { "gemv" }
    public var dispatchDimension: MetalDispatchDimension {
        .gemv(outputDimension: outputDimension, inputDimension: inputDimension)
    }
    public var weightSlots: [MetalWeightSlot] { [MetalWeightSlot(field: field, role: .weight)] }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        fatalError("[Compiler] LinearFragment is dispatched via .projection, not .fragment")
    }
}
