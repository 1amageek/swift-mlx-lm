/// Runtime quantization of KV cache entries.
public struct QuantizeKVFragment: PrimitiveMetalKernelFragment {
    public let totalElements: Int
    public let groupSize: Int
    public let bytesPerBlock: Int

    public init(totalElements: Int, groupSize: Int, bytesPerBlock: Int) {
        self.totalElements = totalElements
        self.groupSize = groupSize
        self.bytesPerBlock = bytesPerBlock
    }

    public var isFusable: Bool { false }
    public func kernelName(context: KernelContext) -> String { "quantize_kv" }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: totalElements / groupSize) }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        fatalError("[Compiler] QuantizeKVFragment decode bindings not yet implemented")
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        fatalError("[Compiler] QuantizeKVFragment prefill steps not yet implemented")
    }
}
