/// DeltaNet/Mamba state-space model recurrence step.
public struct SSMRecurrenceFragment: PrimitiveMetalKernelFragment {
    public let headCount: Int
    public let keyHeadDimension: Int
    public let valueHeadDimension: Int

    public init(headCount: Int, keyHeadDimension: Int, valueHeadDimension: Int) {
        self.headCount = headCount
        self.keyHeadDimension = keyHeadDimension
        self.valueHeadDimension = valueHeadDimension
    }

    public var isFusable: Bool { false }
    public func kernelName(context: KernelContext) -> String { "ssm_recurrence" }
    public var dispatchDimension: MetalDispatchDimension { .perHead(headCount: headCount) }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        fatalError("[Compiler] SSMRecurrenceFragment decode bindings not yet implemented")
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        fatalError("[Compiler] SSMRecurrenceFragment prefill steps not yet implemented")
    }
}
