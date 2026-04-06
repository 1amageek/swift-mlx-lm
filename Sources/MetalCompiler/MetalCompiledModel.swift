import Metal

/// Opaque compiled runtime artifact produced by `MetalInferenceCompiler`.
///
/// This bundles the decode plan together with the optional prefill plan so
/// higher layers do not need to wire the two artifacts manually.
public struct MetalCompiledModel: @unchecked Sendable {
    public let decodePlan: MetalDispatchPlan
    public let prefillPlan: MetalPrefillPlan?
    public let auxiliaryPipelines: [String: MTLComputePipelineState]

    public init(
        decodePlan: MetalDispatchPlan,
        prefillPlan: MetalPrefillPlan? = nil,
        auxiliaryPipelines: [String: MTLComputePipelineState] = [:]
    ) {
        self.decodePlan = decodePlan
        self.prefillPlan = prefillPlan
        self.auxiliaryPipelines = auxiliaryPipelines
    }

    public var steps: [MetalDispatchStep] { decodePlan.steps }
    public var buffers: MetalBufferSet { decodePlan.buffers }
    public var unfusedEntryCount: Int { decodePlan.unfusedEntryCount }
    public var fusedEntryCount: Int { decodePlan.fusedEntryCount }

    public func withPrefillPlan(_ prefillPlan: MetalPrefillPlan?) -> Self {
        Self(
            decodePlan: decodePlan,
            prefillPlan: prefillPlan,
            auxiliaryPipelines: auxiliaryPipelines
        )
    }
}
