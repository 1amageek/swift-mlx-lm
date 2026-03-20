import LMIR

struct ProjectionWeightAccessPolicyOverride: Sendable {
    struct Context: Sendable {
        let entry: DispatchEntry
        let role: String
        let binding: ParameterBinding
        let executionPhase: STAFWeightExecutionPhase
        let stafWeightStore: STAFWeightStore
        let schemeIdentifier: QuantizationSchemeIdentifier
        let weightFormat: WeightFormat?
        let defaultPreference: STAFWeightLayoutPreference
    }

    let resolveLayoutPreference: @Sendable (Context) -> STAFWeightLayoutPreference?

    init(
        resolveLayoutPreference: @escaping @Sendable (Context) -> STAFWeightLayoutPreference?
    ) {
        self.resolveLayoutPreference = resolveLayoutPreference
    }

    func layoutPreference(for context: Context) -> STAFWeightLayoutPreference? {
        resolveLayoutPreference(context)
    }
}

extension ProjectionWeightAccessPolicyOverride {
    static func prefer(
        _ preference: STAFWeightLayoutPreference,
        forTensorNames tensorNames: Set<String>,
        executionPhase: STAFWeightExecutionPhase = .decode
    ) -> Self {
        Self { context in
            guard context.executionPhase == executionPhase else {
                return nil
            }
            guard tensorNames.contains(context.binding.tensorName) else {
                return nil
            }
            return preference
        }
    }
}
