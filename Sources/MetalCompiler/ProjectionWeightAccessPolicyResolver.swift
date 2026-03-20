import LMIR

struct ProjectionWeightAccessPolicyResolver: Sendable {
    private let override: ProjectionWeightAccessPolicyOverride?

    init(override: ProjectionWeightAccessPolicyOverride? = nil) {
        self.override = override
    }

    func accessRequest(
        for entry: DispatchEntry,
        role: String,
        binding: ParameterBinding,
        executionPhase: STAFWeightExecutionPhase,
        stafWeightStore: STAFWeightStore
    ) -> STAFWeightAccessRequest {
        STAFWeightAccessRequest(
            tensorName: binding.tensorName,
            executionPhase: executionPhase,
            layoutPreference: layoutPreference(
                for: entry,
                role: role,
                binding: binding,
                executionPhase: executionPhase,
                stafWeightStore: stafWeightStore
            )
        )
    }

    func layoutPreference(
        for entry: DispatchEntry,
        role: String,
        binding: ParameterBinding,
        executionPhase: STAFWeightExecutionPhase,
        stafWeightStore: STAFWeightStore
    ) -> STAFWeightLayoutPreference {
        guard role == binding.role else {
            return .canonicalRowMajor
        }
        guard case .projection(let projection, _) = entry.kind,
              projection.field == role else {
            return .canonicalRowMajor
        }

        let schemeIdentifier = stafWeightStore
            .tensor(for: binding.tensorName)?
            .format
            .schemeIdentifier ?? .fp16RowMajor
        guard schemeIdentifier == .fp16RowMajor || schemeIdentifier == .bf16RowMajor else {
            return .canonicalRowMajor
        }

        let weightFormat: WeightFormat = schemeIdentifier == .bf16RowMajor
            ? .bfloat16
            : .float16

        let layoutPolicy: Input2048WeightLayoutPolicy?
        if projection.inputDimension == 2_048 && projection.outputDimension == 2_048 {
            layoutPolicy = Input2048GEMVSourcePolicy.square(weightFormat: weightFormat).weightLayoutPolicy
        } else if projection.inputDimension == 2_048 && projection.outputDimension == 6_144 {
            layoutPolicy = Input2048GEMVSourcePolicy.expanded6144(weightFormat: weightFormat).weightLayoutPolicy
        } else if projection.inputDimension == 2_048 && projection.outputDimension == 8_192 {
            layoutPolicy = Input2048GEMVSourcePolicy.expanded8192(weightFormat: weightFormat).weightLayoutPolicy
        } else {
            layoutPolicy = nil
        }

        let defaultPreference: STAFWeightLayoutPreference
        if let layoutPolicy {
            let layout = layoutPolicy.stafWeightLayout
            if layout == .rowMajor {
                defaultPreference = .canonicalRowMajor
            } else {
                defaultPreference = .optimized(layout)
            }
        } else {
            defaultPreference = .canonicalRowMajor
        }

        guard let override else {
            return defaultPreference
        }
        return override.layoutPreference(for: .init(
            entry: entry,
            role: role,
            binding: binding,
            executionPhase: executionPhase,
            stafWeightStore: stafWeightStore,
            schemeIdentifier: schemeIdentifier,
            weightFormat: weightFormat,
            defaultPreference: defaultPreference
        )) ?? defaultPreference
    }
}
