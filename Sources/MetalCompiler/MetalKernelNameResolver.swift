struct MetalKernelNameResolver {
    let stafWeightStore: STAFWeightStore?
    let weightAccessPolicyOverride: ProjectionWeightAccessPolicyOverride?

    static func argumentTableVariantKernelName(for kernelName: String) -> String {
        kernelName + "_argbuf"
    }

    func resolveModelWeightFormat() -> WeightFormat {
        guard let stafWeightStore else { return .float16 }
        for name in stafWeightStore.entries.keys {
            if let info = stafWeightStore.tensor(for: name) {
                return info.format.schemeIdentifier == .bf16RowMajor ? .bfloat16 : .float16
            }
        }
        return .float16
    }

    func preferredDecodeBufferPrecision(for weightFormat: WeightFormat) -> BufferPrecision {
        _ = weightFormat
        return .float16
    }

    func resolvedInput2048SourcePolicy(
        for projection: MetalProjection,
        entry: DispatchEntry,
        role: String,
        weightFormat: WeightFormat,
        accessPolicyResolver: ProjectionWeightAccessPolicyResolver
    ) -> Input2048GEMVSourcePolicy? {
        guard
            projection.inputDimension == 2_048,
            let defaultPolicy = Self.defaultInput2048SourcePolicy(
                outputDimension: projection.outputDimension,
                weightFormat: weightFormat
            )
        else {
            return nil
        }
        guard
            let stafWeightStore,
            let binding = entry.parameterBindings.first(where: { $0.role == role })
        else {
            return defaultPolicy
        }

        let request = accessPolicyResolver.accessRequest(
            for: entry,
            role: role,
            binding: binding,
            executionPhase: .decode,
            stafWeightStore: stafWeightStore
        )
        let resolvedLayout = stafWeightStore
            .resolvedBufferAccess(for: request)?
            .layout ?? request.preferredLayout
        let resolvedLayoutPolicy = Input2048WeightLayoutPolicy(
            stafWeightLayout: resolvedLayout
        )
        return defaultPolicy.with(weightLayoutPolicy: resolvedLayoutPolicy)
    }

    func kernelName(
        for entry: DispatchEntry,
        kernelContext: KernelContext
    ) -> String {
        let weightFormatResolver = KernelWeightFormatResolver(stafWeightStore: stafWeightStore)
        let isBF16 = kernelContext.weightFormat == .bfloat16
        let bf16Suffix = isBF16 ? "_bf16" : ""
        let isPrefill = kernelContext.bufferPrecision == .float32
        let accessPolicyResolver = ProjectionWeightAccessPolicyResolver(
            override: weightAccessPolicyOverride
        )

        switch entry.kind {
        case .projection(let projection, _):
            if let binding = entry.parameterBindings.first(where: { $0.role == projection.field }),
               let stafWeightStore,
               let tensorInfo = stafWeightStore.tensor(for: binding.tensorName) {
                if !isPrefill,
                   let family = Self.denseDecodeProjectionFamily(
                    outputDimension: projection.outputDimension,
                    inputDimension: projection.inputDimension,
                    schemeIdentifier: tensorInfo.format.schemeIdentifier
                   ) {
                    let weightFormat: WeightFormat = tensorInfo.format.schemeIdentifier == .bf16RowMajor
                        ? .bfloat16
                        : .float16
                    if let sourcePolicy = resolvedInput2048SourcePolicy(
                        for: projection,
                        entry: entry,
                        role: projection.field,
                        weightFormat: weightFormat,
                        accessPolicyResolver: accessPolicyResolver
                    ) {
                        let baseName = family.kernelBaseName + sourcePolicy.weightLayoutPolicy.kernelNameSuffix
                        return tensorInfo.format.schemeIdentifier == .bf16RowMajor
                            ? baseName + "_bf16"
                            : baseName
                    }
                    return tensorInfo.format.schemeIdentifier == .bf16RowMajor
                        ? family.kernelBaseName + "_bf16"
                        : family.kernelBaseName
                }
                return isPrefill
                    ? tensorInfo.format.gemmKernelName(bufferPrecision: kernelContext.bufferPrecision)
                    : tensorInfo.format.gemvKernelName
            }

            if !isPrefill,
               let family = Self.denseDecodeProjectionFamily(
                outputDimension: projection.outputDimension,
                inputDimension: projection.inputDimension,
                schemeIdentifier: isBF16 ? .bf16RowMajor : .fp16RowMajor
               ) {
                if let sourcePolicy = resolvedInput2048SourcePolicy(
                    for: projection,
                    entry: entry,
                    role: projection.field,
                    weightFormat: isBF16 ? .bfloat16 : .float16,
                    accessPolicyResolver: accessPolicyResolver
                ) {
                    let baseName = family.kernelBaseName + sourcePolicy.weightLayoutPolicy.kernelNameSuffix
                    return isBF16 ? baseName + "_bf16" : baseName
                }
                return isBF16 ? family.kernelBaseName + "_bf16" : family.kernelBaseName
            }
            return isPrefill ? (isBF16 ? "gemm_bf16_f32s" : "gemm_f32s") : "gemv"

        case .fragment(let fragment):
            let fragmentWeightFormat = weightFormatResolver.resolve(
                forFragment: fragment,
                entry: entry
            )
            let fragmentContext = KernelContext(
                bufferPrecision: kernelContext.bufferPrecision,
                weightFormat: fragmentWeightFormat
            )
            return fragment.kernelName(context: fragmentContext)

        case .fusedCopyNorm:
            let weightFormat = weightFormatResolver.resolve(role: "scale", entry: entry)
            return weightFormat == .bfloat16
                ? "fused_copy_rms_norm_bf16"
                : "fused_copy_rms_norm"
        case .fusedResidualAddCopyNorm:
            let weightFormat = weightFormatResolver.resolve(role: "scale", entry: entry)
            return weightFormat == .bfloat16
                ? "fused_residual_add_copy_rms_norm_bf16"
                : "fused_residual_add_copy_rms_norm"
        case .fusedResidualAddNorm:
            let weightFormat = weightFormatResolver.resolve(role: "scale", entry: entry)
            return weightFormat == .bfloat16
                ? "fused_residual_add_rms_norm_bf16"
                : "fused_residual_add_rms_norm"

        case .fusedSwiGLUProjection(let fused):
            let weightFormat = weightFormatResolver.resolve(role: fused.gateField, entry: entry)
            let family = FusedSwiGLUProjectionFamily.resolve(
                inputDimension: fused.inputDimension,
                outputDimension: fused.outputDimension
            )
            return weightFormat == .bfloat16 ? family.kernelBaseName + "_bf16" : family.kernelBaseName

        case .batchedProjection(let batched):
            return "batched_gemv\(batched.projections.count)" + bf16Suffix

        case .batchedFragment(let batch):
            let fragmentWeightFormat = weightFormatResolver.resolve(
                forFragment: batch.fragments[0],
                entry: entry
            )
            let fragmentContext = KernelContext(
                bufferPrecision: kernelContext.bufferPrecision,
                weightFormat: fragmentWeightFormat
            )
            let baseName = batch.fragments[0].kernelName(context: fragmentContext)
            return "batched_\(baseName)_\(batch.fragments.count)"

        case .structuralCopy:
            return isPrefill ? "copy_buffer_seq_f32" : "copy_buffer"
        case .structuralAdd:
            return isPrefill ? "residual_add_seq_f32" : "residual_add"
        }
    }

    private static func denseDecodeProjectionFamily(
        outputDimension: Int,
        inputDimension: Int,
        schemeIdentifier: QuantizationSchemeIdentifier
    ) -> DecodeProjectionShapeFamily? {
        guard schemeIdentifier == .fp16RowMajor || schemeIdentifier == .bf16RowMajor else {
            return nil
        }
        return DecodeProjectionShapeFamily.resolve(
            outputDimension: outputDimension,
            inputDimension: inputDimension
        )
    }

    private static func defaultInput2048SourcePolicy(
        outputDimension: Int,
        weightFormat: WeightFormat
    ) -> Input2048GEMVSourcePolicy? {
        switch outputDimension {
        case 2_048:
            return .square(weightFormat: weightFormat)
        case 6_144:
            return .expanded6144(weightFormat: weightFormat)
        case 8_192:
            return .expanded8192(weightFormat: weightFormat)
        default:
            return nil
        }
    }
}
