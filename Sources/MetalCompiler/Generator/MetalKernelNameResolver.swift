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
                switch info.format.schemeIdentifier {
                case .bf16RowMajor:
                    return .bfloat16
                case .fp16RowMajor:
                    return .float16
                default:
                    continue
                }
            }
        }
        return .float16
    }

    func preferredDecodeBufferPrecision(for weightFormat: WeightFormat) -> BufferPrecision {
        switch weightFormat {
        case .bfloat16:
            return .bfloat16
        case .float32:
            return .float32
        case .float16,
             .quantized2Bit, .quantized3Bit, .quantized4Bit, .quantized5Bit, .quantized6Bit, .quantized8Bit:
            return .float16
        }
    }

    func resolvedInput2048SourcePolicy(
        for linear: LinearFragment,
        entry: DispatchEntry,
        role: String,
        weightFormat: WeightFormat,
        accessPolicyResolver: ProjectionWeightAccessPolicyResolver
    ) -> Input2048GEMVSourcePolicy? {
        guard
            linear.inputDimension == 2_048,
            let defaultPolicy = Self.defaultInput2048SourcePolicy(
                outputDimension: linear.outputDimension,
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

        if let projection = entry.fragment as? LinearFragment {
            if let binding = entry.parameterBindings.first(where: { $0.role == projection.field }),
               let stafWeightStore,
               let tensorInfo = stafWeightStore.tensor(for: binding.tensorName) {
                if isPrefill,
                   projection.isOutput,
                   projection.field == "weight",
                   projection.outputDimension > projection.inputDimension,
                   (
                    tensorInfo.format.schemeIdentifier == .fp16RowMajor
                    || tensorInfo.format.schemeIdentifier == .bf16RowMajor
                    || tensorInfo.format.schemeIdentifier == .fp32RowMajor
                   ) {
                    return tensorInfo.format.schemeIdentifier == .bf16RowMajor
                        ? "gemv_bf16_f32s"
                        : "gemv_f32s"
                }
                if !isPrefill,
                   let family = Self.denseDecodeProjectionFamily(
                    outputDimension: projection.outputDimension,
                    inputDimension: projection.inputDimension,
                    schemeIdentifier: tensorInfo.format.schemeIdentifier
                   ) {
                    let weightFormat: WeightFormat = tensorInfo.format.schemeIdentifier == .bf16RowMajor
                        ? .bfloat16
                        : (tensorInfo.format.schemeIdentifier == .fp32RowMajor ? .float32 : .float16)
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

            if isPrefill,
               projection.isOutput,
               projection.field == "weight",
               projection.outputDimension > projection.inputDimension {
                return isBF16 ? "gemv_bf16_f32s" : "gemv_f32s"
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
        }

        let fragment = entry.fragment
        let fragmentWeightFormat = weightFormatResolver.resolve(
            forFragment: fragment,
            entry: entry
        )
        let fragmentContext = KernelContext(
            bufferPrecision: kernelContext.bufferPrecision,
            weightFormat: fragmentWeightFormat
        )
        return fragment.kernelName(context: fragmentContext)
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

    private func resolveWeightFormat(
        forRole role: String,
        entry: DispatchEntry,
        fallback: WeightFormat
    ) -> WeightFormat {
        guard
            let stafWeightStore,
            let binding = entry.parameterBindings.first(where: { $0.role == role }),
            let tensorInfo = stafWeightStore.tensor(for: binding.tensorName)
        else {
            return fallback
        }
        switch tensorInfo.format.schemeIdentifier {
        case .bf16RowMajor:
            return .bfloat16
        case .fp32RowMajor:
            return .float32
        default:
            return .float16
        }
    }
}
