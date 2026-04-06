import Metal

struct KernelWeightFormatResolver {
    let stafWeightStore: STAFWeightStore?

    func resolve(role: String, entry: DispatchEntry) -> MetalSourceGenerator.WeightFormat {
        guard let staf = stafWeightStore,
              let binding = entry.parameterBindings.first(where: { $0.role == role }),
              let info = staf.tensor(for: binding.tensorName) else { return .float16 }
        return info.format.schemeIdentifier == .bf16RowMajor ? .bfloat16 : .float16
    }

    func resolve(forFragment fragment: any PrimitiveMetalKernelFragment, entry: DispatchEntry) -> MetalSourceGenerator.WeightFormat {
        let slotRoles = fragment.weightSlots.compactMap { slot -> String? in
            if let field = slot.field {
                return field
            }
            switch slot.role {
            case .weight:
                return "weight"
            case .scale:
                return "scale"
            case .embeddingTable:
                return "embedding_table"
            }
        }
        let roles = slotRoles + ["scale", "embedding_table", "conv_weight"]
        for role in roles {
            let format = resolve(role: role, entry: entry)
            if format == .bfloat16 {
                return .bfloat16
            }
        }
        return .float16
    }
}

struct GeneratedKernelSources {
    let baseSource: String
    let mppSources: [String]
    let mppKernelNames: Set<String>
}

struct MetalKernelSourceCatalog {
    let stafWeightStore: STAFWeightStore?
    let modelWeightFormat: WeightFormat
    let bufferPrecision: MetalSourceGenerator.BufferPrecision
    let accessPolicyResolver: ProjectionWeightAccessPolicyResolver
    let kernelNameResolver: MetalKernelNameResolver

    func generateSources(entries: [DispatchEntry]) -> GeneratedKernelSources {
        let weightFormatResolver = KernelWeightFormatResolver(stafWeightStore: stafWeightStore)
        var sources: [String] = [MetalSourceGenerator.commonHeader]
        var generatedNames: Set<String> = []
        var mppGEMMNames: Set<String> = []
        var mppSources: [String] = []
        var mppGEMMWeightFormat: MetalSourceGenerator.WeightFormat = .float16
        var needsFlashAttnHelper = false
        var needsSSMHelpers = false
        var ssmConvSiluWeightFormats: [MetalSourceGenerator.WeightFormat] = []

        for entry in entries {
            let name: String
            switch entry.kind {
            case .projection(let projection, _):
                let weightFormat = weightFormatResolver.resolve(role: projection.field, entry: entry)
                let decodeFamily = bufferPrecision == .float32
                    ? nil
                    : DecodeProjectionShapeFamily.resolve(
                        outputDimension: projection.outputDimension,
                        inputDimension: projection.inputDimension
                    )
                if let decodeFamily {
                    if let sourcePolicy = kernelNameResolver.resolvedInput2048SourcePolicy(
                        for: projection,
                        entry: entry,
                        role: projection.field,
                        weightFormat: weightFormat,
                        accessPolicyResolver: accessPolicyResolver
                    ) {
                        let baseName = decodeFamily.kernelBaseName + sourcePolicy.weightLayoutPolicy.kernelNameSuffix
                        name = weightFormat == .bfloat16 ? baseName + "_bf16" : baseName
                    } else {
                        name = weightFormat == .bfloat16
                            ? decodeFamily.kernelBaseName + "_bf16"
                            : decodeFamily.kernelBaseName
                    }
                } else {
                    name = weightFormat == .bfloat16 ? "gemv_bf16" : "gemv"
                }
                let isSequenceKernel = bufferPrecision == .float32
                let emittedName = isSequenceKernel
                    ? name.replacingOccurrences(of: "gemv", with: "gemm") + "_f32s"
                    : name
                if generatedNames.insert(emittedName).inserted {
                    if isSequenceKernel {
                        let gemmName = weightFormat == .bfloat16 ? "gemm_bf16_f32s" : "gemm_f32s"
                        mppGEMMNames.insert(gemmName)
                        mppGEMMWeightFormat = weightFormat
                        sources.append(MetalSourceGenerator.generateGEMM(
                            name: gemmName,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat))
                    } else if decodeFamily == .vocabDense {
                        sources.append(MetalSourceGenerator.generateVocabGEMV(
                            name: name,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat))
                        let argumentKernelName = MetalKernelNameResolver.argumentTableVariantKernelName(for: name)
                        if generatedNames.insert(argumentKernelName).inserted {
                            sources.append(MetalSourceGenerator.generateVocabGEMVArgumentTableVariant(
                                name: argumentKernelName,
                                argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat))
                        }
                    } else if decodeFamily == .input8192Tiled {
                        sources.append(MetalSourceGenerator.generateInput8192TiledGEMV(
                            name: name,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat,
                            tileElements: 1_024,
                            unrollFactor: 4))
                        let argumentKernelName = MetalKernelNameResolver.argumentTableVariantKernelName(for: name)
                        if generatedNames.insert(argumentKernelName).inserted {
                            sources.append(MetalSourceGenerator.generateInput8192TiledGEMVArgumentTableVariant(
                                name: argumentKernelName,
                                argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat,
                                tileElements: 1_024,
                                unrollFactor: 4))
                        }
                    } else if decodeFamily == .input2048SquareDense {
                        let sourcePolicy = kernelNameResolver.resolvedInput2048SourcePolicy(
                            for: projection,
                            entry: entry,
                            role: projection.field,
                            weightFormat: weightFormat,
                            accessPolicyResolver: accessPolicyResolver
                        ) ?? Input2048GEMVSourcePolicy.square(weightFormat: weightFormat)
                        sources.append(MetalSourceGenerator.generateInput2048GEMV(
                            name: name,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat,
                            fixedOutputDimension: sourcePolicy.fixedOutputDimension,
                            fixedRowsPerThreadgroup: sourcePolicy.fixedRowsPerThreadgroup,
                            stagesInputAsFloat: sourcePolicy.stagesInputAsFloat,
                            weightLayoutPolicy: sourcePolicy.weightLayoutPolicy,
                            unrollFactor: sourcePolicy.unrollFactor))
                        let argumentKernelName = MetalKernelNameResolver.argumentTableVariantKernelName(for: name)
                        if generatedNames.insert(argumentKernelName).inserted {
                            sources.append(MetalSourceGenerator.generateInput2048GEMVArgumentTableVariant(
                                name: argumentKernelName,
                                argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat,
                                fixedOutputDimension: sourcePolicy.fixedOutputDimension,
                                includesDimensionBindings: false,
                                fixedRowsPerThreadgroup: sourcePolicy.fixedRowsPerThreadgroup,
                                fixedSimdgroups: sourcePolicy.fixedSimdgroups,
                                stagesInputAsFloat: sourcePolicy.stagesInputAsFloat,
                                weightLayoutPolicy: sourcePolicy.weightLayoutPolicy,
                                bf16ArgumentReadPolicy: sourcePolicy.bf16ArgumentReadPolicy,
                                unrollFactor: sourcePolicy.unrollFactor))
                        }
                    } else if decodeFamily == .input20486144Dense {
                        let sourcePolicy = kernelNameResolver.resolvedInput2048SourcePolicy(
                            for: projection,
                            entry: entry,
                            role: projection.field,
                            weightFormat: weightFormat,
                            accessPolicyResolver: accessPolicyResolver
                        ) ?? Input2048GEMVSourcePolicy.expanded6144(weightFormat: weightFormat)
                        sources.append(MetalSourceGenerator.generateInput2048GEMV(
                            name: name,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat,
                            fixedOutputDimension: sourcePolicy.fixedOutputDimension,
                            fixedRowsPerThreadgroup: sourcePolicy.fixedRowsPerThreadgroup,
                            stagesInputAsFloat: sourcePolicy.stagesInputAsFloat,
                            weightLayoutPolicy: sourcePolicy.weightLayoutPolicy,
                            unrollFactor: sourcePolicy.unrollFactor))
                        let argumentKernelName = MetalKernelNameResolver.argumentTableVariantKernelName(for: name)
                        if generatedNames.insert(argumentKernelName).inserted {
                            sources.append(MetalSourceGenerator.generateInput2048GEMVArgumentTableVariant(
                                name: argumentKernelName,
                                argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat,
                                fixedOutputDimension: sourcePolicy.fixedOutputDimension,
                                includesDimensionBindings: false,
                                fixedRowsPerThreadgroup: sourcePolicy.fixedRowsPerThreadgroup,
                                fixedSimdgroups: sourcePolicy.fixedSimdgroups,
                                stagesInputAsFloat: sourcePolicy.stagesInputAsFloat,
                                weightLayoutPolicy: sourcePolicy.weightLayoutPolicy,
                                bf16ArgumentReadPolicy: sourcePolicy.bf16ArgumentReadPolicy,
                                unrollFactor: sourcePolicy.unrollFactor))
                        }
                    } else if decodeFamily == .input20488192Dense {
                        let sourcePolicy = kernelNameResolver.resolvedInput2048SourcePolicy(
                            for: projection,
                            entry: entry,
                            role: projection.field,
                            weightFormat: weightFormat,
                            accessPolicyResolver: accessPolicyResolver
                        ) ?? Input2048GEMVSourcePolicy.expanded8192(weightFormat: weightFormat)
                        sources.append(MetalSourceGenerator.generateInput2048GEMV(
                            name: name,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat,
                            fixedOutputDimension: sourcePolicy.fixedOutputDimension,
                            fixedRowsPerThreadgroup: sourcePolicy.fixedRowsPerThreadgroup,
                            stagesInputAsFloat: sourcePolicy.stagesInputAsFloat,
                            weightLayoutPolicy: sourcePolicy.weightLayoutPolicy,
                            unrollFactor: sourcePolicy.unrollFactor))
                    } else if decodeFamily == .input2048ExpandedDense {
                        sources.append(MetalSourceGenerator.generateInput2048GEMV(
                            name: name,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat,
                            unrollFactor: 4))
                    } else {
                        sources.append(MetalSourceGenerator.generateGEMV(
                            name: name,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat,
                            tileElements: decodeFamily?.tileElements ?? 256))
                        let argumentKernelName = MetalKernelNameResolver.argumentTableVariantKernelName(for: name)
                        if generatedNames.insert(argumentKernelName).inserted {
                            sources.append(MetalSourceGenerator.generateGEMVArgumentTableVariant(
                                name: argumentKernelName,
                                argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat,
                                tileElements: decodeFamily?.tileElements ?? 256))
                        }
                    }
                }

            case .fragment(let fragment):
                let weightFormat = weightFormatResolver.resolve(forFragment: fragment, entry: entry)
                let fragmentContext = KernelContext(bufferPrecision: bufferPrecision, weightFormat: weightFormat)
                let kernelName = fragment.kernelName(context: fragmentContext)
                if generatedNames.insert(kernelName).inserted {
                    let source = fragment.kernelSource(
                        name: kernelName,
                        bufferPrecision: bufferPrecision,
                        weightFormat: weightFormat)
                    if fragment.cacheSlots.contains(where: { $0.kind == .kv }) {
                        needsFlashAttnHelper = true
                    }
                    sources.append(source)
                    if fragment is SSMRecurrenceFragment {
                        needsSSMHelpers = true
                        if !ssmConvSiluWeightFormats.contains(where: { $0 == weightFormat }) {
                            ssmConvSiluWeightFormats.append(weightFormat)
                        }
                    }
                    if bufferPrecision == .float32, let ssmFragment = fragment as? SSMRecurrenceFragment {
                        let sequenceKernelName = "ssm_recurrence_seq_f32"
                        if generatedNames.insert(sequenceKernelName).inserted {
                            sources.append(MetalSourceGenerator.generateSSMRecurrenceSequence(
                                name: sequenceKernelName,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat,
                                convDimension: ssmFragment.convDimension,
                                maxThreadgroupSize: SSMRecurrenceFragment.maxThreadgroupSize))
                        }
                    }
                    if bufferPrecision != .float32 {
                        let argumentKernelName = MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
                        if argumentKernelName != kernelName, generatedNames.insert(argumentKernelName).inserted {
                            switch kernelName {
                            case "embedding_lookup", "embedding_lookup_bf16":
                                sources.append(MetalSourceGenerator.generateEmbeddingLookupArgumentTableVariant(
                                    name: argumentKernelName,
                                    argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                    bufferPrecision: bufferPrecision,
                                    weightFormat: weightFormat))
                            case "argmax":
                                sources.append(MetalSourceGenerator.generateArgmaxArgumentTableVariant(
                                    name: argumentKernelName,
                                    argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                    bufferPrecision: bufferPrecision))
                            case "rms_norm", "rms_norm_bf16":
                                sources.append(MetalSourceGenerator.generateReductionArgumentTableVariant(
                                    name: argumentKernelName,
                                    argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                    bufferPrecision: bufferPrecision,
                                    weightFormat: weightFormat))
                            case "qk_rms_norm", "qk_rms_norm_bf16":
                                sources.append(MetalSourceGenerator.generateQKNormArgumentTableVariant(
                                    name: argumentKernelName,
                                    argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                    bufferPrecision: bufferPrecision,
                                    weightFormat: weightFormat))
                            case "rope":
                                sources.append(MetalSourceGenerator.generateRoPEArgumentTableVariant(
                                    name: argumentKernelName,
                                    argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                    bufferPrecision: bufferPrecision))
                            case "flash_attn_decode":
                                sources.append(MetalSourceGenerator.generateFlashAttentionArgumentTableVariant(
                                    name: argumentKernelName,
                                    argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                    bufferPrecision: bufferPrecision))
                            case "conv_state_update_bf16", "conv_state_update":
                                sources.append(MetalSourceGenerator.generateConvStateUpdateArgumentTableVariant(
                                    name: argumentKernelName,
                                    argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                    bufferPrecision: bufferPrecision,
                                    weightFormat: weightFormat))
                            default:
                                break
                            }
                        }
                    }
                }
                if fragment.cacheSlots.contains(where: { $0.kind == .conv }) && bufferPrecision == .float32 {
                    let extractName = "extract_conv_state_f32"
                    if generatedNames.insert(extractName).inserted {
                        sources.append(MetalSourceGenerator.generateExtractConvState(
                            name: extractName,
                            bufferPrecision: bufferPrecision))
                    }
                }
                if fragment.cacheSlots.contains(where: { $0.kind == .kv }) && bufferPrecision == .float32 {
                    for helperName in ["kv_cache_fill_seq_f32", "flash_attn_batch_f32"] {
                        if generatedNames.insert(helperName).inserted {
                            if helperName.contains("kv_cache_fill") {
                                sources.append(MetalSourceGenerator.generateKVCacheFillSeq(
                                    name: helperName,
                                    bufferPrecision: bufferPrecision))
                            } else {
                                sources.append(MetalSourceGenerator.generateBatchFlashAttention(
                                    name: helperName,
                                    bufferPrecision: bufferPrecision))
                            }
                        }
                    }
                }

            case .fusedCopyNorm:
                let weightFormat = weightFormatResolver.resolve(role: "scale", entry: entry)
                if bufferPrecision != .float32 {
                    let kernelName = weightFormat == .bfloat16 ? "fused_copy_rms_norm_bf16" : "fused_copy_rms_norm"
                    if generatedNames.insert(kernelName).inserted {
                        sources.append(MetalSourceGenerator.generateFusedCopyRMSNorm(
                            name: kernelName,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat))
                        let argumentKernelName = MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
                        if generatedNames.insert(argumentKernelName).inserted {
                            sources.append(MetalSourceGenerator.generateFusedCopyRMSNormArgumentTableVariant(
                                name: argumentKernelName,
                                argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat))
                        }
                    }
                } else {
                    let copyName = "copy_buffer_seq_f32"
                    let normName = weightFormat == .bfloat16 ? "rms_norm_seq_bf16_f32_inplace" : "rms_norm_seq_f32_inplace"
                    if generatedNames.insert(copyName).inserted {
                        sources.append(MetalSourceGenerator.generateCopy(name: copyName, bufferPrecision: bufferPrecision))
                    }
                    if generatedNames.insert(normName).inserted {
                        sources.append(MetalSourceGenerator.generateReduction(
                            name: normName,
                            dimension: 0,
                            epsilon: 0,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat))
                    }
                }

            case .fusedResidualAddCopyNorm:
                let weightFormat = weightFormatResolver.resolve(role: "scale", entry: entry)
                if bufferPrecision != .float32 {
                    let kernelName = weightFormat == .bfloat16 ? "fused_residual_add_copy_rms_norm_bf16" : "fused_residual_add_copy_rms_norm"
                    if generatedNames.insert(kernelName).inserted {
                        sources.append(MetalSourceGenerator.generateFusedResidualAddCopyRMSNorm(
                            name: kernelName,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat))
                        let argumentKernelName = MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
                        if generatedNames.insert(argumentKernelName).inserted {
                            sources.append(MetalSourceGenerator.generateFusedResidualAddCopyRMSNormArgumentTableVariant(
                                name: argumentKernelName,
                                argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat))
                        }
                    }
                } else {
                    let addName = "residual_add_seq_f32"
                    let copyName = "copy_buffer_seq_f32"
                    let normName = weightFormat == .bfloat16 ? "rms_norm_seq_bf16_f32_inplace" : "rms_norm_seq_f32_inplace"
                    if generatedNames.insert(addName).inserted {
                        sources.append(MetalSourceGenerator.generateResidualAdd(name: addName, bufferPrecision: bufferPrecision))
                    }
                    if generatedNames.insert(copyName).inserted {
                        sources.append(MetalSourceGenerator.generateCopy(name: copyName, bufferPrecision: bufferPrecision))
                    }
                    if generatedNames.insert(normName).inserted {
                        sources.append(MetalSourceGenerator.generateReduction(
                            name: normName,
                            dimension: 0,
                            epsilon: 0,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat))
                    }
                }

            case .structuralCopy:
                let kernelName = bufferPrecision == .float32 ? "copy_buffer_seq_f32" : "copy_buffer"
                if generatedNames.insert(kernelName).inserted {
                    sources.append(MetalSourceGenerator.generateCopy(
                        name: kernelName,
                        bufferPrecision: bufferPrecision,
                        isSequence: bufferPrecision == .float32))
                }

            case .structuralAdd:
                let kernelName = bufferPrecision == .float32 ? "residual_add_seq_f32" : "residual_add"
                if generatedNames.insert(kernelName).inserted {
                    sources.append(MetalSourceGenerator.generateResidualAdd(
                        name: kernelName,
                        bufferPrecision: bufferPrecision,
                        isSequence: bufferPrecision == .float32))
                    if bufferPrecision != .float32 {
                        let argumentKernelName = MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
                        if generatedNames.insert(argumentKernelName).inserted {
                            sources.append(MetalSourceGenerator.generateResidualAddArgumentTableVariant(
                                name: argumentKernelName,
                                argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                bufferPrecision: bufferPrecision))
                        }
                    }
                }

            case .fusedResidualAddNorm:
                let weightFormat = weightFormatResolver.resolve(role: "scale", entry: entry)
                let kernelName = weightFormat == .bfloat16 ? "fused_residual_add_rms_norm_bf16" : "fused_residual_add_rms_norm"
                if bufferPrecision != .float32, generatedNames.insert(kernelName).inserted {
                    sources.append(MetalSourceGenerator.generateFusedResidualAddRMSNorm(
                        name: kernelName,
                        bufferPrecision: bufferPrecision,
                        weightFormat: weightFormat))
                    let argumentKernelName = MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
                    if generatedNames.insert(argumentKernelName).inserted {
                        sources.append(MetalSourceGenerator.generateFusedResidualAddRMSNormArgumentTableVariant(
                            name: argumentKernelName,
                            argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat))
                    }
                }

            case .fusedSwiGLUProjection(let fused):
                let weightFormat = weightFormatResolver.resolve(role: fused.gateField, entry: entry)
                if bufferPrecision != .float32 {
                    let family = FusedSwiGLUProjectionFamily.resolve(
                        inputDimension: fused.inputDimension,
                        outputDimension: fused.outputDimension)
                    let kernelName = weightFormat == .bfloat16
                        ? family.kernelBaseName + "_bf16"
                        : family.kernelBaseName
                    if generatedNames.insert(kernelName).inserted {
                        if family == .input2048Dense {
                            sources.append(MetalSourceGenerator.generateInput2048FusedSwiGLUProjection(
                                name: kernelName,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat,
                                fixedRowsPerThreadgroup: 8,
                                fixedSimdgroups: 8,
                                unrollFactor: 8))
                            let argumentKernelName = MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
                            if generatedNames.insert(argumentKernelName).inserted {
                                sources.append(MetalSourceGenerator.generateInput2048FusedSwiGLUProjectionArgumentTableVariant(
                                    name: argumentKernelName,
                                    argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                    bufferPrecision: bufferPrecision,
                                    weightFormat: weightFormat,
                                    stagesInputAsFloat: false,
                                    fixedRowsPerThreadgroup: 8,
                                    fixedSimdgroups: 8,
                                    unrollFactor: 8))
                            }
                        } else {
                            sources.append(MetalSourceGenerator.generateFusedSwiGLUProjection(
                                name: kernelName,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat))
                        }
                    }
                } else {
                    let gemmName = weightFormat == .bfloat16 ? "gemm_bf16_f32s" : "gemm_f32s"
                    mppGEMMNames.insert(gemmName)
                    mppGEMMWeightFormat = weightFormat
                    if generatedNames.insert(gemmName).inserted {
                        sources.append(MetalSourceGenerator.generateGEMM(
                            name: gemmName,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat))
                    }
                    if generatedNames.insert("swiglu_seq_f32").inserted {
                        sources.append(MetalSourceGenerator.generateSwiGLU(
                            name: "swiglu_seq_f32",
                            bufferPrecision: bufferPrecision))
                    }
                }

            case .batchedProjection(let batched):
                let count = batched.projections.count
                let weightFormat = weightFormatResolver.resolve(role: batched.projections[0].field, entry: entry)
                if bufferPrecision != .float32 {
                    let suffix = weightFormat == .bfloat16 ? "_bf16" : ""
                    let kernelName = "batched_gemv\(count)\(suffix)"
                    if generatedNames.insert(kernelName).inserted {
                        if count == 2 {
                            sources.append(MetalSourceGenerator.generateBatchedGEMV2(
                                name: kernelName,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat))
                            let argumentKernelName = MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
                            if generatedNames.insert(argumentKernelName).inserted {
                                sources.append(MetalSourceGenerator.generateBatchedGEMV2ArgumentTableVariant(
                                    name: argumentKernelName,
                                    argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                    bufferPrecision: bufferPrecision,
                                    weightFormat: weightFormat))
                            }
                        } else if count == 3 {
                            sources.append(MetalSourceGenerator.generateBatchedGEMV3(
                                name: kernelName,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat))
                            let argumentKernelName = MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
                            if generatedNames.insert(argumentKernelName).inserted {
                                sources.append(MetalSourceGenerator.generateBatchedGEMV3ArgumentTableVariant(
                                    name: argumentKernelName,
                                    argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                    bufferPrecision: bufferPrecision,
                                    weightFormat: weightFormat))
                            }
                        } else {
                            sources.append(MetalSourceGenerator.generateBatchedGEMV4(
                                name: kernelName,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat))
                            let argumentKernelName = MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
                            if generatedNames.insert(argumentKernelName).inserted {
                                sources.append(MetalSourceGenerator.generateBatchedGEMV4ArgumentTableVariant(
                                    name: argumentKernelName,
                                    argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                    bufferPrecision: bufferPrecision,
                                    weightFormat: weightFormat))
                            }
                        }
                    }
                } else {
                    let gemmName = weightFormat == .bfloat16 ? "gemm_bf16_f32s" : "gemm_f32s"
                    mppGEMMNames.insert(gemmName)
                    mppGEMMWeightFormat = weightFormat
                    if generatedNames.insert(gemmName).inserted {
                        sources.append(MetalSourceGenerator.generateGEMM(
                            name: gemmName,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat))
                    }
                }

            case .batchedFragment(let batch):
                if bufferPrecision != .float32 {
                    let kernelContext = KernelContext(
                        bufferPrecision: bufferPrecision,
                        weightFormat: modelWeightFormat)
                    let kernelName = kernelNameResolver.kernelName(for: entry, kernelContext: kernelContext)
                    if generatedNames.insert(kernelName).inserted {
                        let weightFormat = weightFormatResolver.resolve(role: "q_layernorm", entry: entry)
                        if batch.fragments.count == 2, case .perHead = batch.dispatchDimension {
                            sources.append(MetalSourceGenerator.generateBatchedPerHead2(
                                name: kernelName,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat))
                            let argumentKernelName = MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
                            if generatedNames.insert(argumentKernelName).inserted {
                                sources.append(MetalSourceGenerator.generateBatchedPerHead2ArgumentTableVariant(
                                    name: argumentKernelName,
                                    argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                    bufferPrecision: bufferPrecision,
                                    weightFormat: weightFormat))
                            }
                        }
                    }
                } else {
                    let weightFormat = weightFormatResolver.resolve(role: "q_layernorm", entry: entry)
                    let normName = "qk_rms_norm_seq_f32"
                    if generatedNames.insert(normName).inserted {
                        sources.append(MetalSourceGenerator.generateQKNormSeq(
                            name: normName,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat))
                    }
                }
            }
        }

        if needsFlashAttnHelper {
            sources.insert(MetalSourceGenerator.flashAttentionHelperSource, at: 1)
        }

        if needsSSMHelpers {
            sources.insert(MetalSourceGenerator.generateSSMWeightIndependentHelpers(), at: 1)
            for format in ssmConvSiluWeightFormats {
                sources.insert(MetalSourceGenerator.generateSSMConvSiluHelper(weightFormat: format), at: 2)
            }
        }

        let hiddenCopyName = bufferPrecision == .bfloat16
            ? "hidden_copy_from_float_bf16"
            : bufferPrecision == .float32
            ? "hidden_copy_from_float_f32"
            : "hidden_copy_from_float"
        if generatedNames.insert(hiddenCopyName).inserted {
            sources.append(MetalSourceGenerator.generateHiddenCopyFromFloat(
                name: hiddenCopyName,
                bufferPrecision: bufferPrecision
            ))
        }

        let hiddenAddName = bufferPrecision == .bfloat16
            ? "hidden_add_from_float_bf16"
            : bufferPrecision == .float32
            ? "hidden_add_from_float_f32"
            : "hidden_add_from_float"
        if generatedNames.insert(hiddenAddName).inserted {
            sources.append(MetalSourceGenerator.generateHiddenAddFromFloat(
                name: hiddenAddName,
                bufferPrecision: bufferPrecision
            ))
        }

        for name in mppGEMMNames.sorted() {
            mppSources.append(MetalSourceGenerator.generateMPPGEMM(
                name: name,
                bufferPrecision: bufferPrecision,
                weightFormat: mppGEMMWeightFormat))
        }

        return GeneratedKernelSources(
            baseSource: sources.joined(separator: "\n\n"),
            mppSources: mppSources,
            mppKernelNames: mppGEMMNames)
    }

    func format(_ generated: GeneratedKernelSources) -> String {
        var lines: [String] = []
        lines.append("=== BASE LIBRARY ===")
        lines.append(generated.baseSource)
        if !generated.mppSources.isEmpty {
            lines.append("=== MPP LIBRARY ===")
            lines.append(generated.mppSources.joined(separator: "\n\n"))
        }
        return lines.joined(separator: "\n\n")
    }
}
