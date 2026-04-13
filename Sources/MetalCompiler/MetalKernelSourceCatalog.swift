import Metal

struct KernelWeightFormatResolver {
    let stafWeightStore: STAFWeightStore?

    func resolve(role: String, entry: DispatchEntry) -> MetalSourceGenerator.WeightFormat {
        guard let staf = stafWeightStore,
              let binding = entry.parameterBindings.first(where: { $0.role == role }),
              let info = staf.tensor(for: binding.tensorName) else { return .float16 }
        switch info.format.schemeIdentifier {
        case .bf16RowMajor:
            return .bfloat16
        case .fp32RowMajor:
            return .float32
        case .q4Group64ScaleF16:
            return .quantized4Bit(groupSize: 64)
        case .q4Group128ScaleF16:
            return .quantized4Bit(groupSize: 128)
        case .q8Group32ScaleF16:
            return .quantized8Bit(groupSize: 32)
        case .q8Group64ScaleF16:
            return .quantized8Bit(groupSize: 64)
        default:
            return .float16
        }
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
            if format != .float16 {
                return format
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
        let sourceEntries = sourceGenerationEntries(from: entries)
        var sources: [String] = [MetalSourceGenerator.commonHeader]
        var generatedNames: Set<String> = []
        var mppGEMMNames: Set<String> = []
        var mppSources: [String] = []
        var mppGEMMWeightFormat: MetalSourceGenerator.WeightFormat = .float16
        var needsFlashAttnHelper = false
        var needsSSMHelpers = false
        var ssmConvSiluWeightFormats: [MetalSourceGenerator.WeightFormat] = []

        // Generate batched Q4 GEMM kernels from original (pre-expansion) entries.
        // Both explicit batchedProjection entries and implicit batching via
        // fusedSwiGLUProjection (which the step builder decomposes into
        // batchedProjection at dispatch time) must be handled here.
        if bufferPrecision == .float32 {
            for entry in entries {
                let batchedRole: String
                let batchedCount: Int
                switch entry.kind {
                case .batchedProjection(let batched):
                    batchedRole = batched.projections[0].field
                    batchedCount = batched.projections.count
                case .fusedSwiGLUProjection(let fused):
                    batchedRole = fused.gateField
                    batchedCount = 2
                default:
                    continue
                }
                let weightFormat = weightFormatResolver.resolve(role: batchedRole, entry: entry)
                if let batchedQ4Name = batchedQuantizedGEMMKernelName(for: weightFormat, count: batchedCount) {
                    if generatedNames.insert(batchedQ4Name).inserted,
                       let source = batchedQuantizedGEMMSource(
                        named: batchedQ4Name, weightFormat: weightFormat,
                        count: batchedCount, bufferPrecision: bufferPrecision
                       ) {
                        sources.append(source)
                    }
                }
            }
        }

        for entry in sourceEntries {
            let name: String
            switch entry.kind {
            case .projection(let projection, let isOutput):
                let weightFormat = weightFormatResolver.resolve(role: projection.field, entry: entry)
                if let quantizedKernelName = quantizedProjectionKernelName(
                    for: weightFormat,
                    bufferPrecision: bufferPrecision
                ) {
                    if generatedNames.insert(quantizedKernelName).inserted,
                       let source = quantizedProjectionSource(
                        named: quantizedKernelName,
                        weightFormat: weightFormat,
                        bufferPrecision: bufferPrecision
                       ) {
                        sources.append(source)
                    }
                    // Q4+float32: also generate dequant + BF16 MPP GEMM below.
                    // Other quantized formats: skip the rest (decode GEMV only).
                    if !(bufferPrecision == .float32 && weightFormat.isQuantized) {
                        continue
                    }
                }

                // For Q4 prefill: generate dequant kernel and treat as BF16 for MPP GEMM.
                // The dequant kernel unpacks Q4→BF16, then the standard BF16 MPP path handles GEMM.
                let needsDequantForAMX = bufferPrecision == .float32 && weightFormat.isQuantized
                if needsDequantForAMX {
                    if case .quantized4Bit(let groupSize) = weightFormat {
                        let dequantName = "dequant_q4_g\(groupSize)_bf16"
                        if generatedNames.insert(dequantName).inserted {
                            sources.append(MetalSourceGenerator.generateDequantQ4ToBFloat(
                                name: dequantName,
                                groupSize: groupSize
                            ))
                        }
                    }
                }
                let effectiveWeightFormat: WeightFormat = needsDequantForAMX ? .bfloat16 : weightFormat

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
                        weightFormat: effectiveWeightFormat,
                        accessPolicyResolver: accessPolicyResolver
                    ) {
                        let baseName = decodeFamily.kernelBaseName + sourcePolicy.weightLayoutPolicy.kernelNameSuffix
                        name = effectiveWeightFormat == .bfloat16 ? baseName + "_bf16" : baseName
                    } else {
                        name = effectiveWeightFormat == .bfloat16
                            ? decodeFamily.kernelBaseName + "_bf16"
                            : decodeFamily.kernelBaseName
                    }
                } else {
                    name = effectiveWeightFormat == .bfloat16 ? "gemv_bf16" : "gemv"
                }
                let usesSequenceGEMV =
                    bufferPrecision == .float32
                    && isOutput
                    && projection.field == "weight"
                    && projection.outputDimension > projection.inputDimension
                let isSequenceKernel = bufferPrecision == .float32 && !usesSequenceGEMV
                let emittedName = isSequenceKernel
                    ? name.replacingOccurrences(of: "gemv", with: "gemm") + "_f32s"
                    : (usesSequenceGEMV
                        ? (effectiveWeightFormat == .bfloat16 ? "gemv_bf16_f32s" : "gemv_f32s")
                        : name)
                if generatedNames.insert(emittedName).inserted {
                    if isSequenceKernel {
                        let gemmName = effectiveWeightFormat == .bfloat16 ? "gemm_bf16_f32s" : "gemm_f32s"
                        mppGEMMNames.insert(gemmName)
                        mppGEMMWeightFormat = effectiveWeightFormat
                        sources.append(MetalSourceGenerator.generateGEMM(
                            name: gemmName,
                            bufferPrecision: bufferPrecision,
                            weightFormat: effectiveWeightFormat))
                    } else if usesSequenceGEMV {
                        sources.append(MetalSourceGenerator.generateGEMV(
                            name: emittedName,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat,
                            tileElements: 256))
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
                    // Fused RoPE+flash_attn: prefill path still needs the standalone rope_seq kernel.
                    if let flashFragment = fragment as? FlashAttentionFragment, flashFragment.hasInlineRoPE {
                        let ropeF32 = "rope_seq_f32"
                        if generatedNames.insert(ropeF32).inserted {
                            sources.append(MetalSourceGenerator.generateRoPESeq(
                                name: ropeF32, bufferPrecision: .float32))
                        }
                        let ropeF16 = "rope"
                        if generatedNames.insert(ropeF16).inserted {
                            sources.append(MetalSourceGenerator.generateRoPE(
                                name: ropeF16, bufferPrecision: bufferPrecision))
                        }
                    }
                    if bufferPrecision == .float32, let ssmFragment = fragment as? SSMRecurrenceFragment {
                        let sequenceKernelName = SSMRecurrenceFragment.sequenceKernelName(
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat
                        )
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
                            case "rope_flash_attn_decode":
                                // Fused RoPE+attention — argument table variant not yet implemented.
                                // Falls back to per-buffer encode path.
                                break
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
                if bufferPrecision == .float32 {
                    // Sequence-aware fused kernel: single dispatch for prefill
                    let seqFusedName = weightFormat == .bfloat16
                        ? "fused_copy_rms_norm_seq_bf16_f32"
                        : "fused_copy_rms_norm_seq_f32"
                    if generatedNames.insert(seqFusedName).inserted {
                        sources.append(MetalSourceGenerator.generateFusedCopyRMSNormSequence(
                            name: seqFusedName,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat))
                    }
                    // Fallback individual kernels (kept for compatibility)
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

            case .fusedResidualAddCopyNorm(let fusedOp):
                let weightFormat = weightFormatResolver.resolve(role: "scale", entry: entry)

                if fusedOp.preNorm != nil {
                    // PreNorm variant: 4→1 fusion (preNorm + add + copy + outputNorm)
                    let kernelName = weightFormat == .bfloat16
                        ? "fused_pre_norm_residual_add_copy_rms_norm_bf16"
                        : "fused_pre_norm_residual_add_copy_rms_norm"
                    if generatedNames.insert(kernelName).inserted {
                        sources.append(MetalSourceGenerator.generateFusedPreNormResidualAddCopyRMSNorm(
                            name: kernelName,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat))
                        let argumentKernelName = MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
                        if generatedNames.insert(argumentKernelName).inserted {
                            sources.append(MetalSourceGenerator.generateFusedPreNormResidualAddCopyRMSNormArgumentTableVariant(
                                name: argumentKernelName,
                                argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat))
                        }
                    }
                    if bufferPrecision == .float32 {
                        let seqFusedName = weightFormat == .bfloat16
                            ? "fused_pre_norm_residual_add_copy_rms_norm_seq_bf16_f32"
                            : "fused_pre_norm_residual_add_copy_rms_norm_seq_f32"
                        if generatedNames.insert(seqFusedName).inserted {
                            sources.append(MetalSourceGenerator.generateFusedPreNormResidualAddCopyRMSNormSequence(
                                name: seqFusedName,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat))
                        }
                    }
                } else {
                    // Standard variant (no preNorm)
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
                }
                if bufferPrecision == .float32 {
                    if fusedOp.preNorm == nil {
                        // Sequence-aware fused kernel: single dispatch for prefill (no preNorm)
                        let seqFusedName = weightFormat == .bfloat16
                            ? "fused_residual_add_copy_rms_norm_seq_bf16_f32"
                            : "fused_residual_add_copy_rms_norm_seq_f32"
                        if generatedNames.insert(seqFusedName).inserted {
                            sources.append(MetalSourceGenerator.generateFusedResidualAddCopyRMSNormSequence(
                                name: seqFusedName,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat))
                        }
                    }
                    // Fallback individual kernels (kept for compatibility)
                    let addName = "residual_add_seq_f32"
                    let inplaceAddName = "residual_add_inplace_seq_f32"
                    let copyName = "copy_buffer_seq_f32"
                    let normName = weightFormat == .bfloat16 ? "rms_norm_seq_bf16_f32_inplace" : "rms_norm_seq_f32_inplace"
                    if generatedNames.insert(addName).inserted {
                        sources.append(MetalSourceGenerator.generateResidualAdd(name: addName, bufferPrecision: bufferPrecision))
                    }
                    if generatedNames.insert(inplaceAddName).inserted {
                        sources.append(MetalSourceGenerator.generateResidualAddInPlace(name: inplaceAddName, bufferPrecision: bufferPrecision))
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
                let inplaceKernelName = bufferPrecision == .float32 ? "residual_add_inplace_seq_f32" : "residual_add_inplace"
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
                if generatedNames.insert(inplaceKernelName).inserted {
                    sources.append(MetalSourceGenerator.generateResidualAddInPlace(
                        name: inplaceKernelName,
                        bufferPrecision: bufferPrecision,
                        isSequence: bufferPrecision == .float32))
                    if bufferPrecision != .float32 {
                        let argumentKernelName = MetalKernelNameResolver.argumentTableVariantKernelName(for: inplaceKernelName)
                        if generatedNames.insert(argumentKernelName).inserted {
                            sources.append(MetalSourceGenerator.generateResidualAddInPlaceArgumentTableVariant(
                                name: argumentKernelName,
                                argumentBufferIndex: MetalInferenceCompiler.argumentTableBindingIndex,
                                bufferPrecision: bufferPrecision))
                        }
                    }
                }

            case .fusedResidualAddNorm(let fusedOp):
                let weightFormat = weightFormatResolver.resolve(role: "scale", entry: entry)

                if fusedOp.preNorm != nil {
                    // PreNorm variant: 3→1 fusion (preNorm + add + outputNorm, no copy)
                    // Decode kernel not needed for this pattern (only occurs at model end)
                    if bufferPrecision == .float32 {
                        let seqFusedName = weightFormat == .bfloat16
                            ? "fused_pre_norm_residual_add_rms_norm_seq_bf16_f32"
                            : "fused_pre_norm_residual_add_rms_norm_seq_f32"
                        if generatedNames.insert(seqFusedName).inserted {
                            sources.append(MetalSourceGenerator.generateFusedPreNormResidualAddRMSNormSequence(
                                name: seqFusedName,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat))
                        }
                    }
                } else {
                    let kernelName = weightFormat == .bfloat16 ? "fused_residual_add_rms_norm_bf16" : "fused_residual_add_rms_norm"
                    if generatedNames.insert(kernelName).inserted {
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
                }
                if bufferPrecision == .float32 {
                    if fusedOp.preNorm == nil {
                        let seqFusedName = weightFormat == .bfloat16
                            ? "fused_residual_add_rms_norm_seq_bf16_f32"
                            : "fused_residual_add_rms_norm_seq_f32"
                        if generatedNames.insert(seqFusedName).inserted {
                            sources.append(MetalSourceGenerator.generateFusedResidualAddRMSNormSequence(
                                name: seqFusedName,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat))
                        }
                    }
                    // Fallback individual kernels
                    let addName = "residual_add_seq_f32"
                    let inplaceAddName = "residual_add_inplace_seq_f32"
                    if generatedNames.insert(addName).inserted {
                        sources.append(MetalSourceGenerator.generateResidualAdd(name: addName, bufferPrecision: bufferPrecision))
                    }
                    if generatedNames.insert(inplaceAddName).inserted {
                        sources.append(MetalSourceGenerator.generateResidualAddInPlace(name: inplaceAddName, bufferPrecision: bufferPrecision))
                    }
                    let normName = weightFormat == .bfloat16 ? "rms_norm_seq_bf16_f32_inplace" : "rms_norm_seq_f32_inplace"
                    if generatedNames.insert(normName).inserted {
                        sources.append(MetalSourceGenerator.generateReduction(
                            name: normName, dimension: 0, epsilon: 0,
                            bufferPrecision: bufferPrecision, weightFormat: weightFormat))
                    }
                }

            case .fusedSwiGLUProjection(let fused):
                let weightFormat = weightFormatResolver.resolve(role: fused.gateField, entry: entry)
                let family = FusedSwiGLUProjectionFamily.resolve(
                    inputDimension: fused.inputDimension,
                    outputDimension: fused.outputDimension)
                let baseName = family.kernelBaseName(activation: fused.activation)
                let kernelName = weightFormat == .bfloat16
                    ? baseName + "_bf16"
                    : baseName
                if generatedNames.insert(kernelName).inserted {
                    if family == .input2048Dense {
                        sources.append(MetalSourceGenerator.generateInput2048FusedSwiGLUProjection(
                            name: kernelName,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat,
                            activation: fused.activation,
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
                                activation: fused.activation,
                                stagesInputAsFloat: false,
                                fixedRowsPerThreadgroup: 8,
                                fixedSimdgroups: 8,
                                unrollFactor: 8))
                        }
                    } else {
                        sources.append(MetalSourceGenerator.generateFusedSwiGLUProjection(
                            name: kernelName,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat,
                            activation: fused.activation))
                    }
                }
                if bufferPrecision == .float32 {
                    let gemmName = weightFormat == .bfloat16 ? "gemm_bf16_f32s" : "gemm_f32s"
                    mppGEMMNames.insert(gemmName)
                    mppGEMMWeightFormat = weightFormat
                    if generatedNames.insert(gemmName).inserted {
                        sources.append(MetalSourceGenerator.generateGEMM(
                            name: gemmName,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat))
                    }
                    let activationKernelName: String = switch fused.activation {
                    case .silu: "swiglu_seq_f32"
                    case .geluTanh: "geglu_seq_f32"
                    }
                    if generatedNames.insert(activationKernelName).inserted {
                        sources.append(MetalSourceGenerator.generateGatedActivation(
                            name: activationKernelName,
                            bufferPrecision: bufferPrecision,
                            activation: fused.activation))
                    }
                }

            case .batchedProjection(let batched):
                let count = batched.projections.count
                let weightFormat = weightFormatResolver.resolve(role: batched.projections[0].field, entry: entry)
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
                if bufferPrecision == .float32 {
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
                if bufferPrecision == .float32 {
                    let weightFormat = weightFormatResolver.resolve(role: "q_layernorm", entry: entry)
                    // Batched prefill kernel for QK norm (single dispatch for Q+K)
                    if batch.fragments.count == 2,
                       batch.fragments.allSatisfy({ $0 is QKNormFragment }),
                       case .perHead = batch.dispatchDimension {
                        let batchedSeqName = weightFormat == .bfloat16
                            ? "batched_qk_rms_norm_seq_bf16_f32"
                            : "batched_qk_rms_norm_seq_f32"
                        if generatedNames.insert(batchedSeqName).inserted {
                            sources.append(MetalSourceGenerator.generateBatchedQKNormSequence(
                                name: batchedSeqName,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat))
                        }
                    }
                    // Fallback individual kernel
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

        let kvTransferName = "kv_cache_transfer"
        if generatedNames.insert(kvTransferName).inserted {
            needsFlashAttnHelper = true
            sources.append(MetalSourceGenerator.generateKVCacheTransfer(name: kvTransferName))
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

    private func sourceGenerationEntries(from entries: [DispatchEntry]) -> [DispatchEntry] {
        guard bufferPrecision == .float32 else { return entries }
        return entries.flatMap { entry in
            switch entry.kind {
            case .batchedProjection(let batched):
                return batched.projections.map { projection in
                    DispatchEntry(
                        index: entry.index,
                        kind: .projection(
                            MetalProjection(
                                field: projection.field,
                                inputDimension: projection.inputDimension,
                                outputDimension: projection.outputDimension
                            ),
                            isOutput: false
                        ),
                        parameterBindings: entry.parameterBindings,
                        layerIndex: entry.layerIndex,
                        compositeID: entry.compositeID
                    )
                }
            case .fusedSwiGLUProjection(let fused):
                let elementwiseKind: ElementwiseFragment.ElementwiseKind = switch fused.activation {
                case .silu:
                    .swiglu
                case .geluTanh:
                    .geluGated
                }
                return [
                    DispatchEntry(
                        index: entry.index,
                        kind: .projection(
                            MetalProjection(
                                field: fused.gateField,
                                inputDimension: fused.inputDimension,
                                outputDimension: fused.outputDimension
                            ),
                            isOutput: false
                        ),
                        parameterBindings: entry.parameterBindings,
                        layerIndex: entry.layerIndex,
                        compositeID: entry.compositeID
                    ),
                    DispatchEntry(
                        index: entry.index,
                        kind: .projection(
                            MetalProjection(
                                field: fused.upField,
                                inputDimension: fused.inputDimension,
                                outputDimension: fused.outputDimension
                            ),
                            isOutput: false
                        ),
                        parameterBindings: entry.parameterBindings,
                        layerIndex: entry.layerIndex,
                        compositeID: entry.compositeID
                    ),
                    DispatchEntry(
                        index: entry.index,
                        kind: .fragment(
                            ElementwiseFragment(
                                count: fused.outputDimension,
                                kind: elementwiseKind
                            )
                        ),
                        parameterBindings: entry.parameterBindings,
                        layerIndex: entry.layerIndex,
                        compositeID: entry.compositeID
                    ),
                ]
            default:
                return [entry]
            }
        }
    }

    private func quantizedProjectionKernelName(
        for weightFormat: WeightFormat,
        bufferPrecision: BufferPrecision
    ) -> String? {
        switch (weightFormat, bufferPrecision) {
        case (.quantized4Bit(let groupSize), _):
            switch groupSize {
            case 64:
                return "gemv_q4_g64"
            case 128:
                return "gemv_q4_g128"
            default:
                return nil
            }
        case (.quantized8Bit(let groupSize), _):
            switch groupSize {
            case 32:
                return "gemv_q8_g32"
            case 64:
                return "gemv_q8_g64"
            default:
                return nil
            }
        default:
            return nil
        }
    }

    private func quantizedProjectionSource(
        named kernelName: String,
        weightFormat: WeightFormat,
        bufferPrecision: BufferPrecision
    ) -> String? {
        switch (weightFormat, bufferPrecision) {
        case (.quantized4Bit(let groupSize), .float32):
            guard groupSize == 64 || groupSize == 128 else { return nil }
            return MetalSourceGenerator.generateQuantizedGEMM_Q4(
                name: kernelName,
                bufferPrecision: bufferPrecision,
                groupSize: groupSize
            )
        case (.quantized4Bit(let groupSize), _):
            switch groupSize {
            case 64:
                return MetalSourceGenerator.generateQuantizedGEMV_Q4G64(
                    name: kernelName,
                    bufferPrecision: bufferPrecision
                )
            case 128:
                return MetalSourceGenerator.generateQuantizedGEMV_Q4G128(
                    name: kernelName,
                    bufferPrecision: bufferPrecision
                )
            default:
                return nil
            }
        case (.quantized8Bit(let groupSize), _):
            guard groupSize == 32 || groupSize == 64 else { return nil }
            return MetalSourceGenerator.generateQuantizedGEMV_Q8(
                name: kernelName,
                bufferPrecision: bufferPrecision,
                groupSize: groupSize
            )
        default:
            return nil
        }
    }

    // MARK: - Batched Quantized GEMM (Prefill)

    private func batchedQuantizedGEMMKernelName(
        for weightFormat: MetalSourceGenerator.WeightFormat,
        count: Int
    ) -> String? {
        switch weightFormat {
        case .quantized4Bit(let groupSize):
            return "batched_gemm_q4_g\(groupSize)_\(count)"
        default:
            return nil
        }
    }

    private func batchedQuantizedGEMMSource(
        named kernelName: String,
        weightFormat: MetalSourceGenerator.WeightFormat,
        count: Int,
        bufferPrecision: MetalSourceGenerator.BufferPrecision
    ) -> String? {
        guard case .quantized4Bit(let groupSize) = weightFormat else { return nil }
        switch count {
        case 2:
            return MetalSourceGenerator.generateBatchedQuantizedGEMM_Q4_2(
                name: kernelName, bufferPrecision: bufferPrecision, groupSize: groupSize)
        case 3:
            return MetalSourceGenerator.generateBatchedQuantizedGEMM_Q4_3(
                name: kernelName, bufferPrecision: bufferPrecision, groupSize: groupSize)
        default:
            return nil
        }
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
