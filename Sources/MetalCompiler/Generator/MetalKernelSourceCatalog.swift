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

        // Generate batched prefill GEMM kernels from original (pre-expansion) entries.
        // Q4 packed weights → batched_gemm_q4_g*_{count} (block-level inline dequant).
        // BF16 / FP16 dense weights → batched_gemm_bf16_f32s_{count} / batched_gemm_f32s_{count}
        // (MPP matmul2d with shared input A across projections).
        var batchedMPPGEMMRequests: [(name: String, count: Int, weightFormat: MetalSourceGenerator.WeightFormat)] = []
        if bufferPrecision == .float32 {
            for entry in entries {
                let batchedRole: String
                let batchedCount: Int
                if let batched = entry.fragment as? BatchedProjection {
                    batchedRole = batched.projections[0].field
                    batchedCount = batched.projections.count
                } else {
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
                } else if let batchedMPPName = batchedMPPGEMMKernelName(for: weightFormat, count: batchedCount) {
                    if generatedNames.insert(batchedMPPName).inserted {
                        batchedMPPGEMMRequests.append((name: batchedMPPName, count: batchedCount, weightFormat: weightFormat))
                    }
                }
            }
        }

        for entry in sourceEntries {
            let name: String
            if let projection = entry.fragment as? LinearFragment {
                let isOutput = projection.isOutput
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

            } else {
                let fragment = entry.fragment
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
                                maxThreadgroupSize: SSMRecurrenceFragment.maxThreadgroupSize,
                                headCount: ssmFragment.headCount,
                                groupCount: ssmFragment.groupCount,
                                keyHeadDimension: ssmFragment.keyHeadDimension,
                                valueHeadDimension: ssmFragment.valueHeadDimension))
                        }
                    }
                    if bufferPrecision != .float32 {
                        let argumentKernelName = MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
                        if argumentKernelName != kernelName, generatedNames.insert(argumentKernelName).inserted {
                            if let argumentVariantSource = generateArgumentTableVariant(
                                kernelName: kernelName,
                                argumentKernelName: argumentKernelName,
                                fragment: fragment,
                                bufferPrecision: bufferPrecision,
                                weightFormat: weightFormat
                            ) {
                                sources.append(argumentVariantSource)
                            }
                        }
                    }
                }
                if fragment is BatchedProjection, bufferPrecision == .float32 {
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
                if let batch = fragment as? BatchedFragment, bufferPrecision == .float32 {
                    // Batched prefill kernel for QK norm
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
                // Fused batched QK norm + RoPE for prefill. Emitted alongside the
                // non-fused batched decode kernel (`batched_qk_rms_norm_2`),
                // which this fragment still delegates to for the decode path.
                if fragment is BatchedQKNormRoPEFragment, bufferPrecision == .float32 {
                    let fusedName = weightFormat == .bfloat16
                        ? "batched_qk_rms_norm_rope_seq_bf16_f32"
                        : "batched_qk_rms_norm_rope_seq_f32"
                    if generatedNames.insert(fusedName).inserted {
                        sources.append(MetalSourceGenerator.generateBatchedQKNormRoPESequence(
                            name: fusedName,
                            bufferPrecision: bufferPrecision,
                            weightFormat: weightFormat))
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
                    if let flashAttn = fragment as? FlashAttentionFragment, flashAttn.directScratchMode {
                        // Direct-scratch mode: attention reads K/V from scratch, no KV cache needed.
                        let scratchName = "flash_attn_batch_scratch_f32"
                        if generatedNames.insert(scratchName).inserted {
                            sources.append(MetalSourceGenerator.generateDirectScratchBatchFlashAttention(
                                name: scratchName,
                                bufferPrecision: bufferPrecision))
                        }
                    } else {
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

        // Emit tile-size variants (_mtile16 / _mtile32 / _mtile64) per GEMM
        // kernel name so the runtime can pick the shape that matches the
        // actual sequence length. The base name is kept as an alias for the
        // default tile size so existing name lookups continue to work.
        var emittedMPPKernelNames: Set<String> = []
        for name in mppGEMMNames.sorted() {
            for tileSize in MetalSourceGenerator.mppGEMMTileSizes {
                let variantName = MetalSourceGenerator.mppGEMMVariantName(
                    baseName: name, tileSize: tileSize)
                mppSources.append(MetalSourceGenerator.generateMPPGEMM(
                    name: variantName,
                    bufferPrecision: bufferPrecision,
                    weightFormat: mppGEMMWeightFormat,
                    mTile: tileSize))
                emittedMPPKernelNames.insert(variantName)
            }
            // Also emit the canonical base name at the default tile size so
            // legacy lookups (pipelineCache[baseName]) resolve.
            mppSources.append(MetalSourceGenerator.generateMPPGEMM(
                name: name,
                bufferPrecision: bufferPrecision,
                weightFormat: mppGEMMWeightFormat,
                mTile: MetalSourceGenerator.mppGEMMDefaultTileSize))
            emittedMPPKernelNames.insert(name)
        }

        // Batched MPP GEMM kernels (BF16 / FP16 / FP32 dense weights with multiple
        // projections sharing the same input). Emitted alongside mppGEMMNames so
        // they land in the MPP library and are compiled with the correct include
        // headers for matmul2d.
        for request in batchedMPPGEMMRequests {
            for tileSize in MetalSourceGenerator.mppGEMMTileSizes {
                let variantName = MetalSourceGenerator.mppGEMMVariantName(
                    baseName: request.name, tileSize: tileSize)
                mppSources.append(MetalSourceGenerator.generateBatchedMPPGEMM(
                    name: variantName,
                    count: request.count,
                    bufferPrecision: bufferPrecision,
                    weightFormat: request.weightFormat,
                    mTile: tileSize))
                emittedMPPKernelNames.insert(variantName)
            }
            // Canonical base name alias at the default tile size.
            mppSources.append(MetalSourceGenerator.generateBatchedMPPGEMM(
                name: request.name,
                count: request.count,
                bufferPrecision: bufferPrecision,
                weightFormat: request.weightFormat,
                mTile: MetalSourceGenerator.mppGEMMDefaultTileSize))
            emittedMPPKernelNames.insert(request.name)
        }

        return GeneratedKernelSources(
            baseSource: sources.joined(separator: "\n\n"),
            mppSources: mppSources,
            mppKernelNames: emittedMPPKernelNames)
    }

    private func sourceGenerationEntries(from entries: [DispatchEntry]) -> [DispatchEntry] {
        guard bufferPrecision == .float32 else { return entries }
        return entries.flatMap { entry in
            if let batched = entry.fragment as? BatchedProjection {
                return batched.projections.map { projection in
                    DispatchEntry(
                        index: entry.index,
                        fragment: LinearFragment(
                            field: projection.field,
                            inputDimension: projection.inputDimension,
                            outputDimension: projection.outputDimension,
                            isOutput: false
                        ),
                        parameterBindings: entry.parameterBindings,
                        layerIndex: entry.layerIndex,
                        compositeID: entry.compositeID
                    )
                }
            }
            return [entry]
        }
    }

    private func generateArgumentTableVariant(
        kernelName: String,
        argumentKernelName: String,
        fragment: any PrimitiveMetalKernelFragment,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String? {
        let argumentBufferIndex = MetalInferenceCompiler.argumentTableBindingIndex

        if let batched = fragment as? BatchedProjection {
            switch batched.projections.count {
            case 2:
                return MetalSourceGenerator.generateBatchedGEMV2ArgumentTableVariant(
                    name: argumentKernelName,
                    argumentBufferIndex: argumentBufferIndex,
                    bufferPrecision: bufferPrecision,
                    weightFormat: weightFormat
                )
            case 3:
                return MetalSourceGenerator.generateBatchedGEMV3ArgumentTableVariant(
                    name: argumentKernelName,
                    argumentBufferIndex: argumentBufferIndex,
                    bufferPrecision: bufferPrecision,
                    weightFormat: weightFormat
                )
            default:
                return MetalSourceGenerator.generateBatchedGEMV4ArgumentTableVariant(
                    name: argumentKernelName,
                    argumentBufferIndex: argumentBufferIndex,
                    bufferPrecision: bufferPrecision,
                    weightFormat: weightFormat
                )
            }
        }

        if fragment is BatchedFragment {
            return MetalSourceGenerator.generateBatchedPerHead2ArgumentTableVariant(
                name: argumentKernelName,
                argumentBufferIndex: argumentBufferIndex,
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat
            )
        }

        // Other fragments: ATB variants come from the standard library or
        // are not applicable. Return nil.
        return nil
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

    // MARK: - Batched MPP GEMM (Prefill BF16 / FP16 / FP32)

    /// Kernel name for the batched MPP GEMM kernel that handles multiple dense-weight
    /// projections sharing one input. Returns nil for weight formats where a dense
    /// matmul2d-based kernel does not apply (e.g. packed Q4 — use `batchedQuantizedGEMMKernelName`).
    func batchedMPPGEMMKernelName(
        for weightFormat: MetalSourceGenerator.WeightFormat,
        count: Int
    ) -> String? {
        guard count >= 2 && count <= 4 else { return nil }
        switch weightFormat {
        case .bfloat16:
            return "batched_gemm_bf16_f32s_\(count)"
        case .float16:
            return "batched_gemm_f16_f32s_\(count)"
        case .float32:
            return "batched_gemm_f32_f32s_\(count)"
        case .quantized4Bit, .quantized8Bit:
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
