import LMIR

extension MetalSourceGenerator {

    /// Generate complete MSL source for ALL kernels (decode + prefill).
    ///
    /// This replaces both MetalKernelSource and GeneratedKernelLibrary.
    /// Specialized kernels (flash_attn, ssm, quantized GEMV) are included
    /// as parameterized templates. Simple kernels are generated from parameters.
    ///
    /// - Parameters:
    ///   - weightFormat: Primary weight format (determines weight reader code)
    public static func generateCompleteLibrary(
        weightFormat: WeightFormat
    ) -> String {
        var sources: [String] = [commonHeader]
        let decode = BufferPrecision.float16
        let prefill = BufferPrecision.float32

        // === Decode kernels (F16 buffers, single token) ===
        sources.append(generateReduction(name: "rms_norm", dimension: 0, epsilon: 0, bufferPrecision: decode, weightFormat: .float16, isSequence: false))
        sources.append(generateReduction(name: "rms_norm_bf16", dimension: 0, epsilon: 0, bufferPrecision: decode, weightFormat: .bfloat16, isSequence: false))
        sources.append(generatePerLayerInputModulation(
            name: "per_layer_input_modulation",
            bufferPrecision: decode,
            activation: .custom("gelu_pytorch_tanh"),
            isSequence: false
        ))
        sources.append(generateSwiGLU(name: "swiglu", bufferPrecision: decode, isSequence: false))
        sources.append(generateCopy(name: "copy_buffer", bufferPrecision: decode, isSequence: false))
        sources.append(generateResidualAdd(name: "residual_add", bufferPrecision: decode, isSequence: false))
        sources.append(generateResidualAddArgumentTableVariant(name: "residual_add_argbuf", argumentBufferIndex: 30, bufferPrecision: decode))
        sources.append(generateGEMV(name: "gemv", bufferPrecision: decode, weightFormat: .float16, tileElements: 256))
        sources.append(generateGEMV(name: "gemv_bf16", bufferPrecision: decode, weightFormat: .bfloat16, tileElements: 256))
        sources.append(generateGEMVArgumentTableVariant(name: "gemv_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .float16, tileElements: 256))
        sources.append(generateGEMVArgumentTableVariant(name: "gemv_bf16_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .bfloat16, tileElements: 256))
        sources.append(generateGEMV(name: "gemv_large", bufferPrecision: decode, weightFormat: .float16, tileElements: 512))
        sources.append(generateGEMV(name: "gemv_large_bf16", bufferPrecision: decode, weightFormat: .bfloat16, tileElements: 512))
        sources.append(generateInput8192TiledGEMV(name: "gemv_8192_tiled", bufferPrecision: decode, weightFormat: .float16, tileElements: 1_024, unrollFactor: 4))
        sources.append(generateInput8192TiledGEMV(name: "gemv_8192_tiled_bf16", bufferPrecision: decode, weightFormat: .bfloat16, tileElements: 1_024, unrollFactor: 4))
        sources.append(generateInput8192TiledGEMVArgumentTableVariant(name: "gemv_8192_tiled_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .float16, tileElements: 1_024, unrollFactor: 4))
        sources.append(generateInput8192TiledGEMVArgumentTableVariant(name: "gemv_8192_tiled_bf16_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .bfloat16, tileElements: 1_024, unrollFactor: 4))
        sources.append(generateInput2048GEMV(name: "gemv_2048", bufferPrecision: decode, weightFormat: .float16, unrollFactor: 4))
        sources.append(generateInput2048GEMV(name: "gemv_2048_bf16", bufferPrecision: decode, weightFormat: .bfloat16, unrollFactor: 4))
        let input2048SquareF16 = Input2048GEMVSourcePolicy.square(weightFormat: .float16)
        let input2048SquareBF16 = Input2048GEMVSourcePolicy.square(weightFormat: .bfloat16)
        let input20486144F16 = Input2048GEMVSourcePolicy.expanded6144(weightFormat: .float16)
        let input20486144BF16 = Input2048GEMVSourcePolicy.expanded6144(weightFormat: .bfloat16)
        let input20488192F16 = Input2048GEMVSourcePolicy.expanded8192(weightFormat: .float16)
        let input20488192BF16 = Input2048GEMVSourcePolicy.expanded8192(weightFormat: .bfloat16)
        sources.append(generateInput2048GEMV(name: "gemv_2048_sq", bufferPrecision: decode, weightFormat: .float16, fixedOutputDimension: input2048SquareF16.fixedOutputDimension, fixedRowsPerThreadgroup: input2048SquareF16.fixedRowsPerThreadgroup, fixedSimdgroups: input2048SquareF16.fixedSimdgroups, stagesInputAsFloat: input2048SquareF16.stagesInputAsFloat, weightLayoutPolicy: input2048SquareF16.weightLayoutPolicy, unrollFactor: input2048SquareF16.unrollFactor))
        sources.append(generateInput2048GEMV(name: "gemv_2048_sq_bf16", bufferPrecision: decode, weightFormat: .bfloat16, fixedOutputDimension: input2048SquareBF16.fixedOutputDimension, fixedRowsPerThreadgroup: input2048SquareBF16.fixedRowsPerThreadgroup, fixedSimdgroups: input2048SquareBF16.fixedSimdgroups, stagesInputAsFloat: input2048SquareBF16.stagesInputAsFloat, weightLayoutPolicy: input2048SquareBF16.weightLayoutPolicy, unrollFactor: input2048SquareBF16.unrollFactor))
        sources.append(generateInput2048GEMVArgumentTableVariant(name: "gemv_2048_sq_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .float16, fixedOutputDimension: input2048SquareF16.fixedOutputDimension, includesDimensionBindings: false, fixedRowsPerThreadgroup: input2048SquareF16.fixedRowsPerThreadgroup, fixedSimdgroups: input2048SquareF16.fixedSimdgroups, stagesInputAsFloat: input2048SquareF16.stagesInputAsFloat, weightLayoutPolicy: input2048SquareF16.weightLayoutPolicy, bf16ArgumentReadPolicy: input2048SquareF16.bf16ArgumentReadPolicy, unrollFactor: input2048SquareF16.unrollFactor))
        sources.append(generateInput2048GEMVArgumentTableVariant(name: "gemv_2048_sq_bf16_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .bfloat16, fixedOutputDimension: input2048SquareBF16.fixedOutputDimension, includesDimensionBindings: false, fixedRowsPerThreadgroup: input2048SquareBF16.fixedRowsPerThreadgroup, fixedSimdgroups: input2048SquareBF16.fixedSimdgroups, stagesInputAsFloat: input2048SquareBF16.stagesInputAsFloat, weightLayoutPolicy: input2048SquareBF16.weightLayoutPolicy, bf16ArgumentReadPolicy: input2048SquareBF16.bf16ArgumentReadPolicy, unrollFactor: input2048SquareBF16.unrollFactor))
        sources.append(generateInput2048GEMV(name: "gemv_2048_6144", bufferPrecision: decode, weightFormat: .float16, fixedOutputDimension: input20486144F16.fixedOutputDimension, fixedRowsPerThreadgroup: input20486144F16.fixedRowsPerThreadgroup, fixedSimdgroups: input20486144F16.fixedSimdgroups, stagesInputAsFloat: input20486144F16.stagesInputAsFloat, weightLayoutPolicy: input20486144F16.weightLayoutPolicy, unrollFactor: input20486144F16.unrollFactor))
        sources.append(generateInput2048GEMV(name: "gemv_2048_6144_bf16", bufferPrecision: decode, weightFormat: .bfloat16, fixedOutputDimension: input20486144BF16.fixedOutputDimension, fixedRowsPerThreadgroup: input20486144BF16.fixedRowsPerThreadgroup, fixedSimdgroups: input20486144BF16.fixedSimdgroups, stagesInputAsFloat: input20486144BF16.stagesInputAsFloat, weightLayoutPolicy: input20486144BF16.weightLayoutPolicy, unrollFactor: input20486144BF16.unrollFactor))
        sources.append(generateInput2048GEMVArgumentTableVariant(name: "gemv_2048_6144_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .float16, fixedOutputDimension: input20486144F16.fixedOutputDimension, includesDimensionBindings: false, fixedRowsPerThreadgroup: input20486144F16.fixedRowsPerThreadgroup, fixedSimdgroups: input20486144F16.fixedSimdgroups, stagesInputAsFloat: input20486144F16.stagesInputAsFloat, weightLayoutPolicy: input20486144F16.weightLayoutPolicy, bf16ArgumentReadPolicy: input20486144F16.bf16ArgumentReadPolicy, unrollFactor: input20486144F16.unrollFactor))
        sources.append(generateInput2048GEMVArgumentTableVariant(name: "gemv_2048_6144_bf16_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .bfloat16, fixedOutputDimension: input20486144BF16.fixedOutputDimension, includesDimensionBindings: false, fixedRowsPerThreadgroup: input20486144BF16.fixedRowsPerThreadgroup, fixedSimdgroups: input20486144BF16.fixedSimdgroups, stagesInputAsFloat: input20486144BF16.stagesInputAsFloat, weightLayoutPolicy: input20486144BF16.weightLayoutPolicy, bf16ArgumentReadPolicy: input20486144BF16.bf16ArgumentReadPolicy, unrollFactor: input20486144BF16.unrollFactor))
        sources.append(generateInput2048GEMV(name: "gemv_2048_8192", bufferPrecision: decode, weightFormat: .float16, fixedOutputDimension: input20488192F16.fixedOutputDimension, fixedRowsPerThreadgroup: input20488192F16.fixedRowsPerThreadgroup, stagesInputAsFloat: input20488192F16.stagesInputAsFloat, weightLayoutPolicy: input20488192F16.weightLayoutPolicy, unrollFactor: input20488192F16.unrollFactor))
        sources.append(generateInput2048GEMV(name: "gemv_2048_8192_bf16", bufferPrecision: decode, weightFormat: .bfloat16, fixedOutputDimension: input20488192BF16.fixedOutputDimension, fixedRowsPerThreadgroup: input20488192BF16.fixedRowsPerThreadgroup, stagesInputAsFloat: input20488192BF16.stagesInputAsFloat, weightLayoutPolicy: input20488192BF16.weightLayoutPolicy, unrollFactor: input20488192BF16.unrollFactor))
        sources.append(generateVocabGEMV(name: "gemv_vocab", bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateVocabGEMV(name: "gemv_vocab_bf16", bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateVocabGEMVArgumentTableVariant(name: "gemv_vocab_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateVocabGEMVArgumentTableVariant(name: "gemv_vocab_bf16_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateEmbeddingLookup(name: "embedding_lookup", bufferPrecision: decode, weightFormat: .float16, isSequence: false))
        sources.append(generateEmbeddingLookup(name: "embedding_lookup_bf16", bufferPrecision: decode, weightFormat: .bfloat16, isSequence: false))
        sources.append(generateEmbeddingLookupArgumentTableVariant(name: "embedding_lookup_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateEmbeddingLookupArgumentTableVariant(name: "embedding_lookup_bf16_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateArgmax(name: "argmax", bufferPrecision: decode))
        sources.append(generateArgmaxArgumentTableVariant(name: "argmax_argbuf", argumentBufferIndex: 30, bufferPrecision: decode))
        sources.append(generateFusedCopyRMSNorm(name: "fused_copy_rms_norm", bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateFusedCopyRMSNorm(name: "fused_copy_rms_norm_bf16", bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateFusedCopyRMSNormArgumentTableVariant(name: "fused_copy_rms_norm_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateFusedCopyRMSNormArgumentTableVariant(name: "fused_copy_rms_norm_bf16_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateFusedResidualAddCopyRMSNorm(name: "fused_residual_add_copy_rms_norm", bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateFusedResidualAddCopyRMSNorm(name: "fused_residual_add_copy_rms_norm_bf16", bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateFusedResidualAddCopyRMSNormArgumentTableVariant(name: "fused_residual_add_copy_rms_norm_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateFusedResidualAddCopyRMSNormArgumentTableVariant(name: "fused_residual_add_copy_rms_norm_bf16_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateFusedResidualAddRMSNorm(name: "fused_residual_add_rms_norm", bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateFusedResidualAddRMSNorm(name: "fused_residual_add_rms_norm_bf16", bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateFusedResidualAddRMSNormArgumentTableVariant(name: "fused_residual_add_rms_norm_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateFusedResidualAddRMSNormArgumentTableVariant(name: "fused_residual_add_rms_norm_bf16_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateBatchedGEMV2(name: "batched_gemv2", bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateBatchedGEMV2(name: "batched_gemv2_bf16", bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateBatchedGEMV2ArgumentTableVariant(name: "batched_gemv2_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateBatchedGEMV2ArgumentTableVariant(name: "batched_gemv2_bf16_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateBatchedGEMV3(name: "batched_gemv3", bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateBatchedGEMV3(name: "batched_gemv3_bf16", bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateBatchedGEMV3ArgumentTableVariant(name: "batched_gemv3_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateBatchedGEMV3ArgumentTableVariant(name: "batched_gemv3_bf16_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateBatchedGEMV4(name: "batched_gemv4", bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateBatchedGEMV4(name: "batched_gemv4_bf16", bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateBatchedGEMV4ArgumentTableVariant(name: "batched_gemv4_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateBatchedGEMV4ArgumentTableVariant(name: "batched_gemv4_bf16_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateBatchedPerHead2(name: "batched_qk_rms_norm_2", bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateBatchedPerHead2(name: "batched_qk_rms_norm_bf16_2", bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateBatchedPerHead2ArgumentTableVariant(name: "batched_qk_rms_norm_2_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateBatchedPerHead2ArgumentTableVariant(name: "batched_qk_rms_norm_bf16_2_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateFusedSwiGLUProjection(name: "fused_swiglu_projection", bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateFusedSwiGLUProjection(name: "fused_swiglu_projection_bf16", bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateInput2048FusedSwiGLUProjection(name: "fused_swiglu_projection_2048", bufferPrecision: decode, weightFormat: .float16, stagesInputAsFloat: false, fixedRowsPerThreadgroup: 8, fixedSimdgroups: 8, unrollFactor: 8))
        sources.append(generateInput2048FusedSwiGLUProjection(name: "fused_swiglu_projection_2048_bf16", bufferPrecision: decode, weightFormat: .bfloat16, stagesInputAsFloat: false, fixedRowsPerThreadgroup: 8, fixedSimdgroups: 8, unrollFactor: 8))
        sources.append(generateInput2048FusedSwiGLUProjectionArgumentTableVariant(name: "fused_swiglu_projection_2048_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .float16, stagesInputAsFloat: false, fixedRowsPerThreadgroup: 8, fixedSimdgroups: 8, unrollFactor: 8))
        sources.append(generateInput2048FusedSwiGLUProjectionArgumentTableVariant(name: "fused_swiglu_projection_2048_bf16_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .bfloat16, stagesInputAsFloat: false, fixedRowsPerThreadgroup: 8, fixedSimdgroups: 8, unrollFactor: 8))
        sources.append(generateQKNorm(name: "qk_rms_norm", bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateQKNorm(name: "qk_rms_norm_bf16", bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateQKNormArgumentTableVariant(name: "qk_rms_norm_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateQKNormArgumentTableVariant(name: "qk_rms_norm_bf16_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateReductionArgumentTableVariant(name: "rms_norm_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateReductionArgumentTableVariant(name: "rms_norm_bf16_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateRoPE(name: "rope", bufferPrecision: decode))
        sources.append(generateRoPEArgumentTableVariant(name: "rope_argbuf", argumentBufferIndex: 30, bufferPrecision: decode))
        sources.append(generateConvStateUpdate(name: "conv_state_update_bf16", bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateConvStateUpdateArgumentTableVariant(name: "conv_state_update_bf16_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .bfloat16))

        // === Prefill kernels (F32 buffers, sequence) ===
        sources.append(generateReduction(name: "rms_norm_seq_f32_inplace", dimension: 0, epsilon: 0, bufferPrecision: prefill, weightFormat: .float16))
        sources.append(generateReduction(name: "rms_norm_seq_bf16_f32_inplace", dimension: 0, epsilon: 0, bufferPrecision: prefill, weightFormat: .bfloat16))
        sources.append(generateReduction(name: "rms_norm_seq_f32s", dimension: 0, epsilon: 0, bufferPrecision: prefill, weightFormat: .float16))
        sources.append(generateReduction(name: "rms_norm_seq_bf16_f32s", dimension: 0, epsilon: 0, bufferPrecision: prefill, weightFormat: .bfloat16))
        sources.append(generatePerLayerInputModulation(
            name: "per_layer_input_modulation_seq_f32",
            bufferPrecision: prefill,
            activation: .custom("gelu_pytorch_tanh"),
            isSequence: true
        ))
        sources.append(generateSwiGLU(name: "swiglu_seq_f32", bufferPrecision: prefill))
        sources.append(generateCopy(name: "copy_buffer_seq_f32", bufferPrecision: prefill))
        sources.append(generateResidualAdd(name: "residual_add_seq_f32", bufferPrecision: prefill))
        sources.append(generateGEMM(name: "gemm_f32s", bufferPrecision: prefill, weightFormat: .float16))
        sources.append(generateGEMM(name: "gemm_bf16_f32s", bufferPrecision: prefill, weightFormat: .bfloat16))
        sources.append(generateEmbeddingLookup(name: "embedding_lookup_seq_f32", bufferPrecision: prefill, weightFormat: .float16))
        sources.append(generateEmbeddingLookup(name: "embedding_lookup_seq_bf16_f32", bufferPrecision: prefill, weightFormat: .bfloat16))
        sources.append(generateQKNormSeq(name: "qk_rms_norm_seq_f32", bufferPrecision: prefill, weightFormat: .bfloat16))
        sources.append(generateRoPESeq(name: "rope_seq_f32", bufferPrecision: prefill))
        sources.append(generateConv1dCausalSeq(name: "conv1d_causal_seq_f32", bufferPrecision: prefill, weightFormat: .bfloat16))
        sources.append(generateExtractConvState(name: "extract_conv_state_f32", bufferPrecision: prefill))
        sources.append(generateArgmax(name: "argmax_f32", bufferPrecision: prefill))

        // === F16 seq variants (legacy prefill path, non-F32 models) ===
        let f16 = BufferPrecision.float16
        sources.append(generateReduction(name: "rms_norm_seq", dimension: 0, epsilon: 0, bufferPrecision: f16, weightFormat: .float16))
        sources.append(generateReduction(name: "rms_norm_seq_bf16", dimension: 0, epsilon: 0, bufferPrecision: f16, weightFormat: .bfloat16))
        sources.append(generateSwiGLU(name: "swiglu_seq", bufferPrecision: f16))
        sources.append(generateCopy(name: "copy_buffer_seq", bufferPrecision: f16))
        sources.append(generateResidualAdd(name: "residual_add_seq", bufferPrecision: f16))
        sources.append(generateEmbeddingLookup(name: "embedding_lookup_seq", bufferPrecision: f16, weightFormat: .float16))
        sources.append(generateEmbeddingLookup(name: "embedding_lookup_seq_bf16", bufferPrecision: f16, weightFormat: .bfloat16))
        sources.append(generateRoPESeq(name: "rope_seq", bufferPrecision: f16))
        sources.append(generateQKNormSeq(name: "qk_rms_norm_seq", bufferPrecision: f16, weightFormat: .float16))
        sources.append(generateQKNormSeq(name: "qk_rms_norm_seq_bf16", bufferPrecision: f16, weightFormat: .bfloat16))
        sources.append(generateConv1dCausalSeq(name: "conv1d_causal_seq", bufferPrecision: f16, weightFormat: .bfloat16))
        sources.append(generateExtractConvState(name: "extract_conv_state", bufferPrecision: f16))

        // === F16 GEMM variants ===
        sources.append(generateGEMM(name: "gemm", bufferPrecision: f16, weightFormat: .float16))
        sources.append(generateGEMM(name: "gemm_bf16", bufferPrecision: f16, weightFormat: .bfloat16))

        // === Flash Attention ===
        sources.append(flashAttentionHelperSource)  // helper functions once
        sources.append(generateFlashAttentionKernel(name: "flash_attn_decode", bufferPrecision: decode))
        sources.append(generateFlashAttentionArgumentTableVariant(name: "flash_attn_decode_argbuf", argumentBufferIndex: 30, bufferPrecision: decode))
        sources.append(generateFlashAttentionKernel(name: "flash_attn_decode_f32", bufferPrecision: prefill))
        sources.append(generateBatchFlashAttention(name: "flash_attn_batch_f32", bufferPrecision: prefill))
        sources.append(generateKVCacheFillSeq(name: "kv_cache_fill_seq_f32", bufferPrecision: prefill))

        // === Quantized GEMV (decode) ===
        sources.append(generateQuantizedGEMV_Q4G64(name: "gemv_q4_g64", bufferPrecision: decode))
        sources.append(generateQuantizedGEMV_Q4G128(name: "gemv_q4_g128", bufferPrecision: decode))
        sources.append(generateQuantizedGEMV_Q8(name: "gemv_q8_g32", bufferPrecision: decode, groupSize: 32))
        sources.append(generateQuantizedGEMV_Q8(name: "gemv_q8_g64", bufferPrecision: decode, groupSize: 64))

        // === Quantized GEMM (prefill) ===
        sources.append(generateQuantizedGEMM_Q4(name: "gemm_q4_g64", bufferPrecision: decode, groupSize: 64))
        sources.append(generateQuantizedGEMM_Q4(name: "gemm_q4_g128", bufferPrecision: decode, groupSize: 128))
        sources.append(generateQuantizedGEMM_Q4(name: "gemm_q4_g64_f32s", bufferPrecision: prefill, groupSize: 64))
        sources.append(generateQuantizedGEMM_Q4(name: "gemm_q4_g128_f32s", bufferPrecision: prefill, groupSize: 128))

        // === Misc decode kernels ===
        sources.append(generateSigmoidGate(name: "sigmoid_gate", bufferPrecision: decode))
        sources.append(generateLayerNorm(name: "layer_norm", bufferPrecision: decode, weightFormat: .float16))

        // === Mixed-precision GEMM variants ===
        // gemm_bf16_f32_to_half: F32 input, BF16 weight, F16 output
        // gemm_bf16_f32s_halfout: F32 input, BF16 weight, F16 output (with seqLen)
        // These are transitional — will be replaced by F32 prefill path

        // === SSM recurrence (DeltaNet/Mamba) ===
        sources.append(generateSSMHelperSource(weightFormat: .float16))
        sources.append(generateSSMRecurrence(name: "ssm_recurrence", bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateSSMRecurrence(name: "ssm_recurrence_f32", bufferPrecision: prefill, weightFormat: .float16))
        sources.append(generateSSMRecurrenceSequence(name: "ssm_recurrence_seq", bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateSSMRecurrenceSequence(name: "ssm_recurrence_seq_f32", bufferPrecision: prefill, weightFormat: .float16))

        // === KV cache quantization ===
        sources.append(kvQuantizationSource)

        // === Mixed-precision GEMM (transitional) ===
        sources.append(gemmMixedPrecisionSource)

        return sources.joined(separator: "\n\n")
    }

    // MARK: - Fragment-Driven Generation



    // MARK: - Common Header
}
