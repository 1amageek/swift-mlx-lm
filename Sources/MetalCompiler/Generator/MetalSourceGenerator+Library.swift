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
        sources.append(generateEmbeddingLookup(name: "embedding_lookup_fp32", bufferPrecision: decode, weightFormat: .float32, isSequence: false))
        sources.append(generateQuantizedEmbeddingLookupQ4(name: "embedding_lookup_q4_g64", bufferPrecision: decode, groupSize: 64, isSequence: false))
        sources.append(generateQuantizedEmbeddingLookupQ4(name: "embedding_lookup_q4_g128", bufferPrecision: decode, groupSize: 128, isSequence: false))
        sources.append(generateQuantizedEmbeddingLookupQ8(name: "embedding_lookup_q8_g32", bufferPrecision: decode, groupSize: 32, isSequence: false))
        sources.append(generateQuantizedEmbeddingLookupQ8(name: "embedding_lookup_q8_g64", bufferPrecision: decode, groupSize: 64, isSequence: false))
        sources.append(generateEmbeddingLookupArgumentTableVariant(name: "embedding_lookup_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateEmbeddingLookupArgumentTableVariant(name: "embedding_lookup_bf16_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateEmbeddingLookupArgumentTableVariant(name: "embedding_lookup_fp32_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .float32))
        sources.append(generateArgmax(name: "argmax", bufferPrecision: decode))
        sources.append(generateArgmaxArgumentTableVariant(name: "argmax_argbuf", argumentBufferIndex: 30, bufferPrecision: decode))
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
        sources.append(generateQKNorm(name: "qk_rms_norm", bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateQKNorm(name: "qk_rms_norm_bf16", bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateQKNormArgumentTableVariant(name: "qk_rms_norm_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateQKNormArgumentTableVariant(name: "qk_rms_norm_bf16_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generatePerHeadRMSNorm(name: "per_head_rms_norm", bufferPrecision: decode, isSequence: false))
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
        sources.append(generateRoundFloatToFloat16(name: "round_f16_seq_f32"))
        sources.append(generateRoundFloatToBFloat16(name: "round_bf16_seq_f32"))
        sources.append(generateGEMM(name: "gemm_f32s", bufferPrecision: prefill, weightFormat: .float16))
        sources.append(generateGEMM(name: "gemm_bf16_f32s", bufferPrecision: prefill, weightFormat: .bfloat16))
        sources.append(generateGEMV(name: "gemv_f32s", bufferPrecision: prefill, weightFormat: .float16, tileElements: 256))
        sources.append(generateGEMV(name: "gemv_bf16_f32s", bufferPrecision: prefill, weightFormat: .bfloat16, tileElements: 256))
        sources.append(generateSequenceGEMV(name: "gemv_seq_f32s", bufferPrecision: prefill, weightFormat: .float16, tileElements: 256))
        sources.append(generateSequenceGEMV(name: "gemv_seq_bf16_f32s", bufferPrecision: prefill, weightFormat: .bfloat16, tileElements: 256))
        sources.append(generateBatchedSequenceGEMV(name: "batched_gemv2_seq_f32s", count: 2, bufferPrecision: prefill, weightFormat: .float16))
        sources.append(generateBatchedSequenceGEMV(name: "batched_gemv2_seq_bf16_f32s", count: 2, bufferPrecision: prefill, weightFormat: .bfloat16))
        sources.append(generateBatchedSequenceGEMV(name: "batched_gemv2_seq_fp32_f32s", count: 2, bufferPrecision: prefill, weightFormat: .float32))
        sources.append(generateBatchedSequenceGEMV(name: "batched_gemv3_seq_f32s", count: 3, bufferPrecision: prefill, weightFormat: .float16))
        sources.append(generateBatchedSequenceGEMV(name: "batched_gemv3_seq_bf16_f32s", count: 3, bufferPrecision: prefill, weightFormat: .bfloat16))
        sources.append(generateBatchedSequenceGEMV(name: "batched_gemv3_seq_fp32_f32s", count: 3, bufferPrecision: prefill, weightFormat: .float32))
        sources.append(generateBatchedSequenceGEMV(name: "batched_gemv4_seq_f32s", count: 4, bufferPrecision: prefill, weightFormat: .float16))
        sources.append(generateBatchedSequenceGEMV(name: "batched_gemv4_seq_bf16_f32s", count: 4, bufferPrecision: prefill, weightFormat: .bfloat16))
        sources.append(generateBatchedSequenceGEMV(name: "batched_gemv4_seq_fp32_f32s", count: 4, bufferPrecision: prefill, weightFormat: .float32))
        sources.append(generateEmbeddingLookup(name: "embedding_lookup_seq_f32", bufferPrecision: prefill, weightFormat: .float16))
        sources.append(generateEmbeddingLookup(name: "embedding_lookup_seq_bf16_f32", bufferPrecision: prefill, weightFormat: .bfloat16))
        sources.append(generateEmbeddingLookup(name: "embedding_lookup_seq_fp32_f32", bufferPrecision: prefill, weightFormat: .float32))
        sources.append(generateQuantizedEmbeddingLookupQ4(name: "embedding_lookup_seq_q4_g64_f32", bufferPrecision: prefill, groupSize: 64))
        sources.append(generateQuantizedEmbeddingLookupQ4(name: "embedding_lookup_seq_q4_g128_f32", bufferPrecision: prefill, groupSize: 128))
        sources.append(generateQuantizedEmbeddingLookupQ8(name: "embedding_lookup_seq_q8_g32_f32", bufferPrecision: prefill, groupSize: 32))
        sources.append(generateQuantizedEmbeddingLookupQ8(name: "embedding_lookup_seq_q8_g64_f32", bufferPrecision: prefill, groupSize: 64))
        sources.append(generateQKNormSeq(name: "qk_rms_norm_seq_f32", bufferPrecision: prefill, weightFormat: .bfloat16))
        sources.append(generatePerHeadRMSNorm(name: "per_head_rms_norm_seq_f32", bufferPrecision: prefill, isSequence: true))
        sources.append(generateRoPESeq(name: "rope_seq_f32", bufferPrecision: prefill))
        sources.append(generateConv1dCausalSeq(name: "conv1d_causal_seq_f32", bufferPrecision: prefill, weightFormat: .bfloat16))
        sources.append(generateExtractConvState(name: "extract_conv_state_f32", bufferPrecision: prefill, weightFormat: .bfloat16))
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
        sources.append(generateExtractConvState(name: "extract_conv_state", bufferPrecision: f16, weightFormat: .bfloat16))

        // === F16 GEMM variants ===
        sources.append(generateGEMM(name: "gemm", bufferPrecision: f16, weightFormat: .float16))
        sources.append(generateGEMM(name: "gemm_bf16", bufferPrecision: f16, weightFormat: .bfloat16))

        // === Flash Attention ===
        sources.append(flashAttentionHelperSource)  // helper functions once
        sources.append(generateFlashAttentionKernel(name: "flash_attn_decode", bufferPrecision: decode))
        sources.append(generateFlashAttentionArgumentTableVariant(name: "flash_attn_decode_argbuf", argumentBufferIndex: 30, bufferPrecision: decode))
        sources.append(generateFlashAttentionKernel(name: "flash_attn_decode_f32", bufferPrecision: prefill))
        sources.append(generateBatchFlashAttention(
            name: "flash_attn_batch_f32",
            bufferPrecision: prefill,
            sequenceStorageFormat: .float16
        ))
        sources.append(generateBatchFlashAttention(
            name: "flash_attn_batch_bf16_f32",
            bufferPrecision: prefill,
            sequenceStorageFormat: .bfloat16
        ))
        sources.append(generateDirectScratchBatchFlashAttention(name: "flash_attn_batch_scratch_f32", bufferPrecision: prefill))
        sources.append(generateKVCacheFillSeq(name: "kv_cache_fill_seq_f32", bufferPrecision: prefill))

        // === Quantized GEMV (decode) === format-driven unified scaffold
        sources.append(generateUnifiedQuantizedGEMV(name: "gemv_q4_g64", format: AffineQ4Group64Format(), bufferPrecision: decode))
        sources.append(generateUnifiedQuantizedGEMV(name: "gemv_q4_g128", format: AffineQ4Group128Format(), bufferPrecision: decode))
        sources.append(generateUnifiedQuantizedGEMV(name: "gemv_q8_g32", format: AffineQ8Group32Format(), bufferPrecision: decode))
        sources.append(generateUnifiedQuantizedGEMV(name: "gemv_q8_g64", format: AffineQ8Group64Format(), bufferPrecision: decode))

        // === Quantized GEMM (prefill) ===
        sources.append(generateQuantizedGEMM_Q4(name: "gemm_q4_g64", bufferPrecision: decode, groupSize: 64))
        sources.append(generateQuantizedGEMM_Q4(name: "gemm_q4_g128", bufferPrecision: decode, groupSize: 128))
        sources.append(generateQuantizedGEMM_Q4(name: "gemm_q4_g64_f32s", bufferPrecision: prefill, groupSize: 64))
        sources.append(generateQuantizedGEMM_Q4(name: "gemm_q4_g128_f32s", bufferPrecision: prefill, groupSize: 128))
        sources.append(generateQuantizedGEMM_Q8(name: "gemm_q8_g32", bufferPrecision: decode, groupSize: 32))
        sources.append(generateQuantizedGEMM_Q8(name: "gemm_q8_g64", bufferPrecision: decode, groupSize: 64))
        sources.append(generateQuantizedGEMM_Q8(name: "gemm_q8_g32_f32s", bufferPrecision: prefill, groupSize: 32))
        sources.append(generateQuantizedGEMM_Q8(name: "gemm_q8_g64_f32s", bufferPrecision: prefill, groupSize: 64))

        // === Misc decode kernels ===
        sources.append(generateSigmoidGate(name: "sigmoid_gate", bufferPrecision: decode))
        sources.append(generatePackedSigmoidGate(name: "packed_sigmoid_gate", bufferPrecision: decode, isSequence: false))
        sources.append(generatePackedSigmoidGate(name: "packed_sigmoid_gate_seq_f32", bufferPrecision: prefill))
        sources.append(generatePackedQueryExtract(name: "packed_query_extract", bufferPrecision: decode, isSequence: false))
        sources.append(generatePackedQueryExtract(name: "packed_query_extract_seq_f32", bufferPrecision: prefill))
        sources.append(generateLayerNorm(name: "layer_norm", bufferPrecision: decode, weightFormat: .float16))

        // === Mixed-precision GEMM variants ===
        // gemm_bf16_f32_to_half: F32 input, BF16 weight, F16 output
        // gemm_bf16_f32s_halfout: F32 input, BF16 weight, F16 output (with seqLen)
        // These are transitional — will be replaced by F32 prefill path

        // === SSM recurrence (DeltaNet/Mamba) ===
        sources.append(generateSSMWeightIndependentHelpers())
        sources.append(generateSSMConvSiluHelper(weightFormat: .float16))
        sources.append(generateSSMConvSiluHelper(weightFormat: .bfloat16))
        // Default SSM dimensions for library generation (LFM2-style: 16 groups × 128 dk, 16 heads × 128 dv)
        let defaultHeadCount = 16
        let defaultGroupCount = 16
        let defaultKeyHeadDimension = 128
        let defaultValueHeadDimension = 128
        let defaultConvDimension = 2 * defaultGroupCount * defaultKeyHeadDimension + defaultHeadCount * defaultValueHeadDimension
        let defaultMaxThreadgroupSize = SSMRecurrenceFragment.maxThreadgroupSize
        sources.append(generateSSMRecurrence(name: "ssm_recurrence", bufferPrecision: decode, weightFormat: .float16, convDimension: defaultConvDimension, maxThreadgroupSize: defaultMaxThreadgroupSize, headCount: defaultHeadCount, groupCount: defaultGroupCount, keyHeadDimension: defaultKeyHeadDimension, valueHeadDimension: defaultValueHeadDimension))
        sources.append(generateSSMRecurrence(name: "ssm_recurrence_f32", bufferPrecision: prefill, weightFormat: .float16, convDimension: defaultConvDimension, maxThreadgroupSize: defaultMaxThreadgroupSize, headCount: defaultHeadCount, groupCount: defaultGroupCount, keyHeadDimension: defaultKeyHeadDimension, valueHeadDimension: defaultValueHeadDimension))
        sources.append(generateSSMRecurrenceSequence(name: "ssm_recurrence_seq", bufferPrecision: decode, weightFormat: .float16, convDimension: defaultConvDimension, maxThreadgroupSize: defaultMaxThreadgroupSize, headCount: defaultHeadCount, groupCount: defaultGroupCount, keyHeadDimension: defaultKeyHeadDimension, valueHeadDimension: defaultValueHeadDimension))
        sources.append(generateSSMRecurrenceSequence(name: "ssm_recurrence_seq_f32", bufferPrecision: prefill, weightFormat: .float16, convDimension: defaultConvDimension, maxThreadgroupSize: defaultMaxThreadgroupSize, headCount: defaultHeadCount, groupCount: defaultGroupCount, keyHeadDimension: defaultKeyHeadDimension, valueHeadDimension: defaultValueHeadDimension))
        sources.append(generateSSMRecurrence(name: "ssm_recurrence_bf16", bufferPrecision: decode, weightFormat: .bfloat16, convDimension: defaultConvDimension, maxThreadgroupSize: defaultMaxThreadgroupSize, headCount: defaultHeadCount, groupCount: defaultGroupCount, keyHeadDimension: defaultKeyHeadDimension, valueHeadDimension: defaultValueHeadDimension))
        sources.append(generateSSMRecurrence(name: "ssm_recurrence_bf16_f32", bufferPrecision: prefill, weightFormat: .bfloat16, convDimension: defaultConvDimension, maxThreadgroupSize: defaultMaxThreadgroupSize, headCount: defaultHeadCount, groupCount: defaultGroupCount, keyHeadDimension: defaultKeyHeadDimension, valueHeadDimension: defaultValueHeadDimension))
        sources.append(generateSSMRecurrenceSequence(name: "ssm_recurrence_seq_bf16", bufferPrecision: decode, weightFormat: .bfloat16, convDimension: defaultConvDimension, maxThreadgroupSize: defaultMaxThreadgroupSize, headCount: defaultHeadCount, groupCount: defaultGroupCount, keyHeadDimension: defaultKeyHeadDimension, valueHeadDimension: defaultValueHeadDimension))
        sources.append(generateSSMRecurrenceSequence(name: "ssm_recurrence_seq_bf16_f32", bufferPrecision: prefill, weightFormat: .bfloat16, convDimension: defaultConvDimension, maxThreadgroupSize: defaultMaxThreadgroupSize, headCount: defaultHeadCount, groupCount: defaultGroupCount, keyHeadDimension: defaultKeyHeadDimension, valueHeadDimension: defaultValueHeadDimension))

        // === KV cache quantization ===
        sources.append(kvQuantizationSource)

        // === Mixed-precision GEMM (transitional) ===
        sources.append(gemmMixedPrecisionSource)

        return sources.joined(separator: "\n\n")
    }

    // MARK: - Fragment-Driven GenerationEvent



    // MARK: - Common Header
}
