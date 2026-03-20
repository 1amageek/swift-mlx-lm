/// Generates Metal Shading Language (MSL) source from kernel fragments.
///
/// The generator handles three orthogonal concerns:
/// 1. **Computation** — from the fragment (Reduction, Elementwise, etc.)
/// 2. **Weight reading** — from QuantizationFormat (BF16, FP16, Q4, etc.)
/// 3. **Buffer precision** — from the execution mode (F16 decode, F32 prefill)
///
/// No dtype/precision variants are hardcoded. All combinations are generated
/// from these three independent inputs.
public struct MetalSourceGenerator: Sendable {

    // BufferPrecision and WeightFormat are top-level types in MetalDispatchTypes.swift.
    // These typealiases preserve source compatibility within MetalSourceGenerator.
    public typealias BufferPrecision = MetalCompiler.BufferPrecision
    public typealias WeightFormat = MetalCompiler.WeightFormat

    private enum SpecializedDenseInputStaging {
        case bufferPrecision
        case float

        var stagesAsFloat: Bool {
            switch self {
            case .bufferPrecision:
                return false
            case .float:
                return true
            }
        }
    }

    private enum SpecializedDenseAccumulationStyle {
        case indexed
        case pointerIncrement
    }

    // MARK: - Complete Library Generation

    /// Generate complete MSL source for ALL kernels (decode + prefill).
    ///
    /// This replaces both MetalKernelSource and GeneratedKernelLibrary.
    /// Specialized kernels (flash_attn, ssm, quantized GEMV) are included
    /// as parameterized templates. Simple kernels are generated from parameters.
    ///
    /// - Parameters:
    ///   - weightFormat: Primary weight format (determines weight reader code)
    ///   - specializedSources: Additional MSL source for complex kernels not yet generated
    public static func generateCompleteLibrary(
        weightFormat: WeightFormat
    ) -> String {
        var sources: [String] = [commonHeader]
        let decode = BufferPrecision.float16
        let prefill = BufferPrecision.float32

        // === Decode kernels (F16 buffers, single token) ===
        sources.append(generateReduction(name: "rms_norm", dimension: 0, epsilon: 0, bufferPrecision: decode, weightFormat: .float16, isSequence: false))
        sources.append(generateReduction(name: "rms_norm_bf16", dimension: 0, epsilon: 0, bufferPrecision: decode, weightFormat: .bfloat16, isSequence: false))
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
        sources.append(generateBatchedPerHead2(name: "batched_qk_rms_norm_2", bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateBatchedPerHead2(name: "batched_qk_rms_norm_bf16_2", bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateBatchedPerHead2ArgumentTableVariant(name: "batched_qk_rms_norm_2_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateBatchedPerHead2ArgumentTableVariant(name: "batched_qk_rms_norm_bf16_2_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateFusedSwiGLUProjection(name: "fused_swiglu_projection", bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateFusedSwiGLUProjection(name: "fused_swiglu_projection_bf16", bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateInput2048FusedSwiGLUProjection(name: "fused_swiglu_projection_2048", bufferPrecision: decode, weightFormat: .float16, stagesInputAsFloat: false, unrollFactor: 8))
        sources.append(generateInput2048FusedSwiGLUProjection(name: "fused_swiglu_projection_2048_bf16", bufferPrecision: decode, weightFormat: .bfloat16, stagesInputAsFloat: false, unrollFactor: 8))
        sources.append(generateInput2048FusedSwiGLUProjectionArgumentTableVariant(name: "fused_swiglu_projection_2048_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .float16, stagesInputAsFloat: false, unrollFactor: 8))
        sources.append(generateInput2048FusedSwiGLUProjectionArgumentTableVariant(name: "fused_swiglu_projection_2048_bf16_argbuf", argumentBufferIndex: 30, bufferPrecision: decode, weightFormat: .bfloat16, stagesInputAsFloat: false, unrollFactor: 8))
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
        sources.append(ssmRecurrenceSource)

        // === KV cache quantization ===
        sources.append(kvQuantizationSource)

        // === Mixed-precision GEMM (transitional) ===
        sources.append(gemmMixedPrecisionSource)

        return sources.joined(separator: "\n\n")
    }

    // MARK: - Fragment-Driven Generation



    // MARK: - Common Header

    /// Shared MSL declarations used by all generated kernels.
    public static let commonHeader = """
    #include <metal_stdlib>
    using namespace metal;

    constant constexpr uint SIMD_WIDTH = 32;

    /// BFloat16 → Float32: BF16 is the upper 16 bits of Float32.
    /// Zero-extend to 32 bits — no precision loss, no computation.
    inline float bf16_to_float(uint16_t bf16) {
        uint32_t f32_bits = uint32_t(bf16) << 16;
        return as_type<float>(f32_bits);
    }

    inline uint16_t float_to_bf16(float value) {
        uint32_t bits = as_type<uint32_t>(value);
        uint32_t lsb = (bits >> 16) & 1;
        uint32_t roundingBias = 0x7FFF + lsb;
        uint32_t rounded = bits + roundingBias;
        return uint16_t(rounded >> 16);
    }

    inline float2 bf16x2_to_float2(ushort2 bf16) {
        return float2(bf16_to_float(bf16.x), bf16_to_float(bf16.y));
    }

    inline float4 bf16x4_to_float4(ushort4 bf16) {
        return float4(
            bf16_to_float(bf16.x),
            bf16_to_float(bf16.y),
            bf16_to_float(bf16.z),
            bf16_to_float(bf16.w)
        );
    }

    """

    // MARK: - Generation

    /// Generate MSL source for a reduction kernel (RMSNorm, LayerNorm, Argmax).
    ///
    /// The generated kernel:
    /// 1. Reads input from the buffer using `bufferPrecision` type
    /// 2. Reads weight using `weightFormat` conversion
    /// 3. Performs SIMD reduction across the dimension
    /// 4. Writes output using `bufferPrecision` type
    /// Generate MSL source for a reduction kernel (RMSNorm).
    ///
    /// - Parameter isSequence: true for prefill (operates on [seqLen × dim]), false for decode (single token)
    public static func generateReduction(
        name: String,
        dimension: Int,
        epsilon: Float,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        isSequence: Bool = true
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        if isSequence {
            return """
            kernel void \(name)(
                device const \(bt)* input       [[buffer(0)]],
                device const \(wt)* weight      [[buffer(1)]],
                device \(bt)* output            [[buffer(2)]],
                constant uint& dimension        [[buffer(3)]],
                constant float& epsilon         [[buffer(4)]],
                constant uint& sequenceLength   [[buffer(5)]],
                uint gid_x                      [[threadgroup_position_in_grid]],
                uint tid                        [[thread_index_in_threadgroup]],
                uint threadgroupSize            [[threads_per_threadgroup]]
            ) {
                uint seqPos = gid_x;
                if (seqPos >= sequenceLength) return;

                device const \(bt)* row = input + seqPos * dimension;
                float sumSquared = 0.0f;
                for (uint i = tid; i < dimension; i += threadgroupSize) {
                    float v = float(row[i]);
                    sumSquared += v * v;
                }
                sumSquared = simd_sum(sumSquared);

                threadgroup float shared[32];
                uint simdIndex = tid / SIMD_WIDTH;
                if (tid % SIMD_WIDTH == 0) shared[simdIndex] = sumSquared;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (tid == 0) {
                    float total = 0.0f;
                    uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                    for (uint s = 0; s < sgCount; s++) total += shared[s];
                    shared[0] = rsqrt(total / float(dimension) + epsilon);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float scale = shared[0];
                device \(bt)* outRow = output + seqPos * dimension;
                for (uint i = tid; i < dimension; i += threadgroupSize) {
                    outRow[i] = \(bt)(float(row[i]) * scale * \(readWeight("weight[i]")));
                }
            }
            """
        } else {
            // Decode: single token, in-place on a single hidden buffer.
            // Using one device pointer avoids ambiguous aliasing between
            // separate input/output bindings when the compiler routes both
            // to the same MTLBuffer.
            return """
            kernel void \(name)(
                device \(bt)* data              [[buffer(0)]],
                device const \(wt)* weight      [[buffer(1)]],
                constant uint& dimension        [[buffer(2)]],
                constant float& epsilon         [[buffer(3)]],
                uint tid                        [[thread_index_in_threadgroup]],
                uint threadgroupSize            [[threads_per_threadgroup]]
            ) {
                float sumSquared = 0.0f;
                for (uint i = tid; i < dimension; i += threadgroupSize) {
                    float v = float(data[i]);
                    sumSquared += v * v;
                }
                sumSquared = simd_sum(sumSquared);

                threadgroup float shared[32];
                uint simdIndex = tid / SIMD_WIDTH;
                if (tid % SIMD_WIDTH == 0) shared[simdIndex] = sumSquared;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (tid == 0) {
                    float total = 0.0f;
                    uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                    for (uint s = 0; s < sgCount; s++) total += shared[s];
                    shared[0] = rsqrt(total / float(dimension) + epsilon);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float scale = shared[0];
                for (uint i = tid; i < dimension; i += threadgroupSize) {
                    data[i] = \(bt)(float(data[i]) * scale * \(readWeight("weight[i]")));
                }
            }
            """
        }
    }

    public static func generateReductionArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let inputStructName = "\(name)_args"

        return """
        struct \(inputStructName) {
            device \(bt)* data [[id(0)]];
            device const \(wt)* weight [[id(1)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& dimension                  [[buffer(2)]],
            constant float& epsilon                   [[buffer(3)]],
            uint tid                                  [[thread_index_in_threadgroup]],
            uint threadgroupSize                      [[threads_per_threadgroup]]
        ) {
            float sumSquared = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float v = float(args.data[i]);
                sumSquared += v * v;
            }
            sumSquared = simd_sum(sumSquared);

            threadgroup float shared[32];
            uint simdIndex = tid / SIMD_WIDTH;
            if (tid % SIMD_WIDTH == 0) shared[simdIndex] = sumSquared;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0) {
                float total = 0.0f;
                uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                for (uint s = 0; s < sgCount; s++) total += shared[s];
                shared[0] = rsqrt(total / float(dimension) + epsilon);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                args.data[i] = \(bt)(float(args.data[i]) * scale * \(readWeight("args.weight[i]")));
            }
        }
        """
    }

    /// Generate MSL source for an elementwise kernel (SwiGLU).
    public static func generateSwiGLU(
        name: String,
        bufferPrecision: BufferPrecision,
        isSequence: Bool = true
    ) -> String {
        let bt = bufferPrecision.metalType

        if isSequence {
            return """
            kernel void \(name)(
                device const \(bt)* gate         [[buffer(0)]],
                device const \(bt)* up           [[buffer(1)]],
                device \(bt)* output             [[buffer(2)]],
                constant uint& dimension         [[buffer(3)]],
                constant uint& sequenceLength    [[buffer(4)]],
                uint2 gid                        [[thread_position_in_grid]]
            ) {
                uint i = gid.x;
                uint seqPos = gid.y;
                if (i >= dimension || seqPos >= sequenceLength) return;

                uint idx = seqPos * dimension + i;
                float g = float(gate[idx]);
                float sigmoid = 1.0f / (1.0f + exp(-g));
                output[idx] = \(bt)(g * sigmoid * float(up[idx]));
            }
            """
        } else {
            return """
            kernel void \(name)(
                device const \(bt)* gate         [[buffer(0)]],
                device const \(bt)* up           [[buffer(1)]],
                device \(bt)* output             [[buffer(2)]],
                constant uint& dimension         [[buffer(3)]],
                uint gid                         [[thread_position_in_grid]]
            ) {
                if (gid >= dimension) return;
                float g = float(gate[gid]);
                float sigmoid = 1.0f / (1.0f + exp(-g));
                output[gid] = \(bt)(g * sigmoid * float(up[gid]));
            }
            """
        }
    }

    /// Generate MSL source for a GEMM kernel using Metal Performance Primitives matmul2d.
    ///
    /// Uses Apple's optimized AMX paths via `mpp::tensor_ops::matmul2d`.
    /// Requires Metal language version 4.0.
    /// Buffer layout matches the naive GEMM: input[seqLen×inputDim], weight[outputDim×inputDim], output[seqLen×outputDim].
    public static func generateMPPGEMM(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        // MPP tensor type for weight: bfloat for BF16, half for FP16
        let tensorWeightType = weightFormat == .bfloat16 ? "bfloat" : bt

        return """
        #include <metal_stdlib>
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        using namespace metal;

        kernel void \(name)(
            device \(bt)* input              [[buffer(0)]],
            device \(tensorWeightType)* weight [[buffer(1)]],
            device \(bt)* output             [[buffer(2)]],
            constant uint& inputDimension    [[buffer(3)]],
            constant uint& outputDimension   [[buffer(4)]],
            constant uint& sequenceLength    [[buffer(5)]],
            uint2 tgid [[threadgroup_position_in_grid]]
        ) {
            using namespace mpp::tensor_ops;

            auto A = tensor<device \(bt), dextents<int32_t, 2>, tensor_inline>(
                input, dextents<int32_t, 2>(inputDimension, sequenceLength));
            auto B = tensor<device \(tensorWeightType), dextents<int32_t, 2>, tensor_inline>(
                weight, dextents<int32_t, 2>(inputDimension, outputDimension));
            auto C = tensor<device \(bt), dextents<int32_t, 2>, tensor_inline>(
                output, dextents<int32_t, 2>(outputDimension, sequenceLength));

            constexpr auto desc = matmul2d_descriptor(
                64, 32, dynamic_length_v<int>,
                false, true, false,
                matmul2d_descriptor::mode::multiply);
            matmul2d<desc, execution_simdgroups<4>> op;

            auto mA = A.slice(0, tgid.y * 64);
            auto mB = B.slice(0, tgid.x * 32);
            auto mC = C.slice(tgid.x * 32, tgid.y * 64);
            op.run(mA, mB, mC);
        }
        """
    }

    /// Generate MSL source for a GEMM kernel (prefill projection, naive fallback).
    public static func generateGEMM(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        return """
        kernel void \(name)(
            device const \(bt)* input              [[buffer(0)]],
            device const \(wt)* weight             [[buffer(1)]],
            device \(bt)* output                   [[buffer(2)]],
            constant uint& inputDimension          [[buffer(3)]],
            constant uint& outputDimension         [[buffer(4)]],
            constant uint& sequenceLength          [[buffer(5)]],
            uint2 gid                              [[threadgroup_position_in_grid]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]]
        ) {
            const uint rowsPerThreadgroup = 2;
            const uint row = gid.x * rowsPerThreadgroup + sgitg;
            const uint seqPos = gid.y;
            if (row >= outputDimension || seqPos >= sequenceLength) return;

            float sum = 0.0f;
            device const \(bt)* inputRow = input + seqPos * inputDimension;
            device const \(wt)* weightRow = weight + row * inputDimension;
            for (uint j = tiisg; j < inputDimension; j += SIMD_WIDTH) {
                sum += \(readWeight("weightRow[j]")) * float(inputRow[j]);
            }
            sum = simd_sum(sum);
            if (tiisg == 0) {
                output[seqPos * outputDimension + row] = \(bt)(sum);
            }
        }
        """
    }

    /// Generate MSL source for a GEMV kernel (decode projection, single token).
    ///
    /// Optimization: input is staged into threadgroup memory in tiles and reused
    /// by all rows in the threadgroup. This cuts repeated input reads on the
    /// decode hot path where multiple output rows share the same activation.
    public static func generateGEMV(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        tileElements: Int = 128
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        return """
        kernel void \(name)(
            device const \(bt)* input              [[buffer(0)]],
            device const \(wt)* weight             [[buffer(1)]],
            device \(bt)* output                   [[buffer(2)]],
            constant uint& inputDimension          [[buffer(3)]],
            constant uint& outputDimension         [[buffer(4)]],
            uint gid                               [[threadgroup_position_in_grid]],
            uint tid                               [[thread_index_in_threadgroup]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]],
            uint threadsPerThreadgroup             [[threads_per_threadgroup]]
        ) {
            const uint tileElements = \(tileElements);
            const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup / SIMD_WIDTH);
            const uint row = gid * rowsPerThreadgroup + sgitg;
            if (row >= outputDimension) return;

            threadgroup \(bt) inputTile[tileElements];
            float sum = 0.0f;
            device const \(wt)* weightRow = weight + row * inputDimension;
            for (uint base = 0; base < inputDimension; base += tileElements) {
                for (uint j = tid; j < tileElements; j += threadsPerThreadgroup) {
                    const uint inputIndex = base + j;
                    inputTile[j] = inputIndex < inputDimension ? input[inputIndex] : \(bt)(0.0f);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint tileCount = min(tileElements, inputDimension - base);
                for (uint j = tiisg; j < tileCount; j += SIMD_WIDTH) {
                    sum += \(readWeight("weightRow[base + j]")) * float(inputTile[j]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            sum = simd_sum(sum);
            if (tiisg == 0) {
                output[row] = \(bt)(sum);
            }
        }
        """
    }

    /// Generate a GEMV kernel specialized for vocab/output-head style projections.
    ///
    /// The input dimension is expected to be 2048. The entire input vector is staged
    /// into threadgroup memory once, avoiding the repeated tile barriers used by the
    /// generic large GEMV path.
    public static func generateVocabGEMV(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        return generateSpecializedDenseGEMV(
            name: name,
            bufferPrecision: bufferPrecision,
            weightFormat: weightFormat,
            stagedInputElements: 2_048,
            fixedInputDimension: 2_048,
            inputStaging: .bufferPrecision,
            accumulationStyle: .pointerIncrement
        )
    }

    /// Generate a GEMV kernel specialized for decode projections with inputDimension=2048.
    ///
    /// This family stages the full hidden vector once into threadgroup memory and reuses it
    /// across all rows in the threadgroup. It is used both for the output head and for the
    /// common 2048→{2048,6144,8192} decode projections.
    public static func generateInput2048GEMV(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        fixedOutputDimension: Int? = nil,
        fixedRowsPerThreadgroup: Int? = nil,
        fixedSimdgroups: Int? = nil,
        stagesInputAsFloat: Bool = true,
        weightLayoutPolicy: Input2048WeightLayoutPolicy = .rowMajor,
        unrollFactor: Int = 4
    ) -> String {
        _ = weightLayoutPolicy
        return generateSpecializedDenseGEMV(
            name: name,
            bufferPrecision: bufferPrecision,
            weightFormat: weightFormat,
            stagedInputElements: 2_048,
            fixedInputDimension: 2_048,
            fixedOutputDimension: fixedOutputDimension,
            fixedRowsPerThreadgroup: fixedRowsPerThreadgroup,
            fixedSimdgroups: fixedSimdgroups,
            inputStaging: stagesInputAsFloat ? .float : .bufferPrecision,
            accumulationStyle: .indexed,
            unrollFactor: unrollFactor
        )
    }

    public static func generateInput2048GEMVArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        fixedOutputDimension: Int,
        includesDimensionBindings: Bool = true,
        fixedRowsPerThreadgroup: Int? = nil,
        fixedSimdgroups: Int? = nil,
        stagesInputAsFloat: Bool = true,
        weightLayoutPolicy: Input2048WeightLayoutPolicy = .rowMajor,
        bf16ArgumentReadPolicy: Input2048BF16ArgumentReadPolicy = .scalar,
        unrollFactor: Int = 4
    ) -> String {
        _ = weightLayoutPolicy
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let stagedInputType = stagesInputAsFloat ? "float" : bt
        let stagedInputRead = { (index: String) in "float(inputTile[\(index)])" }
        let stagedInputStore = stagesInputAsFloat ? "inputTile[j] = float(args.input[j]);" : "inputTile[j] = args.input[j];"
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let effectiveUnroll = max(1, unrollFactor)
        let effectiveThreadsPerThreadgroup = fixedSimdgroups.map { "SIMD_WIDTH * \($0)u" } ?? "threadsPerThreadgroup"
        let inputStructName = "\(name)_args"
        let usesPairwiseWeightRead =
            (bf16ArgumentReadPolicy == .pairwise ||
             bf16ArgumentReadPolicy == .pairwisePointerInput ||
             bf16ArgumentReadPolicy == .pairwisePointerFloatInput) &&
            weightFormat == .bfloat16 &&
            effectiveUnroll.isMultiple(of: 2)
        let usesPointerInputRead =
            bf16ArgumentReadPolicy == .pairwisePointerInput
        let usesPointerFloatInputRead = bf16ArgumentReadPolicy == .pairwisePointerFloatInput
        let usesPacked4PointerInputRead =
            (bf16ArgumentReadPolicy == .packed4PointerInput ||
             bf16ArgumentReadPolicy == .packed4FixedPointerInput ||
             bf16ArgumentReadPolicy == .packed4ThreadgroupFixedPointerInput) &&
            weightFormat == .bfloat16 &&
            effectiveUnroll == 4
        let usesPacked4FixedPointerInputRead =
            (bf16ArgumentReadPolicy == .packed4FixedPointerInput ||
             bf16ArgumentReadPolicy == .packed4ThreadgroupFixedPointerInput) &&
            weightFormat == .bfloat16 &&
            effectiveUnroll == 4
        let usesPacked4ThreadgroupFixedPointerInputRead =
            bf16ArgumentReadPolicy == .packed4ThreadgroupFixedPointerInput &&
            weightFormat == .bfloat16 &&
            effectiveUnroll == 4
        let usesBlockedRows8Tile128Layout =
            weightLayoutPolicy == .blockedRows8Tiles128 &&
            usesPairwiseWeightRead &&
            fixedRowsPerThreadgroup == 8 &&
            fixedOutputDimension.isMultiple(of: 8)
        let usesBlockedRows4Tile128Layout =
            weightLayoutPolicy == .blockedRows4Tiles128 &&
            usesPacked4PointerInputRead &&
            fixedRowsPerThreadgroup == 4 &&
            fixedOutputDimension.isMultiple(of: 4)
        let weightRowDeclaration = (usesPairwiseWeightRead || usesPacked4PointerInputRead)
            ? ""
            : "device const \(wt)* weightRow = args.weight + row * 2048u;"
        let pairCount = effectiveUnroll / 2
        let nextIndices = (1..<effectiveUnroll).map { lane in
            "const uint next\(lane) = j + \(lane);"
        }.joined(separator: "\n")
        let unrolledAccumulate = (0..<effectiveUnroll).map { lane -> String in
            if lane == 0 {
                return "sum += \(readWeight("weightRow[j]")) * \(stagedInputRead("j"));"
            }
            return "sum += \(readWeight("weightRow[next\(lane)]")) * \(stagedInputRead("next\(lane)"));"
        }.joined(separator: "\n")
        let pairwiseAccumulate = (0..<pairCount).map { pair -> String in
            let base = pair * 2
            if usesPointerInputRead {
                return """
                    float2 w\(pair) = bf16x2_to_float2(weightLane[\(pair)]);
                    sum += w\(pair).x * float(inputLane[\(base)]);
                    sum += w\(pair).y * float(inputLane[\(base + 1)]);
                    """
            }
            if usesPointerFloatInputRead {
                return """
                    float2 w\(pair) = bf16x2_to_float2(weightLane[\(pair)]);
                    sum += w\(pair).x * inputLane[\(base)];
                    sum += w\(pair).y * inputLane[\(base + 1)];
                    """
            }
            return """
                float2 w\(pair) = bf16x2_to_float2(weightLane[\(pair)]);
                sum += w\(pair).x * \(stagedInputRead("j + \(base)"));
                sum += w\(pair).y * \(stagedInputRead("j + \(base + 1)"));
                """
        }.joined(separator: "\n")
        return """
        struct \(inputStructName) {
            device const \(bt)* input [[id(0)]];
            device const \(wt)* weight [[id(1)]];
            device \(bt)* output [[id(2)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            \(includesDimensionBindings ? "constant uint& inputDimension             [[buffer(3)]],\n            constant uint& outputDimension            [[buffer(4)]]," : "")
            uint gid                                  [[threadgroup_position_in_grid]],
            uint tid                                  [[thread_index_in_threadgroup]],
            uint tiisg                                [[thread_index_in_simdgroup]],
            uint sgitg                                [[simdgroup_index_in_threadgroup]],
            uint threadsPerThreadgroup                [[threads_per_threadgroup]]
        ) {
            const uint stagedInputElements = 2048u;
            const uint rowsPerThreadgroup = \(fixedRowsPerThreadgroup.map { "\($0)u" } ?? "max(1u, threadsPerThreadgroup / SIMD_WIDTH)");
            const uint row = gid * rowsPerThreadgroup + sgitg;
            if (row >= \(fixedOutputDimension)u) return;

            threadgroup \(stagedInputType) inputTile[stagedInputElements];
            for (uint j = tid; j < stagedInputElements; j += \(effectiveThreadsPerThreadgroup)) {
                \(stagedInputStore)
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float sum = 0.0f;
            \(weightRowDeclaration)
            \(usesPacked4PointerInputRead ? """
            \(usesBlockedRows4Tile128Layout
                ? "device const ushort4* weightLane = (device const ushort4*)args.weight + gid * 2048u + sgitg * 32u + tiisg;"
                : usesPacked4ThreadgroupFixedPointerInputRead
                    ? "device const ushort4* weightLane = (device const ushort4*)args.weight + gid * 2048u + sgitg * 512u + tiisg;"
                : "device const ushort4* weightLane = (device const ushort4*)(args.weight + row * 2048u) + tiisg;")
            threadgroup const \(bt)* inputLane = inputTile + tiisg * 4;
            \(usesPacked4FixedPointerInputRead
                ? "for (uint iteration = 0; iteration < stagedInputElements / (SIMD_WIDTH * 4u); ++iteration) {"
                : "for (uint j = tiisg * 4; j < 2048u; j += SIMD_WIDTH * 4) {")
                float4 w = bf16x4_to_float4(weightLane[0]);
                sum += w.x * float(inputLane[0]);
                sum += w.y * float(inputLane[1]);
                sum += w.z * float(inputLane[2]);
                sum += w.w * float(inputLane[3]);
                weightLane += \(usesBlockedRows4Tile128Layout ? "128u" : "SIMD_WIDTH");
                inputLane += SIMD_WIDTH * 4;
            }
            """ : usesPairwiseWeightRead ? """
            \(usesBlockedRows8Tile128Layout
                ? "device const ushort2* weightLane = (device const ushort2*)args.weight + gid * 8192u + sgitg * 64u + tiisg * \(pairCount);"
                : "device const ushort2* weightLane = (device const ushort2*)(args.weight + row * 2048u) + tiisg * \(pairCount);")
            \(usesPointerInputRead ? "threadgroup const \(bt)* inputLane = inputTile + tiisg * \(effectiveUnroll);" : "")
            \(usesPointerFloatInputRead ? "threadgroup const float* inputLane = inputTile + tiisg * \(effectiveUnroll);" : "")
            for (uint j = tiisg * \(effectiveUnroll); j < 2048u; j += SIMD_WIDTH * \(effectiveUnroll)) {
                \(pairwiseAccumulate)
                weightLane += \(usesBlockedRows8Tile128Layout ? "512u" : "SIMD_WIDTH * \(pairCount)");
                \(usesPointerInputRead || usesPointerFloatInputRead ? "inputLane += SIMD_WIDTH * \(effectiveUnroll);" : "")
            }
            """ : """
            for (uint j = tiisg * \(effectiveUnroll); j < 2048u; j += SIMD_WIDTH * \(effectiveUnroll)) {
                \(nextIndices)
                \(unrolledAccumulate)
            }
            """)
            sum = simd_sum(sum);
            if (tiisg == 0) {
                args.output[row] = \(bt)(sum);
            }
        }
        """
    }

    public static func generateVocabGEMVArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let inputStructName = "\(name)_args"

        return """
        struct \(inputStructName) {
            device const \(bt)* input [[id(0)]];
            device const \(wt)* weight [[id(1)]];
            device \(bt)* output [[id(2)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& inputDimension             [[buffer(3)]],
            constant uint& outputDimension            [[buffer(4)]],
            uint gid                                  [[threadgroup_position_in_grid]],
            uint tid                                  [[thread_index_in_threadgroup]],
            uint tiisg                                [[thread_index_in_simdgroup]],
            uint sgitg                                [[simdgroup_index_in_threadgroup]],
            uint threadsPerThreadgroup                [[threads_per_threadgroup]]
        ) {
            const uint stagedInputElements = 2048u;
            const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup / SIMD_WIDTH);
            const uint row = gid * rowsPerThreadgroup + sgitg;
            if (row >= outputDimension) return;

            threadgroup \(bt) inputTile[stagedInputElements];
            for (uint j = tid; j < stagedInputElements; j += threadsPerThreadgroup) {
                inputTile[j] = args.input[j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float sum = 0.0f;
            device const \(wt)* weightRow = args.weight + row * 2048u + tiisg;
            threadgroup const \(bt)* inputLane = inputTile + tiisg;
            for (uint j = tiisg; j < 2048u; j += SIMD_WIDTH) {
                sum += \(readWeight("weightRow[0]")) * float(inputLane[0]);
                weightRow += SIMD_WIDTH;
                inputLane += SIMD_WIDTH;
            }
            sum = simd_sum(sum);
            if (tiisg == 0) {
                args.output[row] = \(bt)(sum);
            }
        }
        """
    }

    /// Generate a GEMV kernel specialized for decode projections with inputDimension=8192.
    ///
    /// This family keeps the tiled structure to preserve occupancy, but fixes the
    /// input dimension and tile size so the inner loop can avoid dynamic bounds work.
    public static func generateInput8192TiledGEMV(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        stagesInputAsFloat: Bool = true,
        fixedOutputDimension: Int? = nil,
        tileElements: Int = 1_024,
        unrollFactor: Int = 4
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let stagedInputType = stagesInputAsFloat ? "float" : bt
        let stagedInputRead = stagesInputAsFloat ? "" : "float"
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let effectiveUnroll = max(1, unrollFactor)
        let unrolledAccumulate = (0..<effectiveUnroll).map { lane -> String in
            if lane == 0 {
                return "sum += \(readWeight("tileWeight[0]")) * \(stagedInputRead)(tileInput[0]);"
            }
            let offset = "\(lane)"
            return "sum += \(readWeight("tileWeight[\(offset)]")) * \(stagedInputRead)(tileInput[\(offset)]);"
        }.joined(separator: "\n")

        return """
        kernel void \(name)(
            device const \(bt)* input              [[buffer(0)]],
            device const \(wt)* weight             [[buffer(1)]],
            device \(bt)* output                   [[buffer(2)]],
            constant uint& inputDimension          [[buffer(3)]],
            constant uint& outputDimension         [[buffer(4)]],
            uint gid                               [[threadgroup_position_in_grid]],
            uint tid                               [[thread_index_in_threadgroup]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]],
            uint threadsPerThreadgroup             [[threads_per_threadgroup]]
        ) {
            const uint fixedInputDimension = 8192u;
            const uint stagedInputElements = \(tileElements);
            const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup / SIMD_WIDTH);
            const uint row = gid * rowsPerThreadgroup + sgitg;
            if (row >= \(fixedOutputDimension.map { "\($0)u" } ?? "outputDimension")) return;

            threadgroup \(stagedInputType) inputTile[stagedInputElements];
            float sum = 0.0f;
            device const \(wt)* weightRow = weight + row * fixedInputDimension;
            for (uint base = 0; base < fixedInputDimension; base += stagedInputElements) {
                device const \(bt)* inputTileSource = input + base + tid;
                for (uint j = tid; j < stagedInputElements; j += threadsPerThreadgroup) {
                    inputTile[j] = \(stagesInputAsFloat ? "float(inputTileSource[0])" : "inputTileSource[0]");
                    inputTileSource += threadsPerThreadgroup;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                device const \(wt)* tileWeight = weightRow + base + tiisg * \(effectiveUnroll);
                threadgroup const \(stagedInputType)* tileInput = inputTile + tiisg * \(effectiveUnroll);
                for (uint j = tiisg * \(effectiveUnroll); j < stagedInputElements; j += SIMD_WIDTH * \(effectiveUnroll)) {
                    \(unrolledAccumulate)
                    tileWeight += SIMD_WIDTH * \(effectiveUnroll);
                    tileInput += SIMD_WIDTH * \(effectiveUnroll);
                }
                if (base + stagedInputElements < fixedInputDimension) {
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            sum = simd_sum(sum);
            if (tiisg == 0) {
                output[row] = \(bt)(sum);
            }
        }
        """
    }

    public static func generateInput8192TiledGEMVArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        fixedOutputDimension: Int? = nil,
        includesDimensionBindings: Bool = true,
        tileElements: Int = 1_024,
        unrollFactor: Int = 4
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let stagedInputType = "float"
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let effectiveUnroll = max(1, unrollFactor)
        let inputStructName = "\(name)_args"
        let unrolledAccumulate = (0..<effectiveUnroll).map { lane -> String in
            "sum += \(readWeight("tileWeight[\(lane)]")) * (tileInput[\(lane)]);"
        }.joined(separator: "\n")

        return """
        struct \(inputStructName) {
            device const \(bt)* input [[id(0)]];
            device const \(wt)* weight [[id(1)]];
            device \(bt)* output [[id(2)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            \(includesDimensionBindings ? "constant uint& inputDimension             [[buffer(3)]],\n            constant uint& outputDimension            [[buffer(4)]]," : "")
            uint gid                                  [[threadgroup_position_in_grid]],
            uint tid                                  [[thread_index_in_threadgroup]],
            uint tiisg                                [[thread_index_in_simdgroup]],
            uint sgitg                                [[simdgroup_index_in_threadgroup]],
            uint threadsPerThreadgroup                [[threads_per_threadgroup]]
        ) {
            const uint fixedInputDimension = 8192u;
            const uint stagedInputElements = \(tileElements);
            const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup / SIMD_WIDTH);
            const uint row = gid * rowsPerThreadgroup + sgitg;
            if (row >= \(fixedOutputDimension.map { "\($0)u" } ?? "outputDimension")) return;

            threadgroup \(stagedInputType) inputTile[stagedInputElements];
            float sum = 0.0f;
            device const \(wt)* weightRow = args.weight + row * fixedInputDimension;
            for (uint base = 0; base < fixedInputDimension; base += stagedInputElements) {
                device const \(bt)* inputTileSource = args.input + base + tid;
                for (uint j = tid; j < stagedInputElements; j += threadsPerThreadgroup) {
                    inputTile[j] = float(inputTileSource[0]);
                    inputTileSource += threadsPerThreadgroup;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                device const \(wt)* tileWeight = weightRow + base + tiisg * \(effectiveUnroll);
                threadgroup const \(stagedInputType)* tileInput = inputTile + tiisg * \(effectiveUnroll);
                for (uint j = tiisg * \(effectiveUnroll); j < stagedInputElements; j += SIMD_WIDTH * \(effectiveUnroll)) {
                    \(unrolledAccumulate)
                    tileWeight += SIMD_WIDTH * \(effectiveUnroll);
                    tileInput += SIMD_WIDTH * \(effectiveUnroll);
                }
                if (base + stagedInputElements < fixedInputDimension) {
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            sum = simd_sum(sum);
            if (tiisg == 0) {
                args.output[row] = \(bt)(sum);
            }
        }
        """
    }

    public static func generateGEMVArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        tileElements: Int = 256
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let inputStructName = "\(name)_args"
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        return """
        struct \(inputStructName) {
            device const \(bt)* input [[id(0)]];
            device const \(wt)* weight [[id(1)]];
            device \(bt)* output [[id(2)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& inputDimension             [[buffer(3)]],
            constant uint& outputDimension            [[buffer(4)]],
            uint gid                                  [[threadgroup_position_in_grid]],
            uint tid                                  [[thread_index_in_threadgroup]],
            uint tiisg                                [[thread_index_in_simdgroup]],
            uint sgitg                                [[simdgroup_index_in_threadgroup]],
            uint threadsPerThreadgroup                [[threads_per_threadgroup]]
        ) {
            const uint tileElements = \(tileElements);
            const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup / SIMD_WIDTH);
            const uint row = gid * rowsPerThreadgroup + sgitg;
            if (row >= outputDimension) return;

            threadgroup \(bt) inputTile[tileElements];
            float sum = 0.0f;
            device const \(wt)* weightRow = args.weight + row * inputDimension;
            for (uint base = 0; base < inputDimension; base += tileElements) {
                for (uint j = tid; j < tileElements; j += threadsPerThreadgroup) {
                    const uint inputIndex = base + j;
                    inputTile[j] = inputIndex < inputDimension ? args.input[inputIndex] : \(bt)(0.0f);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint tileCount = min(tileElements, inputDimension - base);
                for (uint j = tiisg; j < tileCount; j += SIMD_WIDTH) {
                    sum += \(readWeight("weightRow[base + j]")) * float(inputTile[j]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            sum = simd_sum(sum);
            if (tiisg == 0) {
                args.output[row] = \(bt)(sum);
            }
        }
        """
    }

    private static func generateSpecializedDenseGEMV(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        stagedInputElements: Int,
        fixedInputDimension: Int? = nil,
        fixedOutputDimension: Int? = nil,
        fixedRowsPerThreadgroup: Int? = nil,
        fixedSimdgroups: Int? = nil,
        inputStaging: SpecializedDenseInputStaging = .bufferPrecision,
        accumulationStyle: SpecializedDenseAccumulationStyle = .indexed,
        unrollFactor: Int = 4,
        forcePointerIncrementLoop: Bool = false
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let stagesInputAsFloat = inputStaging.stagesAsFloat
        let stagedInputType = stagesInputAsFloat ? "float" : bt
        let stagedInputRead = stagesInputAsFloat ? "" : "float"
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let effectiveUnroll = max(1, unrollFactor)
        let inputDimensionExpr = fixedInputDimension.map { "\($0)u" } ?? "inputDimension"
        let outputDimensionExpr = fixedOutputDimension.map { "\($0)u" } ?? "outputDimension"
        let effectiveThreadsPerThreadgroupExpr = fixedSimdgroups.map { "SIMD_WIDTH * \($0)u" } ?? "threadsPerThreadgroup"
        let rowsPerThreadgroupExpr = fixedRowsPerThreadgroup.map { "\($0)u" } ?? "max(1u, threadsPerThreadgroup / SIMD_WIDTH)"
        let canElideInputBounds = if let fixedInputDimension {
            fixedInputDimension % (32 * effectiveUnroll) == 0
        } else {
            false
        }
        let unrolledAccumulate = (0..<effectiveUnroll).map { lane -> String in
            let accumulator = "sum"
            if lane == 0 {
                return "\(accumulator) += \(readWeight("weightRow[j]")) * \(stagedInputRead)(inputTile[j]);"
            }
            let offset = "\(lane)"
            let nextName = "next\(lane)"
            if canElideInputBounds {
                return "\(accumulator) += \(readWeight("weightRow[j + \(offset)]")) * \(stagedInputRead)(inputTile[j + \(offset)]);"
            }
            return """
                const uint \(nextName) = j + \(offset);
                if (\(nextName) < \(inputDimensionExpr)) {
                    \(accumulator) += \(readWeight("weightRow[\(nextName)]")) * \(stagedInputRead)(inputTile[\(nextName)]);
                }
                """
        }.joined(separator: "\n")
        let pointerAccumulate = (0..<effectiveUnroll).map { lane -> String in
            "sum += \(readWeight("weightLane[\(lane)]")) * \(stagedInputRead)(inputLane[\(lane)]);"
        }.joined(separator: "\n")
        let inputTileLoad: String
        if let fixedInputDimension, fixedInputDimension == stagedInputElements {
            inputTileLoad = stagesInputAsFloat ? "inputTile[j] = float(input[j]);" : "inputTile[j] = input[j];"
        } else {
            inputTileLoad = stagesInputAsFloat
                ? "inputTile[j] = j < \(inputDimensionExpr) ? float(input[j]) : 0.0f;"
                : "inputTile[j] = j < \(inputDimensionExpr) ? input[j] : \(bt)(0.0f);"
        }
        let usePointerIncrementLoop: Bool
        switch accumulationStyle {
        case .indexed:
            usePointerIncrementLoop = canElideInputBounds && forcePointerIncrementLoop
        case .pointerIncrement:
            usePointerIncrementLoop = canElideInputBounds
        }
        return """
        kernel void \(name)(
            device const \(bt)* input              [[buffer(0)]],
            device const \(wt)* weight             [[buffer(1)]],
            device \(bt)* output                   [[buffer(2)]],
            constant uint& inputDimension          [[buffer(3)]],
            constant uint& outputDimension         [[buffer(4)]],
            uint gid                               [[threadgroup_position_in_grid]],
            uint tid                               [[thread_index_in_threadgroup]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]],
            uint threadsPerThreadgroup             [[threads_per_threadgroup]]
        ) {
            const uint stagedInputElements = \(stagedInputElements);
            const uint rowsPerThreadgroup = \(rowsPerThreadgroupExpr);
            const uint row = gid * rowsPerThreadgroup + sgitg;
            if (row >= \(outputDimensionExpr)) return;

            threadgroup \(stagedInputType) inputTile[stagedInputElements];
            for (uint j = tid; j < stagedInputElements; j += \(effectiveThreadsPerThreadgroupExpr)) {
                \(inputTileLoad)
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float sum = 0.0f;
            device const \(wt)* weightRow = weight + row * \(inputDimensionExpr);
            \(usePointerIncrementLoop ? """
            device const \(wt)* weightLane = weightRow + tiisg * \(effectiveUnroll);
            threadgroup const \(stagedInputType)* inputLane = inputTile + tiisg * \(effectiveUnroll);
            for (uint j = tiisg * \(effectiveUnroll); j < \(inputDimensionExpr); j += SIMD_WIDTH * \(effectiveUnroll)) {
                \(pointerAccumulate)
                weightLane += SIMD_WIDTH * \(effectiveUnroll);
                inputLane += SIMD_WIDTH * \(effectiveUnroll);
            }
            """ : """
            for (uint j = tiisg * \(effectiveUnroll); j < \(inputDimensionExpr); j += SIMD_WIDTH * \(effectiveUnroll)) {
                \(unrolledAccumulate)
            }
            """)
            sum = simd_sum(sum);
            if (tiisg == 0) {
                output[row] = \(bt)(sum);
            }
        }
        """
    }

    /// Generate MSL source for buffer copy.
    public static func generateCopy(
        name: String,
        bufferPrecision: BufferPrecision,
        isSequence: Bool = true
    ) -> String {
        let bt = bufferPrecision.metalType

        if isSequence {
            return """
            kernel void \(name)(
                device const \(bt)* source       [[buffer(0)]],
                device \(bt)* destination        [[buffer(1)]],
                constant uint& dimension         [[buffer(2)]],
                constant uint& sequenceLength    [[buffer(3)]],
                uint2 gid                        [[thread_position_in_grid]]
            ) {
                uint i = gid.x;
                uint seqPos = gid.y;
                if (i >= dimension || seqPos >= sequenceLength) return;
                destination[seqPos * dimension + i] = source[seqPos * dimension + i];
            }
            """
        } else {
            return """
            kernel void \(name)(
                device const \(bt)* input        [[buffer(0)]],
                device \(bt)* output             [[buffer(1)]],
                constant uint& count             [[buffer(2)]],
                uint gid                         [[thread_position_in_grid]]
            ) {
                if (gid >= count) return;
                output[gid] = input[gid];
            }
            """
        }
    }

    /// Generate MSL source for residual add.
    public static func generateResidualAdd(
        name: String,
        bufferPrecision: BufferPrecision,
        isSequence: Bool = true
    ) -> String {
        let bt = bufferPrecision.metalType

        if isSequence {
            return """
            kernel void \(name)(
                device \(bt)* hidden             [[buffer(0)]],
                device const \(bt)* residual     [[buffer(1)]],
                device \(bt)* output             [[buffer(2)]],
                constant uint& dimension         [[buffer(3)]],
                constant uint& sequenceLength    [[buffer(4)]],
                uint2 gid                        [[thread_position_in_grid]]
            ) {
                uint i = gid.x;
                uint seqPos = gid.y;
                if (i >= dimension || seqPos >= sequenceLength) return;
                uint idx = seqPos * dimension + i;
                output[idx] = \(bt)(float(hidden[idx]) + float(residual[idx]));
            }
            """
        } else {
            return """
            kernel void \(name)(
                device const \(bt)* input        [[buffer(0)]],
                device const \(bt)* residual     [[buffer(1)]],
                device \(bt)* output             [[buffer(2)]],
                constant uint& count             [[buffer(3)]],
                uint gid                         [[thread_position_in_grid]]
            ) {
                if (gid >= count) return;
                output[gid] = \(bt)(float(input[gid]) + float(residual[gid]));
            }
            """
        }
    }

    public static func generateResidualAddArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision
    ) -> String {
        let bt = bufferPrecision.metalType
        let inputStructName = "\(name)_args"

        return """
        struct \(inputStructName) {
            device const \(bt)* input [[id(0)]];
            device const \(bt)* residual [[id(1)]];
            device \(bt)* output [[id(2)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& count                      [[buffer(3)]],
            uint gid                                  [[thread_position_in_grid]]
        ) {
            if (gid >= count) return;
            args.output[gid] = \(bt)(float(args.input[gid]) + float(args.residual[gid]));
        }
        """
    }

    /// Generate MSL source for embedding lookup.
    public static func generateEmbeddingLookup(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        isSequence: Bool = true
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        if isSequence {
            return """
            kernel void \(name)(
                device const int* tokenIDs       [[buffer(0)]],
                device const \(wt)* table        [[buffer(1)]],
                device \(bt)* output             [[buffer(2)]],
                constant uint& embeddingDim      [[buffer(3)]],
                constant uint& sequenceLength    [[buffer(4)]],
                uint2 gid                        [[thread_position_in_grid]]
            ) {
                uint dim = gid.x;
                uint seqPos = gid.y;
                if (dim >= embeddingDim || seqPos >= sequenceLength) return;
                int tokenID = tokenIDs[seqPos];
                output[seqPos * embeddingDim + dim] = \(bt)(\(readWeight("table[tokenID * embeddingDim + dim]")));
            }
            """
        } else {
            return """
            kernel void \(name)(
                device const int* tokenID        [[buffer(0)]],
                device const \(wt)* table        [[buffer(1)]],
                device \(bt)* output             [[buffer(2)]],
                constant uint& embeddingDim      [[buffer(3)]],
                uint gid                         [[thread_position_in_grid]]
            ) {
                if (gid >= embeddingDim) return;
                output[gid] = \(bt)(\(readWeight("table[tokenID[0] * embeddingDim + gid]")));
            }
            """
        }
    }

    public static func generateEmbeddingLookupArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let inputStructName = "\(name)_args"

        return """
        struct \(inputStructName) {
            device const int* tokenID [[id(0)]];
            device const \(wt)* table [[id(1)]];
            device \(bt)* output [[id(2)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& embeddingDim               [[buffer(3)]],
            uint gid                                  [[thread_position_in_grid]]
        ) {
            if (gid >= embeddingDim) return;
            args.output[gid] = \(bt)(\(readWeight("args.table[args.tokenID[0] * embeddingDim + gid]")));
        }
        """
    }

    public static func generateArgmaxArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision
    ) -> String {
        let bt = bufferPrecision.metalType
        let inputStructName = "\(name)_args"

        return """
        struct \(inputStructName) {
            device const \(bt)* logits [[id(0)]];
            device int* result [[id(1)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& vocabularySize             [[buffer(2)]],
            uint tid                                  [[thread_index_in_threadgroup]],
            uint threadgroupSize                      [[threads_per_threadgroup]]
        ) {
            threadgroup float sharedValues[32];
            threadgroup int sharedIndices[32];

            float localMax = -HUGE_VALF;
            int localIndex = 0;
            for (uint i = tid; i < vocabularySize; i += threadgroupSize) {
                float value = float(args.logits[i]);
                if (value > localMax) { localMax = value; localIndex = int(i); }
            }

            for (uint offset = SIMD_WIDTH / 2; offset > 0; offset >>= 1) {
                float otherValue = simd_shuffle_down(localMax, offset);
                int otherIndex = simd_shuffle_down(localIndex, offset);
                if (otherValue > localMax) { localMax = otherValue; localIndex = otherIndex; }
            }

            uint simdIndex = tid / SIMD_WIDTH;
            if (tid % SIMD_WIDTH == 0) {
                sharedValues[simdIndex] = localMax;
                sharedIndices[simdIndex] = localIndex;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0) {
                float bestValue = -HUGE_VALF;
                int bestIndex = 0;
                uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                for (uint i = 0; i < sgCount; i++) {
                    if (sharedValues[i] > bestValue) {
                        bestValue = sharedValues[i];
                        bestIndex = sharedIndices[i];
                    }
                }
                args.result[0] = bestIndex;
            }
        }
        """
    }

    // MARK: - Fused Kernels

    /// Generate fused copy + RMSNorm: copy(hidden → residual) then norm(hidden → scratch).
    public static func generateFusedCopyRMSNorm(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        return """
        kernel void \(name)(
            device const \(bt)* hidden       [[buffer(0)]],
            device \(bt)* residual           [[buffer(1)]],
            device const \(wt)* weight       [[buffer(2)]],
            device \(bt)* scratch            [[buffer(3)]],
            constant uint& dimension         [[buffer(4)]],
            constant float& epsilon          [[buffer(5)]],
            uint tid                         [[thread_index_in_threadgroup]],
            uint tiisg                       [[thread_index_in_simdgroup]],
            uint sgitg                       [[simdgroup_index_in_threadgroup]],
            uint threadgroupSize             [[threads_per_threadgroup]]
        ) {
            // Copy hidden → residual
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                residual[i] = hidden[i];
            }
            threadgroup_barrier(mem_flags::mem_device);

            // RMSNorm hidden → scratch
            float sumSquared = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float v = float(hidden[i]);
                sumSquared += v * v;
            }
            sumSquared = simd_sum(sumSquared);

            threadgroup float shared[32];
            if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = sumSquared;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sgitg == 0) {
                const uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                float total = tiisg < sgCount ? shared[tiisg] : 0.0f;
                total = simd_sum(total);
                if (tiisg == 0) {
                shared[0] = rsqrt(total / float(dimension) + epsilon);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                scratch[i] = \(bt)(float(hidden[i]) * scale * \(readWeight("weight[i]")));
            }
        }
        """
    }

    public static func generateFusedCopyRMSNormArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let inputStructName = "\(name)_args"

        return """
        struct \(inputStructName) {
            device const \(bt)* hidden [[id(0)]];
            device \(bt)* residual [[id(1)]];
            device const \(wt)* weight [[id(2)]];
            device \(bt)* scratch [[id(3)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& dimension                  [[buffer(4)]],
            constant float& epsilon                   [[buffer(5)]],
            uint tid                                  [[thread_index_in_threadgroup]],
            uint tiisg                                [[thread_index_in_simdgroup]],
            uint sgitg                                [[simdgroup_index_in_threadgroup]],
            uint threadgroupSize                      [[threads_per_threadgroup]]
        ) {
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                args.residual[i] = args.hidden[i];
            }
            threadgroup_barrier(mem_flags::mem_device);

            float sumSquared = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float v = float(args.hidden[i]);
                sumSquared += v * v;
            }
            sumSquared = simd_sum(sumSquared);

            threadgroup float shared[32];
            if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = sumSquared;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sgitg == 0) {
                const uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                float total = tiisg < sgCount ? shared[tiisg] : 0.0f;
                total = simd_sum(total);
                if (tiisg == 0) {
                    shared[0] = rsqrt(total / float(dimension) + epsilon);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                args.scratch[i] = \(bt)(float(args.hidden[i]) * scale * \(readWeight("args.weight[i]")));
            }
        }
        """
    }

    /// Generate fused residualAdd + copy + RMSNorm.
    public static func generateFusedResidualAddCopyRMSNorm(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        return """
        kernel void \(name)(
            device \(bt)* hidden             [[buffer(0)]],
            device \(bt)* residual           [[buffer(1)]],
            device const \(wt)* weight       [[buffer(2)]],
            device \(bt)* scratch            [[buffer(3)]],
            constant uint& dimension         [[buffer(4)]],
            constant float& epsilon          [[buffer(5)]],
            uint tid                         [[thread_index_in_threadgroup]],
            uint tiisg                       [[thread_index_in_simdgroup]],
            uint sgitg                       [[simdgroup_index_in_threadgroup]],
            uint threadgroupSize             [[threads_per_threadgroup]]
        ) {
            // Compute RMS on the unrounded residual sum to avoid accumulating
            // an extra F16 rounding step at every pre-norm block.
            float sumSquared = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(hidden[i]) + float(residual[i]);
                sumSquared += h * h;
            }
            sumSquared = simd_sum(sumSquared);

            threadgroup float shared[32];
            if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = sumSquared;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sgitg == 0) {
                const uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                float total = tiisg < sgCount ? shared[tiisg] : 0.0f;
                total = simd_sum(total);
                if (tiisg == 0) {
                shared[0] = rsqrt(total / float(dimension) + epsilon);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(hidden[i]) + float(residual[i]);
                hidden[i] = \(bt)(h);
                residual[i] = \(bt)(h);
                scratch[i] = \(bt)(h * scale * \(readWeight("weight[i]")));
            }
        }
        """
    }

    public static func generateFusedResidualAddCopyRMSNormArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let inputStructName = "\(name)_args"
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        return """
        struct \(inputStructName) {
            device \(bt)* hidden [[id(0)]];
            device \(bt)* residual [[id(1)]];
            device const \(wt)* weight [[id(2)]];
            device \(bt)* scratch [[id(3)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& dimension                  [[buffer(4)]],
            constant float& epsilon                   [[buffer(5)]],
            uint tid                                  [[thread_index_in_threadgroup]],
            uint tiisg                                [[thread_index_in_simdgroup]],
            uint sgitg                                [[simdgroup_index_in_threadgroup]],
            uint threadgroupSize                      [[threads_per_threadgroup]]
        ) {
            float sumSquared = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(args.hidden[i]) + float(args.residual[i]);
                sumSquared += h * h;
            }
            sumSquared = simd_sum(sumSquared);

            threadgroup float shared[32];
            if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = sumSquared;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sgitg == 0) {
                const uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                float total = tiisg < sgCount ? shared[tiisg] : 0.0f;
                total = simd_sum(total);
                if (tiisg == 0) {
                    shared[0] = rsqrt(total / float(dimension) + epsilon);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(args.hidden[i]) + float(args.residual[i]);
                args.hidden[i] = \(bt)(h);
                args.residual[i] = \(bt)(h);
                args.scratch[i] = \(bt)(h * scale * \(readWeight("args.weight[i]")));
            }
        }
        """
    }

    // MARK: - Fused Residual Add + RMS Norm (no copy)

    /// Generate fused residual add + RMS norm kernel (no copy to residual).
    /// Used at model end where no next residual block follows.
    public static func generateFusedResidualAddRMSNorm(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        return """
        kernel void \(name)(
            device \(bt)* hidden             [[buffer(0)]],
            device const \(bt)* residual     [[buffer(1)]],
            device const \(wt)* weight       [[buffer(2)]],
            device \(bt)* output             [[buffer(3)]],
            constant uint& dimension         [[buffer(4)]],
            constant float& epsilon          [[buffer(5)]],
            uint tid                         [[thread_index_in_threadgroup]],
            uint tiisg                       [[thread_index_in_simdgroup]],
            uint sgitg                       [[simdgroup_index_in_threadgroup]],
            uint threadgroupSize             [[threads_per_threadgroup]]
        ) {
            // Compute RMS on the unrounded residual sum to avoid a redundant
            // activation quantization step before normalization.
            float sumSquared = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(hidden[i]) + float(residual[i]);
                sumSquared += h * h;
            }
            sumSquared = simd_sum(sumSquared);

            threadgroup float shared[32];
            if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = sumSquared;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sgitg == 0) {
                const uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                float total = tiisg < sgCount ? shared[tiisg] : 0.0f;
                total = simd_sum(total);
                if (tiisg == 0) {
                shared[0] = rsqrt(total / float(dimension) + epsilon);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(hidden[i]) + float(residual[i]);
                hidden[i] = \(bt)(h);
                output[i] = \(bt)(h * scale * \(readWeight("weight[i]")));
            }
        }
        """
    }

    public static func generateFusedResidualAddRMSNormArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let inputStructName = "\(name)_args"

        return """
        struct \(inputStructName) {
            device \(bt)* hidden [[id(0)]];
            device const \(bt)* residual [[id(1)]];
            device const \(wt)* weight [[id(2)]];
            device \(bt)* output [[id(3)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args        [[buffer(\(argumentBufferIndex))]],
            constant uint& dimension                 [[buffer(4)]],
            constant float& epsilon                 [[buffer(5)]],
            uint tid                                [[thread_index_in_threadgroup]],
            uint tiisg                              [[thread_index_in_simdgroup]],
            uint sgitg                              [[simdgroup_index_in_threadgroup]],
            uint threadgroupSize                    [[threads_per_threadgroup]]
        ) {
            float sumSquared = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(args.hidden[i]) + float(args.residual[i]);
                sumSquared += h * h;
            }
            sumSquared = simd_sum(sumSquared);

            threadgroup float shared[32];
            if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = sumSquared;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sgitg == 0) {
                const uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                float total = tiisg < sgCount ? shared[tiisg] : 0.0f;
                total = simd_sum(total);
                if (tiisg == 0) {
                    shared[0] = rsqrt(total / float(dimension) + epsilon);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(args.hidden[i]) + float(args.residual[i]);
                args.hidden[i] = \(bt)(h);
                args.output[i] = \(bt)(h * scale * \(readWeight("args.weight[i]")));
            }
        }
        """
    }

    // MARK: - Batched GEMV

    /// Generate fused gate_proj + up_proj + SwiGLU kernel for decode.
    ///
    /// Each row computes both branch projections from the shared input vector,
    /// applies `silu(gate) * up`, and writes the activated intermediate directly
    /// to scratch[0]. This avoids storing the two branch projections separately.
    public static func generateFusedSwiGLUProjection(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        return """
        kernel void \(name)(
            device const \(bt)* input          [[buffer(0)]],
            device const \(wt)* gateWeight     [[buffer(1)]],
            device const \(wt)* upWeight       [[buffer(2)]],
            device \(bt)* output               [[buffer(3)]],
            constant uint& inputDimension      [[buffer(4)]],
            constant uint& outputDimension     [[buffer(5)]],
            uint gid                           [[threadgroup_position_in_grid]],
            uint tid                           [[thread_index_in_threadgroup]],
            uint tiisg                         [[thread_index_in_simdgroup]],
            uint sgitg                         [[simdgroup_index_in_threadgroup]],
            uint threadsPerThreadgroup         [[threads_per_threadgroup]]
        ) {
            const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup / SIMD_WIDTH);
            const uint row = gid * rowsPerThreadgroup + sgitg;
            if (row >= outputDimension) return;

            float gateSum = 0.0f;
            float upSum = 0.0f;
            device const \(wt)* gateRow = gateWeight + row * inputDimension;
            device const \(wt)* upRow = upWeight + row * inputDimension;
            for (uint j = tiisg; j < inputDimension; j += SIMD_WIDTH) {
                float x = float(input[j]);
                gateSum += \(readWeight("gateRow[j]")) * x;
                upSum += \(readWeight("upRow[j]")) * x;
            }
            gateSum = simd_sum(gateSum);
            upSum = simd_sum(upSum);
            if (tiisg == 0) {
                float sig = 1.0f / (1.0f + exp(-gateSum));
                output[row] = \(bt)(gateSum * sig * upSum);
            }
        }
        """
    }

    /// Generate a decode-only fused gate/up projection specialized for inputDimension=2048.
    ///
    /// This mirrors the exact-shape dense GEMV family: the full hidden vector is staged once
    /// into threadgroup memory, then both branch projections accumulate from that shared tile.
    public static func generateInput2048FusedSwiGLUProjection(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        stagesInputAsFloat: Bool = true,
        unrollFactor: Int = 4
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let stagedInputType = stagesInputAsFloat ? "float" : bt
        let stagedInputRead = stagesInputAsFloat ? "" : "float"
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let effectiveUnroll = max(1, unrollFactor)
        let gateAccumulate = (0..<effectiveUnroll).map { lane -> String in
            if lane == 0 {
                return "gateSum += \(readWeight("gateRow[0]")) * \(stagedInputRead)(inputLane[0]);"
            }
            let offset = "\(lane)"
            return "gateSum += \(readWeight("gateRow[\(offset)]")) * \(stagedInputRead)(inputLane[\(offset)]);"
        }.joined(separator: "\n")
        let upAccumulate = (0..<effectiveUnroll).map { lane -> String in
            if lane == 0 {
                return "upSum += \(readWeight("upRow[0]")) * \(stagedInputRead)(inputLane[0]);"
            }
            let offset = "\(lane)"
            return "upSum += \(readWeight("upRow[\(offset)]")) * \(stagedInputRead)(inputLane[\(offset)]);"
        }.joined(separator: "\n")

        return """
        kernel void \(name)(
            device const \(bt)* input          [[buffer(0)]],
            device const \(wt)* gateWeight     [[buffer(1)]],
            device const \(wt)* upWeight       [[buffer(2)]],
            device \(bt)* output               [[buffer(3)]],
            constant uint& inputDimension      [[buffer(4)]],
            constant uint& outputDimension     [[buffer(5)]],
            uint gid                           [[threadgroup_position_in_grid]],
            uint tid                           [[thread_index_in_threadgroup]],
            uint tiisg                         [[thread_index_in_simdgroup]],
            uint sgitg                         [[simdgroup_index_in_threadgroup]],
            uint threadsPerThreadgroup         [[threads_per_threadgroup]]
        ) {
            const uint fixedInputDimension = 2048u;
            const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup / SIMD_WIDTH);
            const uint row = gid * rowsPerThreadgroup + sgitg;
            if (row >= outputDimension) return;

            threadgroup \(stagedInputType) inputTile[fixedInputDimension];
            for (uint j = tid; j < fixedInputDimension; j += threadsPerThreadgroup) {
                inputTile[j] = \(stagesInputAsFloat ? "float(input[j])" : "input[j]");
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float gateSum = 0.0f;
            float upSum = 0.0f;
            device const \(wt)* gateRow = gateWeight + row * fixedInputDimension + tiisg * \(effectiveUnroll);
            device const \(wt)* upRow = upWeight + row * fixedInputDimension + tiisg * \(effectiveUnroll);
            threadgroup const \(stagedInputType)* inputLane = inputTile + tiisg * \(effectiveUnroll);
            for (uint j = tiisg * \(effectiveUnroll); j < fixedInputDimension; j += SIMD_WIDTH * \(effectiveUnroll)) {
                \(gateAccumulate)
                \(upAccumulate)
                gateRow += SIMD_WIDTH * \(effectiveUnroll);
                upRow += SIMD_WIDTH * \(effectiveUnroll);
                inputLane += SIMD_WIDTH * \(effectiveUnroll);
            }
            gateSum = simd_sum(gateSum);
            upSum = simd_sum(upSum);
            if (tiisg == 0) {
                float sig = 1.0f / (1.0f + fast::exp(-gateSum));
                output[row] = \(bt)(gateSum * sig * upSum);
            }
        }
        """
    }

    public static func generateInput2048FusedSwiGLUProjectionArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        stagesInputAsFloat: Bool = true,
        unrollFactor: Int = 4
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let stagedInputType = stagesInputAsFloat ? "float" : bt
        let stagedInputRead = stagesInputAsFloat ? "" : "float"
        let stagedInputStore = stagesInputAsFloat ? "inputTile[j] = float(args.input[j]);" : "inputTile[j] = args.input[j];"
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let effectiveUnroll = max(1, unrollFactor)
        let inputStructName = "\(name)_args"
        let gateAccumulate = (0..<effectiveUnroll).map { lane -> String in
            if lane == 0 {
                return "gateSum += \(readWeight("gateRow[0]")) * \(stagedInputRead)(inputLane[0]);"
            }
            return "gateSum += \(readWeight("gateRow[\(lane)]")) * \(stagedInputRead)(inputLane[\(lane)]);"
        }.joined(separator: "\n")
        let upAccumulate = (0..<effectiveUnroll).map { lane -> String in
            if lane == 0 {
                return "upSum += \(readWeight("upRow[0]")) * \(stagedInputRead)(inputLane[0]);"
            }
            return "upSum += \(readWeight("upRow[\(lane)]")) * \(stagedInputRead)(inputLane[\(lane)]);"
        }.joined(separator: "\n")

        return """
        struct \(inputStructName) {
            device const \(bt)* input [[id(0)]];
            device const \(wt)* gateWeight [[id(1)]];
            device const \(wt)* upWeight [[id(2)]];
            device \(bt)* output [[id(3)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& inputDimension             [[buffer(4)]],
            constant uint& outputDimension            [[buffer(5)]],
            uint gid                                  [[threadgroup_position_in_grid]],
            uint tid                                  [[thread_index_in_threadgroup]],
            uint tiisg                                [[thread_index_in_simdgroup]],
            uint sgitg                                [[simdgroup_index_in_threadgroup]],
            uint threadsPerThreadgroup                [[threads_per_threadgroup]]
        ) {
            const uint fixedInputDimension = 2048u;
            const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup / SIMD_WIDTH);
            const uint row = gid * rowsPerThreadgroup + sgitg;
            if (row >= outputDimension) return;

            threadgroup \(stagedInputType) inputTile[fixedInputDimension];
            for (uint j = tid; j < fixedInputDimension; j += threadsPerThreadgroup) {
                \(stagedInputStore)
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float gateSum = 0.0f;
            float upSum = 0.0f;
            device const \(wt)* gateRow = args.gateWeight + row * fixedInputDimension + tiisg * \(effectiveUnroll);
            device const \(wt)* upRow = args.upWeight + row * fixedInputDimension + tiisg * \(effectiveUnroll);
            threadgroup const \(stagedInputType)* inputLane = inputTile + tiisg * \(effectiveUnroll);
            for (uint j = tiisg * \(effectiveUnroll); j < fixedInputDimension; j += SIMD_WIDTH * \(effectiveUnroll)) {
                \(gateAccumulate)
                \(upAccumulate)
                gateRow += SIMD_WIDTH * \(effectiveUnroll);
                upRow += SIMD_WIDTH * \(effectiveUnroll);
                inputLane += SIMD_WIDTH * \(effectiveUnroll);
            }
            gateSum = simd_sum(gateSum);
            upSum = simd_sum(upSum);
            if (tiisg == 0) {
                float sig = 1.0f / (1.0f + fast::exp(-gateSum));
                args.output[row] = \(bt)(gateSum * sig * upSum);
            }
        }
        """
    }

    /// Generate batched GEMV kernel for 2 projections sharing the same input.
    public static func generateBatchedGEMV2(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        return """
        kernel void \(name)(
            device const \(bt)* input          [[buffer(0)]],
            device const \(wt)* weight0        [[buffer(1)]],
            device const \(wt)* weight1        [[buffer(2)]],
            device \(bt)* output0              [[buffer(3)]],
            device \(bt)* output1              [[buffer(4)]],
            constant uint& inputDimension      [[buffer(5)]],
            constant uint& outputDim0          [[buffer(6)]],
            constant uint& outputDim1          [[buffer(7)]],
            uint gid                           [[threadgroup_position_in_grid]],
            uint tid                           [[thread_index_in_threadgroup]],
            uint tiisg                         [[thread_index_in_simdgroup]],
            uint sgitg                         [[simdgroup_index_in_threadgroup]],
            uint threadsPerThreadgroup         [[threads_per_threadgroup]]
        ) {
            const uint tileElements = 256;
            const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup / SIMD_WIDTH);
            const uint globalRow = gid * rowsPerThreadgroup + sgitg;
            const uint totalRows = outputDim0 + outputDim1;
            if (globalRow >= totalRows) return;

            const bool isSecond = (globalRow >= outputDim0);
            const uint localRow = isSecond ? (globalRow - outputDim0) : globalRow;
            device const \(wt)* weight = isSecond ? weight1 : weight0;
            device \(bt)* output = isSecond ? output1 : output0;

            threadgroup \(bt) inputTile[tileElements];
            float sum = 0.0f;
            device const \(wt)* weightRow = weight + localRow * inputDimension;
            for (uint base = 0; base < inputDimension; base += tileElements) {
                for (uint j = tid; j < tileElements; j += threadsPerThreadgroup) {
                    const uint inputIndex = base + j;
                    inputTile[j] = inputIndex < inputDimension ? input[inputIndex] : \(bt)(0.0f);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint tileCount = min(tileElements, inputDimension - base);
                for (uint j = tiisg; j < tileCount; j += SIMD_WIDTH) {
                    sum += \(readWeight("weightRow[base + j]")) * float(inputTile[j]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            sum = simd_sum(sum);
            if (tiisg == 0) {
                output[localRow] = \(bt)(sum);
            }
        }
        """
    }

    public static func generateBatchedGEMV2ArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let inputStructName = "\(name)_args"

        return """
        struct \(inputStructName) {
            device const \(bt)* input [[id(0)]];
            device const \(wt)* weight0 [[id(1)]];
            device const \(wt)* weight1 [[id(2)]];
            device \(bt)* output0 [[id(3)]];
            device \(bt)* output1 [[id(4)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args        [[buffer(\(argumentBufferIndex))]],
            constant uint& inputDimension            [[buffer(5)]],
            constant uint& outputDim0               [[buffer(6)]],
            constant uint& outputDim1               [[buffer(7)]],
            uint gid                                 [[threadgroup_position_in_grid]],
            uint tid                                 [[thread_index_in_threadgroup]],
            uint tiisg                               [[thread_index_in_simdgroup]],
            uint sgitg                               [[simdgroup_index_in_threadgroup]],
            uint threadsPerThreadgroup               [[threads_per_threadgroup]]
        ) {
            const uint tileElements = 256;
            const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup / SIMD_WIDTH);
            const uint globalRow = gid * rowsPerThreadgroup + sgitg;
            const uint totalRows = outputDim0 + outputDim1;
            if (globalRow >= totalRows) return;

            const bool isSecond = (globalRow >= outputDim0);
            const uint localRow = isSecond ? (globalRow - outputDim0) : globalRow;
            device const \(wt)* weight = isSecond ? args.weight1 : args.weight0;
            device \(bt)* output = isSecond ? args.output1 : args.output0;

            threadgroup \(bt) inputTile[tileElements];
            float sum = 0.0f;
            device const \(wt)* weightRow = weight + localRow * inputDimension;
            for (uint base = 0; base < inputDimension; base += tileElements) {
                for (uint j = tid; j < tileElements; j += threadsPerThreadgroup) {
                    const uint inputIndex = base + j;
                    inputTile[j] = inputIndex < inputDimension ? args.input[inputIndex] : \(bt)(0.0f);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint tileCount = min(tileElements, inputDimension - base);
                for (uint j = tiisg; j < tileCount; j += SIMD_WIDTH) {
                    sum += \(readWeight("weightRow[base + j]")) * float(inputTile[j]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            sum = simd_sum(sum);
            if (tiisg == 0) {
                output[localRow] = \(bt)(sum);
            }
        }
        """
    }

    /// Generate batched GEMV kernel for 3 projections sharing the same input.
    public static func generateBatchedGEMV3(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        return """
        kernel void \(name)(
            device const \(bt)* input          [[buffer(0)]],
            device const \(wt)* weight0        [[buffer(1)]],
            device const \(wt)* weight1        [[buffer(2)]],
            device const \(wt)* weight2        [[buffer(3)]],
            device \(bt)* output0              [[buffer(4)]],
            device \(bt)* output1              [[buffer(5)]],
            device \(bt)* output2              [[buffer(6)]],
            constant uint& inputDimension      [[buffer(7)]],
            constant uint& outputDim0          [[buffer(8)]],
            constant uint& outputDim1          [[buffer(9)]],
            constant uint& outputDim2          [[buffer(10)]],
            uint gid                           [[threadgroup_position_in_grid]],
            uint tid                           [[thread_index_in_threadgroup]],
            uint tiisg                         [[thread_index_in_simdgroup]],
            uint sgitg                         [[simdgroup_index_in_threadgroup]],
            uint threadsPerThreadgroup         [[threads_per_threadgroup]]
        ) {
            const uint tileElements = 256;
            const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup / SIMD_WIDTH);
            const uint globalRow = gid * rowsPerThreadgroup + sgitg;
            const uint totalRows = outputDim0 + outputDim1 + outputDim2;
            if (globalRow >= totalRows) return;

            device const \(wt)* weight;
            device \(bt)* output;
            uint localRow;
            if (globalRow < outputDim0) {
                weight = weight0; output = output0; localRow = globalRow;
            } else if (globalRow < outputDim0 + outputDim1) {
                weight = weight1; output = output1; localRow = globalRow - outputDim0;
            } else {
                weight = weight2; output = output2; localRow = globalRow - outputDim0 - outputDim1;
            }

            threadgroup \(bt) inputTile[tileElements];
            float sum = 0.0f;
            device const \(wt)* weightRow = weight + localRow * inputDimension;
            for (uint base = 0; base < inputDimension; base += tileElements) {
                for (uint j = tid; j < tileElements; j += threadsPerThreadgroup) {
                    const uint inputIndex = base + j;
                    inputTile[j] = inputIndex < inputDimension ? input[inputIndex] : \(bt)(0.0f);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint tileCount = min(tileElements, inputDimension - base);
                for (uint j = tiisg; j < tileCount; j += SIMD_WIDTH) {
                    sum += \(readWeight("weightRow[base + j]")) * float(inputTile[j]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            sum = simd_sum(sum);
            if (tiisg == 0) {
                output[localRow] = \(bt)(sum);
            }
        }
        """
    }

    public static func generateBatchedGEMV3ArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let inputStructName = "\(name)_args"

        return """
        struct \(inputStructName) {
            device const \(bt)* input [[id(0)]];
            device const \(wt)* weight0 [[id(1)]];
            device const \(wt)* weight1 [[id(2)]];
            device const \(wt)* weight2 [[id(3)]];
            device \(bt)* output0 [[id(4)]];
            device \(bt)* output1 [[id(5)]];
            device \(bt)* output2 [[id(6)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args        [[buffer(\(argumentBufferIndex))]],
            constant uint& inputDimension            [[buffer(7)]],
            constant uint& outputDim0               [[buffer(8)]],
            constant uint& outputDim1               [[buffer(9)]],
            constant uint& outputDim2               [[buffer(10)]],
            uint gid                                 [[threadgroup_position_in_grid]],
            uint tid                                 [[thread_index_in_threadgroup]],
            uint tiisg                               [[thread_index_in_simdgroup]],
            uint sgitg                               [[simdgroup_index_in_threadgroup]],
            uint threadsPerThreadgroup               [[threads_per_threadgroup]]
        ) {
            const uint tileElements = 256;
            const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup / SIMD_WIDTH);
            const uint globalRow = gid * rowsPerThreadgroup + sgitg;
            const uint totalRows = outputDim0 + outputDim1 + outputDim2;
            if (globalRow >= totalRows) return;

            device const \(wt)* weight;
            device \(bt)* output;
            uint localRow;
            if (globalRow < outputDim0) {
                weight = args.weight0; output = args.output0; localRow = globalRow;
            } else if (globalRow < outputDim0 + outputDim1) {
                weight = args.weight1; output = args.output1; localRow = globalRow - outputDim0;
            } else {
                weight = args.weight2; output = args.output2; localRow = globalRow - outputDim0 - outputDim1;
            }

            threadgroup \(bt) inputTile[tileElements];
            float sum = 0.0f;
            device const \(wt)* weightRow = weight + localRow * inputDimension;
            for (uint base = 0; base < inputDimension; base += tileElements) {
                for (uint j = tid; j < tileElements; j += threadsPerThreadgroup) {
                    const uint inputIndex = base + j;
                    inputTile[j] = inputIndex < inputDimension ? args.input[inputIndex] : \(bt)(0.0f);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint tileCount = min(tileElements, inputDimension - base);
                for (uint j = tiisg; j < tileCount; j += SIMD_WIDTH) {
                    sum += \(readWeight("weightRow[base + j]")) * float(inputTile[j]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            sum = simd_sum(sum);
            if (tiisg == 0) {
                output[localRow] = \(bt)(sum);
            }
        }
        """
    }

    // MARK: - Batched Per-Head Fragment

    /// Generate batched per-head kernel for 2 independent in-place operations.
    /// Routes threadgroups to the correct data/weight buffers based on head index.
    public static func generateBatchedPerHead2(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        return """
        kernel void \(name)(
            device \(bt)* data0            [[buffer(0)]],
            device \(bt)* data1            [[buffer(1)]],
            device const \(wt)* weight0    [[buffer(2)]],
            device const \(wt)* weight1    [[buffer(3)]],
            constant uint& count0          [[buffer(4)]],
            constant uint& count1          [[buffer(5)]],
            constant uint& headDim         [[buffer(6)]],
            constant float& epsilon        [[buffer(7)]],
            uint headIndex                 [[threadgroup_position_in_grid]],
            uint tid                       [[thread_index_in_threadgroup]],
            uint threadgroupSize           [[threads_per_threadgroup]]
        ) {
            // Route: threadgroups [0..count0) → data0/weight0, [count0..count0+count1) → data1/weight1
            device \(bt)* data;
            device const \(wt)* weight;
            uint localHead;
            if (headIndex < count0) {
                data = data0; weight = weight0; localHead = headIndex;
            } else {
                data = data1; weight = weight1; localHead = headIndex - count0;
                if (localHead >= count1) return;
            }
            uint offset = localHead * headDim;

            // Per-head RMS norm (in-place)
            float sumSq = 0.0f;
            for (uint i = tid; i < headDim; i += threadgroupSize) {
                float v = float(data[offset + i]);
                sumSq += v * v;
            }
            sumSq = simd_sum(sumSq);

            threadgroup float shared[32];
            if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = sumSq;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0) {
                float total = 0.0f;
                uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                for (uint i = 0; i < sgCount; i++) total += shared[i];
                shared[0] = rsqrt(total / float(headDim) + epsilon);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scale = shared[0];
            for (uint i = tid; i < headDim; i += threadgroupSize) {
                data[offset + i] = \(bt)(float(data[offset + i]) * scale * \(readWeight("weight[i]")));
            }
        }
        """
    }

    public static func generateBatchedPerHead2ArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let inputStructName = "\(name)_args"

        return """
        struct \(inputStructName) {
            device \(bt)* data0 [[id(0)]];
            device \(bt)* data1 [[id(1)]];
            device const \(wt)* weight0 [[id(2)]];
            device const \(wt)* weight1 [[id(3)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args        [[buffer(\(argumentBufferIndex))]],
            constant uint& count0                    [[buffer(4)]],
            constant uint& count1                    [[buffer(5)]],
            constant uint& headDim                   [[buffer(6)]],
            constant float& epsilon                  [[buffer(7)]],
            uint headIndex                           [[threadgroup_position_in_grid]],
            uint tid                                 [[thread_index_in_threadgroup]],
            uint threadgroupSize                     [[threads_per_threadgroup]]
        ) {
            device \(bt)* data;
            device const \(wt)* weight;
            uint localHead;
            if (headIndex < count0) {
                data = args.data0; weight = args.weight0; localHead = headIndex;
            } else {
                data = args.data1; weight = args.weight1; localHead = headIndex - count0;
                if (localHead >= count1) return;
            }
            uint offset = localHead * headDim;

            float sumSq = 0.0f;
            for (uint i = tid; i < headDim; i += threadgroupSize) {
                float v = float(data[offset + i]);
                sumSq += v * v;
            }
            sumSq = simd_sum(sumSq);

            threadgroup float shared[32];
            if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = sumSq;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0) {
                float total = 0.0f;
                uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                for (uint i = 0; i < sgCount; i++) total += shared[i];
                shared[0] = rsqrt(total / float(headDim) + epsilon);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scale = shared[0];
            for (uint i = tid; i < headDim; i += threadgroupSize) {
                data[offset + i] = \(bt)(float(data[offset + i]) * scale * \(readWeight("weight[i]")));
            }
        }
        """
    }

    // MARK: - QK Norm

    /// Generate QK RMSNorm (per-head normalization for Q/K projections).
    public static func generateQKNorm(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        return """
        kernel void \(name)(
            device \(bt)* data               [[buffer(0)]],
            device const \(wt)* weight       [[buffer(1)]],
            constant uint& headCount         [[buffer(2)]],
            constant uint& headDim           [[buffer(3)]],
            constant float& epsilon          [[buffer(4)]],
            uint headIndex                   [[threadgroup_position_in_grid]],
            uint tid                         [[thread_index_in_threadgroup]],
            uint threadgroupSize             [[threads_per_threadgroup]]
        ) {
            if (headIndex >= headCount) return;
            uint offset = headIndex * headDim;

            float sumSq = 0.0f;
            for (uint i = tid; i < headDim; i += threadgroupSize) {
                float v = float(data[offset + i]);
                sumSq += v * v;
            }
            sumSq = simd_sum(sumSq);

            threadgroup float shared[32];
            if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = sumSq;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0) {
                float total = 0.0f;
                uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                for (uint i = 0; i < sgCount; i++) total += shared[i];
                shared[0] = rsqrt(total / float(headDim) + epsilon);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scale = shared[0];
            for (uint i = tid; i < headDim; i += threadgroupSize) {
                data[offset + i] = \(bt)(float(data[offset + i]) * scale * \(readWeight("weight[i]")));
            }
        }
        """
    }

    public static func generateQKNormArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let inputStructName = "\(name)_args"

        return """
        struct \(inputStructName) {
            device \(bt)* data [[id(0)]];
            device const \(wt)* weight [[id(1)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& headCount                  [[buffer(2)]],
            constant uint& headDim                    [[buffer(3)]],
            constant float& epsilon                   [[buffer(4)]],
            uint headIndex                            [[threadgroup_position_in_grid]],
            uint tid                                  [[thread_index_in_threadgroup]],
            uint threadgroupSize                      [[threads_per_threadgroup]]
        ) {
            if (headIndex >= headCount) return;
            uint offset = headIndex * headDim;

            float sumSq = 0.0f;
            for (uint i = tid; i < headDim; i += threadgroupSize) {
                float v = float(args.data[offset + i]);
                sumSq += v * v;
            }
            sumSq = simd_sum(sumSq);

            threadgroup float shared[32];
            if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = sumSq;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0) {
                float total = 0.0f;
                uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                for (uint i = 0; i < sgCount; i++) total += shared[i];
                shared[0] = rsqrt(total / float(headDim) + epsilon);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scale = shared[0];
            for (uint i = tid; i < headDim; i += threadgroupSize) {
                args.data[offset + i] = \(bt)(float(args.data[offset + i]) * scale * \(readWeight("args.weight[i]")));
            }
        }
        """
    }

    /// Generate sequence-aware QK RMSNorm (prefill).
    public static func generateQKNormSeq(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        return """
        kernel void \(name)(
            device \(bt)* data               [[buffer(0)]],
            device const \(wt)* weight       [[buffer(1)]],
            constant uint& headCount         [[buffer(2)]],
            constant uint& headDimension     [[buffer(3)]],
            constant float& epsilon          [[buffer(4)]],
            constant uint& sequenceLength    [[buffer(5)]],
            constant uint& totalDimension    [[buffer(6)]],
            uint2 gid                        [[threadgroup_position_in_grid]],
            uint tid                         [[thread_index_in_threadgroup]]
        ) {
            uint head = gid.x;
            uint seqPos = gid.y;
            if (head >= headCount || seqPos >= sequenceLength) return;

            uint offset = seqPos * totalDimension + head * headDimension;

            float sumSq = 0.0f;
            for (uint i = tid; i < headDimension; i += SIMD_WIDTH) {
                float v = float(data[offset + i]);
                sumSq += v * v;
            }
            sumSq = simd_sum(sumSq);

            threadgroup float sharedRMS[1];
            if (tid == 0) {
                sharedRMS[0] = rsqrt(sumSq / float(headDimension) + epsilon);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float rms = sharedRMS[0];
            for (uint i = tid; i < headDimension; i += SIMD_WIDTH) {
                data[offset + i] = \(bt)(float(data[offset + i]) * rms * \(readWeight("weight[i]")));
            }
        }
        """
    }

    // MARK: - RoPE

    /// Generate RoPE kernel (decode: single token, position from buffer).
    public static func generateRoPE(
        name: String,
        bufferPrecision: BufferPrecision
    ) -> String {
        let bt = bufferPrecision.metalType

        return """
        kernel void \(name)(
            device \(bt)* query                  [[buffer(0)]],
            device \(bt)* key                    [[buffer(1)]],
            device const uint* positionBuffer    [[buffer(2)]],
            constant uint& headCount             [[buffer(3)]],
            constant uint& kvHeadCount           [[buffer(4)]],
            constant uint& headDimension         [[buffer(5)]],
            constant uint& ropeDimension         [[buffer(6)]],
            constant float& ropeBase             [[buffer(7)]],
            uint headIndex                       [[threadgroup_position_in_grid]],
            uint tid                             [[thread_index_in_threadgroup]]
        ) {
            const uint halfRopeDim = ropeDimension / 2;
            if (tid >= halfRopeDim) return;

            const uint position = positionBuffer[0];
            const float theta = float(position) * pow(ropeBase, -2.0f * float(tid) / float(ropeDimension));
            const float cosTheta = cos(theta);
            const float sinTheta = sin(theta);

            if (headIndex < headCount) {
                uint qOffset = headIndex * headDimension + tid;
                float q0 = float(query[qOffset]);
                float q1 = float(query[qOffset + halfRopeDim]);
                query[qOffset] = \(bt)(q0 * cosTheta - q1 * sinTheta);
                query[qOffset + halfRopeDim] = \(bt)(q0 * sinTheta + q1 * cosTheta);
            }

            if (headIndex < kvHeadCount) {
                uint kOffset = headIndex * headDimension + tid;
                float k0 = float(key[kOffset]);
                float k1 = float(key[kOffset + halfRopeDim]);
                key[kOffset] = \(bt)(k0 * cosTheta - k1 * sinTheta);
                key[kOffset + halfRopeDim] = \(bt)(k0 * sinTheta + k1 * cosTheta);
            }
        }
        """
    }

    public static func generateRoPEArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision
    ) -> String {
        let bt = bufferPrecision.metalType
        let inputStructName = "\(name)_args"

        return """
        struct \(inputStructName) {
            device \(bt)* query [[id(0)]];
            device \(bt)* key [[id(1)]];
            device const uint* positionBuffer [[id(2)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& headCount                  [[buffer(3)]],
            constant uint& kvHeadCount                [[buffer(4)]],
            constant uint& headDimension              [[buffer(5)]],
            constant uint& ropeDimension              [[buffer(6)]],
            constant float& ropeBase                  [[buffer(7)]],
            uint headIndex                            [[threadgroup_position_in_grid]],
            uint tid                                  [[thread_index_in_threadgroup]]
        ) {
            const uint halfRopeDim = ropeDimension / 2;
            if (tid >= halfRopeDim) return;

            const uint position = args.positionBuffer[0];
            const float theta = float(position) * pow(ropeBase, -2.0f * float(tid) / float(ropeDimension));
            const float cosTheta = cos(theta);
            const float sinTheta = sin(theta);

            if (headIndex < headCount) {
                uint qOffset = headIndex * headDimension + tid;
                float q0 = float(args.query[qOffset]);
                float q1 = float(args.query[qOffset + halfRopeDim]);
                args.query[qOffset] = \(bt)(q0 * cosTheta - q1 * sinTheta);
                args.query[qOffset + halfRopeDim] = \(bt)(q0 * sinTheta + q1 * cosTheta);
            }

            if (headIndex < kvHeadCount) {
                uint kOffset = headIndex * headDimension + tid;
                float k0 = float(args.key[kOffset]);
                float k1 = float(args.key[kOffset + halfRopeDim]);
                args.key[kOffset] = \(bt)(k0 * cosTheta - k1 * sinTheta);
                args.key[kOffset + halfRopeDim] = \(bt)(k0 * sinTheta + k1 * cosTheta);
            }
        }
        """
    }

    /// Generate RoPE kernel (prefill: sequence-aware).
    public static func generateRoPESeq(
        name: String,
        bufferPrecision: BufferPrecision
    ) -> String {
        let bt = bufferPrecision.metalType

        return """
        kernel void \(name)(
            device \(bt)* Q              [[buffer(0)]],
            device \(bt)* K              [[buffer(1)]],
            device const uint* positions [[buffer(2)]],
            constant uint& headCount     [[buffer(3)]],
            constant uint& kvHeadCount   [[buffer(4)]],
            constant uint& headDimension [[buffer(5)]],
            constant uint& ropeDimension [[buffer(6)]],
            constant float& base         [[buffer(7)]],
            constant uint& sequenceLength [[buffer(8)]],
            uint2 gid                    [[threadgroup_position_in_grid]],
            uint tid                     [[thread_index_in_threadgroup]]
        ) {
            uint head = gid.x;
            uint seqPos = gid.y;
            if (seqPos >= sequenceLength) return;
            uint position = positions[seqPos];
            uint totalHeads = headCount + kvHeadCount;
            if (head >= totalHeads) return;
            uint qkvDimension = (head < headCount) ? headCount * headDimension : kvHeadCount * headDimension;
            device \(bt)* data = (head < headCount) ? Q : K;
            uint localHead = (head < headCount) ? head : (head - headCount);
            uint offset = seqPos * qkvDimension + localHead * headDimension;
            uint halfRope = ropeDimension / 2;
            for (uint i = tid; i < halfRope; i += SIMD_WIDTH) {
                float theta = float(position) / pow(base, float(2 * i) / float(ropeDimension));
                float cosTheta = cos(theta);
                float sinTheta = sin(theta);
                float x0 = float(data[offset + i]);
                float x1 = float(data[offset + i + halfRope]);
                data[offset + i] = \(bt)(x0 * cosTheta - x1 * sinTheta);
                data[offset + i + halfRope] = \(bt)(x1 * cosTheta + x0 * sinTheta);
            }
        }
        """
    }

    // MARK: - Conv1d

    /// Generate conv_state_update kernel (decode: single token with persistent state).
    public static func generateConvStateUpdate(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        return """
        kernel void \(name)(
            device half* convState           [[buffer(0)]],
            device const \(bt)* inProjOutput [[buffer(1)]],
            device const \(wt)* weight       [[buffer(2)]],
            device \(bt)* output             [[buffer(3)]],
            constant uint& dimension         [[buffer(4)]],
            constant uint& kernelSize        [[buffer(5)]],
            uint gid                         [[thread_position_in_grid]]
        ) {
            if (gid >= dimension) return;

            float B = float(inProjOutput[gid]);
            float C = float(inProjOutput[dimension + gid]);
            float x = float(inProjOutput[2 * dimension + gid]);
            float Bx = B * x;

            for (uint k = 0; k < kernelSize - 1; k++) {
                convState[k * dimension + gid] = convState[(k + 1) * dimension + gid];
            }
            convState[(kernelSize - 1) * dimension + gid] = half(Bx);

            float convOut = 0.0f;
            for (uint k = 0; k + 1 < kernelSize; k++) {
                convOut += float(convState[k * dimension + gid]) * \(readWeight("weight[gid * kernelSize + k]"));
            }
            convOut += Bx * \(readWeight("weight[gid * kernelSize + (kernelSize - 1)]"));

            output[gid] = \(bt)(C * convOut);
        }
        """
    }

    public static func generateConvStateUpdateArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let inputStructName = "\(name)_args"
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        return """
        struct \(inputStructName) {
            device half* convState [[id(0)]];
            device const \(bt)* inProjOutput [[id(1)]];
            device const \(wt)* weight [[id(2)]];
            device \(bt)* output [[id(3)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& dimension                  [[buffer(4)]],
            constant uint& kernelSize                 [[buffer(5)]],
            uint gid                                  [[thread_position_in_grid]]
        ) {
            if (gid >= dimension) return;

            float B = float(args.inProjOutput[gid]);
            float C = float(args.inProjOutput[dimension + gid]);
            float x = float(args.inProjOutput[2 * dimension + gid]);
            float Bx = B * x;

            for (uint k = 0; k < kernelSize - 1; k++) {
                args.convState[k * dimension + gid] = args.convState[(k + 1) * dimension + gid];
            }
            args.convState[(kernelSize - 1) * dimension + gid] = half(Bx);

            float convOut = 0.0f;
            for (uint k = 0; k + 1 < kernelSize; k++) {
                convOut += float(args.convState[k * dimension + gid]) * \(readWeight("args.weight[gid * kernelSize + k]"));
            }
            convOut += Bx * \(readWeight("args.weight[gid * kernelSize + (kernelSize - 1)]"));

            args.output[gid] = \(bt)(C * convOut);
        }
        """
    }

    /// Generate conv1d_causal_seq kernel (prefill: temporal conv across positions).
    public static func generateConv1dCausalSeq(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        return """
        kernel void \(name)(
            device const \(bt)* input      [[buffer(0)]],
            device const \(wt)* weight     [[buffer(1)]],
            device \(bt)* output           [[buffer(2)]],
            constant uint& convDim         [[buffer(3)]],
            constant uint& inProjDim       [[buffer(4)]],
            constant uint& kernelSize      [[buffer(5)]],
            constant uint& sequenceLength  [[buffer(6)]],
            uint2 gid                      [[thread_position_in_grid]]
        ) {
            uint ch = gid.x;
            uint pos = gid.y;
            if (ch >= convDim || pos >= sequenceLength) return;

            float convOut = 0.0f;
            for (uint k = 0; k < kernelSize; k++) {
                int srcPos = int(pos) - int(kernelSize - 1) + int(k);
                if (srcPos >= 0) {
                    float B = float(input[uint(srcPos) * inProjDim + ch]);
                    float x = float(input[uint(srcPos) * inProjDim + 2 * convDim + ch]);
                    convOut += B * x * \(readWeight("weight[ch * kernelSize + k]"));
                }
            }

            float C = float(input[pos * inProjDim + convDim + ch]);
            output[pos * convDim + ch] = \(bt)(C * convOut);
        }
        """
    }

    /// Generate extract_conv_state kernel (saves last kernelSize positions' B*x to conv_state).
    public static func generateExtractConvState(
        name: String,
        bufferPrecision: BufferPrecision
    ) -> String {
        let bt = bufferPrecision.metalType

        return """
        kernel void \(name)(
            device const \(bt)* inProjOutput   [[buffer(0)]],
            device half* convState             [[buffer(1)]],
            constant uint& convDim             [[buffer(2)]],
            constant uint& inProjDim           [[buffer(3)]],
            constant uint& kernelSize          [[buffer(4)]],
            constant uint& sequenceLength      [[buffer(5)]],
            uint2 gid                          [[thread_position_in_grid]]
        ) {
            uint ch = gid.x;
            uint k = gid.y;
            if (ch >= convDim || k >= kernelSize) return;
            int srcPos = int(sequenceLength) - int(kernelSize) + int(k);
            if (srcPos >= 0 && uint(srcPos) < sequenceLength) {
                float B = float(inProjOutput[uint(srcPos) * inProjDim + ch]);
                float x = float(inProjOutput[uint(srcPos) * inProjDim + 2 * convDim + ch]);
                convState[k * convDim + ch] = half(B * x);
            } else {
                convState[k * convDim + ch] = 0;
            }
        }
        """
    }

    // MARK: - Flash Attention

    /// Generate flash attention decode kernel with parameterized Q/K/V/output buffer type.
    /// KV cache buffer type is always uchar* (handles both FP16 and Q8).
    /// Generate flash attention kernel function (without helper functions).
    /// Call `flashAttentionHelperSource` once, then this for each precision variant.
    public static func generateFlashAttentionKernel(
        name: String,
        bufferPrecision: BufferPrecision
    ) -> String {
        let bt = bufferPrecision.metalType
        let castIn: (String) -> String = { expr in
            bufferPrecision == .float32 ? "(\(expr))" : "float(\(expr))"
        }
        let castOut: (String) -> String = { expr in
            bufferPrecision == .float32 ? "(\(expr))" : "\(bt)(\(expr))"
        }

        return """
        kernel void \(name)(
            device const \(bt)* query             [[buffer(0)]],
            device const \(bt)* newKey            [[buffer(1)]],
            device const \(bt)* newValue          [[buffer(2)]],
            device uchar* keyCache               [[buffer(3)]],
            device uchar* valueCache             [[buffer(4)]],
            device \(bt)* output                  [[buffer(5)]],
            device const uint* positionBuffer    [[buffer(6)]],
            constant uint& headCount             [[buffer(7)]],
            constant uint& kvHeadCount           [[buffer(8)]],
            constant uint& headDimension         [[buffer(9)]],
            constant float& scale                [[buffer(10)]],
            constant uint& layoutMode            [[buffer(11)]],
            constant uint& maxSequenceLength     [[buffer(12)]],
            constant uint& kQuantScheme          [[buffer(13)]],
            constant uint& vQuantScheme          [[buffer(14)]],
            constant uint& kHeadSlotBytes        [[buffer(15)]],
            constant uint& vHeadSlotBytes        [[buffer(16)]],
            uint headIndex                       [[threadgroup_position_in_grid]],
            uint tid                             [[thread_index_in_threadgroup]],
            uint tiisg                           [[thread_index_in_simdgroup]],
            uint sgitg                           [[simdgroup_index_in_threadgroup]],
            uint threadgroupSize                 [[threads_per_threadgroup]]
        ) {
            const uint position = positionBuffer[0];
            const uint sequenceLength = position + 1;
            const uint headDim = headDimension;
            const uint kvHeadIndex = headIndex * kvHeadCount / headCount;
            const uint kvIn = kvHeadIndex * headDim;
            const uint canonicalWriterHead = kvHeadIndex * headCount / kvHeadCount;
            const bool writesCurrentKV = (headIndex == canonicalWriterHead);

            // --- Step 1: Append new K/V to cache ---
            // Only one query head per GQA group writes the current token's KV.
            // Other heads consume newKey/newValue directly for t == position.
            if (writesCurrentKV) {
                uint kWriteByteOffset;
                if (layoutMode == 0) {
                    kWriteByteOffset = kvHeadIndex * maxSequenceLength * kHeadSlotBytes
                        + position * kHeadSlotBytes;
                } else {
                    kWriteByteOffset = position * kvHeadCount * kHeadSlotBytes
                        + kvHeadIndex * kHeadSlotBytes;
                }

                if (kQuantScheme == 0x00 || kQuantScheme == 0x01 || kQuantScheme == 0x02) {
                    for (uint d = tid; d < headDim; d += threadgroupSize) {
                        write_kv_element_dense(
                            keyCache + kWriteByteOffset, d, \(castIn("newKey[kvIn + d]")), kQuantScheme);
                    }
                } else {
                    const uint groupSize = 32;
                    const uint bytesPerBlock = 36;
                    const uint numGroups = (headDim + groupSize - 1) / groupSize;

                    for (uint g = 0; g < numGroups; g++) {
                        uint groupStart = g * groupSize;
                        float localMin = HUGE_VALF;
                        float localMax = -HUGE_VALF;
                        for (uint i = tid; i < groupSize && (groupStart + i) < headDim; i += threadgroupSize) {
                            float val = \(castIn("newKey[kvIn + groupStart + i]"));
                            localMin = min(localMin, val);
                            localMax = max(localMax, val);
                        }
                        localMin = simd_min(localMin);
                        localMax = simd_max(localMax);
                        threadgroup float sharedMin[32], sharedMax[32];
                        if (tiisg == 0) { sharedMin[sgitg] = localMin; sharedMax[sgitg] = localMax; }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        if (tid == 0) {
                            uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                            float gMin = sharedMin[0], gMax = sharedMax[0];
                            for (uint s = 1; s < sgCount; s++) {
                                gMin = min(gMin, sharedMin[s]);
                                gMax = max(gMax, sharedMax[s]);
                            }
                            sharedMin[0] = gMin; sharedMax[0] = gMax;
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        float groupMin = sharedMin[0];
                        float groupMax = sharedMax[0];
                        float groupScale = (groupMax - groupMin) / 255.0f;
                        if (groupScale < 1e-10f) groupScale = 1e-10f;

                        device uchar* blockOutput = keyCache + kWriteByteOffset + g * bytesPerBlock;
                        if (tid == 0) {
                            *(device half*)(blockOutput) = half(groupScale);
                            *(device half*)(blockOutput + 2) = half(groupMin);
                        }
                        threadgroup_barrier(mem_flags::mem_device);
                        for (uint i = tid; i < groupSize && (groupStart + i) < headDim; i += threadgroupSize) {
                            float val = \(castIn("newKey[kvIn + groupStart + i]"));
                            int quantized = int(round((val - groupMin) / groupScale));
                            quantized = clamp(quantized, 0, 255);
                            *(device char*)(blockOutput + 4 + i) = char(quantized);
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_device);

                uint vWriteByteOffset;
                if (layoutMode == 0) {
                    vWriteByteOffset = kvHeadIndex * maxSequenceLength * vHeadSlotBytes
                        + position * vHeadSlotBytes;
                } else {
                    vWriteByteOffset = position * kvHeadCount * vHeadSlotBytes
                        + kvHeadIndex * vHeadSlotBytes;
                }

                if (vQuantScheme == 0x00 || vQuantScheme == 0x01 || vQuantScheme == 0x02) {
                    for (uint d = tid; d < headDim; d += threadgroupSize) {
                        write_kv_element_dense(
                            valueCache + vWriteByteOffset, d, \(castIn("newValue[kvIn + d]")), vQuantScheme);
                    }
                } else {
                    const uint groupSize = 32;
                    const uint bytesPerBlock = 36;
                    const uint numGroups = (headDim + groupSize - 1) / groupSize;
                    for (uint g = 0; g < numGroups; g++) {
                        uint groupStart = g * groupSize;
                        float localMin = HUGE_VALF, localMax = -HUGE_VALF;
                        for (uint i = tid; i < groupSize && (groupStart + i) < headDim; i += threadgroupSize) {
                            float val = \(castIn("newValue[kvIn + groupStart + i]"));
                            localMin = min(localMin, val); localMax = max(localMax, val);
                        }
                        localMin = simd_min(localMin); localMax = simd_max(localMax);
                        threadgroup float sharedMin[32], sharedMax[32];
                        if (tiisg == 0) { sharedMin[sgitg] = localMin; sharedMax[sgitg] = localMax; }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        if (tid == 0) {
                            uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                            float gMin = sharedMin[0], gMax = sharedMax[0];
                            for (uint s = 1; s < sgCount; s++) { gMin = min(gMin, sharedMin[s]); gMax = max(gMax, sharedMax[s]); }
                            sharedMin[0] = gMin; sharedMax[0] = gMax;
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        float groupMin = sharedMin[0], groupMax = sharedMax[0];
                        float groupScale = (groupMax - groupMin) / 255.0f;
                        if (groupScale < 1e-10f) groupScale = 1e-10f;
                        device uchar* blockOutput = valueCache + vWriteByteOffset + g * bytesPerBlock;
                        if (tid == 0) {
                            *(device half*)(blockOutput) = half(groupScale);
                            *(device half*)(blockOutput + 2) = half(groupMin);
                        }
                        threadgroup_barrier(mem_flags::mem_device);
                        for (uint i = tid; i < groupSize && (groupStart + i) < headDim; i += threadgroupSize) {
                            float val = \(castIn("newValue[kvIn + groupStart + i]"));
                            int quantized = int(round((val - groupMin) / groupScale));
                            quantized = clamp(quantized, 0, 255);
                            *(device char*)(blockOutput + 4 + i) = char(quantized);
                        }
                    }
                }
            }

            // --- Step 2: Compute attention scores ---
            const uint queryOffset = headIndex * headDim;

            float maxScore = -HUGE_VALF;
            float sumExp = 0.0f;

            threadgroup float sharedOutput[4096]; // max headDim

            for (uint d = tid; d < headDim; d += threadgroupSize) {
                sharedOutput[d] = 0.0f;
            }

            for (uint t = 0; t < sequenceLength; t++) {
                uint kByteOffset;
                if (layoutMode == 0) {
                    kByteOffset = kvHeadIndex * maxSequenceLength * kHeadSlotBytes + t * kHeadSlotBytes;
                } else {
                    kByteOffset = t * kvHeadCount * kHeadSlotBytes + kvHeadIndex * kHeadSlotBytes;
                }

                float score = 0.0f;
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    float q = \(castIn("query[queryOffset + d]"));
                    float k;
                    if (t == position) {
                        k = \(castIn("newKey[kvIn + d]"));
                    } else {
                        k = read_kv_element(keyCache + kByteOffset, d, kQuantScheme, kHeadSlotBytes, headDim);
                    }
                    score += q * k;
                }
                score = simd_sum(score);
                threadgroup float sharedScore[32];
                if (tiisg == 0) sharedScore[sgitg] = score;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0) {
                    float total = 0.0f;
                    uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                    for (uint s = 0; s < sgCount; s++) total += sharedScore[s];
                    sharedScore[0] = total * scale;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                score = sharedScore[0];

                float oldMax = maxScore;
                maxScore = max(maxScore, score);
                float correction = exp(oldMax - maxScore);
                sumExp = sumExp * correction + exp(score - maxScore);

                uint vByteOffset;
                if (layoutMode == 0) {
                    vByteOffset = kvHeadIndex * maxSequenceLength * vHeadSlotBytes + t * vHeadSlotBytes;
                } else {
                    vByteOffset = t * kvHeadCount * vHeadSlotBytes + kvHeadIndex * vHeadSlotBytes;
                }

                float weight = exp(score - maxScore);
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    float v;
                    if (t == position) {
                        v = \(castIn("newValue[kvIn + d]"));
                    } else {
                        v = read_kv_element(valueCache + vByteOffset, d, vQuantScheme, vHeadSlotBytes, headDim);
                    }
                    sharedOutput[d] = sharedOutput[d] * correction + weight * v;
                }
            }

            // --- Step 3: Write output ---
            float invSum = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f;
            for (uint d = tid; d < headDim; d += threadgroupSize) {
                output[queryOffset + d] = \(castOut("sharedOutput[d] * invSum"));
            }
        }
        """
    }

    public static func generateFlashAttentionArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision
    ) -> String {
        let bt = bufferPrecision.metalType
        let inputStructName = "\(name)_args"
        let castIn: (String) -> String = { expr in
            bufferPrecision == .float32 ? "(\(expr))" : "float(\(expr))"
        }
        let castOut: (String) -> String = { expr in
            bufferPrecision == .float32 ? "(\(expr))" : "\(bt)(\(expr))"
        }

        return """
        struct \(inputStructName) {
            device const \(bt)* query [[id(0)]];
            device const \(bt)* newKey [[id(1)]];
            device const \(bt)* newValue [[id(2)]];
            device uchar* keyCache [[id(3)]];
            device uchar* valueCache [[id(4)]];
            device \(bt)* output [[id(5)]];
            device const uint* positionBuffer [[id(6)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& headCount                  [[buffer(7)]],
            constant uint& kvHeadCount                [[buffer(8)]],
            constant uint& headDimension              [[buffer(9)]],
            constant float& scale                     [[buffer(10)]],
            constant uint& layoutMode                 [[buffer(11)]],
            constant uint& maxSequenceLength          [[buffer(12)]],
            constant uint& kQuantScheme               [[buffer(13)]],
            constant uint& vQuantScheme               [[buffer(14)]],
            constant uint& kHeadSlotBytes             [[buffer(15)]],
            constant uint& vHeadSlotBytes             [[buffer(16)]],
            uint headIndex                            [[threadgroup_position_in_grid]],
            uint tid                                  [[thread_index_in_threadgroup]],
            uint tiisg                                [[thread_index_in_simdgroup]],
            uint sgitg                                [[simdgroup_index_in_threadgroup]],
            uint threadgroupSize                      [[threads_per_threadgroup]]
        ) {
            const uint position = args.positionBuffer[0];
            const uint sequenceLength = position + 1;
            const uint headDim = headDimension;
            const uint kvHeadIndex = headIndex * kvHeadCount / headCount;
            const uint kvIn = kvHeadIndex * headDim;
            const uint canonicalWriterHead = kvHeadIndex * headCount / kvHeadCount;
            const bool writesCurrentKV = (headIndex == canonicalWriterHead);

            if (writesCurrentKV) {
                uint kWriteByteOffset;
                if (layoutMode == 0) {
                    kWriteByteOffset = kvHeadIndex * maxSequenceLength * kHeadSlotBytes + position * kHeadSlotBytes;
                } else {
                    kWriteByteOffset = position * kvHeadCount * kHeadSlotBytes + kvHeadIndex * kHeadSlotBytes;
                }

                if (kQuantScheme == 0x00 || kQuantScheme == 0x01 || kQuantScheme == 0x02) {
                    for (uint d = tid; d < headDim; d += threadgroupSize) {
                        write_kv_element_dense(args.keyCache + kWriteByteOffset, d, \(castIn("args.newKey[kvIn + d]")), kQuantScheme);
                    }
                } else {
                    const uint groupSize = 32;
                    const uint bytesPerBlock = 36;
                    const uint numGroups = (headDim + groupSize - 1) / groupSize;

                    for (uint g = 0; g < numGroups; g++) {
                        uint groupStart = g * groupSize;
                        float localMin = HUGE_VALF;
                        float localMax = -HUGE_VALF;
                        for (uint i = tid; i < groupSize && (groupStart + i) < headDim; i += threadgroupSize) {
                            float val = \(castIn("args.newKey[kvIn + groupStart + i]"));
                            localMin = min(localMin, val);
                            localMax = max(localMax, val);
                        }
                        localMin = simd_min(localMin);
                        localMax = simd_max(localMax);
                        threadgroup float sharedMin[32], sharedMax[32];
                        if (tiisg == 0) { sharedMin[sgitg] = localMin; sharedMax[sgitg] = localMax; }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        if (tid == 0) {
                            uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                            float gMin = sharedMin[0], gMax = sharedMax[0];
                            for (uint s = 1; s < sgCount; s++) {
                                gMin = min(gMin, sharedMin[s]);
                                gMax = max(gMax, sharedMax[s]);
                            }
                            sharedMin[0] = gMin; sharedMax[0] = gMax;
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        float groupMin = sharedMin[0];
                        float groupMax = sharedMax[0];
                        float groupScale = (groupMax - groupMin) / 255.0f;
                        if (groupScale < 1e-10f) groupScale = 1e-10f;

                        device uchar* blockOutput = args.keyCache + kWriteByteOffset + g * bytesPerBlock;
                        if (tid == 0) {
                            *(device half*)(blockOutput) = half(groupScale);
                            *(device half*)(blockOutput + 2) = half(groupMin);
                        }
                        threadgroup_barrier(mem_flags::mem_device);
                        for (uint i = tid; i < groupSize && (groupStart + i) < headDim; i += threadgroupSize) {
                            float val = \(castIn("args.newKey[kvIn + groupStart + i]"));
                            int quantized = int(round((val - groupMin) / groupScale));
                            quantized = clamp(quantized, 0, 255);
                            *(device char*)(blockOutput + 4 + i) = char(quantized);
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_device);

                uint vWriteByteOffset;
                if (layoutMode == 0) {
                    vWriteByteOffset = kvHeadIndex * maxSequenceLength * vHeadSlotBytes + position * vHeadSlotBytes;
                } else {
                    vWriteByteOffset = position * kvHeadCount * vHeadSlotBytes + kvHeadIndex * vHeadSlotBytes;
                }

                if (vQuantScheme == 0x00 || vQuantScheme == 0x01 || vQuantScheme == 0x02) {
                    for (uint d = tid; d < headDim; d += threadgroupSize) {
                        write_kv_element_dense(args.valueCache + vWriteByteOffset, d, \(castIn("args.newValue[kvIn + d]")), vQuantScheme);
                    }
                } else {
                    const uint groupSize = 32;
                    const uint bytesPerBlock = 36;
                    const uint numGroups = (headDim + groupSize - 1) / groupSize;
                    for (uint g = 0; g < numGroups; g++) {
                        uint groupStart = g * groupSize;
                        float localMin = HUGE_VALF, localMax = -HUGE_VALF;
                        for (uint i = tid; i < groupSize && (groupStart + i) < headDim; i += threadgroupSize) {
                            float val = \(castIn("args.newValue[kvIn + groupStart + i]"));
                            localMin = min(localMin, val); localMax = max(localMax, val);
                        }
                        localMin = simd_min(localMin); localMax = simd_max(localMax);
                        threadgroup float sharedMin[32], sharedMax[32];
                        if (tiisg == 0) { sharedMin[sgitg] = localMin; sharedMax[sgitg] = localMax; }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        if (tid == 0) {
                            uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                            float gMin = sharedMin[0], gMax = sharedMax[0];
                            for (uint s = 1; s < sgCount; s++) { gMin = min(gMin, sharedMin[s]); gMax = max(gMax, sharedMax[s]); }
                            sharedMin[0] = gMin; sharedMax[0] = gMax;
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        float groupMin = sharedMin[0], groupMax = sharedMax[0];
                        float groupScale = (groupMax - groupMin) / 255.0f;
                        if (groupScale < 1e-10f) groupScale = 1e-10f;
                        device uchar* blockOutput = args.valueCache + vWriteByteOffset + g * bytesPerBlock;
                        if (tid == 0) {
                            *(device half*)(blockOutput) = half(groupScale);
                            *(device half*)(blockOutput + 2) = half(groupMin);
                        }
                        threadgroup_barrier(mem_flags::mem_device);
                        for (uint i = tid; i < groupSize && (groupStart + i) < headDim; i += threadgroupSize) {
                            float val = \(castIn("args.newValue[kvIn + groupStart + i]"));
                            int quantized = int(round((val - groupMin) / groupScale));
                            quantized = clamp(quantized, 0, 255);
                            *(device char*)(blockOutput + 4 + i) = char(quantized);
                        }
                    }
                }
            }

            const uint queryOffset = headIndex * headDim;

            float maxScore = -HUGE_VALF;
            float sumExp = 0.0f;

            threadgroup float sharedOutput[4096];
            for (uint d = tid; d < headDim; d += threadgroupSize) {
                sharedOutput[d] = 0.0f;
            }

            for (uint t = 0; t < sequenceLength; t++) {
                uint kByteOffset;
                if (layoutMode == 0) {
                    kByteOffset = kvHeadIndex * maxSequenceLength * kHeadSlotBytes + t * kHeadSlotBytes;
                } else {
                    kByteOffset = t * kvHeadCount * kHeadSlotBytes + kvHeadIndex * kHeadSlotBytes;
                }

                float score = 0.0f;
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    float q = \(castIn("args.query[queryOffset + d]"));
                    float k = (t == position)
                        ? \(castIn("args.newKey[kvIn + d]"))
                        : read_kv_element(args.keyCache + kByteOffset, d, kQuantScheme, kHeadSlotBytes, headDim);
                    score += q * k;
                }
                score = simd_sum(score);

                threadgroup float sharedScore[32];
                if (tiisg == 0) sharedScore[sgitg] = score;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0) {
                    float totalScore = 0.0f;
                    uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                    for (uint s = 0; s < sgCount; s++) totalScore += sharedScore[s];
                    sharedScore[0] = totalScore * scale;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                score = sharedScore[0];

                float prevMax = maxScore;
                maxScore = max(maxScore, score);
                float expScale = exp(prevMax - maxScore);
                float expScore = exp(score - maxScore);
                sumExp = sumExp * expScale + expScore;

                uint vByteOffset;
                if (layoutMode == 0) {
                    vByteOffset = kvHeadIndex * maxSequenceLength * vHeadSlotBytes + t * vHeadSlotBytes;
                } else {
                    vByteOffset = t * kvHeadCount * vHeadSlotBytes + kvHeadIndex * vHeadSlotBytes;
                }

                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    float v = (t == position)
                        ? \(castIn("args.newValue[kvIn + d]"))
                        : read_kv_element(args.valueCache + vByteOffset, d, vQuantScheme, vHeadSlotBytes, headDim);
                    sharedOutput[d] = sharedOutput[d] * expScale + expScore * v;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float invSum = 1.0f / max(sumExp, 1e-20f);
            const uint outputOffset = headIndex * headDim;
            for (uint d = tid; d < headDim; d += threadgroupSize) {
                args.output[outputOffset + d] = \(castOut("sharedOutput[d] * invSum"));
            }
        }
        """
    }

    /// Generate KV cache fill kernel for prefill (1D flat dispatch).
    public static func generateKVCacheFillSeq(
        name: String,
        bufferPrecision: BufferPrecision
    ) -> String {
        let bt = bufferPrecision.metalType
        return """
        kernel void \(name)(
            device const \(bt)* newKeys            [[buffer(0)]],
            device const \(bt)* newValues          [[buffer(1)]],
            device uchar* keyCache                 [[buffer(2)]],
            device uchar* valueCache               [[buffer(3)]],
            constant uint& kvHeadCount             [[buffer(4)]],
            constant uint& headDimension           [[buffer(5)]],
            constant uint& maxSequenceLength       [[buffer(6)]],
            constant uint& sequenceLength          [[buffer(7)]],
            constant uint& layoutMode              [[buffer(8)]],
            constant uint& kQuantScheme            [[buffer(9)]],
            constant uint& vQuantScheme            [[buffer(10)]],
            constant uint& kHeadSlotBytes          [[buffer(11)]],
            constant uint& vHeadSlotBytes          [[buffer(12)]],
            uint groupId                            [[threadgroup_position_in_grid]],
            uint tid                               [[thread_index_in_threadgroup]]
        ) {
            uint d = tid;
            uint pos = groupId;
            if (d >= headDimension || pos >= sequenceLength) return;

            for (uint kvHead = 0; kvHead < kvHeadCount; kvHead++) {
                uint inputIdx = pos * kvHeadCount * headDimension + kvHead * headDimension + d;
                float kVal = float(newKeys[inputIdx]);
                float vVal = float(newValues[inputIdx]);

                uint kByteOffset;
                uint vByteOffset;
                if (layoutMode == 0) {
                    kByteOffset = kvHead * maxSequenceLength * kHeadSlotBytes + pos * kHeadSlotBytes;
                    vByteOffset = kvHead * maxSequenceLength * vHeadSlotBytes + pos * vHeadSlotBytes;
                } else {
                    kByteOffset = pos * kvHeadCount * kHeadSlotBytes + kvHead * kHeadSlotBytes;
                    vByteOffset = pos * kvHeadCount * vHeadSlotBytes + kvHead * vHeadSlotBytes;
                }
                write_kv_element_dense(keyCache + kByteOffset, d, kVal, kQuantScheme);
                write_kv_element_dense(valueCache + vByteOffset, d, vVal, vQuantScheme);
            }
        }
        """
    }

    /// Generate batch causal flash attention kernel for prefill.
    ///
    /// Unlike the per-position decode kernel, this processes ALL positions
    /// in a SINGLE dispatch with grid (headCount, seqLen, 1).
    /// Each threadgroup handles one (head, position) pair with causal masking.
    /// KV cache must be pre-filled by a separate kv_cache_fill dispatch.
    public static func generateBatchFlashAttention(
        name: String,
        bufferPrecision: BufferPrecision
    ) -> String {
        let bt = bufferPrecision.metalType
        let castOut: (String) -> String = { expr in
            bufferPrecision == .float32 ? "(\(expr))" : "\(bt)(\(expr))"
        }

        return """
        kernel void \(name)(
            device const \(bt)* query             [[buffer(0)]],
            device const uchar* keyCache          [[buffer(1)]],
            device const uchar* valueCache        [[buffer(2)]],
            device \(bt)* output                  [[buffer(3)]],
            constant uint& headCount              [[buffer(4)]],
            constant uint& kvHeadCount            [[buffer(5)]],
            constant uint& headDimension          [[buffer(6)]],
            constant float& scale                 [[buffer(7)]],
            constant uint& layoutMode             [[buffer(8)]],
            constant uint& maxSequenceLength      [[buffer(9)]],
            constant uint& sequenceLength         [[buffer(10)]],
            constant uint& kQuantScheme           [[buffer(11)]],
            constant uint& vQuantScheme           [[buffer(12)]],
            constant uint& kHeadSlotBytes         [[buffer(13)]],
            constant uint& vHeadSlotBytes         [[buffer(14)]],
            uint flatGroupId                      [[threadgroup_position_in_grid]],
            uint tid                              [[thread_index_in_threadgroup]],
            uint tiisg                            [[thread_index_in_simdgroup]],
            uint sgitg                            [[simdgroup_index_in_threadgroup]],
            uint threadgroupSize                  [[threads_per_threadgroup]]
        ) {
            const uint headIndex = flatGroupId % headCount;
            const uint posId = flatGroupId / headCount;
            if (headIndex >= headCount || posId >= sequenceLength) return;

            const uint headDim = headDimension;
            const uint kvHeadIndex = headIndex * kvHeadCount / headCount;
            const uint queryOffset = posId * headCount * headDim + headIndex * headDim;

            // Online softmax over positions [0..posId] (causal)
            float maxScore = -HUGE_VALF;
            float sumExp = 0.0f;

            threadgroup float sharedOutput[4096];
            for (uint d = tid; d < headDim; d += threadgroupSize) {
                sharedOutput[d] = 0.0f;
            }

            for (uint t = 0; t <= posId; t++) {
                uint kByteOffset;
                if (layoutMode == 0) {
                    kByteOffset = kvHeadIndex * maxSequenceLength * kHeadSlotBytes + t * kHeadSlotBytes;
                } else {
                    kByteOffset = t * kvHeadCount * kHeadSlotBytes + kvHeadIndex * kHeadSlotBytes;
                }

                float score = 0.0f;
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    float q = float(query[queryOffset + d]);
                    float k = read_kv_element(keyCache + kByteOffset, d, kQuantScheme, kHeadSlotBytes, headDim);
                    score += q * k;
                }
                score = simd_sum(score);
                threadgroup float sharedScore[32];
                if (tiisg == 0) sharedScore[sgitg] = score;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0) {
                    float total = 0.0f;
                    uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                    for (uint s = 0; s < sgCount; s++) total += sharedScore[s];
                    sharedScore[0] = total * scale;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                score = sharedScore[0];

                float oldMax = maxScore;
                maxScore = max(maxScore, score);
                float correction = exp(oldMax - maxScore);
                sumExp = sumExp * correction + exp(score - maxScore);

                uint vByteOffset;
                if (layoutMode == 0) {
                    vByteOffset = kvHeadIndex * maxSequenceLength * vHeadSlotBytes + t * vHeadSlotBytes;
                } else {
                    vByteOffset = t * kvHeadCount * vHeadSlotBytes + kvHeadIndex * vHeadSlotBytes;
                }

                float weight = exp(score - maxScore);
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    float v = read_kv_element(valueCache + vByteOffset, d, vQuantScheme, vHeadSlotBytes, headDim);
                    sharedOutput[d] = sharedOutput[d] * correction + weight * v;
                }
            }

            float invSum = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f;
            for (uint d = tid; d < headDim; d += threadgroupSize) {
                output[queryOffset + d] = \(castOut("sharedOutput[d] * invSum"));
            }
        }
        """
    }

    /// Helper functions for KV cache read/write (shared by all flash_attn variants).
    public static let flashAttentionHelperSource = """
    inline float read_kv_element(
        device const uchar* cache, uint elementIndex, uint kvQuantScheme,
        uint headSlotBytes, uint headDim
    ) {
        if (kvQuantScheme == 0x00) {
            return float(((device const half*)cache)[elementIndex]);
        }
        if (kvQuantScheme == 0x01) {
            return bf16_to_float(((device const uint16_t*)cache)[elementIndex]);
        }
        if (kvQuantScheme == 0x02) {
            return ((device const float*)cache)[elementIndex];
        }
        const uint groupSize = 32;
        const uint bytesPerBlock = 36;
        uint group = elementIndex / groupSize;
        uint indexInGroup = elementIndex % groupSize;
        uint blockOffset = group * bytesPerBlock;
        float scale = float(*(device const half*)(cache + blockOffset));
        float zero = float(*(device const half*)(cache + blockOffset + 2));
        char quantized = *(device const char*)(cache + blockOffset + 4 + indexInGroup);
        return scale * float(quantized) + zero;
    }

    inline void write_kv_element_dense(
        device uchar* cache, uint elementIndex, float value, uint kvQuantScheme
    ) {
        if (kvQuantScheme == 0x01) {
            ((device uint16_t*)cache)[elementIndex] = float_to_bf16(value);
            return;
        }
        if (kvQuantScheme == 0x02) {
            ((device float*)cache)[elementIndex] = value;
            return;
        }
        ((device half*)cache)[elementIndex] = half(value);
    }
    """

    // MARK: - Specialized Kernels

    /// Generate quantized GEMV (Q4 group 64).
    public static func generateQuantizedGEMV_Q4G64(
        name: String,
        bufferPrecision: BufferPrecision
    ) -> String {
        let bt = bufferPrecision.metalType
        return """
        kernel void \(name)(
            device const \(bt)* input       [[buffer(0)]],
            device const uchar* weight     [[buffer(1)]],
            device \(bt)* output            [[buffer(2)]],
            constant uint& inputDimension  [[buffer(3)]],
            constant uint& outputDimension [[buffer(4)]],
            uint2 gid                      [[threadgroup_position_in_grid]],
            uint tiisg                     [[thread_index_in_simdgroup]],
            uint sgitg                     [[simdgroup_index_in_threadgroup]]
        ) {
            const uint WEIGHTS_PER_BLOCK = 64;
            const uint BYTES_PER_BLOCK = 36;
            const uint rowsPerThreadgroup = 2;
            const uint row = gid.x * rowsPerThreadgroup + sgitg;
            if (row >= outputDimension) return;

            const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
            device const uchar* rowBase = weight + row * blocksPerRow * BYTES_PER_BLOCK;
            float sum = 0.0f;

            for (uint b = tiisg; b < blocksPerRow; b += SIMD_WIDTH) {
                device const uchar* block = rowBase + b * BYTES_PER_BLOCK;
                float blockScale = float(*(device const half*)(block));
                float blockZero = float(*(device const half*)(block + 2));
                device const uchar* nibbles = block + 4;
                uint startWeight = b * WEIGHTS_PER_BLOCK;
                for (uint i = 0; i < WEIGHTS_PER_BLOCK / 2; i++) {
                    uchar packed = nibbles[i];
                    float w0 = float(packed & 0x0F) * blockScale + blockZero;
                    float w1 = float(packed >> 4) * blockScale + blockZero;
                    sum += w0 * float(input[startWeight + i * 2]);
                    sum += w1 * float(input[startWeight + i * 2 + 1]);
                }
            }
            sum = simd_sum(sum);
            if (tiisg == 0) output[row] = \(bt)(sum);
        }
        """
    }

    /// Generate quantized GEMV (Q4 group 128).
    public static func generateQuantizedGEMV_Q4G128(
        name: String,
        bufferPrecision: BufferPrecision
    ) -> String {
        let bt = bufferPrecision.metalType
        return """
        kernel void \(name)(
            device const \(bt)* input       [[buffer(0)]],
            device const uchar* weight     [[buffer(1)]],
            device \(bt)* output            [[buffer(2)]],
            constant uint& inputDimension  [[buffer(3)]],
            constant uint& outputDimension [[buffer(4)]],
            uint2 gid                      [[threadgroup_position_in_grid]],
            uint tiisg                     [[thread_index_in_simdgroup]],
            uint sgitg                     [[simdgroup_index_in_threadgroup]]
        ) {
            const uint WEIGHTS_PER_BLOCK = 128;
            const uint BYTES_PER_BLOCK = 68;
            const uint rowsPerThreadgroup = 2;
            const uint row = gid.x * rowsPerThreadgroup + sgitg;
            if (row >= outputDimension) return;

            const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
            device const uchar* rowBase = weight + row * blocksPerRow * BYTES_PER_BLOCK;
            float sum = 0.0f;

            for (uint b = tiisg; b < blocksPerRow; b += SIMD_WIDTH) {
                device const uchar* block = rowBase + b * BYTES_PER_BLOCK;
                float blockScale = float(*(device const half*)(block));
                float blockZero = float(*(device const half*)(block + 2));
                device const uchar* nibbles = block + 4;
                uint startWeight = b * WEIGHTS_PER_BLOCK;
                for (uint i = 0; i < WEIGHTS_PER_BLOCK / 2; i++) {
                    uchar packed = nibbles[i];
                    float w0 = float(packed & 0x0F) * blockScale + blockZero;
                    float w1 = float(packed >> 4) * blockScale + blockZero;
                    sum += w0 * float(input[startWeight + i * 2]);
                    sum += w1 * float(input[startWeight + i * 2 + 1]);
                }
            }
            sum = simd_sum(sum);
            if (tiisg == 0) output[row] = \(bt)(sum);
        }
        """
    }

    /// Generate quantized GEMV (Q8 with configurable group size).
    public static func generateQuantizedGEMV_Q8(
        name: String,
        bufferPrecision: BufferPrecision,
        groupSize: Int
    ) -> String {
        let bt = bufferPrecision.metalType
        let bytesPerBlock = 4 + groupSize  // scale(f16) + zero(f16) + int8 × groupSize
        return """
        kernel void \(name)(
            device const \(bt)* input       [[buffer(0)]],
            device const uchar* weight     [[buffer(1)]],
            device \(bt)* output            [[buffer(2)]],
            constant uint& inputDimension  [[buffer(3)]],
            constant uint& outputDimension [[buffer(4)]],
            uint2 gid                      [[threadgroup_position_in_grid]],
            uint tiisg                     [[thread_index_in_simdgroup]],
            uint sgitg                     [[simdgroup_index_in_threadgroup]]
        ) {
            const uint GROUP_SIZE = \(groupSize);
            const uint BYTES_PER_BLOCK = \(bytesPerBlock);
            const uint rowsPerThreadgroup = 2;
            const uint row = gid.x * rowsPerThreadgroup + sgitg;
            if (row >= outputDimension) return;

            const uint blocksPerRow = inputDimension / GROUP_SIZE;
            device const uchar* rowBase = weight + row * blocksPerRow * BYTES_PER_BLOCK;
            float sum = 0.0f;

            for (uint b = tiisg; b < blocksPerRow; b += SIMD_WIDTH) {
                device const uchar* block = rowBase + b * BYTES_PER_BLOCK;
                float blockScale = float(*(device const half*)(block));
                float blockZero = float(*(device const half*)(block + 2));
                device const char* quantized = (device const char*)(block + 4);
                uint startWeight = b * GROUP_SIZE;
                for (uint i = 0; i < GROUP_SIZE; i++) {
                    float w = blockScale * float(quantized[i]) + blockZero;
                    sum += w * float(input[startWeight + i]);
                }
            }
            sum = simd_sum(sum);
            if (tiisg == 0) output[row] = \(bt)(sum);
        }
        """
    }

    /// Generate quantized GEMM (Q4 group, prefill sequence).
    public static func generateQuantizedGEMM_Q4(
        name: String,
        bufferPrecision: BufferPrecision,
        groupSize: Int
    ) -> String {
        let bt = bufferPrecision.metalType
        let weightsPerBlock = groupSize
        let bytesPerBlock = 4 + groupSize / 2  // scale(f16) + zero(f16) + nibbles
        return """
        kernel void \(name)(
            device const \(bt)* input       [[buffer(0)]],
            device const uchar* weight     [[buffer(1)]],
            device \(bt)* output            [[buffer(2)]],
            constant uint& inputDimension  [[buffer(3)]],
            constant uint& outputDimension [[buffer(4)]],
            constant uint& sequenceLength  [[buffer(5)]],
            uint2 gid                      [[threadgroup_position_in_grid]],
            uint tiisg                     [[thread_index_in_simdgroup]],
            uint sgitg                     [[simdgroup_index_in_threadgroup]]
        ) {
            const uint WEIGHTS_PER_BLOCK = \(weightsPerBlock);
            const uint BYTES_PER_BLOCK = \(bytesPerBlock);
            const uint rowsPerThreadgroup = 2;
            const uint row = gid.x * rowsPerThreadgroup + sgitg;
            const uint seqPos = gid.y;
            if (row >= outputDimension || seqPos >= sequenceLength) return;

            const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
            device const uchar* rowBase = weight + row * blocksPerRow * BYTES_PER_BLOCK;
            device const \(bt)* inputRow = input + seqPos * inputDimension;
            float sum = 0.0f;

            for (uint b = tiisg; b < blocksPerRow; b += SIMD_WIDTH) {
                device const uchar* block = rowBase + b * BYTES_PER_BLOCK;
                float blockScale = float(*(device const half*)(block));
                float blockZero = float(*(device const half*)(block + 2));
                device const uchar* nibbles = block + 4;
                uint startWeight = b * WEIGHTS_PER_BLOCK;
                for (uint i = 0; i < WEIGHTS_PER_BLOCK / 2; i++) {
                    uchar packed = nibbles[i];
                    float w0 = float(packed & 0x0F) * blockScale + blockZero;
                    float w1 = float(packed >> 4) * blockScale + blockZero;
                    sum += w0 * float(inputRow[startWeight + i * 2]);
                    sum += w1 * float(inputRow[startWeight + i * 2 + 1]);
                }
            }
            sum = simd_sum(sum);
            if (tiisg == 0) output[seqPos * outputDimension + row] = \(bt)(sum);
        }
        """
    }

    /// Generate sigmoid gate kernel.
    public static func generateSigmoidGate(
        name: String,
        bufferPrecision: BufferPrecision
    ) -> String {
        let bt = bufferPrecision.metalType
        return """
        kernel void \(name)(
            device const \(bt)* input    [[buffer(0)]],
            device const \(bt)* gate     [[buffer(1)]],
            device \(bt)* output         [[buffer(2)]],
            constant uint& dimension     [[buffer(3)]],
            uint gid                     [[thread_position_in_grid]]
        ) {
            if (gid >= dimension) return;
            float g = float(gate[gid]);
            output[gid] = \(bt)(float(input[gid]) * (1.0f / (1.0f + exp(-g))));
        }
        """
    }

    /// Generate layer norm kernel (decode, single token).
    public static func generateLayerNorm(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        return """
        kernel void \(name)(
            device const \(bt)* input       [[buffer(0)]],
            device const \(wt)* weight      [[buffer(1)]],
            device const \(wt)* bias        [[buffer(2)]],
            device \(bt)* output            [[buffer(3)]],
            constant uint& dimension        [[buffer(4)]],
            constant float& epsilon         [[buffer(5)]],
            uint tid                        [[thread_index_in_threadgroup]],
            uint threadgroupSize            [[threads_per_threadgroup]]
        ) {
            // Compute mean
            float sum = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) sum += float(input[i]);
            sum = simd_sum(sum);
            threadgroup float shared[32];
            if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                float total = 0.0f;
                for (uint s = 0; s < (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH; s++) total += shared[s];
                shared[0] = total / float(dimension);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float mean = shared[0];

            // Compute variance
            float varSum = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float diff = float(input[i]) - mean;
                varSum += diff * diff;
            }
            varSum = simd_sum(varSum);
            if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = varSum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                float total = 0.0f;
                for (uint s = 0; s < (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH; s++) total += shared[s];
                shared[0] = rsqrt(total / float(dimension) + epsilon);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float scale = shared[0];

            for (uint i = tid; i < dimension; i += threadgroupSize) {
                output[i] = \(bt)((float(input[i]) - mean) * scale * \(readWeight("weight[i]")) + \(readWeight("bias[i]")));
            }
        }
        """
    }

    // MARK: - Inline Specialized Sources (complex kernels not yet parameterized)

    static let ssmRecurrenceSource = """
    inline float stable_softplus(float x) { return max(x, 0.0f) + log(1.0f + exp(-abs(x))); }
    inline float compute_l2_inv_norm(threadgroup float* vec, uint dim, uint tid, uint tgSize, threadgroup float* scratch) {
        float sumSq = 0.0f;
        for (uint d = tid; d < dim; d += tgSize) sumSq += vec[d] * vec[d];
        sumSq = simd_sum(sumSq);
        if (tid % SIMD_WIDTH == 0) scratch[tid / SIMD_WIDTH] = sumSq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) { float t = 0; for (uint i = 0; i < (tgSize + SIMD_WIDTH - 1) / SIMD_WIDTH; i++) t += scratch[i]; scratch[0] = rsqrt(t + 1e-6f); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return scratch[0];
    }
    kernel void ssm_recurrence(
        device const half* projectedQKV [[buffer(0)]], device const half* projectedZ [[buffer(1)]],
        device const half* projectedBeta [[buffer(2)]], device const half* projectedAlpha [[buffer(3)]],
        device const half* convWeight [[buffer(4)]], device const half* normWeight [[buffer(5)]],
        device const half* dtBias [[buffer(6)]], device const half* aLog [[buffer(7)]],
        device float* recurrentState [[buffer(8)]], device half* convState [[buffer(9)]],
        device half* output [[buffer(10)]], constant uint& numHeads [[buffer(11)]],
        constant uint& groupCount [[buffer(12)]], constant uint& keyDimension [[buffer(13)]],
        constant uint& valueDimension [[buffer(14)]], constant uint& convKernelSize [[buffer(15)]],
        uint headIndex [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint threadgroupSize [[threads_per_threadgroup]]
    ) {
        const uint dk = keyDimension, dv = valueDimension;
        const uint keyGroupDim = groupCount * dk, convDim = 2 * keyGroupDim + numHeads * dv;
        const uint keyGroupIndex = headIndex / (numHeads / groupCount);
        device float* sharedConvOut = (device float*)output;
        for (uint ch = tid; ch < convDim; ch += threadgroupSize) {
            for (uint k = 0; k < convKernelSize - 1; k++) convState[ch * convKernelSize + k] = convState[ch * convKernelSize + k + 1];
            convState[ch * convKernelSize + convKernelSize - 1] = projectedQKV[ch];
            float sum = 0.0f;
            for (uint k = 0; k < convKernelSize; k++) sum += float(convState[ch * convKernelSize + k]) * float(convWeight[ch * convKernelSize + k]);
            sharedConvOut[ch] = sum / (1.0f + exp(-sum));
        }
        threadgroup_barrier(mem_flags::mem_device);
        threadgroup float sharedQ[256], sharedK[256], sharedV[256];
        for (uint d = tid; d < dk; d += threadgroupSize) { sharedK[d] = sharedConvOut[keyGroupIndex * dk + d]; sharedQ[d] = sharedConvOut[keyGroupDim + keyGroupIndex * dk + d]; }
        for (uint d = tid; d < dv; d += threadgroupSize) sharedV[d] = sharedConvOut[2 * keyGroupDim + headIndex * dv + d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float normScratch[32];
        float qInv = compute_l2_inv_norm(sharedQ, dk, tid, threadgroupSize, normScratch);
        for (uint d = tid; d < dk; d += threadgroupSize) sharedQ[d] *= qInv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float kInv = compute_l2_inv_norm(sharedK, dk, tid, threadgroupSize, normScratch);
        for (uint d = tid; d < dk; d += threadgroupSize) sharedK[d] *= kInv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float qScale = rsqrt(float(dk));
        for (uint d = tid; d < dk; d += threadgroupSize) sharedQ[d] *= qScale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float decay = exp(-exp(float(aLog[headIndex])) * stable_softplus(float(projectedAlpha[headIndex]) + float(dtBias[headIndex])));
        float beta = 1.0f / (1.0f + exp(-float(projectedBeta[headIndex])));
        device float* S = recurrentState + headIndex * dk * dv;
        for (uint idx = tid; idx < dk * dv; idx += threadgroupSize) S[idx] *= decay;
        threadgroup_barrier(mem_flags::mem_device);
        threadgroup float sharedKVMem[256];
        for (uint d = tid; d < dv; d += threadgroupSize) { float dot = 0; for (uint j = 0; j < dk; j++) dot += S[j * dv + d] * sharedK[j]; sharedKVMem[d] = dot; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float sharedDelta[256];
        for (uint d = tid; d < dv; d += threadgroupSize) sharedDelta[d] = beta * (sharedV[d] - sharedKVMem[d]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < dk * dv; idx += threadgroupSize) S[idx] += sharedK[idx / dv] * sharedDelta[idx % dv];
        threadgroup_barrier(mem_flags::mem_device);
        threadgroup float sharedOutput[256];
        for (uint d = tid; d < dv; d += threadgroupSize) { float dot = 0; for (uint j = 0; j < dk; j++) dot += S[j * dv + d] * sharedQ[j]; sharedOutput[d] = dot; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sumSq = 0; for (uint d = tid; d < dv; d += threadgroupSize) sumSq += sharedOutput[d] * sharedOutput[d];
        sumSq = simd_sum(sumSq); threadgroup float normReduce[32];
        if (tid % SIMD_WIDTH == 0) normReduce[tid / SIMD_WIDTH] = sumSq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) { float t = 0; for (uint i = 0; i < (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH; i++) t += normReduce[i]; normReduce[0] = rsqrt(t / float(dv) + 1e-6f); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float rmsScale = normReduce[0];
        for (uint d = tid; d < dv; d += threadgroupSize) {
            float normed = sharedOutput[d] * rmsScale * float(normWeight[d]);
            float z = float(projectedZ[headIndex * dv + d]);
            output[headIndex * dv + d] = half(normed * z / (1.0f + exp(-z)));
        }
    }
    """

    static let kvQuantizationSource = """
    kernel void quantize_kv_q8(
        device const half* input [[buffer(0)]], device uchar* output [[buffer(1)]],
        constant uint& totalElements [[buffer(2)]], constant uint& groupSize [[buffer(3)]],
        constant uint& bytesPerBlock [[buffer(4)]], uint gid [[thread_position_in_grid]]
    ) {
        uint blocksTotal = totalElements / groupSize;
        if (gid >= blocksTotal) return;
        device const half* groupInput = input + gid * groupSize;
        float minV = HUGE_VALF, maxV = -HUGE_VALF;
        for (uint i = 0; i < groupSize; i++) { float v = float(groupInput[i]); minV = min(minV, v); maxV = max(maxV, v); }
        float scale = (maxV - minV) / 255.0f; float zero = minV;
        if (scale < 1e-10f) scale = 1e-10f;
        device uchar* blockOut = output + gid * bytesPerBlock;
        *(device half*)(blockOut) = half(scale); *(device half*)(blockOut + 2) = half(zero);
        for (uint i = 0; i < groupSize; i++) { int q = int(round((float(groupInput[i]) - zero) / scale)); *(device char*)(blockOut + 4 + i) = char(clamp(q, 0, 255)); }
    }
    kernel void dequantize_kv_q8(
        device const uchar* input [[buffer(0)]], device half* output [[buffer(1)]],
        constant uint& totalElements [[buffer(2)]], constant uint& groupSize [[buffer(3)]],
        constant uint& bytesPerBlock [[buffer(4)]], uint gid [[thread_position_in_grid]]
    ) {
        if (gid >= totalElements / groupSize) return;
        device const uchar* block = input + gid * bytesPerBlock;
        float scale = float(*(device const half*)(block)); float zero = float(*(device const half*)(block + 2));
        for (uint i = 0; i < groupSize; i++) output[gid * groupSize + i] = half(scale * float(*(device const char*)(block + 4 + i)) + zero);
    }
    """

    static let gemmMixedPrecisionSource = """
    kernel void gemm_bf16_f32_to_half(
        device const float* input [[buffer(0)]], device const uint16_t* weight [[buffer(1)]],
        device half* output [[buffer(2)]], constant uint& inputDimension [[buffer(3)]],
        constant uint& outputDimension [[buffer(4)]],
        uint2 gid [[threadgroup_position_in_grid]], uint tiisg [[thread_index_in_simdgroup]], uint sgitg [[simdgroup_index_in_threadgroup]]
    ) {
        const uint row = gid.x * 2 + sgitg; if (row >= outputDimension) return;
        float sum = 0.0f;
        for (uint j = tiisg; j < inputDimension; j += SIMD_WIDTH) sum += bf16_to_float(weight[row * inputDimension + j]) * input[j];
        sum = simd_sum(sum); if (tiisg == 0) output[row] = half(sum);
    }
    kernel void gemm_bf16_f32s_halfout(
        device const float* input [[buffer(0)]], device const uint16_t* weight [[buffer(1)]],
        device half* output [[buffer(2)]], constant uint& inputDimension [[buffer(3)]],
        constant uint& outputDimension [[buffer(4)]], constant uint& sequenceLength [[buffer(5)]],
        uint2 gid [[threadgroup_position_in_grid]], uint tiisg [[thread_index_in_simdgroup]], uint sgitg [[simdgroup_index_in_threadgroup]]
    ) {
        const uint row = gid.x * 2 + sgitg, seqPos = gid.y;
        if (row >= outputDimension || seqPos >= sequenceLength) return;
        float sum = 0.0f;
        for (uint j = tiisg; j < inputDimension; j += SIMD_WIDTH) sum += bf16_to_float(weight[row * inputDimension + j]) * input[seqPos * inputDimension + j];
        sum = simd_sum(sum); if (tiisg == 0) output[seqPos * outputDimension + row] = half(sum);
    }
    """

    // MARK: - Argmax

    /// Generate MSL source for argmax.
    public static func generateArgmax(
        name: String,
        bufferPrecision: BufferPrecision
    ) -> String {
        let bt = bufferPrecision.metalType

        return """
        kernel void \(name)(
            device const \(bt)* logits       [[buffer(0)]],
            device int* result               [[buffer(1)]],
            constant uint& vocabularySize    [[buffer(2)]],
            uint tid                         [[thread_index_in_threadgroup]],
            uint threadgroupSize             [[threads_per_threadgroup]]
        ) {
            threadgroup float sharedValues[32];
            threadgroup int sharedIndices[32];

            float localMax = -HUGE_VALF;
            int localIndex = 0;
            for (uint i = tid; i < vocabularySize; i += threadgroupSize) {
                float value = float(logits[i]);
                if (value > localMax) { localMax = value; localIndex = int(i); }
            }

            for (uint offset = SIMD_WIDTH / 2; offset > 0; offset >>= 1) {
                float otherValue = simd_shuffle_down(localMax, offset);
                int otherIndex = simd_shuffle_down(localIndex, offset);
                if (otherValue > localMax) { localMax = otherValue; localIndex = otherIndex; }
            }

            uint simdIndex = tid / SIMD_WIDTH;
            if (tid % SIMD_WIDTH == 0) { sharedValues[simdIndex] = localMax; sharedIndices[simdIndex] = localIndex; }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0) {
                float bestValue = -HUGE_VALF;
                int bestIndex = 0;
                uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                for (uint i = 0; i < sgCount; i++) {
                    if (sharedValues[i] > bestValue) { bestValue = sharedValues[i]; bestIndex = sharedIndices[i]; }
                }
                result[0] = bestIndex;
            }
        }
        """
    }
}
