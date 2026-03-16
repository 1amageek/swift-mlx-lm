/// Compiler-owned Metal Shading Language (MSL) kernel source.
///
/// All kernel source code lives here. MetalComponent does NOT provide kernel source.
/// The compiler maps `MetalComputeOperation.kernelName` and `MetalProjection` to
/// the corresponding MSL function in this registry.
///
/// Follows llama.cpp's design: ggml-metal.metal has ALL kernels in one place.
public enum MetalKernelSource {

    /// Common MSL header included before all kernel source.
    public static let commonHeader = """
    #include <metal_stdlib>
    using namespace metal;

    constant constexpr uint SIMD_WIDTH = 32;

    /// BF16 → float32 conversion. BF16 is stored as uint16, shift left 16 to get float32.
    inline float bf16_to_float(uint16_t bf16) {
        uint32_t f32_bits = uint32_t(bf16) << 16;
        return as_type<float>(f32_bits);
    }
    """

    /// All unfused kernel source concatenated.
    public static var allKernelSource: String {
        ([commonHeader] + allKernels.map(\.source)).joined(separator: "\n\n")
    }

    /// Lookup kernel source by function name.
    public static func source(forKernelName name: String) -> String? {
        allKernels.first(where: { $0.functionName == name })?.source
    }

    // MARK: - Kernel Registry

    private struct KernelRegistryEntry {
        let functionName: String
        let source: String
    }

    private static let allKernels: [KernelRegistryEntry] = [
        // GEMV (shared by all projections)
        KernelRegistryEntry(functionName: "gemv", source: gemvSource),
        // Normalization
        KernelRegistryEntry(functionName: "rms_norm", source: rmsNormSource),
        KernelRegistryEntry(functionName: "layer_norm", source: layerNormSource),
        // Activation
        KernelRegistryEntry(functionName: "swiglu", source: swigluSource),
        // Attention
        KernelRegistryEntry(functionName: "flash_attn_decode", source: flashAttentionDecodeSource),
        KernelRegistryEntry(functionName: "rope", source: ropeSource),
        // QK normalization (per-head RMS norm)
        KernelRegistryEntry(functionName: "qk_rms_norm", source: qkRMSNormSource),
        KernelRegistryEntry(functionName: "qk_rms_norm_bf16", source: qkRMSNormBF16Source),
        // BF16 variants
        KernelRegistryEntry(functionName: "rms_norm_bf16", source: rmsNormBF16Source),
        KernelRegistryEntry(functionName: "embedding_lookup_bf16", source: embeddingLookupBF16Source),
        KernelRegistryEntry(functionName: "fused_copy_rms_norm_bf16", source: fusedCopyRMSNormBF16Source),
        KernelRegistryEntry(functionName: "fused_residual_add_copy_rms_norm_bf16", source: fusedResidualAddCopyRMSNormBF16Source),
        // Embedding / Output
        KernelRegistryEntry(functionName: "embedding_lookup", source: embeddingLookupSource),
        KernelRegistryEntry(functionName: "argmax", source: argmaxSource),
        // Convolution
        KernelRegistryEntry(functionName: "conv1d_gated", source: conv1dGatedSource),
        // State space
        KernelRegistryEntry(functionName: "ssm_recurrence", source: ssmRecurrenceSource),
        // Gate
        KernelRegistryEntry(functionName: "sigmoid_gate", source: sigmoidGateSource),
        // Structural
        KernelRegistryEntry(functionName: "copy_buffer", source: copyBufferSource),
        KernelRegistryEntry(functionName: "residual_add", source: residualAddSource),
        // Quantized GEMV
        KernelRegistryEntry(functionName: "gemv_q4_g64", source: gemvQ4Group64Source),
        KernelRegistryEntry(functionName: "gemv_q4_g128", source: gemvQ4Group128Source),
        KernelRegistryEntry(functionName: "gemv_q8_g32", source: gemvQ8Group32Source),
        KernelRegistryEntry(functionName: "gemv_q8_g64", source: gemvQ8Group64Source),
        // BF16 GEMV
        KernelRegistryEntry(functionName: "gemv_bf16", source: gemvBF16Source),
        // Runtime KV cache quantization
        KernelRegistryEntry(functionName: "quantize_kv_q8", source: quantizeKVQ8Source),
        KernelRegistryEntry(functionName: "dequantize_kv_q8", source: dequantizeKVQ8Source),
        // Fused variants
        KernelRegistryEntry(functionName: "fused_residual_add_copy_rms_norm", source: fusedResidualAddCopyRMSNormSource),
        KernelRegistryEntry(functionName: "fused_copy_rms_norm", source: fusedCopyRMSNormSource),
        // Sequence-aware prefill kernels (operate on [seqLen × dim])
        KernelRegistryEntry(functionName: "gemm", source: gemmSource),
        KernelRegistryEntry(functionName: "gemm_bf16", source: gemmBF16Source),
        KernelRegistryEntry(functionName: "gemm_q4_g64", source: gemmQ4G64Source),
        KernelRegistryEntry(functionName: "gemm_q4_g128", source: gemmQ4G128Source),
        KernelRegistryEntry(functionName: "embedding_lookup_seq", source: embeddingLookupSeqSource),
        KernelRegistryEntry(functionName: "embedding_lookup_seq_bf16", source: embeddingLookupSeqBF16Source),
        KernelRegistryEntry(functionName: "rms_norm_seq", source: rmsNormSeqSource),
        KernelRegistryEntry(functionName: "rms_norm_seq_bf16", source: rmsNormSeqBF16Source),
        KernelRegistryEntry(functionName: "swiglu_seq", source: swigluSeqSource),
        KernelRegistryEntry(functionName: "copy_buffer_seq", source: copyBufferSeqSource),
        KernelRegistryEntry(functionName: "residual_add_seq", source: residualAddSeqSource),
        KernelRegistryEntry(functionName: "rope_seq", source: ropeSeqSource),
        KernelRegistryEntry(functionName: "qk_rms_norm_seq", source: qkRMSNormSeqSource),
        KernelRegistryEntry(functionName: "qk_rms_norm_seq_bf16", source: qkRMSNormSeqBF16Source),
        // Float32 scratch variants — scratch buffer is float32 to prevent overflow
        KernelRegistryEntry(functionName: "gemm_f32s", source: gemmF32ScratchSource),
        KernelRegistryEntry(functionName: "gemm_bf16_f32s", source: gemmBF16F32ScratchSource),
        KernelRegistryEntry(functionName: "gemm_q4_g64_f32s", source: gemmQ4G64F32ScratchSource),
        KernelRegistryEntry(functionName: "gemm_q4_g128_f32s", source: gemmQ4G128F32ScratchSource),
        KernelRegistryEntry(functionName: "embedding_lookup_seq_f32", source: embeddingLookupSeqF32Source),
        KernelRegistryEntry(functionName: "embedding_lookup_seq_bf16_f32", source: embeddingLookupSeqBF16F32Source),
        KernelRegistryEntry(functionName: "rms_norm_seq_f32_inplace", source: rmsNormSeqF32InplaceSource),
        KernelRegistryEntry(functionName: "rms_norm_seq_bf16_f32_inplace", source: rmsNormSeqBF16F32InplaceSource),
        KernelRegistryEntry(functionName: "copy_buffer_seq_f32", source: copyBufferSeqF32Source),
        KernelRegistryEntry(functionName: "residual_add_seq_f32", source: residualAddSeqF32Source),
        KernelRegistryEntry(functionName: "rms_norm_seq_f32s", source: rmsNormSeqF32ScratchSource),
        KernelRegistryEntry(functionName: "rms_norm_seq_bf16_f32s", source: rmsNormSeqBF16F32ScratchSource),
        KernelRegistryEntry(functionName: "swiglu_seq_f32", source: swigluSeqF32Source),
        KernelRegistryEntry(functionName: "rope_seq_f32", source: ropeSeqF32Source),
        KernelRegistryEntry(functionName: "qk_rms_norm_seq_f32", source: qkRMSNormSeqF32Source),
        KernelRegistryEntry(functionName: "conv1d_f32", source: conv1dF32Source),
        KernelRegistryEntry(functionName: "flash_attn_decode_f32", source: flashAttentionDecodeF32Source),
        KernelRegistryEntry(functionName: "gemm_bf16_f32_to_half", source: gemmBF16F32ToHalfSource),
    ]

    // MARK: - GEMV

    private static let gemvSource = """
    /// Matrix-vector multiply: output[row] = dot(weight[row], input)
    /// Grid: (outputDimension / rowsPerThreadgroup, 1, 1)
    /// Threadgroup: (simdgroupCount * simdWidth, 1, 1)
    kernel void gemv(
        device const half* input       [[buffer(0)]],
        device const half* weight      [[buffer(1)]],
        device half* output            [[buffer(2)]],
        constant uint& inputDimension  [[buffer(3)]],
        constant uint& outputDimension [[buffer(4)]],
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint tiisg                     [[thread_index_in_simdgroup]],
        uint sgitg                     [[simdgroup_index_in_threadgroup]]
    ) {
        const uint rowsPerThreadgroup = 2;
        const uint row = gid.x * rowsPerThreadgroup + sgitg;
        if (row >= outputDimension) return;

        float sum = 0.0f;
        device const half* weightRow = weight + row * inputDimension;
        for (uint j = tiisg; j < inputDimension; j += SIMD_WIDTH) {
            sum += float(weightRow[j]) * float(input[j]);
        }
        sum = simd_sum(sum);
        if (tiisg == 0) {
            output[row] = half(sum);
        }
    }
    """

    // MARK: - GEMV BF16

    private static let gemvBF16Source = """
    /// GEMV for BFloat16 weights.
    kernel void gemv_bf16(
        device const half* input               [[buffer(0)]],
        device const uint16_t* weight          [[buffer(1)]],
        device half* output                    [[buffer(2)]],
        constant uint& inputDimension          [[buffer(3)]],
        constant uint& outputDimension         [[buffer(4)]],
        uint2 gid                              [[threadgroup_position_in_grid]],
        uint tiisg                             [[thread_index_in_simdgroup]],
        uint sgitg                             [[simdgroup_index_in_threadgroup]]
    ) {
        const uint rowsPerThreadgroup = 2;
        const uint row = gid.x * rowsPerThreadgroup + sgitg;
        if (row >= outputDimension) return;

        float sum = 0.0f;
        device const uint16_t* weightRow = weight + row * inputDimension;
        for (uint j = tiisg; j < inputDimension; j += SIMD_WIDTH) {
            sum += bf16_to_float(weightRow[j]) * float(input[j]);
        }
        sum = simd_sum(sum);
        if (tiisg == 0) {
            output[row] = half(sum);
        }
    }
    """

    // MARK: - RMS Norm

    private static let rmsNormSource = """
    /// RMS normalization: output = (input / rms) * weight
    /// Grid: (1, 1, 1), Threadgroup: (threads, 1, 1)
    kernel void rms_norm(
        device const half* input     [[buffer(0)]],
        device const half* weight    [[buffer(1)]],
        device half* output          [[buffer(2)]],
        constant uint& dimension     [[buffer(3)]],
        constant float& epsilon      [[buffer(4)]],
        uint tid                     [[thread_index_in_threadgroup]],
        uint threadgroupSize         [[threads_per_threadgroup]]
    ) {
        float sumSquared = 0.0f;
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            float value = float(input[i]);
            sumSquared += value * value;
        }
        sumSquared = simd_sum(sumSquared);

        threadgroup float shared[32];
        uint simdIndex = tid / SIMD_WIDTH;
        uint laneIndex = tid % SIMD_WIDTH;
        if (laneIndex == 0) shared[simdIndex] = sumSquared;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float total = 0.0f;
            uint simdgroupCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
            for (uint i = 0; i < simdgroupCount; i++) total += shared[i];
            shared[0] = rsqrt(total / float(dimension) + epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float scale = shared[0];

        for (uint i = tid; i < dimension; i += threadgroupSize) {
            output[i] = half(float(input[i]) * scale * float(weight[i]));
        }
    }
    """

    // MARK: - Layer Norm

    private static let layerNormSource = """
    kernel void layer_norm(
        device const half* input     [[buffer(0)]],
        device const half* weight    [[buffer(1)]],
        device half* output          [[buffer(2)]],
        constant uint& dimension     [[buffer(3)]],
        constant float& epsilon      [[buffer(4)]],
        uint tid                     [[thread_index_in_threadgroup]],
        uint threadgroupSize         [[threads_per_threadgroup]]
    ) {
        // Mean
        float sum = 0.0f;
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            sum += float(input[i]);
        }
        sum = simd_sum(sum);
        threadgroup float shared[32];
        uint simdIndex = tid / SIMD_WIDTH;
        if (tid % SIMD_WIDTH == 0) shared[simdIndex] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0.0f;
            uint simdgroupCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
            for (uint i = 0; i < simdgroupCount; i++) total += shared[i];
            shared[0] = total / float(dimension);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mean = shared[0];

        // Variance
        float varSum = 0.0f;
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            float diff = float(input[i]) - mean;
            varSum += diff * diff;
        }
        varSum = simd_sum(varSum);
        if (tid % SIMD_WIDTH == 0) shared[simdIndex] = varSum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0.0f;
            uint simdgroupCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
            for (uint i = 0; i < simdgroupCount; i++) total += shared[i];
            shared[0] = rsqrt(total / float(dimension) + epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float invStd = shared[0];

        for (uint i = tid; i < dimension; i += threadgroupSize) {
            output[i] = half((float(input[i]) - mean) * invStd * float(weight[i]));
        }
    }
    """

    // MARK: - SwiGLU

    private static let swigluSource = """
    /// Fused SwiGLU: output = silu(gate) * up
    kernel void swiglu(
        device const half* gate   [[buffer(0)]],
        device const half* up     [[buffer(1)]],
        device half* output       [[buffer(2)]],
        constant uint& count      [[buffer(3)]],
        uint gid                  [[thread_position_in_grid]]
    ) {
        if (gid >= count) return;
        float g = float(gate[gid]);
        float s = g / (1.0f + exp(-g));
        output[gid] = half(s * float(up[gid]));
    }
    """

    // MARK: - Flash Attention Decode

    private static let flashAttentionDecodeSource = """
    /// Read one element from KV cache, dequantizing Q8 blocks on-the-fly if needed.
    ///
    /// kvQuantScheme: 0x00 = FP16 (direct read), 0x10 = Q8_G32 (dequantize)
    /// For FP16: cache is device half*, offset is in half elements.
    /// For Q8_G32: cache is device uchar*, layout is interleaved blocks:
    ///   [scale(f16)][zero(f16)][int8 × 32] per group of 32 elements.
    inline float read_kv_element(
        device const uchar* cache,
        uint elementIndex,
        uint kvQuantScheme,
        uint headSlotBytes,  // bytes per head per token slot
        uint headDim
    ) {
        if (kvQuantScheme == 0x00) {
            // FP16: direct read
            device const half* fp16Cache = (device const half*)cache;
            return float(fp16Cache[elementIndex]);
        }
        // Q8_G32: dequantize from interleaved block
        // Block layout: [scale(f16)][zero(f16)][int8 × 32] = 36 bytes per 32 elements
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

    /// Write one element to KV cache, quantizing to Q8 on-the-fly if needed.
    ///
    /// For Q8: computes per-group scale/zero cooperatively using threadgroup shared memory.
    /// Must be called by all threads in the threadgroup simultaneously for the quantization
    /// reduction to work.
    inline void write_kv_element_fp16(
        device uchar* cache,
        uint elementIndex,
        float value
    ) {
        device half* fp16Cache = (device half*)cache;
        fp16Cache[elementIndex] = half(value);
    }

    /// Flash attention decode: single-token query against KV cache.
    ///
    /// Supports FP16 and Q8 quantized KV cache.
    ///   - FP16 (kvQuantScheme=0x00): direct read/write, no overhead
    ///   - Q8_G32 (kvQuantScheme=0x10): dequantize-on-read, quantize-on-write
    ///     Scale/zero computed per group using threadgroup cooperation.
    ///
    /// Dimension-parallel design:
    ///   - Each thread handles a slice of headDim
    ///   - All threads process one token at a time
    ///   - Coalesced memory access (adjacent threads → adjacent addresses)
    ///
    /// Grid: (headCount, 1, 1), Threadgroup: (threads, 1, 1)
    kernel void flash_attn_decode(
        device const half* query             [[buffer(0)]],
        device const half* newKey            [[buffer(1)]],
        device const half* newValue          [[buffer(2)]],
        device uchar* keyCache               [[buffer(3)]],
        device uchar* valueCache             [[buffer(4)]],
        device half* output                  [[buffer(5)]],
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

        // Head slot strides depend on layout mode and quantization
        // sequence-major [head][seq][dim]: headStride = maxSeqLen * headSlotBytes
        // head-major [seq][head][dim]:     seqStride = kvHeadCount * headSlotBytes

        // --- Step 1: Append new K/V to cache ---
        const bool isWriter = (headIndex == kvHeadIndex * headCount / kvHeadCount);
        if (isWriter) {
            const uint kvIn = kvHeadIndex * headDim;

            // K cache write offset (in bytes)
            uint kWriteByteOffset;
            if (layoutMode == 0) {
                kWriteByteOffset = kvHeadIndex * maxSequenceLength * kHeadSlotBytes
                    + position * kHeadSlotBytes;
            } else {
                kWriteByteOffset = position * kvHeadCount * kHeadSlotBytes
                    + kvHeadIndex * kHeadSlotBytes;
            }

            if (kQuantScheme == 0x00) {
                // FP16: direct write
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    write_kv_element_fp16(keyCache + kWriteByteOffset, d, float(newKey[kvIn + d]));
                }
            } else {
                // Q8_G32: quantize per group, then write interleaved block
                const uint groupSize = 32;
                const uint bytesPerBlock = 36;
                const uint numGroups = (headDim + groupSize - 1) / groupSize;

                for (uint g = 0; g < numGroups; g++) {
                    uint groupStart = g * groupSize;
                    // Find min/max for this group (all threads cooperate)
                    float localMin = HUGE_VALF;
                    float localMax = -HUGE_VALF;
                    for (uint i = tid; i < groupSize && (groupStart + i) < headDim; i += threadgroupSize) {
                        float val = float(newKey[kvIn + groupStart + i]);
                        localMin = min(localMin, val);
                        localMax = max(localMax, val);
                    }
                    localMin = simd_min(localMin);
                    localMax = simd_max(localMax);
                    // Cross-simdgroup reduction
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
                        sharedMin[0] = gMin;
                        sharedMax[0] = gMax;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    float groupMin = sharedMin[0];
                    float groupScale = (sharedMax[0] - groupMin) / 255.0f;
                    if (groupScale < 1e-10f) groupScale = 1e-10f;

                    // Write block header (scale + zero)
                    device uchar* blockPtr = keyCache + kWriteByteOffset + g * bytesPerBlock;
                    if (tid == 0) {
                        *(device half*)(blockPtr) = half(groupScale);
                        *(device half*)(blockPtr + 2) = half(groupMin);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    // Write quantized values
                    for (uint i = tid; i < groupSize && (groupStart + i) < headDim; i += threadgroupSize) {
                        float val = float(newKey[kvIn + groupStart + i]);
                        int q = int(round((val - groupMin) / groupScale));
                        q = clamp(q, 0, 255);
                        *(device char*)(blockPtr + 4 + i) = char(q);
                    }
                }
            }

            // V cache write (same pattern)
            uint vWriteByteOffset;
            if (layoutMode == 0) {
                vWriteByteOffset = kvHeadIndex * maxSequenceLength * vHeadSlotBytes
                    + position * vHeadSlotBytes;
            } else {
                vWriteByteOffset = position * kvHeadCount * vHeadSlotBytes
                    + kvHeadIndex * vHeadSlotBytes;
            }

            if (vQuantScheme == 0x00) {
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    write_kv_element_fp16(valueCache + vWriteByteOffset, d, float(newValue[kvIn + d]));
                }
            } else {
                const uint groupSize = 32;
                const uint bytesPerBlock = 36;
                const uint numGroups = (headDim + groupSize - 1) / groupSize;

                for (uint g = 0; g < numGroups; g++) {
                    uint groupStart = g * groupSize;
                    float localMin = HUGE_VALF, localMax = -HUGE_VALF;
                    for (uint i = tid; i < groupSize && (groupStart + i) < headDim; i += threadgroupSize) {
                        float val = float(newValue[kvIn + groupStart + i]);
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
                    float groupScale = (sharedMax[0] - groupMin) / 255.0f;
                    if (groupScale < 1e-10f) groupScale = 1e-10f;

                    device uchar* blockPtr = valueCache + vWriteByteOffset + g * bytesPerBlock;
                    if (tid == 0) {
                        *(device half*)(blockPtr) = half(groupScale);
                        *(device half*)(blockPtr + 2) = half(groupMin);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint i = tid; i < groupSize && (groupStart + i) < headDim; i += threadgroupSize) {
                        float val = float(newValue[kvIn + groupStart + i]);
                        int q = int(round((val - groupMin) / groupScale));
                        q = clamp(q, 0, 255);
                        *(device char*)(blockPtr + 4 + i) = char(q);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // --- Step 2: Dimension-parallel attention ---
        const uint queryOffset = headIndex * headDim;
        const uint dimsPerThread = (headDim + threadgroupSize - 1) / threadgroupSize;

        float qLocal[8];
        for (uint i = 0; i < dimsPerThread; i++) {
            uint d = tid + i * threadgroupSize;
            qLocal[i] = (d < headDim) ? float(query[queryOffset + d]) : 0.0f;
        }

        float runningMax = -HUGE_VALF;
        float runningSum = 0.0f;
        float accumOutput[8];
        for (uint i = 0; i < dimsPerThread; i++) {
            accumOutput[i] = 0.0f;
        }

        threadgroup float sharedScores[32];

        for (uint t = 0; t < sequenceLength; t++) {
            // K cache byte offset for token t
            uint kByteOffset;
            if (layoutMode == 0) {
                kByteOffset = kvHeadIndex * maxSequenceLength * kHeadSlotBytes
                    + t * kHeadSlotBytes;
            } else {
                kByteOffset = t * kvHeadCount * kHeadSlotBytes
                    + kvHeadIndex * kHeadSlotBytes;
            }

            // Partial dot product with on-the-fly dequantization
            float partialScore = 0.0f;
            for (uint i = 0; i < dimsPerThread; i++) {
                uint d = tid + i * threadgroupSize;
                if (d < headDim) {
                    float kVal = read_kv_element(
                        keyCache + kByteOffset, d, kQuantScheme, kHeadSlotBytes, headDim);
                    partialScore += qLocal[i] * kVal;
                }
            }

            // Reduce to full score
            float score = simd_sum(partialScore);
            if (tiisg == 0) { sharedScores[sgitg] = score; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                float total = 0.0f;
                uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                for (uint s = 0; s < sgCount; s++) total += sharedScores[s];
                sharedScores[0] = total * scale;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            score = sharedScores[0];

            // Online softmax
            float newMax = max(runningMax, score);
            float correction = exp(runningMax - newMax);
            float expScore = exp(score - newMax);
            runningSum = runningSum * correction + expScore;
            runningMax = newMax;

            // Weighted V accumulation with dequantization
            uint vByteOffset;
            if (layoutMode == 0) {
                vByteOffset = kvHeadIndex * maxSequenceLength * vHeadSlotBytes
                    + t * vHeadSlotBytes;
            } else {
                vByteOffset = t * kvHeadCount * vHeadSlotBytes
                    + kvHeadIndex * vHeadSlotBytes;
            }

            for (uint i = 0; i < dimsPerThread; i++) {
                uint d = tid + i * threadgroupSize;
                if (d < headDim) {
                    float vVal = read_kv_element(
                        valueCache + vByteOffset, d, vQuantScheme, vHeadSlotBytes, headDim);
                    accumOutput[i] = accumOutput[i] * correction + expScore * vVal;
                }
            }
        }

        // --- Step 3: Write output ---
        float invSum = (runningSum > 0.0f) ? (1.0f / runningSum) : 0.0f;
        for (uint i = 0; i < dimsPerThread; i++) {
            uint d = tid + i * threadgroupSize;
            if (d < headDim) {
                output[queryOffset + d] = half(accumOutput[i] * invSum);
            }
        }
    }
    """

    // MARK: - RoPE

    private static let ropeSource = """
    /// Rotary Position Embedding applied in-place to Q and K.
    ///
    /// Grid: (max(headCount, kvHeadCount), 1, 1)
    /// Threadgroup: (ropeDimension/2, 1, 1) — one thread per dimension pair
    ///
    /// Position is read from a shared MTLBuffer (runtime value, not compile-time).
    /// GQA: Q has headCount heads, K has kvHeadCount heads.
    /// Each threadgroup applies RoPE to Q[headIndex] if headIndex < headCount,
    /// and to K[headIndex] if headIndex < kvHeadCount.
    ///
    /// Partial rotary: if ropeDimension < headDimension, only the first
    /// ropeDimension elements are rotated, the rest are untouched.
    kernel void rope(
        device half* query                   [[buffer(0)]],   // [headCount * headDim] in-place
        device half* key                     [[buffer(1)]],   // [kvHeadCount * headDim] in-place
        device const uint* positionBuffer    [[buffer(2)]],   // current position (shared, runtime)
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

        // Apply to Q (if this head exists in Q)
        if (headIndex < headCount) {
            uint qOffset = headIndex * headDimension + tid;
            float q0 = float(query[qOffset]);
            float q1 = float(query[qOffset + halfRopeDim]);
            query[qOffset] = half(q0 * cosTheta - q1 * sinTheta);
            query[qOffset + halfRopeDim] = half(q0 * sinTheta + q1 * cosTheta);
        }

        // Apply to K (if this head exists in K)
        if (headIndex < kvHeadCount) {
            uint kOffset = headIndex * headDimension + tid;
            float k0 = float(key[kOffset]);
            float k1 = float(key[kOffset + halfRopeDim]);
            key[kOffset] = half(k0 * cosTheta - k1 * sinTheta);
            key[kOffset + halfRopeDim] = half(k0 * sinTheta + k1 * cosTheta);
        }
    }
    """

    // MARK: - QK RMS Norm (Per-Head)

    private static let qkRMSNormSource = """
    /// Per-head RMS normalization for QK norm in attention.
    /// Normalizes each head independently: data[head*headDim..(head+1)*headDim].
    /// Applied in-place to Q or K projection output.
    ///
    /// Grid: (headCount, 1, 1) — one threadgroup per head
    /// Threadgroup: (threads, 1, 1)
    kernel void qk_rms_norm(
        device half* data              [[buffer(0)]],  // in-place Q or K
        device const half* weight      [[buffer(1)]],  // norm weight [headDim]
        constant uint& headCount       [[buffer(2)]],
        constant uint& headDim         [[buffer(3)]],
        constant float& epsilon        [[buffer(4)]],
        uint headIndex                 [[threadgroup_position_in_grid]],
        uint tid                       [[thread_index_in_threadgroup]],
        uint threadgroupSize           [[threads_per_threadgroup]]
    ) {
        if (headIndex >= headCount) return;
        uint offset = headIndex * headDim;

        // Compute sum of squares for this head
        float sumSquared = 0.0f;
        for (uint i = tid; i < headDim; i += threadgroupSize) {
            float v = float(data[offset + i]);
            sumSquared += v * v;
        }
        sumSquared = simd_sum(sumSquared);

        threadgroup float shared[32];
        uint simdIndex = tid / SIMD_WIDTH;
        if (tid % SIMD_WIDTH == 0) shared[simdIndex] = sumSquared;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float total = 0.0f;
            uint simdgroupCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
            for (uint i = 0; i < simdgroupCount; i++) total += shared[i];
            shared[0] = rsqrt(total / float(headDim) + epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float scale = shared[0];

        // Apply norm + weight
        for (uint i = tid; i < headDim; i += threadgroupSize) {
            data[offset + i] = half(float(data[offset + i]) * scale * float(weight[i]));
        }
    }
    """

    private static let qkRMSNormBF16Source = """
    /// Per-head RMS normalization with BF16 weight for QK norm in attention.
    kernel void qk_rms_norm_bf16(
        device half* data              [[buffer(0)]],
        device const uint16_t* weight  [[buffer(1)]],
        constant uint& headCount       [[buffer(2)]],
        constant uint& headDim         [[buffer(3)]],
        constant float& epsilon        [[buffer(4)]],
        uint headIndex                 [[threadgroup_position_in_grid]],
        uint tid                       [[thread_index_in_threadgroup]],
        uint threadgroupSize           [[threads_per_threadgroup]]
    ) {
        if (headIndex >= headCount) return;
        uint offset = headIndex * headDim;

        float sumSquared = 0.0f;
        for (uint i = tid; i < headDim; i += threadgroupSize) {
            float v = float(data[offset + i]);
            sumSquared += v * v;
        }
        sumSquared = simd_sum(sumSquared);

        threadgroup float shared[32];
        uint simdIndex = tid / SIMD_WIDTH;
        if (tid % SIMD_WIDTH == 0) shared[simdIndex] = sumSquared;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float total = 0.0f;
            uint simdgroupCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
            for (uint i = 0; i < simdgroupCount; i++) total += shared[i];
            shared[0] = rsqrt(total / float(headDim) + epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float scale = shared[0];

        for (uint i = tid; i < headDim; i += threadgroupSize) {
            data[offset + i] = half(float(data[offset + i]) * scale * bf16_to_float(weight[i]));
        }
    }
    """

    // MARK: - Embedding

    private static let embeddingLookupSource = """
    kernel void embedding_lookup(
        device const int* tokenID            [[buffer(0)]],
        device const half* embeddingTable    [[buffer(1)]],
        device half* output                  [[buffer(2)]],
        constant uint& embeddingDimension    [[buffer(3)]],
        uint gid                             [[thread_position_in_grid]]
    ) {
        if (gid >= embeddingDimension) return;
        int token = tokenID[0];
        output[gid] = embeddingTable[token * embeddingDimension + gid];
    }
    """

    // MARK: - Argmax

    private static let argmaxSource = """
    kernel void argmax(
        device const half* logits    [[buffer(0)]],
        device int* result           [[buffer(1)]],
        constant uint& vocabularySize [[buffer(2)]],
        uint tid                     [[thread_index_in_threadgroup]],
        uint threadgroupSize         [[threads_per_threadgroup]]
    ) {
        threadgroup float sharedValues[32];
        threadgroup int sharedIndices[32];

        float localMax = -HUGE_VALF;
        int localIndex = 0;
        for (uint i = tid; i < vocabularySize; i += threadgroupSize) {
            float value = float(logits[i]);
            if (value > localMax) {
                localMax = value;
                localIndex = int(i);
            }
        }

        // SIMD reduction
        for (uint offset = SIMD_WIDTH / 2; offset > 0; offset >>= 1) {
            float otherValue = simd_shuffle_down(localMax, offset);
            int otherIndex = simd_shuffle_down(localIndex, offset);
            if (otherValue > localMax) {
                localMax = otherValue;
                localIndex = otherIndex;
            }
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
            uint simdgroupCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
            for (uint i = 0; i < simdgroupCount; i++) {
                if (sharedValues[i] > bestValue) {
                    bestValue = sharedValues[i];
                    bestIndex = sharedIndices[i];
                }
            }
            result[0] = bestIndex;
        }
    }
    """

    // MARK: - Conv1d Gated

    private static let conv1dGatedSource = """
    kernel void conv1d_gated(
        device const half* input     [[buffer(0)]],
        device const half* weight    [[buffer(1)]],
        device half* output          [[buffer(2)]],
        constant uint& dimension     [[buffer(3)]],
        constant uint& kernelSize    [[buffer(4)]],
        uint gid                     [[thread_position_in_grid]]
    ) {
        if (gid >= dimension) return;
        float sum = 0.0f;
        for (uint k = 0; k < kernelSize; k++) {
            sum += float(input[k * dimension + gid]) * float(weight[gid * kernelSize + k]);
        }
        output[gid] = half(sum);
    }
    """

    // MARK: - SSM Recurrence (DeltaNet Decode)

    private static let ssmRecurrenceSource = """
    /// Stable softplus: max(x, 0) + log1p(exp(-abs(x)))
    inline float stable_softplus(float x) {
        return max(x, 0.0f) + log(1.0f + exp(-abs(x)));
    }

    /// L2 normalize helper — computes inverse norm via simd reduction.
    /// threadgroup scratch must be provided by the caller (kernel function).
    inline float compute_l2_inv_norm(
        threadgroup float* vec, uint dim, uint tid, uint tgSize,
        threadgroup float* scratch
    ) {
        float sumSq = 0.0f;
        for (uint d = tid; d < dim; d += tgSize) {
            sumSq += vec[d] * vec[d];
        }
        sumSq = simd_sum(sumSq);
        uint si = tid / SIMD_WIDTH;
        if (tid % SIMD_WIDTH == 0) scratch[si] = sumSq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0.0f;
            uint sgCount = (tgSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
            for (uint i = 0; i < sgCount; i++) total += scratch[i];
            scratch[0] = rsqrt(total + 1e-6f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return scratch[0];
    }

    /// DeltaNet decode: single-step recurrence with conv1d, gating, and state update.
    ///
    /// Grid: (numHeads, 1, 1) — one threadgroup per value head.
    ///
    /// After 4 GEMV projections, the scratch buffers contain:
    ///   projQKV: mixed Q/K/V for conv1d input
    ///   projZ:   output gate values
    ///   projB:   beta gate (sigmoid → state update weight)
    ///   projA:   alpha gate (softplus → decay computation)
    ///
    /// The kernel performs conv1d, activation, DeltaNet recurrence, and norm+gate.
    kernel void ssm_recurrence(
        device const half* projectedQKV      [[buffer(0)]],   // [groupCount*(2*dk) + numHeads*dv]
        device const half* projectedZ        [[buffer(1)]],   // [numHeads*dv]
        device const half* projectedBeta     [[buffer(2)]],   // [numHeads]
        device const half* projectedAlpha    [[buffer(3)]],   // [numHeads]
        device const half* convWeight        [[buffer(4)]],   // [convDim, kernelSize]
        device const half* normWeight        [[buffer(5)]],   // [dv]
        device const half* dtBias            [[buffer(6)]],   // [numHeads]
        device const half* aLog              [[buffer(7)]],   // [numHeads]
        device float* recurrentState         [[buffer(8)]],   // [numHeads, dk, dv] float32
        device half* convState               [[buffer(9)]],   // [convDim, kernelSize]
        device half* output                  [[buffer(10)]],  // [numHeads*dv]
        constant uint& numHeads              [[buffer(11)]],
        constant uint& groupCount            [[buffer(12)]],
        constant uint& keyDimension          [[buffer(13)]],
        constant uint& valueDimension        [[buffer(14)]],
        constant uint& convKernelSize        [[buffer(15)]],
        uint headIndex                       [[threadgroup_position_in_grid]],
        uint tid                             [[thread_index_in_threadgroup]],
        uint threadgroupSize                 [[threads_per_threadgroup]]
    ) {
        const uint dk = keyDimension;
        const uint dv = valueDimension;
        const uint keyGroupDim = groupCount * dk;
        const uint convDim = 2 * keyGroupDim + numHeads * dv;

        // Compute which key group this head belongs to (for GQA expansion)
        const uint repeatFactor = numHeads / groupCount;
        const uint keyGroupIndex = headIndex / repeatFactor;

        // --- Step 1: Conv1d (decode, T=1) ---
        // Update conv state: shift left, append new values.
        // Then compute depthwise conv1d: dot(convState[channel], convWeight[channel])
        //
        // Use threadgroup shared memory for the activated conv output.
        // convDim can be large (e.g., 2*16*128 + 16*128 = 6144), so each thread
        // handles multiple channels.

        // Conv output stored in device memory scratch instead of threadgroup
        // to avoid exceeding 32KB threadgroup memory limit.
        // The kernel uses the output buffer temporarily for conv results.
        device float* sharedConvOut = (device float*)output;  // reuse output buffer as scratch

        for (uint ch = tid; ch < convDim; ch += threadgroupSize) {
            // Shift conv state left and append new input
            for (uint k = 0; k < convKernelSize - 1; k++) {
                convState[ch * convKernelSize + k] = convState[ch * convKernelSize + k + 1];
            }
            convState[ch * convKernelSize + convKernelSize - 1] = projectedQKV[ch];

            // Depthwise conv1d: dot product along kernel dimension
            float sum = 0.0f;
            for (uint k = 0; k < convKernelSize; k++) {
                sum += float(convState[ch * convKernelSize + k]) * float(convWeight[ch * convKernelSize + k]);
            }

            // SiLU activation
            float activated = sum / (1.0f + exp(-sum));
            sharedConvOut[ch] = activated;
        }
        threadgroup_barrier(mem_flags::mem_device);

        // --- Step 2: Split into Q, K, V ---
        // Layout after conv+activation:
        //   [0 .. keyGroupDim-1]:          K for all groups
        //   [keyGroupDim .. 2*keyGroupDim-1]: Q for all groups (in GGUF order)
        //   [2*keyGroupDim ..]:            V for all heads
        //
        // Wait — actually the HF/GGUF convention packs as [K, Q, V] or [Q, K, V].
        // We follow the LMIR convention: QKV mixed.
        // For DeltaNet: the first groupCount*dk is K, next groupCount*dk is Q,
        // last numHeads*dv is V.

        threadgroup float sharedQ[256];   // max dk
        threadgroup float sharedK[256];   // max dk
        threadgroup float sharedV[256];   // max dv

        // Extract this head's Q, K, V from the conv output
        for (uint d = tid; d < dk; d += threadgroupSize) {
            sharedK[d] = sharedConvOut[keyGroupIndex * dk + d];
            sharedQ[d] = sharedConvOut[keyGroupDim + keyGroupIndex * dk + d];
        }
        for (uint d = tid; d < dv; d += threadgroupSize) {
            sharedV[d] = sharedConvOut[2 * keyGroupDim + headIndex * dv + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Step 3: L2 normalize Q, K; scale Q ---
        threadgroup float normScratch[32];
        float qInvNorm = compute_l2_inv_norm(sharedQ, dk, tid, threadgroupSize, normScratch);
        for (uint d = tid; d < dk; d += threadgroupSize) { sharedQ[d] *= qInvNorm; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float kInvNorm = compute_l2_inv_norm(sharedK, dk, tid, threadgroupSize, normScratch);
        for (uint d = tid; d < dk; d += threadgroupSize) { sharedK[d] *= kInvNorm; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float qScale = rsqrt(float(dk));
        for (uint d = tid; d < dk; d += threadgroupSize) {
            sharedQ[d] *= qScale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Step 4: Compute decay and beta ---
        // decay = exp(-exp(A_log[h]) * softplus(alpha[h] + dtBias[h]))
        // beta = sigmoid(projectedBeta[h])
        // Scalars — all threads compute the same value, no threadgroup needed.
        float a = float(aLog[headIndex]);
        float alpha = float(projectedAlpha[headIndex]);
        float dt = float(dtBias[headIndex]);
        float g = -exp(a) * stable_softplus(alpha + dt);
        float decay = exp(g);
        float beta = 1.0f / (1.0f + exp(-float(projectedBeta[headIndex])));

        // --- Step 5: DeltaNet recurrence ---
        // State S: [dk, dv] matrix in float32
        // S = decay * S + beta * k ⊗ (v - decay * S^T * k)
        //
        // For decode, each thread handles a slice of the state matrix.
        // Thread cooperation: each thread owns a row (or partial row) of S.

        device float* S = recurrentState + headIndex * dk * dv;

        // Step 5a: Decay state
        for (uint idx = tid; idx < dk * dv; idx += threadgroupSize) {
            S[idx] *= decay;
        }
        threadgroup_barrier(mem_flags::mem_device);

        // Step 5b: Memory readout: kv_mem = S^T @ k → [dv]
        // For each dv dimension, compute dot(S[:, d], k[:])
        threadgroup float sharedKVMem[256];  // max dv
        for (uint d = tid; d < dv; d += threadgroupSize) {
            float dot = 0.0f;
            for (uint j = 0; j < dk; j++) {
                dot += S[j * dv + d] * sharedK[j];
            }
            sharedKVMem[d] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 5c: State delta: delta = beta * (v - kv_mem)
        threadgroup float sharedDelta[256];  // max dv
        for (uint d = tid; d < dv; d += threadgroupSize) {
            sharedDelta[d] = beta * (sharedV[d] - sharedKVMem[d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 5d: State update: S += k ⊗ delta (outer product)
        for (uint idx = tid; idx < dk * dv; idx += threadgroupSize) {
            uint j = idx / dv;  // key dim
            uint d = idx % dv;  // value dim
            S[idx] += sharedK[j] * sharedDelta[d];
        }
        threadgroup_barrier(mem_flags::mem_device);

        // Step 5e: Output readout: o = S^T @ q → [dv]
        threadgroup float sharedOutput[256];  // max dv
        for (uint d = tid; d < dv; d += threadgroupSize) {
            float dot = 0.0f;
            for (uint j = 0; j < dk; j++) {
                dot += S[j * dv + d] * sharedQ[j];
            }
            sharedOutput[d] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Step 6: RMS norm + z gate ---
        // output = rms_norm(o) * silu(z)
        // RMS norm: o / sqrt(mean(o^2) + eps) * weight
        float sumSq = 0.0f;
        for (uint d = tid; d < dv; d += threadgroupSize) {
            sumSq += sharedOutput[d] * sharedOutput[d];
        }
        sumSq = simd_sum(sumSq);
        threadgroup float normReduce[32];
        uint si = tid / SIMD_WIDTH;
        if (tid % SIMD_WIDTH == 0) normReduce[si] = sumSq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0.0f;
            uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
            for (uint i = 0; i < sgCount; i++) total += normReduce[i];
            normReduce[0] = rsqrt(total / float(dv) + 1e-6f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float rmsScale = normReduce[0];

        // Write final output: normed * weight * silu(z)
        for (uint d = tid; d < dv; d += threadgroupSize) {
            float normed = sharedOutput[d] * rmsScale * float(normWeight[d]);
            float z = float(projectedZ[headIndex * dv + d]);
            float siluZ = z / (1.0f + exp(-z));
            output[headIndex * dv + d] = half(normed * siluZ);
        }
    }
    """

    // MARK: - Sigmoid Gate

    private static let sigmoidGateSource = """
    kernel void sigmoid_gate(
        device const half* input     [[buffer(0)]],
        device const half* gate      [[buffer(1)]],
        device half* output          [[buffer(2)]],
        constant uint& dimension     [[buffer(3)]],
        uint gid                     [[thread_position_in_grid]]
    ) {
        if (gid >= dimension) return;
        float g = float(gate[gid]);
        float s = 1.0f / (1.0f + exp(-g));
        output[gid] = half(float(input[gid]) * s);
    }
    """

    // MARK: - Structural

    private static let copyBufferSource = """
    kernel void copy_buffer(
        device const half* input     [[buffer(0)]],
        device half* output          [[buffer(1)]],
        constant uint& count         [[buffer(2)]],
        uint gid                     [[thread_position_in_grid]]
    ) {
        if (gid >= count) return;
        output[gid] = input[gid];
    }
    """

    private static let residualAddSource = """
    kernel void residual_add(
        device const half* input     [[buffer(0)]],
        device const half* residual  [[buffer(1)]],
        device half* output          [[buffer(2)]],
        constant uint& count         [[buffer(3)]],
        uint gid                     [[thread_position_in_grid]]
    ) {
        if (gid >= count) return;
        output[gid] = half(float(input[gid]) + float(residual[gid]));
    }
    """

    // MARK: - Fused Variants

    private static let fusedResidualAddCopyRMSNormSource = """
    /// Fused: residual_add + copy + rms_norm
    /// Reads hidden + residual, writes normed output + saves new residual.
    kernel void fused_residual_add_copy_rms_norm(
        device half* hidden          [[buffer(0)]],
        device half* residual        [[buffer(1)]],
        device const half* weight    [[buffer(2)]],
        device half* output          [[buffer(3)]],
        constant uint& dimension     [[buffer(4)]],
        constant float& epsilon      [[buffer(5)]],
        uint tid                     [[thread_index_in_threadgroup]],
        uint threadgroupSize         [[threads_per_threadgroup]]
    ) {
        // Step 1: residual add + copy
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            float sum = float(hidden[i]) + float(residual[i]);
            hidden[i] = half(sum);
            residual[i] = half(sum);
        }
        threadgroup_barrier(mem_flags::mem_device);

        // Step 2: RMS norm
        float sumSquared = 0.0f;
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            float value = float(hidden[i]);
            sumSquared += value * value;
        }
        sumSquared = simd_sum(sumSquared);

        threadgroup float shared[32];
        uint simdIndex = tid / SIMD_WIDTH;
        if (tid % SIMD_WIDTH == 0) shared[simdIndex] = sumSquared;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float total = 0.0f;
            uint simdgroupCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
            for (uint i = 0; i < simdgroupCount; i++) total += shared[i];
            shared[0] = rsqrt(total / float(dimension) + epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float scale = shared[0];

        for (uint i = tid; i < dimension; i += threadgroupSize) {
            output[i] = half(float(hidden[i]) * scale * float(weight[i]));
        }
    }
    """

    private static let fusedCopyRMSNormSource = """
    /// Fused: copy + rms_norm
    /// Saves residual copy and produces normed output in one dispatch.
    kernel void fused_copy_rms_norm(
        device const half* input     [[buffer(0)]],
        device half* residual        [[buffer(1)]],
        device const half* weight    [[buffer(2)]],
        device half* output          [[buffer(3)]],
        constant uint& dimension     [[buffer(4)]],
        constant float& epsilon      [[buffer(5)]],
        uint tid                     [[thread_index_in_threadgroup]],
        uint threadgroupSize         [[threads_per_threadgroup]]
    ) {
        // Step 1: copy to residual
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            residual[i] = input[i];
        }

        // Step 2: RMS norm (reads from input, not residual)
        float sumSquared = 0.0f;
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            float value = float(input[i]);
            sumSquared += value * value;
        }
        sumSquared = simd_sum(sumSquared);

        threadgroup float shared[32];
        uint simdIndex = tid / SIMD_WIDTH;
        if (tid % SIMD_WIDTH == 0) shared[simdIndex] = sumSquared;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float total = 0.0f;
            uint simdgroupCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
            for (uint i = 0; i < simdgroupCount; i++) total += shared[i];
            shared[0] = rsqrt(total / float(dimension) + epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float scale = shared[0];

        for (uint i = tid; i < dimension; i += threadgroupSize) {
            output[i] = half(float(input[i]) * scale * float(weight[i]));
        }
    }
    """

    // MARK: - Quantized GEMV (STAF Interleaved Blocks)

    /// STAF block struct for 4-bit quantization, shared across group sizes.
    private static let stafBlockStructs = """
    /// STAF interleaved block: [scale (f16)][zero (f16)][packed quants]
    /// Scale and quants are in the same cache line for efficient access.
    struct BlockQ4Header {
        half scale;
        half zero;
    };
    """

    /// Q4 group=64: 36 bytes/block (4B header + 32B quants), 64 weights
    private static let gemvQ4Group64Source = """
    \(stafBlockStructs)

    /// Quantized GEMV for 4-bit affine, group size 64.
    /// Reads STAF interleaved blocks: [scale|zero|64 packed 4-bit weights]
    /// Dequantization: w = scale * q + zero  (q is 0..15)
    kernel void gemv_q4_g64(
        device const half* input              [[buffer(0)]],
        device const uchar* weight            [[buffer(1)]],
        device half* output                   [[buffer(2)]],
        constant uint& inputDimension         [[buffer(3)]],
        constant uint& outputDimension        [[buffer(4)]],
        uint2 gid                             [[threadgroup_position_in_grid]],
        uint tiisg                            [[thread_index_in_simdgroup]],
        uint sgitg                            [[simdgroup_index_in_threadgroup]]
    ) {
        const uint WEIGHTS_PER_BLOCK = 64;
        const uint BYTES_PER_BLOCK = 36;  // 4 + 32
        const uint rowsPerThreadgroup = 2;
        const uint row = gid.x * rowsPerThreadgroup + sgitg;
        if (row >= outputDimension) return;

        const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
        device const uchar* rowBase = weight + row * blocksPerRow * BYTES_PER_BLOCK;

        float sum = 0.0f;

        for (uint b = tiisg; b < blocksPerRow; b += SIMD_WIDTH) {
            device const uchar* blockPtr = rowBase + b * BYTES_PER_BLOCK;

            // Read header: scale (2B) + zero (2B)
            float scale = float(*(device const half*)(blockPtr));
            float zero = float(*(device const half*)(blockPtr + 2));

            // Read 32 bytes of packed 4-bit quants (64 weights)
            device const uchar* quants = blockPtr + 4;
            uint baseIndex = b * WEIGHTS_PER_BLOCK;

            for (uint i = 0; i < 32; i++) {
                uchar packed = quants[i];
                float w0 = scale * float(packed & 0xF) + zero;
                float w1 = scale * float(packed >> 4) + zero;
                sum += w0 * float(input[baseIndex + i * 2]);
                sum += w1 * float(input[baseIndex + i * 2 + 1]);
            }
        }

        sum = simd_sum(sum);
        if (tiisg == 0) {
            output[row] = half(sum);
        }
    }
    """

    /// Q4 group=128: 68 bytes/block (4B header + 64B quants), 128 weights
    private static let gemvQ4Group128Source = """
    kernel void gemv_q4_g128(
        device const half* input              [[buffer(0)]],
        device const uchar* weight            [[buffer(1)]],
        device half* output                   [[buffer(2)]],
        constant uint& inputDimension         [[buffer(3)]],
        constant uint& outputDimension        [[buffer(4)]],
        uint2 gid                             [[threadgroup_position_in_grid]],
        uint tiisg                            [[thread_index_in_simdgroup]],
        uint sgitg                            [[simdgroup_index_in_threadgroup]]
    ) {
        const uint WEIGHTS_PER_BLOCK = 128;
        const uint BYTES_PER_BLOCK = 68;  // 4 + 64
        const uint rowsPerThreadgroup = 2;
        const uint row = gid.x * rowsPerThreadgroup + sgitg;
        if (row >= outputDimension) return;

        const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
        device const uchar* rowBase = weight + row * blocksPerRow * BYTES_PER_BLOCK;

        float sum = 0.0f;

        for (uint b = tiisg; b < blocksPerRow; b += SIMD_WIDTH) {
            device const uchar* blockPtr = rowBase + b * BYTES_PER_BLOCK;

            float scale = float(*(device const half*)(blockPtr));
            float zero = float(*(device const half*)(blockPtr + 2));

            device const uchar* quants = blockPtr + 4;
            uint baseIndex = b * WEIGHTS_PER_BLOCK;

            for (uint i = 0; i < 64; i++) {
                uchar packed = quants[i];
                float w0 = scale * float(packed & 0xF) + zero;
                float w1 = scale * float(packed >> 4) + zero;
                sum += w0 * float(input[baseIndex + i * 2]);
                sum += w1 * float(input[baseIndex + i * 2 + 1]);
            }
        }

        sum = simd_sum(sum);
        if (tiisg == 0) {
            output[row] = half(sum);
        }
    }
    """

    /// Q8 group=32: 36 bytes/block (4B header + 32B int8), 32 weights
    private static let gemvQ8Group32Source = """
    kernel void gemv_q8_g32(
        device const half* input              [[buffer(0)]],
        device const uchar* weight            [[buffer(1)]],
        device half* output                   [[buffer(2)]],
        constant uint& inputDimension         [[buffer(3)]],
        constant uint& outputDimension        [[buffer(4)]],
        uint2 gid                             [[threadgroup_position_in_grid]],
        uint tiisg                            [[thread_index_in_simdgroup]],
        uint sgitg                            [[simdgroup_index_in_threadgroup]]
    ) {
        const uint WEIGHTS_PER_BLOCK = 32;
        const uint BYTES_PER_BLOCK = 36;  // 4 + 32
        const uint rowsPerThreadgroup = 2;
        const uint row = gid.x * rowsPerThreadgroup + sgitg;
        if (row >= outputDimension) return;

        const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
        device const uchar* rowBase = weight + row * blocksPerRow * BYTES_PER_BLOCK;

        float sum = 0.0f;

        for (uint b = tiisg; b < blocksPerRow; b += SIMD_WIDTH) {
            device const uchar* blockPtr = rowBase + b * BYTES_PER_BLOCK;

            float scale = float(*(device const half*)(blockPtr));
            float zero = float(*(device const half*)(blockPtr + 2));

            device const char* quants = (device const char*)(blockPtr + 4);
            uint baseIndex = b * WEIGHTS_PER_BLOCK;

            for (uint i = 0; i < 32; i++) {
                float w = scale * float(quants[i]) + zero;
                sum += w * float(input[baseIndex + i]);
            }
        }

        sum = simd_sum(sum);
        if (tiisg == 0) {
            output[row] = half(sum);
        }
    }
    """

    /// Q8 group=64: 68 bytes/block (4B header + 64B int8), 64 weights
    private static let gemvQ8Group64Source = """
    kernel void gemv_q8_g64(
        device const half* input              [[buffer(0)]],
        device const uchar* weight            [[buffer(1)]],
        device half* output                   [[buffer(2)]],
        constant uint& inputDimension         [[buffer(3)]],
        constant uint& outputDimension        [[buffer(4)]],
        uint2 gid                             [[threadgroup_position_in_grid]],
        uint tiisg                            [[thread_index_in_simdgroup]],
        uint sgitg                            [[simdgroup_index_in_threadgroup]]
    ) {
        const uint WEIGHTS_PER_BLOCK = 64;
        const uint BYTES_PER_BLOCK = 68;  // 4 + 64
        const uint rowsPerThreadgroup = 2;
        const uint row = gid.x * rowsPerThreadgroup + sgitg;
        if (row >= outputDimension) return;

        const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
        device const uchar* rowBase = weight + row * blocksPerRow * BYTES_PER_BLOCK;

        float sum = 0.0f;

        for (uint b = tiisg; b < blocksPerRow; b += SIMD_WIDTH) {
            device const uchar* blockPtr = rowBase + b * BYTES_PER_BLOCK;

            float scale = float(*(device const half*)(blockPtr));
            float zero = float(*(device const half*)(blockPtr + 2));

            device const char* quants = (device const char*)(blockPtr + 4);
            uint baseIndex = b * WEIGHTS_PER_BLOCK;

            for (uint i = 0; i < 64; i++) {
                float w = scale * float(quants[i]) + zero;
                sum += w * float(input[baseIndex + i]);
            }
        }

        sum = simd_sum(sum);
        if (tiisg == 0) {
            output[row] = half(sum);
        }
    }
    """

    // MARK: - Runtime KV Cache Quantization

    /// Quantize FP16 K/V projection output to Q8 interleaved blocks for cache storage.
    ///
    /// Called after each decode step's K/V projection to quantize before cache append.
    /// Produces interleaved blocks: [scale (f16)][zero (f16)][int8 × groupSize]
    private static let quantizeKVQ8Source = """
    /// Quantize FP16 values to Q8 interleaved blocks.
    /// Input:  FP16 array [kvHeadCount * headDimension]
    /// Output: Q8 interleaved blocks at cache position offset
    kernel void quantize_kv_q8(
        device const half* input        [[buffer(0)]],
        device uchar* output            [[buffer(1)]],
        constant uint& totalElements    [[buffer(2)]],
        constant uint& groupSize        [[buffer(3)]],
        constant uint& bytesPerBlock    [[buffer(4)]],
        uint gid                        [[thread_position_in_grid]]
    ) {
        uint blockIndex = gid;
        uint blocksTotal = totalElements / groupSize;
        if (blockIndex >= blocksTotal) return;

        uint elementStart = blockIndex * groupSize;
        device const half* groupInput = input + elementStart;

        // Compute scale and zero for this group
        float minValue = HUGE_VALF;
        float maxValue = -HUGE_VALF;
        for (uint i = 0; i < groupSize; i++) {
            float value = float(groupInput[i]);
            minValue = min(minValue, value);
            maxValue = max(maxValue, value);
        }

        float scale = (maxValue - minValue) / 255.0f;
        float zero = minValue;
        if (scale < 1e-10f) scale = 1e-10f;  // avoid division by zero

        // Write interleaved block: [scale (f16)][zero (f16)][int8 × groupSize]
        device uchar* blockOutput = output + blockIndex * bytesPerBlock;
        *(device half*)(blockOutput) = half(scale);
        *(device half*)(blockOutput + 2) = half(zero);

        device char* quantOutput = (device char*)(blockOutput + 4);
        for (uint i = 0; i < groupSize; i++) {
            float value = float(groupInput[i]);
            int quantized = int(round((value - zero) / scale));
            quantized = clamp(quantized, 0, 255);
            quantOutput[i] = char(quantized);
        }
    }
    """

    /// Dequantize Q8 interleaved blocks back to FP16 for attention computation.
    private static let dequantizeKVQ8Source = """
    /// Dequantize Q8 interleaved blocks to FP16.
    /// Input:  Q8 interleaved blocks
    /// Output: FP16 array [kvHeadCount * headDimension]
    kernel void dequantize_kv_q8(
        device const uchar* input       [[buffer(0)]],
        device half* output             [[buffer(1)]],
        constant uint& totalElements    [[buffer(2)]],
        constant uint& groupSize        [[buffer(3)]],
        constant uint& bytesPerBlock    [[buffer(4)]],
        uint gid                        [[thread_position_in_grid]]
    ) {
        uint blockIndex = gid;
        uint blocksTotal = totalElements / groupSize;
        if (blockIndex >= blocksTotal) return;

        device const uchar* blockInput = input + blockIndex * bytesPerBlock;
        float scale = float(*(device const half*)(blockInput));
        float zero = float(*(device const half*)(blockInput + 2));

        device const char* quantInput = (device const char*)(blockInput + 4);
        uint elementStart = blockIndex * groupSize;

        for (uint i = 0; i < groupSize; i++) {
            float value = scale * float(quantInput[i]) + zero;
            output[elementStart + i] = half(value);
        }
    }
    """

    // MARK: - BF16 Variant Kernels

    private static let rmsNormBF16Source = """
    /// RMS norm with BF16 weight.
    kernel void rms_norm_bf16(
        device const half* input     [[buffer(0)]],
        device const uint16_t* weight [[buffer(1)]],
        device half* output          [[buffer(2)]],
        constant uint& dimension     [[buffer(3)]],
        constant float& epsilon      [[buffer(4)]],
        uint tid                     [[thread_index_in_threadgroup]],
        uint threadgroupSize         [[threads_per_threadgroup]]
    ) {
        float sumSquared = 0.0f;
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            float value = float(input[i]);
            sumSquared += value * value;
        }
        sumSquared = simd_sum(sumSquared);

        threadgroup float shared[32];
        uint simdIndex = tid / SIMD_WIDTH;
        uint laneIndex = tid % SIMD_WIDTH;
        if (laneIndex == 0) shared[simdIndex] = sumSquared;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float total = 0.0f;
            uint simdgroupCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
            for (uint i = 0; i < simdgroupCount; i++) total += shared[i];
            shared[0] = rsqrt(total / float(dimension) + epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float scale = shared[0];

        for (uint i = tid; i < dimension; i += threadgroupSize) {
            output[i] = half(float(input[i]) * scale * bf16_to_float(weight[i]));
        }
    }
    """

    private static let fusedCopyRMSNormBF16Source = """
    kernel void fused_copy_rms_norm_bf16(
        device const half* input     [[buffer(0)]],
        device half* residual        [[buffer(1)]],
        device const uint16_t* weight [[buffer(2)]],
        device half* output          [[buffer(3)]],
        constant uint& dimension     [[buffer(4)]],
        constant float& epsilon      [[buffer(5)]],
        uint tid                     [[thread_index_in_threadgroup]],
        uint threadgroupSize         [[threads_per_threadgroup]]
    ) {
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            residual[i] = input[i];
        }
        float sumSquared = 0.0f;
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            float value = float(input[i]);
            sumSquared += value * value;
        }
        sumSquared = simd_sum(sumSquared);
        threadgroup float shared[32];
        uint simdIndex = tid / SIMD_WIDTH;
        if (tid % SIMD_WIDTH == 0) shared[simdIndex] = sumSquared;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0.0f;
            uint simdgroupCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
            for (uint i = 0; i < simdgroupCount; i++) total += shared[i];
            shared[0] = rsqrt(total / float(dimension) + epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float scale = shared[0];
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            output[i] = half(float(input[i]) * scale * bf16_to_float(weight[i]));
        }
    }
    """

    private static let fusedResidualAddCopyRMSNormBF16Source = """
    kernel void fused_residual_add_copy_rms_norm_bf16(
        device half* hidden          [[buffer(0)]],
        device half* residual        [[buffer(1)]],
        device const uint16_t* weight [[buffer(2)]],
        device half* output          [[buffer(3)]],
        constant uint& dimension     [[buffer(4)]],
        constant float& epsilon      [[buffer(5)]],
        uint tid                     [[thread_index_in_threadgroup]],
        uint threadgroupSize         [[threads_per_threadgroup]]
    ) {
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            float sum = float(hidden[i]) + float(residual[i]);
            hidden[i] = half(sum);
            residual[i] = half(sum);
        }
        threadgroup_barrier(mem_flags::mem_device);
        float sumSquared = 0.0f;
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            float value = float(hidden[i]);
            sumSquared += value * value;
        }
        sumSquared = simd_sum(sumSquared);
        threadgroup float shared[32];
        uint simdIndex = tid / SIMD_WIDTH;
        if (tid % SIMD_WIDTH == 0) shared[simdIndex] = sumSquared;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0.0f;
            uint simdgroupCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
            for (uint i = 0; i < simdgroupCount; i++) total += shared[i];
            shared[0] = rsqrt(total / float(dimension) + epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float scale = shared[0];
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            output[i] = half(float(hidden[i]) * scale * bf16_to_float(weight[i]));
        }
    }
    """

    private static let embeddingLookupBF16Source = """
    /// Embedding lookup with BF16 table.
    kernel void embedding_lookup_bf16(
        device const int* tokenID            [[buffer(0)]],
        device const uint16_t* embeddingTable [[buffer(1)]],
        device half* output                  [[buffer(2)]],
        constant uint& embeddingDimension    [[buffer(3)]],
        uint gid                             [[thread_position_in_grid]]
    ) {
        if (gid >= embeddingDimension) return;
        int token = tokenID[0];
        output[gid] = half(bf16_to_float(embeddingTable[token * embeddingDimension + gid]));
    }
    """

    // MARK: - Sequence-Aware Prefill Kernels

    /// GEMM: [seqLen × inputDim] × [outputDim × inputDim]^T → [seqLen × outputDim]
    /// Grid.x = (outputDim + 1) / 2, Grid.y = seqLen.
    /// Each threadgroup computes 2 output rows for 1 sequence position.
    /// Weight rows are shared across all seq positions → good cache reuse.
    private static let gemmSource = """
    kernel void gemm(
        device const half* input       [[buffer(0)]],
        device const half* weight      [[buffer(1)]],
        device half* output            [[buffer(2)]],
        constant uint& inputDimension  [[buffer(3)]],
        constant uint& outputDimension [[buffer(4)]],
        constant uint& sequenceLength  [[buffer(5)]],
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint tiisg                     [[thread_index_in_simdgroup]],
        uint sgitg                     [[simdgroup_index_in_threadgroup]]
    ) {
        const uint rowsPerThreadgroup = 2;
        const uint row = gid.x * rowsPerThreadgroup + sgitg;
        const uint seqPos = gid.y;
        if (row >= outputDimension || seqPos >= sequenceLength) return;

        float sum = 0.0f;
        device const half* inputRow = input + seqPos * inputDimension;
        device const half* weightRow = weight + row * inputDimension;
        for (uint j = tiisg; j < inputDimension; j += SIMD_WIDTH) {
            sum += float(weightRow[j]) * float(inputRow[j]);
        }
        sum = simd_sum(sum);
        if (tiisg == 0) {
            output[seqPos * outputDimension + row] = half(sum);
        }
    }
    """

    private static let gemmBF16Source = """
    kernel void gemm_bf16(
        device const half* input               [[buffer(0)]],
        device const uint16_t* weight          [[buffer(1)]],
        device half* output                    [[buffer(2)]],
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
        device const half* inputRow = input + seqPos * inputDimension;
        device const uint16_t* weightRow = weight + row * inputDimension;
        for (uint j = tiisg; j < inputDimension; j += SIMD_WIDTH) {
            sum += bf16_to_float(weightRow[j]) * float(inputRow[j]);
        }
        sum = simd_sum(sum);
        if (tiisg == 0) {
            output[seqPos * outputDimension + row] = half(sum);
        }
    }
    """

    private static let gemmQ4G64Source = """
    kernel void gemm_q4_g64(
        device const half* input              [[buffer(0)]],
        device const uchar* weight            [[buffer(1)]],
        device half* output                   [[buffer(2)]],
        constant uint& inputDimension         [[buffer(3)]],
        constant uint& outputDimension        [[buffer(4)]],
        constant uint& sequenceLength         [[buffer(5)]],
        uint2 gid                             [[threadgroup_position_in_grid]],
        uint tiisg                            [[thread_index_in_simdgroup]],
        uint sgitg                            [[simdgroup_index_in_threadgroup]]
    ) {
        const uint WEIGHTS_PER_BLOCK = 64;
        const uint BYTES_PER_BLOCK = 36;
        const uint rowsPerThreadgroup = 2;
        const uint row = gid.x * rowsPerThreadgroup + sgitg;
        const uint seqPos = gid.y;
        if (row >= outputDimension || seqPos >= sequenceLength) return;

        const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
        device const uchar* rowBase = weight + row * blocksPerRow * BYTES_PER_BLOCK;
        device const half* inputRow = input + seqPos * inputDimension;
        float sum = 0.0f;

        for (uint b = tiisg; b < blocksPerRow; b += SIMD_WIDTH) {
            device const uchar* blockPtr = rowBase + b * BYTES_PER_BLOCK;
            float scale = float(*(device const half*)(blockPtr));
            float zero = float(*(device const half*)(blockPtr + 2));
            device const uchar* quants = blockPtr + 4;
            uint baseIndex = b * WEIGHTS_PER_BLOCK;
            for (uint i = 0; i < 32; i++) {
                uchar packed = quants[i];
                float w0 = scale * float(packed & 0xF) + zero;
                float w1 = scale * float(packed >> 4) + zero;
                sum += w0 * float(inputRow[baseIndex + i * 2]);
                sum += w1 * float(inputRow[baseIndex + i * 2 + 1]);
            }
        }
        sum = simd_sum(sum);
        if (tiisg == 0) {
            output[seqPos * outputDimension + row] = half(sum);
        }
    }
    """

    private static let gemmQ4G128Source = """
    kernel void gemm_q4_g128(
        device const half* input              [[buffer(0)]],
        device const uchar* weight            [[buffer(1)]],
        device half* output                   [[buffer(2)]],
        constant uint& inputDimension         [[buffer(3)]],
        constant uint& outputDimension        [[buffer(4)]],
        constant uint& sequenceLength         [[buffer(5)]],
        uint2 gid                             [[threadgroup_position_in_grid]],
        uint tiisg                            [[thread_index_in_simdgroup]],
        uint sgitg                            [[simdgroup_index_in_threadgroup]]
    ) {
        const uint WEIGHTS_PER_BLOCK = 128;
        const uint BYTES_PER_BLOCK = 68;
        const uint rowsPerThreadgroup = 2;
        const uint row = gid.x * rowsPerThreadgroup + sgitg;
        const uint seqPos = gid.y;
        if (row >= outputDimension || seqPos >= sequenceLength) return;

        const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
        device const uchar* rowBase = weight + row * blocksPerRow * BYTES_PER_BLOCK;
        device const half* inputRow = input + seqPos * inputDimension;
        float sum = 0.0f;

        for (uint b = tiisg; b < blocksPerRow; b += SIMD_WIDTH) {
            device const uchar* blockPtr = rowBase + b * BYTES_PER_BLOCK;
            float scale = float(*(device const half*)(blockPtr));
            float zero = float(*(device const half*)(blockPtr + 2));
            device const uchar* quants = blockPtr + 4;
            uint baseIndex = b * WEIGHTS_PER_BLOCK;
            for (uint i = 0; i < 64; i++) {
                uchar packed = quants[i];
                float w0 = scale * float(packed & 0xF) + zero;
                float w1 = scale * float(packed >> 4) + zero;
                sum += w0 * float(inputRow[baseIndex + i * 2]);
                sum += w1 * float(inputRow[baseIndex + i * 2 + 1]);
            }
        }
        sum = simd_sum(sum);
        if (tiisg == 0) {
            output[seqPos * outputDimension + row] = half(sum);
        }
    }
    """

    /// Batched embedding: tokenIDs[seqLen] → output[seqLen × embDim]
    /// Grid: (ceil(embDim / tgSize), seqLen, 1)
    private static let embeddingLookupSeqSource = """
    kernel void embedding_lookup_seq(
        device const int* tokenIDs            [[buffer(0)]],
        device const half* embeddingTable     [[buffer(1)]],
        device half* output                   [[buffer(2)]],
        constant uint& embeddingDimension     [[buffer(3)]],
        constant uint& sequenceLength         [[buffer(4)]],
        uint2 gid                             [[thread_position_in_grid]]
    ) {
        uint dim = gid.x;
        uint seqPos = gid.y;
        if (dim >= embeddingDimension || seqPos >= sequenceLength) return;
        int token = tokenIDs[seqPos];
        output[seqPos * embeddingDimension + dim] = embeddingTable[token * embeddingDimension + dim];
    }
    """

    /// Batched embedding with BF16 table: tokenIDs[seqLen] → output[seqLen × embDim]
    private static let embeddingLookupSeqBF16Source = """
    kernel void embedding_lookup_seq_bf16(
        device const int* tokenIDs               [[buffer(0)]],
        device const uint16_t* embeddingTable    [[buffer(1)]],
        device half* output                      [[buffer(2)]],
        constant uint& embeddingDimension        [[buffer(3)]],
        constant uint& sequenceLength            [[buffer(4)]],
        uint2 gid                                [[thread_position_in_grid]]
    ) {
        uint dim = gid.x;
        uint seqPos = gid.y;
        if (dim >= embeddingDimension || seqPos >= sequenceLength) return;
        int token = tokenIDs[seqPos];
        output[seqPos * embeddingDimension + dim] = half(bf16_to_float(embeddingTable[token * embeddingDimension + dim]));
    }
    """

    /// Batched RMS norm: normalize each row of [seqLen × dim]
    /// Grid: (seqLen, 1, 1), Threadgroup: (threads, 1, 1)
    private static let rmsNormSeqSource = """
    kernel void rms_norm_seq(
        device const half* input     [[buffer(0)]],
        device const half* weight    [[buffer(1)]],
        device half* output          [[buffer(2)]],
        constant uint& dimension     [[buffer(3)]],
        constant float& epsilon      [[buffer(4)]],
        constant uint& sequenceLength [[buffer(5)]],
        uint gid_x                   [[threadgroup_position_in_grid]],
        uint tid                     [[thread_index_in_threadgroup]],
        uint threadgroupSize         [[threads_per_threadgroup]]
    ) {
        uint seqPos = gid_x;
        if (seqPos >= sequenceLength) return;

        device const half* row = input + seqPos * dimension;

        float sumSquared = 0.0f;
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            float v = float(row[i]);
            sumSquared += v * v;
        }
        sumSquared = simd_sum(sumSquared);

        threadgroup float shared[32];
        uint simdIndex = tid / SIMD_WIDTH;
        uint simdLane = tid % SIMD_WIDTH;
        if (simdLane == 0) shared[simdIndex] = sumSquared;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float total = 0.0f;
            uint simdCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
            for (uint s = 0; s < simdCount; s++) total += shared[s];
            shared[0] = rsqrt(total / float(dimension) + epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float scale = shared[0];
        device half* outRow = output + seqPos * dimension;
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            outRow[i] = half(float(row[i]) * scale * float(weight[i]));
        }
    }
    """

    /// Batched RMS norm with BF16 weight
    private static let rmsNormSeqBF16Source = """
    kernel void rms_norm_seq_bf16(
        device const half* input         [[buffer(0)]],
        device const uint16_t* weight    [[buffer(1)]],
        device half* output              [[buffer(2)]],
        constant uint& dimension         [[buffer(3)]],
        constant float& epsilon          [[buffer(4)]],
        constant uint& sequenceLength    [[buffer(5)]],
        uint gid_x                       [[threadgroup_position_in_grid]],
        uint tid                         [[thread_index_in_threadgroup]],
        uint threadgroupSize             [[threads_per_threadgroup]]
    ) {
        uint seqPos = gid_x;
        if (seqPos >= sequenceLength) return;

        device const half* row = input + seqPos * dimension;

        float sumSquared = 0.0f;
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            float v = float(row[i]);
            sumSquared += v * v;
        }
        sumSquared = simd_sum(sumSquared);

        threadgroup float shared[32];
        uint simdIndex = tid / SIMD_WIDTH;
        uint simdLane = tid % SIMD_WIDTH;
        if (simdLane == 0) shared[simdIndex] = sumSquared;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float total = 0.0f;
            uint simdCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
            for (uint s = 0; s < simdCount; s++) total += shared[s];
            shared[0] = rsqrt(total / float(dimension) + epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float scale = shared[0];
        device half* outRow = output + seqPos * dimension;
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            outRow[i] = half(float(row[i]) * scale * bf16_to_float(weight[i]));
        }
    }
    """

    /// Batched SwiGLU on [seqLen × dim]
    /// Grid: (ceil(dim / tgSize), seqLen, 1)
    private static let swigluSeqSource = """
    kernel void swiglu_seq(
        device const half* gate      [[buffer(0)]],
        device const half* up        [[buffer(1)]],
        device half* output          [[buffer(2)]],
        constant uint& dimension     [[buffer(3)]],
        constant uint& sequenceLength [[buffer(4)]],
        uint2 gid                    [[thread_position_in_grid]]
    ) {
        uint i = gid.x;
        uint seqPos = gid.y;
        if (i >= dimension || seqPos >= sequenceLength) return;

        uint idx = seqPos * dimension + i;
        float g = float(gate[idx]);
        float sigmoid = 1.0f / (1.0f + exp(-g));
        output[idx] = half(g * sigmoid * float(up[idx]));
    }
    """

    /// Batched copy: hidden[seqLen × dim] → residual[seqLen × dim]
    private static let copyBufferSeqSource = """
    kernel void copy_buffer_seq(
        device const half* source    [[buffer(0)]],
        device half* destination     [[buffer(1)]],
        constant uint& dimension     [[buffer(2)]],
        constant uint& sequenceLength [[buffer(3)]],
        uint2 gid                    [[thread_position_in_grid]]
    ) {
        uint i = gid.x;
        uint seqPos = gid.y;
        if (i >= dimension || seqPos >= sequenceLength) return;
        destination[seqPos * dimension + i] = source[seqPos * dimension + i];
    }
    """

    /// Batched residual add: hidden[seqLen × dim] += residual[seqLen × dim]
    private static let residualAddSeqSource = """
    kernel void residual_add_seq(
        device half* hidden          [[buffer(0)]],
        device const half* residual  [[buffer(1)]],
        device half* output          [[buffer(2)]],
        constant uint& dimension     [[buffer(3)]],
        constant uint& sequenceLength [[buffer(4)]],
        uint2 gid                    [[thread_position_in_grid]]
    ) {
        uint i = gid.x;
        uint seqPos = gid.y;
        if (i >= dimension || seqPos >= sequenceLength) return;
        uint idx = seqPos * dimension + i;
        output[idx] = half(float(hidden[idx]) + float(residual[idx]));
    }
    """

    /// Batched RoPE on [seqLen × heads × headDim] with positions[seqLen]
    private static let ropeSeqSource = """
    kernel void rope_seq(
        device half* Q               [[buffer(0)]],
        device half* K               [[buffer(1)]],
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
        device half* data = (head < headCount) ? Q : K;
        uint localHead = (head < headCount) ? head : (head - headCount);
        uint offset = seqPos * qkvDimension + localHead * headDimension;

        uint halfRope = ropeDimension / 2;
        for (uint i = tid; i < halfRope; i += SIMD_WIDTH) {
            float theta = float(position) / pow(base, float(2 * i) / float(ropeDimension));
            float cosTheta = cos(theta);
            float sinTheta = sin(theta);
            float x0 = float(data[offset + i]);
            float x1 = float(data[offset + i + halfRope]);
            data[offset + i] = half(x0 * cosTheta - x1 * sinTheta);
            data[offset + i + halfRope] = half(x1 * cosTheta + x0 * sinTheta);
        }
    }
    """

    /// Batched per-head QK RMS norm on [seqLen × heads × headDim] with BF16 weight
    private static let qkRMSNormSeqBF16Source = """
    kernel void qk_rms_norm_seq_bf16(
        device half* data              [[buffer(0)]],
        device const uint16_t* weight  [[buffer(1)]],
        constant uint& headCount       [[buffer(2)]],
        constant uint& headDimension   [[buffer(3)]],
        constant float& epsilon        [[buffer(4)]],
        constant uint& sequenceLength  [[buffer(5)]],
        constant uint& totalDimension  [[buffer(6)]],
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint tid                       [[thread_index_in_threadgroup]]
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
            data[offset + i] = half(float(data[offset + i]) * rms * bf16_to_float(weight[i]));
        }
    }
    """

    /// Batched per-head QK RMS norm on [seqLen × heads × headDim]
    /// Grid: (totalHeads, seqLen, 1)
    private static let qkRMSNormSeqSource = """
    kernel void qk_rms_norm_seq(
        device half* data              [[buffer(0)]],
        device const half* weight      [[buffer(1)]],
        constant uint& headCount       [[buffer(2)]],
        constant uint& headDimension   [[buffer(3)]],
        constant float& epsilon        [[buffer(4)]],
        constant uint& sequenceLength  [[buffer(5)]],
        constant uint& totalDimension  [[buffer(6)]],
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint tid                       [[thread_index_in_threadgroup]]
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
            data[offset + i] = half(float(data[offset + i]) * rms * float(weight[i]));
        }
    }
    """

    // =========================================================================
    // MARK: - Float32 Scratch Variants
    //
    // Scratch buffer is float32 to prevent Float16 overflow in models with
    // large intermediate_size (e.g., 12288). Hidden/residual stay Float16.
    // =========================================================================

    /// Embedding: FP16 table → float32 hidden output
    private static let embeddingLookupSeqF32Source = """
    kernel void embedding_lookup_seq_f32(
        device const int* tokenIDs           [[buffer(0)]],
        device const half* embeddingTable    [[buffer(1)]],
        device float* output                 [[buffer(2)]],
        constant uint& embeddingDimension    [[buffer(3)]],
        constant uint& sequenceLength        [[buffer(4)]],
        uint2 gid                            [[thread_position_in_grid]]
    ) {
        uint dim = gid.x;
        uint seqPos = gid.y;
        if (dim >= embeddingDimension || seqPos >= sequenceLength) return;
        int token = tokenIDs[seqPos];
        output[seqPos * embeddingDimension + dim] = float(embeddingTable[token * embeddingDimension + dim]);
    }
    """

    /// Embedding: BF16 table → float32 hidden output
    private static let embeddingLookupSeqBF16F32Source = """
    kernel void embedding_lookup_seq_bf16_f32(
        device const int* tokenIDs               [[buffer(0)]],
        device const uint16_t* embeddingTable    [[buffer(1)]],
        device float* output                     [[buffer(2)]],
        constant uint& embeddingDimension        [[buffer(3)]],
        constant uint& sequenceLength            [[buffer(4)]],
        uint2 gid                                [[thread_position_in_grid]]
    ) {
        uint dim = gid.x;
        uint seqPos = gid.y;
        if (dim >= embeddingDimension || seqPos >= sequenceLength) return;
        int token = tokenIDs[seqPos];
        output[seqPos * embeddingDimension + dim] = bf16_to_float(embeddingTable[token * embeddingDimension + dim]);
    }
    """

    /// RMS norm: float32 hidden in-place, half weight
    private static let rmsNormSeqF32InplaceSource = """
    kernel void rms_norm_seq_f32_inplace(
        device float* data           [[buffer(0)]],
        device const half* weight    [[buffer(1)]],
        device float* output         [[buffer(2)]],
        constant uint& dimension     [[buffer(3)]],
        constant float& epsilon      [[buffer(4)]],
        constant uint& sequenceLength [[buffer(5)]],
        uint gid_x                   [[threadgroup_position_in_grid]],
        uint tid                     [[thread_index_in_threadgroup]],
        uint threadgroupSize         [[threads_per_threadgroup]]
    ) {
        uint seqPos = gid_x;
        if (seqPos >= sequenceLength) return;
        device float* row = data + seqPos * dimension;
        float sumSquared = 0.0f;
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            sumSquared += row[i] * row[i];
        }
        sumSquared = simd_sum(sumSquared);
        threadgroup float shared[32];
        if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = sumSquared;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0.0f;
            for (uint s = 0; s < (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH; s++) total += shared[s];
            shared[0] = rsqrt(total / float(dimension) + epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float scale = shared[0];
        device float* outRow = output + seqPos * dimension;
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            outRow[i] = row[i] * scale * float(weight[i]);
        }
    }
    """

    /// RMS norm: float32 hidden in-place, BF16 weight
    private static let rmsNormSeqBF16F32InplaceSource = """
    kernel void rms_norm_seq_bf16_f32_inplace(
        device float* data               [[buffer(0)]],
        device const uint16_t* weight    [[buffer(1)]],
        device float* output             [[buffer(2)]],
        constant uint& dimension         [[buffer(3)]],
        constant float& epsilon          [[buffer(4)]],
        constant uint& sequenceLength    [[buffer(5)]],
        uint gid_x                       [[threadgroup_position_in_grid]],
        uint tid                         [[thread_index_in_threadgroup]],
        uint threadgroupSize             [[threads_per_threadgroup]]
    ) {
        uint seqPos = gid_x;
        if (seqPos >= sequenceLength) return;
        device float* row = data + seqPos * dimension;
        float sumSquared = 0.0f;
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            sumSquared += row[i] * row[i];
        }
        sumSquared = simd_sum(sumSquared);
        threadgroup float shared[32];
        if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = sumSquared;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0.0f;
            for (uint s = 0; s < (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH; s++) total += shared[s];
            shared[0] = rsqrt(total / float(dimension) + epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float scale = shared[0];
        device float* outRow = output + seqPos * dimension;
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            outRow[i] = row[i] * scale * bf16_to_float(weight[i]);
        }
    }
    """

    /// Copy: float32 → float32
    private static let copyBufferSeqF32Source = """
    kernel void copy_buffer_seq_f32(
        device const float* source   [[buffer(0)]],
        device float* destination    [[buffer(1)]],
        constant uint& dimension     [[buffer(2)]],
        constant uint& sequenceLength [[buffer(3)]],
        uint2 gid                    [[thread_position_in_grid]]
    ) {
        uint i = gid.x;
        uint seqPos = gid.y;
        if (i >= dimension || seqPos >= sequenceLength) return;
        destination[seqPos * dimension + i] = source[seqPos * dimension + i];
    }
    """

    /// Residual add: float32
    private static let residualAddSeqF32Source = """
    kernel void residual_add_seq_f32(
        device float* hidden         [[buffer(0)]],
        device const float* residual [[buffer(1)]],
        device float* output         [[buffer(2)]],
        constant uint& dimension     [[buffer(3)]],
        constant uint& sequenceLength [[buffer(4)]],
        uint2 gid                    [[thread_position_in_grid]]
    ) {
        uint i = gid.x;
        uint seqPos = gid.y;
        if (i >= dimension || seqPos >= sequenceLength) return;
        uint idx = seqPos * dimension + i;
        output[idx] = hidden[idx] + residual[idx];
    }
    """

    /// GEMM: float32 input → half logits output (for lastToken output head)
    private static let gemmBF16F32ToHalfSource = """
    kernel void gemm_bf16_f32_to_half(
        device const float* input              [[buffer(0)]],
        device const uint16_t* weight          [[buffer(1)]],
        device half* output                    [[buffer(2)]],
        constant uint& inputDimension          [[buffer(3)]],
        constant uint& outputDimension         [[buffer(4)]],
        uint2 gid                              [[threadgroup_position_in_grid]],
        uint tiisg                             [[thread_index_in_simdgroup]],
        uint sgitg                             [[simdgroup_index_in_threadgroup]]
    ) {
        const uint rowsPerThreadgroup = 2;
        const uint row = gid.x * rowsPerThreadgroup + sgitg;
        if (row >= outputDimension) return;
        float sum = 0.0f;
        device const uint16_t* weightRow = weight + row * inputDimension;
        for (uint j = tiisg; j < inputDimension; j += SIMD_WIDTH) {
            sum += bf16_to_float(weightRow[j]) * input[j];
        }
        sum = simd_sum(sum);
        if (tiisg == 0) {
            output[row] = half(sum);
        }
    }
    """

    /// GEMM: float32 scratch input → float32 scratch output, FP16 weight
    private static let gemmF32ScratchSource = """
    kernel void gemm_f32s(
        device const float* input      [[buffer(0)]],
        device const half* weight      [[buffer(1)]],
        device float* output           [[buffer(2)]],
        constant uint& inputDimension  [[buffer(3)]],
        constant uint& outputDimension [[buffer(4)]],
        constant uint& sequenceLength  [[buffer(5)]],
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint tiisg                     [[thread_index_in_simdgroup]],
        uint sgitg                     [[simdgroup_index_in_threadgroup]]
    ) {
        const uint rowsPerThreadgroup = 2;
        const uint row = gid.x * rowsPerThreadgroup + sgitg;
        const uint seqPos = gid.y;
        if (row >= outputDimension || seqPos >= sequenceLength) return;
        float sum = 0.0f;
        device const float* inputRow = input + seqPos * inputDimension;
        device const half* weightRow = weight + row * inputDimension;
        for (uint j = tiisg; j < inputDimension; j += SIMD_WIDTH) {
            sum += float(weightRow[j]) * inputRow[j];
        }
        sum = simd_sum(sum);
        if (tiisg == 0) {
            output[seqPos * outputDimension + row] = sum;
        }
    }
    """

    /// GEMM: float32 scratch input → float32 scratch output, BF16 weight
    private static let gemmBF16F32ScratchSource = """
    kernel void gemm_bf16_f32s(
        device const float* input              [[buffer(0)]],
        device const uint16_t* weight          [[buffer(1)]],
        device float* output                   [[buffer(2)]],
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
        device const float* inputRow = input + seqPos * inputDimension;
        device const uint16_t* weightRow = weight + row * inputDimension;
        for (uint j = tiisg; j < inputDimension; j += SIMD_WIDTH) {
            sum += bf16_to_float(weightRow[j]) * inputRow[j];
        }
        sum = simd_sum(sum);
        if (tiisg == 0) {
            output[seqPos * outputDimension + row] = sum;
        }
    }
    """

    /// GEMM Q4G64: float32 scratch input → float32 scratch output
    private static let gemmQ4G64F32ScratchSource = """
    kernel void gemm_q4_g64_f32s(
        device const float* input             [[buffer(0)]],
        device const uchar* weight            [[buffer(1)]],
        device float* output                  [[buffer(2)]],
        constant uint& inputDimension         [[buffer(3)]],
        constant uint& outputDimension        [[buffer(4)]],
        constant uint& sequenceLength         [[buffer(5)]],
        uint2 gid                             [[threadgroup_position_in_grid]],
        uint tiisg                            [[thread_index_in_simdgroup]],
        uint sgitg                            [[simdgroup_index_in_threadgroup]]
    ) {
        const uint WEIGHTS_PER_BLOCK = 64;
        const uint BYTES_PER_BLOCK = 36;
        const uint rowsPerThreadgroup = 2;
        const uint row = gid.x * rowsPerThreadgroup + sgitg;
        const uint seqPos = gid.y;
        if (row >= outputDimension || seqPos >= sequenceLength) return;
        const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
        device const uchar* rowBase = weight + row * blocksPerRow * BYTES_PER_BLOCK;
        device const float* inputRow = input + seqPos * inputDimension;
        float sum = 0.0f;
        for (uint b = tiisg; b < blocksPerRow; b += SIMD_WIDTH) {
            device const uchar* blockPtr = rowBase + b * BYTES_PER_BLOCK;
            float scale = float(*(device const half*)(blockPtr));
            float zero = float(*(device const half*)(blockPtr + 2));
            device const uchar* quants = blockPtr + 4;
            uint baseIndex = b * WEIGHTS_PER_BLOCK;
            for (uint i = 0; i < 32; i++) {
                uchar packed = quants[i];
                float w0 = scale * float(packed & 0xF) + zero;
                float w1 = scale * float(packed >> 4) + zero;
                sum += w0 * inputRow[baseIndex + i * 2];
                sum += w1 * inputRow[baseIndex + i * 2 + 1];
            }
        }
        sum = simd_sum(sum);
        if (tiisg == 0) { output[seqPos * outputDimension + row] = sum; }
    }
    """

    /// GEMM Q4G128: float32 scratch input → float32 scratch output
    private static let gemmQ4G128F32ScratchSource = """
    kernel void gemm_q4_g128_f32s(
        device const float* input             [[buffer(0)]],
        device const uchar* weight            [[buffer(1)]],
        device float* output                  [[buffer(2)]],
        constant uint& inputDimension         [[buffer(3)]],
        constant uint& outputDimension        [[buffer(4)]],
        constant uint& sequenceLength         [[buffer(5)]],
        uint2 gid                             [[threadgroup_position_in_grid]],
        uint tiisg                            [[thread_index_in_simdgroup]],
        uint sgitg                            [[simdgroup_index_in_threadgroup]]
    ) {
        const uint WEIGHTS_PER_BLOCK = 128;
        const uint BYTES_PER_BLOCK = 68;
        const uint rowsPerThreadgroup = 2;
        const uint row = gid.x * rowsPerThreadgroup + sgitg;
        const uint seqPos = gid.y;
        if (row >= outputDimension || seqPos >= sequenceLength) return;
        const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
        device const uchar* rowBase = weight + row * blocksPerRow * BYTES_PER_BLOCK;
        device const float* inputRow = input + seqPos * inputDimension;
        float sum = 0.0f;
        for (uint b = tiisg; b < blocksPerRow; b += SIMD_WIDTH) {
            device const uchar* blockPtr = rowBase + b * BYTES_PER_BLOCK;
            float scale = float(*(device const half*)(blockPtr));
            float zero = float(*(device const half*)(blockPtr + 2));
            device const uchar* quants = blockPtr + 4;
            uint baseIndex = b * WEIGHTS_PER_BLOCK;
            for (uint i = 0; i < 64; i++) {
                uchar packed = quants[i];
                float w0 = scale * float(packed & 0xF) + zero;
                float w1 = scale * float(packed >> 4) + zero;
                sum += w0 * inputRow[baseIndex + i * 2];
                sum += w1 * inputRow[baseIndex + i * 2 + 1];
            }
        }
        sum = simd_sum(sum);
        if (tiisg == 0) { output[seqPos * outputDimension + row] = sum; }
    }
    """

    /// GEMM: float32 scratch input → half hidden output, BF16 weight (output projections)
    private static let gemmBF16F32ScratchHalfOutSource = """
    kernel void gemm_bf16_f32s_halfout(
        device const float* input              [[buffer(0)]],
        device const uint16_t* weight          [[buffer(1)]],
        device half* output                    [[buffer(2)]],
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
        device const float* inputRow = input + seqPos * inputDimension;
        device const uint16_t* weightRow = weight + row * inputDimension;
        for (uint j = tiisg; j < inputDimension; j += SIMD_WIDTH) {
            sum += bf16_to_float(weightRow[j]) * inputRow[j];
        }
        sum = simd_sum(sum);
        if (tiisg == 0) {
            output[seqPos * outputDimension + row] = half(sum);
        }
    }
    """

    /// RMS norm: half input → float32 scratch output
    private static let rmsNormSeqF32ScratchSource = """
    kernel void rms_norm_seq_f32s(
        device const half* input     [[buffer(0)]],
        device const half* weight    [[buffer(1)]],
        device float* output         [[buffer(2)]],
        constant uint& dimension     [[buffer(3)]],
        constant float& epsilon      [[buffer(4)]],
        constant uint& sequenceLength [[buffer(5)]],
        uint gid_x                   [[threadgroup_position_in_grid]],
        uint tid                     [[thread_index_in_threadgroup]],
        uint threadgroupSize         [[threads_per_threadgroup]]
    ) {
        uint seqPos = gid_x;
        if (seqPos >= sequenceLength) return;
        device const half* row = input + seqPos * dimension;
        float sumSquared = 0.0f;
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            float v = float(row[i]);
            sumSquared += v * v;
        }
        sumSquared = simd_sum(sumSquared);
        threadgroup float shared[32];
        uint simdIndex = tid / SIMD_WIDTH;
        uint simdLane = tid % SIMD_WIDTH;
        if (simdLane == 0) shared[simdIndex] = sumSquared;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0.0f;
            uint simdCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
            for (uint s = 0; s < simdCount; s++) total += shared[s];
            shared[0] = rsqrt(total / float(dimension) + epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float scale = shared[0];
        device float* outRow = output + seqPos * dimension;
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            outRow[i] = float(row[i]) * scale * float(weight[i]);
        }
    }
    """

    /// RMS norm: half input, BF16 weight → float32 scratch output
    private static let rmsNormSeqBF16F32ScratchSource = """
    kernel void rms_norm_seq_bf16_f32s(
        device const half* input         [[buffer(0)]],
        device const uint16_t* weight    [[buffer(1)]],
        device float* output             [[buffer(2)]],
        constant uint& dimension         [[buffer(3)]],
        constant float& epsilon          [[buffer(4)]],
        constant uint& sequenceLength    [[buffer(5)]],
        uint gid_x                       [[threadgroup_position_in_grid]],
        uint tid                         [[thread_index_in_threadgroup]],
        uint threadgroupSize             [[threads_per_threadgroup]]
    ) {
        uint seqPos = gid_x;
        if (seqPos >= sequenceLength) return;
        device const half* row = input + seqPos * dimension;
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
        device float* outRow = output + seqPos * dimension;
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            outRow[i] = float(row[i]) * scale * bf16_to_float(weight[i]);
        }
    }
    """

    /// SwiGLU: float32 scratch I/O
    private static let swigluSeqF32Source = """
    kernel void swiglu_seq_f32(
        device const float* gate     [[buffer(0)]],
        device const float* up       [[buffer(1)]],
        device float* output         [[buffer(2)]],
        constant uint& dimension     [[buffer(3)]],
        constant uint& sequenceLength [[buffer(4)]],
        uint2 gid                    [[thread_position_in_grid]]
    ) {
        uint i = gid.x;
        uint seqPos = gid.y;
        if (i >= dimension || seqPos >= sequenceLength) return;
        uint idx = seqPos * dimension + i;
        float g = gate[idx];
        float sigmoid = 1.0f / (1.0f + exp(-g));
        output[idx] = g * sigmoid * up[idx];
    }
    """

    /// RoPE: float32 scratch Q/K
    private static let ropeSeqF32Source = """
    kernel void rope_seq_f32(
        device float* Q              [[buffer(0)]],
        device float* K              [[buffer(1)]],
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
        device float* data = (head < headCount) ? Q : K;
        uint localHead = (head < headCount) ? head : (head - headCount);
        uint offset = seqPos * qkvDimension + localHead * headDimension;
        uint halfRope = ropeDimension / 2;
        for (uint i = tid; i < halfRope; i += SIMD_WIDTH) {
            float theta = float(position) / pow(base, float(2 * i) / float(ropeDimension));
            float cosTheta = cos(theta);
            float sinTheta = sin(theta);
            float x0 = data[offset + i];
            float x1 = data[offset + i + halfRope];
            data[offset + i] = x0 * cosTheta - x1 * sinTheta;
            data[offset + i + halfRope] = x1 * cosTheta + x0 * sinTheta;
        }
    }
    """

    /// QK RMS norm: float32 scratch data, half weight
    private static let qkRMSNormSeqF32Source = """
    kernel void qk_rms_norm_seq_f32(
        device float* data             [[buffer(0)]],
        device const half* weight      [[buffer(1)]],
        constant uint& headCount       [[buffer(2)]],
        constant uint& headDimension   [[buffer(3)]],
        constant float& epsilon        [[buffer(4)]],
        constant uint& sequenceLength  [[buffer(5)]],
        constant uint& totalDimension  [[buffer(6)]],
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint tid                       [[thread_index_in_threadgroup]]
    ) {
        uint head = gid.x;
        uint seqPos = gid.y;
        if (head >= headCount || seqPos >= sequenceLength) return;
        uint offset = seqPos * totalDimension + head * headDimension;
        float sumSq = 0.0f;
        for (uint i = tid; i < headDimension; i += SIMD_WIDTH) {
            float v = data[offset + i];
            sumSq += v * v;
        }
        sumSq = simd_sum(sumSq);
        threadgroup float sharedRMS[1];
        if (tid == 0) { sharedRMS[0] = rsqrt(sumSq / float(headDimension) + epsilon); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float rms = sharedRMS[0];
        for (uint i = tid; i < headDimension; i += SIMD_WIDTH) {
            data[offset + i] = data[offset + i] * rms * float(weight[i]);
        }
    }
    """

    /// Conv1d: float32 scratch I/O
    private static let conv1dF32Source = """
    kernel void conv1d_f32(
        device const float* input    [[buffer(0)]],
        device const half* weight    [[buffer(1)]],
        device float* output         [[buffer(2)]],
        constant uint& dimension     [[buffer(3)]],
        constant uint& kernelSize    [[buffer(4)]],
        uint gid                     [[thread_position_in_grid]]
    ) {
        if (gid >= dimension) return;
        float sum = 0.0f;
        for (uint k = 0; k < kernelSize; k++) {
            sum += input[k * dimension + gid] * float(weight[gid * kernelSize + k]);
        }
        output[gid] = sum;
    }
    """

    /// Flash attention decode with float32 Q/K/V/output (for prefill with float32 scratch).
    /// KV cache stays in its native format (FP16 or Q8).
    /// Only the kernel function is redefined — helper functions (read_kv_element, etc.) are shared.
    private static let flashAttentionDecodeF32Source: String = {
        // Extract only the kernel function (not the helper functions that precede it)
        var src = flashAttentionDecodeSource
        // Remove helper function definitions that would cause redefinition errors
        if let kernelRange = src.range(of: "kernel void flash_attn_decode(") {
            src = String(src[kernelRange.lowerBound...])
        }
        src = src.replacingOccurrences(of: "kernel void flash_attn_decode(", with: "kernel void flash_attn_decode_f32(")
        src = src.replacingOccurrences(of: "device const half* query", with: "device const float* query")
        src = src.replacingOccurrences(of: "device const half* newKey", with: "device const float* newKey")
        src = src.replacingOccurrences(of: "device const half* newValue", with: "device const float* newValue")
        src = src.replacingOccurrences(of: "device half* output", with: "device float* output")
        src = src.replacingOccurrences(of: "float(query[", with: "(query[")
        src = src.replacingOccurrences(of: "float(newKey[", with: "(newKey[")
        src = src.replacingOccurrences(of: "float(newValue[", with: "(newValue[")
        src = src.replacingOccurrences(of: "output[queryOffset + d] = half(", with: "output[queryOffset + d] = (")
        return src
    }()
}
