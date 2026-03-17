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

    /// Buffer precision for intermediate values (hidden, scratch, residual).
    public enum BufferPrecision: Sendable {
        /// Float16 — used in decode (single token, no accumulation)
        case float16
        /// Float32 — used in prefill (multi-token, prevents accumulation error)
        case float32

        public var metalType: String {
            switch self {
            case .float16: return "half"
            case .float32: return "float"
            }
        }

        public var byteSize: Int {
            switch self {
            case .float16: return 2
            case .float32: return 4
            }
        }
    }

    /// Weight data format determines how weight bytes are read and converted to float.
    public enum WeightFormat: Sendable, Equatable {
        /// Float16 — direct read as half, convert to float
        case float16
        /// BFloat16 — read as uint16_t, shift left 16 to get float32
        case bfloat16
        /// Quantized 4-bit with group size
        case quantized4Bit(groupSize: Int)
        /// Quantized 8-bit with group size
        case quantized8Bit(groupSize: Int)

        /// MSL type for the weight buffer parameter
        public var bufferType: String {
            switch self {
            case .float16: return "half"
            case .bfloat16: return "uint16_t"
            case .quantized4Bit, .quantized8Bit: return "uchar"
            }
        }

        /// MSL expression to convert a weight value to float
        /// - Parameter expr: The MSL expression accessing the raw weight value
        public func readExpression(_ expr: String) -> String {
            switch self {
            case .float16: return "float(\(expr))"
            case .bfloat16: return "bf16_to_float(\(expr))"
            case .quantized4Bit, .quantized8Bit:
                // Quantized reading is more complex — handled by block dequantization
                return "dequantize(\(expr))"
            }
        }
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
        sources.append(generateGEMV(name: "gemv", bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateGEMV(name: "gemv_bf16", bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateEmbeddingLookup(name: "embedding_lookup", bufferPrecision: decode, weightFormat: .float16, isSequence: false))
        sources.append(generateEmbeddingLookup(name: "embedding_lookup_bf16", bufferPrecision: decode, weightFormat: .bfloat16, isSequence: false))
        sources.append(generateArgmax(name: "argmax", bufferPrecision: decode))
        sources.append(generateFusedCopyRMSNorm(name: "fused_copy_rms_norm", bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateFusedCopyRMSNorm(name: "fused_copy_rms_norm_bf16", bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateFusedResidualAddCopyRMSNorm(name: "fused_residual_add_copy_rms_norm", bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateFusedResidualAddCopyRMSNorm(name: "fused_residual_add_copy_rms_norm_bf16", bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateQKNorm(name: "qk_rms_norm", bufferPrecision: decode, weightFormat: .float16))
        sources.append(generateQKNorm(name: "qk_rms_norm_bf16", bufferPrecision: decode, weightFormat: .bfloat16))
        sources.append(generateRoPE(name: "rope", bufferPrecision: decode))
        sources.append(generateConvStateUpdate(name: "conv_state_update", bufferPrecision: decode, weightFormat: .bfloat16))

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
        sources.append(generateFlashAttentionKernel(name: "flash_attn_decode_f32", bufferPrecision: prefill))

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

    /// Generate MSL for a primitive fragment with the given precision and weight format.
    /// Returns nil if the fragment type is not recognized.
    public static func generateForFragment(
        _ fragment: any PrimitiveMetalKernelFragment,
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String? {
        let isSeq = bufferPrecision == .float32  // F32 = prefill = sequence

        switch fragment {
        case is Reduction:
            return generateReduction(name: name, dimension: 0, epsilon: 0,
                                     bufferPrecision: bufferPrecision, weightFormat: weightFormat, isSequence: isSeq)
        case is ElementwiseFragment:
            return generateSwiGLU(name: name, bufferPrecision: bufferPrecision, isSequence: isSeq)
        case is GatherFragment:
            return generateEmbeddingLookup(name: name, bufferPrecision: bufferPrecision, weightFormat: weightFormat, isSequence: isSeq)
        case is ArgmaxFragment:
            return generateArgmax(name: name, bufferPrecision: bufferPrecision)
        case is FlashAttentionFragment:
            return generateFlashAttentionKernel(name: name, bufferPrecision: bufferPrecision)
        case is RoPEFragment:
            return isSeq ? generateRoPESeq(name: name, bufferPrecision: bufferPrecision)
                         : generateRoPE(name: name, bufferPrecision: bufferPrecision)
        case is QKNormFragment:
            return isSeq ? generateQKNormSeq(name: name, bufferPrecision: bufferPrecision, weightFormat: weightFormat)
                         : generateQKNorm(name: name, bufferPrecision: bufferPrecision, weightFormat: weightFormat)
        case is Conv1dFragment:
            return isSeq ? generateConv1dCausalSeq(name: name, bufferPrecision: bufferPrecision, weightFormat: weightFormat)
                         : generateConvStateUpdate(name: name, bufferPrecision: bufferPrecision, weightFormat: weightFormat)
        case is SSMRecurrenceFragment:
            return ssmRecurrenceSource
        case is SigmoidGateFragment:
            return generateSigmoidGate(name: name, bufferPrecision: bufferPrecision)
        default:
            return nil
        }
    }

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
            // Decode: single token, no sequenceLength parameter
            return """
            kernel void \(name)(
                device const \(bt)* input       [[buffer(0)]],
                device const \(wt)* weight      [[buffer(1)]],
                device \(bt)* output            [[buffer(2)]],
                constant uint& dimension        [[buffer(3)]],
                constant float& epsilon         [[buffer(4)]],
                uint tid                        [[thread_index_in_threadgroup]],
                uint threadgroupSize            [[threads_per_threadgroup]]
            ) {
                float sumSquared = 0.0f;
                for (uint i = tid; i < dimension; i += threadgroupSize) {
                    float v = float(input[i]);
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
                    output[i] = \(bt)(float(input[i]) * scale * \(readWeight("weight[i]")));
                }
            }
            """
        }
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

    /// Generate MSL source for a GEMM kernel (prefill projection).
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
    public static func generateGEMV(
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
            uint2 gid                              [[threadgroup_position_in_grid]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]]
        ) {
            const uint rowsPerThreadgroup = 2;
            const uint row = gid.x * rowsPerThreadgroup + sgitg;
            if (row >= outputDimension) return;

            float sum = 0.0f;
            device const \(wt)* weightRow = weight + row * inputDimension;
            for (uint j = tiisg; j < inputDimension; j += SIMD_WIDTH) {
                sum += \(readWeight("weightRow[j]")) * float(input[j]);
            }
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

            if (tid == 0) {
                float total = 0.0f;
                for (uint s = 0; s < (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH; s++) total += shared[s];
                shared[0] = rsqrt(total / float(dimension) + epsilon);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                scratch[i] = \(bt)(float(hidden[i]) * scale * \(readWeight("weight[i]")));
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
            uint threadgroupSize             [[threads_per_threadgroup]]
        ) {
            // Residual add: hidden += residual, then copy hidden → residual
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(hidden[i]) + float(residual[i]);
                hidden[i] = \(bt)(h);
                residual[i] = \(bt)(h);
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

            if (tid == 0) {
                float total = 0.0f;
                for (uint s = 0; s < (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH; s++) total += shared[s];
                shared[0] = rsqrt(total / float(dimension) + epsilon);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                scratch[i] = \(bt)(float(hidden[i]) * scale * \(readWeight("weight[i]")));
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

            half Bx = half(B * x);

            for (uint k = 0; k < kernelSize - 1; k++) {
                convState[k * dimension + gid] = convState[(k + 1) * dimension + gid];
            }
            convState[(kernelSize - 1) * dimension + gid] = Bx;

            float convOut = 0.0f;
            for (uint k = 0; k < kernelSize; k++) {
                convOut += float(convState[k * dimension + gid]) * \(readWeight("weight[gid * kernelSize + k]"));
            }

            output[gid] = \(bt)(C * convOut);
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

            // --- Step 1: Append new K/V to cache ---
            const bool isWriter = (headIndex == kvHeadIndex * headCount / kvHeadCount);
            if (isWriter) {
                const uint kvIn = kvHeadIndex * headDim;

                uint kWriteByteOffset;
                if (layoutMode == 0) {
                    kWriteByteOffset = kvHeadIndex * maxSequenceLength * kHeadSlotBytes
                        + position * kHeadSlotBytes;
                } else {
                    kWriteByteOffset = position * kvHeadCount * kHeadSlotBytes
                        + kvHeadIndex * kHeadSlotBytes;
                }

                if (kQuantScheme == 0x00) {
                    for (uint d = tid; d < headDim; d += threadgroupSize) {
                        write_kv_element_fp16(keyCache + kWriteByteOffset, d, \(castIn("newKey[kvIn + d]")));
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

                if (vQuantScheme == 0x00) {
                    for (uint d = tid; d < headDim; d += threadgroupSize) {
                        write_kv_element_fp16(valueCache + vWriteByteOffset, d, \(castIn("newValue[kvIn + d]")));
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
            threadgroup_barrier(mem_flags::mem_device);

            // --- Step 2: Compute attention scores ---
            const uint queryOffset = headIndex * headDim;

            float maxScore = -HUGE_VALF;
            float sumExp = 0.0f;

            threadgroup float sharedMaxScore[1];
            threadgroup float sharedSumExp[1];
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

            // --- Step 3: Write output ---
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

    inline void write_kv_element_fp16(device uchar* cache, uint elementIndex, float value) {
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
