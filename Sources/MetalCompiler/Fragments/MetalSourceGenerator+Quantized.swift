extension MetalSourceGenerator {
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
            device const uchar* quantized = (device const uchar*)(block + 4);
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
    for (uint i = 0; i < groupSize; i++) { int q = int(round((float(groupInput[i]) - zero) / scale)); *(device uchar*)(blockOut + 4 + i) = uchar(clamp(q, 0, 255)); }
}
kernel void dequantize_kv_q8(
    device const uchar* input [[buffer(0)]], device half* output [[buffer(1)]],
    constant uint& totalElements [[buffer(2)]], constant uint& groupSize [[buffer(3)]],
    constant uint& bytesPerBlock [[buffer(4)]], uint gid [[thread_position_in_grid]]
) {
    if (gid >= totalElements / groupSize) return;
    device const uchar* block = input + gid * bytesPerBlock;
    float scale = float(*(device const half*)(block)); float zero = float(*(device const half*)(block + 2));
    for (uint i = 0; i < groupSize; i++) output[gid * groupSize + i] = half(scale * float(*(device const uchar*)(block + 4 + i)) + zero);
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

}
