extension MetalSourceGenerator {
/// Generate quantized GEMV (Q4 group 64).
public static func generateQuantizedGEMV_Q4G64(
    name: String,
    bufferPrecision: BufferPrecision
) -> String {
    let bt = bufferPrecision.metalType
    let tileElements = 256
    return """
    kernel void \(name)(
        device const \(bt)* input       [[buffer(0)]],
        device const uchar* weight     [[buffer(1)]],
        device \(bt)* output            [[buffer(2)]],
        constant uint& inputDimension  [[buffer(3)]],
        constant uint& outputDimension [[buffer(4)]],
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint tid                       [[thread_index_in_threadgroup]],
        uint tiisg                     [[thread_index_in_simdgroup]],
        uint sgitg                     [[simdgroup_index_in_threadgroup]],
        uint2 tptg                     [[threads_per_threadgroup]]
    ) {
        const uint WEIGHTS_PER_BLOCK = 64;
        const uint BYTES_PER_BLOCK = 36;
        const uint THREADS_PER_THREADGROUP = tptg.x;
        const uint rowsPerThreadgroup = THREADS_PER_THREADGROUP / SIMD_WIDTH;
        const uint TILE_ELEMENTS = \(tileElements);
        const uint row = gid.x * rowsPerThreadgroup + sgitg;
        if (row >= outputDimension) return;

        const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
        device const uchar* rowBase = weight + row * blocksPerRow * BYTES_PER_BLOCK;
        threadgroup \(bt) inputTile[TILE_ELEMENTS];
        float sum = 0.0f;

        for (uint base = 0; base < inputDimension; base += TILE_ELEMENTS) {
            const uint tileCount = min(TILE_ELEMENTS, inputDimension - base);
            for (uint j = tid; j < tileCount; j += THREADS_PER_THREADGROUP) {
                inputTile[j] = input[base + j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint blockBase = base / WEIGHTS_PER_BLOCK;
            const uint blockCount = tileCount / WEIGHTS_PER_BLOCK;
            for (uint localBlock = 0; localBlock < blockCount; localBlock++) {
                device const uchar* block = rowBase + (blockBase + localBlock) * BYTES_PER_BLOCK;
                float blockScale = float(*(device const half*)(block));
                float blockZero = float(*(device const half*)(block + 2));
                device const uchar* nibbles = block + 4;
                const uint tileOffset = localBlock * WEIGHTS_PER_BLOCK;
                for (uint i = tiisg; i < WEIGHTS_PER_BLOCK / 2; i += SIMD_WIDTH) {
                    uchar packed = nibbles[i];
                    const uint inputOffset = tileOffset + i * 2;
                float w0 = float(packed & 0x0F) * blockScale + blockZero;
                float w1 = float(packed >> 4) * blockScale + blockZero;
                    sum += w0 * float(inputTile[inputOffset]);
                    sum += w1 * float(inputTile[inputOffset + 1]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
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
    let tileElements = 256
    return """
    kernel void \(name)(
        device const \(bt)* input       [[buffer(0)]],
        device const uchar* weight     [[buffer(1)]],
        device \(bt)* output            [[buffer(2)]],
        constant uint& inputDimension  [[buffer(3)]],
        constant uint& outputDimension [[buffer(4)]],
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint tid                       [[thread_index_in_threadgroup]],
        uint tiisg                     [[thread_index_in_simdgroup]],
        uint sgitg                     [[simdgroup_index_in_threadgroup]],
        uint2 tptg                     [[threads_per_threadgroup]]
    ) {
        const uint WEIGHTS_PER_BLOCK = 128;
        const uint BYTES_PER_BLOCK = 68;
        const uint THREADS_PER_THREADGROUP = tptg.x;
        const uint rowsPerThreadgroup = THREADS_PER_THREADGROUP / SIMD_WIDTH;
        const uint TILE_ELEMENTS = \(tileElements);
        const uint row = gid.x * rowsPerThreadgroup + sgitg;
        if (row >= outputDimension) return;

        const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
        device const uchar* rowBase = weight + row * blocksPerRow * BYTES_PER_BLOCK;
        threadgroup \(bt) inputTile[TILE_ELEMENTS];
        float sum = 0.0f;

        for (uint base = 0; base < inputDimension; base += TILE_ELEMENTS) {
            const uint tileCount = min(TILE_ELEMENTS, inputDimension - base);
            for (uint j = tid; j < tileCount; j += THREADS_PER_THREADGROUP) {
                inputTile[j] = input[base + j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint blockBase = base / WEIGHTS_PER_BLOCK;
            const uint blockCount = tileCount / WEIGHTS_PER_BLOCK;
            for (uint localBlock = 0; localBlock < blockCount; localBlock++) {
                device const uchar* block = rowBase + (blockBase + localBlock) * BYTES_PER_BLOCK;
                float blockScale = float(*(device const half*)(block));
                float blockZero = float(*(device const half*)(block + 2));
                device const uchar* nibbles = block + 4;
                const uint tileOffset = localBlock * WEIGHTS_PER_BLOCK;
                for (uint i = tiisg; i < WEIGHTS_PER_BLOCK / 2; i += SIMD_WIDTH) {
                    uchar packed = nibbles[i];
                    const uint inputOffset = tileOffset + i * 2;
                float w0 = float(packed & 0x0F) * blockScale + blockZero;
                float w1 = float(packed >> 4) * blockScale + blockZero;
                    sum += w0 * float(inputTile[inputOffset]);
                    sum += w1 * float(inputTile[inputOffset + 1]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
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
        uint sgitg                     [[simdgroup_index_in_threadgroup]],
        uint2 tptg                     [[threads_per_threadgroup]]
    ) {
        const uint GROUP_SIZE = \(groupSize);
        const uint BYTES_PER_BLOCK = \(bytesPerBlock);
        const uint rowsPerThreadgroup = tptg.x / SIMD_WIDTH;
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

/// Generate quantized GEMM (Q8 group, multi-row prefill sequence).
///
/// Signature matches `generateQuantizedGEMM_Q4` exactly so dispatch builder
/// can route any Q* scheme through the same buffer-binding convention.
///
/// - Buffer 0: input  (F16 decode / F32 prefill)
/// - Buffer 1: packed weights (uchar, 36/68 bytes per block for Q8G32/Q8G64)
/// - Buffer 2: output (F16 decode / F32 prefill)
/// - Buffer 3: inputDimension   (uint32)
/// - Buffer 4: outputDimension  (uint32)
/// - Buffer 5: sequenceLength   (uint32)
/// - Buffer 6: inputRowStride   (uint32)
///
/// Block layout (per MLX Q8 affine):
/// ```
/// ┌──────────┬──────────┬──────────────────────────┐
/// │scale (2B)│ zero (2B)│ packed quants (groupSize B) │
/// └──────────┴──────────┴──────────────────────────┘
/// ```
/// Each quantized value is stored as uint8 (0..255). Dequant: `w = scale*q + zero`.
public static func generateQuantizedGEMM_Q8(
    name: String,
    bufferPrecision: BufferPrecision,
    groupSize: Int
) -> String {
    let bt = bufferPrecision.metalType
    let bytesPerBlock = 4 + groupSize  // scale(f16) + zero(f16) + uint8 × groupSize
    let tileElements = max(groupSize * 2, 256)
    return """
    kernel void \(name)(
        device const \(bt)* input       [[buffer(0)]],
        device const uchar* weight     [[buffer(1)]],
        device \(bt)* output            [[buffer(2)]],
        constant uint& inputDimension  [[buffer(3)]],
        constant uint& outputDimension [[buffer(4)]],
        constant uint& sequenceLength  [[buffer(5)]],
        constant uint& inputRowStride  [[buffer(6)]],
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint tid                       [[thread_index_in_threadgroup]],
        uint tiisg                     [[thread_index_in_simdgroup]],
        uint sgitg                     [[simdgroup_index_in_threadgroup]]
    ) {
        const uint GROUP_SIZE = \(groupSize);
        const uint BYTES_PER_BLOCK = \(bytesPerBlock);
        const uint rowsPerThreadgroup = 2;
        const uint THREADS_PER_THREADGROUP = SIMD_WIDTH * rowsPerThreadgroup;
        const uint TILE_ELEMENTS = \(tileElements);
        const uint row = gid.x * rowsPerThreadgroup + sgitg;
        const uint seqPos = gid.y;
        if (row >= outputDimension || seqPos >= sequenceLength) return;

        const uint blocksPerRow = inputDimension / GROUP_SIZE;
        device const uchar* rowBase = weight + row * blocksPerRow * BYTES_PER_BLOCK;
        device const \(bt)* inputRow = input + seqPos * inputRowStride;
        threadgroup \(bt) inputTile[TILE_ELEMENTS];
        float sum = 0.0f;

        for (uint base = 0; base < inputDimension; base += TILE_ELEMENTS) {
            const uint tileCount = min(TILE_ELEMENTS, inputDimension - base);
            for (uint j = tid; j < tileCount; j += THREADS_PER_THREADGROUP) {
                inputTile[j] = inputRow[base + j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint blockBase = base / GROUP_SIZE;
            const uint blockCount = tileCount / GROUP_SIZE;
            for (uint localBlock = 0; localBlock < blockCount; localBlock++) {
                device const uchar* block = rowBase + (blockBase + localBlock) * BYTES_PER_BLOCK;
                float blockScale = float(*(device const half*)(block));
                float blockZero = float(*(device const half*)(block + 2));
                device const uchar* quantized = block + 4;
                const uint tileOffset = localBlock * GROUP_SIZE;
                for (uint i = tiisg; i < GROUP_SIZE; i += SIMD_WIDTH) {
                    float w = blockScale * float(quantized[i]) + blockZero;
                    sum += w * float(inputTile[tileOffset + i]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        sum = simd_sum(sum);
        if (tiisg == 0) output[seqPos * outputDimension + row] = \(bt)(sum);
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
    let tileElements = max(groupSize * 2, 256)
    return """
    kernel void \(name)(
        device const \(bt)* input       [[buffer(0)]],
        device const uchar* weight     [[buffer(1)]],
        device \(bt)* output            [[buffer(2)]],
        constant uint& inputDimension  [[buffer(3)]],
        constant uint& outputDimension [[buffer(4)]],
        constant uint& sequenceLength  [[buffer(5)]],
        constant uint& inputRowStride  [[buffer(6)]],
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint tid                       [[thread_index_in_threadgroup]],
        uint tiisg                     [[thread_index_in_simdgroup]],
        uint sgitg                     [[simdgroup_index_in_threadgroup]]
    ) {
        const uint WEIGHTS_PER_BLOCK = \(weightsPerBlock);
        const uint BYTES_PER_BLOCK = \(bytesPerBlock);
        const uint rowsPerThreadgroup = 2;
        const uint THREADS_PER_THREADGROUP = SIMD_WIDTH * rowsPerThreadgroup;
        const uint TILE_ELEMENTS = \(tileElements);
        const uint row = gid.x * rowsPerThreadgroup + sgitg;
        const uint seqPos = gid.y;
        if (row >= outputDimension || seqPos >= sequenceLength) return;

        const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
        device const uchar* rowBase = weight + row * blocksPerRow * BYTES_PER_BLOCK;
        device const \(bt)* inputRow = input + seqPos * inputRowStride;
        threadgroup \(bt) inputTile[TILE_ELEMENTS];
        float sum = 0.0f;

        for (uint base = 0; base < inputDimension; base += TILE_ELEMENTS) {
            const uint tileCount = min(TILE_ELEMENTS, inputDimension - base);
            for (uint j = tid; j < tileCount; j += THREADS_PER_THREADGROUP) {
                inputTile[j] = inputRow[base + j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint blockBase = base / WEIGHTS_PER_BLOCK;
            const uint blockCount = tileCount / WEIGHTS_PER_BLOCK;
            for (uint localBlock = 0; localBlock < blockCount; localBlock++) {
                device const uchar* block = rowBase + (blockBase + localBlock) * BYTES_PER_BLOCK;
                float blockScale = float(*(device const half*)(block));
                float blockZero = float(*(device const half*)(block + 2));
                device const uchar* nibbles = block + 4;
                const uint tileOffset = localBlock * WEIGHTS_PER_BLOCK;
                for (uint i = tiisg; i < WEIGHTS_PER_BLOCK / 2; i += SIMD_WIDTH) {
                    uchar packed = nibbles[i];
                    const uint inputOffset = tileOffset + i * 2;
                float w0 = float(packed & 0x0F) * blockScale + blockZero;
                float w1 = float(packed >> 4) * blockScale + blockZero;
                    sum += w0 * float(inputTile[inputOffset]);
                    sum += w1 * float(inputTile[inputOffset + 1]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        sum = simd_sum(sum);
        if (tiisg == 0) output[seqPos * outputDimension + row] = \(bt)(sum);
    }
    """
}

// MARK: - Batched Quantized GEMM (Prefill)

/// Generate batched Q4 GEMM kernel for 2 projections sharing the same input.
/// Combines Q4 block unpacking (from generateQuantizedGEMM_Q4) with
/// multi-projection routing (from generateBatchedGEMV3).
///
/// Grid: (ceil(totalOutputDim/2), seqLen, 1)
/// Threadgroup: (SIMD_WIDTH * 2, 1, 1)
public static func generateBatchedQuantizedGEMM_Q4_2(
    name: String,
    bufferPrecision: BufferPrecision,
    groupSize: Int
) -> String {
    let bt = bufferPrecision.metalType
    let weightsPerBlock = groupSize
    let bytesPerBlock = 4 + groupSize / 2
    let tileElements = max(groupSize * 2, 256)
    return """
    kernel void \(name)(
        device const \(bt)* input       [[buffer(0)]],
        device const uchar* weight0    [[buffer(1)]],
        device const uchar* weight1    [[buffer(2)]],
        device \(bt)* output0           [[buffer(3)]],
        device \(bt)* output1           [[buffer(4)]],
        constant uint& inputDimension  [[buffer(5)]],
        constant uint& outputDim0      [[buffer(6)]],
        constant uint& outputDim1      [[buffer(7)]],
        constant uint& sequenceLength  [[buffer(8)]],
        constant uint& inputRowStride  [[buffer(9)]],
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint tid                       [[thread_index_in_threadgroup]],
        uint tiisg                     [[thread_index_in_simdgroup]],
        uint sgitg                     [[simdgroup_index_in_threadgroup]]
    ) {
        const uint WEIGHTS_PER_BLOCK = \(weightsPerBlock);
        const uint BYTES_PER_BLOCK = \(bytesPerBlock);
        const uint rowsPerThreadgroup = 2;
        const uint THREADS_PER_THREADGROUP = SIMD_WIDTH * rowsPerThreadgroup;
        const uint TILE_ELEMENTS = \(tileElements);
        const uint globalRow = gid.x * rowsPerThreadgroup + sgitg;
        const uint totalRows = outputDim0 + outputDim1;
        const uint seqPos = gid.y;
        if (globalRow >= totalRows || seqPos >= sequenceLength) return;

        device const uchar* weight;
        device \(bt)* output;
        uint localRow;
        uint outputDim;
        if (globalRow < outputDim0) {
            weight = weight0; output = output0; localRow = globalRow; outputDim = outputDim0;
        } else {
            weight = weight1; output = output1; localRow = globalRow - outputDim0; outputDim = outputDim1;
        }

        const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
        device const uchar* rowBase = weight + localRow * blocksPerRow * BYTES_PER_BLOCK;
        device const \(bt)* inputRow = input + seqPos * inputRowStride;
        threadgroup \(bt) inputTile[TILE_ELEMENTS];
        float sum = 0.0f;

        for (uint base = 0; base < inputDimension; base += TILE_ELEMENTS) {
            const uint tileCount = min(TILE_ELEMENTS, inputDimension - base);
            for (uint j = tid; j < tileCount; j += THREADS_PER_THREADGROUP) {
                inputTile[j] = inputRow[base + j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint blockBase = base / WEIGHTS_PER_BLOCK;
            const uint blockCount = tileCount / WEIGHTS_PER_BLOCK;
            for (uint localBlock = 0; localBlock < blockCount; localBlock++) {
                device const uchar* block = rowBase + (blockBase + localBlock) * BYTES_PER_BLOCK;
                float blockScale = float(*(device const half*)(block));
                float blockZero = float(*(device const half*)(block + 2));
                device const uchar* nibbles = block + 4;
                const uint tileOffset = localBlock * WEIGHTS_PER_BLOCK;
                for (uint i = tiisg; i < WEIGHTS_PER_BLOCK / 2; i += SIMD_WIDTH) {
                    uchar packed = nibbles[i];
                    const uint inputOffset = tileOffset + i * 2;
                    float w0 = float(packed & 0x0F) * blockScale + blockZero;
                    float w1 = float(packed >> 4) * blockScale + blockZero;
                    sum += w0 * float(inputTile[inputOffset]);
                    sum += w1 * float(inputTile[inputOffset + 1]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        sum = simd_sum(sum);
        if (tiisg == 0) output[seqPos * outputDim + localRow] = \(bt)(sum);
    }
    """
}

/// Generate batched Q4 GEMM kernel for 3 projections sharing the same input.
public static func generateBatchedQuantizedGEMM_Q4_3(
    name: String,
    bufferPrecision: BufferPrecision,
    groupSize: Int
) -> String {
    let bt = bufferPrecision.metalType
    let weightsPerBlock = groupSize
    let bytesPerBlock = 4 + groupSize / 2
    let tileElements = max(groupSize * 2, 256)
    return """
    kernel void \(name)(
        device const \(bt)* input       [[buffer(0)]],
        device const uchar* weight0    [[buffer(1)]],
        device const uchar* weight1    [[buffer(2)]],
        device const uchar* weight2    [[buffer(3)]],
        device \(bt)* output0           [[buffer(4)]],
        device \(bt)* output1           [[buffer(5)]],
        device \(bt)* output2           [[buffer(6)]],
        constant uint& inputDimension  [[buffer(7)]],
        constant uint& outputDim0      [[buffer(8)]],
        constant uint& outputDim1      [[buffer(9)]],
        constant uint& outputDim2      [[buffer(10)]],
        constant uint& sequenceLength  [[buffer(11)]],
        constant uint& inputRowStride  [[buffer(12)]],
        uint2 gid                      [[threadgroup_position_in_grid]],
        uint tid                       [[thread_index_in_threadgroup]],
        uint tiisg                     [[thread_index_in_simdgroup]],
        uint sgitg                     [[simdgroup_index_in_threadgroup]]
    ) {
        const uint WEIGHTS_PER_BLOCK = \(weightsPerBlock);
        const uint BYTES_PER_BLOCK = \(bytesPerBlock);
        const uint rowsPerThreadgroup = 2;
        const uint THREADS_PER_THREADGROUP = SIMD_WIDTH * rowsPerThreadgroup;
        const uint TILE_ELEMENTS = \(tileElements);
        const uint globalRow = gid.x * rowsPerThreadgroup + sgitg;
        const uint totalRows = outputDim0 + outputDim1 + outputDim2;
        const uint seqPos = gid.y;
        if (globalRow >= totalRows || seqPos >= sequenceLength) return;

        device const uchar* weight;
        device \(bt)* output;
        uint localRow;
        uint outputDim;
        if (globalRow < outputDim0) {
            weight = weight0; output = output0; localRow = globalRow; outputDim = outputDim0;
        } else if (globalRow < outputDim0 + outputDim1) {
            weight = weight1; output = output1; localRow = globalRow - outputDim0; outputDim = outputDim1;
        } else {
            weight = weight2; output = output2; localRow = globalRow - outputDim0 - outputDim1; outputDim = outputDim2;
        }

        const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
        device const uchar* rowBase = weight + localRow * blocksPerRow * BYTES_PER_BLOCK;
        device const \(bt)* inputRow = input + seqPos * inputRowStride;
        threadgroup \(bt) inputTile[TILE_ELEMENTS];
        float sum = 0.0f;

        for (uint base = 0; base < inputDimension; base += TILE_ELEMENTS) {
            const uint tileCount = min(TILE_ELEMENTS, inputDimension - base);
            for (uint j = tid; j < tileCount; j += THREADS_PER_THREADGROUP) {
                inputTile[j] = inputRow[base + j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint blockBase = base / WEIGHTS_PER_BLOCK;
            const uint blockCount = tileCount / WEIGHTS_PER_BLOCK;
            for (uint localBlock = 0; localBlock < blockCount; localBlock++) {
                device const uchar* block = rowBase + (blockBase + localBlock) * BYTES_PER_BLOCK;
                float blockScale = float(*(device const half*)(block));
                float blockZero = float(*(device const half*)(block + 2));
                device const uchar* nibbles = block + 4;
                const uint tileOffset = localBlock * WEIGHTS_PER_BLOCK;
                for (uint i = tiisg; i < WEIGHTS_PER_BLOCK / 2; i += SIMD_WIDTH) {
                    uchar packed = nibbles[i];
                    const uint inputOffset = tileOffset + i * 2;
                    float w0 = float(packed & 0x0F) * blockScale + blockZero;
                    float w1 = float(packed >> 4) * blockScale + blockZero;
                    sum += w0 * float(inputTile[inputOffset]);
                    sum += w1 * float(inputTile[inputOffset + 1]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        sum = simd_sum(sum);
        if (tiisg == 0) output[seqPos * outputDim + localRow] = \(bt)(sum);
    }
    """
}

// MARK: - Dequant Q4 → BF16

/// Dequantize Q4 weight matrix to BFloat16 for AMX matmul2d consumption.
///
/// Grid (threadgroups): (outputDimension, 1, 1) — one threadgroup per output row
/// Threadgroup: (256, 1, 1) — 256 threads process all blocks in the row
///
/// Each threadgroup unpacks an entire row of Q4 blocks into BF16 values.
/// Output layout is N-major (row = outputDim, col = inputDim), matching
/// the `tensor_inline` layout expected by MPP GEMM.
///
/// Previous design dispatched (blocksPerRow × outputDimension) threadgroups of
/// (groupSize/2) threads each, resulting in ~36K tiny threadgroups for a 1536×1536
/// matrix. This version collapses to outputDimension threadgroups of 256 threads,
/// reducing dispatch overhead by ~24x.
public static func generateDequantQ4ToBFloat(
    name: String,
    groupSize: Int
) -> String {
    let bytesPerBlock = 4 + groupSize / 2  // scale(f16) + zero(f16) + nibbles
    let nibbleBytesPerBlock = groupSize / 2
    return """
    #include <metal_stdlib>
    using namespace metal;

    kernel void \(name)(
        device const uchar* packed       [[buffer(0)]],
        device bfloat* output            [[buffer(1)]],
        constant uint& inputDimension    [[buffer(2)]],
        constant uint& outputDimension   [[buffer(3)]],
        uint tgpos [[threadgroup_position_in_grid]],
        uint tid   [[thread_index_in_threadgroup]]
    ) {
        const uint WEIGHTS_PER_BLOCK = \(groupSize);
        const uint BYTES_PER_BLOCK = \(bytesPerBlock);
        const uint NIBBLE_BYTES_PER_BLOCK = \(nibbleBytesPerBlock);
        const uint THREADS_PER_TG = 256;
        const uint row = tgpos;
        if (row >= outputDimension) return;

        const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
        const uint totalNibbleBytes = blocksPerRow * NIBBLE_BYTES_PER_BLOCK;
        device const uchar* rowBase = packed + row * blocksPerRow * BYTES_PER_BLOCK;
        device bfloat* outRow = output + row * inputDimension;

        for (uint byteIdx = tid; byteIdx < totalNibbleBytes; byteIdx += THREADS_PER_TG) {
            uint blockIdx = byteIdx / NIBBLE_BYTES_PER_BLOCK;
            uint localByte = byteIdx % NIBBLE_BYTES_PER_BLOCK;

            device const uchar* block = rowBase + blockIdx * BYTES_PER_BLOCK;
            float scale = float(*(device const half*)(block));
            float zero  = float(*(device const half*)(block + 2));
            uchar packed_byte = block[4 + localByte];

            float w0 = float(packed_byte & 0x0F) * scale + zero;
            float w1 = float(packed_byte >> 4)   * scale + zero;
            uint col = blockIdx * WEIGHTS_PER_BLOCK + localByte * 2;
            outRow[col]     = bfloat(w0);
            outRow[col + 1] = bfloat(w1);
        }
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
