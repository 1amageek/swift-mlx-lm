extension MetalSourceGenerator {
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
    inline uint \(name)_mrope_axis(
        uint halfDimIndex,
        uint temporalSections,
        uint heightSections,
        uint widthSections,
        bool interleaved
    ) {
        if (temporalSections == 0 && heightSections == 0 && widthSections == 0) {
            return 0;
        }
        if (!interleaved) {
            if (halfDimIndex < temporalSections) return 0;
            if (halfDimIndex < temporalSections + heightSections) return 1;
            return 2;
        }
        if (halfDimIndex < heightSections * 3 && halfDimIndex % 3 == 1) return 1;
        if (halfDimIndex < widthSections * 3 && halfDimIndex % 3 == 2) return 2;
        return 0;
    }

    kernel void \(name)(
        device \(bt)* query                  [[buffer(0)]],
        device \(bt)* key                    [[buffer(1)]],
        device const uint* positionAxesBuffer [[buffer(2)]],
        constant uint& headCount             [[buffer(3)]],
        constant uint& kvHeadCount           [[buffer(4)]],
        constant uint& headDimension         [[buffer(5)]],
        constant uint& ropeDimension         [[buffer(6)]],
        constant float& ropeBase             [[buffer(7)]],
        constant uint& temporalSections      [[buffer(8)]],
        constant uint& heightSections        [[buffer(9)]],
        constant uint& widthSections         [[buffer(10)]],
        constant uint& mropeInterleaved      [[buffer(11)]],
        uint headIndex                       [[threadgroup_position_in_grid]],
        uint tid                             [[thread_index_in_threadgroup]]
    ) {
        const uint halfRopeDim = ropeDimension / 2;
        if (tid >= halfRopeDim) return;

        const uint axis = \(name)_mrope_axis(
            tid,
            temporalSections,
            heightSections,
            widthSections,
            mropeInterleaved != 0
        );
        const uint position = positionAxesBuffer[axis];
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
        device const uint* positionAxesBuffer [[id(2)]];
    };

    inline uint \(name)_mrope_axis(
        uint halfDimIndex,
        uint temporalSections,
        uint heightSections,
        uint widthSections,
        bool interleaved
    ) {
        if (temporalSections == 0 && heightSections == 0 && widthSections == 0) {
            return 0;
        }
        if (!interleaved) {
            if (halfDimIndex < temporalSections) return 0;
            if (halfDimIndex < temporalSections + heightSections) return 1;
            return 2;
        }
        if (halfDimIndex < heightSections * 3 && halfDimIndex % 3 == 1) return 1;
        if (halfDimIndex < widthSections * 3 && halfDimIndex % 3 == 2) return 2;
        return 0;
    }

    kernel void \(name)(
        constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
        constant uint& headCount                  [[buffer(3)]],
        constant uint& kvHeadCount                [[buffer(4)]],
        constant uint& headDimension              [[buffer(5)]],
        constant uint& ropeDimension              [[buffer(6)]],
        constant float& ropeBase                  [[buffer(7)]],
        constant uint& temporalSections           [[buffer(8)]],
        constant uint& heightSections             [[buffer(9)]],
        constant uint& widthSections              [[buffer(10)]],
        constant uint& mropeInterleaved           [[buffer(11)]],
        uint headIndex                            [[threadgroup_position_in_grid]],
        uint tid                                  [[thread_index_in_threadgroup]]
    ) {
        const uint halfRopeDim = ropeDimension / 2;
        if (tid >= halfRopeDim) return;

        const uint axis = \(name)_mrope_axis(
            tid,
            temporalSections,
            heightSections,
            widthSections,
            mropeInterleaved != 0
        );
        const uint position = args.positionAxesBuffer[axis];
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
    inline uint \(name)_mrope_axis(
        uint halfDimIndex,
        uint temporalSections,
        uint heightSections,
        uint widthSections,
        bool interleaved
    ) {
        if (temporalSections == 0 && heightSections == 0 && widthSections == 0) {
            return 0;
        }
        if (!interleaved) {
            if (halfDimIndex < temporalSections) return 0;
            if (halfDimIndex < temporalSections + heightSections) return 1;
            return 2;
        }
        if (halfDimIndex < heightSections * 3 && halfDimIndex % 3 == 1) return 1;
        if (halfDimIndex < widthSections * 3 && halfDimIndex % 3 == 2) return 2;
        return 0;
    }

    kernel void \(name)(
        device \(bt)* Q              [[buffer(0)]],
        device \(bt)* K              [[buffer(1)]],
        device const uint* positionAxesBuffer [[buffer(2)]],
        constant uint& headCount     [[buffer(3)]],
        constant uint& kvHeadCount   [[buffer(4)]],
        constant uint& headDimension [[buffer(5)]],
        constant uint& ropeDimension [[buffer(6)]],
        constant float& base         [[buffer(7)]],
        constant uint& temporalSections [[buffer(8)]],
        constant uint& heightSections [[buffer(9)]],
        constant uint& widthSections [[buffer(10)]],
        constant uint& mropeInterleaved [[buffer(11)]],
        constant uint& sequenceLength [[buffer(12)]],
        uint2 gid                    [[threadgroup_position_in_grid]],
        uint tid                     [[thread_index_in_threadgroup]]
    ) {
        uint head = gid.x;
        uint seqPos = gid.y;
        if (seqPos >= sequenceLength) return;
        uint totalHeads = headCount + kvHeadCount;
        if (head >= totalHeads) return;
        uint qkvDimension = (head < headCount) ? headCount * headDimension : kvHeadCount * headDimension;
        device \(bt)* data = (head < headCount) ? Q : K;
        uint localHead = (head < headCount) ? head : (head - headCount);
        uint offset = seqPos * qkvDimension + localHead * headDimension;
        uint halfRope = ropeDimension / 2;
        for (uint i = tid; i < halfRope; i += SIMD_WIDTH) {
            const uint axis = \(name)_mrope_axis(
                i,
                temporalSections,
                heightSections,
                widthSections,
                mropeInterleaved != 0
            );
            uint position = positionAxesBuffer[seqPos * 3 + axis];
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

}
