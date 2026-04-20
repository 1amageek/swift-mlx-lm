extension MetalSourceGenerator {
// MARK: - QK Norm

/// Generate QK RMSNorm (per-head normalization for Q/K projections).
public static func generateQKNorm(
    name: String,
    bufferPrecision: BufferPrecision,
    weightFormat: WeightFormat,
    weightBias: Float = 0
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
        constant float& weightBias       [[buffer(5)]],
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
            float affine = \(readWeight("weight[i]")) + weightBias;
            data[offset + i] = \(bt)(float(data[offset + i]) * scale * affine);
        }
    }
    """
}

public static func generateQKNormArgumentTableVariant(
    name: String,
    argumentBufferIndex: Int,
    bufferPrecision: BufferPrecision,
    weightFormat: WeightFormat,
    weightBias: Float = 0
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
        constant float& weightBias                [[buffer(5)]],
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
            float affine = \(readWeight("args.weight[i]")) + weightBias;
            args.data[offset + i] = \(bt)(float(args.data[offset + i]) * scale * affine);
        }
    }
    """
}

/// Generate sequence-aware QK RMSNorm (prefill).
public static func generateQKNormSeq(
    name: String,
    bufferPrecision: BufferPrecision,
    weightFormat: WeightFormat,
    weightBias: Float = 0
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
        constant float& weightBias       [[buffer(5)]],
        constant uint& sequenceLength    [[buffer(6)]],
        constant uint& totalDimension    [[buffer(7)]],
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
            float affine = \(readWeight("weight[i]")) + weightBias;
            data[offset + i] = \(bt)(float(data[offset + i]) * rms * affine);
        }
    }
    """
}

public static func generatePerHeadRMSNorm(
    name: String,
    bufferPrecision: BufferPrecision,
    isSequence: Bool
) -> String {
    let bt = bufferPrecision.metalType

    if isSequence {
        return """
        kernel void \(name)(
            device \(bt)* data               [[buffer(0)]],
            constant uint& headCount         [[buffer(1)]],
            constant uint& headDimension     [[buffer(2)]],
            constant float& epsilon          [[buffer(3)]],
            constant uint& sequenceLength    [[buffer(4)]],
            constant uint& totalDimension    [[buffer(5)]],
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
                data[offset + i] = \(bt)(float(data[offset + i]) * rms);
            }
        }
        """
    }

    return """
    kernel void \(name)(
        device \(bt)* data               [[buffer(0)]],
        constant uint& headCount         [[buffer(1)]],
        constant uint& headDimension     [[buffer(2)]],
        constant float& epsilon          [[buffer(3)]],
        uint headIndex                   [[threadgroup_position_in_grid]],
        uint tid                         [[thread_index_in_threadgroup]],
        uint threadgroupSize             [[threads_per_threadgroup]]
    ) {
        if (headIndex >= headCount) return;
        uint offset = headIndex * headDimension;

        float sumSq = 0.0f;
        for (uint i = tid; i < headDimension; i += threadgroupSize) {
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
            shared[0] = rsqrt(total / float(headDimension) + epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float scale = shared[0];
        for (uint i = tid; i < headDimension; i += threadgroupSize) {
            data[offset + i] = \(bt)(float(data[offset + i]) * scale);
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
    inline uint2 \(name)_mrope_mapping(
        uint halfDimIndex,
        uint temporalSections,
        uint heightSections,
        uint widthSections,
        bool interleaved
    ) {
        if (temporalSections == 0 && heightSections == 0 && widthSections == 0) {
            return uint2(0, halfDimIndex);
        }
        if (!interleaved) {
            if (halfDimIndex < temporalSections) return uint2(0, halfDimIndex);
            if (halfDimIndex < temporalSections + heightSections) {
                return uint2(1, halfDimIndex - temporalSections);
            }
            return uint2(2, halfDimIndex - temporalSections - heightSections);
        }
        if ((halfDimIndex % 3) == 1 && halfDimIndex < heightSections * 3) {
            return uint2(1, halfDimIndex);
        }
        if ((halfDimIndex % 3) == 2 && halfDimIndex < widthSections * 3) {
            return uint2(2, halfDimIndex);
        }
        return uint2(0, halfDimIndex);
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
        constant uint& proportionalRoPE      [[buffer(12)]],
        uint headIndex                       [[threadgroup_position_in_grid]],
        uint tid                             [[thread_index_in_threadgroup]]
    ) {
        const bool useProportional = proportionalRoPE != 0;
        const uint pairCount = useProportional ? (headDimension / 2) : (ropeDimension / 2);
        const uint rotatedPairs = ropeDimension / 2;
        if (tid >= pairCount) return;

        const uint2 ropeMapping = \(name)_mrope_mapping(
            tid,
            temporalSections,
            heightSections,
            widthSections,
            mropeInterleaved != 0
        );
        const uint axis = ropeMapping.x;
        const uint position = positionAxesBuffer[axis];
        float cosTheta = 1.0f;
        float sinTheta = 0.0f;
        if (!useProportional || tid < rotatedPairs) {
            const float thetaDenominator = useProportional ? float(headDimension) : float(ropeDimension);
            const float theta = float(position) * pow(ropeBase, -2.0f * float(ropeMapping.y) / thetaDenominator);
            cosTheta = cos(theta);
            sinTheta = sin(theta);
        }

        if (headIndex < headCount) {
            uint qOffset = headIndex * headDimension + tid;
            float q0 = float(query[qOffset]);
            float q1 = float(query[qOffset + pairCount]);
            query[qOffset] = \(bt)(q0 * cosTheta - q1 * sinTheta);
            query[qOffset + pairCount] = \(bt)(q0 * sinTheta + q1 * cosTheta);
        }

        if (headIndex < kvHeadCount) {
            uint kOffset = headIndex * headDimension + tid;
            float k0 = float(key[kOffset]);
            float k1 = float(key[kOffset + pairCount]);
            key[kOffset] = \(bt)(k0 * cosTheta - k1 * sinTheta);
            key[kOffset + pairCount] = \(bt)(k0 * sinTheta + k1 * cosTheta);
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

    inline uint2 \(name)_mrope_mapping(
        uint halfDimIndex,
        uint temporalSections,
        uint heightSections,
        uint widthSections,
        bool interleaved
    ) {
        if (temporalSections == 0 && heightSections == 0 && widthSections == 0) {
            return uint2(0, halfDimIndex);
        }
        if (!interleaved) {
            if (halfDimIndex < temporalSections) return uint2(0, halfDimIndex);
            if (halfDimIndex < temporalSections + heightSections) {
                return uint2(1, halfDimIndex - temporalSections);
            }
            return uint2(2, halfDimIndex - temporalSections - heightSections);
        }
        if ((halfDimIndex % 3) == 1 && halfDimIndex < heightSections * 3) {
            return uint2(1, halfDimIndex);
        }
        if ((halfDimIndex % 3) == 2 && halfDimIndex < widthSections * 3) {
            return uint2(2, halfDimIndex);
        }
        return uint2(0, halfDimIndex);
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
        constant uint& proportionalRoPE           [[buffer(12)]],
        uint headIndex                            [[threadgroup_position_in_grid]],
        uint tid                                  [[thread_index_in_threadgroup]]
    ) {
        const bool useProportional = proportionalRoPE != 0;
        const uint pairCount = useProportional ? (headDimension / 2) : (ropeDimension / 2);
        const uint rotatedPairs = ropeDimension / 2;
        if (tid >= pairCount) return;

        const uint2 ropeMapping = \(name)_mrope_mapping(
            tid,
            temporalSections,
            heightSections,
            widthSections,
            mropeInterleaved != 0
        );
        const uint axis = ropeMapping.x;
        const uint position = args.positionAxesBuffer[axis];
        float cosTheta = 1.0f;
        float sinTheta = 0.0f;
        if (!useProportional || tid < rotatedPairs) {
            const float thetaDenominator = useProportional ? float(headDimension) : float(ropeDimension);
            const float theta = float(position) * pow(ropeBase, -2.0f * float(ropeMapping.y) / thetaDenominator);
            cosTheta = cos(theta);
            sinTheta = sin(theta);
        }

        if (headIndex < headCount) {
            uint qOffset = headIndex * headDimension + tid;
            float q0 = float(args.query[qOffset]);
            float q1 = float(args.query[qOffset + pairCount]);
            args.query[qOffset] = \(bt)(q0 * cosTheta - q1 * sinTheta);
            args.query[qOffset + pairCount] = \(bt)(q0 * sinTheta + q1 * cosTheta);
        }

        if (headIndex < kvHeadCount) {
            uint kOffset = headIndex * headDimension + tid;
            float k0 = float(args.key[kOffset]);
            float k1 = float(args.key[kOffset + pairCount]);
            args.key[kOffset] = \(bt)(k0 * cosTheta - k1 * sinTheta);
            args.key[kOffset + pairCount] = \(bt)(k0 * sinTheta + k1 * cosTheta);
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
    inline uint2 \(name)_mrope_mapping(
        uint halfDimIndex,
        uint temporalSections,
        uint heightSections,
        uint widthSections,
        bool interleaved
    ) {
        if (temporalSections == 0 && heightSections == 0 && widthSections == 0) {
            return uint2(0, halfDimIndex);
        }
        if (!interleaved) {
            if (halfDimIndex < temporalSections) return uint2(0, halfDimIndex);
            if (halfDimIndex < temporalSections + heightSections) {
                return uint2(1, halfDimIndex - temporalSections);
            }
            return uint2(2, halfDimIndex - temporalSections - heightSections);
        }
        if ((halfDimIndex % 3) == 1 && halfDimIndex < heightSections * 3) {
            return uint2(1, halfDimIndex);
        }
        if ((halfDimIndex % 3) == 2 && halfDimIndex < widthSections * 3) {
            return uint2(2, halfDimIndex);
        }
        return uint2(0, halfDimIndex);
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
        constant uint& proportionalRoPE [[buffer(13)]],
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
        const bool useProportional = proportionalRoPE != 0;
        const uint pairCount = useProportional ? (headDimension / 2) : (ropeDimension / 2);
        const uint rotatedPairs = ropeDimension / 2;
        for (uint i = tid; i < pairCount; i += SIMD_WIDTH) {
            const uint2 ropeMapping = \(name)_mrope_mapping(
                i,
                temporalSections,
                heightSections,
                widthSections,
                mropeInterleaved != 0
            );
            const uint axis = ropeMapping.x;
            uint position = positionAxesBuffer[seqPos * 3 + axis];
            float cosTheta = 1.0f;
            float sinTheta = 0.0f;
            if (!useProportional || i < rotatedPairs) {
                const float thetaDenominator = useProportional ? float(headDimension) : float(ropeDimension);
                const float theta = float(position) / pow(base, float(2 * ropeMapping.y) / thetaDenominator);
                cosTheta = cos(theta);
                sinTheta = sin(theta);
            }
            float x0 = float(data[offset + i]);
            float x1 = float(data[offset + i + pairCount]);
            data[offset + i] = \(bt)(x0 * cosTheta - x1 * sinTheta);
            data[offset + i + pairCount] = \(bt)(x1 * cosTheta + x0 * sinTheta);
        }
    }
    """
}

// MARK: - Flash Attention

/// Generate flash attention decode kernel with Clifford rotor rotation + QJL correction.
///
/// When `inlineRoPE` is true, the kernel includes inline RoPE rotation for Q and K,
/// eliminating the need for a separate RoPE dispatch and its barrier.
public static func generateFlashAttentionKernel(
    name: String,
    bufferPrecision: BufferPrecision,
    inlineRoPE: Bool = false
) -> String {
    let bt = bufferPrecision.metalType
    let castIn: (String) -> String = { expr in
        bufferPrecision == .float32 ? "(\(expr))" : "float(\(expr))"
    }
    let castOut: (String) -> String = { expr in
        bufferPrecision == .float32 ? "(\(expr))" : "\(bt)(\(expr))"
    }

    let ropeKernelParams = inlineRoPE ? """
            device const uint* ropePositionAxes  [[buffer(22)]],
            constant uint& ropeDimension         [[buffer(23)]],
            constant float& ropeBase_            [[buffer(24)]],
            constant uint& temporalSections      [[buffer(25)]],
            constant uint& heightSections        [[buffer(26)]],
            constant uint& widthSections         [[buffer(27)]],
            constant uint& mropeInterleaved      [[buffer(28)]],
        """ : ""

    let mropeAxisHelper = inlineRoPE ? """
        inline uint2 \(name)_mrope_mapping(uint h, uint tS, uint hS, uint wS, bool interleaved) {
            if (tS == 0 && hS == 0 && wS == 0) return uint2(0, h);
            if (!interleaved) {
                if (h < tS) return uint2(0, h);
                if (h < tS + hS) return uint2(1, h - tS);
                return uint2(2, h - tS - hS);
            }
            if ((h % 3) == 1 && h < hS * 3) return uint2(1, h);
            if ((h % 3) == 2 && h < wS * 3) return uint2(2, h);
            return uint2(0, h);
        }

        """ : ""

    // RoPE code block: load vector → rotBuf, apply rotation in-place
    let ropeApplyK = inlineRoPE ? """
                    // Inline RoPE: load K → rotBuf, apply rotation
                    for (uint d = tid; d < headDim; d += threadgroupSize) {
                        rotBuf[d] = \(castIn("newKey[kvIn + d]"));
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    {
                        const bool useProportional = proportionalRoPE != 0;
                        const uint pairCount = useProportional ? (headDim / 2) : (ropeDimension / 2);
                        const uint rotatedPairs = ropeDimension / 2;
                        for (uint h = tid; h < pairCount; h += threadgroupSize) {
                            uint2 ropeMapping = \(name)_mrope_mapping(h, temporalSections, heightSections, widthSections, mropeInterleaved != 0);
                            uint axis = ropeMapping.x;
                            float pos = float(ropePositionAxes[axis]);
                            float cosT = 1.0f, sinT = 0.0f;
                            if (!useProportional || h < rotatedPairs) {
                                float theta = pos * pow(
                                    ropeBase_,
                                    -2.0f * float(ropeMapping.y) / float(useProportional ? headDim : ropeDimension)
                                );
                                cosT = cos(theta);
                                sinT = sin(theta);
                            }
                            float k0 = rotBuf[h], k1 = rotBuf[h + pairCount];
                            rotBuf[h] = k0 * cosT - k1 * sinT;
                            rotBuf[h + pairCount] = k0 * sinT + k1 * cosT;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
        """ : ""

    let ropeApplyQ = inlineRoPE ? """
                // Inline RoPE: load Q → rotQuery, apply rotation
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    rotQuery[d] = \(castIn("query[queryOffset + d]"));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                {
                    const bool useProportional = proportionalRoPE != 0;
                    const uint pairCount = useProportional ? (headDim / 2) : (ropeDimension / 2);
                    const uint rotatedPairs = ropeDimension / 2;
                    for (uint h = tid; h < pairCount; h += threadgroupSize) {
                        uint2 ropeMapping = \(name)_mrope_mapping(h, temporalSections, heightSections, widthSections, mropeInterleaved != 0);
                        uint axis = ropeMapping.x;
                        float pos = float(ropePositionAxes[axis]);
                        float cosT = 1.0f, sinT = 0.0f;
                        if (!useProportional || h < rotatedPairs) {
                            float theta = pos * pow(
                                ropeBase_,
                                -2.0f * float(ropeMapping.y) / float(useProportional ? headDim : ropeDimension)
                            );
                            cosT = cos(theta);
                            sinT = sin(theta);
                        }
                        float q0 = rotQuery[h], q1 = rotQuery[h + pairCount];
                        rotQuery[h] = q0 * cosT - q1 * sinT;
                        rotQuery[h + pairCount] = q0 * sinT + q1 * cosT;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
        """ : ""

    return """
    \(mropeAxisHelper)
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
        device const half* rotorParams       [[buffer(17)]],
        device const half* qjlMatrix         [[buffer(18)]],
        device half* qjlResidualK            [[buffer(19)]],
        constant uint& numRotorGroups        [[buffer(20)]],
        constant uint& qjlDimension         [[buffer(21)]],
        constant uint& executionFlags        [[buffer(29)]],
    \(ropeKernelParams)    uint headIndex                       [[threadgroup_position_in_grid]],
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
        const bool readsExistingKV = (executionFlags & 1u) != 0;
        const bool proportionalRoPE = (executionFlags & 2u) != 0;
        const bool writesCurrentKV = !readsExistingKV && (headIndex == canonicalWriterHead);
        const bool kRotor = is_rotor_scheme(kQuantScheme);
        const bool vRotor = is_rotor_scheme(vQuantScheme);

        // Rotor params pointer for this KV head
        device const half* headRotors = rotorParams + kvHeadIndex * numRotorGroups * 4;

        // --- Step 1: Append new K/V to cache ---
        threadgroup float rotBuf[512];
        threadgroup float currentKey[512];
        threadgroup float currentValue[512];
        threadgroup float quantSMin[32], quantSMax[32];

        if (!readsExistingKV) {
        \(ropeApplyK)
        if (kRotor) {
            for (uint g = tid; g < numRotorGroups; g += threadgroupSize) {
                uint base = g * 3;
                \(inlineRoPE ? """
                float v1 = (base < headDim) ? rotBuf[base] : 0.0f;
                float v2 = (base + 1 < headDim) ? rotBuf[base + 1] : 0.0f;
                float v3 = (base + 2 < headDim) ? rotBuf[base + 2] : 0.0f;
                """ : """
                float v1 = (base < headDim) ? \(castIn("newKey[kvIn + base]")) : 0.0f;
                float v2 = (base + 1 < headDim) ? \(castIn("newKey[kvIn + base + 1]")) : 0.0f;
                float v3 = (base + 2 < headDim) ? \(castIn("newKey[kvIn + base + 2]")) : 0.0f;
                """)
                float4 R = float4(float(headRotors[g * 4]), float(headRotors[g * 4 + 1]),
                                  float(headRotors[g * 4 + 2]), float(headRotors[g * 4 + 3]));
                float3 r = rotor_sandwich(R, float3(v1, v2, v3));
                if (base < headDim) currentKey[base] = r.x;
                if (base + 1 < headDim) currentKey[base + 1] = r.y;
                if (base + 2 < headDim) currentKey[base + 2] = r.z;
            }
        \(inlineRoPE ? """
        } else {
            for (uint d = tid; d < headDim; d += threadgroupSize) {
                currentKey[d] = rotBuf[d];
            }
        """ : """
        } else {
            for (uint d = tid; d < headDim; d += threadgroupSize) {
                currentKey[d] = \(castIn("newKey[kvIn + d]"));
            }
        """
        )
        }
        if (vRotor) {
            for (uint g = tid; g < numRotorGroups; g += threadgroupSize) {
                uint base = g * 3;
                float v1 = (base < headDim) ? \(castIn("newValue[kvIn + base]")) : 0.0f;
                float v2 = (base + 1 < headDim) ? \(castIn("newValue[kvIn + base + 1]")) : 0.0f;
                float v3 = (base + 2 < headDim) ? \(castIn("newValue[kvIn + base + 2]")) : 0.0f;
                float4 R = float4(float(headRotors[g * 4]), float(headRotors[g * 4 + 1]),
                                  float(headRotors[g * 4 + 2]), float(headRotors[g * 4 + 3]));
                float3 r = rotor_sandwich(R, float3(v1, v2, v3));
                if (base < headDim) currentValue[base] = r.x;
                if (base + 1 < headDim) currentValue[base + 1] = r.y;
                if (base + 2 < headDim) currentValue[base + 2] = r.z;
            }
        } else {
            for (uint d = tid; d < headDim; d += threadgroupSize) {
                currentValue[d] = \(castIn("newValue[kvIn + d]"));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (writesCurrentKV) {
            uint kWriteByteOffset;
            if (layoutMode == 0) {
                kWriteByteOffset = kvHeadIndex * maxSequenceLength * kHeadSlotBytes
                    + position * kHeadSlotBytes;
            } else {
                kWriteByteOffset = position * kvHeadCount * kHeadSlotBytes
                    + kvHeadIndex * kHeadSlotBytes;
            }
                if (kRotor) {
                    uint kBase = rotor_base_scheme(kQuantScheme);
                    if (kBase == 0x40) {
                        write_kv_quantized_q4(currentKey, keyCache + kWriteByteOffset, headDim, tid, threadgroupSize, tiisg, sgitg, quantSMin, quantSMax);
                    } else {
                        write_kv_quantized_q8(currentKey, keyCache + kWriteByteOffset, headDim, tid, threadgroupSize, tiisg, sgitg, quantSMin, quantSMax);
                    }
                    if (qjlDimension > 0) {
                        threadgroup_barrier(mem_flags::mem_device);
                        device half* qjlOut = qjlResidualK + (position * kvHeadCount + kvHeadIndex) * qjlDimension;
                        qjl_compute_residual(currentKey, keyCache + kWriteByteOffset, kQuantScheme,
                            qjlMatrix, qjlOut, headDim, qjlDimension, kHeadSlotBytes, tid, threadgroupSize);
                    }
    \(inlineRoPE ? """
                } else if (kQuantScheme == 0x00 || kQuantScheme == 0x01 || kQuantScheme == 0x02) {
                    for (uint d = tid; d < headDim; d += threadgroupSize) {
                        write_kv_element_dense(
                            keyCache + kWriteByteOffset, d, currentKey[d], kQuantScheme);
                    }
                } else {
                    const uint groupSize = 32;
                    const uint bytesPerBlock = 36;
                    const uint numGroups = (headDim + groupSize - 1) / groupSize;
                    for (uint g = 0; g < numGroups; g++) {
                        uint groupStart = g * groupSize;
                        float localMin = HUGE_VALF, localMax = -HUGE_VALF;
                        for (uint i = tid; i < groupSize && (groupStart + i) < headDim; i += threadgroupSize) {
                            float val = currentKey[groupStart + i];
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
                        device uchar* blockOutput = keyCache + kWriteByteOffset + g * bytesPerBlock;
                        if (tid == 0) {
                            *(device half*)(blockOutput) = half(groupScale);
                            *(device half*)(blockOutput + 2) = half(groupMin);
                        }
                        threadgroup_barrier(mem_flags::mem_device);
                        for (uint i = tid; i < groupSize && (groupStart + i) < headDim; i += threadgroupSize) {
                            float val = currentKey[groupStart + i];
                            int quantized = int(round((val - groupMin) / groupScale));
                            quantized = clamp(quantized, 0, 255);
                            *(device uchar*)(blockOutput + 4 + i) = uchar(quantized);
                        }
                    }
                }
    """ : """
                } else if (kQuantScheme == 0x00 || kQuantScheme == 0x01 || kQuantScheme == 0x02) {
                    for (uint d = tid; d < headDim; d += threadgroupSize) {
                        write_kv_element_dense(
                            keyCache + kWriteByteOffset, d, currentKey[d], kQuantScheme);
                    }
                } else {
                    const uint groupSize = 32;
                    const uint bytesPerBlock = 36;
                    const uint numGroups = (headDim + groupSize - 1) / groupSize;
                    for (uint g = 0; g < numGroups; g++) {
                        uint groupStart = g * groupSize;
                        float localMin = HUGE_VALF, localMax = -HUGE_VALF;
                        for (uint i = tid; i < groupSize && (groupStart + i) < headDim; i += threadgroupSize) {
                            float val = currentKey[groupStart + i];
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
                        device uchar* blockOutput = keyCache + kWriteByteOffset + g * bytesPerBlock;
                        if (tid == 0) {
                            *(device half*)(blockOutput) = half(groupScale);
                            *(device half*)(blockOutput + 2) = half(groupMin);
                        }
                        threadgroup_barrier(mem_flags::mem_device);
                        for (uint i = tid; i < groupSize && (groupStart + i) < headDim; i += threadgroupSize) {
                            float val = currentKey[groupStart + i];
                            int quantized = int(round((val - groupMin) / groupScale));
                            quantized = clamp(quantized, 0, 255);
                            *(device uchar*)(blockOutput + 4 + i) = uchar(quantized);
                        }
                    }
                }
    """)
            threadgroup_barrier(mem_flags::mem_device);

            uint vWriteByteOffset;
            if (layoutMode == 0) {
                vWriteByteOffset = kvHeadIndex * maxSequenceLength * vHeadSlotBytes
                    + position * vHeadSlotBytes;
            } else {
                vWriteByteOffset = position * kvHeadCount * vHeadSlotBytes
                    + kvHeadIndex * vHeadSlotBytes;
            }

            if (vRotor) {
                uint vBase = rotor_base_scheme(vQuantScheme);
                if (vBase == 0x40) {
                    write_kv_quantized_q4(currentValue, valueCache + vWriteByteOffset, headDim, tid, threadgroupSize, tiisg, sgitg, quantSMin, quantSMax);
                } else {
                    write_kv_quantized_q8(currentValue, valueCache + vWriteByteOffset, headDim, tid, threadgroupSize, tiisg, sgitg, quantSMin, quantSMax);
                }
            } else if (vQuantScheme == 0x00 || vQuantScheme == 0x01 || vQuantScheme == 0x02) {
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    write_kv_element_dense(
                        valueCache + vWriteByteOffset, d, currentValue[d], vQuantScheme);
                }
            } else {
                const uint groupSize = 32;
                const uint bytesPerBlock = 36;
                const uint numGroups = (headDim + groupSize - 1) / groupSize;
                for (uint g = 0; g < numGroups; g++) {
                    uint groupStart = g * groupSize;
                    float localMin = HUGE_VALF, localMax = -HUGE_VALF;
                    for (uint i = tid; i < groupSize && (groupStart + i) < headDim; i += threadgroupSize) {
                        float val = currentValue[groupStart + i];
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
                        float val = currentValue[groupStart + i];
                        int quantized = int(round((val - groupMin) / groupScale));
                        quantized = clamp(quantized, 0, 255);
                        *(device uchar*)(blockOutput + 4 + i) = uchar(quantized);
                    }
                }
            }
        }

        // --- Step 2: Compute attention scores ---
        // \(inlineRoPE ? "Inline RoPE + optional RotorQuant on Q." : "RotorQuant K: pre-rotate Q via Clifford rotor so Q'·K' = Q·K.")
        const uint queryOffset = headIndex * headDim;

        threadgroup float rotQuery[512];
    \(ropeApplyQ)
        if (kRotor) {
            // \(inlineRoPE ? "Rotor sandwich on RoPE-rotated Q in rotQuery" : "Fused load + rotate")
            for (uint g = tid; g < numRotorGroups; g += threadgroupSize) {
                uint base = g * 3;
    \(inlineRoPE ? """
                float v1 = (base < headDim) ? rotQuery[base] : 0.0f;
                float v2 = (base + 1 < headDim) ? rotQuery[base + 1] : 0.0f;
                float v3 = (base + 2 < headDim) ? rotQuery[base + 2] : 0.0f;
    """ : """
                float v1 = (base < headDim) ? \(castIn("query[queryOffset + base]")) : 0.0f;
                float v2 = (base + 1 < headDim) ? \(castIn("query[queryOffset + base + 1]")) : 0.0f;
                float v3 = (base + 2 < headDim) ? \(castIn("query[queryOffset + base + 2]")) : 0.0f;
    """)
                float4 R = float4(float(headRotors[g * 4]), float(headRotors[g * 4 + 1]),
                                  float(headRotors[g * 4 + 2]), float(headRotors[g * 4 + 3]));
                float3 r = rotor_sandwich(R, float3(v1, v2, v3));
                if (base < headDim) rotQuery[base] = r.x;
                if (base + 1 < headDim) rotQuery[base + 1] = r.y;
                if (base + 2 < headDim) rotQuery[base + 2] = r.z;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // QJL: pre-compute Φ·Q_rotated for correction
        threadgroup float qjlQueryProj[64]; // max qjlDim
        if (kRotor && qjlDimension > 0) {
            qjl_project_query(rotQuery, qjlMatrix, qjlQueryProj, headDim, qjlDimension, tid, threadgroupSize);
        }

        float maxScore = -HUGE_VALF;
        float sumExp = 0.0f;

        threadgroup float sharedOutput[512];

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
                float q = \(inlineRoPE ? "rotQuery[d]" : "kRotor ? rotQuery[d] : \(castIn("query[queryOffset + d]"))");
                float k = (!readsExistingKV && t == position)
                    ? currentKey[d]
                    : read_kv_element(keyCache + kByteOffset, d, kQuantScheme, kHeadSlotBytes, headDim);
                score += q * k;
            }

            // QJL correction for Q·K
            if (kRotor && qjlDimension > 0 && (readsExistingKV || t != position)) {
                device const half* qjlRes = qjlResidualK + (t * kvHeadCount + kvHeadIndex) * qjlDimension;
                score += qjl_score_correction(qjlQueryProj, qjlRes, qjlDimension, tid, threadgroupSize);
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
            float expCorrection = exp(oldMax - maxScore);
            sumExp = sumExp * expCorrection + exp(score - maxScore);

            uint vByteOffset;
            if (layoutMode == 0) {
                vByteOffset = kvHeadIndex * maxSequenceLength * vHeadSlotBytes + t * vHeadSlotBytes;
            } else {
                vByteOffset = t * kvHeadCount * vHeadSlotBytes + kvHeadIndex * vHeadSlotBytes;
            }

            float weight = exp(score - maxScore);
            for (uint d = tid; d < headDim; d += threadgroupSize) {
                float v = (!readsExistingKV && t == position)
                    ? currentValue[d]
                    : read_kv_element(valueCache + vByteOffset, d, vQuantScheme, vHeadSlotBytes, headDim);
                sharedOutput[d] = sharedOutput[d] * expCorrection + weight * v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // --- Step 3: Write output ---
        // RotorQuant V: post-rotate output with inverse Clifford rotation.
        float invSum = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f;
        for (uint d = tid; d < headDim; d += threadgroupSize) {
            sharedOutput[d] *= invSum;
        }

        if (vRotor) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            rotor_apply_inverse(sharedOutput, headRotors, headDim, numRotorGroups, tid, threadgroupSize);
        }

        for (uint d = tid; d < headDim; d += threadgroupSize) {
            output[queryOffset + d] = \(castOut("sharedOutput[d]"));
        }
    }
    """
}

/// Generate flash attention argument table variant with Clifford rotor + QJL.
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
        device const half* rotorParams [[id(17)]];
        device const half* qjlMatrix [[id(18)]];
        device half* qjlResidualK [[id(19)]];
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
        constant uint& numRotorGroups             [[buffer(20)]],
        constant uint& qjlDimension               [[buffer(21)]],
        constant uint& useExistingKV              [[buffer(29)]],
        constant uint& windowLeft                 [[buffer(30)]],
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
        const bool readsExistingKV = (useExistingKV != 0);
        const bool writesCurrentKV = !readsExistingKV && (headIndex == canonicalWriterHead);
        const bool kRotor = is_rotor_scheme(kQuantScheme);
        const bool vRotor = is_rotor_scheme(vQuantScheme);

        device const half* headRotors = args.rotorParams + kvHeadIndex * numRotorGroups * 4;

        threadgroup float rotBuf[512];
        threadgroup float currentKey[512];
        threadgroup float currentValue[512];
        threadgroup float quantSMin[32], quantSMax[32];

        if (!readsExistingKV) {
        if (kRotor) {
            for (uint g = tid; g < numRotorGroups; g += threadgroupSize) {
                uint base = g * 3;
                float v1 = (base < headDim) ? \(castIn("args.newKey[kvIn + base]")) : 0.0f;
                float v2 = (base + 1 < headDim) ? \(castIn("args.newKey[kvIn + base + 1]")) : 0.0f;
                float v3 = (base + 2 < headDim) ? \(castIn("args.newKey[kvIn + base + 2]")) : 0.0f;
                float4 R = float4(float(headRotors[g * 4]), float(headRotors[g * 4 + 1]),
                                  float(headRotors[g * 4 + 2]), float(headRotors[g * 4 + 3]));
                float3 r = rotor_sandwich(R, float3(v1, v2, v3));
                if (base < headDim) currentKey[base] = r.x;
                if (base + 1 < headDim) currentKey[base + 1] = r.y;
                if (base + 2 < headDim) currentKey[base + 2] = r.z;
            }
        } else {
            for (uint d = tid; d < headDim; d += threadgroupSize) {
                currentKey[d] = \(castIn("args.newKey[kvIn + d]"));
            }
        }
        if (vRotor) {
            for (uint g = tid; g < numRotorGroups; g += threadgroupSize) {
                uint base = g * 3;
                float v1 = (base < headDim) ? \(castIn("args.newValue[kvIn + base]")) : 0.0f;
                float v2 = (base + 1 < headDim) ? \(castIn("args.newValue[kvIn + base + 1]")) : 0.0f;
                float v3 = (base + 2 < headDim) ? \(castIn("args.newValue[kvIn + base + 2]")) : 0.0f;
                float4 R = float4(float(headRotors[g * 4]), float(headRotors[g * 4 + 1]),
                                  float(headRotors[g * 4 + 2]), float(headRotors[g * 4 + 3]));
                float3 r = rotor_sandwich(R, float3(v1, v2, v3));
                if (base < headDim) currentValue[base] = r.x;
                if (base + 1 < headDim) currentValue[base + 1] = r.y;
                if (base + 2 < headDim) currentValue[base + 2] = r.z;
            }
        } else {
            for (uint d = tid; d < headDim; d += threadgroupSize) {
                currentValue[d] = \(castIn("args.newValue[kvIn + d]"));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (writesCurrentKV) {
            uint kWriteByteOffset;
            if (layoutMode == 0) {
                kWriteByteOffset = kvHeadIndex * maxSequenceLength * kHeadSlotBytes + position * kHeadSlotBytes;
            } else {
                kWriteByteOffset = position * kvHeadCount * kHeadSlotBytes + kvHeadIndex * kHeadSlotBytes;
            }

            if (kRotor) {
                uint kBase = rotor_base_scheme(kQuantScheme);
                if (kBase == 0x40) {
                    write_kv_quantized_q4(currentKey, args.keyCache + kWriteByteOffset, headDim, tid, threadgroupSize, tiisg, sgitg, quantSMin, quantSMax);
                } else {
                    write_kv_quantized_q8(currentKey, args.keyCache + kWriteByteOffset, headDim, tid, threadgroupSize, tiisg, sgitg, quantSMin, quantSMax);
                }
                if (qjlDimension > 0) {
                    threadgroup_barrier(mem_flags::mem_device);
                    device half* qjlOut = args.qjlResidualK + (position * kvHeadCount + kvHeadIndex) * qjlDimension;
                    qjl_compute_residual(currentKey, args.keyCache + kWriteByteOffset, kQuantScheme,
                        args.qjlMatrix, qjlOut, headDim, qjlDimension, kHeadSlotBytes, tid, threadgroupSize);
                }
            } else if (kQuantScheme == 0x00 || kQuantScheme == 0x01 || kQuantScheme == 0x02) {
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    write_kv_element_dense(args.keyCache + kWriteByteOffset, d, currentKey[d], kQuantScheme);
                }
            } else {
                const uint groupSize = 32;
                const uint bytesPerBlock = 36;
                const uint numGroups = (headDim + groupSize - 1) / groupSize;
                for (uint g = 0; g < numGroups; g++) {
                    uint groupStart = g * groupSize;
                    float localMin = HUGE_VALF, localMax = -HUGE_VALF;
                    for (uint i = tid; i < groupSize && (groupStart + i) < headDim; i += threadgroupSize) {
                        float val = currentKey[groupStart + i];
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
                    device uchar* blockOutput = args.keyCache + kWriteByteOffset + g * bytesPerBlock;
                    if (tid == 0) {
                        *(device half*)(blockOutput) = half(groupScale);
                        *(device half*)(blockOutput + 2) = half(groupMin);
                    }
                    threadgroup_barrier(mem_flags::mem_device);
                    for (uint i = tid; i < groupSize && (groupStart + i) < headDim; i += threadgroupSize) {
                        float val = currentKey[groupStart + i];
                        int quantized = int(round((val - groupMin) / groupScale));
                        quantized = clamp(quantized, 0, 255);
                        *(device uchar*)(blockOutput + 4 + i) = uchar(quantized);
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

            if (vRotor) {
                uint vBase = rotor_base_scheme(vQuantScheme);
                if (vBase == 0x40) {
                    write_kv_quantized_q4(currentValue, args.valueCache + vWriteByteOffset, headDim, tid, threadgroupSize, tiisg, sgitg, quantSMin, quantSMax);
                } else {
                    write_kv_quantized_q8(currentValue, args.valueCache + vWriteByteOffset, headDim, tid, threadgroupSize, tiisg, sgitg, quantSMin, quantSMax);
                }
            } else if (vQuantScheme == 0x00 || vQuantScheme == 0x01 || vQuantScheme == 0x02) {
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    write_kv_element_dense(args.valueCache + vWriteByteOffset, d, currentValue[d], vQuantScheme);
                }
            } else {
                const uint groupSize = 32;
                const uint bytesPerBlock = 36;
                const uint numGroups = (headDim + groupSize - 1) / groupSize;
                for (uint g = 0; g < numGroups; g++) {
                    uint groupStart = g * groupSize;
                    float localMin = HUGE_VALF, localMax = -HUGE_VALF;
                    for (uint i = tid; i < groupSize && (groupStart + i) < headDim; i += threadgroupSize) {
                        float val = currentValue[groupStart + i];
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
                        float val = currentValue[groupStart + i];
                        int quantized = int(round((val - groupMin) / groupScale));
                        quantized = clamp(quantized, 0, 255);
                        *(device uchar*)(blockOutput + 4 + i) = uchar(quantized);
                    }
                }
            }
        }

        const uint queryOffset = headIndex * headDim;

        threadgroup float rotQuery[512];
        if (kRotor) {
            // Fused load + rotate
            for (uint g = tid; g < numRotorGroups; g += threadgroupSize) {
                uint base = g * 3;
                float v1 = (base < headDim) ? \(castIn("args.query[queryOffset + base]")) : 0.0f;
                float v2 = (base + 1 < headDim) ? \(castIn("args.query[queryOffset + base + 1]")) : 0.0f;
                float v3 = (base + 2 < headDim) ? \(castIn("args.query[queryOffset + base + 2]")) : 0.0f;
                float4 R = float4(float(headRotors[g * 4]), float(headRotors[g * 4 + 1]),
                                  float(headRotors[g * 4 + 2]), float(headRotors[g * 4 + 3]));
                float3 r = rotor_sandwich(R, float3(v1, v2, v3));
                if (base < headDim) rotQuery[base] = r.x;
                if (base + 1 < headDim) rotQuery[base + 1] = r.y;
                if (base + 2 < headDim) rotQuery[base + 2] = r.z;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        threadgroup float qjlQueryProj[64];
        if (kRotor && qjlDimension > 0) {
            qjl_project_query(rotQuery, args.qjlMatrix, qjlQueryProj, headDim, qjlDimension, tid, threadgroupSize);
        }

        float maxScore = -HUGE_VALF;
        float sumExp = 0.0f;

        threadgroup float sharedOutput[512];
        for (uint d = tid; d < headDim; d += threadgroupSize) {
            sharedOutput[d] = 0.0f;
        }

        const uint attentionStart = (windowLeft == 0xFFFFFFFFu)
            ? 0u
            : ((position + 1 > windowLeft) ? (position - windowLeft + 1u) : 0u);

        for (uint t = attentionStart; t < sequenceLength; t++) {
            uint kByteOffset;
            if (layoutMode == 0) {
                kByteOffset = kvHeadIndex * maxSequenceLength * kHeadSlotBytes + t * kHeadSlotBytes;
            } else {
                kByteOffset = t * kvHeadCount * kHeadSlotBytes + kvHeadIndex * kHeadSlotBytes;
            }

            float score = 0.0f;
            for (uint d = tid; d < headDim; d += threadgroupSize) {
                float q = kRotor ? rotQuery[d] : \(castIn("args.query[queryOffset + d]"));
                float k = (!readsExistingKV && t == position)
                    ? currentKey[d]
                    : read_kv_element(args.keyCache + kByteOffset, d, kQuantScheme, kHeadSlotBytes, headDim);
                score += q * k;
            }

            if (kRotor && qjlDimension > 0 && (readsExistingKV || t != position)) {
                device const half* qjlRes = args.qjlResidualK + (t * kvHeadCount + kvHeadIndex) * qjlDimension;
                score += qjl_score_correction(qjlQueryProj, qjlRes, qjlDimension, tid, threadgroupSize);
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
                float v = (!readsExistingKV && t == position)
                    ? currentValue[d]
                    : read_kv_element(args.valueCache + vByteOffset, d, vQuantScheme, vHeadSlotBytes, headDim);
                sharedOutput[d] = sharedOutput[d] * expScale + expScore * v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float invSum = 1.0f / max(sumExp, 1e-20f);
        const uint outputOffset = headIndex * headDim;
        for (uint d = tid; d < headDim; d += threadgroupSize) {
            sharedOutput[d] *= invSum;
        }

        if (vRotor) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            rotor_apply_inverse(sharedOutput, headRotors, headDim, numRotorGroups, tid, threadgroupSize);
        }

        for (uint d = tid; d < headDim; d += threadgroupSize) {
            args.output[outputOffset + d] = \(castOut("sharedOutput[d]"));
        }
    }
    """
}

/// Generate KV cache fill kernel for prefill with Clifford rotor rotation + QJL.
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
        device const half* rotorParams         [[buffer(13)]],
        device const half* qjlMatrix           [[buffer(14)]],
        device half* qjlResidualK              [[buffer(15)]],
        constant uint& numRotorGroups          [[buffer(16)]],
        constant uint& qjlDimension            [[buffer(17)]],
        uint groupId                            [[threadgroup_position_in_grid]],
        uint tid                               [[thread_index_in_threadgroup]],
        uint tiisg                             [[thread_index_in_simdgroup]],
        uint sgitg                             [[simdgroup_index_in_threadgroup]],
        uint threadgroupSize                   [[threads_per_threadgroup]]
    ) {
        uint pos = groupId;
        if (pos >= sequenceLength) return;
        const uint headDim = headDimension;
        const bool kRotor = is_rotor_scheme(kQuantScheme);
        const bool vRotor = is_rotor_scheme(vQuantScheme);

        if (!kRotor && !vRotor) {
            for (uint kvHead = 0; kvHead < kvHeadCount; kvHead++) {
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    uint inputIdx = pos * kvHeadCount * headDim + kvHead * headDim + d;
                    float kVal = float(newKeys[inputIdx]);
                    float vVal = float(newValues[inputIdx]);
                    uint kByteOffset, vByteOffset;
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
            return;
        }

        // RotorQuant path: load full head → Clifford rotor rotate → quantize
        threadgroup float rotatedK[512];
        threadgroup float rotatedV[512];
        threadgroup float quantSMin[32], quantSMax[32];

        for (uint kvHead = 0; kvHead < kvHeadCount; kvHead++) {
            uint baseIdx = pos * kvHeadCount * headDim + kvHead * headDim;
            device const half* headRotors = rotorParams + kvHead * numRotorGroups * 4;

            // Fused load + rotate for K and V
            if (kRotor) {
                for (uint g = tid; g < numRotorGroups; g += threadgroupSize) {
                    uint base = g * 3;
                    float v1 = (base < headDim) ? float(newKeys[baseIdx + base]) : 0.0f;
                    float v2 = (base + 1 < headDim) ? float(newKeys[baseIdx + base + 1]) : 0.0f;
                    float v3 = (base + 2 < headDim) ? float(newKeys[baseIdx + base + 2]) : 0.0f;
                    float4 R = float4(float(headRotors[g * 4]), float(headRotors[g * 4 + 1]),
                                      float(headRotors[g * 4 + 2]), float(headRotors[g * 4 + 3]));
                    float3 r = rotor_sandwich(R, float3(v1, v2, v3));
                    if (base < headDim) rotatedK[base] = r.x;
                    if (base + 1 < headDim) rotatedK[base + 1] = r.y;
                    if (base + 2 < headDim) rotatedK[base + 2] = r.z;
                }
            } else {
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    rotatedK[d] = float(newKeys[baseIdx + d]);
                }
            }
            if (vRotor) {
                for (uint g = tid; g < numRotorGroups; g += threadgroupSize) {
                    uint base = g * 3;
                    float v1 = (base < headDim) ? float(newValues[baseIdx + base]) : 0.0f;
                    float v2 = (base + 1 < headDim) ? float(newValues[baseIdx + base + 1]) : 0.0f;
                    float v3 = (base + 2 < headDim) ? float(newValues[baseIdx + base + 2]) : 0.0f;
                    float4 R = float4(float(headRotors[g * 4]), float(headRotors[g * 4 + 1]),
                                      float(headRotors[g * 4 + 2]), float(headRotors[g * 4 + 3]));
                    float3 r = rotor_sandwich(R, float3(v1, v2, v3));
                    if (base < headDim) rotatedV[base] = r.x;
                    if (base + 1 < headDim) rotatedV[base + 1] = r.y;
                    if (base + 2 < headDim) rotatedV[base + 2] = r.z;
                }
            } else {
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    rotatedV[d] = float(newValues[baseIdx + d]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint kByteOffset, vByteOffset;
            if (layoutMode == 0) {
                kByteOffset = kvHead * maxSequenceLength * kHeadSlotBytes + pos * kHeadSlotBytes;
                vByteOffset = kvHead * maxSequenceLength * vHeadSlotBytes + pos * vHeadSlotBytes;
            } else {
                kByteOffset = pos * kvHeadCount * kHeadSlotBytes + kvHead * kHeadSlotBytes;
                vByteOffset = pos * kvHeadCount * vHeadSlotBytes + kvHead * vHeadSlotBytes;
            }

            if (kRotor) {
                uint kBase = rotor_base_scheme(kQuantScheme);
                if (kBase == 0x40) {
                    write_kv_quantized_q4(rotatedK, keyCache + kByteOffset, headDim, tid, threadgroupSize, tiisg, sgitg, quantSMin, quantSMax);
                } else {
                    write_kv_quantized_q8(rotatedK, keyCache + kByteOffset, headDim, tid, threadgroupSize, tiisg, sgitg, quantSMin, quantSMax);
                }
                // QJL residual for K
                if (qjlDimension > 0) {
                    threadgroup_barrier(mem_flags::mem_device);
                    device half* qjlOut = qjlResidualK + (pos * kvHeadCount + kvHead) * qjlDimension;
                    qjl_compute_residual(rotatedK, keyCache + kByteOffset, kQuantScheme,
                        qjlMatrix, qjlOut, headDim, qjlDimension, kHeadSlotBytes, tid, threadgroupSize);
                }
            } else {
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    write_kv_element_dense(keyCache + kByteOffset, d, rotatedK[d], kQuantScheme);
                }
            }
            threadgroup_barrier(mem_flags::mem_device);

            if (vRotor) {
                uint vBase = rotor_base_scheme(vQuantScheme);
                if (vBase == 0x40) {
                    write_kv_quantized_q4(rotatedV, valueCache + vByteOffset, headDim, tid, threadgroupSize, tiisg, sgitg, quantSMin, quantSMax);
                } else {
                    write_kv_quantized_q8(rotatedV, valueCache + vByteOffset, headDim, tid, threadgroupSize, tiisg, sgitg, quantSMin, quantSMax);
                }
            } else {
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    write_kv_element_dense(valueCache + vByteOffset, d, rotatedV[d], vQuantScheme);
                }
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
    }
    """
}

/// Generate batch causal flash attention kernel for prefill with Clifford rotor + QJL.
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
        device const half* rotorParams        [[buffer(15)]],
        device const half* qjlMatrix          [[buffer(16)]],
        device const half* qjlResidualK       [[buffer(17)]],
        constant uint& numRotorGroups         [[buffer(18)]],
        constant uint& qjlDimension           [[buffer(19)]],
        constant uint& causal                 [[buffer(20)]],
        constant uint& windowLeft             [[buffer(21)]],
        constant uint& windowRight            [[buffer(22)]],
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
        const bool kRotor = is_rotor_scheme(kQuantScheme);
        const bool vRotor = is_rotor_scheme(vQuantScheme);

        device const half* headRotors = rotorParams + kvHeadIndex * numRotorGroups * 4;

        threadgroup float rotQuery[512];
        if (kRotor) {
            // Fused load + rotate
            for (uint g = tid; g < numRotorGroups; g += threadgroupSize) {
                uint base = g * 3;
                float v1 = (base < headDim) ? float(query[queryOffset + base]) : 0.0f;
                float v2 = (base + 1 < headDim) ? float(query[queryOffset + base + 1]) : 0.0f;
                float v3 = (base + 2 < headDim) ? float(query[queryOffset + base + 2]) : 0.0f;
                float4 R = float4(float(headRotors[g * 4]), float(headRotors[g * 4 + 1]),
                                  float(headRotors[g * 4 + 2]), float(headRotors[g * 4 + 3]));
                float3 r = rotor_sandwich(R, float3(v1, v2, v3));
                if (base < headDim) rotQuery[base] = r.x;
                if (base + 1 < headDim) rotQuery[base + 1] = r.y;
                if (base + 2 < headDim) rotQuery[base + 2] = r.z;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        threadgroup float qjlQueryProj[64];
        if (kRotor && qjlDimension > 0) {
            qjl_project_query(rotQuery, qjlMatrix, qjlQueryProj, headDim, qjlDimension, tid, threadgroupSize);
        }

        float maxScore = -HUGE_VALF;
        float sumExp = 0.0f;

        threadgroup float sharedOutput[512];
        for (uint d = tid; d < headDim; d += threadgroupSize) {
            sharedOutput[d] = 0.0f;
        }

        const bool isCausal = (causal != 0);
        const uint leftReach = (windowLeft == 0xFFFFFFFFu || windowLeft == 0u)
            ? posId
            : min(posId, windowLeft - 1u);
        const uint attentionStart = posId - leftReach;
        const uint attentionEnd = isCausal
            ? posId
            : (
                (windowRight == 0xFFFFFFFFu)
                    ? (sequenceLength - 1u)
                    : min(sequenceLength - 1u, posId + (windowRight == 0u ? 0u : windowRight - 1u))
            );

        for (uint t = attentionStart; t <= attentionEnd; t++) {
            uint kByteOffset;
            if (layoutMode == 0) {
                kByteOffset = kvHeadIndex * maxSequenceLength * kHeadSlotBytes + t * kHeadSlotBytes;
            } else {
                kByteOffset = t * kvHeadCount * kHeadSlotBytes + kvHeadIndex * kHeadSlotBytes;
            }

            float score = 0.0f;
            for (uint d = tid; d < headDim; d += threadgroupSize) {
                float q = kRotor ? rotQuery[d] : float(query[queryOffset + d]);
                float k = read_kv_element(keyCache + kByteOffset, d, kQuantScheme, kHeadSlotBytes, headDim);
                score += q * k;
            }

            if (kRotor && qjlDimension > 0) {
                device const half* qjlRes = qjlResidualK + (t * kvHeadCount + kvHeadIndex) * qjlDimension;
                score += qjl_score_correction(qjlQueryProj, qjlRes, qjlDimension, tid, threadgroupSize);
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
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float invSum = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f;
        for (uint d = tid; d < headDim; d += threadgroupSize) {
            sharedOutput[d] *= invSum;
        }

        if (vRotor) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            rotor_apply_inverse(sharedOutput, headRotors, headDim, numRotorGroups, tid, threadgroupSize);
        }

        for (uint d = tid; d < headDim; d += threadgroupSize) {
            output[queryOffset + d] = \(castOut("sharedOutput[d]"));
        }
    }
    """
}

/// Generate post-prefill KV transfer kernel.
///
/// This supports the deferred-quantization path used when prefill keeps a dense
/// KV cache for quality, then converts into the decode cache format before the
/// first decode step.
public static func generateKVCacheTransfer(name: String) -> String {
    """
    struct KVCacheTransferParams {
        uint layerCount;
        uint kvHeadCount;
        uint headDimension;
        uint maxSequenceLength;
        uint sequenceLength;
        uint layoutMode;
        uint sourceKScheme;
        uint sourceVScheme;
        uint destinationKScheme;
        uint destinationVScheme;
        uint sourceKHeadSlotBytes;
        uint sourceVHeadSlotBytes;
        uint destinationKHeadSlotBytes;
        uint destinationVHeadSlotBytes;
        uint numRotorGroups;
        uint qjlDimension;
    };

    kernel void \(name)(
        device const uchar* sourceKeyCache    [[buffer(0)]],
        device const uchar* sourceValueCache  [[buffer(1)]],
        device uchar* destinationKeyCache     [[buffer(2)]],
        device uchar* destinationValueCache   [[buffer(3)]],
        constant KVCacheTransferParams& params [[buffer(4)]],
        device const half* rotorParams        [[buffer(5)]],
        device const half* qjlMatrix          [[buffer(6)]],
        device half* qjlResidualK             [[buffer(7)]],
        uint flatGroupId                      [[threadgroup_position_in_grid]],
        uint tid                              [[thread_index_in_threadgroup]],
        uint tiisg                            [[thread_index_in_simdgroup]],
        uint sgitg                            [[simdgroup_index_in_threadgroup]],
        uint threadgroupSize                  [[threads_per_threadgroup]]
    ) {
        if (params.sequenceLength == 0) return;

        const uint layerIndex = flatGroupId / params.sequenceLength;
        const uint positionIndex = flatGroupId % params.sequenceLength;
        if (layerIndex >= params.layerCount || positionIndex >= params.sequenceLength) return;

        const uint headDim = params.headDimension;
        const bool destinationKRotor = is_rotor_scheme(params.destinationKScheme);
        const bool destinationVRotor = is_rotor_scheme(params.destinationVScheme);
        const uint sourceKLayerBytes = params.maxSequenceLength * params.kvHeadCount * params.sourceKHeadSlotBytes;
        const uint sourceVLayerBytes = params.maxSequenceLength * params.kvHeadCount * params.sourceVHeadSlotBytes;
        const uint destinationKLayerBytes = params.maxSequenceLength * params.kvHeadCount * params.destinationKHeadSlotBytes;
        const uint destinationVLayerBytes = params.maxSequenceLength * params.kvHeadCount * params.destinationVHeadSlotBytes;

        threadgroup float convertedK[512];
        threadgroup float convertedV[512];
        threadgroup float quantSMin[32], quantSMax[32];

        for (uint kvHead = 0; kvHead < params.kvHeadCount; kvHead++) {
            uint sourceKByteOffset, sourceVByteOffset;
            uint destinationKByteOffset, destinationVByteOffset;
            if (params.layoutMode == 0) {
                sourceKByteOffset = layerIndex * sourceKLayerBytes + kvHead * params.maxSequenceLength * params.sourceKHeadSlotBytes + positionIndex * params.sourceKHeadSlotBytes;
                sourceVByteOffset = layerIndex * sourceVLayerBytes + kvHead * params.maxSequenceLength * params.sourceVHeadSlotBytes + positionIndex * params.sourceVHeadSlotBytes;
                destinationKByteOffset = layerIndex * destinationKLayerBytes + kvHead * params.maxSequenceLength * params.destinationKHeadSlotBytes + positionIndex * params.destinationKHeadSlotBytes;
                destinationVByteOffset = layerIndex * destinationVLayerBytes + kvHead * params.maxSequenceLength * params.destinationVHeadSlotBytes + positionIndex * params.destinationVHeadSlotBytes;
            } else {
                sourceKByteOffset = layerIndex * sourceKLayerBytes + positionIndex * params.kvHeadCount * params.sourceKHeadSlotBytes + kvHead * params.sourceKHeadSlotBytes;
                sourceVByteOffset = layerIndex * sourceVLayerBytes + positionIndex * params.kvHeadCount * params.sourceVHeadSlotBytes + kvHead * params.sourceVHeadSlotBytes;
                destinationKByteOffset = layerIndex * destinationKLayerBytes + positionIndex * params.kvHeadCount * params.destinationKHeadSlotBytes + kvHead * params.destinationKHeadSlotBytes;
                destinationVByteOffset = layerIndex * destinationVLayerBytes + positionIndex * params.kvHeadCount * params.destinationVHeadSlotBytes + kvHead * params.destinationVHeadSlotBytes;
            }

            for (uint d = tid; d < headDim; d += threadgroupSize) {
                convertedK[d] = read_kv_element(
                    sourceKeyCache + sourceKByteOffset,
                    d,
                    params.sourceKScheme,
                    params.sourceKHeadSlotBytes,
                    headDim
                );
                convertedV[d] = read_kv_element(
                    sourceValueCache + sourceVByteOffset,
                    d,
                    params.sourceVScheme,
                    params.sourceVHeadSlotBytes,
                    headDim
                );
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (destinationKRotor || destinationVRotor) {
                device const half* headRotors = rotorParams + ((layerIndex * params.kvHeadCount + kvHead) * params.numRotorGroups * 4);
                if (destinationKRotor) {
                    rotor_apply_forward(convertedK, headRotors, headDim, params.numRotorGroups, tid, threadgroupSize);
                }
                if (destinationVRotor) {
                    rotor_apply_forward(convertedV, headRotors, headDim, params.numRotorGroups, tid, threadgroupSize);
                }
            }

            const uint destinationKBaseScheme = rotor_base_scheme(params.destinationKScheme);
            if (destinationKBaseScheme == 0x40) {
                write_kv_quantized_q4(
                    convertedK,
                    destinationKeyCache + destinationKByteOffset,
                    headDim,
                    tid,
                    threadgroupSize,
                    tiisg,
                    sgitg,
                    quantSMin,
                    quantSMax
                );
            } else if (destinationKBaseScheme == 0x10) {
                write_kv_quantized_q8(
                    convertedK,
                    destinationKeyCache + destinationKByteOffset,
                    headDim,
                    tid,
                    threadgroupSize,
                    tiisg,
                    sgitg,
                    quantSMin,
                    quantSMax
                );
            } else {
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    write_kv_element_dense(
                        destinationKeyCache + destinationKByteOffset,
                        d,
                        convertedK[d],
                        params.destinationKScheme
                    );
                }
            }

            if (params.qjlDimension > 0) {
                threadgroup_barrier(mem_flags::mem_device);
                device half* qjlOut = qjlResidualK
                    + ((layerIndex * params.maxSequenceLength + positionIndex) * params.kvHeadCount + kvHead) * params.qjlDimension;
                qjl_compute_residual(
                    convertedK,
                    destinationKeyCache + destinationKByteOffset,
                    params.destinationKScheme,
                    qjlMatrix,
                    qjlOut,
                    headDim,
                    params.qjlDimension,
                    params.destinationKHeadSlotBytes,
                    tid,
                    threadgroupSize
                );
            }
            threadgroup_barrier(mem_flags::mem_device);

            const uint destinationVBaseScheme = rotor_base_scheme(params.destinationVScheme);
            if (destinationVBaseScheme == 0x40) {
                write_kv_quantized_q4(
                    convertedV,
                    destinationValueCache + destinationVByteOffset,
                    headDim,
                    tid,
                    threadgroupSize,
                    tiisg,
                    sgitg,
                    quantSMin,
                    quantSMax
                );
            } else if (destinationVBaseScheme == 0x10) {
                write_kv_quantized_q8(
                    convertedV,
                    destinationValueCache + destinationVByteOffset,
                    headDim,
                    tid,
                    threadgroupSize,
                    tiisg,
                    sgitg,
                    quantSMin,
                    quantSMax
                );
            } else {
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    write_kv_element_dense(
                        destinationValueCache + destinationVByteOffset,
                        d,
                        convertedV[d],
                        params.destinationVScheme
                    );
                }
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
    }
    """
}

/// Helper functions for KV cache read/write (shared by all flash_attn variants).
///
/// RotorQuant uses Clifford algebra Cl(3,0) rotors: groups of 3 dimensions are
/// Rotor rotation helper functions only (no KV cache quantization).
/// Used for standalone Metal shader compilation in tests.
public static let rotorHelperSource = """
// --- RotorQuant scheme detection ---

inline bool is_rotor_scheme(uint scheme) {
    return scheme == 0x70 || scheme == 0x71;
}

inline uint rotor_base_scheme(uint scheme) {
    if (scheme == 0x70) return 0x10;  // RotorQ8 → Q8
    if (scheme == 0x71) return 0x40;  // RotorQ4 → Q4
    return scheme;
}

// --- Clifford Cl(3,0) rotor sandwich product ---

inline float3 rotor_sandwich(float4 R, float3 v) {
    float s = R.x;
    float3 p = float3(R.w, -R.z, R.y);
    float3 t = 2.0f * cross(p, v);
    return v + s * t + cross(p, t);
}

inline float3 rotor_sandwich_inverse(float4 R, float3 v) {
    float s = R.x;
    float3 p = float3(-R.w, R.z, -R.y);
    float3 t = 2.0f * cross(p, v);
    return v + s * t + cross(p, t);
}

inline void rotor_apply_forward(
    threadgroup float* data,
    device const half* rotors,
    uint headDim,
    uint numGroups,
    uint tid,
    uint threadgroupSize
) {
    for (uint g = tid; g < numGroups; g += threadgroupSize) {
        uint base = g * 3;
        float v1 = (base < headDim) ? data[base] : 0.0f;
        float v2 = (base + 1 < headDim) ? data[base + 1] : 0.0f;
        float v3 = (base + 2 < headDim) ? data[base + 2] : 0.0f;
        float4 R = float4(
            float(rotors[g * 4]),
            float(rotors[g * 4 + 1]),
            float(rotors[g * 4 + 2]),
            float(rotors[g * 4 + 3])
        );
        float3 r = rotor_sandwich(R, float3(v1, v2, v3));
        if (base < headDim) data[base] = r.x;
        if (base + 1 < headDim) data[base + 1] = r.y;
        if (base + 2 < headDim) data[base + 2] = r.z;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void rotor_apply_inverse(
    threadgroup float* data,
    device const half* rotors,
    uint headDim,
    uint numGroups,
    uint tid,
    uint threadgroupSize
) {
    for (uint g = tid; g < numGroups; g += threadgroupSize) {
        uint base = g * 3;
        float v1 = (base < headDim) ? data[base] : 0.0f;
        float v2 = (base + 1 < headDim) ? data[base + 1] : 0.0f;
        float v3 = (base + 2 < headDim) ? data[base + 2] : 0.0f;
        float4 R = float4(
            float(rotors[g * 4]),
            float(rotors[g * 4 + 1]),
            float(rotors[g * 4 + 2]),
            float(rotors[g * 4 + 3])
        );
        float3 r = rotor_sandwich_inverse(R, float3(v1, v2, v3));
        if (base < headDim) data[base] = r.x;
        if (base + 1 < headDim) data[base + 1] = r.y;
        if (base + 2 < headDim) data[base + 2] = r.z;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
"""

/// Full flash attention helper source including RotorQuant, QJL, and KV cache functions.
///
/// rotated via sandwich product RvR̃ before quantization. Each rotor has 4
/// components [s, b₁₂, b₁₃, b₂₃] stored as half precision.
///
/// QJL (Quantized Johnson-Lindenstrauss) correction stores per-token projected
/// K quantization residuals for unbiased inner product estimation.
public static let flashAttentionHelperSource = """
// --- RotorQuant scheme detection ---

inline bool is_rotor_scheme(uint scheme) {
    return scheme == 0x70 || scheme == 0x71;
}

inline uint rotor_base_scheme(uint scheme) {
    if (scheme == 0x70) return 0x10;  // RotorQ8 → Q8
    if (scheme == 0x71) return 0x40;  // RotorQ4 → Q4
    return scheme;
}

// --- Clifford Cl(3,0) rotor sandwich product ---
//
// For grade-1 multivector v = (v��, v₂, v₃) and rotor R = (s, b₁₂, b₁₃, b₂₃):
// RvR̃ is computed via quaternion equivalence q = (s, b₂₃, -b₁₃, b₁₂).
// Uses the cross-product form: v' = v + s·t + cross(p, t) where t = 2·cross(p, v).

inline float3 rotor_sandwich(float4 R, float3 v) {
    float s = R.x;
    float3 p = float3(R.w, -R.z, R.y);  // (b₂₃, -b₁₃, b₁₂)
    float3 t = 2.0f * cross(p, v);
    return v + s * t + cross(p, t);
}

/// Inverse sandwich R̃vR (conjugate rotor).
inline float3 rotor_sandwich_inverse(float4 R, float3 v) {
    float s = R.x;
    float3 p = float3(-R.w, R.z, -R.y);  // (-b₂₃, b₁���, -b₁₂)
    float3 t = 2.0f * cross(p, v);
    return v + s * t + cross(p, t);
}

/// Apply forward Clifford rotation to a full head vector in threadgroup memory.
/// `data` has `headDim` elements. `rotors` points to [numGroups × 4] half values
/// for the current KV head (already offset by layer and head).
/// Each thread handles one or more groups of 3 dimensions independently.
inline void rotor_apply_forward(
    threadgroup float* data,
    device const half* rotors,
    uint headDim,
    uint numGroups,
    uint tid,
    uint threadgroupSize
) {
    for (uint g = tid; g < numGroups; g += threadgroupSize) {
        uint base = g * 3;
        float v1 = (base < headDim) ? data[base] : 0.0f;
        float v2 = (base + 1 < headDim) ? data[base + 1] : 0.0f;
        float v3 = (base + 2 < headDim) ? data[base + 2] : 0.0f;

        float4 R = float4(
            float(rotors[g * 4]),      // s
            float(rotors[g * 4 + 1]),  // b₁₂
            float(rotors[g * 4 + 2]),  // b₁₃
            float(rotors[g * 4 + 3])   // b₂₃
        );
        float3 r = rotor_sandwich(R, float3(v1, v2, v3));

        if (base < headDim) data[base] = r.x;
        if (base + 1 < headDim) data[base + 1] = r.y;
        if (base + 2 < headDim) data[base + 2] = r.z;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

/// Apply inverse Clifford rotation to a full head vector in threadgroup memory.
inline void rotor_apply_inverse(
    threadgroup float* data,
    device const half* rotors,
    uint headDim,
    uint numGroups,
    uint tid,
    uint threadgroupSize
) {
    for (uint g = tid; g < numGroups; g += threadgroupSize) {
        uint base = g * 3;
        float v1 = (base < headDim) ? data[base] : 0.0f;
        float v2 = (base + 1 < headDim) ? data[base + 1] : 0.0f;
        float v3 = (base + 2 < headDim) ? data[base + 2] : 0.0f;

        float4 R = float4(
            float(rotors[g * 4]),
            float(rotors[g * 4 + 1]),
            float(rotors[g * 4 + 2]),
            float(rotors[g * 4 + 3])
        );
        float3 r = rotor_sandwich_inverse(R, float3(v1, v2, v3));

        if (base < headDim) data[base] = r.x;
        if (base + 1 < headDim) data[base + 1] = r.y;
        if (base + 2 < headDim) data[base + 2] = r.z;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// --- KV cache read/write ---

inline float read_kv_element(
    device const uchar* cache, uint elementIndex, uint kvQuantScheme,
    uint headSlotBytes, uint headDim
) {
    uint baseScheme = is_rotor_scheme(kvQuantScheme) ? rotor_base_scheme(kvQuantScheme) : kvQuantScheme;
    if (baseScheme == 0x00) {
        return float(((device const half*)cache)[elementIndex]);
    }
    if (baseScheme == 0x01) {
        return bf16_to_float(((device const uint16_t*)cache)[elementIndex]);
    }
    if (baseScheme == 0x02) {
        return ((device const float*)cache)[elementIndex];
    }
    if (baseScheme == 0x40) {
        const uint groupSize = 64;
        const uint bytesPerBlock = 36;
        uint group = elementIndex / groupSize;
        uint indexInGroup = elementIndex % groupSize;
        uint blockOffset = group * bytesPerBlock;
        float scale = float(*(device const half*)(cache + blockOffset));
        float zero = float(*(device const half*)(cache + blockOffset + 2));
        uchar packed = *(device const uchar*)(cache + blockOffset + 4 + indexInGroup / 2);
        uchar nibble = (indexInGroup % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
        return scale * float(nibble) + zero;
    }
    // Q8: 32 elements per group
    const uint groupSize = 32;
    const uint bytesPerBlock = 36;
    uint group = elementIndex / groupSize;
    uint indexInGroup = elementIndex % groupSize;
    uint blockOffset = group * bytesPerBlock;
    float scale = float(*(device const half*)(cache + blockOffset));
    float zero = float(*(device const half*)(cache + blockOffset + 2));
    uchar quantized = *(device const uchar*)(cache + blockOffset + 4 + indexInGroup);
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

// --- QJL (Quantized Johnson-Lindenstrauss) correction ---
//
// Projects the quantization residual r = x_rotated - dequantize(quantize(x_rotated))
// using a random Rademacher matrix Φ (±1/√m). The projected residual is stored
// alongside the KV cache for inner product correction at attention time.
//
// Corrected score: Q·K ≈ Q̂·K̂ + (Φ·Q̂)·(Φ·r_K)

/// Compute QJL projected residual from a pre-rotation buffer and the just-quantized cache.
/// Writes `qjlDim` half values to `qjlOut`.
/// `rotatedSrc` is the pre-quantization rotated vector in threadgroup memory.
/// After this call, `rotatedSrc` contains the residual (overwritten in-place).
/// `quantizedSlot` is the cache slot that was just written (for dequantization).
inline void qjl_compute_residual(
    threadgroup float* rotatedSrc,
    device const uchar* quantizedSlot,
    uint quantScheme,
    device const half* qjlMatrix,   // [headDim × qjlDim]
    device half* qjlOut,            // [qjlDim]
    uint headDim,
    uint qjlDim,
    uint headSlotBytes,
    uint tid,
    uint threadgroupSize
) {
    // Step 1: Compute residual in-place (parallel across headDim).
    // Dequantize once per element instead of qjlDim times.
    for (uint d = tid; d < headDim; d += threadgroupSize) {
        float dequantized = read_kv_element(quantizedSlot, d, quantScheme, headSlotBytes, headDim);
        rotatedSrc[d] = rotatedSrc[d] - dequantized;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Project residual via Rademacher matrix (parallel across qjlDim)
    for (uint j = tid; j < qjlDim; j += threadgroupSize) {
        float proj = 0.0f;
        for (uint d = 0; d < headDim; d++) {
            proj += rotatedSrc[d] * float(qjlMatrix[d * qjlDim + j]);
        }
        qjlOut[j] = half(proj);
    }
}

/// Pre-compute QJL projection of a query vector: Φ·Q_rotated.
/// Stores qjlDim float values into threadgroup memory.
inline void qjl_project_query(
    threadgroup const float* rotQuery,
    device const half* qjlMatrix,   // [headDim × qjlDim]
    threadgroup float* qjlQueryProj, // [qjlDim] output in threadgroup
    uint headDim,
    uint qjlDim,
    uint tid,
    uint threadgroupSize
) {
    for (uint j = tid; j < qjlDim; j += threadgroupSize) {
        float proj = 0.0f;
        for (uint d = 0; d < headDim; d++) {
            proj += rotQuery[d] * float(qjlMatrix[d * qjlDim + j]);
        }
        qjlQueryProj[j] = proj;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

/// Compute QJL correction for a single Q·K inner product.
/// Returns the correction term: (Φ·Q)·(Φ·r_K).
inline float qjl_score_correction(
    threadgroup const float* qjlQueryProj, // [qjlDim] pre-computed
    device const half* qjlResidual,        // [qjlDim] stored per-token
    uint qjlDim,
    uint tid,
    uint threadgroupSize
) {
    float correction = 0.0f;
    for (uint j = tid; j < qjlDim; j += threadgroupSize) {
        correction += qjlQueryProj[j] * float(qjlResidual[j]);
    }
    return correction;
}

/// Write a full head vector to KV cache with per-group Q8 quantization.
/// `sMin` and `sMax` must be threadgroup float arrays of at least 32 elements,
/// provided by the calling kernel function.
inline void write_kv_quantized_q8(
    threadgroup const float* src, device uchar* cacheSlot,
    uint headDim, uint tid, uint threadgroupSize,
    uint tiisg, uint sgitg,
    threadgroup float* sMin, threadgroup float* sMax
) {
    const uint groupSize = 32;
    const uint bytesPerBlock = 36;
    const uint numGroups = (headDim + groupSize - 1) / groupSize;
    for (uint g = 0; g < numGroups; g++) {
        uint groupStart = g * groupSize;
        float localMin = HUGE_VALF, localMax = -HUGE_VALF;
        for (uint i = tid; i < groupSize && (groupStart + i) < headDim; i += threadgroupSize) {
            float val = src[groupStart + i];
            localMin = min(localMin, val);
            localMax = max(localMax, val);
        }
        localMin = simd_min(localMin);
        localMax = simd_max(localMax);
        if (tiisg == 0) { sMin[sgitg] = localMin; sMax[sgitg] = localMax; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
            float gMin = sMin[0], gMax = sMax[0];
            for (uint s = 1; s < sgCount; s++) { gMin = min(gMin, sMin[s]); gMax = max(gMax, sMax[s]); }
            sMin[0] = gMin; sMax[0] = gMax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float groupMin = sMin[0], groupMax = sMax[0];
        float groupScale = (groupMax - groupMin) / 255.0f;
        if (groupScale < 1e-10f) groupScale = 1e-10f;
        device uchar* block = cacheSlot + g * bytesPerBlock;
        if (tid == 0) {
            *(device half*)(block) = half(groupScale);
            *(device half*)(block + 2) = half(groupMin);
        }
        threadgroup_barrier(mem_flags::mem_device);
        for (uint i = tid; i < groupSize && (groupStart + i) < headDim; i += threadgroupSize) {
            float val = src[groupStart + i];
            int q = int(round((val - groupMin) / groupScale));
            q = clamp(q, 0, 255);
            *(device uchar*)(block + 4 + i) = uchar(q);
        }
    }
}

/// Write a full head vector to KV cache with per-group Q4 quantization.
/// `sMin` and `sMax` must be threadgroup float arrays of at least 32 elements,
/// provided by the calling kernel function.
inline void write_kv_quantized_q4(
    threadgroup const float* src, device uchar* cacheSlot,
    uint headDim, uint tid, uint threadgroupSize,
    uint tiisg, uint sgitg,
    threadgroup float* sMin, threadgroup float* sMax
) {
    const uint groupSize = 64;
    const uint bytesPerBlock = 36;
    const uint numGroups = (headDim + groupSize - 1) / groupSize;
    for (uint g = 0; g < numGroups; g++) {
        uint groupStart = g * groupSize;
        float localMin = HUGE_VALF, localMax = -HUGE_VALF;
        for (uint i = tid; i < groupSize && (groupStart + i) < headDim; i += threadgroupSize) {
            float val = src[groupStart + i];
            localMin = min(localMin, val);
            localMax = max(localMax, val);
        }
        localMin = simd_min(localMin);
        localMax = simd_max(localMax);
        if (tiisg == 0) { sMin[sgitg] = localMin; sMax[sgitg] = localMax; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
            float gMin = sMin[0], gMax = sMax[0];
            for (uint s = 1; s < sgCount; s++) { gMin = min(gMin, sMin[s]); gMax = max(gMax, sMax[s]); }
            sMin[0] = gMin; sMax[0] = gMax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float groupMin = sMin[0], groupMax = sMax[0];
        float groupScale = (groupMax - groupMin) / 15.0f;
        if (groupScale < 1e-10f) groupScale = 1e-10f;
        device uchar* block = cacheSlot + g * bytesPerBlock;
        if (tid == 0) {
            *(device half*)(block) = half(groupScale);
            *(device half*)(block + 2) = half(groupMin);
        }
        threadgroup_barrier(mem_flags::mem_device);
        for (uint i = tid; i < groupSize / 2 && (groupStart + i * 2) < headDim; i += threadgroupSize) {
            float v0 = src[groupStart + i * 2];
            float v1 = (groupStart + i * 2 + 1 < headDim) ? src[groupStart + i * 2 + 1] : 0.0f;
            uchar q0 = uchar(clamp(int(round((v0 - groupMin) / groupScale)), 0, 15));
            uchar q1 = uchar(clamp(int(round((v1 - groupMin) / groupScale)), 0, 15));
            *(device uchar*)(block + 4 + i) = q0 | (q1 << 4);
        }
    }
}
"""

// MARK: - Direct-Scratch Batch Attention

/// Generate batch attention kernel that reads K/V directly from scratch buffers.
///
/// Used for embedding-only models (no OutputHead) where KV cache is unnecessary.
/// QK normalization and RoPE are handled by their existing separate kernels —
/// this kernel only replaces the attention step's KV source.
///
/// Compared to `generateBatchFlashAttention`, this kernel:
/// - Reads K/V as float from scratch (no quantization, no rotor, no QJL)
/// - Uses element strides (totalQDimension, totalKDimension) instead of byte offsets
/// - Same online softmax + window/causal logic
public static func generateDirectScratchBatchFlashAttention(
    name: String,
    bufferPrecision: BufferPrecision
) -> String {
    let bt = bufferPrecision.metalType
    let castOut: (String) -> String = { expr in
        bufferPrecision == .float32 ? "(\(expr))" : "\(bt)(\(expr))"
    }

    return """
    kernel void \(name)(
        device const \(bt)* query         [[buffer(0)]],
        device const \(bt)* keyScratch    [[buffer(1)]],
        device const \(bt)* valueScratch  [[buffer(2)]],
        device \(bt)* output              [[buffer(3)]],
        constant uint& headCount          [[buffer(4)]],
        constant uint& kvHeadCount        [[buffer(5)]],
        constant uint& headDimension      [[buffer(6)]],
        constant float& scale             [[buffer(7)]],
        constant uint& sequenceLength     [[buffer(8)]],
        constant uint& totalQDimension    [[buffer(9)]],
        constant uint& totalKDimension    [[buffer(10)]],
        constant uint& causal             [[buffer(11)]],
        constant uint& windowLeft         [[buffer(12)]],
        constant uint& windowRight        [[buffer(13)]],
        uint2 gid                         [[threadgroup_position_in_grid]],
        uint tid                          [[thread_index_in_threadgroup]],
        uint tiisg                        [[thread_index_in_simdgroup]],
        uint sgitg                        [[simdgroup_index_in_threadgroup]],
        uint2 tgSize                      [[threads_per_threadgroup]]
    ) {
        // Grid layout: width = headCount, height = sequenceLength (adjusted at runtime).
        const uint headIndex = gid.x;
        const uint posId = gid.y;
        // Threadgroup is 1D (width=threads, height=1, depth=1); only .x is meaningful.
        const uint threadgroupSize = tgSize.x;

        const uint headDim = headDimension;
        const uint kvHeadIndex = headIndex * kvHeadCount / headCount;
        const uint queryOffset = posId * totalQDimension + headIndex * headDim;

        float maxScore = -HUGE_VALF;
        float sumExp = 0.0f;

        threadgroup float sharedOutput[512];
        for (uint d = tid; d < headDim; d += threadgroupSize) {
            sharedOutput[d] = 0.0f;
        }

        const bool isCausal = (causal != 0);
        const uint leftReach = (windowLeft == 0xFFFFFFFFu || windowLeft == 0u)
            ? posId
            : min(posId, windowLeft - 1u);
        const uint attentionStart = posId - leftReach;
        const uint attentionEnd = isCausal
            ? posId
            : (
                (windowRight == 0xFFFFFFFFu)
                    ? (sequenceLength - 1u)
                    : min(sequenceLength - 1u, posId + (windowRight == 0u ? 0u : windowRight - 1u))
            );

        for (uint t = attentionStart; t <= attentionEnd; t++) {
            const uint kOffset = t * totalKDimension + kvHeadIndex * headDim;

            float score = 0.0f;
            for (uint d = tid; d < headDim; d += threadgroupSize) {
                float q = float(query[queryOffset + d]);
                float k = float(keyScratch[kOffset + d]);
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

            float weight = exp(score - maxScore);
            for (uint d = tid; d < headDim; d += threadgroupSize) {
                float v = float(valueScratch[kOffset + d]);
                sharedOutput[d] = sharedOutput[d] * correction + weight * v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float invSum = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f;
        for (uint d = tid; d < headDim; d += threadgroupSize) {
            output[queryOffset + d] = \(castOut("sharedOutput[d] * invSum"));
        }
    }
    """
}

}
