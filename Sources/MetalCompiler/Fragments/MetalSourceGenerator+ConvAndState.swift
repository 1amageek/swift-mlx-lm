extension MetalSourceGenerator {
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

// MARK: - State Space

/// Weight-independent SSM helper functions (emitted once per compilation unit).
public static func generateSSMWeightIndependentHelpers() -> String {
    return """
    inline float stable_softplus(float x) { return max(x, 0.0f) + log(1.0f + exp(-abs(x))); }
    inline float stable_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
    inline float compute_l2_inv_norm(
        threadgroup float* vec,
        uint dim,
        uint tid,
        uint tgSize,
        threadgroup float* scratch
    ) {
        float sumSq = 0.0f;
        for (uint d = tid; d < dim; d += tgSize) sumSq += vec[d] * vec[d];
        sumSq = simd_sum(sumSq);
        if (tid % SIMD_WIDTH == 0) scratch[tid / SIMD_WIDTH] = sumSq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0.0f;
            for (uint i = 0; i < (tgSize + SIMD_WIDTH - 1) / SIMD_WIDTH; ++i) total += scratch[i];
            scratch[0] = rsqrt(total + 1e-6f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return scratch[0];
    }
    """
}

/// Weight-dependent conv_silu overload (emitted per weight format).
/// MSL function overloading resolves the correct variant based on convWeight pointer type.
public static func generateSSMConvSiluHelper(weightFormat: WeightFormat) -> String {
    let wt = weightFormat.bufferType
    let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

    return """
    inline float conv_silu(
        device const half* convState,
        device const \(wt)* convWeight,
        uint channel,
        uint convDimension,
        uint convKernelSize
    ) {
        float sum = 0.0f;
        for (uint k = 0; k < convKernelSize; ++k) {
            sum += float(convState[k * convDimension + channel])
                * \(readWeight("convWeight[channel * convKernelSize + k]"));
        }
        return sum * stable_sigmoid(sum);
    }
    """
}

/// Combined SSM helper source for a single weight format (convenience for library use).
public static func generateSSMHelperSource(weightFormat: WeightFormat) -> String {
    return generateSSMWeightIndependentHelpers() + "\n" + generateSSMConvSiluHelper(weightFormat: weightFormat)
}

public static func generateSSMRecurrence(
    name: String,
    bufferPrecision: BufferPrecision,
    weightFormat: WeightFormat,
    convDimension: Int,
    maxThreadgroupSize: Int
) -> String {
    let bt = bufferPrecision.metalType
    let wt = weightFormat.bufferType
    let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

    return """
    kernel void \(name)(
        device const \(bt)* projectedQKV [[buffer(0)]],
        device const \(bt)* projectedZ [[buffer(1)]],
        device const \(bt)* projectedBeta [[buffer(2)]],
        device const \(bt)* projectedAlpha [[buffer(3)]],
        device const \(wt)* convWeight [[buffer(4)]],
        device const float* normWeight [[buffer(5)]],
        device const \(wt)* dtBias [[buffer(6)]],
        device const float* aLog [[buffer(7)]],
        device float* recurrentState [[buffer(8)]],
        device half* convState [[buffer(9)]],
        device \(bt)* output [[buffer(10)]],
        constant uint& numHeads [[buffer(11)]],
        constant uint& groupCount [[buffer(12)]],
        constant uint& keyDimension [[buffer(13)]],
        constant uint& valueDimension [[buffer(14)]],
        constant uint& convKernelSize [[buffer(15)]],
        uint tid [[thread_index_in_threadgroup]],
        uint tgSize [[threads_per_threadgroup]]
    ) {
        const uint dk = keyDimension;
        const uint dv = valueDimension;
        const uint keyGroupDim = groupCount * dk;
        const uint convDim = 2 * keyGroupDim + numHeads * dv;
        const uint headsPerGroup = max(1u, numHeads / max(groupCount, 1u));

        threadgroup float convSiluCache[\(convDimension)];

        // Fused conv-shift + conv_silu: shift conv state and compute convolution
        // in one pass using registers, eliminating device barrier and redundant reads.
        for (uint channel = tid; channel < convDim; channel += tgSize) {
            float sum = 0.0f;
            for (uint k = 0; k + 1 < convKernelSize; ++k) {
                float val = float(convState[(k + 1) * convDim + channel]);
                convState[k * convDim + channel] = half(val);
                sum += val * \(readWeight("convWeight[channel * convKernelSize + k]"));
            }
            float newVal = float(projectedQKV[channel]);
            convState[(convKernelSize - 1) * convDim + channel] = half(newVal);
            sum += newVal * \(readWeight("convWeight[channel * convKernelSize + convKernelSize - 1]"));
            convSiluCache[channel] = sum * stable_sigmoid(sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 3: Per-head recurrence with multi-thread-per-head parallelism.
        // Distribute dv dimension across multiple threads within each head.
        threadgroup float normPartials[\(maxThreadgroupSize)];
        {
            const uint threadsPerHead = min(tgSize / max(numHeads, 1u), dv);
            const uint activeThreads = numHeads * threadsPerHead;
            if (tid < activeThreads) {
                const uint headIndex = tid / threadsPerHead;
                const uint localTid = tid % threadsPerHead;
                const uint dChunk = dv / threadsPerHead;
                const uint dStart = localTid * dChunk;
                const uint dEnd = (localTid + 1 == threadsPerHead) ? dv : dStart + dChunk;

                const uint keyGroupIndex = min(groupCount - 1, headIndex / headsPerGroup);
                float decay = exp(-exp(aLog[headIndex]) * stable_softplus(float(projectedAlpha[headIndex]) + \(readWeight("dtBias[headIndex]"))));
                float beta = stable_sigmoid(float(projectedBeta[headIndex]));
                device float* state = recurrentState + headIndex * dk * dv;

                const uint qBase = keyGroupIndex * dk;
                const uint kBase = keyGroupDim + keyGroupIndex * dk;
                const uint vBase = 2 * keyGroupDim + headIndex * dv;

                float qNormSq = 0.0f;
                float kNormSq = 0.0f;
                for (uint j = 0; j < dk; ++j) {
                    float q = convSiluCache[qBase + j];
                    float k = convSiluCache[kBase + j];
                    qNormSq += q * q;
                    kNormSq += k * k;
                }
                float qInv = rsqrt(qNormSq + 1e-6f) * rsqrt(float(dk));
                float kInv = rsqrt(kNormSq + 1e-6f);

                // Fully fused per-d: decay+kvmem → delta → update+output
                // kInv/qInv factored out of inner loops to eliminate 2*dk redundant muls per d.
                float localNormSq = 0.0f;
                for (uint d = dStart; d < dEnd; ++d) {
                    float kvmemRaw = 0.0f;
                    for (uint j = 0; j < dk; ++j) {
                        float s = state[j * dv + d] * decay;
                        state[j * dv + d] = s;
                        kvmemRaw += s * convSiluCache[kBase + j];
                    }
                    float delta = beta * (convSiluCache[vBase + d] - kvmemRaw * kInv);
                    float kInvDelta = kInv * delta;
                    float dotRaw = 0.0f;
                    for (uint j = 0; j < dk; ++j) {
                        state[j * dv + d] += convSiluCache[kBase + j] * kInvDelta;
                        dotRaw += state[j * dv + d] * convSiluCache[qBase + j];
                    }
                    float dot = dotRaw * qInv;
                    output[headIndex * dv + d] = \(bt)(dot);
                    localNormSq += dot * dot;
                }

                normPartials[tid] = localNormSq;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            const uint threadsPerHead = min(tgSize / max(numHeads, 1u), dv);
            const uint activeThreads = numHeads * threadsPerHead;
            if (tid < activeThreads) {
                const uint headIndex = tid / threadsPerHead;
                const uint localTid = tid % threadsPerHead;
                const uint dChunk = dv / threadsPerHead;
                const uint dStart = localTid * dChunk;
                const uint dEnd = (localTid + 1 == threadsPerHead) ? dv : dStart + dChunk;

                float totalNormSq = 0.0f;
                for (uint t = 0; t < threadsPerHead; ++t) {
                    totalNormSq += normPartials[headIndex * threadsPerHead + t];
                }
                float rmsScale = rsqrt(totalNormSq / float(dv) + 1e-6f);
                for (uint d = dStart; d < dEnd; ++d) {
                    float normed = float(output[headIndex * dv + d]) * rmsScale * normWeight[d];
                    float z = float(projectedZ[headIndex * dv + d]);
                    output[headIndex * dv + d] = \(bt)(normed * z * stable_sigmoid(z));
                }
            }
        }
    }
    """
}

public static func generateSSMRecurrenceSequence(
    name: String,
    bufferPrecision: BufferPrecision,
    weightFormat: WeightFormat,
    convDimension: Int,
    maxThreadgroupSize: Int
) -> String {
    let bt = bufferPrecision.metalType
    let wt = weightFormat.bufferType
    let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

    return """
    kernel void \(name)(
        device const \(bt)* projectedQKV [[buffer(0)]],
        device const \(bt)* projectedZ [[buffer(1)]],
        device const \(bt)* projectedBeta [[buffer(2)]],
        device const \(bt)* projectedAlpha [[buffer(3)]],
        device const \(wt)* convWeight [[buffer(4)]],
        device const float* normWeight [[buffer(5)]],
        device const \(wt)* dtBias [[buffer(6)]],
        device const float* aLog [[buffer(7)]],
        device float* recurrentState [[buffer(8)]],
        device half* convState [[buffer(9)]],
        device \(bt)* output [[buffer(10)]],
        constant uint& numHeads [[buffer(11)]],
        constant uint& groupCount [[buffer(12)]],
        constant uint& keyDimension [[buffer(13)]],
        constant uint& valueDimension [[buffer(14)]],
        constant uint& convKernelSize [[buffer(15)]],
        constant uint& sequenceLength [[buffer(16)]],
        uint tid [[thread_index_in_threadgroup]],
        uint tgSize [[threads_per_threadgroup]]
    ) {
        const uint dk = keyDimension;
        const uint dv = valueDimension;
        const uint keyGroupDim = groupCount * dk;
        const uint convDim = 2 * keyGroupDim + numHeads * dv;
        const uint outputDim = numHeads * dv;
        const uint safeGroupCount = max(groupCount, 1u);
        const uint headsPerGroup = max(1u, numHeads / safeGroupCount);

        // Precomputed conv_silu values for all channels (eliminates ~33K redundant
        // device memory reads per head per position).
        threadgroup float convSiluCache[\(convDimension)];
        threadgroup float normPartials[\(maxThreadgroupSize)];

        for (uint pos = 0; pos < sequenceLength; ++pos) {
            device const \(bt)* projectedQKVPos = projectedQKV + pos * convDim;
            device const \(bt)* projectedZPos = projectedZ + pos * outputDim;
            device const \(bt)* projectedBetaPos = projectedBeta + pos * numHeads;
            device const \(bt)* projectedAlphaPos = projectedAlpha + pos * numHeads;
            device \(bt)* outputPos = output + pos * outputDim;

            // Fused conv-shift + conv_silu: shift conv state and compute convolution
            // in one pass using registers, eliminating device barrier and redundant reads.
            for (uint channel = tid; channel < convDim; channel += tgSize) {
                float sum = 0.0f;
                for (uint k = 0; k + 1 < convKernelSize; ++k) {
                    float val = float(convState[(k + 1) * convDim + channel]);
                    convState[k * convDim + channel] = half(val);
                    sum += val * \(readWeight("convWeight[channel * convKernelSize + k]"));
                }
                float newVal = float(projectedQKVPos[channel]);
                convState[(convKernelSize - 1) * convDim + channel] = half(newVal);
                sum += newVal * \(readWeight("convWeight[channel * convKernelSize + convKernelSize - 1]"));
                convSiluCache[channel] = sum * stable_sigmoid(sum);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Phase 3: Per-head recurrence with multi-thread-per-head parallelism.
            // Distribute dv dimension across multiple threads within each head.
            {
                const uint threadsPerHead = min(tgSize / max(numHeads, 1u), dv);
                const uint activeThreads = numHeads * threadsPerHead;
                if (tid < activeThreads) {
                    const uint headIndex = tid / threadsPerHead;
                    const uint localTid = tid % threadsPerHead;
                    const uint dChunk = dv / threadsPerHead;
                    const uint dStart = localTid * dChunk;
                    const uint dEnd = (localTid + 1 == threadsPerHead) ? dv : dStart + dChunk;

                    const uint keyGroupIndex = min(groupCount - 1, headIndex / headsPerGroup);
                    float decay = exp(-exp(aLog[headIndex]) * stable_softplus(float(projectedAlphaPos[headIndex]) + \(readWeight("dtBias[headIndex]"))));
                    float beta = stable_sigmoid(float(projectedBetaPos[headIndex]));
                    device float* state = recurrentState + headIndex * dk * dv;

                    const uint qBase = keyGroupIndex * dk;
                    const uint kBase = keyGroupDim + keyGroupIndex * dk;
                    const uint vBase = 2 * keyGroupDim + headIndex * dv;

                    float qNormSq = 0.0f;
                    float kNormSq = 0.0f;
                    for (uint j = 0; j < dk; ++j) {
                        float q = convSiluCache[qBase + j];
                        float k = convSiluCache[kBase + j];
                        qNormSq += q * q;
                        kNormSq += k * k;
                    }
                    float qInv = rsqrt(qNormSq + 1e-6f) * rsqrt(float(dk));
                    float kInv = rsqrt(kNormSq + 1e-6f);

                    // Fully fused per-d: decay+kvmem → delta → update+output
                    // kInv/qInv factored out of inner loops to eliminate 2*dk redundant muls per d.
                    float localNormSq = 0.0f;
                    for (uint d = dStart; d < dEnd; ++d) {
                        float kvmemRaw = 0.0f;
                        for (uint j = 0; j < dk; ++j) {
                            float s = state[j * dv + d] * decay;
                            state[j * dv + d] = s;
                            kvmemRaw += s * convSiluCache[kBase + j];
                        }
                        float delta = beta * (convSiluCache[vBase + d] - kvmemRaw * kInv);
                        float kInvDelta = kInv * delta;
                        float dotRaw = 0.0f;
                        for (uint j = 0; j < dk; ++j) {
                            state[j * dv + d] += convSiluCache[kBase + j] * kInvDelta;
                            dotRaw += state[j * dv + d] * convSiluCache[qBase + j];
                        }
                        float dot = dotRaw * qInv;
                        outputPos[headIndex * dv + d] = \(bt)(dot);
                        localNormSq += dot * dot;
                    }

                    normPartials[tid] = localNormSq;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            {
                const uint threadsPerHead = min(tgSize / max(numHeads, 1u), dv);
                const uint activeThreads = numHeads * threadsPerHead;
                if (tid < activeThreads) {
                    const uint headIndex = tid / threadsPerHead;
                    const uint localTid = tid % threadsPerHead;
                    const uint dChunk = dv / threadsPerHead;
                    const uint dStart = localTid * dChunk;
                    const uint dEnd = (localTid + 1 == threadsPerHead) ? dv : dStart + dChunk;

                    float totalNormSq = 0.0f;
                    for (uint t = 0; t < threadsPerHead; ++t) {
                        totalNormSq += normPartials[headIndex * threadsPerHead + t];
                    }
                    float rmsScale = rsqrt(totalNormSq / float(dv) + 1e-6f);
                    for (uint d = dStart; d < dEnd; ++d) {
                        float normed = float(outputPos[headIndex * dv + d]) * rmsScale * normWeight[d];
                        float z = float(projectedZPos[headIndex * dv + d]);
                        outputPos[headIndex * dv + d] = \(bt)(normed * z * stable_sigmoid(z));
                    }
                }
            }
            // Skip barrier after final position — no subsequent iteration reads this state,
            // and the command encoder's implicit barrier handles cross-dispatch visibility.
            if (pos + 1 < sequenceLength) {
                threadgroup_barrier(mem_flags::mem_device);
            }
        }
    }
    """
}
}
