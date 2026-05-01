extension MetalSourceGenerator {
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

/// Generate batched GEMV kernel for 4 projections sharing the same input.
public static func generateBatchedGEMV4(
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
        device const \(wt)* weight3        [[buffer(4)]],
        device \(bt)* output0              [[buffer(5)]],
        device \(bt)* output1              [[buffer(6)]],
        device \(bt)* output2              [[buffer(7)]],
        device \(bt)* output3              [[buffer(8)]],
        constant uint& inputDimension      [[buffer(9)]],
        constant uint& outputDim0          [[buffer(10)]],
        constant uint& outputDim1          [[buffer(11)]],
        constant uint& outputDim2          [[buffer(12)]],
        constant uint& outputDim3          [[buffer(13)]],
        uint gid                           [[threadgroup_position_in_grid]],
        uint tid                           [[thread_index_in_threadgroup]],
        uint tiisg                         [[thread_index_in_simdgroup]],
        uint sgitg                         [[simdgroup_index_in_threadgroup]],
        uint threadsPerThreadgroup         [[threads_per_threadgroup]]
    ) {
        const uint tileElements = 256;
        const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup / SIMD_WIDTH);
        const uint globalRow = gid * rowsPerThreadgroup + sgitg;
        const uint totalRows = outputDim0 + outputDim1 + outputDim2 + outputDim3;
        if (globalRow >= totalRows) return;

        device const \(wt)* weight;
        device \(bt)* output;
        uint localRow;
        if (globalRow < outputDim0) {
            weight = weight0; output = output0; localRow = globalRow;
        } else if (globalRow < outputDim0 + outputDim1) {
            weight = weight1; output = output1; localRow = globalRow - outputDim0;
        } else if (globalRow < outputDim0 + outputDim1 + outputDim2) {
            weight = weight2; output = output2; localRow = globalRow - outputDim0 - outputDim1;
        } else {
            weight = weight3; output = output3; localRow = globalRow - outputDim0 - outputDim1 - outputDim2;
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

public static func generateBatchedGEMV4ArgumentTableVariant(
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
        device const \(wt)* weight3 [[id(4)]];
        device \(bt)* output0 [[id(5)]];
        device \(bt)* output1 [[id(6)]];
        device \(bt)* output2 [[id(7)]];
        device \(bt)* output3 [[id(8)]];
    };

    kernel void \(name)(
        constant \(inputStructName)& args        [[buffer(\(argumentBufferIndex))]],
        constant uint& inputDimension            [[buffer(9)]],
        constant uint& outputDim0                [[buffer(10)]],
        constant uint& outputDim1                [[buffer(11)]],
        constant uint& outputDim2                [[buffer(12)]],
        constant uint& outputDim3                [[buffer(13)]],
        uint gid                                 [[threadgroup_position_in_grid]],
        uint tid                                 [[thread_index_in_threadgroup]],
        uint tiisg                               [[thread_index_in_simdgroup]],
        uint sgitg                               [[simdgroup_index_in_threadgroup]],
        uint threadsPerThreadgroup               [[threads_per_threadgroup]]
    ) {
        const uint tileElements = 256;
        const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup / SIMD_WIDTH);
        const uint globalRow = gid * rowsPerThreadgroup + sgitg;
        const uint totalRows = outputDim0 + outputDim1 + outputDim2 + outputDim3;
        if (globalRow >= totalRows) return;

        device const \(wt)* weight;
        device \(bt)* output;
        uint localRow;
        if (globalRow < outputDim0) {
            weight = args.weight0; output = args.output0; localRow = globalRow;
        } else if (globalRow < outputDim0 + outputDim1) {
            weight = args.weight1; output = args.output1; localRow = globalRow - outputDim0;
        } else if (globalRow < outputDim0 + outputDim1 + outputDim2) {
            weight = args.weight2; output = args.output2; localRow = globalRow - outputDim0 - outputDim1;
        } else {
            weight = args.weight3; output = args.output3; localRow = globalRow - outputDim0 - outputDim1 - outputDim2;
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

/// Generate a sequence-wide batched GEMV kernel for projections sharing the same input.
///
/// This mirrors the decode `batched_gemv{N}` reduction order for every sequence
/// position while preserving a single sequence dispatch. Hybrid prefill uses it
/// for state-sensitive dense projections so SSM/conv state and KV cache are
/// seeded from decode-equivalent projection values.
public static func generateBatchedSequenceGEMV(
    name: String,
    count: Int,
    bufferPrecision: BufferPrecision,
    weightFormat: WeightFormat,
    tileElements: Int = 256
) -> String {
    precondition((2...4).contains(count), "batched sequence GEMV supports 2...4 projections")
    let bt = bufferPrecision.metalType
    let wt = weightFormat.bufferType
    let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
    let storeValue: (String) -> String = { expr in
        bufferPrecision.isPrefillSequencePrecision
            ? MetalSourceGenerator.sequenceStorageValue(expr, weightFormat: weightFormat)
            : expr
    }

    let weightBindings = (0..<count).map { i in
        "device const \(wt)* weight\(i)        [[buffer(\(1 + i))]],"
    }.joined(separator: "\n        ")
    let outputBindings = (0..<count).map { i in
        "device \(bt)* output\(i)              [[buffer(\(1 + count + i))]],"
    }.joined(separator: "\n        ")
    let outputDimBindings = (0..<count).map { i in
        "constant uint& outputDim\(i)          [[buffer(\(2 + 2 * count + i))]],"
    }.joined(separator: "\n        ")

    let totalRows = (0..<count).map { "outputDim\($0)" }.joined(separator: " + ")
    let branchBlocks = (0..<count).map { i -> String in
        let condition: String
        if i == 0 {
            condition = "if (globalRow < outputDim0)"
        } else if i == count - 1 {
            condition = "else"
        } else {
            let cumulative = (0...i).map { "outputDim\($0)" }.joined(separator: " + ")
            condition = "else if (globalRow < \(cumulative))"
        }
        let prior = i == 0 ? "0u" : (0..<i).map { "outputDim\($0)" }.joined(separator: " + ")
        return """
                \(condition) {
                    weight = weight\(i); output = output\(i); outputDimension = outputDim\(i);
                    localRow = globalRow - (\(prior));
                }
        """
    }.joined(separator: "\n        ")

    return """
    kernel void \(name)(
        device const \(bt)* input              [[buffer(0)]],
        \(weightBindings)
        \(outputBindings)
        constant uint& inputDimension          [[buffer(\(1 + 2 * count))]],
        \(outputDimBindings)
        constant uint& sequenceLength          [[buffer(\(2 + 3 * count))]],
        constant uint& inputRowStride          [[buffer(\(3 + 3 * count))]],
        constant uint& outputRowStride         [[buffer(\(4 + 3 * count))]],
        uint2 gid                              [[threadgroup_position_in_grid]],
        uint tid                               [[thread_index_in_threadgroup]],
        uint tiisg                             [[thread_index_in_simdgroup]],
        uint sgitg                             [[simdgroup_index_in_threadgroup]],
        uint2 threadsPerThreadgroup            [[threads_per_threadgroup]]
    ) {
        const uint tileElements = \(tileElements);
        const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup.x / SIMD_WIDTH);
        const uint globalRow = gid.x * rowsPerThreadgroup + sgitg;
        const uint seqPos = gid.y;
        const uint totalRows = \(totalRows);
        if (globalRow >= totalRows || seqPos >= sequenceLength) return;

        device const \(wt)* weight;
        device \(bt)* output;
        uint outputDimension;
        uint localRow;
        \(branchBlocks)

        threadgroup \(bt) inputTile[tileElements];
        float sum = 0.0f;
        device const \(bt)* inputRow = input + seqPos * inputRowStride;
        device const \(wt)* weightRow = weight + localRow * inputDimension;
        for (uint base = 0; base < inputDimension; base += tileElements) {
            for (uint j = tid; j < tileElements; j += threadsPerThreadgroup.x) {
                const uint inputIndex = base + j;
                inputTile[j] = inputIndex < inputDimension ? inputRow[inputIndex] : \(bt)(0.0f);
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
            output[seqPos * outputRowStride + localRow] = \(bt)(\(storeValue("sum")));
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
        constant float& weightBias     [[buffer(8)]],
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
            float affine = \(readWeight("weight[i]")) + weightBias;
            data[offset + i] = \(bt)(float(data[offset + i]) * scale * affine);
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
        constant float& weightBias               [[buffer(8)]],
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
            float affine = \(readWeight("weight[i]")) + weightBias;
            data[offset + i] = \(bt)(float(data[offset + i]) * scale * affine);
        }
    }
    """
}

/// Generate batched QK norm for prefill (sequence mode).
///
/// Single dispatch applies per-head RMS norm to both Q and K projections.
/// Threadgroups [0..qHeadCount) → Q data/weights, [qHeadCount..total) → K data/weights.
///
/// Grid: (qHeadCount + kHeadCount, sequenceLength, 1)
/// Threadgroup: (SIMD_WIDTH, 1, 1) — one SIMD group per head.
public static func generateBatchedQKNormSequence(
    name: String,
    bufferPrecision: BufferPrecision,
    weightFormat: WeightFormat
) -> String {
    let bt = bufferPrecision.metalType
    let wt = weightFormat.bufferType
    let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
    let storeValue: (String) -> String = { expr in
        bufferPrecision.isPrefillSequencePrecision
            ? MetalSourceGenerator.sequenceStorageValue(expr, weightFormat: weightFormat)
            : expr
    }

    return """
    kernel void \(name)(
        device \(bt)* qData              [[buffer(0)]],
        device \(bt)* kData              [[buffer(1)]],
        device const \(wt)* qWeight      [[buffer(2)]],
        device const \(wt)* kWeight      [[buffer(3)]],
        constant uint& qHeadCount        [[buffer(4)]],
        constant uint& kHeadCount        [[buffer(5)]],
        constant uint& headDimension     [[buffer(6)]],
        constant float& epsilon          [[buffer(7)]],
        constant float& weightBias       [[buffer(8)]],
        constant uint& sequenceLength    [[buffer(9)]],
        constant uint& qTotalDimension   [[buffer(10)]],
        constant uint& kTotalDimension   [[buffer(11)]],
        uint2 gid                        [[threadgroup_position_in_grid]],
        uint tid                         [[thread_index_in_threadgroup]],
        uint2 threadgroupSize            [[threads_per_threadgroup]]
    ) {
        uint head = gid.x;
        uint seqPos = gid.y;
        if (seqPos >= sequenceLength) return;

        device \(bt)* data;
        device const \(wt)* weight;
        uint localHead;
        uint totalDim;

        if (head < qHeadCount) {
            data = qData; weight = qWeight; localHead = head; totalDim = qTotalDimension;
        } else {
            localHead = head - qHeadCount;
            if (localHead >= kHeadCount) return;
            data = kData; weight = kWeight; totalDim = kTotalDimension;
        }

        uint offset = seqPos * totalDim + localHead * headDimension;

        float sumSq = 0.0f;
        uint threads = threadgroupSize.x;
        for (uint i = tid; i < headDimension; i += threads) {
            float v = float(data[offset + i]);
            sumSq += v * v;
        }
        sumSq = simd_sum(sumSq);

        threadgroup float shared[32];
        if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = sumSq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float total = 0.0f;
            uint sgCount = (threads + SIMD_WIDTH - 1) / SIMD_WIDTH;
            for (uint i = 0; i < sgCount; i++) total += shared[i];
            shared[0] = rsqrt(total / float(headDimension) + epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float rms = shared[0];
        for (uint i = tid; i < headDimension; i += threads) {
            float affine = \(readWeight("weight[i]")) + weightBias;
            float normalized = float(data[offset + i]) * rms * affine;
            data[offset + i] = \(bt)(\(storeValue("normalized")));
        }
    }
    """
}

/// Fused per-head QK RMS norm + RoPE for prefill (sequence mode).
///
/// Merges two sequential dispatches — `batched_qk_rms_norm_seq_f32` + `rope_seq_f32` —
/// into a single dispatch, saving one barrier per layer. RoPE reads the values
/// written by the norm phase in the same thread (safe when pairCount % SIMD_WIDTH == 0,
/// with a device-memory barrier as a guard when the assumption is violated).
///
/// Grid: (qHeadCount + kHeadCount, sequenceLength, 1)
/// Threadgroup: (SIMD_WIDTH, 1, 1) — one SIMD group per head.
public static func generateBatchedQKNormRoPESequence(
    name: String,
    bufferPrecision: BufferPrecision,
    weightFormat: WeightFormat
) -> String {
    let bt = bufferPrecision.metalType
    let wt = weightFormat.bufferType
    let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
    let normStoreValue: (String) -> String = { expr in
        bufferPrecision.isPrefillSequencePrecision
            ? MetalSourceGenerator.sequenceStorageValue(expr, weightFormat: weightFormat)
            : expr
    }

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
        device \(bt)* qData                    [[buffer(0)]],
        device \(bt)* kData                    [[buffer(1)]],
        device const \(wt)* qWeight            [[buffer(2)]],
        device const \(wt)* kWeight            [[buffer(3)]],
        device const uint* positionAxesBuffer  [[buffer(4)]],
        constant uint& qHeadCount              [[buffer(5)]],
        constant uint& kHeadCount              [[buffer(6)]],
        constant uint& headDimension           [[buffer(7)]],
        constant uint& ropeDimension           [[buffer(8)]],
        constant float& epsilon                [[buffer(9)]],
        constant float& weightBias             [[buffer(10)]],
        constant float& ropeBase               [[buffer(11)]],
        constant uint& temporalSections        [[buffer(12)]],
        constant uint& heightSections          [[buffer(13)]],
        constant uint& widthSections           [[buffer(14)]],
        constant uint& mropeInterleaved        [[buffer(15)]],
        constant uint& sequenceLength          [[buffer(16)]],
        constant uint& qTotalDimension         [[buffer(17)]],
        constant uint& kTotalDimension         [[buffer(18)]],
        constant uint& proportionalRoPE        [[buffer(19)]],
        constant uint& qRowStride              [[buffer(20)]],
        constant uint& kRowStride              [[buffer(21)]],
        uint2 gid                              [[threadgroup_position_in_grid]],
        uint tid                               [[thread_index_in_threadgroup]],
        uint2 threadgroupSize                  [[threads_per_threadgroup]]
    ) {
        uint head = gid.x;
        uint seqPos = gid.y;
        if (seqPos >= sequenceLength) return;

        device \(bt)* data;
        device const \(wt)* weight;
        uint localHead;
        uint rowStride;

        if (head < qHeadCount) {
            data = qData; weight = qWeight; localHead = head; rowStride = qRowStride;
        } else {
            localHead = head - qHeadCount;
            if (localHead >= kHeadCount) return;
            data = kData; weight = kWeight; rowStride = kRowStride;
        }

        uint offset = seqPos * rowStride + localHead * headDimension;

        // Phase 1: per-head RMS norm (in-place).
        float sumSq = 0.0f;
        uint threads = threadgroupSize.x;
        for (uint i = tid; i < headDimension; i += threads) {
            float v = float(data[offset + i]);
            sumSq += v * v;
        }
        sumSq = simd_sum(sumSq);

        threadgroup float shared[32];
        if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = sumSq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float total = 0.0f;
            uint sgCount = (threads + SIMD_WIDTH - 1) / SIMD_WIDTH;
            for (uint i = 0; i < sgCount; i++) total += shared[i];
            shared[0] = rsqrt(total / float(headDimension) + epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float rms = shared[0];
        for (uint i = tid; i < headDimension; i += threads) {
            float affine = \(readWeight("weight[i]")) + weightBias;
            float normalized = float(data[offset + i]) * rms * affine;
            data[offset + i] = \(bt)(\(normStoreValue("normalized")));
        }

        // Cross-SIMD device-memory visibility before RoPE reads paired indices.
        threadgroup_barrier(mem_flags::mem_device);

        // Phase 2: RoPE rotation (in-place).
        const bool useProportional = proportionalRoPE != 0;
        const uint pairCount = useProportional ? (headDimension / 2) : (ropeDimension / 2);
        const uint rotatedPairs = ropeDimension / 2;
        for (uint i = tid; i < pairCount; i += threads) {
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
                const float theta = float(position) * pow(ropeBase, -2.0f * float(ropeMapping.y) / thetaDenominator);
                cosTheta = cos(theta);
                sinTheta = sin(theta);
            }
            float x0 = float(data[offset + i]);
            float x1 = float(data[offset + i + pairCount]);
            float rotated0 = x0 * cosTheta - x1 * sinTheta;
            float rotated1 = x0 * sinTheta + x1 * cosTheta;
            data[offset + i] = \(bt)(rotated0);
            data[offset + i + pairCount] = \(bt)(rotated1);
        }
    }
    """
}

}
