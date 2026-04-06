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

}
