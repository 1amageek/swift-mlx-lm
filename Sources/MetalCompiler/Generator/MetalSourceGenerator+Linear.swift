extension MetalSourceGenerator {
// MARK: - Linear Kernels

    /// Supported MPP GEMM M-tile sizes. Callers emit one kernel per size and
    /// select the appropriate pipeline at dispatch time based on runtime
    /// sequence length, trading off padding waste (small seq → small tile) and
    /// per-threadgroup work (long seq → large tile).
    public static let mppGEMMTileSizes: [Int] = [16, 32, 64]

    /// Default M-tile used when only a single variant is needed (long-seq baseline).
    public static let mppGEMMDefaultTileSize: Int = 64

    /// Suffix for a tile-specific kernel variant name.
    public static func mppGEMMVariantName(baseName: String, tileSize: Int) -> String {
        "\(baseName)_mtile\(tileSize)"
    }

    public static func generateMPPGEMM(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        mTile: Int = mppGEMMDefaultTileSize
    ) -> String {
        let bt = bufferPrecision.metalType
        let tensorWeightType: String = switch weightFormat {
        case .bfloat16:
            "bfloat"
        case .float16:
            "half"
        case .float32:
            "float"
        case .quantized2Bit, .quantized3Bit, .quantized4Bit, .quantized5Bit, .quantized6Bit, .quantized8Bit:
            bt
        }

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
            constant uint& inputRowStride    [[buffer(6)]],
            uint2 tgid [[threadgroup_position_in_grid]]
        ) {
            using namespace mpp::tensor_ops;
            (void)inputRowStride;

            // Pad sequence extent to the M-tile boundary so that edge
            // threadgroups never slice beyond the tensor declaration.
            // The backing buffers are allocated for maximumSequenceLength,
            // which is always >= paddedSeqLen.
            constexpr uint M_TILE = \(mTile);
            const uint paddedSeqLen = ((sequenceLength + M_TILE - 1) / M_TILE) * M_TILE;

            auto A = tensor<device \(bt), dextents<int32_t, 2>, tensor_inline>(
                input, dextents<int32_t, 2>(inputDimension, paddedSeqLen));
            auto B = tensor<device \(tensorWeightType), dextents<int32_t, 2>, tensor_inline>(
                weight, dextents<int32_t, 2>(inputDimension, outputDimension));
            auto C = tensor<device \(bt), dextents<int32_t, 2>, tensor_inline>(
                output, dextents<int32_t, 2>(outputDimension, paddedSeqLen));

            constexpr auto desc = matmul2d_descriptor(
                M_TILE, 32, dynamic_length_v<int>,
                false, true, false,
                matmul2d_descriptor::mode::multiply);
            matmul2d<desc, execution_simdgroups<4>> op;

            auto mA = A.slice(0, tgid.y * M_TILE);
            auto mB = B.slice(0, tgid.x * 32);
            auto mC = C.slice(tgid.x * 32, tgid.y * M_TILE);
            op.run(mA, mB, mC);
        }
        """
    }

    /// Generate MSL source for a batched MPP GEMM kernel with BF16 or native dense weights.
    ///
    /// All `count` projections share the same input `A`. Each projection has its
    /// own weight matrix and output buffer. tgid.x linearly indexes N-tiles across
    /// all projections; each threadgroup maps to one projection based on
    /// cumulative N-tile counts. tgid.y indexes the M-tile (sequence position).
    ///
    /// Emitting one kernel for multiple projections removes barriers between
    /// them and reduces dispatch encoding cost on the CPU side, which is the
    /// dominant cost for short-sequence prefill on Apple Silicon.
    ///
    /// Assumptions:
    /// - Every `outputDim_i` is a multiple of 32 (N_TILE).
    /// - Count is 2 or 3 (used for gate/up and Q/K/V respectively).
    public static func generateBatchedMPPGEMM(
        name: String,
        count: Int,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        mTile: Int = mppGEMMDefaultTileSize
    ) -> String {
        precondition(count >= 2, "batched MPP GEMM requires count >= 2")
        let bt = bufferPrecision.metalType
        let tensorWeightType: String = switch weightFormat {
        case .bfloat16:
            "bfloat"
        case .float16:
            "half"
        case .float32:
            "float"
        case .quantized2Bit, .quantized3Bit, .quantized4Bit, .quantized5Bit, .quantized6Bit, .quantized8Bit:
            bt
        }

        // Buffer binding layout:
        //   0           : input
        //   1..count    : weight_i
        //   count+1..2*count : output_i
        //   2*count+1   : inputDimension
        //   2*count+2..3*count+1 : outputDim_i
        //   3*count+2   : sequenceLength
        //   3*count+3   : inputRowStride
        let weightBindings = (0..<count).map { i in
            "device \(tensorWeightType)* weight\(i)      [[buffer(\(1 + i))]],"
        }.joined(separator: "\n    ")
        let outputBindings = (0..<count).map { i in
            "device \(bt)* output\(i)         [[buffer(\(1 + count + i))]],"
        }.joined(separator: "\n    ")
        let outputDimBindings = (0..<count).map { i in
            "constant uint& outputDim\(i)     [[buffer(\(2 + 2 * count + i))]],"
        }.joined(separator: "\n    ")

        // Per-projection run blocks. Each block constructs local B/C tensor
        // slices from the projection-local N-tile index and runs matmul2d.
        var runBlocks: [String] = []
        for i in 0..<count {
            let priorTilesExpr: String
            if i == 0 {
                priorTilesExpr = "0u"
            } else {
                priorTilesExpr = (0..<i).map { "(outputDim\($0) / 32)" }.joined(separator: " + ")
            }
            // nTileLimit excludes the early-return case — we only enter this
            // branch when nTile is in this projection's range.
            let conditionExpr: String
            if i == 0 {
                conditionExpr = "if (nTile < outputDim0 / 32)"
            } else if i == count - 1 {
                conditionExpr = "else"
            } else {
                let cumulative = (0...i).map { "(outputDim\($0) / 32)" }.joined(separator: " + ")
                conditionExpr = "else if (nTile < \(cumulative))"
            }
            runBlocks.append("""
                \(conditionExpr) {
                    const uint localNTile = nTile - (\(priorTilesExpr));
                    auto B = tensor<device \(tensorWeightType), dextents<int32_t, 2>, tensor_inline>(
                        weight\(i), dextents<int32_t, 2>(inputDimension, outputDim\(i)));
                    auto C = tensor<device \(bt), dextents<int32_t, 2>, tensor_inline>(
                        output\(i), dextents<int32_t, 2>(outputDim\(i), paddedSeqLen));
                    auto mB = B.slice(0, localNTile * 32);
                    auto mC = C.slice(localNTile * 32, tgid.y * M_TILE);
                    op.run(mA, mB, mC);
                }
                """)
        }
        let runBody = runBlocks.joined(separator: "\n            ")

        return """
        #include <metal_stdlib>
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        using namespace metal;

        kernel void \(name)(
            device \(bt)* input              [[buffer(0)]],
            \(weightBindings)
            \(outputBindings)
            constant uint& inputDimension    [[buffer(\(1 + 2 * count))]],
            \(outputDimBindings)
            constant uint& sequenceLength    [[buffer(\(2 + 3 * count))]],
            constant uint& inputRowStride    [[buffer(\(3 + 3 * count))]],
            uint2 tgid [[threadgroup_position_in_grid]]
        ) {
            using namespace mpp::tensor_ops;
            (void)inputRowStride;

            constexpr uint M_TILE = \(mTile);
            constexpr uint N_TILE = 32;
            const uint paddedSeqLen = ((sequenceLength + M_TILE - 1) / M_TILE) * M_TILE;

            auto A = tensor<device \(bt), dextents<int32_t, 2>, tensor_inline>(
                input, dextents<int32_t, 2>(inputDimension, paddedSeqLen));

            constexpr auto desc = matmul2d_descriptor(
                M_TILE, N_TILE, dynamic_length_v<int>,
                false, true, false,
                matmul2d_descriptor::mode::multiply);
            matmul2d<desc, execution_simdgroups<4>> op;

            auto mA = A.slice(0, tgid.y * M_TILE);

            const uint nTile = tgid.x;
            \(runBody)
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
            constant uint& inputRowStride         [[buffer(6)]],
            uint2 gid                              [[threadgroup_position_in_grid]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]]
        ) {
            const uint rowsPerThreadgroup = 2;
            const uint row = gid.x * rowsPerThreadgroup + sgitg;
            const uint seqPos = gid.y;
            if (row >= outputDimension || seqPos >= sequenceLength) return;

            float sum = 0.0f;
            device const \(bt)* inputRow = input + seqPos * inputRowStride;
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
}
