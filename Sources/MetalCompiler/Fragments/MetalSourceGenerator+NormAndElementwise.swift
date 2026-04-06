import LMIR

extension MetalSourceGenerator {
    /// Generate MSL source for a reduction kernel such as RMSNorm or LayerNorm.
    ///
    /// The generated kernel:
    /// 1. Reads input from the buffer using `bufferPrecision` type.
    /// 2. Reads weight values using `weightFormat` conversion when applicable.
    /// 3. Performs SIMD reduction across `dimension`.
    /// 4. Writes output using `bufferPrecision` type.
    ///
    /// - Parameters:
    ///   - name: Kernel function name emitted into the generated source.
    ///   - dimension: Logical reduction dimension for the kernel.
    ///   - epsilon: Epsilon used by normalization-style reductions.
    ///   - bufferPrecision: Buffer element precision used for inputs and outputs.
    ///   - weightFormat: Weight element precision used by scale reads.
    ///   - isSequence: `true` for prefill kernels operating on `[seqLen × dim]`, `false` for single-token decode kernels.
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
            return """
            kernel void \(name)(
                device \(bt)* data              [[buffer(0)]],
                device const \(wt)* weight      [[buffer(1)]],
                constant uint& dimension        [[buffer(2)]],
                constant float& epsilon         [[buffer(3)]],
                uint tid                        [[thread_index_in_threadgroup]],
                uint threadgroupSize            [[threads_per_threadgroup]]
            ) {
                float sumSquared = 0.0f;
                for (uint i = tid; i < dimension; i += threadgroupSize) {
                    float v = float(data[i]);
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
                    data[i] = \(bt)(float(data[i]) * scale * \(readWeight("weight[i]")));
                }
            }
            """
        }
    }

    public static func generateReductionArgumentTableVariant(
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
            constant uint& dimension                  [[buffer(2)]],
            constant float& epsilon                   [[buffer(3)]],
            uint tid                                  [[thread_index_in_threadgroup]],
            uint threadgroupSize                      [[threads_per_threadgroup]]
        ) {
            float sumSquared = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float v = float(args.data[i]);
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
                args.data[i] = \(bt)(float(args.data[i]) * scale * \(readWeight("args.weight[i]")));
            }
        }
        """
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

    public static func generatePerLayerInputModulation(
        name: String,
        bufferPrecision: BufferPrecision,
        activation: ActivationKind,
        isSequence: Bool
    ) -> String {
        let bt = bufferPrecision.metalType
        let activated = switch activation {
        case .custom(let kind) where kind == "gelu_pytorch_tanh" || kind == "gelu_new" || kind == "gelu_fast":
            "0.5f * gateValue * (1.0f + tanh(0.7978845608f * (gateValue + 0.044715f * gateValue * gateValue * gateValue)))"
        case .gelu:
            "0.5f * gateValue * (1.0f + erf(gateValue * 0.70710678118f))"
        default:
            "gateValue"
        }

        if isSequence {
            return """
            kernel void \(name)(
                device const \(bt)* gate         [[buffer(0)]],
                device const float* perLayer     [[buffer(1)]],
                device \(bt)* output             [[buffer(2)]],
                constant uint& dimension         [[buffer(3)]],
                constant uint& sequenceLength    [[buffer(4)]],
                uint2 gid                        [[thread_position_in_grid]]
            ) {
                uint i = gid.x;
                uint seqPos = gid.y;
                if (i >= dimension || seqPos >= sequenceLength) return;
                uint idx = seqPos * dimension + i;
                float gateValue = float(gate[idx]);
                float activatedGate = \(activated);
                output[idx] = \(bt)(activatedGate * perLayer[idx]);
            }
            """
        }

        return """
        kernel void \(name)(
            device const \(bt)* gate         [[buffer(0)]],
            device const float* perLayer     [[buffer(1)]],
            device \(bt)* output             [[buffer(2)]],
            constant uint& dimension         [[buffer(3)]],
            uint gid                         [[thread_position_in_grid]]
        ) {
            if (gid >= dimension) return;
            float gateValue = float(gate[gid]);
            float activatedGate = \(activated);
            output[gid] = \(bt)(activatedGate * perLayer[gid]);
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

    public static func generateHiddenCopyFromFloat(
        name: String,
        bufferPrecision: BufferPrecision
    ) -> String {
        let bt = bufferPrecision.metalType
        return """
        kernel void \(name)(
            device \(bt)* hidden               [[buffer(0)]],
            device const float* source         [[buffer(1)]],
            constant uint& count               [[buffer(2)]],
            uint gid                           [[thread_position_in_grid]]
        ) {
            if (gid >= count) return;
            hidden[gid] = \(bt)(source[gid]);
        }
        """
    }

    public static func generateHiddenAddFromFloat(
        name: String,
        bufferPrecision: BufferPrecision
    ) -> String {
        let bt = bufferPrecision.metalType
        return """
        kernel void \(name)(
            device \(bt)* hidden               [[buffer(0)]],
            device const float* delta          [[buffer(1)]],
            constant uint& count               [[buffer(2)]],
            uint gid                           [[thread_position_in_grid]]
        ) {
            if (gid >= count) return;
            hidden[gid] = \(bt)(float(hidden[gid]) + delta[gid]);
        }
        """
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

    public static func generateResidualAddArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision
    ) -> String {
        let bt = bufferPrecision.metalType
        let inputStructName = "\(name)_args"

        return """
        struct \(inputStructName) {
            device const \(bt)* input [[id(0)]];
            device const \(bt)* residual [[id(1)]];
            device \(bt)* output [[id(2)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& count                      [[buffer(3)]],
            uint gid                                  [[thread_position_in_grid]]
        ) {
            if (gid >= count) return;
            args.output[gid] = \(bt)(float(args.input[gid]) + float(args.residual[gid]));
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
            uint tiisg                       [[thread_index_in_simdgroup]],
            uint sgitg                       [[simdgroup_index_in_threadgroup]],
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

            if (sgitg == 0) {
                const uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                float total = tiisg < sgCount ? shared[tiisg] : 0.0f;
                total = simd_sum(total);
                if (tiisg == 0) {
                shared[0] = rsqrt(total / float(dimension) + epsilon);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                scratch[i] = \(bt)(float(hidden[i]) * scale * \(readWeight("weight[i]")));
            }
        }
        """
    }

    public static func generateFusedCopyRMSNormArgumentTableVariant(
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
            device const \(bt)* hidden [[id(0)]];
            device \(bt)* residual [[id(1)]];
            device const \(wt)* weight [[id(2)]];
            device \(bt)* scratch [[id(3)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& dimension                  [[buffer(4)]],
            constant float& epsilon                   [[buffer(5)]],
            uint tid                                  [[thread_index_in_threadgroup]],
            uint tiisg                                [[thread_index_in_simdgroup]],
            uint sgitg                                [[simdgroup_index_in_threadgroup]],
            uint threadgroupSize                      [[threads_per_threadgroup]]
        ) {
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                args.residual[i] = args.hidden[i];
            }
            threadgroup_barrier(mem_flags::mem_device);

            float sumSquared = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float v = float(args.hidden[i]);
                sumSquared += v * v;
            }
            sumSquared = simd_sum(sumSquared);

            threadgroup float shared[32];
            if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = sumSquared;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sgitg == 0) {
                const uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                float total = tiisg < sgCount ? shared[tiisg] : 0.0f;
                total = simd_sum(total);
                if (tiisg == 0) {
                    shared[0] = rsqrt(total / float(dimension) + epsilon);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                args.scratch[i] = \(bt)(float(args.hidden[i]) * scale * \(readWeight("args.weight[i]")));
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
            uint tiisg                       [[thread_index_in_simdgroup]],
            uint sgitg                       [[simdgroup_index_in_threadgroup]],
            uint threadgroupSize             [[threads_per_threadgroup]]
        ) {
            // Compute RMS on the unrounded residual sum to avoid accumulating
            // an extra F16 rounding step at every pre-norm block.
            float sumSquared = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(hidden[i]) + float(residual[i]);
                sumSquared += h * h;
            }
            sumSquared = simd_sum(sumSquared);

            threadgroup float shared[32];
            if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = sumSquared;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sgitg == 0) {
                const uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                float total = tiisg < sgCount ? shared[tiisg] : 0.0f;
                total = simd_sum(total);
                if (tiisg == 0) {
                shared[0] = rsqrt(total / float(dimension) + epsilon);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(hidden[i]) + float(residual[i]);
                hidden[i] = \(bt)(h);
                residual[i] = \(bt)(h);
                scratch[i] = \(bt)(h * scale * \(readWeight("weight[i]")));
            }
        }
        """
    }

    public static func generateFusedResidualAddCopyRMSNormArgumentTableVariant(
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
            device \(bt)* hidden [[id(0)]];
            device \(bt)* residual [[id(1)]];
            device const \(wt)* weight [[id(2)]];
            device \(bt)* scratch [[id(3)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& dimension                  [[buffer(4)]],
            constant float& epsilon                   [[buffer(5)]],
            uint tid                                  [[thread_index_in_threadgroup]],
            uint tiisg                                [[thread_index_in_simdgroup]],
            uint sgitg                                [[simdgroup_index_in_threadgroup]],
            uint threadgroupSize                      [[threads_per_threadgroup]]
        ) {
            float sumSquared = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(args.hidden[i]) + float(args.residual[i]);
                sumSquared += h * h;
            }
            sumSquared = simd_sum(sumSquared);

            threadgroup float shared[32];
            if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = sumSquared;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sgitg == 0) {
                const uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                float total = tiisg < sgCount ? shared[tiisg] : 0.0f;
                total = simd_sum(total);
                if (tiisg == 0) {
                    shared[0] = rsqrt(total / float(dimension) + epsilon);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(args.hidden[i]) + float(args.residual[i]);
                args.hidden[i] = \(bt)(h);
                args.residual[i] = \(bt)(h);
                args.scratch[i] = \(bt)(h * scale * \(readWeight("args.weight[i]")));
            }
        }
        """
    }

    // MARK: - Fused Residual Add + RMS Norm (no copy)

    /// Generate fused residual add + RMS norm kernel (no copy to residual).
    /// Used at model end where no next residual block follows.
    public static func generateFusedResidualAddRMSNorm(
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
            device const \(bt)* residual     [[buffer(1)]],
            device const \(wt)* weight       [[buffer(2)]],
            device \(bt)* output             [[buffer(3)]],
            constant uint& dimension         [[buffer(4)]],
            constant float& epsilon          [[buffer(5)]],
            uint tid                         [[thread_index_in_threadgroup]],
            uint tiisg                       [[thread_index_in_simdgroup]],
            uint sgitg                       [[simdgroup_index_in_threadgroup]],
            uint threadgroupSize             [[threads_per_threadgroup]]
        ) {
            // Compute RMS on the unrounded residual sum to avoid a redundant
            // activation quantization step before normalization.
            float sumSquared = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(hidden[i]) + float(residual[i]);
                sumSquared += h * h;
            }
            sumSquared = simd_sum(sumSquared);

            threadgroup float shared[32];
            if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = sumSquared;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sgitg == 0) {
                const uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                float total = tiisg < sgCount ? shared[tiisg] : 0.0f;
                total = simd_sum(total);
                if (tiisg == 0) {
                shared[0] = rsqrt(total / float(dimension) + epsilon);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(hidden[i]) + float(residual[i]);
                hidden[i] = \(bt)(h);
                output[i] = \(bt)(h * scale * \(readWeight("weight[i]")));
            }
        }
        """
    }

    public static func generateFusedResidualAddRMSNormArgumentTableVariant(
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
            device \(bt)* hidden [[id(0)]];
            device const \(bt)* residual [[id(1)]];
            device const \(wt)* weight [[id(2)]];
            device \(bt)* output [[id(3)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args        [[buffer(\(argumentBufferIndex))]],
            constant uint& dimension                 [[buffer(4)]],
            constant float& epsilon                 [[buffer(5)]],
            uint tid                                [[thread_index_in_threadgroup]],
            uint tiisg                              [[thread_index_in_simdgroup]],
            uint sgitg                              [[simdgroup_index_in_threadgroup]],
            uint threadgroupSize                    [[threads_per_threadgroup]]
        ) {
            float sumSquared = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(args.hidden[i]) + float(args.residual[i]);
                sumSquared += h * h;
            }
            sumSquared = simd_sum(sumSquared);

            threadgroup float shared[32];
            if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = sumSquared;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sgitg == 0) {
                const uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                float total = tiisg < sgCount ? shared[tiisg] : 0.0f;
                total = simd_sum(total);
                if (tiisg == 0) {
                    shared[0] = rsqrt(total / float(dimension) + epsilon);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(args.hidden[i]) + float(args.residual[i]);
                args.hidden[i] = \(bt)(h);
                args.output[i] = \(bt)(h * scale * \(readWeight("args.weight[i]")));
            }
        }
        """
    }

    // MARK: - Batched GEMV

    /// Generate fused gate_proj + up_proj + SwiGLU kernel for decode.
    ///
    /// Each row computes both branch projections from the shared input vector,
    /// applies `silu(gate) * up`, and writes the activated intermediate directly
    /// to scratch[0]. This avoids storing the two branch projections separately.

}
