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
        weightBias: Float = 0,
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
                constant float& weightBias      [[buffer(5)]],
                constant uint& sequenceLength   [[buffer(6)]],
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
                    float affine = \(readWeight("weight[i]")) + weightBias;
                    outRow[i] = \(bt)(float(row[i]) * scale * affine);
                }
            }
            """
        } else {
            return """
            kernel void \(name)(
                device const \(bt)* input       [[buffer(0)]],
                device const \(wt)* weight      [[buffer(1)]],
                device \(bt)* output            [[buffer(2)]],
                constant uint& dimension        [[buffer(3)]],
                constant float& epsilon         [[buffer(4)]],
                constant float& weightBias      [[buffer(5)]],
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
                    float affine = \(readWeight("weight[i]")) + weightBias;
                    output[i] = \(bt)(float(input[i]) * scale * affine);
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
            device const \(bt)* input [[id(0)]];
            device const \(wt)* weight [[id(1)]];
            device \(bt)* output [[id(2)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& dimension                  [[buffer(3)]],
            constant float& epsilon                   [[buffer(4)]],
            constant float& weightBias                [[buffer(5)]],
            uint tid                                  [[thread_index_in_threadgroup]],
            uint threadgroupSize                      [[threads_per_threadgroup]]
        ) {
            float sumSquared = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float v = float(args.input[i]);
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
                float affine = \(readWeight("args.weight[i]")) + weightBias;
                args.output[i] = \(bt)(float(args.input[i]) * scale * affine);
            }
        }
        """
    }

    /// Activation function applied inside gated MLP kernels.
    public enum GatedActivation: Sendable {
        /// SiLU: x * sigmoid(x) — used by SwiGLU (Llama, LFM2)
        case silu
        /// GELU (tanh approximation): 0.5x(1+tanh(sqrt(2/π)(x+0.044715x³))) — used by GEGLU (Gemma4)
        case geluTanh
    }

    /// Generate MSL source for an elementwise kernel (SwiGLU).
    public static func generateSwiGLU(
        name: String,
        bufferPrecision: BufferPrecision,
        isSequence: Bool = true
    ) -> String {
        generateGatedActivation(name: name, bufferPrecision: bufferPrecision, activation: .silu, isSequence: isSequence)
    }

    /// Generate MSL source for a gated activation kernel.
    ///
    /// Computes `activation(gate) * up` element-wise.
    /// SwiGLU uses `.silu`, GEGLU uses `.geluTanh`.
    public static func generateGatedActivation(
        name: String,
        bufferPrecision: BufferPrecision,
        activation: GatedActivation,
        isSequence: Bool = true
    ) -> String {
        let bt = bufferPrecision.metalType
        let activationExpr = switch activation {
        case .silu:
            "g * (1.0f / (1.0f + exp(-g)))"
        case .geluTanh:
            "0.5f * g * (1.0f + precise::tanh(0.7978845608f * (g + 0.044715f * g * g * g)))"
        }

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
                float activated = \(activationExpr);
                output[idx] = \(bt)(activated * float(up[idx]));
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
                float activated = \(activationExpr);
                output[gid] = \(bt)(activated * float(up[gid]));
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
            "0.5f * gateValue * (1.0f + precise::tanh(0.7978845608f * (gateValue + 0.044715f * gateValue * gateValue * gateValue)))"
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

    public static func generateScalarMultiply(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        isSequence: Bool
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let scalar = weightFormat.readExpression("weight[0]")

        if isSequence {
            return """
            kernel void \(name)(
                device const \(bt)* input        [[buffer(0)]],
                device const \(wt)* weight       [[buffer(1)]],
                device \(bt)* output             [[buffer(2)]],
                constant uint& dimension         [[buffer(3)]],
                constant uint& sequenceLength    [[buffer(4)]],
                uint2 gid                        [[thread_position_in_grid]]
            ) {
                uint i = gid.x;
                uint seqPos = gid.y;
                if (i >= dimension || seqPos >= sequenceLength) return;
                uint idx = seqPos * dimension + i;
                float scale = \(scalar);
                output[idx] = \(bt)(float(input[idx]) * scale);
            }
            """
        }

        return """
        kernel void \(name)(
            device const \(bt)* input        [[buffer(0)]],
            device const \(wt)* weight       [[buffer(1)]],
            device \(bt)* output             [[buffer(2)]],
            constant uint& count             [[buffer(3)]],
            uint gid                         [[thread_position_in_grid]]
        ) {
            if (gid >= count) return;
            float scale = \(scalar);
            output[gid] = \(bt)(float(input[gid]) * scale);
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

    public static func generateResidualAddInPlace(
        name: String,
        bufferPrecision: BufferPrecision,
        isSequence: Bool = true
    ) -> String {
        let bt = bufferPrecision.metalType

        if isSequence {
            return """
            kernel void \(name)(
                device \(bt)* inputOutput        [[buffer(0)]],
                device const \(bt)* residual     [[buffer(1)]],
                constant uint& dimension         [[buffer(2)]],
                constant uint& sequenceLength    [[buffer(3)]],
                uint2 gid                        [[thread_position_in_grid]]
            ) {
                uint i = gid.x;
                uint seqPos = gid.y;
                if (i >= dimension || seqPos >= sequenceLength) return;
                uint idx = seqPos * dimension + i;
                inputOutput[idx] = \(bt)(float(inputOutput[idx]) + float(residual[idx]));
            }
            """
        } else {
            return """
            kernel void \(name)(
                device \(bt)* inputOutput        [[buffer(0)]],
                device const \(bt)* residual     [[buffer(1)]],
                constant uint& count             [[buffer(2)]],
                uint gid                         [[thread_position_in_grid]]
            ) {
                if (gid >= count) return;
                inputOutput[gid] = \(bt)(float(inputOutput[gid]) + float(residual[gid]));
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

    public static func generateResidualAddInPlaceArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision
    ) -> String {
        let bt = bufferPrecision.metalType
        let inputStructName = "\(name)_args"

        return """
        struct \(inputStructName) {
            device \(bt)* inputOutput [[id(0)]];
            device const \(bt)* residual [[id(1)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& count                      [[buffer(2)]],
            uint gid                                  [[thread_position_in_grid]]
        ) {
            if (gid >= count) return;
            args.inputOutput[gid] = \(bt)(float(args.inputOutput[gid]) + float(args.residual[gid]));
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

    public static func generatePackedSigmoidGate(
        name: String,
        bufferPrecision: BufferPrecision,
        isSequence: Bool = true
    ) -> String {
        let bt = bufferPrecision.metalType

        if isSequence {
            return """
            kernel void \(name)(
                device const \(bt)* input        [[buffer(0)]],
                device const \(bt)* packed       [[buffer(1)]],
                device \(bt)* output             [[buffer(2)]],
                constant uint& dimension         [[buffer(3)]],
                constant uint& headDimension     [[buffer(4)]],
                constant uint& packedHeadStride  [[buffer(5)]],
                constant uint& gateHeadOffset    [[buffer(6)]],
                constant uint& packedRowStride   [[buffer(7)]],
                constant uint& outputRowStride   [[buffer(8)]],
                constant uint& sequenceLength    [[buffer(9)]],
                uint2 gid                        [[thread_position_in_grid]]
            ) {
                uint i = gid.x;
                uint seqPos = gid.y;
                if (i >= dimension || seqPos >= sequenceLength) return;

                uint headIndex = i / headDimension;
                uint lane = i % headDimension;
                uint packedBase = seqPos * packedRowStride;
                uint outputBase = seqPos * outputRowStride;
                float g = float(packed[packedBase + headIndex * packedHeadStride + gateHeadOffset + lane]);
                output[outputBase + i] = \(bt)(float(input[outputBase + i]) * (1.0f / (1.0f + exp(-g))));
            }
            """
        }

        return """
        kernel void \(name)(
            device const \(bt)* input        [[buffer(0)]],
            device const \(bt)* packed       [[buffer(1)]],
            device \(bt)* output             [[buffer(2)]],
            constant uint& dimension         [[buffer(3)]],
            constant uint& headDimension     [[buffer(4)]],
            constant uint& packedHeadStride  [[buffer(5)]],
            constant uint& gateHeadOffset    [[buffer(6)]],
            uint gid                         [[thread_position_in_grid]]
        ) {
            if (gid >= dimension) return;
            uint headIndex = gid / headDimension;
            uint lane = gid % headDimension;
            float g = float(packed[headIndex * packedHeadStride + gateHeadOffset + lane]);
            output[gid] = \(bt)(float(input[gid]) * (1.0f / (1.0f + exp(-g))));
        }
        """
    }

    public static func generatePackedQueryExtract(
        name: String,
        bufferPrecision: BufferPrecision,
        isSequence: Bool = true
    ) -> String {
        let bt = bufferPrecision.metalType

        if isSequence {
            return """
            kernel void \(name)(
                device const \(bt)* packed       [[buffer(0)]],
                device \(bt)* output             [[buffer(1)]],
                constant uint& headCount         [[buffer(2)]],
                constant uint& headDimension     [[buffer(3)]],
                constant uint& packedRowStride   [[buffer(4)]],
                constant uint& outputRowStride   [[buffer(5)]],
                constant uint& sequenceLength    [[buffer(6)]],
                uint2 gid                        [[thread_position_in_grid]]
            ) {
                uint element = gid.x;
                uint seqPos = gid.y;
                uint totalDimension = headCount * headDimension;
                if (element >= totalDimension || seqPos >= sequenceLength) return;

                uint headIndex = element / headDimension;
                uint lane = element % headDimension;
                uint packedBase = seqPos * packedRowStride;
                uint outputBase = seqPos * outputRowStride;
                uint packedIndex = packedBase + headIndex * (2 * headDimension) + lane;
                output[outputBase + element] = packed[packedIndex];
            }
            """
        }

        return """
        kernel void \(name)(
            device const \(bt)* packed       [[buffer(0)]],
            device \(bt)* output             [[buffer(1)]],
            constant uint& headCount         [[buffer(2)]],
            constant uint& headDimension     [[buffer(3)]],
            uint gid                         [[thread_position_in_grid]]
        ) {
            uint totalDimension = headCount * headDimension;
            if (gid >= totalDimension) return;

            uint headIndex = gid / headDimension;
            uint lane = gid % headDimension;
            output[gid] = packed[headIndex * (2 * headDimension) + lane];
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
            constant float& weightBias       [[buffer(6)]],
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
                float w = \(readWeight("weight[i]")) + weightBias;
                scratch[i] = \(bt)(float(hidden[i]) * scale * w);
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
            constant float& weightBias                [[buffer(6)]],
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
                float w = \(readWeight("args.weight[i]")) + weightBias;
                args.scratch[i] = \(bt)(float(args.hidden[i]) * scale * w);
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
            constant float& weightBias       [[buffer(6)]],
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
                float w = \(readWeight("weight[i]")) + weightBias;
                scratch[i] = \(bt)(h * scale * w);
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
            constant float& weightBias                [[buffer(6)]],
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
                float w = \(readWeight("args.weight[i]")) + weightBias;
                args.scratch[i] = \(bt)(h * scale * w);
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
            constant float& weightBias       [[buffer(6)]],
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
                float w = \(readWeight("weight[i]")) + weightBias;
                output[i] = \(bt)(h * scale * w);
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
            constant float& weightBias              [[buffer(6)]],
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
                float w = \(readWeight("args.weight[i]")) + weightBias;
                args.output[i] = \(bt)(h * scale * w);
            }
        }
        """
    }

    // MARK: - Sequence-Aware Fused Kernels (Prefill)

    /// Generate fused copy + RMSNorm for prefill (sequence mode).
    ///
    /// Single dispatch replaces copy_buffer_seq_f32 + rms_norm_seq_f32:
    ///   residual[pos] = hidden[pos]       (copy)
    ///   hidden[pos]   = RMSNorm(hidden[pos])  (in-place norm)
    ///
    /// Grid: (sequenceLength, 1, 1) — one threadgroup per sequence position.
    public static func generateFusedCopyRMSNormSequence(
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
            constant uint& dimension         [[buffer(3)]],
            constant float& epsilon          [[buffer(4)]],
            constant uint& sequenceLength    [[buffer(5)]],
            constant float& weightBias       [[buffer(6)]],
            uint gid_x                       [[threadgroup_position_in_grid]],
            uint tid                         [[thread_index_in_threadgroup]],
            uint threadgroupSize             [[threads_per_threadgroup]]
        ) {
            uint seqPos = gid_x;
            if (seqPos >= sequenceLength) return;

            device \(bt)* h_row = hidden + seqPos * dimension;
            device \(bt)* r_row = residual + seqPos * dimension;

            // Pass 1: copy hidden → residual, compute sum of squares for RMSNorm
            float sumSquared = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float v = float(h_row[i]);
                r_row[i] = h_row[i];
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

            // Pass 2: write normalized output to hidden (in-place)
            float scale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float w = \(readWeight("weight[i]")) + weightBias;
                h_row[i] = \(bt)(float(h_row[i]) * scale * w);
            }
        }
        """
    }

    /// Generate fused residualAdd + copy + RMSNorm for prefill (sequence mode).
    ///
    /// Single dispatch replaces residual_add_inplace_seq_f32 + copy_buffer_seq_f32 + rms_norm_seq_f32:
    ///   h = hidden[pos] + residual[pos]   (add)
    ///   residual[pos] = h                 (copy sum for next residual block)
    ///   hidden[pos]   = RMSNorm(h)        (normalized output)
    ///
    /// Grid: (sequenceLength, 1, 1) — one threadgroup per sequence position.
    public static func generateFusedResidualAddCopyRMSNormSequence(
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
            constant uint& dimension         [[buffer(3)]],
            constant float& epsilon          [[buffer(4)]],
            constant uint& sequenceLength    [[buffer(5)]],
            constant float& weightBias       [[buffer(6)]],
            uint gid_x                       [[threadgroup_position_in_grid]],
            uint tid                         [[thread_index_in_threadgroup]],
            uint threadgroupSize             [[threads_per_threadgroup]]
        ) {
            uint seqPos = gid_x;
            if (seqPos >= sequenceLength) return;

            device \(bt)* h_row = hidden + seqPos * dimension;
            device \(bt)* r_row = residual + seqPos * dimension;

            // Pass 1: compute residual sum and RMS
            float sumSquared = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(h_row[i]) + float(r_row[i]);
                sumSquared += h * h;
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

            // Pass 2: write sum to residual, normalized to hidden
            float scale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(h_row[i]) + float(r_row[i]);
                r_row[i] = \(bt)(h);
                float w = \(readWeight("weight[i]")) + weightBias;
                h_row[i] = \(bt)(h * scale * w);
            }
        }
        """
    }

    /// Generate fused residualAdd + RMSNorm for prefill (sequence mode, no residual copy).
    ///
    /// Single dispatch replaces residual_add_inplace_seq_f32 + rms_norm_seq_f32:
    ///   h = hidden[pos] + residual[pos]   (add)
    ///   hidden[pos] = RMSNorm(h)          (normalized output)
    ///
    /// Used at final layer boundary where no subsequent residual block follows.
    /// Grid: (sequenceLength, 1, 1) — one threadgroup per sequence position.
    public static func generateFusedResidualAddRMSNormSequence(
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
            constant uint& dimension         [[buffer(3)]],
            constant float& epsilon          [[buffer(4)]],
            constant uint& sequenceLength    [[buffer(5)]],
            constant float& weightBias       [[buffer(6)]],
            uint gid_x                       [[threadgroup_position_in_grid]],
            uint tid                         [[thread_index_in_threadgroup]],
            uint threadgroupSize             [[threads_per_threadgroup]]
        ) {
            uint seqPos = gid_x;
            if (seqPos >= sequenceLength) return;

            device \(bt)* h_row = hidden + seqPos * dimension;
            device const \(bt)* r_row = residual + seqPos * dimension;

            // Pass 1: compute residual sum and RMS
            float sumSquared = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(h_row[i]) + float(r_row[i]);
                sumSquared += h * h;
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

            // Pass 2: write normalized sum to hidden
            float scale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(h_row[i]) + float(r_row[i]);
                float w = \(readWeight("weight[i]")) + weightBias;
                h_row[i] = \(bt)(h * scale * w);
            }
        }
        """
    }

    // MARK: - PreNorm Fused Kernels (Prefill)

    /// Generate fused preNorm + residualAdd + copy + outputNorm for prefill (sequence mode).
    ///
    /// Single dispatch replaces 4 operations:
    ///   Phase 1: hidden[pos] = RMSNorm_pre(hidden[pos])       — in-place pre-norm
    ///   Phase 2: h = hidden[pos] + residual[pos]              — residual add
    ///            residual[pos] = h                             — copy for next residual
    ///   Phase 3: hidden[pos] = RMSNorm_out(h)                 — output norm
    ///
    /// Buffer layout:
    ///   [0] hidden (read/write), [1] residual (read/write),
    ///   [2] preNormWeight (read), [3] outputNormWeight (read),
    ///   [4] dimension, [5] preNormEpsilon, [6] sequenceLength,
    ///   [7] preNormWeightBias, [8] outputEpsilon, [9] outputWeightBias
    ///
    /// Grid: (sequenceLength, 1, 1) — one threadgroup per sequence position.
    public static func generateFusedPreNormResidualAddCopyRMSNormSequence(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        return """
        kernel void \(name)(
            device \(bt)* hidden                  [[buffer(0)]],
            device \(bt)* residual                [[buffer(1)]],
            device const \(wt)* preNormWeight     [[buffer(2)]],
            device const \(wt)* outputNormWeight  [[buffer(3)]],
            constant uint& dimension              [[buffer(4)]],
            constant float& preNormEpsilon        [[buffer(5)]],
            constant uint& sequenceLength         [[buffer(6)]],
            constant float& preNormWeightBias     [[buffer(7)]],
            constant float& outputEpsilon         [[buffer(8)]],
            constant float& outputWeightBias      [[buffer(9)]],
            uint gid_x                            [[threadgroup_position_in_grid]],
            uint tid                              [[thread_index_in_threadgroup]],
            uint threadgroupSize                  [[threads_per_threadgroup]]
        ) {
            uint seqPos = gid_x;
            if (seqPos >= sequenceLength) return;

            device \(bt)* h_row = hidden + seqPos * dimension;
            device \(bt)* r_row = residual + seqPos * dimension;

            threadgroup float shared[32];

            // Phase 1: in-place preNorm on hidden
            float sumSq1 = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float v = float(h_row[i]);
                sumSq1 += v * v;
            }
            sumSq1 = simd_sum(sumSq1);
            uint simdIndex = tid / SIMD_WIDTH;
            if (tid % SIMD_WIDTH == 0) shared[simdIndex] = sumSq1;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0) {
                float total = 0.0f;
                uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                for (uint s = 0; s < sgCount; s++) total += shared[s];
                shared[0] = rsqrt(total / float(dimension) + preNormEpsilon);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float preScale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float v = float(h_row[i]);
                float w = \(readWeight("preNormWeight[i]")) + preNormWeightBias;
                h_row[i] = \(bt)(v * preScale * w);
            }
            threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

            // Phase 2+3: residual add + copy + output norm
            float sumSq2 = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(h_row[i]) + float(r_row[i]);
                sumSq2 += h * h;
            }
            sumSq2 = simd_sum(sumSq2);
            if (tid % SIMD_WIDTH == 0) shared[simdIndex] = sumSq2;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0) {
                float total = 0.0f;
                uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                for (uint s = 0; s < sgCount; s++) total += shared[s];
                shared[0] = rsqrt(total / float(dimension) + outputEpsilon);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float outScale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(h_row[i]) + float(r_row[i]);
                r_row[i] = \(bt)(h);
                float w = \(readWeight("outputNormWeight[i]")) + outputWeightBias;
                h_row[i] = \(bt)(h * outScale * w);
            }
        }
        """
    }

    /// Generate fused preNorm + residualAdd + outputNorm for prefill (sequence mode, no copy).
    ///
    /// Single dispatch replaces 3 operations:
    ///   Phase 1: hidden[pos] = RMSNorm_pre(hidden[pos])       — in-place pre-norm
    ///   Phase 2: h = hidden[pos] + residual[pos]              — residual add
    ///   Phase 3: hidden[pos] = RMSNorm_out(h)                 — output norm
    ///
    /// Buffer layout:
    ///   [0] hidden (read/write), [1] residual (read),
    ///   [2] preNormWeight (read), [3] outputNormWeight (read),
    ///   [4] dimension, [5] preNormEpsilon, [6] sequenceLength,
    ///   [7] preNormWeightBias, [8] outputEpsilon, [9] outputWeightBias
    ///
    /// Grid: (sequenceLength, 1, 1) — one threadgroup per sequence position.
    public static func generateFusedPreNormResidualAddRMSNormSequence(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        return """
        kernel void \(name)(
            device \(bt)* hidden                  [[buffer(0)]],
            device const \(bt)* residual          [[buffer(1)]],
            device const \(wt)* preNormWeight     [[buffer(2)]],
            device const \(wt)* outputNormWeight  [[buffer(3)]],
            constant uint& dimension              [[buffer(4)]],
            constant float& preNormEpsilon        [[buffer(5)]],
            constant uint& sequenceLength         [[buffer(6)]],
            constant float& preNormWeightBias     [[buffer(7)]],
            constant float& outputEpsilon         [[buffer(8)]],
            constant float& outputWeightBias      [[buffer(9)]],
            uint gid_x                            [[threadgroup_position_in_grid]],
            uint tid                              [[thread_index_in_threadgroup]],
            uint threadgroupSize                  [[threads_per_threadgroup]]
        ) {
            uint seqPos = gid_x;
            if (seqPos >= sequenceLength) return;

            device \(bt)* h_row = hidden + seqPos * dimension;
            device const \(bt)* r_row = residual + seqPos * dimension;

            threadgroup float shared[32];

            // Phase 1: in-place preNorm on hidden
            float sumSq1 = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float v = float(h_row[i]);
                sumSq1 += v * v;
            }
            sumSq1 = simd_sum(sumSq1);
            uint simdIndex = tid / SIMD_WIDTH;
            if (tid % SIMD_WIDTH == 0) shared[simdIndex] = sumSq1;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0) {
                float total = 0.0f;
                uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                for (uint s = 0; s < sgCount; s++) total += shared[s];
                shared[0] = rsqrt(total / float(dimension) + preNormEpsilon);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float preScale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float v = float(h_row[i]);
                float w = \(readWeight("preNormWeight[i]")) + preNormWeightBias;
                h_row[i] = \(bt)(v * preScale * w);
            }
            threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

            // Phase 2+3: residual add + output norm (no copy)
            float sumSq2 = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(h_row[i]) + float(r_row[i]);
                sumSq2 += h * h;
            }
            sumSq2 = simd_sum(sumSq2);
            if (tid % SIMD_WIDTH == 0) shared[simdIndex] = sumSq2;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0) {
                float total = 0.0f;
                uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                for (uint s = 0; s < sgCount; s++) total += shared[s];
                shared[0] = rsqrt(total / float(dimension) + outputEpsilon);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float outScale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(h_row[i]) + float(r_row[i]);
                float w = \(readWeight("outputNormWeight[i]")) + outputWeightBias;
                h_row[i] = \(bt)(h * outScale * w);
            }
        }
        """
    }

    /// Generate fused preNorm + residualAdd + copy + outputNorm for decode (single token).
    ///
    /// Single dispatch replaces 4 operations:
    ///   Phase 1: hidden[i] = RMSNorm_pre(hidden[i])    — in-place pre-norm
    ///   Phase 2: h = hidden[i] + residual[i]
    ///            hidden[i] = h; residual[i] = h         — add + copy
    ///   Phase 3: scratch[i] = RMSNorm_out(h)            — output norm to scratch
    ///
    /// Buffer layout:
    ///   [0] hidden, [1] residual, [2] preNormWeight, [3] outputNormWeight,
    ///   [4] scratch, [5] dimension, [6] preNormEpsilon,
    ///   [7] preNormWeightBias, [8] outputEpsilon, [9] outputWeightBias
    public static func generateFusedPreNormResidualAddCopyRMSNorm(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }

        return """
        kernel void \(name)(
            device \(bt)* hidden                  [[buffer(0)]],
            device \(bt)* residual                [[buffer(1)]],
            device const \(wt)* preNormWeight     [[buffer(2)]],
            device const \(wt)* outputNormWeight  [[buffer(3)]],
            device \(bt)* scratch                 [[buffer(4)]],
            constant uint& dimension              [[buffer(5)]],
            constant float& preNormEpsilon        [[buffer(6)]],
            constant float& preNormWeightBias     [[buffer(7)]],
            constant float& outputEpsilon         [[buffer(8)]],
            constant float& outputWeightBias      [[buffer(9)]],
            uint tid                              [[thread_index_in_threadgroup]],
            uint tiisg                            [[thread_index_in_simdgroup]],
            uint sgitg                            [[simdgroup_index_in_threadgroup]],
            uint threadgroupSize                  [[threads_per_threadgroup]]
        ) {
            threadgroup float shared[32];

            // Phase 1: in-place preNorm on hidden
            float sumSq1 = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float v = float(hidden[i]);
                sumSq1 += v * v;
            }
            sumSq1 = simd_sum(sumSq1);
            if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = sumSq1;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sgitg == 0) {
                const uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                float total = tiisg < sgCount ? shared[tiisg] : 0.0f;
                total = simd_sum(total);
                if (tiisg == 0) shared[0] = rsqrt(total / float(dimension) + preNormEpsilon);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float preScale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float v = float(hidden[i]);
                float w = \(readWeight("preNormWeight[i]")) + preNormWeightBias;
                hidden[i] = \(bt)(v * preScale * w);
            }
            threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

            // Phase 2+3: residual add + copy + output norm
            float sumSq2 = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(hidden[i]) + float(residual[i]);
                sumSq2 += h * h;
            }
            sumSq2 = simd_sum(sumSq2);
            if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = sumSq2;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sgitg == 0) {
                const uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                float total = tiisg < sgCount ? shared[tiisg] : 0.0f;
                total = simd_sum(total);
                if (tiisg == 0) shared[0] = rsqrt(total / float(dimension) + outputEpsilon);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float outScale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(hidden[i]) + float(residual[i]);
                hidden[i] = \(bt)(h);
                residual[i] = \(bt)(h);
                float w = \(readWeight("outputNormWeight[i]")) + outputWeightBias;
                scratch[i] = \(bt)(h * outScale * w);
            }
        }
        """
    }

    /// Generate argument table variant of fused preNorm + residualAdd + copy + outputNorm for decode.
    public static func generateFusedPreNormResidualAddCopyRMSNormArgumentTableVariant(
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
            device const \(wt)* preNormWeight [[id(2)]];
            device const \(wt)* outputNormWeight [[id(3)]];
            device \(bt)* scratch [[id(4)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& dimension                  [[buffer(5)]],
            constant float& preNormEpsilon            [[buffer(6)]],
            constant float& preNormWeightBias         [[buffer(7)]],
            constant float& outputEpsilon             [[buffer(8)]],
            constant float& outputWeightBias          [[buffer(9)]],
            uint tid                                  [[thread_index_in_threadgroup]],
            uint tiisg                                [[thread_index_in_simdgroup]],
            uint sgitg                                [[simdgroup_index_in_threadgroup]],
            uint threadgroupSize                      [[threads_per_threadgroup]]
        ) {
            threadgroup float shared[32];

            // Phase 1: in-place preNorm on hidden
            float sumSq1 = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float v = float(args.hidden[i]);
                sumSq1 += v * v;
            }
            sumSq1 = simd_sum(sumSq1);
            if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = sumSq1;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sgitg == 0) {
                const uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                float total = tiisg < sgCount ? shared[tiisg] : 0.0f;
                total = simd_sum(total);
                if (tiisg == 0) shared[0] = rsqrt(total / float(dimension) + preNormEpsilon);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float preScale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float v = float(args.hidden[i]);
                float w = \(readWeight("args.preNormWeight[i]")) + preNormWeightBias;
                args.hidden[i] = \(bt)(v * preScale * w);
            }
            threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

            // Phase 2+3: residual add + copy + output norm
            float sumSq2 = 0.0f;
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(args.hidden[i]) + float(args.residual[i]);
                sumSq2 += h * h;
            }
            sumSq2 = simd_sum(sumSq2);
            if (tid % SIMD_WIDTH == 0) shared[tid / SIMD_WIDTH] = sumSq2;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sgitg == 0) {
                const uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
                float total = tiisg < sgCount ? shared[tiisg] : 0.0f;
                total = simd_sum(total);
                if (tiisg == 0) shared[0] = rsqrt(total / float(dimension) + outputEpsilon);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float outScale = shared[0];
            for (uint i = tid; i < dimension; i += threadgroupSize) {
                float h = float(args.hidden[i]) + float(args.residual[i]);
                args.hidden[i] = \(bt)(h);
                args.residual[i] = \(bt)(h);
                float w = \(readWeight("args.outputNormWeight[i]")) + outputWeightBias;
                args.scratch[i] = \(bt)(h * outScale * w);
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
