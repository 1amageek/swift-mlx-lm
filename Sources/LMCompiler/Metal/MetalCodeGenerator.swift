/// Generates Metal shader source code from a sequence of fusable operations.
///
/// The compiler uses this to dynamically emit fused Metal kernels from IR,
/// eliminating multiple dispatch overhead for element-wise operation chains.
///
/// ## Design
///
/// Each `FusableOp` maps to a single line (or small block) of Metal code.
/// The generator concatenates these into a complete kernel body that
/// `MLXFast.metalKernel()` wraps with the function signature.
///
/// Variables are thread-local (registers). No shared memory is used.
/// Each thread processes one element (indexed by `thread_position_in_grid.x`).
///
/// ## Usage
///
/// ```swift
/// let ops: [FusableOp] = [
///     .load("g", input: 0),
///     .load("u", input: 1),
///     .silu("a", "g"),
///     .multiply("o", "a", "u"),
///     .store("o", output: 0),
/// ]
/// let source = MetalCodeGenerator.generate(ops: ops)
/// // → "uint idx = thread_position_in_grid.x;\nT g = gate[idx]; ..."
/// ```
public enum MetalCodeGenerator {

    /// A fusable operation that maps to Metal code.
    public enum FusableOp: Sendable {

        // MARK: - Memory access

        /// Load element from input buffer: `T variable = inputN[idx]`
        case load(variable: String, input: Int)

        /// Store element to output buffer: `outputN[idx] = variable`
        case store(variable: String, output: Int)

        /// Load element from a split partition of an input buffer.
        /// `T variable = inputN[partIndex * stride + idx]` where stride = totalSize / partCount
        case splitLoad(variable: String, input: Int, partIndex: Int, partCount: Int)

        // MARK: - Unary operations

        /// SiLU activation: `dst = src / (T(1) + exp(-src))`
        case silu(dst: String, src: String)

        /// GELU activation: `dst = src * T(0.5) * (T(1) + erf(src * T(0.7071067811865476)))`
        case gelu(dst: String, src: String)

        /// ReLU activation: `dst = max(src, T(0))`
        case relu(dst: String, src: String)

        /// Sigmoid: `dst = T(1) / (T(1) + exp(-src))`
        case sigmoid(dst: String, src: String)

        /// Negation: `dst = -src`
        case negate(dst: String, src: String)

        /// Exponential: `dst = exp(src)`
        case exp(dst: String, src: String)

        // MARK: - Normalization

        /// RMS normalization: `dst = src / sqrt(mean(src²) + eps) * weight`
        /// Weight is loaded from a separate input buffer at `weightInput[idx % D]`.
        /// Requires HIDDEN_D template parameter.
        /// NOTE: This is a per-element approximation that works when each thread
        /// processes one element. For full RMSNorm the reduction across D is needed.
        /// Use `rmsNormFull` for correct implementation with threadgroup reduction.
        case rmsNormLoad(dst: String, src: String, weightInput: Int, eps: Float)

        // MARK: - Residual

        /// Residual addition: `dst = lhs + rhs`
        /// Typically used as the final op: `output = residual + op(norm(input))`
        case residualAdd(dst: String, residual: String, operand: String)

        // MARK: - Binary operations

        /// Element-wise multiply: `dst = lhs * rhs`
        case multiply(dst: String, lhs: String, rhs: String)

        /// Element-wise add: `dst = lhs + rhs`
        case add(dst: String, lhs: String, rhs: String)

        // MARK: - Convolution (fixed kernel, channel-parallel)

        /// Depthwise conv1d dot product over cache window.
        /// Reads state[0..K-2] and appends newVal as the last element.
        /// `dst = sum(state[k] * weight[k]) + newVal * weight[K-1]`
        /// state layout: [K-1, D] row-major, weight layout: [D, K] row-major.
        case convDot(dst: String, stateInput: Int, weightInput: Int, newVal: String, kernelSize: Int)

        /// Cache shift: shift state left by 1, append newVal at end.
        /// `for k in 0..<K-2: new_state[k] = state[k+1]; new_state[K-2] = newVal`
        case cacheShift(output: Int, stateInput: Int, newVal: String, kernelSize: Int)
    }

    /// Generate Metal kernel body from a sequence of fusable operations.
    ///
    /// The generated code uses `thread_position_in_grid.x` as the element index.
    /// All variables are thread-local `T` (dtype determined by template parameter).
    ///
    /// - Parameters:
    ///   - ops: sequence of operations to fuse
    ///   - inputNames: names for input buffers (must match MLXFast.metalKernel inputNames)
    ///   - outputNames: names for output buffers
    /// - Returns: Metal source code string (kernel body only)
    public static func generate(
        ops: [FusableOp],
        inputNames: [String],
        outputNames: [String]
    ) -> String {
        var lines: [String] = []
        lines.append("uint idx = thread_position_in_grid.x;")

        for op in ops {
            switch op {

            case .load(let variable, let input):
                lines.append("T \(variable) = \(inputNames[input])[idx];")

            case .store(let variable, let output):
                lines.append("\(outputNames[output])[idx] = \(variable);")

            case .splitLoad(let variable, let input, let partIndex, let partCount):
                // input is [partCount * D] contiguous. Each part is D elements.
                // For split part i: offset = i * (totalSize / partCount) + idx
                // Since we use per-element threading, D = grid size.
                // So: inputN[partIndex * D + idx] where D comes from template HIDDEN_D.
                lines.append("T \(variable) = \(inputNames[input])[\(partIndex) * HIDDEN_D + idx];")

            case .silu(let dst, let src):
                lines.append("T \(dst) = \(src) / (T(1) + exp(-\(src)));")

            case .gelu(let dst, let src):
                lines.append("T \(dst) = \(src) * T(0.5) * (T(1) + precise::erf(\(src) * T(0.7071067811865476)));")

            case .relu(let dst, let src):
                lines.append("T \(dst) = max(\(src), T(0));")

            case .sigmoid(let dst, let src):
                lines.append("T \(dst) = T(1) / (T(1) + exp(-\(src)));")

            case .negate(let dst, let src):
                lines.append("T \(dst) = -\(src);")

            case .exp(let dst, let src):
                lines.append("T \(dst) = exp(\(src));")

            case .rmsNormLoad(let dst, let src, let weightInput, let eps):
                // RMS norm is a reduction — cannot be done per-element in a single thread.
                // Instead, we emit code that reads from a pre-normed input.
                // The actual RMSNorm is called via MLXFast.rmsNorm BEFORE this kernel,
                // and the result is passed as an input buffer.
                // This case loads the normed value and applies the weight.
                lines.append("T \(dst) = \(src) * \(inputNames[weightInput])[idx % HIDDEN_D];")
                _ = eps  // eps is applied in the MLXFast.rmsNorm call

            case .residualAdd(let dst, let residual, let operand):
                lines.append("T \(dst) = \(residual) + \(operand);")

            case .multiply(let dst, let lhs, let rhs):
                lines.append("T \(dst) = \(lhs) * \(rhs);")

            case .add(let dst, let lhs, let rhs):
                lines.append("T \(dst) = \(lhs) + \(rhs);")

            case .convDot(let dst, let stateInput, let weightInput, let newVal, let K):
                // Depthwise conv1d: dot product over [state[0..K-2], newVal] * weight[0..K-1]
                // state: [K-1, D] row-major → state[k * D + idx]
                // weight: [D, K] row-major → weight[idx * K + k]
                lines.append("T \(dst) = T(0);")
                lines.append("for (int k = 0; k < \(K - 1); k++) {")
                lines.append("    \(dst) += \(inputNames[stateInput])[k * HIDDEN_D + idx] * \(inputNames[weightInput])[idx * \(K) + k];")
                lines.append("}")
                lines.append("\(dst) += \(newVal) * \(inputNames[weightInput])[idx * \(K) + \(K - 1)];")

            case .cacheShift(let output, let stateInput, let newVal, let K):
                // Shift cache left by 1, append newVal
                lines.append("for (int k = 0; k < \(K - 2); k++) {")
                lines.append("    \(outputNames[output])[k * HIDDEN_D + idx] = \(inputNames[stateInput])[(k + 1) * HIDDEN_D + idx];")
                lines.append("}")
                lines.append("\(outputNames[output])[\(K - 2) * HIDDEN_D + idx] = \(newVal);")
            }
        }

        return lines.joined(separator: "\n")
    }
}
