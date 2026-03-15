@preconcurrency import MLX
import MLXFast

/// Fused Metal kernel for SiLU activation + gate multiply.
///
/// Combines `silu(gate) * up` into a single Metal kernel dispatch.
/// Eliminates 2 intermediate dispatches per MLP layer.
///
/// ## Operations fused
///
/// 1. SiLU activation: gate / (1 + exp(-gate))
/// 2. Element-wise multiply with up projection output
///
/// ## Grid layout
///
/// - `thread_position_in_grid.x` = element index (0..<N)
/// - Each thread processes one element independently
enum FusedSiLUGateKernel {

    // MARK: - Kernel factory

    private static let kernel: MLXFastKernel = MLXFast.metalKernel(
        name: "fused_silu_gate",
        inputNames: ["gate", "up"],
        outputNames: ["out"],
        source: metalSource
    )

    // MARK: - Public API

    /// Execute fused SiLU(gate) * up kernel.
    ///
    /// - Parameters:
    ///   - gate: gate projection output (any shape, will be flattened)
    ///   - up: up projection output (same shape as gate)
    /// - Returns: silu(gate) * up (same shape as inputs)
    static func call(gate: MLXArray, up: MLXArray) -> MLXArray {
        let shape = gate.shape
        let N = shape.reduce(1, *)
        let dtype = gate.dtype

        let results = kernel(
            [gate.reshaped(-1), up.reshaped(-1)],
            template: [("T", dtype)],
            grid: (N, 1, 1),
            threadGroup: (min(N, 1024), 1, 1),
            outputShapes: [shape],
            outputDTypes: [dtype]
        )

        return results[0]
    }

    // MARK: - Metal source

    private static let metalSource = """
    uint idx = thread_position_in_grid.x;

    T g = gate[idx];
    T u = up[idx];

    // SiLU(g) = g * sigmoid(g) = g / (1 + exp(-g))
    T silu_g = g / (T(1) + exp(-g));

    out[idx] = silu_g * u;
    """
}
