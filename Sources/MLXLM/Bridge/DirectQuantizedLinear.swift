import MLX
import MLXNN

/// Quantized linear layer initialized from pre-packed GGUF/MLX arrays.
///
/// Unlike `QuantizedLinear`, this type can execute GGUF-native layouts that
/// MLX's current `quantizedMM` kernels do not support directly, notably
/// `Q6_K` (`groupSize = 16`). The packed representation stays intact in
/// storage; unsupported kernels fall back to a transient dequantize + matmul
/// in `callAsFunction(_:)`.
final class DirectQuantizedLinear: QuantizedLinear {

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        if groupSize >= 32 {
            return super.callAsFunction(x)
        }

        let denseWeight = dequantized(
            weight,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )

        if let bias {
            return addMM(bias, x, denseWeight.T)
        } else {
            return matmul(x, denseWeight.T)
        }
    }
}
