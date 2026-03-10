import MLX
import MLXNN

/// Quantized embedding initialized from pre-packed GGUF/MLX arrays.
///
/// Unlike `QuantizedEmbedding`, this initializer does not call `MLX.quantized()`.
/// It accepts already-packed `weight/scales/biases`, which is required for GGUF
/// direct-pack formats such as `Q6_K` that use group sizes not accepted by
/// `MLX.quantized()` placeholder construction.
final class DirectQuantizedEmbedding: Embedding, Quantized {

    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode
    let scales: MLXArray
    let biases: MLXArray?

    override var shape: (Int, Int) {
        let (embeddingCount, packedDimensions) = super.shape
        return (embeddingCount, packedDimensions * 32 / bits)
    }

    init(
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.scales = scales
        self.biases = biases
        super.init(weight: weight)
        self.freeze()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let indices = x.flattened()
        let out = dequantized(
            weight[indices],
            scales: scales[indices],
            biases: biases == nil ? nil : biases![indices],
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        return out.reshaped(shape + [-1])
    }

    override func asLinear(_ x: MLXArray) -> MLXArray {
        if groupSize >= 32 {
            return quantizedMM(
                x,
                weight,
                scales: scales,
                biases: biases,
                transpose: true,
                groupSize: groupSize,
                bits: bits,
                mode: mode
            )
        }

        let denseWeight = dequantized(
            weight,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        return matmul(x, denseWeight.T)
    }
}
