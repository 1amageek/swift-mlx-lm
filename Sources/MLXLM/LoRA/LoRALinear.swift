import Foundation
import MLX
import MLXNN

/// LoRA adapter for `Linear` layers.
///
/// Adds low-rank matrices A and B such that:
/// `output = base_forward(x) + scale * (x @ A @ B)`
///
/// - SeeAlso: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
class LoRALinear: Linear, LoRALayer {

    let scale: Float

    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray

    required init(
        _ inputDimensions: Int, _ outputDimensions: Int,
        rank: Int = 8, scale: Float = 20.0, linear: Linear
    ) {
        self.scale = scale

        let loraScale: Float = 1 / Foundation.sqrt(Float(inputDimensions))
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputDimensions])

        super.init(weight: linear.weight, bias: linear.bias)

        freeze()
    }

    override func freeze(
        recursive: Bool = true, keys: [String]? = nil, strict: Bool = false
    ) throws {
        let keys =
            (keys ?? self.filterMap(filter: Self.filterLocalParameters).flattened().map { $0.0 })
            .filter { $0 != "lora_a" && $0 != "lora_b" }
        try super.freeze(recursive: recursive, keys: keys, strict: strict)
    }

    /// Wrap a Linear (or QuantizedLinear) in a LoRA adapter.
    static func from(linear: Linear, rank: Int = 8, scale: Float = 20.0) -> LoRALayer {
        if let quantized = linear as? QuantizedLinear {
            return QLoRALinear.from(linear: quantized, rank: rank, scale: scale)
        }
        let (outputDimensions, inputDimensions) = linear.shape
        return LoRALinear(
            inputDimensions, outputDimensions, rank: rank, scale: scale, linear: linear)
    }

    /// Fuse LoRA weights into the base weight and return a plain Linear.
    func fused() -> Module {
        let dtype = weight.dtype
        let loraB = (scale * loraB.T).asType(dtype)
        let loraA = loraA.T.asType(dtype)
        return Linear(weight: weight + matmul(loraB, loraA), bias: bias)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = super.callAsFunction(x.asType(weight.dtype))
        let z = matmul(matmul(x, self.loraA), self.loraB)
        return y + scale * z
    }
}

/// LoRA adapter for `QuantizedLinear` layers (QLoRA).
///
/// - SeeAlso: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
class QLoRALinear: QuantizedLinear, LoRALayer {

    let scale: Float

    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray

    required init(
        _ inputDimensions: Int, _ outputDimensions: Int,
        rank: Int = 8, scale: Float = 20.0, linear: QuantizedLinear
    ) {
        self.scale = scale

        let loraScale: Float = 1 / Foundation.sqrt(Float(inputDimensions))
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputDimensions])

        super.init(
            weight: linear.weight, bias: linear.bias,
            scales: linear.scales, biases: linear.biases,
            groupSize: linear.groupSize, bits: linear.bits)

        freeze()
    }

    override func freeze(
        recursive: Bool = true, keys: [String]? = nil, strict: Bool = false
    ) throws {
        let keys =
            (keys ?? self.filterMap(filter: Self.filterLocalParameters).flattened().map { $0.0 })
            .filter { $0 != "lora_a" && $0 != "lora_b" }
        try super.freeze(recursive: recursive, keys: keys, strict: strict)
    }

    /// Wrap a QuantizedLinear in a QLoRA adapter.
    static func from(
        linear: QuantizedLinear, rank: Int = 8, scale: Float = 20.0
    ) -> LoRALayer {
        let (outputDimensions, inputDimensions) = linear.shape
        return QLoRALinear(
            inputDimensions, outputDimensions, rank: rank, scale: scale, linear: linear)
    }

    /// Fuse LoRA weights into the dequantized base, then re-quantize.
    func fused() -> Module {
        let weight = dequantizedWeight
        let dtype = weight.dtype
        let loraB = (scale * loraB.T).asType(dtype)
        let loraA = loraA.T.asType(dtype)
        return QuantizedLinear(
            weight: weight + matmul(loraB, loraA),
            bias: bias,
            groupSize: groupSize,
            bits: bits
        )
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = super.callAsFunction(x.asType(scales.dtype))
        let z = matmul(matmul(x, self.loraA), self.loraB)
        return y + scale * z
    }
}
