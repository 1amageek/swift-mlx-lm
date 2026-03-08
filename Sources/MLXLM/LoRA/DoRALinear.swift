import Foundation
import MLX
import MLXLinalg
import MLXNN

/// DoRA forward pass shared by DoRALinear and QDoRALinear.
private func doraForward(
    x: MLXArray, y: MLXArray,
    weight: MLXArray, bias: MLXArray?,
    loraA: MLXArray, loraB: MLXArray,
    scale: Float, magnitude: MLXArray
) -> MLXArray {
    let z = matmul(matmul(x, loraA), loraB)
    var out = y + (scale * z).asType(x.dtype)

    let adapted = weight + matmul(scale * loraB.T, loraA.T)
    let denom = norm(adapted, axis: 1)
    out *= (magnitude / denom).asType(x.dtype)

    if let bias {
        return out + bias
    } else {
        return out
    }
}

/// Fuse DoRA weights into the base weight.
private func doraFuse(
    weight: MLXArray,
    loraA: MLXArray, loraB: MLXArray,
    scale: Float, magnitude: MLXArray
) -> MLXArray {
    let loraA = loraA.T.asType(weight.dtype)
    let loraB = (scale * loraB.T).asType(weight.dtype)

    var adapted = weight + matmul(loraB, loraA)
    let denom = norm(adapted, axis: 1)
    adapted *= (magnitude / denom).reshaped([-1, 1])

    return adapted
}

/// Filter out DoRA-specific trainable parameter keys.
private func doraFreezeKeys(from module: Module, keys: [String]?) -> [String] {
    return
        (keys
        ?? module.filterMap(filter: type(of: module).filterLocalParameters)
            .flattened()
            .map { $0.0 })
        .filter { !["lora_a", "lora_b", "m"].contains($0) }
}

/// DoRA (Weight-Decomposed Low-Rank Adaptation) adapter for `Linear` layers.
///
/// Adds magnitude normalization on top of LoRA:
/// `output = (magnitude / norm(W + scale * B^T A^T)) * (Wx + scale * x A B)`
///
/// - SeeAlso: [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
class DoRALinear: Linear, LoRALayer {

    let scale: Float

    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray
    @ParameterInfo(key: "m") var magnitude: MLXArray

    required init(linear: Linear, rank: Int = 8, scale: Float = 20.0) {
        let (_, inputDimensions) = linear.shape
        let loraScale: Float = 1 / Foundation.sqrt(Float(inputDimensions))

        self.scale = scale
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, linear.shape.0])
        self._magnitude.wrappedValue = MLXLinalg.norm(linear.weight, axis: 1)

        super.init(weight: linear.weight, bias: linear.bias)

        freeze()
    }

    /// Wrap a Linear (or QuantizedLinear) in a DoRA adapter.
    static func from(linear: Linear, rank: Int = 8, scale: Float = 20.0) -> LoRALayer {
        if let quantized = linear as? QuantizedLinear {
            return QDoRALinear(linear: quantized, rank: rank, scale: scale)
        }
        return DoRALinear(linear: linear, rank: rank, scale: scale)
    }

    override func freeze(
        recursive: Bool = true, keys: [String]? = nil, strict: Bool = false
    ) throws {
        let keys = doraFreezeKeys(from: self, keys: keys)
        try super.freeze(recursive: recursive, keys: keys, strict: strict)
    }

    func fused() -> Module {
        Linear(
            weight: doraFuse(
                weight: weight, loraA: loraA, loraB: loraB,
                scale: scale, magnitude: magnitude),
            bias: bias
        )
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = matmul(x, weight.T)
        return doraForward(
            x: x, y: y,
            weight: weight, bias: bias,
            loraA: loraA, loraB: loraB,
            scale: scale, magnitude: magnitude
        )
    }
}

/// DoRA adapter for `QuantizedLinear` layers.
class QDoRALinear: QuantizedLinear, LoRALayer {

    let scale: Float

    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray
    @ParameterInfo(key: "m") var magnitude: MLXArray

    required init(linear: QuantizedLinear, rank: Int = 8, scale: Float = 20.0) {
        let (_, inputDimensions) = linear.shape
        let loraScale: Float = 1 / Foundation.sqrt(Float(inputDimensions))

        self.scale = scale
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, linear.shape.0])
        self._magnitude.wrappedValue = MLXLinalg.norm(linear.dequantizedWeight, axis: 1)

        super.init(
            weight: linear.weight, bias: linear.bias,
            scales: linear.scales, biases: linear.biases,
            groupSize: linear.groupSize, bits: linear.bits
        )

        freeze()
    }

    override func freeze(
        recursive: Bool = true, keys: [String]? = nil, strict: Bool = false
    ) throws {
        let keys = doraFreezeKeys(from: self, keys: keys)
        try super.freeze(recursive: recursive, keys: keys, strict: strict)
    }

    func fused() -> Module {
        QuantizedLinear(
            weight: doraFuse(
                weight: dequantizedWeight, loraA: loraA, loraB: loraB,
                scale: scale, magnitude: magnitude),
            bias: bias, groupSize: groupSize, bits: bits
        )
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = quantizedMM(
            x, weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits)
        return doraForward(
            x: x, y: y,
            weight: dequantizedWeight, bias: bias,
            loraA: loraA, loraB: loraB,
            scale: scale, magnitude: magnitude
        )
    }
}
