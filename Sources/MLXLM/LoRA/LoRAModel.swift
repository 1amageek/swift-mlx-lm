import MLX
import MLXNN

/// Protocol for models that support LoRA adapter injection.
///
/// Conforming models expose their transformer layers so that
/// `LoRAContainer` can replace Linear modules with LoRA variants.
protocol LoRAModel {

    /// Transformer layers eligible for LoRA adaptation.
    var loraLayers: [Module] { get }

    /// Default Linear layer keys to apply LoRA adapters to.
    ///
    /// Used when `LoRAConfiguration.loraParameters.keys` is nil.
    var loraDefaultKeys: [String] { get }
}

extension LoRAModel {

    /// By default, apply LoRA to all Linear layers found in `loraLayers`.
    var loraDefaultKeys: [String] {
        let namedModules = loraLayers.flatMap { $0.namedModules() }
        let linearKeys = namedModules.compactMap { key, module in
            if module is Linear {
                return key
            } else {
                return nil
            }
        }
        let unique = Set(linearKeys)
        return Array(unique)
    }
}

/// A module that includes a LoRA adapter and can be converted
/// back to its original, unadapted form.
protocol LoRALayer: Module {

    /// Returns a module with the LoRA adapter permanently fused into the base weights.
    func fused() -> Module

    /// Returns the original module without the LoRA adapter.
    func reverted() -> Module
}

/// Default `reverted()` implementation for LoRA layers that inherit from Linear.
extension LoRALayer where Self: Linear {
    func reverted() -> Module {
        if let quantized = self as? QuantizedLinear {
            return QuantizedLinear(
                weight: quantized.weight, bias: quantized.bias,
                scales: quantized.scales, biases: quantized.biases,
                groupSize: quantized.groupSize, bits: quantized.bits
            )
        } else {
            return Linear(weight: weight, bias: bias)
        }
    }
}

/// Helper to access the dequantized weight matrix from a QuantizedLinear layer.
extension QuantizedLinear {
    var dequantizedWeight: MLXArray {
        dequantized(
            weight,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits
        )
    }
}
