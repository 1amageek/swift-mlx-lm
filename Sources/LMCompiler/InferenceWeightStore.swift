import MLX
import SwiftLM

/// Quantization-aware weight store for the inference compiler.
///
/// Unlike `MLXWeightStore` which extracts bare `MLXArray` from `BoundWeights`
/// (losing quantization metadata), this store preserves `MLXTensorStorage`
/// variants so the compiler can select `matmul` vs `quantizedMatmul` at
/// compile time.
public struct InferenceWeightStore: @unchecked Sendable {

    private let weights: [ParameterSlot: MLXTensorStorage]

    /// Initialize from a pre-built dictionary.
    public init(weights: [ParameterSlot: MLXTensorStorage]) {
        self.weights = weights
    }

    /// Initialize from `BoundWeights`, extracting `MLXTensorStorage` or `MLXArray`.
    ///
    /// If `TensorData.storage` is `MLXTensorStorage`, it is used directly.
    /// If `TensorData.storage` is `MLXArray`, it is wrapped as `.dense`.
    /// Otherwise, an error is thrown.
    public init(boundWeights: BoundWeights) throws {
        var converted: [ParameterSlot: MLXTensorStorage] = [:]
        converted.reserveCapacity(boundWeights.tensors.count)
        for (slot, tensorData) in boundWeights.tensors {
            if let storage = tensorData.storage as? MLXTensorStorage {
                converted[slot] = storage
            } else if let array = tensorData.storage as? MLXArray {
                converted[slot] = .dense(array)
            } else {
                throw CompilerError.invalidWeightStorage(
                    slot, "Expected MLXTensorStorage or MLXArray, got \(type(of: tensorData.storage))")
            }
        }
        self.weights = converted
    }

    /// Look up a weight, returning nil if not found.
    public func get(_ slot: ParameterSlot) -> MLXTensorStorage? {
        weights[slot]
    }

    /// Look up a weight, throwing if not found.
    public func require(_ slot: ParameterSlot) throws -> MLXTensorStorage {
        guard let w = weights[slot] else {
            throw CompilerError.missingWeight(slot)
        }
        return w
    }

    /// Look up a weight that must be dense (e.g., norm scales, embedding tables).
    ///
    /// Non-quantizable parameters like RMSNorm weights and embedding tables
    /// should always be dense. This method extracts the `MLXArray` directly.
    public func requireDense(_ slot: ParameterSlot) throws -> MLXArray {
        guard let w = weights[slot] else {
            throw CompilerError.missingWeight(slot)
        }
        switch w {
        case .dense(let array):
            return array
        case .affineQuantized:
            throw CompilerError.invalidWeightStorage(
                slot, "Expected dense weight, got affineQuantized")
        }
    }

    /// Look up a dense weight, returning nil if not found.
    public func getDense(_ slot: ParameterSlot) -> MLXArray? {
        guard let w = weights[slot] else { return nil }
        switch w {
        case .dense(let array):
            return array
        case .affineQuantized:
            return nil
        }
    }

    /// All stored parameter slots.
    public var allSlots: Dictionary<ParameterSlot, MLXTensorStorage>.Keys {
        weights.keys
    }

    /// Number of stored weights.
    public var count: Int { weights.count }
}
