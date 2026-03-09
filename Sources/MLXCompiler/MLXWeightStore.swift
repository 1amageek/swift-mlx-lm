import MLX
import SwiftLM

/// Typed weight storage that provides MLXArray lookup by ParameterSlot.
///
/// Wraps `BoundWeights` (which uses opaque `TensorData.storage`) and extracts
/// concrete MLXArrays for use by the executor.
///
/// - Note: Marked `@unchecked Sendable` because `MLXArray` is not declared
///   `Sendable`. Weight storage is immutable after construction and only
///   accessed through read-only lookups, so concurrent access is safe.
public struct MLXWeightStore: @unchecked Sendable {

    private let weights: [ParameterSlot: MLXArray]

    /// Initialize from `BoundWeights`, extracting MLXArrays from TensorData storage.
    public init(boundWeights: BoundWeights) throws {
        var converted: [ParameterSlot: MLXArray] = [:]
        converted.reserveCapacity(boundWeights.tensors.count)
        for (slot, tensorData) in boundWeights.tensors {
            guard let array = tensorData.storage as? MLXArray else {
                throw CompilerError.invalidWeightStorage(
                    slot, "Expected MLXArray, got \(type(of: tensorData.storage))")
            }
            converted[slot] = array
        }
        self.weights = converted
    }

    /// Initialize directly from a pre-built dictionary.
    public init(weights: [ParameterSlot: MLXArray]) {
        self.weights = weights
    }

    /// Look up a weight, returning nil if not found.
    public func get(_ slot: ParameterSlot) -> MLXArray? {
        weights[slot]
    }

    /// Look up a weight, throwing if not found.
    public func require(_ slot: ParameterSlot) throws -> MLXArray {
        guard let w = weights[slot] else {
            throw CompilerError.missingWeight(slot)
        }
        return w
    }

    /// All stored parameter slots.
    public var allSlots: Dictionary<ParameterSlot, MLXArray>.Keys {
        weights.keys
    }

    /// Number of stored weights.
    public var count: Int { weights.count }
}
