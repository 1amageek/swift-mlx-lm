import MLX
import MLXFast

// MARK: - Protocol

/// Key-value cache for transformer attention layers.
public protocol KVCache: AnyObject {

    /// Number of tokens currently cached.
    var offset: Int { get }

    /// Maximum cache size (nil = unlimited).
    var maxSize: Int? { get }

    /// Append new keys/values, return full key/value sequences for attention.
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)

    /// Whether this cache supports trimming.
    var isTrimmable: Bool { get }

    /// Remove the last `n` tokens from the cache. Returns actual count trimmed.
    @discardableResult
    func trim(_ n: Int) -> Int

    /// Serializable state for snapshot/restore.
    var state: [MLXArray] { get set }

    /// String metadata for snapshot/restore.
    var metaState: [String] { get set }

    /// All MLXArrays that should be evaluated.
    func innerState() -> [MLXArray]

    /// Create an attention mask for the current cache state.
    func makeMask(queryLength: Int) -> MLXFast.ScaledDotProductAttentionMaskMode
}

// MARK: - KVCacheSimple

/// Simple append-only KV cache with pre-allocated growth.
final class KVCacheSimple: KVCache {

    private var keys: MLXArray?
    private var values: MLXArray?
    private var step: Int
    private(set) var offset: Int = 0

    var maxSize: Int? { nil }
    var isTrimmable: Bool { true }

    init(step: Int = 256) {
        self.step = step
    }

    func innerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }

    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let B = newKeys.dim(0)
        let heads = newKeys.dim(1)
        let headDim = newKeys.dim(3)
        let nTokens = newKeys.dim(2)

        if self.keys == nil {
            let capacity = alignUp(nTokens + step, to: step)
            let kShape = [B, heads, capacity, headDim]
            self.keys = MLXArray.zeros(kShape, dtype: newKeys.dtype)
            self.values = MLXArray.zeros(kShape, dtype: newValues.dtype)
            self.offset = 0
        }

        let currentCapacity = self.keys!.dim(2)
        let needed = offset + nTokens
        if needed > currentCapacity {
            let newCapacity = alignUp(needed + step, to: step)
            let extShape = [B, heads, newCapacity - currentCapacity, headDim]
            self.keys = concatenated([self.keys!, MLXArray.zeros(extShape, dtype: newKeys.dtype)], axis: 2)
            self.values = concatenated([self.values!, MLXArray.zeros(extShape, dtype: newValues.dtype)], axis: 2)
        }

        self.keys![0..., 0..., offset..<(offset + nTokens), 0...] = newKeys
        self.values![0..., 0..., offset..<(offset + nTokens), 0...] = newValues
        offset += nTokens

        return (self.keys![0..., 0..., 0..<offset, 0...], self.values![0..., 0..., 0..<offset, 0...])
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        let trimmed = min(n, offset)
        offset -= trimmed
        return trimmed
    }

    var state: [MLXArray] {
        get { [keys, values].compactMap { $0 } }
        set {
            guard newValue.count >= 2 else { return }
            keys = newValue[0]
            values = newValue[1]
        }
    }

    var metaState: [String] {
        get { [String(offset)] }
        set {
            if let first = newValue.first, let v = Int(first) {
                offset = v
            }
        }
    }

    func makeMask(queryLength n: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n > 1 {
            return .causal
        }
        return .none
    }

    private func alignUp(_ value: Int, to alignment: Int) -> Int {
        ((value + alignment - 1) / alignment) * alignment
    }
}

// MARK: - RotatingKVCache

/// KV cache with a maximum size. Evicts middle tokens to maintain the size limit
/// while preserving the initial `keep` tokens.
final class RotatingKVCache: KVCache {

    private var keys: MLXArray?
    private var values: MLXArray?
    private let keep: Int
    private let maxCacheSize: Int
    private let step: Int
    private var idx: Int = 0
    private(set) var offset: Int = 0

    var maxSize: Int? { maxCacheSize }
    var isTrimmable: Bool { offset < maxCacheSize }

    init(maxSize: Int, keep: Int = 0, step: Int = 256) {
        self.maxCacheSize = maxSize
        self.keep = keep
        self.step = step
    }

    func innerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }

    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let nTokens = newKeys.dim(2)

        if nTokens == 1 && offset >= maxCacheSize {
            return updateInPlace(keys: newKeys, values: newValues)
        } else {
            return updateConcat(keys: newKeys, values: newValues)
        }
    }

    private func updateInPlace(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let insertIdx = keep + ((idx - keep) % (maxCacheSize - keep))
        self.keys![0..., 0..., insertIdx..<(insertIdx + 1), 0...] = newKeys
        self.values![0..., 0..., insertIdx..<(insertIdx + 1), 0...] = newValues
        offset += 1
        idx += 1
        return (self.keys!, self.values!)
    }

    private func updateConcat(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if self.keys == nil {
            self.keys = newKeys
            self.values = newValues
        } else {
            var k = temporalOrder(self.keys!)
            var v = temporalOrder(self.values!)
            k = concatenated([k, newKeys], axis: 2)
            v = concatenated([v, newValues], axis: 2)

            let total = k.dim(2)
            if total > maxCacheSize {
                let trimCount = total - maxCacheSize
                if keep > 0 {
                    let keptK = k[0..., 0..., 0..<keep, 0...]
                    let restK = k[0..., 0..., (keep + trimCount)..., 0...]
                    k = concatenated([keptK, restK], axis: 2)
                    let keptV = v[0..., 0..., 0..<keep, 0...]
                    let restV = v[0..., 0..., (keep + trimCount)..., 0...]
                    v = concatenated([keptV, restV], axis: 2)
                } else {
                    k = k[0..., 0..., trimCount..., 0...]
                    v = v[0..., 0..., trimCount..., 0...]
                }
            }
            self.keys = k
            self.values = v
        }

        let nTokens = newKeys.dim(2)
        offset += nTokens
        idx = self.keys!.dim(2)
        return (self.keys!, self.values!)
    }

    private func temporalOrder(_ array: MLXArray) -> MLXArray {
        let total = array.dim(2)
        guard offset > total, keep < total else { return array }
        let rotateAt = keep + ((idx - keep) % (total - keep))
        if rotateAt == total { return array }
        let part1 = array[0..., 0..., 0..<keep, 0...]
        let part2 = array[0..., 0..., rotateAt..., 0...]
        let part3 = array[0..., 0..., keep..<rotateAt, 0...]
        return concatenated([part1, part2, part3], axis: 2)
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        let trimmed = min(n, offset)
        offset -= trimmed
        idx -= trimmed
        return trimmed
    }

    var state: [MLXArray] {
        get { [keys, values].compactMap { $0 } }
        set {
            guard newValue.count >= 2 else { return }
            keys = newValue[0]
            values = newValue[1]
        }
    }

    var metaState: [String] {
        get { [String(keep), String(maxCacheSize), String(step), String(offset), String(idx)] }
        set {
            guard newValue.count >= 5 else { return }
            offset = Int(newValue[3]) ?? 0
            idx = Int(newValue[4]) ?? 0
        }
    }

    func makeMask(queryLength n: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n > 1 {
            return .causal
        }
        return .none
    }
}

// MARK: - QuantizedKVCache

/// KV cache that stores keys and values in quantized form to reduce memory.
final class QuantizedKVCache: KVCache {

    private var keys: (MLXArray, MLXArray, MLXArray?)?
    private var values: (MLXArray, MLXArray, MLXArray?)?
    private let step: Int
    let groupSize: Int
    let bits: Int
    private(set) var offset: Int = 0

    var maxSize: Int? { nil }
    var isTrimmable: Bool { true }

    init(groupSize: Int = 64, bits: Int = 8) {
        self.groupSize = groupSize
        self.bits = bits
        self.step = 256
    }

    func innerState() -> [MLXArray] {
        var arrays: [MLXArray] = []
        if let k = keys {
            arrays.append(k.0)
            arrays.append(k.1)
            if let b = k.2 { arrays.append(b) }
        }
        if let v = values {
            arrays.append(v.0)
            arrays.append(v.1)
            if let b = v.2 { arrays.append(b) }
        }
        return arrays
    }

    /// Quantize and store new keys/values. Returns dequantized full sequences.
    func updateQuantized(keys newKeys: MLXArray, values newValues: MLXArray)
        -> ((MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?))
    {
        let (kq, ks, kb) = MLX.quantized(newKeys, groupSize: groupSize, bits: bits)
        let (vq, vs, vb) = MLX.quantized(newValues, groupSize: groupSize, bits: bits)
        let newK = (kq, ks, kb as MLXArray?)
        let newV = (vq, vs, vb as MLXArray?)

        if self.keys == nil {
            self.keys = newK
            self.values = newV
        } else {
            self.keys = (
                concatenated([self.keys!.0, newK.0], axis: 2),
                concatenated([self.keys!.1, newK.1], axis: 2),
                optConcat(self.keys!.2, newK.2, axis: 2)
            )
            self.values = (
                concatenated([self.values!.0, newV.0], axis: 2),
                concatenated([self.values!.1, newV.1], axis: 2),
                optConcat(self.values!.2, newV.2, axis: 2)
            )
        }

        offset += newKeys.dim(2)
        return (self.keys!, self.values!)
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("Use updateQuantized(keys:values:) for QuantizedKVCache")
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        let trimmed = min(n, offset)
        offset -= trimmed
        return trimmed
    }

    var state: [MLXArray] {
        get {
            var arrays: [MLXArray] = []
            if let k = keys { arrays += [k.0, k.1]; if let b = k.2 { arrays.append(b) } }
            if let v = values { arrays += [v.0, v.1]; if let b = v.2 { arrays.append(b) } }
            return arrays
        }
        set {
            guard newValue.count >= 4 else { return }
            keys = (newValue[0], newValue[1], newValue.count > 4 ? newValue[2] : nil)
            let vStart = newValue.count > 4 ? 3 : 2
            values = (newValue[vStart], newValue[vStart + 1],
                     newValue.count > vStart + 2 ? newValue[vStart + 2] : nil)
        }
    }

    var metaState: [String] {
        get { [String(step), String(offset), String(groupSize), String(bits)] }
        set {
            if newValue.count >= 2 {
                offset = Int(newValue[1]) ?? 0
            }
        }
    }

    func makeMask(queryLength n: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n > 1 {
            return .causal
        }
        return .none
    }

    private func optConcat(_ a: MLXArray?, _ b: MLXArray?, axis: Int) -> MLXArray? {
        guard let a, let b else { return a ?? b }
        return concatenated([a, b], axis: axis)
    }
}

// MARK: - Factory

/// Create a cache array for each layer of a model.
func createKVCaches(
    layerCount: Int,
    parameters: GenerateParameters
) -> [KVCache] {
    (0..<layerCount).map { layerIndex in
        if let maxSize = parameters.maxKVSize {
            return RotatingKVCache(maxSize: maxSize)
        }
        if let bits = parameters.kvBits, layerIndex >= parameters.quantizedKVStart {
            return QuantizedKVCache(groupSize: parameters.kvGroupSize, bits: bits)
        }
        return KVCacheSimple()
    }
}
