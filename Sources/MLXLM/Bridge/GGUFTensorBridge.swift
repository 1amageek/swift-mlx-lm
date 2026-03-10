import Foundation
import MLX
import GGUFParser
import MLXCompiler

/// Result of converting a GGUF tensor to MLX format.
enum ConvertedTensor {
    /// Unquantized F16 tensor (for embeddings, norms, unsupported quant types).
    case float16(MLXArray)

    /// Directly packed MLX quantized tensor — no F16 intermediate.
    ///
    /// `weight` is packed UInt32, `scales` and `biases` are Float16.
    /// Ready to load into `QuantizedLinear` via `model.update(parameters:)`.
    case quantized(weight: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int, bits: Int)
}

/// Converts GGUF tensor data to MLXArray values.
///
/// All 2D weight tensors with supported quantization types are directly packed
/// to MLX native affine format (UInt32 weights + Float16 scales/biases).
///
/// Three packing strategies:
/// - **Tier 0-1 (Direct)**: GGUF affine blocks → MLX affine. Zero quality loss.
///   Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q8_0, Q8_1, Q5_0, Q5_1, Q8_K, Q2_K, Q3_K, TQ2_0.
/// - **Tier 2-3 (Re-quantize)**: Non-linear LUT/grid → decode → per-group min-max affine.
///   IQ4_NL, IQ4_XS, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M.
/// - **Tier 4 (Ternary)**: Ternary trit → decode → 2-bit affine. TQ1_0.
///
/// 1D tensors (norms, biases) always fall back to F16.
struct GGUFTensorBridge {

    init() {}

    /// Convert a GGUF tensor directly to MLX native quantized format.
    ///
    /// All supported GGUF quantization types (23 types) are packed to MLX affine format.
    /// Only 1D tensors and unquantized types (f32, f16, bf16) fall back to F16.
    func convertDirect(tensor: GGUFTensorInfo, data: Data) throws -> ConvertedTensor {
        let qtype = tensor.quantizationType
        let shape = tensor.dimensions.reversed().map { Int($0) }

        // Only 2D weight tensors can be directly packed for QuantizedLinear.
        // 1D tensors (norms, biases) always go to F16.
        guard shape.count >= 2 else {
            return .float16(try convertToFloat16(qtype: qtype, data: data, shape: shape))
        }

        switch qtype {
        // Tier 0: Existing direct affine packing
        case .q4_0:
            return packQ4_0(data: data, shape: shape)
        case .q4_1:
            return packQ4_1(data: data, shape: shape)
        case .q4_K:
            return packQ4_K(data: data, shape: shape)
        case .q5_K:
            return packQ5_K(data: data, shape: shape)
        case .q6_K:
            return packQ6_K(data: data, shape: shape)
        case .q8_0:
            return packQ8_0(data: data, shape: shape)
        case .q8_1:
            return packQ8_1(data: data, shape: shape)

        // Tier 1: New direct affine packing (zero loss)
        case .q5_0:
            return packQ5_0(data: data, shape: shape)
        case .q5_1:
            return packQ5_1(data: data, shape: shape)
        case .q8_K:
            return packQ8_K(data: data, shape: shape)
        case .q2_K:
            return packQ2_K(data: data, shape: shape)
        case .q3_K:
            return packQ3_K(data: data, shape: shape)
        case .tq2_0:
            return packTQ2_0(data: data, shape: shape)

        // Tier 2: LUT decode → affine re-quantize
        case .iq4_NL:
            return packIQ4_NL(data: data, shape: shape)
        case .iq4_XS:
            return packIQ4_XS(data: data, shape: shape)

        // Tier 3: Grid decode → affine re-quantize
        case .iq2_XXS:
            return packIQ2_XXS(data: data, shape: shape)
        case .iq2_XS:
            return packIQ2_XS(data: data, shape: shape)
        case .iq2_S:
            return packIQ2_S(data: data, shape: shape)
        case .iq3_XXS:
            return packIQ3_XXS(data: data, shape: shape)
        case .iq3_S:
            return packIQ3_S(data: data, shape: shape)
        case .iq1_S:
            return packIQ1_S(data: data, shape: shape)
        case .iq1_M:
            return packIQ1_M(data: data, shape: shape)

        // Tier 4: Ternary decode → affine re-quantize
        case .tq1_0:
            return packTQ1_0(data: data, shape: shape)

        // Unquantized types and integer types → F16
        default:
            return .float16(try convertToFloat16(qtype: qtype, data: data, shape: shape))
        }
    }

    /// Convert a GGUF tensor to `MLXTensorStorage`, preserving quantization metadata.
    ///
    /// For directly packable types (Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q8_0, Q8_1), returns
    /// `.affineQuantized` with pre-packed weights ready for `quantizedMM`.
    /// For other types, returns `.dense` with an F16 array.
    func convertToTensorStorage(tensor: GGUFTensorInfo, data: Data) throws -> MLXTensorStorage {
        let result = try convertDirect(tensor: tensor, data: data)
        switch result {
        case .float16(let array):
            return .dense(array)
        case .quantized(let weight, let scales, let biases, let groupSize, let bits):
            let shape = tensor.dimensions.reversed().map { Int($0) }
            return .affineQuantized(AffineQuantizedTensor(
                logicalShape: shape,
                packedWeight: weight,
                scales: scales,
                zeroBiases: biases,
                groupSize: groupSize,
                bits: bits,
                origin: originFromQuantType(tensor.quantizationType)
            ))
        }
    }

    /// Map GGUF quantization type to diagnostic origin tag.
    private func originFromQuantType(_ qtype: GGUFQuantizationType) -> QuantizationOrigin {
        switch qtype {
        case .q4_0: return .ggufQ4_0
        case .q4_1: return .ggufQ4_1
        case .q4_K: return .ggufQ4_K
        case .q5_K: return .ggufQ5_K
        case .q6_K: return .ggufQ6_K
        case .q8_0: return .ggufQ8_0
        case .q8_1: return .ggufQ8_1
        case .q5_0: return .ggufQ5_0
        case .q5_1: return .ggufQ5_1
        case .q8_K: return .ggufQ8_K
        case .q2_K: return .ggufQ2_K
        case .q3_K: return .ggufQ3_K
        case .tq2_0: return .ggufTQ2_0
        case .iq4_NL: return .ggufIQ4_NL
        case .iq4_XS: return .ggufIQ4_XS
        case .iq2_XXS: return .ggufIQ2_XXS
        case .iq2_XS: return .ggufIQ2_XS
        case .iq2_S: return .ggufIQ2_S
        case .iq3_XXS: return .ggufIQ3_XXS
        case .iq3_S: return .ggufIQ3_S
        case .iq1_S: return .ggufIQ1_S
        case .iq1_M: return .ggufIQ1_M
        case .tq1_0: return .ggufTQ1_0
        default: return .unknown
        }
    }

    /// Convert a GGUF tensor to an F16 MLXArray (legacy path, used as fallback).
    func convert(tensor: GGUFTensorInfo, data: Data) throws -> MLXArray {
        let qtype = tensor.quantizationType
        let shape = tensor.dimensions.reversed().map { Int($0) }
        return try convertToFloat16(qtype: qtype, data: data, shape: shape)
    }

    private func convertToFloat16(
        qtype: GGUFQuantizationType, data: Data, shape: [Int]
    ) throws -> MLXArray {
        switch qtype {
        case .f32:
            return loadFloat32(data: data, shape: shape)
        case .f16:
            return loadFloat16(data: data, shape: shape)
        case .bf16:
            return loadBFloat16(data: data, shape: shape)
        case .q4_0:
            return try dequantizeQ4_0(data: data, shape: shape)
        case .q8_0:
            return try dequantizeQ8_0(data: data, shape: shape)
        case .q4_K:
            return try dequantizeQ4_K(data: data, shape: shape)
        case .q6_K:
            return try dequantizeQ6_K(data: data, shape: shape)
        case .q2_K:
            return try dequantizeQ2_K(data: data, shape: shape)
        case .q3_K:
            return try dequantizeQ3_K(data: data, shape: shape)
        case .q4_1:
            return try dequantizeQ4_1(data: data, shape: shape)
        case .q5_0:
            return try dequantizeQ5_0(data: data, shape: shape)
        case .q5_1:
            return try dequantizeQ5_1(data: data, shape: shape)
        case .q5_K:
            return try dequantizeQ5_K(data: data, shape: shape)
        case .q8_1:
            return try dequantizeQ8_1(data: data, shape: shape)
        case .q8_K:
            return try dequantizeQ8_K(data: data, shape: shape)
        case .iq4_NL:
            return try dequantizeIQ4_NL(data: data, shape: shape)
        case .iq4_XS:
            return try dequantizeIQ4_XS(data: data, shape: shape)
        case .iq2_XXS:
            return try dequantizeIQ2_XXS(data: data, shape: shape)
        case .iq2_XS:
            return try dequantizeIQ2_XS(data: data, shape: shape)
        case .iq2_S:
            return try dequantizeIQ2_S(data: data, shape: shape)
        case .iq3_XXS:
            return try dequantizeIQ3_XXS(data: data, shape: shape)
        case .iq3_S:
            return try dequantizeIQ3_S(data: data, shape: shape)
        case .iq1_S:
            return try dequantizeIQ1_S(data: data, shape: shape)
        case .iq1_M:
            return try dequantizeIQ1_M(data: data, shape: shape)
        case .tq1_0:
            return try dequantizeTQ1_0(data: data, shape: shape)
        case .tq2_0:
            return try dequantizeTQ2_0(data: data, shape: shape)
        default:
            throw GGUFLoadError.unsupportedQuantization(qtype.rawValue)
        }
    }

    // MARK: - Direct GGUF → MLX Quantized Packing
    //
    // These methods convert GGUF quantized blocks directly to MLX's native
    // packed UInt32 format + Float16 scales/biases, avoiding F16 intermediates.
    //
    // MLX 4-bit packing: 8 values per UInt32, LSB-first.
    //   UInt32 = v[0] | (v[1]<<4) | (v[2]<<8) | ... | (v[7]<<28)
    //
    // MLX 8-bit packing: 4 values per UInt32, LSB-first.
    //   UInt32 = v[0] | (v[1]<<8) | (v[2]<<16) | (v[3]<<24)
    //
    // MLX affine dequantization: value = q * scale + bias

    /// Pack Q4_0 → MLX 4-bit, groupSize=32.
    ///
    /// GGUF: value = (q - 8) * scale → MLX: value = q * scale + (-8 * scale)
    private func packQ4_0(data: Data, shape: [Int]) -> ConvertedTensor {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 32

        var packed = [UInt32](repeating: 0, count: totalElements / 8)
        var scales = [Float](repeating: 0, count: blockCount)
        var biases = [Float](repeating: 0, count: blockCount)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 18
                let scale = readFloat16(bytes, offset)
                scales[block] = scale
                biases[block] = -8.0 * scale

                let packedBase = block * 4
                let qs = offset + 2
                packed[packedBase + 0] = packLowNibbles8(bytes, qs, 0)
                packed[packedBase + 1] = packLowNibbles8(bytes, qs, 8)
                packed[packedBase + 2] = packHighNibbles8(bytes, qs, 0)
                packed[packedBase + 3] = packHighNibbles8(bytes, qs, 8)
            }
        }

        return makeQuantizedResult(
            packed: packed, scales: scales, biases: biases,
            shape: shape, groupSize: 32, bits: 4)
    }

    /// Pack Q4_1 → MLX 4-bit, groupSize=32.
    ///
    /// GGUF: value = q * d + m → MLX: value = q * d + m (direct mapping)
    private func packQ4_1(data: Data, shape: [Int]) -> ConvertedTensor {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 32

        var packed = [UInt32](repeating: 0, count: totalElements / 8)
        var scales = [Float](repeating: 0, count: blockCount)
        var biases = [Float](repeating: 0, count: blockCount)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 20
                scales[block] = readFloat16(bytes, offset)
                biases[block] = readFloat16(bytes, offset + 2)

                let packedBase = block * 4
                let qs = offset + 4
                packed[packedBase + 0] = packLowNibbles8(bytes, qs, 0)
                packed[packedBase + 1] = packLowNibbles8(bytes, qs, 8)
                packed[packedBase + 2] = packHighNibbles8(bytes, qs, 0)
                packed[packedBase + 3] = packHighNibbles8(bytes, qs, 8)
            }
        }

        return makeQuantizedResult(
            packed: packed, scales: scales, biases: biases,
            shape: shape, groupSize: 32, bits: 4)
    }

    /// Pack Q4_K → MLX 4-bit, groupSize=32.
    ///
    /// GGUF Q4_K layout: 4 chunks of 32 bytes per 256-element super-block.
    /// Each 32-byte chunk encodes 64 elements:
    ///   - Low nibbles of 32 bytes → first 32 elements (scale[chunk*2])
    ///   - High nibbles of 32 bytes → next 32 elements (scale[chunk*2+1])
    ///
    /// MLX affine: value = q * scale + bias, where bias = -min.
    private func packQ4_K(data: Data, shape: [Int]) -> ConvertedTensor {
        let totalElements = shape.reduce(1, *)
        let superBlockCount = totalElements / 256

        var packed = [UInt32](repeating: 0, count: totalElements / 8)
        var scales = [Float](repeating: 0, count: totalElements / 32)
        var biases = [Float](repeating: 0, count: totalElements / 32)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for sb in 0..<superBlockCount {
                let offset = sb * 144
                let d = readFloat16(bytes, offset)
                let dmin = readFloat16(bytes, offset + 2)
                let scOffset = offset + 4

                // Extract 8 sub-block scales and mins (get_scale_min_k4)
                var subScales = [Float](repeating: 0, count: 8)
                var subMins = [Float](repeating: 0, count: 8)
                extractScaleMinK4(bytes: bytes, offset: scOffset, d: d, dmin: dmin,
                                  scales: &subScales, mins: &subMins)

                let qsBase = offset + 16
                // 4 chunks of 32 bytes, each encoding 64 elements
                for chunk in 0..<4 {
                    let qs = qsBase + chunk * 32

                    // Low nibbles → group (chunk*2): 32 elements
                    let groupLow = sb * 8 + chunk * 2
                    scales[groupLow] = subScales[chunk * 2]
                    biases[groupLow] = -subMins[chunk * 2]
                    let packedLow = sb * 32 + (chunk * 2) * 4
                    packed[packedLow + 0] = packLowNibbles8(bytes, qs, 0)
                    packed[packedLow + 1] = packLowNibbles8(bytes, qs, 8)
                    packed[packedLow + 2] = packLowNibbles8(bytes, qs, 16)
                    packed[packedLow + 3] = packLowNibbles8(bytes, qs, 24)

                    // High nibbles → group (chunk*2+1): 32 elements
                    let groupHigh = sb * 8 + chunk * 2 + 1
                    scales[groupHigh] = subScales[chunk * 2 + 1]
                    biases[groupHigh] = -subMins[chunk * 2 + 1]
                    let packedHigh = sb * 32 + (chunk * 2 + 1) * 4
                    packed[packedHigh + 0] = packHighNibbles8(bytes, qs, 0)
                    packed[packedHigh + 1] = packHighNibbles8(bytes, qs, 8)
                    packed[packedHigh + 2] = packHighNibbles8(bytes, qs, 16)
                    packed[packedHigh + 3] = packHighNibbles8(bytes, qs, 24)
                }
            }
        }

        return makeQuantizedResult(
            packed: packed, scales: scales, biases: biases,
            shape: shape, groupSize: 32, bits: 4)
    }

    /// Pack Q8_0 → MLX 8-bit, groupSize=32.
    ///
    /// GGUF: value = q_signed * scale → MLX: value = q_unsigned * scale + (-128 * scale)
    /// where q_unsigned = q_signed + 128
    private func packQ8_0(data: Data, shape: [Int]) -> ConvertedTensor {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 32

        var packed = [UInt32](repeating: 0, count: totalElements / 4)
        var scales = [Float](repeating: 0, count: blockCount)
        var biases = [Float](repeating: 0, count: blockCount)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 34
                let scale = readFloat16(bytes, offset)
                scales[block] = scale
                biases[block] = -128.0 * scale

                let packedBase = block * 8
                let qs = offset + 2
                // Pack 32 signed int8 as unsigned (add 128) into 8 UInt32
                for k in 0..<8 {
                    let b0 = UInt32(UInt8(bitPattern: Int8(bitPattern: bytes[qs + k * 4 + 0]) &+ 127 &+ 1))
                    let b1 = UInt32(UInt8(bitPattern: Int8(bitPattern: bytes[qs + k * 4 + 1]) &+ 127 &+ 1))
                    let b2 = UInt32(UInt8(bitPattern: Int8(bitPattern: bytes[qs + k * 4 + 2]) &+ 127 &+ 1))
                    let b3 = UInt32(UInt8(bitPattern: Int8(bitPattern: bytes[qs + k * 4 + 3]) &+ 127 &+ 1))
                    packed[packedBase + k] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
                }
            }
        }

        return makeQuantizedResult(
            packed: packed, scales: scales, biases: biases,
            shape: shape, groupSize: 32, bits: 8)
    }

    // MARK: - Packing Helpers

    /// Pack 8 low nibbles from consecutive bytes into one UInt32.
    @inline(__always)
    private func packLowNibbles8(
        _ bytes: UnsafeBufferPointer<UInt8>, _ base: Int, _ start: Int
    ) -> UInt32 {
        var v: UInt32 = 0
        for j in 0..<8 {
            v |= UInt32(bytes[base + start + j] & 0x0F) << (j * 4)
        }
        return v
    }

    /// Pack 8 high nibbles from consecutive bytes into one UInt32.
    @inline(__always)
    private func packHighNibbles8(
        _ bytes: UnsafeBufferPointer<UInt8>, _ base: Int, _ start: Int
    ) -> UInt32 {
        var v: UInt32 = 0
        for j in 0..<8 {
            v |= UInt32((bytes[base + start + j] >> 4) & 0x0F) << (j * 4)
        }
        return v
    }

    /// Pack quantized unsigned values into UInt32 words, LSB-first.
    ///
    /// MLX expects quantized weights packed little-endian within each UInt32:
    /// the first logical value occupies the least-significant bits.
    @inline(__always)
    private func packUnsignedValues(_ values: [UInt8], bits: Int) -> [UInt32] {
        let totalBits = values.count * bits
        let wordCount = totalBits / 32
        var words = [UInt32](repeating: 0, count: wordCount)

        for (index, rawValue) in values.enumerated() {
            let value = UInt32(rawValue)
            let bitIndex = index * bits
            let wordIndex = bitIndex / 32
            let bitOffset = bitIndex % 32

            words[wordIndex] |= value << bitOffset

            let spill = bitOffset + bits - 32
            if spill > 0 {
                words[wordIndex + 1] |= value >> (bits - spill)
            }
        }

        return words
    }

    /// Extract Q4_K / Q5_K scale and min values from 12-byte packed encoding.
    private func extractScaleMinK4(
        bytes: UnsafeBufferPointer<UInt8>, offset q: Int,
        d: Float, dmin: Float,
        scales: inout [Float], mins: inout [Float]
    ) {
        for j in 0..<8 {
            if j < 4 {
                scales[j] = Float(bytes[q + j] & 63) * d
                mins[j] = Float(bytes[q + j + 4] & 63) * dmin
            } else {
                let sc = (bytes[q + j + 4] & 0x0F) | ((bytes[q + j - 4] >> 6) << 4)
                let mn = (bytes[q + j + 4] >> 4) | ((bytes[q + j] >> 6) << 4)
                scales[j] = Float(sc) * d
                mins[j] = Float(mn) * dmin
            }
        }
    }

    /// Pack Q5_K → MLX 5-bit affine, groupSize=32.
    ///
    /// GGUF: value = scale * q - min, where q ∈ [0, 31]
    /// MLX:  value = q * scale + (-min)
    private func packQ5_K(data: Data, shape: [Int]) -> ConvertedTensor {
        let totalElements = shape.reduce(1, *)
        let superBlockCount = totalElements / 256
        let groupCount = totalElements / 32

        var packed = [UInt32](repeating: 0, count: groupCount * 5)
        var scales = [Float](repeating: 0, count: groupCount)
        var biases = [Float](repeating: 0, count: groupCount)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<superBlockCount {
                let offset = block * 176
                let d = readFloat16(bytes, offset)
                let dmin = readFloat16(bytes, offset + 2)
                let qOffset = offset + 4
                let qhOffset = offset + 16
                let qsOffset = offset + 48

                var blockScales = [Float](repeating: 0, count: 8)
                var blockMins = [Float](repeating: 0, count: 8)
                extractScaleMinK4(
                    bytes: bytes, offset: qOffset, d: d, dmin: dmin,
                    scales: &blockScales, mins: &blockMins
                )

                for chunk in 0..<4 {
                    let qsBase = qsOffset + chunk * 32
                    var lowValues = [UInt8](repeating: 0, count: 32)
                    var highValues = [UInt8](repeating: 0, count: 32)

                    for l in 0..<32 {
                        let byte = bytes[qsBase + l]
                        let qh = bytes[qhOffset + l]

                        let low = Int(byte & 0x0F) | (Int((qh >> (chunk * 2)) & 1) << 4)
                        let high = Int((byte >> 4) & 0x0F) | (Int((qh >> (chunk * 2 + 1)) & 1) << 4)

                        lowValues[l] = UInt8(low)
                        highValues[l] = UInt8(high)
                    }

                    let group0 = block * 8 + chunk * 2
                    let group1 = group0 + 1

                    scales[group0] = blockScales[chunk * 2]
                    biases[group0] = -blockMins[chunk * 2]
                    scales[group1] = blockScales[chunk * 2 + 1]
                    biases[group1] = -blockMins[chunk * 2 + 1]

                    let packedLow = packUnsignedValues(lowValues, bits: 5)
                    let packedHigh = packUnsignedValues(highValues, bits: 5)
                    packed.replaceSubrange(group0 * 5..<(group0 * 5 + 5), with: packedLow)
                    packed.replaceSubrange(group1 * 5..<(group1 * 5 + 5), with: packedHigh)
                }
            }
        }

        return makeQuantizedResult(
            packed: packed, scales: scales, biases: biases,
            shape: shape, groupSize: 32, bits: 5
        )
    }

    /// Pack Q6_K → MLX 6-bit affine, groupSize=16.
    ///
    /// GGUF: value = scale * (q - 32), where q ∈ [0, 63]
    /// MLX:  value = q * scale + (-32 * scale)
    private func packQ6_K(data: Data, shape: [Int]) -> ConvertedTensor {
        let totalElements = shape.reduce(1, *)
        let superBlockCount = totalElements / 256
        let groupCount = totalElements / 16

        var packed = [UInt32](repeating: 0, count: groupCount * 3)
        var scales = [Float](repeating: 0, count: groupCount)
        var biases = [Float](repeating: 0, count: groupCount)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<superBlockCount {
                let offset = block * 210
                let scalesOffset = offset + 192
                let d = readFloat16(bytes, offset + 208)

                for half in 0..<2 {
                    let qlBase = offset + half * 64
                    let qhBase = offset + 128 + half * 32
                    let scBase = scalesOffset + half * 8
                    var groups = Array(repeating: [UInt8](repeating: 0, count: 16), count: 8)

                    for l in 0..<32 {
                        let q1 = Int(bytes[qlBase + l] & 0x0F) | (Int((bytes[qhBase + l] >> 0) & 3) << 4)
                        let q2 = Int(bytes[qlBase + l + 32] & 0x0F) | (Int((bytes[qhBase + l] >> 2) & 3) << 4)
                        let q3 = Int(bytes[qlBase + l] >> 4) | (Int((bytes[qhBase + l] >> 4) & 3) << 4)
                        let q4 = Int(bytes[qlBase + l + 32] >> 4) | (Int((bytes[qhBase + l] >> 6) & 3) << 4)

                        let groupIndex = l / 16
                        let elementIndex = l % 16
                        groups[groupIndex][elementIndex] = UInt8(q1)
                        groups[groupIndex + 2][elementIndex] = UInt8(q2)
                        groups[groupIndex + 4][elementIndex] = UInt8(q3)
                        groups[groupIndex + 6][elementIndex] = UInt8(q4)
                    }

                    for localGroup in 0..<8 {
                        let groupIndex = block * 16 + half * 8 + localGroup
                        let scale = d * Float(Int8(bitPattern: bytes[scBase + localGroup]))
                        scales[groupIndex] = scale
                        biases[groupIndex] = -32.0 * scale

                        let packedGroup = packUnsignedValues(groups[localGroup], bits: 6)
                        packed.replaceSubrange(groupIndex * 3..<(groupIndex * 3 + 3), with: packedGroup)
                    }
                }
            }
        }

        return makeQuantizedResult(
            packed: packed, scales: scales, biases: biases,
            shape: shape, groupSize: 16, bits: 6
        )
    }

    /// Create ConvertedTensor.quantized from flat arrays.
    private func makeQuantizedResult(
        packed: [UInt32], scales: [Float], biases: [Float],
        shape: [Int], groupSize: Int, bits: Int
    ) -> ConvertedTensor {
        let packedColumns = shape.last! * bits / 32
        let weightShape = shape.dropLast().map { $0 } + [packedColumns]
        let scaleShape = shape.dropLast().map { $0 } + [shape.last! / groupSize]

        let weightData = packed.withUnsafeBytes { Data($0) }
        let weightArray = MLXArray(weightData, weightShape, type: UInt32.self)
        let scalesArray = MLXArray(scales).reshaped(scaleShape).asType(.float16)
        let biasesArray = MLXArray(biases).reshaped(scaleShape).asType(.float16)

        return .quantized(
            weight: weightArray, scales: scalesArray, biases: biasesArray,
            groupSize: groupSize, bits: bits)
    }

    // MARK: - Tier 1 Direct Affine Packing (Zero Loss)

    /// Pack Q5_0 → MLX 5-bit affine, groupSize=32.
    ///
    /// GGUF Q5_0 block: 22 bytes / 32 elements.
    /// Layout: d(f16, 2) + qh(uint32, 4) + qs(16).
    /// value = (q5 - 16) * d → MLX: q * scale + bias, scale = d, bias = -16 * d
    private func packQ5_0(data: Data, shape: [Int]) -> ConvertedTensor {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 32
        let groupCount = blockCount

        // 5-bit: 32 values × 5 bits = 160 bits = 5 UInt32 words per group
        var packed = [UInt32](repeating: 0, count: groupCount * 5)
        var scales = [Float](repeating: 0, count: groupCount)
        var biases = [Float](repeating: 0, count: groupCount)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 22
                let d = readFloat16(bytes, offset)
                scales[block] = d
                biases[block] = -16.0 * d

                // Read qh as uint32 (little-endian)
                let qh = UInt32(bytes[offset + 2])
                    | (UInt32(bytes[offset + 3]) << 8)
                    | (UInt32(bytes[offset + 4]) << 16)
                    | (UInt32(bytes[offset + 5]) << 24)

                var values = [UInt8](repeating: 0, count: 32)
                for j in 0..<16 {
                    let byte = bytes[offset + 6 + j]
                    let x0 = (Int(byte) & 0x0F) | (Int((qh >> UInt32(j)) & 1) << 4)
                    let x1 = (Int(byte >> 4) & 0x0F) | (Int((qh >> UInt32(j + 16)) & 1) << 4)
                    values[j] = UInt8(x0)
                    values[j + 16] = UInt8(x1)
                }

                let packedGroup = packUnsignedValues(values, bits: 5)
                packed.replaceSubrange(block * 5..<(block * 5 + 5), with: packedGroup)
            }
        }

        return makeQuantizedResult(
            packed: packed, scales: scales, biases: biases,
            shape: shape, groupSize: 32, bits: 5
        )
    }

    /// Pack Q5_1 → MLX 5-bit affine, groupSize=32.
    ///
    /// GGUF Q5_1 block: 24 bytes / 32 elements.
    /// Layout: d(f16, 2) + m(f16, 2) + qh(uint32, 4) + qs(16).
    /// value = q5 * d + m → MLX: q * scale + bias, scale = d, bias = m (direct)
    private func packQ5_1(data: Data, shape: [Int]) -> ConvertedTensor {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 32
        let groupCount = blockCount

        var packed = [UInt32](repeating: 0, count: groupCount * 5)
        var scales = [Float](repeating: 0, count: groupCount)
        var biases = [Float](repeating: 0, count: groupCount)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 24
                let d = readFloat16(bytes, offset)
                let m = readFloat16(bytes, offset + 2)
                scales[block] = d
                biases[block] = m

                // qh starts at offset+4
                let qh = UInt32(bytes[offset + 4])
                    | (UInt32(bytes[offset + 5]) << 8)
                    | (UInt32(bytes[offset + 6]) << 16)
                    | (UInt32(bytes[offset + 7]) << 24)

                var values = [UInt8](repeating: 0, count: 32)
                for j in 0..<16 {
                    let byte = bytes[offset + 8 + j]
                    let x0 = (Int(byte) & 0x0F) | (Int((qh >> UInt32(j)) & 1) << 4)
                    let x1 = (Int(byte >> 4) & 0x0F) | (Int((qh >> UInt32(j + 16)) & 1) << 4)
                    values[j] = UInt8(x0)
                    values[j + 16] = UInt8(x1)
                }

                let packedGroup = packUnsignedValues(values, bits: 5)
                packed.replaceSubrange(block * 5..<(block * 5 + 5), with: packedGroup)
            }
        }

        return makeQuantizedResult(
            packed: packed, scales: scales, biases: biases,
            shape: shape, groupSize: 32, bits: 5
        )
    }

    /// Pack Q8_K → MLX 8-bit affine, groupSize=32.
    ///
    /// GGUF Q8_K super-block: 292 bytes / 256 elements.
    /// Layout: d(f32, 4) + qs(256 int8) + bsums(32 ignored).
    /// Single f32 scale for all 256 elements, split into 8 sub-groups of 32.
    /// value = q_signed * d → MLX: q_unsigned * scale + bias,
    /// where q_unsigned = q_signed + 128, bias = -128 * d
    private func packQ8_K(data: Data, shape: [Int]) -> ConvertedTensor {
        let totalElements = shape.reduce(1, *)
        let superBlockCount = totalElements / 256
        let groupCount = totalElements / 32

        var packed = [UInt32](repeating: 0, count: totalElements / 4)
        var scales = [Float](repeating: 0, count: groupCount)
        var biases = [Float](repeating: 0, count: groupCount)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for sb in 0..<superBlockCount {
                let offset = sb * 292
                let d = readFloat32(bytes, offset)

                for subGroup in 0..<8 {
                    let groupIndex = sb * 8 + subGroup
                    scales[groupIndex] = d
                    biases[groupIndex] = -128.0 * d

                    let qsBase = offset + 4 + subGroup * 32
                    let packedBase = groupIndex * 8
                    for k in 0..<8 {
                        let b0 = UInt32(UInt8(bitPattern: Int8(bitPattern: bytes[qsBase + k * 4 + 0]) &+ 127 &+ 1))
                        let b1 = UInt32(UInt8(bitPattern: Int8(bitPattern: bytes[qsBase + k * 4 + 1]) &+ 127 &+ 1))
                        let b2 = UInt32(UInt8(bitPattern: Int8(bitPattern: bytes[qsBase + k * 4 + 2]) &+ 127 &+ 1))
                        let b3 = UInt32(UInt8(bitPattern: Int8(bitPattern: bytes[qsBase + k * 4 + 3]) &+ 127 &+ 1))
                        packed[packedBase + k] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
                    }
                }
            }
        }

        return makeQuantizedResult(
            packed: packed, scales: scales, biases: biases,
            shape: shape, groupSize: 32, bits: 8
        )
    }

    /// Pack Q2_K → MLX 2-bit affine, groupSize=16.
    ///
    /// GGUF Q2_K super-block: 84 bytes / 256 elements.
    /// Layout: scales(16) + qs(64) + d(f16, 2) + dmin(f16, 2).
    /// 16 sub-groups of 16 elements each.
    /// value = sub_scale * q - sub_min → MLX: q * sub_scale + (-sub_min)
    private func packQ2_K(data: Data, shape: [Int]) -> ConvertedTensor {
        let totalElements = shape.reduce(1, *)
        let superBlockCount = totalElements / 256
        let groupCount = totalElements / 16

        // 2-bit: 16 values × 2 bits = 32 bits = 1 UInt32 word per group
        var packed = [UInt32](repeating: 0, count: groupCount)
        var scales = [Float](repeating: 0, count: groupCount)
        var biases = [Float](repeating: 0, count: groupCount)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for sb in 0..<superBlockCount {
                let offset = sb * 84
                let scalesOffset = offset
                let qsOffset = offset + 16
                let d = readFloat16(bytes, offset + 80)
                let dmin = readFloat16(bytes, offset + 82)

                var groupIndex = sb * 16
                var scaleIdx = 0

                // Two 128-element halves, each with 32 qs bytes
                for half in 0..<2 {
                    let qBase = qsOffset + half * 32
                    var shift = 0

                    for _ in 0..<4 {
                        // First 16 elements: q[0..15] >> shift
                        let scByte1 = bytes[scalesOffset + scaleIdx]
                        let sc1 = Float(scByte1 & 0x0F) * d
                        let mn1 = Float((scByte1 >> 4) & 0x0F) * dmin
                        scaleIdx += 1

                        scales[groupIndex] = sc1
                        biases[groupIndex] = -mn1

                        var values1 = [UInt8](repeating: 0, count: 16)
                        for l in 0..<16 {
                            values1[l] = (bytes[qBase + l] >> shift) & 0x03
                        }
                        let packedGroup1 = packUnsignedValues(values1, bits: 2)
                        packed[groupIndex] = packedGroup1[0]
                        groupIndex += 1

                        // Next 16 elements: q[16..31] >> shift
                        let scByte2 = bytes[scalesOffset + scaleIdx]
                        let sc2 = Float(scByte2 & 0x0F) * d
                        let mn2 = Float((scByte2 >> 4) & 0x0F) * dmin
                        scaleIdx += 1

                        scales[groupIndex] = sc2
                        biases[groupIndex] = -mn2

                        var values2 = [UInt8](repeating: 0, count: 16)
                        for l in 0..<16 {
                            values2[l] = (bytes[qBase + 16 + l] >> shift) & 0x03
                        }
                        let packedGroup2 = packUnsignedValues(values2, bits: 2)
                        packed[groupIndex] = packedGroup2[0]
                        groupIndex += 1

                        shift += 2
                    }
                }
            }
        }

        return makeQuantizedResult(
            packed: packed, scales: scales, biases: biases,
            shape: shape, groupSize: 16, bits: 2
        )
    }

    /// Pack Q3_K → MLX 3-bit affine, groupSize=16.
    ///
    /// GGUF Q3_K super-block: 110 bytes / 256 elements.
    /// Layout: hmask(32) + qs(64) + scales(12) + d(f16, 2).
    /// 16 sub-groups of 16 elements each.
    /// 3-bit unsigned = qval (2-bit from qs) + hmBit (1-bit from hmask).
    /// value = effective_scale * (q3 - 4) → MLX: q * scale + bias,
    /// where scale = effective_scale, bias = -4 * effective_scale
    private func packQ3_K(data: Data, shape: [Int]) -> ConvertedTensor {
        let totalElements = shape.reduce(1, *)
        let superBlockCount = totalElements / 256
        let groupCount = totalElements / 16
        // 3-bit packing: 256 values × 3 bits = 768 bits = 24 UInt32 words per super-block.
        // Groups of 16 are not word-aligned (48 bits each), so pack all 256 values
        // per super-block as a contiguous bitstream.
        let wordsPerSuperBlock = 256 * 3 / 32  // = 24
        var packed = [UInt32](repeating: 0, count: superBlockCount * wordsPerSuperBlock)
        var scales = [Float](repeating: 0, count: groupCount)
        var biases = [Float](repeating: 0, count: groupCount)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for sb in 0..<superBlockCount {
                let offset = sb * 110
                let hmaskOffset = offset
                let qsOffset = offset + 32
                let scalesOffset = offset + 96
                let d = readFloat16(bytes, offset + 108)

                // Unpack 12 bytes → 16 6-bit scale values
                var rawScales = [Int](repeating: 0, count: 16)
                unpackQ3KScales(bytes: bytes, offset: scalesOffset, scales: &rawScales)

                // Collect all 256 values in output order
                var allValues = [UInt8](repeating: 0, count: 256)
                var valueIdx = 0
                var scaleIdx = 0
                var m: UInt8 = 1
                var groupIndex = sb * 16

                // Two 128-element halves, each with 32 qs bytes
                for _ in 0..<2 {
                    let qBase = qsOffset + (valueIdx / 128) * 32
                    var shift = 0

                    for _ in 0..<4 {
                        // First 16 elements: q[0..15] >> shift
                        let effectiveScale = d * Float(rawScales[scaleIdx] - 32)
                        scaleIdx += 1
                        scales[groupIndex] = effectiveScale
                        biases[groupIndex] = -4.0 * effectiveScale
                        groupIndex += 1

                        for l in 0..<16 {
                            let qval = Int((bytes[qBase + l] >> shift) & 0x03)
                            let hmBit = (bytes[hmaskOffset + l] & m) != 0
                            let q3 = hmBit ? (qval + 4) : qval
                            allValues[valueIdx] = UInt8(q3)
                            valueIdx += 1
                        }

                        // Next 16 elements: q[16..31] >> shift
                        let effectiveScale2 = d * Float(rawScales[scaleIdx] - 32)
                        scaleIdx += 1
                        scales[groupIndex] = effectiveScale2
                        biases[groupIndex] = -4.0 * effectiveScale2
                        groupIndex += 1

                        for l in 0..<16 {
                            let qval = Int((bytes[qBase + 16 + l] >> shift) & 0x03)
                            let hmBit = (bytes[hmaskOffset + 16 + l] & m) != 0
                            let q3 = hmBit ? (qval + 4) : qval
                            allValues[valueIdx] = UInt8(q3)
                            valueIdx += 1
                        }

                        shift += 2
                        m <<= 1
                    }
                }

                // Pack all 256 values as contiguous 3-bit bitstream
                let packedBlock = packUnsignedValues(allValues, bits: 3)
                let base = sb * wordsPerSuperBlock
                packed.replaceSubrange(base..<(base + wordsPerSuperBlock), with: packedBlock)
            }
        }

        return makeQuantizedResult(
            packed: packed, scales: scales, biases: biases,
            shape: shape, groupSize: 16, bits: 3
        )
    }

    /// Pack TQ2_0 → MLX 2-bit affine, groupSize=32.
    ///
    /// GGUF TQ2_0 super-block: 66 bytes / 256 elements.
    /// Layout: qs(64) + d(f16, 2).
    /// value = (q - 1) * d → MLX: q * scale + bias, scale = d, bias = -d
    /// 8 sub-groups of 32, all with same d.
    private func packTQ2_0(data: Data, shape: [Int]) -> ConvertedTensor {
        let totalElements = shape.reduce(1, *)
        let superBlockCount = totalElements / 256
        let groupCount = totalElements / 32

        // 2-bit: 32 values × 2 bits = 64 bits = 2 UInt32 words per group
        var packed = [UInt32](repeating: 0, count: groupCount * 2)
        var scales = [Float](repeating: 0, count: groupCount)
        var biases = [Float](repeating: 0, count: groupCount)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for sb in 0..<superBlockCount {
                let offset = sb * 66
                let d = readFloat16(bytes, offset + 64)
                let qsBase = offset

                var groupIndex = sb * 8

                // Process in 32-byte chunks with 4 bit-shift passes each
                var j = 0
                while j < 64 {
                    let chunkSize = min(32, 64 - j)
                    for l in 0..<4 {
                        scales[groupIndex] = d
                        biases[groupIndex] = -d

                        var values = [UInt8](repeating: 0, count: 32)
                        for m in 0..<chunkSize {
                            values[m] = (bytes[qsBase + j + m] >> (l * 2)) & 3
                        }
                        // Pad remaining values with 0 if chunkSize < 32
                        // (shouldn't happen since 64 / 32 = 2 even chunks)

                        let packedGroup = packUnsignedValues(values, bits: 2)
                        packed.replaceSubrange(groupIndex * 2..<(groupIndex * 2 + 2), with: packedGroup)
                        groupIndex += 1
                    }
                    j += 32
                }
            }
        }

        return makeQuantizedResult(
            packed: packed, scales: scales, biases: biases,
            shape: shape, groupSize: 32, bits: 2
        )
    }

    // MARK: - Re-quantize Decoded Values to Affine

    /// Pack decoded Float values into MLX affine quantized format.
    ///
    /// Used by Tier 2-4 pack functions (IQ, TQ types) that must decode non-linear
    /// representations before re-quantizing to affine.
    ///
    /// For each group of `groupSize` elements, computes optimal min-max affine
    /// parameters and quantizes to `bits`-bit unsigned integers.
    private func packDecodedToAffine(
        _ decoded: [Float], shape: [Int], groupSize: Int, bits: Int
    ) -> ConvertedTensor {
        let totalElements = decoded.count
        let groupCount = totalElements / groupSize
        let maxQ = Float((1 << bits) - 1)

        var packed = [UInt8]()
        packed.reserveCapacity(totalElements)
        var scales = [Float](repeating: 0, count: groupCount)
        var biases = [Float](repeating: 0, count: groupCount)

        for g in 0..<groupCount {
            let base = g * groupSize
            var vmin = decoded[base]
            var vmax = decoded[base]
            for i in 1..<groupSize {
                let v = decoded[base + i]
                if v < vmin { vmin = v }
                if v > vmax { vmax = v }
            }

            let range = vmax - vmin
            let scale: Float
            if range == 0 {
                scale = 0
            } else {
                scale = range / maxQ
            }
            scales[g] = scale
            biases[g] = vmin

            let invScale = scale == 0 ? Float(0) : 1.0 / scale
            for i in 0..<groupSize {
                let q = (decoded[base + i] - vmin) * invScale
                let clamped = min(max(q.rounded(), 0), maxQ)
                packed.append(UInt8(clamped))
            }
        }

        let packedWords = packUnsignedValues(packed, bits: bits)
        return makeQuantizedResult(
            packed: packedWords, scales: scales, biases: biases,
            shape: shape, groupSize: groupSize, bits: bits)
    }

    // MARK: - Tier 2: LUT Decode → 4-bit Affine

    /// Pack IQ4_NL → MLX 4-bit affine, groupSize=32.
    ///
    /// Decodes non-linear LUT values then re-quantizes to affine.
    /// Block = 18 bytes/32 elements: d(f16) + qs[16].
    private func packIQ4_NL(data: Data, shape: [Int]) -> ConvertedTensor {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 32
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 18
                let d = readFloat16(bytes, offset)

                for j in 0..<16 {
                    let byte = bytes[offset + 2 + j]
                    let lo = Int(byte & 0x0F)
                    let hi = Int((byte >> 4) & 0x0F)
                    result[block * 32 + j] = d * Float(Self.kvaluesIQ4NL[lo])
                    result[block * 32 + j + 16] = d * Float(Self.kvaluesIQ4NL[hi])
                }
            }
        }

        return packDecodedToAffine(result, shape: shape, groupSize: 32, bits: 4)
    }

    /// Pack IQ4_XS → MLX 4-bit affine, groupSize=32.
    ///
    /// Super-block = 136 bytes/256 elements: d(f16) + scales_h(2) + scales_l(4) + qs[128].
    /// Each 32-element group has a 6-bit scale; values decoded via IQ4_NL LUT.
    private func packIQ4_XS(data: Data, shape: [Int]) -> ConvertedTensor {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 136
                let d = readFloat16(bytes, offset)
                let scalesH = UInt16(bytes[offset + 2]) | (UInt16(bytes[offset + 3]) << 8)

                for ib in 0..<8 {
                    let lsByte = bytes[offset + 4 + ib / 2]
                    let ls = Int((lsByte >> (4 * (ib % 2))) & 0x0F) | (Int((scalesH >> (2 * ib)) & 3) << 4)
                    let dl = d * Float(ls - 32)
                    let qsBase = offset + 8 + 16 * ib
                    let outBase = block * 256 + ib * 32

                    for j in 0..<16 {
                        let byte = bytes[qsBase + j]
                        result[outBase + j] = dl * Float(Self.kvaluesIQ4NL[Int(byte & 0x0F)])
                        result[outBase + j + 16] = dl * Float(Self.kvaluesIQ4NL[Int((byte >> 4) & 0x0F)])
                    }
                }
            }
        }

        return packDecodedToAffine(result, shape: shape, groupSize: 32, bits: 4)
    }

    // MARK: - Tier 3: Grid Decode → 4-bit Affine

    /// Pack IQ2_XXS → MLX 4-bit affine, groupSize=32.
    ///
    /// Super-block = 66 bytes/256 elements: d(f16) + qs[64].
    /// Each group: aux32[0] = 4 grid indices, aux32[1] = 4x7-bit sign indices + 4-bit scale.
    private func packIQ2_XXS(data: Data, shape: [Int]) -> ConvertedTensor {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 66
                let d = readFloat16(bytes, offset)
                let qBase = offset + 2

                for ib32 in 0..<8 {
                    let aux32_0 = readUInt32LE(bytes, qBase + ib32 * 8)
                    let aux32_1 = readUInt32LE(bytes, qBase + ib32 * 8 + 4)
                    let db = d * (0.5 + Float(aux32_1 >> 28)) * 0.25

                    for l in 0..<4 {
                        let gridIdx = Int((aux32_0 >> (8 * l)) & 0xFF)
                        let grid = IQLookupTables.iq2xxsGrid[gridIdx]
                        let signs = IQLookupTables.ksignsIQ2XS[Int((aux32_1 >> (7 * l)) & 127)]

                        let outBase = block * 256 + ib32 * 32 + l * 8
                        for j in 0..<8 {
                            let gridVal = Float(UInt8((grid >> (8 * j)) & 0xFF))
                            let sign: Float = (signs & IQLookupTables.kmaskIQ2XS[j]) != 0 ? -1.0 : 1.0
                            result[outBase + j] = db * gridVal * sign
                        }
                    }
                }
            }
        }

        return packDecodedToAffine(result, shape: shape, groupSize: 32, bits: 4)
    }

    /// Pack IQ2_XS → MLX 4-bit affine, groupSize=32.
    ///
    /// Super-block = 74 bytes/256 elements: d(f16) + qs[32] as uint16 + scales[8].
    /// Each uint16: grid_index(9 bits) + sign_index(7 bits).
    private func packIQ2_XS(data: Data, shape: [Int]) -> ConvertedTensor {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 74
                let d = readFloat16(bytes, offset)
                let qsBase = offset + 2
                let scBase = offset + 66

                for ib32 in 0..<8 {
                    let db0 = d * (0.5 + Float(bytes[scBase + ib32] & 0x0F)) * 0.25
                    let db1 = d * (0.5 + Float(bytes[scBase + ib32] >> 4)) * 0.25

                    for l in 0..<4 {
                        let qs = readUInt16LE(bytes, qsBase + (4 * ib32 + l) * 2)
                        let gridIdx = Int(qs & 511)
                        let signs = IQLookupTables.ksignsIQ2XS[Int(qs >> 9)]
                        let grid = IQLookupTables.iq2xsGrid[gridIdx]
                        let dl = l < 2 ? db0 : db1

                        let outBase = block * 256 + (ib32 * 4 + l) * 8
                        for j in 0..<8 {
                            let gridVal = Float(UInt8((grid >> (8 * j)) & 0xFF))
                            let sign: Float = (signs & IQLookupTables.kmaskIQ2XS[j]) != 0 ? -1.0 : 1.0
                            result[outBase + j] = dl * gridVal * sign
                        }
                    }
                }
            }
        }

        return packDecodedToAffine(result, shape: shape, groupSize: 32, bits: 4)
    }

    /// Pack IQ2_S → MLX 4-bit affine, groupSize=32.
    ///
    /// Super-block = 82 bytes/256 elements: d(f16) + qs[64] + qh[8] + scales[8].
    /// qs[0..31] = grid indices, qs[32..63] = sign bytes.
    private func packIQ2_S(data: Data, shape: [Int]) -> ConvertedTensor {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 82
                let d = readFloat16(bytes, offset)
                let qsStart = offset + 2
                let qhBase = offset + 2 + 64
                let scBase = offset + 2 + 64 + 8

                var qsOff = qsStart
                var signOff = qsStart + 32

                for ib32 in 0..<8 {
                    let db0 = d * (0.5 + Float(bytes[scBase + ib32] & 0x0F)) * 0.25
                    let db1 = d * (0.5 + Float(bytes[scBase + ib32] >> 4)) * 0.25

                    for l in 0..<4 {
                        let dl = l < 2 ? db0 : db1
                        let qsVal = Int(bytes[qsOff + l])
                        let qhVal = Int(bytes[qhBase + ib32])
                        let gridIdx = qsVal | ((qhVal << (8 - 2 * l)) & 0x300)
                        let grid = IQLookupTables.iq2sGrid[min(gridIdx, 1023)]
                        let signByte = bytes[signOff + l]

                        let outBase = block * 256 + ib32 * 32 + l * 8
                        for j in 0..<8 {
                            let gridVal = Float(UInt8((grid >> (8 * j)) & 0xFF))
                            let sign: Float = (signByte & IQLookupTables.kmaskIQ2XS[j]) != 0 ? -1.0 : 1.0
                            result[outBase + j] = dl * gridVal * sign
                        }
                    }
                    qsOff += 4
                    signOff += 4
                }
            }
        }

        return packDecodedToAffine(result, shape: shape, groupSize: 32, bits: 4)
    }

    /// Pack IQ3_XXS → MLX 4-bit affine, groupSize=32.
    ///
    /// Super-block = 98 bytes/256 elements: d(f16) + qs[96].
    /// First 64 bytes = grid byte indices, last 32 bytes = sign/scale uint32s.
    private func packIQ3_XXS(data: Data, shape: [Int]) -> ConvertedTensor {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 98
                let d = readFloat16(bytes, offset)
                var qsOff = offset + 2
                let scaleSignBase = offset + 2 + 64

                for ib32 in 0..<8 {
                    let aux32 = readUInt32LE(bytes, scaleSignBase + 4 * ib32)
                    let db = d * (0.5 + Float(aux32 >> 28)) * 0.5

                    for l in 0..<4 {
                        let signs = IQLookupTables.ksignsIQ2XS[Int((aux32 >> (7 * l)) & 127)]
                        let grid1Idx = Int(bytes[qsOff + 2 * l])
                        let grid2Idx = Int(bytes[qsOff + 2 * l + 1])
                        let grid1 = IQLookupTables.iq3xxsGrid[grid1Idx]
                        let grid2 = IQLookupTables.iq3xxsGrid[grid2Idx]

                        let outBase = block * 256 + ib32 * 32 + l * 8
                        for j in 0..<4 {
                            let g1 = Float(UInt8((grid1 >> (8 * j)) & 0xFF))
                            let s1: Float = (signs & IQLookupTables.kmaskIQ2XS[j]) != 0 ? -1.0 : 1.0
                            result[outBase + j] = db * g1 * s1
                            let g2 = Float(UInt8((grid2 >> (8 * j)) & 0xFF))
                            let s2: Float = (signs & IQLookupTables.kmaskIQ2XS[j + 4]) != 0 ? -1.0 : 1.0
                            result[outBase + j + 4] = db * g2 * s2
                        }
                    }
                    qsOff += 8
                }
            }
        }

        return packDecodedToAffine(result, shape: shape, groupSize: 32, bits: 4)
    }

    /// Pack IQ3_S → MLX 4-bit affine, groupSize=32.
    ///
    /// Super-block = 110 bytes/256 elements: d(f16) + qs[64] + qh[8] + signs[32] + scales[4].
    /// Grid index: 8 bits from qs + 1 bit from qh = 9 bits into iq3s_grid[512].
    private func packIQ3_S(data: Data, shape: [Int]) -> ConvertedTensor {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 110
                let d = readFloat16(bytes, offset)
                var qsOff = offset + 2
                var qhOff = offset + 2 + 64
                var signOff = offset + 2 + 64 + 8
                let scBase = offset + 2 + 64 + 8 + 32

                var ib32 = 0
                while ib32 < 8 {
                    let scByte = bytes[scBase + ib32 / 2]
                    let db1 = d * Float(1 + 2 * Int(scByte & 0x0F))
                    let db2 = d * Float(1 + 2 * Int(scByte >> 4))

                    // First 32-element group
                    for l in 0..<4 {
                        let qsVal0 = Int(bytes[qsOff + 2 * l])
                        let qsVal1 = Int(bytes[qsOff + 2 * l + 1])
                        let qhByte = bytes[qhOff]
                        let gridIdx1 = qsVal0 | ((Int(qhByte) << (8 - 2 * l)) & 256)
                        let gridIdx2 = qsVal1 | ((Int(qhByte) << (7 - 2 * l)) & 256)
                        let grid1 = IQLookupTables.iq3sGrid[min(gridIdx1, 511)]
                        let grid2 = IQLookupTables.iq3sGrid[min(gridIdx2, 511)]
                        let signByte = bytes[signOff + l]

                        let outBase = block * 256 + ib32 * 32 + l * 8
                        for j in 0..<4 {
                            let g1 = Float(UInt8((grid1 >> (8 * j)) & 0xFF))
                            let s1: Float = (signByte & IQLookupTables.kmaskIQ2XS[j]) != 0 ? -1.0 : 1.0
                            result[outBase + j] = db1 * g1 * s1
                            let g2 = Float(UInt8((grid2 >> (8 * j)) & 0xFF))
                            let s2: Float = (signByte & IQLookupTables.kmaskIQ2XS[j + 4]) != 0 ? -1.0 : 1.0
                            result[outBase + j + 4] = db1 * g2 * s2
                        }
                    }
                    qsOff += 8
                    signOff += 4

                    // Second 32-element group
                    for l in 0..<4 {
                        let qsVal0 = Int(bytes[qsOff + 2 * l])
                        let qsVal1 = Int(bytes[qsOff + 2 * l + 1])
                        let qhByte = bytes[qhOff + 1]
                        let gridIdx1 = qsVal0 | ((Int(qhByte) << (8 - 2 * l)) & 256)
                        let gridIdx2 = qsVal1 | ((Int(qhByte) << (7 - 2 * l)) & 256)
                        let grid1 = IQLookupTables.iq3sGrid[min(gridIdx1, 511)]
                        let grid2 = IQLookupTables.iq3sGrid[min(gridIdx2, 511)]
                        let signByte = bytes[signOff + l]

                        let outBase = block * 256 + (ib32 + 1) * 32 + l * 8
                        for j in 0..<4 {
                            let g1 = Float(UInt8((grid1 >> (8 * j)) & 0xFF))
                            let s1: Float = (signByte & IQLookupTables.kmaskIQ2XS[j]) != 0 ? -1.0 : 1.0
                            result[outBase + j] = db2 * g1 * s1
                            let g2 = Float(UInt8((grid2 >> (8 * j)) & 0xFF))
                            let s2: Float = (signByte & IQLookupTables.kmaskIQ2XS[j + 4]) != 0 ? -1.0 : 1.0
                            result[outBase + j + 4] = db2 * g2 * s2
                        }
                    }
                    qhOff += 2
                    qsOff += 8
                    signOff += 4
                    ib32 += 2
                }
            }
        }

        return packDecodedToAffine(result, shape: shape, groupSize: 32, bits: 4)
    }

    /// Pack IQ1_S → MLX 4-bit affine, groupSize=32.
    ///
    /// Super-block = 50 bytes/256 elements: d(f16) + qs[32] + qh[8] as uint16.
    /// Grid index: 8 bits from qs + 3 bits from qh = 11 bits into iq1s_grid[2048].
    private func packIQ1_S(data: Data, shape: [Int]) -> ConvertedTensor {
        let IQ1S_DELTA: Float = 0.125
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 50
                let d = readFloat16(bytes, offset)
                var qsOff = offset + 2
                let qhBase = offset + 34

                for ib in 0..<8 {
                    let qh = readUInt16LE(bytes, qhBase + ib * 2)
                    let dl = d * Float(2 * Int((qh >> 12) & 7) + 1)
                    let delta: Float = (qh & 0x8000) != 0 ? -IQ1S_DELTA : IQ1S_DELTA

                    for l in 0..<4 {
                        let qs = Int(bytes[qsOff + l])
                        let gridIdx = qs | (Int((qh >> (3 * l)) & 7) << 8)
                        let grid = IQLookupTables.iq1sGrid[min(gridIdx, 2047)]

                        let outBase = block * 256 + (ib * 4 + l) * 8
                        for j in 0..<8 {
                            let gridVal = Int8(bitPattern: UInt8((grid >> (8 * j)) & 0xFF))
                            result[outBase + j] = dl * (Float(gridVal) + delta)
                        }
                    }
                    qsOff += 4
                }
            }
        }

        return packDecodedToAffine(result, shape: shape, groupSize: 32, bits: 4)
    }

    /// Pack IQ1_M → MLX 4-bit affine, groupSize=32.
    ///
    /// Super-block = 56 bytes/256 elements: qs[32] + qh[16] + scales[8].
    /// d is packed across the top 4 bits of each scale uint16.
    private func packIQ1_M(data: Data, shape: [Int]) -> ConvertedTensor {
        let IQ1M_DELTA: Float = 0.125
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 56
                let qsBase = offset
                let qhBase = offset + 32
                let scBase = offset + 48

                let sc0 = readUInt16LE(bytes, scBase)
                let sc1 = readUInt16LE(bytes, scBase + 2)
                let sc2 = readUInt16LE(bytes, scBase + 4)
                let sc3 = readUInt16LE(bytes, scBase + 6)
                let dBits = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0) | ((sc2 >> 4) & 0x0F00) | (sc3 & 0xF000)
                let d = Float(Float16(bitPattern: dBits))

                let sc = [sc0, sc1, sc2, sc3]
                var qsOff = qsBase
                var qhOff = qhBase

                for ib in 0..<8 {
                    let scWord = sc[ib / 2]
                    let scShift1 = 6 * (ib % 2)
                    let scShift2 = 6 * (ib % 2) + 3
                    let dl1 = d * Float(2 * Int((scWord >> scShift1) & 0x7) + 1)
                    let dl2 = d * Float(2 * Int((scWord >> scShift2) & 0x7) + 1)

                    let qh0 = bytes[qhOff]
                    let qh1 = bytes[qhOff + 1]

                    let idx0 = Int(bytes[qsOff]) | ((Int(qh0) << 8) & 0x700)
                    let idx1 = Int(bytes[qsOff + 1]) | ((Int(qh0) << 4) & 0x700)
                    let idx2 = Int(bytes[qsOff + 2]) | ((Int(qh1) << 8) & 0x700)
                    let idx3 = Int(bytes[qsOff + 3]) | ((Int(qh1) << 4) & 0x700)

                    let delta0: Float = (qh0 & 0x08) != 0 ? -IQ1M_DELTA : IQ1M_DELTA
                    let delta1: Float = (qh0 & 0x80) != 0 ? -IQ1M_DELTA : IQ1M_DELTA
                    let delta2: Float = (qh1 & 0x08) != 0 ? -IQ1M_DELTA : IQ1M_DELTA
                    let delta3: Float = (qh1 & 0x80) != 0 ? -IQ1M_DELTA : IQ1M_DELTA

                    let indices = [idx0, idx1, idx2, idx3]
                    let deltas = [delta0, delta1, delta2, delta3]

                    for l in 0..<2 {
                        let grid = IQLookupTables.iq1sGrid[min(indices[l], 2047)]
                        let outBase = block * 256 + (ib * 4 + l) * 8
                        for j in 0..<8 {
                            let gridVal = Int8(bitPattern: UInt8((grid >> (8 * j)) & 0xFF))
                            result[outBase + j] = dl1 * (Float(gridVal) + deltas[l])
                        }
                    }
                    for l in 2..<4 {
                        let grid = IQLookupTables.iq1sGrid[min(indices[l], 2047)]
                        let outBase = block * 256 + (ib * 4 + l) * 8
                        for j in 0..<8 {
                            let gridVal = Int8(bitPattern: UInt8((grid >> (8 * j)) & 0xFF))
                            result[outBase + j] = dl2 * (Float(gridVal) + deltas[l])
                        }
                    }
                    qsOff += 4
                    qhOff += 2
                }
            }
        }

        return packDecodedToAffine(result, shape: shape, groupSize: 32, bits: 4)
    }

    // MARK: - Tier 4: Ternary Decode → 2-bit Affine

    /// Pack TQ1_0 → MLX 2-bit affine, groupSize=32.
    ///
    /// Ternary quantization. Super-block = 54 bytes/256 elements:
    /// qs[48] + qh[4] + d(f16). Uses pow3 multiplication for base-3 trit extraction.
    private func packTQ1_0(data: Data, shape: [Int]) -> ConvertedTensor {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)
        let pow3: [UInt8] = [1, 3, 9, 27, 81, 243]

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 54
                let d = readFloat16(bytes, offset + 52)
                let qsBase = offset
                let qhBase = offset + 48

                var outIdx = block * 256

                // Phase 1: First 32 bytes of qs (32 bytes x 5 trits = 160 values)
                for n in 0..<5 {
                    for m in 0..<32 {
                        let q = UInt16(bytes[qsBase + m]) &* UInt16(pow3[n])
                        let xi = Int16((q &* 3) >> 8)
                        result[outIdx] = Float(xi - 1) * d
                        outIdx += 1
                    }
                }
                // Phase 2: Next 16 bytes of qs (16 bytes x 5 trits = 80 values)
                for n in 0..<5 {
                    for m in 0..<16 {
                        let q = UInt16(bytes[qsBase + 32 + m]) &* UInt16(pow3[n])
                        let xi = Int16((q &* 3) >> 8)
                        result[outIdx] = Float(xi - 1) * d
                        outIdx += 1
                    }
                }
                // Phase 3: qh[4] (4 bytes x 4 trits = 16 values)
                for n in 0..<4 {
                    for j in 0..<4 {
                        let q = UInt16(bytes[qhBase + j]) &* UInt16(pow3[n])
                        let xi = Int16((q &* 3) >> 8)
                        result[outIdx] = Float(xi - 1) * d
                        outIdx += 1
                    }
                }
            }
        }

        return packDecodedToAffine(result, shape: shape, groupSize: 32, bits: 2)
    }

    // MARK: - Unquantized Loaders

    private func loadFloat32(data: Data, shape: [Int]) -> MLXArray {
        MLXArray(data, shape, type: Float.self).asType(.float16)
    }

    private func loadFloat16(data: Data, shape: [Int]) -> MLXArray {
        MLXArray(data, shape, type: Float16.self)
    }

    private func loadBFloat16(data: Data, shape: [Int]) -> MLXArray {
        MLXArray(data, shape, type: UInt16.self).view(dtype: .bfloat16).asType(.float16)
    }

    // MARK: - Q4_0 Dequantization

    /// Q4_0: Block of 32 elements = 2 bytes (f16 scale) + 16 bytes (packed 4-bit)
    private func dequantizeQ4_0(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 32
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 18
                let scale = readFloat16(bytes, offset)

                for j in 0..<16 {
                    let byte = bytes[offset + 2 + j]
                    result[block * 32 + j] = Float(Int(byte & 0x0F) - 8) * scale
                    result[block * 32 + j + 16] = Float(Int((byte >> 4) & 0x0F) - 8) * scale
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - Q4_1 Dequantization

    /// Q4_1: Block of 32 elements = 20 bytes (f16 scale + f16 min + 16 packed 4-bit)
    private func dequantizeQ4_1(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 32
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 20
                let d = readFloat16(bytes, offset)
                let m = readFloat16(bytes, offset + 2)

                for j in 0..<16 {
                    let byte = bytes[offset + 4 + j]
                    result[block * 32 + j] = Float(byte & 0x0F) * d + m
                    result[block * 32 + j + 16] = Float((byte >> 4) & 0x0F) * d + m
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - Q5_0 Dequantization

    /// Q5_0: Block of 32 elements = 22 bytes (f16 scale + 4 high-bit bytes + 16 packed 4-bit)
    private func dequantizeQ5_0(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 32
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 22
                let d = readFloat16(bytes, offset)

                // Read 32 high bits as uint32 (little-endian)
                let qh = UInt32(bytes[offset + 2])
                    | (UInt32(bytes[offset + 3]) << 8)
                    | (UInt32(bytes[offset + 4]) << 16)
                    | (UInt32(bytes[offset + 5]) << 24)

                for j in 0..<16 {
                    let byte = bytes[offset + 6 + j]
                    let xh0 = Int((qh >> UInt32(j)) & 1) << 4
                    let xh1 = Int((qh >> UInt32(j + 16)) & 1) << 4

                    let x0 = Int(byte & 0x0F) | xh0
                    let x1 = Int((byte >> 4) & 0x0F) | xh1

                    result[block * 32 + j] = Float(x0 - 16) * d
                    result[block * 32 + j + 16] = Float(x1 - 16) * d
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - Q5_1 Dequantization

    /// Q5_1: Block of 32 elements = 24 bytes (f16 scale + f16 min + 4 high-bit bytes + 16 packed 4-bit)
    private func dequantizeQ5_1(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 32
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 24
                let d = readFloat16(bytes, offset)
                let m = readFloat16(bytes, offset + 2)

                let qh = UInt32(bytes[offset + 4])
                    | (UInt32(bytes[offset + 5]) << 8)
                    | (UInt32(bytes[offset + 6]) << 16)
                    | (UInt32(bytes[offset + 7]) << 24)

                for j in 0..<16 {
                    let byte = bytes[offset + 8 + j]
                    let xh0 = Int((qh >> UInt32(j)) & 1) << 4
                    let xh1 = Int((qh >> UInt32(j + 16)) & 1) << 4

                    let x0 = Int(byte & 0x0F) | xh0
                    let x1 = Int((byte >> 4) & 0x0F) | xh1

                    result[block * 32 + j] = Float(x0) * d + m
                    result[block * 32 + j + 16] = Float(x1) * d + m
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - Q8_0 Dequantization

    /// Q8_0: Block of 32 elements = 2 bytes (f16 scale) + 32 bytes (int8 values)
    private func dequantizeQ8_0(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 32
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 34
                let scale = readFloat16(bytes, offset)

                for j in 0..<32 {
                    let q = Int8(bitPattern: bytes[offset + 2 + j])
                    result[block * 32 + j] = Float(q) * scale
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - Q4_K Dequantization

    /// Q4_K: Super-block of 256 elements = 144 bytes
    /// Layout: d(2) + dmin(2) + scales(12) + qs(128)
    ///
    /// Scale encoding (12 bytes → 8 scales + 8 mins, each 6-bit):
    ///   bytes 0-3:  lower 6 bits of scales[0..3], bits 6-7 = upper 2 bits of scales[4..7]
    ///   bytes 4-7:  lower 6 bits of mins[0..3],   bits 6-7 = upper 2 bits of mins[4..7]
    ///   bytes 8-11: lower 4 bits = scales[4..7] lower 4, upper 4 bits = mins[4..7] lower 4
    private func dequantizeQ4_K(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 144
                let d = readFloat16(bytes, offset)
                let dmin = readFloat16(bytes, offset + 2)
                let q = offset + 4  // scales[12] start

                var scales = [Float](repeating: 0, count: 8)
                var mins = [Float](repeating: 0, count: 8)

                // Matches llama.cpp get_scale_min_k4
                for j in 0..<8 {
                    if j < 4 {
                        scales[j] = Float(bytes[q + j] & 63) * d
                        mins[j] = Float(bytes[q + j + 4] & 63) * dmin
                    } else {
                        // scale: lower 4 bits from q[j+4], upper 2 bits from q[j-4] bits 6-7
                        let sc = (bytes[q + j + 4] & 0x0F) | ((bytes[q + j - 4] >> 6) << 4)
                        // min: lower 4 bits from q[j+4] >> 4, upper 2 bits from q[j] bits 6-7
                        let mn = (bytes[q + j + 4] >> 4) | ((bytes[q + j] >> 6) << 4)
                        scales[j] = Float(sc) * d
                        mins[j] = Float(mn) * dmin
                    }
                }

                let qsOffset = offset + 16
                // 4 chunks of 32 bytes, each encoding 64 elements
                for chunk in 0..<4 {
                    let sc0 = scales[chunk * 2]
                    let mn0 = mins[chunk * 2]
                    let sc1 = scales[chunk * 2 + 1]
                    let mn1 = mins[chunk * 2 + 1]
                    let baseIdx = block * 256 + chunk * 64
                    let qs = qsOffset + chunk * 32

                    for j in 0..<32 {
                        let byte = bytes[qs + j]
                        result[baseIdx + j] = sc0 * Float(byte & 0x0F) - mn0
                        result[baseIdx + j + 32] = sc1 * Float((byte >> 4) & 0x0F) - mn1
                    }
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - Q6_K Dequantization

    /// Q6_K: Super-block of 256 elements = 210 bytes
    /// Layout: ql(128) + qh(64) + scales(16) + d(2)
    ///
    /// Each element = 6-bit signed value (subtract 32 → range [-32, +31]).
    /// Lower 4 bits in ql[], upper 2 bits in qh[].
    /// qh mapping (per 128-element half, l=0..31):
    ///   qh[l] bits 0-1 → element l+0   (ql[l] low nibble)
    ///   qh[l] bits 2-3 → element l+32  (ql[l+32] low nibble)
    ///   qh[l] bits 4-5 → element l+64  (ql[l] high nibble)
    ///   qh[l] bits 6-7 → element l+96  (ql[l+32] high nibble)
    private func dequantizeQ6_K(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 210
                let scalesOffset = offset + 192
                let d = readFloat16(bytes, offset + 208)

                // Process two 128-element halves (matches llama.cpp outer loop)
                for half in 0..<2 {
                    let qlBase = offset + half * 64
                    let qhBase = offset + 128 + half * 32
                    let scBase = scalesOffset + half * 8
                    let outBase = block * 256 + half * 128

                    for l in 0..<32 {
                        let scIdx = l / 16

                        let q1ql = Int(bytes[qlBase + l] & 0x0F)
                        let q1qh = Int((bytes[qhBase + l] >> 0) & 3) << 4
                        let q1 = (q1ql | q1qh) - 32

                        let q2ql = Int(bytes[qlBase + l + 32] & 0x0F)
                        let q2qh = Int((bytes[qhBase + l] >> 2) & 3) << 4
                        let q2 = (q2ql | q2qh) - 32

                        let q3ql = Int(bytes[qlBase + l] >> 4)
                        let q3qh = Int((bytes[qhBase + l] >> 4) & 3) << 4
                        let q3 = (q3ql | q3qh) - 32

                        let q4ql = Int(bytes[qlBase + l + 32] >> 4)
                        let q4qh = Int((bytes[qhBase + l] >> 6) & 3) << 4
                        let q4 = (q4ql | q4qh) - 32

                        let sc0 = Float(Int8(bitPattern: bytes[scBase + scIdx]))
                        let sc2 = Float(Int8(bitPattern: bytes[scBase + scIdx + 2]))
                        let sc4 = Float(Int8(bitPattern: bytes[scBase + scIdx + 4]))
                        let sc6 = Float(Int8(bitPattern: bytes[scBase + scIdx + 6]))

                        result[outBase + l] = d * sc0 * Float(q1)
                        result[outBase + l + 32] = d * sc2 * Float(q2)
                        result[outBase + l + 64] = d * sc4 * Float(q3)
                        result[outBase + l + 96] = d * sc6 * Float(q4)
                    }
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - Q2_K Dequantization

    /// Q2_K: Super-block of 256 elements = 84 bytes
    /// Layout: scales(16) + qs(64) + d(2) + dmin(2)
    ///
    /// Matches llama.cpp dequantize_row_q2_K: two 128-element halves,
    /// each processing 32 qs bytes with 4 shift levels (0,2,4,6).
    /// Each shift level produces 2 groups of 16 elements (q[0..15] and q[16..31]).
    private func dequantizeQ2_K(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 84
                let scalesOffset = offset
                let qsOffset = offset + 16
                let d = readFloat16(bytes, offset + 80)
                let dmin = readFloat16(bytes, offset + 82)

                var outIdx = block * 256
                var scaleIdx = 0

                // Two 128-element halves, each with 32 qs bytes
                for half in 0..<2 {
                    let qBase = qsOffset + half * 32
                    var shift = 0

                    for _ in 0..<4 {
                        // First 16 elements: q[0..15] >> shift
                        let scByte1 = bytes[scalesOffset + scaleIdx]
                        let sc1 = Float(scByte1 & 0x0F) * d
                        let mn1 = Float((scByte1 >> 4) & 0x0F) * dmin
                        scaleIdx += 1

                        for l in 0..<16 {
                            let q = Int((bytes[qBase + l] >> shift) & 0x03)
                            result[outIdx] = sc1 * Float(q) - mn1
                            outIdx += 1
                        }

                        // Next 16 elements: q[16..31] >> shift
                        let scByte2 = bytes[scalesOffset + scaleIdx]
                        let sc2 = Float(scByte2 & 0x0F) * d
                        let mn2 = Float((scByte2 >> 4) & 0x0F) * dmin
                        scaleIdx += 1

                        for l in 0..<16 {
                            let q = Int((bytes[qBase + 16 + l] >> shift) & 0x03)
                            result[outIdx] = sc2 * Float(q) - mn2
                            outIdx += 1
                        }

                        shift += 2
                    }
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - Q3_K Dequantization

    /// Q3_K: Super-block of 256 elements = 110 bytes
    /// Layout: hmask(32) + qs(64) + scales(12) + d(2)
    ///
    /// Matches llama.cpp dequantize_row_q3_K: two 128-element halves,
    /// each processing 32 qs bytes with 4 shift levels (0,2,4,6).
    /// hmask bitmask `m` starts at 1 and shifts left once per shift level
    /// (first half uses bits 0-3, second half uses bits 4-7).
    /// hmask pointer does NOT advance — both halves read hmask[0..31].
    ///
    /// Scale encoding (12 bytes → 16 6-bit signed values):
    /// Treated as three uint32 words via kmask approach.
    private func dequantizeQ3_K(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 110
                let hmaskOffset = offset
                let qsOffset = offset + 32
                let scalesOffset = offset + 96
                let d = readFloat16(bytes, offset + 108)

                // Unpack 12 bytes → 16 6-bit scales (matches llama.cpp kmask approach)
                var scales = [Int](repeating: 0, count: 16)
                unpackQ3KScales(bytes: bytes, offset: scalesOffset, scales: &scales)

                var outIdx = block * 256
                var scaleIdx = 0
                var m: UInt8 = 1

                // Two 128-element halves, each with 32 qs bytes
                // hm does NOT advance — both halves use hmask[0..31]
                for half in 0..<2 {
                    let qBase = qsOffset + half * 32
                    var shift = 0

                    for _ in 0..<4 {
                        // First 16 elements: q[0..15] >> shift
                        let sc = d * Float(scales[scaleIdx] - 32)
                        scaleIdx += 1

                        for l in 0..<16 {
                            let qval = Int((bytes[qBase + l] >> shift) & 0x03)
                            let hmBit = (bytes[hmaskOffset + l] & m) != 0
                            result[outIdx] = sc * Float(qval - (hmBit ? 0 : 4))
                            outIdx += 1
                        }

                        // Next 16 elements: q[16..31] >> shift
                        let sc2 = d * Float(scales[scaleIdx] - 32)
                        scaleIdx += 1

                        for l in 0..<16 {
                            let qval = Int((bytes[qBase + 16 + l] >> shift) & 0x03)
                            let hmBit = (bytes[hmaskOffset + 16 + l] & m) != 0
                            result[outIdx] = sc2 * Float(qval - (hmBit ? 0 : 4))
                            outIdx += 1
                        }

                        shift += 2
                        m <<= 1
                    }
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    /// Unpack 12-byte Q3_K scale section into 16 6-bit values.
    ///
    /// Layout mirrors llama.cpp's uint32 kmask approach:
    ///   aux[0] bytes: low nibble = scales[0..3], high nibble = scales[8..11]
    ///   aux[1] bytes: low nibble = scales[4..7], high nibble = scales[12..15]
    ///   aux[2] bytes: 2-bit pairs = upper bits for all 16 scales
    private func unpackQ3KScales(bytes: UnsafeBufferPointer<UInt8>, offset: Int, scales: inout [Int]) {
        // Read raw bytes as 4 groups of 4 bytes
        var raw = [UInt8](repeating: 0, count: 16)
        for i in 0..<12 {
            raw[i] = bytes[offset + i]
        }

        // Treat as uint32 words (little-endian)
        let aux0_bytes = (0..<4).map { raw[$0] }
        let aux1_bytes = (4..<8).map { raw[$0] }
        let tmp_bytes = (8..<12).map { raw[$0] }

        // scales[0..3]: low nibble of aux[0] | bits 0-1 of tmp
        for i in 0..<4 {
            let lo4 = Int(aux0_bytes[i] & 0x0F)
            let hi2 = Int((tmp_bytes[i] >> 0) & 0x03) << 4
            scales[i] = lo4 | hi2
        }
        // scales[4..7]: low nibble of aux[1] | bits 2-3 of tmp
        for i in 0..<4 {
            let lo4 = Int(aux1_bytes[i] & 0x0F)
            let hi2 = Int((tmp_bytes[i] >> 2) & 0x03) << 4
            scales[4 + i] = lo4 | hi2
        }
        // scales[8..11]: high nibble of aux[0] | bits 4-5 of tmp
        for i in 0..<4 {
            let lo4 = Int((aux0_bytes[i] >> 4) & 0x0F)
            let hi2 = Int((tmp_bytes[i] >> 4) & 0x03) << 4
            scales[8 + i] = lo4 | hi2
        }
        // scales[12..15]: high nibble of aux[1] | bits 6-7 of tmp
        for i in 0..<4 {
            let lo4 = Int((aux1_bytes[i] >> 4) & 0x0F)
            let hi2 = Int((tmp_bytes[i] >> 6) & 0x03) << 4
            scales[12 + i] = lo4 | hi2
        }
    }

    // MARK: - Q5_K Dequantization

    /// Q5_K: Super-block of 256 elements = 176 bytes
    /// Layout: d(2) + dmin(2) + scales(12) + qh(32) + qs(128)
    ///
    /// Scale encoding: same 12-byte format as Q4_K (get_scale_min_k4).
    private func dequantizeQ5_K(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 176
                let dVal = readFloat16(bytes, offset)
                let dmin = readFloat16(bytes, offset + 2)
                let q = offset + 4  // scales[12] start
                let qhOffset = offset + 16
                let qsOffset = offset + 48

                // Decode scales using get_scale_min_k4 pattern (same as Q4_K)
                var scales = [Float](repeating: 0, count: 8)
                var mins = [Float](repeating: 0, count: 8)
                for j in 0..<8 {
                    if j < 4 {
                        scales[j] = Float(bytes[q + j] & 63) * dVal
                        mins[j] = Float(bytes[q + j + 4] & 63) * dmin
                    } else {
                        let sc = (bytes[q + j + 4] & 0x0F) | ((bytes[q + j - 4] >> 6) << 4)
                        let mn = (bytes[q + j + 4] >> 4) | ((bytes[q + j] >> 6) << 4)
                        scales[j] = Float(sc) * dVal
                        mins[j] = Float(mn) * dmin
                    }
                }

                // 4 chunks of 32 qs bytes, each encoding 64 elements.
                // qh layout: qh[l] bit (chunk*2) → low nibble high bit,
                //             qh[l] bit (chunk*2+1) → high nibble high bit.
                for chunk in 0..<4 {
                    let sc0 = scales[chunk * 2]
                    let mn0 = mins[chunk * 2]
                    let sc1 = scales[chunk * 2 + 1]
                    let mn1 = mins[chunk * 2 + 1]
                    let baseIdx = block * 256 + chunk * 64
                    let qs = qsOffset + chunk * 32

                    for l in 0..<32 {
                        let byte = bytes[qs + l]
                        let qhBit0 = Int((bytes[qhOffset + l] >> (chunk * 2)) & 1) << 4
                        let qhBit1 = Int((bytes[qhOffset + l] >> (chunk * 2 + 1)) & 1) << 4

                        result[baseIdx + l] = sc0 * Float(Int(byte & 0x0F) | qhBit0) - mn0
                        result[baseIdx + l + 32] = sc1 * Float(Int((byte >> 4) & 0x0F) | qhBit1) - mn1
                    }
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - Q8_1 Dequantization

    /// Q8_1: Block of 32 elements = 36 bytes (f16 d + f16 s + 32 int8)
    /// s is d*sum(qs), used for dot product optimization, not needed for dequant.
    private func dequantizeQ8_1(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 32
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 36
                let d = readFloat16(bytes, offset)
                // s at offset+2 is not needed for element-wise dequant
                for j in 0..<32 {
                    let q = Int8(bitPattern: bytes[offset + 4 + j])
                    result[block * 32 + j] = d * Float(q)
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - Q8_K Dequantization

    /// Q8_K: Super-block of 256 elements = 292 bytes (f32 d + 256 int8 qs + 16 int16 bsums)
    /// bsums are for dot product optimization, not needed for dequant.
    private func dequantizeQ8_K(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 292
                let d = readFloat32(bytes, offset)
                for j in 0..<256 {
                    let q = Int8(bitPattern: bytes[offset + 4 + j])
                    result[block * 256 + j] = d * Float(q)
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    /// Pack Q8_1 → MLX 8-bit, groupSize=32.
    ///
    /// Same as Q8_0: value = q_signed * d → MLX: q_unsigned * d + (-128 * d)
    private func packQ8_1(data: Data, shape: [Int]) -> ConvertedTensor {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 32

        var packed = [UInt32](repeating: 0, count: totalElements / 4)
        var scales = [Float](repeating: 0, count: blockCount)
        var biases = [Float](repeating: 0, count: blockCount)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 36  // 2(d) + 2(s) + 32(qs)
                let d = readFloat16(bytes, offset)
                scales[block] = d
                biases[block] = -128.0 * d

                let packedBase = block * 8
                let qs = offset + 4
                for k in 0..<8 {
                    let b0 = UInt32(UInt8(bitPattern: Int8(bitPattern: bytes[qs + k * 4 + 0]) &+ 127 &+ 1))
                    let b1 = UInt32(UInt8(bitPattern: Int8(bitPattern: bytes[qs + k * 4 + 1]) &+ 127 &+ 1))
                    let b2 = UInt32(UInt8(bitPattern: Int8(bitPattern: bytes[qs + k * 4 + 2]) &+ 127 &+ 1))
                    let b3 = UInt32(UInt8(bitPattern: Int8(bitPattern: bytes[qs + k * 4 + 3]) &+ 127 &+ 1))
                    packed[packedBase + k] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
                }
            }
        }

        return makeQuantizedResult(
            packed: packed, scales: scales, biases: biases,
            shape: shape, groupSize: 32, bits: 8)
    }

    // MARK: - IQ4_NL Dequantization

    /// IQ4_NL non-linear lookup table (from llama.cpp kvalues_iq4nl).
    /// Maps 4-bit indices [0..15] to non-linearly spaced int8 values.
    private static let kvaluesIQ4NL: [Int8] = [
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113
    ]

    /// IQ4_NL: Block of 32 elements = 18 bytes (same layout as Q4_0 but uses non-linear LUT).
    private func dequantizeIQ4_NL(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 32
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 18
                let d = readFloat16(bytes, offset)

                for j in 0..<16 {
                    let byte = bytes[offset + 2 + j]
                    let lo = Int(byte & 0x0F)
                    let hi = Int((byte >> 4) & 0x0F)
                    result[block * 32 + j] = d * Float(Self.kvaluesIQ4NL[lo])
                    result[block * 32 + j + 16] = d * Float(Self.kvaluesIQ4NL[hi])
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - IQ4_XS Dequantization

    /// IQ4_XS: Super-block of 256 elements = 136 bytes.
    /// Layout: d(2) + scales_h(2) + scales_l(4) + qs(128)
    /// Each 32-element group has a 6-bit scale; values decoded via IQ4_NL LUT.
    private func dequantizeIQ4_XS(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 136
                let d = readFloat16(bytes, offset)
                let scalesH = UInt16(bytes[offset + 2]) | (UInt16(bytes[offset + 3]) << 8)

                for ib in 0..<8 {
                    // 6-bit scale: lower 4 bits from scales_l, upper 2 from scales_h
                    let lsByte = bytes[offset + 4 + ib / 2]
                    let ls = Int((lsByte >> (4 * (ib % 2))) & 0x0F) | (Int((scalesH >> (2 * ib)) & 3) << 4)
                    let dl = d * Float(ls - 32)
                    let qsBase = offset + 8 + 16 * ib
                    let outBase = block * 256 + ib * 32

                    for j in 0..<16 {
                        let byte = bytes[qsBase + j]
                        result[outBase + j] = dl * Float(Self.kvaluesIQ4NL[Int(byte & 0x0F)])
                        result[outBase + j + 16] = dl * Float(Self.kvaluesIQ4NL[Int((byte >> 4) & 0x0F)])
                    }
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - IQ2_XXS Dequantization

    /// IQ2_XXS: Super-block of 256 elements = 66 bytes.
    /// Layout: d(2) + qs(64) where qs is treated as 8 groups of 8 bytes.
    /// Each group: aux32[0] = 4 grid indices, aux32[1] = 4×7-bit sign indices + 4-bit scale.
    private func dequantizeIQ2_XXS(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 66
                let d = readFloat16(bytes, offset)
                let qBase = offset + 2

                for ib32 in 0..<8 {
                    let aux32_0 = readUInt32LE(bytes, qBase + ib32 * 8)
                    let aux32_1 = readUInt32LE(bytes, qBase + ib32 * 8 + 4)
                    let db = d * (0.5 + Float(aux32_1 >> 28)) * 0.25

                    for l in 0..<4 {
                        let gridIdx = Int((aux32_0 >> (8 * l)) & 0xFF)
                        let grid = IQLookupTables.iq2xxsGrid[gridIdx]
                        let signs = IQLookupTables.ksignsIQ2XS[Int((aux32_1 >> (7 * l)) & 127)]

                        let outBase = block * 256 + ib32 * 32 + l * 8
                        for j in 0..<8 {
                            let gridVal = Float(UInt8((grid >> (8 * j)) & 0xFF))
                            let sign: Float = (signs & IQLookupTables.kmaskIQ2XS[j]) != 0 ? -1.0 : 1.0
                            result[outBase + j] = db * gridVal * sign
                        }
                    }
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - IQ2_XS Dequantization

    /// IQ2_XS: Super-block of 256 elements = 74 bytes.
    /// Layout: d(2) + qs[32] as uint16 + scales[8].
    /// Each uint16: grid_index(9 bits) + sign_index(7 bits).
    private func dequantizeIQ2_XS(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 74
                let d = readFloat16(bytes, offset)
                let qsBase = offset + 2
                let scBase = offset + 66

                for ib32 in 0..<8 {
                    let db0 = d * (0.5 + Float(bytes[scBase + ib32] & 0x0F)) * 0.25
                    let db1 = d * (0.5 + Float(bytes[scBase + ib32] >> 4)) * 0.25

                    for l in 0..<4 {
                        let qs = readUInt16LE(bytes, qsBase + (4 * ib32 + l) * 2)
                        let gridIdx = Int(qs & 511)
                        let signs = IQLookupTables.ksignsIQ2XS[Int(qs >> 9)]
                        let grid = IQLookupTables.iq2xsGrid[gridIdx]
                        let dl = l < 2 ? db0 : db1

                        let outBase = block * 256 + (ib32 * 4 + l) * 8
                        for j in 0..<8 {
                            let gridVal = Float(UInt8((grid >> (8 * j)) & 0xFF))
                            let sign: Float = (signs & IQLookupTables.kmaskIQ2XS[j]) != 0 ? -1.0 : 1.0
                            result[outBase + j] = dl * gridVal * sign
                        }
                    }
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - IQ2_S Dequantization

    /// IQ2_S: Super-block of 256 elements = 82 bytes.
    /// Layout: d(2) + qs[64] + qh[8] + scales[8].
    /// qs[0..31] = grid indices, qs[32..63] = sign bytes.
    private func dequantizeIQ2_S(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 82
                let d = readFloat16(bytes, offset)
                let qsStart = offset + 2
                let qhBase = offset + 2 + 64  // after qs[64]
                let scBase = offset + 2 + 64 + 8  // after qh[8]

                var qsOff = qsStart
                var signOff = qsStart + 32  // signs = qs + QK_K/8

                for ib32 in 0..<8 {
                    let db0 = d * (0.5 + Float(bytes[scBase + ib32] & 0x0F)) * 0.25
                    let db1 = d * (0.5 + Float(bytes[scBase + ib32] >> 4)) * 0.25

                    for l in 0..<4 {
                        let dl = l < 2 ? db0 : db1
                        let qsVal = Int(bytes[qsOff + l])
                        let qhVal = Int(bytes[qhBase + ib32])
                        let gridIdx = qsVal | ((qhVal << (8 - 2 * l)) & 0x300)
                        let grid = IQLookupTables.iq2sGrid[min(gridIdx, 1023)]
                        let signByte = bytes[signOff + l]

                        let outBase = block * 256 + ib32 * 32 + l * 8
                        for j in 0..<8 {
                            let gridVal = Float(UInt8((grid >> (8 * j)) & 0xFF))
                            let sign: Float = (signByte & IQLookupTables.kmaskIQ2XS[j]) != 0 ? -1.0 : 1.0
                            result[outBase + j] = dl * gridVal * sign
                        }
                    }
                    qsOff += 4
                    signOff += 4
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - IQ3_XXS Dequantization

    /// IQ3_XXS: Super-block of 256 elements = 98 bytes.
    /// Layout: d(2) + qs[3*QK_K/8=96].
    /// First 64 bytes = grid byte indices, last 32 bytes = sign/scale uint32s.
    private func dequantizeIQ3_XXS(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 98
                let d = readFloat16(bytes, offset)
                var qsOff = offset + 2
                let scaleSignBase = offset + 2 + 64  // scales_and_signs = qs + QK_K/4

                for ib32 in 0..<8 {
                    let aux32 = readUInt32LE(bytes, scaleSignBase + 4 * ib32)
                    let db = d * (0.5 + Float(aux32 >> 28)) * 0.5

                    for l in 0..<4 {
                        let signs = IQLookupTables.ksignsIQ2XS[Int((aux32 >> (7 * l)) & 127)]
                        let grid1Idx = Int(bytes[qsOff + 2 * l])
                        let grid2Idx = Int(bytes[qsOff + 2 * l + 1])
                        let grid1 = IQLookupTables.iq3xxsGrid[grid1Idx]
                        let grid2 = IQLookupTables.iq3xxsGrid[grid2Idx]

                        let outBase = block * 256 + ib32 * 32 + l * 8
                        for j in 0..<4 {
                            let g1 = Float(UInt8((grid1 >> (8 * j)) & 0xFF))
                            let s1: Float = (signs & IQLookupTables.kmaskIQ2XS[j]) != 0 ? -1.0 : 1.0
                            result[outBase + j] = db * g1 * s1
                            let g2 = Float(UInt8((grid2 >> (8 * j)) & 0xFF))
                            let s2: Float = (signs & IQLookupTables.kmaskIQ2XS[j + 4]) != 0 ? -1.0 : 1.0
                            result[outBase + j + 4] = db * g2 * s2
                        }
                    }
                    qsOff += 8
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - IQ3_S Dequantization

    /// IQ3_S: Super-block of 256 elements = 110 bytes.
    /// Layout: d(2) + qs[64] + qh[8] + signs[32] + scales[4].
    /// Grid index: 8 bits from qs + 1 bit from qh = 9 bits into iq3s_grid[512].
    private func dequantizeIQ3_S(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 110
                let d = readFloat16(bytes, offset)
                var qsOff = offset + 2
                var qhOff = offset + 2 + 64        // qh[8]
                var signOff = offset + 2 + 64 + 8  // signs[32]
                let scBase = offset + 2 + 64 + 8 + 32  // scales[4]

                // Process pairs of 32-element groups
                var ib32 = 0
                while ib32 < 8 {
                    let scByte = bytes[scBase + ib32 / 2]
                    let db1 = d * Float(1 + 2 * Int(scByte & 0x0F))
                    let db2 = d * Float(1 + 2 * Int(scByte >> 4))

                    // First 32-element group
                    for l in 0..<4 {
                        let qsVal0 = Int(bytes[qsOff + 2 * l])
                        let qsVal1 = Int(bytes[qsOff + 2 * l + 1])
                        let qhByte = bytes[qhOff]
                        let gridIdx1 = qsVal0 | ((Int(qhByte) << (8 - 2 * l)) & 256)
                        let gridIdx2 = qsVal1 | ((Int(qhByte) << (7 - 2 * l)) & 256)
                        let grid1 = IQLookupTables.iq3sGrid[min(gridIdx1, 511)]
                        let grid2 = IQLookupTables.iq3sGrid[min(gridIdx2, 511)]
                        let signByte = bytes[signOff + l]

                        let outBase = block * 256 + ib32 * 32 + l * 8
                        for j in 0..<4 {
                            let g1 = Float(UInt8((grid1 >> (8 * j)) & 0xFF))
                            let s1: Float = (signByte & IQLookupTables.kmaskIQ2XS[j]) != 0 ? -1.0 : 1.0
                            result[outBase + j] = db1 * g1 * s1
                            let g2 = Float(UInt8((grid2 >> (8 * j)) & 0xFF))
                            let s2: Float = (signByte & IQLookupTables.kmaskIQ2XS[j + 4]) != 0 ? -1.0 : 1.0
                            result[outBase + j + 4] = db1 * g2 * s2
                        }
                    }
                    qsOff += 8
                    signOff += 4

                    // Second 32-element group
                    for l in 0..<4 {
                        let qsVal0 = Int(bytes[qsOff + 2 * l])
                        let qsVal1 = Int(bytes[qsOff + 2 * l + 1])
                        let qhByte = bytes[qhOff + 1]
                        let gridIdx1 = qsVal0 | ((Int(qhByte) << (8 - 2 * l)) & 256)
                        let gridIdx2 = qsVal1 | ((Int(qhByte) << (7 - 2 * l)) & 256)
                        let grid1 = IQLookupTables.iq3sGrid[min(gridIdx1, 511)]
                        let grid2 = IQLookupTables.iq3sGrid[min(gridIdx2, 511)]
                        let signByte = bytes[signOff + l]

                        let outBase = block * 256 + (ib32 + 1) * 32 + l * 8
                        for j in 0..<4 {
                            let g1 = Float(UInt8((grid1 >> (8 * j)) & 0xFF))
                            let s1: Float = (signByte & IQLookupTables.kmaskIQ2XS[j]) != 0 ? -1.0 : 1.0
                            result[outBase + j] = db2 * g1 * s1
                            let g2 = Float(UInt8((grid2 >> (8 * j)) & 0xFF))
                            let s2: Float = (signByte & IQLookupTables.kmaskIQ2XS[j + 4]) != 0 ? -1.0 : 1.0
                            result[outBase + j + 4] = db2 * g2 * s2
                        }
                    }
                    qhOff += 2
                    qsOff += 8
                    signOff += 4
                    ib32 += 2
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - IQ1_S Dequantization

    /// IQ1_S: Super-block of 256 elements = 50 bytes.
    /// Layout: d(2) + qs[32] + qh[8] as uint16.
    /// Grid index: 8 bits from qs + 3 bits from qh = 11 bits into iq1s_grid[2048].
    private func dequantizeIQ1_S(data: Data, shape: [Int]) throws -> MLXArray {
        let IQ1S_DELTA: Float = 0.125
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 50
                let d = readFloat16(bytes, offset)
                var qsOff = offset + 2
                let qhBase = offset + 34

                for ib in 0..<8 {
                    let qh = readUInt16LE(bytes, qhBase + ib * 2)
                    let dl = d * Float(2 * Int((qh >> 12) & 7) + 1)
                    let delta: Float = (qh & 0x8000) != 0 ? -IQ1S_DELTA : IQ1S_DELTA

                    for l in 0..<4 {
                        let qs = Int(bytes[qsOff + l])
                        let gridIdx = qs | (Int((qh >> (3 * l)) & 7) << 8)
                        let grid = IQLookupTables.iq1sGrid[min(gridIdx, 2047)]

                        let outBase = block * 256 + (ib * 4 + l) * 8
                        for j in 0..<8 {
                            let gridVal = Int8(bitPattern: UInt8((grid >> (8 * j)) & 0xFF))
                            result[outBase + j] = dl * (Float(gridVal) + delta)
                        }
                    }
                    qsOff += 4
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - IQ1_M Dequantization

    /// IQ1_M: Super-block of 256 elements = 56 bytes.
    /// Layout: qs[32] + qh[16] + scales[8].
    /// d is packed across the top 4 bits of each scale uint16.
    private func dequantizeIQ1_M(data: Data, shape: [Int]) throws -> MLXArray {
        let IQ1M_DELTA: Float = 0.125
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 56
                let qsBase = offset
                let qhBase = offset + 32
                let scBase = offset + 48

                // Extract d from top 4 bits of each scale uint16
                let sc0 = readUInt16LE(bytes, scBase)
                let sc1 = readUInt16LE(bytes, scBase + 2)
                let sc2 = readUInt16LE(bytes, scBase + 4)
                let sc3 = readUInt16LE(bytes, scBase + 6)
                let dBits = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0) | ((sc2 >> 4) & 0x0F00) | (sc3 & 0xF000)
                let d = Float(Float16(bitPattern: dBits))

                let sc = [sc0, sc1, sc2, sc3]
                var qsOff = qsBase
                var qhOff = qhBase

                for ib in 0..<8 {
                    let scWord = sc[ib / 2]
                    let scShift1 = 6 * (ib % 2)
                    let scShift2 = 6 * (ib % 2) + 3
                    let dl1 = d * Float(2 * Int((scWord >> scShift1) & 0x7) + 1)
                    let dl2 = d * Float(2 * Int((scWord >> scShift2) & 0x7) + 1)

                    let qh0 = bytes[qhOff]
                    let qh1 = bytes[qhOff + 1]

                    let idx0 = Int(bytes[qsOff]) | ((Int(qh0) << 8) & 0x700)
                    let idx1 = Int(bytes[qsOff + 1]) | ((Int(qh0) << 4) & 0x700)
                    let idx2 = Int(bytes[qsOff + 2]) | ((Int(qh1) << 8) & 0x700)
                    let idx3 = Int(bytes[qsOff + 3]) | ((Int(qh1) << 4) & 0x700)

                    let delta0: Float = (qh0 & 0x08) != 0 ? -IQ1M_DELTA : IQ1M_DELTA
                    let delta1: Float = (qh0 & 0x80) != 0 ? -IQ1M_DELTA : IQ1M_DELTA
                    let delta2: Float = (qh1 & 0x08) != 0 ? -IQ1M_DELTA : IQ1M_DELTA
                    let delta3: Float = (qh1 & 0x80) != 0 ? -IQ1M_DELTA : IQ1M_DELTA

                    let indices = [idx0, idx1, idx2, idx3]
                    let deltas = [delta0, delta1, delta2, delta3]

                    // First two sub-groups use dl1
                    for l in 0..<2 {
                        let grid = IQLookupTables.iq1sGrid[min(indices[l], 2047)]
                        let outBase = block * 256 + (ib * 4 + l) * 8
                        for j in 0..<8 {
                            let gridVal = Int8(bitPattern: UInt8((grid >> (8 * j)) & 0xFF))
                            result[outBase + j] = dl1 * (Float(gridVal) + deltas[l])
                        }
                    }
                    // Last two sub-groups use dl2
                    for l in 2..<4 {
                        let grid = IQLookupTables.iq1sGrid[min(indices[l], 2047)]
                        let outBase = block * 256 + (ib * 4 + l) * 8
                        for j in 0..<8 {
                            let gridVal = Int8(bitPattern: UInt8((grid >> (8 * j)) & 0xFF))
                            result[outBase + j] = dl2 * (Float(gridVal) + deltas[l])
                        }
                    }
                    qsOff += 4
                    qhOff += 2
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - TQ1_0 Dequantization

    /// TQ1_0: Ternary quantization. Super-block of 256 elements.
    /// Layout: qs[(256-16)/5=48] + qh[4] + d(2).
    /// Uses pow3 multiplication for base-3 trit extraction.
    private func dequantizeTQ1_0(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        // Block size: qs=48, qh=4, d=2 → 54 bytes
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)
        let pow3: [UInt8] = [1, 3, 9, 27, 81, 243]

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 54
                // d is after qs[48] + qh[4]
                let d = readFloat16(bytes, offset + 52)
                let qsBase = offset
                let qhBase = offset + 48

                var outIdx = block * 256

                // Phase 1: First 32 bytes of qs (32 bytes × 5 trits = 160 values)
                for n in 0..<5 {
                    for m in 0..<32 {
                        let q = UInt16(bytes[qsBase + m]) &* UInt16(pow3[n])
                        let xi = Int16((q &* 3) >> 8)
                        result[outIdx] = Float(xi - 1) * d
                        outIdx += 1
                    }
                }
                // Phase 2: Next 16 bytes of qs (16 bytes × 5 trits = 80 values)
                for n in 0..<5 {
                    for m in 0..<16 {
                        let q = UInt16(bytes[qsBase + 32 + m]) &* UInt16(pow3[n])
                        let xi = Int16((q &* 3) >> 8)
                        result[outIdx] = Float(xi - 1) * d
                        outIdx += 1
                    }
                }
                // Phase 3: qh[4] (4 bytes × 4 trits = 16 values)
                for n in 0..<4 {
                    for j in 0..<4 {
                        let q = UInt16(bytes[qhBase + j]) &* UInt16(pow3[n])
                        let xi = Int16((q &* 3) >> 8)
                        result[outIdx] = Float(xi - 1) * d
                        outIdx += 1
                    }
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - TQ2_0 Dequantization

    /// TQ2_0: Ternary quantization. Super-block of 256 elements = 66 bytes.
    /// Layout: qs[64] + d(2).
    /// Each byte encodes 4 values (2 bits each). Processed in 32-byte chunks × 4 bit-shift passes.
    private func dequantizeTQ2_0(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 66
                let d = readFloat16(bytes, offset + 64)
                let qsBase = offset

                var outIdx = block * 256
                // Process in 32-byte chunks with 4 bit-shift passes each
                var j = 0
                while j < 64 {
                    let chunkSize = min(32, 64 - j)
                    for l in 0..<4 {
                        for m in 0..<chunkSize {
                            let q = Int((bytes[qsBase + j + m] >> (l * 2)) & 3)
                            result[outIdx] = Float(q - 1) * d
                            outIdx += 1
                        }
                    }
                    j += 32
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - Helpers

    /// Read a little-endian float16 from a byte buffer at the given offset.
    func readFloat16(_ bytes: UnsafeBufferPointer<UInt8>, _ offset: Int) -> Float {
        let bits = UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
        return Float(Float16(bitPattern: bits))
    }

    /// Read a little-endian float32 from a byte buffer at the given offset.
    func readFloat32(_ bytes: UnsafeBufferPointer<UInt8>, _ offset: Int) -> Float {
        let bits = UInt32(bytes[offset])
            | (UInt32(bytes[offset + 1]) << 8)
            | (UInt32(bytes[offset + 2]) << 16)
            | (UInt32(bytes[offset + 3]) << 24)
        return Float(bitPattern: bits)
    }

    /// Read a little-endian uint16 from a byte buffer at the given offset.
    func readUInt16LE(_ bytes: UnsafeBufferPointer<UInt8>, _ offset: Int) -> UInt16 {
        UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
    }

    /// Read a little-endian uint32 from a byte buffer at the given offset.
    func readUInt32LE(_ bytes: UnsafeBufferPointer<UInt8>, _ offset: Int) -> UInt32 {
        UInt32(bytes[offset])
            | (UInt32(bytes[offset + 1]) << 8)
            | (UInt32(bytes[offset + 2]) << 16)
            | (UInt32(bytes[offset + 3]) << 24)
    }
}
