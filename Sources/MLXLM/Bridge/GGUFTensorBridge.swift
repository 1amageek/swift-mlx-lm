import Foundation
import MLX
import GGUFParser

/// Converts GGUF tensor data to MLXArray values.
///
/// Phase 1: Dequantizes all formats to F16 for simplicity.
/// Future: Direct quantized-to-QuantizedLinear conversion for memory efficiency.
struct GGUFTensorBridge {

    init() {}

    /// Convert a GGUF tensor to an MLXArray.
    ///
    /// - Parameters:
    ///   - tensor: Tensor metadata from the GGUF file.
    ///   - data: Raw tensor data bytes.
    /// - Returns: MLXArray with shape derived from GGUF dimensions.
    func convert(tensor: GGUFTensorInfo, data: Data) throws -> MLXArray {
        let qtype = tensor.quantizationType

        // GGUF dimensions: ne[0] is innermost, ne[1] is next, etc.
        // MLX uses row-major with last dim fastest, so reverse.
        let shape = tensor.dimensions.reversed().map { Int($0) }

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
        case .q5_K:
            return try dequantizeQ5_K(data: data, shape: shape)
        default:
            throw GGUFLoadError.unsupportedQuantization(qtype.rawValue)
        }
    }

    // MARK: - Unquantized Loaders

    private func loadFloat32(data: Data, shape: [Int]) -> MLXArray {
        MLXArray(data, shape, type: Float.self).asType(.float16)
    }

    private func loadFloat16(data: Data, shape: [Int]) -> MLXArray {
        MLXArray(data, shape, type: Float16.self)
    }

    private func loadBFloat16(data: Data, shape: [Int]) -> MLXArray {
        // No native BFloat16 type in Swift; load as UInt16 pairs and reinterpret
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
                let scale = float16ToFloat32(hi: bytes[offset + 1], lo: bytes[offset])

                for j in 0..<16 {
                    let byte = bytes[offset + 2 + j]
                    let lo = Int(byte & 0x0F)
                    let hi = Int((byte >> 4) & 0x0F)
                    result[block * 32 + j] = Float(lo - 8) * scale
                    result[block * 32 + j + 16] = Float(hi - 8) * scale
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
                let scale = float16ToFloat32(hi: bytes[offset + 1], lo: bytes[offset])

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
    private func dequantizeQ4_K(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 144
                let d = float16ToFloat32(hi: bytes[offset + 1], lo: bytes[offset])
                let dmin = float16ToFloat32(hi: bytes[offset + 3], lo: bytes[offset + 2])

                // Decode scales (12 bytes → 8 sub-blocks, each has scale + min)
                var scales = [Float](repeating: 0, count: 8)
                var mins = [Float](repeating: 0, count: 8)
                let scalesOffset = offset + 4

                for i in 0..<8 {
                    if i < 4 {
                        scales[i] = Float(bytes[scalesOffset + i] & 0x3F) * d
                        mins[i] = Float(bytes[scalesOffset + i + 4] & 0x3F) * dmin
                    } else {
                        let hiScale = (bytes[scalesOffset + (i - 4)] >> 6) |
                                     ((bytes[scalesOffset + (i - 4) + 4] >> 6) << 2)
                        scales[i] = Float(bytes[scalesOffset + i + 4] & 0x3F |
                                         (hiScale << 6)) * d
                        mins[i] = Float(bytes[scalesOffset + i + 8] & 0x3F) * dmin
                    }
                }

                // Decode 4-bit quantized values
                let qsOffset = offset + 16
                for subBlock in 0..<8 {
                    let sc = scales[subBlock]
                    let mn = mins[subBlock]
                    let baseIdx = block * 256 + subBlock * 32

                    for j in 0..<16 {
                        let byte = bytes[qsOffset + subBlock * 16 + j]
                        result[baseIdx + j] = sc * Float(byte & 0x0F) - mn
                        result[baseIdx + j + 16] = sc * Float((byte >> 4) & 0x0F) - mn
                    }
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - Q6_K Dequantization

    /// Q6_K: Super-block of 256 elements = 210 bytes
    /// Layout: ql(128) + qh(64) + scales(16) + d(2)
    private func dequantizeQ6_K(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 210
                let qlOffset = offset
                let qhOffset = offset + 128
                let scalesOffset = offset + 192
                let dOffset = offset + 208
                let d = float16ToFloat32(hi: bytes[dOffset + 1], lo: bytes[dOffset])

                for subBlock in 0..<16 {
                    let sc = Int8(bitPattern: bytes[scalesOffset + subBlock])
                    let baseIdx = block * 256 + subBlock * 16

                    for j in 0..<16 {
                        let qlIdx = subBlock * 16 + j
                        let ql: UInt8
                        if subBlock < 8 {
                            ql = bytes[qlOffset + qlIdx] & 0x0F
                        } else {
                            ql = (bytes[qlOffset + qlIdx - 128] >> 4) & 0x0F
                        }

                        let qhBitIdx = (subBlock % 8) * 16 + j
                        let qhByte = bytes[qhOffset + qhBitIdx / 4]
                        let qhShift = (subBlock / 8) * 2 + (qhBitIdx % 4 >= 2 ? 1 : 0)
                        let qh = (qhByte >> (qhShift * 2)) & 0x03

                        let q = Int(ql) | (Int(qh) << 4)
                        result[baseIdx + j] = d * Float(sc) * Float(q - 32)
                    }
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - Q2_K Dequantization

    /// Q2_K: Super-block of 256 elements = 84 bytes
    private func dequantizeQ2_K(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 84
                let scalesOffset = offset        // 16 bytes
                let qsOffset = offset + 16       // 64 bytes
                let dOffset = offset + 80         // 2 bytes
                let dminOffset = offset + 82      // 2 bytes
                let d = float16ToFloat32(hi: bytes[dOffset + 1], lo: bytes[dOffset])
                let dmin = float16ToFloat32(hi: bytes[dminOffset + 1], lo: bytes[dminOffset])

                for subBlock in 0..<16 {
                    let scByte = bytes[scalesOffset + subBlock]
                    let sc = Float(scByte & 0x0F) * d
                    let mn = Float((scByte >> 4) & 0x0F) * dmin
                    let baseIdx = block * 256 + subBlock * 16

                    for j in 0..<16 {
                        let qByteIdx = qsOffset + (subBlock * 16 + j) / 4
                        let qShift = ((subBlock * 16 + j) % 4) * 2
                        let q = Int((bytes[qByteIdx] >> qShift) & 0x03)
                        result[baseIdx + j] = sc * Float(q) - mn
                    }
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - Q3_K Dequantization

    /// Q3_K: Super-block of 256 elements = 110 bytes
    private func dequantizeQ3_K(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 110
                let hmaskOffset = offset          // 32 bytes
                let qsOffset = offset + 32        // 64 bytes
                let scalesOffset = offset + 96    // 12 bytes
                let dOffset = offset + 108        // 2 bytes
                let d = float16ToFloat32(hi: bytes[dOffset + 1], lo: bytes[dOffset])

                // Decode 6-bit scales from 12 bytes → 16 values
                var scales = [Int8](repeating: 0, count: 16)
                for i in 0..<16 {
                    if i < 8 {
                        scales[i] = Int8(bitPattern: bytes[scalesOffset + i])
                    } else {
                        let lo4 = bytes[scalesOffset + i - 8] & 0xF0
                        let hi2 = bytes[scalesOffset + 8 + (i - 8) / 2]
                        let shift = ((i - 8) % 2) * 4
                        scales[i] = Int8(bitPattern: (lo4 >> 4) | ((hi2 >> shift) & 0x0F) << 4)
                    }
                }

                for subBlock in 0..<16 {
                    let sc = Float(scales[subBlock])
                    let baseIdx = block * 256 + subBlock * 16

                    for j in 0..<16 {
                        let elemIdx = subBlock * 16 + j
                        let qByteIdx = qsOffset + elemIdx / 4
                        let qShift = (elemIdx % 4) * 2
                        let q2 = Int((bytes[qByteIdx] >> qShift) & 0x03)

                        let hmBit = Int((bytes[hmaskOffset + elemIdx / 8] >> (elemIdx % 8)) & 1)
                        let q = q2 | (hmBit << 2)

                        result[baseIdx + j] = d * sc * Float(q - 4)
                    }
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - Q5_K Dequantization

    /// Q5_K: Super-block of 256 elements = 176 bytes
    private func dequantizeQ5_K(data: Data, shape: [Int]) throws -> MLXArray {
        let totalElements = shape.reduce(1, *)
        let blockCount = totalElements / 256
        var result = [Float](repeating: 0, count: totalElements)

        data.withUnsafeBytes { buffer in
            let bytes = buffer.bindMemory(to: UInt8.self)
            for block in 0..<blockCount {
                let offset = block * 176
                let dVal = float16ToFloat32(hi: bytes[offset + 1], lo: bytes[offset])
                let dmin = float16ToFloat32(hi: bytes[offset + 3], lo: bytes[offset + 2])
                let scalesOffset = offset + 4     // 12 bytes
                let qhOffset = offset + 16        // 32 bytes
                let qsOffset = offset + 48        // 128 bytes

                // Decode scales (same pattern as Q4_K)
                var scales = [Float](repeating: 0, count: 8)
                var mins = [Float](repeating: 0, count: 8)
                for i in 0..<8 {
                    if i < 4 {
                        scales[i] = Float(bytes[scalesOffset + i] & 0x3F) * dVal
                        mins[i] = Float(bytes[scalesOffset + i + 4] & 0x3F) * dmin
                    } else {
                        scales[i] = Float(bytes[scalesOffset + i + 4] & 0x3F) * dVal
                        mins[i] = Float(bytes[scalesOffset + i + 8] & 0x3F) * dmin
                    }
                }

                for subBlock in 0..<8 {
                    let sc = scales[subBlock]
                    let mn = mins[subBlock]
                    let baseIdx = block * 256 + subBlock * 32

                    for j in 0..<16 {
                        let byte = bytes[qsOffset + subBlock * 16 + j]
                        let qhBit0 = Int((bytes[qhOffset + j] >> subBlock) & 1) << 4
                        let qhBit1 = Int((bytes[qhOffset + j + 16] >> subBlock) & 1) << 4

                        result[baseIdx + j] = sc * Float(Int(byte & 0x0F) | qhBit0) - mn
                        result[baseIdx + j + 16] = sc * Float(Int((byte >> 4) & 0x0F) | qhBit1) - mn
                    }
                }
            }
        }

        return MLXArray(result).reshaped(shape).asType(.float16)
    }

    // MARK: - Helpers

    /// Convert two bytes (little-endian f16) to Float.
    private func float16ToFloat32(hi: UInt8, lo: UInt8) -> Float {
        let bits = UInt16(hi) << 8 | UInt16(lo)
        return float16BitsToFloat(bits)
    }

    /// Convert f16 bit pattern to Float32.
    private func float16BitsToFloat(_ bits: UInt16) -> Float {
        let sign = (bits >> 15) & 1
        let exp = (bits >> 10) & 0x1F
        let frac = bits & 0x3FF

        if exp == 0 {
            if frac == 0 { return sign == 0 ? 0.0 : -0.0 }
            // Subnormal
            let value = Float(frac) / 1024.0 * pow(2.0, -14.0)
            return sign == 0 ? value : -value
        }
        if exp == 0x1F {
            if frac == 0 { return sign == 0 ? Float.infinity : -Float.infinity }
            return Float.nan
        }

        let value = (1.0 + Float(frac) / 1024.0) * pow(2.0, Float(Int(exp) - 15))
        return sign == 0 ? value : -value
    }
}
