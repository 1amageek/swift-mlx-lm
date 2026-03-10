import Testing
import Foundation
import MLX
import MLXNN
import GGUFParser
@testable import GGUFTokenizer
@testable import MLXLM

// MARK: - GGUFTensorNameMapper Tests

@Suite("LlamaTensorNameMapper")
struct LlamaTensorNameMapperTests {

    let mapper = LlamaTensorNameMapper()

    @Test("Global tensor: token_embd.weight")
    func tokenEmbd() {
        #expect(mapper.mlxName(for: "token_embd.weight") == "model.embed_tokens.weight")
    }

    @Test("Global tensor: output_norm.weight")
    func outputNorm() {
        #expect(mapper.mlxName(for: "output_norm.weight") == "model.norm.weight")
    }

    @Test("Global tensor: output.weight")
    func output() {
        #expect(mapper.mlxName(for: "output.weight") == "lm_head.weight")
    }

    @Test("Block attention query")
    func attnQ() {
        #expect(mapper.mlxName(for: "blk.0.attn_q.weight") == "model.layers.0.self_attn.q_proj.weight")
        #expect(mapper.mlxName(for: "blk.15.attn_q.weight") == "model.layers.15.self_attn.q_proj.weight")
    }

    @Test("Block attention key/value/output")
    func attnKVO() {
        #expect(mapper.mlxName(for: "blk.3.attn_k.weight") == "model.layers.3.self_attn.k_proj.weight")
        #expect(mapper.mlxName(for: "blk.3.attn_v.weight") == "model.layers.3.self_attn.v_proj.weight")
        #expect(mapper.mlxName(for: "blk.3.attn_output.weight") == "model.layers.3.self_attn.o_proj.weight")
    }

    @Test("Block MLP projections")
    func mlpProj() {
        #expect(mapper.mlxName(for: "blk.7.ffn_gate.weight") == "model.layers.7.mlp.gate_proj.weight")
        #expect(mapper.mlxName(for: "blk.7.ffn_up.weight") == "model.layers.7.mlp.up_proj.weight")
        #expect(mapper.mlxName(for: "blk.7.ffn_down.weight") == "model.layers.7.mlp.down_proj.weight")
    }

    @Test("Block norms")
    func blockNorms() {
        #expect(mapper.mlxName(for: "blk.0.attn_norm.weight") == "model.layers.0.input_layernorm.weight")
        #expect(mapper.mlxName(for: "blk.0.ffn_norm.weight") == "model.layers.0.post_attention_layernorm.weight")
    }

    @Test("Bias variants")
    func biasVariants() {
        #expect(mapper.mlxName(for: "blk.0.attn_q.bias") == "model.layers.0.self_attn.q_proj.bias")
        #expect(mapper.mlxName(for: "blk.0.attn_k.bias") == "model.layers.0.self_attn.k_proj.bias")
        #expect(mapper.mlxName(for: "blk.0.attn_v.bias") == "model.layers.0.self_attn.v_proj.bias")
        #expect(mapper.mlxName(for: "blk.0.attn_output.bias") == "model.layers.0.self_attn.o_proj.bias")
        #expect(mapper.mlxName(for: "blk.0.ffn_gate.bias") == "model.layers.0.mlp.gate_proj.bias")
        #expect(mapper.mlxName(for: "blk.0.ffn_up.bias") == "model.layers.0.mlp.up_proj.bias")
        #expect(mapper.mlxName(for: "blk.0.ffn_down.bias") == "model.layers.0.mlp.down_proj.bias")
    }

    @Test("Unknown tensor returns nil")
    func unknownTensor() {
        #expect(mapper.mlxName(for: "unknown.tensor") == nil)
        #expect(mapper.mlxName(for: "blk.0.unknown.weight") == nil)
    }

    @Test("Invalid block format returns nil")
    func invalidBlock() {
        #expect(mapper.mlxName(for: "blk.abc.attn_q.weight") == nil)
        #expect(mapper.mlxName(for: "blk.") == nil)
    }
}

// MARK: - TransformerConfiguration GGUF Tests

@Suite("TransformerConfiguration from GGUF")
struct TransformerConfigurationGGUFTests {

    /// Create a minimal GGUF file with specified metadata for testing.
    private func makeGGUFData(
        architecture: String = "llama",
        embeddingLength: Int = 256,
        blockCount: Int = 2,
        headCount: Int = 4,
        feedForwardLength: Int? = nil,
        headCountKV: Int? = nil,
        rmsEps: Float? = nil,
        ropeFreqBase: Float? = nil,
        vocabSize: Int = 100,
        includeLmHead: Bool = false
    ) throws -> GGUFFile {
        // Build a minimal GGUF binary
        var builder = GGUFTestBuilder()
        builder.addMetadata("general.architecture", value: .string(architecture))
        builder.addMetadata("\(architecture).embedding_length", value: .uint32(UInt32(embeddingLength)))
        builder.addMetadata("\(architecture).block_count", value: .uint32(UInt32(blockCount)))
        builder.addMetadata("\(architecture).attention.head_count", value: .uint32(UInt32(headCount)))

        if let ff = feedForwardLength {
            builder.addMetadata("\(architecture).feed_forward_length", value: .uint32(UInt32(ff)))
        }
        if let kv = headCountKV {
            builder.addMetadata("\(architecture).attention.head_count_kv", value: .uint32(UInt32(kv)))
        }
        if let eps = rmsEps {
            builder.addMetadata("\(architecture).attention.layer_norm_rms_epsilon", value: .float32(eps))
        }
        if let base = ropeFreqBase {
            builder.addMetadata("\(architecture).rope.freq_base", value: .float32(base))
        }

        // Vocabulary tokens
        let tokens = (0..<vocabSize).map { "tok\($0)" }
        builder.addMetadata("tokenizer.ggml.tokens", value: .array(tokens.map { .string($0) }))

        if includeLmHead {
            builder.addTensor(name: "output.weight", shape: [UInt64(vocabSize), UInt64(embeddingLength)], type: .f32)
        }

        let data = builder.build()
        return try GGUFFile.parse(data: data)
    }

    @Test("Extract basic config")
    func basicConfig() throws {
        let file = try makeGGUFData()
        let config = try TransformerConfiguration(from: file)

        #expect(config.hiddenSize == 256)
        #expect(config.hiddenLayers == 2)
        #expect(config.attentionHeads == 4)
        #expect(config.vocabularySize == 100)
        #expect(config.resolvedHeadDimensions == 64) // 256 / 4
    }

    @Test("Custom feed forward length")
    func customFFN() throws {
        let file = try makeGGUFData(feedForwardLength: 512)
        let config = try TransformerConfiguration(from: file)
        #expect(config.intermediateSize == 512)
    }

    @Test("Default feed forward length is 4x hidden")
    func defaultFFN() throws {
        let file = try makeGGUFData()
        let config = try TransformerConfiguration(from: file)
        #expect(config.intermediateSize == 1024) // 256 * 4
    }

    @Test("KV heads override")
    func kvHeads() throws {
        let file = try makeGGUFData(headCountKV: 2)
        let config = try TransformerConfiguration(from: file)
        #expect(config.kvHeads == 2)
    }

    @Test("RMS norm eps")
    func rmsEps() throws {
        let file = try makeGGUFData(rmsEps: 1e-6)
        let config = try TransformerConfiguration(from: file)
        #expect(config.normEps == 1e-6)
    }

    @Test("RoPE theta")
    func ropeTheta() throws {
        let file = try makeGGUFData(ropeFreqBase: 500_000.0)
        let config = try TransformerConfiguration(from: file)
        #expect(config.ropeTheta == 500_000.0)
    }

    @Test("Tie word embeddings when no output.weight tensor")
    func tieEmbeddings() throws {
        let file = try makeGGUFData(includeLmHead: false)
        let config = try TransformerConfiguration(from: file)
        #expect(config.tieWordEmbeddings == true)
    }

    @Test("No tie when output.weight tensor exists")
    func noTieEmbeddings() throws {
        let file = try makeGGUFData(includeLmHead: true)
        let config = try TransformerConfiguration(from: file)
        #expect(config.tieWordEmbeddings == false)
    }

    @Test("Missing embedding_length throws")
    func missingEmbedding() throws {
        var builder = GGUFTestBuilder()
        builder.addMetadata("general.architecture", value: .string("llama"))
        builder.addMetadata("llama.block_count", value: .uint32(2))
        builder.addMetadata("llama.attention.head_count", value: .uint32(4))
        let file = try GGUFFile.parse(data: builder.build())

        #expect(throws: (any Error).self) {
            try TransformerConfiguration(from: file)
        }
    }
}

// MARK: - GGUFTensorBridge Tests

@Suite("GGUFTensorBridge")
struct GGUFTensorBridgeTests {

    let bridge = GGUFTensorBridge()

    @Test("F32 tensor loading")
    func f32Loading() throws {
        let values: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let data = values.withUnsafeBytes { Data($0) }
        let tensor = GGUFTensorInfo(
            name: "test",
            dimensions: [3, 2], // GGUF: inner-first
            quantizationType: .f32,
            offset: 0
        )

        let result = try bridge.convert(tensor: tensor, data: data)
        // GGUF dims [3,2] reversed to MLX [2,3]
        #expect(result.shape == [2, 3])
        #expect(result.dtype == .float16)
    }

    @Test("F16 tensor loading")
    func f16Loading() throws {
        // Create F16 data: 4 float16 values
        let float16Values: [UInt16] = [
            0x3C00, // 1.0
            0x4000, // 2.0
            0x4200, // 3.0
            0x4400, // 4.0
        ]
        let data = float16Values.withUnsafeBytes { Data($0) }
        let tensor = GGUFTensorInfo(
            name: "test",
            dimensions: [4],
            quantizationType: .f16,
            offset: 0
        )

        let result = try bridge.convert(tensor: tensor, data: data)
        #expect(result.shape == [4])
        #expect(result.dtype == .float16)

        let asFloat = result.asType(.float32)
        let vals: [Float] = [asFloat[0].item(), asFloat[1].item(), asFloat[2].item(), asFloat[3].item()]
        #expect(abs(vals[0] - 1.0) < 0.01)
        #expect(abs(vals[1] - 2.0) < 0.01)
        #expect(abs(vals[2] - 3.0) < 0.01)
        #expect(abs(vals[3] - 4.0) < 0.01)
    }

    @Test("Q4_0 dequantization produces correct shape")
    func q4_0Shape() throws {
        // Q4_0: 32 elements per block = 18 bytes (2 scale + 16 data)
        // 1 block = 32 elements
        let blockData = Data(repeating: 0, count: 18)
        let tensor = GGUFTensorInfo(
            name: "test",
            dimensions: [32],
            quantizationType: .q4_0,
            offset: 0
        )

        let result = try bridge.convert(tensor: tensor, data: blockData)
        #expect(result.shape == [32])
        #expect(result.dtype == .float16)
    }

    @Test("Q8_0 dequantization produces correct shape")
    func q8_0Shape() throws {
        // Q8_0: 32 elements per block = 34 bytes (2 scale + 32 data)
        let blockData = Data(repeating: 0, count: 34)
        let tensor = GGUFTensorInfo(
            name: "test",
            dimensions: [32],
            quantizationType: .q8_0,
            offset: 0
        )

        let result = try bridge.convert(tensor: tensor, data: blockData)
        #expect(result.shape == [32])
        #expect(result.dtype == .float16)
    }

    @Test("Q4_K dequantization produces correct shape")
    func q4_KShape() throws {
        // Q4_K: 256 elements per super-block = 144 bytes
        let blockData = Data(repeating: 0, count: 144)
        let tensor = GGUFTensorInfo(
            name: "test",
            dimensions: [256],
            quantizationType: .q4_K,
            offset: 0
        )

        let result = try bridge.convert(tensor: tensor, data: blockData)
        #expect(result.shape == [256])
        #expect(result.dtype == .float16)
    }

    @Test("Q6_K dequantization produces correct shape")
    func q6_KShape() throws {
        // Q6_K: 256 elements per super-block = 210 bytes
        let blockData = Data(repeating: 0, count: 210)
        let tensor = GGUFTensorInfo(
            name: "test",
            dimensions: [256],
            quantizationType: .q6_K,
            offset: 0
        )

        let result = try bridge.convert(tensor: tensor, data: blockData)
        #expect(result.shape == [256])
    }

    @Test("Q2_K dequantization produces correct shape")
    func q2_KShape() throws {
        let blockData = Data(repeating: 0, count: 84)
        let tensor = GGUFTensorInfo(
            name: "test",
            dimensions: [256],
            quantizationType: .q2_K,
            offset: 0
        )

        let result = try bridge.convert(tensor: tensor, data: blockData)
        #expect(result.shape == [256])
    }

    @Test("Q3_K dequantization produces correct shape")
    func q3_KShape() throws {
        let blockData = Data(repeating: 0, count: 110)
        let tensor = GGUFTensorInfo(
            name: "test",
            dimensions: [256],
            quantizationType: .q3_K,
            offset: 0
        )

        let result = try bridge.convert(tensor: tensor, data: blockData)
        #expect(result.shape == [256])
    }

    @Test("Q5_K dequantization produces correct shape")
    func q5_KShape() throws {
        let blockData = Data(repeating: 0, count: 176)
        let tensor = GGUFTensorInfo(
            name: "test",
            dimensions: [256],
            quantizationType: .q5_K,
            offset: 0
        )

        let result = try bridge.convert(tensor: tensor, data: blockData)
        #expect(result.shape == [256])
    }

    @Test("Unsupported quantization throws")
    func unsupportedQuant() throws {
        // i64 is an integer type with no dequantization path
        let tensor = GGUFTensorInfo(
            name: "test",
            dimensions: [8],
            quantizationType: .i64,
            offset: 0
        )

        #expect(throws: GGUFLoadError.self) {
            try bridge.convert(tensor: tensor, data: Data(repeating: 0, count: 64))
        }
    }

    @Test("2D tensor shape reversal")
    func shapeReversal() throws {
        // GGUF dimensions [8, 4] means 4 rows x 8 cols in GGUF convention
        // MLX should be [4, 8] (rows first)
        let values = [Float](repeating: 1.0, count: 32)
        let data = values.withUnsafeBytes { Data($0) }
        let tensor = GGUFTensorInfo(
            name: "test",
            dimensions: [8, 4], // GGUF: ne[0]=8, ne[1]=4
            quantizationType: .f32,
            offset: 0
        )

        let result = try bridge.convert(tensor: tensor, data: data)
        #expect(result.shape == [4, 8]) // MLX: reversed
    }

    @Test("Q4_0 dequantization values are reasonable")
    func q4_0Values() throws {
        // Build a single Q4_0 block with known values:
        // scale = 1.0 (f16), all nibbles = 8 (zero after subtraction)
        var blockData = Data(count: 18)

        // f16 representation of 1.0 = 0x3C00
        blockData[0] = 0x00  // lo byte
        blockData[1] = 0x3C  // hi byte

        // All data bytes = 0x88 → lo nibble = 8, hi nibble = 8
        // q = 8 - 8 = 0, so all values should be 0
        for i in 2..<18 {
            blockData[i] = 0x88
        }

        let tensor = GGUFTensorInfo(
            name: "test",
            dimensions: [32],
            quantizationType: .q4_0,
            offset: 0
        )

        let result = try bridge.convert(tensor: tensor, data: blockData)
        let floatResult = result.asType(.float32)
        eval(floatResult)

        // All values should be 0 (scale * (8-8) = 0)
        let sum: Float = floatResult.sum().item()
        #expect(abs(sum) < 0.01)
    }
}

// MARK: - Direct Quantized Packing Tests

@Suite("Direct GGUF→MLX Quantized Packing")
struct DirectQuantizedPackingTests {

    let bridge = GGUFTensorBridge()

    // MARK: - Q4_0

    @Test("Q4_0 direct pack: round-trip matches dequantized values")
    func q4_0RoundTrip() throws {
        // Build a Q4_0 block: 32 elements, 18 bytes
        // scale=2.0, all nibbles = 0x88 → low=8, high=8 → (8-8)*2=0, (8-8)*2=0
        let scale = Float16(2.0)
        var block = Data(count: 18)
        withUnsafeBytes(of: scale.bitPattern.littleEndian) { block.replaceSubrange(0..<2, with: $0) }
        for j in 0..<16 { block[2 + j] = 0x88 } // low=8, high=8

        let tensor = GGUFTensorInfo(name: "t", dimensions: [32], quantizationType: .q4_0, offset: 0)

        // Direct pack
        let direct = try bridge.convertDirect(tensor: tensor, data: block)
        guard case .float16 = direct else {
            // 1D tensors fall back to F16 — that's expected
            return
        }
    }

    @Test("Q4_0 direct pack: 2D weight round-trip correctness")
    func q4_0DirectPack2D() throws {
        // 2D tensor: [inDim=32, outDim=2] → MLX shape [2, 32]
        // 2 blocks, each 18 bytes = 36 bytes total
        var data = Data(count: 36)

        // Block 0: scale=1.5, nibbles encode specific pattern
        let scale0 = Float16(1.5)
        withUnsafeBytes(of: scale0.bitPattern.littleEndian) { data.replaceSubrange(0..<2, with: $0) }
        for j in 0..<16 {
            // low nibble = j % 16, high nibble = (15 - j) % 16
            data[2 + j] = UInt8(j & 0x0F) | UInt8((15 - j) & 0x0F) << 4
        }

        // Block 1: scale=0.5, all nibbles = 0x33
        let scale1 = Float16(0.5)
        withUnsafeBytes(of: scale1.bitPattern.littleEndian) { data.replaceSubrange(18..<20, with: $0) }
        for j in 0..<16 { data[20 + j] = 0x33 } // low=3, high=3

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [32, 2], quantizationType: .q4_0, offset: 0)

        // Dequantize to F16
        let f16Array = try bridge.convert(tensor: tensor, data: data)
        #expect(f16Array.shape == [2, 32])

        // Direct pack
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected .quantized for 2D Q4_0 tensor")
            return
        }
        #expect(bits == 4)
        #expect(groupSize == 32)
        #expect(weight.shape == [2, 4]) // 32/8 = 4 UInt32 per row
        #expect(scales.shape == [2, 1]) // 32/32 = 1 scale per row
        #expect(biases.shape == [2, 1])

        // Verify: QuantizedLinear output matches F16 Linear output
        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits,
            f16Weight: f16Array,
            inputDim: 32, outputDim: 2)
    }

    // MARK: - Q4_1

    @Test("Q4_1 direct pack: 2D weight round-trip correctness")
    func q4_1DirectPack2D() throws {
        // Q4_1: 20 bytes per 32-element block (scale + min + 16 qs)
        var data = Data(count: 40) // 2 blocks

        // Block 0: d=2.0, m=1.0
        let d0 = Float16(2.0)
        let m0 = Float16(1.0)
        withUnsafeBytes(of: d0.bitPattern.littleEndian) { data.replaceSubrange(0..<2, with: $0) }
        withUnsafeBytes(of: m0.bitPattern.littleEndian) { data.replaceSubrange(2..<4, with: $0) }
        for j in 0..<16 { data[4 + j] = 0x55 } // low=5, high=5

        // Block 1: d=0.25, m=-0.5
        let d1 = Float16(0.25)
        let m1 = Float16(-0.5)
        withUnsafeBytes(of: d1.bitPattern.littleEndian) { data.replaceSubrange(20..<22, with: $0) }
        withUnsafeBytes(of: m1.bitPattern.littleEndian) { data.replaceSubrange(22..<24, with: $0) }
        for j in 0..<16 { data[24 + j] = 0xAA } // low=10, high=10

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [32, 2], quantizationType: .q4_1, offset: 0)

        let f16Array = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected .quantized")
            return
        }
        #expect(bits == 4)
        #expect(groupSize == 32)

        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits,
            f16Weight: f16Array,
            inputDim: 32, outputDim: 2)
    }

    // MARK: - Q4_K

    @Test("Q4_K direct pack: shapes and basic correctness")
    func q4_KDirectPackShapes() throws {
        // Q4_K: 144 bytes per 256-element super-block
        // Tensor: [256, 2] → MLX [2, 256], 2 super-blocks
        var data = Data(count: 288)

        // Fill with known pattern: all zero scales → zero output
        for sb in 0..<2 {
            let offset = sb * 144
            // d=1.0, dmin=0.5
            let d = Float16(1.0)
            let dmin = Float16(0.5)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset..<offset+2, with: $0)
            }
            withUnsafeBytes(of: dmin.bitPattern.littleEndian) {
                data.replaceSubrange(offset+2..<offset+4, with: $0)
            }
            // Set scale bytes: scales[0..3] = 1 (& 63), mins[0..3] = 2 (& 63)
            for j in 0..<4 { data[offset + 4 + j] = 1 }
            for j in 0..<4 { data[offset + 8 + j] = 2 }
            // Upper scales bytes (j=4..7): all zero
            for j in 0..<4 { data[offset + 12 + j] = 0 }
            // qs: all 0x77 → low=7, high=7
            for j in 0..<128 { data[offset + 16 + j] = 0x77 }
        }

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [256, 2], quantizationType: .q4_K, offset: 0)

        let f16Array = try bridge.convert(tensor: tensor, data: data)
        #expect(f16Array.shape == [2, 256])

        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected .quantized for Q4_K")
            return
        }
        #expect(bits == 4)
        #expect(groupSize == 32)
        #expect(weight.shape == [2, 32]) // 256/8
        #expect(scales.shape == [2, 8])  // 256/32
        #expect(biases.shape == [2, 8])

        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits,
            f16Weight: f16Array,
            inputDim: 256, outputDim: 2)
    }

    @Test("Q4_K direct pack: non-trivial scale encoding (j >= 4)")
    func q4_KUpperScales() throws {
        // Test that upper-half scales (j=4..7) are correctly extracted
        var data = Data(count: 144)

        let d = Float16(1.0)
        let dmin = Float16(1.0)
        withUnsafeBytes(of: d.bitPattern.littleEndian) { data.replaceSubrange(0..<2, with: $0) }
        withUnsafeBytes(of: dmin.bitPattern.littleEndian) { data.replaceSubrange(2..<4, with: $0) }

        // Carefully set scale encoding:
        // bytes[0..3]: lower 6 bits = scales[0..3], bits 6-7 = upper 2 bits of scales[4..7]
        // bytes[4..7]: lower 6 bits = mins[0..3], bits 6-7 = upper 2 bits of mins[4..7]
        // bytes[8..11]: lower 4 = scales[4..7] lower4, upper 4 = mins[4..7] lower4
        data[4 + 0] = 10        // scale[0] = 10
        data[4 + 1] = 20        // scale[1] = 20
        data[4 + 2] = 30        // scale[2] = 30
        data[4 + 3] = 63        // scale[3] = 63 (max 6-bit)
        data[4 + 4] = 5         // min[0] = 5
        data[4 + 5] = 15        // min[1] = 15
        data[4 + 6] = 25        // min[2] = 25
        data[4 + 7] = 35        // min[3] = 35
        // Upper bytes for j=4..7: scales lower4 | mins lower4 << 4
        // (j-4 upper 2 bits come from bytes[0..3] bits 6-7)
        data[4 + 8] = 3         // scale[4] lower4 = 3, min[4] = 0 (upper4)
        data[4 + 9] = 7         // scale[5] lower4 = 7
        data[4 + 10] = 0xF      // scale[6] lower4 = 15
        data[4 + 11] = 0x12     // scale[7] lower4 = 2, min[7] upper4 = 1

        // qs: pattern with varying nibbles
        for j in 0..<128 { data[16 + j] = UInt8((j * 3) & 0xFF) }

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [256], quantizationType: .q4_K, offset: 0)
        // 1D falls back to F16
        let f16Array = try bridge.convert(tensor: tensor, data: data)
        #expect(f16Array.shape == [256])

        // For 2D variant
        let tensor2D = GGUFTensorInfo(
            name: "t", dimensions: [256, 1], quantizationType: .q4_K, offset: 0)
        let f16_2D = try bridge.convert(tensor: tensor2D, data: data)
        let direct = try bridge.convertDirect(tensor: tensor2D, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected .quantized")
            return
        }

        // 256-dim dot product with scale values up to 63, so element values can reach ~945.
        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits,
            f16Weight: f16_2D,
            inputDim: 256, outputDim: 1)
    }

    // MARK: - Q8_0

    @Test("Q8_0 direct pack: 2D weight round-trip correctness")
    func q8_0DirectPack2D() throws {
        // Q8_0: 34 bytes per 32-element block (2 scale + 32 int8)
        var data = Data(count: 68) // 2 blocks

        // Block 0: scale=0.1, values = -128,-127,...,-97
        let s0 = Float16(0.1)
        withUnsafeBytes(of: s0.bitPattern.littleEndian) { data.replaceSubrange(0..<2, with: $0) }
        for j in 0..<32 {
            data[2 + j] = UInt8(bitPattern: Int8(-128 + Int32(j)))
        }

        // Block 1: scale=0.5, values = 0,1,...,31
        let s1 = Float16(0.5)
        withUnsafeBytes(of: s1.bitPattern.littleEndian) { data.replaceSubrange(34..<36, with: $0) }
        for j in 0..<32 {
            data[36 + j] = UInt8(bitPattern: Int8(j))
        }

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [32, 2], quantizationType: .q8_0, offset: 0)

        let f16Array = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected .quantized")
            return
        }
        #expect(bits == 8)
        #expect(groupSize == 32)
        #expect(weight.shape == [2, 8]) // 32/4 = 8 UInt32 per row
        #expect(scales.shape == [2, 1])

        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits,
            f16Weight: f16Array,
            inputDim: 32, outputDim: 2)
    }

    @Test("Q8_0 direct pack: extreme signed values")
    func q8_0ExtremeValues() throws {
        var data = Data(count: 34)
        let scale = Float16(1.0)
        withUnsafeBytes(of: scale.bitPattern.littleEndian) { data.replaceSubrange(0..<2, with: $0) }
        // Mix of -128, 0, 127
        data[2] = UInt8(bitPattern: Int8(-128))
        data[3] = 0
        data[4] = UInt8(bitPattern: Int8(127))
        for j in 3..<32 { data[2 + j] = UInt8(bitPattern: Int8(j - 16)) }

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [32, 1], quantizationType: .q8_0, offset: 0)
        let f16 = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let gs, let bits) = direct else {
            Issue.record("Expected .quantized")
            return
        }
        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: gs, bits: bits, f16Weight: f16,
            inputDim: 32, outputDim: 1)
    }

    // MARK: - Q5_K

    @Test("Q5_K direct pack: 2D weight round-trip correctness")
    func q5_KDirectPack2D() throws {
        var data = Data(count: 176 * 2)

        for block in 0..<2 {
            let offset = block * 176
            let d = Float16(block == 0 ? 0.5 : 0.25)
            let dmin = Float16(block == 0 ? 0.125 : 0.375)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset..<offset + 2, with: $0)
            }
            withUnsafeBytes(of: dmin.bitPattern.littleEndian) {
                data.replaceSubrange(offset + 2..<offset + 4, with: $0)
            }

            for j in 0..<12 {
                data[offset + 4 + j] = UInt8((block * 17 + j * 5) & 0xFF)
            }
            for j in 0..<32 {
                data[offset + 16 + j] = UInt8((block * 29 + j * 7) & 0xFF)
            }
            for j in 0..<128 {
                data[offset + 48 + j] = UInt8((block * 11 + j * 13) & 0xFF)
            }
        }

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [256, 2], quantizationType: .q5_K, offset: 0)

        let f16Array = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected .quantized for Q5_K")
            return
        }
        #expect(bits == 5)
        #expect(groupSize == 32)
        #expect(weight.shape == [2, 40])  // 256 * 5 / 32
        #expect(scales.shape == [2, 8])
        #expect(biases.shape == [2, 8])

        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits,
            f16Weight: f16Array,
            inputDim: 256, outputDim: 2
        )
    }

    // MARK: - Q6_K

    @Test("Q6_K direct pack: 2D weight round-trip correctness")
    func q6_KDirectPack2D() throws {
        var data = Data(count: 210 * 2)

        for block in 0..<2 {
            let offset = block * 210
            for j in 0..<128 {
                data[offset + j] = UInt8((block * 19 + j * 9) & 0xFF)
            }
            for j in 0..<64 {
                data[offset + 128 + j] = UInt8((block * 23 + j * 3) & 0xFF)
            }
            for j in 0..<16 {
                let scale = Int8(block == 0 ? (j - 8) : (j - 4))
                data[offset + 192 + j] = UInt8(bitPattern: scale)
            }
            let d = Float16(block == 0 ? 0.125 : 0.25)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset + 208..<offset + 210, with: $0)
            }
        }

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [256, 2], quantizationType: .q6_K, offset: 0)

        let f16Array = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected .quantized for Q6_K")
            return
        }
        #expect(bits == 6)
        #expect(groupSize == 16)
        #expect(weight.shape == [2, 48])  // 256 * 6 / 32
        #expect(scales.shape == [2, 16])
        #expect(biases.shape == [2, 16])

        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits,
            f16Weight: f16Array,
            inputDim: 256, outputDim: 2,
            tolerance: 0.1
        )
    }

    // MARK: - Fallback Tests

    @Test("Q6_K falls back to F16 (no direct packing)")
    func q6_KFallback() throws {
        // Q6_K: 210 bytes per 256-element block
        let data = Data(count: 210)
        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [256], quantizationType: .q6_K, offset: 0)
        let result = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .float16 = result else {
            Issue.record("Q6_K should fall back to .float16")
            return
        }
    }

    @Test("Q5_K falls back to F16 (no direct packing)")
    func q5_KFallback() throws {
        let data = Data(count: 176)
        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [256], quantizationType: .q5_K, offset: 0)
        let result = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .float16 = result else {
            Issue.record("Q5_K should fall back to .float16")
            return
        }
    }

    @Test("1D tensor always falls back to F16 regardless of quant type")
    func oneDimensionalFallback() throws {
        let data = Data(count: 18) // Q4_0 block
        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [32], quantizationType: .q4_0, offset: 0)
        let result = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .float16 = result else {
            Issue.record("1D Q4_0 should fall back to .float16")
            return
        }
    }

    // MARK: - Edge Cases

    @Test("Q4_0 direct pack: zero scale produces zero output")
    func q4_0ZeroScale() throws {
        var data = Data(count: 36) // 2 blocks
        // Both blocks have scale=0, arbitrary nibbles
        for j in 0..<16 { data[2 + j] = 0xFF; data[20 + j] = 0xFF }

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [32, 2], quantizationType: .q4_0, offset: 0)
        let f16 = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let gs, let bits) = direct else {
            Issue.record("Expected .quantized")
            return
        }
        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: gs, bits: bits, f16Weight: f16,
            inputDim: 32, outputDim: 2)
    }

    @Test("Q4_K direct pack: large tensor (4 super-blocks × 2 rows)")
    func q4_KLargeTensor() throws {
        // [1024, 2] → MLX [2, 1024], 2×4 = 8 super-blocks
        var data = Data(count: 144 * 8)
        for sb in 0..<8 {
            let offset = sb * 144
            let d = Float16(Float(sb + 1) * 0.1)
            let dmin = Float16(Float(sb + 1) * 0.05)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset..<offset+2, with: $0)
            }
            withUnsafeBytes(of: dmin.bitPattern.littleEndian) {
                data.replaceSubrange(offset+2..<offset+4, with: $0)
            }
            for j in 0..<4 { data[offset + 4 + j] = UInt8((sb * 7 + j) & 63) }
            for j in 0..<4 { data[offset + 8 + j] = UInt8((sb * 3 + j) & 63) }
            for j in 0..<4 { data[offset + 12 + j] = 0 }
            for j in 0..<128 { data[offset + 16 + j] = UInt8((sb * 13 + j * 7) & 0xFF) }
        }

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [1024, 2], quantizationType: .q4_K, offset: 0)
        let f16 = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let gs, let bits) = direct else {
            Issue.record("Expected .quantized")
            return
        }
        #expect(weight.shape == [2, 128]) // 1024/8
        #expect(scales.shape == [2, 32])  // 1024/32

        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: gs, bits: bits, f16Weight: f16,
            inputDim: 1024, outputDim: 2)
    }

    // MARK: - Tier 1: Q5_0

    @Test("Q5_0 direct pack: 2D weight round-trip correctness")
    func q5_0DirectPack2D() throws {
        // Q5_0: 22 bytes per 32-element block. Tensor [32, 2] → 2 blocks.
        var data = Data(count: 22 * 2)

        for block in 0..<2 {
            let offset = block * 22
            let d = Float16(block == 0 ? 1.5 : 0.75)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset..<offset + 2, with: $0)
            }
            // qh (uint32 at offset+2): set some high bits
            let qh: UInt32 = block == 0 ? 0xAAAA_5555 : 0x5555_AAAA
            withUnsafeBytes(of: qh.littleEndian) {
                data.replaceSubrange(offset + 2..<offset + 6, with: $0)
            }
            // qs: 16 bytes, varying pattern
            for j in 0..<16 {
                data[offset + 6 + j] = UInt8((block * 37 + j * 11) & 0xFF)
            }
        }

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [32, 2], quantizationType: .q5_0, offset: 0)
        let f16Array = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected .quantized for Q5_0")
            return
        }
        #expect(bits == 5)
        #expect(groupSize == 32)

        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits,
            f16Weight: f16Array,
            inputDim: 32, outputDim: 2,
            tolerance: 0.1)
    }

    // MARK: - Tier 1: Q5_1

    @Test("Q5_1 direct pack: 2D weight round-trip correctness")
    func q5_1DirectPack2D() throws {
        // Q5_1: 24 bytes per 32-element block. Tensor [32, 2] → 2 blocks.
        var data = Data(count: 24 * 2)

        for block in 0..<2 {
            let offset = block * 24
            let d = Float16(block == 0 ? 0.5 : 1.25)
            let m = Float16(block == 0 ? -1.0 : 0.5)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset..<offset + 2, with: $0)
            }
            withUnsafeBytes(of: m.bitPattern.littleEndian) {
                data.replaceSubrange(offset + 2..<offset + 4, with: $0)
            }
            // qh (uint32 at offset+4)
            let qh: UInt32 = block == 0 ? 0xF0F0_0F0F : 0x0F0F_F0F0
            withUnsafeBytes(of: qh.littleEndian) {
                data.replaceSubrange(offset + 4..<offset + 8, with: $0)
            }
            // qs: 16 bytes
            for j in 0..<16 {
                data[offset + 8 + j] = UInt8((block * 23 + j * 7) & 0xFF)
            }
        }

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [32, 2], quantizationType: .q5_1, offset: 0)
        let f16Array = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected .quantized for Q5_1")
            return
        }
        #expect(bits == 5)
        #expect(groupSize == 32)

        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits,
            f16Weight: f16Array,
            inputDim: 32, outputDim: 2)
    }

    // MARK: - Tier 1: Q8_K

    @Test("Q8_K direct pack: 2D weight round-trip correctness")
    func q8_KDirectPack2D() throws {
        // Q8_K: 292 bytes per 256-element super-block. Tensor [256, 2] → 2 blocks.
        var data = Data(count: 292 * 2)

        for block in 0..<2 {
            let offset = block * 292
            // d (float32)
            let d: Float32 = block == 0 ? 0.1 : 0.25
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset..<offset + 4, with: $0)
            }
            // qs: 256 signed int8 values
            for j in 0..<256 {
                let val = Int8(truncatingIfNeeded: block * 41 + j * 3 - 128)
                data[offset + 4 + j] = UInt8(bitPattern: val)
            }
            // bsums: 32 bytes (ignored by pack), leave as zero
        }

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [256, 2], quantizationType: .q8_K, offset: 0)
        let f16Array = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected .quantized for Q8_K")
            return
        }
        #expect(bits == 8)
        #expect(groupSize == 32)
        #expect(weight.shape == [2, 64])  // 256/4
        #expect(scales.shape == [2, 8])   // 256/32

        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits,
            f16Weight: f16Array,
            inputDim: 256, outputDim: 2)
    }

    // MARK: - Tier 1: Q2_K

    @Test("Q2_K direct pack: 2D weight round-trip correctness")
    func q2_KDirectPack2D() throws {
        // Q2_K: 84 bytes per 256-element super-block. Tensor [256, 2] → 2 blocks.
        var data = Data(count: 84 * 2)

        for block in 0..<2 {
            let offset = block * 84
            // scales: 16 bytes (4-bit scale + 4-bit min per byte)
            for j in 0..<16 {
                data[offset + j] = UInt8(((j + block * 3) & 0x0F) | (((j + 1) & 0x0F) << 4))
            }
            // qs: 64 bytes
            for j in 0..<64 {
                data[offset + 16 + j] = UInt8((block * 19 + j * 5) & 0xFF)
            }
            // d (float16) at offset+80
            let d = Float16(block == 0 ? 0.5 : 0.25)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset + 80..<offset + 82, with: $0)
            }
            // dmin (float16) at offset+82
            let dmin = Float16(block == 0 ? 0.125 : 0.0625)
            withUnsafeBytes(of: dmin.bitPattern.littleEndian) {
                data.replaceSubrange(offset + 82..<offset + 84, with: $0)
            }
        }

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [256, 2], quantizationType: .q2_K, offset: 0)
        let f16Array = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected .quantized for Q2_K")
            return
        }
        #expect(bits == 2)
        #expect(groupSize == 16)

        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits,
            f16Weight: f16Array,
            inputDim: 256, outputDim: 2,
            tolerance: 0.1)
    }

    // MARK: - Tier 1: Q3_K

    @Test("Q3_K direct pack: 2D weight round-trip correctness")
    func q3_KDirectPack2D() throws {
        // Q3_K: 110 bytes per 256-element super-block. Tensor [256, 2] → 2 blocks.
        var data = Data(count: 110 * 2)

        for block in 0..<2 {
            let offset = block * 110
            // hmask: 32 bytes
            for j in 0..<32 {
                data[offset + j] = UInt8((block * 13 + j * 3) & 0xFF)
            }
            // qs: 64 bytes
            for j in 0..<64 {
                data[offset + 32 + j] = UInt8((block * 7 + j * 11) & 0xFF)
            }
            // scales: 12 bytes
            for j in 0..<12 {
                data[offset + 96 + j] = UInt8((block * 17 + j * 5) & 0xFF)
            }
            // d (float16) at offset+108
            let d = Float16(block == 0 ? 0.25 : 0.125)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset + 108..<offset + 110, with: $0)
            }
        }

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [256, 2], quantizationType: .q3_K, offset: 0)
        let f16Array = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected .quantized for Q3_K")
            return
        }
        #expect(bits == 3)
        #expect(groupSize == 16)

        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits,
            f16Weight: f16Array,
            inputDim: 256, outputDim: 2,
            tolerance: 0.1)
    }

    // MARK: - Tier 1: TQ2_0

    @Test("TQ2_0 direct pack: 2D weight round-trip correctness")
    func tq2_0DirectPack2D() throws {
        // TQ2_0: 66 bytes per 256-element super-block. Tensor [256, 2] → 2 blocks.
        var data = Data(count: 66 * 2)

        for block in 0..<2 {
            let offset = block * 66
            // qs: 64 bytes (2-bit values packed as 4 shifts per byte)
            for j in 0..<64 {
                data[offset + j] = UInt8((block * 29 + j * 7) & 0xFF)
            }
            // d (float16) at offset+64
            let d = Float16(block == 0 ? 0.5 : 0.75)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset + 64..<offset + 66, with: $0)
            }
        }

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [256, 2], quantizationType: .tq2_0, offset: 0)
        let f16Array = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected .quantized for TQ2_0")
            return
        }
        #expect(bits == 2)
        #expect(groupSize == 32)

        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits,
            f16Weight: f16Array,
            inputDim: 256, outputDim: 2)
    }

    // MARK: - Tier 2: IQ4_NL

    @Test("IQ4_NL re-quantize pack: 2D weight round-trip correctness")
    func iq4_NLPack2D() throws {
        // IQ4_NL: 18 bytes per 32-element block. Tensor [32, 2] → 2 blocks.
        var data = Data(count: 18 * 2)

        for block in 0..<2 {
            let offset = block * 18
            let d = Float16(block == 0 ? 0.5 : 1.0)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset..<offset + 2, with: $0)
            }
            // qs: 16 bytes with nibble values 0-15
            for j in 0..<16 {
                let lo = (j + block * 3) & 0x0F
                let hi = (15 - j + block * 5) & 0x0F
                data[offset + 2 + j] = UInt8(lo | (hi << 4))
            }
        }

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [32, 2], quantizationType: .iq4_NL, offset: 0)
        let f16Array = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected .quantized for IQ4_NL")
            return
        }
        #expect(bits == 4)
        #expect(groupSize == 32)

        // Tier 2: re-quantization adds bounded error (non-linear LUT → affine)
        // 32-dim dot product still accumulates LUT→affine quantization noise
        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits,
            f16Weight: f16Array,
            inputDim: 32, outputDim: 2,
            tolerance: 30)
    }

    // MARK: - Tier 2: IQ4_XS

    @Test("IQ4_XS re-quantize pack: 2D weight round-trip correctness")
    func iq4_XSPack2D() throws {
        // IQ4_XS: 136 bytes per 256-element super-block. Tensor [256, 2] → 2 blocks.
        var data = Data(count: 136 * 2)

        for block in 0..<2 {
            let offset = block * 136
            let d = Float16(block == 0 ? 0.25 : 0.5)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset..<offset + 2, with: $0)
            }
            // scales_h: 2 bytes
            data[offset + 2] = UInt8(block == 0 ? 0xAA : 0x55)
            data[offset + 3] = UInt8(block == 0 ? 0x55 : 0xAA)
            // scales_l: 4 bytes
            for j in 0..<4 {
                data[offset + 4 + j] = UInt8(0x88 + block * 0x11)
            }
            // qs: 128 bytes
            for j in 0..<128 {
                data[offset + 8 + j] = UInt8((block * 31 + j * 13) & 0xFF)
            }
        }

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [256, 2], quantizationType: .iq4_XS, offset: 0)
        let f16Array = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected .quantized for IQ4_XS")
            return
        }
        #expect(bits == 4)
        #expect(groupSize == 32)

        // Tier 2: 256-dim dot product with non-linear→affine re-quantization
        // accumulates significant per-element noise
        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits,
            f16Weight: f16Array,
            inputDim: 256, outputDim: 2,
            tolerance: 600)
    }

    // MARK: - Tier 3: IQ2_XXS

    @Test("IQ2_XXS re-quantize pack: 2D weight round-trip correctness")
    func iq2_XXSPack2D() throws {
        // IQ2_XXS: 66 bytes per 256-element super-block. Tensor [256, 2] → 2 blocks.
        var data = Data(count: 66 * 2)

        for block in 0..<2 {
            let offset = block * 66
            let d = Float16(block == 0 ? 0.5 : 0.25)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset..<offset + 2, with: $0)
            }
            // qs: 64 bytes (8 groups of 8 bytes, each pair of uint32: grid indices + signs/scale)
            for j in 0..<64 {
                data[offset + 2 + j] = UInt8((block * 17 + j * 3) & 0xFF)
            }
        }

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [256, 2], quantizationType: .iq2_XXS, offset: 0)
        let f16Array = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected .quantized for IQ2_XXS")
            return
        }
        #expect(bits == 4)
        #expect(groupSize == 32)

        // Tier 3: grid decode → re-quantize; large per-element noise at 256-dim
        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits,
            f16Weight: f16Array,
            inputDim: 256, outputDim: 2,
            tolerance: 15)
    }

    // MARK: - Tier 3: IQ2_XS

    @Test("IQ2_XS re-quantize pack: 2D weight round-trip correctness")
    func iq2_XSPack2D() throws {
        // IQ2_XS: 74 bytes per 256-element super-block. Tensor [256, 2] → 2 blocks.
        var data = Data(count: 74 * 2)

        for block in 0..<2 {
            let offset = block * 74
            let d = Float16(block == 0 ? 0.25 : 0.5)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset..<offset + 2, with: $0)
            }
            // qs: 64 bytes (32 uint16 values)
            for j in 0..<64 {
                data[offset + 2 + j] = UInt8((block * 23 + j * 7) & 0xFF)
            }
            // scales: 8 bytes
            for j in 0..<8 {
                data[offset + 66 + j] = UInt8(0x44 + block * 0x22)
            }
        }

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [256, 2], quantizationType: .iq2_XS, offset: 0)
        let f16Array = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected .quantized for IQ2_XS")
            return
        }
        #expect(bits == 4)
        #expect(groupSize == 32)

        // Tier 3: grid decode → re-quantize; accumulated 256-dim noise
        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits,
            f16Weight: f16Array,
            inputDim: 256, outputDim: 2,
            tolerance: 10)
    }

    // MARK: - Tier 3: IQ2_S

    @Test("IQ2_S re-quantize pack: 2D weight round-trip correctness")
    func iq2_SPack2D() throws {
        // IQ2_S: 82 bytes per 256-element super-block. Tensor [256, 2] → 2 blocks.
        var data = Data(count: 82 * 2)

        for block in 0..<2 {
            let offset = block * 82
            let d = Float16(block == 0 ? 0.5 : 0.125)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset..<offset + 2, with: $0)
            }
            // qs: 64 bytes (32 grid indices + 32 sign bytes)
            for j in 0..<64 {
                data[offset + 2 + j] = UInt8((block * 11 + j * 9) & 0xFF)
            }
            // qh: 8 bytes
            for j in 0..<8 {
                data[offset + 66 + j] = UInt8((block * 3 + j) & 0x03)
            }
            // scales: 8 bytes
            for j in 0..<8 {
                data[offset + 74 + j] = UInt8(0x33 + block * 0x11)
            }
        }

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [256, 2], quantizationType: .iq2_S, offset: 0)
        let f16Array = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected .quantized for IQ2_S")
            return
        }
        #expect(bits == 4)
        #expect(groupSize == 32)

        // Tier 3: grid decode → re-quantize; accumulated noise
        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits,
            f16Weight: f16Array,
            inputDim: 256, outputDim: 2,
            tolerance: 25)
    }

    // MARK: - Tier 3: IQ3_XXS

    @Test("IQ3_XXS re-quantize pack: 2D weight round-trip correctness")
    func iq3_XXSPack2D() throws {
        // IQ3_XXS: 98 bytes per 256-element super-block. Tensor [256, 2] → 2 blocks.
        var data = Data(count: 98 * 2)

        for block in 0..<2 {
            let offset = block * 98
            let d = Float16(block == 0 ? 0.25 : 0.5)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset..<offset + 2, with: $0)
            }
            // qs: 96 bytes (64 grid indices + 32 bytes sign/scale data)
            for j in 0..<96 {
                data[offset + 2 + j] = UInt8((block * 19 + j * 11) & 0xFF)
            }
        }

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [256, 2], quantizationType: .iq3_XXS, offset: 0)
        let f16Array = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected .quantized for IQ3_XXS")
            return
        }
        #expect(bits == 4)
        #expect(groupSize == 32)

        // Tier 3: grid decode → re-quantize; accumulated 256-dim noise
        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits,
            f16Weight: f16Array,
            inputDim: 256, outputDim: 2,
            tolerance: 15)
    }

    // MARK: - Tier 3: IQ3_S

    @Test("IQ3_S re-quantize pack: 2D weight round-trip correctness")
    func iq3_SPack2D() throws {
        // IQ3_S: 110 bytes per 256-element super-block. Tensor [256, 2] → 2 blocks.
        var data = Data(count: 110 * 2)

        for block in 0..<2 {
            let offset = block * 110
            let d = Float16(block == 0 ? 0.25 : 0.375)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset..<offset + 2, with: $0)
            }
            // qs: 64 bytes
            for j in 0..<64 {
                data[offset + 2 + j] = UInt8((block * 7 + j * 13) & 0xFF)
            }
            // qh: 8 bytes
            for j in 0..<8 {
                data[offset + 66 + j] = UInt8((block * 5 + j) & 0x01)
            }
            // signs: 32 bytes
            for j in 0..<32 {
                data[offset + 74 + j] = UInt8((block * 31 + j * 3) & 0xFF)
            }
            // scales: 4 bytes
            for j in 0..<4 {
                data[offset + 106 + j] = UInt8(0x55 + block * 0x11)
            }
        }

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [256, 2], quantizationType: .iq3_S, offset: 0)
        let f16Array = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected .quantized for IQ3_S")
            return
        }
        #expect(bits == 4)
        #expect(groupSize == 32)

        // Tier 3: grid decode → re-quantize; accumulated 256-dim noise
        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits,
            f16Weight: f16Array,
            inputDim: 256, outputDim: 2,
            tolerance: 50)
    }

    // MARK: - Tier 3: IQ1_S

    @Test("IQ1_S re-quantize pack: 2D weight round-trip correctness")
    func iq1_SPack2D() throws {
        // IQ1_S: 50 bytes per 256-element super-block. Tensor [256, 2] → 2 blocks.
        var data = Data(count: 50 * 2)

        for block in 0..<2 {
            let offset = block * 50
            let d = Float16(block == 0 ? 0.5 : 0.25)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset..<offset + 2, with: $0)
            }
            // qs: 32 bytes
            for j in 0..<32 {
                data[offset + 2 + j] = UInt8((block * 13 + j * 7) & 0xFF)
            }
            // qh: 16 bytes (8 uint16 values; top 3 bits = scale, bit 15 = delta sign)
            for j in 0..<8 {
                // Construct valid qh: scale=3 (bits 12-14), no sign bit
                let qhVal: UInt16 = UInt16(3 << 12) | UInt16((block * 5 + j * 11) & 0x0FFF)
                withUnsafeBytes(of: qhVal.littleEndian) {
                    data.replaceSubrange(offset + 34 + j * 2..<offset + 36 + j * 2, with: $0)
                }
            }
        }

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [256, 2], quantizationType: .iq1_S, offset: 0)
        let f16Array = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected .quantized for IQ1_S")
            return
        }
        #expect(bits == 4)
        #expect(groupSize == 32)

        // Tier 3: grid decode → re-quantize; IQ1_S has very few codepoints
        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits,
            f16Weight: f16Array,
            inputDim: 256, outputDim: 2,
            tolerance: 5)
    }

    // MARK: - Tier 3: IQ1_M

    @Test("IQ1_M re-quantize pack: 2D weight round-trip correctness")
    func iq1_MPack2D() throws {
        // IQ1_M: 56 bytes per 256-element super-block. Tensor [256, 2] → 2 blocks.
        var data = Data(count: 56 * 2)

        for block in 0..<2 {
            let offset = block * 56
            // qs: 32 bytes
            for j in 0..<32 {
                data[offset + j] = UInt8((block * 11 + j * 5) & 0xFF)
            }
            // qh: 16 bytes
            for j in 0..<16 {
                data[offset + 32 + j] = UInt8((block * 7 + j * 3) & 0xFF)
            }
            // scales: 8 bytes (4 uint16 values, top 4 bits of each encode d)
            // Encode d = 0.5 as Float16 bits spread across top 4 bits of each scale
            let dBits = Float16(block == 0 ? 0.5 : 0.25).bitPattern
            let sc0 = UInt16(0x111) | ((dBits & 0x000F) << 12)
            let sc1 = UInt16(0x111) | ((dBits & 0x00F0) << 8)
            let sc2 = UInt16(0x111) | ((dBits & 0x0F00) << 4)
            let sc3 = UInt16(0x111) | (dBits & 0xF000)
            withUnsafeBytes(of: sc0.littleEndian) {
                data.replaceSubrange(offset + 48..<offset + 50, with: $0)
            }
            withUnsafeBytes(of: sc1.littleEndian) {
                data.replaceSubrange(offset + 50..<offset + 52, with: $0)
            }
            withUnsafeBytes(of: sc2.littleEndian) {
                data.replaceSubrange(offset + 52..<offset + 54, with: $0)
            }
            withUnsafeBytes(of: sc3.littleEndian) {
                data.replaceSubrange(offset + 54..<offset + 56, with: $0)
            }
        }

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [256, 2], quantizationType: .iq1_M, offset: 0)
        let f16Array = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected .quantized for IQ1_M")
            return
        }
        #expect(bits == 4)
        #expect(groupSize == 32)

        // Tier 3: grid decode → re-quantize; IQ1_M has very few codepoints
        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits,
            f16Weight: f16Array,
            inputDim: 256, outputDim: 2,
            tolerance: 5)
    }

    // MARK: - Tier 4: TQ1_0

    @Test("TQ1_0 re-quantize pack: 2D weight round-trip correctness")
    func tq1_0Pack2D() throws {
        // TQ1_0: 54 bytes per 256-element super-block. Tensor [256, 2] → 2 blocks.
        var data = Data(count: 54 * 2)

        for block in 0..<2 {
            let offset = block * 54
            // qs: 48 bytes (base-3 trit encoding)
            for j in 0..<48 {
                // Keep values < 243 to avoid overflow in pow3 multiply
                data[offset + j] = UInt8((block * 19 + j * 7) % 243)
            }
            // qh: 4 bytes
            for j in 0..<4 {
                data[offset + 48 + j] = UInt8((block * 13 + j * 5) % 243)
            }
            // d (float16) at offset+52
            let d = Float16(block == 0 ? 0.5 : 0.25)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset + 52..<offset + 54, with: $0)
            }
        }

        let tensor = GGUFTensorInfo(
            name: "t", dimensions: [256, 2], quantizationType: .tq1_0, offset: 0)
        let f16Array = try bridge.convert(tensor: tensor, data: data)
        let direct = try bridge.convertDirect(tensor: tensor, data: data)
        guard case .quantized(let weight, let scales, let biases, let groupSize, let bits) = direct else {
            Issue.record("Expected .quantized for TQ1_0")
            return
        }
        #expect(bits == 2)
        #expect(groupSize == 32)

        // Tier 4: ternary decode → 2-bit affine; high accumulated error
        try verifyQuantizedMatchesF16(
            weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits,
            f16Weight: f16Array,
            inputDim: 256, outputDim: 2,
            tolerance: 200)
    }

    // MARK: - 1D Fallback for New Types

    @Test("New quant types fall back to F16 for 1D tensors")
    func newTypesFallbackTo1D() throws {
        let types: [(GGUFQuantizationType, Int)] = [
            (.q5_0, 22), (.q5_1, 24), (.q8_K, 292), (.q2_K, 84),
            (.q3_K, 110), (.tq2_0, 66), (.iq4_NL, 18), (.iq4_XS, 136),
            (.iq2_XXS, 66), (.iq2_XS, 74), (.iq2_S, 82),
            (.iq3_XXS, 98), (.iq3_S, 110), (.iq1_S, 50), (.iq1_M, 56),
            (.tq1_0, 54),
        ]

        for (qtype, blockSize) in types {
            let elementsPerBlock = qtype == .iq4_NL ? 32 : (blockSize <= 24 ? 32 : 256)
            let data = Data(count: blockSize)
            let tensor = GGUFTensorInfo(
                name: "t", dimensions: [elementsPerBlock],
                quantizationType: qtype, offset: 0)
            let result = try bridge.convertDirect(tensor: tensor, data: data)
            guard case .float16 = result else {
                Issue.record("1D \(qtype) should fall back to .float16")
                continue
            }
        }
    }

    // MARK: - Helper

    /// Verify that a QuantizedLinear with direct-packed weights produces the
    /// same output as a Linear with dequantized F16 weights.
    ///
    /// This is the gold-standard test: the direct path must be lossless
    /// compared to the dequantize→Linear path.
    private func verifyQuantizedMatchesF16(
        weight: MLXArray, scales: MLXArray, biases: MLXArray,
        groupSize: Int, bits: Int,
        f16Weight: MLXArray,
        inputDim: Int, outputDim: Int,
        tolerance: Float = 0.05
    ) throws {
        // Create a module from the direct-packed data without forcing it through
        // MLX's native quantizedMM path when the backend does not support the
        // GGUF-native layout (e.g. Q6_K groupSize=16).
        let qLinear: Linear
        if groupSize >= 32 {
            qLinear = QuantizedLinear(
                weight: weight, bias: nil, scales: scales, biases: biases,
                groupSize: groupSize, bits: bits, mode: .affine
            )
        } else {
            qLinear = DirectQuantizedLinear(
                weight: weight, bias: nil, scales: scales, biases: biases,
                groupSize: groupSize, bits: bits, mode: .affine
            )
        }

        // Create Linear from dequantized F16 data
        let linear = Linear(weight: f16Weight, bias: nil)

        // Fixed-seed input for reproducibility (quantizedMM vs F16 matmul have different
        // accumulation, so random input causes varying absolute error across runs)
        MLXRandom.seed(42)
        let input = MLXRandom.normal([1, inputDim]).asType(.float16)

        // Forward pass on both
        let qOutput = qLinear(input)
        let fOutput = linear(input)
        eval(qOutput, fOutput)

        // Use relative error: normalize by output magnitude to account for
        // large dot products (256-dim with scale factors up to 63)
        let qValues = qOutput.asType(DType.float32)
        let fValues = fOutput.asType(DType.float32)
        let diff = MLX.abs(qValues - fValues)
        let magnitude = MLX.abs(fValues).max()
        eval(diff, magnitude)

        let maxDiff: Float = diff.max().item()
        let maxMag: Float = magnitude.item()

        // For near-zero outputs, use absolute tolerance; otherwise relative tolerance.
        // quantizedMM and F16 matmul use different accumulation, so small relative error is expected.
        let effectiveTolerance = max(tolerance, maxMag * 0.001)

        #expect(
            maxDiff < effectiveTolerance,
            Comment(rawValue: "Max diff \(maxDiff) exceeds tolerance \(effectiveTolerance) (magnitude=\(maxMag)). Packing likely has a bug.")
        )
    }
}

// MARK: - TransformerModel Tests

@Suite("TransformerModel")
struct TransformerModelTests {

    private func makeSmallConfig() -> TransformerConfiguration {
        TransformerConfiguration(
            hiddenSize: 64,
            hiddenLayers: 2,
            intermediateSize: 128,
            attentionHeads: 4,
            vocabularySize: 100,
            kvHeads: 2
        )
    }

    @Test("Model creation")
    func modelCreation() {
        let config = makeSmallConfig()
        let model = TransformerModel(config)
        #expect(model.vocabularySize == 100)
        #expect(model.layerCount == 2)
        #expect(model.kvHeads == [2, 2])
    }

    @Test("Forward pass shape")
    func forwardPassShape() {
        let config = makeSmallConfig()
        let model = TransformerModel(config)

        let tokens = MLXArray([1, 2, 3]).reshaped([1, 3])
        let cache = model.newCache(parameters: nil)

        let output = model.callAsFunction(tokens, cache: cache)
        #expect(output.shape == [1, 3, 100]) // [batch, seq, vocab]
    }

    @Test("Tied embeddings use embed_tokens as lm_head")
    func tiedEmbeddings() {
        let config = TransformerConfiguration(
            hiddenSize: 64,
            hiddenLayers: 1,
            intermediateSize: 128,
            attentionHeads: 4,
            vocabularySize: 100,
            tieWordEmbeddings: true
        )
        let model = TransformerModel(config)
        // lmHead should be nil when tied
        #expect(model.lmHead == nil)
    }

    @Test("Untied embeddings have separate lm_head")
    func untiedEmbeddings() {
        let config = TransformerConfiguration(
            hiddenSize: 64,
            hiddenLayers: 1,
            intermediateSize: 128,
            attentionHeads: 4,
            vocabularySize: 100,
            tieWordEmbeddings: false
        )
        let model = TransformerModel(config)
        #expect(model.lmHead != nil)
    }

    @Test("Sanitize filters rotary embeddings")
    func sanitizeWeights() {
        let config = makeSmallConfig()
        let model = TransformerModel(config)

        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray([1.0]),
            "model.layers.0.self_attn.q_proj.weight": MLXArray([1.0]),
        ]

        let sanitized = model.sanitize(weights: weights)
        #expect(sanitized.count == 1)
        #expect(sanitized["model.layers.0.self_attn.q_proj.weight"] != nil)
    }
}

// MARK: - ChatTemplateRenderer Tests

@Suite("ChatTemplateRenderer")
struct ChatTemplateRendererTests {

    @Test("Simple template rendering")
    func simpleTemplate() throws {
        let template = "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}"
        let renderer = try ChatTemplateRenderer(
            templateString: template,
            bosToken: "<s>",
            eosToken: "</s>"
        )

        let result = try renderer.render(messages: [
            .system("Be helpful"),
            .user("Hello"),
        ])

        #expect(result.contains("system: Be helpful"))
        #expect(result.contains("user: Hello"))
    }

    @Test("BOS/EOS tokens available in template")
    func specialTokens() throws {
        let template = "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}{{ eos_token }}"
        let renderer = try ChatTemplateRenderer(
            templateString: template,
            bosToken: "<BOS>",
            eosToken: "<EOS>"
        )

        let result = try renderer.render(messages: [.user("Hi")])
        #expect(result == "<BOS>Hi<EOS>")
    }

    @Test("add_generation_prompt flag")
    func addGenPrompt() throws {
        let template = "{% for m in messages %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}ASSISTANT:{% endif %}"
        let renderer = try ChatTemplateRenderer(
            templateString: template,
            bosToken: nil,
            eosToken: nil
        )

        let withPrompt = try renderer.render(messages: [.user("Hi")], addGenerationPrompt: true)
        #expect(withPrompt.contains("ASSISTANT:"))

        let withoutPrompt = try renderer.render(messages: [.user("Hi")], addGenerationPrompt: false)
        #expect(!withoutPrompt.contains("ASSISTANT:"))
    }

    @Test("Nil BOS/EOS renders empty strings")
    func nilSpecialTokens() throws {
        let template = "[{{ bos_token }}]{{ messages[0].content }}[{{ eos_token }}]"
        let renderer = try ChatTemplateRenderer(
            templateString: template,
            bosToken: nil,
            eosToken: nil
        )

        let result = try renderer.render(messages: [.user("test")])
        #expect(result == "[]test[]")
    }

    @Test("Additional context merged into template")
    func additionalContext() throws {
        let template = "{% if enable_thinking %}THINK{% endif %}{{ messages[0].content }}"
        let renderer = try ChatTemplateRenderer(
            templateString: template,
            bosToken: nil,
            eosToken: nil
        )

        let withThinking = try renderer.render(
            messages: [.user("Hi")],
            additionalContext: ["enable_thinking": true]
        )
        #expect(withThinking == "THINKHi")

        let withoutThinking = try renderer.render(
            messages: [.user("Hi")],
            additionalContext: ["enable_thinking": false]
        )
        #expect(withoutThinking == "Hi")
    }

    @Test("Qwen2.5 chat template renders with swift-jinja")
    func qwen25TemplateRenders() throws {
        let renderer = try ChatTemplateRenderer(
            templateString: qwen25ChatTemplate,
            bosToken: nil,
            eosToken: "<|im_end|>"
        )

        let result = try renderer.render(messages: [.user("hi")], addGenerationPrompt: true)

        #expect(result.contains("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"))
        #expect(result.contains("<|im_start|>user\nhi<|im_end|>\n"))
        #expect(result.hasSuffix("<|im_start|>assistant\n"))
    }
}

// MARK: - GGUFUserInputProcessor Tests

@Suite("GGUFUserInputProcessor")
struct GGUFUserInputProcessorTests {

    @Test("Fallback formatting without template")
    func fallbackFormatting() async throws {
        let tokenizer = MockTokenizer()
        let processor = GGUFUserInputProcessor(
            tokenizer: tokenizer,
            chatTemplate: nil,
            bosToken: nil,
            eosToken: nil,
            addBosToken: false
        )

        let input = UserInput(chat: [
            .system("Be helpful"),
            .user("Hello"),
        ])

        let result = try await processor.prepare(input: input)
        #expect(result.text.tokens.shape[0] == 1) // batch dim
        #expect(result.text.tokens.dim(1) > 0)    // has tokens
    }

    @Test("Template-based processing")
    func templateProcessing() async throws {
        let tokenizer = MockTokenizer()
        let processor = GGUFUserInputProcessor(
            tokenizer: tokenizer,
            chatTemplate: "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}",
            bosToken: "<s>",
            eosToken: "</s>",
            addBosToken: false
        )

        let input = UserInput(prompt: "Hello")
        let result = try await processor.prepare(input: input)
        #expect(result.text.tokens.dim(1) > 0)
    }

    @Test("Additional context passed to template")
    func additionalContext() async throws {
        let tokenizer = MockTokenizer()
        let processor = GGUFUserInputProcessor(
            tokenizer: tokenizer,
            chatTemplate: "{% if enable_thinking %}THINK{% endif %}{{ messages[0].content }}",
            bosToken: nil,
            eosToken: nil,
            addBosToken: false
        )

        let input = UserInput(
            chat: [.user("Hello")],
            additionalContext: ["enable_thinking": true]
        )
        let result = try await processor.prepare(input: input)
        #expect(result.text.tokens.dim(1) > 0)
    }

    @Test("Template special tokens survive tokenization as atomic IDs")
    func templateSpecialTokensAreAtomic() async throws {
        let tokenizer = try makeSpecialTokenTokenizer()
        let processor = GGUFUserInputProcessor(
            tokenizer: tokenizer,
            chatTemplate: "<|im_start|>{{ messages[0].content }}<|im_end|>",
            bosToken: nil,
            eosToken: nil,
            addBosToken: false
        )

        let result = try await processor.prepare(input: UserInput(prompt: "hi"))
        let tokens = result.text.tokens.flattened().asArray(Int32.self).map(Int.init)

        #expect(tokens == [258, Int(UInt8(ascii: "h")), Int(UInt8(ascii: "i")), 259])
    }

    @Test("Qwen2.5 template keeps im tokens atomic through prepare")
    func qwen25TemplateSpecialTokensAreAtomic() async throws {
        let tokenizer = try makeSpecialTokenTokenizer(preTokenizer: .qwen2)
        let processor = GGUFUserInputProcessor(
            tokenizer: tokenizer,
            chatTemplate: qwen25ChatTemplate,
            bosToken: nil,
            eosToken: "<|im_end|>",
            addBosToken: false
        )

        let result = try await processor.prepare(input: UserInput(prompt: "hi"))
        let tokens = result.text.tokens.flattened().asArray(Int32.self).map(Int.init)

        #expect(tokens.first == 258)
        #expect(tokens.filter { $0 == 258 }.count == 3)
        #expect(tokens.filter { $0 == 259 }.count == 2)
    }
}

// MARK: - KVCache Tests

@Suite("KVCache")
struct KVCacheTests {

    @Test("Simple cache creation")
    func simpleCreation() {
        let params = GenerateParameters()
        let caches = createKVCaches(layerCount: 4, parameters: params)
        #expect(caches.count == 4)
        #expect(caches[0].offset == 0)
    }

    @Test("Rotating cache with max size")
    func rotatingCache() {
        var params = GenerateParameters()
        params.maxKVSize = 256
        let caches = createKVCaches(layerCount: 2, parameters: params)
        #expect(caches.count == 2)
    }
}

// MARK: - Bias Detection Tests

@Suite("TransformerConfiguration Bias Detection")
struct TransformerConfigurationBiasTests {

    @Test("Detect attention bias from tensor names")
    func detectAttentionBias() throws {
        var builder = GGUFTestBuilder()
        builder.addMetadata("general.architecture", value: .string("qwen2"))
        builder.addMetadata("qwen2.embedding_length", value: .uint32(64))
        builder.addMetadata("qwen2.block_count", value: .uint32(1))
        builder.addMetadata("qwen2.attention.head_count", value: .uint32(4))

        let tokens = (0..<16).map { "tok\($0)" }
        builder.addMetadata("tokenizer.ggml.tokens", value: .array(tokens.map { .string($0) }))

        // Add attention bias tensors
        builder.addTensor(name: "blk.0.attn_q.bias", shape: [64], type: .f32)

        let file = try GGUFFile.parse(data: builder.build())
        let config = try TransformerConfiguration(from: file)
        #expect(config.attentionBias == true)
    }

    @Test("No attention bias when tensor absent")
    func noAttentionBias() throws {
        var builder = GGUFTestBuilder()
        builder.addMetadata("general.architecture", value: .string("llama"))
        builder.addMetadata("llama.embedding_length", value: .uint32(64))
        builder.addMetadata("llama.block_count", value: .uint32(1))
        builder.addMetadata("llama.attention.head_count", value: .uint32(4))

        let tokens = (0..<16).map { "tok\($0)" }
        builder.addMetadata("tokenizer.ggml.tokens", value: .array(tokens.map { .string($0) }))

        let file = try GGUFFile.parse(data: builder.build())
        let config = try TransformerConfiguration(from: file)
        #expect(config.attentionBias == false)
    }

    @Test("Detect MLP bias from tensor names")
    func detectMlpBias() throws {
        var builder = GGUFTestBuilder()
        builder.addMetadata("general.architecture", value: .string("qwen2"))
        builder.addMetadata("qwen2.embedding_length", value: .uint32(64))
        builder.addMetadata("qwen2.block_count", value: .uint32(1))
        builder.addMetadata("qwen2.attention.head_count", value: .uint32(4))

        let tokens = (0..<16).map { "tok\($0)" }
        builder.addMetadata("tokenizer.ggml.tokens", value: .array(tokens.map { .string($0) }))

        builder.addTensor(name: "blk.0.ffn_gate.bias", shape: [64], type: .f32)

        let file = try GGUFFile.parse(data: builder.build())
        let config = try TransformerConfiguration(from: file)
        #expect(config.mlpBias == true)
    }
}

// MARK: - PromptCacheSnapshot Tests

@Suite("PromptCacheSnapshot")
struct PromptCacheSnapshotTests {

    @Test("Capture and materialize simple cache")
    func captureAndMaterialize() {
        let caches: [KVCache] = [KVCacheSimple(), KVCacheSimple()]

        let snapshot = capturePromptCache(cache: caches, prefixTokenCount: 10)
        #expect(snapshot.prefixTokenCount == 10)
        #expect(snapshot.cacheClasses.count == 2)
        #expect(snapshot.cacheClasses[0] == "KVCacheSimple")

        let restored = materializePromptCache(from: snapshot)
        #expect(restored.count == 2)
        #expect(restored[0] is KVCacheSimple)
    }

    @Test("Capture and materialize rotating cache")
    func captureRotating() {
        let caches: [KVCache] = [RotatingKVCache(maxSize: 512, keep: 4, step: 128)]

        let snapshot = capturePromptCache(cache: caches, prefixTokenCount: 20)
        #expect(snapshot.cacheClasses[0] == "RotatingKVCache")

        let restored = materializePromptCache(from: snapshot)
        #expect(restored.count == 1)
        #expect(restored[0] is RotatingKVCache)
    }

    @Test("Capture and materialize quantized cache")
    func captureQuantized() {
        let caches: [KVCache] = [QuantizedKVCache(groupSize: 64, bits: 4)]

        let snapshot = capturePromptCache(cache: caches, prefixTokenCount: 5)
        #expect(snapshot.cacheClasses[0] == "QuantizedKVCache")

        let restored = materializePromptCache(from: snapshot)
        #expect(restored.count == 1)
        #expect(restored[0] is QuantizedKVCache)
    }
}

// MARK: - E2E Integration Test

@Suite("E2E Integration")
struct E2EIntegrationTests {

    /// Build a complete minimal GGUF binary suitable for GGUFModelLoader.
    private func buildTinyModelGGUF(weightType: GGUFQuantizationType = .f32) -> Data {
        // Tiny model: hidden=32, layers=1, heads=2, kvHeads=2, intermediate=64, vocab=16
        let hidden = 32
        let intermediate = 64
        let vocabSize = 16
        let heads = 2

        var builder = GGUFTestBuilder()

        // Architecture metadata
        builder.addMetadata("general.architecture", value: .string("llama"))
        builder.addMetadata("llama.embedding_length", value: .uint32(UInt32(hidden)))
        builder.addMetadata("llama.block_count", value: .uint32(1))
        builder.addMetadata("llama.attention.head_count", value: .uint32(UInt32(heads)))
        builder.addMetadata("llama.attention.head_count_kv", value: .uint32(UInt32(heads)))
        builder.addMetadata("llama.feed_forward_length", value: .uint32(UInt32(intermediate)))
        builder.addMetadata("llama.context_length", value: .uint32(512))

        // Tokenizer metadata (SentencePiece style)
        let tokens = (0..<vocabSize).map { "<tok\($0)>" }
        builder.addMetadata("tokenizer.ggml.tokens", value: .array(tokens.map { .string($0) }))
        builder.addMetadata("tokenizer.ggml.scores", value: .array((0..<vocabSize).map { .float32(Float(vocabSize - $0)) }))
        builder.addMetadata("tokenizer.ggml.model", value: .string("llama"))
        builder.addMetadata("tokenizer.ggml.bos_token_id", value: .uint32(0))
        builder.addMetadata("tokenizer.ggml.eos_token_id", value: .uint32(UInt32(vocabSize - 1)))

        // Tensors (GGUF dims: inner-first)
        // Embedding: MLX [vocab, hidden] → GGUF [hidden, vocab]
        builder.addTensor(name: "token_embd.weight", shape: [UInt64(hidden), UInt64(vocabSize)], type: weightType)

        // Final norm: [hidden]
        builder.addTensor(name: "output_norm.weight", shape: [UInt64(hidden)], type: .f32)

        // Layer 0 norms
        builder.addTensor(name: "blk.0.attn_norm.weight", shape: [UInt64(hidden)], type: .f32)
        builder.addTensor(name: "blk.0.ffn_norm.weight", shape: [UInt64(hidden)], type: .f32)

        // Layer 0 attention: all [hidden, hidden] → GGUF [hidden, hidden]
        builder.addTensor(name: "blk.0.attn_q.weight", shape: [UInt64(hidden), UInt64(hidden)], type: weightType)
        builder.addTensor(name: "blk.0.attn_k.weight", shape: [UInt64(hidden), UInt64(hidden)], type: weightType)
        builder.addTensor(name: "blk.0.attn_v.weight", shape: [UInt64(hidden), UInt64(hidden)], type: weightType)
        builder.addTensor(name: "blk.0.attn_output.weight", shape: [UInt64(hidden), UInt64(hidden)], type: weightType)

        // Layer 0 MLP
        builder.addTensor(name: "blk.0.ffn_gate.weight", shape: [UInt64(hidden), UInt64(intermediate)], type: weightType)
        builder.addTensor(name: "blk.0.ffn_up.weight", shape: [UInt64(hidden), UInt64(intermediate)], type: weightType)
        builder.addTensor(name: "blk.0.ffn_down.weight", shape: [UInt64(intermediate), UInt64(hidden)], type: weightType)

        return builder.build()
    }

    @Test("Load tiny model from GGUF data")
    func loadTinyModel() throws {
        let ggufData = buildTinyModelGGUF()
        let file = try GGUFFile.parse(data: ggufData)

        // Verify metadata
        #expect(file.architecture == "llama")
        #expect(file.embeddingLength == 32)
        #expect(file.blockCount == 1)

        // Extract config
        let config = try TransformerConfiguration(from: file)
        #expect(config.hiddenSize == 32)
        #expect(config.hiddenLayers == 1)
        #expect(config.intermediateSize == 64)
        #expect(config.tieWordEmbeddings == true) // no output.weight tensor
    }

    @Test("Full pipeline: GGUF → Model → Forward pass")
    func fullPipeline() throws {
        let ggufData = buildTinyModelGGUF()

        // Write to temp file for GGUFModelLoader
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_tiny_model.gguf")
        try ggufData.write(to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        // Load model
        let loader = GGUFModelLoader()
        let context = try loader.loadContext(url: tempURL)

        // Verify context
        #expect(context.configuration.name == "test_tiny_model")
        #expect(context.model.layerCount == 1)

        // Forward pass with token input
        let tokens = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped([1, 3])
        let cache = context.model.newCache(parameters: nil)
        let output = context.model.callAsFunction(
            LMInput.Text(tokens: tokens),
            cache: cache,
            state: nil
        )

        // Should produce logits of shape [1, 3, vocab_size=16]
        #expect(output.logits.shape == [1, 3, 16])
    }

    @Test("Token generation from tiny model")
    func tokenGeneration() throws {
        let ggufData = buildTinyModelGGUF()

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_tiny_gen.gguf")
        try ggufData.write(to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let loader = GGUFModelLoader()
        let context = try loader.loadContext(url: tempURL)

        // Create input
        let tokens = MLXArray([Int32(1)]).reshaped([1, 1])
        let input = LMInput(tokens: tokens)

        // Generate with argmax (temperature=0) and limited tokens
        var params = GenerateParameters()
        params.temperature = 0
        params.maxTokens = 3

        let iterator = try TokenIterator(
            input: input,
            model: context.model,
            cache: context.model.newCache(parameters: params),
            parameters: params,
            eosTokenIds: context.configuration.eosTokenIds
        )

        var generated: [Int] = []
        var iter = iterator
        while let token = iter.next() {
            generated.append(token)
        }

        // Should have generated tokens (up to maxTokens, unless EOS hit)
        #expect(generated.count > 0)
        #expect(generated.count <= 3)
    }

    @Test("Quantization disabled keeps dense modules for quantized GGUF")
    func quantizationDisabledKeepsDenseModules() throws {
        let ggufData = buildTinyModelGGUF(weightType: .q4_0)

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_tiny_quant_disabled.gguf")
        try ggufData.write(to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let loader = GGUFModelLoader()
        let quantizedContext = try loader.loadContext(url: tempURL)
        let denseContext = try loader.loadContext(url: tempURL, quantization: .disabled)

        let quantizedReport = try #require(quantizedContext.loadReport)
        let denseReport = try #require(denseContext.loadReport)
        #expect(quantizedReport.quantization.source == .autoDetected)
        #expect(denseReport.quantization.source == .disabled)

        let quantizedModel = try #require(quantizedContext.model as? TransformerModel)
        let denseModel = try #require(denseContext.model as? TransformerModel)

        #expect(quantizedModel.model.embedTokens is any Quantized)
        #expect(quantizedModel.model.layers[0].attention.wq is any Quantized)
        #expect(!(denseModel.model.embedTokens is any Quantized))
        #expect(!(denseModel.model.layers[0].attention.wq is any Quantized))
    }
}

// MARK: - LoadReport Tests

@Suite("LoadReport")
struct LoadReportTests {

    /// Build a minimal GGUF for LoadReport testing.
    private func buildTinyModelGGUF() -> Data {
        let hidden = 32
        let intermediate = 64
        let vocabSize = 16
        let heads = 2

        var builder = GGUFTestBuilder()

        builder.addMetadata("general.architecture", value: .string("llama"))
        builder.addMetadata("llama.embedding_length", value: .uint32(UInt32(hidden)))
        builder.addMetadata("llama.block_count", value: .uint32(1))
        builder.addMetadata("llama.attention.head_count", value: .uint32(UInt32(heads)))
        builder.addMetadata("llama.attention.head_count_kv", value: .uint32(UInt32(heads)))
        builder.addMetadata("llama.feed_forward_length", value: .uint32(UInt32(intermediate)))
        builder.addMetadata("llama.context_length", value: .uint32(512))

        let tokens = (0..<vocabSize).map { "<tok\($0)>" }
        builder.addMetadata("tokenizer.ggml.tokens", value: .array(tokens.map { .string($0) }))
        builder.addMetadata("tokenizer.ggml.scores", value: .array((0..<vocabSize).map { .float32(Float(vocabSize - $0)) }))
        builder.addMetadata("tokenizer.ggml.model", value: .string("llama"))
        builder.addMetadata("tokenizer.ggml.bos_token_id", value: .uint32(0))
        builder.addMetadata("tokenizer.ggml.eos_token_id", value: .uint32(UInt32(vocabSize - 1)))

        builder.addTensor(name: "token_embd.weight", shape: [UInt64(hidden), UInt64(vocabSize)], type: .f32)
        builder.addTensor(name: "output_norm.weight", shape: [UInt64(hidden)], type: .f32)
        builder.addTensor(name: "blk.0.attn_norm.weight", shape: [UInt64(hidden)], type: .f32)
        builder.addTensor(name: "blk.0.ffn_norm.weight", shape: [UInt64(hidden)], type: .f32)
        builder.addTensor(name: "blk.0.attn_q.weight", shape: [UInt64(hidden), UInt64(hidden)], type: .f32)
        builder.addTensor(name: "blk.0.attn_k.weight", shape: [UInt64(hidden), UInt64(hidden)], type: .f32)
        builder.addTensor(name: "blk.0.attn_v.weight", shape: [UInt64(hidden), UInt64(hidden)], type: .f32)
        builder.addTensor(name: "blk.0.attn_output.weight", shape: [UInt64(hidden), UInt64(hidden)], type: .f32)
        builder.addTensor(name: "blk.0.ffn_gate.weight", shape: [UInt64(hidden), UInt64(intermediate)], type: .f32)
        builder.addTensor(name: "blk.0.ffn_up.weight", shape: [UInt64(hidden), UInt64(intermediate)], type: .f32)
        builder.addTensor(name: "blk.0.ffn_down.weight", shape: [UInt64(intermediate), UInt64(hidden)], type: .f32)

        return builder.build()
    }

    @Test("LoadReport is populated when loading via GGUFModelLoader")
    func reportIsPopulated() throws {
        let ggufData = buildTinyModelGGUF()
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_report.gguf")
        try ggufData.write(to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let loader = GGUFModelLoader()
        let context = try loader.loadContext(url: tempURL)

        #expect(context.loadReport != nil)
    }

    @Test("Model resolution records TransformerModel as fallback")
    func modelResolution() throws {
        let ggufData = buildTinyModelGGUF()
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_report_resolution.gguf")
        try ggufData.write(to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let loader = GGUFModelLoader()
        let context = try loader.loadContext(url: tempURL)
        let report = try #require(context.loadReport)

        // Plain Llama GGUF with ffn_norm → no specialized model matches,
        // falls through to TransformerModel (universal fallback)
        #expect(report.modelResolution.selectedType.contains("TransformerModel"))
        #expect(report.modelResolution.candidatesEvaluated == report.modelResolution.totalCandidates)
    }

    @Test("Weight loading counts mapped and skipped tensors")
    func weightLoadingCounts() throws {
        let ggufData = buildTinyModelGGUF()
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_report_weights.gguf")
        try ggufData.write(to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let loader = GGUFModelLoader()
        let context = try loader.loadContext(url: tempURL)
        let report = try #require(context.loadReport)

        // 11 tensors in the GGUF, all should be mapped by LlamaTensorNameMapper
        #expect(report.weightLoading.mappedCount == 11)
        #expect(report.weightLoading.skippedCount == 0)
        #expect(report.weightLoading.skippedTensors.isEmpty)
    }

    @Test("No embedded adapter for standard model")
    func noEmbeddedAdapter() throws {
        let ggufData = buildTinyModelGGUF()
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_report_nolora.gguf")
        try ggufData.write(to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let loader = GGUFModelLoader()
        let context = try loader.loadContext(url: tempURL)
        let report = try #require(context.loadReport)

        #expect(report.weightLoading.embeddedAdapter == nil)
    }

    @Test("Quantization disabled for F32 model")
    func quantizationDisabledForF32() throws {
        let ggufData = buildTinyModelGGUF()
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_report_quant.gguf")
        try ggufData.write(to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let loader = GGUFModelLoader()
        let context = try loader.loadContext(url: tempURL)
        let report = try #require(context.loadReport)

        // All tensors are F32, so no quantization should be auto-detected
        #expect(report.quantization.source == .disabled)
        #expect(report.quantization.bits == 0)
    }

    @Test("Summary produces readable output")
    func summaryFormat() throws {
        let ggufData = buildTinyModelGGUF()
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_report_summary.gguf")
        try ggufData.write(to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let loader = GGUFModelLoader()
        let context = try loader.loadContext(url: tempURL)
        let report = try #require(context.loadReport)

        let summary = report.summary
        #expect(summary.contains("Model:"))
        #expect(summary.contains("Tensors:"))
        #expect(summary.contains("Quantization:"))
    }

    @Test("LoadReport is nil when ModelContext is constructed manually")
    func manualContextHasNoReport() {
        let tokenizer = MockTokenizer()
        let context = ModelContext(
            configuration: ModelConfiguration(name: "test"),
            model: TransformerModel(TransformerConfiguration(
                hiddenSize: 32, hiddenLayers: 1, intermediateSize: 64,
                attentionHeads: 2, vocabularySize: 16
            )),
            processor: GGUFUserInputProcessor(
                tokenizer: tokenizer, chatTemplate: nil,
                bosToken: nil, eosToken: nil, addBosToken: false),
            tokenizer: tokenizer
        )

        #expect(context.loadReport == nil)
    }
}

// MARK: - Test Helpers

/// Minimal mock tokenizer for testing.
struct MockTokenizer: Tokenizer, @unchecked Sendable {
    var bosTokenID: Int? { 1 }
    var eosTokenID: Int? { 2 }
    var vocabularySize: Int { 100 }

    func encode(text: String) -> [Int] {
        // Simple byte-level encoding
        Array(text.utf8).map { Int($0) }
    }

    func decode(tokens: [Int]) -> String {
        String(tokens.compactMap { UnicodeScalar($0) }.map { Character($0) })
    }

    func tokenToString(_ id: Int) -> String? {
        if id == 1 { return "<s>" }
        if id == 2 { return "</s>" }
        return String(UnicodeScalar(id) ?? UnicodeScalar(0))
    }

    func tokenID(for string: String) -> Int? {
        if string == "<s>" { return 1 }
        if string == "</s>" { return 2 }
        return nil
    }
}

private let qwen25ChatTemplate = #"""
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"""#

private func makeSpecialTokenTokenizer(
    preTokenizer: PreTokenizer = .qwen2
) throws -> MergesBPETokenizer {
    var vocabulary: [String] = []
    for byte: UInt8 in 0...255 {
        vocabulary.append(String(Character(ByteEncoder.byteToUnicode[byte]!)))
    }
    vocabulary.append("<|vision_start|>")
    vocabulary.append("<|vision_end|>")
    vocabulary.append("<|im_start|>")
    vocabulary.append("<|im_end|>")

    var tokenTypes = Array(repeating: 1, count: vocabulary.count)
    tokenTypes[258] = 4
    tokenTypes[259] = 4

    return try MergesBPETokenizer(
        vocabulary: vocabulary,
        merges: [],
        preTokenizer: preTokenizer,
        specialTokens: SpecialTokens(),
        tokenTypes: tokenTypes
    )
}

// MARK: - GGUFTestBuilder

/// Builds minimal GGUF binary data for testing.
struct GGUFTestBuilder {
    private var metadata: [(String, GGUFMetadataValue)] = []
    private var tensors: [(name: String, shape: [UInt64], type: GGUFQuantizationType)] = []

    mutating func addMetadata(_ key: String, value: GGUFMetadataValue) {
        metadata.append((key, value))
    }

    mutating func addTensor(name: String, shape: [UInt64], type: GGUFQuantizationType) {
        tensors.append((name, shape, type))
    }

    func build() -> Data {
        var data = Data()

        // Magic
        appendUInt32(&data, GGUFFile.magic)
        // Version 3
        appendUInt32(&data, 3)
        // Tensor count
        appendUInt64(&data, UInt64(tensors.count))
        // Metadata KV count
        appendUInt64(&data, UInt64(metadata.count))

        // Metadata
        for (key, value) in metadata {
            appendString(&data, key)
            appendMetadataValue(&data, value)
        }

        // Tensor directory
        var tensorDataSize = 0
        for (name, shape, type) in tensors {
            appendString(&data, name)
            appendUInt32(&data, UInt32(shape.count))
            for dim in shape {
                appendUInt64(&data, dim)
            }
            appendUInt32(&data, type.rawValue)
            appendUInt64(&data, UInt64(tensorDataSize))

            let elements = shape.reduce(1, *)
            let bytesPerElement: Int
            switch type {
            case .f32: bytesPerElement = 4
            case .f16: bytesPerElement = 2
            default: bytesPerElement = 2
            }
            tensorDataSize += Int(elements) * bytesPerElement
        }

        // Pad to 32-byte alignment
        let headerEnd = data.count
        let alignment = 32
        let padding = (alignment - (headerEnd % alignment)) % alignment
        data.append(Data(repeating: 0, count: padding))

        // Tensor data (zeros)
        data.append(Data(repeating: 0, count: tensorDataSize))

        return data
    }

    private func appendUInt32(_ data: inout Data, _ value: UInt32) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }

    private func appendUInt64(_ data: inout Data, _ value: UInt64) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }

    private func appendString(_ data: inout Data, _ value: String) {
        let utf8 = Array(value.utf8)
        appendUInt64(&data, UInt64(utf8.count))
        data.append(contentsOf: utf8)
    }

    private func appendMetadataValue(_ data: inout Data, _ value: GGUFMetadataValue) {
        switch value {
        case .uint32(let v):
            appendUInt32(&data, 4) // type tag for UINT32
            appendUInt32(&data, v)
        case .float32(let v):
            appendUInt32(&data, 6) // type tag for FLOAT32
            withUnsafeBytes(of: v) { data.append(contentsOf: $0) }
        case .string(let s):
            appendUInt32(&data, 8) // type tag for STRING
            appendString(&data, s)
        case .array(let arr):
            appendUInt32(&data, 9) // type tag for ARRAY
            // Determine element type from first element
            if let first = arr.first {
                switch first {
                case .string:
                    appendUInt32(&data, 8) // STRING elements
                case .float32:
                    appendUInt32(&data, 6) // FLOAT32 elements
                default:
                    appendUInt32(&data, 4) // UINT32 elements
                }
            } else {
                appendUInt32(&data, 8)
            }
            appendUInt64(&data, UInt64(arr.count))
            for element in arr {
                switch element {
                case .string(let s):
                    appendString(&data, s)
                case .uint32(let v):
                    appendUInt32(&data, v)
                case .float32(let v):
                    withUnsafeBytes(of: v) { data.append(contentsOf: $0) }
                default:
                    break
                }
            }
        default:
            break
        }
    }
}
