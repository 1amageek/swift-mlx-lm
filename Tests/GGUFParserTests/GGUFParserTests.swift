import Testing
@testable import GGUFParser
import Foundation

@Suite("GGUF Metadata Value")
struct GGUFMetadataValueTests {

    @Test("Integer coercion")
    func integerCoercion() {
        let uint32 = GGUFMetadataValue.uint32(42)
        #expect(uint32.intValue == 42)
        #expect(uint32.uint32Value == 42)
        #expect(uint32.stringValue == nil)

        let int8 = GGUFMetadataValue.int8(-1)
        #expect(int8.intValue == -1)
    }

    @Test("String value")
    func stringValue() {
        let value = GGUFMetadataValue.string("llama")
        #expect(value.stringValue == "llama")
        #expect(value.intValue == nil)
    }

    @Test("Array value")
    func arrayValue() {
        let arr = GGUFMetadataValue.array([.string("hello"), .string("world")])
        #expect(arr.arrayValue?.count == 2)
        #expect(arr.arrayValue?[0].stringValue == "hello")
    }

    @Test("Float coercion")
    func floatCoercion() {
        let f32 = GGUFMetadataValue.float32(3.14)
        #expect(f32.float32Value != nil)
        #expect(f32.doubleValue != nil)
    }

    @Test("Bool value")
    func boolValue() {
        let value = GGUFMetadataValue.bool(true)
        #expect(value.boolValue == true)
    }
}

@Suite("GGUF Quantization Type")
struct GGUFQuantizationTypeTests {

    @Test("Q4_0 block properties")
    func q4_0Properties() {
        let q = GGUFQuantizationType.q4_0
        #expect(q.blockSize == 18)
        #expect(q.elementsPerBlock == 32)
        #expect(q.isUnquantized == false)
    }

    @Test("F16 is unquantized")
    func f16Properties() {
        let q = GGUFQuantizationType.f16
        #expect(q.blockSize == 2)
        #expect(q.elementsPerBlock == 1)
        #expect(q.isUnquantized == true)
    }

    @Test("Q4_K block properties")
    func q4KProperties() {
        let q = GGUFQuantizationType.q4_K
        #expect(q.elementsPerBlock == 256)
        #expect(q.isUnquantized == false)
    }
}

@Suite("GGUF Tensor Info")
struct GGUFTensorInfoTests {

    @Test("Element count and data size")
    func tensorInfoCalculations() {
        let info = GGUFTensorInfo(
            name: "test.weight",
            dimensions: [4096, 4096],
            quantizationType: .f16,
            offset: 0
        )
        #expect(info.elementCount == 4096 * 4096)
        #expect(info.dataSize == 4096 * 4096 * 2)
    }

    @Test("Quantized data size")
    func quantizedDataSize() {
        let info = GGUFTensorInfo(
            name: "test.weight",
            dimensions: [4096, 4096],
            quantizationType: .q4_0,
            offset: 0
        )
        let totalElements = 4096 * 4096
        let blocks = totalElements / 32
        #expect(info.dataSize == blocks * 18)
    }
}

@Suite("GGUF File Parsing")
struct GGUFFileParsingTests {

    @Test("Parse minimal GGUF")
    func parseMinimalGGUF() throws {
        let data = buildMinimalGGUF()
        let file = try GGUFFile.parse(data: data)
        #expect(file.version == 3)
        #expect(file.metadata["general.architecture"]?.stringValue == "llama")
        #expect(file.tensors.isEmpty)
    }

    @Test("Invalid magic throws")
    func invalidMagicThrows() {
        var data = Data(count: 32)
        data[0] = 0x00
        data[1] = 0x00
        data[2] = 0x00
        data[3] = 0x00
        #expect(throws: GGUFError.self) {
            _ = try GGUFFile.parse(data: data)
        }
    }

    @Test("Architecture accessor")
    func architectureAccessor() throws {
        let data = buildMinimalGGUF()
        let file = try GGUFFile.parse(data: data)
        #expect(file.architecture == "llama")
    }
}

// MARK: - Test Helpers

/// Build a minimal valid GGUF v3 file with one metadata entry.
private func buildMinimalGGUF() -> Data {
    var data = Data()

    // Magic: "GGUF" (little-endian: 0x47 0x47 0x55 0x46)
    appendUInt32(&data, 0x4655_4747)
    // Version: 3
    appendUInt32(&data, 3)
    // Tensor count: 0
    appendUInt64(&data, 0)
    // Metadata KV count: 1
    appendUInt64(&data, 1)

    // Metadata: general.architecture = "llama"
    appendString(&data, "general.architecture")
    appendUInt32(&data, GGUFMetadataValueType.string.rawValue)
    appendString(&data, "llama")

    return data
}

private func appendUInt32(_ data: inout Data, _ value: UInt32) {
    var v = value.littleEndian
    data.append(Data(bytes: &v, count: 4))
}

private func appendUInt64(_ data: inout Data, _ value: UInt64) {
    var v = value.littleEndian
    data.append(Data(bytes: &v, count: 8))
}

private func appendString(_ data: inout Data, _ string: String) {
    let utf8 = string.utf8
    appendUInt64(&data, UInt64(utf8.count))
    data.append(contentsOf: utf8)
}
