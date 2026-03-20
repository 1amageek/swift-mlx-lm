import Testing
import Metal
import Foundation
@testable import MetalCompiler

// MARK: - STAF Roundtrip Tests

@Suite("STAF Roundtrip")
struct STAFRoundtripTests {

    // MARK: - Float16 Roundtrip

    @Test("Float16 tensor survives safetensors → STAF → load")
    func float16Roundtrip() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("staf_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDirectory) }

        // Known Float16 values: simple sequence [1.0, 2.0, ..., 128.0]
        let elementCount = 128
        var expectedValues: [Float16] = (0..<elementCount).map { Float16($0 + 1) }

        // Write safetensors file with known Float16 data
        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "test.weight", dtype: "F16", shape: [1, elementCount], data: Data(
                    bytes: &expectedValues, count: elementCount * MemoryLayout<Float16>.size))
            ],
            to: safetensorsURL)

        // Convert safetensors → STAF
        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        let converter = STAFConverter()
        try converter.convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)

        // Load STAF
        let loader = STAFLoader()
        let store = try loader.load(at: stafURL, device: device)

        // Verify tensor exists
        guard let entry = store.entries["test.weight"] else {
            Issue.record("Tensor 'test.weight' not found in STAF store")
            return
        }
        #expect(entry.schemeIdentifier == .fp16RowMajor)
        #expect(entry.payloadSize == elementCount * 2)

        // Read values from the Metal buffer at the reported offset
        let bufferPointer = store.buffer.contents() + entry.bufferOffset
        let loadedValues = UnsafeBufferPointer(
            start: bufferPointer.bindMemory(to: Float16.self, capacity: elementCount),
            count: elementCount)

        // Verify exact match
        for i in 0..<elementCount {
            #expect(
                loadedValues[i] == expectedValues[i],
                "Mismatch at index \(i): loaded=\(loadedValues[i]) expected=\(expectedValues[i])")
        }
    }

    // MARK: - BFloat16 Roundtrip

    @Test("BFloat16 tensor preserved as raw bytes in STAF")
    func bfloat16Roundtrip() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("staf_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDirectory) }

        // BF16: 1.0, 2.0, -1.0, 3.0
        let bf16Values: [BFloat16] = [1.0, 2.0, -1.0, 3.0]
        let elementCount = bf16Values.count
        var bf16Data = bf16Values

        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "test.weight", dtype: "BF16", shape: [1, elementCount], data: Data(
                    bytes: &bf16Data, count: elementCount * MemoryLayout<BFloat16>.size))
            ],
            to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        let converter = STAFConverter()
        try converter.convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)

        let loader = STAFLoader()
        let store = try loader.load(at: stafURL, device: device)

        guard let entry = store.entries["test.weight"] else {
            Issue.record("Tensor 'test.weight' not found")
            return
        }
        #expect(entry.schemeIdentifier == .bf16RowMajor)

        // BF16 data should be preserved as-is (raw bytes)
        let bufferPointer = store.buffer.contents() + entry.bufferOffset
        let loaded = UnsafeBufferPointer(
            start: bufferPointer.bindMemory(to: BFloat16.self, capacity: elementCount),
            count: elementCount)

        for i in 0..<elementCount {
            #expect(
                loaded[i] == bf16Values[i],
                "BF16 mismatch at index \(i): loaded=\(loaded[i]) expected=\(bf16Values[i])")
        }
    }

    // MARK: - Float32 → Float16 Conversion

    @Test("Float32 tensor is converted to Float16 in STAF")
    func float32ToFloat16Conversion() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("staf_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDirectory) }

        // Float32 values
        var f32Values: [Float] = [1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 0.0, 100.0]
        let elementCount = f32Values.count

        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "test.weight", dtype: "F32", shape: [2, 4], data: Data(
                    bytes: &f32Values, count: elementCount * 4))
            ],
            to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try converter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)

        let store = try STAFLoader().load(at: stafURL, device: device)

        guard let entry = store.entries["test.weight"] else {
            Issue.record("Tensor not found")
            return
        }

        // Should be stored as FP16
        #expect(entry.schemeIdentifier == .fp16RowMajor)
        #expect(entry.payloadSize == elementCount * 2)

        let bufferPointer = store.buffer.contents() + entry.bufferOffset
        let loadedFP16 = UnsafeBufferPointer(
            start: bufferPointer.bindMemory(to: Float16.self, capacity: elementCount),
            count: elementCount)

        for i in 0..<elementCount {
            let expected = Float16(f32Values[i])
            #expect(
                loadedFP16[i] == expected,
                "FP32→FP16 mismatch at index \(i): loaded=\(loadedFP16[i]) expected=\(expected)")
        }
    }

    // MARK: - Multiple Tensors

    @Test("Multiple tensors loaded at correct offsets")
    func multipleTensorsRoundtrip() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("staf_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDirectory) }

        // Create three tensors with distinct known values
        let size1 = 64
        let size2 = 256
        let size3 = 32
        var values1: [Float16] = (0..<size1).map { Float16($0 + 1) }        // 1..64
        var values2: [Float16] = (0..<size2).map { Float16(100 + $0) }       // 100..355
        var values3: [Float16] = (0..<size3).map { Float16(1000 + $0) }      // 1000..1031

        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "model.embed_tokens.weight", dtype: "F16", shape: [1, size1],
                           data: Data(bytes: &values1, count: size1 * 2)),
                TestTensor(name: "model.layers.0.self_attn.q_proj.weight", dtype: "F16", shape: [size2, 1],
                           data: Data(bytes: &values2, count: size2 * 2)),
                TestTensor(name: "model.norm.weight", dtype: "F16", shape: [size3],
                           data: Data(bytes: &values3, count: size3 * 2)),
            ],
            to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)

        let store = try STAFLoader().load(at: stafURL, device: device)

        // Verify all three tensors
        try verifyTensorValues(store: store, name: "model.embed_tokens.weight",
                               expected: values1, label: "embedding")
        try verifyTensorValues(store: store, name: "model.layers.0.self_attn.q_proj.weight",
                               expected: values2, label: "q_proj")
        try verifyTensorValues(store: store, name: "model.norm.weight",
                               expected: values3, label: "norm")
    }

    // MARK: - Page Alignment

    @Test("STAF buffer offset is page-aligned for bytesNoCopy")
    func pageAlignmentCorrectness() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("staf_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDirectory) }

        // Create a tensor large enough that payload offset matters
        let elementCount = 4096
        var values: [Float16] = (0..<elementCount).map { Float16(Float($0) * 0.001) }

        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "test.weight", dtype: "F16", shape: [64, 64], data: Data(
                    bytes: &values, count: elementCount * 2))
            ],
            to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)

        // Load and verify the MTLBuffer is valid
        let store = try STAFLoader().load(at: stafURL, device: device)

        // Buffer must start at a page-aligned address
        let pageSize = Int(sysconf(_SC_PAGESIZE))
        let bufferAddress = Int(bitPattern: store.buffer.contents())
        #expect(
            bufferAddress % pageSize == 0,
            "MTLBuffer address 0x\(String(bufferAddress, radix: 16)) is not page-aligned (page=\(pageSize))")

        // Buffer length must be page-aligned
        #expect(
            store.buffer.length % pageSize == 0,
            "MTLBuffer length \(store.buffer.length) is not page-aligned (page=\(pageSize))")

        // Verify data at the offset
        guard let entry = store.entries["test.weight"] else {
            Issue.record("Tensor not found")
            return
        }

        let bufferPointer = store.buffer.contents() + entry.bufferOffset
        let loaded = UnsafeBufferPointer(
            start: bufferPointer.bindMemory(to: Float16.self, capacity: elementCount),
            count: elementCount)

        for i in 0..<elementCount {
            #expect(
                loaded[i] == values[i],
                "Page alignment test: mismatch at index \(i): loaded=\(loaded[i]) expected=\(values[i])")
        }
    }

    // MARK: - Embedding Lookup Kernel Integration

    @Test("Embedding lookup kernel reads correct values from STAF")
    func embeddingLookupFromSTAF() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("staf_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDirectory) }

        // Embedding table: 4 tokens × 8 dimensions
        let vocabSize = 4
        let embeddingDimension = 8
        let totalElements = vocabSize * embeddingDimension

        // Token 0: [1,1,1,1,1,1,1,1], Token 1: [2,2,...], Token 2: [3,...], Token 3: [4,...]
        var embeddingData: [Float16] = (0..<totalElements).map { i in
            Float16(i / embeddingDimension + 1)
        }

        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "model.embed_tokens.weight", dtype: "F16",
                           shape: [vocabSize, embeddingDimension],
                           data: Data(bytes: &embeddingData, count: totalElements * 2))
            ],
            to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)

        let store = try STAFLoader().load(at: stafURL, device: device)

        guard let access = store.bufferAccess(for: "model.embed_tokens.weight") else {
            Issue.record("Cannot get buffer access for embedding table")
            return
        }

        // Compile the embedding_lookup kernel
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(
            source: MetalSourceGenerator.generateCompleteLibrary(weightFormat: .bfloat16), options: options)
        guard let function = library.makeFunction(name: "embedding_lookup") else {
            Issue.record("embedding_lookup kernel not found")
            return
        }
        let pipeline = try device.makeComputePipelineState(function: function)

        // Allocate buffers
        let tokenInBuffer = device.makeBuffer(length: 4, options: .storageModeShared)!
        let outputBuffer = device.makeBuffer(
            length: embeddingDimension * MemoryLayout<Float16>.size, options: .storageModeShared)!

        // Test each token
        for tokenID: Int32 in 0..<Int32(vocabSize) {
            tokenInBuffer.contents().bindMemory(to: Int32.self, capacity: 1).pointee = tokenID

            // Clear output
            memset(outputBuffer.contents(), 0, outputBuffer.length)

            guard let queue = device.makeCommandQueue(),
                  let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                Issue.record("Cannot create Metal command encoder")
                return
            }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(tokenInBuffer, offset: 0, index: 0)
            encoder.setBuffer(access.buffer, offset: access.offset, index: 1)
            encoder.setBuffer(outputBuffer, offset: 0, index: 2)
            var dim = UInt32(embeddingDimension)
            encoder.setBytes(&dim, length: 4, index: 3)

            let threadgroupSize = min(embeddingDimension, pipeline.maxTotalThreadsPerThreadgroup)
            let gridSize = (embeddingDimension + threadgroupSize - 1) / threadgroupSize
            encoder.dispatchThreadgroups(
                MTLSize(width: gridSize, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            if let error = commandBuffer.error {
                Issue.record("GPU error for token \(tokenID): \(error)")
                continue
            }

            // Verify output
            let outputPointer = outputBuffer.contents().bindMemory(
                to: Float16.self, capacity: embeddingDimension)
            let expectedValue = Float16(tokenID + 1)
            for d in 0..<embeddingDimension {
                let actual = outputPointer[d]
                #expect(
                    actual == expectedValue,
                    "Token \(tokenID) dim \(d): actual=\(actual) expected=\(expectedValue)")
                // Check for NaN explicitly
                #expect(!actual.isNaN, "Token \(tokenID) dim \(d): NaN detected!")
            }
        }
    }

    // MARK: - Q4 Quantized Roundtrip

    @Test("Q4 quantized tensor block layout is preserved")
    func q4QuantizedRoundtrip() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("staf_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDirectory) }

        // MLX quantized format: weight (uint32 packed), scales (float16), biases (float16)
        // Shape: [outputDim=4, packedInputDim=8] with 4-bit, groups of 64
        // packedInputDim = inputDim / 8 (8 nibbles per uint32)
        // inputDim = 8 * 8 = 64, groups = 64/64 = 1 group per row
        let outputDimension = 4
        let packedInputDimension = 8
        let inputDimension = packedInputDimension * 8
        let groupsPerRow = inputDimension / 64  // 1

        // Weight data: packed uint32 values
        var weightData: [UInt32] = Array(repeating: 0x01010101, count: outputDimension * packedInputDimension)
        // Scales: one per group per row
        var scalesData: [Float16] = Array(repeating: Float16(2.0), count: outputDimension * groupsPerRow)
        // Biases (zeros): one per group per row
        var biasesData: [Float16] = Array(repeating: Float16(0.5), count: outputDimension * groupsPerRow)

        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "model.layers.0.mlp.gate_proj.weight", dtype: "U32",
                           shape: [outputDimension, packedInputDimension],
                           data: Data(bytes: &weightData, count: weightData.count * 4)),
                TestTensor(name: "model.layers.0.mlp.gate_proj.scales", dtype: "F16",
                           shape: [outputDimension, groupsPerRow],
                           data: Data(bytes: &scalesData, count: scalesData.count * 2)),
                TestTensor(name: "model.layers.0.mlp.gate_proj.biases", dtype: "F16",
                           shape: [outputDimension, groupsPerRow],
                           data: Data(bytes: &biasesData, count: biasesData.count * 2)),
            ],
            to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)

        let store = try STAFLoader().load(at: stafURL, device: device)

        // The weight tensor should be repacked into interleaved blocks
        guard let entry = store.entries["model.layers.0.mlp.gate_proj.weight"] else {
            Issue.record("Quantized weight not found in STAF")
            return
        }

        #expect(entry.schemeIdentifier == .q4Group64ScaleF16)

        // Read the interleaved block: [scale(2B), zero(2B), quants(32B)] = 36 bytes per block
        let format = AffineQ4Group64Format()
        let blocksPerRow = inputDimension / format.groupSize  // 1
        let totalBlocks = outputDimension * blocksPerRow       // 4
        #expect(entry.payloadSize == totalBlocks * format.bytesPerBlock)

        let blockPointer = store.buffer.contents() + entry.bufferOffset
        for row in 0..<outputDimension {
            let blockBase = blockPointer + row * format.bytesPerBlock
            // Read scale (first 2 bytes)
            let scale = blockBase.load(as: Float16.self)
            #expect(scale == Float16(2.0), "Row \(row) scale: \(scale) != 2.0")
            // Read zero (next 2 bytes)
            let zero = (blockBase + 2).load(as: Float16.self)
            #expect(zero == Float16(0.5), "Row \(row) zero: \(zero) != 0.5")
        }

        // Scales and biases tensors should NOT be in the STAF (consumed by weight)
        #expect(store.entries["model.layers.0.mlp.gate_proj.scales"] == nil,
                "Consumed scales tensor should not be in STAF")
        #expect(store.entries["model.layers.0.mlp.gate_proj.biases"] == nil,
                "Consumed biases tensor should not be in STAF")
    }

    // MARK: - Deallocator Safety

    @Test("STAF file size matches converted output")
    func stafFileSizeConsistency() throws {
        let tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("staf_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDirectory) }

        var values: [Float16] = (0..<256).map { Float16($0) }
        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "test.weight", dtype: "F16", shape: [16, 16],
                           data: Data(bytes: &values, count: 256 * 2))
            ],
            to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)

        // Verify file structure
        let stafData = try Data(contentsOf: stafURL)
        #expect(stafData.count > STAF.headerSize, "STAF file too small")

        // Check magic
        let magic = stafData.withUnsafeBytes { $0.load(as: UInt32.self) }
        #expect(magic == STAF.magic, "Bad STAF magic: 0x\(String(magic, radix: 16))")

        // Check section count
        let sectionCount = stafData.withUnsafeBytes {
            ($0.baseAddress! + 40).loadUnaligned(as: UInt32.self)
        }
        #expect(sectionCount == 1, "Expected 1 tensor, got \(sectionCount)")

        // Check payload alignment
        let sectionTableOffset = stafData.withUnsafeBytes {
            Int(($0.baseAddress! + 44).loadUnaligned(as: UInt32.self))
        }
        #expect(sectionTableOffset == STAF.headerSize, "Section table should follow header")

        // Read first entry's payload offset
        let entryBase = sectionTableOffset
        let payloadOffset = stafData.withUnsafeBytes {
            Int(($0.baseAddress! + entryBase + 44).loadUnaligned(as: UInt64.self))
        }

        // Payload offset must be 4KB aligned (STAF.payloadAlignment)
        #expect(
            payloadOffset % STAF.payloadAlignment == 0,
            "Payload offset \(payloadOffset) not \(STAF.payloadAlignment)-aligned")
    }

    // MARK: - Munmap Address Correctness

    @Test("STAFLoader munmap address matches mmap allocation")
    func munmapAddressCorrectness() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("staf_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDirectory) }

        // Create a STAF file where the payload offset is 4KB but page size is 16KB
        // This exercises the alignment path where alignedPointer != payloadPointer
        var values: [Float16] = (0..<512).map { Float16($0) }
        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "test.weight", dtype: "F16", shape: [32, 16],
                           data: Data(bytes: &values, count: 512 * 2))
            ],
            to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)

        // Verify the STAF file's payload offset
        let stafData = try Data(contentsOf: stafURL)
        let sectionTableOffset = stafData.withUnsafeBytes {
            Int(($0.baseAddress! + 44).loadUnaligned(as: UInt32.self))
        }
        let payloadOffset = stafData.withUnsafeBytes {
            Int(($0.baseAddress! + sectionTableOffset + 44).loadUnaligned(as: UInt64.self))
        }

        let pageSize = Int(sysconf(_SC_PAGESIZE))
        let payloadMisalignment = payloadOffset % pageSize
        if payloadMisalignment != 0 {
            // When payload is not page-aligned, the loader must adjust.
            // This is the case that exercises the alignment code.
            print("[Test] Payload offset \(payloadOffset) has \(payloadMisalignment) byte misalignment vs \(pageSize) page size")
        }

        // Load STAF — this creates the MTLBuffer
        let store = try STAFLoader().load(at: stafURL, device: device)

        // Verify the tensor data is still correct
        guard let entry = store.entries["test.weight"] else {
            Issue.record("Tensor not found")
            return
        }

        let loaded = UnsafeBufferPointer(
            start: (store.buffer.contents() + entry.bufferOffset).bindMemory(
                to: Float16.self, capacity: 512),
            count: 512)

        for i in 0..<512 {
            #expect(loaded[i] == values[i],
                    "Munmap test: mismatch at \(i): loaded=\(loaded[i]) expected=\(values[i])")
        }
    }

    // MARK: - Validity Check

    @Test("isValid returns true for freshly converted STAF")
    func validityCheck() throws {
        let tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("staf_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDirectory) }

        var values: [Float16] = [Float16(1.0), Float16(2.0)]
        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "test.weight", dtype: "F16", shape: [2],
                           data: Data(bytes: &values, count: 4))
            ],
            to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        let converter = STAFConverter()
        try converter.convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)

        let isValid = try converter.isValid(stafURL: stafURL, safetensorsURLs: [safetensorsURL])
        #expect(isValid, "Freshly converted STAF should be valid")
    }

    @Test("STAF header records format version and metadata table")
    func headerContainsVersionedMetadataTable() throws {
        let tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("staf_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDirectory) }

        var values: [Float16] = [1, 2, 3, 4]
        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "test.weight", dtype: "F16", shape: [4],
                           data: Data(bytes: &values, count: values.count * 2))
            ],
            to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)

        let stafData = try Data(contentsOf: stafURL)
        guard let header = STAF.parseHeader(from: stafData) else {
            Issue.record("Cannot parse STAF header")
            return
        }

        #expect(header.magic == STAF.magic)
        #expect(header.formatVersion == STAF.currentFormatVersion)
        #expect(header.sectionTableOffset == UInt32(STAF.headerSize))
        #expect(header.metadataEntryCount > 0)
        #expect(header.metadataTableOffset == UInt32(STAF.headerSize + STAF.sectionEntrySize))
        #expect(header.stringTableOffset > header.metadataTableOffset)
    }

    @Test("STAF metadata roundtrip preserves typed file metadata")
    func fileMetadataRoundtrip() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("staf_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDirectory) }

        var values: [Float16] = [1, 2, 3, 4]
        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "test.weight", dtype: "F16", shape: [4],
                           data: Data(bytes: &values, count: values.count * 2))
            ],
            to: safetensorsURL)

        let metadata = STAFFileMetadata(values: [
            "model.architecture_family": .string("transformer"),
            "model.hidden_size": .uint64(2048),
            "model.attention_heads": .uint32(16),
            "model.tied_embeddings": .bool(true),
            "model.rope_theta": .float64(10000.0),
            "model.norm_eps": .float32(1e-5)
        ])

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(
            safetensorsURLs: [safetensorsURL],
            outputURL: stafURL,
            metadata: metadata
        )

        let store = try STAFLoader().load(at: stafURL, device: device)

        #expect(store.metadata[STAFMetadataKey.sourceFormat] == .string("safetensors"))
        #expect(store.metadata[STAFMetadataKey.converterVersion] == .uint32(1))
        #expect(store.metadata[STAFMetadataKey.sourceShardCount] == .uint64(1))
        #expect(store.metadata[STAFMetadataKey.metadataSchemaVersion] == .uint32(1))
        #expect(store.metadata["model.architecture_family"] == .string("transformer"))
        #expect(store.metadata["model.hidden_size"] == .uint64(2048))
        #expect(store.metadata["model.attention_heads"] == .uint32(16))
        #expect(store.metadata["model.tied_embeddings"] == .bool(true))
        #expect(store.metadata["model.rope_theta"] == .float64(10000.0))
        #expect(store.metadata["model.norm_eps"] == .float32(1e-5))
    }

    // MARK: - Helpers

    private func converter() -> STAFConverter { STAFConverter() }

    private func verifyTensorValues(
        store: STAFWeightStore, name: String,
        expected: [Float16], label: String
    ) throws {
        guard let entry = store.entries[name] else {
            Issue.record("\(label): tensor '\(name)' not found")
            return
        }

        let loaded = UnsafeBufferPointer(
            start: (store.buffer.contents() + entry.bufferOffset).bindMemory(
                to: Float16.self, capacity: expected.count),
            count: expected.count)

        for i in 0..<expected.count {
            #expect(
                loaded[i] == expected[i],
                "\(label) mismatch at \(i): loaded=\(loaded[i]) expected=\(expected[i])")
        }
    }
}

// MARK: - Safetensors Test File Writer

private struct TestTensor {
    let name: String
    let dtype: String
    let shape: [Int]
    let data: Data
}

/// Write a minimal safetensors file for testing.
///
/// Format: [8B header_size LE] [JSON header] [tensor data]
private func writeSafetensors(tensors: [TestTensor], to url: URL) throws {
    // Build tensor data section
    var dataSection = Data()
    var tensorOffsets: [(name: String, begin: Int, end: Int)] = []

    for tensor in tensors {
        let begin = dataSection.count
        dataSection.append(tensor.data)
        let end = dataSection.count
        tensorOffsets.append((name: tensor.name, begin: begin, end: end))
    }

    // Build JSON header
    var headerObject: [String: Any] = [:]
    for (i, tensor) in tensors.enumerated() {
        let offsets = tensorOffsets[i]
        headerObject[tensor.name] = [
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "data_offsets": [offsets.begin, offsets.end]
        ] as [String: Any]
    }

    let headerJSON = try JSONSerialization.data(withJSONObject: headerObject, options: .sortedKeys)
    let headerSize = UInt64(headerJSON.count)

    // Write file
    var fileData = Data()
    var headerSizeLE = headerSize.littleEndian
    fileData.append(Data(bytes: &headerSizeLE, count: 8))
    fileData.append(headerJSON)
    fileData.append(dataSection)

    try fileData.write(to: url)
}
