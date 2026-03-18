import Testing
import Metal
import Foundation
@testable import MetalCompiler

// MARK: - STAF Deep Tests

/// Thorough STAF tests covering correctness at scale, quantization end-to-end,
/// boundary conditions, error handling, and structural invariants.
@Suite("STAF Deep")
struct STAFDeepTests {

    // MARK: - Q4 GEMV End-to-End

    @Test("Q4 GEMV kernel produces correct dot product from STAF weights")
    func q4GEMVEndToEnd() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = makeTempDirectory()
        defer { cleanup(tempDirectory) }

        // Build a 2×64 quantized weight matrix where we know the dequantized values.
        // Block layout: [scale(2B), zero(2B), quants(32B)] = 36 bytes per block.
        // With group=64, each row has 1 block.
        //
        // Row 0: scale=1.0, zero=0.0, all quants=0x33 → nibbles (3,3) → dequant = 1.0*3 + 0.0 = 3.0
        // Row 1: scale=2.0, zero=1.0, all quants=0x22 → nibbles (2,2) → dequant = 2.0*2 + 1.0 = 5.0
        let outputDimension = 2
        let inputDimension = 64
        let packedInputDimension = inputDimension / 8  // 8

        // Write safetensors with MLX quantized format
        var weightData: [UInt32] = Array(repeating: 0x33333333, count: packedInputDimension)
            + Array(repeating: 0x22222222, count: packedInputDimension)
        var scalesData: [Float16] = [Float16(1.0), Float16(2.0)]
        var biasesData: [Float16] = [Float16(0.0), Float16(1.0)]

        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "test.weight", dtype: "U32",
                           shape: [outputDimension, packedInputDimension],
                           data: Data(bytes: &weightData, count: weightData.count * 4)),
                TestTensor(name: "test.scales", dtype: "F16",
                           shape: [outputDimension, 1],
                           data: Data(bytes: &scalesData, count: scalesData.count * 2)),
                TestTensor(name: "test.biases", dtype: "F16",
                           shape: [outputDimension, 1],
                           data: Data(bytes: &biasesData, count: biasesData.count * 2)),
            ],
            to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)
        let store = try STAFLoader().load(at: stafURL, device: device)

        guard let access = store.bufferAccess(for: "test.weight") else {
            Issue.record("Weight not found in STAF")
            return
        }

        // Compile gemv_q4_g64 kernel
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(
            source: MetalSourceGenerator.generateCompleteLibrary(weightFormat: .bfloat16), options: options)
        guard let function = library.makeFunction(name: "gemv_q4_g64") else {
            Issue.record("gemv_q4_g64 kernel not found")
            return
        }
        let pipeline = try device.makeComputePipelineState(function: function)

        // Input: all 1.0
        var inputValues: [Float16] = Array(repeating: Float16(1.0), count: inputDimension)
        let inputBuffer = device.makeBuffer(
            bytes: &inputValues, length: inputDimension * 2, options: .storageModeShared)!
        let outputBuffer = device.makeBuffer(
            length: outputDimension * 2, options: .storageModeShared)!

        guard let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            Issue.record("Cannot create Metal encoder")
            return
        }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(inputBuffer, offset: 0, index: 0)
        enc.setBuffer(access.buffer, offset: access.offset, index: 1)
        enc.setBuffer(outputBuffer, offset: 0, index: 2)
        var inDim = UInt32(inputDimension)
        var outDim = UInt32(outputDimension)
        enc.setBytes(&inDim, length: 4, index: 3)
        enc.setBytes(&outDim, length: 4, index: 4)

        let simdWidth = pipeline.threadExecutionWidth
        let threads = min(2 * simdWidth, pipeline.maxTotalThreadsPerThreadgroup)
        let groups = (outputDimension + 1) / 2
        enc.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        if let error = cb.error {
            Issue.record("GPU error: \(error)")
            return
        }

        let resultPointer = outputBuffer.contents().bindMemory(to: Float16.self, capacity: outputDimension)

        // Row 0: all weights dequant to 3.0, input all 1.0 → dot = 3.0 * 64 = 192.0
        let row0 = Float(resultPointer[0])
        #expect(abs(row0 - 192.0) < 1.0, "Row 0: expected ~192.0, got \(row0)")

        // Row 1: all weights dequant to 5.0, input all 1.0 → dot = 5.0 * 64 = 320.0
        let row1 = Float(resultPointer[1])
        #expect(abs(row1 - 320.0) < 1.0, "Row 1: expected ~320.0, got \(row1)")
    }

    // MARK: - FP16 GEMV End-to-End

    @Test("FP16 GEMV kernel produces correct dot product from STAF weights")
    func fp16GEMVEndToEnd() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = makeTempDirectory()
        defer { cleanup(tempDirectory) }

        // 3×4 weight matrix, known values
        // Row 0: [1, 0, 0, 0]
        // Row 1: [0, 1, 0, 0]
        // Row 2: [1, 1, 1, 1]
        let outputDimension = 3
        let inputDimension = 4
        var weights: [Float16] = [
            1, 0, 0, 0,
            0, 1, 0, 0,
            1, 1, 1, 1,
        ]

        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "test.weight", dtype: "F16",
                           shape: [outputDimension, inputDimension],
                           data: Data(bytes: &weights, count: weights.count * 2)),
            ],
            to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)
        let store = try STAFLoader().load(at: stafURL, device: device)

        guard let access = store.bufferAccess(for: "test.weight") else {
            Issue.record("Weight not found")
            return
        }

        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(
            source: MetalSourceGenerator.generateCompleteLibrary(weightFormat: .bfloat16), options: options)
        let pipeline = try device.makeComputePipelineState(
            function: library.makeFunction(name: "gemv")!)

        // Input: [2, 3, 5, 7]
        var inputValues: [Float16] = [2, 3, 5, 7]
        let inputBuffer = device.makeBuffer(
            bytes: &inputValues, length: inputDimension * 2, options: .storageModeShared)!
        let outputBuffer = device.makeBuffer(
            length: outputDimension * 2, options: .storageModeShared)!

        guard let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            Issue.record("Cannot create Metal encoder")
            return
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(inputBuffer, offset: 0, index: 0)
        enc.setBuffer(access.buffer, offset: access.offset, index: 1)
        enc.setBuffer(outputBuffer, offset: 0, index: 2)
        var inDim = UInt32(inputDimension)
        var outDim = UInt32(outputDimension)
        enc.setBytes(&inDim, length: 4, index: 3)
        enc.setBytes(&outDim, length: 4, index: 4)
        let simdWidth = pipeline.threadExecutionWidth
        let threads = min(2 * simdWidth, pipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreadgroups(
            MTLSize(width: (outputDimension + 1) / 2, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let result = UnsafeBufferPointer(
            start: outputBuffer.contents().bindMemory(to: Float16.self, capacity: outputDimension),
            count: outputDimension)

        // Row 0: dot([1,0,0,0], [2,3,5,7]) = 2.0
        #expect(result[0] == Float16(2.0), "Row 0: expected 2.0, got \(result[0])")
        // Row 1: dot([0,1,0,0], [2,3,5,7]) = 3.0
        #expect(result[1] == Float16(3.0), "Row 1: expected 3.0, got \(result[1])")
        // Row 2: dot([1,1,1,1], [2,3,5,7]) = 17.0
        #expect(result[2] == Float16(17.0), "Row 2: expected 17.0, got \(result[2])")
    }

    // MARK: - Many Tensors (No Offset Overlap)

    @Test("100 tensors load at non-overlapping offsets")
    func manyTensorsNoOverlap() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = makeTempDirectory()
        defer { cleanup(tempDirectory) }

        // Create 100 tensors of varying sizes
        let tensorCount = 100
        var tensors: [TestTensor] = []
        var expectedData: [String: [Float16]] = [:]

        for i in 0..<tensorCount {
            let size = 32 + (i * 7) % 256  // 32..287 elements
            let name = "model.layers.\(i / 4).block.\(i % 4).weight"
            let values: [Float16] = (0..<size).map { Float16(Float(i * 1000 + $0) * 0.01) }
            var mutableValues = values
            tensors.append(TestTensor(
                name: name, dtype: "F16", shape: [size],
                data: Data(bytes: &mutableValues, count: size * 2)))
            expectedData[name] = values
        }

        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(tensors: tensors, to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)
        let store = try STAFLoader().load(at: stafURL, device: device)

        #expect(store.entries.count == tensorCount,
                "Expected \(tensorCount) tensors, got \(store.entries.count)")

        // Verify no offset overlaps
        var ranges: [(name: String, start: Int, end: Int)] = []
        for (name, entry) in store.entries {
            ranges.append((name: name, start: entry.bufferOffset, end: entry.bufferOffset + entry.payloadSize))
        }
        ranges.sort { $0.start < $1.start }
        for i in 1..<ranges.count {
            #expect(
                ranges[i].start >= ranges[i - 1].end,
                "Overlap: '\(ranges[i-1].name)' [..\(ranges[i-1].end)] vs '\(ranges[i].name)' [\(ranges[i].start)..]")
        }

        // Verify all values
        for (name, expected) in expectedData {
            guard let entry = store.entries[name] else {
                Issue.record("Tensor '\(name)' missing")
                continue
            }
            let loaded = UnsafeBufferPointer(
                start: (store.buffer.contents() + entry.bufferOffset).bindMemory(
                    to: Float16.self, capacity: expected.count),
                count: expected.count)
            for j in 0..<expected.count {
                if loaded[j] != expected[j] {
                    Issue.record("\(name)[\(j)]: loaded=\(loaded[j]) expected=\(expected[j])")
                    break
                }
            }
        }
    }

    // MARK: - Large Tensor

    @Test("Large tensor (512K elements) survives roundtrip")
    func largeTensorRoundtrip() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = makeTempDirectory()
        defer { cleanup(tempDirectory) }

        let elementCount = 512 * 1024  // 512K elements = 1MB
        var values: [Float16] = (0..<elementCount).map { Float16(Float($0) * 0.001) }

        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "large.weight", dtype: "F16", shape: [512, 1024],
                           data: Data(bytes: &values, count: elementCount * 2))
            ],
            to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)
        let store = try STAFLoader().load(at: stafURL, device: device)

        guard let entry = store.entries["large.weight"] else {
            Issue.record("Large tensor not found")
            return
        }
        #expect(entry.payloadSize == elementCount * 2)

        let loaded = UnsafeBufferPointer(
            start: (store.buffer.contents() + entry.bufferOffset).bindMemory(
                to: Float16.self, capacity: elementCount),
            count: elementCount)

        // Spot check (checking all 512K is slow, check boundaries + stride)
        let checkIndices = [0, 1, 2, 100, 1000, 10000, elementCount / 2,
                            elementCount - 3, elementCount - 2, elementCount - 1]
        for i in checkIndices {
            #expect(loaded[i] == values[i], "Large tensor[\(i)]: loaded=\(loaded[i]) expected=\(values[i])")
        }

        // Check a random stride through the middle
        for i in stride(from: 0, to: elementCount, by: 1024) {
            #expect(loaded[i] == values[i], "Large tensor stride[\(i)]: loaded=\(loaded[i]) expected=\(values[i])")
        }
    }

    // MARK: - Nibble Repacking Correctness

    @Test("Q4 nibble repacking produces correct byte patterns")
    func nibbleRepackingCorrectness() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = makeTempDirectory()
        defer { cleanup(tempDirectory) }

        // Q4 G64 requires inputDimension=64 minimum (1 group of 64 weights).
        // MLX packs 8 × 4-bit values per uint32 → packedInputDimension = 64/8 = 8.
        //
        // First uint32 = 0x76543210 → nibbles: 0,1,2,3,4,5,6,7
        // STAF repacks to uint8 pairs (low nibble first):
        //   byte 0: (nibble 0) | (nibble 1 << 4) = 0x10
        //   byte 1: (nibble 2) | (nibble 3 << 4) = 0x32
        //   byte 2: (nibble 4) | (nibble 5 << 4) = 0x54
        //   byte 3: (nibble 6) | (nibble 7 << 4) = 0x76
        let outputDimension = 1
        let packedInputDimension = 8  // 8 uint32 = 64 nibbles = 64 weights
        let groupsPerRow = 1  // 64 / 64

        // First uint32 has the known pattern, rest are zero
        var weightData: [UInt32] = [0x76543210] + Array(repeating: 0, count: packedInputDimension - 1)
        var scalesData: [Float16] = [Float16(1.0)]
        var biasesData: [Float16] = [Float16(0.0)]

        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "test.weight", dtype: "U32",
                           shape: [outputDimension, packedInputDimension],
                           data: Data(bytes: &weightData, count: weightData.count * 4)),
                TestTensor(name: "test.scales", dtype: "F16",
                           shape: [outputDimension, groupsPerRow],
                           data: Data(bytes: &scalesData, count: scalesData.count * 2)),
                TestTensor(name: "test.biases", dtype: "F16",
                           shape: [outputDimension, groupsPerRow],
                           data: Data(bytes: &biasesData, count: biasesData.count * 2)),
            ],
            to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)
        let store = try STAFLoader().load(at: stafURL, device: device)

        guard let entry = store.entries["test.weight"] else {
            Issue.record("Weight not found")
            return
        }
        #expect(entry.schemeIdentifier == .q4Group64ScaleF16)

        // Block layout: [scale(2B), zero(2B), quants(32B)] = 36 bytes per block
        let blockBase = store.buffer.contents() + entry.bufferOffset
        let scale = blockBase.load(as: Float16.self)
        #expect(scale == Float16(1.0), "Scale: \(scale)")
        let zero = (blockBase + 2).load(as: Float16.self)
        #expect(zero == Float16(0.0), "Zero: \(zero)")

        // Check the first 4 repacked bytes from the known uint32 pattern
        // These are the first 4 bytes of the 32-byte quant section (after 4-byte header)
        let quantBytes = (blockBase + 4).bindMemory(to: UInt8.self, capacity: 32)
        let expectedFirst4: [UInt8] = [0x10, 0x32, 0x54, 0x76]
        for i in 0..<4 {
            #expect(
                quantBytes[i] == expectedFirst4[i],
                "Nibble byte \(i): 0x\(String(quantBytes[i], radix: 16)) expected 0x\(String(expectedFirst4[i], radix: 16))")
        }

        // Remaining 28 bytes should be 0x00 (from zero uint32s)
        for i in 4..<32 {
            #expect(quantBytes[i] == 0x00,
                    "Nibble byte \(i) should be 0x00, got 0x\(String(quantBytes[i], radix: 16))")
        }
    }

    // MARK: - Q4 G128 GEMV End-to-End

    @Test("Q4 G128 GEMV kernel produces correct dot product from STAF weights")
    func q4g128GEMVEndToEnd() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = makeTempDirectory()
        defer { cleanup(tempDirectory) }

        // Q4 G128: 128 weights per block, 68 bytes per block (4B header + 64B quants)
        let outputDimension = 2
        let inputDimension = 128
        let packedInputDimension = inputDimension / 8  // 16
        let groupsPerRow = inputDimension / 128  // 1

        // Row 0: scale=1.0, zero=0.0, all quants=0x55 → nibbles (5,5) → dequant = 5.0
        // Row 1: scale=0.5, zero=2.0, all quants=0xAA → nibbles (10,10) → dequant = 0.5*10+2.0 = 7.0
        var weightData: [UInt32] = Array(repeating: 0x55555555, count: packedInputDimension)
            + Array(repeating: 0xAAAAAAAA, count: packedInputDimension)
        var scalesData: [Float16] = [Float16(1.0), Float16(0.5)]
        var biasesData: [Float16] = [Float16(0.0), Float16(2.0)]

        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "test.weight", dtype: "U32",
                           shape: [outputDimension, packedInputDimension],
                           data: Data(bytes: &weightData, count: weightData.count * 4)),
                TestTensor(name: "test.scales", dtype: "F16",
                           shape: [outputDimension, groupsPerRow],
                           data: Data(bytes: &scalesData, count: scalesData.count * 2)),
                TestTensor(name: "test.biases", dtype: "F16",
                           shape: [outputDimension, groupsPerRow],
                           data: Data(bytes: &biasesData, count: biasesData.count * 2)),
            ],
            to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)
        let store = try STAFLoader().load(at: stafURL, device: device)

        guard let entry = store.entries["test.weight"] else {
            Issue.record("Weight not found")
            return
        }
        #expect(entry.schemeIdentifier == .q4Group128ScaleF16)

        guard let access = store.bufferAccess(for: "test.weight") else {
            Issue.record("Buffer access failed")
            return
        }

        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(
            source: MetalSourceGenerator.generateCompleteLibrary(weightFormat: .bfloat16), options: options)
        let pipeline = try device.makeComputePipelineState(
            function: library.makeFunction(name: "gemv_q4_g128")!)

        var inputValues: [Float16] = Array(repeating: Float16(1.0), count: inputDimension)
        let inputBuffer = device.makeBuffer(
            bytes: &inputValues, length: inputDimension * 2, options: .storageModeShared)!
        let outputBuffer = device.makeBuffer(
            length: outputDimension * 2, options: .storageModeShared)!

        guard let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            Issue.record("Cannot create encoder")
            return
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(inputBuffer, offset: 0, index: 0)
        enc.setBuffer(access.buffer, offset: access.offset, index: 1)
        enc.setBuffer(outputBuffer, offset: 0, index: 2)
        var inDim = UInt32(inputDimension)
        var outDim = UInt32(outputDimension)
        enc.setBytes(&inDim, length: 4, index: 3)
        enc.setBytes(&outDim, length: 4, index: 4)
        let simdWidth = pipeline.threadExecutionWidth
        let threads = min(2 * simdWidth, pipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreadgroups(
            MTLSize(width: (outputDimension + 1) / 2, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let result = outputBuffer.contents().bindMemory(to: Float16.self, capacity: outputDimension)
        // Row 0: 5.0 * 128 = 640.0
        let row0 = Float(result[0])
        #expect(abs(row0 - 640.0) < 2.0, "Q4G128 Row 0: expected ~640.0, got \(row0)")
        // Row 1: 7.0 * 128 = 896.0
        let row1 = Float(result[1])
        #expect(abs(row1 - 896.0) < 2.0, "Q4G128 Row 1: expected ~896.0, got \(row1)")
    }

    // MARK: - BF16 GEMV End-to-End

    @Test("BF16 GEMV kernel produces correct dot product from STAF weights")
    func bf16GEMVEndToEnd() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = makeTempDirectory()
        defer { cleanup(tempDirectory) }

        // BF16 identity-like matrix: 2×4
        // Row 0: [1, 0, 0, 0]
        // Row 1: [0, 0, 1, 0]
        let outputDimension = 2
        let inputDimension = 4
        var bf16Weights: [BFloat16] = [
            .one, .zero, .zero, .zero,  // [1, 0, 0, 0]
            .zero, .zero, .one, .zero,  // [0, 0, 1, 0]
        ]

        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "test.weight", dtype: "BF16",
                           shape: [outputDimension, inputDimension],
                           data: Data(bytes: &bf16Weights, count: bf16Weights.count * MemoryLayout<BFloat16>.size))
            ],
            to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)
        let store = try STAFLoader().load(at: stafURL, device: device)

        guard let entry = store.entries["test.weight"] else {
            Issue.record("Weight not found")
            return
        }
        #expect(entry.schemeIdentifier == .bf16RowMajor)

        guard let access = store.bufferAccess(for: "test.weight") else {
            Issue.record("Buffer access failed")
            return
        }

        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(
            source: MetalSourceGenerator.generateCompleteLibrary(weightFormat: .bfloat16), options: options)
        let pipeline = try device.makeComputePipelineState(
            function: library.makeFunction(name: "gemv_bf16")!)

        // Input: [2, 3, 5, 7]
        var inputValues: [Float16] = [2, 3, 5, 7]
        let inputBuffer = device.makeBuffer(
            bytes: &inputValues, length: inputDimension * 2, options: .storageModeShared)!
        let outputBuffer = device.makeBuffer(
            length: outputDimension * 2, options: .storageModeShared)!

        guard let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            Issue.record("Cannot create encoder")
            return
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(inputBuffer, offset: 0, index: 0)
        enc.setBuffer(access.buffer, offset: access.offset, index: 1)
        enc.setBuffer(outputBuffer, offset: 0, index: 2)
        var inDim = UInt32(inputDimension)
        var outDim = UInt32(outputDimension)
        enc.setBytes(&inDim, length: 4, index: 3)
        enc.setBytes(&outDim, length: 4, index: 4)
        let simdWidth = pipeline.threadExecutionWidth
        let threads = min(2 * simdWidth, pipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreadgroups(
            MTLSize(width: (outputDimension + 1) / 2, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let result = outputBuffer.contents().bindMemory(to: Float16.self, capacity: outputDimension)
        // Row 0: dot([1,0,0,0], [2,3,5,7]) = 2.0
        #expect(result[0] == Float16(2.0), "BF16 Row 0: expected 2.0, got \(result[0])")
        // Row 1: dot([0,0,1,0], [2,3,5,7]) = 5.0
        #expect(result[1] == Float16(5.0), "BF16 Row 1: expected 5.0, got \(result[1])")
    }

    // MARK: - All QuantizationFormat Scheme Detection

    @Test("Converter correctly detects all quantization schemes")
    func allQuantizationSchemeDetection() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = makeTempDirectory()
        defer { cleanup(tempDirectory) }

        // FP16 dense
        var fp16Values: [Float16] = Array(repeating: Float16(1.0), count: 8)
        // BF16 dense
        var bf16Values: [BFloat16] = Array(repeating: .one, count: 8)
        // FP32 dense (converted to FP16)
        var fp32Values: [Float] = Array(repeating: 1.0, count: 8)
        // Q4 G64: needs packedDim=8 (64 elements) and companion scales/biases
        var q4g64Weight: [UInt32] = Array(repeating: 0, count: 8)
        var q4g64Scales: [Float16] = [Float16(1.0)]
        var q4g64Biases: [Float16] = [Float16(0.0)]
        // Q4 G128: needs packedDim=16 (128 elements) and companion scales/biases
        var q4g128Weight: [UInt32] = Array(repeating: 0, count: 16)
        var q4g128Scales: [Float16] = [Float16(1.0)]
        var q4g128Biases: [Float16] = [Float16(0.0)]

        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "dense_fp16.weight", dtype: "F16", shape: [2, 4],
                           data: Data(bytes: &fp16Values, count: 16)),
                TestTensor(name: "dense_bf16.weight", dtype: "BF16", shape: [2, 4],
                           data: Data(bytes: &bf16Values, count: 16)),
                TestTensor(name: "dense_fp32.weight", dtype: "F32", shape: [2, 4],
                           data: Data(bytes: &fp32Values, count: 32)),
                TestTensor(name: "quant_g64.weight", dtype: "U32", shape: [1, 8],
                           data: Data(bytes: &q4g64Weight, count: 32)),
                TestTensor(name: "quant_g64.scales", dtype: "F16", shape: [1, 1],
                           data: Data(bytes: &q4g64Scales, count: 2)),
                TestTensor(name: "quant_g64.biases", dtype: "F16", shape: [1, 1],
                           data: Data(bytes: &q4g64Biases, count: 2)),
                TestTensor(name: "quant_g128.weight", dtype: "U32", shape: [1, 16],
                           data: Data(bytes: &q4g128Weight, count: 64)),
                TestTensor(name: "quant_g128.scales", dtype: "F16", shape: [1, 1],
                           data: Data(bytes: &q4g128Scales, count: 2)),
                TestTensor(name: "quant_g128.biases", dtype: "F16", shape: [1, 1],
                           data: Data(bytes: &q4g128Biases, count: 2)),
            ],
            to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)
        let store = try STAFLoader().load(at: stafURL, device: device)

        // Dense formats
        #expect(store.entries["dense_fp16.weight"]?.schemeIdentifier == .fp16RowMajor)
        #expect(store.entries["dense_bf16.weight"]?.schemeIdentifier == .bf16RowMajor)
        #expect(store.entries["dense_fp32.weight"]?.schemeIdentifier == .fp16RowMajor)  // F32→F16

        // Quantized formats
        #expect(store.entries["quant_g64.weight"]?.schemeIdentifier == .q4Group64ScaleF16)
        #expect(store.entries["quant_g128.weight"]?.schemeIdentifier == .q4Group128ScaleF16)

        // Companion tensors consumed
        #expect(store.entries["quant_g64.scales"] == nil)
        #expect(store.entries["quant_g64.biases"] == nil)
        #expect(store.entries["quant_g128.scales"] == nil)
        #expect(store.entries["quant_g128.biases"] == nil)

        // QuantizationFormatRegistry resolves correctly
        for (name, entry) in store.entries {
            let format = QuantizationFormatRegistry.format(for: entry.schemeIdentifier)
            #expect(format != nil, "No format registered for '\(name)' scheme \(entry.schemeIdentifier)")
        }
    }

    // MARK: - Tensor Alignment

    @Test("Each tensor payload starts at 256-byte aligned offset within file")
    func tensorPayloadAlignment() throws {
        let tempDirectory = makeTempDirectory()
        defer { cleanup(tempDirectory) }

        // Create tensors of varying sizes to stress alignment
        var tensors: [TestTensor] = []
        let sizes = [1, 3, 7, 15, 31, 63, 127, 255, 511, 1023]
        for (i, size) in sizes.enumerated() {
            var values: [Float16] = Array(repeating: Float16(Float(i)), count: size)
            tensors.append(TestTensor(
                name: "t\(i)", dtype: "F16", shape: [size],
                data: Data(bytes: &values, count: size * 2)))
        }

        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(tensors: tensors, to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)

        // Read raw STAF and check alignment
        let stafData = try Data(contentsOf: stafURL)
        let sectionCount = stafData.withUnsafeBytes {
            Int(($0.baseAddress! + 40).loadUnaligned(as: UInt32.self))
        }
        let sectionTableOffset = stafData.withUnsafeBytes {
            Int(($0.baseAddress! + 44).loadUnaligned(as: UInt32.self))
        }

        for i in 0..<sectionCount {
            let entryBase = sectionTableOffset + i * 128
            let payloadOffset = stafData.withUnsafeBytes {
                Int(($0.baseAddress! + entryBase + 44).loadUnaligned(as: UInt64.self))
            }
            #expect(
                payloadOffset % STAF.tensorAlignment == 0,
                "Tensor \(i) payloadOffset \(payloadOffset) not \(STAF.tensorAlignment)-byte aligned")
        }
    }

    // MARK: - Shape Preservation

    @Test("Tensor shapes survive roundtrip")
    func shapePreservation() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = makeTempDirectory()
        defer { cleanup(tempDirectory) }

        let testCases: [(name: String, shape: [Int])] = [
            ("scalar", [1]),
            ("vector", [128]),
            ("matrix", [64, 32]),
            ("tensor3d", [12, 8, 64]),
            ("tensor4d", [2, 12, 8, 64]),
        ]

        var tensors: [TestTensor] = []
        for tc in testCases {
            let elementCount = tc.shape.reduce(1, *)
            var values: [Float16] = Array(repeating: Float16(1.0), count: elementCount)
            tensors.append(TestTensor(
                name: tc.name, dtype: "F16", shape: tc.shape,
                data: Data(bytes: &values, count: elementCount * 2)))
        }

        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(tensors: tensors, to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)
        let store = try STAFLoader().load(at: stafURL, device: device)

        for tc in testCases {
            guard let entry = store.entries[tc.name] else {
                Issue.record("Tensor '\(tc.name)' missing")
                continue
            }
            #expect(entry.shape == tc.shape,
                    "'\(tc.name)' shape: loaded=\(entry.shape) expected=\(tc.shape)")
        }
    }

    // MARK: - Float16 Edge Values

    @Test("Float16 edge values (0, -0, max, min, denormal) survive roundtrip")
    func float16EdgeValues() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = makeTempDirectory()
        defer { cleanup(tempDirectory) }

        var values: [Float16] = [
            Float16(0.0),           // +0
            -Float16(0.0),          // -0
            Float16.greatestFiniteMagnitude,
            -Float16.greatestFiniteMagnitude,
            Float16.leastNormalMagnitude,
            Float16.leastNonzeroMagnitude,  // denormal
            Float16(1.0),
            Float16(-1.0),
        ]

        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "edge.weight", dtype: "F16", shape: [values.count],
                           data: Data(bytes: &values, count: values.count * 2))
            ],
            to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)
        let store = try STAFLoader().load(at: stafURL, device: device)

        guard let entry = store.entries["edge.weight"] else {
            Issue.record("Tensor not found")
            return
        }

        let loaded = UnsafeBufferPointer(
            start: (store.buffer.contents() + entry.bufferOffset).bindMemory(
                to: Float16.self, capacity: values.count),
            count: values.count)

        // Compare bitwise (to distinguish +0 from -0)
        for i in 0..<values.count {
            let loadedBits = loaded[i].bitPattern
            let expectedBits = values[i].bitPattern
            #expect(
                loadedBits == expectedBits,
                "Edge value[\(i)]: loaded bits=0x\(String(loadedBits, radix: 16)) expected=0x\(String(expectedBits, radix: 16))")
        }
    }

    // MARK: - Multi-Shard Safetensors

    @Test("Multiple safetensors shards merge into single STAF")
    func multiShardConversion() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = makeTempDirectory()
        defer { cleanup(tempDirectory) }

        // Shard 1: embedding
        var shard1Values: [Float16] = (0..<64).map { Float16($0 + 1) }
        let shard1URL = tempDirectory.appendingPathComponent("model-00001-of-00002.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "model.embed_tokens.weight", dtype: "F16", shape: [8, 8],
                           data: Data(bytes: &shard1Values, count: 64 * 2))
            ],
            to: shard1URL)

        // Shard 2: norm
        var shard2Values: [Float16] = (0..<32).map { Float16(100 + $0) }
        let shard2URL = tempDirectory.appendingPathComponent("model-00002-of-00002.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "model.norm.weight", dtype: "F16", shape: [32],
                           data: Data(bytes: &shard2Values, count: 32 * 2))
            ],
            to: shard2URL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(
            safetensorsURLs: [shard1URL, shard2URL], outputURL: stafURL)
        let store = try STAFLoader().load(at: stafURL, device: device)

        // Both tensors must exist
        #expect(store.entries.count == 2, "Expected 2 tensors from 2 shards")

        // Verify shard 1 values
        guard let embEntry = store.entries["model.embed_tokens.weight"] else {
            Issue.record("Embedding not found")
            return
        }
        let embLoaded = UnsafeBufferPointer(
            start: (store.buffer.contents() + embEntry.bufferOffset).bindMemory(
                to: Float16.self, capacity: 64),
            count: 64)
        for i in 0..<64 {
            #expect(embLoaded[i] == shard1Values[i], "Shard 1 [\(i)] mismatch")
        }

        // Verify shard 2 values
        guard let normEntry = store.entries["model.norm.weight"] else {
            Issue.record("Norm not found")
            return
        }
        let normLoaded = UnsafeBufferPointer(
            start: (store.buffer.contents() + normEntry.bufferOffset).bindMemory(
                to: Float16.self, capacity: 32),
            count: 32)
        for i in 0..<32 {
            #expect(normLoaded[i] == shard2Values[i], "Shard 2 [\(i)] mismatch")
        }
    }

    // MARK: - Corrupted File Handling

    @Test("STAFLoader rejects truncated file")
    func rejectsTruncatedFile() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = makeTempDirectory()
        defer { cleanup(tempDirectory) }

        // Write a file that's too small to be valid STAF
        let truncatedURL = tempDirectory.appendingPathComponent("truncated.staf")
        try Data(count: 32).write(to: truncatedURL)

        #expect(throws: STAFLoadError.self) {
            _ = try STAFLoader().load(at: truncatedURL, device: device)
        }
    }

    @Test("STAFLoader rejects bad magic")
    func rejectsBadMagic() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = makeTempDirectory()
        defer { cleanup(tempDirectory) }

        // Write a valid-sized file with wrong magic
        var badData = Data(count: STAF.headerSize + 256)
        badData.withUnsafeMutableBytes { buf in
            buf.storeBytes(of: UInt32(0xDEADBEEF), toByteOffset: 0, as: UInt32.self)
        }
        let badURL = tempDirectory.appendingPathComponent("bad_magic.staf")
        try badData.write(to: badURL)

        #expect(throws: STAFLoadError.self) {
            _ = try STAFLoader().load(at: badURL, device: device)
        }
    }

    // MARK: - Tensor Name with Special Characters

    @Test("Long tensor names survive roundtrip")
    func longTensorNames() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = makeTempDirectory()
        defer { cleanup(tempDirectory) }

        let longName = "model.layers.99.block_sparse_moe.experts.7.feed_forward.down_proj.weight"
        var values: [Float16] = [Float16(42.0), Float16(43.0)]

        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: longName, dtype: "F16", shape: [2],
                           data: Data(bytes: &values, count: 4))
            ],
            to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)
        let store = try STAFLoader().load(at: stafURL, device: device)

        guard let entry = store.entries[longName] else {
            Issue.record("Long-named tensor not found")
            return
        }
        #expect(entry.name == longName)

        let loaded = (store.buffer.contents() + entry.bufferOffset).bindMemory(
            to: Float16.self, capacity: 2)
        #expect(loaded[0] == Float16(42.0))
        #expect(loaded[1] == Float16(43.0))
    }

    // MARK: - bufferAccess and tensor API Consistency

    @Test("bufferAccess and tensor return consistent offsets")
    func bufferAccessConsistency() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = makeTempDirectory()
        defer { cleanup(tempDirectory) }

        var values: [Float16] = [Float16(1.0), Float16(2.0), Float16(3.0), Float16(4.0)]
        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "test.weight", dtype: "F16", shape: [4],
                           data: Data(bytes: &values, count: 8))
            ],
            to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)
        let store = try STAFLoader().load(at: stafURL, device: device)

        // bufferAccess API
        guard let access = store.bufferAccess(for: "test.weight") else {
            Issue.record("bufferAccess returned nil")
            return
        }

        // tensor API
        guard let tensorResult = store.tensor(for: "test.weight") else {
            Issue.record("tensor returned nil")
            return
        }

        // Both should return the same offset
        #expect(access.offset == tensorResult.offset, "Offsets differ: access=\(access.offset) tensor=\(tensorResult.offset)")
    }

    // MARK: - STAF isValid with Modified Source

    @Test("isValid returns false when source is newer than STAF")
    func isValidDetectsStaleCache() throws {
        let tempDirectory = makeTempDirectory()
        defer { cleanup(tempDirectory) }

        var values: [Float16] = [Float16(1.0)]
        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "t", dtype: "F16", shape: [1],
                           data: Data(bytes: &values, count: 2))
            ],
            to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)

        // Touch source file to make it newer
        Thread.sleep(forTimeInterval: 0.1)
        try "".data(using: .utf8)!.write(to: safetensorsURL)
        try writeSafetensors(
            tensors: [
                TestTensor(name: "t", dtype: "F16", shape: [1],
                           data: Data(bytes: &values, count: 2))
            ],
            to: safetensorsURL)

        let isValid = try STAFConverter().isValid(stafURL: stafURL, safetensorsURLs: [safetensorsURL])
        #expect(!isValid, "STAF should be invalid when source is newer")
    }

    // MARK: - Helpers

    private func makeTempDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("staf_deep_\(UUID().uuidString)")
        try! FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func cleanup(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }
}

// MARK: - Safetensors File Writer (shared)

private struct TestTensor {
    let name: String
    let dtype: String
    let shape: [Int]
    let data: Data
}

private func writeSafetensors(tensors: [TestTensor], to url: URL) throws {
    var dataSection = Data()
    var tensorOffsets: [(name: String, begin: Int, end: Int)] = []

    for tensor in tensors {
        let begin = dataSection.count
        dataSection.append(tensor.data)
        let end = dataSection.count
        tensorOffsets.append((name: tensor.name, begin: begin, end: end))
    }

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

    var fileData = Data()
    var headerSizeLE = headerSize.littleEndian
    fileData.append(Data(bytes: &headerSizeLE, count: 8))
    fileData.append(headerJSON)
    fileData.append(dataSection)

    try fileData.write(to: url)
}
