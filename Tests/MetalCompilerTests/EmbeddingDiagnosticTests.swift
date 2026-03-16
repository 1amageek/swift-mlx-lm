import Testing
import Metal
import Foundation
@testable import MetalCompiler

/// Diagnostic test: verify embedding lookup kernel works with real STAF data formats.
/// Tests BF16, FP16, and quantized embeddings in isolation.
@Suite("Embedding Diagnostic")
struct EmbeddingDiagnosticTests {

    /// Test BF16 embedding → FP16 output via embedding_lookup_bf16 kernel
    @Test("BF16 embedding lookup produces finite values, not NaN")
    func bf16EmbeddingProducesFiniteValues() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let embeddingDimension = 64
        let vocabSize = 100

        // Create BF16 embedding table with known values
        // BF16(1.0) = 0x3F80, BF16(2.0) = 0x4000, BF16(0.5) = 0x3F00
        var bf16Table: [UInt16] = (0..<vocabSize * embeddingDimension).map { i in
            // Each token has a unique value: token_id * 0.01 + dim * 0.001
            let tokenID = i / embeddingDimension
            let dim = i % embeddingDimension
            let value = Float(tokenID) * 0.01 + Float(dim) * 0.001
            // Convert to BF16: float32 → truncate lower 16 bits
            let f32Bits = value.bitPattern
            return UInt16(f32Bits >> 16)
        }

        let tableBuffer = device.makeBuffer(
            bytes: &bf16Table,
            length: bf16Table.count * 2,
            options: .storageModeShared)!

        let options = MTLCompileOptions()
        options.languageVersion = .version3_0
        let library = try device.makeLibrary(
            source: MetalKernelSource.allKernelSource, options: options)

        // Test both single-token and seq BF16 kernels
        let kernelNames = ["embedding_lookup_bf16", "embedding_lookup_seq_bf16"]

        for kernelName in kernelNames {
            guard let function = library.makeFunction(name: kernelName) else {
                Issue.record("\(kernelName) not found in library")
                continue
            }
            let pipeline = try device.makeComputePipelineState(function: function)

            let tokenID: Int32 = 42
            let tokenBuffer = device.makeBuffer(length: 4, options: .storageModeShared)!
            tokenBuffer.contents().bindMemory(to: Int32.self, capacity: 1).pointee = tokenID

            let outputBuffer = device.makeBuffer(
                length: embeddingDimension * 2, options: .storageModeShared)!
            memset(outputBuffer.contents(), 0, outputBuffer.length)

            guard let queue = device.makeCommandQueue(),
                  let cb = queue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else {
                Issue.record("Cannot create encoder")
                continue
            }

            enc.setComputePipelineState(pipeline)
            enc.setBuffer(tokenBuffer, offset: 0, index: 0)
            enc.setBuffer(tableBuffer, offset: 0, index: 1)
            enc.setBuffer(outputBuffer, offset: 0, index: 2)
            var dim = UInt32(embeddingDimension)
            enc.setBytes(&dim, length: 4, index: 3)

            if kernelName.contains("seq") {
                var seqLen: UInt32 = 1
                enc.setBytes(&seqLen, length: 4, index: 4)
                let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
                let gridX = (embeddingDimension + tgSize - 1) / tgSize
                enc.dispatchThreadgroups(
                    MTLSize(width: gridX, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            } else {
                let tgSize = min(embeddingDimension, pipeline.maxTotalThreadsPerThreadgroup)
                let gridX = (embeddingDimension + tgSize - 1) / tgSize
                enc.dispatchThreadgroups(
                    MTLSize(width: gridX, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            }

            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()

            if let error = cb.error {
                Issue.record("\(kernelName) GPU error: \(error)")
                continue
            }

            let resultPointer = outputBuffer.contents().bindMemory(
                to: Float16.self, capacity: embeddingDimension)

            var nanCount = 0
            var zeroCount = 0
            var finiteCount = 0
            for i in 0..<embeddingDimension {
                let val = resultPointer[i]
                if val.isNaN { nanCount += 1 }
                else if val == 0 { zeroCount += 1 }
                else { finiteCount += 1 }
            }

            let sample = (0..<min(4, embeddingDimension)).map { Float(resultPointer[$0]) }
            print("[\(kernelName)] token=\(tokenID) output[0..3]=\(sample) nan=\(nanCount) zero=\(zeroCount) finite=\(finiteCount)")

            #expect(nanCount == 0, "\(kernelName): \(nanCount) NaN values in embedding output")
            #expect(finiteCount > 0, "\(kernelName): no finite values in output")
        }
    }

    /// Test BF16 embedding → FP16 output via seq kernel with multiple tokens
    @Test("BF16 seq embedding lookup handles multiple tokens correctly")
    func bf16SeqEmbeddingMultipleTokens() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let embeddingDimension = 32
        let vocabSize = 10
        let seqLen = 5

        // BF16 embedding: token i → all dims = BF16(float(i + 1))
        var bf16Table = [UInt16](repeating: 0, count: vocabSize * embeddingDimension)
        for tokenID in 0..<vocabSize {
            let value = Float(tokenID + 1)
            let bf16 = UInt16(value.bitPattern >> 16)
            for dim in 0..<embeddingDimension {
                bf16Table[tokenID * embeddingDimension + dim] = bf16
            }
        }

        let tableBuffer = device.makeBuffer(
            bytes: &bf16Table, length: bf16Table.count * 2, options: .storageModeShared)!

        // Input tokens: [0, 1, 2, 3, 4]
        var tokenIDs: [Int32] = [0, 1, 2, 3, 4]
        let tokenBuffer = device.makeBuffer(
            bytes: &tokenIDs, length: seqLen * 4, options: .storageModeShared)!

        let outputBuffer = device.makeBuffer(
            length: seqLen * embeddingDimension * 2, options: .storageModeShared)!

        let options = MTLCompileOptions()
        options.languageVersion = .version3_0
        let library = try device.makeLibrary(
            source: MetalKernelSource.allKernelSource, options: options)
        let pipeline = try device.makeComputePipelineState(
            function: library.makeFunction(name: "embedding_lookup_seq_bf16")!)

        guard let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            Issue.record("Cannot create encoder")
            return
        }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(tokenBuffer, offset: 0, index: 0)
        enc.setBuffer(tableBuffer, offset: 0, index: 1)
        enc.setBuffer(outputBuffer, offset: 0, index: 2)
        var dim = UInt32(embeddingDimension)
        enc.setBytes(&dim, length: 4, index: 3)
        var sl = UInt32(seqLen)
        enc.setBytes(&sl, length: 4, index: 4)

        let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let gridX = (embeddingDimension + tgSize - 1) / tgSize
        enc.dispatchThreadgroups(
            MTLSize(width: gridX, height: seqLen, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        if let error = cb.error {
            Issue.record("GPU error: \(error)")
            return
        }

        let result = outputBuffer.contents().bindMemory(
            to: Float16.self, capacity: seqLen * embeddingDimension)

        for pos in 0..<seqLen {
            let expectedApprox = Float(pos + 1)  // token i → value i+1
            let actual = Float(result[pos * embeddingDimension])
            // BF16→FP16 conversion loses precision, allow tolerance
            #expect(
                abs(actual - expectedApprox) < 0.1,
                "pos=\(pos) token=\(pos): expected ~\(expectedApprox), got \(actual)")
            #expect(!result[pos * embeddingDimension].isNaN, "pos=\(pos): NaN!")
        }
    }

    /// Test that BF16 rms_norm_seq_bf16 produces finite values
    @Test("BF16 RMS norm seq produces finite values")
    func bf16RMSNormSeqFinite() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let dimension = 64
        let seqLen = 4

        // Input: known finite FP16 values
        var input: [Float16] = (0..<seqLen * dimension).map { Float16(Float($0 % dimension + 1) * 0.1) }
        let inputBuffer = device.makeBuffer(
            bytes: &input, length: input.count * 2, options: .storageModeShared)!

        // BF16 weight: all 1.0
        let bf16One = UInt16(Float(1.0).bitPattern >> 16)
        var bf16Weight = [UInt16](repeating: bf16One, count: dimension)
        let weightBuffer = device.makeBuffer(
            bytes: &bf16Weight, length: dimension * 2, options: .storageModeShared)!

        let outputBuffer = device.makeBuffer(
            length: seqLen * dimension * 2, options: .storageModeShared)!

        let options = MTLCompileOptions()
        options.languageVersion = .version3_0
        let library = try device.makeLibrary(
            source: MetalKernelSource.allKernelSource, options: options)
        let pipeline = try device.makeComputePipelineState(
            function: library.makeFunction(name: "rms_norm_seq_bf16")!)

        guard let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            Issue.record("Cannot create encoder")
            return
        }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(inputBuffer, offset: 0, index: 0)
        enc.setBuffer(weightBuffer, offset: 0, index: 1)
        enc.setBuffer(outputBuffer, offset: 0, index: 2)
        var dim = UInt32(dimension)
        var eps: Float = 1e-5
        var sl = UInt32(seqLen)
        enc.setBytes(&dim, length: 4, index: 3)
        enc.setBytes(&eps, length: 4, index: 4)
        enc.setBytes(&sl, length: 4, index: 5)

        let threads = min(dimension, pipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreadgroups(
            MTLSize(width: seqLen, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let result = outputBuffer.contents().bindMemory(
            to: Float16.self, capacity: seqLen * dimension)

        for pos in 0..<seqLen {
            let val = Float(result[pos * dimension])
            #expect(!val.isNaN, "rms_norm_seq_bf16 pos=\(pos): NaN")
            #expect(val.isFinite, "rms_norm_seq_bf16 pos=\(pos): not finite (\(val))")
        }
    }

    /// Test GEMM with Q4 G64 weights (the most common quantized format)
    @Test("Q4 G64 GEMM seq produces finite values")
    func q4g64GEMMSeqFinite() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gemm_diag_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDirectory) }

        let outputDimension = 4
        let inputDimension = 64
        let seqLen = 3
        let packedInputDimension = inputDimension / 8

        // Weight: all quants = 0x33 → nibbles (3,3) → dequant = scale*3 + zero
        var weightData: [UInt32] = Array(repeating: 0x33333333, count: outputDimension * packedInputDimension)
        var scalesData: [Float16] = Array(repeating: Float16(1.0), count: outputDimension)
        var biasesData: [Float16] = Array(repeating: Float16(0.0), count: outputDimension)

        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(name: "w.weight", dtype: "U32",
                           shape: [outputDimension, packedInputDimension],
                           data: Data(bytes: &weightData, count: weightData.count * 4)),
                TestTensor(name: "w.scales", dtype: "F16",
                           shape: [outputDimension, 1],
                           data: Data(bytes: &scalesData, count: scalesData.count * 2)),
                TestTensor(name: "w.biases", dtype: "F16",
                           shape: [outputDimension, 1],
                           data: Data(bytes: &biasesData, count: biasesData.count * 2)),
            ],
            to: safetensorsURL)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)
        let store = try STAFLoader().load(at: stafURL, device: device)

        guard let access = store.bufferAccess(for: "w.weight") else {
            Issue.record("Weight not found")
            return
        }

        let options = MTLCompileOptions()
        options.languageVersion = .version3_0
        let library = try device.makeLibrary(
            source: MetalKernelSource.allKernelSource, options: options)
        let pipeline = try device.makeComputePipelineState(
            function: library.makeFunction(name: "gemm_q4_g64")!)

        // Input: [seqLen × inputDim] all 1.0
        var inputValues = [Float16](repeating: Float16(1.0), count: seqLen * inputDimension)
        let inputBuffer = device.makeBuffer(
            bytes: &inputValues, length: inputValues.count * 2, options: .storageModeShared)!
        let outputBuffer = device.makeBuffer(
            length: seqLen * outputDimension * 2, options: .storageModeShared)!

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
        var sl = UInt32(seqLen)
        enc.setBytes(&inDim, length: 4, index: 3)
        enc.setBytes(&outDim, length: 4, index: 4)
        enc.setBytes(&sl, length: 4, index: 5)

        let simdWidth = pipeline.threadExecutionWidth
        let threads = min(2 * simdWidth, pipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreadgroups(
            MTLSize(width: (outputDimension + 1) / 2, height: seqLen, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let result = outputBuffer.contents().bindMemory(
            to: Float16.self, capacity: seqLen * outputDimension)

        // Each row: dot product of all-1 input × all-3 weights = 3.0 * 64 = 192.0
        for pos in 0..<seqLen {
            for row in 0..<outputDimension {
                let val = Float(result[pos * outputDimension + row])
                #expect(!val.isNaN, "gemm_q4_g64 pos=\(pos) row=\(row): NaN")
                #expect(abs(val - 192.0) < 1.0,
                        "gemm_q4_g64 pos=\(pos) row=\(row): expected ~192.0, got \(val)")
            }
        }
    }
}

// MARK: - Helpers

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
        tensorOffsets.append((name: tensor.name, begin: begin, end: dataSection.count))
    }
    var headerObject: [String: Any] = [:]
    for (i, tensor) in tensors.enumerated() {
        headerObject[tensor.name] = [
            "dtype": tensor.dtype, "shape": tensor.shape,
            "data_offsets": [tensorOffsets[i].begin, tensorOffsets[i].end]
        ] as [String: Any]
    }
    let headerJSON = try JSONSerialization.data(withJSONObject: headerObject, options: .sortedKeys)
    var fileData = Data()
    var headerSizeLE = UInt64(headerJSON.count).littleEndian
    fileData.append(Data(bytes: &headerSizeLE, count: 8))
    fileData.append(headerJSON)
    fileData.append(dataSection)
    try fileData.write(to: url)
}
