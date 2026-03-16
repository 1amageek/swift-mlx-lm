import Testing
import Metal
import Foundation
@testable import MetalCompiler
import LMArchitecture
import ModelDeclarations
import LMIR

/// Diagnostic test using real-scale BF16 weights (hiddenSize=2048, 16 layers).
/// Reproduces the exact conditions of the app failure.
@Suite("Real Scale Diagnostic")
struct RealModelDiagnosticTests {

    @Test("Step-by-step prefill at real scale (hiddenSize=2048, 16 layers)")
    func realScaleStepByStep() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        // Match real LFM2.5-1.2B-Thinking dimensions
        let hiddenSize = 2048
        let intermediateSize = 5632
        let vocabSize = 1000  // small vocab to fit in temp disk; dimensions match real model
        let attentionHeads = 32
        let kvHeads = 8
        let headDim = 64
        let convKernelSize = 3
        let seqLen = 8  // small seq for diagnostic (not 986)

        let tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("real_diag_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDirectory) }

        // Create BF16 weights at real scale for 1 conv layer + 1 attn layer
        try createRealScaleWeights(
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            attentionHeads: attentionHeads,
            kvHeads: kvHeads,
            headDim: headDim,
            convKernelSize: convKernelSize,
            to: tempDirectory)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        let store = try STAFLoader().load(at: stafURL, device: device)
        print("[Real diag] STAF loaded: \(store.entries.count) tensors")

        // Build LFM2-like model with 2 layers (1 conv + 1 attn)
        let config = ModelConfig(
            hiddenSize: hiddenSize, layerCount: 2, intermediateSize: intermediateSize,
            vocabSize: vocabSize, attentionHeads: attentionHeads, kvHeads: kvHeads, headDim: headDim,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 10000, ropeDimension: headDim,
            ropeScaling: nil, tiedEmbeddings: false,
            expertCount: nil, expertsPerToken: nil, qkNorm: true,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: convKernelSize,
            partialRotaryFactor: nil, slidingWindow: nil,
            layerTypes: ["conv", "attention"]
        )
        let graph = try LFM2(config: config).makeModelGraph()
        let resolved = ParameterResolver().resolve(graph: graph, convention: .lfm2Family)

        let compiler = MetalInferenceCompiler()
        let prefillPlan = try compiler.compilePrefill(
            graph: resolved,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            maximumSequenceLength: 32,
            stafWeightStore: store,
            device: device)

        print("[Real diag] Prefill plan: \(prefillPlan.stepCount) steps")

        // Fill tokens and positions
        let tokenPtr = prefillPlan.buffers.tokenIDs.contents().bindMemory(to: Int32.self, capacity: seqLen)
        let posPtr = prefillPlan.buffers.positions.contents().bindMemory(to: UInt32.self, capacity: seqLen)
        for i in 0..<seqLen {
            tokenPtr[i] = Int32(i + 1)
            posPtr[i] = UInt32(i)
        }

        // Run each step individually, check NaN after each
        var nanFoundAtStep = -1
        for stepIndex in 0..<prefillPlan.steps.count {
            let step = prefillPlan.steps[stepIndex]

            guard let queue = device.makeCommandQueue(),
                  let cb = queue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else {
                Issue.record("Cannot create encoder at step \(stepIndex)")
                return
            }
            enc.memoryBarrier(scope: .buffers)

            switch step.mode {
            case .batch:
                enc.setComputePipelineState(step.pipeline)
                for (index, buffer, offset) in step.bufferBindings {
                    enc.setBuffer(buffer, offset: offset, index: index)
                }
                for (index, value) in step.bytesBindings {
                    value.withUnsafeBufferPointer { ptr in
                        enc.setBytes(ptr.baseAddress!, length: ptr.count, index: index)
                    }
                }
                if let seqLenIndex = step.sequenceLengthBindingIndex {
                    var sl = UInt32(seqLen)
                    withUnsafeBytes(of: &sl) { enc.setBytes($0.baseAddress!, length: $0.count, index: seqLenIndex) }
                }
                var grid = step.gridSize
                if step.sequenceLengthBindingIndex != nil && grid.height > 1 {
                    grid = MTLSize(width: grid.width, height: seqLen, depth: grid.depth)
                }
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: step.threadgroupSize)

            case .perPosition:
                for pos in 0..<seqLen {
                    enc.setComputePipelineState(step.pipeline)
                    for (index, buffer, baseOffset) in step.bufferBindings {
                        let stride = step.perPositionStrides[index] ?? 0
                        enc.setBuffer(buffer, offset: baseOffset + pos * stride, index: index)
                    }
                    for (index, value) in step.bytesBindings {
                        value.withUnsafeBufferPointer { ptr in
                            enc.setBytes(ptr.baseAddress!, length: ptr.count, index: index)
                        }
                    }
                    if let posIndex = step.positionBufferIndex {
                        var posValue = UInt32(pos)
                        withUnsafeBytes(of: &posValue) { enc.setBytes($0.baseAddress!, length: $0.count, index: posIndex) }
                    }
                    enc.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
                }

            case .lastToken:
                enc.setComputePipelineState(step.pipeline)
                for (index, buffer, baseOffset) in step.bufferBindings {
                    let stride = step.perPositionStrides[index] ?? 0
                    enc.setBuffer(buffer, offset: baseOffset + (seqLen - 1) * stride, index: index)
                }
                for (index, value) in step.bytesBindings {
                    value.withUnsafeBufferPointer { ptr in
                        enc.setBytes(ptr.baseAddress!, length: ptr.count, index: index)
                    }
                }
                enc.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
            }

            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()

            if let error = cb.error {
                print("[Real diag] GPU ERROR at step \(stepIndex): \(error)")
                Issue.record("GPU error at step \(stepIndex): \(error)")
                return
            }

            // Check hidden at position 0
            let hiddenPtr = prefillPlan.buffers.hidden.contents().bindMemory(to: Float16.self, capacity: hiddenSize)
            let hasNaN = (0..<hiddenSize).contains { hiddenPtr[$0].isNaN }
            let sample = (0..<min(4, hiddenSize)).map { Float(hiddenPtr[$0]) }

            // Check scratch at position 0
            let scratchPtr = prefillPlan.buffers.scratch.contents().bindMemory(to: Float16.self, capacity: hiddenSize)
            let scratchNaN = (0..<hiddenSize).contains { scratchPtr[$0].isNaN }

            let modeStr: String
            switch step.mode {
            case .batch: modeStr = "batch"
            case .perPosition: modeStr = "perPos"
            case .lastToken: modeStr = "lastTok"
            }

            // Log every step for first NaN search
            print("[Real diag] step \(stepIndex) (\(modeStr)): hidden[0..3]=\(sample) hasNaN=\(hasNaN) scratchNaN=\(scratchNaN)")

            if hasNaN && nanFoundAtStep < 0 {
                nanFoundAtStep = stepIndex
                print("[Real diag] *** FIRST NaN at step \(stepIndex) (\(modeStr)) ***")

                // Dump buffer bindings for this step
                for (index, buffer, offset) in step.bufferBindings {
                    print("[Real diag]   binding[\(index)]: offset=\(offset) bufferLen=\(buffer.length)")
                }

                // Check if NaN is in scratch too
                let scratchSample = (0..<min(8, hiddenSize)).map { Float(scratchPtr[$0]) }
                print("[Real diag]   scratch[0..7]=\(scratchSample)")
            }
        }

        if nanFoundAtStep >= 0 {
            Issue.record("NaN first appeared at step \(nanFoundAtStep)")
        } else {
            print("[Real diag] SUCCESS: No NaN in any step at real scale!")
        }
    }
}

// MARK: - Real-Scale Weight Generation

private func createRealScaleWeights(
    hiddenSize: Int, intermediateSize: Int, vocabSize: Int,
    attentionHeads: Int, kvHeads: Int, headDim: Int,
    convKernelSize: Int, to directory: URL
) throws {
    var tensors: [TestTensor] = []

    func bf16Tensor(name: String, shape: [Int]) {
        let count = shape.reduce(1, *)
        var data = [UInt16](repeating: 0, count: count)
        // Small values to avoid overflow: ~0.01 scale
        for i in 0..<count {
            let value = Float(((i * 7 + 13) % 200)) * 0.0001 - 0.01
            data[i] = UInt16(value.bitPattern >> 16)
        }
        tensors.append(TestTensor(
            name: name, dtype: "BF16", shape: shape,
            data: Data(bytes: &data, count: count * 2)))
    }

    // Embedding
    bf16Tensor(name: "model.embed_tokens.weight", shape: [vocabSize, hiddenSize])
    bf16Tensor(name: "model.embedding_norm.weight", shape: [hiddenSize])

    // Layer 0: Conv
    let p0 = "model.layers.0"
    bf16Tensor(name: "\(p0).operator_norm.weight", shape: [hiddenSize])
    bf16Tensor(name: "\(p0).conv.in_proj.weight", shape: [hiddenSize * 3, hiddenSize])
    bf16Tensor(name: "\(p0).conv.conv.weight", shape: [hiddenSize, convKernelSize])
    bf16Tensor(name: "\(p0).conv.out_proj.weight", shape: [hiddenSize, hiddenSize])
    bf16Tensor(name: "\(p0).ffn_norm.weight", shape: [hiddenSize])
    bf16Tensor(name: "\(p0).feed_forward.w1.weight", shape: [intermediateSize, hiddenSize])
    bf16Tensor(name: "\(p0).feed_forward.w3.weight", shape: [intermediateSize, hiddenSize])
    bf16Tensor(name: "\(p0).feed_forward.w2.weight", shape: [hiddenSize, intermediateSize])

    // Layer 1: Attention
    let p1 = "model.layers.1"
    bf16Tensor(name: "\(p1).operator_norm.weight", shape: [hiddenSize])
    bf16Tensor(name: "\(p1).self_attn.q_proj.weight", shape: [attentionHeads * headDim, hiddenSize])
    bf16Tensor(name: "\(p1).self_attn.k_proj.weight", shape: [kvHeads * headDim, hiddenSize])
    bf16Tensor(name: "\(p1).self_attn.v_proj.weight", shape: [kvHeads * headDim, hiddenSize])
    bf16Tensor(name: "\(p1).self_attn.out_proj.weight", shape: [hiddenSize, attentionHeads * headDim])
    bf16Tensor(name: "\(p1).self_attn.q_layernorm.weight", shape: [headDim])
    bf16Tensor(name: "\(p1).self_attn.k_layernorm.weight", shape: [headDim])
    bf16Tensor(name: "\(p1).ffn_norm.weight", shape: [hiddenSize])
    bf16Tensor(name: "\(p1).feed_forward.w1.weight", shape: [intermediateSize, hiddenSize])
    bf16Tensor(name: "\(p1).feed_forward.w3.weight", shape: [intermediateSize, hiddenSize])
    bf16Tensor(name: "\(p1).feed_forward.w2.weight", shape: [hiddenSize, intermediateSize])

    // LM head
    bf16Tensor(name: "lm_head.weight", shape: [vocabSize, hiddenSize])

    let safetensorsURL = directory.appendingPathComponent("model.safetensors")
    try writeSafetensors(tensors: tensors, to: safetensorsURL)

    let stafURL = directory.appendingPathComponent("model.staf")
    try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)
}

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
