import Testing
import Metal
import Foundation
@testable import MetalCompiler
import LMArchitecture
import ModelDeclarations

/// Step-by-step NaN diagnostic for LFM2-like BF16 model prefill.
/// Runs each prefill step individually and checks for NaN after each one.
@Suite("Prefill NaN Diagnostic")
struct PrefillNaNDiagnosticTests {

    @Test("Step-by-step prefill identifies NaN source",
          .disabled("Pre-migration diagnostic: accesses storageModePrivate scratch buffer via contents()"))
    func stepByStepNaNDiagnostic() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        // Build a minimal model matching LFM2 structure:
        // TokenEmbedding → RMSNorm (embedding_norm) → Residual { RMSNorm + MLP } → RMSNorm → OutputHead
        let hiddenSize = 64
        let intermediateSize = 128
        let vocabSize = 100
        let seqLen = 4

        // Create BF16 STAF weights
        let tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("nan_diag_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDirectory) }

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try createBF16Weights(
            hiddenSize: hiddenSize, intermediateSize: intermediateSize,
            vocabSize: vocabSize, to: tempDirectory)

        let store = try STAFLoader().load(at: stafURL, device: device)
        print("[NaN diag] STAF loaded: \(store.entries.count) tensors")
        for (name, entry) in store.entries.sorted(by: { $0.key < $1.key }) {
            print("[NaN diag]   \(name): scheme=\(entry.schemeIdentifier) size=\(entry.payloadSize) offset=\(entry.bufferOffset)")
        }

        // Build model graph (same structure as LFM2 but simplified — no conv, just MLP)
        let config = ModelConfig(
            hiddenSize: hiddenSize, layerCount: 1, intermediateSize: intermediateSize,
            vocabSize: vocabSize, attentionHeads: 4, kvHeads: 4, headDim: 16,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 10000, ropeDimension: 16,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: false,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: nil,
            partialRotaryFactor: nil, slidingWindow: nil
        )
        let graph = try ModelGraph(Transformer(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .llamaFamily)

        // Compile prefill plan
        let compiler = MetalInferenceCompiler()
        let prefillPlan = try compiler.compilePrefill(
            graph: resolved,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 32),
            stafWeightStore: store,
            device: device)

        print("[NaN diag] Prefill plan: \(prefillPlan.stepCount) steps")

        // Fill token IDs and positions
        let tokenPtr = prefillPlan.buffers.tokenIDs.contents().bindMemory(to: Int32.self, capacity: seqLen)
        let posPtr = prefillPlan.buffers.positions.contents().bindMemory(to: UInt32.self, capacity: seqLen)
        for i in 0..<seqLen {
            tokenPtr[i] = Int32(i + 1)  // tokens 1,2,3,4
            posPtr[i] = UInt32(i)
        }

        // Run each step one at a time and check for NaN
        let elementSize = MemoryLayout<Float16>.size
        var nanFoundAtStep = -1

        for stepIndex in 0..<prefillPlan.steps.count {
            let step = prefillPlan.steps[stepIndex]

            guard let queue = device.makeCommandQueue(),
                  let cb = queue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else {
                Issue.record("Cannot create encoder at step \(stepIndex)")
                return
            }

            switch step.mode {
            case .batch:
                // Dispatch batch step with actual seqLen
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(step.pipeline)
                for (index, buffer, offset) in step.bufferBindings {
                    enc.setBuffer(buffer, offset: offset, index: index)
                }
                for (index, value) in step.bytesBindings {
                    value.withUnsafeBufferPointer { ptr in
                        enc.setBytes(ptr.baseAddress!, length: ptr.count, index: index)
                    }
                }
                if let bindingIndex = step.sequenceLengthPolicy.bindingIndex {
                    var seqLenValue = UInt32(seqLen)
                    withUnsafeBytes(of: &seqLenValue) { raw in
                        enc.setBytes(raw.baseAddress!, length: raw.count, index: bindingIndex)
                    }
                }
                let grid = step.resolvedGridSize(sequenceLength: seqLen)
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: step.threadgroupSize)

            case .perPosition:
                // Run for all positions
                for pos in 0..<seqLen {
                    enc.memoryBarrier(scope: .buffers)
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
                        withUnsafeBytes(of: &posValue) { raw in
                            enc.setBytes(raw.baseAddress!, length: raw.count, index: posIndex)
                        }
                    }
                    enc.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
                }

            case .lastToken:
                enc.memoryBarrier(scope: .buffers)
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
                print("[NaN diag] GPU error at step \(stepIndex): \(error)")
                Issue.record("GPU error at step \(stepIndex)")
                return
            }

            // Check hidden buffer for NaN at position 0
            let hiddenPtr = prefillPlan.buffers.hidden.contents().bindMemory(
                to: Float.self, capacity: hiddenSize)
            let hasNaN = (0..<hiddenSize).contains { hiddenPtr[$0].isNaN }
            let hasFinite = (0..<hiddenSize).contains { hiddenPtr[$0].isFinite && hiddenPtr[$0] != 0 }
            let sample = (0..<min(4, hiddenSize)).map { Float(hiddenPtr[$0]) }

            // Also check scratch
            let scratchPtr = prefillPlan.buffers.scratch.contents().bindMemory(
                to: Float.self, capacity: hiddenSize)
            let scratchHasNaN = (0..<hiddenSize).contains { scratchPtr[$0].isNaN }
            let scratchSample = (0..<min(4, hiddenSize)).map { Float(scratchPtr[$0]) }

            let modeStr: String
            switch step.mode {
            case .batch: modeStr = "batch"
            case .perPosition: modeStr = "perPos"
            case .lastToken: modeStr = "lastTok"
            }

            print("[NaN diag] step \(stepIndex) (\(modeStr)): hidden[0..3]=\(sample) hasNaN=\(hasNaN) hasFinite=\(hasFinite) scratch[0..3]=\(scratchSample) scratchNaN=\(scratchHasNaN)")

            if hasNaN && nanFoundAtStep < 0 {
                nanFoundAtStep = stepIndex
                print("[NaN diag] *** FIRST NaN at step \(stepIndex) ***")
                // Don't break — continue to see if it was already NaN from input
            }
        }

        if nanFoundAtStep >= 0 {
            Issue.record("NaN first appeared at step \(nanFoundAtStep)")
        } else {
            print("[NaN diag] No NaN found in any step — prefill is correct!")
        }
    }
    @Test("Step-by-step prefill with ShortConv layer (LFM2-like)",
          .disabled("Pre-migration diagnostic: accesses storageModePrivate scratch buffer via contents()"))
    func stepByStepWithShortConv() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        // LFM2-like: TokenEmbedding → RMSNorm → Residual { RMSNorm + ShortConv } → Residual { RMSNorm + MLP } → RMSNorm → OutputHead
        let hiddenSize = 64
        let intermediateSize = 128
        let vocabSize = 100
        let seqLen = 4
        let convKernelSize = 3

        let tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("nan_conv_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDirectory) }

        // Create BF16 weights including conv weights
        try createBF16WeightsWithConv(
            hiddenSize: hiddenSize, intermediateSize: intermediateSize,
            vocabSize: vocabSize, convKernelSize: convKernelSize,
            to: tempDirectory)

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        let store = try STAFLoader().load(at: stafURL, device: device)
        print("[Conv diag] STAF: \(store.entries.count) tensors")

        // Build LFM2-like model graph using LMArchitecture components
        let config = ModelConfig(
            hiddenSize: hiddenSize, layerCount: 1, intermediateSize: intermediateSize,
            vocabSize: vocabSize, attentionHeads: 4, kvHeads: 4, headDim: 16,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 10000, ropeDimension: 16,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: false,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: convKernelSize, convLCache: convKernelSize,
            partialRotaryFactor: nil, slidingWindow: nil,
            layerTypes: ["conv"]
        )
        let graph = try ModelGraph(LFM2(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .lfm2Family)

        // Compile prefill plan
        let prefillPlan = try MetalInferenceCompiler().compilePrefill(
            graph: resolved,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 32),
            stafWeightStore: store,
            device: device)

        print("[Conv diag] Prefill plan: \(prefillPlan.stepCount) steps")

        // Fill tokens
        let tokenPtr = prefillPlan.buffers.tokenIDs.contents().bindMemory(to: Int32.self, capacity: seqLen)
        let posPtr = prefillPlan.buffers.positions.contents().bindMemory(to: UInt32.self, capacity: seqLen)
        for i in 0..<seqLen {
            tokenPtr[i] = Int32(i + 1)
            posPtr[i] = UInt32(i)
        }

        // Run step by step
        var nanFoundAtStep = -1
        for stepIndex in 0..<prefillPlan.steps.count {
            let step = prefillPlan.steps[stepIndex]

            guard let queue = device.makeCommandQueue(),
                  let cb = queue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else {
                Issue.record("Cannot create encoder")
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
                if let bindingIndex = step.sequenceLengthPolicy.bindingIndex {
                    var seqLenValue = UInt32(seqLen)
                    withUnsafeBytes(of: &seqLenValue) { raw in
                        enc.setBytes(raw.baseAddress!, length: raw.count, index: bindingIndex)
                    }
                }
                let grid = step.resolvedGridSize(sequenceLength: seqLen)
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
                print("[Conv diag] GPU ERROR at step \(stepIndex): \(error)")
                Issue.record("GPU error at step \(stepIndex)")
                return
            }

            let hiddenPtr = prefillPlan.buffers.hidden.contents().bindMemory(to: Float.self, capacity: hiddenSize)
            let hasNaN = (0..<hiddenSize).contains { hiddenPtr[$0].isNaN }
            let sample = (0..<min(4, hiddenSize)).map { Float(hiddenPtr[$0]) }

            let modeStr: String
            switch step.mode {
            case .batch: modeStr = "batch"
            case .perPosition: modeStr = "perPos"
            case .lastToken: modeStr = "lastTok"
            }
            print("[Conv diag] step \(stepIndex) (\(modeStr)): hidden[0..3]=\(sample) hasNaN=\(hasNaN)")

            if hasNaN && nanFoundAtStep < 0 {
                nanFoundAtStep = stepIndex
                print("[Conv diag] *** FIRST NaN at step \(stepIndex) ***")
            }
        }

        if nanFoundAtStep >= 0 {
            Issue.record("NaN first appeared at step \(nanFoundAtStep)")
        }
    }
}

// MARK: - BF16 Weight GenerationEvent

/// Create a safetensors file with BF16 weights matching Transformer (llama-family) naming.
private func createBF16Weights(
    hiddenSize: Int, intermediateSize: Int, vocabSize: Int,
    to directory: URL
) throws {
    var tensors: [TestTensor] = []

    // Helper: create BF16 tensor with small random-ish values
    func bf16Tensor(name: String, shape: [Int]) {
        let count = shape.reduce(1, *)
        var data = [BFloat16](repeating: .zero, count: count)
        for i in 0..<count {
            // Small values around 0.01 to avoid overflow
            let value = Float(((i * 7 + 13) % 100)) * 0.001 - 0.05
            data[i] = BFloat16(value)
        }
        tensors.append(TestTensor(
            name: name, dtype: "BF16", shape: shape,
            data: Data(bytes: &data, count: count * MemoryLayout<BFloat16>.size)))
    }

    // Embedding
    bf16Tensor(name: "model.embed_tokens.weight", shape: [vocabSize, hiddenSize])

    // Layer 0
    let prefix = "model.layers.0"
    bf16Tensor(name: "\(prefix).input_layernorm.weight", shape: [hiddenSize])
    bf16Tensor(name: "\(prefix).self_attn.q_proj.weight", shape: [hiddenSize, hiddenSize])
    bf16Tensor(name: "\(prefix).self_attn.k_proj.weight", shape: [hiddenSize / 4, hiddenSize])
    bf16Tensor(name: "\(prefix).self_attn.v_proj.weight", shape: [hiddenSize / 4, hiddenSize])
    bf16Tensor(name: "\(prefix).self_attn.o_proj.weight", shape: [hiddenSize, hiddenSize])
    bf16Tensor(name: "\(prefix).post_attention_layernorm.weight", shape: [hiddenSize])
    bf16Tensor(name: "\(prefix).mlp.gate_proj.weight", shape: [intermediateSize, hiddenSize])
    bf16Tensor(name: "\(prefix).mlp.up_proj.weight", shape: [intermediateSize, hiddenSize])
    bf16Tensor(name: "\(prefix).mlp.down_proj.weight", shape: [hiddenSize, intermediateSize])

    // Final norm (reuse embedding for tied weights)
    bf16Tensor(name: "model.norm.weight", shape: [hiddenSize])

    let safetensorsURL = directory.appendingPathComponent("model.safetensors")
    try writeSafetensors(tensors: tensors, to: safetensorsURL)

    let stafURL = directory.appendingPathComponent("model.staf")
    try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)
}

/// Create BF16 weights for LFM2-like model with ShortConv + MLP.
private func createBF16WeightsWithConv(
    hiddenSize: Int, intermediateSize: Int, vocabSize: Int,
    convKernelSize: Int, to directory: URL
) throws {
    var tensors: [TestTensor] = []

    func bf16Tensor(name: String, shape: [Int]) {
        let count = shape.reduce(1, *)
        var data = [BFloat16](repeating: .zero, count: count)
        for i in 0..<count {
            let value = Float(((i * 7 + 13) % 100)) * 0.001 - 0.05
            data[i] = BFloat16(value)
        }
        tensors.append(TestTensor(
            name: name, dtype: "BF16", shape: shape,
            data: Data(bytes: &data, count: count * MemoryLayout<BFloat16>.size)))
    }

    bf16Tensor(name: "model.embed_tokens.weight", shape: [vocabSize, hiddenSize])
    bf16Tensor(name: "model.embedding_norm.weight", shape: [hiddenSize])

    let prefix = "model.layers.0"
    // Conv layer weights
    bf16Tensor(name: "\(prefix).operator_norm.weight", shape: [hiddenSize])
    bf16Tensor(name: "\(prefix).conv.in_proj.weight", shape: [hiddenSize * 3, hiddenSize])
    bf16Tensor(name: "\(prefix).conv.conv.weight", shape: [hiddenSize, convKernelSize])
    bf16Tensor(name: "\(prefix).conv.out_proj.weight", shape: [hiddenSize, hiddenSize])
    // MLP weights
    bf16Tensor(name: "\(prefix).ffn_norm.weight", shape: [hiddenSize])
    bf16Tensor(name: "\(prefix).feed_forward.w1.weight", shape: [intermediateSize, hiddenSize])
    bf16Tensor(name: "\(prefix).feed_forward.w3.weight", shape: [intermediateSize, hiddenSize])
    bf16Tensor(name: "\(prefix).feed_forward.w2.weight", shape: [hiddenSize, intermediateSize])

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
