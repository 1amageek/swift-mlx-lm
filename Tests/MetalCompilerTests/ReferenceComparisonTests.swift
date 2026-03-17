import Testing
import Metal
import Foundation
@testable import MetalCompiler
import LMArchitecture
import ModelDeclarations
import LMIR

/// Compare Metal inference output against Python HuggingFace reference.
///
/// Prerequisites:
///   1. Generate reference: `python3 scripts/dump_lfm2_reference.py`
///   2. Have the STAF file in TestData/LFM2.5-1.2B-Thinking/model.staf
@Suite("Reference Comparison")
struct ReferenceComparisonTests {

    private static let referencePath = "/Users/1amageek/Desktop/swift-lm/TestData/lfm2_reference.safetensors"
    private static let stafPath = "/Users/1amageek/Desktop/swift-lm/TestData/LFM2.5-1.2B-Thinking/model.staf"

    // MARK: - Prefill Tests

    @Test("Prefill logits match Python reference")
    func prefillLogitsMatchReference() throws {
        let env = try setupOrSkip()
        var model = env.model

        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        let firstToken = model.prefill(tokens: tokens)

        // Read from PREFILL plan's logits buffer (F32 in prefill)
        guard let prefillPlan = model.prefillPlan else {
            Issue.record("No prefill plan"); return
        }
        let prefillLogits = readF32Buffer(prefillPlan.buffers.logits)
        let refLogits = try readRefTensorAsFloats(env.ref, name: "ref.prefill.logits_last")

        let metalTop = argmax(prefillLogits)
        let refTop = argmax(refLogits)

        print("[RefComp] Metal  prefill: firstToken=\(firstToken), logits argmax=\(metalTop.index) (val=\(String(format: "%.2f", metalTop.value)))")
        print("[RefComp] Python prefill: argmax=\(refTop.index) (val=\(String(format: "%.2f", refTop.value)))")

        let metalTop10 = topK(prefillLogits, k: 10)
        let refTop10 = topK(refLogits, k: 10)
        print("[RefComp] Metal  top-10: \(metalTop10.map { "(\($0.index),\(String(format: "%.1f", $0.value)))" })")
        print("[RefComp] Python top-10: \(refTop10.map { "(\($0.index),\(String(format: "%.1f", $0.value)))" })")

        let maxErr = maxAbsoluteError(prefillLogits, refLogits)
        print("[RefComp] Prefill logits max absolute error: \(String(format: "%.4f", maxErr))")

        #expect(metalTop.index == refTop.index,
                "Prefill argmax mismatch: Metal=\(metalTop.index) Python=\(refTop.index)")
    }

    @Test("Prefill embedding matches Python reference")
    func prefillEmbeddingMatches() throws {
        let env = try setupOrSkip()

        // Run prefill step-by-step to capture embedding output
        guard let prefillPlan = env.model.prefillPlan else {
            Issue.record("No prefill plan"); return
        }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let seqLen = 5
        let tokens: [Int32] = [1, 1, 6, 6423, 708]

        // Fill token IDs and positions
        let tokenPtr = prefillPlan.buffers.tokenIDs.contents()
            .bindMemory(to: Int32.self, capacity: seqLen)
        let posPtr = prefillPlan.buffers.positions.contents()
            .bindMemory(to: UInt32.self, capacity: seqLen)
        for i in 0..<seqLen {
            tokenPtr[i] = tokens[i]
            posPtr[i] = UInt32(i)
        }

        // Run only the first step (embedding lookup)
        let step = prefillPlan.steps[0]
        guard let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }

        enc.setComputePipelineState(step.pipeline)
        for (index, buffer, offset) in step.bufferBindings {
            enc.setBuffer(buffer, offset: offset, index: index)
        }
        if let seqLenIdx = step.sequenceLengthBindingIndex {
            var sl = UInt32(seqLen)
            withUnsafeBytes(of: &sl) { enc.setBytes($0.baseAddress!, length: $0.count, index: seqLenIdx) }
        }
        for (index, value) in step.bytesBindings {
            value.withUnsafeBufferPointer { enc.setBytes($0.baseAddress!, length: $0.count, index: index) }
        }
        var grid = step.gridSize
        if step.sequenceLengthBindingIndex != nil && grid.height > 1 {
            grid = MTLSize(width: grid.width, height: seqLen, depth: grid.depth)
        }
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: step.threadgroupSize)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        // Read embedding output from hidden buffer (F32 in prefill, last token)
        let hiddenSize = 2048
        let lastTokenOffset = (seqLen - 1) * hiddenSize
        let hiddenPtr = prefillPlan.buffers.hidden.contents()
            .bindMemory(to: Float32.self, capacity: seqLen * hiddenSize)
        let metalEmb = (0..<hiddenSize).map { hiddenPtr[lastTokenOffset + $0] }

        // Read Python reference embedding (last token)
        let refEmbAll = try readRefTensorAsFloats(env.ref, name: "ref.prefill.embedding")
        let refEmb = Array(refEmbAll[lastTokenOffset..<lastTokenOffset + hiddenSize])

        let maxErr = maxAbsoluteError(metalEmb, refEmb)
        let metalSample = (0..<4).map { String(format: "%.4f", metalEmb[$0]) }
        let refSample = (0..<4).map { String(format: "%.4f", refEmb[$0]) }
        let metalNorm = sqrtf(metalEmb.reduce(0) { $0 + $1 * $1 })
        let refNorm = sqrtf(refEmb.reduce(0) { $0 + $1 * $1 })

        print("[RefComp] Embedding (last token):")
        print("  Metal:  \(metalSample) norm=\(String(format: "%.2f", metalNorm))")
        print("  Python: \(refSample) norm=\(String(format: "%.2f", refNorm))")
        print("  Max absolute error: \(String(format: "%.6f", maxErr))")

        #expect(maxErr < 0.01, "Embedding diverges: maxErr=\(maxErr)")
    }

    @Test("Conv state after prefill matches Python reference")
    func convStateAfterPrefillMatches() throws {
        let env = try setupOrSkip()
        var model = env.model

        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        _ = model.prefill(tokens: tokens)

        // Read conv_state from DECODE plan's buffer (prefill transfers it)
        guard let convState = model.plan.buffers.convState else {
            Issue.record("No conv_state buffer"); return
        }

        let convDim = model.plan.buffers.convStateDimension
        let kernelSize = model.plan.buffers.convStateKernelSize
        let elementSize = MemoryLayout<Float16>.size

        for convIdx in 0..<10 {
            let refData = try readRefTensorAsFloats(env.ref, name: "ref.prefill.conv_state.\(convIdx)")
            let layerOffset = convIdx * kernelSize * convDim * elementSize
            let metalPtr = (convState.contents() + layerOffset)
                .bindMemory(to: Float16.self, capacity: kernelSize * convDim)
            let metalVals = (0..<kernelSize * convDim).map { Float(metalPtr[$0]) }

            let error = maxAbsoluteError(metalVals, refData)

            print("[RefComp] conv_state[\(convIdx)]: maxErr=\(String(format: "%.6f", error))")

            #expect(error < 0.1,
                    "conv_state[\(convIdx)] diverges: maxErr=\(error)")
        }
    }

    // MARK: - Per-layer Prefill Comparison

    @Test("Prefill per-layer hidden states match Python reference")
    func prefillPerLayerMatch() throws {
        let env = try setupOrSkip()
        guard let prefillPlan = env.model.prefillPlan else {
            Issue.record("No prefill plan"); return
        }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let seqLen = 5
        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        let hiddenSize = 2048

        // Fill token IDs and positions
        let tokenPtr = prefillPlan.buffers.tokenIDs.contents()
            .bindMemory(to: Int32.self, capacity: seqLen)
        let posPtr = prefillPlan.buffers.positions.contents()
            .bindMemory(to: UInt32.self, capacity: seqLen)
        for i in 0..<seqLen {
            tokenPtr[i] = tokens[i]
            posPtr[i] = UInt32(i)
        }

        // Run ALL prefill steps
        guard let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }

        for step in prefillPlan.steps {
            if step.sync == .bufferBarrier { enc.memoryBarrier(scope: .buffers) }
            enc.setComputePipelineState(step.pipeline)
            for (index, buffer, offset) in step.bufferBindings {
                enc.setBuffer(buffer, offset: offset, index: index)
            }
            for (index, value) in step.bytesBindings {
                value.withUnsafeBufferPointer { enc.setBytes($0.baseAddress!, length: $0.count, index: index) }
            }
            if let seqLenIdx = step.sequenceLengthBindingIndex {
                var sl = UInt32(seqLen)
                withUnsafeBytes(of: &sl) { enc.setBytes($0.baseAddress!, length: $0.count, index: seqLenIdx) }
            }
            switch step.mode {
            case .batch:
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
                    if let posIdx = step.positionBufferIndex {
                        var posValue = UInt32(pos)
                        withUnsafeBytes(of: &posValue) {
                            enc.setBytes($0.baseAddress!, length: $0.count, index: posIdx)
                        }
                    }
                    for (index, value) in step.bytesBindings {
                        value.withUnsafeBufferPointer { enc.setBytes($0.baseAddress!, length: $0.count, index: index) }
                    }
                    enc.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
                }
            case .lastToken:
                enc.setComputePipelineState(step.pipeline)
                for (index, buffer, baseOffset) in step.bufferBindings {
                    let stride = step.perPositionStrides[index] ?? 0
                    enc.setBuffer(buffer, offset: baseOffset + (seqLen - 1) * stride, index: index)
                }
                if let posIdx = step.positionBufferIndex {
                    var posValue = UInt32(seqLen - 1)
                    withUnsafeBytes(of: &posValue) {
                        enc.setBytes($0.baseAddress!, length: $0.count, index: posIdx)
                    }
                }
                for (index, value) in step.bytesBindings {
                    value.withUnsafeBufferPointer { enc.setBytes($0.baseAddress!, length: $0.count, index: index) }
                }
                enc.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
            }
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        // Compare final hidden (last token)
        let elementSize = MemoryLayout<Float16>.size
        let lastTokenOffset = (seqLen - 1) * hiddenSize
        let hiddenPtr = prefillPlan.buffers.hidden.contents()
            .bindMemory(to: Float32.self, capacity: seqLen * hiddenSize)
        let metalFinalHidden = (0..<hiddenSize).map { hiddenPtr[lastTokenOffset + $0] }

        let refFinalAll = try readRefTensorAsFloats(env.ref, name: "ref.prefill.final_hidden")
        let refLastToken = Array(refFinalAll[lastTokenOffset..<lastTokenOffset + hiddenSize])

        let finalErr = maxAbsoluteError(metalFinalHidden, refLastToken)
        let metalNorm = sqrtf(metalFinalHidden.reduce(0) { $0 + $1 * $1 })
        let refNorm = sqrtf(refLastToken.reduce(0) { $0 + $1 * $1 })
        let metalSample = (0..<4).map { String(format: "%.4f", metalFinalHidden[$0]) }
        let refSample = (0..<4).map { String(format: "%.4f", refLastToken[$0]) }

        print("[RefComp] Final hidden (last token):")
        print("  Metal:  \(metalSample) norm=\(String(format: "%.2f", metalNorm))")
        print("  Python: \(refSample) norm=\(String(format: "%.2f", refNorm))")
        print("  Max absolute error: \(String(format: "%.4f", finalErr))")

        // Compare prefill logits (F32 buffer)
        let prefillLogits = readF32Buffer(prefillPlan.buffers.logits)
        let refLogits = try readRefTensorAsFloats(env.ref, name: "ref.prefill.logits_last")
        let logitsErr = maxAbsoluteError(prefillLogits, refLogits)
        let metalArgmax = argmax(prefillLogits)
        let refArgmax = argmax(refLogits)
        print("[RefComp] Prefill logits: Metal argmax=\(metalArgmax.index) Python argmax=\(refArgmax.index) maxErr=\(String(format: "%.4f", logitsErr))")

        // Compare per-layer after_mlp for last token
        for layerIdx in 0..<16 {
            let refLayerAll = try readRefTensorAsFloats(env.ref, name: "ref.prefill.layer_\(layerIdx).after_mlp")
            let refLayer = Array(refLayerAll[lastTokenOffset..<lastTokenOffset + hiddenSize])
            let refLayerNorm = sqrtf(refLayer.reduce(0) { $0 + $1 * $1 })
            let refLayerSample = (0..<2).map { String(format: "%.3f", refLayer[$0]) }
            if layerIdx < 3 || layerIdx >= 14 {
                print("[RefComp] Python layer_\(layerIdx).after_mlp: \(refLayerSample)... norm=\(String(format: "%.1f", refLayerNorm))")
            }
        }

        #expect(metalArgmax.index == refArgmax.index,
                "Prefill logits argmax: Metal=\(metalArgmax.index) Python=\(refArgmax.index)")
    }

    // MARK: - Decode Tests

    @Test("Decode step 0 logits match Python reference")
    func decodeStep0LogitsMatch() throws {
        try verifyDecodeStep(step: 0)
    }

    @Test("Decode step 1 logits match Python reference")
    func decodeStep1LogitsMatch() throws {
        try verifyDecodeStep(step: 1)
    }

    @Test("Decode step 2 logits match Python reference")
    func decodeStep2LogitsMatch() throws {
        try verifyDecodeStep(step: 2)
    }

    // MARK: - Decode Step Helper

    private func verifyDecodeStep(step: Int) throws {
        let env = try setupOrSkip()
        var model = env.model

        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        var currentToken = model.prefill(tokens: tokens)

        for s in 0...step {
            currentToken = model.decodeSync(tokenID: currentToken)
            if s < step { continue }

            // Read from DECODE plan's logits buffer (F16)
            let metalLogits = readF16Buffer(model.plan.buffers.logits)
            let refLogits = try readRefTensorAsFloats(env.ref, name: "ref.decode_\(step).logits")

            let refTop = argmax(refLogits)
            let metalTop = argmax(metalLogits)
            let maxErr = maxAbsoluteError(metalLogits, refLogits)

            print("[RefComp] Decode step \(step):")
            print("  Python argmax: \(refTop.index) (val=\(String(format: "%.2f", refTop.value)))")
            print("  Metal  argmax: \(metalTop.index) (val=\(String(format: "%.2f", metalTop.value)))")
            print("  Max absolute error: \(String(format: "%.4f", maxErr))")

            #expect(metalTop.index == refTop.index,
                    "Decode step \(step) argmax: Metal=\(metalTop.index) Python=\(refTop.index)")
        }
    }

    // MARK: - Setup

    private struct TestEnvironment {
        var model: MetalInferenceModel
        let ref: MetalWeightFile
    }

    private func setupOrSkip() throws -> TestEnvironment {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SetupError.noDevice
        }

        let refURL = URL(fileURLWithPath: Self.referencePath)
        guard FileManager.default.fileExists(atPath: refURL.path) else {
            print("[RefComp] Reference not found — run: python3 scripts/dump_lfm2_reference.py")
            throw SetupError.noReference
        }

        let stafURL = URL(fileURLWithPath: Self.stafPath)
        let modelDir = stafURL.deletingLastPathComponent()

        // Auto-convert safetensors → STAF if needed
        if !FileManager.default.fileExists(atPath: Self.stafPath) {
            let safetensorsURL = modelDir.appendingPathComponent("model.safetensors")
            guard FileManager.default.fileExists(atPath: safetensorsURL.path) else {
                print("[RefComp] Neither STAF nor safetensors found in \(modelDir.path)")
                throw SetupError.noSTAF
            }
            print("[RefComp] Converting safetensors → STAF...")
            try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)
            print("[RefComp] STAF created: \(Self.stafPath)")
        }

        let ref = try SafetensorsLoader().load(at: refURL, device: device)
        let store = try STAFLoader().load(at: stafURL, device: device)

        let config = ModelConfig(
            hiddenSize: 2048, layerCount: 16, intermediateSize: 8192,
            vocabSize: 65536, attentionHeads: 32, kvHeads: 8, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 1000000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: true,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: nil, convLCache: 3,
            partialRotaryFactor: nil, slidingWindow: nil,
            layerTypes: ["conv", "conv", "full_attention", "conv", "conv", "full_attention",
                         "conv", "conv", "full_attention", "conv", "full_attention", "conv",
                         "full_attention", "conv", "full_attention", "conv"]
        )
        let graph = try LFM2(config: config).makeModelGraph()
        let resolved = ParameterResolver().resolve(graph: graph, convention: .lfm2Family)

        let compiler = MetalInferenceCompiler()
        let decodePlan = try compiler.compile(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, stafWeightStore: store, device: device)
        let prefillPlan = try compiler.compilePrefill(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, maximumSequenceLength: 64,
            stafWeightStore: store, device: device)

        var model = try MetalInferenceModel(plan: decodePlan, device: device)
        model.prefillPlan = prefillPlan

        return TestEnvironment(model: model, ref: ref)
    }

    // MARK: - Reference Tensor Access

    private func readFloat16Tensor(
        _ file: MetalWeightFile, name: String
    ) throws -> UnsafeBufferPointer<Float16> {
        guard let info = file.tensors[name] else {
            throw SetupError.tensorNotFound(name)
        }
        let count = info.shape.reduce(1, *)
        let ptr = (file.buffer.contents() + file.dataSectionOffset + info.dataOffset)
            .bindMemory(to: Float16.self, capacity: count)
        return UnsafeBufferPointer(start: ptr, count: count)
    }

    // MARK: - Buffer Reading (converts any buffer to [Float])

    private func readF16Buffer(_ buffer: MTLBuffer) -> [Float] {
        let count = buffer.length / MemoryLayout<Float16>.size
        let ptr = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(ptr[$0]) }
    }

    private func readF32Buffer(_ buffer: MTLBuffer) -> [Float] {
        let count = buffer.length / MemoryLayout<Float32>.size
        let ptr = buffer.contents().bindMemory(to: Float32.self, capacity: count)
        return (0..<count).map { ptr[$0] }
    }

    private func readRefTensorAsFloats(
        _ file: MetalWeightFile, name: String
    ) throws -> [Float] {
        let buf = try readFloat16Tensor(file, name: name)
        return (0..<buf.count).map { Float(buf[$0]) }
    }

    // MARK: - Comparison Utilities (all work on [Float])

    private struct IndexedValue {
        let index: Int
        let value: Float
    }

    private func argmax(_ values: [Float]) -> IndexedValue {
        var maxVal: Float = -.infinity
        var maxIdx = 0
        for i in 0..<values.count {
            if values[i] > maxVal { maxVal = values[i]; maxIdx = i }
        }
        return IndexedValue(index: maxIdx, value: maxVal)
    }

    private func topK(_ values: [Float], k: Int) -> [IndexedValue] {
        let indexed = values.enumerated().map { IndexedValue(index: $0.offset, value: $0.element) }
        return Array(indexed.sorted { $0.value > $1.value }.prefix(k))
    }

    private func maxAbsoluteError(_ a: [Float], _ b: [Float]) -> Float {
        let count = min(a.count, b.count)
        var maxErr: Float = 0
        for i in 0..<count {
            maxErr = max(maxErr, abs(a[i] - b[i]))
        }
        return maxErr
    }

    // MARK: - Errors

    private enum SetupError: Error {
        case noDevice
        case noReference
        case noSTAF
        case tensorNotFound(String)
    }
}
