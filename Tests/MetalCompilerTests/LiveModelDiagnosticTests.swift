import Testing
import Metal
import Foundation
@testable import MetalCompiler
@testable import SwiftLM
import LMArchitecture
import ModelDeclarations
import LMIR
import Tokenizers
import Jinja
import OrderedCollections

/// Diagnostic using the ACTUAL model STAF file on disk.
/// Loads real LFM2.5-1.2B-Thinking weights and runs prefill step-by-step.
@Suite("Live Model Diagnostic")
struct LiveModelDiagnosticTests {

    static let stafPath = "/Users/1amageek/Desktop/swift-lm/TestData/LFM2.5-1.2B-Thinking/model.staf"

    @Test("Single command buffer prefill with real STAF (matches app behavior)")
    func liveModelSingleCommandBuffer() throws {
        guard let resources = try RealModelTestSupport.loadOrSkip(skipMessage: "[Live CB] STAF not found — skipping") else {
            return
        }
        defer { resources.release() }

        let device = resources.device
        let store = resources.store
        let config = Self.makeConfig()
        let graph = try LFM2(config: config).makeModelGraph()
        let resolved = ParameterResolver().resolve(graph: graph, convention: .lfm2Family)
        let seqLen = 986

        let prefillPlan = try MetalInferenceCompiler().compilePrefill(
            graph: resolved,
            hiddenSize: 2048, intermediateSize: 5632, vocabSize: 128256,
            inferencePolicy: InferencePolicy(maximumSequenceLength: seqLen),
            stafWeightStore: store, device: device)

        print("[Live CB] plan: \(prefillPlan.stepCount) steps, seqLen=\(seqLen)")

        let tokenPtr = prefillPlan.buffers.tokenIDs.contents().bindMemory(to: Int32.self, capacity: seqLen)
        let posPtr = prefillPlan.buffers.positions.contents().bindMemory(to: UInt32.self, capacity: seqLen)
        for i in 0..<seqLen {
            tokenPtr[i] = Int32(i + 1)
            posPtr[i] = UInt32(i)
        }

        // Run ALL steps in ONE command buffer — same as app
        guard let queue = device.makeCommandQueue(),
              let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            Issue.record("Cannot create encoder")
            return
        }

        for step in prefillPlan.steps {
            switch step.mode {
            case .batch:
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(step.pipeline)
                for (index, buffer, offset) in step.bufferBindings { enc.setBuffer(buffer, offset: offset, index: index) }
                for (index, value) in step.bytesBindings { value.withUnsafeBufferPointer { enc.setBytes($0.baseAddress!, length: $0.count, index: index) } }
                step.bindRuntimeArguments(encoder: enc, sequenceLength: UInt32(seqLen))
                let grid = step.resolvedGridSize(sequenceLength: seqLen)
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: step.threadgroupSize)

            case .perPosition:
                for pos in 0..<seqLen {
                    enc.memoryBarrier(scope: .buffers)
                    enc.setComputePipelineState(step.pipeline)
                    for (index, buffer, base) in step.bufferBindings { enc.setBuffer(buffer, offset: base + pos * (step.perPositionStrides[index] ?? 0), index: index) }
                    for (index, value) in step.bytesBindings { value.withUnsafeBufferPointer { enc.setBytes($0.baseAddress!, length: $0.count, index: index) } }
                    if let pi = step.positionBufferIndex { var pv = UInt32(pos); withUnsafeBytes(of: &pv) { enc.setBytes($0.baseAddress!, length: $0.count, index: pi) } }
                    enc.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
                }

            case .lastToken:
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(step.pipeline)
                for (index, buffer, base) in step.bufferBindings { enc.setBuffer(buffer, offset: base + (seqLen-1) * (step.perPositionStrides[index] ?? 0), index: index) }
                for (index, value) in step.bytesBindings { value.withUnsafeBufferPointer { enc.setBytes($0.baseAddress!, length: $0.count, index: index) } }
                enc.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
            }
        }

        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        if let error = cb.error {
            print("[Live CB] GPU ERROR: \(error)")
            Issue.record("GPU error: \(error)")
            return
        }

        // Check results
        let hp = prefillPlan.buffers.hidden.contents().bindMemory(to: Float.self, capacity: 2048)
        let sample = (0..<4).map { Float(hp[$0]) }
        let hasNaN = (0..<2048).contains { hp[$0].isNaN }
        print("[Live CB] hidden[pos=0][0..3] = \(sample) NaN=\(hasNaN)")

        let lastBase = 985 * 2048
        let hp2 = prefillPlan.buffers.hidden.contents().bindMemory(to: Float.self, capacity: lastBase + 2048)
        let lastSample = (0..<4).map { Float(hp2[lastBase + $0]) }
        let lastNaN = (0..<2048).contains { hp2[lastBase + $0].isNaN }
        print("[Live CB] hidden[pos=985][0..3] = \(lastSample) NaN=\(lastNaN)")

        if hasNaN || lastNaN {
            Issue.record("NaN detected in single command buffer execution")
        } else {
            print("[Live CB] SUCCESS: No NaN in single command buffer!")
        }
    }

    /// Config matching the ACTUAL LFM2.5-1.2B-Thinking config.json on disk.
    static func makeConfig() -> ModelConfig {
        ModelConfig(
            hiddenSize: 2048, layerCount: 16, intermediateSize: 8192,
            vocabSize: 65536, attentionHeads: 32, kvHeads: 8, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 1000000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: true,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: 3, convLCache: 3,
            partialRotaryFactor: nil, slidingWindow: nil,
            layerTypes: ["conv", "conv", "full_attention", "conv", "conv", "full_attention",
                         "conv", "conv", "full_attention", "conv", "full_attention", "conv",
                         "full_attention", "conv", "full_attention", "conv"]
        )
    }

    @Test("Compile prefill plan and dump projection details")
    func dumpProjectionDetails() throws {
        guard let resources = try RealModelTestSupport.loadOrSkip(skipMessage: "[Proj dump] STAF not found") else {
            return
        }
        defer { resources.release() }

        let device = resources.device
        let store = resources.store
        let config = Self.makeConfig()
        let graph = try LFM2(config: config).makeModelGraph()
        let resolved = ParameterResolver().resolve(graph: graph, convention: .lfm2Family)

        // Print the same params the app uses
        print("[Proj dump] hidden=\(config.hiddenSize) intermediate=\(config.intermediateSize) vocab=\(config.vocabSize)")
        let seqLen = 986

        // compilePrefill will print projection details via the logging we added
        let prefillPlan = try MetalInferenceCompiler().compilePrefill(
            graph: resolved,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: seqLen),
            stafWeightStore: store, device: device)

        print("[Proj dump] plan: \(prefillPlan.stepCount) steps, scratch=\(prefillPlan.buffers.scratch.length)")

        // Now run step-by-step and log step index alongside projection info
        let tokenPtr = prefillPlan.buffers.tokenIDs.contents().bindMemory(to: Int32.self, capacity: seqLen)
        let posPtr = prefillPlan.buffers.positions.contents().bindMemory(to: UInt32.self, capacity: seqLen)
        for i in 0..<seqLen { tokenPtr[i] = Int32(i + 1); posPtr[i] = UInt32(i) }

        let hiddenSize = config.hiddenSize
        var nanStep = -1

        for (stepIndex, step) in prefillPlan.steps.enumerated() {
            guard let queue = device.makeCommandQueue(),
                  let cb = queue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else { break }
            enc.memoryBarrier(scope: .buffers)

            switch step.mode {
            case .batch:
                enc.setComputePipelineState(step.pipeline)
                for (i, buf, off) in step.bufferBindings { enc.setBuffer(buf, offset: off, index: i) }
                for (i, val) in step.bytesBindings { val.withUnsafeBufferPointer { enc.setBytes($0.baseAddress!, length: $0.count, index: i) } }
                step.bindRuntimeArguments(encoder: enc, sequenceLength: UInt32(seqLen))
                let g = step.resolvedGridSize(sequenceLength: seqLen)
                enc.dispatchThreadgroups(g, threadsPerThreadgroup: step.threadgroupSize)
            case .perPosition:
                for pos in 0..<seqLen {
                    enc.setComputePipelineState(step.pipeline)
                    for (i, buf, base) in step.bufferBindings { enc.setBuffer(buf, offset: base + pos * (step.perPositionStrides[i] ?? 0), index: i) }
                    for (i, val) in step.bytesBindings { val.withUnsafeBufferPointer { enc.setBytes($0.baseAddress!, length: $0.count, index: i) } }
                    if let pi = step.positionBufferIndex { var pv = UInt32(pos); withUnsafeBytes(of: &pv) { enc.setBytes($0.baseAddress!, length: $0.count, index: pi) } }
                    enc.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
                }
            case .lastToken:
                enc.setComputePipelineState(step.pipeline)
                for (i, buf, base) in step.bufferBindings { enc.setBuffer(buf, offset: base + (seqLen-1) * (step.perPositionStrides[i] ?? 0), index: i) }
                for (i, val) in step.bytesBindings { val.withUnsafeBufferPointer { enc.setBytes($0.baseAddress!, length: $0.count, index: i) } }
                enc.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
            }

            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()

            if let error = cb.error {
                print("[Proj dump] step \(stepIndex) GPU ERROR: \(error)")
                Issue.record("GPU error at step \(stepIndex)")
                return
            }

            let hp = prefillPlan.buffers.hidden.contents().bindMemory(to: Float.self, capacity: hiddenSize)
            let hNaN = (0..<hiddenSize).contains { hp[$0].isNaN }
            let sp = prefillPlan.buffers.scratch.contents().bindMemory(to: Float.self, capacity: hiddenSize)
            let sNaN = (0..<hiddenSize).contains { sp[$0].isNaN }
            let hs = (0..<min(4, hiddenSize)).map { Float(hp[$0]) }
            let m: String
            switch step.mode {
            case .batch:
                m = "B"
            case .perPosition:
                m = "P"
            case .lastToken:
                m = "L"
            }

            if stepIndex < 20 || hNaN || sNaN {
                // Dump bytes bindings to see inputDim/outputDim
                let bytesDesc = step.bytesBindings.map { (idx, val) -> String in
                    if val.count == 4 {
                        let v = val.withUnsafeBufferPointer { $0.baseAddress!.withMemoryRebound(to: UInt32.self, capacity: 1) { $0.pointee } }
                        return "b\(idx)=\(v)"
                    }
                    return "b\(idx)=[\(val.count)B]"
                }.joined(separator: " ")
                let bindsDesc = step.bufferBindings.map { "[\($0.0)]off=\($0.2)" }.joined(separator: " ")
                print("[Proj dump] step \(stepIndex) (\(m)): h=\(hs) hNaN=\(hNaN) sNaN=\(sNaN) | \(bindsDesc) | \(bytesDesc)")
            }

            if (hNaN || sNaN) && nanStep < 0 {
                nanStep = stepIndex
                print("[Proj dump] *** NaN at step \(stepIndex) ***")

                // Dump weight data at this step's weight binding
                for (idx, buf, off) in step.bufferBindings {
                    if buf.length > 100_000_000 {  // STAF weight buffer is large
                        let wp = (buf.contents() + off).bindMemory(to: BFloat16.self, capacity: 8)
                        let wSample = (0..<8).map { String(format: "0x%04x", wp[$0].bitPattern) }
                        let wFloat = (0..<8).map { wp[$0].floatValue }
                        print("[Proj dump] weight bind[\(idx)] raw=\(wSample) asF16=\(wFloat)")
                        // Check for NaN in weights
                        let wSize = min(buf.length - off, 50_000_000) / 2
                        let wAll = (buf.contents() + off).bindMemory(to: Float16.self, capacity: wSize)
                        let wNaN = (0..<wSize).contains { wAll[$0].isNaN }
                        let wInf = (0..<wSize).contains { wAll[$0].isInfinite }
                        print("[Proj dump] weight has NaN=\(wNaN) Inf=\(wInf) (checked \(wSize) elements)")
                    }
                }

                // Dump input (scratch[0]) values at a few positions
                let scratchCheck = prefillPlan.buffers.scratch.contents().bindMemory(to: Float.self, capacity: 12288 * 2)
                let pos0 = (0..<min(8, 12288)).map { scratchCheck[$0] }
                let pos0Max = (0..<12288).map { abs(scratchCheck[$0]) }.max() ?? 0
                print("[Proj dump] scratch[pos0][0..7]=\(pos0) maxAbs=\(pos0Max)")
                let pos1Base = 12288
                let pos1 = (0..<min(8, 12288)).map { scratchCheck[pos1Base + $0] }
                print("[Proj dump] scratch[pos1][0..7]=\(pos1)")
            }
            if nanStep >= 0 && stepIndex >= nanStep + 3 { break }
        }

        if nanStep >= 0 {
            Issue.record("NaN at step \(nanStep)")
        } else {
            print("[Proj dump] SUCCESS: no NaN")
        }
    }

    static let modelDir = "/Users/1amageek/Desktop/swift-lm/TestData/LFM2.5-1.2B-Thinking"

    @Test("Run prefill via MetalInferenceModel with REAL tokens from tokenizer")
    func prefillViaInferenceModel() async throws {
        guard let resources = try RealModelTestSupport.loadOrSkip(skipMessage: "STAF not found") else {
            return
        }
        defer { resources.release() }

        let device = resources.device

        // Load tokenizer from model directory (same as app's ModelBundleLoader)
        let modelDirURL = URL(fileURLWithPath: Self.modelDir)
        let tokenizer = try await AutoTokenizer.from(modelFolder: modelDirURL)

        // Load and apply chat template (same as app)
        let templateURL = URL(fileURLWithPath: Self.modelDir + "/chat_template.jinja")
        let templateString = try String(contentsOf: templateURL, encoding: .utf8)
        let template = try Template(templateString)

        let messages: [[String: String]] = [["role": "user", "content": "hi"]]
        var ordered1 = OrderedCollections.OrderedDictionary<String, Value>()
        ordered1["role"] = .string("user")
        ordered1["content"] = .string("hi")
        let jinjaMessages: [Value] = [.object(ordered1)]
        let context: [String: Value] = [
            "messages": .array(jinjaMessages),
            "add_generation_prompt": .boolean(true),
            "bos_token": .string(tokenizer.bosToken ?? ""),
            "eos_token": .string(tokenizer.eosToken ?? ""),
        ]
        let renderedPrompt = try template.render(context)
        let tokenIDs = tokenizer.encode(text: renderedPrompt)
        let tokens = tokenIDs.map { Int32($0) }

        print("[InfModel] rendered prompt: \(renderedPrompt.count) chars, \(tokens.count) tokens")
        print("[InfModel] tokens[0..<20]=\(Array(tokens.prefix(20)))")
        print("[InfModel] tokens[\(tokens.count-5)..<\(tokens.count)]=\(Array(tokens.suffix(5)))")

        // Use exact app tokens from file (986 tokens from chat template)
        let tokenFileURL = URL(fileURLWithPath: "/tmp/lfm2_test_tokens.txt")
        let realTokens: [Int32]
        if FileManager.default.fileExists(atPath: tokenFileURL.path) {
            let tokenString = try String(contentsOf: tokenFileURL, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
            realTokens = tokenString.split(separator: ",").compactMap { Int32($0) }
            print("[InfModel] loaded \(realTokens.count) real tokens from file")
        } else {
            realTokens = tokens
            print("[InfModel] token file not found, using tokenizer tokens (\(tokens.count))")
        }

        // Compile model (same as app)
        let config = Self.makeConfig()
        let graph = try LFM2(config: config).makeModelGraph()
        let resolved = ParameterResolver().resolve(graph: graph, convention: .lfm2Family)

        let compiler = MetalInferenceCompiler()
        let store = resources.store
        let decodePlan = try compiler.compile(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, stafWeightStore: store, device: device)
        let prefillPlan = try compiler.compilePrefill(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, inferencePolicy: InferencePolicy(maximumSequenceLength: max(1, realTokens.count)),
            stafWeightStore: store, device: device)

        print("[InfModel] decode=\(decodePlan.steps.count) prefill=\(prefillPlan.stepCount) scratch=\(prefillPlan.buffers.scratch.length) hidden=\(prefillPlan.buffers.hidden.length)")

        var model = try MetalInferenceModel(plan: decodePlan, device: device)
        model.prefillPlan = prefillPlan
        model.prefill(tokens: realTokens)

        print("[InfModel] position=\(model.position)")
        print("[InfModel] position=\(model.position)")
        #expect(model.position == realTokens.count, "Prefill should advance position to \(realTokens.count)")
    }

    @Test("Step-by-step prefill with real STAF finds NaN source")
    func liveModelStepByStep() throws {
        guard let resources = try RealModelTestSupport.loadOrSkip(skipMessage: "[Live diag] STAF file not found — skipping") else {
            return
        }
        defer { resources.release() }

        let device = resources.device
        let store = resources.store
        print("[Live diag] STAF loaded: \(store.entries.count) tensors")

        let config = Self.makeConfig()

        let graph = try LFM2(config: config).makeModelGraph()
        let resolved = ParameterResolver().resolve(graph: graph, convention: .lfm2Family)

        let seqLen = 986

        let prefillPlan = try MetalInferenceCompiler().compilePrefill(
            graph: resolved,
            hiddenSize: 2048, intermediateSize: 5632, vocabSize: 128256,
            inferencePolicy: InferencePolicy(maximumSequenceLength: seqLen),
            stafWeightStore: store,
            device: device)

        print("[Live diag] Prefill plan: \(prefillPlan.stepCount) steps")

        // Fill tokens
        let tokenPtr = prefillPlan.buffers.tokenIDs.contents().bindMemory(to: Int32.self, capacity: seqLen)
        let posPtr = prefillPlan.buffers.positions.contents().bindMemory(to: UInt32.self, capacity: seqLen)
        for i in 0..<seqLen {
            tokenPtr[i] = Int32(i + 1)
            posPtr[i] = UInt32(i)
        }

        let hiddenSize = 2048
        let elementSize = MemoryLayout<Float16>.size
        let hiddenElements = hiddenSize

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
                step.bindRuntimeArguments(encoder: enc, sequenceLength: UInt32(seqLen))
                let grid = step.resolvedGridSize(sequenceLength: seqLen)
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: step.threadgroupSize)

            case .perPosition:
                for pos in 0..<seqLen {
                    enc.setComputePipelineState(step.pipeline)
                    for (index, buffer, base) in step.bufferBindings {
                        let stride = step.perPositionStrides[index] ?? 0
                        enc.setBuffer(buffer, offset: base + pos * stride, index: index)
                    }
                    for (index, value) in step.bytesBindings {
                        value.withUnsafeBufferPointer { ptr in
                            enc.setBytes(ptr.baseAddress!, length: ptr.count, index: index)
                        }
                    }
                    if let pi = step.positionBufferIndex {
                        var pv = UInt32(pos)
                        withUnsafeBytes(of: &pv) { enc.setBytes($0.baseAddress!, length: $0.count, index: pi) }
                    }
                    enc.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
                }

            case .lastToken:
                enc.setComputePipelineState(step.pipeline)
                for (index, buffer, base) in step.bufferBindings {
                    let stride = step.perPositionStrides[index] ?? 0
                    enc.setBuffer(buffer, offset: base + (seqLen - 1) * stride, index: index)
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
                print("[Live diag] GPU ERROR at step \(stepIndex): \(error)")
                Issue.record("GPU error at step \(stepIndex)")
                return
            }

            // Check hidden and scratch for NaN
            let hp = prefillPlan.buffers.hidden.contents().bindMemory(to: Float.self, capacity: hiddenElements)
            let hiddenNaN = (0..<hiddenElements).contains { hp[$0].isNaN }
            let sp = prefillPlan.buffers.scratch.contents().bindMemory(to: Float.self, capacity: hiddenElements)
            let scratchNaN = (0..<hiddenElements).contains { sp[$0].isNaN }
            let sample = (0..<min(4, hiddenElements)).map { Float(hp[$0]) }
            let scratchSample = (0..<min(4, hiddenElements)).map { Float(sp[$0]) }

            let modeStr: String
            switch step.mode {
            case .batch: modeStr = "batch"
            case .perPosition: modeStr = "perPos"
            case .lastToken: modeStr = "lastTok"
            }

            print("[Live diag] step \(stepIndex) (\(modeStr)): hidden=\(sample) hNaN=\(hiddenNaN) scratch=\(scratchSample) sNaN=\(scratchNaN)")

            if (hiddenNaN || scratchNaN) && nanFoundAtStep < 0 {
                nanFoundAtStep = stepIndex
                print("[Live diag] *** FIRST NaN at step \(stepIndex) (\(modeStr)) ***")
                for (index, buffer, offset) in step.bufferBindings {
                    print("[Live diag]   bind[\(index)]: offset=\(offset) len=\(buffer.length)")
                }
                // Don't break — show a couple more steps for context
                if stepIndex > prefillPlan.steps.count - 3 { break }
            }
            if nanFoundAtStep >= 0 && stepIndex > nanFoundAtStep + 2 { break }
        }

        if nanFoundAtStep >= 0 {
            Issue.record("NaN first appeared at step \(nanFoundAtStep)")
        } else {
            print("[Live diag] SUCCESS: No NaN with real model weights!")
        }
    }
}
