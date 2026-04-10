import Testing
import Metal
import Foundation
@testable import MetalCompiler
import LMArchitecture
import ModelDeclarations
import LMIR

/// Focused diagnostic: run prefill step-by-step and dump per-layer hidden state
/// compared against Python reference. Writes results to a file for reliable reading.
@Suite("Prefill Diagnostic")
struct PrefillDiagnosticTest {

    private static let referencePath = "/Users/1amageek/Desktop/swift-lm/TestData/lfm2_reference.safetensors"
    private static let stafPath = "/Users/1amageek/Desktop/swift-lm/TestData/LFM2.5-1.2B-Thinking/model.staf"
    private static let outputPath = "/Users/1amageek/Desktop/swift-lm/TestData/prefill_diagnostic.txt"

    @Test("Step-by-step prefill diagnostic with per-layer comparison",
          .disabled("Pre-migration diagnostic: accesses storageModePrivate scratch buffer via contents()"))
    func stepByStepDiagnostic() throws {
        guard let resources = try RealModelTestSupport.loadOrSkip(skipMessage: "STAF not found — skipping") else {
            return
        }
        defer { resources.release() }

        let device = resources.device

        let refURL = URL(fileURLWithPath: Self.referencePath)
        guard FileManager.default.fileExists(atPath: refURL.path) else { throw DiagError.noFile }

        let ref = try SafetensorsLoader().load(at: refURL, device: device)
        let store = resources.store

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
        let prefillPlan = try compiler.compilePrefill(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store, device: device)

        let seqLen = 5
        let hiddenSize = 2048
        let tokens: [Int32] = [1, 1, 6, 6423, 708]

        // Fill inputs
        let tokenPtr = prefillPlan.buffers.tokenIDs.contents()
            .bindMemory(to: Int32.self, capacity: seqLen)
        let posPtr = prefillPlan.buffers.positions.contents()
            .bindMemory(to: UInt32.self, capacity: seqLen)
        for i in 0..<seqLen { tokenPtr[i] = tokens[i]; posPtr[i] = UInt32(i) }

        var output = "=== Prefill Step-by-Step Diagnostic ===\n"
        output += "Total steps: \(prefillPlan.steps.count)\n\n"

        // Dump STAF weight bindings for layer 0 norms
        let normNames = [
            "model.layers.0.operator_norm.weight",
            "model.layers.0.ffn_norm.weight",
            "model.layers.1.operator_norm.weight",
        ]
        for name in normNames {
            if let access = store.bufferAccess(for: name) {
                let ptr = (access.buffer.contents() + access.offset)
                    .bindMemory(to: BFloat16.self, capacity: 4)
                let vals = (0..<4).map { Float(ptr[$0]) }
                output += "STAF \(name): \(vals.map { String(format: "%.4f", $0) })\n"
            } else {
                output += "STAF \(name): NOT FOUND\n"
            }
        }
        output += "\n"

        // Run step by step
        guard let queue = device.makeCommandQueue() else { throw DiagError.noDevice }

        for stepIndex in 0..<prefillPlan.steps.count {
            let step = prefillPlan.steps[stepIndex]

            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!

            if step.sync == .bufferBarrier { enc.memoryBarrier(scope: .buffers) }
            enc.setComputePipelineState(step.pipeline)

            for (index, buffer, offset) in step.bufferBindings {
                enc.setBuffer(buffer, offset: offset, index: index)
            }
            for (index, value) in step.bytesBindings {
                value.withUnsafeBufferPointer {
                    enc.setBytes($0.baseAddress!, length: $0.count, index: index)
                }
            }

            switch step.mode {
            case .batch:
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
                    if let posIdx = step.positionBufferIndex {
                        var posValue = UInt32(pos)
                        withUnsafeBytes(of: &posValue) {
                            enc.setBytes($0.baseAddress!, length: $0.count, index: posIdx)
                        }
                    }
                    for (index, value) in step.bytesBindings {
                        value.withUnsafeBufferPointer {
                            enc.setBytes($0.baseAddress!, length: $0.count, index: index)
                        }
                    }
                    enc.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
                }

            case .lastToken:
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
                enc.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
            }

            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()

            // Read hidden state at last token after every step (F32 buffers in prefill)
            let lastOffset = (seqLen - 1) * hiddenSize
            let hiddenPtr = prefillPlan.buffers.hidden.contents()
                .bindMemory(to: Float32.self, capacity: seqLen * hiddenSize)

            let norm = l2normF32(hiddenPtr + lastOffset, count: hiddenSize)
            let sample = (0..<4).map { String(format: "%.4f", hiddenPtr[lastOffset + $0]) }
            let nanCount = (0..<hiddenSize).filter { hiddenPtr[lastOffset + $0].isNaN }.count

            // Also read scratch buffer at slot 0 (last token, F32)
            let scratchPtr = prefillPlan.buffers.scratch.contents()
                .bindMemory(to: Float32.self, capacity: seqLen * hiddenSize)
            let scratchNorm = l2normF32(scratchPtr + lastOffset, count: hiddenSize)
            let scratchSample = (0..<4).map { String(format: "%.4f", scratchPtr[lastOffset + $0]) }

            let kernelName = step.pipeline.label ?? "unknown"
            // For norm steps, dump weight buffer first 4 values as BF16
            var weightInfo = ""
            if kernelName.contains("norm") && step.bufferBindings.count >= 2 {
                let (_, wBuf, wOff) = step.bufferBindings[1]
                let wPtr = (wBuf.contents() + wOff).bindMemory(to: BFloat16.self, capacity: 4)
                let wVals = (0..<4).map { String(format: "%.4f", Float(wPtr[$0])) }
                weightInfo = " w=\(wVals)"
            }
            output += "step \(stepIndex) [\(kernelName)] mode=\(step.mode): hidden norm=\(String(format: "%.2f", norm)) scratch0 norm=\(String(format: "%.2f", scratchNorm)) sample_h=\(sample) sample_s=\(scratchSample)\(weightInfo) nan=\(nanCount)\n"
        }

        // Compare final results with Python reference
        output += "\n=== Final Comparison ===\n"

        // Embedding
        let embRef = loadTensor(ref, name: "ref.prefill.embedding")
        if let embRef = embRef {
            let lastOffset = (seqLen - 1) * hiddenSize
            let refNorm = l2norm(embRef + lastOffset, count: hiddenSize)
            let refSample = (0..<4).map { String(format: "%.4f", Float(embRef[lastOffset + $0])) }
            output += "Python embedding (last token): norm=\(String(format: "%.2f", refNorm)) sample=\(refSample)\n"
        }

        // Per-layer after_mlp
        for layerIdx in 0..<16 {
            let refLayer = loadTensor(ref, name: "ref.prefill.layer_\(layerIdx).after_mlp")
            if let refPtr = refLayer {
                let lastOffset = (seqLen - 1) * hiddenSize
                let refNorm = l2norm(refPtr + lastOffset, count: hiddenSize)
                let refSample = (0..<4).map { String(format: "%.4f", Float(refPtr[lastOffset + $0])) }
                output += "Python layer_\(layerIdx).after_mlp: norm=\(String(format: "%.2f", refNorm)) sample=\(refSample)\n"
            }
        }

        // Final hidden
        let refFinal = loadTensor(ref, name: "ref.prefill.final_hidden")
        if let refPtr = refFinal {
            let lastOffset = (seqLen - 1) * hiddenSize
            let refNorm = l2norm(refPtr + lastOffset, count: hiddenSize)
            let refSample = (0..<4).map { String(format: "%.4f", Float(refPtr[lastOffset + $0])) }
            output += "Python final_hidden: norm=\(String(format: "%.2f", refNorm)) sample=\(refSample)\n"
        }

        // Logits (F32 in prefill)
        let logitsPtr = prefillPlan.buffers.logits.contents()
            .bindMemory(to: Float32.self, capacity: 65536)
        var metalMax: Float = -.infinity; var metalMaxIdx = 0
        for i in 0..<65536 {
            let v = logitsPtr[i]
            if v > metalMax { metalMax = v; metalMaxIdx = i }
        }
        let refLogits = loadTensor(ref, name: "ref.prefill.logits_last")
        var refMax: Float = -.infinity; var refMaxIdx = 0
        if let refPtr = refLogits {
            for i in 0..<65536 {
                let v = Float(refPtr[i])
                if v > refMax { refMax = v; refMaxIdx = i }
            }
        }
        output += "\nMetal logits argmax=\(metalMaxIdx) (val=\(String(format: "%.2f", metalMax)))\n"
        output += "Python logits argmax=\(refMaxIdx) (val=\(String(format: "%.2f", refMax)))\n"

        // Write to file
        try output.write(toFile: Self.outputPath, atomically: true, encoding: .utf8)
        print("[Diag] Results written to \(Self.outputPath)")

        #expect(metalMaxIdx == refMaxIdx, "Logits argmax: Metal=\(metalMaxIdx) Python=\(refMaxIdx)")
    }

    // MARK: - Helpers

    private func loadTensor(_ file: MetalWeightFile, name: String) -> UnsafeMutablePointer<Float16>? {
        guard let info = file.tensors[name] else { return nil }
        let count = info.shape.reduce(1, *)
        return (file.buffer.contents() + file.dataSectionOffset + info.dataOffset)
            .bindMemory(to: Float16.self, capacity: count)
    }

    private func l2norm(_ ptr: UnsafePointer<Float16>, count: Int) -> Float {
        var sum: Float = 0
        for i in 0..<count {
            let v = Float(ptr[i])
            sum += v * v
        }
        return sqrtf(sum)
    }

    private func l2normF32(_ ptr: UnsafePointer<Float32>, count: Int) -> Float {
        var sum: Float = 0
        for i in 0..<count { sum += ptr[i] * ptr[i] }
        return sqrtf(sum)
    }


    private enum DiagError: Error {
        case noDevice
        case noFile
    }
}
