import Foundation
import Metal
import Testing
@testable import MetalCompiler

#if ENABLE_METAL_PROBES
@Suite("Decode Layerwise Diagnostics", .serialized)
struct DecodeLayerwiseDiagnosticTests {
    private static let referencePath = "/Users/1amageek/Desktop/swift-lm/TestData/lfm2_reference.safetensors"
    private static let stafPath = "/Users/1amageek/Desktop/swift-lm/TestData/LFM2.5-1.2B-Thinking/model.staf"
    private static let promptTokens: [Int32] = [1, 1, 6, 6423, 708]

    private struct TestEnvironment {
        var model: MetalInferenceModel
        let ref: MetalWeightFile
    }

    private struct ParsedDispatchEntry {
        let layer: Int?
        let kind: String
    }

    @Test("Decode step 1 layerwise diagnostic")
    func decodeStep1LayerwiseDiagnostic() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let env = try Self.buildEnvironment()
        let compiler = MetalInferenceCompiler()
        let dispatchDump = try makeDispatchDump(compiler: compiler)
        let entries = parseDispatchEntries(from: dispatchDump)
        var model = env.model

        var currentToken = model.prefill(tokens: Self.promptTokens)
        currentToken = model.decodeSync(tokenID: currentToken)

        model.buffers.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(model.position)
        model.buffers.tokenIn.contents().bindMemory(to: Int32.self, capacity: 1).pointee = currentToken

        var currentLayer = 0
        var waitingForOperatorResidual = false
        var submission = try MetalSubmissionContext(device: model.device)

        for (stepIndex, step) in model.decodePlan.steps.enumerated() {
            guard stepIndex < entries.count else { break }
            if currentLayer == 13 {
                print("[DecodeLayerwise] step=\(stepIndex) metaLayer=\(String(describing: step.metadata.layerIndex)) kernel=\(step.metadata.kernelName) entry=\(entries[stepIndex].kind)")
            }

            try submission.withCompute { encoder, argumentTable in
                MetalDecodeEncoder.encodeStep(
                    step: step,
                    encoder: encoder,
                    argumentTable: argumentTable
                )
            }

            let entry = entries[stepIndex]
            if entry.kind.contains("projection(o_proj") || entry.kind.contains("projection(out_proj") {
                let metal = readDecodeBuffer(model.buffers.hidden, precision: model.buffers.bufferPrecision)
                let ref = try readRefTensorAsFloats(env.ref, name: "ref.decode_1.layer_\(currentLayer).after_op")
                let err = maxAbsoluteError(metal, ref)
                print("[DecodeLayerwise] layer=\(currentLayer) phase=after_op kind=\(entry.kind) maxErr=\(String(format: "%.4f", err))")
                waitingForOperatorResidual = true
            }

            if entry.kind.contains("projection(down_proj") {
                let metal = readDecodeBuffer(model.buffers.hidden, precision: model.buffers.bufferPrecision)
                let ref = try readRefTensorAsFloats(env.ref, name: "ref.decode_1.layer_\(currentLayer).mlp_out")
                let err = maxAbsoluteError(metal, ref)
                print("[DecodeLayerwise] layer=\(currentLayer) phase=mlp_out kind=\(entry.kind) maxErr=\(String(format: "%.4f", err))")
            }

            if entry.kind.contains("ResidualAddFragment") || entry.kind.contains("synthesized_3way") {
                if waitingForOperatorResidual {
                    waitingForOperatorResidual = false
                } else {
                    let metal = readDecodeBuffer(model.buffers.hidden, precision: model.buffers.bufferPrecision)
                    let ref = try readRefTensorAsFloats(env.ref, name: "ref.decode_1.layer_\(currentLayer).after_mlp")
                    let err = maxAbsoluteError(metal, ref)
                    print("[DecodeLayerwise] layer=\(currentLayer) phase=after_mlp kind=\(entry.kind) maxErr=\(String(format: "%.4f", err))")
                    currentLayer += 1
                }
            }
        }
    }

    @Test("Decode step 1 layer13 ffn_norm output matches CPU reference")
    func decodeStep1Layer13FFNNormOutputMatchesCPUReference() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let env = try Self.buildEnvironment()
        let compiler = MetalInferenceCompiler()
        let dispatchDump = try makeDispatchDump(compiler: compiler)
        let entries = parseDispatchEntries(from: dispatchDump)
        var model = env.model

        var currentToken = model.prefill(tokens: Self.promptTokens)
        currentToken = model.decodeSync(tokenID: currentToken)

        model.buffers.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(model.position)
        model.buffers.tokenIn.contents().bindMemory(to: Int32.self, capacity: 1).pointee = currentToken

        let previousLayerOutput = try readRefTensorAsFloats(env.ref, name: "ref.decode_1.layer_12.after_mlp")
        let currentAfterOp = try readRefTensorAsFloats(env.ref, name: "ref.decode_1.layer_13.after_op")
        let preNorm = zip(previousLayerOutput, currentAfterOp).map(+)

        var currentLayer = 0
        var waitingForOperatorResidual = false
        var submission = try MetalSubmissionContext(device: model.device)

        for (stepIndex, step) in model.decodePlan.steps.enumerated() {
            guard stepIndex < entries.count else { break }

            let entry = entries[stepIndex]
            if entry.kind.contains("projection(o_proj") || entry.kind.contains("projection(out_proj") {
                waitingForOperatorResidual = true
            }

            guard currentLayer == 13, waitingForOperatorResidual, entry.kind.contains("ResidualAddFragment") || entry.kind.contains("synthesized_3way") else {
                try submission.withCompute { encoder, argumentTable in
                    MetalDecodeEncoder.encodeStep(
                        step: step,
                        encoder: encoder,
                        argumentTable: argumentTable
                    )
                }
                if entry.kind.contains("ResidualAddFragment") || entry.kind.contains("synthesized_3way") {
                    if waitingForOperatorResidual {
                        waitingForOperatorResidual = false
                    } else {
                        currentLayer += 1
                    }
                }
                continue
            }

            let scaleBinding = try #require(step.bufferBindings.first(where: { $0.index == 2 }))
            let epsilon = step.bytesBindings.first(where: { $0.index == 5 }).map { readFloatBinding($0.value) } ?? 1e-5
            let scale = readBufferSlice(
                buffer: scaleBinding.buffer,
                offset: scaleBinding.offset,
                count: preNorm.count,
                precision: .bfloat16
            )
            let preHidden = Array(readDecodeBuffer(model.buffers.hidden, precision: model.buffers.bufferPrecision).prefix(preNorm.count))
            let preResidual = Array(readDecodeBuffer(model.buffers.residual, precision: model.buffers.bufferPrecision).prefix(preNorm.count))
            let actualPreNorm = zip(preHidden, preResidual).map(+)

            try submission.withCompute { encoder, argumentTable in
                MetalDecodeEncoder.encodeStep(
                    step: step,
                    encoder: encoder,
                    argumentTable: argumentTable
                )
            }

            let reference = rmsNorm(preNorm, scale: scale, epsilon: epsilon)
            let actual = Array(readDecodeBuffer(model.buffers.scratch, precision: model.buffers.bufferPrecision).prefix(preNorm.count))
            let err = maxAbsoluteError(actual, reference)
            let hiddenErr = maxAbsoluteError(preHidden, currentAfterOp)
            let residualErr = maxAbsoluteError(preResidual, previousLayerOutput)
            let actualInputReference = rmsNorm(actualPreNorm, scale: scale, epsilon: epsilon)
            let kernelErrFromActualInputs = maxAbsoluteError(actual, actualInputReference)

            print("[DecodeLayerwise] layer=13 phase=ffn_norm kind=\(entry.kind) maxErr=\(String(format: "%.4f", err))")
            print("[DecodeLayerwise] layer=13 ffn_norm hiddenErr=\(String(format: "%.4f", hiddenErr)) residualErr=\(String(format: "%.4f", residualErr)) kernelErr(actualInputs)=\(String(format: "%.4f", kernelErrFromActualInputs))")
            let actualSample = actual.prefix(4).map { String(format: "%.4f", $0) }
            let referenceSample = reference.prefix(4).map { String(format: "%.4f", $0) }
            print("[DecodeLayerwise] layer=13 ffn_norm actual[0..3]=\(actualSample)")
            print("[DecodeLayerwise] layer=13 ffn_norm ref[0..3]=\(referenceSample)")
            #expect(err < 0.05, "layer13 ffn_norm drifted: maxErr=\(err)")
            return
        }

        Issue.record("Could not locate layer13 residual add step")
    }

    private static func buildEnvironment() throws -> TestEnvironment {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SetupError.noDevice
        }

        let refURL = URL(fileURLWithPath: Self.referencePath)
        guard FileManager.default.fileExists(atPath: refURL.path) else {
            throw SetupError.noReference
        }

        let stafURL = URL(fileURLWithPath: Self.stafPath)
        guard FileManager.default.fileExists(atPath: stafURL.path) else {
            throw SetupError.noSTAF
        }

        let ref = try SafetensorsLoader().load(at: refURL, device: device)
        let store = try STAFLoader().load(at: stafURL, device: device)
        let spec = try BenchmarkSupport.loadLFM25ModelSpec()
        let compiler = MetalInferenceCompiler()
        let decodePlan = try compiler.compile(
            graph: spec.resolved,
            hiddenSize: spec.config.hiddenSize,
            intermediateSize: spec.config.intermediateSize,
            vocabSize: spec.config.vocabSize,
            stafWeightStore: store,
            device: device
        )
        let prefillPlan = try compiler.compilePrefill(
            graph: spec.resolved,
            hiddenSize: spec.config.hiddenSize,
            intermediateSize: spec.config.intermediateSize,
            vocabSize: spec.config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store,
            sharedKVCache: decodePlan.buffers.kvCache,
            sharedConvState: decodePlan.buffers.convState,
            sharedConvStateDimension: decodePlan.buffers.convStateDimension,
            sharedConvStateKernelSize: decodePlan.buffers.convStateKernelSize,
            sharedRecurrentState: decodePlan.buffers.recurrentState,
            sharedRecurrentStateBytesPerLayer: decodePlan.buffers.recurrentStateBytesPerLayer,
            device: device
        )

        var model = try MetalInferenceModel(plan: decodePlan, device: device)
        model.prefillPlan = prefillPlan
        return TestEnvironment(model: model, ref: ref)
    }

    private func makeDispatchDump(compiler: MetalInferenceCompiler) throws -> String {
        let spec = try BenchmarkSupport.loadLFM25ModelSpec()
        return compiler.dumpDispatchEntries(
            graph: spec.resolved,
            hiddenSize: spec.config.hiddenSize
        )
    }

    private func parseDispatchEntries(from dump: String) -> [ParsedDispatchEntry] {
        dump.split(separator: "\n").compactMap { line in
            guard let bracketEnd = line.firstIndex(of: "]") else { return nil }
            let tail = line[line.index(after: bracketEnd)...].trimmingCharacters(in: .whitespaces)
            if tail.hasPrefix("-- ") {
                return ParsedDispatchEntry(layer: nil, kind: String(tail.dropFirst(3)))
            }
            guard tail.first == "L" else { return nil }
            let pieces = tail.split(separator: " ", maxSplits: 1).map(String.init)
            guard pieces.count == 2, let layer = Int(pieces[0].dropFirst()) else { return nil }
            return ParsedDispatchEntry(layer: layer, kind: pieces[1])
        }
    }

    private func readDecodeBuffer(_ buffer: MTLBuffer, precision: BufferPrecision) -> [Float] {
        if buffer.storageMode == .private {
            let device = buffer.device
            guard let staging = device.makeBuffer(length: buffer.length, options: .storageModeShared),
                  let queue = device.makeCommandQueue(),
                  let commandBuffer = queue.makeCommandBuffer(),
                  let blit = commandBuffer.makeBlitCommandEncoder() else { return [] }
            blit.copy(from: buffer, sourceOffset: 0, to: staging, destinationOffset: 0, size: buffer.length)
            blit.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            return readSharedDecodeBuffer(staging, precision: precision)
        }
        return readSharedDecodeBuffer(buffer, precision: precision)
    }

    private func readSharedDecodeBuffer(_ buffer: MTLBuffer, precision: BufferPrecision) -> [Float] {
        let count = buffer.length / precision.byteSize
        switch precision {
        case .float16:
            let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
            return (0..<count).map { Float(pointer[$0]) }
        case .bfloat16:
            let pointer = buffer.contents().bindMemory(to: BFloat16.self, capacity: count)
            return (0..<count).map { Float(pointer[$0]) }
        case .float32:
            let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: pointer, count: count))
        }
    }

    private func readRefTensorAsFloats(_ file: MetalWeightFile, name: String) throws -> [Float] {
        guard let info = file.tensors[name] else {
            throw SetupError.tensorNotFound(name)
        }
        let count = info.shape.reduce(1, *)
        let pointer = (file.buffer.contents() + file.dataSectionOffset + info.dataOffset)
            .bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(pointer[$0]) }
    }

    private func maxAbsoluteError(_ lhs: [Float], _ rhs: [Float]) -> Float {
        zip(lhs, rhs).reduce(Float.zero) { current, pair in
            max(current, abs(pair.0 - pair.1))
        }
    }

    private func readBufferSlice(
        buffer: MTLBuffer,
        offset: Int,
        count: Int,
        precision: BufferPrecision
    ) -> [Float] {
        if buffer.storageMode == .private {
            let device = buffer.device
            guard let staging = device.makeBuffer(length: count * precision.byteSize, options: .storageModeShared),
                  let queue = device.makeCommandQueue(),
                  let commandBuffer = queue.makeCommandBuffer(),
                  let blit = commandBuffer.makeBlitCommandEncoder() else { return [] }
            blit.copy(from: buffer, sourceOffset: offset, to: staging, destinationOffset: 0, size: count * precision.byteSize)
            blit.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            return readSharedDecodeBuffer(staging, precision: precision)
        }

        switch precision {
        case .float16:
            let pointer = (buffer.contents() + offset).bindMemory(to: Float16.self, capacity: count)
            return (0..<count).map { Float(pointer[$0]) }
        case .bfloat16:
            let pointer = (buffer.contents() + offset).bindMemory(to: BFloat16.self, capacity: count)
            return (0..<count).map { Float(pointer[$0]) }
        case .float32:
            let pointer = (buffer.contents() + offset).bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: pointer, count: count))
        }
    }

    private func rmsNorm(_ input: [Float], scale: [Float], epsilon: Float) -> [Float] {
        let meanSquare = input.reduce(Float.zero) { $0 + $1 * $1 } / Float(input.count)
        let invRMS = 1.0 / sqrtf(meanSquare + epsilon)
        return zip(input, scale).map { $0 * invRMS * $1 }
    }

    private func readFloatBinding(_ bytes: [UInt8]) -> Float {
        precondition(bytes.count == MemoryLayout<Float>.size)
        return bytes.withUnsafeBytes { $0.load(as: Float.self) }
    }

    private enum SetupError: Error {
        case noDevice
        case noReference
        case noSTAF
        case tensorNotFound(String)
    }
}
#endif
