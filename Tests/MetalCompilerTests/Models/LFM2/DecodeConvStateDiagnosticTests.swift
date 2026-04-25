import Foundation
import Metal
import Testing
@testable import MetalCompiler

#if ENABLE_METAL_PROBES
@Suite("Decode ConvState Diagnostics", .serialized)
struct DecodeConvStateDiagnosticTests {
    private static let referencePath = "/Users/1amageek/Desktop/swift-lm/TestData/lfm2_reference.safetensors"
    private static let stafPath = "/Users/1amageek/Desktop/swift-lm/TestData/LFM2.5-1.2B-Thinking/model.staf"
    private static let promptTokens: [Int32] = [1, 1, 6, 6423, 708]

    private struct TestEnvironment {
        var model: MetalInferenceModel
        let ref: MetalWeightFile
    }

    private struct StepDiagnostic {
        let argmax: Int
        let refArgmax: Int
        let logitsMaxErr: Float
        let finalHiddenMaxErr: Float
    }

    @Test("Decode step 1 conv-state injection diagnostic")
    func decodeStep1ConvStateInjectionDiagnostic() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let baseline = try autoreleasepool {
            try runDecodeDiagnostic(step: 1, beforeFinalStep: nil)
        }
        let injected = try autoreleasepool {
            try runDecodeDiagnostic(
                step: 1,
                beforeFinalStep: { model, ref in
                    try injectReferenceConvState(step: 0, into: &model, ref: ref)
                }
            )
        }

        let baselineLogitsErr = String(format: "%.4f", baseline.logitsMaxErr)
        let baselineHiddenErr = String(format: "%.4f", baseline.finalHiddenMaxErr)
        let injectedLogitsErr = String(format: "%.4f", injected.logitsMaxErr)
        let injectedHiddenErr = String(format: "%.4f", injected.finalHiddenMaxErr)
        print("[DecodeConvStateOnly] step=1 baseline argmax=\(baseline.argmax) ref=\(baseline.refArgmax) logitsErr=\(baselineLogitsErr) finalHiddenErr=\(baselineHiddenErr)")
        print("[DecodeConvStateOnly] step=1 injected argmax=\(injected.argmax) ref=\(injected.refArgmax) logitsErr=\(injectedLogitsErr) finalHiddenErr=\(injectedHiddenErr)")

        #expect(baseline.refArgmax == injected.refArgmax)
    }

    @Test("Decode step 2 conv-state injection diagnostic")
    func decodeStep2ConvStateInjectionDiagnostic() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let baseline = try autoreleasepool {
            try runDecodeDiagnostic(step: 2, beforeFinalStep: nil)
        }
        let injected = try autoreleasepool {
            try runDecodeDiagnostic(
                step: 2,
                beforeFinalStep: { model, ref in
                    try injectReferenceConvState(step: 1, into: &model, ref: ref)
                }
            )
        }

        let baselineLogitsErr = String(format: "%.4f", baseline.logitsMaxErr)
        let baselineHiddenErr = String(format: "%.4f", baseline.finalHiddenMaxErr)
        let injectedLogitsErr = String(format: "%.4f", injected.logitsMaxErr)
        let injectedHiddenErr = String(format: "%.4f", injected.finalHiddenMaxErr)
        print("[DecodeConvStateOnly] step=2 baseline argmax=\(baseline.argmax) ref=\(baseline.refArgmax) logitsErr=\(baselineLogitsErr) finalHiddenErr=\(baselineHiddenErr)")
        print("[DecodeConvStateOnly] step=2 injected argmax=\(injected.argmax) ref=\(injected.refArgmax) logitsErr=\(injectedLogitsErr) finalHiddenErr=\(injectedHiddenErr)")

        #expect(baseline.refArgmax == injected.refArgmax)
    }

    private func runDecodeDiagnostic(
        step: Int,
        beforeFinalStep: ((inout MetalInferenceModel, MetalWeightFile) throws -> Void)?
    ) throws -> StepDiagnostic {
        let env = try Self.buildEnvironment()
        var model = env.model
        let firstToken = model.prefill(tokens: Self.promptTokens)
        var currentToken = firstToken
        for currentStep in 0...step {
            if currentStep == step, let beforeFinalStep {
                try beforeFinalStep(&model, env.ref)
            }
            currentToken = model.decodeSync(tokenID: currentToken)
        }

        let refLogits = try readRefTensorAsFloats(env.ref, name: "ref.decode_\(step).logits")
        let metalLogits = readDecodeBuffer(model.buffers.logits, precision: model.buffers.bufferPrecision)
        let refHidden = try readRefTensorAsFloats(env.ref, name: "ref.decode_\(step).final_hidden")
        let metalHidden = readDecodeBuffer(
            model.decodePlan.outputHeadInputBinding().buffer,
            precision: model.buffers.bufferPrecision
        )

        return StepDiagnostic(
            argmax: argmax(metalLogits).index,
            refArgmax: argmax(refLogits).index,
            logitsMaxErr: maxAbsoluteError(metalLogits, refLogits),
            finalHiddenMaxErr: maxAbsoluteError(metalHidden, refHidden)
        )
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

    private func injectReferenceConvState(
        step: Int,
        into model: inout MetalInferenceModel,
        ref: MetalWeightFile
    ) throws {
        guard let convStateBuffer = model.buffers.convState else {
            return
        }
        let convDim = model.buffers.convStateDimension
        let kernelSize = model.buffers.convStateKernelSize
        let valuesPerLayer = convDim * kernelSize
        let layerCount = convStateBuffer.length / (valuesPerLayer * MemoryLayout<Float16>.stride)
        let totalValueCount = layerCount * valuesPerLayer
        var packed = Array<Float16>(repeating: .zero, count: totalValueCount)

        for convIdx in 0..<layerCount {
            let values = try readRefTensorAsFloats(ref, name: "ref.decode_\(step).conv_state.\(convIdx)")
            let base = convIdx * valuesPerLayer
            let limit = min(values.count, valuesPerLayer)
            for valueIndex in 0..<limit {
                packed[base + valueIndex] = Float16(values[valueIndex])
            }
        }

        let byteCount = packed.count * MemoryLayout<Float16>.stride
        if convStateBuffer.storageMode == .private {
            guard let staging = model.device.makeBuffer(length: byteCount, options: .storageModeShared) else {
                throw SetupError.noDevice
            }
            let pointer = staging.contents().bindMemory(to: Float16.self, capacity: packed.count)
            packed.withUnsafeBufferPointer { source in
                guard let sourceBaseAddress = source.baseAddress else { return }
                pointer.update(from: sourceBaseAddress, count: source.count)
            }
            var submission = try MetalSubmissionContext(device: model.device)
            try submission.copyBuffers([
                (
                    from: staging,
                    sourceOffset: 0,
                    to: convStateBuffer,
                    destinationOffset: 0,
                    size: byteCount
                )
            ])
            return
        }

        let pointer = convStateBuffer.contents().bindMemory(to: Float16.self, capacity: packed.count)
        packed.withUnsafeBufferPointer { source in
            guard let sourceBaseAddress = source.baseAddress else { return }
            pointer.update(from: sourceBaseAddress, count: source.count)
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

    private func argmax(_ values: [Float]) -> (index: Int, value: Float) {
        values.enumerated().max(by: { $0.element < $1.element }).map {
            (index: $0.offset, value: $0.element)
        } ?? (index: 0, value: -.infinity)
    }

    private func maxAbsoluteError(_ lhs: [Float], _ rhs: [Float]) -> Float {
        zip(lhs, rhs).reduce(Float.zero) { current, pair in
            max(current, abs(pair.0 - pair.1))
        }
    }

    private enum SetupError: Error {
        case noDevice
        case noReference
        case noSTAF
        case tensorNotFound(String)
    }
}
#endif
