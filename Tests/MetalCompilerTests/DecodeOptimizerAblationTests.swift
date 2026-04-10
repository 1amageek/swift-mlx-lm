import Foundation
import Metal
import Testing
@testable import MetalCompiler

@Suite("Decode Optimizer Ablation", .serialized)
struct DecodeOptimizerAblationTests {
    private static let referencePath = "/Users/1amageek/Desktop/swift-lm/TestData/lfm2_reference.safetensors"
    private static let stafPath = "/Users/1amageek/Desktop/swift-lm/TestData/LFM2.5-1.2B-Thinking/model.staf"
    private static let promptTokens: [Int32] = [1, 1, 6, 6423, 708]

    private struct TestEnvironment {
        var model: MetalInferenceModel
        let ref: MetalWeightFile
    }

    @Test("Decode step 1 compares aggressive vs none optimizer")
    func decodeStep1ComparesAggressiveVsNoneOptimizer() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let aggressive = try runDecodeStep1(optimizer: AggressiveOptimizer())
        let none = try runDecodeStep1(optimizer: NoOptimizer())

        print("[DecodeOptAblation] aggressive argmax=\(aggressive.argmax) ref=\(aggressive.refArgmax) logitsErr=\(String(format: "%.4f", aggressive.logitsErr)) hiddenErr=\(String(format: "%.4f", aggressive.hiddenErr))")
        print("[DecodeOptAblation] none       argmax=\(none.argmax) ref=\(none.refArgmax) logitsErr=\(String(format: "%.4f", none.logitsErr)) hiddenErr=\(String(format: "%.4f", none.hiddenErr))")

        #expect(aggressive.refArgmax == none.refArgmax)
    }

    private func runDecodeStep1(optimizer: any DispatchOptimizer) throws -> (argmax: Int, refArgmax: Int, logitsErr: Float, hiddenErr: Float) {
        let env = try Self.buildEnvironment(optimizer: optimizer)
        var model = env.model

        var currentToken = model.prefill(tokens: Self.promptTokens)
        for _ in 0...1 {
            currentToken = model.decodeSync(tokenID: currentToken)
        }

        let refLogits = try readRefTensorAsFloats(env.ref, name: "ref.decode_1.logits")
        let metalLogits = readDecodeBuffer(model.buffers.logits, precision: model.buffers.bufferPrecision)
        let refHidden = try readRefTensorAsFloats(env.ref, name: "ref.decode_1.final_hidden")
        let metalHidden = readDecodeBuffer(model.decodePlan.outputHeadInputBinding().buffer, precision: model.buffers.bufferPrecision)

        return (
            argmax: argmax(metalLogits).index,
            refArgmax: argmax(refLogits).index,
            logitsErr: maxAbsoluteError(metalLogits, refLogits),
            hiddenErr: maxAbsoluteError(metalHidden, refHidden)
        )
    }

    private static func buildEnvironment(optimizer: any DispatchOptimizer) throws -> TestEnvironment {
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
        let compiler = MetalInferenceCompiler(optimizer: optimizer)
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
