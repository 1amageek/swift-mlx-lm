import Testing
import Metal
import Foundation
@testable import MetalCompiler

#if ENABLE_METAL_PROBES
@Suite("Decode Precision Diagnostics", .serialized)
struct DecodePrecisionDiagnosticTests {
    private static let referencePath = "/Users/1amageek/Desktop/swift-lm/TestData/lfm2_reference.safetensors"
    private static let stafPath = "/Users/1amageek/Desktop/swift-lm/TestData/LFM2.5-1.2B-Thinking/model.staf"

    private struct TestEnvironment {
        var model: MetalInferenceModel
        let ref: MetalWeightFile
    }

    private struct StepResult {
        let step: Int
        let decodePrecision: BufferPrecision
        let metalCount: Int
        let referenceCount: Int
        let metalNaNCount: Int
        let metalArgmax: Int
        let metalArgmaxValue: Float
        let refArgmax: Int
        let refArgmaxValue: Float
        let maxErr: Float
    }

    @Test("Float32 decode improves step 1 reference agreement")
    func float32DecodeImprovesStep1() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let baseline = try runDecodeDiagnostic(step: 1, decodeBufferPrecisionOverride: nil)
        let float32 = try runDecodeDiagnostic(step: 1, decodeBufferPrecisionOverride: .float32)
        let baselineErr = String(format: "%.4f", baseline.maxErr)
        let float32Err = String(format: "%.4f", float32.maxErr)

        print("[DecodePrecision] step=1 baseline precision=\(baseline.decodePrecision) counts=\(baseline.metalCount)/\(baseline.referenceCount) nan=\(baseline.metalNaNCount) metalArgmax=\(baseline.metalArgmax) refArgmax=\(baseline.refArgmax) maxErr=\(baselineErr)")
        print("[DecodePrecision] step=1 float32 precision=\(float32.decodePrecision) counts=\(float32.metalCount)/\(float32.referenceCount) nan=\(float32.metalNaNCount) metalArgmax=\(float32.metalArgmax) refArgmax=\(float32.refArgmax) maxErr=\(float32Err)")

        #expect(baseline.metalCount == baseline.referenceCount)
        #expect(float32.metalCount == float32.referenceCount)
        #expect(float32.metalNaNCount == 0)
        #expect(float32.maxErr < baseline.maxErr,
                "Float32 decode did not improve step 1: baseline=\(baseline.maxErr) float32=\(float32.maxErr)")
    }

    @Test("Float32 decode improves step 2 reference agreement")
    func float32DecodeImprovesStep2() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let baseline = try runDecodeDiagnostic(step: 2, decodeBufferPrecisionOverride: nil)
        let float32 = try runDecodeDiagnostic(step: 2, decodeBufferPrecisionOverride: .float32)
        let baselineErr = String(format: "%.4f", baseline.maxErr)
        let float32Err = String(format: "%.4f", float32.maxErr)

        print("[DecodePrecision] step=2 baseline precision=\(baseline.decodePrecision) counts=\(baseline.metalCount)/\(baseline.referenceCount) nan=\(baseline.metalNaNCount) metalArgmax=\(baseline.metalArgmax) refArgmax=\(baseline.refArgmax) maxErr=\(baselineErr)")
        print("[DecodePrecision] step=2 float32 precision=\(float32.decodePrecision) counts=\(float32.metalCount)/\(float32.referenceCount) nan=\(float32.metalNaNCount) metalArgmax=\(float32.metalArgmax) refArgmax=\(float32.refArgmax) maxErr=\(float32Err)")

        #expect(baseline.metalCount == baseline.referenceCount)
        #expect(float32.metalCount == float32.referenceCount)
        #expect(float32.metalNaNCount == 0)
        #expect(float32.maxErr < baseline.maxErr,
                "Float32 decode did not improve step 2: baseline=\(baseline.maxErr) float32=\(float32.maxErr)")
    }

    @Test("Float16 decode improves step 1 reference agreement")
    func float16DecodeImprovesStep1() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let baseline = try runDecodeDiagnostic(step: 1, decodeBufferPrecisionOverride: nil)
        let float16 = try runDecodeDiagnostic(step: 1, decodeBufferPrecisionOverride: .float16)
        let baselineErr = String(format: "%.4f", baseline.maxErr)
        let float16Err = String(format: "%.4f", float16.maxErr)

        print("[DecodePrecision] step=1 baseline precision=\(baseline.decodePrecision) counts=\(baseline.metalCount)/\(baseline.referenceCount) nan=\(baseline.metalNaNCount) metalArgmax=\(baseline.metalArgmax) refArgmax=\(baseline.refArgmax) maxErr=\(baselineErr)")
        print("[DecodePrecision] step=1 float16 precision=\(float16.decodePrecision) counts=\(float16.metalCount)/\(float16.referenceCount) nan=\(float16.metalNaNCount) metalArgmax=\(float16.metalArgmax) refArgmax=\(float16.refArgmax) maxErr=\(float16Err)")

        #expect(baseline.metalCount == baseline.referenceCount)
        #expect(float16.metalCount == float16.referenceCount)
        #expect(float16.metalNaNCount == 0)
        #expect(float16.maxErr < baseline.maxErr,
                "Float16 decode did not improve step 1: baseline=\(baseline.maxErr) float16=\(float16.maxErr)")
    }

    @Test("Float16 decode improves step 2 reference agreement")
    func float16DecodeImprovesStep2() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let baseline = try runDecodeDiagnostic(step: 2, decodeBufferPrecisionOverride: nil)
        let float16 = try runDecodeDiagnostic(step: 2, decodeBufferPrecisionOverride: .float16)
        let baselineErr = String(format: "%.4f", baseline.maxErr)
        let float16Err = String(format: "%.4f", float16.maxErr)

        print("[DecodePrecision] step=2 baseline precision=\(baseline.decodePrecision) counts=\(baseline.metalCount)/\(baseline.referenceCount) nan=\(baseline.metalNaNCount) metalArgmax=\(baseline.metalArgmax) refArgmax=\(baseline.refArgmax) maxErr=\(baselineErr)")
        print("[DecodePrecision] step=2 float16 precision=\(float16.decodePrecision) counts=\(float16.metalCount)/\(float16.referenceCount) nan=\(float16.metalNaNCount) metalArgmax=\(float16.metalArgmax) refArgmax=\(float16.refArgmax) maxErr=\(float16Err)")

        #expect(baseline.metalCount == baseline.referenceCount)
        #expect(float16.metalCount == float16.referenceCount)
        #expect(float16.metalNaNCount == 0)
        #expect(float16.maxErr < baseline.maxErr,
                "Float16 decode did not improve step 2: baseline=\(baseline.maxErr) float16=\(float16.maxErr)")
    }

    private func runDecodeDiagnostic(
        step: Int,
        decodeBufferPrecisionOverride: BufferPrecision?
    ) throws -> StepResult {
        let env = try Self.buildEnvironment(decodeBufferPrecisionOverride: decodeBufferPrecisionOverride)
        var model = env.model
        let tokens: [Int32] = [1, 1, 6, 6423, 708]

        var currentToken = model.prefill(tokens: tokens)
        for _ in 0...step {
            currentToken = model.decodeSync(tokenID: currentToken)
        }

        let metalLogits = readDecodeBuffer(model.buffers.logits, precision: model.buffers.bufferPrecision)
        let refLogits = try readRefTensorAsFloats(env.ref, name: "ref.decode_\(step).logits")
        let metalTop = argmax(metalLogits)
        let refTop = argmax(refLogits)
        return StepResult(
            step: step,
            decodePrecision: model.buffers.bufferPrecision,
            metalCount: metalLogits.count,
            referenceCount: refLogits.count,
            metalNaNCount: metalLogits.filter(\.isNaN).count,
            metalArgmax: metalTop.index,
            metalArgmaxValue: metalTop.value,
            refArgmax: refTop.index,
            refArgmaxValue: refTop.value,
            maxErr: maxAbsoluteError(metalLogits, refLogits)
        )
    }

    private static func buildEnvironment(
        decodeBufferPrecisionOverride: BufferPrecision?
    ) throws -> TestEnvironment {
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
        let compiler = MetalInferenceCompiler(decodeBufferPrecisionOverride: decodeBufferPrecisionOverride)
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
        let count = buffer.length / precision.byteSize
        switch precision {
        case .float16:
            let ptr = buffer.contents().bindMemory(to: Float16.self, capacity: count)
            return (0..<count).map { Float(ptr[$0]) }
        case .bfloat16:
            let ptr = buffer.contents().bindMemory(to: BFloat16.self, capacity: count)
            return (0..<count).map { Float(ptr[$0]) }
        case .float32:
            let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: ptr, count: count))
        }
    }

    private func readRefTensorAsFloats(
        _ file: MetalWeightFile,
        name: String
    ) throws -> [Float] {
        guard let info = file.tensors[name] else {
            throw SetupError.tensorNotFound(name)
        }
        let count = info.shape.reduce(1, *)
        let ptr = (file.buffer.contents() + file.dataSectionOffset + info.dataOffset)
            .bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(ptr[$0]) }
    }

    private func maxAbsoluteError(_ lhs: [Float], _ rhs: [Float]) -> Float {
        zip(lhs, rhs).reduce(Float.zero) { current, pair in
            max(current, abs(pair.0 - pair.1))
        }
    }

    private func argmax(_ values: [Float]) -> (index: Int, value: Float) {
        values.enumerated().max(by: { $0.element < $1.element }).map {
            (index: $0.offset, value: $0.element)
        } ?? (index: 0, value: -.infinity)
    }

    private enum SetupError: Error {
        case noDevice
        case noReference
        case noSTAF
        case tensorNotFound(String)
    }
}
#endif
