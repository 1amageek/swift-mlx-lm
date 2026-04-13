import Testing
import Metal
import Foundation
@testable import MetalCompiler

@Suite("Decode Barrier Diagnostics", .serialized)
struct DecodeBarrierDiagnosticTests {
    private static let referencePath = "/Users/1amageek/Desktop/swift-lm/TestData/lfm2_reference.safetensors"
    private static let stafPath = "/Users/1amageek/Desktop/swift-lm/TestData/LFM2.5-1.2B-Thinking/model.staf"
    private static let promptTokens: [Int32] = [1, 1, 6, 6423, 708]

    private struct TestEnvironment {
        var model: MetalInferenceModel
        let ref: MetalWeightFile
        let device: MTLDevice
    }

    private struct StepDiagnostic {
        let step: Int
        let tag: String
        let argmax: Int
        let refArgmax: Int
        let logitsMaxErr: Float
        let finalHiddenMaxErr: Float
    }

    @Test("Decode step 1 barrier variants diagnostic")
    func decodeStep1BarrierVariantsDiagnostic() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let baseline = try runDecodeDiagnostic(step: 1, tag: "baseline")
        let forcedBarriers = try runDecodeDiagnostic(
            step: 1,
            tag: "forced-barriers",
            mutatePlan: { forceBarrierPolicy(on: $0, barrierPolicy: .bufferBarrier) }
        )
        let forcedBarriersDevice = try runDecodeDiagnostic(
            step: 1,
            tag: "forced-barriers-device",
            mutatePlan: { forceBarrierPolicy(on: $0, barrierPolicy: .bufferBarrier) },
            visibilityOptions: .device
        )

        print("[DecodeBarrier] step=1 baseline argmax=\(baseline.argmax) ref=\(baseline.refArgmax) logitsErr=\(String(format: "%.4f", baseline.logitsMaxErr)) finalHiddenErr=\(String(format: "%.4f", baseline.finalHiddenMaxErr))")
        print("[DecodeBarrier] step=1 forced-barriers argmax=\(forcedBarriers.argmax) ref=\(forcedBarriers.refArgmax) logitsErr=\(String(format: "%.4f", forcedBarriers.logitsMaxErr)) finalHiddenErr=\(String(format: "%.4f", forcedBarriers.finalHiddenMaxErr))")
        print("[DecodeBarrier] step=1 forced-barriers-device argmax=\(forcedBarriersDevice.argmax) ref=\(forcedBarriersDevice.refArgmax) logitsErr=\(String(format: "%.4f", forcedBarriersDevice.logitsMaxErr)) finalHiddenErr=\(String(format: "%.4f", forcedBarriersDevice.finalHiddenMaxErr))")

        #expect(baseline.refArgmax == forcedBarriers.refArgmax)
        #expect(baseline.refArgmax == forcedBarriersDevice.refArgmax)
    }

    @Test("Decode step 2 barrier variants diagnostic")
    func decodeStep2BarrierVariantsDiagnostic() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let baseline = try runDecodeDiagnostic(step: 2, tag: "baseline")
        let forcedBarriers = try runDecodeDiagnostic(
            step: 2,
            tag: "forced-barriers",
            mutatePlan: { forceBarrierPolicy(on: $0, barrierPolicy: .bufferBarrier) }
        )
        let forcedBarriersDevice = try runDecodeDiagnostic(
            step: 2,
            tag: "forced-barriers-device",
            mutatePlan: { forceBarrierPolicy(on: $0, barrierPolicy: .bufferBarrier) },
            visibilityOptions: .device
        )

        print("[DecodeBarrier] step=2 baseline argmax=\(baseline.argmax) ref=\(baseline.refArgmax) logitsErr=\(String(format: "%.4f", baseline.logitsMaxErr)) finalHiddenErr=\(String(format: "%.4f", baseline.finalHiddenMaxErr))")
        print("[DecodeBarrier] step=2 forced-barriers argmax=\(forcedBarriers.argmax) ref=\(forcedBarriers.refArgmax) logitsErr=\(String(format: "%.4f", forcedBarriers.logitsMaxErr)) finalHiddenErr=\(String(format: "%.4f", forcedBarriers.finalHiddenMaxErr))")
        print("[DecodeBarrier] step=2 forced-barriers-device argmax=\(forcedBarriersDevice.argmax) ref=\(forcedBarriersDevice.refArgmax) logitsErr=\(String(format: "%.4f", forcedBarriersDevice.logitsMaxErr)) finalHiddenErr=\(String(format: "%.4f", forcedBarriersDevice.finalHiddenMaxErr))")

        #expect(baseline.refArgmax == forcedBarriers.refArgmax)
        #expect(baseline.refArgmax == forcedBarriersDevice.refArgmax)
    }

    @Test("Decode step 1 optimizer variants diagnostic")
    func decodeStep1OptimizerVariantsDiagnostic() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let aggressive = try runDecodeDiagnostic(
            step: 1,
            tag: "aggressive",
            optimizer: AggressiveOptimizer()
        )
        let standard = try runDecodeDiagnostic(
            step: 1,
            tag: "standard",
            optimizer: StandardOptimizer()
        )

        print("[DecodeOptimizer] step=1 aggressive argmax=\(aggressive.argmax) ref=\(aggressive.refArgmax) logitsErr=\(String(format: "%.4f", aggressive.logitsMaxErr)) finalHiddenErr=\(String(format: "%.4f", aggressive.finalHiddenMaxErr))")
        print("[DecodeOptimizer] step=1 standard argmax=\(standard.argmax) ref=\(standard.refArgmax) logitsErr=\(String(format: "%.4f", standard.logitsMaxErr)) finalHiddenErr=\(String(format: "%.4f", standard.finalHiddenMaxErr))")

        #expect(aggressive.refArgmax == standard.refArgmax)
    }

    @Test("Decode step 2 optimizer variants diagnostic")
    func decodeStep2OptimizerVariantsDiagnostic() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let aggressive = try runDecodeDiagnostic(
            step: 2,
            tag: "aggressive",
            optimizer: AggressiveOptimizer()
        )
        let standard = try runDecodeDiagnostic(
            step: 2,
            tag: "standard",
            optimizer: StandardOptimizer()
        )

        print("[DecodeOptimizer] step=2 aggressive argmax=\(aggressive.argmax) ref=\(aggressive.refArgmax) logitsErr=\(String(format: "%.4f", aggressive.logitsMaxErr)) finalHiddenErr=\(String(format: "%.4f", aggressive.finalHiddenMaxErr))")
        print("[DecodeOptimizer] step=2 standard argmax=\(standard.argmax) ref=\(standard.refArgmax) logitsErr=\(String(format: "%.4f", standard.logitsMaxErr)) finalHiddenErr=\(String(format: "%.4f", standard.finalHiddenMaxErr))")

        #expect(aggressive.refArgmax == standard.refArgmax)
    }

    @Test("Decode step 1 weight layout variants diagnostic")
    func decodeStep1WeightLayoutVariantsDiagnostic() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let baseline = try runDecodeDiagnostic(step: 1, tag: "baseline")
        let canonical = try runDecodeDiagnostic(
            step: 1,
            tag: "canonical-row-major",
            weightAccessPolicyOverride: canonicalRowMajorDecodeOverride()
        )

        print("[DecodeWeightLayout] step=1 baseline argmax=\(baseline.argmax) ref=\(baseline.refArgmax) logitsErr=\(String(format: "%.4f", baseline.logitsMaxErr)) finalHiddenErr=\(String(format: "%.4f", baseline.finalHiddenMaxErr))")
        print("[DecodeWeightLayout] step=1 canonical-row-major argmax=\(canonical.argmax) ref=\(canonical.refArgmax) logitsErr=\(String(format: "%.4f", canonical.logitsMaxErr)) finalHiddenErr=\(String(format: "%.4f", canonical.finalHiddenMaxErr))")

        #expect(baseline.refArgmax == canonical.refArgmax)
    }

    @Test("Decode step 2 weight layout variants diagnostic")
    func decodeStep2WeightLayoutVariantsDiagnostic() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let baseline = try runDecodeDiagnostic(step: 2, tag: "baseline")
        let canonical = try runDecodeDiagnostic(
            step: 2,
            tag: "canonical-row-major",
            weightAccessPolicyOverride: canonicalRowMajorDecodeOverride()
        )

        print("[DecodeWeightLayout] step=2 baseline argmax=\(baseline.argmax) ref=\(baseline.refArgmax) logitsErr=\(String(format: "%.4f", baseline.logitsMaxErr)) finalHiddenErr=\(String(format: "%.4f", baseline.finalHiddenMaxErr))")
        print("[DecodeWeightLayout] step=2 canonical-row-major argmax=\(canonical.argmax) ref=\(canonical.refArgmax) logitsErr=\(String(format: "%.4f", canonical.logitsMaxErr)) finalHiddenErr=\(String(format: "%.4f", canonical.finalHiddenMaxErr))")

        #expect(baseline.refArgmax == canonical.refArgmax)
    }

    @Test("Decode step 1 conv-state injection diagnostic")
    func decodeStep1ConvStateInjectionDiagnostic() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let baseline = try runDecodeDiagnostic(step: 1, tag: "baseline")
        let injected = try runDecodeDiagnostic(
            step: 1,
            tag: "python-conv-state",
            beforeFinalStep: { model, ref in
                try injectReferenceConvState(step: 0, into: &model, ref: ref)
            }
        )

        print("[DecodeConvState] step=1 baseline argmax=\(baseline.argmax) ref=\(baseline.refArgmax) logitsErr=\(String(format: "%.4f", baseline.logitsMaxErr)) finalHiddenErr=\(String(format: "%.4f", baseline.finalHiddenMaxErr))")
        print("[DecodeConvState] step=1 python-conv-state argmax=\(injected.argmax) ref=\(injected.refArgmax) logitsErr=\(String(format: "%.4f", injected.logitsMaxErr)) finalHiddenErr=\(String(format: "%.4f", injected.finalHiddenMaxErr))")

        #expect(baseline.refArgmax == injected.refArgmax)
    }

    @Test("Decode step 2 conv-state injection diagnostic")
    func decodeStep2ConvStateInjectionDiagnostic() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let baseline = try runDecodeDiagnostic(step: 2, tag: "baseline")
        let injected = try runDecodeDiagnostic(
            step: 2,
            tag: "python-conv-state",
            beforeFinalStep: { model, ref in
                try injectReferenceConvState(step: 1, into: &model, ref: ref)
            }
        )

        print("[DecodeConvState] step=2 baseline argmax=\(baseline.argmax) ref=\(baseline.refArgmax) logitsErr=\(String(format: "%.4f", baseline.logitsMaxErr)) finalHiddenErr=\(String(format: "%.4f", baseline.finalHiddenMaxErr))")
        print("[DecodeConvState] step=2 python-conv-state argmax=\(injected.argmax) ref=\(injected.refArgmax) logitsErr=\(String(format: "%.4f", injected.logitsMaxErr)) finalHiddenErr=\(String(format: "%.4f", injected.finalHiddenMaxErr))")

        #expect(baseline.refArgmax == injected.refArgmax)
    }

    private func runDecodeDiagnostic(
        step: Int,
        tag: String,
        optimizer: (any DispatchOptimizer)? = nil,
        weightAccessPolicyOverride: ProjectionWeightAccessPolicyOverride? = nil,
        beforeFinalStep: ((inout MetalInferenceModel, MetalWeightFile) throws -> Void)? = nil,
        mutatePlan: ((MetalDispatchPlan) -> MetalDispatchPlan)? = nil,
        visibilityOptions: MTL4VisibilityOptions = []
    ) throws -> StepDiagnostic {
        let env = try Self.buildEnvironment(
            optimizer: optimizer,
            weightAccessPolicyOverride: weightAccessPolicyOverride
        )
        var model = env.model
        if let mutatePlan {
            let mutatedPlan = mutatePlan(model.decodePlan)
            var mutatedModel = try MetalInferenceModel(plan: mutatedPlan, device: env.device)
            mutatedModel.prefillPlan = model.prefillPlan
            model = mutatedModel
        }

        let firstToken = model.prefill(tokens: Self.promptTokens)
        var currentToken = firstToken
        for currentStep in 0...step {
            if currentStep == step, let beforeFinalStep {
                try beforeFinalStep(&model, env.ref)
            }
            if currentStep == step, visibilityOptions != [] {
                currentToken = try runDecodeStep(
                    model: &model,
                    tokenID: currentToken,
                    visibilityOptions: visibilityOptions
                )
            } else {
                currentToken = model.decodeSync(tokenID: currentToken)
            }
        }

        let refLogits = try readRefTensorAsFloats(env.ref, name: "ref.decode_\(step).logits")
        let metalLogits = readDecodeBuffer(model.buffers.logits, precision: model.buffers.bufferPrecision)
        let refHidden = try readRefTensorAsFloats(env.ref, name: "ref.decode_\(step).final_hidden")
        let metalHidden = readDecodeBuffer(
            model.decodePlan.outputHeadInputBinding().buffer,
            precision: model.buffers.bufferPrecision
        )

        return StepDiagnostic(
            step: step,
            tag: tag,
            argmax: argmax(metalLogits).index,
            refArgmax: argmax(refLogits).index,
            logitsMaxErr: maxAbsoluteError(metalLogits, refLogits),
            finalHiddenMaxErr: maxAbsoluteError(metalHidden, refHidden)
        )
    }

    private func runDecodeStep(
        model: inout MetalInferenceModel,
        tokenID: Int32,
        visibilityOptions: MTL4VisibilityOptions
    ) throws -> Int32 {
        let buffers = model.buffers
        buffers.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(model.position)
        let ropeAxes = buffers.ropePositionAxes.contents().bindMemory(to: UInt32.self, capacity: 3)
        ropeAxes[0] = UInt32(model.position)
        ropeAxes[1] = UInt32(model.position)
        ropeAxes[2] = UInt32(model.position)
        buffers.tokenIn.contents().bindMemory(to: Int32.self, capacity: 1).pointee = tokenID

        var submission = try MetalSubmissionContext(device: model.device)
        try submission.withCompute { encoder, argumentTable in
            MetalDecodeEncoder.encodeSteps(
                plan: model.decodePlan,
                encoder: encoder,
                argumentTable: argumentTable
            )
        }
        model.position += 1
        return buffers.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
    }

    private func forceBarrierPolicy(
        on plan: MetalDispatchPlan,
        barrierPolicy: MetalBarrierPolicy
    ) -> MetalDispatchPlan {
        let steps = plan.steps.enumerated().map { index, step in
            guard index > 0 else { return step }
            let descriptor = MetalDispatchDescriptor(
                pipeline: step.pipeline,
                gridSize: step.gridSize,
                threadgroupSize: step.threadgroupSize,
                threadgroupMemoryLength: step.threadgroupMemoryLength,
                barrierPolicy: barrierPolicy
            )
            return MetalDispatchStep(
                descriptor: descriptor,
                bindings: step.bindings,
                bufferAccesses: step.bufferAccesses,
                metadata: step.metadata
            )
        }
        return MetalDispatchPlan(
            steps: steps,
            buffers: plan.buffers,
            unfusedEntryCount: plan.unfusedEntryCount,
            fusedEntryCount: plan.fusedEntryCount,
            supplementalResidencyBuffers: plan.supplementalResidencyBuffers
        )
    }

    private static func buildEnvironment(
        optimizer: (any DispatchOptimizer)? = nil,
        weightAccessPolicyOverride: ProjectionWeightAccessPolicyOverride? = nil
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
        let compiler = MetalInferenceCompiler(
            optimizer: optimizer,
            weightAccessPolicyOverride: weightAccessPolicyOverride
        )
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
        return TestEnvironment(model: model, ref: ref, device: device)
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
                pointer.update(from: source.baseAddress!, count: source.count)
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
            pointer.update(from: source.baseAddress!, count: source.count)
        }
    }

    private func canonicalRowMajorDecodeOverride() -> ProjectionWeightAccessPolicyOverride {
        ProjectionWeightAccessPolicyOverride { context in
            guard context.executionPhase == .decode else {
                return nil
            }
            guard case .projection = context.entry.kind else {
                return nil
            }
            return .canonicalRowMajor
        }
    }

    private enum SetupError: Error {
        case noDevice
        case noReference
        case noSTAF
        case tensorNotFound(String)
    }
}
