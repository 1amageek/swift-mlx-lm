import Foundation
import Metal
import LMArchitecture
import LMIR
import ModelDeclarations
@testable import MetalCompiler

enum BenchmarkSupport {
    private struct LoadedStore {
        let device: MTLDevice
        let store: STAFWeightStore
    }

    struct StepProfile {
        let index: Int
        let kernelName: String
        let gridSize: MTLSize
        let threadgroupSize: MTLSize
        var totalMicroseconds: Double = 0
    }

    struct DecodeSyncBreakdown {
        var cpuWriteMicroseconds: Double = 0
        var encodeSubmitMicroseconds: Double = 0
        var waitMicroseconds: Double = 0
        var readbackMicroseconds: Double = 0
        var gpuMicroseconds: Double = 0
        var totalMicroseconds: Double = 0

        func averaged(over iterations: Int) -> Self {
            let scale = 1.0 / Double(iterations)
            return Self(
                cpuWriteMicroseconds: cpuWriteMicroseconds * scale,
                encodeSubmitMicroseconds: encodeSubmitMicroseconds * scale,
                waitMicroseconds: waitMicroseconds * scale,
                readbackMicroseconds: readbackMicroseconds * scale,
                gpuMicroseconds: gpuMicroseconds * scale,
                totalMicroseconds: totalMicroseconds * scale
            )
        }

        var hostOverheadMicroseconds: Double {
            totalMicroseconds - gpuMicroseconds
        }
    }

    static let stafPath = "/Users/1amageek/Desktop/swift-lm/TestData/LFM2.5-1.2B-Thinking/model.staf"
    static let outputPath = "/Users/1amageek/Desktop/swift-lm/TestData/benchmark.txt"
    static let squareQProjBlockedSafePrefixTensorNames: Set<String> = [
        "model.layers.10.self_attn.q_proj.weight",
        "model.layers.12.self_attn.q_proj.weight",
        "model.layers.14.self_attn.q_proj.weight",
    ]
    static let squareQProjBlockedSafePrefix2TensorNames: Set<String> = [
        "model.layers.10.self_attn.q_proj.weight",
        "model.layers.12.self_attn.q_proj.weight",
    ]
    static let squareSingleQProjBlockedTensorName = "model.layers.10.self_attn.q_proj.weight"
    private static let loadedStoreLock = NSLock()
    nonisolated(unsafe) private static var loadedStoreCache: LoadedStore?

    static func isHotExactShapeGEMVKernel(_ name: String) -> Bool {
        (name.hasPrefix("gemv_2048_6144") || name.hasPrefix("gemv_2048_sq")) &&
        name.hasSuffix("_argbuf")
    }

    static func isHot6144GEMVKernel(_ name: String) -> Bool {
        name.hasPrefix("gemv_2048_6144") && name.hasSuffix("_argbuf")
    }

    static func isHotSquareGEMVKernel(_ name: String) -> Bool {
        name.hasPrefix("gemv_2048_sq") && name.hasSuffix("_argbuf")
    }

    static func canonicalHotExactShapeGEMVFamily(for name: String) -> String {
        if name.hasPrefix("gemv_2048_6144") {
            return "gemv_2048_6144_bf16_argbuf"
        }
        if name.hasPrefix("gemv_2048_sq") {
            return "gemv_2048_sq_bf16_argbuf"
        }
        return name
    }

    static func blocked6144DecodeOverride() -> ProjectionWeightAccessPolicyOverride {
        ProjectionWeightAccessPolicyOverride { context in
            guard context.executionPhase == .decode else {
                return nil
            }
            guard context.schemeIdentifier == .bf16RowMajor,
                  context.role == "in_proj",
                  case .projection(let projection, _) = context.entry.kind,
                  projection.inputDimension == 2_048,
                  projection.outputDimension == 6_144 else {
                return nil
            }
            return .optimized(.blockedRows8Tiles128)
        }
    }

    static func blocked4x1286144DecodeOverride() -> ProjectionWeightAccessPolicyOverride {
        ProjectionWeightAccessPolicyOverride { context in
            guard context.executionPhase == .decode else {
                return nil
            }
            guard context.schemeIdentifier == .bf16RowMajor,
                  context.role == "in_proj",
                  case .projection(let projection, _) = context.entry.kind,
                  projection.inputDimension == 2_048,
                  projection.outputDimension == 6_144 else {
                return nil
            }
            return .optimized(.blockedRows4Tiles128)
        }
    }

    static func blocked8x128SquareDecodeOverride() -> ProjectionWeightAccessPolicyOverride {
        ProjectionWeightAccessPolicyOverride { context in
            guard context.executionPhase == .decode else {
                return nil
            }
            guard context.schemeIdentifier == .bf16RowMajor,
                  case .projection(let projection, _) = context.entry.kind,
                  projection.inputDimension == 2_048,
                  projection.outputDimension == 2_048 else {
                return nil
            }
            return .optimized(.blockedRows8Tiles128)
        }
    }

    static func blocked8x128SquareQProjDecodeOverride() -> ProjectionWeightAccessPolicyOverride {
        ProjectionWeightAccessPolicyOverride { context in
            guard context.executionPhase == .decode else {
                return nil
            }
            guard context.schemeIdentifier == .bf16RowMajor,
                  context.role == "q_proj",
                  case .projection(let projection, _) = context.entry.kind,
                  projection.inputDimension == 2_048,
                  projection.outputDimension == 2_048 else {
                return nil
            }
            return .optimized(.blockedRows8Tiles128)
        }
    }

    static func blocked8x128SquareQProjPrefix3DecodeOverride() -> ProjectionWeightAccessPolicyOverride {
        ProjectionWeightAccessPolicyOverride.prefer(
            .optimized(.blockedRows8Tiles128),
            forTensorNames: squareQProjBlockedSafePrefixTensorNames
        )
    }

    static func blocked8x128SquareQProjPrefix2DecodeOverride() -> ProjectionWeightAccessPolicyOverride {
        ProjectionWeightAccessPolicyOverride.prefer(
            .optimized(.blockedRows8Tiles128),
            forTensorNames: squareQProjBlockedSafePrefix2TensorNames
        )
    }

    static func blocked8x128SquareSingleQProjDecodeOverride() -> ProjectionWeightAccessPolicyOverride {
        ProjectionWeightAccessPolicyOverride.prefer(
            .optimized(.blockedRows8Tiles128),
            forTensorNames: [squareSingleQProjBlockedTensorName]
        )
    }

    static func settleGPU() {
        Thread.sleep(forTimeInterval: 0.2)
    }

    static func currentTimeNanoseconds() -> UInt64 {
        DispatchTime.now().uptimeNanoseconds
    }

    static func elapsedMicroseconds(from start: UInt64, to end: UInt64) -> Double {
        Double(end - start) / 1_000.0
    }

    static func setupOrSkip(
        optimizer: (any DispatchOptimizer)? = nil,
        weightAccessPolicyOverride: ProjectionWeightAccessPolicyOverride? = nil
    ) throws -> (MetalInferenceModel, STAFWeightStore) {
        let (device, store) = try loadStoreOrSkip()

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

        let compiler: MetalInferenceCompiler
        if let weightAccessPolicyOverride {
            compiler = MetalInferenceCompiler(
                optimizer: optimizer,
                weightAccessPolicyOverride: weightAccessPolicyOverride
            )
        } else {
            compiler = MetalInferenceCompiler(optimizer: optimizer)
        }
        let decodePlan = try compiler.compile(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, stafWeightStore: store, device: device)
        let prefillPlan = try compiler.compilePrefill(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, maximumSequenceLength: 64,
            stafWeightStore: store,
            sharedKVCache: decodePlan.buffers.kvCache,
            sharedConvState: decodePlan.buffers.convState,
            sharedConvStateDimension: decodePlan.buffers.convStateDimension,
            sharedConvStateKernelSize: decodePlan.buffers.convStateKernelSize,
            device: device)

        var model = try MetalInferenceModel(plan: decodePlan, device: device)
        model.prefillPlan = prefillPlan

        return (model, store)
    }

    static func loadStoreOrSkip() throws -> (MTLDevice, STAFWeightStore) {
        loadedStoreLock.lock()
        defer { loadedStoreLock.unlock() }

        if let cached = loadedStoreCache {
            return (cached.device, cached.store)
        }

        guard let device = MTLCreateSystemDefaultDevice() else { throw BenchError.noDevice }

        let stafURL = URL(fileURLWithPath: stafPath)
        if !FileManager.default.fileExists(atPath: stafURL.path) {
            let safetensorsURL = stafURL.deletingLastPathComponent()
                .appendingPathComponent("model.safetensors")
            guard FileManager.default.fileExists(atPath: safetensorsURL.path) else {
                throw BenchError.noModel
            }
            try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)
        }
        let store = try STAFLoader().load(at: stafURL, device: device)
        loadedStoreCache = LoadedStore(device: device, store: store)
        return (device, store)
    }

    static func profileDecodeSteps(
        model: inout MetalInferenceModel,
        queue: MTLCommandQueue,
        iterations: Int,
        filter: (MetalDispatchStep) -> Bool
    ) throws -> [StepProfile] {
        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        var currentToken = model.prefill(tokens: promptTokens)
        for _ in 0..<3 { currentToken = model.decodeSync(tokenID: currentToken) }

        let steps = model.plan.steps.enumerated().filter { filter($0.element) }
        model.plan.buffers.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(model.position)
        model.plan.buffers.tokenIn.contents().bindMemory(to: Int32.self, capacity: 1).pointee = currentToken

        var profiles: [StepProfile] = steps.map { index, step in
            StepProfile(
                index: index,
                kernelName: step.pipeline.label ?? "step_\(index)",
                gridSize: step.gridSize,
                threadgroupSize: step.threadgroupSize)
        }

        for (_, step) in steps {
            try executeProfiledStep(step, queue: queue)
        }

        for _ in 0..<iterations {
            for (profileIndex, (_, step)) in steps.enumerated() {
                let elapsedMicroseconds = try executeProfiledStep(step, queue: queue)
                profiles[profileIndex].totalMicroseconds += elapsedMicroseconds
            }
        }

        return profiles
    }

    @discardableResult
    static func executeProfiledStep(
        _ step: MetalDispatchStep,
        queue: MTLCommandQueue
    ) throws -> Double {
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw BenchError.noDevice
        }
        if step.sync == .bufferBarrier {
            encoder.memoryBarrier(scope: .buffers)
        }
        encoder.setComputePipelineState(step.pipeline)
        for (index, buffer, offset) in step.bufferBindings {
            encoder.setBuffer(buffer, offset: offset, index: index)
        }
        for (index, value) in step.bytesBindings {
            value.withUnsafeBufferPointer { pointer in
                if let baseAddress = pointer.baseAddress {
                    encoder.setBytes(baseAddress, length: pointer.count, index: index)
                }
            }
        }
        if step.threadgroupMemoryLength > 0 {
            encoder.setThreadgroupMemoryLength(step.threadgroupMemoryLength, index: 0)
        }
        encoder.dispatchThreadgroups(step.gridSize, threadsPerThreadgroup: step.threadgroupSize)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1_000_000
    }

    static func measureDecodeSyncBreakdown(
        model: inout MetalInferenceModel,
        iterations: Int
    ) throws -> DecodeSyncBreakdown {
        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        var currentToken = model.prefill(tokens: promptTokens)
        for _ in 0..<3 { currentToken = model.decodeSync(tokenID: currentToken) }

        let buffers = model.plan.buffers
        var breakdown = DecodeSyncBreakdown()

        for _ in 0..<iterations {
            let totalStart = currentTimeNanoseconds()

            let writeStart = currentTimeNanoseconds()
            buffers.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(model.position)
            buffers.tokenIn.contents().bindMemory(to: Int32.self, capacity: 1).pointee = currentToken
            let writeEnd = currentTimeNanoseconds()

            let encodeStart = currentTimeNanoseconds()
            guard let commandBuffer = model.commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw BenchError.noDevice
            }
            for step in model.plan.steps {
                step.bindings.bind(to: encoder)
                step.descriptor.encode(on: encoder)
            }
            encoder.endEncoding()
            commandBuffer.commit()
            let submitEnd = currentTimeNanoseconds()

            commandBuffer.waitUntilCompleted()
            let waitEnd = currentTimeNanoseconds()

            currentToken = buffers.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
            model.position += 1
            let readEnd = currentTimeNanoseconds()

            breakdown.cpuWriteMicroseconds += elapsedMicroseconds(from: writeStart, to: writeEnd)
            breakdown.encodeSubmitMicroseconds += elapsedMicroseconds(from: encodeStart, to: submitEnd)
            breakdown.waitMicroseconds += elapsedMicroseconds(from: submitEnd, to: waitEnd)
            breakdown.readbackMicroseconds += elapsedMicroseconds(from: waitEnd, to: readEnd)
            breakdown.totalMicroseconds += elapsedMicroseconds(from: totalStart, to: readEnd)
            breakdown.gpuMicroseconds += (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1_000_000
        }

        return breakdown.averaged(over: iterations)
    }

    static func decodeTokenTrace(
        model: inout MetalInferenceModel,
        promptTokens: [Int32],
        predecodeSteps: Int = 0,
        decodeSteps: Int
    ) -> [Int32] {
        var trace: [Int32] = []
        var currentToken = model.prefill(tokens: promptTokens)
        trace.append(currentToken)
        for _ in 0..<predecodeSteps {
            currentToken = model.decodeSync(tokenID: currentToken)
            trace.append(currentToken)
        }
        for _ in 0..<decodeSteps {
            currentToken = model.decodeSync(tokenID: currentToken)
            trace.append(currentToken)
        }
        return trace
    }

    enum BenchError: Error {
        case noDevice
        case noModel
    }
}
