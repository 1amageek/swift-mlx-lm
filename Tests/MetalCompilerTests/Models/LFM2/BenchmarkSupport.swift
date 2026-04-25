import Foundation
import Metal
import LMArchitecture
import LMIR
import ModelDeclarations
@testable import MetalCompiler
@testable import SwiftLM

enum BenchmarkSupport {
    private struct LoadedStore {
        let device: MTLDevice
        let store: STAFWeightStore
    }

    struct CollectedPrefillEntries {
        let device: MTLDevice
        let store: STAFWeightStore
        let spec: LFM25ModelSpec
        let context: CompileContext
        let walkContext: WalkContext
        let fusedEntries: [DispatchEntry]
    }

    struct SetupWithCollectedPrefillEntries {
        let model: MetalInferenceModel
        let collected: CollectedPrefillEntries
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

    private static let repositoryRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()

    static let testDataPath = repositoryRoot.appendingPathComponent("TestData").path
    static let lfmBundlePath = repositoryRoot
        .appendingPathComponent("TestData/LFM2.5-1.2B-Thinking")
        .path
    static let stafPath = repositoryRoot
        .appendingPathComponent("TestData/LFM2.5-1.2B-Thinking/model.staf")
        .path
    static let outputPath = repositoryRoot
        .appendingPathComponent("TestData/benchmark.txt")
        .path
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
    private static let lfmSpecLock = NSLock()
    nonisolated(unsafe) private static var lfmSpecCache: LFM25ModelSpec?

    struct LFM25ModelSpec {
        let modelType: String
        let config: ModelConfig
        let resolved: ModelGraph
        let convention: any WeightNamingConvention
    }

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
                  let projection = context.entry.fragment as? LinearFragment,
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
                  let projection = context.entry.fragment as? LinearFragment,
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
                  let projection = context.entry.fragment as? LinearFragment,
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
                  let projection = context.entry.fragment as? LinearFragment,
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
        weightAccessPolicyOverride: ProjectionWeightAccessPolicyOverride? = nil,
        decodeBufferPrecisionOverride: BufferPrecision? = nil,
        useCachedStore: Bool = true
    ) throws -> (MetalInferenceModel, STAFWeightStore) {
        try autoreleasepool {
            let (device, store): (MTLDevice, STAFWeightStore)
            if useCachedStore {
                (device, store) = try loadStoreOrSkip()
            } else {
                (device, store) = try loadFreshStoreOrSkip()
            }
            let spec = try loadLFM25ModelSpec()

            let compiler: MetalInferenceCompiler
            if let weightAccessPolicyOverride {
                compiler = MetalInferenceCompiler(
                    weightAccessPolicyOverride: weightAccessPolicyOverride,
                    decodeBufferPrecisionOverride: decodeBufferPrecisionOverride
                )
            } else if let decodeBufferPrecisionOverride {
                compiler = MetalInferenceCompiler(
                    decodeBufferPrecisionOverride: decodeBufferPrecisionOverride
                )
            } else {
                compiler = MetalInferenceCompiler()
            }
            let decodePlan = try compiler.compile(
                graph: spec.resolved,
                hiddenSize: spec.config.hiddenSize,
                intermediateSize: spec.config.intermediateSize,
                vocabSize: spec.config.vocabSize,
                stafWeightStore: store,
                device: device)
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
                device: device)

            var model = try MetalInferenceModel(plan: decodePlan, device: device)
            model.prefillPlan = prefillPlan

            return (model, store)
        }
    }

    static func collectPrefillEntriesOrSkip(
        weightAccessPolicyOverride: ProjectionWeightAccessPolicyOverride? = nil,
        decodeBufferPrecisionOverride: BufferPrecision? = nil,
        useCachedStore: Bool = true
    ) throws -> CollectedPrefillEntries {
        try autoreleasepool {
            let (device, store): (MTLDevice, STAFWeightStore)
            if useCachedStore {
                (device, store) = try loadStoreOrSkip()
            } else {
                (device, store) = try loadFreshStoreOrSkip()
            }
            let spec = try loadLFM25ModelSpec()

            let kernelNameResolver = MetalKernelNameResolver(
                stafWeightStore: store,
                weightAccessPolicyOverride: weightAccessPolicyOverride
            )
            let context = CompileContext(
                graph: spec.resolved,
                hiddenSize: spec.config.hiddenSize,
                intermediateSize: spec.config.intermediateSize,
                vocabSize: spec.config.vocabSize,
                inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
                stafWeightStore: store,
                device: device,
                weightFormat: kernelNameResolver.resolveModelWeightFormat(),
                decodeBufferPrecision: decodeBufferPrecisionOverride ?? kernelNameResolver.preferredDecodeBufferPrecision(
                    for: kernelNameResolver.resolveModelWeightFormat()
                ),
                accessPolicyResolver: ProjectionWeightAccessPolicyResolver(
                    override: weightAccessPolicyOverride
                )
            )
            let collector = MetalEntryCollector()
            let optimization = collector.collect(
                using: context,
                kernelContext: context.prefillKernelContext
            )
            return CollectedPrefillEntries(
                device: device,
                store: store,
                spec: spec,
                context: context,
                walkContext: optimization.walkContext,
                fusedEntries: optimization.fusedEntries
            )
        }
    }

    static func setupWithCollectedPrefillEntriesOrSkip(
        weightAccessPolicyOverride: ProjectionWeightAccessPolicyOverride? = nil,
        decodeBufferPrecisionOverride: BufferPrecision? = nil,
        useCachedStore: Bool = true
    ) throws -> SetupWithCollectedPrefillEntries {
        try autoreleasepool {
            let (device, store): (MTLDevice, STAFWeightStore)
            if useCachedStore {
                (device, store) = try loadStoreOrSkip()
            } else {
                (device, store) = try loadFreshStoreOrSkip()
            }
            let spec = try loadLFM25ModelSpec()

            let compiler: MetalInferenceCompiler
            if let weightAccessPolicyOverride {
                compiler = MetalInferenceCompiler(
                    weightAccessPolicyOverride: weightAccessPolicyOverride,
                    decodeBufferPrecisionOverride: decodeBufferPrecisionOverride
                )
            } else if let decodeBufferPrecisionOverride {
                compiler = MetalInferenceCompiler(
                    decodeBufferPrecisionOverride: decodeBufferPrecisionOverride
                )
            } else {
                compiler = MetalInferenceCompiler()
            }
            let decodePlan = try compiler.compile(
                graph: spec.resolved,
                hiddenSize: spec.config.hiddenSize,
                intermediateSize: spec.config.intermediateSize,
                vocabSize: spec.config.vocabSize,
                stafWeightStore: store,
                device: device)
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
                device: device)

            var model = try MetalInferenceModel(plan: decodePlan, device: device)
            model.prefillPlan = prefillPlan

            let kernelNameResolver = MetalKernelNameResolver(
                stafWeightStore: store,
                weightAccessPolicyOverride: weightAccessPolicyOverride
            )
            let context = CompileContext(
                graph: spec.resolved,
                hiddenSize: spec.config.hiddenSize,
                intermediateSize: spec.config.intermediateSize,
                vocabSize: spec.config.vocabSize,
                inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
                stafWeightStore: store,
                device: device,
                weightFormat: kernelNameResolver.resolveModelWeightFormat(),
                decodeBufferPrecision: decodeBufferPrecisionOverride ?? kernelNameResolver.preferredDecodeBufferPrecision(
                    for: kernelNameResolver.resolveModelWeightFormat()
                ),
                accessPolicyResolver: ProjectionWeightAccessPolicyResolver(
                    override: weightAccessPolicyOverride
                )
            )
            let collector = MetalEntryCollector()
            let optimization = collector.collect(
                using: context,
                kernelContext: context.prefillKernelContext
            )

            return SetupWithCollectedPrefillEntries(
                model: model,
                collected: CollectedPrefillEntries(
                    device: device,
                    store: store,
                    spec: spec,
                    context: context,
                    walkContext: optimization.walkContext,
                    fusedEntries: optimization.fusedEntries
                )
            )
        }
    }

    static func loadLFM25ModelSpec() throws -> LFM25ModelSpec {
        lfmSpecLock.lock()
        defer { lfmSpecLock.unlock() }

        if let cached = lfmSpecCache {
            return cached
        }

        let configURL = URL(fileURLWithPath: lfmBundlePath).appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let decoder = HFConfigDecoder()
        let modelType = try decoder.modelType(from: configData)
        let config = try decoder.decode(from: configData)
        let resolver = ModelGraphResolver()
        let graph = try resolver.resolveModelGraph(modelType: modelType, config: config)
        let convention = resolver.namingConvention(for: modelType)
        let resolved = ParameterResolver().resolve(graph: graph, convention: convention)
        let spec = LFM25ModelSpec(
            modelType: modelType,
            config: config,
            resolved: resolved,
            convention: convention
        )
        lfmSpecCache = spec
        return spec
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

    static func loadFreshStoreOrSkip() throws -> (MTLDevice, STAFWeightStore) {
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
        return (device, store)
    }

    static func profileDecodeSteps(
        model: inout MetalInferenceModel,
        device: MTLDevice,
        iterations: Int,
        filter: (MetalDispatchStep) -> Bool
    ) throws -> [StepProfile] {
        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        var currentToken = model.prefill(tokens: promptTokens)
        for _ in 0..<3 { currentToken = model.decodeSync(tokenID: currentToken) }

        let steps = model.decodePlan.steps.enumerated().filter { filter($0.element) }
        model.buffers.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(model.position)
        model.buffers.tokenIn.contents().bindMemory(to: Int32.self, capacity: 1).pointee = currentToken

        var profiles: [StepProfile] = steps.map { index, step in
            StepProfile(
                index: index,
                kernelName: step.pipeline.label ?? "step_\(index)",
                gridSize: step.gridSize,
                threadgroupSize: step.threadgroupSize)
        }

        var submission = try MetalSubmissionContext(device: device)

        for (_, step) in steps {
            try executeProfiledStep(step, submission: &submission)
        }

        for _ in 0..<iterations {
            for (profileIndex, (_, step)) in steps.enumerated() {
                let elapsedMicroseconds = try executeProfiledStep(step, submission: &submission)
                profiles[profileIndex].totalMicroseconds += elapsedMicroseconds
            }
        }

        return profiles
    }

    @discardableResult
    static func executeProfiledStep(
        _ step: MetalDispatchStep,
        submission: inout MetalSubmissionContext
    ) throws -> Double {
        let timing = try submission.withComputeTimed { encoder, argumentTable in
            MetalDecodeEncoder.encodeStep(
                step: step,
                encoder: encoder,
                argumentTable: argumentTable
            )
        }
        return (timing.gpuEndTime - timing.gpuStartTime) * 1_000_000
    }

    static func measureDecodeSyncBreakdown(
        model: inout MetalInferenceModel,
        iterations: Int
    ) throws -> DecodeSyncBreakdown {
        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        var currentToken = model.prefill(tokens: promptTokens)
        for _ in 0..<3 { currentToken = model.decodeSync(tokenID: currentToken) }

        var breakdown = DecodeSyncBreakdown()

        for _ in 0..<iterations {
            let totalStart = currentTimeNanoseconds()

            // decodeSyncTimed handles CPU write, encode, submit, and wait internally.
            // Granular CPU write vs encode vs wait breakdown is not available with Metal 4
            // reusable command buffer. Measure total wall time and GPU time.
            let encodeStart = currentTimeNanoseconds()
            let timedResult = model.decodeSyncTimed(tokenID: currentToken)
            let submitEnd = currentTimeNanoseconds()

            currentToken = timedResult.token
            let readEnd = currentTimeNanoseconds()

            breakdown.encodeSubmitMicroseconds += elapsedMicroseconds(from: encodeStart, to: submitEnd)
            breakdown.readbackMicroseconds += elapsedMicroseconds(from: submitEnd, to: readEnd)
            breakdown.totalMicroseconds += elapsedMicroseconds(from: totalStart, to: readEnd)
            breakdown.gpuMicroseconds += (timedResult.gpuEndTime - timedResult.gpuStartTime) * 1_000_000
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

    // MARK: - Generic Bundle Setup

    private static let bundleStoreLock = NSLock()
    nonisolated(unsafe) private static var bundleStoreCache: [String: LoadedStore] = [:]

    /// Set up a model from a HuggingFace-format bundle directory.
    ///
    /// Reads config.json, resolves the model graph via `ModelGraphResolver`,
    /// loads/converts STAF weights, and compiles decode + prefill plans.
    static func setupFromBundle(
        bundlePath: String,
        maximumPrefillLength: Int = 64
    ) throws -> (MetalInferenceModel, STAFWeightStore, String) {
        try setupFromBundle(
            bundlePath: bundlePath,
            inferencePolicy: InferencePolicy(maximumSequenceLength: maximumPrefillLength)
        )
    }

    static func setupFromBundle(
        bundlePath: String,
        inferencePolicy: InferencePolicy
    ) throws -> (MetalInferenceModel, STAFWeightStore, String) {
        let (device, store) = try loadBundleStoreOrSkip(bundlePath: bundlePath)

        let configURL = URL(fileURLWithPath: bundlePath).appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let decoder = HFConfigDecoder()
        let modelType = try decoder.modelType(from: configData)
        let config = try decoder.decode(from: configData)

        let resolver = ModelGraphResolver()
        let graph = try resolver.resolveModelGraph(modelType: modelType, config: config)
        let convention = resolver.namingConvention(for: modelType)
        let resolved = ParameterResolver().resolve(graph: graph, convention: convention)

        let compiler = MetalInferenceCompiler()
        let compiled = try compiler.compile(
            graph: resolved,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: inferencePolicy,
            stafWeightStore: store,
            device: device
        )
        let prefillPlan = try compiler.compilePrefill(
            graph: resolved,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: inferencePolicy,
            stafWeightStore: store,
            sharedKVCache: compiled.decodePlan.buffers.kvCache,
            sharedConvState: compiled.decodePlan.buffers.convState,
            sharedConvStateDimension: compiled.decodePlan.buffers.convStateDimension,
            sharedConvStateKernelSize: compiled.decodePlan.buffers.convStateKernelSize,
            sharedRecurrentState: compiled.decodePlan.buffers.recurrentState,
            sharedRecurrentStateBytesPerLayer: compiled.decodePlan.buffers.recurrentStateBytesPerLayer,
            device: device
        )
        let finalCompiled = compiled.withPrefillPlan(prefillPlan)
        let model = try MetalInferenceModel(compiledModel: finalCompiled, device: device)

        return (model, store, modelType)
    }

    static func loadBundleStoreOrSkip(bundlePath: String) throws -> (MTLDevice, STAFWeightStore) {
        bundleStoreLock.lock()
        defer { bundleStoreLock.unlock() }

        if let cached = bundleStoreCache[bundlePath] {
            return (cached.device, cached.store)
        }

        guard let device = MTLCreateSystemDefaultDevice() else { throw BenchError.noDevice }

        let bundleURL = URL(fileURLWithPath: bundlePath)
        let stafURL = bundleURL.appendingPathComponent("model.staf")
        let safetensorsURLs = try resolveSafetensorsFiles(in: bundleURL)
        guard !safetensorsURLs.isEmpty else { throw BenchError.noModel }

        // Reconvert when the STAF file is missing OR stale (e.g. header
        // format version changed). STAFConverter.isValid() checks header
        // magic, format version, table layout, and safetensors mtime.
        let converter = STAFConverter()
        let needsConversion: Bool
        if FileManager.default.fileExists(atPath: stafURL.path) {
            needsConversion = !(try converter.isValid(
                stafURL: stafURL,
                safetensorsURLs: safetensorsURLs,
                expectedMetadata: nil
            ))
        } else {
            needsConversion = true
        }

        if needsConversion {
            let configURL = bundleURL.appendingPathComponent("config.json")
            let quantization: MLXQuantizationHint?
            if FileManager.default.fileExists(atPath: configURL.path) {
                let configData = try Data(contentsOf: configURL)
                quantization = try HFConfigDecoder().quantizationHint(from: configData)
            } else {
                quantization = nil
            }
            try converter.convert(
                safetensorsURLs: safetensorsURLs,
                outputURL: stafURL,
                quantization: quantization
            )
        }

        let store = try STAFLoader().load(at: stafURL, device: device)
        bundleStoreCache[bundlePath] = LoadedStore(device: device, store: store)
        return (device, store)
    }

    /// Resolve safetensors files: single model.safetensors or sharded via index.
    private static func resolveSafetensorsFiles(in bundleURL: URL) throws -> [URL] {
        let singleURL = bundleURL.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: singleURL.path) {
            return [singleURL]
        }
        let indexURL = bundleURL.appendingPathComponent("model.safetensors.index.json")
        guard FileManager.default.fileExists(atPath: indexURL.path) else {
            return []
        }
        let indexData = try Data(contentsOf: indexURL)
        guard let indexJSON = try JSONSerialization.jsonObject(with: indexData) as? [String: Any],
              let weightMap = indexJSON["weight_map"] as? [String: String] else {
            return []
        }
        let shardFiles = Set(weightMap.values)
        let shardURLs = shardFiles.sorted().map { bundleURL.appendingPathComponent($0) }
        for url in shardURLs {
            guard FileManager.default.fileExists(atPath: url.path) else { return [] }
        }
        return shardURLs
    }

    enum BenchError: Error {
        case noDevice
        case noModel
    }
}
