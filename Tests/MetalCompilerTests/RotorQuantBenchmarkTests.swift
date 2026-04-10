import Foundation
import Metal
import Testing
@testable import MetalCompiler
@testable import LMIR
@testable import LMArchitecture
@testable import ModelDeclarations

/// Benchmark tests comparing KV cache quantization strategies.
///
/// Measures throughput, memory, and quality impact of RotorQuant
/// (Hadamard-rotated quantization) versus baseline FP16 and raw Q8 KV caches.
@Suite("RotorQuant KV Cache Benchmark", .serialized)
struct RotorQuantBenchmarkTests {

    // MARK: - Setup Helpers

    /// Compile a model with a specific KV cache quantization policy.
    private static func setupWithPolicy(
        inferencePolicy: InferencePolicy,
        optimizer: (any DispatchOptimizer)? = nil
    ) throws -> (MetalInferenceModel, STAFWeightStore) {
        let (device, store) = try BenchmarkSupport.loadStoreOrSkip()

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

        let compiler = MetalInferenceCompiler(optimizer: optimizer)
        let decodePlan = try compiler.compile(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, inferencePolicy: inferencePolicy,
            stafWeightStore: store, device: device)
        let prefillPlan = try compiler.compilePrefill(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, inferencePolicy: inferencePolicy,
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

    private static func printPerKernelProfile(
        title: String,
        profiles: [BenchmarkSupport.StepProfile],
        iterations: Int
    ) {
        struct KernelAggregate {
            var totalMicroseconds: Double = 0
            var count: Int = 0
            var gridSample: MTLSize = .init()
            var tgSample: MTLSize = .init()
        }

        var aggregates: [String: KernelAggregate] = [:]
        let totalMicroseconds = profiles.reduce(0.0) { $0 + $1.totalMicroseconds }
        let avgTotalUs = totalMicroseconds / Double(iterations)

        for profile in profiles {
            let avgUs = profile.totalMicroseconds / Double(iterations)
            aggregates[profile.kernelName, default: KernelAggregate()].totalMicroseconds += avgUs
            aggregates[profile.kernelName, default: KernelAggregate()].count += 1
            if aggregates[profile.kernelName]?.gridSample.width == 0 {
                aggregates[profile.kernelName]?.gridSample = profile.gridSize
                aggregates[profile.kernelName]?.tgSample = profile.threadgroupSize
            }
        }

        let sorted = aggregates.sorted { $0.value.totalMicroseconds > $1.value.totalMicroseconds }

        print("\n=== \(title) ===")
        print("Total steps: \(profiles.count)")
        print("Total: \(String(format: "%.0f", avgTotalUs)) us (\(String(format: "%.1f", avgTotalUs / 1000)) ms)")
        print("")
        let header = "Kernel".padding(toLength: 40, withPad: " ", startingAt: 0)
            + "Count  Total us     %  Grid          TG"
        print(header)
        print(String(repeating: "-", count: 100))

        for (name, aggregate) in sorted.prefix(15) {
            let pct = aggregate.totalMicroseconds / avgTotalUs * 100
            let grid = "\(aggregate.gridSample.width)x\(aggregate.gridSample.height)x\(aggregate.gridSample.depth)"
            let tg = "\(aggregate.tgSample.width)"
            let pad = name.padding(toLength: 40, withPad: " ", startingAt: 0)
            print("\(pad)\(String(format: "%5d %9.0f %5.1f%%", aggregate.count, aggregate.totalMicroseconds, pct))  \(grid.padding(toLength: 14, withPad: " ", startingAt: 0))\(tg)")
        }

        let attentionMicroseconds = aggregates
            .filter { $0.key.localizedCaseInsensitiveContains("attn") }
            .reduce(0.0) { $0 + $1.value.totalMicroseconds }
        let gemvMicroseconds = aggregates
            .filter { $0.key.localizedCaseInsensitiveContains("gemv") }
            .reduce(0.0) { $0 + $1.value.totalMicroseconds }
        let projectionMicroseconds = aggregates
            .filter { $0.key.localizedCaseInsensitiveContains("projection") }
            .reduce(0.0) { $0 + $1.value.totalMicroseconds }
        print("")
        print("Attention share: \(String(format: "%.1f", attentionMicroseconds / avgTotalUs * 100))%")
        print("GEMV share:      \(String(format: "%.1f", gemvMicroseconds / avgTotalUs * 100))%")
        print("Projection share:\(String(format: "%.1f", projectionMicroseconds / avgTotalUs * 100))%")
    }

    // MARK: - KV Cache Memory Comparison

    @Test("KV cache memory: FP16 vs Q8 vs RotorQ8 vs RotorQ4")
    func kvCacheMemoryComparison() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let maxSeqLen = 4096
        let kvHeads = 8
        let headDim = 64
        let layerCount = 16

        let schemes: [(String, QuantizationSchemeIdentifier)] = [
            ("FP16", .fp16RowMajor),
            ("BF16", .bf16RowMajor),
            ("Q8g32", .q8Group32ScaleF16),
            ("RotorQ8", .rotorQ8Group32ScaleF16),
            ("Q4g64", .q4Group64ScaleF16),
            ("RotorQ4", .rotorQ4Group64ScaleF16),
        ]

        print("\n=== KV Cache Memory Comparison ===")
        print("  maxSeqLen=\(maxSeqLen)  kvHeads=\(kvHeads)  headDim=\(headDim)  layers=\(layerCount)")
        print(String(repeating: "-", count: 60))
        print("Scheme     headSlot  tokenSlot  layer(KB)  total(MB)  ratio")
        print(String(repeating: "-", count: 60))

        let fp16Spec = KVCacheSpecification(
            keyQuantizationScheme: .fp16RowMajor,
            valueQuantizationScheme: .fp16RowMajor,
            kvHeadCount: kvHeads, headDimension: headDim,
            maximumSequenceLength: maxSeqLen)
        let fp16Total = fp16Spec.totalBufferSize(scheme: .fp16RowMajor) * 2

        for (name, scheme) in schemes {
            let spec = KVCacheSpecification(
                keyQuantizationScheme: scheme,
                valueQuantizationScheme: scheme,
                kvHeadCount: kvHeads, headDimension: headDim,
                maximumSequenceLength: maxSeqLen)
            let headSlot = spec.bytesPerHeadSlot(scheme: scheme)
            let tokenSlot = spec.bytesPerTokenSlot(scheme: scheme)
            let layerBytes = spec.bytesPerLayer(scheme: scheme)
            let totalBytes = spec.totalBufferSize(scheme: scheme) * 2  // K + V
            let ratio = Double(totalBytes) / Double(fp16Total)
            let pad = name.padding(toLength: 10, withPad: " ", startingAt: 0)
            let line = String(format: "%8d %10d %10.1f %10.2f  %.2fx", headSlot, tokenSlot, Double(layerBytes) / 1024, Double(totalBytes) / 1024 / 1024, ratio)
            print("\(pad) \(line)")
        }
    }

    // MARK: - Decode Throughput Comparison

    @Test("Decode throughput: FP16 vs Q8 vs RotorQ8")
    func decodeComparisonBenchmark() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let decodeSteps = 50
        let iterations = 5
        let maxSeqLen = 256

        let policies: [(String, InferencePolicy)] = [
            ("FP16", InferencePolicy(
                maximumSequenceLength: maxSeqLen,
                kvCache: .automatic)),
            ("Q8g32", InferencePolicy(
                maximumSequenceLength: maxSeqLen,
                kvCache: KVCachePolicy(
                    keyScheme: .fixed(.q8Group32ScaleF16),
                    valueScheme: .fixed(.q8Group32ScaleF16)))),
            ("RotorQ8", InferencePolicy(
                maximumSequenceLength: maxSeqLen,
                kvCache: KVCachePolicy(
                    keyScheme: .fixed(.rotorQ8Group32ScaleF16),
                    valueScheme: .fixed(.rotorQ8Group32ScaleF16)))),
            ("RotorQ4", InferencePolicy(
                maximumSequenceLength: maxSeqLen,
                kvCache: KVCachePolicy(
                    keyScheme: .fixed(.rotorQ4Group64ScaleF16),
                    valueScheme: .fixed(.rotorQ4Group64ScaleF16)))),
        ]

        print("\n=== Decode Throughput Comparison: LFM2.5-1.2B ===")
        print("KV Policy   tok/s      ms/tok  KV(KB)  ratio")
        print(String(repeating: "-", count: 55))

        var baselineTokPerSec: Double = 0

        for (name, policy) in policies {
            let (model, _) = try Self.setupWithPolicy(inferencePolicy: policy)
            var m = model

            let kvBytes = (m.buffers.kvCache?.keys.length ?? 0)
                        + (m.buffers.kvCache?.values.length ?? 0)

            var results: [Double] = []
            BenchmarkSupport.settleGPU()

            for _ in 0..<iterations {
                m.resetCaches()
                var currentToken = m.prefill(tokens: promptTokens)
                for _ in 0..<3 { currentToken = m.decodeSync(tokenID: currentToken) }

                let start = CFAbsoluteTimeGetCurrent()
                for _ in 0..<decodeSteps {
                    currentToken = m.decodeSync(tokenID: currentToken)
                }
                results.append(CFAbsoluteTimeGetCurrent() - start)
            }

            let median = results.sorted()[iterations / 2]
            let tokPerSec = Double(decodeSteps) / median
            let msPerTok = median / Double(decodeSteps) * 1000
            if name == "FP16" { baselineTokPerSec = tokPerSec }
            let ratio = baselineTokPerSec > 0 ? tokPerSec / baselineTokPerSec : 1.0

            let pad = name.padding(toLength: 12, withPad: " ", startingAt: 0)
            let line = String(format: "%7.1f %10.2f %7.1f  %.2fx", tokPerSec, msPerTok, Double(kvBytes) / 1024, ratio)
            print("\(pad) \(line)")

            BenchmarkSupport.settleGPU()
        }
    }

    // MARK: - Prefill Throughput Comparison

    @Test("Prefill throughput: FP16 vs RotorQ8")
    func prefillComparisonBenchmark() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let prefillLengths = [16, 32, 64]
        let iterations = 5

        let policies: [(String, InferencePolicy)] = [
            ("FP16", InferencePolicy(
                maximumSequenceLength: 64,
                kvCache: .automatic)),
            ("RotorQ8", InferencePolicy(
                maximumSequenceLength: 64,
                kvCache: KVCachePolicy(
                    keyScheme: .fixed(.rotorQ8Group32ScaleF16),
                    valueScheme: .fixed(.rotorQ8Group32ScaleF16)))),
            ("RotorQ4", InferencePolicy(
                maximumSequenceLength: 64,
                kvCache: KVCachePolicy(
                    keyScheme: .fixed(.rotorQ4Group64ScaleF16),
                    valueScheme: .fixed(.rotorQ4Group64ScaleF16)))),
        ]

        print("\n=== Prefill Throughput Comparison: LFM2.5-1.2B ===")

        for (name, policy) in policies {
            let (model, _) = try Self.setupWithPolicy(inferencePolicy: policy)
            var m = model

            print("\n[\(name)]")
            print("  SeqLen  tok/s    ms")
            BenchmarkSupport.settleGPU()

            for length in prefillLengths {
                var results: [Double] = []
                for _ in 0..<iterations {
                    m.resetCaches()
                    let tokens = [Int32](repeating: 1, count: length)
                    let start = CFAbsoluteTimeGetCurrent()
                    _ = m.prefill(tokens: tokens)
                    results.append(CFAbsoluteTimeGetCurrent() - start)
                }
                let median = results.sorted()[iterations / 2]
                let tokPerSec = Double(length) / median
                print("  \(String(format: "%6d  %7.1f  %6.1f", length, tokPerSec, median * 1000))")
            }
            BenchmarkSupport.settleGPU()
        }
    }

    // MARK: - Asymmetric K/V Quantization

    @Test("Asymmetric policy: RotorQ4-K + FP16-V vs symmetric RotorQ4")
    func asymmetricPolicyBenchmark() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let decodeSteps = 50
        let iterations = 5
        let maxSeqLen = 256

        let policies: [(String, InferencePolicy)] = [
            ("FP16/FP16", InferencePolicy(
                maximumSequenceLength: maxSeqLen,
                kvCache: .automatic)),
            ("RQ4/FP16", InferencePolicy(
                maximumSequenceLength: maxSeqLen,
                kvCache: KVCachePolicy(
                    keyScheme: .fixed(.rotorQ4Group64ScaleF16),
                    valueScheme: .automatic))),
            ("RQ4/RQ8", InferencePolicy(
                maximumSequenceLength: maxSeqLen,
                kvCache: KVCachePolicy(
                    keyScheme: .fixed(.rotorQ4Group64ScaleF16),
                    valueScheme: .fixed(.rotorQ8Group32ScaleF16)))),
            ("RQ4/RQ4", InferencePolicy(
                maximumSequenceLength: maxSeqLen,
                kvCache: KVCachePolicy(
                    keyScheme: .fixed(.rotorQ4Group64ScaleF16),
                    valueScheme: .fixed(.rotorQ4Group64ScaleF16)))),
        ]

        print("\n=== Asymmetric KV Quantization: LFM2.5-1.2B ===")
        print("K/V Policy   tok/s      ms/tok  KV(KB)")
        print(String(repeating: "-", count: 50))

        for (name, policy) in policies {
            let (model, _) = try Self.setupWithPolicy(inferencePolicy: policy)
            var m = model

            let kvBytes = (m.buffers.kvCache?.keys.length ?? 0)
                        + (m.buffers.kvCache?.values.length ?? 0)

            var results: [Double] = []
            BenchmarkSupport.settleGPU()

            for _ in 0..<iterations {
                m.resetCaches()
                var currentToken = m.prefill(tokens: promptTokens)
                for _ in 0..<3 { currentToken = m.decodeSync(tokenID: currentToken) }

                let start = CFAbsoluteTimeGetCurrent()
                for _ in 0..<decodeSteps {
                    currentToken = m.decodeSync(tokenID: currentToken)
                }
                results.append(CFAbsoluteTimeGetCurrent() - start)
            }

            let median = results.sorted()[iterations / 2]
            let tokPerSec = Double(decodeSteps) / median
            let msPerTok = median / Double(decodeSteps) * 1000

            let pad = name.padding(toLength: 13, withPad: " ", startingAt: 0)
            let line = String(format: "%7.1f %10.2f %7.1f", tokPerSec, msPerTok, Double(kvBytes) / 1024)
            print("\(pad) \(line)")

            BenchmarkSupport.settleGPU()
        }
    }

    // MARK: - Gemma4 (Pure Transformer) Benchmarks

    static let gemma4BundlePath = "/Users/1amageek/Desktop/swift-lm/TestData/gemma-4-E2B-it"

    @Test("Gemma4 decode throughput: FP16 vs RotorQ8 vs RotorQ4")
    func gemma4DecodeComparison() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let decodeSteps = 50
        let iterations = 3
        let maxSeqLen = 256

        let policies: [(String, InferencePolicy)] = [
            ("FP16", InferencePolicy(
                maximumSequenceLength: maxSeqLen,
                kvCache: .automatic)),
            ("RotorQ8", InferencePolicy(
                maximumSequenceLength: maxSeqLen,
                kvCache: KVCachePolicy(
                    keyScheme: .fixed(.rotorQ8Group32ScaleF16),
                    valueScheme: .fixed(.rotorQ8Group32ScaleF16)))),
            ("RotorQ4", InferencePolicy(
                maximumSequenceLength: maxSeqLen,
                kvCache: KVCachePolicy(
                    keyScheme: .fixed(.rotorQ4Group64ScaleF16),
                    valueScheme: .fixed(.rotorQ4Group64ScaleF16)))),
            ("RQ4/FP16", InferencePolicy(
                maximumSequenceLength: maxSeqLen,
                kvCache: KVCachePolicy(
                    keyScheme: .fixed(.rotorQ4Group64ScaleF16),
                    valueScheme: .automatic))),
        ]

        print("\n=== Gemma4-E2B Decode Throughput (35 attention layers) ===")
        print("KV Policy   tok/s      ms/tok  KV(KB)  ratio")
        print(String(repeating: "-", count: 55))

        var baselineTokPerSec: Double = 0

        for (name, policy) in policies {
            let (model, _, _) = try BenchmarkSupport.setupFromBundle(
                bundlePath: Self.gemma4BundlePath,
                inferencePolicy: policy)
            var m = model

            let kvBytes = (m.buffers.kvCache?.keys.length ?? 0)
                        + (m.buffers.kvCache?.values.length ?? 0)

            var results: [Double] = []
            BenchmarkSupport.settleGPU()

            for _ in 0..<iterations {
                m.resetCaches()
                var currentToken = m.prefill(tokens: promptTokens)
                for _ in 0..<3 { currentToken = m.decodeSync(tokenID: currentToken) }

                let start = CFAbsoluteTimeGetCurrent()
                for _ in 0..<decodeSteps {
                    currentToken = m.decodeSync(tokenID: currentToken)
                }
                results.append(CFAbsoluteTimeGetCurrent() - start)
            }

            let median = results.sorted()[iterations / 2]
            let tokPerSec = Double(decodeSteps) / median
            let msPerTok = median / Double(decodeSteps) * 1000
            if name == "FP16" { baselineTokPerSec = tokPerSec }
            let ratio = baselineTokPerSec > 0 ? tokPerSec / baselineTokPerSec : 1.0

            let pad = name.padding(toLength: 12, withPad: " ", startingAt: 0)
            let line = String(format: "%7.1f %10.2f %7.1f  %.2fx", tokPerSec, msPerTok, Double(kvBytes) / 1024, ratio)
            print("\(pad) \(line)")

            BenchmarkSupport.settleGPU()
        }
    }

    @Test("Gemma4 prefill throughput: FP16 vs RotorQ8")
    func gemma4PrefillComparison() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let prefillLengths = [16, 32, 64]
        let iterations = 3

        let policies: [(String, InferencePolicy)] = [
            ("FP16", InferencePolicy(
                maximumSequenceLength: 64,
                kvCache: .automatic)),
            ("RotorQ8", InferencePolicy(
                maximumSequenceLength: 64,
                kvCache: KVCachePolicy(
                    keyScheme: .fixed(.rotorQ8Group32ScaleF16),
                    valueScheme: .fixed(.rotorQ8Group32ScaleF16)))),
        ]

        print("\n=== Gemma4-E2B Prefill Throughput (35 attention layers) ===")

        for (name, policy) in policies {
            let (model, _, _) = try BenchmarkSupport.setupFromBundle(
                bundlePath: Self.gemma4BundlePath,
                inferencePolicy: policy)
            var m = model

            print("\n[\(name)]")
            print("  SeqLen  tok/s    ms")
            BenchmarkSupport.settleGPU()

            for length in prefillLengths {
                var results: [Double] = []
                for _ in 0..<iterations {
                    m.resetCaches()
                    let tokens = [Int32](repeating: 1, count: length)
                    let start = CFAbsoluteTimeGetCurrent()
                    _ = m.prefill(tokens: tokens)
                    results.append(CFAbsoluteTimeGetCurrent() - start)
                }
                let median = results.sorted()[iterations / 2]
                let tokPerSec = Double(length) / median
                print("  \(String(format: "%6d  %7.1f  %6.1f", length, tokPerSec, median * 1000))")
            }
            BenchmarkSupport.settleGPU()
        }
    }

    // MARK: - Quality Evaluation

    @Test("Gemma4 token quality: FP16 vs Q8 vs RotorQ8 vs RotorQ4")
    func gemma4TokenQualityComparison() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let decodeSteps = 100

        // Multiple prompts to reduce single-sequence bias
        let prompts: [(String, [Int32])] = [
            ("prompt_A", [1, 1, 6, 6423, 708]),
            ("prompt_B", [1, 2, 1681, 3, 4, 5]),
            ("prompt_C", [1, 1, 1, 1, 1, 1, 1, 1]),
        ]

        let policies: [(String, InferencePolicy)] = [
            ("FP16", InferencePolicy(
                maximumSequenceLength: 256,
                kvCache: .automatic)),
            ("Q8", InferencePolicy(
                maximumSequenceLength: 256,
                kvCache: KVCachePolicy(
                    keyScheme: .fixed(.q8Group32ScaleF16),
                    valueScheme: .fixed(.q8Group32ScaleF16)))),
            ("RotorQ8", InferencePolicy(
                maximumSequenceLength: 256,
                kvCache: KVCachePolicy(
                    keyScheme: .fixed(.rotorQ8Group32ScaleF16),
                    valueScheme: .fixed(.rotorQ8Group32ScaleF16)))),
            ("RotorQ4", InferencePolicy(
                maximumSequenceLength: 256,
                kvCache: KVCachePolicy(
                    keyScheme: .fixed(.rotorQ4Group64ScaleF16),
                    valueScheme: .fixed(.rotorQ4Group64ScaleF16)))),
            ("RQ4/FP16", InferencePolicy(
                maximumSequenceLength: 256,
                kvCache: KVCachePolicy(
                    keyScheme: .fixed(.rotorQ4Group64ScaleF16),
                    valueScheme: .automatic))),
        ]

        // Collect traces: [policyName: [promptName: tokenTrace]]
        var allTraces: [(String, [(String, [Int32])])] = []

        for (policyName, policy) in policies {
            let (model, _, _) = try BenchmarkSupport.setupFromBundle(
                bundlePath: Self.gemma4BundlePath,
                inferencePolicy: policy)
            var m = model

            var promptTraces: [(String, [Int32])] = []
            for (promptName, promptTokens) in prompts {
                m.resetCaches()
                BenchmarkSupport.settleGPU()
                let trace = BenchmarkSupport.decodeTokenTrace(
                    model: &m,
                    promptTokens: promptTokens,
                    decodeSteps: decodeSteps
                )
                promptTraces.append((promptName, trace))
            }
            allTraces.append((policyName, promptTraces))
        }

        // FP16 is ground truth (index 0)
        let baselineTraces = allTraces[0].1

        print("\n=== Gemma4-E2B Token Quality Evaluation ===")
        print("  decode_steps=\(decodeSteps)  prompts=\(prompts.count)")

        // Token diversity per trace
        print("\nToken Diversity:")
        for (policyName, promptTraces) in allTraces {
            let pad = policyName.padding(toLength: 10, withPad: " ", startingAt: 0)
            for (promptName, trace) in promptTraces {
                let unique = Set(trace).count
                print("  \(pad) \(promptName): \(unique) unique tokens in \(trace.count) total")
            }
        }

        print(String(repeating: "-", count: 75))
        print("Policy       Prompt    Match   Rate    First_Div  Trace(first 20)")
        print(String(repeating: "-", count: 75))

        var policySummary: [(String, Int, Int)] = []  // (name, totalMatch, totalTokens)

        for (_, (policyName, promptTraces)) in allTraces.enumerated() {
            var policyMatchTotal = 0
            var policyTokenTotal = 0

            for (promptIndex, (promptName, trace)) in promptTraces.enumerated() {
                let baseline = baselineTraces[promptIndex].1
                let compareLength = min(trace.count, baseline.count)

                var matchCount = 0
                var firstDivergence = -1
                for i in 0..<compareLength {
                    if trace[i] == baseline[i] {
                        matchCount += 1
                    } else if firstDivergence < 0 {
                        firstDivergence = i
                    }
                }

                let rate = compareLength > 0 ? Double(matchCount) / Double(compareLength) : 0
                policyMatchTotal += matchCount
                policyTokenTotal += compareLength

                let divStr = (firstDivergence >= 0 ? String(firstDivergence) : "none")
                    .padding(toLength: 9, withPad: " ", startingAt: 0)
                let tracePreview = trace.prefix(20).map { String($0) }.joined(separator: ",")
                let pad = policyName.padding(toLength: 13, withPad: " ", startingAt: 0)
                print("\(pad)\(promptName)  \(String(format: "%3d/%3d", matchCount, compareLength))  \(String(format: "%5.1f%%", rate * 100))  \(divStr)  [\(tracePreview)]")
            }

            policySummary.append((policyName, policyMatchTotal, policyTokenTotal))
        }

        // Aggregate summary
        print(String(repeating: "-", count: 75))
        print("\nAggregate Token Agreement with FP16 Baseline:")
        for (name, matchTotal, tokenTotal) in policySummary {
            let rate = tokenTotal > 0 ? Double(matchTotal) / Double(tokenTotal) * 100 : 0
            let pad = name.padding(toLength: 13, withPad: " ", startingAt: 0)
            print("  \(pad) \(String(format: "%3d/%3d", matchTotal, tokenTotal))  (\(String(format: "%.1f%%", rate)))")
        }

        // Divergence position histogram
        print("\nDivergence Positions (vs FP16):")
        for (policyIndex, (policyName, promptTraces)) in allTraces.enumerated() {
            if policyIndex == 0 { continue }  // skip FP16 vs itself
            var divergences: [Int] = []
            for (promptIndex, (_, trace)) in promptTraces.enumerated() {
                let baseline = baselineTraces[promptIndex].1
                let compareLength = min(trace.count, baseline.count)
                for i in 0..<compareLength {
                    if trace[i] != baseline[i] {
                        divergences.append(i)
                    }
                }
            }
            let pad = policyName.padding(toLength: 13, withPad: " ", startingAt: 0)
            if divergences.isEmpty {
                print("  \(pad) no divergences")
            } else {
                let positions = divergences.map { String($0) }.joined(separator: ", ")
                print("  \(pad) \(divergences.count) divergences at positions: [\(positions)]")
            }
        }

        // FP16 must be 100% with itself
        #expect(policySummary[0].1 == policySummary[0].2,
            "FP16 baseline should match itself 100%")
    }

    // MARK: - Throughput vs Context Length

    @Test("Gemma4 decode throughput vs KV cache fill level")
    func gemma4ThroughputVsContextLength() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let fillLevels = [64, 256, 512]
        let maxSeqLen = 600  // headroom for fill + decode + warmup
        let decodeSteps = 20
        let warmupSteps = 5
        let iterations = 3

        let policies: [(String, InferencePolicy)] = [
            ("FP16", InferencePolicy(
                maximumSequenceLength: maxSeqLen,
                kvCache: .automatic)),
            ("RotorQ8", InferencePolicy(
                maximumSequenceLength: maxSeqLen,
                kvCache: KVCachePolicy(
                    keyScheme: .fixed(.rotorQ8Group32ScaleF16),
                    valueScheme: .fixed(.rotorQ8Group32ScaleF16)))),
            ("RotorQ4", InferencePolicy(
                maximumSequenceLength: maxSeqLen,
                kvCache: KVCachePolicy(
                    keyScheme: .fixed(.rotorQ4Group64ScaleF16),
                    valueScheme: .fixed(.rotorQ4Group64ScaleF16)))),
        ]

        // GPU warm-up: build a temporary model, run 20 decode steps, then discard
        do {
            let (warmModel, _, _) = try BenchmarkSupport.setupFromBundle(
                bundlePath: Self.gemma4BundlePath,
                inferencePolicy: policies[0].1)
            var m = warmModel
            m.resetCaches()
            let warmTokens = [Int32](repeating: 1, count: 64)
            var tok = m.prefill(tokens: warmTokens)
            for _ in 0..<20 { tok = m.decodeSync(tokenID: tok) }
            Thread.sleep(forTimeInterval: 0.5)
        }

        print("\n=== Gemma4-E2B Decode Throughput vs Context Length ===")
        print("  KV cache filled via prefill, then \(decodeSteps) decode steps measured")
        print("  \(iterations) iterations, median reported")
        print("  Single model alive at a time to avoid GPU memory interference")
        print(String(repeating: "-", count: 65))

        // Collect: [policyIndex][fillIndex] = median tok/s
        var allResults: [[Double]] = Array(repeating: Array(repeating: 0.0, count: fillLevels.count), count: policies.count)

        // Build and measure one model at a time to avoid GPU buffer interference.
        // Multiple simultaneous hazardTrackingModeUntracked models cause anomalous
        // GPU cache behavior at high context lengths.
        for (fillIndex, fillLevel) in fillLevels.enumerated() {
            for (policyIndex, (_, policy)) in policies.enumerated() {
                let (model, _, _) = try BenchmarkSupport.setupFromBundle(
                    bundlePath: Self.gemma4BundlePath,
                    inferencePolicy: policy)
                var m = model
                var iterResults: [Double] = []

                for _ in 0..<iterations {
                    m.resetCaches()
                    BenchmarkSupport.settleGPU()

                    // Fill KV cache via prefill
                    let fillTokens = [Int32](repeating: 1, count: fillLevel)
                    var currentToken = m.prefill(tokens: fillTokens)

                    // Warmup decode steps (not measured, stabilizes GPU pipeline)
                    for _ in 0..<warmupSteps {
                        currentToken = m.decodeSync(tokenID: currentToken)
                    }

                    // Measure decode at this fill level
                    let start = CFAbsoluteTimeGetCurrent()
                    for _ in 0..<decodeSteps {
                        currentToken = m.decodeSync(tokenID: currentToken)
                    }
                    iterResults.append(CFAbsoluteTimeGetCurrent() - start)
                }

                let median = iterResults.sorted()[iterations / 2]
                let tokPerSec = Double(decodeSteps) / median
                allResults[policyIndex][fillIndex] = tokPerSec
            }
        }

        // Print results table
        let fillHeader = fillLevels.map { String(format: "%7d", $0) }.joined(separator: " ")
        print("Policy       \(fillHeader)   (tok/s at each fill level)")
        print(String(repeating: "-", count: 65))

        for (policyIndex, (policyName, _)) in policies.enumerated() {
            let pad = policyName.padding(toLength: 13, withPad: " ", startingAt: 0)
            let values = allResults[policyIndex].map { String(format: "%7.1f", $0) }.joined(separator: " ")
            print("\(pad)\(values)")
        }

        // Ratio table (vs FP16)
        if allResults.count >= 2 {
            print("\nRatio vs FP16:")
            let fp16Results = allResults[0]
            for (policyIndex, (policyName, _)) in policies.enumerated() {
                if policyIndex == 0 { continue }
                let ratios = zip(allResults[policyIndex], fp16Results).map { q, f in
                    f > 0 ? q / f : 0
                }
                let pad = policyName.padding(toLength: 13, withPad: " ", startingAt: 0)
                let values = ratios.map { String(format: "%7.2fx", $0) }.joined(separator: " ")
                print("\(pad)\(values)")
            }

            // Monotonicity check: tok/s should decrease (or stay flat) as fill increases
            print("\nMonotonicity check (tok/s should not increase with fill level):")
            for (policyIndex, (policyName, _)) in policies.enumerated() {
                let results = allResults[policyIndex]
                var violations: [String] = []
                for i in 1..<results.count {
                    if results[i] > results[i - 1] * 1.05 {  // 5% tolerance for noise
                        violations.append("fill \(fillLevels[i-1])→\(fillLevels[i]): \(String(format: "%.1f", results[i-1]))→\(String(format: "%.1f", results[i])) tok/s")
                    }
                }
                let pad = policyName.padding(toLength: 13, withPad: " ", startingAt: 0)
                if violations.isEmpty {
                    print("  \(pad) OK (monotonically decreasing)")
                } else {
                    print("  \(pad) ANOMALY: \(violations.joined(separator: ", "))")
                }
            }

            // Bandwidth analysis
            print("\nEstimated KV cache bandwidth fraction:")
            let kvHeads = 4
            let headDim = 64
            let layers = 35
            let weightBytes: Double = 2_000_000_000  // ~2B params, Q8 weights
            for fillLevel in fillLevels {
                let kvBytesPerToken = Double(fillLevel * layers * kvHeads * headDim * 2 * 2)  // K+V, FP16
                let fraction = kvBytesPerToken / (kvBytesPerToken + weightBytes) * 100
                print("  fill=\(String(format: "%-5d", fillLevel)): KV read=\(String(format: "%7.1f", kvBytesPerToken / 1024 / 1024)) MB  fraction=\(String(format: "%5.1f%%", fraction))")
            }
        }
    }

    // MARK: - Compilation Validation

    @Test("RotorQuant KVCacheSpecification buffer size")
    func kvCacheSpecificationSizes() throws {
        let spec = KVCacheSpecification(
            keyQuantizationScheme: .rotorQ8Group32ScaleF16,
            valueQuantizationScheme: .rotorQ4Group64ScaleF16,
            kvHeadCount: 8,
            headDimension: 128,
            maximumSequenceLength: 4096)

        let keyHeadSlot = spec.bytesPerHeadSlot(scheme: .rotorQ8Group32ScaleF16)
        let valueHeadSlot = spec.bytesPerHeadSlot(scheme: .rotorQ4Group64ScaleF16)

        // RotorQ8 group32: (128/32) * 36 = 144 bytes, aligned to 64 → 192
        #expect(keyHeadSlot > 0)
        // RotorQ4 group64: (128/64) * 36 = 72 bytes, aligned to 64 → 128
        #expect(valueHeadSlot > 0)
        #expect(keyHeadSlot != valueHeadSlot, "K and V should have different slot sizes for asymmetric schemes")

        let keyTotal = spec.totalBufferSize(scheme: .rotorQ8Group32ScaleF16)
        let valueTotal = spec.totalBufferSize(scheme: .rotorQ4Group64ScaleF16)
        let fp16Total = spec.totalBufferSize(scheme: .fp16RowMajor)

        // Quantized should be smaller than FP16
        #expect(keyTotal < fp16Total, "RotorQ8 total should be < FP16")
        #expect(valueTotal < fp16Total, "RotorQ4 total should be < FP16")

        print("KVCacheSpec (kvHeads=8, headDim=128, maxSeq=4096):")
        print("  RotorQ8 K: headSlot=\(keyHeadSlot)B, total=\(String(format: "%.1f", Double(keyTotal) / 1024 / 1024))MB")
        print("  RotorQ4 V: headSlot=\(valueHeadSlot)B, total=\(String(format: "%.1f", Double(valueTotal) / 1024 / 1024))MB")
        print("  FP16 ref:  total=\(String(format: "%.1f", Double(fp16Total) / 1024 / 1024))MB")
    }

    // MARK: - Per-Kernel Profiling

    @Test("Gemma4 FP16 per-kernel decode profile")
    func gemma4PerKernelProfile() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let policy = InferencePolicy(
            maximumSequenceLength: 256,
            kvCache: .init(keyScheme: .automatic, valueScheme: .automatic))

        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: Self.gemma4BundlePath,
            inferencePolicy: policy)
        var m = model

        let iterations = 10
        let profiles = try BenchmarkSupport.profileDecodeSteps(
            model: &m,
            device: device,
            iterations: iterations,
            filter: { _ in true })

        struct KernelAggregate {
            var totalMicroseconds: Double = 0
            var count: Int = 0
            var gridSample: MTLSize = MTLSize(width: 0, height: 0, depth: 0)
            var tgSample: MTLSize = MTLSize(width: 0, height: 0, depth: 0)
        }
        var aggregates: [String: KernelAggregate] = [:]
        let totalMicroseconds = profiles.reduce(0.0) { $0 + $1.totalMicroseconds }

        for p in profiles {
            let avgUs = p.totalMicroseconds / Double(iterations)
            aggregates[p.kernelName, default: KernelAggregate()].totalMicroseconds += avgUs
            aggregates[p.kernelName, default: KernelAggregate()].count += 1
            if aggregates[p.kernelName]?.gridSample.width == 0 {
                aggregates[p.kernelName]?.gridSample = p.gridSize
                aggregates[p.kernelName]?.tgSample = p.threadgroupSize
            }
        }

        let avgTotalUs = totalMicroseconds / Double(iterations)
        let sorted = aggregates.sorted { $0.value.totalMicroseconds > $1.value.totalMicroseconds }

        print("\n=== Per-Kernel Decode Profile: Gemma4-E2B FP16 (avg of \(iterations) runs) ===")
        print("Total steps: \(profiles.count)")
        print("Total: \(String(format: "%.0f", avgTotalUs)) us (\(String(format: "%.1f", avgTotalUs / 1000)) ms)")
        print("")
        let header = "Kernel".padding(toLength: 40, withPad: " ", startingAt: 0)
            + "Count  Total us     %  Grid          TG"
        print(header)
        print(String(repeating: "-", count: 100))

        for (name, agg) in sorted {
            let pct = agg.totalMicroseconds / avgTotalUs * 100
            let grid = "\(agg.gridSample.width)x\(agg.gridSample.height)x\(agg.gridSample.depth)"
            let tg = "\(agg.tgSample.width)"
            let pad = name.padding(toLength: 40, withPad: " ", startingAt: 0)
            print("\(pad)\(String(format: "%5d %9.0f %5.1f%%", agg.count, agg.totalMicroseconds, pct))  \(grid.padding(toLength: 14, withPad: " ", startingAt: 0))\(tg)")
        }

        print("\n--- Top 30 individual steps by time ---")
        let topSteps = profiles.sorted { $0.totalMicroseconds > $1.totalMicroseconds }.prefix(30)
        for p in topSteps {
            let avgUs = p.totalMicroseconds / Double(iterations)
            let pct = avgUs / avgTotalUs * 100
            let grid = "\(p.gridSize.width)x\(p.gridSize.height)"
            let name = p.kernelName.padding(toLength: 35, withPad: " ", startingAt: 0)
            print("  [\(String(format: "%3d", p.index))] \(name) \(String(format: "%7.0f", avgUs)) us (\(String(format: "%4.1f", pct))%)  grid=\(grid) tg=\(p.threadgroupSize.width)")
        }

        // CPU/GPU breakdown
        print("\n=== CPU/GPU Breakdown (50 iterations) ===")
        let breakdown = try BenchmarkSupport.measureDecodeSyncBreakdown(
            model: &m, iterations: 50)
        print("  CPU write:      \(String(format: "%7.0f", breakdown.cpuWriteMicroseconds)) us")
        print("  Encode+submit:  \(String(format: "%7.0f", breakdown.encodeSubmitMicroseconds)) us")
        print("  GPU wait:       \(String(format: "%7.0f", breakdown.waitMicroseconds)) us")
        print("  Readback:       \(String(format: "%7.0f", breakdown.readbackMicroseconds)) us")
        print("  GPU time:       \(String(format: "%7.0f", breakdown.gpuMicroseconds)) us")
        print("  Total:          \(String(format: "%7.0f", breakdown.totalMicroseconds)) us")
        let hostOverhead = breakdown.totalMicroseconds - breakdown.gpuMicroseconds
        let hostPct = hostOverhead / breakdown.totalMicroseconds * 100
        print("  Host overhead:  \(String(format: "%7.0f", hostOverhead)) us (\(String(format: "%.1f", hostPct))%)")
    }

    @Test("LFM FP16 vs RotorQ4 per-kernel decode profile")
    func lfmPerKernelProfile() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let fp16Policy = InferencePolicy(
            maximumSequenceLength: 256,
            kvCache: .init(keyScheme: .automatic, valueScheme: .automatic))
        let rotorQ4Policy = InferencePolicy(
            maximumSequenceLength: 256,
            kvCache: .init(
                keyScheme: .fixed(.rotorQ4Group64ScaleF16),
                valueScheme: .fixed(.rotorQ4Group64ScaleF16)))

        let iterations = 10

        do {
            let (model, _) = try Self.setupWithPolicy(inferencePolicy: fp16Policy)
            var m = model
            let profiles = try BenchmarkSupport.profileDecodeSteps(
                model: &m,
                device: device,
                iterations: iterations,
                filter: { _ in true })
            Self.printPerKernelProfile(
                title: "Per-Kernel Decode Profile: LFM2.5-1.2B FP16 (avg of \(iterations) runs)",
                profiles: profiles,
                iterations: iterations
            )
            print("\n=== LFM FP16 CPU/GPU Breakdown (50 iterations) ===")
            let breakdown = try BenchmarkSupport.measureDecodeSyncBreakdown(model: &m, iterations: 50)
            print("  Encode+submit:  \(String(format: "%7.0f", breakdown.encodeSubmitMicroseconds)) us")
            print("  GPU time:       \(String(format: "%7.0f", breakdown.gpuMicroseconds)) us")
            print("  Total:          \(String(format: "%7.0f", breakdown.totalMicroseconds)) us")
        }

        do {
            let (model, _) = try Self.setupWithPolicy(inferencePolicy: rotorQ4Policy)
            var m = model
            let profiles = try BenchmarkSupport.profileDecodeSteps(
                model: &m,
                device: device,
                iterations: iterations,
                filter: { _ in true })
            Self.printPerKernelProfile(
                title: "Per-Kernel Decode Profile: LFM2.5-1.2B RotorQ4 (avg of \(iterations) runs)",
                profiles: profiles,
                iterations: iterations
            )
            print("\n=== LFM RotorQ4 CPU/GPU Breakdown (50 iterations) ===")
            let breakdown = try BenchmarkSupport.measureDecodeSyncBreakdown(model: &m, iterations: 50)
            print("  Encode+submit:  \(String(format: "%7.0f", breakdown.encodeSubmitMicroseconds)) us")
            print("  GPU time:       \(String(format: "%7.0f", breakdown.gpuMicroseconds)) us")
            print("  Total:          \(String(format: "%7.0f", breakdown.totalMicroseconds)) us")
        }
    }
}
