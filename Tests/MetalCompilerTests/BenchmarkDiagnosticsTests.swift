import Foundation
import Metal
import Testing
import LMArchitecture
import LMIR
import ModelDeclarations
@testable import MetalCompiler

/// GPU-heavy diagnostic benchmarks kept separate from throughput acceptance.
@Suite("Benchmark Diagnostics", .serialized)
struct BenchmarkDiagnosticsTests {
    @Test("Decode sync vs pipelined decode throughput")
    func decodeSyncVsPipelinedThroughput() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let decodeSteps = 50
        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let syncResult = try measureDecodeThroughput(
            mode: .sync,
            optimizer: AggressiveOptimizer(),
            decodeSteps: decodeSteps,
            promptTokens: promptTokens
        )
        let pipelinedResult = try measureDecodeThroughput(
            mode: .pipelined,
            optimizer: AggressiveOptimizer(),
            decodeSteps: decodeSteps,
            promptTokens: promptTokens
        )

        let syncTokPerSec = syncResult.tokensPerSecond
        let pipelinedTokPerSec = pipelinedResult.tokensPerSecond
        let deltaPct = syncTokPerSec > 0 ? (pipelinedTokPerSec - syncTokPerSec) / syncTokPerSec * 100 : 0

        print("\n=== Decode Sync vs Pipelined Decode (aggressive) ===")
        print("sync:      \(String(format: "%.1f", syncTokPerSec)) tok/s (\(String(format: "%.2f", syncResult.millisecondsPerToken)) ms/tok)")
        print("pipelined: \(String(format: "%.1f", pipelinedTokPerSec)) tok/s (\(String(format: "%.2f", pipelinedResult.millisecondsPerToken)) ms/tok)")
        print("delta:     \(String(format: "%+.1f", deltaPct))%")
    }

    @Test("Decode sync host-overhead breakdown")
    func decodeSyncHostOverheadBreakdown() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let (model, _) = try BenchmarkSupport.setupOrSkip(optimizer: AggressiveOptimizer())
        var inferenceModel = model
        let iterations = 50
        let breakdown = try BenchmarkSupport.measureDecodeSyncBreakdown(
            model: &inferenceModel,
            iterations: iterations
        )

        let hostOverhead = breakdown.hostOverheadMicroseconds
        let hostPct = breakdown.totalMicroseconds > 0
            ? hostOverhead / breakdown.totalMicroseconds * 100
            : 0
        let waitPct = breakdown.totalMicroseconds > 0
            ? breakdown.waitMicroseconds / breakdown.totalMicroseconds * 100
            : 0

        print("\n=== Decode Sync Host Overhead (\(iterations) iterations, aggressive) ===")
        print("total:         \(String(format: "%.1f", breakdown.totalMicroseconds)) us/token")
        print("gpu:           \(String(format: "%.1f", breakdown.gpuMicroseconds)) us/token")
        print("cpu write:     \(String(format: "%.1f", breakdown.cpuWriteMicroseconds)) us/token")
        print("encode+submit: \(String(format: "%.1f", breakdown.encodeSubmitMicroseconds)) us/token")
        print("wait:          \(String(format: "%.1f", breakdown.waitMicroseconds)) us/token")
        print("readback:      \(String(format: "%.1f", breakdown.readbackMicroseconds)) us/token")
        print("host overhead: \(String(format: "%.1f", hostOverhead)) us/token (\(String(format: "%.1f", hostPct))%)")
        print("wait share:    \(String(format: "%.1f", waitPct))%")
    }

    @Test("Optimizer decode token trace comparison")
    func optimizerDecodeTokenTraceComparison() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let variants: [(String, any DispatchOptimizer)] = [
            ("none", NoOptimizer()),
            ("standard", StandardOptimizer()),
            ("aggressive", AggressiveOptimizer()),
        ]

        var traces: [(String, [Int32])] = []
        for (name, optimizer) in variants {
            let (model, _) = try BenchmarkSupport.setupOrSkip(optimizer: optimizer)
            var inferenceModel = model
            let trace = BenchmarkSupport.decodeTokenTrace(
                model: &inferenceModel,
                promptTokens: promptTokens,
                predecodeSteps: 3,
                decodeSteps: 5
            )
            traces.append((name, trace))
        }

        print("\n=== Optimizer Decode Token Trace ===")
        for (name, trace) in traces {
            print("\(name): \(trace)")
        }

        #expect(traces.count == 3)
        #expect(traces[1].1 == traces[2].1, "standard and aggressive traces should match")
    }

    @Test("Compilation time (IR → dispatch plan)")
    func compilationBenchmark() throws {
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

        let start = CFAbsoluteTimeGetCurrent()
        let compiler = MetalInferenceCompiler()
        let decodePlan = try compiler.compile(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, stafWeightStore: store, device: device)
        let compileTime = CFAbsoluteTimeGetCurrent() - start

        let prefillStart = CFAbsoluteTimeGetCurrent()
        let prefillPlan = try compiler.compilePrefill(
            graph: resolved, hiddenSize: 2048, intermediateSize: 8192,
            vocabSize: 65536, maximumSequenceLength: 64,
            stafWeightStore: store, device: device)
        let prefillCompileTime = CFAbsoluteTimeGetCurrent() - prefillStart

        print("[Benchmark] compilation:")
        print("  decode plan: \(decodePlan.fusedEntryCount) dispatches, \(String(format: "%.0f", compileTime * 1000))ms")
        print("  prefill plan: \(prefillPlan.stepCount) steps, \(String(format: "%.0f", prefillCompileTime * 1000))ms")

        let optimizers: [any DispatchOptimizer] = [
            NoOptimizer(),
            StandardOptimizer(),
            AggressiveOptimizer(),
        ]
        print("\n[Benchmark] optimizer comparison:")
        for opt in optimizers {
            let comp = MetalInferenceCompiler(optimizer: opt)
            let report = comp.analyzeOptimization(graph: resolved, hiddenSize: 2048)
            report.printReport()
        }
    }

    @Test("Per-step decode profiling")
    func perStepDecodeProfile() throws {
        let (model, _) = try BenchmarkSupport.setupOrSkip()
        var m = model
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw BenchmarkSupport.BenchError.noDevice
        }

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        var currentToken = m.prefill(tokens: promptTokens)
        for _ in 0..<3 { currentToken = m.decodeSync(tokenID: currentToken) }

        let plan = m.plan
        plan.buffers.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(m.position)
        plan.buffers.tokenIn.contents().bindMemory(to: Int32.self, capacity: 1).pointee = currentToken

        let iterations = 10
        let profiles = try BenchmarkSupport.profileDecodeSteps(
            model: &m,
            queue: queue,
            iterations: iterations,
            filter: { _ in true })

        struct KernelAggregate {
            var totalMicroseconds: Double = 0
            var count: Int = 0
            var gridSample: MTLSize = MTLSize(width: 0, height: 0, depth: 0)
        }
        var aggregates: [String: KernelAggregate] = [:]
        let totalMicroseconds = profiles.reduce(0.0) { $0 + $1.totalMicroseconds }

        for p in profiles {
            let avgUs = p.totalMicroseconds / Double(iterations)
            aggregates[p.kernelName, default: KernelAggregate()].totalMicroseconds += avgUs
            aggregates[p.kernelName, default: KernelAggregate()].count += 1
            if aggregates[p.kernelName]?.gridSample.width == 0 {
                aggregates[p.kernelName]?.gridSample = p.gridSize
            }
        }

        let avgTotalUs = totalMicroseconds / Double(iterations)
        let sorted = aggregates.sorted { $0.value.totalMicroseconds > $1.value.totalMicroseconds }

        print("\n=== Per-Step Decode Profile: LFM2.5-1.2B (avg of \(iterations) runs) ===")
        print("Total: \(String(format: "%.0f", avgTotalUs)) us (\(String(format: "%.1f", avgTotalUs / 1000)) ms)")
        print("")
        let header = "Kernel".padding(toLength: 35, withPad: " ", startingAt: 0)
            + "Count Total us     %  Grid"
        print(header)
        print(String(repeating: "-", count: 80))

        for (name, agg) in sorted {
            let pct = agg.totalMicroseconds / avgTotalUs * 100
            let grid = "\(agg.gridSample.width)x\(agg.gridSample.height)x\(agg.gridSample.depth)"
            let pad = name.padding(toLength: 35, withPad: " ", startingAt: 0)
            print("\(pad)\(String(format: "%5d %8.0f %5.1f%%", agg.count, agg.totalMicroseconds, pct))  \(grid)")
        }

        print("\n--- Top 20 individual steps ---")
        let topSteps = profiles.sorted { $0.totalMicroseconds > $1.totalMicroseconds }.prefix(20)
        for p in topSteps {
            let avgUs = p.totalMicroseconds / Double(iterations)
            let pct = avgUs / avgTotalUs * 100
            let grid = "\(p.gridSize.width)x\(p.gridSize.height)"
            let name = p.kernelName.padding(toLength: 30, withPad: " ", startingAt: 0)
            print("  [\(String(format: "%3d", p.index))] \(name) \(String(format: "%6.0f", avgUs)) us (\(String(format: "%4.1f", pct))%)  grid=\(grid)")
        }
    }

    @Test("Hot exact-shape GEMV family microbench")
    func hotExactShapeGEMVMicrobench() throws {
        let (model, _) = try BenchmarkSupport.setupOrSkip()
        var m = model
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw BenchmarkSupport.BenchError.noDevice
        }

        let iterations = 20
        let profiles = try BenchmarkSupport.profileDecodeSteps(
            model: &m,
            queue: queue,
            iterations: iterations,
            filter: { step in
                BenchmarkSupport.isHotExactShapeGEMVKernel(step.pipeline.label ?? "")
            })

        let totalMicroseconds = profiles.reduce(0.0) { $0 + $1.totalMicroseconds }
        let avgTotalUs = totalMicroseconds / Double(iterations)

        struct KernelFamilyAggregate {
            var totalMicroseconds: Double = 0
            var count: Int = 0
        }

        var aggregates: [String: KernelFamilyAggregate] = [:]
        for profile in profiles {
            let averageMicroseconds = profile.totalMicroseconds / Double(iterations)
            let family = BenchmarkSupport.canonicalHotExactShapeGEMVFamily(for: profile.kernelName)
            aggregates[family, default: KernelFamilyAggregate()].totalMicroseconds += averageMicroseconds
            aggregates[family, default: KernelFamilyAggregate()].count += 1
        }

        let sorted = aggregates.sorted { $0.value.totalMicroseconds > $1.value.totalMicroseconds }

        print("\n=== Hot Exact-Shape GEMV Microbench: LFM2.5-1.2B (avg of \(iterations) runs) ===")
        print("Target families total: \(String(format: "%.0f", avgTotalUs)) us")
        for (name, aggregate) in sorted {
            let pct = avgTotalUs > 0 ? aggregate.totalMicroseconds / avgTotalUs * 100 : 0
            let averagePerStep = aggregate.totalMicroseconds / Double(aggregate.count)
            print("  \(name): count=\(aggregate.count) family=\(String(format: "%.0f", aggregate.totalMicroseconds)) us step=\(String(format: "%.1f", averagePerStep)) us share=\(String(format: "%.1f", pct))%")
        }
    }

    @Test("Square exact-shape GEMV role breakdown")
    func squareExactShapeGEMVRoleBreakdown() throws {
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
        let compiler = MetalInferenceCompiler()
        let dump = compiler.dumpDispatchEntries(graph: graph, hiddenSize: 2048)

        struct Aggregate {
            var count: Int = 0
        }

        let square = dump
            .split(separator: "\n")
            .map(String.init)
            .filter { line in
                line.contains("projection(") &&
                line.contains("in=2048") &&
                line.contains("out=2048")
            }
        var byRole: [String: Aggregate] = [:]
        for line in square {
            guard
                let roleStart = line.range(of: "projection(")?.upperBound,
                let roleEnd = line[roleStart...].firstIndex(of: ",")
            else {
                continue
            }
            let role = String(line[roleStart..<roleEnd])
            byRole[role, default: Aggregate()].count += 1
        }

        print("\n=== Square Exact-Shape GEMV Role Breakdown ===")
        print("square steps: \(square.count)")
        for (role, aggregate) in byRole.sorted(by: { $0.key < $1.key }) {
            print("  role=\(role) count=\(aggregate.count)")
        }
        for line in square {
            print("  \(line)")
        }
    }

    @Test("Square exact-shape GEMV role microbench")
    func squareExactShapeGEMVRoleMicrobench() throws {
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
        let compiler = MetalInferenceCompiler()
        let dump = compiler.dumpDispatchEntries(graph: graph, hiddenSize: 2048)
        let squareRoles = dump
            .split(separator: "\n")
            .map(String.init)
            .filter { line in
                line.contains("projection(") &&
                line.contains("in=2048") &&
                line.contains("out=2048")
            }
            .compactMap { line -> String? in
                guard
                    let roleStart = line.range(of: "projection(")?.upperBound,
                    let roleEnd = line[roleStart...].firstIndex(of: ",")
                else {
                    return nil
                }
                return String(line[roleStart..<roleEnd])
            }

        let (model, _) = try BenchmarkSupport.setupOrSkip()
        var baseline = model
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw BenchmarkSupport.BenchError.noDevice
        }

        let iterations = 20
        let profiles = try BenchmarkSupport.profileDecodeSteps(
            model: &baseline,
            queue: queue,
            iterations: iterations,
            filter: { step in
                BenchmarkSupport.isHotSquareGEMVKernel(step.pipeline.label ?? "")
            })

        struct Aggregate {
            var totalMicroseconds: Double = 0
            var count: Int = 0
        }

        var byRole: [String: Aggregate] = [:]
        for (role, profile) in zip(squareRoles, profiles) {
            byRole[role, default: Aggregate()].totalMicroseconds += profile.totalMicroseconds / Double(iterations)
            byRole[role, default: Aggregate()].count += 1
        }

        let total = byRole.values.reduce(0.0) { $0 + $1.totalMicroseconds }

        print("\n=== Square Exact-Shape GEMV Role Microbench: LFM2.5-1.2B (avg of \(iterations) runs) ===")
        print("square family total: \(String(format: "%.0f", total)) us")
        for (role, aggregate) in byRole.sorted(by: { $0.value.totalMicroseconds > $1.value.totalMicroseconds }) {
            let share = total > 0 ? aggregate.totalMicroseconds / total * 100 : 0
            let stepAverage = aggregate.count > 0 ? aggregate.totalMicroseconds / Double(aggregate.count) : 0
            print("  role=\(role) count=\(aggregate.count) family=\(String(format: "%.0f", aggregate.totalMicroseconds)) us step=\(String(format: "%.1f", stepAverage)) us share=\(String(format: "%.1f", share))%")
        }
    }

    @Test("Blocked layout override: 2048->6144 hot-family microbench")
    func blocked6144HotFamilyMicrobench() throws {
        let override = BenchmarkSupport.blocked6144DecodeOverride()
        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let iterations = 20
        let baseline = try profileDecodeVariant(
            promptTokens: promptTokens,
            iterations: iterations,
            filter: { BenchmarkSupport.isHot6144GEMVKernel($0.pipeline.label ?? "") }
        )
        let specialized = try profileDecodeVariant(
            weightAccessPolicyOverride: override,
            promptTokens: promptTokens,
            iterations: iterations,
            filter: { BenchmarkSupport.isHot6144GEMVKernel($0.pipeline.label ?? "") }
        )

        let baselineTotal = baseline.profiles.reduce(0.0) { $0 + $1.totalMicroseconds } / Double(iterations)
        let specializedTotal = specialized.profiles.reduce(0.0) { $0 + $1.totalMicroseconds } / Double(iterations)
        let baselinePerStep = baselineTotal / Double(baseline.profiles.count)
        let specializedPerStep = specializedTotal / Double(specialized.profiles.count)
        let delta = specializedTotal - baselineTotal
        let deltaPct = baselineTotal > 0 ? delta / baselineTotal * 100 : 0

        print("\n=== Blocked Layout Override: 2048->6144 Hot Family ===")
        print("baseline tokens:    \(baseline.sequence)")
        print("specialized tokens: \(specialized.sequence)")
        print("baseline:    total=\(String(format: "%.0f", baselineTotal)) us step=\(String(format: "%.1f", baselinePerStep)) us count=\(baseline.profiles.count)")
        print("specialized: total=\(String(format: "%.0f", specializedTotal)) us step=\(String(format: "%.1f", specializedPerStep)) us count=\(specialized.profiles.count)")
        print("delta:       \(String(format: "%+.0f", delta)) us (\(String(format: "%+.1f", deltaPct))%)")
    }

    @Test("Blocked 4x128 layout override: 2048->6144 hot-family microbench")
    func blocked4x1286144HotFamilyMicrobench() throws {
        let override = BenchmarkSupport.blocked4x1286144DecodeOverride()
        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let iterations = 20
        let baseline = try profileDecodeVariant(
            promptTokens: promptTokens,
            iterations: iterations,
            filter: { BenchmarkSupport.isHot6144GEMVKernel($0.pipeline.label ?? "") }
        )
        let specialized = try profileDecodeVariant(
            weightAccessPolicyOverride: override,
            promptTokens: promptTokens,
            iterations: iterations,
            filter: { BenchmarkSupport.isHot6144GEMVKernel($0.pipeline.label ?? "") }
        )

        let baselineTotal = baseline.profiles.reduce(0.0) { $0 + $1.totalMicroseconds } / Double(iterations)
        let specializedTotal = specialized.profiles.reduce(0.0) { $0 + $1.totalMicroseconds } / Double(iterations)
        let baselinePerStep = baselineTotal / Double(baseline.profiles.count)
        let specializedPerStep = specializedTotal / Double(specialized.profiles.count)
        let delta = specializedTotal - baselineTotal
        let deltaPct = baselineTotal > 0 ? delta / baselineTotal * 100 : 0

        print("\n=== Blocked 4x128 Layout Override: 2048->6144 Hot Family ===")
        print("baseline tokens:    \(baseline.sequence)")
        print("specialized tokens: \(specialized.sequence)")
        print("baseline:    total=\(String(format: "%.0f", baselineTotal)) us step=\(String(format: "%.1f", baselinePerStep)) us count=\(baseline.profiles.count)")
        print("specialized: total=\(String(format: "%.0f", specializedTotal)) us step=\(String(format: "%.1f", specializedPerStep)) us count=\(specialized.profiles.count)")
        print("delta:       \(String(format: "%+.0f", delta)) us (\(String(format: "%+.1f", deltaPct))%)")
    }

    @Test("Blocked 8x128 layout override: 2048->2048 hot-family microbench")
    func blocked8x128SquareHotFamilyMicrobench() throws {
        let override = BenchmarkSupport.blocked8x128SquareDecodeOverride()
        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let iterations = 20
        let baseline = try profileDecodeVariant(
            promptTokens: promptTokens,
            iterations: iterations,
            filter: { BenchmarkSupport.isHotSquareGEMVKernel($0.pipeline.label ?? "") }
        )
        let specialized = try profileDecodeVariant(
            weightAccessPolicyOverride: override,
            promptTokens: promptTokens,
            iterations: iterations,
            filter: { BenchmarkSupport.isHotSquareGEMVKernel($0.pipeline.label ?? "") }
        )

        let baselineTotal = baseline.profiles.reduce(0.0) { $0 + $1.totalMicroseconds } / Double(iterations)
        let specializedTotal = specialized.profiles.reduce(0.0) { $0 + $1.totalMicroseconds } / Double(iterations)
        let baselinePerStep = baselineTotal / Double(baseline.profiles.count)
        let specializedPerStep = specializedTotal / Double(specialized.profiles.count)
        let delta = specializedTotal - baselineTotal
        let deltaPct = baselineTotal > 0 ? delta / baselineTotal * 100 : 0

        print("\n=== Blocked 8x128 Layout Override: 2048->2048 Hot Family ===")
        print("baseline tokens:    \(baseline.sequence)")
        print("specialized tokens: \(specialized.sequence)")
        print("baseline:    total=\(String(format: "%.0f", baselineTotal)) us step=\(String(format: "%.1f", baselinePerStep)) us count=\(baseline.profiles.count)")
        print("specialized: total=\(String(format: "%.0f", specializedTotal)) us step=\(String(format: "%.1f", specializedPerStep)) us count=\(specialized.profiles.count)")
        print("delta:       \(String(format: "%+.0f", delta)) us (\(String(format: "%+.1f", deltaPct))%)")
    }

    @Test("Blocked 8x128 layout override: square q_proj-only diagnostic")
    func blocked8x128SquareQProjOnlyDiagnostic() throws {
        let override = BenchmarkSupport.blocked8x128SquareQProjDecodeOverride()
        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let iterations = 20
        let baseline = try profileDecodeVariant(
            promptTokens: promptTokens,
            iterations: iterations,
            filter: { BenchmarkSupport.isHotSquareGEMVKernel($0.pipeline.label ?? "") }
        )
        let specialized = try profileDecodeVariant(
            weightAccessPolicyOverride: override,
            promptTokens: promptTokens,
            iterations: iterations,
            filter: { BenchmarkSupport.isHotSquareGEMVKernel($0.pipeline.label ?? "") }
        )

        let baselineTotal = baseline.profiles.reduce(0.0) { $0 + $1.totalMicroseconds } / Double(iterations)
        let specializedTotal = specialized.profiles.reduce(0.0) { $0 + $1.totalMicroseconds } / Double(iterations)
        let baselinePerStep = baselineTotal / Double(baseline.profiles.count)
        let specializedPerStep = specializedTotal / Double(specialized.profiles.count)
        let delta = specializedTotal - baselineTotal
        let deltaPct = baselineTotal > 0 ? delta / baselineTotal * 100 : 0

        print("\n=== Blocked 8x128 Layout Override: Square q_proj-only ===")
        print("baseline tokens:    \(baseline.sequence)")
        print("specialized tokens: \(specialized.sequence)")
        print("baseline:    total=\(String(format: "%.0f", baselineTotal)) us step=\(String(format: "%.1f", baselinePerStep)) us count=\(baseline.profiles.count)")
        print("specialized: total=\(String(format: "%.0f", specializedTotal)) us step=\(String(format: "%.1f", specializedPerStep)) us count=\(specialized.profiles.count)")
        print("delta:       \(String(format: "%+.0f", delta)) us (\(String(format: "%+.1f", deltaPct))%)")
    }

    @Test("Blocked 8x128 layout override: square q_proj prefix-3 diagnostic")
    func blocked8x128SquareQProjPrefix3Diagnostic() throws {
        let override = BenchmarkSupport.blocked8x128SquareQProjPrefix3DecodeOverride()
        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let iterations = 20
        let baseline = try profileDecodeVariant(
            promptTokens: promptTokens,
            iterations: iterations,
            filter: { BenchmarkSupport.isHotSquareGEMVKernel($0.pipeline.label ?? "") }
        )
        let specialized = try profileDecodeVariant(
            weightAccessPolicyOverride: override,
            promptTokens: promptTokens,
            iterations: iterations,
            filter: { BenchmarkSupport.isHotSquareGEMVKernel($0.pipeline.label ?? "") }
        )

        let baselineTotal = baseline.profiles.reduce(0.0) { $0 + $1.totalMicroseconds } / Double(iterations)
        let specializedTotal = specialized.profiles.reduce(0.0) { $0 + $1.totalMicroseconds } / Double(iterations)
        let baselinePerStep = baselineTotal / Double(baseline.profiles.count)
        let specializedPerStep = specializedTotal / Double(specialized.profiles.count)
        let delta = specializedTotal - baselineTotal
        let deltaPct = baselineTotal > 0 ? delta / baselineTotal * 100 : 0

        print("\n=== Blocked 8x128 Layout Override: Square q_proj Prefix-3 ===")
        print("tensors:      \(Array(BenchmarkSupport.squareQProjBlockedSafePrefixTensorNames).sorted())")
        print("baseline tokens:    \(baseline.sequence)")
        print("specialized tokens: \(specialized.sequence)")
        print("baseline:    total=\(String(format: "%.0f", baselineTotal)) us step=\(String(format: "%.1f", baselinePerStep)) us count=\(baseline.profiles.count)")
        print("specialized: total=\(String(format: "%.0f", specializedTotal)) us step=\(String(format: "%.1f", specializedPerStep)) us count=\(specialized.profiles.count)")
        print("delta:       \(String(format: "%+.0f", delta)) us (\(String(format: "%+.1f", deltaPct))%)")
    }

    @Test("Blocked 8x128 layout override: square q_proj prefix-2 diagnostic")
    func blocked8x128SquareQProjPrefix2Diagnostic() throws {
        let override = BenchmarkSupport.blocked8x128SquareQProjPrefix2DecodeOverride()
        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let iterations = 20
        let baseline = try profileDecodeVariant(
            promptTokens: promptTokens,
            iterations: iterations,
            filter: { BenchmarkSupport.isHotSquareGEMVKernel($0.pipeline.label ?? "") }
        )
        let specialized = try profileDecodeVariant(
            weightAccessPolicyOverride: override,
            promptTokens: promptTokens,
            iterations: iterations,
            filter: { BenchmarkSupport.isHotSquareGEMVKernel($0.pipeline.label ?? "") }
        )

        let baselineTotal = baseline.profiles.reduce(0.0) { $0 + $1.totalMicroseconds } / Double(iterations)
        let specializedTotal = specialized.profiles.reduce(0.0) { $0 + $1.totalMicroseconds } / Double(iterations)
        let baselinePerStep = baselineTotal / Double(baseline.profiles.count)
        let specializedPerStep = specializedTotal / Double(specialized.profiles.count)
        let delta = specializedTotal - baselineTotal
        let deltaPct = baselineTotal > 0 ? delta / baselineTotal * 100 : 0

        print("\n=== Blocked 8x128 Layout Override: Square q_proj Prefix-2 ===")
        print("tensors:      \(Array(BenchmarkSupport.squareQProjBlockedSafePrefix2TensorNames).sorted())")
        print("baseline tokens:    \(baseline.sequence)")
        print("specialized tokens: \(specialized.sequence)")
        print("baseline:    total=\(String(format: "%.0f", baselineTotal)) us step=\(String(format: "%.1f", baselinePerStep)) us count=\(baseline.profiles.count)")
        print("specialized: total=\(String(format: "%.0f", specializedTotal)) us step=\(String(format: "%.1f", specializedPerStep)) us count=\(specialized.profiles.count)")
        print("delta:       \(String(format: "%+.0f", delta)) us (\(String(format: "%+.1f", deltaPct))%)")
    }

    @Test("Blocked 8x128 layout override: single square q_proj multi-token diagnostic")
    func blocked8x128SquareSingleQProjMultiTokenDiagnostic() throws {
        let override = BenchmarkSupport.blocked8x128SquareSingleQProjDecodeOverride()
        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let iterations = 20
        let baseline = try profileDecodeVariant(
            promptTokens: promptTokens,
            iterations: iterations,
            filter: { BenchmarkSupport.isHotSquareGEMVKernel($0.pipeline.label ?? "") }
        )
        let specialized = try profileDecodeVariant(
            weightAccessPolicyOverride: override,
            promptTokens: promptTokens,
            iterations: iterations,
            filter: { BenchmarkSupport.isHotSquareGEMVKernel($0.pipeline.label ?? "") }
        )

        let baselineTotal = baseline.profiles.reduce(0.0) { $0 + $1.totalMicroseconds } / Double(iterations)
        let specializedTotal = specialized.profiles.reduce(0.0) { $0 + $1.totalMicroseconds } / Double(iterations)
        let baselinePerStep = baselineTotal / Double(baseline.profiles.count)
        let specializedPerStep = specializedTotal / Double(specialized.profiles.count)
        let delta = specializedTotal - baselineTotal
        let deltaPct = baselineTotal > 0 ? delta / baselineTotal * 100 : 0

        print("\n=== Blocked 8x128 Layout Override: Single Square q_proj Multi-token ===")
        print("tensor:       \(BenchmarkSupport.squareSingleQProjBlockedTensorName)")
        print("baseline tokens:    \(baseline.sequence)")
        print("specialized tokens: \(specialized.sequence)")
        print("baseline:    total=\(String(format: "%.0f", baselineTotal)) us step=\(String(format: "%.1f", baselinePerStep)) us count=\(baseline.profiles.count)")
        print("specialized: total=\(String(format: "%.0f", specializedTotal)) us step=\(String(format: "%.1f", specializedPerStep)) us count=\(specialized.profiles.count)")
        print("delta:       \(String(format: "%+.0f", delta)) us (\(String(format: "%+.1f", deltaPct))%)")
    }

    private enum DecodeMode {
        case sync
        case pipelined
    }

    private struct DecodeThroughputResult {
        let tokensPerSecond: Double
        let millisecondsPerToken: Double
    }

    private struct ProfileVariantResult {
        let sequence: [Int32]
        let profiles: [BenchmarkSupport.StepProfile]
    }

    private func measureDecodeThroughput(
        mode: DecodeMode,
        optimizer: (any DispatchOptimizer)? = nil,
        decodeSteps: Int,
        promptTokens: [Int32]
    ) throws -> DecodeThroughputResult {
        let (model, _) = try BenchmarkSupport.setupOrSkip(optimizer: optimizer)
        var inference = model
        defer {
            inference.resetCaches()
            BenchmarkSupport.settleGPU()
        }

        switch mode {
        case .sync:
            var token = inference.prefill(tokens: promptTokens)
            for _ in 0..<3 { token = inference.decodeSync(tokenID: token) }

            let start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<decodeSteps {
                token = inference.decodeSync(tokenID: token)
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            return DecodeThroughputResult(
                tokensPerSecond: Double(decodeSteps) / elapsed,
                millisecondsPerToken: elapsed / Double(decodeSteps) * 1000
            )

        case .pipelined:
            var token = inference.prefill(tokens: promptTokens)
            for _ in 0..<3 {
                _ = inference.decode(tokenID: token)
                token = inference.flush()
            }

            let start = CFAbsoluteTimeGetCurrent()
            _ = inference.decode(tokenID: token)
            for step in 0..<decodeSteps {
                token = inference.flush()
                if step + 1 < decodeSteps {
                    _ = inference.decode(tokenID: token)
                }
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            return DecodeThroughputResult(
                tokensPerSecond: Double(decodeSteps) / elapsed,
                millisecondsPerToken: elapsed / Double(decodeSteps) * 1000
            )
        }
    }

    private func profileDecodeVariant(
        weightAccessPolicyOverride: ProjectionWeightAccessPolicyOverride? = nil,
        promptTokens: [Int32],
        decodeSteps: Int = 3,
        iterations: Int,
        filter: (MetalDispatchStep) -> Bool
    ) throws -> ProfileVariantResult {
        let (model, _) = try BenchmarkSupport.setupOrSkip(
            weightAccessPolicyOverride: weightAccessPolicyOverride
        )
        var inference = model
        defer {
            inference.resetCaches()
            BenchmarkSupport.settleGPU()
        }

        let first = inference.prefill(tokens: promptTokens)
        var sequence: [Int32] = [first]
        var token = first
        for _ in 0..<decodeSteps {
            token = inference.decodeSync(tokenID: token)
            sequence.append(token)
        }

        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw BenchmarkSupport.BenchError.noDevice
        }

        let profiles = try BenchmarkSupport.profileDecodeSteps(
            model: &inference,
            queue: queue,
            iterations: iterations,
            filter: filter
        )
        return ProfileVariantResult(sequence: sequence, profiles: profiles)
    }
}
