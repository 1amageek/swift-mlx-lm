import Foundation
import Metal
import Testing
@testable import MetalCompiler

/// Gemma 4 E2B throughput benchmarks.
///
/// Mirrors the LFM2.5 benchmark suite structure so results are directly comparable.
/// The model bundle is expected at TestData/gemma-4-E2B-it/.
@Suite("Gemma4 Benchmark", .serialized)
struct Gemma4BenchmarkTests {

    static let bundlePath = "/Users/1amageek/Desktop/swift-lm/TestData/gemma-4-E2B-it"
    static let modelLabel = "Gemma4-E2B"

    @Test("Prefill throughput (tok/s)")
    func prefillBenchmark() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: Self.bundlePath
        )
        var inferenceModel = model

        let warmupTokens: [Int32] = [1, 1, 6]
        _ = inferenceModel.prefill(tokens: warmupTokens)
        inferenceModel.resetCaches()

        let lengths = [16, 32, 64]
        for length in lengths {
            inferenceModel.resetCaches()
            let tokens = [Int32](repeating: 1, count: length)

            let start = CFAbsoluteTimeGetCurrent()
            _ = inferenceModel.prefill(tokens: tokens)
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            let tokPerSec = Double(length) / elapsed
            print("[Benchmark/\(Self.modelLabel)] prefill \(length) tokens: \(String(format: "%.1f", tokPerSec)) tok/s (\(String(format: "%.3f", elapsed * 1000))ms)")
        }
    }

    @Test("Decode throughput (tok/s)")
    func decodeBenchmark() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: Self.bundlePath
        )
        var inferenceModel = model

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        var currentToken = inferenceModel.prefill(tokens: promptTokens)
        for _ in 0..<3 {
            currentToken = inferenceModel.decodeSync(tokenID: currentToken)
        }

        let decodeSteps = 50
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<decodeSteps {
            currentToken = inferenceModel.decodeSync(tokenID: currentToken)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        let tokPerSec = Double(decodeSteps) / elapsed
        let msPerToken = elapsed / Double(decodeSteps) * 1000
        print("[Benchmark/\(Self.modelLabel)] decode \(decodeSteps) tokens: \(String(format: "%.1f", tokPerSec)) tok/s (\(String(format: "%.2f", msPerToken)) ms/tok)")
    }

    @Test("Aggressive optimizer: decode throughput")
    func aggressiveDecodeBenchmark() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: Self.bundlePath,
            optimizer: AggressiveOptimizer()
        )
        var inferenceModel = model

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        var currentToken = inferenceModel.prefill(tokens: promptTokens)
        for _ in 0..<3 {
            currentToken = inferenceModel.decodeSync(tokenID: currentToken)
        }

        let decodeSteps = 50
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<decodeSteps {
            currentToken = inferenceModel.decodeSync(tokenID: currentToken)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        let tokPerSec = Double(decodeSteps) / elapsed
        let msPerToken = elapsed / Double(decodeSteps) * 1000
        print("[Benchmark/\(Self.modelLabel)/aggressive] decode \(decodeSteps) tokens: \(String(format: "%.1f", tokPerSec)) tok/s (\(String(format: "%.2f", msPerToken)) ms/tok)")
        print("[Benchmark/\(Self.modelLabel)/aggressive] dispatches: \(inferenceModel.decodePlan.fusedEntryCount) (from \(inferenceModel.decodePlan.unfusedEntryCount))")
    }

    @Test("End-to-end: prefill + decode with memory diagnostics")
    func endToEndBenchmark() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: Self.bundlePath
        )
        var inferenceModel = model

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let generateCount = 100

        let b = inferenceModel.buffers
        let hiddenBytes = b.hidden.length
        let scratchBytes = b.scratch.length
        let residualBytes = b.residual.length
        let logitsBytes = b.logits.length
        let kvBytes = (b.kvCache?.keys.length ?? 0) + (b.kvCache?.values.length ?? 0)
        let convBytes = b.convState?.length ?? 0
        let recurrentBytes = b.recurrentState?.length ?? 0
        let weightBytes = b.weights.reduce(0) { $0 + $1.length }
        let totalGPUBytes = hiddenBytes + scratchBytes + residualBytes + logitsBytes + kvBytes + convBytes + recurrentBytes
        let totalWeightBytes = weightBytes

        func storageMode(_ buf: MTLBuffer) -> String {
            switch buf.storageMode {
            case .shared: return "shared"
            case .private: return "private"
            case .memoryless: return "memoryless"
            @unknown default: return "unknown"
            }
        }

        let totalStart = CFAbsoluteTimeGetCurrent()
        let prefillStart = CFAbsoluteTimeGetCurrent()
        var currentToken = inferenceModel.prefill(tokens: promptTokens)
        let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStart

        let decodeStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<generateCount {
            currentToken = inferenceModel.decodeSync(tokenID: currentToken)
        }
        let decodeTime = CFAbsoluteTimeGetCurrent() - decodeStart
        let totalTime = CFAbsoluteTimeGetCurrent() - totalStart
        let totalTokens = promptTokens.count + generateCount

        let bytesPerToken = Double(totalWeightBytes) * 2.0
        let decodeBandwidth = bytesPerToken * Double(generateCount) / decodeTime / 1e9

        var report = "=== Benchmark: \(Self.modelLabel) ===\n\n"
        report += "Memory Layout:\n"
        report += "  hidden:      \(String(format: "%6.1f", Double(hiddenBytes) / 1024)) KB (\(storageMode(b.hidden)))\n"
        report += "  scratch:     \(String(format: "%6.1f", Double(scratchBytes) / 1024)) KB (\(storageMode(b.scratch)))\n"
        report += "  residual:    \(String(format: "%6.1f", Double(residualBytes) / 1024)) KB (\(storageMode(b.residual)))\n"
        report += "  logits:      \(String(format: "%6.1f", Double(logitsBytes) / 1024)) KB (\(storageMode(b.logits)))\n"
        report += "  KV cache:    \(String(format: "%6.1f", Double(kvBytes) / 1024 / 1024)) MB (\(b.kvCache.map { storageMode($0.keys) } ?? "none"))\n"
        report += "  conv:        \(String(format: "%6.1f", Double(convBytes) / 1024)) KB (\(b.convState.map { storageMode($0) } ?? "none"))\n"
        report += "  recurrent:   \(String(format: "%6.1f", Double(recurrentBytes) / 1024)) KB (\(b.recurrentState.map { storageMode($0) } ?? "none"))\n"
        report += "  per-layer:   \(b.perLayerInputs.map { String(format: "%.1f", Double($0.length) / 1024) + " KB" } ?? "none")\n"
        report += "  weights:     \(String(format: "%6.1f", Double(weightBytes) / 1024 / 1024)) MB (\(b.weights.first.map { storageMode($0) } ?? "none"))\n"
        report += "  total intermediate: \(String(format: "%.1f", Double(totalGPUBytes) / 1024 / 1024)) MB\n"
        report += "  total weights:      \(String(format: "%.1f", Double(totalWeightBytes) / 1024 / 1024)) MB\n"
        report += "\nDispatch:\n"
        report += "  decode steps: \(inferenceModel.decodePlan.steps.count)\n"
        report += "  fused entries: \(inferenceModel.decodePlan.fusedEntryCount) (from \(inferenceModel.decodePlan.unfusedEntryCount))\n"
        report += "\nThroughput:\n"
        report += "  prefill: \(promptTokens.count) tokens, \(String(format: "%.1f", Double(promptTokens.count) / prefillTime)) tok/s\n"
        report += "  decode:  \(generateCount) tokens, \(String(format: "%.1f", Double(generateCount) / decodeTime)) tok/s (\(String(format: "%.2f", decodeTime / Double(generateCount) * 1000)) ms/tok)\n"
        report += "  total:   \(totalTokens) tokens in \(String(format: "%.0f", totalTime * 1000))ms\n"
        report += "\nBandwidth (estimate):\n"
        report += "  decode: \(String(format: "%.1f", decodeBandwidth)) GB/s (2× weight read per token)\n"

        print(report)
    }

    @Test("Decode kernel breakdown by type")
    func decodeKernelBreakdown() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: Self.bundlePath,
            optimizer: AggressiveOptimizer()
        )
        var inferenceModel = model

        let profiles = try BenchmarkSupport.profileDecodeSteps(
            model: &inferenceModel,
            device: inferenceModel.device,
            iterations: 20,
            filter: { _ in true }
        )

        // Classify each kernel by category
        struct KernelCategory {
            var name: String
            var totalMicroseconds: Double = 0
            var count: Int = 0
        }
        var categories: [String: KernelCategory] = [:]

        for profile in profiles {
            let name = profile.kernelName.lowercased()
            let category: String
            if name.contains("gemv") || name.contains("gemm") {
                category = "GEMV/GEMM"
            } else if name.contains("flash_attn") || name.contains("attention") {
                category = "Attention"
            } else if name.contains("norm") || name.contains("rms") {
                category = "Norm"
            } else if name.contains("rope") {
                category = "RoPE"
            } else if name.contains("swiglu") || name.contains("gelu") || name.contains("elementwise") {
                category = "Elementwise"
            } else if name.contains("embed") || name.contains("gather") {
                category = "Embedding"
            } else if name.contains("argmax") {
                category = "Argmax"
            } else if name.contains("copy") || name.contains("add") || name.contains("residual") {
                category = "Structural"
            } else if name.contains("per_layer") || name.contains("pli") {
                category = "PerLayerInput"
            } else {
                category = "Other(\(profile.kernelName))"
            }

            var cat = categories[category] ?? KernelCategory(name: category)
            cat.totalMicroseconds += profile.totalMicroseconds / Double(20)
            cat.count += 1
            categories[category] = cat
        }

        let totalMicroseconds = categories.values.reduce(0.0) { $0 + $1.totalMicroseconds }
        let sorted = categories.values.sorted { $0.totalMicroseconds > $1.totalMicroseconds }

        print("\n=== Gemma4-E2B Decode Kernel Breakdown (aggressive, \(profiles.count) steps) ===")
        print("Category             Time(µs)  Count  Avg(µs)    Pct")
        print(String(repeating: "-", count: 54))
        for cat in sorted {
            let avg = cat.count > 0 ? cat.totalMicroseconds / Double(cat.count) : 0
            let pct = totalMicroseconds > 0 ? cat.totalMicroseconds / totalMicroseconds * 100 : 0
            let padded = cat.name.padding(toLength: 20, withPad: " ", startingAt: 0)
            print("\(padded) \(String(format: "%8.0f %6d %8.1f %5.1f%%", cat.totalMicroseconds, cat.count, avg, pct))")
        }
        print(String(repeating: "-", count: 54))
        print("TOTAL                \(String(format: "%8.0f %6d", totalMicroseconds, profiles.count))")
        print("  → \(String(format: "%.2f", totalMicroseconds / 1000)) ms/token")
    }

    @Test("CPU/GPU breakdown: where is decode time spent?")
    func cpuGpuBreakdown() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: Self.bundlePath
        )
        var inferenceModel = model

        let breakdown = try BenchmarkSupport.measureDecodeSyncBreakdown(
            model: &inferenceModel, iterations: 50)

        print("\n=== Gemma4-E2B CPU/GPU Breakdown (50 iterations) ===")
        print("  CPU write:      \(String(format: "%7.0f", breakdown.cpuWriteMicroseconds)) us")
        print("  Encode+submit:  \(String(format: "%7.0f", breakdown.encodeSubmitMicroseconds)) us")
        print("  GPU wait:       \(String(format: "%7.0f", breakdown.waitMicroseconds)) us")
        print("  Readback:       \(String(format: "%7.0f", breakdown.readbackMicroseconds)) us")
        print("  GPU time:       \(String(format: "%7.0f", breakdown.gpuMicroseconds)) us")
        print("  Total:          \(String(format: "%7.0f", breakdown.totalMicroseconds)) us")
        let hostOverhead = breakdown.totalMicroseconds - breakdown.gpuMicroseconds
        let hostPct = hostOverhead / breakdown.totalMicroseconds * 100
        print("  Host overhead:  \(String(format: "%7.0f", hostOverhead)) us (\(String(format: "%.1f", hostPct))%)")
        let steps = inferenceModel.decodePlan.steps
        let barrierCount = steps.filter { $0.barrierPolicy.isBarrier }.count
        let pipelineNames = Set(steps.map { $0.pipeline.label ?? "(none)" })
        print("  Decode steps:   \(steps.count)")
        print("  Barriers:       \(barrierCount) (\(String(format: "%.0f", Double(barrierCount) / Double(steps.count) * 100))%)")
        print("  Unique pipelines: \(pipelineNames.count)")
        print("  us/step (encode): \(String(format: "%.1f", breakdown.encodeSubmitMicroseconds / Double(steps.count)))")

        // Per-decode GPU time using decodeSyncTimed (Metal 4 reusable command buffer)
        inferenceModel.resetCaches()
        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        var tok = inferenceModel.prefill(tokens: promptTokens)
        for _ in 0..<3 { tok = inferenceModel.decodeSync(tokenID: tok) }

        let singleBufIterations = 20
        var gpuTimes: [Double] = []
        for _ in 0..<singleBufIterations {
            let result = inferenceModel.decodeSyncTimed(tokenID: tok)
            tok = result.token
            let gpuMs = (result.gpuEndTime - result.gpuStartTime) * 1000
            gpuTimes.append(gpuMs)
        }
        let medianGpu = gpuTimes.sorted()[gpuTimes.count / 2]
        print("\n  Metal 4 GPU time (median of \(singleBufIterations)): \(String(format: "%.2f", medianGpu)) ms")
    }

    @Test("Prefill completes without error")
    func prefillPipelineDiagnostics() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: Self.bundlePath
        )
        var m = model

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let tok = m.prefill(tokens: promptTokens)
        #expect(m.position == promptTokens.count, "Position should advance by token count")

        // Verify decode chain runs without error
        let tok2 = m.decodeSync(tokenID: tok)
        _ = m.decodeSync(tokenID: tok2)
        #expect(m.position == promptTokens.count + 2, "Position should advance after decode")
    }
}
