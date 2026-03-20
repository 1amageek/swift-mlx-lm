import Foundation
import Metal
import Testing
@testable import MetalCompiler

/// Throughput acceptance benchmarks.
///
/// This suite intentionally excludes heavy profiling/diagnostic tests so the
/// reported tok/s is not contaminated by earlier GPU-heavy diagnostics.
@Suite("Benchmark", .serialized)
struct BenchmarkTests {
    @Test("Optimizer comparison: prefill + decode throughput")
    func optimizerComparisonBenchmark() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        let optimizers: [(any DispatchOptimizer, String)] = [
            (NoOptimizer(), "none"),
            (StandardOptimizer(), "standard"),
            (AggressiveOptimizer(), "aggressive"),
        ]

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let prefillLength = 64
        let decodeSteps = 50
        let iterations = 5

        print("\n=== Optimizer Comparison: LFM2.5-1.2B (\(iterations) iterations) ===")
        print("Optimizer     Decode Pfill  Dec tok/s Pfl tok/s Dec ms/tk Pfl ms/tk")
        print(String(repeating: "-", count: 72))

        for (opt, name) in optimizers {
            let (model, _) = try BenchmarkSupport.setupOrSkip(optimizer: opt)
            var m = model

            var decResults: [Double] = []
            var pfResults: [Double] = []
            BenchmarkSupport.settleGPU()

            for _ in 0..<iterations {
                m.resetCaches()
                let prefillTokens = [Int32](repeating: 1, count: prefillLength)
                let pfStart = CFAbsoluteTimeGetCurrent()
                _ = m.prefill(tokens: prefillTokens)
                let pfTime = CFAbsoluteTimeGetCurrent() - pfStart
                pfResults.append(pfTime)

                m.resetCaches()
                var currentToken = m.prefill(tokens: promptTokens)
                for _ in 0..<3 { currentToken = m.decodeSync(tokenID: currentToken) }

                let decStart = CFAbsoluteTimeGetCurrent()
                for _ in 0..<decodeSteps { currentToken = m.decodeSync(tokenID: currentToken) }
                let decTime = CFAbsoluteTimeGetCurrent() - decStart
                decResults.append(decTime)
            }

            let decMedian = decResults.sorted()[iterations / 2]
            let pfMedian = pfResults.sorted()[iterations / 2]

            let decTokPerSec = Double(decodeSteps) / decMedian
            let pfTokPerSec = Double(prefillLength) / pfMedian
            let decMsPerTok = decMedian / Double(decodeSteps) * 1000
            let pfMsPerTok = pfMedian / Double(prefillLength) * 1000
            let decDispatches = m.plan.fusedEntryCount
            let pfSteps = m.prefillPlan?.stepCount ?? 0

            let pad = name.padding(toLength: 12, withPad: " ", startingAt: 0)
            print("\(pad) \(String(format: "%6d %5d %9.1f %9.1f %9.2f %9.2f", decDispatches, pfSteps, decTokPerSec, pfTokPerSec, decMsPerTok, pfMsPerTok))")

            BenchmarkSupport.settleGPU()
        }
    }

    @Test("Prefill throughput (tok/s)")
    func prefillBenchmark() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()
        let (model, _) = try BenchmarkSupport.setupOrSkip()
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
            print("[Benchmark] prefill \(length) tokens: \(String(format: "%.1f", tokPerSec)) tok/s (\(String(format: "%.3f", elapsed * 1000))ms)")
        }
    }

    @Test("Decode throughput (tok/s)")
    func decodeBenchmark() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()
        let (model, _) = try BenchmarkSupport.setupOrSkip()
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
        print("[Benchmark] decode \(decodeSteps) tokens: \(String(format: "%.1f", tokPerSec)) tok/s (\(String(format: "%.2f", msPerToken)) ms/tok)")
    }

    @Test("Aggressive optimizer: decode throughput")
    func aggressiveDecodeBenchmark() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()
        let (model, _) = try BenchmarkSupport.setupOrSkip(optimizer: AggressiveOptimizer())
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
        print("[Benchmark/aggressive] decode \(decodeSteps) tokens: \(String(format: "%.1f", tokPerSec)) tok/s (\(String(format: "%.2f", msPerToken)) ms/tok)")
        print("[Benchmark/aggressive] dispatches: \(inferenceModel.plan.fusedEntryCount) (from \(inferenceModel.plan.unfusedEntryCount))")
    }

    @Test("End-to-end: prefill + decode with memory diagnostics")
    func endToEndBenchmark() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()
        let (model, _) = try BenchmarkSupport.setupOrSkip()
        var inferenceModel = model

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let generateCount = 100

        let b = inferenceModel.plan.buffers
        let hiddenBytes = b.hidden.length
        let scratchBytes = b.scratch.length
        let residualBytes = b.residual.length
        let logitsBytes = b.logits.length
        let kvBytes = (b.kvCache?.keys.length ?? 0) + (b.kvCache?.values.length ?? 0)
        let convBytes = b.convState?.length ?? 0
        let weightBytes = b.weights.reduce(0) { $0 + $1.length }
        let totalGPUBytes = hiddenBytes + scratchBytes + residualBytes + logitsBytes + kvBytes + convBytes
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

        var report = "=== Benchmark: LFM2.5-1.2B ===\n\n"
        report += "Memory Layout:\n"
        report += "  hidden:   \(String(format: "%6.1f", Double(hiddenBytes) / 1024)) KB (\(storageMode(b.hidden)))\n"
        report += "  scratch:  \(String(format: "%6.1f", Double(scratchBytes) / 1024)) KB (\(storageMode(b.scratch)))\n"
        report += "  residual: \(String(format: "%6.1f", Double(residualBytes) / 1024)) KB (\(storageMode(b.residual)))\n"
        report += "  logits:   \(String(format: "%6.1f", Double(logitsBytes) / 1024)) KB (\(storageMode(b.logits)))\n"
        report += "  KV cache: \(String(format: "%6.1f", Double(kvBytes) / 1024 / 1024)) MB (\(b.kvCache.map { storageMode($0.keys) } ?? "none"))\n"
        report += "  conv:     \(String(format: "%6.1f", Double(convBytes) / 1024)) KB (\(b.convState.map { storageMode($0) } ?? "none"))\n"
        report += "  weights:  \(String(format: "%6.1f", Double(weightBytes) / 1024 / 1024)) MB (\(b.weights.first.map { storageMode($0) } ?? "none"))\n"
        report += "  total intermediate: \(String(format: "%.1f", Double(totalGPUBytes) / 1024 / 1024)) MB\n"
        report += "  total weights:      \(String(format: "%.1f", Double(totalWeightBytes) / 1024 / 1024)) MB\n"
        report += "\nDispatch:\n"
        report += "  decode steps: \(inferenceModel.plan.steps.count)\n"
        report += "  fused entries: \(inferenceModel.plan.fusedEntryCount) (from \(inferenceModel.plan.unfusedEntryCount))\n"
        report += "\nThroughput:\n"
        report += "  prefill: \(promptTokens.count) tokens, \(String(format: "%.1f", Double(promptTokens.count) / prefillTime)) tok/s\n"
        report += "  decode:  \(generateCount) tokens, \(String(format: "%.1f", Double(generateCount) / decodeTime)) tok/s (\(String(format: "%.2f", decodeTime / Double(generateCount) * 1000)) ms/tok)\n"
        report += "  total:   \(totalTokens) tokens in \(String(format: "%.0f", totalTime * 1000))ms\n"
        report += "\nBandwidth (estimate):\n"
        report += "  decode: \(String(format: "%.1f", decodeBandwidth)) GB/s (2× weight read per token)\n"

        print(report)
        try report.write(toFile: BenchmarkSupport.outputPath, atomically: true, encoding: .utf8)
    }
}
