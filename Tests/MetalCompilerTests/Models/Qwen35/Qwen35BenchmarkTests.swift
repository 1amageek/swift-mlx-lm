import Foundation
import Metal
import Testing
@testable import MetalCompiler

/// Qwen 3.5 0.8B throughput benchmarks.
///
/// Designed for apples-to-apples comparison with mlx-swift-lm's Qwen35BenchmarkTests.
/// Both stacks load the same underlying weights (Qwen/Qwen3.5-0.8B BF16, text path only).
///
/// Bundle resolution:
///   - Direct: $SWIFTLM_QWEN35_BUNDLE  (env override)
///   - Cache : ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/<hash>/
///
/// If neither is present the test is skipped (`Issue.record`).
#if ENABLE_METAL_PROBES
@Suite("Qwen35 Benchmark", .serialized)
struct Qwen35BenchmarkTests {

    static let modelLabel = "Qwen3.5-0.8B"

    @Test("MLX-aligned prefill + decode throughput (3-run median)")
    func mlxAlignedBenchmark() throws {
        guard let bundlePath = try Self.resolveBundlePath() else {
            Issue.record("Qwen3.5-0.8B bundle not found. Expected ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B or $SWIFTLM_QWEN35_BUNDLE.")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: bundlePath,
            maximumPrefillLength: 128
        )
        var inferenceModel = model

        // Warmup — resident caches, hot Metal kernels.
        do {
            let warmupTokens: [Int32] = Array(repeating: 1, count: 8)
            var tok = inferenceModel.prefill(tokens: warmupTokens)
            for _ in 0..<4 { tok = inferenceModel.decodeSync(tokenID: tok) }
            inferenceModel.resetState()
        }

        print("=== \(Self.modelLabel) BF16 swift-lm benchmark (MLX-aligned) ===")
        print("bundle: \(bundlePath)")
        print("runs per measurement: 3")
        print()

        print("PREFILL (tok/s — prompt tokens divided by time-to-first-token)")
        let prefillLengths = [16, 32, 64, 128]
        for length in prefillLengths {
            var tps: [Double] = []
            var msList: [Double] = []
            for _ in 0..<3 {
                inferenceModel.resetState()
                let tokens = [Int32](repeating: 1, count: length)
                let start = CFAbsoluteTimeGetCurrent()
                _ = inferenceModel.prefill(tokens: tokens)
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                tps.append(Double(length) / elapsed)
                msList.append(elapsed * 1000)
            }
            let s = BenchStats(tps)
            let m = BenchStats(msList)
            print(String(
                format: "  len %3d: median %6.1f tok/s, mean %6.1f ±%.2f (σ/μ %.2f%%) | %.2f ms median",
                length, s.median, s.mean, s.stddev, s.relStddev * 100, m.median))
        }

        print()
        print("DECODE (tok/s — steady-state token generation after prefill)")
        let decodeSteps = 100
        var dtps: [Double] = []
        var dms: [Double] = []
        for _ in 0..<3 {
            inferenceModel.resetState()
            let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
            var tok = inferenceModel.prefill(tokens: promptTokens)
            for _ in 0..<3 { tok = inferenceModel.decodeSync(tokenID: tok) }

            let start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<decodeSteps {
                tok = inferenceModel.decodeSync(tokenID: tok)
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            dtps.append(Double(decodeSteps) / elapsed)
            dms.append(elapsed * 1000 / Double(decodeSteps))
        }
        let ds = BenchStats(dtps)
        let dm = BenchStats(dms)
        print(String(
            format: "  %3d steps: median %5.1f tok/s, mean %5.1f ±%.2f (σ/μ %.2f%%) | %.2f ms/tok median",
            decodeSteps, ds.median, ds.mean, ds.stddev, ds.relStddev * 100, dm.median))
        print()
    }

    // MARK: - Bundle resolution

    private static func resolveBundlePath() throws -> String? {
        if let override = ProcessInfo.processInfo.environment["SWIFTLM_QWEN35_BUNDLE"],
           !override.trimmingCharacters(in: .whitespaces).isEmpty {
            return NSString(string: override).expandingTildeInPath
        }
        let hubRoot = NSString(string: "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots").expandingTildeInPath
        guard FileManager.default.fileExists(atPath: hubRoot) else { return nil }
        let entries = try FileManager.default.contentsOfDirectory(atPath: hubRoot).sorted()
        for entry in entries {
            let candidate = "\(hubRoot)/\(entry)"
            let cfg = "\(candidate)/config.json"
            if FileManager.default.fileExists(atPath: cfg) {
                return candidate
            }
        }
        return nil
    }

    private struct BenchStats {
        let mean: Double
        let median: Double
        let stddev: Double
        init(_ values: [Double]) {
            precondition(!values.isEmpty)
            let sorted = values.sorted()
            let count = values.count
            let sum = values.reduce(0, +)
            let meanValue = sum / Double(count)
            let stddevValue: Double
            if count > 1 {
                let variance = values.reduce(0.0) { acc, v in
                    acc + (v - meanValue) * (v - meanValue)
                } / Double(count - 1)
                stddevValue = variance.squareRoot()
            } else {
                stddevValue = 0
            }
            self.mean = meanValue
            self.median = sorted[count / 2]
            self.stddev = stddevValue
        }
        var relStddev: Double { mean == 0 ? 0 : stddev / mean }
    }
}
#endif
