import Foundation
import Metal
import Testing
@testable import MetalCompiler

#if ENABLE_METAL_PROBES
/// Gemma 4 E2B Q6-weight vs BF16-weight token quality.
///
/// Purpose: Verify the Q6 decode path produces healthy output, not a collapsed
/// degenerate loop. Q6 is the quality sweet spot — near-BF16 output at ~44%
/// weight size. If diversity collapses, the MLX → STAF Q6 repacking or the
/// unified Q6 GEMV dispatch is broken.
///
/// Why not aggregate argmax match: see `Gemma4Q4AgreementTests` — greedy
/// decoding is hyper-sensitive to sub-logit differences. Q6 drifts less than
/// Q4 but still cannot hit 100% against BF16. Assert on diversity only.
@Suite("Gemma4 Q6 Agreement", .serialized)
struct Gemma4Q6AgreementTests {

    static let bf16BundlePath = "/Users/1amageek/Desktop/swift-lm/TestData/gemma-4-E2B-it"
    static let q6BundlePath = "/Users/1amageek/Desktop/swift-lm/TestData/gemma-4-E2B-it-6bit"

    @Test("Q6 vs BF16 token diversity (3 prompts × 30 decode steps)")
    func q6VersusBFloat16Agreement() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let decodeSteps = 30
        let prompts: [(String, [Int32])] = [
            ("japan     ", [818, 5279, 529, 6056, 563]),
            ("fibonacci ", [2063, 10779, 78113, 236769, 236749, 1473]),
            ("hello     ", [9259, 236764, 1217, 659, 611, 3124, 236881]),
        ]

        let policy = InferencePolicy(maximumSequenceLength: 256)

        var baselineTraces: [(String, [Int32])] = []
        try autoreleasepool {
            let (model, _, _) = try BenchmarkSupport.setupFromBundle(
                bundlePath: Self.bf16BundlePath,
                inferencePolicy: policy)
            var m = model
            for (name, tokens) in prompts {
                m.resetState()
                let trace = BenchmarkSupport.decodeTokenTrace(
                    model: &m,
                    promptTokens: tokens,
                    decodeSteps: decodeSteps)
                baselineTraces.append((name, trace))
            }
        }

        var q6Traces: [(String, [Int32])] = []
        try autoreleasepool {
            let (model, _, _) = try BenchmarkSupport.setupFromBundle(
                bundlePath: Self.q6BundlePath,
                inferencePolicy: policy)
            var m = model
            for (name, tokens) in prompts {
                m.resetState()
                let trace = BenchmarkSupport.decodeTokenTrace(
                    model: &m,
                    promptTokens: tokens,
                    decodeSteps: decodeSteps)
                q6Traces.append((name, trace))
            }
        }

        print("\n=== Gemma4-E2B Q6 vs BF16 Token Agreement ===")
        print("  decode_steps=\(decodeSteps)  prompts=\(prompts.count)")
        print(String(repeating: "-", count: 75))
        print("Prompt     Match   Rate    First_Div")
        print(String(repeating: "-", count: 75))

        let minimumUniqueTokens = 20
        var totalMatch = 0
        var totalCompare = 0
        for (index, (name, q6Trace)) in q6Traces.enumerated() {
            let bf16Trace = baselineTraces[index].1
            let compareLength = min(q6Trace.count, bf16Trace.count)

            var matchCount = 0
            var firstDivergence = -1
            for i in 0..<compareLength {
                if q6Trace[i] == bf16Trace[i] {
                    matchCount += 1
                } else if firstDivergence < 0 {
                    firstDivergence = i
                }
            }
            totalMatch += matchCount
            totalCompare += compareLength

            let rate = compareLength > 0 ? Double(matchCount) / Double(compareLength) * 100 : 0
            let divStr = firstDivergence >= 0 ? String(firstDivergence) : "none"
            let bf16Preview = bf16Trace.prefix(15).map { String($0) }.joined(separator: ",")
            let q6Preview = q6Trace.prefix(15).map { String($0) }.joined(separator: ",")
            let padName = name.padding(toLength: 10, withPad: " ", startingAt: 0)
            print("\(padName) \(String(format: "%3d/%3d", matchCount, compareLength))  \(String(format: "%5.1f%%", rate))  \(divStr)")
            print("  BF16: [\(bf16Preview)]")
            print("  Q6  : [\(q6Preview)]")
            let bf16Unique = Set(bf16Trace).count
            let q6Unique = Set(q6Trace).count
            print("  diversity: BF16 \(bf16Unique) unique, Q6 \(q6Unique) unique (of \(compareLength) tokens)")

            #expect(
                bf16Unique >= minimumUniqueTokens,
                "BF16 trace for '\(name.trimmingCharacters(in: .whitespaces))' collapsed: only \(bf16Unique) unique tokens of \(compareLength) (threshold \(minimumUniqueTokens))")
            #expect(
                q6Unique >= minimumUniqueTokens,
                "Q6 trace for '\(name.trimmingCharacters(in: .whitespaces))' collapsed: only \(q6Unique) unique tokens of \(compareLength) (threshold \(minimumUniqueTokens))")
        }

        print(String(repeating: "-", count: 75))
        let aggregate = totalCompare > 0 ? Double(totalMatch) / Double(totalCompare) * 100 : 0
        print(String(format: "Aggregate: %d/%d  (%.2f%%) — informational only, not asserted", totalMatch, totalCompare, aggregate))
        print()
    }
}
#endif
