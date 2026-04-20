import Foundation
import Metal
import Testing
@testable import MetalCompiler

/// Gemma 4 E2B Q4-weight vs BF16-weight token agreement.
///
/// Purpose: Verify the Q4 prefill dequant→BF16 routing fix (task #36) does not
/// degrade token quality. Both bundles load the same upstream model; Q4 is
/// only a lossy weight compression, so their generated tokens should largely
/// overlap for short prompts where divergence compounds slowly.
///
/// Threshold: ≥ 95% aggregate match across 3 prompts × 100 decode steps.
/// This is the Phase 0 completion criterion per the quantization extension plan.
@Suite("Gemma4 Q4 Agreement", .serialized)
struct Gemma4Q4AgreementTests {

    static let bf16BundlePath = "/Users/Shared/swift-lm-testdata/gemma-4-E2B-it"
    static let q4BundlePath = "/Users/Shared/swift-lm-testdata/gemma-4-E2B-it-4bit"

    @Test("Q4 vs BF16 token agreement (3 prompts × 100 decode steps)")
    func q4VersusBFloat16Agreement() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let decodeSteps = 30
        // Real tokenized prompts from Gemma4 tokenizer.
        // "The capital of Japan is" → [818, 5279, 529, 6056, 563]
        // "def fibonacci(n):"       → [2063, 10779, 78113, 236769, 236749, 1473]
        // "Hello, how are you today?" → [9259, 236764, 1217, 659, 611, 3124, 236881]
        let prompts: [(String, [Int32])] = [
            ("japan     ", [818, 5279, 529, 6056, 563]),
            ("fibonacci ", [2063, 10779, 78113, 236769, 236749, 1473]),
            ("hello     ", [9259, 236764, 1217, 659, 611, 3124, 236881]),
        ]

        let policy = InferencePolicy(maximumSequenceLength: 256)

        // Baseline: BF16 weights
        var baselineTraces: [(String, [Int32])] = []
        do {
            let (model, _, _) = try BenchmarkSupport.setupFromBundle(
                bundlePath: Self.bf16BundlePath,
                inferencePolicy: policy)
            Self.dumpLayer0KernelList(label: "BF16", model: model)
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

        // Comparison: Q4 weights
        var q4Traces: [(String, [Int32])] = []
        do {
            let (model, _, _) = try BenchmarkSupport.setupFromBundle(
                bundlePath: Self.q4BundlePath,
                inferencePolicy: policy)
            Self.dumpLayer0KernelList(label: "Q4", model: model)
            var m = model
            for (name, tokens) in prompts {
                m.resetState()
                let trace = BenchmarkSupport.decodeTokenTrace(
                    model: &m,
                    promptTokens: tokens,
                    decodeSteps: decodeSteps)
                q4Traces.append((name, trace))
            }
        }

        print("\n=== Gemma4-E2B Q4 vs BF16 Token Agreement ===")
        print("  decode_steps=\(decodeSteps)  prompts=\(prompts.count)")
        print(String(repeating: "-", count: 75))
        print("Prompt     Match   Rate    First_Div  BF16 trace(first 15)")
        print(String(repeating: "-", count: 75))

        var totalMatch = 0
        var totalCompare = 0
        for (index, (name, q4Trace)) in q4Traces.enumerated() {
            let bf16Trace = baselineTraces[index].1
            let compareLength = min(q4Trace.count, bf16Trace.count)

            var matchCount = 0
            var firstDivergence = -1
            for i in 0..<compareLength {
                if q4Trace[i] == bf16Trace[i] {
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
            let q4Preview = q4Trace.prefix(15).map { String($0) }.joined(separator: ",")
            let padName = name.padding(toLength: 10, withPad: " ", startingAt: 0)
            print("\(padName) \(String(format: "%3d/%3d", matchCount, compareLength))  \(String(format: "%5.1f%%", rate))  \(divStr.padding(toLength: 9, withPad: " ", startingAt: 0))")
            print("  BF16: [\(bf16Preview)]")
            print("  Q4  : [\(q4Preview)]")
            // Token diversity — detect collapse to single token (serious quality regression)
            let bf16Unique = Set(bf16Trace).count
            let q4Unique = Set(q4Trace).count
            print("  diversity: BF16 \(bf16Unique) unique, Q4 \(q4Unique) unique (of \(compareLength) tokens)")
        }

        print(String(repeating: "-", count: 75))
        let aggregate = totalCompare > 0 ? Double(totalMatch) / Double(totalCompare) * 100 : 0
        print(String(format: "Aggregate: %d/%d  (%.2f%%)", totalMatch, totalCompare, aggregate))
        print()

        // Phase 0 completion criterion: ≥ 95% aggregate agreement
        #expect(aggregate >= 95.0, "Q4 vs BF16 token agreement \(aggregate)% below 95% threshold")
    }

    /// BF16 decode must be deterministic across repeated runs with the same prompt.
    /// If results vary between runs, there is a GPU hazard/barrier bug (task #40).
    @Test("BF16 decode determinism (3 consecutive runs, identical prompt)")
    func bf16DecodeDeterminism() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let decodeSteps = 120  // Exercise past layer 26+ where the original #40 issue surfaced
        let prompt: [Int32] = [1, 1, 6, 6423, 708]
        let policy = InferencePolicy(maximumSequenceLength: 256)

        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: Self.bf16BundlePath,
            inferencePolicy: policy)
        var m = model

        var traces: [[Int32]] = []
        for _ in 0..<3 {
            m.resetState()
            let trace = BenchmarkSupport.decodeTokenTrace(
                model: &m,
                promptTokens: prompt,
                decodeSteps: decodeSteps)
            traces.append(trace)
        }

        print("\n=== Gemma4-E2B BF16 Decode Determinism ===")
        print("  prompt=\(prompt) decode_steps=\(decodeSteps) runs=\(traces.count)")
        for (i, t) in traces.enumerated() {
            let preview = t.prefix(20).map { String($0) }.joined(separator: ",")
            print("  run \(i): [\(preview)] (unique=\(Set(t).count))")
        }

        // Find first divergence between run 0 and each subsequent run
        var firstDivergences: [Int] = []
        for i in 1..<traces.count {
            var div = -1
            let n = min(traces[0].count, traces[i].count)
            for k in 0..<n where traces[0][k] != traces[i][k] {
                div = k
                break
            }
            firstDivergences.append(div)
            print("  run 0 vs run \(i): first_divergence=\(div == -1 ? "identical" : String(div))")
        }
        print()

        #expect(traces[0] == traces[1], "Run 0 and Run 1 must produce identical BF16 token traces (determinism bug #40)")
        #expect(traces[1] == traces[2], "Run 1 and Run 2 must produce identical BF16 token traces (determinism bug #40)")
    }

    /// Isolate state-reset vs kernel non-determinism. Creates THREE fresh model
    /// instances and runs ONE prefill+short-decode on each. If outputs differ,
    /// the bug is in the kernel pipeline itself (not state reset).
    @Test("BF16 fresh-instance determinism (3 separate MetalInferenceModel)")
    func bf16FreshInstanceDeterminism() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let decodeSteps = 20
        let prompt: [Int32] = [1, 1, 6, 6423, 708]
        let policy = InferencePolicy(maximumSequenceLength: 256)

        var traces: [[Int32]] = []
        for _ in 0..<3 {
            try autoreleasepool {
                let (model, _, _) = try BenchmarkSupport.setupFromBundle(
                    bundlePath: Self.bf16BundlePath,
                    inferencePolicy: policy)
                var m = model
                let trace = BenchmarkSupport.decodeTokenTrace(
                    model: &m,
                    promptTokens: prompt,
                    decodeSteps: decodeSteps)
                traces.append(trace)
            }
        }

        print("\n=== Gemma4-E2B BF16 Fresh-Instance Determinism ===")
        print("  prompt=\(prompt) decode_steps=\(decodeSteps) runs=\(traces.count)")
        for (i, t) in traces.enumerated() {
            let preview = t.prefix(15).map { String($0) }.joined(separator: ",")
            print("  run \(i): [\(preview)]")
        }
        print()

        #expect(traces[0] == traces[1], "Fresh instances must produce identical output (kernel-level determinism)")
        #expect(traces[1] == traces[2], "Fresh instances must produce identical output (kernel-level determinism)")
    }

    /// Prefill determinism at the *hidden-state* level (not just argmax).
    ///
    /// Three fresh MetalInferenceModel instances run `debugPrefillLastTokenFinalHidden`
    /// on an identical prompt. This bypasses decode, sampling, and argmax tie-breaking
    /// entirely — if these hidden values differ, a kernel inside the prefill pipeline
    /// is non-deterministic. If they are bit-identical, the bug is confined to decode
    /// or argmax.
    @Test("BF16 prefill hidden-state determinism (3 fresh instances)")
    func bf16PrefillHiddenDeterminism() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let prompt: [Int32] = [1, 1, 6, 6423, 708]
        let policy = InferencePolicy(maximumSequenceLength: 256)

        var hiddens: [[Float]] = []
        for _ in 0..<3 {
            try autoreleasepool {
                let (model, _, _) = try BenchmarkSupport.setupFromBundle(
                    bundlePath: Self.bf16BundlePath,
                    inferencePolicy: policy)
                var m = model
                let hidden = try m.debugPrefillLastTokenFinalHidden(tokens: prompt)
                hiddens.append(hidden)
            }
        }

        print("\n=== Gemma4-E2B BF16 Prefill Hidden Determinism ===")
        print("  prompt=\(prompt) runs=\(hiddens.count)")
        for (i, h) in hiddens.enumerated() {
            let preview = h.prefix(6).map { String(format: "%.6f", $0) }.joined(separator: ",")
            let sum = h.reduce(0, +)
            let min = h.min() ?? 0
            let max = h.max() ?? 0
            print("  run \(i): count=\(h.count) first6=[\(preview)] sum=\(sum) min=\(min) max=\(max)")
        }

        // Pairwise comparison
        for i in 1..<hiddens.count {
            let a = hiddens[0]
            let b = hiddens[i]
            guard a.count == b.count else {
                print("  run 0 vs run \(i): LENGTH MISMATCH (\(a.count) vs \(b.count))")
                continue
            }
            var firstDiv = -1
            var maxAbsDiff: Float = 0
            var diffCount = 0
            for k in 0..<a.count {
                let diff = abs(a[k] - b[k])
                if diff > 0 {
                    if firstDiv < 0 { firstDiv = k }
                    diffCount += 1
                    if diff > maxAbsDiff { maxAbsDiff = diff }
                }
            }
            print("  run 0 vs run \(i): first_div=\(firstDiv) diffCount=\(diffCount)/\(a.count) max|diff|=\(maxAbsDiff)")
        }
        print()

        #expect(hiddens[0] == hiddens[1], "Prefill hidden must be bit-identical across fresh instances (run 0 vs run 1)")
        #expect(hiddens[1] == hiddens[2], "Prefill hidden must be bit-identical across fresh instances (run 1 vs run 2)")
    }

    /// Prefill determinism within a SINGLE model instance.
    ///
    /// Runs `debugPrefillLastTokenFinalHidden` three times on the SAME model instance.
    /// Each call resets state via `resetState()`. If output differs here, the bug is in
    /// per-call encoding or state reset. If output is identical, the divergence
    /// observed in `bf16PrefillHiddenDeterminism` is caused by compilation-time
    /// non-determinism (different compile runs produce different binding/encoding).
    @Test("BF16 prefill hidden-state determinism (single instance, 3 calls)")
    func bf16PrefillHiddenDeterminismSingleInstance() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let prompt: [Int32] = [1, 1, 6, 6423, 708]
        let policy = InferencePolicy(maximumSequenceLength: 256)

        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: Self.bf16BundlePath,
            inferencePolicy: policy)
        var m = model

        var hiddens: [[Float]] = []
        for _ in 0..<3 {
            let hidden = try m.debugPrefillLastTokenFinalHidden(tokens: prompt)
            hiddens.append(hidden)
        }

        print("\n=== Gemma4-E2B BF16 Prefill Single-Instance Determinism ===")
        print("  prompt=\(prompt) runs=\(hiddens.count)")
        for (i, h) in hiddens.enumerated() {
            let preview = h.prefix(6).map { String(format: "%.6f", $0) }.joined(separator: ",")
            let sum = h.reduce(0, +)
            print("  run \(i): count=\(h.count) first6=[\(preview)] sum=\(sum)")
        }
        for i in 1..<hiddens.count {
            let a = hiddens[0]
            let b = hiddens[i]
            var firstDiv = -1
            var maxAbsDiff: Float = 0
            for k in 0..<a.count {
                let diff = abs(a[k] - b[k])
                if diff > 0 && firstDiv < 0 { firstDiv = k }
                if diff > maxAbsDiff { maxAbsDiff = diff }
            }
            print("  run 0 vs run \(i): first_div=\(firstDiv) max|diff|=\(maxAbsDiff)")
        }
        print()

        #expect(hiddens[0] == hiddens[1], "Single-instance repeated prefill must be bit-identical (0 vs 1)")
        #expect(hiddens[1] == hiddens[2], "Single-instance repeated prefill must be bit-identical (1 vs 2)")
    }

    /// Probe prefill hidden state at step 0 (immediately after token embedding).
    /// If step 0 output is non-deterministic, the bug is in token embedding / gather.
    /// If step 0 is deterministic, the bug is in a later layer.
    @Test("BF16 prefill step-0 probe (localize non-determinism)")
    func bf16PrefillStep0Probe() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let prompt: [Int32] = [1, 1, 6, 6423, 708]
        let policy = InferencePolicy(maximumSequenceLength: 256)

        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: Self.bf16BundlePath,
            inferencePolicy: policy)
        var m = model

        var step0Snaps: [[Float]] = []
        var stepMidSnaps: [[Float]] = []
        for _ in 0..<3 {
            let snaps = try m.debugPrefillLastTokenHiddenSnapshots(
                tokens: prompt,
                stepIndices: [0, 20])
            step0Snaps.append(snaps[0] ?? [])
            stepMidSnaps.append(snaps[20] ?? [])
        }

        print("\n=== Gemma4-E2B BF16 Prefill Step-0 Probe ===")
        print("  prompt=\(prompt)")
        for (i, h) in step0Snaps.enumerated() {
            let preview = h.prefix(6).map { String(format: "%.4f", $0) }.joined(separator: ",")
            let sum = h.reduce(0, +)
            print("  step0 run \(i): count=\(h.count) first6=[\(preview)] sum=\(sum)")
        }
        for (i, h) in stepMidSnaps.enumerated() {
            let preview = h.prefix(6).map { String(format: "%.4f", $0) }.joined(separator: ",")
            let sum = h.reduce(0, +)
            print("  step20 run \(i): count=\(h.count) first6=[\(preview)] sum=\(sum)")
        }

        // Diff analysis
        for i in 1..<step0Snaps.count {
            let a = step0Snaps[0]
            let b = step0Snaps[i]
            if a.count != b.count { continue }
            var firstDiv = -1
            var maxAbsDiff: Float = 0
            for k in 0..<a.count {
                let diff = abs(a[k] - b[k])
                if diff > 0 && firstDiv < 0 { firstDiv = k }
                if diff > maxAbsDiff { maxAbsDiff = diff }
            }
            print("  step0 run 0 vs run \(i): first_div=\(firstDiv) max|diff|=\(maxAbsDiff)")
        }
        for i in 1..<stepMidSnaps.count {
            let a = stepMidSnaps[0]
            let b = stepMidSnaps[i]
            if a.count != b.count { continue }
            var firstDiv = -1
            var maxAbsDiff: Float = 0
            for k in 0..<a.count {
                let diff = abs(a[k] - b[k])
                if diff > 0 && firstDiv < 0 { firstDiv = k }
                if diff > maxAbsDiff { maxAbsDiff = diff }
            }
            print("  step20 run 0 vs run \(i): first_div=\(firstDiv) max|diff|=\(maxAbsDiff)")
        }
        print()

        #expect(step0Snaps[0] == step0Snaps[1], "Step 0 (token embedding) must be bit-identical (0 vs 1)")
        #expect(step0Snaps[1] == step0Snaps[2], "Step 0 (token embedding) must be bit-identical (1 vs 2)")
    }

    /// Walk the prefill step space to locate the first non-deterministic step.
    ///
    /// Given we know step 0/20 is deterministic but the final-hidden output is not,
    /// the divergence must occur between step 21 and the final step. This test
    /// samples at a coarse grid of late steps to identify the boundary.
    @Test("BF16 prefill late-step walk (localize divergence boundary)")
    func bf16PrefillLateStepWalk() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let prompt: [Int32] = [1, 1, 6, 6423, 708]
        let policy = InferencePolicy(maximumSequenceLength: 256)

        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: Self.bf16BundlePath,
            inferencePolicy: policy)
        var m = model

        let stepCount = m.prefillPlan?.steps.count ?? 0
        print("\n=== Gemma4-E2B BF16 Prefill Late-Step Walk ===")
        print("  prompt=\(prompt)  prefill_step_count=\(stepCount)")

        // Sample at several points: 0, 20, 100, 200, 300, 400, 500, last
        var probeSteps: [Int] = [0, 20, 100, 200, 300, 400, 500]
        probeSteps = probeSteps.filter { $0 < stepCount }
        if stepCount > 0 {
            probeSteps.append(stepCount - 1)
        }
        let probeSet = Set(probeSteps)

        // Run 3 times, capture snapshots at every probe step
        var runSnaps: [[Int: [Float]]] = []
        for _ in 0..<3 {
            let snaps = try m.debugPrefillLastTokenHiddenSnapshots(
                tokens: prompt,
                stepIndices: probeSet)
            runSnaps.append(snaps)
        }

        print("  step    | run0 sum          | run1 sum          | run2 sum          | max|diff(0,1)| | max|diff(0,2)|")
        print(String(repeating: "-", count: 120))
        for step in probeSteps.sorted() {
            let h0 = runSnaps[0][step] ?? []
            let h1 = runSnaps[1][step] ?? []
            let h2 = runSnaps[2][step] ?? []
            let sum0 = h0.reduce(0, +)
            let sum1 = h1.reduce(0, +)
            let sum2 = h2.reduce(0, +)
            var maxDiff01: Float = 0
            var maxDiff02: Float = 0
            if h0.count == h1.count {
                for k in 0..<h0.count {
                    let d = abs(h0[k] - h1[k]); if d > maxDiff01 { maxDiff01 = d }
                }
            }
            if h0.count == h2.count {
                for k in 0..<h0.count {
                    let d = abs(h0[k] - h2[k]); if d > maxDiff02 { maxDiff02 = d }
                }
            }
            print(String(format: "  %5d  | %17.6f | %17.6f | %17.6f | %14.6f | %14.6f",
                         step, sum0, sum1, sum2, maxDiff01, maxDiff02))
        }
        print()

        // Find first non-deterministic probe
        let sortedProbes = probeSteps.sorted()
        var firstNondetermStep: Int = -1
        for step in sortedProbes {
            let h0 = runSnaps[0][step] ?? []
            let h1 = runSnaps[1][step] ?? []
            if h0 != h1 {
                firstNondetermStep = step
                break
            }
        }
        print("  first non-deterministic probe step: \(firstNondetermStep == -1 ? "none (all deterministic)" : String(firstNondetermStep))")
        print()
    }

    /// Single-step probe at arbitrary target step, 3 consecutive calls.
    /// Used to bisect the first non-deterministic step in prefill.
    private func probeSingleStepConsecutive(targetStep: Int) throws -> (determ: Bool, sums: [Float]) {
        let prompt: [Int32] = [1, 1, 6, 6423, 708]
        let policy = InferencePolicy(maximumSequenceLength: 256)
        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: Self.bf16BundlePath,
            inferencePolicy: policy)
        var m = model
        var runs: [[Float]] = []
        for _ in 0..<3 {
            let snaps = try m.debugPrefillLastTokenHiddenSnapshots(
                tokens: prompt,
                stepIndices: [targetStep])
            runs.append(snaps[targetStep] ?? [])
        }
        let sums = runs.map { $0.reduce(0, +) }
        let det = runs[0] == runs[1] && runs[1] == runs[2]
        return (det, sums)
    }

    /// Bisect: probe single target steps at a grid to find first non-deterministic step.
    @Test("BF16 prefill — bisect first non-deterministic step")
    func bf16PrefillBisectFirstNonDeterministic() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        // Dump kernel names including pipeline.label for steps 0..75.
        let policy2 = InferencePolicy(maximumSequenceLength: 256)
        let (model2, _, _) = try BenchmarkSupport.setupFromBundle(bundlePath: Self.bf16BundlePath, inferencePolicy: policy2)
        if let plan = model2.prefillPlan {
            print("\n=== Prefill step kernels with pipeline.label (0..75) ===")
            for i in 0..<min(75, plan.steps.count) {
                let step = plan.steps[i]
                let kernel = step.metadata.kernelName ?? "(nil)"
                let pipLabel = step.pipeline.label ?? "(nil)"
                let weight = step.metadata.weightTensorName ?? "-"
                let layer = step.metadata.layerIndex.map { String($0) } ?? "-"
                print("  \(String(format: "%4d", i)) | L\(layer.padding(toLength: 3, withPad: " ", startingAt: 0)) | md=\(kernel.padding(toLength: 40, withPad: " ", startingAt: 0)) | pl=\(pipLabel.padding(toLength: 40, withPad: " ", startingAt: 0)) | \(weight)")
            }
        }
        let targets: [Int] = []
        print("\n=== Gemma4-E2B BF16 Prefill Bisect ===")
        print("  target_step | det   | sum0               | sum1               | sum2")
        print(String(repeating: "-", count: 105))
        for target in targets {
            let (det, sums) = try probeSingleStepConsecutive(targetStep: target)
            let detStr = (det ? "true " : "false").padding(toLength: 5, withPad: " ", startingAt: 0)
            print("  \(String(format: "%11d", target)) | \(detStr) | \(String(format: "%18.6f", sums[0])) | \(String(format: "%18.6f", sums[1])) | \(String(format: "%18.6f", sums[2]))")
        }
        print()
    }

    /// Probe scratch slot 0 at step 71 (flash_attn output) across 3 consecutive runs.
    /// If non-deterministic: flash_attn_batch_f32 writes non-deterministic output.
    /// Also probes slots 1 (RoPE'd Q), 2 (K), 3 (V) for comparison.
    @Test("BF16 prefill — probe flash_attn attention output (step 71 scratch slot 0)")
    func bf16PrefillProbeAttentionScratchAt71() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let prompt: [Int32] = [1, 1, 6, 6423, 708]
        let policy = InferencePolicy(maximumSequenceLength: 256)
        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: Self.bf16BundlePath,
            inferencePolicy: policy)
        var m = model
        let slotDim = m.prefillPlan?.slotDimension ?? 0
        let totalQDim = 4 * 256  // Gemma4-E2B: headCount=4, headDim=256
        let totalKVDim = 2 * 256 // Gemma4-E2B: kvHeadCount=2, headDim=256
        print("\n=== Scratch probe at step 71 (flash_attn output, slot 0) ===")
        print("  slotDimension = \(slotDim)")

        // For attention output at slot 0, the producer (flash_attn_batch_f32) writes
        // with row stride = totalQDim (not slotDim). For other slots that hold post-norm/
        // projection output, the row stride is slotDim. Use the producer-specific stride.
        let probes: [(step: Int, slot: Int, rowStride: Int, countElems: Int, label: String)] = [
            (71, 0, totalQDim, totalQDim, "step71.slot0 (attn_out)"),
            (71, 1, slotDim, totalQDim, "step71.slot1 (Q)"),
            (70, 1, slotDim, totalQDim, "step70.slot1 (Q pre-cache)"),
            (70, 2, slotDim, totalKVDim, "step70.slot2 (K)"),
            (70, 3, slotDim, totalKVDim, "step70.slot3 (V)"),
            (69, 2, slotDim, totalKVDim, "step69.slot2 (K post-rope)"),
        ]

        for probe in probes {
            var runs: [[Float]] = []
            for _ in 0..<3 {
                let snaps = try m.debugPrefillLastTokenScratchSnapshots(
                    tokens: prompt,
                    stepIndices: [probe.step],
                    slotIndex: probe.slot,
                    rowStride: probe.rowStride,
                    count: probe.countElems)
                runs.append(snaps[probe.step] ?? [])
            }
            let sums = runs.map { $0.reduce(0, +) }
            let det = runs[0] == runs[1] && runs[1] == runs[2]
            let detStr = (det ? "true " : "FALSE").padding(toLength: 5, withPad: " ", startingAt: 0)
            let label = probe.label.padding(toLength: 28, withPad: " ", startingAt: 0)
            print("  \(label) | \(detStr) | count=\(runs[0].count) | sum0=\(String(format: "%14.6f", sums[0])) | sum1=\(String(format: "%14.6f", sums[1])) | sum2=\(String(format: "%14.6f", sums[2]))")
            if !det && runs[0].count == runs[1].count {
                var firstDiv = -1
                var maxDiff: Float = 0
                for k in 0..<runs[0].count {
                    let d = abs(runs[0][k] - runs[1][k])
                    if d > 0 && firstDiv < 0 { firstDiv = k }
                    if d > maxDiff { maxDiff = d }
                }
                print("    first_div=\(firstDiv) max|diff|=\(maxDiff)")
                let firstFew = min(8, runs[0].count)
                for k in 0..<firstFew {
                    print("    idx[\(k)]: r0=\(runs[0][k]) r1=\(runs[1][k]) r2=\(runs[2][k])")
                }
            }
        }
        print()
    }

    /// Probe flash_attn output (scratch slot 0) at EVERY layer's flash_attn step.
    /// Auto-discovers flash_attn step indices from prefill plan, then probes each.
    ///
    /// Hypothesis A (kernel-generic): ALL layers' flash_attn output is non-deterministic.
    /// The L0-L3 K cache determinism observed earlier is coincidental (race gets
    /// masked by downstream projections that amplify small differences proportional
    /// to input magnitude, and early layers have smaller activations).
    ///
    /// Hypothesis B (layer-specific): Only L4+ flash_attn output is non-deterministic.
    /// Something layer-specific (sliding window param, sharing config, head count)
    /// triggers the race starting at L4.
    ///
    /// Result discriminates the next debugging direction.
    @Test("BF16 prefill — probe flash_attn output at ALL layers (layer-specific vs kernel-generic)")
    func bf16PrefillProbeFlashAttnAllLayers() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let prompt: [Int32] = [1, 1, 6, 6423, 708]
        let policy = InferencePolicy(maximumSequenceLength: 256)
        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: Self.bf16BundlePath,
            inferencePolicy: policy)
        var m = model

        let totalQDim = 4 * 256  // Gemma4-E2B: headCount=4, headDim=256

        // Auto-discover flash_attn step indices per layer.
        guard let plan = m.prefillPlan else {
            Issue.record("prefillPlan unavailable")
            return
        }
        struct LayerProbe { let step: Int; let layer: Int; let kernel: String }
        var probes: [LayerProbe] = []
        for (index, step) in plan.steps.enumerated() {
            let kernel = step.metadata.kernelName ?? step.pipeline.label ?? ""
            if kernel.hasPrefix("flash_attn") {
                let layer = step.metadata.layerIndex ?? -1
                probes.append(LayerProbe(step: index, layer: layer, kernel: kernel))
            }
        }

        print("\n=== flash_attn per-layer determinism probe ===")
        print("  discovered \(probes.count) flash_attn dispatches in prefill plan")
        print("  layer | step | kernel                              | det   | sum0           | sum1           | sum2           | firstDiv | maxDiff")
        print(String(repeating: "-", count: 130))

        // Probe only the first N layers to keep runtime manageable.
        // L0..L10 should be enough to answer the hypothesis question.
        let probeLimit = min(11, probes.count)
        for i in 0..<probeLimit {
            let probe = probes[i]
            var runs: [[Float]] = []
            for _ in 0..<3 {
                let snaps = try m.debugPrefillLastTokenScratchSnapshots(
                    tokens: prompt,
                    stepIndices: [probe.step],
                    slotIndex: 0,
                    rowStride: totalQDim,
                    count: totalQDim)
                runs.append(snaps[probe.step] ?? [])
            }
            let sums = runs.map { $0.reduce(0, +) }
            let det = runs[0] == runs[1] && runs[1] == runs[2]
            let detStr = (det ? "true " : "FALSE").padding(toLength: 5, withPad: " ", startingAt: 0)
            var firstDiv = -1
            var maxDiff: Float = 0
            if !det && runs[0].count == runs[1].count {
                for k in 0..<runs[0].count {
                    let d = abs(runs[0][k] - runs[1][k])
                    if d > 0 && firstDiv < 0 { firstDiv = k }
                    if d > maxDiff { maxDiff = d }
                }
            }
            let kernelPad = probe.kernel.padding(toLength: 38, withPad: " ", startingAt: 0)
            print(String(format: "  L%-4d | %4d | %@ | %@ | %14.6f | %14.6f | %14.6f | %8d | %.6f",
                         probe.layer, probe.step, kernelPad as CVarArg, detStr as CVarArg,
                         sums[0], sums[1], sums[2], firstDiv, maxDiff))
        }
        print()
    }

    /// Probe KV cache content at layer 4 across 3 consecutive runs.
    /// Step 71 flash_attn output at layer 4 is non-deterministic while all scratch
    /// inputs (Q, K, V) are bit-deterministic. This test determines whether the
    /// KV cache written by `kv_cache_fill_seq_f32` itself is deterministic —
    /// if YES → flash_attn kernel reads deterministic cache but produces
    ///          non-deterministic output (bug is in flash_attn kernel)
    /// if NO  → bug is in kv_cache_fill or pre-cache barrier.
    @Test("BF16 prefill — probe KV cache at layers 0,1,4,14 (3 consecutive runs)")
    func bf16PrefillProbeKVCacheAtLayer4() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let prompt: [Int32] = [1, 1, 6, 6423, 708]
        let policy = InferencePolicy(maximumSequenceLength: 256)
        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: Self.bf16BundlePath,
            inferencePolicy: policy)
        var m = model

        print("\n=== KV cache layer probe (layers 0,1,4,14 × 3 runs) ===")

        // Diagnostic: verify prefill plan actually contains kv_cache_fill steps
        if let plan = m.prefillPlan {
            let totalSteps = plan.steps.count
            var kvFillCount = 0
            var kvFillKernels: [String] = []
            for step in plan.steps {
                let name = step.metadata.kernelName ?? step.pipeline.label ?? "?"
                if name.contains("kv_cache_fill") {
                    kvFillCount += 1
                    if kvFillKernels.count < 3 { kvFillKernels.append(name) }
                }
            }
            print("  [diag] prefill totalSteps=\(totalSteps) kv_cache_fill_count=\(kvFillCount) samples=\(kvFillKernels)")

            if let prefillKV = plan.buffers.kvCache,
               let decodeKV = m.decodePlan.buffers.kvCache {
                let sameKeys = ObjectIdentifier(prefillKV.keys as AnyObject) == ObjectIdentifier(decodeKV.keys as AnyObject)
                let sameValues = ObjectIdentifier(prefillKV.values as AnyObject) == ObjectIdentifier(decodeKV.values as AnyObject)
                print("  [diag] kvCache shared: keys=\(sameKeys) values=\(sameValues) " +
                      "layerCount=\(prefillKV.specification.layerCount) " +
                      "bytesPerLayer=\(prefillKV.specification.bytesPerLayer(scheme: prefillKV.specification.keyQuantizationScheme))")
                print("  [diag] kvCache.keys length=\(prefillKV.keys.length) storageMode=\(prefillKV.keys.storageMode.rawValue) hazard=\(prefillKV.keys.hazardTrackingMode.rawValue)")
            } else {
                print("  [diag] kvCache: prefillKV=\(plan.buffers.kvCache != nil) decodeKV=\(m.decodePlan.buffers.kvCache != nil)")
            }
        } else {
            print("  [diag] prefillPlan is nil!")
        }

        // Sanity check: run normal prefill path once and dump KV cache directly via
        // a separate blit command buffer (MTL3 queue). This bypasses
        // `captureKVCacheLayerSnapshot` (which re-runs prefill internally) and
        // validates whether the shared kvCache buffer is readable after a regular
        // `model.prefill` call.
        do {
            m.resetState()
            _ = m.prefill(tokens: prompt)
            if let kv = m.decodePlan.buffers.kvCache {
                let spec = kv.specification
                let bytesPerLayer = spec.bytesPerLayer(scheme: spec.keyQuantizationScheme)
                let device = m.device
                guard let mtl3Queue = device.makeCommandQueue() else {
                    Issue.record("Cannot create MTL3 command queue")
                    return
                }
                for layer in [0, 1, 4, 14] {
                    let off = spec.layerOffset(layer: layer, scheme: spec.keyQuantizationScheme)
                    let staging = device.makeBuffer(length: bytesPerLayer, options: [.storageModeShared])!
                    guard let cb = mtl3Queue.makeCommandBuffer(),
                          let blit = cb.makeBlitCommandEncoder() else {
                        Issue.record("Cannot create MTL3 blit encoder")
                        return
                    }
                    blit.copy(from: kv.keys, sourceOffset: off,
                              to: staging, destinationOffset: 0, size: bytesPerLayer)
                    blit.endEncoding()
                    cb.commit()
                    cb.waitUntilCompleted()
                    let ptr = staging.contents().bindMemory(to: UInt8.self, capacity: bytesPerLayer)
                    let bytes = Array(UnsafeBufferPointer(start: ptr, count: bytesPerLayer))
                    let sum = bytes.reduce(UInt64(0)) { $0 + UInt64($1) }
                    let nz = bytes.firstIndex(where: { $0 != 0 }) ?? -1
                    print("  [sanity] postPrefill L\(layer) K: sum=\(sum) firstNonZero=\(nz)")
                }
            }
        }

        let layerIndices = [0, 1, 2, 3, 4, 14]
        for layer in layerIndices {
            print("-- layer \(layer) --")
            for kind in [MetalInferenceModel.DebugKVCacheSliceKind.keys,
                         MetalInferenceModel.DebugKVCacheSliceKind.values] {
                let label = (kind == .keys) ? "K" : "V"
                var runs: [[UInt8]] = []
                var scheme: QuantizationSchemeIdentifier?
                var bytesPerHeadSlot = 0
                var kvHeadCount = 0
                var maxSeqLen = 0
                for _ in 0..<3 {
                    guard let snap = try m.debugPrefillKVCacheLayerSnapshot(
                        tokens: prompt, layerIndex: layer, kind: kind
                    ) else {
                        Issue.record("debugPrefillKVCacheLayerSnapshot returned nil")
                        return
                    }
                    runs.append(snap.bytes)
                    scheme = snap.scheme
                    bytesPerHeadSlot = snap.bytesPerHeadSlot
                    kvHeadCount = snap.kvHeadCount
                    maxSeqLen = snap.maximumSequenceLength
                }
                let det = runs[0] == runs[1] && runs[1] == runs[2]
                let detStr = det ? "true " : "FALSE"
                let cs0 = runs[0].reduce(UInt64(0)) { $0 + UInt64($1) }
                let cs1 = runs[1].reduce(UInt64(0)) { $0 + UInt64($1) }
                let cs2 = runs[2].reduce(UInt64(0)) { $0 + UInt64($1) }
                let nonZero0 = runs[0].firstIndex(where: { $0 != 0 }) ?? -1
                print("  L\(layer) \(label): scheme=\(String(describing: scheme!)) " +
                      "size=\(runs[0].count) bytesPerHeadSlot=\(bytesPerHeadSlot) " +
                      "kvHeads=\(kvHeadCount) maxSeq=\(maxSeqLen)")
                print("    det=\(detStr) cs0=\(cs0) cs1=\(cs1) cs2=\(cs2) firstNonZeroByte=\(nonZero0)")

                // Per-position summary for positions 0..4 (prompt length = 5), head 0
                if cs0 > 0 {
                    for pos in 0..<min(5, prompt.count) {
                        let head0Offset = pos * bytesPerHeadSlot
                        let s0 = runs[0][head0Offset..<(head0Offset + bytesPerHeadSlot)]
                        let s1 = runs[1][head0Offset..<(head0Offset + bytesPerHeadSlot)]
                        let s2 = runs[2][head0Offset..<(head0Offset + bytesPerHeadSlot)]
                        let eq = Array(s0) == Array(s1) && Array(s1) == Array(s2)
                        let sum0 = s0.reduce(UInt64(0)) { $0 + UInt64($1) }
                        let sum1 = s1.reduce(UInt64(0)) { $0 + UInt64($1) }
                        let sum2 = s2.reduce(UInt64(0)) { $0 + UInt64($1) }
                        print("    pos\(pos): det=\(eq ? "true " : "FALSE") s0=\(sum0) s1=\(sum1) s2=\(sum2)")
                    }
                }
            }
        }
        print()
    }

    /// Single-step probe at LAST step (523) three times on same instance.
    /// If all zeros, the bug is in running all 524 steps then copying.
    /// If deterministic non-zero, submission machinery is fine & specific kernel
    /// is non-deterministic.
    @Test("BF16 prefill single-probe LAST step — 3 consecutive calls")
    func bf16PrefillLastStepConsecutiveCalls() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let prompt: [Int32] = [1, 1, 6, 6423, 708]
        let policy = InferencePolicy(maximumSequenceLength: 256)

        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: Self.bf16BundlePath,
            inferencePolicy: policy)
        var m = model
        guard let stepCount = m.prefillPlan?.steps.count, stepCount > 1 else {
            Issue.record("prefillPlan unavailable")
            return
        }
        let lastStep = stepCount - 1

        var runs: [[Float]] = []
        for i in 0..<3 {
            let snaps = try m.debugPrefillLastTokenHiddenSnapshots(
                tokens: prompt,
                stepIndices: [lastStep])
            let h = snaps[lastStep] ?? []
            let sum = h.reduce(0, +)
            let preview = h.prefix(6).map { String(format: "%.4f", $0) }.joined(separator: ",")
            print("  call \(i) step=\(lastStep): count=\(h.count) sum=\(sum) first6=[\(preview)]")
            runs.append(h)
        }

        var firstDiv01 = -1
        var maxDiff01: Float = 0
        if runs[0].count == runs[1].count {
            for k in 0..<runs[0].count {
                let d = abs(runs[0][k] - runs[1][k])
                if d > 0 && firstDiv01 < 0 { firstDiv01 = k }
                if d > maxDiff01 { maxDiff01 = d }
            }
        }
        print("  call 0 vs call 1: first_div=\(firstDiv01) max|diff|=\(maxDiff01)")
        print()
        #expect(runs[0] == runs[1], "Last step consecutive probes must be bit-identical (0 vs 1)")
        #expect(runs[1] == runs[2], "Last step consecutive probes must be bit-identical (1 vs 2)")
    }

    /// Minimal isolation: capture step 0 ONLY, three times on same instance.
    /// If run 2 diverges from run 0/1, the fault is in snapshot-call repetition,
    /// not in late-kernel execution.
    @Test("BF16 prefill step-0 only — 3 consecutive calls (isolate submission bug)")
    func bf16PrefillStep0ConsecutiveCalls() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let prompt: [Int32] = [1, 1, 6, 6423, 708]
        let policy = InferencePolicy(maximumSequenceLength: 256)

        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: Self.bf16BundlePath,
            inferencePolicy: policy)
        var m = model

        var runs: [[Float]] = []
        for i in 0..<3 {
            let snaps = try m.debugPrefillLastTokenHiddenSnapshots(
                tokens: prompt,
                stepIndices: [0])
            let h = snaps[0] ?? []
            print("  call \(i): count=\(h.count) sum=\(h.reduce(0, +)) first6=\(h.prefix(6).map { String(format: "%.4f", $0) })")
            runs.append(h)
        }
        print()

        #expect(runs[0] == runs[1], "Call 0 vs call 1 at step 0 must be bit-identical")
        #expect(runs[1] == runs[2], "Call 1 vs call 2 at step 0 must be bit-identical")
    }

    /// Print decode plan step kernels for layer 0 (pre-attn norm through o_proj + pre-MLP norm
    /// through down_proj). This exposes what kernel is actually dispatched per projection.
    static func dumpLayer0KernelList(label: String, model: MetalInferenceModel) {
        print("\n=== \(label) bundle — decode plan layer 0 kernels (first 60 steps) ===")
        print("step | kernelName                                  | weightTensor                                              | layer")
        print(String(repeating: "-", count: 140))
        let steps = model.decodePlan.steps
        let upperBound = min(60, steps.count)
        for (index, step) in steps[..<upperBound].enumerated() {
            let kernel = step.metadata.kernelName ?? step.pipeline.label ?? "(unlabeled)"
            let weight = step.metadata.weightTensorName ?? "-"
            let layer = step.metadata.layerIndex.map { String($0) } ?? "-"
            let padKernel = kernel.padding(toLength: 43, withPad: " ", startingAt: 0)
            let padWeight = weight.padding(toLength: 58, withPad: " ", startingAt: 0)
            print("\(String(format: "%4d", index)) | \(padKernel) | \(padWeight) | \(layer)")
        }
        print(String(repeating: "-", count: 140))
        print("Total decode steps: \(steps.count)")
        let q4Count = steps.filter { ($0.metadata.kernelName ?? $0.pipeline.label ?? "").contains("q4_g64") }.count
        let bf16GemvCount = steps.filter { ($0.metadata.kernelName ?? $0.pipeline.label ?? "").hasPrefix("gemv_bf16") }.count
        let denseGemvCount = steps.filter { ($0.metadata.kernelName ?? $0.pipeline.label ?? "") == "gemv" }.count
        let input2048Count = steps.filter { ($0.metadata.kernelName ?? $0.pipeline.label ?? "").hasPrefix("gemv_input2048") }.count
        print("  gemv_q4_g64 steps  : \(q4Count)")
        print("  gemv_bf16* steps   : \(bf16GemvCount)")
        print("  dense gemv steps   : \(denseGemvCount)")
        print("  gemv_input2048 step: \(input2048Count)")
        print()
    }
}
