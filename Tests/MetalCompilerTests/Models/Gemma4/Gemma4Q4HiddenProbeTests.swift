import Foundation
import Metal
import Testing
import LMArchitecture
import LMIR
import ModelDeclarations
@testable import MetalCompiler
@testable import SwiftLM

/// Per-layer hidden probe to locate where Q4 prefill diverges from BF16.
///
/// Token agreement test shows 0% match with Q4 producing identity-like output
/// (first token = last input token, then collapse to a single token). This
/// suggests all layers collapse to identity — residual path dominates because
/// attention/MLP branch output is zero or near-zero.
///
/// Probe strategy: capture last-token hidden at the end of each transformer
/// layer (after post-MLP residual add). Compare BF16 vs Q4 by cosine and L2.
@Suite("Gemma4 Q4 Hidden Probe", .serialized)
struct Gemma4Q4HiddenProbeTests {

    static let bf16BundlePath = "/Users/1amageek/Desktop/swift-lm/TestData/gemma-4-E2B-it"
    static let q4BundlePath = "/Users/1amageek/Desktop/swift-lm/TestData/gemma-4-E2B-it-4bit"

    @Test("Q4 vs BF16 per-layer hidden divergence (prompt_A)")
    func q4LayerDivergence() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        let policy = InferencePolicy(maximumSequenceLength: 256)

        // BF16: collect hidden at all steps in layer 0 + layer ends
        let (bf16Indices, bf16Hidden, bf16Kernels) = try captureLayerHidden(
            bundlePath: Self.bf16BundlePath,
            policy: policy,
            tokens: tokens,
            label: "BF16")

        // Q4: same capture on the Q4 bundle
        let (q4Indices, q4Hidden, q4Kernels) = try captureLayerHidden(
            bundlePath: Self.q4BundlePath,
            policy: policy,
            tokens: tokens,
            label: "Q4")

        print("\n=== Gemma4-E2B Per-Step Hidden Probe (prompt_A last token) ===")

        print("\nStep-by-step comparison (probing layer 0 + every layer end):")
        print("step | BF16 kernel                          | Q4 kernel                            | BF16 L2    | Q4 L2      | cosine")
        print(String(repeating: "-", count: 140))
        let count = min(bf16Indices.count, q4Indices.count, bf16Hidden.count, q4Hidden.count)
        var firstDiv = -1
        for i in 0..<count {
            let stepIdx = bf16Indices[i]
            let bfKernel = bf16Kernels[i]
            let q4Kernel = i < q4Kernels.count ? q4Kernels[i] : "?"
            let bf = bf16Hidden[i]
            let q4 = q4Hidden[i]
            let bfL2 = l2Norm(bf)
            let q4L2 = l2Norm(q4)
            let cos = cosine(bf, q4)
            let paddedBF = bfKernel.padding(toLength: 38, withPad: " ", startingAt: 0)
            let paddedQ4 = q4Kernel.padding(toLength: 38, withPad: " ", startingAt: 0)
            let numPart = String(format: "%10.4f | %10.4f | %.4f", bfL2, q4L2, cos)
            print("\(String(format: "%4d", stepIdx)) | \(paddedBF) | \(paddedQ4) | \(numPart)")
            if cos < 0.95 && firstDiv < 0 {
                firstDiv = stepIdx
            }
        }
        print(String(repeating: "-", count: 140))
        print("First step with cosine(BF16, Q4) < 0.95: step \(firstDiv)")

        print("\n=== Full kernel names (layer 0, untruncated) ===")
        for i in 0..<min(count, 18) {
            let stepIdx = bf16Indices[i]
            print("step \(stepIdx):")
            print("  BF16: \(bf16Kernels[i])")
            print("  Q4  : \(i < q4Kernels.count ? q4Kernels[i] : "?")")
        }

        #expect(count > 0)
    }

    private func captureLayerHidden(
        bundlePath: String,
        policy: InferencePolicy,
        tokens: [Int32],
        label: String
    ) throws -> (indices: [Int], hidden: [[Float]], kernels: [String]) {
        var result: (indices: [Int], hidden: [[Float]], kernels: [String]) = ([], [], [])
        try autoreleasepool {
            let (model, _, _) = try BenchmarkSupport.setupFromBundle(
                bundlePath: bundlePath,
                inferencePolicy: policy)
            var m = model
            let prefillPlan = try #require(m.prefillPlan, "prefillPlan missing")

            // Probe all steps in layer 0 + last step of each subsequent layer.
            var stepsToProbe: [Int] = []
            var layer0LastStep = -1
            var lastStepByLayer: [Int: Int] = [:]
            for (index, step) in prefillPlan.steps.enumerated() {
                if let layer = step.metadata.layerIndex {
                    lastStepByLayer[layer] = index
                    if layer == 0 {
                        layer0LastStep = index
                    }
                }
            }
            // Probe ALL steps 0..<=layer0LastStep (not just those with layerIndex metadata).
            // Layer 0 includes synthesized/fused kernels that may have no layerIndex.
            if layer0LastStep < 0 { layer0LastStep = 20 }  // fallback
            for s in 0...layer0LastStep {
                stepsToProbe.append(s)
            }
            // Add last step of each layer >= 1
            for (layer, stepIdx) in lastStepByLayer where layer >= 1 {
                stepsToProbe.append(stepIdx)
            }
            stepsToProbe = Array(Set(stepsToProbe)).sorted()

            let stepIndices = Set(stepsToProbe)
            let snapshots = try m.debugPrefillLastTokenHiddenSnapshots(
                tokens: tokens,
                stepIndices: stepIndices)

            var hiddenList: [[Float]] = []
            var kernelList: [String] = []
            var indexList: [Int] = []
            for stepIdx in stepsToProbe {
                guard let h = snapshots[stepIdx] else { continue }
                let step = prefillPlan.steps[stepIdx]
                let kernelName = step.metadata.kernelName ?? step.pipeline.label ?? "?"
                hiddenList.append(h)
                kernelList.append(kernelName)
                indexList.append(stepIdx)
            }

            print("[\(label)] total steps=\(prefillPlan.steps.count) layer0_last=\(layer0LastStep) probed=\(stepsToProbe.count)")
            result = (indexList, hiddenList, kernelList)
        }
        return result
    }

    /// Isolate bug location: transformer stack vs OutputHead (lm_head).
    ///
    /// Captures the last-token hidden AFTER final RMSNorm but BEFORE the lm_head
    /// projection + argmax. Compares BF16 vs Q4:
    ///   - If match (cosine ≥ 0.95) → bug is in OutputHead Q4 LinearFragment or argmax
    ///   - If diverge → bug is somewhere in transformer layers' late computation
    @Test("Q4 vs BF16 final prefill hidden (pre lm_head)")
    func q4FinalHiddenVsBFloat16() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let prompts: [(String, [Int32])] = [
            ("prompt_A", [1, 1, 6, 6423, 708]),
            ("prompt_B", [1, 2, 1681, 3, 4, 5]),
            ("prompt_C", [1, 1, 1, 1, 1, 1, 1, 1]),
        ]
        let policy = InferencePolicy(maximumSequenceLength: 256)

        print("\n=== Gemma4-E2B Final Prefill Hidden (pre lm_head) ===")
        print("prompt     | BF16 L2     | Q4 L2       | cosine(BF16,Q4) | top abs diff")
        print(String(repeating: "-", count: 90))

        for (name, tokens) in prompts {
            let bf16Hidden = try captureFinalHidden(bundle: Self.bf16BundlePath, policy: policy, tokens: tokens)
            let q4Hidden = try captureFinalHidden(bundle: Self.q4BundlePath, policy: policy, tokens: tokens)

            let bfL2 = l2Norm(bf16Hidden)
            let q4L2 = l2Norm(q4Hidden)
            let cos = cosine(bf16Hidden, q4Hidden)
            let n = min(bf16Hidden.count, q4Hidden.count)
            var topDiff: Float = 0
            for i in 0..<n {
                let d = abs(bf16Hidden[i] - q4Hidden[i])
                if d > topDiff { topDiff = d }
            }
            let padName = name.padding(toLength: 10, withPad: " ", startingAt: 0)
            print("\(padName) | \(String(format: "%10.4f", bfL2)) | \(String(format: "%10.4f", q4L2)) | \(String(format: "%14.4f", cos)) | \(String(format: "%.6f", topDiff))")

            // Print first 8 values of each for visual inspection
            let bfHead = bf16Hidden.prefix(8).map { String(format: "%.3f", $0) }.joined(separator: ",")
            let q4Head = q4Hidden.prefix(8).map { String(format: "%.3f", $0) }.joined(separator: ",")
            print("  BF16[0..8]: [\(bfHead)]")
            print("  Q4  [0..8]: [\(q4Head)]")
        }
        print()
    }

    private func captureFinalHidden(
        bundle: String,
        policy: InferencePolicy,
        tokens: [Int32]
    ) throws -> [Float] {
        var out: [Float] = []
        try autoreleasepool {
            let (model, _, _) = try BenchmarkSupport.setupFromBundle(
                bundlePath: bundle, inferencePolicy: policy)
            var m = model
            out = try m.debugPrefillLastTokenFinalHidden(tokens: tokens)
        }
        return out
    }

    /// Per-layer RESIDUAL buffer probe at each layer's final step.
    /// After synthesized_*way_reduction_residualadd, residual buffer holds the
    /// post-add hidden state. This bypasses the probe-runtime hidden buffer
    /// mismatch caused by fusion routing.
    @Test("Q4 vs BF16 per-layer residual probe (prompt_B — worst divergence)")
    func q4PerLayerResidualDivergence() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let tokens: [Int32] = [1, 2, 1681, 3, 4, 5]  // prompt_B: worst cosine (0.12)
        let policy = InferencePolicy(maximumSequenceLength: 256)

        // Locate last step of each layer by metadata.layerIndex
        let (bfSteps, bfResidual, bfKernels) = try captureLayerResidual(
            bundlePath: Self.bf16BundlePath, policy: policy, tokens: tokens)
        let (q4Steps, q4Residual, q4Kernels) = try captureLayerResidual(
            bundlePath: Self.q4BundlePath, policy: policy, tokens: tokens)

        print("\n=== Gemma4-E2B Per-Layer Residual Probe (prompt_B) ===")
        print("step | layer | kernel (BF16/Q4)                       | BF16 L2    | Q4 L2      | cosine")
        print(String(repeating: "-", count: 120))
        let count = min(bfSteps.count, q4Steps.count)
        var firstDiv = -1
        for i in 0..<count {
            let step = bfSteps[i].step
            let layer = bfSteps[i].layer
            let bfK = bfKernels[i]
            let q4K = q4Kernels[i]
            let bf = bfResidual[i]
            let q4 = q4Residual[i]
            let bfL = l2Norm(bf)
            let q4L = l2Norm(q4)
            let cos = cosine(bf, q4)
            let kpad = (bfK == q4K ? bfK : "BF16=\(bfK) Q4=\(q4K)").padding(toLength: 42, withPad: " ", startingAt: 0)
            print("\(String(format: "%4d", step)) | \(String(format: "%5d", layer)) | \(kpad) | \(String(format: "%10.4f", bfL)) | \(String(format: "%10.4f", q4L)) | \(String(format: "%.4f", cos))")
            if cos < 0.95 && firstDiv < 0 { firstDiv = layer }
        }
        print(String(repeating: "-", count: 120))
        print("First layer with cosine(BF16, Q4) < 0.95: layer \(firstDiv)")
        #expect(count > 0)
    }

    private func captureLayerResidual(
        bundlePath: String,
        policy: InferencePolicy,
        tokens: [Int32]
    ) throws -> (steps: [(step: Int, layer: Int)], residuals: [[Float]], kernels: [String]) {
        var result: (steps: [(step: Int, layer: Int)], residuals: [[Float]], kernels: [String]) = ([], [], [])
        try autoreleasepool {
            let (model, _, _) = try BenchmarkSupport.setupFromBundle(
                bundlePath: bundlePath, inferencePolicy: policy)
            var m = model
            let plan = try #require(m.prefillPlan, "prefillPlan missing")

            // Last step of each layer via metadata.layerIndex
            var lastStepOfLayer: [Int: Int] = [:]
            for (idx, step) in plan.steps.enumerated() {
                if let layer = step.metadata.layerIndex {
                    lastStepOfLayer[layer] = idx
                }
            }
            let sorted = lastStepOfLayer.sorted { $0.key < $1.key }
            let targetSteps = Set(sorted.map { $0.value })
            let snapshots = try m.debugPrefillLastTokenResidualSnapshots(
                tokens: tokens, stepIndices: targetSteps)
            var steps: [(step: Int, layer: Int)] = []
            var residuals: [[Float]] = []
            var kernels: [String] = []
            for (layer, stepIdx) in sorted {
                guard let v = snapshots[stepIdx] else { continue }
                steps.append((step: stepIdx, layer: layer))
                residuals.append(v)
                let st = plan.steps[stepIdx]
                kernels.append(st.metadata.kernelName ?? st.pipeline.label ?? "?")
            }
            result = (steps, residuals, kernels)
        }
        return result
    }

    /// Dump plan.steps structure for BF16 and Q4 to compare per-layer step layout.
    /// Diagnoses why BF16 residual probe reads 0 past L10 while Q4 reads valid until L15.
    @Test("Dump BF16 vs Q4 prefill plan.steps layout")
    func dumpPrefillPlanStepLayout() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let policy = InferencePolicy(maximumSequenceLength: 256)

        func dumpPlan(_ label: String, _ bundlePath: String) throws {
            try autoreleasepool {
                let (model, _, _) = try BenchmarkSupport.setupFromBundle(
                    bundlePath: bundlePath, inferencePolicy: policy)
                let plan = try #require(model.prefillPlan, "prefillPlan missing")
                print("\n=== [\(label)] plan.steps.count = \(plan.steps.count) ===")

                // Per-layer last-step index + kernel name (full)
                var lastStepOfLayer: [Int: Int] = [:]
                var firstStepOfLayer: [Int: Int] = [:]
                for (idx, step) in plan.steps.enumerated() {
                    if let layer = step.metadata.layerIndex {
                        lastStepOfLayer[layer] = idx
                        if firstStepOfLayer[layer] == nil { firstStepOfLayer[layer] = idx }
                    }
                }
                let sorted = lastStepOfLayer.sorted { $0.key < $1.key }
                print("layer | first | last | steps | last-step kernel (full)")
                print(String(repeating: "-", count: 100))
                for (layer, lastIdx) in sorted {
                    let firstIdx = firstStepOfLayer[layer] ?? -1
                    let stepCount = lastIdx - firstIdx + 1
                    let step = plan.steps[lastIdx]
                    let kname = step.metadata.kernelName ?? step.pipeline.label ?? "?"
                    print("\(String(format: "%5d", layer)) | \(String(format: "%5d", firstIdx)) | \(String(format: "%4d", lastIdx)) | \(String(format: "%5d", stepCount)) | \(kname)")
                }

                // Dump full unfiltered step list for L10-L16 transition zone
                print("\n--- Full step list [index | layerIdx | kernel name] for L10-L16 range ---")
                var dumpStart = -1
                var dumpEnd = -1
                if let l10 = firstStepOfLayer[10] { dumpStart = l10 }
                if let l16 = lastStepOfLayer[16] { dumpEnd = l16 }
                if dumpStart >= 0, dumpEnd >= 0 {
                    for idx in dumpStart...dumpEnd {
                        let step = plan.steps[idx]
                        let kname = step.metadata.kernelName ?? step.pipeline.label ?? "?"
                        let layer = step.metadata.layerIndex.map { String($0) } ?? "  -"
                        print("\(String(format: "%4d", idx)) | L\(layer) | \(kname)")
                    }
                }
            }
        }

        try dumpPlan("BF16", Self.bf16BundlePath)
        try dumpPlan("Q4", Self.q4BundlePath)
        #expect(true)
    }

    /// Dump Q4 prefill kernel sources for L0-L2 synthesized fusion kernels.
    /// Goal: inspect the MSL of `synthesized_3way_reduction_residualadd_*` and
    /// any fused kernel where Q4 produces anomalous hidden L2=1818.39 in L0.
    @Test("Dump Q4 prefill synthesized kernel sources for Gemma4-E2B")
    func q4DumpPrefillKernelSources() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        try autoreleasepool {
            let (device, store) = try BenchmarkSupport.loadBundleStoreOrSkip(
                bundlePath: Self.q4BundlePath)
            _ = device

            let configURL = URL(fileURLWithPath: Self.q4BundlePath)
                .appendingPathComponent("config.json")
            let configData = try Data(contentsOf: configURL)
            let decoder = HFConfigDecoder()
            let modelType = try decoder.modelType(from: configData)
            let config = try decoder.decode(from: configData)

            let resolver = ModelGraphResolver()
            let graph = try resolver.resolveModelGraph(modelType: modelType, config: config)
            let convention = resolver.namingConvention(for: modelType)
            let resolved = ParameterResolver().resolve(graph: graph, convention: convention)

            let compiler = MetalInferenceCompiler()
            let library = compiler.dumpGeneratedPrefillKernelLibrary(
                graph: resolved,
                hiddenSize: config.hiddenSize,
                stafWeightStore: store
            )

            print("\n=== Q4 Prefill Kernel Library (Gemma4-E2B) ===")
            print("Total characters: \(library.count)")

            let kernelNames: [String] = library.split(separator: "\n")
                .compactMap { line -> String? in
                    guard line.hasPrefix("kernel void ") else { return nil }
                    let afterPrefix = line.dropFirst("kernel void ".count)
                    let beforeParen = afterPrefix.prefix { $0 != "(" }
                    return String(beforeParen)
                }
            print("\nKernel names (\(kernelNames.count)):")
            for name in kernelNames.sorted() { print("  \(name)") }

            // Target kernels of interest
            let targets = [
                "synthesized_3way_reduction_residualadd",
                "synthesized_4way_reduction_residualadd",
                "synthesized_5way_reduction_residualadd",
                "synthesized_2way_copy_reduction",
            ]
            for target in targets {
                if let hit = kernelNames.first(where: { $0.hasPrefix(target) }) {
                    print("\n=== Source of \(hit) ===")
                    let start = library.range(of: "kernel void \(hit)")!
                    let suffix = library[start.lowerBound...]
                    // Naive end: find next `kernel void ` after one line
                    let endMarker = "\nkernel void "
                    let searchStart = suffix.index(start.lowerBound, offsetBy: "kernel void ".count)
                    let endRange = suffix.range(of: endMarker, range: searchStart..<suffix.endIndex)
                    let source = endRange.map { suffix[..<$0.lowerBound] } ?? suffix
                    print(source)
                } else {
                    print("\n[NOT FOUND] \(target)")
                }
            }
            #expect(!library.isEmpty)
        }
    }

    /// Step-by-step probe of ALL steps in L0-L3 for hidden + residual.
    /// Pinpoints the exact kernel within L2 at which Q4 first diverges from BF16.
    /// L0-L1 cos=0.997, L2 cos=0.984 — the bug manifests between L1_end and L2_end.
    /// This probe exposes EVERY step so we can see whether divergence starts in
    /// L2's attn branch (norm/QKV/flash_attn/o_proj) or MLP branch (norm/gate/up/down).
    @Test("Q4 vs BF16 step-by-step in L0-L3 (prompt_B)")
    func q4EarlyLayerStepByStep() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let tokens: [Int32] = [1, 2, 1681, 3, 4, 5]
        let policy = InferencePolicy(maximumSequenceLength: 256)

        let bf = try captureEarlyLayerDetail(
            bundlePath: Self.bf16BundlePath, policy: policy, tokens: tokens, label: "BF16")
        let q4 = try captureEarlyLayerDetail(
            bundlePath: Self.q4BundlePath, policy: policy, tokens: tokens, label: "Q4")

        print("\n=== Gemma4-E2B L0-L3 Step-by-Step Divergence (prompt_B) ===")
        print("step | layer | BF16 kernel                            | Q4 kernel                              | cos(hidden) | bfH L2    | q4H L2")
        print(String(repeating: "-", count: 175))

        let count = min(bf.stepIndices.count, q4.stepIndices.count)
        var firstHiddenDiv = -1
        var firstResidualDiv = -1
        for i in 0..<count {
            let step = bf.stepIndices[i]
            let layer = bf.layers[i] ?? -1
            let bfKernel = bf.kernels[i]
            let q4Kernel = i < q4.kernels.count ? q4.kernels[i] : "?"
            let bfH = bf.hiddens[i]
            let q4H = q4.hiddens[i]
            let bfR = bf.residuals[i]
            let q4R = q4.residuals[i]
            let cosH = cosine(bfH, q4H)
            let cosR = cosine(bfR, q4R)
            let diff = bfKernel == q4Kernel ? " " : "!"
            let bfPad = bfKernel.padding(toLength: 38, withPad: " ", startingAt: 0)
            let q4Pad = q4Kernel.padding(toLength: 38, withPad: " ", startingAt: 0)
            let line = "\(String(format: "%4d", step)) | \(String(format: "%5d", layer)) | \(bfPad) \(diff) \(q4Pad) | \(String(format: "%11.4f", cosH)) | \(String(format: "%9.4f", l2Norm(bfH))) | \(String(format: "%9.4f", l2Norm(q4H)))"
            print(line)
            _ = cosR; _ = bfR; _ = q4R
            if cosH < 0.99 && firstHiddenDiv < 0 { firstHiddenDiv = step }
            if cosR < 0.99 && firstResidualDiv < 0 { firstResidualDiv = step }
        }
        print(String(repeating: "-", count: 160))
        print("First step with cos(hidden) < 0.99:   step \(firstHiddenDiv)")
        print("First step with cos(residual) < 0.99: step \(firstResidualDiv)")
        #expect(count > 0)
    }

    private struct EarlyLayerDetail {
        var stepIndices: [Int] = []
        var layers: [Int?] = []
        var kernels: [String] = []
        var hiddens: [[Float]] = []
        var residuals: [[Float]] = []
    }

    private func captureEarlyLayerDetail(
        bundlePath: String,
        policy: InferencePolicy,
        tokens: [Int32],
        label: String
    ) throws -> EarlyLayerDetail {
        var detail = EarlyLayerDetail()
        try autoreleasepool {
            let (model, _, _) = try BenchmarkSupport.setupFromBundle(
                bundlePath: bundlePath, inferencePolicy: policy)
            var m = model
            let plan = try #require(m.prefillPlan, "prefillPlan missing")

            // Identify the first step belonging to L4 (exclusive upper bound).
            var upperBound = plan.steps.count
            for (idx, step) in plan.steps.enumerated() {
                if let layer = step.metadata.layerIndex, layer >= 4 {
                    upperBound = idx
                    break
                }
            }
            let targetSteps = Array(0..<upperBound)
            let stepIndices = Set(targetSteps)

            let hiddenSnapshots = try m.debugPrefillLastTokenHiddenSnapshots(
                tokens: tokens, stepIndices: stepIndices)
            let residualSnapshots = try m.debugPrefillLastTokenResidualSnapshots(
                tokens: tokens, stepIndices: stepIndices)

            for step in targetSteps {
                let planStep = plan.steps[step]
                let kernel = planStep.metadata.kernelName ?? planStep.pipeline.label ?? "?"
                let hidden = hiddenSnapshots[step] ?? []
                let residual = residualSnapshots[step] ?? []
                detail.stepIndices.append(step)
                detail.layers.append(planStep.metadata.layerIndex)
                detail.kernels.append(kernel)
                detail.hiddens.append(hidden)
                detail.residuals.append(residual)
            }
            print("[\(label)] probed steps 0..<\(upperBound) (L0-L3 boundary)")
        }
        return detail
    }

    private func l2Norm(_ v: [Float]) -> Float {
        var sum: Float = 0
        for x in v { sum += x * x }
        return sum.squareRoot()
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> Float {
        let n = min(a.count, b.count)
        guard n > 0 else { return Float.nan }
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<n {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        let denom = (na.squareRoot() * nb.squareRoot())
        return denom > 0 ? dot / denom : Float.nan
    }
}
