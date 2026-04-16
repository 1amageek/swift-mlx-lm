import Testing
import Metal
import Foundation
@testable import MetalCompiler
import LMArchitecture
import ModelDeclarations
import LMIR

/// Diagnostic tests that dump barrier patterns for analysis.
/// These are not correctness tests — they print detailed information about
/// which steps have barriers and why.
@Suite("Barrier Diagnostics")
struct BarrierDiagnosticTests {

    @Test("Dump per-step barrier analysis for 2-layer Transformer")
    func dumpBarrierAnalysis2Layer() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = ModelConfig(
            hiddenSize: 128, layerCount: 2, intermediateSize: 512,
            vocabSize: 1000, attentionHeads: 4, kvHeads: 4, headDim: 32,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 10000, ropeDimension: 32,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: false,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: nil,
            partialRotaryFactor: nil, slidingWindow: nil
        )
        let graph = try ModelGraph(Transformer(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .llamaFamily)
        let compiled = try MetalInferenceCompiler().compile(
            graph: resolved, hiddenSize: 128, intermediateSize: 512,
            vocabSize: 1000, device: device)

        analyzeBarrierPattern(compiled: compiled, label: "2-layer Transformer (h=128)")
    }

    @Test("Dump per-step barrier analysis for 4-layer Transformer")
    func dumpBarrierAnalysis4Layer() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = ModelConfig(
            hiddenSize: 256, layerCount: 4, intermediateSize: 1024,
            vocabSize: 2000, attentionHeads: 8, kvHeads: 8, headDim: 32,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 10000, ropeDimension: 32,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: false,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: nil,
            partialRotaryFactor: nil, slidingWindow: nil
        )
        let graph = try ModelGraph(Transformer(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .llamaFamily)
        let compiled = try MetalInferenceCompiler().compile(
            graph: resolved, hiddenSize: 256, intermediateSize: 1024,
            vocabSize: 2000, device: device)

        analyzeBarrierPattern(compiled: compiled, label: "4-layer Transformer (h=256)")
    }

    // MARK: - Analysis

    private func analyzeBarrierPattern(compiled: MetalCompiledModel, label: String) {
        let steps = compiled.steps
        let totalSteps = steps.count
        let barrierSteps = steps.filter { $0.barrierPolicy.isBarrier }.count
        let elided = totalSteps - barrierSteps

        let deviceBarrierSteps = steps.filter {
            if case .barrier(let vis) = $0.barrierPolicy { return vis == .device }
            return false
        }.count
        let noneVisibilityBarrierSteps = barrierSteps - deviceBarrierSteps

        print("\n========== BARRIER ANALYSIS: \(label) ==========")
        print("Total steps: \(totalSteps), Barriers: \(barrierSteps), Elided: \(elided)")
        print("Barrier rate: \(String(format: "%.1f%%", Double(barrierSteps) / Double(totalSteps) * 100))")
        print("Device visibility: \(deviceBarrierSteps), None visibility: \(noneVisibilityBarrierSteps)")
        print("Fusion: \(compiled.unfusedEntryCount) unfused → \(compiled.fusedEntryCount) fused")
        print("")

        // Per-step detail
        var pendingWrites = Set<BufferRegion>()
        var consecutiveBarrierRuns: [Int] = []
        var currentRun = 0

        for (index, step) in steps.enumerated() {
            let kernel = step.metadata.kernelName ?? "unknown"
            let layer = step.metadata.layerIndex.map { "L\($0)" } ?? "--"
            let barrier = step.barrierPolicy.isBarrier ? "BARRIER" : "  skip"
            let reads = step.bufferAccesses.reads.count
            let writes = step.bufferAccesses.writes.count
            let isConservative = step.bufferAccesses.reads == step.bufferAccesses.writes && reads > 1

            // Trace the conflict
            let conflictingRegions = pendingWrites.intersection(
                step.bufferAccesses.reads.union(step.bufferAccesses.writes)
            )

            var conflictInfo = ""
            if !conflictingRegions.isEmpty {
                conflictInfo = " ← conflict with \(conflictingRegions.count) region(s)"
            }

            let conservativeTag = isConservative ? " [CONSERVATIVE]" : ""

            print("  [\(String(format: "%2d", index))] \(barrier) \(layer) \(kernel.padding(toLength: 40, withPad: " ", startingAt: 0)) R=\(reads) W=\(writes)\(conservativeTag)\(conflictInfo)")

            if step.barrierPolicy.isBarrier {
                currentRun += 1
                pendingWrites = step.bufferAccesses.writes
            } else {
                if currentRun > 0 {
                    consecutiveBarrierRuns.append(currentRun)
                    currentRun = 0
                }
                pendingWrites.formUnion(step.bufferAccesses.writes)
            }
        }
        if currentRun > 0 { consecutiveBarrierRuns.append(currentRun) }

        // Summary
        print("")
        print("--- SUMMARY ---")

        // Count by kernel type
        var barriersByKernel: [String: (barriers: Int, total: Int)] = [:]
        for step in steps {
            let kernel = step.metadata.kernelName ?? "unknown"
            let prefix = kernelCategory(kernel)
            var entry = barriersByKernel[prefix, default: (0, 0)]
            entry.total += 1
            if step.barrierPolicy.isBarrier { entry.barriers += 1 }
            barriersByKernel[prefix] = entry
        }

        print("Barriers by kernel category:")
        for (category, stats) in barriersByKernel.sorted(by: { $0.value.total > $1.value.total }) {
            let rate = Double(stats.barriers) / Double(stats.total) * 100
            print("  \(category.padding(toLength: 25, withPad: " ", startingAt: 0)) \(stats.barriers)/\(stats.total) (\(String(format: "%.0f%%", rate)))")
        }

        // Conservative step analysis
        let conservativeSteps = steps.filter {
            $0.bufferAccesses.reads == $0.bufferAccesses.writes && $0.bufferAccesses.reads.count > 1
        }
        print("\nConservative (no writeBufferIndices): \(conservativeSteps.count)/\(totalSteps)")
        for step in conservativeSteps {
            print("  - \(step.metadata.kernelName ?? "unknown")")
        }

        // Genuine vs eliminable barriers
        // A barrier is "genuine" if it would still be needed even with perfect write tracking
        // It's "eliminable" if it's caused by conservative tracking or false dependencies
        print("")
    }

    private func kernelCategory(_ name: String) -> String {
        if name.hasPrefix("gemv") { return "projection (GEMV)" }
        if name.hasPrefix("gemm") { return "projection (GEMM)" }
        if name.contains("fused_residual") { return "fused_residual_add_norm" }
        if name.contains("fused_copy") { return "fused_copy_norm" }
        if name.contains("swiglu") || name.contains("geglu") { return "fused_activation_proj" }
        if name.contains("flash_attn") { return "flash_attention" }
        if name.contains("rope") { return "rope" }
        if name.contains("rms_norm") || name.contains("layer_norm") { return "norm" }
        if name.contains("copy") { return "structural_copy" }
        if name.contains("add") { return "structural_add" }
        if name.contains("embed") || name.contains("gather") { return "embedding" }
        if name.contains("argmax") { return "argmax" }
        return name
    }
}
