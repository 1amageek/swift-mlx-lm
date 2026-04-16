import Metal
import Testing
@testable import MetalCompiler
@testable import SwiftLM

/// Isolate the exact prefill step that causes GPU hang by running steps
/// one at a time in individual command buffers.
@Suite("Embedding Prefill Step Isolation", .serialized)
struct EmbeddingPrefillStepIsolationTests {

    /// Step 0 alone — embedding lookup only.
    @Test("Step 0: embedding lookup", .timeLimit(.minutes(1)))
    func step0EmbeddingLookup() async throws {
        let (plan, residency) = try await loadPlan()
        guard let plan, let residency else { return }
        try runSteps(plan: plan, residency: residency, through: 0)
    }

    /// Steps 0-1 — embedding + synthesized copy+norm.
    @Test("Steps 0-1: embedding + norm", .timeLimit(.minutes(1)))
    func steps0to1() async throws {
        let (plan, residency) = try await loadPlan()
        guard let plan, let residency else { return }
        try runSteps(plan: plan, residency: residency, through: 1)
    }

    /// Steps 0-2 — through first batched Q/K/V projection.
    @Test("Steps 0-2: through Q/K/V projection", .timeLimit(.minutes(1)))
    func steps0to2() async throws {
        let (plan, residency) = try await loadPlan()
        guard let plan, let residency else { return }
        try runSteps(plan: plan, residency: residency, through: 2)
    }

    /// Steps 0-5 — through flash attention (first layer).
    @Test("Steps 0-5: through flash attention", .timeLimit(.minutes(1)))
    func steps0to5() async throws {
        let (plan, residency) = try await loadPlan()
        guard let plan, let residency else { return }
        try runSteps(plan: plan, residency: residency, through: 5)
    }

    /// Steps 0-6 — through o_proj GEMV.
    @Test("Steps 0-6: through o_proj", .timeLimit(.minutes(1)))
    func steps0to6() async throws {
        let (plan, residency) = try await loadPlan()
        guard let plan, let residency else { return }
        try runSteps(plan: plan, residency: residency, through: 6)
    }

    /// Steps 0-7 — through residual add + copy + norm (synthesized).
    @Test("Steps 0-7: through synthesized residual", .timeLimit(.minutes(1)))
    func steps0to7() async throws {
        let (plan, residency) = try await loadPlan()
        guard let plan, let residency else { return }
        try runSteps(plan: plan, residency: residency, through: 7)
    }

    /// Steps 0-8 — through gate+up batched projection.
    @Test("Steps 0-8: through gate+up projection", .timeLimit(.minutes(1)))
    func steps0to8() async throws {
        let (plan, residency) = try await loadPlan()
        guard let plan, let residency else { return }
        try runSteps(plan: plan, residency: residency, through: 8)
    }

    /// Steps 0-9 — through GeGLU activation.
    @Test("Steps 0-9: through GeGLU", .timeLimit(.minutes(1)))
    func steps0to9() async throws {
        let (plan, residency) = try await loadPlan()
        guard let plan, let residency else { return }
        try runSteps(plan: plan, residency: residency, through: 9)
    }

    /// Steps 0-10 — through down_proj (first full layer).
    @Test("Steps 0-10: first full layer", .timeLimit(.minutes(1)))
    func steps0to10() async throws {
        let (plan, residency) = try await loadPlan()
        guard let plan, let residency else { return }
        try runSteps(plan: plan, residency: residency, through: 10)
    }

    /// Diagnostic: print step metadata for first 11 steps.
    @Test("Step metadata dump", .timeLimit(.minutes(1)))
    func stepMetadataDump() async throws {
        let (plan, _) = try await loadPlan()
        guard let plan else { return }

        for idx in 0..<min(11, plan.steps.count) {
            let step = plan.steps[idx]
            let kernel = step.metadata.kernelName ?? step.pipeline.label ?? "(unknown)"
            let grid = step.gridSize
            let tg = step.threadgroupSize
            let tgMem = step.threadgroupMemoryLength
            let mode = step.mode
            let bufs = step.bufferBindings.map { "[\($0.index)]:\(Unmanaged.passUnretained($0.buffer).toOpaque())+\($0.offset)" }
            let bytes = step.bytesBindings.map { "[\($0.index)]:\($0.value.count)B" }
            let pipelineLabel = step.pipeline.label ?? "(no label)"
            let maxTPT = step.pipeline.maxTotalThreadsPerThreadgroup
            print("[Dump] step=\(idx) kernel=\(kernel) pipeline=\(pipelineLabel)")
            print("  grid=\(grid.width)x\(grid.height)x\(grid.depth) tg=\(tg.width)x\(tg.height)x\(tg.depth) tgMem=\(tgMem) maxTPT=\(maxTPT)")
            print("  mode=\(mode) bufs=\(bufs.joined(separator: " "))")
            print("  bytes=\(bytes.joined(separator: " "))")
        }
    }

    // MARK: - Helpers

    private func loadPlan() async throws -> (MetalPrefillPlan?, MetalResidencyLease?) {
        guard let container = try await EmbeddingGemmaTestSupport.realEmbeddingGemmaContainer(
            variant: .community4Bit
        ) else {
            print("[StepIsolation.Skip] No Q4 EmbeddingGemma model found")
            return (nil, nil)
        }

        let plan = try container.prefillPlan.makeRuntimeIsolatedCopy(device: container.device)
        let residency = try MetalResidencyLease.combined(
            label: "step-isolation",
            leases: [
                MetalResidencyLease.required(
                    device: container.device,
                    label: "runtime",
                    buffers: plan.buffers.runtimeResidencyBuffers
                ),
                MetalResidencyLease.required(
                    device: container.device,
                    label: "weights",
                    buffers: plan.buffers.weightResidencyBuffers
                ),
                MetalResidencyLease.required(
                    device: container.device,
                    label: "supplemental",
                    buffers: plan.supplementalResidencyBuffers
                ),
            ]
        )
        return (plan, residency)
    }

    private func runSteps(
        plan: MetalPrefillPlan,
        residency: MetalResidencyLease,
        through lastStep: Int
    ) throws {
        let tokenIDs: [Int32] = [2]  // Single token
        let stepIndices = Set(0...min(lastStep, plan.steps.count - 1))

        print("[StepIsolation] Running steps 0...\(lastStep) with \(tokenIDs.count) token(s)")
        for idx in stepIndices.sorted() {
            let kernel = plan.steps[idx].metadata.kernelName ?? "(unknown)"
            print("[StepIsolation]   step=\(idx) kernel=\(kernel)")
        }

        var submission = try MetalSubmissionContext(device: plan.buffers.hidden.device)

        let snapshots = try MetalPrefillExecutor().captureLastTokenHiddenSnapshots(
            prefillPlan: plan,
            submission: &submission,
            position: 0,
            tokens: tokenIDs,
            stepIndices: stepIndices,
            ephemeralResidency: residency
        )

        for idx in stepIndices.sorted() {
            if let values = snapshots[idx] {
                let norm = sqrt(values.reduce(0) { $0 + $1 * $1 })
                let finite = values.allSatisfy { $0.isFinite }
                print("[StepIsolation] step=\(idx) norm=\(String(format: "%.4f", norm)) finite=\(finite)")
            } else {
                print("[StepIsolation] step=\(idx) snapshot=nil")
            }
        }
    }
}
