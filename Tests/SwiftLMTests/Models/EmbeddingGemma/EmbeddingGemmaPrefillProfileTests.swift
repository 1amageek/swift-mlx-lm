import Foundation
import Metal
import Testing
@testable import MetalCompiler
@testable import SwiftLM

@Suite("EmbeddingGemma Prefill Profile", .serialized)
struct EmbeddingGemmaPrefillProfileTests {

    @Test("Q4 vs BF16 prefill step breakdown", .timeLimit(.minutes(10)))
    func q4VsBF16PrefillStepBreakdown() async throws {
        guard let q4Container = try await EmbeddingGemmaTestSupport.realEmbeddingGemmaContainer(
            variant: .community4Bit
        ) else {
            print("[Skip] No Q4 EmbeddingGemma snapshot found")
            return
        }
        guard let bf16Container = try await EmbeddingGemmaTestSupport.realEmbeddingGemmaContainer(
            variant: .communityBF16
        ) else {
            print("[Skip] No BF16 EmbeddingGemma snapshot found")
            return
        }

        let sequenceLength = 32
        let iterations = 5

        // Print plan metadata for cross-validation
        printPlanMetadata(q4Container, label: "Q4")
        printPlanMetadata(bf16Container, label: "BF16")

        let (q4Profiles, _) = try profileContainerWithProfiles(
            q4Container,
            sequenceLength: sequenceLength,
            iterations: iterations
        )
        let (bf16Profiles, _) = try profileContainerWithProfiles(
            bf16Container,
            sequenceLength: sequenceLength,
            iterations: iterations
        )

        let q4Breakdown = PrefillProfilingSupport.aggregateByCategory(
            profiles: q4Profiles, iterations: iterations
        )
        let bf16Breakdown = PrefillProfilingSupport.aggregateByCategory(
            profiles: bf16Profiles, iterations: iterations
        )

        print(PrefillProfilingSupport.formatBreakdown(q4Breakdown, label: "Q4"))
        print(PrefillProfilingSupport.formatBreakdown(bf16Breakdown, label: "BF16"))

        // Print first projection step grid/tg for MPP vs naive validation
        printFirstProjectionStepDetails(q4Profiles, label: "Q4")
        printFirstProjectionStepDetails(bf16Profiles, label: "BF16")

        // Cross-validate: end-to-end wall-clock vs profiled total
        try crossValidateWithEndToEnd(
            container: q4Container,
            profilingTotalMicroseconds: q4Breakdown.totalMicroseconds,
            label: "Q4",
            sequenceLength: sequenceLength
        )
        try crossValidateWithEndToEnd(
            container: bf16Container,
            profilingTotalMicroseconds: bf16Breakdown.totalMicroseconds,
            label: "BF16",
            sequenceLength: sequenceLength
        )

        let q4Projection = q4Breakdown.entries[.projection]?.totalMicroseconds ?? 0
        let bf16Projection = bf16Breakdown.entries[.projection]?.totalMicroseconds ?? 0

        if bf16Projection > 0 {
            let ratio = q4Projection / bf16Projection
            let q4ProjectionMs = String(format: "%.3f", q4Projection / 1000.0)
            let bf16ProjectionMs = String(format: "%.3f", bf16Projection / 1000.0)
            print(
                "[PrefillProfile.Comparison] projection: "
                    + "Q4=\(q4ProjectionMs)ms "
                    + "BF16=\(bf16ProjectionMs)ms "
                    + "ratio=\(String(format: "%.2f", ratio))x"
            )
        }

        // Validate non-projection categories are similar across variants
        let q4Attention = q4Breakdown.entries[.attention]?.totalMicroseconds ?? 0
        let bf16Attention = bf16Breakdown.entries[.attention]?.totalMicroseconds ?? 0
        if q4Attention > 0, bf16Attention > 0 {
            let attentionRatio = q4Attention / bf16Attention
            print(
                "[PrefillProfile.Validation] attention ratio=\(String(format: "%.2f", attentionRatio))x "
                    + "(expected ~1.0 for weight-independent steps)"
            )
            #expect(attentionRatio > 0.7 && attentionRatio < 1.5,
                    "Attention time should be similar across Q4/BF16 variants")
        }

        #expect(q4Breakdown.totalMicroseconds > 0)
        #expect(bf16Breakdown.totalMicroseconds > 0)
    }

    @Test("Q4 prefill projection micro-benchmark", .timeLimit(.minutes(10)))
    func q4PrefillProjectionMicroBenchmark() async throws {
        guard let container = try await EmbeddingGemmaTestSupport.realEmbeddingGemmaContainer(
            variant: .community4Bit
        ) else {
            print("[Skip] No Q4 EmbeddingGemma snapshot found")
            return
        }

        let sequenceLength = 32
        let iterations = 20

        let (profiles, _) = try profileContainerWithProfiles(
            container,
            sequenceLength: sequenceLength,
            iterations: iterations
        )

        let projectionProfiles = profiles.filter { $0.category == .projection }
        guard !projectionProfiles.isEmpty else {
            print("[PrefillProfile.Micro] No projection steps found")
            return
        }

        let totalProjectionMicroseconds = projectionProfiles.reduce(0.0) {
            $0 + $1.totalMicroseconds / Double(iterations)
        }

        print(
            "[PrefillProfile.Micro] projectionSteps=\(projectionProfiles.count) "
                + "totalAvg=\(String(format: "%.3f", totalProjectionMicroseconds / 1000.0))ms "
                + "seqLen=\(sequenceLength) "
                + "iterations=\(iterations)"
        )

        for profile in projectionProfiles {
            let averageMicroseconds = profile.totalMicroseconds / Double(iterations)
            let grid = "(\(profile.gridSize.width)x\(profile.gridSize.height)x\(profile.gridSize.depth))"
            let tg = "(\(profile.threadgroupSize.width)x\(profile.threadgroupSize.height)x\(profile.threadgroupSize.depth))"
            print(
                "[PrefillProfile.Micro] step=\(profile.index) "
                    + "kernel=\(profile.kernelName) "
                    + "avg=\(String(format: "%.3f", averageMicroseconds))us "
                    + "grid=\(grid) tg=\(tg)"
            )
        }

        #expect(totalProjectionMicroseconds > 0)
    }

    // MARK: - Validation Helpers

    /// Print plan metadata to validate MPP state and kernel families.
    private func printPlanMetadata(
        _ container: TextEmbeddingContainer,
        label: String
    ) {
        let plan = container.prefillPlan
        let projectionFamilies = plan.quantizationKernelFamilies(path: "prefillProjection")
        let lookupFamilies = plan.quantizationKernelFamilies(path: "embeddingLookup")
        print(
            "[PrefillProfile.\(label).Meta] usesMPP=\(plan.usesMPP) "
                + "steps=\(plan.steps.count) "
                + "projectionFamilies=[\(projectionFamilies.joined(separator: ","))] "
                + "lookupFamilies=[\(lookupFamilies.joined(separator: ","))]"
        )
    }

    /// Print grid/tg of first projection step to distinguish MPP from naive.
    ///
    /// MPP grid: ((outputDim+31)/32 × (seqLen+63)/64), tg: (simdWidth*4 × 1 × 1)
    /// Q4/Naive grid: (outputDim/2 × seqLen), tg: (64 × 1 × 1)
    private func printFirstProjectionStepDetails(
        _ profiles: [PrefillStepProfile],
        label: String
    ) {
        guard let first = profiles.first(where: { $0.category == .projection }) else { return }
        let grid = "(\(first.gridSize.width)x\(first.gridSize.height)x\(first.gridSize.depth))"
        let tg = "(\(first.threadgroupSize.width)x\(first.threadgroupSize.height)x\(first.threadgroupSize.depth))"
        let isMPPLikely = first.threadgroupSize.width == 128
        print(
            "[PrefillProfile.\(label).Projection] firstStep=\(first.index) "
                + "kernel=\(first.kernelName) "
                + "grid=\(grid) tg=\(tg) "
                + "dispatchPattern=\(isMPPLikely ? "MPP" : "SIMD")"
        )
    }

    /// Cross-validate profiling total against a single end-to-end wall-clock prefill.
    ///
    /// Per-step profiling dispatches each step in a separate command buffer,
    /// which adds submission overhead. The profiled total should be greater than
    /// or comparable to the wall-clock time of a single full prefill.
    private func crossValidateWithEndToEnd(
        container: TextEmbeddingContainer,
        profilingTotalMicroseconds: Double,
        label: String,
        sequenceLength: Int
    ) throws {
        let context = try TextEmbeddingContext(container)
        let dummyTokens = (0..<sequenceLength).map { _ in "test" }.joined(separator: " ")
        let clock = ContinuousClock()
        // Warmup
        _ = try context.embed(dummyTokens)
        // Measured
        let duration = try clock.measure {
            _ = try context.embed(dummyTokens)
        }
        let wallClockMicroseconds =
            Double(duration.components.seconds) * 1_000_000
            + Double(duration.components.attoseconds) / 1_000_000_000_000
        let profilingMs = String(format: "%.3f", profilingTotalMicroseconds / 1000.0)
        let wallClockMs = String(format: "%.3f", wallClockMicroseconds / 1000.0)
        let overheadRatio = profilingTotalMicroseconds / max(wallClockMicroseconds, 1)
        print(
            "[PrefillProfile.\(label).CrossValidation] "
                + "profiled=\(profilingMs)ms "
                + "wallClock=\(wallClockMs)ms "
                + "overheadRatio=\(String(format: "%.2f", overheadRatio))x"
        )
    }

    // MARK: - Profiling Infrastructure

    private func profileContainerWithProfiles(
        _ container: TextEmbeddingContainer,
        sequenceLength: Int,
        iterations: Int
    ) throws -> ([PrefillStepProfile], MetalPrefillPlan) {
        let isolatedPlan = try container.prefillPlan.makeRuntimeIsolatedCopy(
            device: container.device
        )
        let residency = try MetalResidencyLease.combined(
            label: "swift-lm.embeddinggemma.profile",
            leases: [
                MetalResidencyLease.required(
                    device: container.device,
                    label: "swift-lm.embeddinggemma.profile.runtime",
                    buffers: isolatedPlan.buffers.runtimeResidencyBuffers
                ),
                MetalResidencyLease.required(
                    device: container.device,
                    label: "swift-lm.embeddinggemma.profile.weights",
                    buffers: isolatedPlan.buffers.weightResidencyBuffers
                ),
                MetalResidencyLease.required(
                    device: container.device,
                    label: "swift-lm.embeddinggemma.profile.supplemental",
                    buffers: isolatedPlan.supplementalResidencyBuffers
                ),
            ]
        )
        var submission = try MetalSubmissionContext(device: container.device)

        let profiles = try PrefillProfilingSupport.profilePrefillSteps(
            plan: isolatedPlan,
            submission: &submission,
            sequenceLength: sequenceLength,
            iterations: iterations,
            ephemeralResidency: residency
        )

        return (profiles, isolatedPlan)
    }
}
