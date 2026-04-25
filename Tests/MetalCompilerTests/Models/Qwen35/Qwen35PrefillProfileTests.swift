import Foundation
import Metal
import Testing
@testable import MetalCompiler

/// Per-step GPU profiling for Qwen3.5-0.8B prefill.
///
/// Purpose: identify which prefill steps scale with sequence length. MLX keeps prefill
/// at a constant ~41-43 ms for sequence length 16→128, while swift-lm grows linearly
/// at ~2.2 ms/token. This test times each step individually to find the scaling culprit.
#if ENABLE_METAL_PROBES
@Suite("Qwen35 Prefill Profile", .serialized)
struct Qwen35PrefillProfileTests {

    static let sequenceLengths = [16, 64, 128]
    static let iterations = 5

    @Test("Per-step prefill timing at seqLen 16/64/128")
    func perStepPrefillTimingByLength() throws {
        guard let bundlePath = try resolveBundlePath() else {
            Issue.record("Qwen3.5-0.8B bundle not found.")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let (model, _, _) = try BenchmarkSupport.setupFromBundle(
            bundlePath: bundlePath,
            maximumPrefillLength: 128
        )
        guard let plan = model.prefillPlan else {
            Issue.record("No prefill plan")
            return
        }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        // Structural summary
        print("=== Qwen3.5-0.8B Prefill Plan ===")
        print("total steps: \(plan.steps.count)")
        printStepSummary(plan: plan)

        // Clone plan for isolated profiling
        let isolatedPlan = try plan.makeRuntimeIsolatedCopy(device: device)

        let residency = try MetalResidencyLease.combined(
            label: "qwen35.profile",
            leases: [
                MetalResidencyLease.required(
                    device: device,
                    label: "qwen35.profile.runtime",
                    buffers: isolatedPlan.buffers.runtimeResidencyBuffers
                ),
                MetalResidencyLease.required(
                    device: device,
                    label: "qwen35.profile.weights",
                    buffers: isolatedPlan.buffers.weightResidencyBuffers
                ),
                MetalResidencyLease.required(
                    device: device,
                    label: "qwen35.profile.supplemental",
                    buffers: isolatedPlan.supplementalResidencyBuffers
                ),
            ]
        )
        var submission = try MetalSubmissionContext(device: device)

        // Profile at each sequence length and print category breakdown
        var profilesByLength: [Int: [StepProfile]] = [:]
        for seqLen in Self.sequenceLengths {
            let profiles = try profileAll(
                plan: isolatedPlan,
                submission: &submission,
                sequenceLength: seqLen,
                iterations: Self.iterations,
                residency: residency
            )
            profilesByLength[seqLen] = profiles
            printCategoryBreakdown(profiles: profiles, iterations: Self.iterations, seqLen: seqLen)
        }

        // Scaling report: for each step, show time at 16 / 64 / 128 and ratio 128/16
        printScalingReport(profilesByLength: profilesByLength, iterations: Self.iterations)

        #expect(!profilesByLength.isEmpty)
    }

    // MARK: - Bundle resolution

    private func resolveBundlePath() throws -> String? {
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

    // MARK: - Step summary

    private func printStepSummary(plan: MetalPrefillPlan) {
        var kindCounts: [String: Int] = [:]
        for step in plan.steps {
            let name = step.metadata.kernelName ?? step.pipeline.label ?? "(unknown)"
            kindCounts[name, default: 0] += 1
        }
        let sorted = kindCounts.sorted { $0.value > $1.value }
        for (name, count) in sorted {
            print("  \(count)× \(name)")
        }
    }

    // MARK: - Profile types

    private struct StepProfile {
        let index: Int
        let kernelName: String
        let category: String
        let mode: PrefillStepMode
        let gridWidth: Int
        let gridHeight: Int
        let threadgroupWidth: Int
        var totalMicroseconds: Double
    }

    private func classify(_ kernelName: String) -> String {
        let name = kernelName.lowercased()
        if name.hasPrefix("embedding_lookup") || name.contains("gather") { return "embedding" }
        if name.hasPrefix("gemm_") || name.hasPrefix("gemv_") || name.contains("mpp") { return "projection" }
        if name.contains("ssm") || name.contains("delta") || name.contains("recurrence") { return "ssm_recurrence" }
        if name.hasPrefix("rms_norm") || name.contains("qk_rms_norm") || name.contains("_norm_") { return "reduction" }
        if name.hasPrefix("flash_attn") || name.hasPrefix("sdpa") { return "attention" }
        if name.hasPrefix("conv1d") || name.hasPrefix("conv_") { return "conv1d" }
        if name.contains("rope") { return "rope" }
        if name.contains("swiglu") || name.contains("silu") || name.contains("sigmoid") { return "elementwise" }
        if name.hasPrefix("copy_") || name.hasPrefix("add_") || name.hasPrefix("residual_") ||
           name.hasPrefix("fused_") || name.hasPrefix("kv_cache_") { return "structural" }
        return "other"
    }

    // MARK: - Profiling

    private func profileAll(
        plan: MetalPrefillPlan,
        submission: inout MetalSubmissionContext,
        sequenceLength: Int,
        iterations: Int,
        residency: MetalResidencyLease
    ) throws -> [StepProfile] {
        populateInputs(plan: plan, sequenceLength: sequenceLength)

        var profiles: [StepProfile] = plan.steps.enumerated().map { index, step in
            let kernelName = step.metadata.kernelName ?? step.pipeline.label ?? "(unlabeled)"
            let grid = step.resolvedGridSize(sequenceLength: sequenceLength)
            return StepProfile(
                index: index,
                kernelName: kernelName,
                category: classify(kernelName),
                mode: step.mode,
                gridWidth: grid.width,
                gridHeight: grid.height,
                threadgroupWidth: step.threadgroupSize.width,
                totalMicroseconds: 0
            )
        }

        let runtimeConstantBuffer = plan.buffers.runtimeConstantBuffer

        // Warmup (one full pass)
        for step in plan.steps {
            _ = try submission.withComputeTimed(ephemeralResidency: residency) { encoder, argumentTable in
                encodeSingleStep(
                    step,
                    encoder: encoder,
                    argumentTable: argumentTable,
                    runtimeConstantBuffer: runtimeConstantBuffer,
                    sequenceLength: sequenceLength
                )
            }
        }

        // Measured iterations
        for _ in 0..<iterations {
            for (index, step) in plan.steps.enumerated() {
                let timing = try submission.withComputeTimed(ephemeralResidency: residency) { encoder, argumentTable in
                    encodeSingleStep(
                        step,
                        encoder: encoder,
                        argumentTable: argumentTable,
                        runtimeConstantBuffer: runtimeConstantBuffer,
                        sequenceLength: sequenceLength
                    )
                }
                let microseconds = (timing.gpuEndTime - timing.gpuStartTime) * 1_000_000
                profiles[index].totalMicroseconds += microseconds
            }
        }

        return profiles
    }

    private func populateInputs(plan: MetalPrefillPlan, sequenceLength: Int) {
        let tokenPointer = plan.buffers.tokenIDs.contents()
            .bindMemory(to: Int32.self, capacity: sequenceLength)
        let positionPointer = plan.buffers.positions.contents()
            .bindMemory(to: UInt32.self, capacity: sequenceLength)
        let ropeAxesPointer = plan.buffers.ropePositionAxes.contents()
            .bindMemory(to: UInt32.self, capacity: sequenceLength * 3)
        for index in 0..<sequenceLength {
            tokenPointer[index] = Int32(index + 1)
            let position = UInt32(index)
            positionPointer[index] = position
            ropeAxesPointer[index * 3] = position
            ropeAxesPointer[index * 3 + 1] = position
            ropeAxesPointer[index * 3 + 2] = position
        }
        let constantPointer = plan.buffers.runtimeConstantBuffer.contents()
        constantPointer
            .advanced(by: PrefillBufferSet.sequenceLengthOffset)
            .bindMemory(to: UInt32.self, capacity: 1)
            .pointee = UInt32(sequenceLength)
        constantPointer
            .advanced(by: PrefillBufferSet.hiddenConversionCountOffset)
            .bindMemory(to: UInt32.self, capacity: 1)
            .pointee = 0
        for index in 0..<sequenceLength {
            constantPointer
                .advanced(by: PrefillBufferSet.positionOffset(at: index))
                .bindMemory(to: UInt32.self, capacity: 1)
                .pointee = UInt32(index)
        }
    }

    private func encodeSingleStep(
        _ step: MetalPrefillStep,
        encoder: MTL4ComputeCommandEncoder,
        argumentTable: MTL4ArgumentTable,
        runtimeConstantBuffer: MTLBuffer,
        sequenceLength: Int
    ) {
        switch step.mode {
        case .batch:
            step.bindings.bind(to: argumentTable)
            step.bindRuntimeArguments(
                argumentTable: argumentTable,
                runtimeConstantBuffer: runtimeConstantBuffer,
                sequenceLengthOffset: PrefillBufferSet.sequenceLengthOffset
            )
            let gridSize = step.resolvedGridSize(sequenceLength: sequenceLength)
            step.descriptor.encode(on: encoder, argumentTable: argumentTable, gridSize: gridSize)
        case .lastToken:
            let lastPosition = sequenceLength - 1
            step.bindStaticArguments(argumentTable: argumentTable, position: lastPosition)
            step.descriptor.encode(on: encoder, argumentTable: argumentTable)
        case .perPosition:
            for positionOffset in 0..<sequenceLength {
                step.bindStaticArguments(argumentTable: argumentTable, position: positionOffset)
                if let positionBufferIndex = step.positionBufferIndex {
                    argumentTable.setAddress(
                        runtimeConstantBuffer.gpuAddress
                            + UInt64(PrefillBufferSet.positionOffset(at: positionOffset)),
                        index: positionBufferIndex
                    )
                }
                step.descriptor.encode(on: encoder, argumentTable: argumentTable)
            }
        }
    }

    // MARK: - Reporting

    private func printCategoryBreakdown(profiles: [StepProfile], iterations: Int, seqLen: Int) {
        struct Entry { var steps: Int = 0; var totalMicros: Double = 0 }
        var byCategory: [String: Entry] = [:]
        var total: Double = 0
        for p in profiles {
            let avg = p.totalMicroseconds / Double(iterations)
            var e = byCategory[p.category] ?? Entry()
            e.steps += 1
            e.totalMicros += avg
            byCategory[p.category] = e
            total += avg
        }
        print("--- seqLen=\(seqLen) total=\(String(format: "%.3f", total / 1000.0))ms steps=\(profiles.count) ---")
        for (cat, entry) in byCategory.sorted(by: { $0.value.totalMicros > $1.value.totalMicros }) {
            let ms = entry.totalMicros / 1000.0
            let pct = total > 0 ? entry.totalMicros / total * 100.0 : 0
            let catPadded = cat.padding(toLength: 16, withPad: " ", startingAt: 0)
            let msStr = String(format: "%.3f", ms)
            let pctStr = String(format: "%.1f", pct)
            print("  \(catPadded) steps=\(entry.steps)  \(msStr) ms (\(pctStr)%)")
        }
    }

    private func printScalingReport(profilesByLength: [Int: [StepProfile]], iterations: Int) {
        print()
        print("=== Per-kernel scaling (seqLen 16 → 128) ===")
        // Aggregate by kernel name across all steps
        struct Row {
            let kernelName: String
            let category: String
            var times: [Int: Double] = [:]  // seqLen -> avg micros (sum across all steps)
            var counts: [Int: Int] = [:]    // seqLen -> step count
            var firstGrid: (Int, Int, Int)? = nil
        }
        var rows: [String: Row] = [:]
        for (seqLen, profiles) in profilesByLength {
            for p in profiles {
                var row = rows[p.kernelName] ?? Row(kernelName: p.kernelName, category: p.category)
                let avg = p.totalMicroseconds / Double(iterations)
                row.times[seqLen, default: 0] += avg
                row.counts[seqLen, default: 0] += 1
                if row.firstGrid == nil {
                    row.firstGrid = (p.gridWidth, p.gridHeight, p.threadgroupWidth)
                }
                rows[p.kernelName] = row
            }
        }
        let lengths = Self.sequenceLengths
        // Header
        var header = "  kernel                                 cat             n "
        for l in lengths { header += "  \(l)us    " }
        header += "  128/16  grid(w×h,tg)"
        print(header)
        let sorted = rows.values.sorted { ($0.times[128] ?? 0) > ($1.times[128] ?? 0) }
        for row in sorted {
            let count = row.counts[lengths[0]] ?? 0
            let knamePadded = row.kernelName.prefix(38).padding(toLength: 38, withPad: " ", startingAt: 0)
            let catPadded = row.category.padding(toLength: 14, withPad: " ", startingAt: 0)
            var line = "  \(knamePadded) \(catPadded) \(count) "
            for l in lengths {
                let us = row.times[l] ?? 0
                line += String(format: " %7.0f", us)
            }
            let t16 = row.times[16] ?? 0
            let t128 = row.times[128] ?? 0
            let ratio = t16 > 0 ? t128 / t16 : 0
            line += String(format: "  %6.2fx", ratio)
            if let g = row.firstGrid {
                line += "  (\(g.0)×\(g.1), tg=\(g.2))"
            }
            print(line)
        }
    }
}
#endif
