import Foundation
import Metal
@testable import MetalCompiler

// MARK: - Step Category

/// High-level classification for prefill step profiling.
enum PrefillStepCategory: String, CaseIterable, Sendable {
    case embeddingLookup
    case projection
    case reduction
    case attention
    case elementwise
    case structural
    case other

    static func classify(kernelName: String) -> PrefillStepCategory {
        let name = kernelName.lowercased()
        if name.hasPrefix("embedding_lookup") {
            return .embeddingLookup
        }
        if name.hasPrefix("gemm_") || name.hasPrefix("gemv_") || name.contains("mpp") {
            return .projection
        }
        if name.hasPrefix("rms_norm") || name.hasPrefix("layer_norm")
            || name.contains("_norm_") || name.hasPrefix("qk_rms_norm")
        {
            return .reduction
        }
        if name.hasPrefix("flash_attn") || name.hasPrefix("sdpa") || name.contains("attention") {
            return .attention
        }
        if name.hasPrefix("swiglu") || name.hasPrefix("silu") || name.hasPrefix("gelu")
            || name.hasPrefix("sigmoid") || name.hasPrefix("geglu")
        {
            return .elementwise
        }
        if name.hasPrefix("copy_") || name.hasPrefix("add_") || name.hasPrefix("fused_")
            || name.hasPrefix("residual_") || name.hasPrefix("kv_cache_")
            || name.hasPrefix("rope")
        {
            return .structural
        }
        return .other
    }
}

// MARK: - Profile Types

struct PrefillStepProfile: Sendable {
    let index: Int
    let kernelName: String
    let category: PrefillStepCategory
    let mode: PrefillStepMode
    let gridSize: MTLSize
    let threadgroupSize: MTLSize
    var totalMicroseconds: Double = 0
}

struct PrefillCategoryBreakdown: Sendable {
    struct Entry: Sendable {
        var stepCount: Int = 0
        var totalMicroseconds: Double = 0
        var kernelNames: [String: Int] = [:]
    }

    var entries: [PrefillStepCategory: Entry] = [:]
    var totalMicroseconds: Double = 0
}

// MARK: - Profiling Support

enum PrefillProfilingSupport {

    // MARK: Buffer Population

    /// Write dummy token IDs, positions, and runtime constants for profiling.
    ///
    /// This reimplements `MetalPrefillExecutor.populatePrefillInputs` and
    /// `writeRuntimeConstants` using only public buffer APIs from `PrefillBufferSet`.
    /// The values are synthetic — profiling measures kernel execution time,
    /// not output correctness.
    static func populatePrefillInputsForProfiling(
        plan: MetalPrefillPlan,
        sequenceLength: Int
    ) {
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
        // Sequence length at offset 0
        constantPointer
            .advanced(by: PrefillBufferSet.sequenceLengthOffset)
            .bindMemory(to: UInt32.self, capacity: 1)
            .pointee = UInt32(sequenceLength)
        // Hidden conversion count at offset 4
        constantPointer
            .advanced(by: PrefillBufferSet.hiddenConversionCountOffset)
            .bindMemory(to: UInt32.self, capacity: 1)
            .pointee = 0
        // Per-position absolute positions
        for index in 0..<sequenceLength {
            constantPointer
                .advanced(by: PrefillBufferSet.positionOffset(at: index))
                .bindMemory(to: UInt32.self, capacity: 1)
                .pointee = UInt32(index)
        }
    }

    // MARK: Single Step Encoding

    /// Encode a single prefill step into the given encoder.
    ///
    /// Mirrors the per-mode dispatch logic in `MetalPrefillExecutor.encodePrefillSteps`
    /// using public APIs on `MetalPrefillStep`.
    static func encodeSinglePrefillStep(
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
            step.descriptor.encode(
                on: encoder,
                argumentTable: argumentTable,
                gridSize: gridSize
            )
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

    // MARK: Step Profiling

    /// Profile every prefill step individually using GPU timestamps.
    ///
    /// Each step is dispatched in its own command buffer via `withComputeTimed`
    /// to capture per-step GPU execution time. A warmup pass is run first
    /// to stabilize GPU state.
    ///
    /// - Parameters:
    ///   - plan: The prefill plan containing steps to profile.
    ///   - submission: Mutable submission context for command buffer creation.
    ///   - sequenceLength: Sequence length for grid size resolution.
    ///   - iterations: Number of measured iterations (excludes warmup).
    ///   - ephemeralResidency: Residency lease for GPU buffer access.
    /// - Returns: Per-step profiles with accumulated GPU microseconds.
    static func profilePrefillSteps(
        plan: MetalPrefillPlan,
        submission: inout MetalSubmissionContext,
        sequenceLength: Int,
        iterations: Int,
        ephemeralResidency: MetalResidencyLease = .empty
    ) throws -> [PrefillStepProfile] {
        guard sequenceLength > 0, sequenceLength <= plan.maximumSequenceLength else {
            throw PrefillProfilingError.invalidSequenceLength(
                requested: sequenceLength,
                maximum: plan.maximumSequenceLength
            )
        }

        populatePrefillInputsForProfiling(plan: plan, sequenceLength: sequenceLength)

        var profiles: [PrefillStepProfile] = plan.steps.enumerated().map { index, step in
            let kernelName = step.metadata.kernelName ?? step.pipeline.label ?? "(unlabeled)"
            return PrefillStepProfile(
                index: index,
                kernelName: kernelName,
                category: PrefillStepCategory.classify(kernelName: kernelName),
                mode: step.mode,
                gridSize: step.resolvedGridSize(sequenceLength: sequenceLength),
                threadgroupSize: step.descriptor.threadgroupSize
            )
        }

        let runtimeConstantBuffer = plan.buffers.runtimeConstantBuffer

        // Warmup pass
        for step in plan.steps {
            _ = try submission.withComputeTimed(
                ephemeralResidency: ephemeralResidency
            ) { encoder, argumentTable in
                encodeSinglePrefillStep(
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
                let timing = try submission.withComputeTimed(
                    ephemeralResidency: ephemeralResidency
                ) { encoder, argumentTable in
                    encodeSinglePrefillStep(
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

    // MARK: Aggregation

    /// Aggregate step profiles into a per-category breakdown.
    static func aggregateByCategory(
        profiles: [PrefillStepProfile],
        iterations: Int
    ) -> PrefillCategoryBreakdown {
        var breakdown = PrefillCategoryBreakdown()
        for profile in profiles {
            let averageMicroseconds = profile.totalMicroseconds / Double(iterations)
            var entry = breakdown.entries[profile.category] ?? .init()
            entry.stepCount += 1
            entry.totalMicroseconds += averageMicroseconds
            entry.kernelNames[profile.kernelName, default: 0] += 1
            breakdown.entries[profile.category] = entry
            breakdown.totalMicroseconds += averageMicroseconds
        }
        return breakdown
    }

    // MARK: Formatting

    /// Format a category breakdown as a human-readable profiling report.
    static func formatBreakdown(
        _ breakdown: PrefillCategoryBreakdown,
        label: String
    ) -> String {
        guard breakdown.totalMicroseconds > 0 else {
            return "[PrefillProfile.\(label)] (no timing data)"
        }

        let totalMilliseconds = breakdown.totalMicroseconds / 1000.0
        var lines: [String] = []
        lines.append(
            "[PrefillProfile.\(label)] total=\(String(format: "%.3f", totalMilliseconds))ms "
                + "steps=\(breakdown.entries.values.reduce(0) { $0 + $1.stepCount })"
        )

        let sortedEntries = breakdown.entries
            .sorted { $0.value.totalMicroseconds > $1.value.totalMicroseconds }

        for (category, entry) in sortedEntries {
            let categoryMilliseconds = entry.totalMicroseconds / 1000.0
            let percentage = (entry.totalMicroseconds / breakdown.totalMicroseconds) * 100.0
            let kernelSummary = entry.kernelNames
                .sorted { $0.value > $1.value }
                .map { "\($0.key)x\($0.value)" }
                .joined(separator: ",")
            lines.append(
                "[PrefillProfile.\(label)] category=\(category.rawValue) "
                    + "steps=\(entry.stepCount) "
                    + "avg=\(String(format: "%.3f", categoryMilliseconds))ms "
                    + "(\(String(format: "%.1f", percentage))%) "
                    + "kernels=[\(kernelSummary)]"
            )
        }

        return lines.joined(separator: "\n")
    }
}

// MARK: - Error

enum PrefillProfilingError: Error, CustomStringConvertible {
    case invalidSequenceLength(requested: Int, maximum: Int)

    var description: String {
        switch self {
        case .invalidSequenceLength(let requested, let maximum):
            return "Requested sequence length \(requested) exceeds plan maximum \(maximum)"
        }
    }
}
