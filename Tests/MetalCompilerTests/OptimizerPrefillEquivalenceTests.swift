import Testing
import Metal
@testable import MetalCompiler

@Suite("Optimizer Prefill Equivalence", .serialized)
struct OptimizerPrefillEquivalenceTests {
    @Test("Standard prefill final hidden is repeatable across fresh loads on LFM")
    func standardPrefillFinalHiddenIsRepeatableAcrossFreshLoads() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let firstHidden = try capturePrefill(
            optimizer: StandardOptimizer(),
            tokens: promptTokens
        ) { model in
            try model.debugPrefillLastTokenFinalHidden(tokens: promptTokens)
        }
        let secondHidden = try capturePrefill(
            optimizer: StandardOptimizer(),
            tokens: promptTokens
        ) { model in
            try model.debugPrefillLastTokenFinalHidden(tokens: promptTokens)
        }

        let error = maxAbsoluteError(firstHidden, secondHidden)
        #expect(
            error == 0,
            "standard prefill should be repeatable across fresh loads, maxErr=\(error)"
        )
    }

    @Test("Standard and aggressive prefill final hidden match on LFM")
    func standardAndAggressivePrefillFinalHiddenMatch() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let standardHidden = try capturePrefill(
            optimizer: StandardOptimizer(),
            tokens: promptTokens
        ) { model in
            try model.debugPrefillLastTokenFinalHidden(tokens: promptTokens)
        }
        let aggressiveHidden = try capturePrefill(
            optimizer: AggressiveOptimizer(),
            tokens: promptTokens
        ) { model in
            try model.debugPrefillLastTokenFinalHidden(tokens: promptTokens)
        }

        #expect(standardHidden.count == aggressiveHidden.count)
        #expect(
            maxAbsoluteError(standardHidden, aggressiveHidden) == 0,
            "standard and aggressive prefill final hidden should match exactly"
        )
    }

    @Test("Standard and aggressive prefill layer outputs match on LFM")
    func standardAndAggressivePrefillLayerOutputsMatch() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let standardCapture = try capturePrefill(
            optimizer: StandardOptimizer(),
            tokens: promptTokens
        ) { model in
            let capturePoints = layerTerminalSteps(in: try #require(model.prefillPlan))
            let snapshots = try model.debugPrefillLastTokenHiddenSnapshots(
                tokens: promptTokens,
                stepIndices: Set(capturePoints.values)
            )
            return (capturePoints, snapshots)
        }
        let aggressiveCapture = try capturePrefill(
            optimizer: AggressiveOptimizer(),
            tokens: promptTokens
        ) { model in
            let capturePoints = layerTerminalSteps(in: try #require(model.prefillPlan))
            let snapshots = try model.debugPrefillLastTokenHiddenSnapshots(
                tokens: promptTokens,
                stepIndices: Set(capturePoints.values)
            )
            return (capturePoints, snapshots)
        }

        let standardCapturePoints = standardCapture.0
        let aggressiveCapturePoints = aggressiveCapture.0

        #expect(standardCapturePoints.keys.sorted() == aggressiveCapturePoints.keys.sorted())
        let standardSnapshots = standardCapture.1
        let aggressiveSnapshots = aggressiveCapture.1

        for layerIndex in standardCapturePoints.keys.sorted() {
            let standardStep = try #require(standardCapturePoints[layerIndex])
            let aggressiveStep = try #require(aggressiveCapturePoints[layerIndex])
            let standardHidden = try #require(standardSnapshots[standardStep])
            let aggressiveHidden = try #require(aggressiveSnapshots[aggressiveStep])
            let error = maxAbsoluteError(standardHidden, aggressiveHidden)
            #expect(
                error == 0,
                "layer \(layerIndex) mismatch: standard step \(standardStep), aggressive step \(aggressiveStep), maxErr=\(error)"
            )
        }
    }

    @Test("Standard and aggressive prefill final scratch hidden match on LFM")
    func standardAndAggressivePrefillFinalScratchHiddenMatch() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let hiddenSize = 2048
        let standardCapture = try capturePrefill(
            optimizer: StandardOptimizer(),
            tokens: promptTokens
        ) { model in
            let plan = try #require(model.prefillPlan)
            let step = try #require(plan.steps.lastIndex(where: { $0.mode != .lastToken }))
            let scratch = try #require(
                model.debugPrefillLastTokenScratchSnapshots(
                    tokens: promptTokens,
                    stepIndices: [step],
                    slotIndex: 0,
                    rowStride: hiddenSize,
                    count: hiddenSize
                )[step]
            )
            return (step, scratch, describeTail(of: plan))
        }
        let aggressiveCapture = try capturePrefill(
            optimizer: AggressiveOptimizer(),
            tokens: promptTokens
        ) { model in
            let plan = try #require(model.prefillPlan)
            let step = try #require(plan.steps.lastIndex(where: { $0.mode != .lastToken }))
            let scratch = try #require(
                model.debugPrefillLastTokenScratchSnapshots(
                    tokens: promptTokens,
                    stepIndices: [step],
                    slotIndex: 0,
                    rowStride: hiddenSize,
                    count: hiddenSize
                )[step]
            )
            return (step, scratch, describeTail(of: plan))
        }

        let standardStep = standardCapture.0
        let aggressiveStep = aggressiveCapture.0
        let standardScratch = standardCapture.1
        let aggressiveScratch = aggressiveCapture.1

        let error = maxAbsoluteError(standardScratch, aggressiveScratch)
        #expect(
            error == 0,
            """
            final scratch mismatch: standard step \(standardStep), aggressive step \(aggressiveStep), maxErr=\(error)
            standard tail:
            \(standardCapture.2)
            aggressive tail:
            \(aggressiveCapture.2)
            """
        )
    }

    @Test("Standard and aggressive prefill final residual match on LFM")
    func standardAndAggressivePrefillFinalResidualMatch() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let standardCapture = try capturePrefill(
            optimizer: StandardOptimizer(),
            tokens: promptTokens
        ) { model in
            let plan = try #require(model.prefillPlan)
            let step = try #require(plan.steps.lastIndex(where: { $0.mode != .lastToken }))
            let residual = try #require(
                model.debugPrefillLastTokenResidualSnapshots(
                    tokens: promptTokens,
                    stepIndices: [step]
                )[step]
            )
            return (step, residual, describeTail(of: plan))
        }
        let aggressiveCapture = try capturePrefill(
            optimizer: AggressiveOptimizer(),
            tokens: promptTokens
        ) { model in
            let plan = try #require(model.prefillPlan)
            let step = try #require(plan.steps.lastIndex(where: { $0.mode != .lastToken }))
            let residual = try #require(
                model.debugPrefillLastTokenResidualSnapshots(
                    tokens: promptTokens,
                    stepIndices: [step]
                )[step]
            )
            return (step, residual, describeTail(of: plan))
        }

        let standardStep = standardCapture.0
        let aggressiveStep = aggressiveCapture.0
        let standardResidual = standardCapture.1
        let aggressiveResidual = aggressiveCapture.1

        let error = maxAbsoluteError(standardResidual, aggressiveResidual)
        #expect(
            error == 0,
            """
            final residual mismatch: standard step \(standardStep), aggressive step \(aggressiveStep), maxErr=\(error)
            standard tail:
            \(standardCapture.2)
            aggressive tail:
            \(aggressiveCapture.2)
            """
        )
    }

    @Test("Standard and aggressive prefill tail checkpoints match on LFM")
    func standardAndAggressivePrefillTailCheckpointsMatch() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let hiddenSize = 2048
        let standardCapture = try capturePrefill(
            optimizer: StandardOptimizer(),
            tokens: promptTokens
        ) { model in
            let plan = try #require(model.prefillPlan)
            let scratch = try model.debugPrefillLastTokenScratchSnapshots(
                tokens: promptTokens,
                stepIndices: Set([252]),
                slotIndex: 0,
                rowStride: hiddenSize,
                count: hiddenSize
            )
            let hidden = try model.debugPrefillLastTokenHiddenSnapshots(
                tokens: promptTokens,
                stepIndices: Set([253, 254])
            )
            return (scratch, hidden, describeTail(of: plan))
        }
        let aggressiveCapture = try capturePrefill(
            optimizer: AggressiveOptimizer(),
            tokens: promptTokens
        ) { model in
            let plan = try #require(model.prefillPlan)
            let scratch = try model.debugPrefillLastTokenScratchSnapshots(
                tokens: promptTokens,
                stepIndices: Set([252]),
                slotIndex: 0,
                rowStride: hiddenSize,
                count: hiddenSize
            )
            let hidden = try model.debugPrefillLastTokenHiddenSnapshots(
                tokens: promptTokens,
                stepIndices: Set([253, 254])
            )
            return (scratch, hidden, describeTail(of: plan))
        }

        let checkpointSteps = [252, 253, 254]
        let standardScratch = standardCapture.0
        let aggressiveScratch = aggressiveCapture.0
        let standardHidden = standardCapture.1
        let aggressiveHidden = aggressiveCapture.1

        let scratchError = maxAbsoluteError(
            try #require(standardScratch[252]),
            try #require(aggressiveScratch[252])
        )
        let hidden253Error = maxAbsoluteError(
            try #require(standardHidden[253]),
            try #require(aggressiveHidden[253])
        )
        let hidden254Error = maxAbsoluteError(
            try #require(standardHidden[254]),
            try #require(aggressiveHidden[254])
        )

        #expect(
            scratchError == 0 && hidden253Error == 0 && hidden254Error == 0,
            """
            tail checkpoint mismatch:
            step 252 scratch maxErr=\(scratchError)
            step 253 hidden maxErr=\(hidden253Error)
            step 254 hidden maxErr=\(hidden254Error)
            standard tail:
            \(standardCapture.2)
            aggressive tail:
            \(aggressiveCapture.2)
            """
        )
        _ = checkpointSteps
    }

    @Test("Standard and aggressive prefill first hidden divergence is reported on LFM")
    func standardAndAggressivePrefillFirstHiddenDivergenceIsReported() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let standardCapture = try capturePrefill(
            optimizer: StandardOptimizer(),
            tokens: promptTokens
        ) { model in
            let plan = try #require(model.prefillPlan)
            let steps = Set(0..<plan.steps.count)
            let snapshots = try model.debugPrefillLastTokenHiddenSnapshots(
                tokens: promptTokens,
                stepIndices: steps
            )
            return (plan.steps.count, snapshots, describeTail(of: plan))
        }
        let aggressiveCapture = try capturePrefill(
            optimizer: AggressiveOptimizer(),
            tokens: promptTokens
        ) { model in
            let plan = try #require(model.prefillPlan)
            let steps = Set(0..<plan.steps.count)
            let snapshots = try model.debugPrefillLastTokenHiddenSnapshots(
                tokens: promptTokens,
                stepIndices: steps
            )
            return (plan.steps.count, snapshots, describeTail(of: plan))
        }

        let comparedStepCount = min(standardCapture.0, aggressiveCapture.0)
        let standardSnapshots = standardCapture.1
        let aggressiveSnapshots = aggressiveCapture.1

        let firstMismatch = (0..<comparedStepCount).first(where: { stepIndex in
            guard
                let lhs = standardSnapshots[stepIndex],
                let rhs = aggressiveSnapshots[stepIndex]
            else {
                return true
            }
            return maxAbsoluteError(lhs, rhs) != 0
        })

        #expect(
            firstMismatch == nil,
            """
            first hidden divergence step: \(firstMismatch.map(String.init) ?? "none")
            standard tail:
            \(standardCapture.2)
            aggressive tail:
            \(aggressiveCapture.2)
            """
        )
    }

    @Test("Standard and aggressive prefill earliest tail divergence is reported on LFM")
    func standardAndAggressivePrefillEarliestTailDivergenceIsReported() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let hiddenSize = 2048
        let tailMLPDimension = 8192
        let scratchSteps: [(step: Int, slot: Int, count: Int, label: String)] = [
            (249, 0, hiddenSize, "post-final-norm"),
            (250, 1, tailMLPDimension, "gate-proj"),
            (251, 2, tailMLPDimension, "up-proj"),
            (252, 0, tailMLPDimension, "post-swiglu"),
            (255, 0, hiddenSize, "final-hidden"),
        ]
        let hiddenSteps: [(step: Int, label: String)] = [
            (248, "input-to-final-tail"),
            (253, "down-proj"),
            (254, "post-residual-add"),
        ]
        let standardCapture = try capturePrefill(
            optimizer: StandardOptimizer(),
            tokens: promptTokens
        ) { model in
            let plan = try #require(model.prefillPlan)
            let scratch = try Dictionary(
                uniqueKeysWithValues: scratchSteps.map { checkpoint in
                    let values = try #require(
                        model.debugPrefillLastTokenScratchSnapshots(
                            tokens: promptTokens,
                            stepIndices: [checkpoint.step],
                            slotIndex: checkpoint.slot,
                            rowStride: checkpoint.count,
                            count: checkpoint.count
                        )[checkpoint.step]
                    )
                    return (checkpoint.step, values)
                }
            )
            let hidden = try model.debugPrefillLastTokenHiddenSnapshots(
                tokens: promptTokens,
                stepIndices: Set(hiddenSteps.map(\.step))
            )
            return (
                scratch,
                hidden,
                describeRange(of: plan, from: 236, through: 248),
                describeTail(of: plan)
            )
        }
        let aggressiveCapture = try capturePrefill(
            optimizer: AggressiveOptimizer(),
            tokens: promptTokens
        ) { model in
            let plan = try #require(model.prefillPlan)
            let scratch = try Dictionary(
                uniqueKeysWithValues: scratchSteps.map { checkpoint in
                    let values = try #require(
                        model.debugPrefillLastTokenScratchSnapshots(
                            tokens: promptTokens,
                            stepIndices: [checkpoint.step],
                            slotIndex: checkpoint.slot,
                            rowStride: checkpoint.count,
                            count: checkpoint.count
                        )[checkpoint.step]
                    )
                    return (checkpoint.step, values)
                }
            )
            let hidden = try model.debugPrefillLastTokenHiddenSnapshots(
                tokens: promptTokens,
                stepIndices: Set(hiddenSteps.map(\.step))
            )
            return (
                scratch,
                hidden,
                describeRange(of: plan, from: 236, through: 248),
                describeTail(of: plan)
            )
        }

        let standardScratch = standardCapture.0
        let aggressiveScratch = aggressiveCapture.0
        let standardHidden = standardCapture.1
        let aggressiveHidden = aggressiveCapture.1

        var firstMismatch: String?
        var reportLines: [String] = []
        for checkpoint in scratchSteps {
            let error = maxAbsoluteError(
                try #require(standardScratch[checkpoint.step]),
                try #require(aggressiveScratch[checkpoint.step])
            )
            reportLines.append("step \(checkpoint.step) \(checkpoint.label) scratch maxErr=\(error)")
            if error != 0, firstMismatch == nil {
                firstMismatch = "step \(checkpoint.step) \(checkpoint.label)"
            }
        }
        for checkpoint in hiddenSteps {
            let error = maxAbsoluteError(
                try #require(standardHidden[checkpoint.step]),
                try #require(aggressiveHidden[checkpoint.step])
            )
            reportLines.append("step \(checkpoint.step) \(checkpoint.label) hidden maxErr=\(error)")
            if error != 0, firstMismatch == nil {
                firstMismatch = "step \(checkpoint.step) \(checkpoint.label)"
            }
        }

        #expect(
            firstMismatch == nil,
            """
            earliest tail divergence: \(firstMismatch ?? "none")
            \(reportLines.joined(separator: "\n"))
            standard pre-tail:
            \(standardCapture.2)
            aggressive pre-tail:
            \(aggressiveCapture.2)
            standard tail:
            \(standardCapture.3)
            aggressive tail:
            \(aggressiveCapture.3)
            """
        )
    }

    @Test("Standard and aggressive prefill earliest pre-tail divergence is reported on LFM")
    func standardAndAggressivePrefillEarliestPreTailDivergenceIsReported() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let hiddenSize = 2048
        let slotDimension = 6144
        let preTailMLPDimension = 8192

        let hiddenSteps: [(step: Int, label: String)] = [
            (239, "pre-tail mlp down-proj"),
            (240, "pre-tail residual-add"),
            (246, "conv tail projection"),
            (247, "conv tail residual-add"),
        ]
        let scratchSteps: [(step: Int, slot: Int, count: Int, label: String)] = [
            (236, 1, preTailMLPDimension, "pre-tail gate-proj"),
            (237, 2, preTailMLPDimension, "pre-tail up-proj"),
            (238, 0, preTailMLPDimension, "pre-tail post-swiglu"),
            (242, 0, hiddenSize, "conv pre-norm"),
            (244, 0, hiddenSize, "conv1d output"),
        ]

        let standardCapture = try capturePrefill(
            optimizer: StandardOptimizer(),
            tokens: promptTokens
        ) { model in
            let plan = try #require(model.prefillPlan)
            let hidden = try model.debugPrefillLastTokenHiddenSnapshots(
                tokens: promptTokens,
                stepIndices: Set(hiddenSteps.map(\.step))
            )
            let scratch = try Dictionary(
                uniqueKeysWithValues: scratchSteps.map { checkpoint in
                    let values = try #require(
                        model.debugPrefillLastTokenScratchSnapshots(
                            tokens: promptTokens,
                            stepIndices: [checkpoint.step],
                            slotIndex: checkpoint.slot,
                            rowStride: checkpoint.count,
                            count: checkpoint.count
                        )[checkpoint.step]
                    )
                    return (checkpoint.step, values)
                }
            )
            return (hidden, scratch, describeRange(of: plan, from: 236, through: 248))
        }
        let aggressiveCapture = try capturePrefill(
            optimizer: AggressiveOptimizer(),
            tokens: promptTokens
        ) { model in
            let plan = try #require(model.prefillPlan)
            let hidden = try model.debugPrefillLastTokenHiddenSnapshots(
                tokens: promptTokens,
                stepIndices: Set(hiddenSteps.map(\.step))
            )
            let scratch = try Dictionary(
                uniqueKeysWithValues: scratchSteps.map { checkpoint in
                    let values = try #require(
                        model.debugPrefillLastTokenScratchSnapshots(
                            tokens: promptTokens,
                            stepIndices: [checkpoint.step],
                            slotIndex: checkpoint.slot,
                            rowStride: checkpoint.count,
                            count: checkpoint.count
                        )[checkpoint.step]
                    )
                    return (checkpoint.step, values)
                }
            )
            return (hidden, scratch, describeRange(of: plan, from: 236, through: 248))
        }

        let standardHidden = standardCapture.0
        let aggressiveHidden = aggressiveCapture.0
        let standardScratch = standardCapture.1
        let aggressiveScratch = aggressiveCapture.1

        var firstMismatch: String?
        var reportLines: [String] = []
        for checkpoint in hiddenSteps {
            let error = maxAbsoluteError(
                try #require(standardHidden[checkpoint.step]),
                try #require(aggressiveHidden[checkpoint.step])
            )
            reportLines.append("step \(checkpoint.step) \(checkpoint.label) hidden maxErr=\(error)")
            if error != 0, firstMismatch == nil {
                firstMismatch = "step \(checkpoint.step) \(checkpoint.label)"
            }
        }
        for checkpoint in scratchSteps {
            let error = maxAbsoluteError(
                try #require(standardScratch[checkpoint.step]),
                try #require(aggressiveScratch[checkpoint.step])
            )
            reportLines.append("step \(checkpoint.step) \(checkpoint.label) scratch maxErr=\(error)")
            if error != 0, firstMismatch == nil {
                firstMismatch = "step \(checkpoint.step) \(checkpoint.label)"
            }
        }

        #expect(
            firstMismatch == nil,
            """
            earliest pre-tail divergence: \(firstMismatch ?? "none")
            \(reportLines.joined(separator: "\n"))
            standard pre-tail:
            \(standardCapture.2)
            aggressive pre-tail:
            \(aggressiveCapture.2)
            """
        )
        _ = slotDimension
    }

    private func capturePrefill<T>(
        optimizer: any DispatchOptimizer,
        tokens: [Int32],
        body: (inout MetalInferenceModel) throws -> T
    ) throws -> T {
        let _ = tokens
        return try autoreleasepool {
            let (loadedModel, _) = try BenchmarkSupport.setupOrSkip(
                optimizer: optimizer,
                useCachedStore: false
            )
            var model = loadedModel
            let result = try body(&model)
            BenchmarkSupport.settleGPU()
            return result
        }
    }

    private func maxAbsoluteError(_ lhs: [Float], _ rhs: [Float]) -> Float {
        zip(lhs, rhs).reduce(0) { current, pair in
            max(current, abs(pair.0 - pair.1))
        }
    }

    private func layerTerminalSteps(in plan: MetalPrefillPlan) -> [Int: Int] {
        var terminalSteps: [Int: Int] = [:]
        for (stepIndex, step) in plan.steps.enumerated() {
            guard let layerIndex = step.metadata.layerIndex else { continue }
            terminalSteps[layerIndex] = stepIndex
        }
        return terminalSteps
    }

    private func describeTail(of plan: MetalPrefillPlan, count: Int = 10) -> String {
        plan.steps.enumerated().suffix(count).map { index, step in
            describe(step: step, index: index, plan: plan)
        }.joined(separator: "\n")
    }

    private func describeRange(
        of plan: MetalPrefillPlan,
        from startIndex: Int,
        through endIndex: Int
    ) -> String {
        guard !plan.steps.isEmpty else { return "<empty>" }
        let lowerBound = max(startIndex, 0)
        let upperBound = min(endIndex, plan.steps.count - 1)
        guard lowerBound <= upperBound else { return "<out-of-range>" }
        return (lowerBound...upperBound).map { index in
            describe(step: plan.steps[index], index: index, plan: plan)
        }.joined(separator: "\n")
    }

    private func describe(
        step: MetalPrefillStep,
        index: Int,
        plan: MetalPrefillPlan
    ) -> String {
        let kernel = step.pipeline.label ?? "<unnamed>"
        let sync = step.sync
        let arg = step.bindings.argumentPolicy
        let const = step.bindings.constantPolicy
        let inputBinding = step.bufferBindings.first(where: { $0.index == 0 })
        let outputBinding = step.bufferBindings.first(where: { $0.index == 2 })
        let inputLabel = inputBinding.map { bindingDescription($0, plan: plan) } ?? "none"
        let outputLabel = outputBinding.map { bindingDescription($0, plan: plan) } ?? "none"
        let allBindings = step.bufferBindings
            .map { "b\($0.index)=\(bindingDescription($0, plan: plan))" }
            .joined(separator: ", ")
        let bytes = step.bindings.constants
            .sorted { $0.index < $1.index }
            .map { "c\($0.index)=\(describeConstant($0))" }
            .joined(separator: ", ")
        return "[\(index)] mode=\(step.mode) layer=\(step.metadata.layerIndex.map(String.init) ?? "-") kernel=\(kernel) sync=\(sync) arg=\(arg) const=\(const) in=\(inputLabel) out=\(outputLabel) buffers=[\(allBindings)] bytes=[\(bytes)]"
    }

    private func describeConstant(_ binding: MetalConstantBinding) -> String {
        switch binding {
        case .inline(let bytes):
            return describeBytes(bytes.value)
        case .buffer(let bytes):
            let base = bytes.buffer.contents().advanced(by: bytes.offset)
            let value = Array(UnsafeBufferPointer(
                start: base.assumingMemoryBound(to: UInt8.self),
                count: bytes.length
            ))
            return describeBytes(value)
        }
    }

    private func describeBytes(_ value: [UInt8]) -> String {
        switch value.count {
        case MemoryLayout<UInt32>.size:
            let number = value.withUnsafeBytes { $0.load(as: UInt32.self) }
            return "\(number)"
        case MemoryLayout<Float>.size:
            let number = value.withUnsafeBytes { $0.load(as: Float.self) }
            return "\(number)"
        default:
            return value.map(String.init).joined(separator: " ")
        }
    }

    private func bindingDescription(
        _ binding: (index: Int, buffer: MTLBuffer, offset: Int),
        plan: MetalPrefillPlan
    ) -> String {
        let name: String
        if binding.buffer === plan.buffers.hidden {
            name = "hidden"
        } else if binding.buffer === plan.buffers.scratch {
            name = "scratch"
        } else if binding.buffer === plan.buffers.logits {
            name = "logits"
        } else if binding.buffer === plan.buffers.residual {
            name = "residual"
        } else {
            name = "buffer"
        }
        return "\(name)@\(binding.offset)"
    }
}
