import Testing
import Darwin
import Metal
@testable import MetalCompiler

#if ENABLE_METAL_PROBES
@Suite("Optimizer Attention Probes", .serialized)
struct OptimizerAttentionProbeTests {
    @Test("LFM layer2 flash output stays unstable with and without device visibility")
    func lfmLayer2FlashOutputStaysUnstableWithAndWithoutDeviceVisibility() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let flashContext = try withFreshProbeSetup { model, collected in
            try resolvedLayer2FlashContext(
                from: .init(model: model, collected: collected)
            )
        }

        func flashProbe(
            visibilityOptions: MTL4VisibilityOptions = []
        ) throws -> (head: [Float], mid: [Float], tail: [Float], maximum: Float) {
            try withFreshProbeSetup { model, _ in
                try bindingSegmentProbe(
                    model: &model,
                    tokens: promptTokens,
                    stepIndex: flashContext.flashStepIndex,
                    bindingIndex: flashContext.flashOutputBindingIndex,
                    outputDimension: flashContext.flashOutputDimension,
                    visibilityOptions: visibilityOptions
                )
            }
        }

        let defaultFirst = try flashProbe()
        let defaultSecond = try flashProbe()
        let deviceFirst = try flashProbe(visibilityOptions: .device)
        let deviceSecond = try flashProbe(visibilityOptions: .device)

        let defaultErrors = [
            maxAbsoluteError(defaultFirst.head, defaultSecond.head),
            maxAbsoluteError(defaultFirst.mid, defaultSecond.mid),
            maxAbsoluteError(defaultFirst.tail, defaultSecond.tail),
        ]
        let deviceErrors = [
            maxAbsoluteError(deviceFirst.head, deviceSecond.head),
            maxAbsoluteError(deviceFirst.mid, deviceSecond.mid),
            maxAbsoluteError(deviceFirst.tail, deviceSecond.tail),
        ]
        let defaultMaxError = defaultErrors.max() ?? 0
        let deviceMaxError = deviceErrors.max() ?? 0

        #expect(
            defaultFirst.maximum > 0
                && defaultSecond.maximum > 0
                && deviceFirst.maximum > 0
                && deviceSecond.maximum > 0
                && defaultMaxError >= 0.0001
                && deviceMaxError >= 0.0001,
            """
            layer2 flash output unexpectedly stabilized under the default or device-visibility replay
            flashStep=\(flashContext.flashStepIndex)
            outProjStep=\(flashContext.outProjStepIndex)
            cacheFillStep=\(flashContext.cacheFillStepIndex)
            qjlDimension=\(flashContext.qjlDimension)
            kvHeadCount=\(flashContext.kvHeadCount)
            defaultMaxima=[\(defaultFirst.maximum), \(defaultSecond.maximum)]
            deviceMaxima=[\(deviceFirst.maximum), \(deviceSecond.maximum)]
            defaultErrors=\(defaultErrors)
            deviceErrors=\(deviceErrors)
            default head first=\(defaultFirst.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            default head second=\(defaultSecond.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            device head first=\(deviceFirst.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            device head second=\(deviceSecond.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            """
        )
    }

    @Test("LFM layer2 QJL residual path is stable across fresh model instances when enabled")
    func lfmLayer2QJLResidualPathIsStableAcrossFreshModelInstancesWhenEnabled() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let flashContext = try withFreshProbeSetup { model, collected in
            try resolvedLayer2FlashContext(
                from: .init(model: model, collected: collected)
            )
        }
        guard flashContext.qjlDimension > 0 else {
            return
        }
        let totalQJLRows = promptTokens.count * flashContext.kvHeadCount
        let sampledRows = Array(Set([
            0,
            max(flashContext.kvHeadCount - 1, 0),
            max(totalQJLRows - flashContext.kvHeadCount, 0),
            max(totalQJLRows - 1, 0),
        ])).sorted()
        let probeCount = min(flashContext.qjlDimension, 16)
        let writeProbes = rowProbes(
            labelPrefix: "qjl-write-",
            bindingIndex: 15,
            phase: .afterStep,
            rowStride: flashContext.qjlDimension,
            count: probeCount,
            rowIndices: sampledRows
        )
        let readProbes = rowProbes(
            labelPrefix: "qjl-read-",
            bindingIndex: 17,
            phase: .beforeStep,
            rowStride: flashContext.qjlDimension,
            count: probeCount,
            rowIndices: sampledRows
        )

        func qjlSnapshots(
            stepIndex: Int,
            probes: [MetalInferenceModel.DebugPrefillBindingProbe]
        ) throws -> [String: [Float]] {
            try withFreshProbeSetup { model, _ in
                try model.debugPrefillBindingProbes(
                    tokens: promptTokens,
                    stepIndex: stepIndex,
                    probes: probes
                )
            }
        }

        let firstWriteSnapshots = try qjlSnapshots(
            stepIndex: flashContext.cacheFillStepIndex,
            probes: writeProbes
        )
        let secondWriteSnapshots = try qjlSnapshots(
            stepIndex: flashContext.cacheFillStepIndex,
            probes: writeProbes
        )
        let firstReadSnapshots = try qjlSnapshots(
            stepIndex: flashContext.flashStepIndex,
            probes: readProbes
        )
        let secondReadSnapshots = try qjlSnapshots(
            stepIndex: flashContext.flashStepIndex,
            probes: readProbes
        )

        var writeErrors: [Float] = []
        var readErrors: [Float] = []
        var writeMaxima: [Float] = []
        var readMaxima: [Float] = []
        for rowIndex in sampledRows {
            let writeKey = "qjl-write-\(rowIndex)"
            let readKey = "qjl-read-\(rowIndex)"
            let firstWrite = try #require(firstWriteSnapshots[writeKey])
            let secondWrite = try #require(secondWriteSnapshots[writeKey])
            let firstRead = try #require(firstReadSnapshots[readKey])
            let secondRead = try #require(secondReadSnapshots[readKey])
            writeErrors.append(maxAbsoluteError(firstWrite, secondWrite))
            readErrors.append(maxAbsoluteError(firstRead, secondRead))
            writeMaxima.append(maxAbsoluteValue(firstWrite))
            readMaxima.append(maxAbsoluteValue(firstRead))
        }

        #expect(
            writeMaxima.allSatisfy { $0 > 0 }
                && readMaxima.allSatisfy { $0 > 0 }
                && writeErrors.allSatisfy { $0 < 0.0001 }
                && readErrors.allSatisfy { $0 < 0.0001 },
            """
            layer2 QJL residual path is unstable across fresh model instances
            flashStep=\(flashContext.flashStepIndex)
            cacheFillStep=\(flashContext.cacheFillStepIndex)
            qjlDimension=\(flashContext.qjlDimension)
            kvHeadCount=\(flashContext.kvHeadCount)
            sampledRows=\(sampledRows)
            writeMaxima=\(writeMaxima)
            readMaxima=\(readMaxima)
            writeErrors=\(writeErrors)
            readErrors=\(readErrors)
            """
        )
    }

    @Test("LFM layer2 flash instability persists after query and KV head slices stay stable")
    func lfmLayer2FlashInstabilityPersistsAfterQueryAndKVHeadSlicesStayStable() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let flashContext = try withFreshProbeSetup { model, collected in
            try resolvedLayer2FlashContext(
                from: .init(model: model, collected: collected)
            )
        }
        let flashStep = flashContext.plan.steps[flashContext.flashStepIndex]
        let headCount = try #require(uint32BindingValue(in: flashStep, at: 4))
        let kvHeadCount = try #require(uint32BindingValue(in: flashStep, at: 5))
        let headDimension = try #require(uint32BindingValue(in: flashStep, at: 6))

        let firstOutputHeads = try flashOutputHeadSlices(
            promptTokens: promptTokens,
            flashContext: flashContext,
            headCount: headCount,
            headDimension: headDimension
        )
        let secondOutputHeads = try flashOutputHeadSlices(
            promptTokens: promptTokens,
            flashContext: flashContext,
            headCount: headCount,
            headDimension: headDimension
        )

        var outputHeadErrors: [Float] = []
        outputHeadErrors.reserveCapacity(headCount)
        for headIndex in 0..<headCount {
            let firstHead = try requireHeadSlice(firstOutputHeads, headIndex: headIndex)
            let secondHead = try requireHeadSlice(secondOutputHeads, headIndex: headIndex)
            outputHeadErrors.append(maxAbsoluteError(firstHead, secondHead))
        }
        let divergentHeadIndex = try #require(
            outputHeadErrors.enumerated().max(by: { $0.element < $1.element })?.offset
        )
        let divergentHeadError = outputHeadErrors[divergentHeadIndex]
        let firstDivergentOutputHead = try requireHeadSlice(firstOutputHeads, headIndex: divergentHeadIndex)
        let secondDivergentOutputHead = try requireHeadSlice(secondOutputHeads, headIndex: divergentHeadIndex)
        let kvHeadIndex = divergentHeadIndex * kvHeadCount / headCount

        let firstQuerySlice = try flashQueryHeadSlice(
            promptTokens: promptTokens,
            flashContext: flashContext,
            headIndex: divergentHeadIndex,
            headCount: headCount,
            headDimension: headDimension
        )
        let secondQuerySlice = try flashQueryHeadSlice(
            promptTokens: promptTokens,
            flashContext: flashContext,
            headIndex: divergentHeadIndex,
            headCount: headCount,
            headDimension: headDimension
        )
        let firstKCacheRows = try flashKVHeadRows(
            promptTokens: promptTokens,
            flashContext: flashContext,
            bindingIndex: 1,
            kvHeadIndex: kvHeadIndex,
            kvHeadCount: kvHeadCount,
            headDimension: headDimension
        )
        let secondKCacheRows = try flashKVHeadRows(
            promptTokens: promptTokens,
            flashContext: flashContext,
            bindingIndex: 1,
            kvHeadIndex: kvHeadIndex,
            kvHeadCount: kvHeadCount,
            headDimension: headDimension
        )
        let firstVCacheRows = try flashKVHeadRows(
            promptTokens: promptTokens,
            flashContext: flashContext,
            bindingIndex: 2,
            kvHeadIndex: kvHeadIndex,
            kvHeadCount: kvHeadCount,
            headDimension: headDimension
        )
        let secondVCacheRows = try flashKVHeadRows(
            promptTokens: promptTokens,
            flashContext: flashContext,
            bindingIndex: 2,
            kvHeadIndex: kvHeadIndex,
            kvHeadCount: kvHeadCount,
            headDimension: headDimension
        )

        let queryError = maxAbsoluteError(firstQuerySlice, secondQuerySlice)
        let kCacheErrors = try compareRowSlices(
            first: firstKCacheRows,
            second: secondKCacheRows,
            rowIndices: Array(0..<promptTokens.count),
            labelPrefix: "kv-1-row-"
        )
        let vCacheErrors = try compareRowSlices(
            first: firstVCacheRows,
            second: secondVCacheRows,
            rowIndices: Array(0..<promptTokens.count),
            labelPrefix: "kv-2-row-"
        )

        #expect(
            divergentHeadError >= 0.0001
                && queryError < 0.0001
                && kCacheErrors.allSatisfy { $0 < 0.0001 }
                && vCacheErrors.allSatisfy { $0 < 0.0001 },
            """
            layer2 flash divergence was not isolated inside the flash stage
            flashStep=\(flashContext.flashStepIndex)
            divergentHeadIndex=\(divergentHeadIndex)
            kvHeadIndex=\(kvHeadIndex)
            headDimension=\(headDimension)
            outputHeadErrors=\(outputHeadErrors)
            divergentHeadError=\(divergentHeadError)
            queryError=\(queryError)
            kCacheErrors=\(kCacheErrors)
            vCacheErrors=\(vCacheErrors)
            first output head=\(firstDivergentOutputHead.prefix(8).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            second output head=\(secondDivergentOutputHead.prefix(8).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first query=\(firstQuerySlice.prefix(8).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            second query=\(secondQuerySlice.prefix(8).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            """
        )
    }

    @Test("LFM layer2 flash output reaches out-proj input unchanged within the same replay")
    func lfmLayer2FlashOutputReachesOutProjInputUnchangedWithinTheSameReplay() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let flashContext = try withFreshProbeSetup { model, collected in
            try resolvedLayer2FlashContext(
                from: .init(model: model, collected: collected)
            )
        }
        let flashStep = flashContext.plan.steps[flashContext.flashStepIndex]
        let outProjStep = flashContext.plan.steps[flashContext.outProjStepIndex]
        let headCount = try #require(uint32BindingValue(in: flashStep, at: 4))
        let headDimension = try #require(uint32BindingValue(in: flashStep, at: 6))
        let flashOutputBinding = try #require(
            flashStep.bindings.buffers.first(where: { $0.index == flashContext.flashOutputBindingIndex })
        )
        let outProjInputBindingIndex = try #require(
            outProjStep.bindings.buffers.first(where: { binding in
                binding.buffer === flashOutputBinding.buffer
                    && binding.offset == flashOutputBinding.offset
            })?.index
        )
        let probes = (0..<headCount).flatMap { headIndex in
            [
                MetalInferenceModel.DebugPrefillBindingProbe(
                    label: "flash-out-\(headIndex)",
                    stepIndex: flashContext.flashStepIndex,
                    bindingIndex: flashContext.flashOutputBindingIndex,
                    phase: .afterStep,
                    elementOffset: headIndex * headDimension,
                    rowStride: flashContext.flashOutputDimension,
                    count: headDimension
                ),
                MetalInferenceModel.DebugPrefillBindingProbe(
                    label: "out-proj-in-\(headIndex)",
                    stepIndex: flashContext.outProjStepIndex,
                    bindingIndex: outProjInputBindingIndex,
                    phase: .beforeStep,
                    elementOffset: headIndex * headDimension,
                    rowStride: flashContext.flashOutputDimension,
                    count: headDimension
                ),
            ]
        }
        let snapshots = try withFreshProbeSetup { model, _ in
            try model.debugPrefillBindingProbes(
                tokens: promptTokens,
                stepIndex: flashContext.flashStepIndex,
                probes: probes
            )
        }

        var maxima: [Float] = []
        var errors: [Float] = []
        for headIndex in 0..<headCount {
            let flashValues = try #require(snapshots["flash-out-\(headIndex)"])
            let outProjValues = try #require(snapshots["out-proj-in-\(headIndex)"])
            maxima.append(maxAbsoluteValue(flashValues))
            errors.append(maxAbsoluteError(flashValues, outProjValues))
        }

        #expect(
            maxima.allSatisfy { $0 > 0 }
                && errors.allSatisfy { $0 < 0.0001 },
            """
            layer2 flash output did not reach out-proj input unchanged within the same replay
            flashStep=\(flashContext.flashStepIndex)
            outProjStep=\(flashContext.outProjStepIndex)
            flashOutputBindingIndex=\(flashContext.flashOutputBindingIndex)
            outProjInputBindingIndex=\(outProjInputBindingIndex)
            headCount=\(headCount)
            headDimension=\(headDimension)
            maxima=\(maxima)
            errors=\(errors)
            """
        )
    }

    private func withFreshProbeSetup<T>(
        optimizer: (any DispatchOptimizer)? = StandardOptimizer(),
        weightAccessPolicyOverride: ProjectionWeightAccessPolicyOverride? = nil,
        body: (inout MetalInferenceModel, BenchmarkSupport.CollectedPrefillEntries) throws -> T
    ) throws -> T {
        try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            let setup = try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
                optimizer: optimizer,
                weightAccessPolicyOverride: weightAccessPolicyOverride,
                useCachedStore: false
            )
            var model = setup.model
            return try body(&model, setup.collected)
        }
    }

    private func bindingSegmentProbe(
        model: inout MetalInferenceModel,
        tokens: [Int32],
        stepIndex: Int,
        bindingIndex: Int,
        phase: MetalInferenceModel.DebugPrefillProbePhase = .afterStep,
        outputDimension: Int,
        visibilityOptions: MTL4VisibilityOptions = []
    ) throws -> (head: [Float], mid: [Float], tail: [Float], maximum: Float) {
        let segmentCount = min(outputDimension, 32)
        let middleElementOffset = max((outputDimension / 2) - (segmentCount / 2), 0)
        let tailElementOffset = max(outputDimension - segmentCount, 0)
        let probes: [MetalInferenceModel.DebugPrefillBindingProbe] = [
            .init(
                label: "output-head",
                bindingIndex: bindingIndex,
                phase: phase,
                rowStride: outputDimension,
                count: segmentCount
            ),
            .init(
                label: "output-mid",
                bindingIndex: bindingIndex,
                phase: phase,
                elementOffset: middleElementOffset,
                rowStride: outputDimension,
                count: segmentCount
            ),
            .init(
                label: "output-tail",
                bindingIndex: bindingIndex,
                phase: phase,
                elementOffset: tailElementOffset,
                rowStride: outputDimension,
                count: segmentCount
            ),
        ]
        let snapshots = try autoreleasepool {
            try model.debugPrefillBindingProbes(
                tokens: tokens,
                stepIndex: stepIndex,
                probes: probes,
                visibilityOptions: visibilityOptions
            )
        }
        let head = try #require(snapshots["output-head"])
        let mid = try #require(snapshots["output-mid"])
        let tail = try #require(snapshots["output-tail"])
        return (
            head,
            mid,
            tail,
            max(maxAbsoluteValue(head), maxAbsoluteValue(mid), maxAbsoluteValue(tail))
        )
    }

    private func rowProbes(
        labelPrefix: String,
        bindingIndex: Int,
        phase: MetalInferenceModel.DebugPrefillProbePhase,
        rowStride: Int,
        count: Int,
        rowIndices: [Int]
        ) -> [MetalInferenceModel.DebugPrefillBindingProbe] {
        rowIndices.map { rowIndex in
            .init(
                label: "\(labelPrefix)\(rowIndex)",
                bindingIndex: bindingIndex,
                phase: phase,
                rowIndex: rowIndex,
                rowStride: rowStride,
                count: count
            )
        }
    }

    private func flashOutputHeadSlices(
        promptTokens: [Int32],
        flashContext: (
            plan: MetalPrefillPlan,
            outProjStepIndex: Int,
            flashStepIndex: Int,
            cacheFillStepIndex: Int,
            flashOutputBindingIndex: Int,
            flashOutputDimension: Int,
            qjlDimension: Int,
            kvHeadCount: Int
        ),
        headCount: Int,
        headDimension: Int
    ) throws -> [String: [Float]] {
        let probes = (0..<headCount).map { headIndex in
            MetalInferenceModel.DebugPrefillBindingProbe(
                label: "flash-out-\(headIndex)",
                bindingIndex: flashContext.flashOutputBindingIndex,
                phase: .afterStep,
                elementOffset: headIndex * headDimension,
                rowStride: flashContext.flashOutputDimension,
                count: headDimension
            )
        }
        return try withFreshProbeSetup { model, _ in
            try model.debugPrefillBindingProbes(
                tokens: promptTokens,
                stepIndex: flashContext.flashStepIndex,
                probes: probes
            )
        }
    }

    private func flashQueryHeadSlice(
        promptTokens: [Int32],
        flashContext: (
            plan: MetalPrefillPlan,
            outProjStepIndex: Int,
            flashStepIndex: Int,
            cacheFillStepIndex: Int,
            flashOutputBindingIndex: Int,
            flashOutputDimension: Int,
            qjlDimension: Int,
            kvHeadCount: Int
        ),
        headIndex: Int,
        headCount: Int,
        headDimension: Int
    ) throws -> [Float] {
        try withFreshProbeSetup { model, _ in
            let snapshots = try model.debugPrefillBindingProbes(
                tokens: promptTokens,
                stepIndex: flashContext.flashStepIndex,
                probes: [
                    .init(
                        label: "query-head",
                        bindingIndex: 0,
                        phase: .beforeStep,
                        elementOffset: headIndex * headDimension,
                        rowStride: headCount * headDimension,
                        count: headDimension
                    )
                ]
            )
            return try #require(snapshots["query-head"])
        }
    }

    private func flashKVHeadRows(
        promptTokens: [Int32],
        flashContext: (
            plan: MetalPrefillPlan,
            outProjStepIndex: Int,
            flashStepIndex: Int,
            cacheFillStepIndex: Int,
            flashOutputBindingIndex: Int,
            flashOutputDimension: Int,
            qjlDimension: Int,
            kvHeadCount: Int
        ),
        bindingIndex: Int,
        kvHeadIndex: Int,
        kvHeadCount: Int,
        headDimension: Int
    ) throws -> [String: [Float]] {
        let rowIndices = Array(0..<promptTokens.count)
        let probes = rowIndices.map { rowIndex in
            MetalInferenceModel.DebugPrefillBindingProbe(
                label: "kv-\(bindingIndex)-row-\(rowIndex)",
                bindingIndex: bindingIndex,
                phase: .beforeStep,
                rowIndex: rowIndex,
                elementOffset: kvHeadIndex * headDimension,
                rowStride: kvHeadCount * headDimension,
                count: headDimension
            )
        }
        return try withFreshProbeSetup { model, _ in
            try model.debugPrefillBindingProbes(
                tokens: promptTokens,
                stepIndex: flashContext.flashStepIndex,
                probes: probes
            )
        }
    }

    private func compareRowSlices(
        first: [String: [Float]],
        second: [String: [Float]],
        rowIndices: [Int],
        labelPrefix: String
    ) throws -> [Float] {
        try rowIndices.map { rowIndex in
            let key = "\(labelPrefix)\(rowIndex)"
            let firstValues = try #require(first[key])
            let secondValues = try #require(second[key])
            return maxAbsoluteError(firstValues, secondValues)
        }
    }

    private func requireHeadSlice(
        _ snapshots: [String: [Float]],
        headIndex: Int
    ) throws -> [Float] {
        try #require(snapshots["flash-out-\(headIndex)"])
    }

    private func resolvedLayer2FlashContext(
        from setup: BenchmarkSupport.SetupWithCollectedPrefillEntries
    ) throws -> (
        plan: MetalPrefillPlan,
        outProjStepIndex: Int,
        flashStepIndex: Int,
        cacheFillStepIndex: Int,
        flashOutputBindingIndex: Int,
        flashOutputDimension: Int,
        qjlDimension: Int,
        kvHeadCount: Int
    ) {
        let plan = try #require(setup.model.prefillPlan)
        let access = try #require(
            layerProjectionAccess(
                in: setup.collected,
                layerPrefix: "model.layers.2.",
                tensorNameSuffix: ".self_attn.out_proj.weight"
            )
        )
        let projection = try #require(projectionKind(from: access.entry))
        let outProjStepIndex = try #require(
            matchingProjectionStepIndex(
                in: plan,
                entryIndex: access.entry.index,
                tensorName: access.tensorName,
                access: access.access,
                projection: projection
            )
        )
        let outProjStep = plan.steps[outProjStepIndex]
        let flashOutputDimension = try #require(uint32BindingValue(in: outProjStep, at: 3))
        let flashOutputBinding = try #require(
            outProjStep.bindings.buffers.first(where: { $0.index == 0 })
        )
        let flashStepIndex = try #require(
            lastWriterStepIndex(
                in: plan.steps,
                before: outProjStepIndex,
                target: flashOutputBinding
            )
        )
        let flashStep = plan.steps[flashStepIndex]
        let flashOutputBindingIndex = try #require(
            matchingWriteBindingIndex(
                in: flashStep,
                target: flashOutputBinding
            )
        )
        let keyCacheBinding = try #require(
            flashStep.bindings.buffers.first(where: { $0.index == 1 })
        )
        let cacheFillStepIndex = try #require(
            lastWriterStepIndex(
                in: plan.steps,
                before: flashStepIndex,
                target: keyCacheBinding
            )
        )
        let qjlDimension = Int(uint32BindingValue(in: flashStep, at: 19) ?? 0)
        let kvHeadCount = Int(try #require(uint32BindingValue(in: flashStep, at: 5)))
        return (
            plan,
            outProjStepIndex,
            flashStepIndex,
            cacheFillStepIndex,
            flashOutputBindingIndex,
            flashOutputDimension,
            qjlDimension,
            kvHeadCount
        )
    }

    private func layerProjectionAccess(
        in collected: BenchmarkSupport.CollectedPrefillEntries,
        layerPrefix: String,
        tensorNameSuffix: String
    ) -> (entry: DispatchEntry, tensorName: String, access: STAFWeightBufferAccess, schemeIdentifier: QuantizationSchemeIdentifier)? {
        let resolver = ProjectionWeightAccessPolicyResolver()
        return collected.fusedEntries.compactMap { entry -> (entry: DispatchEntry, tensorName: String, access: STAFWeightBufferAccess, schemeIdentifier: QuantizationSchemeIdentifier)? in
            guard
                case .projection = entry.kind,
                let binding = entry.parameterBindings.first(where: {
                    $0.tensorName.contains(layerPrefix) && $0.tensorName.hasSuffix(tensorNameSuffix)
                }),
                let access = collected.store.resolvedBufferAccess(
                    for: resolver.accessRequest(
                        for: entry,
                        role: binding.role,
                        binding: binding,
                        executionPhase: .prefill,
                        stafWeightStore: collected.store
                    )
                ),
                let tensor = collected.store.tensor(for: binding.tensorName)
            else {
                return nil
            }
            return (entry, binding.tensorName, access, tensor.format.schemeIdentifier)
        }.first
    }

    private func projectionKind(from entry: DispatchEntry) -> MetalProjection? {
        guard case .projection(let projection, _) = entry.kind else { return nil }
        return projection
    }

    private func matchingProjectionStepIndex(
        in plan: MetalPrefillPlan,
        entryIndex: Int,
        tensorName: String,
        access: STAFWeightBufferAccess,
        projection: MetalProjection
    ) -> Int? {
        let exactMatch = plan.steps.enumerated().first { _, step in
            step.metadata.entryIndex == entryIndex
                && projectionStepMatches(
                    step,
                    tensorName: tensorName,
                    access: access,
                    projection: projection
                )
        }?.offset
        if let exactMatch {
            return exactMatch
        }
        return plan.steps.enumerated().first { _, step in
            projectionStepMatches(
                step,
                tensorName: tensorName,
                access: access,
                projection: projection
            )
        }?.offset
    }

    private func projectionStepMatches(
        _ step: MetalPrefillStep,
        tensorName: String,
        access: STAFWeightBufferAccess,
        projection: MetalProjection
    ) -> Bool {
        guard step.metadata.weightTensorName == tensorName else {
            return false
        }
        guard
            let weightBinding = step.bindings.buffers.first(where: { $0.index == 1 }),
            weightBinding.buffer === access.buffer,
            weightBinding.offset == access.offset
        else {
            return false
        }
        return uint32BindingValue(in: step, at: 3) == projection.inputDimension
            && uint32BindingValue(in: step, at: 4) == projection.outputDimension
    }

    private func uint32BindingValue(in step: MetalPrefillStep, at index: Int) -> Int? {
        if let binding = step.bytesBindings.first(where: { $0.index == index }) {
            guard binding.value.count == MemoryLayout<UInt32>.stride else {
                return nil
            }
            return binding.value.withUnsafeBytes { rawBuffer in
                Int(rawBuffer.load(as: UInt32.self))
            }
        }
        guard let constant = step.bindings.constants.first(where: { $0.index == index }) else {
            return nil
        }
        guard case .buffer(let binding) = constant else {
            return nil
        }
        guard binding.length == MemoryLayout<UInt32>.stride else {
            return nil
        }
        let pointer = binding.buffer.contents().advanced(by: binding.offset)
        return Int(pointer.assumingMemoryBound(to: UInt32.self).pointee)
    }

    private func lastWriterStepIndex(
        in steps: [MetalPrefillStep],
        before consumerStepIndex: Int,
        target: MetalBufferBinding
    ) -> Int? {
        guard consumerStepIndex > 0 else { return nil }
        for index in stride(from: consumerStepIndex - 1, through: 0, by: -1) {
            let step = steps[index]
            let writeIndices = step.metadata.bufferAccessPattern?.writeIndices
            let matches = step.bindings.buffers.contains { binding in
                (writeIndices?.contains(binding.index) ?? true)
                    && binding.buffer === target.buffer
                    && binding.offset == target.offset
            }
            if matches {
                return index
            }
        }
        return consumerStepIndex - 1
    }

    private func matchingWriteBindingIndex(
        in step: MetalPrefillStep,
        target: MetalBufferBinding
    ) -> Int? {
        let writeIndices = step.metadata.bufferAccessPattern?.writeIndices
        return step.bindings.buffers.first(where: { binding in
            (writeIndices?.contains(binding.index) ?? true)
                && binding.buffer === target.buffer
                && binding.offset == target.offset
        })?.index
    }

    private func maxAbsoluteError(_ lhs: [Float], _ rhs: [Float]) -> Float {
        zip(lhs, rhs).reduce(0) { current, pair in
            max(current, abs(pair.0 - pair.1))
        }
    }

    private func maxAbsoluteValue(_ values: [Float]) -> Float {
        values.reduce(0) { current, value in
            max(current, abs(value))
        }
    }

    private func withTemporaryEnvironment<T>(
        _ key: String,
        _ value: String?,
        body: () throws -> T
    ) throws -> T {
        let original = getenv(key).map { String(cString: $0) }
        if let value {
            setenv(key, value, 1)
        } else {
            unsetenv(key)
        }
        defer {
            if let original {
                setenv(key, original, 1)
            } else {
                unsetenv(key)
            }
        }
        return try autoreleasepool {
            try body()
        }
    }
}
#endif
