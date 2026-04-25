import Testing
import Darwin
import Metal
@testable import MetalCompiler

@Suite("Entry Contracts", .serialized)
struct OptimizerEntryContractTests {
    @Test("LFM nil-layer down-proj bindings resolve to concrete STAF accesses")
    func lfmNilLayerDownProjBindingsResolveToConcreteSTAFAccesses() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let collected = try BenchmarkSupport.collectPrefillEntriesOrSkip(
            useCachedStore: false
        )

        let summaries = try summarizeNilLayerDownProjAccesses(in: collected)
        #expect(!summaries.isEmpty, "expected at least one nil-layer down-proj projection")
        for summary in summaries {
            #expect(summary.bindingExists, "missing binding for \(summary.tensorName)")
            #expect(summary.hasResolvedAccess, "missing STAF access for \(summary.tensorName)")
        }
    }

    @Test("LFM layer14 w1/w3 bindings resolve to concrete STAF accesses")
    func lfmLayer14W1W3BindingsResolveToConcreteSTAFAccesses() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let setup = try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
            useCachedStore: false
        )
        let plan = try #require(setup.model.prefillPlan)
        let collected = setup.collected

        let w1 = try #require(
            layerBatchedProjectionAccess(
                in: collected,
                layerPrefix: "model.layers.14.",
                role: "gate_proj",
                tensorNameSuffix: ".feed_forward.w1.weight"
            )
        )
        let w3 = try #require(
            layerBatchedProjectionAccess(
                in: collected,
                layerPrefix: "model.layers.14.",
                role: "up_proj",
                tensorNameSuffix: ".feed_forward.w3.weight"
            )
        )
        let w1Projection = w1.projection
        let w3Projection = w3.projection
        let projectionCount = try #require((w1.entry.fragment as? BatchedProjection)?.projections.count)
        let w1StepIndex = try #require(
            matchingBatchedProjectionStepIndex(
                in: plan,
                entryIndex: w1.entry.index,
                access: w1.access,
                projection: w1Projection,
                projectionIndex: w1.projectionIndex,
                projectionCount: projectionCount
            )
        )
        let w3StepIndex = try #require(
            matchingBatchedProjectionStepIndex(
                in: plan,
                entryIndex: w3.entry.index,
                access: w3.access,
                projection: w3Projection,
                projectionIndex: w3.projectionIndex,
                projectionCount: projectionCount
            )
        )

        let w1WeightBinding = try #require(plan.steps[w1StepIndex].bindings.buffers.first(where: { $0.index == 1 + w1.projectionIndex }))
        let w3WeightBinding = try #require(plan.steps[w3StepIndex].bindings.buffers.first(where: { $0.index == 1 + w3.projectionIndex }))

        #expect(
            w1WeightBinding.buffer === w1.access.buffer && w1WeightBinding.offset == w1.access.offset,
            """
            w1 step does not bind the resolved STAF access
            step=\(w1StepIndex) tensor=\(w1.tensorName)
            expectedOffset=\(w1.access.offset) actualOffset=\(w1WeightBinding.offset)
            expectedLabel=\(w1.access.buffer.label ?? "(unlabeled)")
            actualLabel=\(w1WeightBinding.buffer.label ?? "(unlabeled)")
            """
        )
        #expect(
            w3WeightBinding.buffer === w3.access.buffer && w3WeightBinding.offset == w3.access.offset,
            """
            w3 step does not bind the resolved STAF access
            step=\(w3StepIndex) tensor=\(w3.tensorName)
            expectedOffset=\(w3.access.offset) actualOffset=\(w3WeightBinding.offset)
            expectedLabel=\(w3.access.buffer.label ?? "(unlabeled)")
            actualLabel=\(w3WeightBinding.buffer.label ?? "(unlabeled)")
            """
        )
    }

    @Test("LFM layer15 conv in-proj binding and constants resolve correctly")
    func lfmLayer15ConvInProjBindingAndConstantsResolveCorrectly() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let setup = try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
            useCachedStore: false
        )
        let collected = setup.collected
        let plan = try #require(setup.model.prefillPlan)

        let access = try #require(
            layerProjectionAccess(
                in: collected,
                layerPrefix: "model.layers.15.",
                tensorNameSuffix: ".conv.in_proj.weight"
            )
        )
        let projection = try #require(projectionKind(from: access.entry))
        let stepIndex = try #require(
            matchingProjectionStepIndex(
                in: plan,
                entryIndex: access.entry.index,
                tensorName: access.tensorName,
                access: access.access,
                projection: projection
            )
        )
        let step = plan.steps[stepIndex]
        let weightBinding = try #require(step.bindings.buffers.first(where: { $0.index == 1 }))

        #expect(
            weightBinding.buffer === access.access.buffer && weightBinding.offset == access.access.offset,
            """
            conv.in_proj step does not bind the resolved STAF access
            step=\(stepIndex) tensor=\(access.tensorName)
            expectedOffset=\(access.access.offset) actualOffset=\(weightBinding.offset)
            expectedLabel=\(access.access.buffer.label ?? "(unlabeled)")
            actualLabel=\(weightBinding.buffer.label ?? "(unlabeled)")
            """
        )
        #expect(
            uint32BindingValue(in: step, at: 3) == projection.inputDimension,
            "conv.in_proj inputDimension mismatch"
        )
        #expect(
            uint32BindingValue(in: step, at: 4) == projection.outputDimension,
            "conv.in_proj outputDimension mismatch"
        )
        let staticSequenceLength = uint32BindingValue(in: step, at: 5)
        #expect(
            staticSequenceLength == plan.maximumSequenceLength,
            "conv.in_proj static sequenceLength mismatch: \(String(describing: staticSequenceLength))"
        )

        let sampleWeightMax = maxAbsoluteWeightSample(
            access: access.access,
            schemeIdentifier: access.schemeIdentifier,
            sampleCount: min(projection.inputDimension * 4, access.access.size / MemoryLayout<UInt16>.stride)
        )
        #expect(
            sampleWeightMax > 0,
            """
            conv.in_proj sampled weights are all zero
            step=\(stepIndex) tensor=\(access.tensorName)
            scheme=\(access.schemeIdentifier)
            """
        )
    }

    #if ENABLE_METAL_PROBES
    @Test("LFM layer15 norm output reaches conv in-proj input unchanged")
    func lfmLayer15NormOutputReachesConvInProjInputUnchanged() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let setup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }

        let collected = setup.collected
        var model = setup.model
        let plan = try #require(model.prefillPlan)
        let access = try #require(
            layerProjectionAccess(
                in: collected,
                layerPrefix: "model.layers.15.",
                tensorNameSuffix: ".conv.in_proj.weight"
            )
        )
        let projection = try #require(projectionKind(from: access.entry))
        let convStepIndex = try #require(
            matchingProjectionStepIndex(
                in: plan,
                entryIndex: access.entry.index,
                tensorName: access.tensorName,
                access: access.access,
                projection: projection
            )
        )
        let normStepIndex = convStepIndex - 1
        #expect(normStepIndex >= 0, "expected norm step before conv.in_proj")
        let normStep = plan.steps[normStepIndex]
        let convStep = plan.steps[convStepIndex]
        let normOutputDimension = try #require(uint32BindingValue(in: normStep, at: 3))
        let convInputDimension = try #require(uint32BindingValue(in: convStep, at: 3))
        let normOutputBinding = try #require(normStep.bindings.buffers.first(where: { $0.index == 2 }))
        let convInputBinding = try #require(convStep.bindings.buffers.first(where: { $0.index == 0 }))

        #expect(
            normOutputDimension == convInputDimension,
            "layer15 norm/conv dimension mismatch: norm=\(normOutputDimension) conv=\(convInputDimension)"
        )
        #expect(
            normOutputBinding.buffer === convInputBinding.buffer
                && normOutputBinding.offset == convInputBinding.offset,
            """
            layer15 norm output binding does not feed conv.in_proj input
            normStep=\(normStepIndex) convStep=\(convStepIndex)
            normKernel=\(normStep.pipeline.label ?? "unlabeled")
            convKernel=\(convStep.pipeline.label ?? "unlabeled")
            normOffset=\(normOutputBinding.offset) convOffset=\(convInputBinding.offset)
            normBuffer=\(normOutputBinding.buffer.label ?? "(unlabeled)")
            convBuffer=\(convInputBinding.buffer.label ?? "(unlabeled)")
            """
        )

        let segmentCount = min(normOutputDimension, 32)
        let middleElementOffset = max((normOutputDimension / 2) - (segmentCount / 2), 0)
        let tailElementOffset = max(normOutputDimension - segmentCount, 0)
        let probe = try model.debugPrefillBindingProbes(
            tokens: promptTokens,
            stepIndex: convStepIndex,
            probes: [
                .init(
                    label: "normOutput-head",
                    stepIndex: normStepIndex,
                    bindingIndex: 2,
                    phase: .afterStep,
                    rowStride: normOutputDimension,
                    count: segmentCount
                ),
                .init(
                    label: "normOutput-mid",
                    stepIndex: normStepIndex,
                    bindingIndex: 2,
                    phase: .afterStep,
                    elementOffset: middleElementOffset,
                    rowStride: normOutputDimension,
                    count: segmentCount
                ),
                .init(
                    label: "normOutput-tail",
                    stepIndex: normStepIndex,
                    bindingIndex: 2,
                    phase: .afterStep,
                    elementOffset: tailElementOffset,
                    rowStride: normOutputDimension,
                    count: segmentCount
                ),
                .init(
                    label: "convInput-head",
                    bindingIndex: 0,
                    phase: .beforeStep,
                    rowStride: convInputDimension,
                    count: segmentCount
                ),
                .init(
                    label: "convInput-mid",
                    bindingIndex: 0,
                    phase: .beforeStep,
                    elementOffset: middleElementOffset,
                    rowStride: convInputDimension,
                    count: segmentCount
                ),
                .init(
                    label: "convInput-tail",
                    bindingIndex: 0,
                    phase: .beforeStep,
                    elementOffset: tailElementOffset,
                    rowStride: convInputDimension,
                    count: segmentCount
                ),
            ]
        )
        let normHead = try #require(probe["normOutput-head"])
        let normMid = try #require(probe["normOutput-mid"])
        let normTail = try #require(probe["normOutput-tail"])
        let inputHead = try #require(probe["convInput-head"])
        let inputMid = try #require(probe["convInput-mid"])
        let inputTail = try #require(probe["convInput-tail"])
        let headError = maxAbsoluteError(normHead, inputHead)
        let midError = maxAbsoluteError(normMid, inputMid)
        let tailError = maxAbsoluteError(normTail, inputTail)
        let normMax = max(maxAbsoluteValue(normHead), maxAbsoluteValue(normMid), maxAbsoluteValue(normTail))
        let inputMax = max(maxAbsoluteValue(inputHead), maxAbsoluteValue(inputMid), maxAbsoluteValue(inputTail))

        #expect(
            headError < 0.0001 && midError < 0.0001 && tailError < 0.0001 && normMax > 0 && inputMax > 0,
            """
            layer15 norm output did not reach conv.in_proj input intact
            normStep=\(normStepIndex) convStep=\(convStepIndex)
            normKernel=\(normStep.pipeline.label ?? "unlabeled")
            convKernel=\(convStep.pipeline.label ?? "unlabeled")
            headError=\(headError) midError=\(midError) tailError=\(tailError)
            normMax=\(normMax) inputMax=\(inputMax)
            first4 norm-head=\(normHead.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 input-head=\(inputHead.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 norm-mid=\(normMid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 input-mid=\(inputMid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 norm-tail=\(normTail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 input-tail=\(inputTail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            """
        )
    }

    @Test("LFM layer15 norm output reaches conv in-proj input with device visibility probes")
    func lfmLayer15NormOutputReachesConvInProjInputWithDeviceVisibilityProbes() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let setup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }

        let collected = setup.collected
        var model = setup.model
        let plan = try #require(model.prefillPlan)
        let access = try #require(
            layerProjectionAccess(
                in: collected,
                layerPrefix: "model.layers.15.",
                tensorNameSuffix: ".conv.in_proj.weight"
            )
        )
        let projection = try #require(projectionKind(from: access.entry))
        let convStepIndex = try #require(
            matchingProjectionStepIndex(
                in: plan,
                entryIndex: access.entry.index,
                tensorName: access.tensorName,
                access: access.access,
                projection: projection
            )
        )
        let normStepIndex = convStepIndex - 1
        let normStep = plan.steps[normStepIndex]
        let convStep = plan.steps[convStepIndex]
        let normOutputDimension = try #require(uint32BindingValue(in: normStep, at: 3))
        let convInputDimension = try #require(uint32BindingValue(in: convStep, at: 3))

        let segmentCount = min(normOutputDimension, 32)
        let middleElementOffset = max((normOutputDimension / 2) - (segmentCount / 2), 0)
        let tailElementOffset = max(normOutputDimension - segmentCount, 0)
        let probe = try model.debugPrefillBindingProbes(
            tokens: promptTokens,
            stepIndex: convStepIndex,
            probes: [
                .init(
                    label: "normOutput-head",
                    stepIndex: normStepIndex,
                    bindingIndex: 2,
                    phase: .afterStep,
                    rowStride: normOutputDimension,
                    count: segmentCount
                ),
                .init(
                    label: "normOutput-mid",
                    stepIndex: normStepIndex,
                    bindingIndex: 2,
                    phase: .afterStep,
                    elementOffset: middleElementOffset,
                    rowStride: normOutputDimension,
                    count: segmentCount
                ),
                .init(
                    label: "normOutput-tail",
                    stepIndex: normStepIndex,
                    bindingIndex: 2,
                    phase: .afterStep,
                    elementOffset: tailElementOffset,
                    rowStride: normOutputDimension,
                    count: segmentCount
                ),
                .init(
                    label: "convInput-head",
                    bindingIndex: 0,
                    phase: .beforeStep,
                    rowStride: convInputDimension,
                    count: segmentCount
                ),
                .init(
                    label: "convInput-mid",
                    bindingIndex: 0,
                    phase: .beforeStep,
                    elementOffset: middleElementOffset,
                    rowStride: convInputDimension,
                    count: segmentCount
                ),
                .init(
                    label: "convInput-tail",
                    bindingIndex: 0,
                    phase: .beforeStep,
                    elementOffset: tailElementOffset,
                    rowStride: convInputDimension,
                    count: segmentCount
                ),
            ],
            visibilityOptions: .device
        )
        let normHead = try #require(probe["normOutput-head"])
        let normMid = try #require(probe["normOutput-mid"])
        let normTail = try #require(probe["normOutput-tail"])
        let inputHead = try #require(probe["convInput-head"])
        let inputMid = try #require(probe["convInput-mid"])
        let inputTail = try #require(probe["convInput-tail"])
        let headError = maxAbsoluteError(normHead, inputHead)
        let midError = maxAbsoluteError(normMid, inputMid)
        let tailError = maxAbsoluteError(normTail, inputTail)
        let normMax = max(maxAbsoluteValue(normHead), maxAbsoluteValue(normMid), maxAbsoluteValue(normTail))
        let inputMax = max(maxAbsoluteValue(inputHead), maxAbsoluteValue(inputMid), maxAbsoluteValue(inputTail))

        #expect(
            headError < 0.0001 && midError < 0.0001 && tailError < 0.0001 && normMax > 0 && inputMax > 0,
            """
            device-visibility probes still lost layer15 norm output before conv.in_proj
            normStep=\(normStepIndex) convStep=\(convStepIndex)
            normKernel=\(normStep.pipeline.label ?? "unlabeled")
            convKernel=\(convStep.pipeline.label ?? "unlabeled")
            headError=\(headError) midError=\(midError) tailError=\(tailError)
            normMax=\(normMax) inputMax=\(inputMax)
            first4 norm-head=\(normHead.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 input-head=\(inputHead.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 norm-mid=\(normMid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 input-mid=\(inputMid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 norm-tail=\(normTail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 input-tail=\(inputTail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            """
        )
    }

    @Test("LFM layer14 SwiGLU probes become non-zero when MPP is disabled")
    func lfmLayer14SwiGLUProbesBecomeNonZeroWhenMPPIsDisabled() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let setup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }

        var model = setup.model
        let plan = try #require(model.prefillPlan)
        let swigluStep = try #require(layer14SwiGLUStepIndex(in: plan))
        let swigluDimension = try #require(uint32BindingValue(in: plan.steps[swigluStep], at: 3))
        let gateProjectionStep = max(swigluStep - 2, 0)
        let gateProjection = plan.steps[gateProjectionStep]
        let naiveGridWidth = (swigluDimension + 1) / 2

        #expect(
            gateProjection.gridSize.width == naiveGridWidth,
            """
            expected naive prefill GEMM shape when MPP is disabled
            step=\(gateProjectionStep)
            kernel=\(gateProjection.pipeline.label ?? "unlabeled")
            grid=\(gateProjection.gridSize.width)x\(gateProjection.gridSize.height)x\(gateProjection.gridSize.depth)
            tg=\(gateProjection.threadgroupSize.width)x\(gateProjection.threadgroupSize.height)x\(gateProjection.threadgroupSize.depth)
            expectedGridWidth=\(naiveGridWidth)
            """
        )

        let segmentCount = min(swigluDimension, 32)
        let middleElementOffset = max((swigluDimension / 2) - (segmentCount / 2), 0)
        let tailElementOffset = max(swigluDimension - segmentCount, 0)
        let probe = try model.debugPrefillBindingProbes(
            tokens: promptTokens,
            stepIndex: swigluStep,
            probes: [
                .init(
                    label: "gate-head",
                    bindingIndex: 0,
                    phase: .beforeStep,
                    rowStride: swigluDimension,
                    count: segmentCount
                ),
                .init(
                    label: "gate-mid",
                    bindingIndex: 0,
                    phase: .beforeStep,
                    elementOffset: middleElementOffset,
                    rowStride: swigluDimension,
                    count: segmentCount
                ),
                .init(
                    label: "gate-tail",
                    bindingIndex: 0,
                    phase: .beforeStep,
                    elementOffset: tailElementOffset,
                    rowStride: swigluDimension,
                    count: segmentCount
                ),
                .init(
                    label: "up-head",
                    bindingIndex: 1,
                    phase: .beforeStep,
                    rowStride: swigluDimension,
                    count: segmentCount
                ),
                .init(
                    label: "up-mid",
                    bindingIndex: 1,
                    phase: .beforeStep,
                    elementOffset: middleElementOffset,
                    rowStride: swigluDimension,
                    count: segmentCount
                ),
                .init(
                    label: "up-tail",
                    bindingIndex: 1,
                    phase: .beforeStep,
                    elementOffset: tailElementOffset,
                    rowStride: swigluDimension,
                    count: segmentCount
                ),
                .init(
                    label: "output-head",
                    bindingIndex: 2,
                    phase: .afterStep,
                    rowStride: swigluDimension,
                    count: segmentCount
                ),
                .init(
                    label: "output-mid",
                    bindingIndex: 2,
                    phase: .afterStep,
                    elementOffset: middleElementOffset,
                    rowStride: swigluDimension,
                    count: segmentCount
                ),
                .init(
                    label: "output-tail",
                    bindingIndex: 2,
                    phase: .afterStep,
                    elementOffset: tailElementOffset,
                    rowStride: swigluDimension,
                    count: segmentCount
                ),
            ]
        )
        let gateHead = try #require(probe["gate-head"])
        let gateMid = try #require(probe["gate-mid"])
        let gateTail = try #require(probe["gate-tail"])
        let upHead = try #require(probe["up-head"])
        let upMid = try #require(probe["up-mid"])
        let upTail = try #require(probe["up-tail"])
        let outputHead = try #require(probe["output-head"])
        let outputMid = try #require(probe["output-mid"])
        let outputTail = try #require(probe["output-tail"])
        let gateMax = max(maxAbsoluteValue(gateHead), maxAbsoluteValue(gateMid), maxAbsoluteValue(gateTail))
        let upMax = max(maxAbsoluteValue(upHead), maxAbsoluteValue(upMid), maxAbsoluteValue(upTail))
        let outputMax = max(maxAbsoluteValue(outputHead), maxAbsoluteValue(outputMid), maxAbsoluteValue(outputTail))

        #expect(
            gateMax > 0 && upMax > 0 && outputMax > 0,
            """
            swiglu probes stayed zero with MPP disabled
            gateMax=\(gateMax) upMax=\(upMax) outputMax=\(outputMax)
            first4 gate-head=\(gateHead.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 gate-mid=\(gateMid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 gate-tail=\(gateTail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 up-head=\(upHead.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 up-mid=\(upMid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 up-tail=\(upTail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 output-head=\(outputHead.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 output-mid=\(outputMid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 output-tail=\(outputTail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            """
        )
    }

    @Test("LFM layer15 conv in-proj remains non-zero with MPP disabled")
    func lfmLayer15ConvInProjRemainsNonZeroWhenMPPIsDisabled() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let setup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }

        let collected = setup.collected
        var model = setup.model
        let plan = try #require(model.prefillPlan)
        let access = try #require(
            layerProjectionAccess(
                in: collected,
                layerPrefix: "model.layers.15.",
                tensorNameSuffix: ".conv.in_proj.weight"
            )
        )
        let projection = try #require(projectionKind(from: access.entry))
        let stepIndex = try #require(
            matchingProjectionStepIndex(
                in: plan,
                entryIndex: access.entry.index,
                tensorName: access.tensorName,
                access: access.access,
                projection: projection
            )
        )
        let step = plan.steps[stepIndex]
        let inputDimension = try #require(uint32BindingValue(in: step, at: 3))
        let outputDimension = try #require(uint32BindingValue(in: step, at: 4))
        let outputRowProbeIndices = Array(promptTokens.indices)
        let inputChunkSize = 128
        let segmentCount = min(outputDimension, 32)
        let middleElementOffset = max((outputDimension / 2) - (segmentCount / 2), 0)
        let tailElementOffset = max(outputDimension - segmentCount, 0)
        let probeRequests: [MetalInferenceModel.DebugPrefillBindingProbe] =
            chunkedVectorProbes(
                labelPrefix: "input-",
                bindingIndex: 0,
                phase: .beforeStep,
                rowStride: inputDimension,
                count: inputDimension,
                chunkSize: inputChunkSize
            ) + [
                .init(
                    label: "output",
                    bindingIndex: 2,
                    phase: .afterStep,
                    rowStride: outputDimension,
                    count: segmentCount
                ),
                .init(
                    label: "output-head32",
                    bindingIndex: 2,
                    phase: .afterStep,
                    rowStride: outputDimension,
                    count: segmentCount
                ),
                .init(
                    label: "output-mid32",
                    bindingIndex: 2,
                    phase: .afterStep,
                    elementOffset: middleElementOffset,
                    rowStride: outputDimension,
                    count: segmentCount
                ),
                .init(
                    label: "output-tail32",
                    bindingIndex: 2,
                    phase: .afterStep,
                    elementOffset: tailElementOffset,
                    rowStride: outputDimension,
                    count: segmentCount
                ),
            ] + rowProbes(
                labelPrefix: "output-row-",
                bindingIndex: 2,
                phase: .afterStep,
                rowStride: outputDimension,
                count: segmentCount,
                rowIndices: outputRowProbeIndices
            )
        let probe = try model.debugPrefillBindingProbes(
            tokens: promptTokens,
            stepIndex: stepIndex,
            probes: probeRequests
        )
        let input = reconstructChunkedVector(
            from: probe,
            prefix: "input-",
            count: inputDimension,
            chunkSize: inputChunkSize
        )
        let output = try #require(probe["output"])
        let outputHead32 = try #require(probe["output-head32"])
        let outputMid32 = try #require(probe["output-mid32"])
        let outputTail32 = try #require(probe["output-tail32"])
        let inputBinding = try #require(step.bindings.buffers.first(where: { $0.index == 0 }))
        let runtimeSequenceLength = plan.buffers.runtimeConstantBuffer.contents()
            .load(fromByteOffset: PrefillBufferSet.sequenceLengthOffset, as: UInt32.self)
        let outputBinding = try #require(step.bindings.buffers.first(where: { $0.index == 2 }))
        let outputRowProbeMaxima = rowMaxima(
            from: probe,
            prefix: "output-row-",
            rowIndices: outputRowProbeIndices
        )
        let isolatedOutput = try isolatedProjectionOutput(
            device: model.device,
            input: input,
            access: access.access,
            inputDimension: inputDimension,
            outputDimension: outputDimension
        )
        let isolatedAliasedOutput = try isolatedAliasedProjectionOutput(
            device: model.device,
            input: input,
            outputOffset: outputBinding.offset,
            access: access.access,
            inputDimension: inputDimension,
            outputDimension: outputDimension
        )
        let isolatedActualLayoutOutput = try isolatedActualLayoutProjectionOutput(
            device: model.device,
            input: input,
            inputOffset: inputBinding.offset,
            outputOffset: outputBinding.offset,
            access: access.access,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            sequenceLength: Int(runtimeSequenceLength),
            activeRowIndex: Int(runtimeSequenceLength) - 1,
            bindSequenceLengthFromRuntimeBuffer: true,
            pipeline: step.pipeline
        )
        let activeRowIndex = Int(runtimeSequenceLength) - 1
        let isolatedSequenceOutput = try isolatedProjectionOutput(
            device: model.device,
            input: input,
            access: access.access,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            sequenceLength: Int(runtimeSequenceLength),
            activeRowIndex: activeRowIndex
        )
        let isolatedSequenceRuntimeBoundOutput = try isolatedProjectionOutput(
            device: model.device,
            input: input,
            access: access.access,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            sequenceLength: Int(runtimeSequenceLength),
            activeRowIndex: activeRowIndex,
            bindSequenceLengthFromRuntimeBuffer: true
        )
        let planPipelineSequenceRuntimeBoundOutput = try isolatedProjectionOutput(
            device: model.device,
            input: input,
            access: access.access,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            sequenceLength: Int(runtimeSequenceLength),
            activeRowIndex: activeRowIndex,
            bindSequenceLengthFromRuntimeBuffer: true,
            pipeline: step.pipeline
        )
        let isolatedAliasedSequenceOutput = try isolatedAliasedProjectionOutput(
            device: model.device,
            input: input,
            outputOffset: outputBinding.offset,
            access: access.access,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            sequenceLength: Int(runtimeSequenceLength),
            activeRowIndex: activeRowIndex
        )
        let isolatedReference = try manualRowMajorDenseProjection(
            input: input,
            access: access.access,
            schemeIdentifier: access.schemeIdentifier,
            outputDimension: outputDimension,
            inputDimension: inputDimension
        )
        let referenceHead32 = segment(
            from: isolatedReference,
            elementOffset: 0,
            count: segmentCount
        )
        let referenceMid32 = segment(
            from: isolatedReference,
            elementOffset: middleElementOffset,
            count: segmentCount
        )
        let referenceTail32 = segment(
            from: isolatedReference,
            elementOffset: tailElementOffset,
            count: segmentCount
        )
        let inputMax = maxAbsoluteValue(input)
        let outputMax = maxAbsoluteValue(output)
        let outputHead32Max = maxAbsoluteValue(outputHead32)
        let outputMid32Max = maxAbsoluteValue(outputMid32)
        let outputTail32Max = maxAbsoluteValue(outputTail32)
        let outputHeadError = maxAbsoluteError(outputHead32, referenceHead32)
        let outputMidError = maxAbsoluteError(outputMid32, referenceMid32)
        let outputTailError = maxAbsoluteError(outputTail32, referenceTail32)
        let isolatedMax = maxAbsoluteValue(isolatedOutput)
        let isolatedError = maxAbsoluteError(isolatedOutput, isolatedReference)
        let isolatedAliasedMax = maxAbsoluteValue(isolatedAliasedOutput)
        let isolatedAliasedError = maxAbsoluteError(isolatedAliasedOutput, isolatedReference)
        let isolatedActualLayoutMax = maxAbsoluteValue(isolatedActualLayoutOutput)
        let isolatedActualLayoutError = maxAbsoluteError(isolatedActualLayoutOutput, isolatedReference)
        let isolatedSequenceMax = maxAbsoluteValue(isolatedSequenceOutput)
        let isolatedSequenceError = maxAbsoluteError(isolatedSequenceOutput, isolatedReference)
        let isolatedSequenceRuntimeBoundMax = maxAbsoluteValue(isolatedSequenceRuntimeBoundOutput)
        let isolatedSequenceRuntimeBoundError = maxAbsoluteError(
            isolatedSequenceRuntimeBoundOutput,
            isolatedReference
        )
        let planPipelineSequenceRuntimeBoundMax = maxAbsoluteValue(planPipelineSequenceRuntimeBoundOutput)
        let planPipelineSequenceRuntimeBoundError = maxAbsoluteError(
            planPipelineSequenceRuntimeBoundOutput,
            isolatedReference
        )
        let isolatedAliasedSequenceMax = maxAbsoluteValue(isolatedAliasedSequenceOutput)
        let isolatedAliasedSequenceError = maxAbsoluteError(
            isolatedAliasedSequenceOutput,
            isolatedReference
        )

        #expect(
            runtimeSequenceLength == UInt32(promptTokens.count),
            "runtime sequenceLength was not updated: \(runtimeSequenceLength)"
        )
        #expect(
            isolatedMax > 0 && isolatedError < 0.05,
            """
            isolated conv.in_proj kernel did not match CPU reference
            isolatedMax=\(isolatedMax) isolatedError=\(isolatedError)
            first4 ref=\(isolatedReference.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 isolated=\(isolatedOutput.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            """
        )
        #expect(
            isolatedAliasedMax > 0 && isolatedAliasedError < 0.05,
            """
            isolated aliased conv.in_proj kernel did not match CPU reference
            outputOffset=\(outputBinding.offset)
            isolatedAliasedMax=\(isolatedAliasedMax) isolatedAliasedError=\(isolatedAliasedError)
            first4 ref=\(isolatedReference.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 aliased=\(isolatedAliasedOutput.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            """
        )
        #expect(
            isolatedSequenceMax > 0 && isolatedSequenceError < 0.05,
            """
            isolated sequence conv.in_proj kernel did not match CPU reference
            sequenceLength=\(runtimeSequenceLength) activeRowIndex=\(activeRowIndex)
            isolatedSequenceMax=\(isolatedSequenceMax) isolatedSequenceError=\(isolatedSequenceError)
            first4 ref=\(isolatedReference.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 isolated-seq=\(isolatedSequenceOutput.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            """
        )
        #expect(
            isolatedSequenceRuntimeBoundMax > 0 && isolatedSequenceRuntimeBoundError < 0.05,
            """
            isolated sequence conv.in_proj kernel with runtime-bound sequenceLength did not match CPU reference
            sequenceLength=\(runtimeSequenceLength) activeRowIndex=\(activeRowIndex)
            isolatedSequenceRuntimeBoundMax=\(isolatedSequenceRuntimeBoundMax)
            isolatedSequenceRuntimeBoundError=\(isolatedSequenceRuntimeBoundError)
            first4 ref=\(isolatedReference.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 isolated-runtime=\(isolatedSequenceRuntimeBoundOutput.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            """
        )
        #expect(
            planPipelineSequenceRuntimeBoundMax > 0 && planPipelineSequenceRuntimeBoundError < 0.05,
            """
            plan pipeline failed even with isolated temp buffers and runtime-bound sequenceLength
            planPipelineSequenceRuntimeBoundMax=\(planPipelineSequenceRuntimeBoundMax)
            planPipelineSequenceRuntimeBoundError=\(planPipelineSequenceRuntimeBoundError)
            first4 ref=\(isolatedReference.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 plan-pipeline=\(planPipelineSequenceRuntimeBoundOutput.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            """
        )
        #expect(
            isolatedAliasedSequenceMax > 0 && isolatedAliasedSequenceError < 0.05,
            """
            isolated aliased sequence conv.in_proj kernel did not match CPU reference
            sequenceLength=\(runtimeSequenceLength) activeRowIndex=\(activeRowIndex) outputOffset=\(outputBinding.offset)
            isolatedAliasedSequenceMax=\(isolatedAliasedSequenceMax)
            isolatedAliasedSequenceError=\(isolatedAliasedSequenceError)
            first4 ref=\(isolatedReference.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 isolated-aliased-seq=\(isolatedAliasedSequenceOutput.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            """
        )

        #expect(
            inputMax > 0
                && outputMax > 0
                && outputHead32Max > 0
                && outputMid32Max > 0
                && outputTail32Max > 0
                && outputHeadError < 0.05
                && outputMidError < 0.05
                && outputTailError < 0.05,
            """
            layer15 conv.in_proj also collapsed with MPP disabled
            step=\(stepIndex)
            kernel=\(step.pipeline.label ?? "unlabeled")
            grid=\(step.gridSize.width)x\(step.gridSize.height)x\(step.gridSize.depth)
            tg=\(step.threadgroupSize.width)x\(step.threadgroupSize.height)x\(step.threadgroupSize.depth)
            inputMax=\(inputMax) outputMax=\(outputMax)
            outputHead32Max=\(outputHead32Max)
            outputMid32Max=\(outputMid32Max)
            outputTail32Max=\(outputTail32Max)
            outputHeadError=\(outputHeadError)
            outputMidError=\(outputMidError)
            outputTailError=\(outputTailError)
            isolatedMax=\(isolatedMax) isolatedError=\(isolatedError)
            isolatedAliasedMax=\(isolatedAliasedMax) isolatedAliasedError=\(isolatedAliasedError)
            isolatedSequenceMax=\(isolatedSequenceMax) isolatedSequenceError=\(isolatedSequenceError)
            isolatedSequenceRuntimeBoundMax=\(isolatedSequenceRuntimeBoundMax) isolatedSequenceRuntimeBoundError=\(isolatedSequenceRuntimeBoundError)
            planPipelineSequenceRuntimeBoundMax=\(planPipelineSequenceRuntimeBoundMax)
            planPipelineSequenceRuntimeBoundError=\(planPipelineSequenceRuntimeBoundError)
            isolatedAliasedSequenceMax=\(isolatedAliasedSequenceMax) isolatedAliasedSequenceError=\(isolatedAliasedSequenceError)
            isolatedActualLayoutMax=\(isolatedActualLayoutMax) isolatedActualLayoutError=\(isolatedActualLayoutError)
            runtimeSequenceLength=\(runtimeSequenceLength)
            inputOffset=\(inputBinding.offset) outputOffset=\(outputBinding.offset)
            sameIOBuffer=\(inputBinding.buffer === outputBinding.buffer)
            outputRowMaxima={\(formatRowMaxima(outputRowProbeMaxima))}
            first4 input=\(input.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 output=\(output.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 output-head32=\(outputHead32.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 output-mid32=\(outputMid32.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 output-tail32=\(outputTail32.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 ref-head32=\(referenceHead32.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 ref-mid32=\(referenceMid32.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 ref-tail32=\(referenceTail32.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 isolated=\(isolatedOutput.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 aliased=\(isolatedAliasedOutput.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 actual-layout=\(isolatedActualLayoutOutput.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 isolated-seq=\(isolatedSequenceOutput.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 isolated-runtime=\(isolatedSequenceRuntimeBoundOutput.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 plan-pipeline=\(planPipelineSequenceRuntimeBoundOutput.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first4 isolated-aliased-seq=\(isolatedAliasedSequenceOutput.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            """
        )
    }

    @Test("LFM layer15 conv in-proj segment probes stay stable across fresh model instances")
    func lfmLayer15ConvInProjSegmentProbesStayStableAcrossFreshModelInstances() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let firstSetup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }
        let secondSetup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }

        func resolvedProbe(
            from setup: BenchmarkSupport.SetupWithCollectedPrefillEntries,
            includeBeforeStepInputProbe: Bool
        ) throws -> (head: [Float], mid: [Float], tail: [Float], maximum: Float) {
            var model = setup.model
            let plan = try #require(model.prefillPlan)
            let access = try #require(
                layerProjectionAccess(
                    in: setup.collected,
                    layerPrefix: "model.layers.15.",
                    tensorNameSuffix: ".conv.in_proj.weight"
                )
            )
            let projection = try #require(projectionKind(from: access.entry))
            let stepIndex = try #require(
                matchingProjectionStepIndex(
                    in: plan,
                    entryIndex: access.entry.index,
                    tensorName: access.tensorName,
                    access: access.access,
                    projection: projection
                )
            )
            let inputDimension = try #require(uint32BindingValue(in: plan.steps[stepIndex], at: 3))
            let outputDimension = try #require(uint32BindingValue(in: plan.steps[stepIndex], at: 4))
            return try convInProjSegmentProbe(
                model: &model,
                tokens: promptTokens,
                stepIndex: stepIndex,
                inputDimension: inputDimension,
                outputDimension: outputDimension,
                includeBeforeStepInputProbe: includeBeforeStepInputProbe
            )
        }

        let first = try resolvedProbe(from: firstSetup, includeBeforeStepInputProbe: false)
        let second = try resolvedProbe(from: secondSetup, includeBeforeStepInputProbe: false)
        let firstFenced = try resolvedProbe(from: firstSetup, includeBeforeStepInputProbe: true)
        let secondFenced = try resolvedProbe(from: secondSetup, includeBeforeStepInputProbe: true)
        let headError = maxAbsoluteError(firstFenced.head, secondFenced.head)
        let midError = maxAbsoluteError(firstFenced.mid, secondFenced.mid)
        let tailError = maxAbsoluteError(firstFenced.tail, secondFenced.tail)
        let unfencedHeadError = maxAbsoluteError(first.head, second.head)
        let unfencedMidError = maxAbsoluteError(first.mid, second.mid)
        let unfencedTailError = maxAbsoluteError(first.tail, second.tail)

        #expect(
            firstFenced.maximum > 0
                && secondFenced.maximum > 0
                && headError < 0.0001
                && midError < 0.0001
                && tailError < 0.0001,
            """
            fresh model instances still disagree even after adding a before-step input probe fence
            unfencedMax=[\(first.maximum), \(second.maximum)]
            unfencedErrors=[\(unfencedHeadError), \(unfencedMidError), \(unfencedTailError)]
            fencedMax=[\(firstFenced.maximum), \(secondFenced.maximum)]
            fencedErrors=[\(headError), \(midError), \(tailError)]
            unfenced first head=\(first.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            unfenced second head=\(second.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            fenced first head=\(firstFenced.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            fenced second head=\(secondFenced.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            unfenced first mid=\(first.mid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            unfenced second mid=\(second.mid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            fenced first mid=\(firstFenced.mid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            fenced second mid=\(secondFenced.mid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            unfenced first tail=\(first.tail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            unfenced second tail=\(second.tail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            fenced first tail=\(firstFenced.tail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            fenced second tail=\(secondFenced.tail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            """
        )
    }

    @Test("LFM layer15 norm output segment probes stay stable across fresh model instances")
    func lfmLayer15NormOutputSegmentProbesStayStableAcrossFreshModelInstances() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let firstSetup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }
        let secondSetup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }

        func resolvedNormProbe(
            from setup: BenchmarkSupport.SetupWithCollectedPrefillEntries,
            includeBeforeStepInputProbe: Bool
        ) throws -> (head: [Float], mid: [Float], tail: [Float], maximum: Float) {
            var model = setup.model
            let plan = try #require(model.prefillPlan)
            let access = try #require(
                layerProjectionAccess(
                    in: setup.collected,
                    layerPrefix: "model.layers.15.",
                    tensorNameSuffix: ".conv.in_proj.weight"
                )
            )
            let projection = try #require(projectionKind(from: access.entry))
            let convStepIndex = try #require(
                matchingProjectionStepIndex(
                    in: plan,
                    entryIndex: access.entry.index,
                    tensorName: access.tensorName,
                    access: access.access,
                    projection: projection
                )
            )
            let normStepIndex = convStepIndex - 1
            let normStep = plan.steps[normStepIndex]
            let inputDimension = try #require(uint32BindingValue(in: normStep, at: 3))
            let outputDimension = try #require(uint32BindingValue(in: normStep, at: 3))
            return try bindingSegmentProbe(
                model: &model,
                tokens: promptTokens,
                stepIndex: normStepIndex,
                bindingIndex: 2,
                inputBindingIndex: includeBeforeStepInputProbe ? 0 : nil,
                inputDimension: inputDimension,
                outputDimension: outputDimension
            )
        }

        let first = try resolvedNormProbe(from: firstSetup, includeBeforeStepInputProbe: false)
        let second = try resolvedNormProbe(from: secondSetup, includeBeforeStepInputProbe: false)
        let firstFenced = try resolvedNormProbe(from: firstSetup, includeBeforeStepInputProbe: true)
        let secondFenced = try resolvedNormProbe(from: secondSetup, includeBeforeStepInputProbe: true)

        let unfencedHeadError = maxAbsoluteError(first.head, second.head)
        let unfencedMidError = maxAbsoluteError(first.mid, second.mid)
        let unfencedTailError = maxAbsoluteError(first.tail, second.tail)
        let fencedHeadError = maxAbsoluteError(firstFenced.head, secondFenced.head)
        let fencedMidError = maxAbsoluteError(firstFenced.mid, secondFenced.mid)
        let fencedTailError = maxAbsoluteError(firstFenced.tail, secondFenced.tail)

        #expect(
            firstFenced.maximum > 0
                && secondFenced.maximum > 0
                && fencedHeadError < 0.0001
                && fencedMidError < 0.0001
                && fencedTailError < 0.0001,
            """
            fresh model instances still disagree on layer15 norm output segments
            unfencedMax=[\(first.maximum), \(second.maximum)]
            unfencedErrors=[\(unfencedHeadError), \(unfencedMidError), \(unfencedTailError)]
            fencedMax=[\(firstFenced.maximum), \(secondFenced.maximum)]
            fencedErrors=[\(fencedHeadError), \(fencedMidError), \(fencedTailError)]
            unfenced first head=\(first.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            unfenced second head=\(second.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            fenced first head=\(firstFenced.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            fenced second head=\(secondFenced.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            unfenced first mid=\(first.mid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            unfenced second mid=\(second.mid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            fenced first mid=\(firstFenced.mid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            fenced second mid=\(secondFenced.mid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            unfenced first tail=\(first.tail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            unfenced second tail=\(second.tail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            fenced first tail=\(firstFenced.tail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            fenced second tail=\(secondFenced.tail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            """
        )
    }

    @Test("LFM layer15 norm input segment probes stay stable across fresh model instances")
    func lfmLayer15NormInputSegmentProbesStayStableAcrossFreshModelInstances() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let firstSetup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }
        let secondSetup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }

        func resolvedNormInputProbe(
            from setup: BenchmarkSupport.SetupWithCollectedPrefillEntries
        ) throws -> (head: [Float], mid: [Float], tail: [Float], maximum: Float) {
            var model = setup.model
            let plan = try #require(model.prefillPlan)
            let access = try #require(
                layerProjectionAccess(
                    in: setup.collected,
                    layerPrefix: "model.layers.15.",
                    tensorNameSuffix: ".conv.in_proj.weight"
                )
            )
            let projection = try #require(projectionKind(from: access.entry))
            let convStepIndex = try #require(
                matchingProjectionStepIndex(
                    in: plan,
                    entryIndex: access.entry.index,
                    tensorName: access.tensorName,
                    access: access.access,
                    projection: projection
                )
            )
            let normStepIndex = convStepIndex - 1
            let normStep = plan.steps[normStepIndex]
            let inputDimension = try #require(uint32BindingValue(in: normStep, at: 3))
            return try bindingSegmentProbe(
                model: &model,
                tokens: promptTokens,
                stepIndex: normStepIndex,
                bindingIndex: 0,
                phase: .beforeStep,
                outputDimension: inputDimension
            )
        }

        let first = try resolvedNormInputProbe(from: firstSetup)
        let second = try resolvedNormInputProbe(from: secondSetup)
        let headError = maxAbsoluteError(first.head, second.head)
        let midError = maxAbsoluteError(first.mid, second.mid)
        let tailError = maxAbsoluteError(first.tail, second.tail)

        #expect(
            first.maximum > 0
                && second.maximum > 0
                && headError < 0.0001
                && midError < 0.0001
                && tailError < 0.0001,
            """
            fresh model instances disagree on layer15 norm input segments
            maxima=[\(first.maximum), \(second.maximum)]
            errors=[\(headError), \(midError), \(tailError)]
            first head=\(first.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            second head=\(second.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first mid=\(first.mid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            second mid=\(second.mid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first tail=\(first.tail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            second tail=\(second.tail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            """
        )
    }

    @Test("LFM layer15 norm producer output segment probes stay stable across fresh model instances")
    func lfmLayer15NormProducerOutputSegmentProbesStayStableAcrossFreshModelInstances() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let firstSetup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }
        let secondSetup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }

        func resolvedProducerProbe(
            from setup: BenchmarkSupport.SetupWithCollectedPrefillEntries
        ) throws -> (
            probe: (head: [Float], mid: [Float], tail: [Float], maximum: Float),
            producerStepIndex: Int,
            writerBindingIndex: Int,
            nearby: [String]
        ) {
            var model = setup.model
            let context = try resolvedLayer15ConvInProjContext(from: setup)
            let producerStepIndex = try #require(
                lastWriterStepIndex(
                    in: context.plan.steps,
                    before: context.normStepIndex,
                    target: context.normInputBinding
                )
            )
            let producerStep = context.plan.steps[producerStepIndex]
            let writerBindingIndex = try #require(
                matchingWriteBindingIndex(
                    in: producerStep,
                    target: context.normInputBinding
                )
            )
            let probe = try bindingSegmentProbe(
                model: &model,
                tokens: promptTokens,
                stepIndex: producerStepIndex,
                bindingIndex: writerBindingIndex,
                outputDimension: context.normInputDimension
            )
            return (
                probe,
                producerStepIndex,
                writerBindingIndex,
                nearbyStepSummaries(
                    in: context.plan.steps,
                    center: context.normStepIndex,
                    target: context.normInputBinding
                )
            )
        }

        let first = try resolvedProducerProbe(from: firstSetup)
        let second = try resolvedProducerProbe(from: secondSetup)
        let headError = maxAbsoluteError(first.probe.head, second.probe.head)
        let midError = maxAbsoluteError(first.probe.mid, second.probe.mid)
        let tailError = maxAbsoluteError(first.probe.tail, second.probe.tail)

        #expect(
            first.probe.maximum > 0
                && second.probe.maximum > 0
                && headError < 0.0001
                && midError < 0.0001
                && tailError < 0.0001,
            """
            fresh model instances disagree on the producer that writes layer15 norm input
            producerStepIndices=[\(first.producerStepIndex), \(second.producerStepIndex)]
            writerBindingIndices=[\(first.writerBindingIndex), \(second.writerBindingIndex)]
            maxima=[\(first.probe.maximum), \(second.probe.maximum)]
            errors=[\(headError), \(midError), \(tailError)]
            first head=\(first.probe.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            second head=\(second.probe.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first mid=\(first.probe.mid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            second mid=\(second.probe.mid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first tail=\(first.probe.tail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            second tail=\(second.probe.tail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            nearby first:
            \(first.nearby.joined(separator: "\n"))
            nearby second:
            \(second.nearby.joined(separator: "\n"))
            """
        )
    }

    @Test("LFM layer15 residual-add inputs stay stable across fresh model instances")
    func lfmLayer15ResidualAddInputsStayStableAcrossFreshModelInstances() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let firstSetup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }
        let secondSetup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }

        func resolvedResidualInputs(
            from setup: BenchmarkSupport.SetupWithCollectedPrefillEntries
        ) throws -> (
            hiddenProbe: (head: [Float], mid: [Float], tail: [Float], maximum: Float),
            residualProbe: (head: [Float], mid: [Float], tail: [Float], maximum: Float),
            producerStepIndex: Int,
            hiddenSourceStepIndex: Int,
            residualSourceStepIndex: Int,
            hiddenNearby: [String],
            residualNearby: [String]
        ) {
            var model = setup.model
            let context = try resolvedLayer15ConvInProjContext(from: setup)
            let producerStepIndex = try #require(
                lastWriterStepIndex(
                    in: context.plan.steps,
                    before: context.normStepIndex,
                    target: context.normInputBinding
                )
            )
            let producerStep = context.plan.steps[producerStepIndex]
            let hiddenInputBinding = try #require(
                producerStep.bindings.buffers.first(where: { $0.index == 0 })
            )
            let residualInputBinding = try #require(
                producerStep.bindings.buffers.first(where: { $0.index == 1 })
            )
            let hiddenSourceStepIndex = try #require(
                lastWriterStepIndex(
                    in: context.plan.steps,
                    before: producerStepIndex,
                    target: hiddenInputBinding
                )
            )
            let residualSourceStepIndex = try #require(
                lastWriterStepIndex(
                    in: context.plan.steps,
                    before: producerStepIndex,
                    target: residualInputBinding
                )
            )
            let hiddenProbe = try bindingSegmentProbe(
                model: &model,
                tokens: promptTokens,
                stepIndex: producerStepIndex,
                bindingIndex: 0,
                phase: .beforeStep,
                outputDimension: context.normInputDimension
            )
            let residualProbe = try bindingSegmentProbe(
                model: &model,
                tokens: promptTokens,
                stepIndex: producerStepIndex,
                bindingIndex: 1,
                phase: .beforeStep,
                outputDimension: context.normInputDimension
            )
            return (
                hiddenProbe,
                residualProbe,
                producerStepIndex,
                hiddenSourceStepIndex,
                residualSourceStepIndex,
                nearbyStepSummaries(
                    in: context.plan.steps,
                    center: producerStepIndex,
                    target: hiddenInputBinding
                ),
                nearbyStepSummaries(
                    in: context.plan.steps,
                    center: residualSourceStepIndex,
                    target: residualInputBinding
                )
            )
        }

        let first = try resolvedResidualInputs(from: firstSetup)
        let second = try resolvedResidualInputs(from: secondSetup)
        let hiddenHeadError = maxAbsoluteError(first.hiddenProbe.head, second.hiddenProbe.head)
        let hiddenMidError = maxAbsoluteError(first.hiddenProbe.mid, second.hiddenProbe.mid)
        let hiddenTailError = maxAbsoluteError(first.hiddenProbe.tail, second.hiddenProbe.tail)
        let residualHeadError = maxAbsoluteError(first.residualProbe.head, second.residualProbe.head)
        let residualMidError = maxAbsoluteError(first.residualProbe.mid, second.residualProbe.mid)
        let residualTailError = maxAbsoluteError(first.residualProbe.tail, second.residualProbe.tail)

        #expect(
            first.hiddenProbe.maximum > 0
                && second.hiddenProbe.maximum > 0
                && first.residualProbe.maximum > 0
                && second.residualProbe.maximum > 0
                && hiddenHeadError < 0.0001
                && hiddenMidError < 0.0001
                && hiddenTailError < 0.0001
                && residualHeadError < 0.0001
                && residualMidError < 0.0001
                && residualTailError < 0.0001,
            """
            fresh model instances disagree on residual_add inputs before layer15 norm
            producerStepIndices=[\(first.producerStepIndex), \(second.producerStepIndex)]
            hiddenSourceStepIndices=[\(first.hiddenSourceStepIndex), \(second.hiddenSourceStepIndex)]
            residualSourceStepIndices=[\(first.residualSourceStepIndex), \(second.residualSourceStepIndex)]
            hiddenMaxima=[\(first.hiddenProbe.maximum), \(second.hiddenProbe.maximum)]
            residualMaxima=[\(first.residualProbe.maximum), \(second.residualProbe.maximum)]
            hiddenErrors=[\(hiddenHeadError), \(hiddenMidError), \(hiddenTailError)]
            residualErrors=[\(residualHeadError), \(residualMidError), \(residualTailError)]
            first hidden head=\(first.hiddenProbe.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            second hidden head=\(second.hiddenProbe.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first residual head=\(first.residualProbe.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            second residual head=\(second.residualProbe.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            hidden nearby first:
            \(first.hiddenNearby.joined(separator: "\n"))
            hidden nearby second:
            \(second.hiddenNearby.joined(separator: "\n"))
            residual nearby first:
            \(first.residualNearby.joined(separator: "\n"))
            residual nearby second:
            \(second.residualNearby.joined(separator: "\n"))
            """
        )
    }

    @Test("LFM layer14 post-attention residual-add inputs stay stable across fresh model instances")
    func lfmLayer14PostAttentionResidualAddInputsStayStableAcrossFreshModelInstances() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let firstSetup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }
        let secondSetup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }

        func resolvedLayer14ResidualAddInputs(
            from setup: BenchmarkSupport.SetupWithCollectedPrefillEntries
        ) throws -> (
            attentionProbe: (head: [Float], mid: [Float], tail: [Float], maximum: Float),
            skipProbe: (head: [Float], mid: [Float], tail: [Float], maximum: Float),
            residualAddStepIndex: Int,
            attentionSourceStepIndex: Int,
            skipSourceStepIndex: Int,
            nearby: [String],
            skipNearby: [String]
        ) {
            var model = setup.model
            let context = try resolvedLayer15ConvInProjContext(from: setup)
            let layer15ResidualAddStepIndex = try #require(
                lastWriterStepIndex(
                    in: context.plan.steps,
                    before: context.normStepIndex,
                    target: context.normInputBinding
                )
            )
            let layer15ResidualAddStep = context.plan.steps[layer15ResidualAddStepIndex]
            let layer15ResidualInputBinding = try #require(
                layer15ResidualAddStep.bindings.buffers.first(where: { $0.index == 1 })
            )
            let residualCopyStepIndex = try #require(
                lastWriterStepIndex(
                    in: context.plan.steps,
                    before: layer15ResidualAddStepIndex,
                    target: layer15ResidualInputBinding
                )
            )
            let residualCopyStep = context.plan.steps[residualCopyStepIndex]
            let copyInputBinding = try #require(
                residualCopyStep.bindings.buffers.first(where: { $0.index == 0 })
            )
            let residualAddStepIndex = try #require(
                lastWriterStepIndex(
                    in: context.plan.steps,
                    before: residualCopyStepIndex,
                    target: copyInputBinding
                )
            )
            let residualAddStep = context.plan.steps[residualAddStepIndex]
            let attentionInputBinding = try #require(
                residualAddStep.bindings.buffers.first(where: { $0.index == 0 })
            )
            let skipInputBinding = try #require(
                residualAddStep.bindings.buffers.first(where: { $0.index == 1 })
            )
            let attentionSourceStepIndex = try #require(
                lastWriterStepIndex(
                    in: context.plan.steps,
                    before: residualAddStepIndex,
                    target: attentionInputBinding
                )
            )
            let skipSourceStepIndex = try #require(
                lastWriterStepIndex(
                    in: context.plan.steps,
                    before: residualAddStepIndex,
                    target: skipInputBinding
                )
            )
            let attentionProbe = try bindingSegmentProbe(
                model: &model,
                tokens: promptTokens,
                stepIndex: residualAddStepIndex,
                bindingIndex: 0,
                phase: .beforeStep,
                outputDimension: context.normInputDimension
            )
            let skipProbe = try bindingSegmentProbe(
                model: &model,
                tokens: promptTokens,
                stepIndex: residualAddStepIndex,
                bindingIndex: 1,
                phase: .beforeStep,
                outputDimension: context.normInputDimension
            )
            return (
                attentionProbe,
                skipProbe,
                residualAddStepIndex,
                attentionSourceStepIndex,
                skipSourceStepIndex,
                nearbyStepSummaries(
                    in: context.plan.steps,
                    center: residualAddStepIndex,
                    target: skipInputBinding
                ),
                nearbyStepSummaries(
                    in: context.plan.steps,
                    center: skipSourceStepIndex,
                    target: skipInputBinding
                )
            )
        }

        let first = try resolvedLayer14ResidualAddInputs(from: firstSetup)
        let second = try resolvedLayer14ResidualAddInputs(from: secondSetup)
        let attentionHeadError = maxAbsoluteError(first.attentionProbe.head, second.attentionProbe.head)
        let attentionMidError = maxAbsoluteError(first.attentionProbe.mid, second.attentionProbe.mid)
        let attentionTailError = maxAbsoluteError(first.attentionProbe.tail, second.attentionProbe.tail)
        let skipHeadError = maxAbsoluteError(first.skipProbe.head, second.skipProbe.head)
        let skipMidError = maxAbsoluteError(first.skipProbe.mid, second.skipProbe.mid)
        let skipTailError = maxAbsoluteError(first.skipProbe.tail, second.skipProbe.tail)

        #expect(
            first.attentionProbe.maximum > 0
                && second.attentionProbe.maximum > 0
                && first.skipProbe.maximum > 0
                && second.skipProbe.maximum > 0
                && attentionHeadError < 0.0001
                && attentionMidError < 0.0001
                && attentionTailError < 0.0001
                && skipHeadError < 0.0001
                && skipMidError < 0.0001
                && skipTailError < 0.0001,
            """
            fresh model instances disagree on layer14 post-attention residual_add inputs
            residualAddStepIndices=[\(first.residualAddStepIndex), \(second.residualAddStepIndex)]
            attentionSourceStepIndices=[\(first.attentionSourceStepIndex), \(second.attentionSourceStepIndex)]
            skipSourceStepIndices=[\(first.skipSourceStepIndex), \(second.skipSourceStepIndex)]
            attentionMaxima=[\(first.attentionProbe.maximum), \(second.attentionProbe.maximum)]
            skipMaxima=[\(first.skipProbe.maximum), \(second.skipProbe.maximum)]
            attentionErrors=[\(attentionHeadError), \(attentionMidError), \(attentionTailError)]
            skipErrors=[\(skipHeadError), \(skipMidError), \(skipTailError)]
            first attention head=\(first.attentionProbe.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            second attention head=\(second.attentionProbe.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            first skip head=\(first.skipProbe.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            second skip head=\(second.skipProbe.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            nearby first:
            \(first.nearby.joined(separator: "\n"))
            nearby second:
            \(second.nearby.joined(separator: "\n"))
            skip nearby first:
            \(first.skipNearby.joined(separator: "\n"))
            skip nearby second:
            \(second.skipNearby.joined(separator: "\n"))
            """
        )
    }

    @Test("LFM layer13 residual-add output feeding layer14 stays stable across fresh model instances")
    func lfmLayer13ResidualAddOutputFeedingLayer14StaysStableAcrossFreshModelInstances() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let firstSetup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }
        let secondSetup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }

        func resolvedBoundaryProbe(
            from setup: BenchmarkSupport.SetupWithCollectedPrefillEntries
        ) throws -> (
            probe: (head: [Float], mid: [Float], tail: [Float], maximum: Float),
            boundaryStepIndex: Int,
            nearby: [String]
        ) {
            var model = setup.model
            let context = try resolvedLayer15ConvInProjContext(from: setup)
            let layer15ResidualAddStepIndex = try #require(
                lastWriterStepIndex(
                    in: context.plan.steps,
                    before: context.normStepIndex,
                    target: context.normInputBinding
                )
            )
            let layer15ResidualAddStep = context.plan.steps[layer15ResidualAddStepIndex]
            let layer15ResidualInputBinding = try #require(
                layer15ResidualAddStep.bindings.buffers.first(where: { $0.index == 1 })
            )
            let residualCopyStepIndex = try #require(
                lastWriterStepIndex(
                    in: context.plan.steps,
                    before: layer15ResidualAddStepIndex,
                    target: layer15ResidualInputBinding
                )
            )
            let residualCopyStep = context.plan.steps[residualCopyStepIndex]
            let copyInputBinding = try #require(
                residualCopyStep.bindings.buffers.first(where: { $0.index == 0 })
            )
            let boundaryStepIndex = try #require(
                lastWriterStepIndex(
                    in: context.plan.steps,
                    before: residualCopyStepIndex,
                    target: copyInputBinding
                )
            )
            let probe = try bindingSegmentProbe(
                model: &model,
                tokens: promptTokens,
                stepIndex: boundaryStepIndex,
                bindingIndex: 2,
                outputDimension: context.normInputDimension
            )
            return (
                probe,
                boundaryStepIndex,
                nearbyStepSummaries(
                    in: context.plan.steps,
                    center: boundaryStepIndex,
                    target: copyInputBinding
                )
            )
        }

        let first = try resolvedBoundaryProbe(from: firstSetup)
        let second = try resolvedBoundaryProbe(from: secondSetup)
        let headError = maxAbsoluteError(first.probe.head, second.probe.head)
        let midError = maxAbsoluteError(first.probe.mid, second.probe.mid)
        let tailError = maxAbsoluteError(first.probe.tail, second.probe.tail)

        #expect(
            first.probe.maximum > 0
                && second.probe.maximum > 0
                && headError < 0.0001
                && midError < 0.0001
                && tailError < 0.0001,
            """
            fresh model instances disagree on the common layer13 residual_add output that feeds layer14
            boundaryStepIndices=[\(first.boundaryStepIndex), \(second.boundaryStepIndex)]
            maxima=[\(first.probe.maximum), \(second.probe.maximum)]
            errors=[\(headError), \(midError), \(tailError)]
            first head=\(first.probe.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            second head=\(second.probe.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            nearby first:
            \(first.nearby.joined(separator: "\n"))
            nearby second:
            \(second.nearby.joined(separator: "\n"))
            """
        )
    }

    @Test("LFM earliest residual-boundary divergence is identified across fresh model instances")
    func lfmEarliestResidualBoundaryDivergenceIsIdentifiedAcrossFreshModelInstances() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let firstSetup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }
        let secondSetup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }

        let context = try resolvedLayer15ConvInProjContext(from: firstSetup)
        let layer15ResidualAddStepIndex = try #require(
            lastWriterStepIndex(
                in: context.plan.steps,
                before: context.normStepIndex,
                target: context.normInputBinding
            )
        )
        let chain = residualBoundaryChain(
            in: context.plan,
            startingAt: layer15ResidualAddStepIndex
        )
        var firstModel = firstSetup.model
        var secondModel = secondSetup.model

        var earliest: (
            stepIndex: Int,
            layerIndex: Int?,
            kernel: String,
            tensor: String?,
            firstMax: Float,
            secondMax: Float,
            headError: Float,
            midError: Float,
            tailError: Float,
            nearby: [String]
        )?

        for stepIndex in chain.reversed() {
            let step = context.plan.steps[stepIndex]
            let first = try bindingSegmentProbe(
                model: &firstModel,
                tokens: promptTokens,
                stepIndex: stepIndex,
                bindingIndex: 2,
                outputDimension: context.normInputDimension
            )
            let second = try bindingSegmentProbe(
                model: &secondModel,
                tokens: promptTokens,
                stepIndex: stepIndex,
                bindingIndex: 2,
                outputDimension: context.normInputDimension
            )
            let headError = maxAbsoluteError(first.head, second.head)
            let midError = maxAbsoluteError(first.mid, second.mid)
            let tailError = maxAbsoluteError(first.tail, second.tail)
            if headError >= 0.0001 || midError >= 0.0001 || tailError >= 0.0001 {
                let outputBinding = try #require(
                    step.bindings.buffers.first(where: { $0.index == 2 })
                )
                earliest = (
                    stepIndex,
                    step.metadata.layerIndex,
                    step.pipeline.label ?? "unlabeled",
                    step.metadata.weightTensorName,
                    first.maximum,
                    second.maximum,
                    headError,
                    midError,
                    tailError,
                    nearbyStepSummaries(
                        in: context.plan.steps,
                        center: stepIndex,
                        target: outputBinding
                    )
                )
                break
            }
        }

        #expect(
            earliest == nil,
            """
            earliest residual-boundary divergence detected
            chain=\(chain.map(String.init).joined(separator: " -> "))
            earliestStep=\(earliest.map { String($0.stepIndex) } ?? "none")
            earliestLayer=\(earliest?.layerIndex.map(String.init) ?? "nil")
            kernel=\(earliest?.kernel ?? "nil")
            tensor=\(earliest?.tensor ?? "nil")
            maxima=[\(earliest?.firstMax ?? 0), \(earliest?.secondMax ?? 0)]
            errors=[\(earliest?.headError ?? 0), \(earliest?.midError ?? 0), \(earliest?.tailError ?? 0)]
            nearby:
            \(earliest?.nearby.joined(separator: "\n") ?? "")
            """
        )
    }

    @Test("LFM earliest divergent residual-boundary inputs identify the unstable branch across fresh model instances")
    func lfmEarliestDivergentResidualBoundaryInputsIdentifyTheUnstableBranchAcrossFreshModelInstances()
        throws
    {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let firstSetup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }
        let secondSetup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }

        let context = try resolvedLayer15ConvInProjContext(from: firstSetup)
        let layer15ResidualAddStepIndex = try #require(
            lastWriterStepIndex(
                in: context.plan.steps,
                before: context.normStepIndex,
                target: context.normInputBinding
            )
        )
        let chain = residualBoundaryChain(
            in: context.plan,
            startingAt: layer15ResidualAddStepIndex
        )
        var firstModel = firstSetup.model
        var secondModel = secondSetup.model

        func earliestDivergentBoundaryStepIndex() throws -> Int? {
            for stepIndex in chain.reversed() {
                let first = try bindingSegmentProbe(
                    model: &firstModel,
                    tokens: promptTokens,
                    stepIndex: stepIndex,
                    bindingIndex: 2,
                    outputDimension: context.normInputDimension
                )
                let second = try bindingSegmentProbe(
                    model: &secondModel,
                    tokens: promptTokens,
                    stepIndex: stepIndex,
                    bindingIndex: 2,
                    outputDimension: context.normInputDimension
                )
                let headError = maxAbsoluteError(first.head, second.head)
                let midError = maxAbsoluteError(first.mid, second.mid)
                let tailError = maxAbsoluteError(first.tail, second.tail)
                if headError >= 0.0001 || midError >= 0.0001 || tailError >= 0.0001 {
                    return stepIndex
                }
            }
            return nil
        }

        let earliestStepCandidate = try earliestDivergentBoundaryStepIndex()
        let earliestStepIndex = try #require(earliestStepCandidate)

        let earliestStep = context.plan.steps[earliestStepIndex]
        let hiddenBinding = try #require(
            earliestStep.bindings.buffers.first(where: { $0.index == 0 })
        )
        let residualBinding = try #require(
            earliestStep.bindings.buffers.first(where: { $0.index == 1 })
        )
        let hiddenSourceStepIndex = try #require(
            lastWriterStepIndex(
                in: context.plan.steps,
                before: earliestStepIndex,
                target: hiddenBinding
            )
        )
        let residualSourceStepIndex = try #require(
            lastWriterStepIndex(
                in: context.plan.steps,
                before: earliestStepIndex,
                target: residualBinding
            )
        )

        let firstHidden = try bindingSegmentProbe(
            model: &firstModel,
            tokens: promptTokens,
            stepIndex: earliestStepIndex,
            bindingIndex: 0,
            phase: .beforeStep,
            outputDimension: context.normInputDimension
        )
        let secondHidden = try bindingSegmentProbe(
            model: &secondModel,
            tokens: promptTokens,
            stepIndex: earliestStepIndex,
            bindingIndex: 0,
            phase: .beforeStep,
            outputDimension: context.normInputDimension
        )
        let firstResidual = try bindingSegmentProbe(
            model: &firstModel,
            tokens: promptTokens,
            stepIndex: earliestStepIndex,
            bindingIndex: 1,
            phase: .beforeStep,
            outputDimension: context.normInputDimension
        )
        let secondResidual = try bindingSegmentProbe(
            model: &secondModel,
            tokens: promptTokens,
            stepIndex: earliestStepIndex,
            bindingIndex: 1,
            phase: .beforeStep,
            outputDimension: context.normInputDimension
        )

        let hiddenHeadError = maxAbsoluteError(firstHidden.head, secondHidden.head)
        let hiddenMidError = maxAbsoluteError(firstHidden.mid, secondHidden.mid)
        let hiddenTailError = maxAbsoluteError(firstHidden.tail, secondHidden.tail)
        let residualHeadError = maxAbsoluteError(firstResidual.head, secondResidual.head)
        let residualMidError = maxAbsoluteError(firstResidual.mid, secondResidual.mid)
        let residualTailError = maxAbsoluteError(firstResidual.tail, secondResidual.tail)

        #expect(
            firstHidden.maximum > 0
                && secondHidden.maximum > 0
                && firstResidual.maximum > 0
                && secondResidual.maximum > 0
                && hiddenHeadError < 0.0001
                && hiddenMidError < 0.0001
                && hiddenTailError < 0.0001
                && residualHeadError < 0.0001
                && residualMidError < 0.0001
                && residualTailError < 0.0001,
            """
            earliest residual-boundary inputs disagree
            earliestStep=\(earliestStepIndex)
            kernel=\(earliestStep.pipeline.label ?? "unlabeled")
            hiddenSourceStepIndex=\(hiddenSourceStepIndex)
            residualSourceStepIndex=\(residualSourceStepIndex)
            hiddenMaxima=[\(firstHidden.maximum), \(secondHidden.maximum)]
            residualMaxima=[\(firstResidual.maximum), \(secondResidual.maximum)]
            hiddenErrors=[\(hiddenHeadError), \(hiddenMidError), \(hiddenTailError)]
            residualErrors=[\(residualHeadError), \(residualMidError), \(residualTailError)]
            hidden nearby:
            \(nearbyStepSummaries(in: context.plan.steps, center: earliestStepIndex, target: hiddenBinding).joined(separator: "\n"))
            residual nearby:
            \(nearbyStepSummaries(in: context.plan.steps, center: residualSourceStepIndex, target: residualBinding).joined(separator: "\n"))
            """
        )
    }

    @Test("LFM earliest divergent attention branch traces to its source output across fresh model instances")
    func lfmEarliestDivergentAttentionBranchTracesToItsSourceOutputAcrossFreshModelInstances() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let firstSetup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }
        let secondSetup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }

        let context = try resolvedLayer15ConvInProjContext(from: firstSetup)
        let layer15ResidualAddStepIndex = try #require(
            lastWriterStepIndex(
                in: context.plan.steps,
                before: context.normStepIndex,
                target: context.normInputBinding
            )
        )
        let chain = residualBoundaryChain(
            in: context.plan,
            startingAt: layer15ResidualAddStepIndex
        )
        var firstModel = firstSetup.model
        var secondModel = secondSetup.model

        func earliestDivergentBoundaryStepIndex() throws -> Int? {
            for stepIndex in chain.reversed() {
                let first = try bindingSegmentProbe(
                    model: &firstModel,
                    tokens: promptTokens,
                    stepIndex: stepIndex,
                    bindingIndex: 2,
                    outputDimension: context.normInputDimension
                )
                let second = try bindingSegmentProbe(
                    model: &secondModel,
                    tokens: promptTokens,
                    stepIndex: stepIndex,
                    bindingIndex: 2,
                    outputDimension: context.normInputDimension
                )
                let headError = maxAbsoluteError(first.head, second.head)
                let midError = maxAbsoluteError(first.mid, second.mid)
                let tailError = maxAbsoluteError(first.tail, second.tail)
                if headError >= 0.0001 || midError >= 0.0001 || tailError >= 0.0001 {
                    return stepIndex
                }
            }
            return nil
        }

        let earliestStepCandidate = try earliestDivergentBoundaryStepIndex()
        let earliestStepIndex = try #require(earliestStepCandidate)

        let earliestStep = context.plan.steps[earliestStepIndex]
        let attentionBinding = try #require(
            earliestStep.bindings.buffers.first(where: { $0.index == 0 })
        )
        let attentionSourceStepIndex = try #require(
            lastWriterStepIndex(
                in: context.plan.steps,
                before: earliestStepIndex,
                target: attentionBinding
            )
        )
        let attentionSourceStep = context.plan.steps[attentionSourceStepIndex]
        let attentionInputDimension = try #require(
            uint32BindingValue(in: attentionSourceStep, at: 3)
        )
        let attentionSourceInputBinding = try #require(
            attentionSourceStep.bindings.buffers.first(where: { $0.index == 0 })
        )
        let upstreamStepIndex = try #require(
            lastWriterStepIndex(
                in: context.plan.steps,
                before: attentionSourceStepIndex,
                target: attentionSourceInputBinding
            )
        )
        let upstreamStep = context.plan.steps[upstreamStepIndex]
        let upstreamWriteBindingIndex = try #require(
            matchingWriteBindingIndex(
                in: upstreamStep,
                target: attentionSourceInputBinding
            )
        )

        let firstAttentionOutput = try bindingSegmentProbe(
            model: &firstModel,
            tokens: promptTokens,
            stepIndex: attentionSourceStepIndex,
            bindingIndex: 2,
            outputDimension: context.normInputDimension
        )
        let secondAttentionOutput = try bindingSegmentProbe(
            model: &secondModel,
            tokens: promptTokens,
            stepIndex: attentionSourceStepIndex,
            bindingIndex: 2,
            outputDimension: context.normInputDimension
        )
        let firstUpstreamOutput = try bindingSegmentProbe(
            model: &firstModel,
            tokens: promptTokens,
            stepIndex: upstreamStepIndex,
            bindingIndex: upstreamWriteBindingIndex,
            outputDimension: attentionInputDimension
        )
        let secondUpstreamOutput = try bindingSegmentProbe(
            model: &secondModel,
            tokens: promptTokens,
            stepIndex: upstreamStepIndex,
            bindingIndex: upstreamWriteBindingIndex,
            outputDimension: attentionInputDimension
        )

        let attentionHeadError = maxAbsoluteError(firstAttentionOutput.head, secondAttentionOutput.head)
        let attentionMidError = maxAbsoluteError(firstAttentionOutput.mid, secondAttentionOutput.mid)
        let attentionTailError = maxAbsoluteError(firstAttentionOutput.tail, secondAttentionOutput.tail)
        let upstreamHeadError = maxAbsoluteError(firstUpstreamOutput.head, secondUpstreamOutput.head)
        let upstreamMidError = maxAbsoluteError(firstUpstreamOutput.mid, secondUpstreamOutput.mid)
        let upstreamTailError = maxAbsoluteError(firstUpstreamOutput.tail, secondUpstreamOutput.tail)

        #expect(
            firstAttentionOutput.maximum > 0
                && secondAttentionOutput.maximum > 0
                && firstUpstreamOutput.maximum > 0
                && secondUpstreamOutput.maximum > 0
                && attentionHeadError < 0.0001
                && attentionMidError < 0.0001
                && attentionTailError < 0.0001
                && upstreamHeadError < 0.0001
                && upstreamMidError < 0.0001
                && upstreamTailError < 0.0001,
            """
            earliest divergent attention branch still disagrees upstream
            earliestStep=\(earliestStepIndex)
            attentionSourceStep=\(attentionSourceStepIndex)
            attentionSourceKernel=\(attentionSourceStep.pipeline.label ?? "unlabeled")
            attentionSourceTensor=\(attentionSourceStep.metadata.weightTensorName ?? "nil")
            upstreamStep=\(upstreamStepIndex)
            upstreamKernel=\(upstreamStep.pipeline.label ?? "unlabeled")
            upstreamTensor=\(upstreamStep.metadata.weightTensorName ?? "nil")
            attentionInputDimension=\(attentionInputDimension)
            attentionErrors=[\(attentionHeadError), \(attentionMidError), \(attentionTailError)]
            upstreamErrors=[\(upstreamHeadError), \(upstreamMidError), \(upstreamTailError)]
            attention nearby:
            \(nearbyStepSummaries(in: context.plan.steps, center: attentionSourceStepIndex, target: attentionBinding).joined(separator: "\n"))
            upstream nearby:
            \(nearbyStepSummaries(in: context.plan.steps, center: upstreamStepIndex, target: attentionSourceInputBinding).joined(separator: "\n"))
            """
        )
    }

    @Test("LFM earliest divergent attention query chain identifies the first unstable source across fresh model instances")
    func lfmEarliestDivergentAttentionQueryChainIdentifiesTheFirstUnstableSourceAcrossFreshModelInstances()
        throws
    {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let firstSetup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }
        let secondSetup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }

        let context = try resolvedLayer15ConvInProjContext(from: firstSetup)
        let layer15ResidualAddStepIndex = try #require(
            lastWriterStepIndex(
                in: context.plan.steps,
                before: context.normStepIndex,
                target: context.normInputBinding
            )
        )
        let chain = residualBoundaryChain(
            in: context.plan,
            startingAt: layer15ResidualAddStepIndex
        )
        var firstModel = firstSetup.model
        var secondModel = secondSetup.model

        func earliestDivergentBoundaryStepIndex() throws -> Int? {
            for stepIndex in chain.reversed() {
                let first = try bindingSegmentProbe(
                    model: &firstModel,
                    tokens: promptTokens,
                    stepIndex: stepIndex,
                    bindingIndex: 2,
                    outputDimension: context.normInputDimension
                )
                let second = try bindingSegmentProbe(
                    model: &secondModel,
                    tokens: promptTokens,
                    stepIndex: stepIndex,
                    bindingIndex: 2,
                    outputDimension: context.normInputDimension
                )
                let headError = maxAbsoluteError(first.head, second.head)
                let midError = maxAbsoluteError(first.mid, second.mid)
                let tailError = maxAbsoluteError(first.tail, second.tail)
                if headError >= 0.0001 || midError >= 0.0001 || tailError >= 0.0001 {
                    return stepIndex
                }
            }
            return nil
        }

        let earliestStepCandidate = try earliestDivergentBoundaryStepIndex()
        let earliestStepIndex = try #require(earliestStepCandidate)
        let earliestStep = context.plan.steps[earliestStepIndex]
        let attentionBinding = try #require(
            earliestStep.bindings.buffers.first(where: { $0.index == 0 })
        )
        let attentionSourceStepIndex = try #require(
            lastWriterStepIndex(
                in: context.plan.steps,
                before: earliestStepIndex,
                target: attentionBinding
            )
        )
        let attentionSourceStep = context.plan.steps[attentionSourceStepIndex]
        let attentionInputDimension = try #require(
            uint32BindingValue(in: attentionSourceStep, at: 3)
        )
        let flashInputBinding = try #require(
            attentionSourceStep.bindings.buffers.first(where: { $0.index == 0 })
        )
        let flashStepIndex = try #require(
            lastWriterStepIndex(
                in: context.plan.steps,
                before: attentionSourceStepIndex,
                target: flashInputBinding
            )
        )

        var consumerStepIndex = flashStepIndex
        var consumerBindingIndex = 0
        var currentDimension = attentionInputDimension
        var records: [String] = []
        var earliest: (
            stepIndex: Int,
            kernel: String,
            tensor: String?,
            headError: Float,
            midError: Float,
            tailError: Float
        )?

        for _ in 0..<8 {
            let consumerStep = context.plan.steps[consumerStepIndex]
            guard let targetBinding = consumerStep.bindings.buffers.first(where: { $0.index == consumerBindingIndex }) else {
                break
            }
            guard let sourceStepIndex = lastWriterStepIndex(
                in: context.plan.steps,
                before: consumerStepIndex,
                target: targetBinding
            ) else {
                break
            }
            let sourceStep = context.plan.steps[sourceStepIndex]
            guard let sourceOutputBindingIndex = matchingWriteBindingIndex(
                in: sourceStep,
                target: targetBinding
            ) else {
                break
            }

            let first = try bindingSegmentProbe(
                model: &firstModel,
                tokens: promptTokens,
                stepIndex: sourceStepIndex,
                bindingIndex: sourceOutputBindingIndex,
                outputDimension: currentDimension
            )
            let second = try bindingSegmentProbe(
                model: &secondModel,
                tokens: promptTokens,
                stepIndex: sourceStepIndex,
                bindingIndex: sourceOutputBindingIndex,
                outputDimension: currentDimension
            )
            let headError = maxAbsoluteError(first.head, second.head)
            let midError = maxAbsoluteError(first.mid, second.mid)
            let tailError = maxAbsoluteError(first.tail, second.tail)
            records.append(
                "[\(sourceStepIndex)] kernel=\(sourceStep.pipeline.label ?? "unlabeled") tensor=\(sourceStep.metadata.weightTensorName ?? "nil") dim=\(currentDimension) errors=[\(headError), \(midError), \(tailError)]"
            )
            if earliest == nil,
               headError >= 0.0001 || midError >= 0.0001 || tailError >= 0.0001 {
                earliest = (
                    sourceStepIndex,
                    sourceStep.pipeline.label ?? "unlabeled",
                    sourceStep.metadata.weightTensorName,
                    headError,
                    midError,
                    tailError
                )
            }

            guard sourceStepIndex > 0 else { break }
            guard sourceStep.bindings.buffers.contains(where: { $0.index == 0 }) else { break }
            consumerStepIndex = sourceStepIndex
            consumerBindingIndex = 0
            currentDimension = uint32BindingValue(in: sourceStep, at: 3) ?? currentDimension
        }

        #expect(
            earliest == nil,
            """
            earliest divergent attention query source detected
            earliestBoundaryStep=\(earliestStepIndex)
            attentionSourceStep=\(attentionSourceStepIndex)
            flashStep=\(flashStepIndex)
            queryTrace:
            \(records.joined(separator: "\n"))
            earliestSourceStep=\(earliest.map { String($0.stepIndex) } ?? "none")
            earliestSourceKernel=\(earliest?.kernel ?? "nil")
            earliestSourceTensor=\(earliest?.tensor ?? "nil")
            earliestSourceErrors=[\(earliest?.headError ?? 0), \(earliest?.midError ?? 0), \(earliest?.tailError ?? 0)]
            """
        )
    }

    @Test("LFM earliest divergent attention KV cache path distinguishes cache-fill from flash instability across fresh model instances")
    func lfmEarliestDivergentAttentionKVCachePathDistinguishesCacheFillFromFlashInstabilityAcrossFreshModelInstances()
        throws
    {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let firstSetup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }
        let secondSetup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }

        let context = try resolvedLayer15ConvInProjContext(from: firstSetup)
        let layer15ResidualAddStepIndex = try #require(
            lastWriterStepIndex(
                in: context.plan.steps,
                before: context.normStepIndex,
                target: context.normInputBinding
            )
        )
        let chain = residualBoundaryChain(
            in: context.plan,
            startingAt: layer15ResidualAddStepIndex
        )
        var firstModel = firstSetup.model
        var secondModel = secondSetup.model

        func earliestDivergentBoundaryStepIndex() throws -> Int? {
            for stepIndex in chain.reversed() {
                let first = try bindingSegmentProbe(
                    model: &firstModel,
                    tokens: promptTokens,
                    stepIndex: stepIndex,
                    bindingIndex: 2,
                    outputDimension: context.normInputDimension
                )
                let second = try bindingSegmentProbe(
                    model: &secondModel,
                    tokens: promptTokens,
                    stepIndex: stepIndex,
                    bindingIndex: 2,
                    outputDimension: context.normInputDimension
                )
                let headError = maxAbsoluteError(first.head, second.head)
                let midError = maxAbsoluteError(first.mid, second.mid)
                let tailError = maxAbsoluteError(first.tail, second.tail)
                if headError >= 0.0001 || midError >= 0.0001 || tailError >= 0.0001 {
                    return stepIndex
                }
            }
            return nil
        }

        let earliestStepCandidate = try earliestDivergentBoundaryStepIndex()
        let earliestStepIndex = try #require(earliestStepCandidate)
        let earliestStep = context.plan.steps[earliestStepIndex]
        let attentionBinding = try #require(
            earliestStep.bindings.buffers.first(where: { $0.index == 0 })
        )
        let attentionSourceStepIndex = try #require(
            lastWriterStepIndex(
                in: context.plan.steps,
                before: earliestStepIndex,
                target: attentionBinding
            )
        )
        let attentionSourceStep = context.plan.steps[attentionSourceStepIndex]
        let flashOutputBinding = try #require(
            attentionSourceStep.bindings.buffers.first(where: { $0.index == 0 })
        )
        let flashStepIndex = try #require(
            lastWriterStepIndex(
                in: context.plan.steps,
                before: attentionSourceStepIndex,
                target: flashOutputBinding
            )
        )
        let flashStep = context.plan.steps[flashStepIndex]

        let kCacheBinding = try #require(
            flashStep.bindings.buffers.first(where: { $0.index == 1 })
        )
        let vCacheBinding = try #require(
            flashStep.bindings.buffers.first(where: { $0.index == 2 })
        )
        let cacheFillForKStepIndex = try #require(
            lastWriterStepIndex(
                in: context.plan.steps,
                before: flashStepIndex,
                target: kCacheBinding
            )
        )
        let cacheFillForVStepIndex = try #require(
            lastWriterStepIndex(
                in: context.plan.steps,
                before: flashStepIndex,
                target: vCacheBinding
            )
        )
        let cacheFillStepIndex = cacheFillForKStepIndex
        let cacheFillStep = context.plan.steps[cacheFillStepIndex]
        #expect(cacheFillForVStepIndex == cacheFillStepIndex)

        let kInputBinding = try #require(
            cacheFillStep.bindings.buffers.first(where: { $0.index == 0 })
        )
        let vInputBinding = try #require(
            cacheFillStep.bindings.buffers.first(where: { $0.index == 1 })
        )
        let kSourceStepIndex = try #require(
            lastWriterStepIndex(
                in: context.plan.steps,
                before: cacheFillStepIndex,
                target: kInputBinding
            )
        )
        let vSourceStepIndex = try #require(
            lastWriterStepIndex(
                in: context.plan.steps,
                before: cacheFillStepIndex,
                target: vInputBinding
            )
        )
        let kSourceStep = context.plan.steps[kSourceStepIndex]
        let vSourceStep = context.plan.steps[vSourceStepIndex]
        let kDimension = try #require(uint32BindingValue(in: kSourceStep, at: 4))
        let vDimension = try #require(uint32BindingValue(in: vSourceStep, at: 4))

        let firstKInput = try bindingSegmentProbe(
            model: &firstModel,
            tokens: promptTokens,
            stepIndex: cacheFillStepIndex,
            bindingIndex: 0,
            phase: .beforeStep,
            outputDimension: kDimension
        )
        let secondKInput = try bindingSegmentProbe(
            model: &secondModel,
            tokens: promptTokens,
            stepIndex: cacheFillStepIndex,
            bindingIndex: 0,
            phase: .beforeStep,
            outputDimension: kDimension
        )
        let firstVInput = try bindingSegmentProbe(
            model: &firstModel,
            tokens: promptTokens,
            stepIndex: cacheFillStepIndex,
            bindingIndex: 1,
            phase: .beforeStep,
            outputDimension: vDimension
        )
        let secondVInput = try bindingSegmentProbe(
            model: &secondModel,
            tokens: promptTokens,
            stepIndex: cacheFillStepIndex,
            bindingIndex: 1,
            phase: .beforeStep,
            outputDimension: vDimension
        )
        let firstKCache = try bindingSegmentProbe(
            model: &firstModel,
            tokens: promptTokens,
            stepIndex: flashStepIndex,
            bindingIndex: 1,
            phase: .beforeStep,
            outputDimension: kDimension
        )
        let secondKCache = try bindingSegmentProbe(
            model: &secondModel,
            tokens: promptTokens,
            stepIndex: flashStepIndex,
            bindingIndex: 1,
            phase: .beforeStep,
            outputDimension: kDimension
        )
        let firstVCache = try bindingSegmentProbe(
            model: &firstModel,
            tokens: promptTokens,
            stepIndex: flashStepIndex,
            bindingIndex: 2,
            phase: .beforeStep,
            outputDimension: vDimension
        )
        let secondVCache = try bindingSegmentProbe(
            model: &secondModel,
            tokens: promptTokens,
            stepIndex: flashStepIndex,
            bindingIndex: 2,
            phase: .beforeStep,
            outputDimension: vDimension
        )

        let kInputErrors = [
            maxAbsoluteError(firstKInput.head, secondKInput.head),
            maxAbsoluteError(firstKInput.mid, secondKInput.mid),
            maxAbsoluteError(firstKInput.tail, secondKInput.tail),
        ]
        let vInputErrors = [
            maxAbsoluteError(firstVInput.head, secondVInput.head),
            maxAbsoluteError(firstVInput.mid, secondVInput.mid),
            maxAbsoluteError(firstVInput.tail, secondVInput.tail),
        ]
        let kCacheErrors = [
            maxAbsoluteError(firstKCache.head, secondKCache.head),
            maxAbsoluteError(firstKCache.mid, secondKCache.mid),
            maxAbsoluteError(firstKCache.tail, secondKCache.tail),
        ]
        let vCacheErrors = [
            maxAbsoluteError(firstVCache.head, secondVCache.head),
            maxAbsoluteError(firstVCache.mid, secondVCache.mid),
            maxAbsoluteError(firstVCache.tail, secondVCache.tail),
        ]

        #expect(
            firstKInput.maximum > 0
                && secondKInput.maximum > 0
                && firstVInput.maximum > 0
                && secondVInput.maximum > 0
                && firstKCache.maximum > 0
                && secondKCache.maximum > 0
                && firstVCache.maximum > 0
                && secondVCache.maximum > 0
                && kInputErrors.allSatisfy { $0 < 0.0001 }
                && vInputErrors.allSatisfy { $0 < 0.0001 }
                && kCacheErrors.allSatisfy { $0 < 0.0001 }
                && vCacheErrors.allSatisfy { $0 < 0.0001 },
            """
            earliest divergent attention KV cache path disagrees
            earliestBoundaryStep=\(earliestStepIndex)
            flashStep=\(flashStepIndex)
            cacheFillStep=\(cacheFillStepIndex)
            kSourceStep=\(kSourceStepIndex) kernel=\(kSourceStep.pipeline.label ?? "unlabeled") tensor=\(kSourceStep.metadata.weightTensorName ?? "nil")
            vSourceStep=\(vSourceStepIndex) kernel=\(vSourceStep.pipeline.label ?? "unlabeled") tensor=\(vSourceStep.metadata.weightTensorName ?? "nil")
            kDim=\(kDimension) vDim=\(vDimension)
            kInputErrors=\(kInputErrors)
            vInputErrors=\(vInputErrors)
            kCacheErrors=\(kCacheErrors)
            vCacheErrors=\(vCacheErrors)
            cacheFill nearby:
            \(nearbyStepSummaries(in: context.plan.steps, center: cacheFillStepIndex, target: kInputBinding).joined(separator: "\n"))
            flash nearby:
            \(nearbyStepSummaries(in: context.plan.steps, center: flashStepIndex, target: kCacheBinding).joined(separator: "\n"))
            """
        )
    }

    @Test("LFM layer15 conv in-proj segment probes stay stable when prefill barriers are forced")
    func lfmLayer15ConvInProjSegmentProbesStayStableWhenPrefillBarriersAreForced() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let setup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }

        let collected = setup.collected
        var model = setup.model
        let plan = try #require(model.prefillPlan)
        let access = try #require(
            layerProjectionAccess(
                in: collected,
                layerPrefix: "model.layers.15.",
                tensorNameSuffix: ".conv.in_proj.weight"
            )
        )
        let projection = try #require(projectionKind(from: access.entry))
        let stepIndex = try #require(
            matchingProjectionStepIndex(
                in: plan,
                entryIndex: access.entry.index,
                tensorName: access.tensorName,
                access: access.access,
                projection: projection
            )
        )
        let step = plan.steps[stepIndex]
        let outputDimension = try #require(uint32BindingValue(in: step, at: 4))
        let optimized = try convInProjSegmentProbe(
            model: &model,
            tokens: promptTokens,
            stepIndex: stepIndex,
            outputDimension: outputDimension
        )
        let optimizedRepeat = try convInProjSegmentProbe(
            model: &model,
            tokens: promptTokens,
            stepIndex: stepIndex,
            outputDimension: outputDimension
        )

        model.prefillPlan = forcedBufferBarrierPlan(from: plan)
        let forcedAll = try convInProjSegmentProbe(
            model: &model,
            tokens: promptTokens,
            stepIndex: stepIndex,
            outputDimension: outputDimension
        )

        model.prefillPlan = planWithBarrierPolicy(
            plan,
            stepIndices: [stepIndex],
            barrierPolicy: .bufferBarrier
        )
        let forcedCurrentOnly = try convInProjSegmentProbe(
            model: &model,
            tokens: promptTokens,
            stepIndex: stepIndex,
            outputDimension: outputDimension
        )

        model.prefillPlan = planWithBarrierPolicy(
            plan,
            stepIndices: [stepIndex + 1],
            barrierPolicy: .bufferBarrier
        )
        let forcedNextOnly = try convInProjSegmentProbe(
            model: &model,
            tokens: promptTokens,
            stepIndex: stepIndex,
            outputDimension: outputDimension
        )

        model.prefillPlan = planWithBarrierPolicy(
            plan,
            stepIndices: [stepIndex, stepIndex + 1],
            barrierPolicy: .bufferBarrier
        )
        let forcedLocalPair = try convInProjSegmentProbe(
            model: &model,
            tokens: promptTokens,
            stepIndex: stepIndex,
            outputDimension: outputDimension
        )

        let optimizedMax = optimized.maximum
        let forcedMax = forcedAll.maximum
        let repeatHeadError = maxAbsoluteError(optimized.head, optimizedRepeat.head)
        let repeatMidError = maxAbsoluteError(optimized.mid, optimizedRepeat.mid)
        let repeatTailError = maxAbsoluteError(optimized.tail, optimizedRepeat.tail)
        let headError = maxAbsoluteError(optimized.head, forcedAll.head)
        let midError = maxAbsoluteError(optimized.mid, forcedAll.mid)
        let tailError = maxAbsoluteError(optimized.tail, forcedAll.tail)
        let currentOnlyHeadError = maxAbsoluteError(optimized.head, forcedCurrentOnly.head)
        let currentOnlyMidError = maxAbsoluteError(optimized.mid, forcedCurrentOnly.mid)
        let currentOnlyTailError = maxAbsoluteError(optimized.tail, forcedCurrentOnly.tail)
        let nextOnlyHeadError = maxAbsoluteError(optimized.head, forcedNextOnly.head)
        let nextOnlyMidError = maxAbsoluteError(optimized.mid, forcedNextOnly.mid)
        let nextOnlyTailError = maxAbsoluteError(optimized.tail, forcedNextOnly.tail)
        let localPairHeadError = maxAbsoluteError(optimized.head, forcedLocalPair.head)
        let localPairMidError = maxAbsoluteError(optimized.mid, forcedLocalPair.mid)
        let localPairTailError = maxAbsoluteError(optimized.tail, forcedLocalPair.tail)

        #expect(
            optimizedMax > 0 && forcedMax > 0 && headError < 0.0001 && midError < 0.0001 && tailError < 0.0001,
            """
            forcing prefill barriers changed reliable conv.in_proj segment probes
            optimizedMax=\(optimizedMax) forcedMax=\(forcedMax)
            repeatErrors=[\(repeatHeadError), \(repeatMidError), \(repeatTailError)]
            headError=\(headError) midError=\(midError) tailError=\(tailError)
            currentOnlyMax=\(forcedCurrentOnly.maximum) currentOnlyErrors=[\(currentOnlyHeadError), \(currentOnlyMidError), \(currentOnlyTailError)]
            nextOnlyMax=\(forcedNextOnly.maximum) nextOnlyErrors=[\(nextOnlyHeadError), \(nextOnlyMidError), \(nextOnlyTailError)]
            localPairMax=\(forcedLocalPair.maximum) localPairErrors=[\(localPairHeadError), \(localPairMidError), \(localPairTailError)]
            optimized head=\(optimized.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            optimized-repeat head=\(optimizedRepeat.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            forced-all head=\(forcedAll.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            forced-current head=\(forcedCurrentOnly.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            forced-next head=\(forcedNextOnly.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            optimized mid=\(optimized.mid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            optimized-repeat mid=\(optimizedRepeat.mid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            forced-all mid=\(forcedAll.mid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            forced-current mid=\(forcedCurrentOnly.mid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            forced-next mid=\(forcedNextOnly.mid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            optimized tail=\(optimized.tail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            optimized-repeat tail=\(optimizedRepeat.tail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            forced-all tail=\(forcedAll.tail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            forced-current tail=\(forcedCurrentOnly.tail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            forced-next tail=\(forcedNextOnly.tail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            """
        )
    }

    @Test("LFM layer15 conv in-proj segment probes stay stable with device visibility barriers")
    func lfmLayer15ConvInProjSegmentProbesStayStableWithDeviceVisibilityBarriers() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let setup = try withTemporaryEnvironment("SWIFTLM_DISABLE_MPP", "1") {
            try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(
    
                useCachedStore: false
            )
        }

        let collected = setup.collected
        var model = setup.model
        let plan = try #require(model.prefillPlan)
        let access = try #require(
            layerProjectionAccess(
                in: collected,
                layerPrefix: "model.layers.15.",
                tensorNameSuffix: ".conv.in_proj.weight"
            )
        )
        let projection = try #require(projectionKind(from: access.entry))
        let stepIndex = try #require(
            matchingProjectionStepIndex(
                in: plan,
                entryIndex: access.entry.index,
                tensorName: access.tensorName,
                access: access.access,
                projection: projection
            )
        )
        let step = plan.steps[stepIndex]
        let outputDimension = try #require(uint32BindingValue(in: step, at: 4))
        let `default` = try convInProjSegmentProbe(
            model: &model,
            tokens: promptTokens,
            stepIndex: stepIndex,
            outputDimension: outputDimension
        )
        let defaultRepeat = try convInProjSegmentProbe(
            model: &model,
            tokens: promptTokens,
            stepIndex: stepIndex,
            outputDimension: outputDimension
        )
        let stepVisible = try convInProjSegmentProbe(
            model: &model,
            tokens: promptTokens,
            stepIndex: stepIndex,
            outputDimension: outputDimension,
            stepVisibilityOptions: .device,
            probeVisibilityOptions: []
        )
        let probeVisible = try convInProjSegmentProbe(
            model: &model,
            tokens: promptTokens,
            stepIndex: stepIndex,
            outputDimension: outputDimension,
            stepVisibilityOptions: [],
            probeVisibilityOptions: .device
        )
        let bothVisible = try convInProjSegmentProbe(
            model: &model,
            tokens: promptTokens,
            stepIndex: stepIndex,
            outputDimension: outputDimension,
            visibilityOptions: .device
        )
        let defaultMax = `default`.maximum
        let deviceMax = bothVisible.maximum
        let repeatHeadError = maxAbsoluteError(`default`.head, defaultRepeat.head)
        let repeatMidError = maxAbsoluteError(`default`.mid, defaultRepeat.mid)
        let repeatTailError = maxAbsoluteError(`default`.tail, defaultRepeat.tail)
        let headError = maxAbsoluteError(`default`.head, bothVisible.head)
        let midError = maxAbsoluteError(`default`.mid, bothVisible.mid)
        let tailError = maxAbsoluteError(`default`.tail, bothVisible.tail)
        let stepHeadError = maxAbsoluteError(`default`.head, stepVisible.head)
        let stepMidError = maxAbsoluteError(`default`.mid, stepVisible.mid)
        let stepTailError = maxAbsoluteError(`default`.tail, stepVisible.tail)
        let probeHeadError = maxAbsoluteError(`default`.head, probeVisible.head)
        let probeMidError = maxAbsoluteError(`default`.mid, probeVisible.mid)
        let probeTailError = maxAbsoluteError(`default`.tail, probeVisible.tail)

        #expect(
            defaultMax > 0 && deviceMax > 0 && headError < 0.0001 && midError < 0.0001 && tailError < 0.0001,
            """
            device visibility barrier changed reliable conv.in_proj segment probes
            defaultMax=\(defaultMax) deviceMax=\(deviceMax)
            repeatErrors=[\(repeatHeadError), \(repeatMidError), \(repeatTailError)]
            headError=\(headError) midError=\(midError) tailError=\(tailError)
            stepOnlyMax=\(stepVisible.maximum) stepOnlyErrors=[\(stepHeadError), \(stepMidError), \(stepTailError)]
            probeOnlyMax=\(probeVisible.maximum) probeOnlyErrors=[\(probeHeadError), \(probeMidError), \(probeTailError)]
            default head=\(`default`.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            default-repeat head=\(defaultRepeat.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            step-only head=\(stepVisible.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            probe-only head=\(probeVisible.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            both head=\(bothVisible.head.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            default mid=\(`default`.mid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            default-repeat mid=\(defaultRepeat.mid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            step-only mid=\(stepVisible.mid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            probe-only mid=\(probeVisible.mid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            both mid=\(bothVisible.mid.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            default tail=\(`default`.tail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            default-repeat tail=\(defaultRepeat.tail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            step-only tail=\(stepVisible.tail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            probe-only tail=\(probeVisible.tail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            both tail=\(bothVisible.tail.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            """
        )
    }
    #endif

    private func entries(
        in entries: [DispatchEntry],
        containingAnyTensorPrefix prefixes: [String]
    ) -> [DispatchEntry] {
        entries.filter { entry in
            entry.parameterBindings.contains { binding in
                prefixes.contains { binding.tensorName.contains($0) }
            }
        }
    }

    private func summarizeNilLayerDownProjAccesses(
        in collected: BenchmarkSupport.CollectedPrefillEntries
    ) throws -> [WeightAccessSummary] {
        let downProjEntries = collected.fusedEntries.compactMap { entry -> DispatchEntry? in
            guard let projection = entry.fragment as? LinearFragment, projection.field == "down_proj" else {
                return nil
            }
            return entry
        }
        let resolver = ProjectionWeightAccessPolicyResolver()

        return downProjEntries.map { entry in
            let binding = entry.parameterBindings.first(where: { $0.role == "down_proj" })
            let tensor = binding?.tensorName ?? "<missing>"
            let request = binding.map {
                resolver.accessRequest(
                    for: entry,
                    role: "down_proj",
                    binding: $0,
                    executionPhase: .prefill,
                    stafWeightStore: collected.store
                )
            }
            let resolved = request.flatMap { collected.store.resolvedBufferAccess(for: $0) }
            let format = binding.flatMap { collected.store.tensor(for: $0.tensorName)?.format.schemeIdentifier }

            return WeightAccessSummary(
                entry: describeEntry(entry),
                tensorName: tensor,
                preferredLayout: request?.preferredLayout,
                resolvedLayout: resolved?.layout,
                resolvedOffset: resolved?.offset,
                formatIdentifier: format,
                bindingExists: binding != nil,
                hasResolvedAccess: resolved != nil
            )
        }
    }

    private func layerDownProjAccesses(
        in collected: BenchmarkSupport.CollectedPrefillEntries,
        layerPrefix: String
    ) -> [(entry: DispatchEntry, tensorName: String, access: STAFWeightBufferAccess, schemeIdentifier: QuantizationSchemeIdentifier)] {
        let resolver = ProjectionWeightAccessPolicyResolver()
        return collected.fusedEntries.compactMap { entry in
            guard
                let projection = entry.fragment as? LinearFragment,
                projection.field == "down_proj",
                let binding = entry.parameterBindings.first(where: {
                    $0.role == "down_proj" && $0.tensorName.contains(layerPrefix)
                }),
                let access = collected.store.resolvedBufferAccess(
                    for: resolver.accessRequest(
                        for: entry,
                        role: "down_proj",
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
        }
    }

    private func layerProjectionAccess(
        in collected: BenchmarkSupport.CollectedPrefillEntries,
        layerPrefix: String,
        tensorNameSuffix: String
    ) -> (entry: DispatchEntry, tensorName: String, access: STAFWeightBufferAccess, schemeIdentifier: QuantizationSchemeIdentifier)? {
        let resolver = ProjectionWeightAccessPolicyResolver()
        return collected.fusedEntries.compactMap { entry -> (entry: DispatchEntry, tensorName: String, access: STAFWeightBufferAccess, schemeIdentifier: QuantizationSchemeIdentifier)? in
            guard
                entry.fragment is LinearFragment,
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

    private func layerBatchedProjectionAccess(
        in collected: BenchmarkSupport.CollectedPrefillEntries,
        layerPrefix: String,
        role: String,
        tensorNameSuffix: String
    ) -> (
        entry: DispatchEntry,
        projection: BatchedProjection.Entry,
        projectionIndex: Int,
        tensorName: String,
        access: STAFWeightBufferAccess,
        schemeIdentifier: QuantizationSchemeIdentifier
    )? {
        let resolver = ProjectionWeightAccessPolicyResolver()
        return collected.fusedEntries.compactMap { entry -> (
            entry: DispatchEntry,
            projection: BatchedProjection.Entry,
            projectionIndex: Int,
            tensorName: String,
            access: STAFWeightBufferAccess,
            schemeIdentifier: QuantizationSchemeIdentifier
        )? in
            guard let batched = entry.fragment as? BatchedProjection else {
                return nil
            }
            guard batched.projections.contains(where: { $0.field == role }) else {
                return nil
            }
            guard let binding = entry.parameterBindings.first(where: {
                $0.role == role
                    && $0.tensorName.contains(layerPrefix)
                    && $0.tensorName.hasSuffix(tensorNameSuffix)
            }) else {
                return nil
            }
            guard let matchedProjectionIndex = batched.projections.firstIndex(where: { $0.field == role }) else {
                return nil
            }
            let matchedProjection = batched.projections[matchedProjectionIndex]
            guard let access = collected.store.resolvedBufferAccess(
                for: resolver.accessRequest(
                    for: entry,
                    role: role,
                    binding: binding,
                    executionPhase: .prefill,
                    stafWeightStore: collected.store
                )
            ),
            let tensor = collected.store.tensor(for: binding.tensorName) else {
                return nil
            }
            return (
                entry,
                matchedProjection,
                matchedProjectionIndex,
                binding.tensorName,
                access,
                tensor.format.schemeIdentifier
            )
        }.first
    }

    private func projectionKind(from entry: DispatchEntry) -> LinearFragment? {
        entry.fragment as? LinearFragment
    }

    private func layer14SwiGLUStepIndex(in plan: MetalPrefillPlan) -> Int? {
        plan.steps.enumerated().first { _, step in
            (step.pipeline.label ?? "").hasPrefix("swiglu_seq")
                && step.bindings.buffers.contains(where: { $0.index == 2 && $0.offset == 0 })
                && step.bindings.buffers.contains(where: { $0.index == 0 && $0.offset == 2_097_152 })
                && step.bindings.buffers.contains(where: { $0.index == 1 && $0.offset == 4_194_304 })
        }?.offset
    }

    private func resolvedLayer15ConvInProjContext(
        from setup: BenchmarkSupport.SetupWithCollectedPrefillEntries
    ) throws -> (
        plan: MetalPrefillPlan,
        convStepIndex: Int,
        normStepIndex: Int,
        normInputBinding: MetalBufferBinding,
        normInputDimension: Int
    ) {
        let plan = try #require(setup.model.prefillPlan)
        let access = try #require(
            layerProjectionAccess(
                in: setup.collected,
                layerPrefix: "model.layers.15.",
                tensorNameSuffix: ".conv.in_proj.weight"
            )
        )
        let projection = try #require(projectionKind(from: access.entry))
        let convStepIndex = try #require(
            matchingProjectionStepIndex(
                in: plan,
                entryIndex: access.entry.index,
                tensorName: access.tensorName,
                access: access.access,
                projection: projection
            )
        )
        let normStepIndex = convStepIndex - 1
        let normStep = plan.steps[normStepIndex]
        let normInputBinding = try #require(
            normStep.bindings.buffers.first(where: { $0.index == 0 })
        )
        let normInputDimension = try #require(uint32BindingValue(in: normStep, at: 3))
        return (plan, convStepIndex, normStepIndex, normInputBinding, normInputDimension)
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

    private func matchingProjectionStepIndex(
        in plan: MetalPrefillPlan,
        entryIndex: Int,
        tensorName: String,
        access: STAFWeightBufferAccess,
        projection: some ProjectionDimensions
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

    private func matchingBatchedProjectionStepIndex(
        in plan: MetalPrefillPlan,
        entryIndex: Int,
        access: STAFWeightBufferAccess,
        projection: BatchedProjection.Entry,
        projectionIndex: Int,
        projectionCount: Int
    ) -> Int? {
        plan.steps.enumerated().first { _, step in
            step.metadata.entryIndex == entryIndex
                && batchedProjectionStepMatches(
                    step,
                    access: access,
                    projection: projection,
                    projectionIndex: projectionIndex,
                    projectionCount: projectionCount
                )
        }?.offset
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

    private func projectionStepMatches(
        _ step: MetalPrefillStep,
        tensorName: String,
        access: STAFWeightBufferAccess,
        projection: some ProjectionDimensions
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

    private func batchedProjectionStepMatches(
        _ step: MetalPrefillStep,
        access: STAFWeightBufferAccess,
        projection: BatchedProjection.Entry,
        projectionIndex: Int,
        projectionCount: Int
    ) -> Bool {
        let weightBindingIndex = 1 + projectionIndex
        guard
            let weightBinding = step.bindings.buffers.first(where: { $0.index == weightBindingIndex }),
            weightBinding.buffer === access.buffer,
            weightBinding.offset == access.offset
        else {
            return false
        }

        let dimensionBase = 1 + 2 * projectionCount
        return uint32BindingValue(in: step, at: dimensionBase) == projection.inputDimension
            && uint32BindingValue(in: step, at: dimensionBase + 1 + projectionIndex) == projection.outputDimension
    }

    private func candidateProjectionStepSummaries(
        in plan: MetalPrefillPlan,
        tensorName: String,
        projection: some ProjectionDimensions
    ) -> [String] {
        plan.steps.enumerated().compactMap { index, step in
            let inputDimension = uint32BindingValue(in: step, at: 3)
            let outputDimension = uint32BindingValue(in: step, at: 4)
            let sameTensor = step.metadata.weightTensorName == tensorName
            let sameShape = inputDimension == projection.inputDimension
                && outputDimension == projection.outputDimension
            guard sameTensor || sameShape else { return nil }
            let weightOffset = step.bindings.buffers.first(where: { $0.index == 1 })?.offset ?? -1
            let kernelName = step.pipeline.label ?? "unlabeled"
            return "[\(index)] kernel=\(kernelName) entry=\(step.metadata.entryIndex.map(String.init) ?? "-") tensor=\(step.metadata.weightTensorName ?? "nil") in=\(inputDimension.map(String.init) ?? "nil") out=\(outputDimension.map(String.init) ?? "nil") weightOffset=\(weightOffset)"
        }
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

    private func residualBoundaryChain(
        in plan: MetalPrefillPlan,
        startingAt stepIndex: Int
    ) -> [Int] {
        guard !plan.steps.isEmpty else { return [] }
        var chain: [Int] = []
        var current = stepIndex
        while current >= 0 && current < plan.steps.count {
            chain.append(current)
            let step = plan.steps[current]
            guard (step.pipeline.label ?? "").hasPrefix("residual_add_seq") else {
                break
            }
            guard let residualBinding = step.bindings.buffers.first(where: { $0.index == 1 }) else {
                break
            }
            guard let copyStepIndex = lastWriterStepIndex(
                in: plan.steps,
                before: current,
                target: residualBinding
            ) else {
                break
            }
            let copyStep = plan.steps[copyStepIndex]
            guard (copyStep.pipeline.label ?? "").hasPrefix("copy_buffer_seq") else {
                break
            }
            guard let copyInputBinding = copyStep.bindings.buffers.first(where: { $0.index == 0 }) else {
                break
            }
            guard let previous = lastWriterStepIndex(
                in: plan.steps,
                before: copyStepIndex,
                target: copyInputBinding
            ), previous < current else {
                break
            }
            current = previous
        }
        return chain
    }

    private func nearbyStepSummaries(
        in steps: [MetalPrefillStep],
        center: Int,
        target: MetalBufferBinding,
        radius: Int = 4
    ) -> [String] {
        guard !steps.isEmpty else { return [] }
        let lower = max(0, center - radius)
        let upper = min(steps.count - 1, center + radius)
        return (lower...upper).map { index in
            let step = steps[index]
            let kernel = step.pipeline.label ?? "unlabeled"
            let writes = step.metadata.bufferAccessPattern?.writeIndices
                .sorted()
                .map(String.init)
                .joined(separator: ",") ?? "nil"
            let reads = step.metadata.bufferAccessPattern?.readIndices
                .sorted()
                .map(String.init)
                .joined(separator: ",") ?? "nil"
            let barrier = describeBarrierPolicy(step.barrierPolicy)
            let grid = "\(step.gridSize.width)x\(step.gridSize.height)x\(step.gridSize.depth)"
            let tg = "\(step.threadgroupSize.width)x\(step.threadgroupSize.height)x\(step.threadgroupSize.depth)"
            let bindings = step.bindings.buffers
                .map { binding in
                    let mark = binding.buffer === target.buffer && binding.offset == target.offset ? "*" : ""
                    return "\(mark)b\(binding.index)@\(binding.offset)"
                }
                .joined(separator: " ")
            return "[\(index)] kernel=\(kernel) barrier=\(barrier) grid=\(grid) tg=\(tg) entry=\(step.metadata.entryIndex.map(String.init) ?? "-") tensor=\(step.metadata.weightTensorName ?? "nil") reads={\(reads)} writes={\(writes)} bindings=[\(bindings)]"
        }
    }

    private func describeBarrierPolicy(_ policy: MetalBarrierPolicy) -> String {
        switch policy {
        case .none:
            return "none"
        case .barrier(let visibility):
            return visibility == .device ? "device" : "none-visibility"
        }
    }

    private func manualRowMajorDenseProjection(
        input: [Float],
        access: STAFWeightBufferAccess,
        schemeIdentifier: QuantizationSchemeIdentifier,
        outputDimension: Int,
        inputDimension: Int
    ) throws -> [Float] {
        guard access.layout == .rowMajor else {
            throw ManualProjectionError.unsupportedLayout
        }
        guard input.count == inputDimension else {
            throw ManualProjectionError.inputCountMismatch
        }

        let basePointer = access.buffer.contents().advanced(by: access.offset)
        switch schemeIdentifier {
        case .bf16RowMajor:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: outputDimension * inputDimension)
            return (0..<outputDimension).map { row in
                let start = row * inputDimension
                return zip(0..<inputDimension, input).reduce(Float.zero) { partial, pair in
                    partial + Float(bitPattern: UInt32(pointer[start + pair.0]) << 16) * pair.1
                }
            }
        case .fp16RowMajor:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: outputDimension * inputDimension)
            return (0..<outputDimension).map { row in
                let start = row * inputDimension
                return zip(0..<inputDimension, input).reduce(Float.zero) { partial, pair in
                    partial + Float(Float16(bitPattern: pointer[start + pair.0])) * pair.1
                }
            }
        case .fp32RowMajor:
            let pointer = basePointer.bindMemory(to: Float.self, capacity: outputDimension * inputDimension)
            return (0..<outputDimension).map { row in
                let start = row * inputDimension
                return zip(0..<inputDimension, input).reduce(Float.zero) { partial, pair in
                    partial + pointer[start + pair.0] * pair.1
                }
            }
        default:
            throw ManualProjectionError.unsupportedScheme(schemeIdentifier)
        }
    }

    private func manualColumnMajorDenseProjection(
        input: [Float],
        access: STAFWeightBufferAccess,
        schemeIdentifier: QuantizationSchemeIdentifier,
        outputDimension: Int,
        inputDimension: Int
    ) throws -> [Float] {
        guard access.layout == .rowMajor else {
            throw ManualProjectionError.unsupportedLayout
        }
        guard input.count == inputDimension else {
            throw ManualProjectionError.inputCountMismatch
        }

        let basePointer = access.buffer.contents().advanced(by: access.offset)
        switch schemeIdentifier {
        case .bf16RowMajor:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: outputDimension * inputDimension)
            return (0..<outputDimension).map { row in
                zip(0..<inputDimension, input).reduce(Float.zero) { partial, pair in
                    partial + Float(bitPattern: UInt32(pointer[pair.0 * outputDimension + row]) << 16) * pair.1
                }
            }
        case .fp16RowMajor:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: outputDimension * inputDimension)
            return (0..<outputDimension).map { row in
                zip(0..<inputDimension, input).reduce(Float.zero) { partial, pair in
                    partial + Float(Float16(bitPattern: pointer[pair.0 * outputDimension + row])) * pair.1
                }
            }
        case .fp32RowMajor:
            let pointer = basePointer.bindMemory(to: Float.self, capacity: outputDimension * inputDimension)
            return (0..<outputDimension).map { row in
                zip(0..<inputDimension, input).reduce(Float.zero) { partial, pair in
                    partial + pointer[pair.0 * outputDimension + row] * pair.1
                }
            }
        default:
            throw ManualProjectionError.unsupportedScheme(schemeIdentifier)
        }
    }

    private func maxAbsoluteError(_ lhs: [Float], _ rhs: [Float]) -> Float {
        zip(lhs, rhs).reduce(0) { current, pair in
            max(current, abs(pair.0 - pair.1))
        }
    }

    private func manualSwiGLU(gate: [Float], up: [Float]) -> [Float] {
        zip(gate, up).map { gate, up in
            let activated = gate * (1 / (1 + exp(-gate)))
            return activated * up
        }
    }

    private func maxAbsoluteValue(_ values: [Float]) -> Float {
        values.reduce(0) { current, value in
            max(current, abs(value))
        }
    }

    private func maxAbsoluteWeightSample(
        access: STAFWeightBufferAccess,
        schemeIdentifier: QuantizationSchemeIdentifier,
        sampleCount: Int
    ) -> Float {
        let basePointer = access.buffer.contents().advanced(by: access.offset)
        switch schemeIdentifier {
        case .bf16RowMajor:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: sampleCount)
            return (0..<sampleCount).reduce(0) { current, index in
                max(current, abs(Float(bitPattern: UInt32(pointer[index]) << 16)))
            }
        case .fp16RowMajor:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: sampleCount)
            return (0..<sampleCount).reduce(0) { current, index in
                max(current, abs(Float(Float16(bitPattern: pointer[index]))))
            }
        case .fp32RowMajor:
            let pointer = basePointer.bindMemory(to: Float.self, capacity: sampleCount)
            return (0..<sampleCount).reduce(0) { current, index in
                max(current, abs(pointer[index]))
            }
        default:
            return .nan
        }
    }

    private func isolatedProjectionOutput(
        device: MTLDevice,
        input: [Float],
        access: STAFWeightBufferAccess,
        inputDimension: Int,
        outputDimension: Int,
        sequenceLength: Int = 1,
        activeRowIndex: Int = 0,
        bindSequenceLengthFromRuntimeBuffer: Bool = false,
        pipeline overridePipeline: MTLComputePipelineState? = nil
    ) throws -> [Float] {
        guard sequenceLength > 0 else {
            throw MetalCompilerError.deviceSetupFailed("isolated projection sequenceLength must be positive")
        }
        guard activeRowIndex >= 0, activeRowIndex < sequenceLength else {
            throw MetalCompilerError.deviceSetupFailed("isolated projection activeRowIndex out of range")
        }
        let inputBuffer = try #require(device.makeBuffer(
            length: inputDimension * sequenceLength * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        inputBuffer.contents().initializeMemory(
            as: UInt8.self,
            repeating: 0,
            count: inputDimension * sequenceLength * MemoryLayout<Float>.stride
        )
        let inputBase = inputBuffer.contents().bindMemory(
            to: Float.self,
            capacity: inputDimension * sequenceLength
        )
        for (index, value) in input.enumerated() where index < inputDimension {
            inputBase[activeRowIndex * inputDimension + index] = value
        }
        let outputBuffer = try #require(device.makeBuffer(
            length: outputDimension * sequenceLength * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        outputBuffer.contents().initializeMemory(
            as: UInt8.self,
            repeating: 0,
            count: outputDimension * sequenceLength * MemoryLayout<Float>.stride
        )
        let runtimeConstantBuffer: MTLBuffer?
        if bindSequenceLengthFromRuntimeBuffer {
            let buffer = try #require(device.makeBuffer(
                length: MemoryLayout<UInt32>.stride,
                options: .storageModeShared
            ))
            buffer.contents().storeBytes(of: UInt32(sequenceLength), as: UInt32.self)
            runtimeConstantBuffer = buffer
        } else {
            runtimeConstantBuffer = nil
        }

        let allocator = MetalConstantBindingAllocator(device: device)
        let bindings = try allocator.makeBindingTable(
            bufferBindings: [
                (index: 0, buffer: inputBuffer, offset: 0),
                (index: 1, buffer: access.buffer, offset: access.offset),
                (index: 2, buffer: outputBuffer, offset: 0),
            ],
            bytesBindings: [
                (index: 3, value: uint32Bytes(UInt32(inputDimension))),
                (index: 4, value: uint32Bytes(UInt32(outputDimension))),
                (index: 5, value: uint32Bytes(UInt32(sequenceLength))),
                (index: 6, value: uint32Bytes(UInt32(inputDimension))),
            ],
            argumentPolicy: .argumentTable
        )
        let residency = try MetalResidencyLease.required(
            device: device,
            label: "swift-lm.test.isolated-projection",
            buffers: [inputBuffer, access.buffer, outputBuffer]
                + (runtimeConstantBuffer.map { [$0] } ?? [])
                + bindings.ownedResidencyBuffers
        )

        let pipeline: MTLComputePipelineState
        if let overridePipeline {
            pipeline = overridePipeline
        } else {
            let source = MetalSourceGenerator.commonHeader + "\n\n"
                + MetalSourceGenerator.generateGEMM(
                    name: "isolated_projection_gemm_bf16_f32s",
                    bufferPrecision: .float32,
                    weightFormat: .bfloat16
                )
            let options = MTLCompileOptions()
            options.languageVersion = .version4_0
            let library = try device.makeLibrary(source: source, options: options)
            let function = try #require(library.makeFunction(name: "isolated_projection_gemm_bf16_f32s"))
            pipeline = try device.makeComputePipelineState(function: function)
        }

        let rowsPerThreadgroup = 2
        let simdWidth = max(pipeline.threadExecutionWidth, 1)
        let threads = min(simdWidth * rowsPerThreadgroup, pipeline.maxTotalThreadsPerThreadgroup)
        let gridSize = MTLSize(
            width: (outputDimension + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            height: sequenceLength,
            depth: 1
        )
        let threadgroupSize = MTLSize(width: threads, height: 1, depth: 1)

        var submission = try MetalSubmissionContext(device: device)
        try submission.withCompute(ephemeralResidency: residency) { encoder, argumentTable in
            bindings.bind(to: argumentTable)
            if let runtimeConstantBuffer {
                argumentTable.setAddress(runtimeConstantBuffer.gpuAddress, index: 5)
            }
            encoder.setArgumentTable(argumentTable)
            encoder.setComputePipelineState(pipeline)
            encoder.dispatchThreadgroups(
                threadgroupsPerGrid: gridSize,
                threadsPerThreadgroup: threadgroupSize
            )
        }

        let outputBase = outputBuffer.contents().bindMemory(
            to: Float.self,
            capacity: outputDimension * sequenceLength
        )
        let outputOffset = activeRowIndex * outputDimension
        return (0..<outputDimension).map { outputBase[outputOffset + $0] }
    }

    private func isolatedAliasedProjectionOutput(
        device: MTLDevice,
        input: [Float],
        outputOffset: Int,
        access: STAFWeightBufferAccess,
        inputDimension: Int,
        outputDimension: Int,
        sequenceLength: Int = 1,
        activeRowIndex: Int = 0,
        bindSequenceLengthFromRuntimeBuffer: Bool = false,
        pipeline overridePipeline: MTLComputePipelineState? = nil
    ) throws -> [Float] {
        guard sequenceLength > 0 else {
            throw MetalCompilerError.deviceSetupFailed("isolated aliased projection sequenceLength must be positive")
        }
        guard activeRowIndex >= 0, activeRowIndex < sequenceLength else {
            throw MetalCompilerError.deviceSetupFailed("isolated aliased projection activeRowIndex out of range")
        }
        let inputLength = inputDimension * sequenceLength * MemoryLayout<Float>.stride
        let outputLength = outputDimension * sequenceLength * MemoryLayout<Float>.stride
        let ioBufferLength = max(inputLength, outputOffset + outputLength)
        let ioBuffer = try #require(device.makeBuffer(
            length: ioBufferLength,
            options: .storageModeShared
        ))
        ioBuffer.contents().initializeMemory(
            as: UInt8.self,
            repeating: 0,
            count: ioBufferLength
        )
        let inputBase = ioBuffer.contents().bindMemory(
            to: Float.self,
            capacity: inputDimension * sequenceLength
        )
        for (index, value) in input.enumerated() where index < inputDimension {
            inputBase[activeRowIndex * inputDimension + index] = value
        }
        let runtimeConstantBuffer: MTLBuffer?
        if bindSequenceLengthFromRuntimeBuffer {
            let buffer = try #require(device.makeBuffer(
                length: MemoryLayout<UInt32>.stride,
                options: .storageModeShared
            ))
            buffer.contents().storeBytes(of: UInt32(sequenceLength), as: UInt32.self)
            runtimeConstantBuffer = buffer
        } else {
            runtimeConstantBuffer = nil
        }

        let allocator = MetalConstantBindingAllocator(device: device)
        let bindings = try allocator.makeBindingTable(
            bufferBindings: [
                (index: 0, buffer: ioBuffer, offset: 0),
                (index: 1, buffer: access.buffer, offset: access.offset),
                (index: 2, buffer: ioBuffer, offset: outputOffset),
            ],
            bytesBindings: [
                (index: 3, value: uint32Bytes(UInt32(inputDimension))),
                (index: 4, value: uint32Bytes(UInt32(outputDimension))),
                (index: 5, value: uint32Bytes(UInt32(sequenceLength))),
                (index: 6, value: uint32Bytes(UInt32(inputDimension))),
            ],
            argumentPolicy: .argumentTable
        )
        let residency = try MetalResidencyLease.required(
            device: device,
            label: "swift-lm.test.isolated-aliased-projection",
            buffers: [ioBuffer, access.buffer]
                + (runtimeConstantBuffer.map { [$0] } ?? [])
                + bindings.ownedResidencyBuffers
        )

        let pipeline: MTLComputePipelineState
        if let overridePipeline {
            pipeline = overridePipeline
        } else {
            let source = MetalSourceGenerator.commonHeader + "\n\n"
                + MetalSourceGenerator.generateGEMM(
                    name: "isolated_aliased_projection_gemm_bf16_f32s",
                    bufferPrecision: .float32,
                    weightFormat: .bfloat16
                )
            let options = MTLCompileOptions()
            options.languageVersion = .version4_0
            let library = try device.makeLibrary(source: source, options: options)
            let function = try #require(library.makeFunction(name: "isolated_aliased_projection_gemm_bf16_f32s"))
            pipeline = try device.makeComputePipelineState(function: function)
        }

        let rowsPerThreadgroup = 2
        let simdWidth = max(pipeline.threadExecutionWidth, 1)
        let threads = min(simdWidth * rowsPerThreadgroup, pipeline.maxTotalThreadsPerThreadgroup)
        let gridSize = MTLSize(
            width: (outputDimension + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            height: sequenceLength,
            depth: 1
        )
        let threadgroupSize = MTLSize(width: threads, height: 1, depth: 1)

        var submission = try MetalSubmissionContext(device: device)
        try submission.withCompute(ephemeralResidency: residency) { encoder, argumentTable in
            bindings.bind(to: argumentTable)
            if let runtimeConstantBuffer {
                argumentTable.setAddress(runtimeConstantBuffer.gpuAddress, index: 5)
            }
            encoder.setArgumentTable(argumentTable)
            encoder.setComputePipelineState(pipeline)
            encoder.dispatchThreadgroups(
                threadgroupsPerGrid: gridSize,
                threadsPerThreadgroup: threadgroupSize
            )
        }

        let outputPointer = ioBuffer.contents()
            .advanced(by: outputOffset + activeRowIndex * outputDimension * MemoryLayout<Float>.stride)
            .bindMemory(to: Float.self, capacity: outputDimension)
        return (0..<outputDimension).map { outputPointer[$0] }
    }

    private func isolatedActualLayoutProjectionOutput(
        device: MTLDevice,
        input: [Float],
        inputOffset: Int,
        outputOffset: Int,
        access: STAFWeightBufferAccess,
        inputDimension: Int,
        outputDimension: Int,
        sequenceLength: Int,
        activeRowIndex: Int,
        bindSequenceLengthFromRuntimeBuffer: Bool,
        pipeline overridePipeline: MTLComputePipelineState? = nil
    ) throws -> [Float] {
        guard sequenceLength > 0 else {
            throw MetalCompilerError.deviceSetupFailed("isolated actual-layout projection sequenceLength must be positive")
        }
        guard activeRowIndex >= 0, activeRowIndex < sequenceLength else {
            throw MetalCompilerError.deviceSetupFailed("isolated actual-layout projection activeRowIndex out of range")
        }
        let inputLength = inputDimension * sequenceLength * MemoryLayout<Float>.stride
        let outputLength = outputDimension * sequenceLength * MemoryLayout<Float>.stride
        let ioBufferLength = max(inputOffset + inputLength, outputOffset + outputLength)
        let ioBuffer = try #require(device.makeBuffer(
            length: ioBufferLength,
            options: .storageModeShared
        ))
        ioBuffer.contents().initializeMemory(
            as: UInt8.self,
            repeating: 0,
            count: ioBufferLength
        )
        let inputPointer = ioBuffer.contents()
            .advanced(by: inputOffset + activeRowIndex * inputDimension * MemoryLayout<Float>.stride)
            .bindMemory(to: Float.self, capacity: inputDimension)
        for (index, value) in input.enumerated() where index < inputDimension {
            inputPointer[index] = value
        }
        let runtimeConstantBuffer: MTLBuffer?
        if bindSequenceLengthFromRuntimeBuffer {
            let buffer = try #require(device.makeBuffer(
                length: MemoryLayout<UInt32>.stride,
                options: .storageModeShared
            ))
            buffer.contents().storeBytes(of: UInt32(sequenceLength), as: UInt32.self)
            runtimeConstantBuffer = buffer
        } else {
            runtimeConstantBuffer = nil
        }

        let allocator = MetalConstantBindingAllocator(device: device)
        let bindings = try allocator.makeBindingTable(
            bufferBindings: [
                (index: 0, buffer: ioBuffer, offset: inputOffset),
                (index: 1, buffer: access.buffer, offset: access.offset),
                (index: 2, buffer: ioBuffer, offset: outputOffset),
            ],
            bytesBindings: [
                (index: 3, value: uint32Bytes(UInt32(inputDimension))),
                (index: 4, value: uint32Bytes(UInt32(outputDimension))),
                (index: 5, value: uint32Bytes(UInt32(sequenceLength))),
                (index: 6, value: uint32Bytes(UInt32(inputDimension))),
            ],
            argumentPolicy: .argumentTable
        )
        let residency = try MetalResidencyLease.required(
            device: device,
            label: "swift-lm.test.isolated-actual-layout-projection",
            buffers: [ioBuffer, access.buffer]
                + (runtimeConstantBuffer.map { [$0] } ?? [])
                + bindings.ownedResidencyBuffers
        )

        let pipeline: MTLComputePipelineState
        if let overridePipeline {
            pipeline = overridePipeline
        } else {
            let source = MetalSourceGenerator.commonHeader + "\n\n"
                + MetalSourceGenerator.generateGEMM(
                    name: "isolated_actual_layout_projection_gemm_bf16_f32s",
                    bufferPrecision: .float32,
                    weightFormat: .bfloat16
                )
            let options = MTLCompileOptions()
            options.languageVersion = .version4_0
            let library = try device.makeLibrary(source: source, options: options)
            let function = try #require(library.makeFunction(name: "isolated_actual_layout_projection_gemm_bf16_f32s"))
            pipeline = try device.makeComputePipelineState(function: function)
        }

        let rowsPerThreadgroup = 2
        let simdWidth = max(pipeline.threadExecutionWidth, 1)
        let threads = min(simdWidth * rowsPerThreadgroup, pipeline.maxTotalThreadsPerThreadgroup)
        let gridSize = MTLSize(
            width: (outputDimension + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            height: sequenceLength,
            depth: 1
        )
        let threadgroupSize = MTLSize(width: threads, height: 1, depth: 1)

        var submission = try MetalSubmissionContext(device: device)
        try submission.withCompute(ephemeralResidency: residency) { encoder, argumentTable in
            bindings.bind(to: argumentTable)
            if let runtimeConstantBuffer {
                argumentTable.setAddress(runtimeConstantBuffer.gpuAddress, index: 5)
            }
            encoder.setArgumentTable(argumentTable)
            encoder.setComputePipelineState(pipeline)
            encoder.dispatchThreadgroups(
                threadgroupsPerGrid: gridSize,
                threadsPerThreadgroup: threadgroupSize
            )
        }

        let outputPointer = ioBuffer.contents()
            .advanced(by: outputOffset + activeRowIndex * outputDimension * MemoryLayout<Float>.stride)
            .bindMemory(to: Float.self, capacity: outputDimension)
        return (0..<outputDimension).map { outputPointer[$0] }
    }

    private func forcedBufferBarrierPlan(from plan: MetalPrefillPlan) -> MetalPrefillPlan {
        planWithBarrierPolicy(
            plan,
            stepIndices: Set(plan.steps.indices),
            barrierPolicy: .bufferBarrier
        )
    }

    private func planWithBarrierPolicy(
        _ plan: MetalPrefillPlan,
        stepIndices: Set<Int>,
        barrierPolicy: MetalBarrierPolicy
    ) -> MetalPrefillPlan {
        let steps = plan.steps.map { step in
            guard stepIndices.contains(step.metadata.entryIndex ?? -1) == false else { return step }
            return step
        }
        let remappedSteps = steps.enumerated().map { index, step in
            guard stepIndices.contains(index) else { return step }
            let descriptor = MetalDispatchDescriptor(
                pipeline: step.pipeline,
                gridSize: step.gridSize,
                threadgroupSize: step.threadgroupSize,
                threadgroupMemoryLength: step.threadgroupMemoryLength,
                barrierPolicy: barrierPolicy
            )
            return MetalPrefillStep(
                descriptor: descriptor,
                bindings: step.bindings,
                mode: step.mode,
                sequenceLengthPolicy: step.sequenceLengthPolicy,
                positionBufferIndex: step.positionBufferIndex,
                perPositionStrides: step.perPositionStrides,
                metadata: step.metadata
            )
        }
        return MetalPrefillPlan(
            steps: remappedSteps,
            buffers: plan.buffers,
            slotDimension: plan.slotDimension,
            maximumSequenceLength: plan.maximumSequenceLength,
            stepCount: remappedSteps.count,
            usesMPP: plan.usesMPP,
            finalHiddenBuffer: plan.finalHiddenBuffer,
            finalHiddenBaseOffset: plan.finalHiddenBaseOffset,
            finalHiddenRowStride: plan.finalHiddenRowStride,
            supplementalResidencyBuffers: plan.supplementalResidencyBuffers
        )
    }

    private func readFloatBuffer(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return (0..<count).map { pointer[$0] }
    }

    private func uint32Bytes(_ value: UInt32) -> [UInt8] {
        withUnsafeBytes(of: value.littleEndian) { rawBuffer in
            Array(rawBuffer)
        }
    }

    #if ENABLE_METAL_PROBES
    private func chunkedVectorProbes(
        labelPrefix: String,
        stepIndex: Int? = nil,
        bindingIndex: Int,
        phase: MetalInferenceModel.DebugPrefillProbePhase,
        rowStride: Int,
        count: Int,
        rowIndex: Int? = nil,
        chunkSize: Int = 128
    ) -> [MetalInferenceModel.DebugPrefillBindingProbe] {
        stride(from: 0, to: count, by: chunkSize).map { elementOffset in
            .init(
                label: "\(labelPrefix)\(elementOffset)",
                stepIndex: stepIndex,
                bindingIndex: bindingIndex,
                phase: phase,
                rowIndex: rowIndex,
                elementOffset: elementOffset,
                rowStride: rowStride,
                count: min(chunkSize, count - elementOffset)
            )
        }
    }

    private func reconstructChunkedVector(
        from snapshots: [String: [Float]],
        prefix: String,
        count: Int,
        chunkSize: Int = 128
    ) -> [Float] {
        var values = Array(repeating: Float.zero, count: count)
        for elementOffset in stride(from: 0, to: count, by: chunkSize) {
            let key = "\(prefix)\(elementOffset)"
            guard let chunk = snapshots[key] else { continue }
            for (index, value) in chunk.enumerated() where elementOffset + index < values.count {
                values[elementOffset + index] = value
            }
        }
        return values
    }

    private func segment(
        from values: [Float],
        elementOffset: Int,
        count: Int
    ) -> [Float] {
        guard count > 0, elementOffset < values.count else { return [] }
        let end = min(values.count, elementOffset + count)
        return Array(values[elementOffset..<end])
    }

    private func bindingSegmentProbe(
        model: inout MetalInferenceModel,
        tokens: [Int32],
        stepIndex: Int,
        bindingIndex: Int,
        phase: MetalInferenceModel.DebugPrefillProbePhase = .afterStep,
        inputBindingIndex: Int? = nil,
        inputDimension: Int? = nil,
        outputDimension: Int,
        visibilityOptions: MTL4VisibilityOptions = [],
        stepVisibilityOptions: MTL4VisibilityOptions? = nil,
        probeVisibilityOptions: MTL4VisibilityOptions? = nil
    ) throws -> (head: [Float], mid: [Float], tail: [Float], maximum: Float) {
        let segmentCount = min(outputDimension, 32)
        let middleElementOffset = max((outputDimension / 2) - (segmentCount / 2), 0)
        let tailElementOffset = max(outputDimension - segmentCount, 0)
        var probes: [MetalInferenceModel.DebugPrefillBindingProbe] = []
        if let inputBindingIndex, let inputDimension, inputDimension > 0 {
            probes.append(
                .init(
                    label: "input-fence",
                    bindingIndex: inputBindingIndex,
                    phase: .beforeStep,
                    rowStride: inputDimension,
                    count: 1
                )
            )
        }
        probes.append(contentsOf: [
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
        ])
        let snapshots = try autoreleasepool {
            try model.debugPrefillBindingProbes(
                tokens: tokens,
                stepIndex: stepIndex,
                probes: probes,
                visibilityOptions: visibilityOptions,
                stepVisibilityOptions: stepVisibilityOptions,
                probeVisibilityOptions: probeVisibilityOptions
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

    private func convInProjSegmentProbe(
        model: inout MetalInferenceModel,
        tokens: [Int32],
        stepIndex: Int,
        inputDimension: Int? = nil,
        outputDimension: Int,
        includeBeforeStepInputProbe: Bool = false,
        visibilityOptions: MTL4VisibilityOptions = [],
        stepVisibilityOptions: MTL4VisibilityOptions? = nil,
        probeVisibilityOptions: MTL4VisibilityOptions? = nil
    ) throws -> (head: [Float], mid: [Float], tail: [Float], maximum: Float) {
        try bindingSegmentProbe(
            model: &model,
            tokens: tokens,
            stepIndex: stepIndex,
            bindingIndex: 2,
            inputBindingIndex: includeBeforeStepInputProbe ? 0 : nil,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            visibilityOptions: visibilityOptions,
            stepVisibilityOptions: stepVisibilityOptions,
            probeVisibilityOptions: probeVisibilityOptions
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

    private func rowMaxima(
        from snapshots: [String: [Float]],
        prefix: String,
        rowIndices: [Int]
    ) -> [Int: Float] {
        var maxima: [Int: Float] = [:]
        maxima.reserveCapacity(rowIndices.count)
        for rowIndex in rowIndices {
            let key = "\(prefix)\(rowIndex)"
            if let values = snapshots[key] {
                maxima[rowIndex] = maxAbsoluteValue(values)
            }
        }
        return maxima
    }
    #endif

    private func rowMaxima(
        from snapshots: [Int: [Float]],
        rowIndices: [Int]
    ) -> [Int: Float] {
        var maxima: [Int: Float] = [:]
        maxima.reserveCapacity(rowIndices.count)
        for rowIndex in rowIndices {
            if let values = snapshots[rowIndex] {
                maxima[rowIndex] = maxAbsoluteValue(values)
            }
        }
        return maxima
    }

    private func formatRowMaxima(_ maxima: [Int: Float]) -> String {
        maxima.keys.sorted().map { rowIndex in
            let value = maxima[rowIndex] ?? 0
            return "\(rowIndex):\(String(format: "%.4f", value))"
        }.joined(separator: ", ")
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

    private func describeEntry(_ entry: DispatchEntry) -> String {
        let layer = entry.layerIndex.map(String.init) ?? "-"
        let kind: String
        if let linear = entry.fragment as? LinearFragment {
            kind = "projection(field=\(linear.field),in=\(linear.inputDimension),out=\(linear.outputDimension),isOutput=\(linear.isOutput))"
        } else {
            kind = "fragment(\(type(of: entry.fragment)))"
        }
        let bindings = entry.parameterBindings
            .map { "\($0.role)=\($0.tensorName)" }
            .joined(separator: ", ")
        return "layer=\(layer) \(kind) bindings=[\(bindings)]"
    }
}

private protocol ProjectionDimensions {
    var inputDimension: Int { get }
    var outputDimension: Int { get }
}
extension LinearFragment: ProjectionDimensions {}
extension BatchedProjection.Entry: ProjectionDimensions {}

private enum ManualProjectionError: Error {
    case unsupportedLayout
    case unsupportedScheme(QuantizationSchemeIdentifier)
    case inputCountMismatch
}

private struct WeightAccessSummary: Equatable, CustomStringConvertible {
    let entry: String
    let tensorName: String
    let preferredLayout: STAFWeightLayout?
    let resolvedLayout: STAFWeightLayout?
    let resolvedOffset: Int?
    let formatIdentifier: QuantizationSchemeIdentifier?
    let bindingExists: Bool
    let hasResolvedAccess: Bool

    var description: String {
        "\(entry) tensor=\(tensorName) preferred=\(preferredLayout.map(String.init(describing:)) ?? "nil") resolved=\(resolvedLayout.map(String.init(describing:)) ?? "nil") offset=\(resolvedOffset.map(String.init) ?? "nil") format=\(formatIdentifier.map { String($0.rawValue) } ?? "nil") bound=\(bindingExists) access=\(hasResolvedAccess)"
    }
}
