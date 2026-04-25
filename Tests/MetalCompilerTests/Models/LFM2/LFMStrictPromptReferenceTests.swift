import Foundation
import Metal
import Testing
@testable import MetalCompiler

@Suite("LFM Strict Prompt Reference", .serialized)
struct LFMStrictPromptReferenceTests {
    private static let referencePath = URL(fileURLWithPath: BenchmarkSupport.testDataPath)
        .appendingPathComponent("lfm2_strict_chat_reference.safetensors")
        .path
    private static let stafPath = BenchmarkSupport.stafPath
    private static let modelDirectoryPath = BenchmarkSupport.lfmBundlePath

    private struct TestEnvironment {
        var model: MetalInferenceModel
        let ref: MetalWeightFile
    }

    @Test("Strict prompt prefill logits match HuggingFace reference", .timeLimit(.minutes(2)))
    func strictPromptPrefillLogitsMatchReference() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let env = try setupOrSkip()
        let tokens = try readInputTokens(env.ref)
        var model = env.model

        let firstToken = model.prefill(tokens: tokens)
        let refLogits = try readRefTensorAsFloats(env.ref, name: "ref.prefill.logits_last")
        let metalLogits = readF32Buffer(try #require(model.prefillPlan).buffers.logits)
        let metalTop = argmax(metalLogits)
        let refTop = argmax(refLogits)

        print("[StrictRef] firstToken metal=\(firstToken)")
        print("[StrictRef] prefill argmax metal=\(metalTop.index) python=\(refTop.index)")
        print("[StrictRef] prefill top-10 metal=\(formatTopK(topK(metalLogits, k: 10)))")
        print("[StrictRef] prefill top-10 python=\(formatTopK(topK(refLogits, k: 10)))")

        let prefillPlan = try #require(model.prefillPlan)
        for (index, step) in prefillPlan.steps.prefix(20).enumerated() {
            let kernel = step.metadata.kernelName ?? step.pipeline.label ?? "unknown"
            print("[StrictRef] step[\(index)] layer=\(step.metadata.layerIndex.map(String.init) ?? "-") kernel=\(kernel) mode=\(step.mode)")
        }
        for (index, step) in prefillPlan.steps.suffix(20).enumerated() {
            let absoluteIndex = prefillPlan.steps.count - 20 + index
            let kernel = step.metadata.kernelName ?? step.pipeline.label ?? "unknown"
            print("[StrictRef] tailStep[\(absoluteIndex)] layer=\(step.metadata.layerIndex.map(String.init) ?? "-") kernel=\(kernel) mode=\(step.mode)")
        }
        let finalHiddenSource = prefillPlan.finalHiddenSource(sequenceLength: tokens.count)
        print("[StrictRef] prefill stepCount=\(prefillPlan.stepCount)")
        print("[StrictRef] finalHiddenSource buffer=\(bufferLabel(finalHiddenSource.buffer, prefillPlan: prefillPlan)) offset=\(finalHiddenSource.offset)")

        #if ENABLE_METAL_PROBES
        let metalFinal = try model.debugPrefillLastTokenFinalHidden(tokens: tokens)
        let refFinalAll = try readRefTensorAsFloats(env.ref, name: "ref.prefill.final_hidden")
        let refFinal = Array(refFinalAll.suffix(2048))
        let finalErr = maxAbsoluteError(metalFinal, refFinal)
        print("[StrictRef] prefill final_hidden maxErr=\(String(format: "%.4f", finalErr))")

        let allCheckpoints: [(name: String, stepCount: Int, threshold: Float)] = [
            ("ref.prefill.layer_0.after_op", 7, 0.25),
            ("ref.prefill.layer_0.mlp_out", 13, 0.50),
            ("ref.prefill.layer_0.after_mlp", 14, 0.50),
            ("ref.prefill.layer_2.after_op", 40, 0.50),
            ("ref.prefill.layer_5.after_op", 87, 0.50),
            ("ref.prefill.layer_8.after_op", 134, 0.50),
            ("ref.prefill.layer_8.after_mlp", 141, 0.50),
            ("ref.prefill.layer_9.after_op", 148, 0.50),
            ("ref.prefill.layer_9.after_mlp", 155, 0.50),
            ("ref.prefill.layer_10.after_op", 167, 0.50),
            ("ref.prefill.layer_10.after_mlp", 174, 0.50),
            ("ref.prefill.layer_11.after_op", 181, 0.50),
            ("ref.prefill.layer_11.after_mlp", 188, 0.50),
            ("ref.prefill.layer_12.after_op", 200, 0.50),
            ("ref.prefill.layer_12.after_mlp", 207, 0.50),
            ("ref.prefill.layer_13.after_op", 214, 0.50),
            ("ref.prefill.layer_13.after_mlp", 221, 0.50),
            ("ref.prefill.layer_14.after_op", 233, 0.50),
            ("ref.prefill.layer_14.after_mlp", 240, 0.50),
            ("ref.prefill.layer_15.after_op", 247, 0.50),
            ("ref.prefill.layer_15.after_mlp", 254, 0.50),
        ]
        let checkpoints = allCheckpoints.filter { $0.stepCount < prefillPlan.steps.count }
        let skippedCheckpoints = allCheckpoints.filter { $0.stepCount >= prefillPlan.steps.count }
        if !skippedCheckpoints.isEmpty {
            print("[StrictRef] skipping \(skippedCheckpoints.count) stale checkpoints for \(prefillPlan.stepCount)-step prefill plan")
        }

        let snapshots = try model.debugPrefillLastTokenHiddenSnapshots(
            tokens: tokens,
            stepIndices: Set(checkpoints.map(\.stepCount))
        )
        for checkpoint in checkpoints {
            let metal = try #require(snapshots[checkpoint.stepCount], "Missing step snapshot \(checkpoint.stepCount)")
            let referenceAll = try readRefTensorAsFloats(env.ref, name: checkpoint.name)
            let reference = Array(referenceAll.suffix(2048))
            let error = maxAbsoluteError(metal, reference)
            print("[StrictRef] \(checkpoint.name) maxErr=\(String(format: "%.4f", error))")
            #expect(error < checkpoint.threshold, "\(checkpoint.name) diverged: maxErr=\(error)")
        }
        #expect(finalErr < 0.25, "Strict prompt final hidden diverged: maxErr=\(finalErr)")
        #endif

        #expect(firstToken == Int32(refTop.index), "Strict prompt first token mismatch")
        #expect(metalTop.index == refTop.index, "Strict prompt prefill argmax mismatch")
    }

    @Test("Layer13 dense MLP STAF rows match safetensors", .timeLimit(.minutes(2)))
    func layer13DenseMLPSTAFRowsMatchSafetensors() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let device = try #require(MTLCreateSystemDefaultDevice())
        let modelDirectory = URL(fileURLWithPath: Self.modelDirectoryPath)
        let safetensorURLs = try FileManager.default.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: nil
        )
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
        let safetensors = try SafetensorsLoader().loadAll(urls: safetensorURLs, device: device)
        let stafStore = try STAFLoader().load(
            at: modelDirectory.appendingPathComponent("model.staf"),
            device: device
        )

        let tensorNames = [
            "model.layers.13.feed_forward.w1.weight",
            "model.layers.13.feed_forward.w2.weight",
            "model.layers.13.feed_forward.w3.weight",
        ]

        for tensorName in tensorNames {
            let safetensorsTensor = try #require(safetensors.tensor(for: tensorName))
            let rowCount = try #require(safetensorsTensor.shape.first)
            let columnCount = try #require(safetensorsTensor.shape.dropFirst().first)
            let sampledRows = Array(Set([0, rowCount / 2, max(rowCount - 1, 0)])).sorted()
            for rowIndex in sampledRows {
                let safetensorsRow = try readTensorRow(
                    tensor: safetensorsTensor,
                    rowIndex: rowIndex,
                    columnCount: columnCount
                )
                let stafRow = try readSTAFRow(
                    tensorName: tensorName,
                    rowIndex: rowIndex,
                    columnCount: columnCount,
                    store: stafStore
                )
                let error = maxAbsoluteError(safetensorsRow, stafRow)
                print("[StrictRef] \(tensorName) row=\(rowIndex) maxErr=\(String(format: "%.6f", error))")
                #expect(error == 0, "\(tensorName) row \(rowIndex) diverged: maxErr=\(error)")
            }
        }
    }

    #if ENABLE_METAL_PROBES
    @Test("Layer13 MLP chain localizes execution drift", .timeLimit(.minutes(2)))
    func layer13MLPChainLocalizesExecutionDrift() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let env = try setupOrSkip()
        let tokens = try readInputTokens(env.ref)
        let device = try #require(MTLCreateSystemDefaultDevice())
        let safetensors = try loadModelSafetensors(device: device)
        let spec = try BenchmarkSupport.loadLFM25ModelSpec()
        var model = env.model

        let prefillPlan = try #require(model.prefillPlan)
        let hiddenRowStride = prefillPlan.buffers.hidden.length
            / MemoryLayout<Float>.stride
            / prefillPlan.maximumSequenceLength
        let scratchRowStride = prefillPlan.slotDimension
        let scratchSlotByteStride = prefillPlan.maximumSequenceLength
            * prefillPlan.slotDimension
            * MemoryLayout<Float>.stride

        let inputSnapshots = try model.debugPrefillLastTokenHiddenSnapshots(
            tokens: tokens,
            stepIndices: [214]
        )
        let input = try #require(inputSnapshots[214])
        let debugPassRanges = prefillPassRanges(for: prefillPlan.steps, within: 0..<(217))
        let debugPassRangeLabels = debugPassRanges.map { "\($0.lowerBound)..<\($0.upperBound)" }
        print("[StrictRef] debugPassRanges(0..<217)=\(debugPassRangeLabels)")
        let normStep = prefillPlan.steps[216]
        dumpStepBindings(normStep, stepIndex: 216, prefillPlan: prefillPlan)
        print("[StrictRef] step[216] barrierPolicy=\(String(describing: normStep.barrierPolicy))")
        let normScaleBinding = try #require(normStep.bindings.buffers.first(where: { $0.index == 1 }))
        let boundNormScale = readBufferSlice(
            buffer: normScaleBinding.buffer,
            offset: normScaleBinding.offset,
            count: 2048,
            precision: .bfloat16
        )
        let boundNormEpsilon = normStep.bytesBindings
            .first(where: { $0.index == 4 })
            .map { readFloatBinding($0.value) }
            ?? spec.config.normEps
        let normOut = try model.debugPrefillLastTokenBufferSnapshot(
            tokens: tokens,
            stepIndex: 216,
            buffer: prefillPlan.buffers.hidden,
            baseOffset: 0,
            rowStride: hiddenRowStride,
            count: 2048
        )
        let gateOut = try model.debugPrefillLastTokenBufferSnapshot(
            tokens: tokens,
            stepIndex: 217,
            buffer: prefillPlan.buffers.scratch,
            baseOffset: scratchSlotByteStride,
            rowStride: scratchRowStride,
            count: 8192
        )
        let upOut = try model.debugPrefillLastTokenBufferSnapshot(
            tokens: tokens,
            stepIndex: 218,
            buffer: prefillPlan.buffers.scratch,
            baseOffset: 2 * scratchSlotByteStride,
            rowStride: scratchRowStride,
            count: 8192
        )
        let swigluOut = try model.debugPrefillLastTokenBufferSnapshot(
            tokens: tokens,
            stepIndex: 219,
            buffer: prefillPlan.buffers.scratch,
            baseOffset: 0,
            rowStride: scratchRowStride,
            count: 8192
        )
        let downOut = try model.debugPrefillLastTokenBufferSnapshot(
            tokens: tokens,
            stepIndex: 220,
            buffer: prefillPlan.buffers.scratch,
            baseOffset: scratchSlotByteStride,
            rowStride: scratchRowStride,
            count: 2048
        )

        let normScale = try readTensorVector(
            try #require(safetensors.tensor(for: "model.layers.13.ffn_norm.weight"))
        )
        let gateTensor = try #require(safetensors.tensor(for: "model.layers.13.feed_forward.w1.weight"))
        let upTensor = try #require(safetensors.tensor(for: "model.layers.13.feed_forward.w3.weight"))
        let downTensor = try #require(safetensors.tensor(for: "model.layers.13.feed_forward.w2.weight"))

        let normReference = manualRMSNorm(
            input: input,
            scale: normScale,
            epsilon: spec.config.normEps
        )
        let boundNormReference = manualRMSNorm(
            input: input,
            scale: boundNormScale,
            epsilon: boundNormEpsilon
        )
        let gateReference = try manualDenseProjection(
            input: normReference,
            tensor: gateTensor,
            outputDimension: 8192,
            inputDimension: 2048
        )
        let upReference = try manualDenseProjection(
            input: normReference,
            tensor: upTensor,
            outputDimension: 8192,
            inputDimension: 2048
        )
        let swigluReference = manualSwiGLU(gate: gateReference, up: upReference)
        let downReference = try manualDenseProjection(
            input: swigluReference,
            tensor: downTensor,
            outputDimension: 2048,
            inputDimension: 8192
        )

        let normErr = maxAbsoluteError(normOut, normReference)
        let normErrFromBoundScale = maxAbsoluteError(normOut, boundNormReference)
        let normScaleErr = maxAbsoluteError(boundNormScale, normScale)
        let gateErr = maxAbsoluteError(gateOut, gateReference)
        let upErr = maxAbsoluteError(upOut, upReference)
        let swigluErr = maxAbsoluteError(swigluOut, swigluReference)
        let downErr = maxAbsoluteError(downOut, downReference)

        print("[StrictRef] layer13 normErr=\(String(format: "%.6f", normErr))")
        print("[StrictRef] layer13 normErr(boundScale)=\(String(format: "%.6f", normErrFromBoundScale))")
        print("[StrictRef] layer13 normScaleErr=\(String(format: "%.6f", normScaleErr))")
        print("[StrictRef] layer13 gateErr=\(String(format: "%.6f", gateErr))")
        print("[StrictRef] layer13 upErr=\(String(format: "%.6f", upErr))")
        print("[StrictRef] layer13 swigluErr=\(String(format: "%.6f", swigluErr))")
        print("[StrictRef] layer13 downErr=\(String(format: "%.6f", downErr))")

        #expect(normErr < 0.01, "layer13 ffn_norm drifted: maxErr=\(normErr)")
        #expect(gateErr < 0.05, "layer13 gate_proj drifted: maxErr=\(gateErr)")
        #expect(upErr < 0.05, "layer13 up_proj drifted: maxErr=\(upErr)")
        #expect(swigluErr < 0.05, "layer13 swiglu drifted: maxErr=\(swigluErr)")
        #expect(downErr < 0.05, "layer13 down_proj drifted: maxErr=\(downErr)")
    }

    @Test("Layer13 ffn_norm scale binding matches safetensors", .timeLimit(.minutes(2)))
    func layer13FFNNormScaleBindingMatchesSafetensors() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let env = try setupOrSkip()
        let device = try #require(MTLCreateSystemDefaultDevice())
        let safetensors = try loadModelSafetensors(device: device)
        let prefillPlan = try #require(env.model.prefillPlan)
        let normStep = prefillPlan.steps[216]
        let normScaleBinding = try #require(normStep.bindings.buffers.first(where: { $0.index == 1 }))
        let boundNormScale = readBufferSlice(
            buffer: normScaleBinding.buffer,
            offset: normScaleBinding.offset,
            count: 2048,
            precision: .bfloat16
        )
        let referenceScale = try readTensorVector(
            try #require(safetensors.tensor(for: "model.layers.13.ffn_norm.weight"))
        )
        let scaleErr = maxAbsoluteError(boundNormScale, referenceScale)
        print("[StrictRef] layer13 ffn_norm scale binding maxErr=\(String(format: "%.6f", scaleErr))")
        #expect(scaleErr < 0.0001, "layer13 ffn_norm scale binding drifted: maxErr=\(scaleErr)")
    }

    @Test("Layer13 ffn_norm manual separate-output dispatch matches reference", .timeLimit(.minutes(2)))
    func layer13FFNNormManualSeparateOutputDispatchMatchesReference() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let env = try setupOrSkip()
        let tokens = try readInputTokens(env.ref)
        let device = try #require(MTLCreateSystemDefaultDevice())
        let safetensors = try loadModelSafetensors(device: device)
        let spec = try BenchmarkSupport.loadLFM25ModelSpec()
        var model = env.model

        let prefillPlan = try #require(model.prefillPlan)
        let normStep = prefillPlan.steps[216]
        let hiddenRowStride = prefillPlan.buffers.hidden.length
            / MemoryLayout<Float>.stride
            / prefillPlan.maximumSequenceLength
        let outputLength = prefillPlan.maximumSequenceLength
            * hiddenRowStride
            * MemoryLayout<Float>.stride
        let outputBuffer = try #require(
            device.makeBuffer(length: outputLength, options: .storageModeShared)
        )
        let manualOutput = try model.debugPrefillLastTokenBufferSnapshotManualDispatch(
            tokens: tokens,
            prefixThroughStepIndex: 215,
            pipeline: normStep.pipeline,
            gridSize: normStep.gridSize,
            threadgroupSize: normStep.threadgroupSize,
            threadgroupMemoryLength: normStep.threadgroupMemoryLength,
            bufferBindings: normStep.bufferBindings.map { binding in
                if binding.index == 2 {
                    return (index: binding.index, buffer: outputBuffer, offset: 0)
                }
                return binding
            },
            bytesBindings: residentBytesBindings(from: normStep.bindings),
            runtimeSequenceLengthBindingIndex: normStep.sequenceLengthPolicy.bindingIndex,
            outputBuffer: outputBuffer,
            outputBaseOffset: 0,
            outputRowStride: hiddenRowStride,
            count: 2048
        )
        let splitPassHiddenOutput = try model.debugPrefillLastTokenBufferSnapshotSplitPass(
            tokens: tokens,
            prefixThroughStepIndex: 215,
            isolatedStepIndex: 216,
            buffer: prefillPlan.buffers.hidden,
            baseOffset: 0,
            rowStride: hiddenRowStride,
            count: 2048
        )
        let inputSnapshots = try model.debugPrefillLastTokenHiddenSnapshots(
            tokens: tokens,
            stepIndices: [214]
        )
        let input = try #require(inputSnapshots[214])
        let normScale = try readTensorVector(
            try #require(safetensors.tensor(for: "model.layers.13.ffn_norm.weight"))
        )
        let reference = manualRMSNorm(
            input: input,
            scale: normScale,
            epsilon: spec.config.normEps
        )
        let manualErr = maxAbsoluteError(manualOutput, reference)
        let splitPassErr = maxAbsoluteError(splitPassHiddenOutput, reference)
        print("[StrictRef] layer13 manual separate-output normErr=\(String(format: "%.6f", manualErr))")
        print("[StrictRef] layer13 split-pass hidden-output normErr=\(String(format: "%.6f", splitPassErr))")
        #expect(manualErr < 0.01, "layer13 manual separate-output norm drifted: maxErr=\(manualErr)")
    }
    #endif

    private func setupOrSkip() throws -> TestEnvironment {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SetupError.noDevice
        }

        let refURL = URL(fileURLWithPath: Self.referencePath)
        guard FileManager.default.fileExists(atPath: refURL.path) else {
            throw SetupError.noReference
        }

        let stafURL = URL(fileURLWithPath: Self.stafPath)
        guard FileManager.default.fileExists(atPath: stafURL.path) else {
            throw SetupError.noSTAF
        }

        let ref = try SafetensorsLoader().load(at: refURL, device: device)
        let store = try STAFLoader().load(at: stafURL, device: device)
        let spec = try BenchmarkSupport.loadLFM25ModelSpec()
        let compiler = MetalInferenceCompiler()
        let decodePlan = try compiler.compile(
            graph: spec.resolved,
            hiddenSize: spec.config.hiddenSize,
            intermediateSize: spec.config.intermediateSize,
            vocabSize: spec.config.vocabSize,
            stafWeightStore: store,
            device: device
        )
        let prefillPlan = try compiler.compilePrefill(
            graph: spec.resolved,
            hiddenSize: spec.config.hiddenSize,
            intermediateSize: spec.config.intermediateSize,
            vocabSize: spec.config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store,
            sharedKVCache: decodePlan.buffers.kvCache,
            sharedConvState: decodePlan.buffers.convState,
            sharedConvStateDimension: decodePlan.buffers.convStateDimension,
            sharedConvStateKernelSize: decodePlan.buffers.convStateKernelSize,
            sharedRecurrentState: decodePlan.buffers.recurrentState,
            sharedRecurrentStateBytesPerLayer: decodePlan.buffers.recurrentStateBytesPerLayer,
            device: device
        )

        var model = try MetalInferenceModel(plan: decodePlan, device: device)
        model.prefillPlan = prefillPlan
        return TestEnvironment(model: model, ref: ref)
    }

    private func readInputTokens(_ file: MetalWeightFile) throws -> [Int32] {
        guard let info = file.tensors["ref.input_tokens"] else {
            throw SetupError.tensorNotFound("ref.input_tokens")
        }
        let count = info.shape.reduce(1, *)
        let pointer = (file.buffer.contents() + file.dataSectionOffset + info.dataOffset)
            .bindMemory(to: Int32.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func readRefTensorAsFloats(_ file: MetalWeightFile, name: String) throws -> [Float] {
        guard let info = file.tensors[name] else {
            throw SetupError.tensorNotFound(name)
        }
        let count = info.shape.reduce(1, *)
        let pointer = (file.buffer.contents() + file.dataSectionOffset + info.dataOffset)
            .bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(pointer[$0]) }
    }

    private func loadModelSafetensors(device: MTLDevice) throws -> MetalWeightStore {
        let modelDirectory = URL(fileURLWithPath: Self.modelDirectoryPath)
        let safetensorURLs = try FileManager.default.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: nil
        )
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
        return try SafetensorsLoader().loadAll(urls: safetensorURLs, device: device)
    }

    private func readTensorRow(
        tensor: MetalTensor,
        rowIndex: Int,
        columnCount: Int
    ) throws -> [Float] {
        guard tensor.shape.count >= 2 else {
            throw SetupError.tensorNotFound("Expected rank-2 tensor row access")
        }
        let rowCount = tensor.shape[0]
        guard rowIndex >= 0, rowIndex < rowCount else {
            throw SetupError.tensorNotFound("Tensor row index out of bounds")
        }
        let basePointer = tensor.buffer.contents().advanced(by: tensor.offset)
        let start = rowIndex * columnCount
        switch tensor.dtype {
        case .float16:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: tensor.shape.reduce(1, *))
            return (0..<columnCount).map { index in
                Float(Float16(bitPattern: pointer[start + index]))
            }
        case .bfloat16:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: tensor.shape.reduce(1, *))
            return (0..<columnCount).map { index in
                Float(bitPattern: UInt32(pointer[start + index]) << 16)
            }
        case .float32:
            let pointer = basePointer.bindMemory(to: Float.self, capacity: tensor.shape.reduce(1, *))
            return (0..<columnCount).map { index in
                pointer[start + index]
            }
        case .quantized:
            throw SetupError.tensorNotFound("Quantized safetensors row access unsupported")
        }
    }

    private func readTensorVector(_ tensor: MetalTensor) throws -> [Float] {
        let count = tensor.shape.reduce(1, *)
        let basePointer = tensor.buffer.contents().advanced(by: tensor.offset)
        switch tensor.dtype {
        case .float16:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: count)
            return (0..<count).map { index in
                Float(Float16(bitPattern: pointer[index]))
            }
        case .bfloat16:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: count)
            return (0..<count).map { index in
                Float(bitPattern: UInt32(pointer[index]) << 16)
            }
        case .float32:
            let pointer = basePointer.bindMemory(to: Float.self, capacity: count)
            return (0..<count).map { index in
                pointer[index]
            }
        case .quantized:
            throw SetupError.tensorNotFound("Quantized safetensors vector access unsupported")
        }
    }

    #if ENABLE_METAL_PROBES
    private func residentBytesBindings(
        from table: MetalBindingTable
    ) -> [(index: Int, value: [UInt8])] {
        table.constants.compactMap { constant in
            guard case .buffer(let binding) = constant else {
                return nil
            }
            let base = binding.buffer.contents().advanced(by: binding.offset)
            let bytes = Array(
                UnsafeBufferPointer(
                    start: base.assumingMemoryBound(to: UInt8.self),
                    count: binding.length
                )
            )
            return (index: binding.index, value: bytes)
        }
        .sorted { $0.index < $1.index }
    }
    #endif

    private func readBufferSlice(
        buffer: MTLBuffer,
        offset: Int,
        count: Int,
        precision: BufferPrecision
    ) -> [Float] {
        switch precision {
        case .float16:
            let pointer = (buffer.contents() + offset).bindMemory(to: Float16.self, capacity: count)
            return (0..<count).map { index in
                Float(pointer[index])
            }
        case .bfloat16:
            let pointer = (buffer.contents() + offset).bindMemory(to: BFloat16.self, capacity: count)
            return (0..<count).map { index in
                Float(pointer[index])
            }
        case .float32:
            let pointer = (buffer.contents() + offset).bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: pointer, count: count))
        }
    }

    private func readFloatBinding(_ bytes: [UInt8]) -> Float {
        precondition(bytes.count == MemoryLayout<Float>.size)
        return bytes.withUnsafeBytes { $0.load(as: Float.self) }
    }

    private func readSTAFRow(
        tensorName: String,
        rowIndex: Int,
        columnCount: Int,
        store: STAFWeightStore
    ) throws -> [Float] {
        guard let entry = store.entries[tensorName],
              let access = store.bufferAccess(for: tensorName),
              let format = QuantizationFormatRegistry.format(for: entry.schemeIdentifier) else {
            throw SetupError.tensorNotFound(tensorName)
        }
        let start = rowIndex * columnCount
        let basePointer = access.buffer.contents().advanced(by: access.offset)
        switch format.schemeIdentifier {
        case .fp16RowMajor:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: entry.shape.reduce(1, *))
            return (0..<columnCount).map { index in
                Float(Float16(bitPattern: pointer[start + index]))
            }
        case .bf16RowMajor:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: entry.shape.reduce(1, *))
            return (0..<columnCount).map { index in
                Float(bitPattern: UInt32(pointer[start + index]) << 16)
            }
        case .fp32RowMajor:
            let pointer = basePointer.bindMemory(to: Float.self, capacity: entry.shape.reduce(1, *))
            return (0..<columnCount).map { index in
                pointer[start + index]
            }
        default:
            throw SetupError.tensorNotFound("Unsupported STAF format for \(tensorName)")
        }
    }

    private func readF32Buffer(_ buffer: MTLBuffer) -> [Float] {
        let count = buffer.length / MemoryLayout<Float>.stride
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func readDecodeBuffer(_ buffer: MTLBuffer, precision: BufferPrecision) -> [Float] {
        if buffer.storageMode == .private {
            let device = buffer.device
            guard let staging = device.makeBuffer(length: buffer.length, options: .storageModeShared),
                  let queue = device.makeCommandQueue(),
                  let commandBuffer = queue.makeCommandBuffer(),
                  let blit = commandBuffer.makeBlitCommandEncoder() else { return [] }
            blit.copy(from: buffer, sourceOffset: 0, to: staging, destinationOffset: 0, size: buffer.length)
            blit.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            return readDecodeBuffer(staging, precision: precision)
        }

        switch precision {
        case .float32:
            return readF32Buffer(buffer)
        case .float16:
            let count = buffer.length / MemoryLayout<Float16>.stride
            let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
            return (0..<count).map { Float(pointer[$0]) }
        case .bfloat16:
            let count = buffer.length / MemoryLayout<UInt16>.stride
            let pointer = buffer.contents().bindMemory(to: UInt16.self, capacity: count)
            return (0..<count).map { Float(bitPattern: UInt32(pointer[$0]) << 16) }
        }
    }

    private func argmax(_ values: [Float]) -> (index: Int, value: Float) {
        var bestIndex = 0
        var bestValue = values.first ?? -.infinity
        for (index, value) in values.enumerated() where value > bestValue {
            bestIndex = index
            bestValue = value
        }
        return (bestIndex, bestValue)
    }

    private func topK(_ values: [Float], k: Int) -> [(index: Int, value: Float)] {
        values.enumerated()
            .sorted { lhs, rhs in
                if lhs.element == rhs.element { return lhs.offset < rhs.offset }
                return lhs.element > rhs.element
            }
            .prefix(k)
            .map { ($0.offset, $0.element) }
    }

    private func formatTopK(_ values: [(index: Int, value: Float)]) -> [String] {
        values.map { entry in
            "(\(entry.index),\(String(format: "%.1f", entry.value)))"
        }
    }

    private func dumpStepBindings(
        _ step: MetalPrefillStep,
        stepIndex: Int,
        prefillPlan: MetalPrefillPlan
    ) {
        let kernel = step.metadata.kernelName ?? step.pipeline.label ?? "unknown"
        let bindings = step.bufferBindings.map { binding in
            "i\(binding.index)=\(bufferLabel(binding.buffer, prefillPlan: prefillPlan))@\(binding.offset)"
        }.joined(separator: ", ")
        print("[StrictRef] step[\(stepIndex)] \(kernel) bindings: \(bindings)")
    }

    private func bufferLabel(
        _ buffer: MTLBuffer,
        prefillPlan: MetalPrefillPlan
    ) -> String {
        if buffer === prefillPlan.buffers.hidden { return "hidden" }
        if buffer === prefillPlan.buffers.scratch { return "scratch" }
        if buffer === prefillPlan.buffers.residual { return "residual" }
        if buffer === prefillPlan.buffers.logits { return "logits" }
        if buffer === prefillPlan.buffers.tokenIDs { return "tokenIDs" }
        if buffer === prefillPlan.buffers.positions { return "positions" }
        if buffer === prefillPlan.buffers.runtimeConstantBuffer { return "runtimeConstant" }
        return "buffer"
    }

    private func maxAbsoluteError(_ lhs: [Float], _ rhs: [Float]) -> Float {
        let count = min(lhs.count, rhs.count)
        guard count > 0 else { return 0 }
        var maxErr: Float = 0
        for index in 0..<count {
            maxErr = max(maxErr, abs(lhs[index] - rhs[index]))
        }
        return maxErr
    }

    private func manualRMSNorm(
        input: [Float],
        scale: [Float],
        epsilon: Float
    ) -> [Float] {
        let meanSquare = input.reduce(Float.zero) { partial, value in
            partial + value * value
        } / Float(max(input.count, 1))
        let invRMS = 1 / sqrt(meanSquare + epsilon)
        return zip(input, scale).map { value, weight in
            value * invRMS * weight
        }
    }

    private func manualDenseProjection(
        input: [Float],
        tensor: MetalTensor,
        outputDimension: Int,
        inputDimension: Int
    ) throws -> [Float] {
        guard tensor.shape.count >= 2 else {
            throw SetupError.tensorNotFound("Expected rank-2 tensor projection access")
        }
        return try (0..<outputDimension).map { rowIndex in
            let row = try readTensorRow(
                tensor: tensor,
                rowIndex: rowIndex,
                columnCount: inputDimension
            )
            var sum: Float = 0
            for index in 0..<inputDimension {
                sum += row[index] * input[index]
            }
            return sum
        }
    }

    private func manualSwiGLU(gate: [Float], up: [Float]) -> [Float] {
        zip(gate, up).map { gate, up in
            let activated = gate * (1 / (1 + exp(-gate)))
            return activated * up
        }
    }

    private enum SetupError: Error {
        case noDevice
        case noReference
        case noSTAF
        case tensorNotFound(String)
    }
}
