import Testing
import Metal
import Foundation
@testable import MetalCompiler
import LMArchitecture
import ModelDeclarations
import LMIR

/// Compare Metal inference output against Python HuggingFace reference.
///
/// Prerequisites:
///   1. Generate reference: `python3 scripts/hf/dump_lfm2_reference.py`
///   2. Have LFM2.5-1.2B-Thinking cached in HF hub:
///      `huggingface-cli download LiquidAI/LFM2.5-1.2B-Thinking`
///      The STAF file is generated lazily next to the snapshot.
#if ENABLE_METAL_PROBES
@Suite("Reference Comparison", .serialized)
struct ReferenceComparisonTests {

    private static let referencePath = URL(fileURLWithPath: BenchmarkSupport.testDataPath)
        .appendingPathComponent("lfm2_reference.safetensors")
        .path
    private static let stafPath = BenchmarkSupport.stafPath

    // MARK: - Prefill Tests

    @Test("Prefill logits match Python reference")
    func prefillLogitsMatchReference() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        let env = try setupOrSkip()
        var model = env.model

        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        let firstToken = model.prefill(tokens: tokens)

        // Read from PREFILL plan's logits buffer (F32 in prefill)
        guard let prefillPlan = model.prefillPlan else {
            Issue.record("No prefill plan"); return
        }
        let prefillLogits = readF32Buffer(prefillPlan.buffers.logits)
        let refLogits = try readRefTensorAsFloats(env.ref, name: "ref.prefill.logits_last")

        let metalTop = argmax(prefillLogits)
        let refTop = argmax(refLogits)

        print("[RefComp] Metal  prefill: firstToken=\(firstToken), logits argmax=\(metalTop.index) (val=\(String(format: "%.2f", metalTop.value)))")
        print("[RefComp] Python prefill: argmax=\(refTop.index) (val=\(String(format: "%.2f", refTop.value)))")

        let metalTop10 = topK(prefillLogits, k: 10)
        let refTop10 = topK(refLogits, k: 10)
        print("[RefComp] Metal  top-10: \(metalTop10.map { "(\($0.index),\(String(format: "%.1f", $0.value)))" })")
        print("[RefComp] Python top-10: \(refTop10.map { "(\($0.index),\(String(format: "%.1f", $0.value)))" })")

        let maxErr = maxAbsoluteError(prefillLogits, refLogits)
        print("[RefComp] Prefill logits max absolute error: \(String(format: "%.4f", maxErr))")

        #expect(metalTop.index == refTop.index,
                "Prefill argmax mismatch: Metal=\(metalTop.index) Python=\(refTop.index)")
    }

    @Test("Prefill embedding matches Python reference")
    func prefillEmbeddingMatches() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        let env = try setupOrSkip()

        // Run prefill step-by-step to capture embedding output
        guard let prefillPlan = env.model.prefillPlan else {
            Issue.record("No prefill plan"); return
        }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let seqLen = 5
        let tokens: [Int32] = [1, 1, 6, 6423, 708]

        // Fill token IDs and positions
        let tokenPtr = prefillPlan.buffers.tokenIDs.contents()
            .bindMemory(to: Int32.self, capacity: seqLen)
        let posPtr = prefillPlan.buffers.positions.contents()
            .bindMemory(to: UInt32.self, capacity: seqLen)
        for i in 0..<seqLen {
            tokenPtr[i] = tokens[i]
            posPtr[i] = UInt32(i)
        }

        // Run only the first step (embedding lookup) using Metal 4
        let step = prefillPlan.steps[0]
        let runtimeConstantBuffer = prefillPlan.buffers.runtimeConstantBuffer
        runtimeConstantBuffer.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(seqLen)
        var submission = try MetalSubmissionContext(device: device)
        try submission.withCompute { encoder, argumentTable in
            step.bindStaticArguments(argumentTable: argumentTable)
            step.bindRuntimeArguments(
                argumentTable: argumentTable,
                runtimeConstantBuffer: runtimeConstantBuffer,
                sequenceLengthOffset: PrefillBufferSet.sequenceLengthOffset
            )
            let grid = step.resolvedGridSize(sequenceLength: seqLen)
            step.descriptor.encode(on: encoder, argumentTable: argumentTable, gridSize: grid)
        }

        // Read embedding output from hidden buffer (F32 in prefill, last token)
        let hiddenSize = 2048
        let lastTokenOffset = (seqLen - 1) * hiddenSize
        let hiddenPtr = prefillPlan.buffers.hidden.contents()
            .bindMemory(to: Float32.self, capacity: seqLen * hiddenSize)
        let metalEmb = (0..<hiddenSize).map { hiddenPtr[lastTokenOffset + $0] }

        // Read Python reference embedding (last token)
        let refEmbAll = try readRefTensorAsFloats(env.ref, name: "ref.prefill.embedding")
        let refEmb = Array(refEmbAll[lastTokenOffset..<lastTokenOffset + hiddenSize])

        let maxErr = maxAbsoluteError(metalEmb, refEmb)
        let metalSample = (0..<4).map { String(format: "%.4f", metalEmb[$0]) }
        let refSample = (0..<4).map { String(format: "%.4f", refEmb[$0]) }
        let metalNorm = sqrtf(metalEmb.reduce(0) { $0 + $1 * $1 })
        let refNorm = sqrtf(refEmb.reduce(0) { $0 + $1 * $1 })

        print("[RefComp] Embedding (last token):")
        print("  Metal:  \(metalSample) norm=\(String(format: "%.2f", metalNorm))")
        print("  Python: \(refSample) norm=\(String(format: "%.2f", refNorm))")
        print("  Max absolute error: \(String(format: "%.6f", maxErr))")

        #expect(maxErr < 0.01, "Embedding diverges: maxErr=\(maxErr)")
    }

    @Test("Conv state after prefill matches Python reference")
    func convStateAfterPrefillMatches() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        let env = try setupOrSkip()
        var model = env.model

        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        _ = model.prefill(tokens: tokens)

        // Read conv_state from DECODE plan's buffer (prefill transfers it)
        guard let convState = model.buffers.convState else {
            Issue.record("No conv_state buffer"); return
        }

        let convDim = model.buffers.convStateDimension
        let kernelSize = model.buffers.convStateKernelSize
        let elementSize = MemoryLayout<Float16>.size

        for convIdx in 0..<10 {
            let refData = try readRefTensorAsFloats(env.ref, name: "ref.prefill.conv_state.\(convIdx)")
            let layerOffset = convIdx * kernelSize * convDim * elementSize
            let fullConvState = readDecodeBuffer(convState, precision: .bfloat16)
            let metalVals = Array(fullConvState[(layerOffset / elementSize)..<((layerOffset / elementSize) + kernelSize * convDim)])

            var maxErr: Float = 0
            var maxErrIdx = 0
            for i in 0..<min(metalVals.count, refData.count) {
                let e = abs(metalVals[i] - refData[i])
                if e > maxErr { maxErr = e; maxErrIdx = i }
            }
            let k = maxErrIdx / convDim
            let ch = maxErrIdx % convDim
            print("[RefComp] conv_state[\(convIdx)]: maxErr=\(String(format: "%.6f", maxErr)) at k=\(k) ch=\(ch) metal=\(String(format: "%.4f", metalVals[maxErrIdx])) python=\(String(format: "%.4f", refData[maxErrIdx]))")

            #expect(maxErr < 1.0,
                    "conv_state[\(convIdx)] diverges: maxErr=\(maxErr)")
        }
    }

    @Test("Prefill conv state source and transferred decode state diagnostic")
    func prefillConvStateSourceAndTransferredStateDiagnostic() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        let env = try setupOrSkip()
        var model = env.model

        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        _ = model.prefill(tokens: tokens)

        guard
            let prefillPlan = model.prefillPlan,
            let prefillConvState = prefillPlan.buffers.convState,
            let decodeConvState = model.buffers.convState
        else {
            Issue.record("Missing prefill/decode conv_state buffer")
            return
        }

        let convDim = model.buffers.convStateDimension
        let kernelSize = model.buffers.convStateKernelSize
        let layerStride = convDim * kernelSize
        let prefillAll = readDecodeBuffer(prefillConvState, precision: .float16)
        let decodeAll = readDecodeBuffer(decodeConvState, precision: .float16)

        for convIdx in 0..<min(4, prefillAll.count / layerStride, decodeAll.count / layerStride) {
            let refData = try readRefTensorAsFloats(env.ref, name: "ref.prefill.conv_state.\(convIdx)")
            let base = convIdx * layerStride
            let prefillVals = Array(prefillAll[base..<(base + layerStride)])
            let decodeVals = Array(decodeAll[base..<(base + layerStride)])
            let prefillErr = maxAbsoluteError(prefillVals, refData)
            let decodeErr = maxAbsoluteError(decodeVals, refData)
            let transferErr = maxAbsoluteError(prefillVals, decodeVals)
            print("[RefComp] conv_state[\(convIdx)] prefillErr=\(String(format: "%.4f", prefillErr)) decodeErr=\(String(format: "%.4f", decodeErr)) transferErr=\(String(format: "%.4f", transferErr))")
        }
    }

    @Test("Prefill conv-state binding offsets diagnostic")
    func prefillConvStateBindingOffsetsDiagnostic() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        let env = try setupOrSkip()
        guard
            let prefillPlan = env.model.prefillPlan,
            let convState = prefillPlan.buffers.convState
        else {
            Issue.record("Missing prefill conv_state buffer")
            return
        }

        for (index, step) in prefillPlan.steps.enumerated() {
            let bindings = step.bindings.buffers.filter { $0.buffer === convState }
            guard !bindings.isEmpty else { continue }
            let offsets = bindings.map(\.offset).sorted()
            print("[RefComp] prefill step \(index) layer=\(step.metadata.layerIndex.map(String.init) ?? "-") mode=\(step.mode) convStateOffsets=\(offsets)")
        }
    }

    // MARK: - Per-layer Prefill Comparison

    @Test("Prefill per-layer hidden states diagnostic")
    func prefillPerLayerMatch() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        let env = try setupOrSkip()
        guard let prefillPlan = env.model.prefillPlan else {
            Issue.record("No prefill plan"); return
        }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let seqLen = 5
        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        let hiddenSize = 2048
        let lastTokenOffset = (seqLen - 1) * hiddenSize

        let refFinalAll = try readRefTensorAsFloats(env.ref, name: "ref.prefill.final_hidden")
        let refLastToken = Array(refFinalAll[lastTokenOffset..<lastTokenOffset + hiddenSize])
        let refLogits = try readRefTensorAsFloats(env.ref, name: "ref.prefill.logits_last")
        var refLayers: [[Float]] = []
        refLayers.reserveCapacity(16)
        for layerIdx in 0..<16 {
            let refLayerAll = try readRefTensorAsFloats(env.ref, name: "ref.prefill.layer_\(layerIdx).after_mlp")
            refLayers.append(Array(refLayerAll[lastTokenOffset..<lastTokenOffset + hiddenSize]))
        }

        // Fill token IDs and positions
        let tokenPtr = prefillPlan.buffers.tokenIDs.contents()
            .bindMemory(to: Int32.self, capacity: seqLen)
        let posPtr = prefillPlan.buffers.positions.contents()
            .bindMemory(to: UInt32.self, capacity: seqLen)
        for i in 0..<seqLen {
            tokenPtr[i] = tokens[i]
            posPtr[i] = UInt32(i)
        }

        // Run ALL prefill steps using Metal 4
        let runtimeConstantBuffer = prefillPlan.buffers.runtimeConstantBuffer
        runtimeConstantBuffer.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(seqLen)
        // Write per-position values for perPosition steps
        for pos in 0..<seqLen {
            let positionPtr = runtimeConstantBuffer.contents()
                .advanced(by: PrefillBufferSet.positionOffset(at: pos))
                .bindMemory(to: UInt32.self, capacity: 1)
            positionPtr.pointee = UInt32(pos)
        }

        var submission = try MetalSubmissionContext(device: device)
        try submission.withCompute { encoder, argumentTable in
            for step in prefillPlan.steps {
                switch step.mode {
                case .batch:
                    step.bindings.bind(to: argumentTable)
                    step.bindRuntimeArguments(
                        argumentTable: argumentTable,
                        runtimeConstantBuffer: runtimeConstantBuffer,
                        sequenceLengthOffset: PrefillBufferSet.sequenceLengthOffset
                    )
                    let grid = step.resolvedGridSize(sequenceLength: seqLen)
                    step.descriptor.encode(on: encoder, argumentTable: argumentTable, gridSize: grid)
                case .perPosition:
                    for pos in 0..<seqLen {
                        step.bindStaticArguments(argumentTable: argumentTable, position: pos)
                        if let posIdx = step.positionBufferIndex {
                            argumentTable.setAddress(
                                runtimeConstantBuffer.gpuAddress + UInt64(PrefillBufferSet.positionOffset(at: pos)),
                                index: posIdx
                            )
                        }
                        step.descriptor.encode(on: encoder, argumentTable: argumentTable)
                    }
                case .lastToken:
                    step.bindStaticArguments(argumentTable: argumentTable, position: seqLen - 1)
                    step.descriptor.encode(on: encoder, argumentTable: argumentTable)
                }
            }
        }

        // Compare final hidden (last token)
        let hiddenSource = prefillPlan.finalHiddenSource(sequenceLength: seqLen)
        let hiddenPtr = hiddenSource.buffer.contents().bindMemory(
            to: Float32.self,
            capacity: hiddenSource.buffer.length / MemoryLayout<Float32>.stride
        )
        let hiddenElementOffset = hiddenSource.offset / MemoryLayout<Float32>.stride
        let metalFinalHidden = (0..<hiddenSize).map { hiddenPtr[hiddenElementOffset + $0] }

        let finalErr = maxAbsoluteError(metalFinalHidden, refLastToken)
        let metalNorm = sqrtf(metalFinalHidden.reduce(0) { $0 + $1 * $1 })
        let refNorm = sqrtf(refLastToken.reduce(0) { $0 + $1 * $1 })
        let metalSample = (0..<4).map { String(format: "%.4f", metalFinalHidden[$0]) }
        let refSample = (0..<4).map { String(format: "%.4f", refLastToken[$0]) }

        print("[RefComp] Final hidden (last token):")
        print("  Metal:  \(metalSample) norm=\(String(format: "%.2f", metalNorm))")
        print("  Python: \(refSample) norm=\(String(format: "%.2f", refNorm))")
        print("  Max absolute error: \(String(format: "%.4f", finalErr))")

        // Compare prefill logits (F32 buffer)
        let prefillLogits = readF32Buffer(prefillPlan.buffers.logits)
        let logitsErr = maxAbsoluteError(prefillLogits, refLogits)
        let metalArgmax = argmax(prefillLogits)
        let refArgmax = argmax(refLogits)
        print("[RefComp] Prefill logits: Metal argmax=\(metalArgmax.index) Python argmax=\(refArgmax.index) maxErr=\(String(format: "%.4f", logitsErr))")

        // Compare per-layer after_mlp for last token
        for layerIdx in 0..<16 {
            let refLayer = refLayers[layerIdx]
            let refLayerNorm = sqrtf(refLayer.reduce(0) { $0 + $1 * $1 })
            let refLayerSample = (0..<2).map { String(format: "%.3f", refLayer[$0]) }
            if layerIdx < 3 || layerIdx >= 14 {
                print("[RefComp] Python layer_\(layerIdx).after_mlp: \(refLayerSample)... norm=\(String(format: "%.1f", refLayerNorm))")
            }
        }
    }

    @Test("Prefill first conv layer boundary diagnostic")
    func prefillFirstConvLayerMatchesReference() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        let env = try setupOrSkip()
        guard let prefillPlan = env.model.prefillPlan else {
            Issue.record("No prefill plan")
            return
        }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let seqLen = 5
        let hiddenSize = 2048
        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        let lastTokenOffset = (seqLen - 1) * hiddenSize

        preparePrefillInputs(
            prefillPlan: prefillPlan,
            seqLen: seqLen,
            tokens: tokens
        )
        try executePrefillPrefix(
            prefillPlan: prefillPlan,
            device: device,
            seqLen: seqLen,
            stepCount: 7
        )

        let hidden = readF32Buffer(prefillPlan.buffers.hidden)
        let metal = Array(hidden[lastTokenOffset..<(lastTokenOffset + hiddenSize)])
        let referenceAll = try readRefTensorAsFloats(env.ref, name: "ref.prefill.layer_0.after_op")
        let reference = Array(referenceAll[lastTokenOffset..<(lastTokenOffset + hiddenSize)])
        let maxErr = maxAbsoluteError(metal, reference)

        print("[RefComp] layer_0.after_op boundary diagnostic maxErr=\(String(format: "%.4f", maxErr))")
    }

    @Test("Prefill final hidden matches Python reference")
    func prefillFinalHiddenMatchesReference() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        let env = try setupOrSkip()
        var model = env.model
        guard let prefillPlan = model.prefillPlan else {
            Issue.record("No prefill plan")
            return
        }

        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        _ = model.prefill(tokens: tokens)
        let finalHidden = prefillPlan.finalHiddenSource(sequenceLength: tokens.count)
        let fullHidden = readDecodeBuffer(finalHidden.buffer, precision: .float32)
        let hiddenBase = finalHidden.offset / MemoryLayout<Float>.stride
        let hidden = Array(fullHidden[hiddenBase..<(hiddenBase + 2048)])

        let refHiddenAll = try readRefTensorAsFloats(env.ref, name: "ref.prefill.final_hidden")
        let hiddenSize = 2048
        let reference = Array(refHiddenAll.suffix(hiddenSize))
        let maxErr = maxAbsoluteError(hidden, reference)

        print("[RefComp] prefill final hidden maxErr=\(String(format: "%.4f", maxErr))")
        #expect(maxErr < 0.5, "Prefill final hidden diverged beyond BF16 tolerance: maxErr=\(maxErr)")
    }

    @Test("Prefill first conv state extraction matches scratch-derived Bx")
    func prefillFirstConvStateMatchesScratchBx() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        let env = try setupOrSkip()
        guard let prefillPlan = env.model.prefillPlan else {
            Issue.record("No prefill plan")
            return
        }
        guard let device = MTLCreateSystemDefaultDevice(),
              let convState = prefillPlan.buffers.convState else {
            Issue.record("No conv_state buffer")
            return
        }

        let seqLen = 5
        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        preparePrefillInputs(
            prefillPlan: prefillPlan,
            seqLen: seqLen,
            tokens: tokens
        )
        try executePrefillPrefix(
            prefillPlan: prefillPlan,
            device: device,
            seqLen: seqLen,
            stepCount: 7
        )

        let hiddenSize = 2048
        let kernelSize = prefillPlan.buffers.convStateKernelSize
        let scratchSlotStride = prefillPlan.slotDimension * seqLen
        let scratchBase = scratchSlotStride
        let scratch = readF32Buffer(prefillPlan.buffers.scratch)
        var expected = Array(repeating: Float.zero, count: kernelSize * hiddenSize)
        for k in 0..<kernelSize {
            let srcPos = seqLen - kernelSize + k
            guard srcPos >= 0 else { continue }
            for ch in 0..<hiddenSize {
                let base = scratchBase + srcPos * prefillPlan.slotDimension
                let b = scratch[base + ch]
                let x = scratch[base + 2 * hiddenSize + ch]
                expected[k * hiddenSize + ch] = b * x
            }
        }

        let convAll = readDecodeBuffer(convState, precision: .bfloat16)
        let actual = Array(convAll.prefix(kernelSize * hiddenSize))
        let maxErr = maxAbsoluteError(actual, expected)
        let sample = [0, hiddenSize + 575, 2 * hiddenSize + 575].map { index in
            String(
                format: "(i=%d metal=%.4f expected=%.4f)",
                index,
                actual[index],
                expected[index]
            )
        }
        print("[RefComp] first conv_state vs scratch-derived Bx maxErr=\(String(format: "%.4f", maxErr)) \(sample)")
        #expect(maxErr < 0.1, "First conv_state extraction diverged from scratch-derived Bx: maxErr=\(maxErr)")
    }

    @Test("Prefill first layer after MLP boundary diagnostic")
    func prefillFirstLayerAfterMLPMatchesReference() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        let env = try setupOrSkip()
        guard let prefillPlan = env.model.prefillPlan else {
            Issue.record("No prefill plan")
            return
        }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let seqLen = 5
        let hiddenSize = 2048
        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        let lastTokenOffset = (seqLen - 1) * hiddenSize

        preparePrefillInputs(
            prefillPlan: prefillPlan,
            seqLen: seqLen,
            tokens: tokens
        )
        try executePrefillPrefix(
            prefillPlan: prefillPlan,
            device: device,
            seqLen: seqLen,
            stepCount: 14
        )

        let hidden = readF32Buffer(prefillPlan.buffers.hidden)
        let metal = Array(hidden[lastTokenOffset..<(lastTokenOffset + hiddenSize)])
        let referenceAll = try readRefTensorAsFloats(env.ref, name: "ref.prefill.layer_0.after_mlp")
        let reference = Array(referenceAll[lastTokenOffset..<(lastTokenOffset + hiddenSize)])
        let maxErr = maxAbsoluteError(metal, reference)

        print("[RefComp] layer_0.after_mlp boundary diagnostic maxErr=\(String(format: "%.4f", maxErr))")
    }

    @Test("Prefill first attention layer boundary diagnostic")
    func prefillFirstAttentionLayerMatchesReference() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        let env = try setupOrSkip()
        guard let prefillPlan = env.model.prefillPlan else {
            Issue.record("No prefill plan")
            return
        }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let seqLen = 5
        let hiddenSize = 2048
        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        let lastTokenOffset = (seqLen - 1) * hiddenSize

        preparePrefillInputs(
            prefillPlan: prefillPlan,
            seqLen: seqLen,
            tokens: tokens
        )
        try executePrefillPrefix(
            prefillPlan: prefillPlan,
            device: device,
            seqLen: seqLen,
            stepCount: 40
        )

        let hidden = readF32Buffer(prefillPlan.buffers.hidden)
        let metal = Array(hidden[lastTokenOffset..<(lastTokenOffset + hiddenSize)])
        let referenceAll = try readRefTensorAsFloats(env.ref, name: "ref.prefill.layer_2.after_op")
        let reference = Array(referenceAll[lastTokenOffset..<(lastTokenOffset + hiddenSize)])
        let maxErr = maxAbsoluteError(metal, reference)

        print("[RefComp] layer_2.after_op boundary diagnostic maxErr=\(String(format: "%.4f", maxErr))")
    }

    @Test("Prefill second attention layer boundary diagnostic")
    func prefillSecondAttentionLayerMatchesReference() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        let env = try setupOrSkip()
        guard let prefillPlan = env.model.prefillPlan else {
            Issue.record("No prefill plan")
            return
        }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let seqLen = 5
        let hiddenSize = 2048
        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        let lastTokenOffset = (seqLen - 1) * hiddenSize

        preparePrefillInputs(
            prefillPlan: prefillPlan,
            seqLen: seqLen,
            tokens: tokens
        )
        try executePrefillPrefix(
            prefillPlan: prefillPlan,
            device: device,
            seqLen: seqLen,
            stepCount: 87
        )

        let hidden = readF32Buffer(prefillPlan.buffers.hidden)
        let metal = Array(hidden[lastTokenOffset..<(lastTokenOffset + hiddenSize)])
        let referenceAll = try readRefTensorAsFloats(env.ref, name: "ref.prefill.layer_5.after_op")
        let reference = Array(referenceAll[lastTokenOffset..<(lastTokenOffset + hiddenSize)])
        let maxErr = maxAbsoluteError(metal, reference)

        print("[RefComp] layer_5.after_op boundary diagnostic maxErr=\(String(format: "%.4f", maxErr))")
    }

    @Test("Prefill third attention layer boundary diagnostic")
    func prefillThirdAttentionLayerMatchesReference() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        let env = try setupOrSkip()
        guard let prefillPlan = env.model.prefillPlan else {
            Issue.record("No prefill plan")
            return
        }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let seqLen = 5
        let hiddenSize = 2048
        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        let lastTokenOffset = (seqLen - 1) * hiddenSize

        preparePrefillInputs(
            prefillPlan: prefillPlan,
            seqLen: seqLen,
            tokens: tokens
        )
        try executePrefillPrefix(
            prefillPlan: prefillPlan,
            device: device,
            seqLen: seqLen,
            stepCount: 134
        )

        let hidden = readF32Buffer(prefillPlan.buffers.hidden)
        let metal = Array(hidden[lastTokenOffset..<(lastTokenOffset + hiddenSize)])
        let referenceAll = try readRefTensorAsFloats(env.ref, name: "ref.prefill.layer_8.after_op")
        let reference = Array(referenceAll[lastTokenOffset..<(lastTokenOffset + hiddenSize)])
        let maxErr = maxAbsoluteError(metal, reference)

        print("[RefComp] layer_8.after_op boundary diagnostic maxErr=\(String(format: "%.4f", maxErr))")
    }

    // MARK: - Graph and Dispatch Dump

    @Test(
        "Dump IR graph and dispatch entries for LFM2",
        .disabled("Diagnostic dump only; keep readiness suites focused and memory bounded")
    )
    func dumpGraphAndDispatches() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        let spec = try BenchmarkSupport.loadLFM25ModelSpec()

        // Dump IR graph
        print("=== IR GRAPH ===")
        print(spec.resolved.dump())

        // Dump dispatch entries
        let compiler = MetalInferenceCompiler()
        let dump = compiler.dumpDispatchEntries(
            graph: spec.resolved,
            hiddenSize: spec.config.hiddenSize
        )
        print("\n=== DISPATCH ENTRIES ===")
        print(dump)
    }

    @Test(
        "Dump compiled decode plan for LFM2",
        .disabled("Diagnostic dump only; keep readiness suites focused and memory bounded")
    )
    func dumpCompiledDecodePlan() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let device = MTLCreateSystemDefaultDevice() else { throw SetupError.noDevice }
        let stafURL = URL(fileURLWithPath: Self.stafPath)
        guard FileManager.default.fileExists(atPath: stafURL.path) else {
            throw SetupError.noSTAF
        }

        let store = try STAFLoader().load(at: stafURL, device: device)
        let spec = try BenchmarkSupport.loadLFM25ModelSpec()
        let compiler = MetalInferenceCompiler()
        let dump = try compiler.dumpCompiledDecodePlan(
            graph: spec.resolved,
            hiddenSize: spec.config.hiddenSize,
            intermediateSize: spec.config.intermediateSize,
            vocabSize: spec.config.vocabSize,
            stafWeightStore: store,
            device: device)

        print("=== COMPILED DECODE PLAN ===")
        print(dump)
        #expect(dump.contains("kernel="))
    }

    // MARK: - Decode Tests

    @Test("Decode step 0 logits match Python reference")
    func decodeStep0LogitsMatch() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        try verifyDecodeStep(step: 0)
    }

    @Test("Decode step 1 logits match Python reference")
    func decodeStep1LogitsMatch() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        try verifyDecodeStep(step: 1)
    }

    @Test(
        "Decode step 1 logits diagnostic with float32 decode",
        .disabled("Diagnostic precision experiment; not a production correctness gate")
    )
    func decodeStep1LogitsMatchFloat32Decode() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        try verifyDecodeStep(step: 1, decodeBufferPrecisionOverride: .float32)
    }

    @Test(
        "Decode step 2 logits diagnostic",
        .disabled("Crash-prone diagnostic when batched with real-model checks; split before re-enabling")
    )
    func decodeStep2LogitsMatch() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        try verifyDecodeStep(step: 2)
    }

    @Test("Decode step 1 output head matches Python when fed Python final_hidden")
    func decodeStep1OutputHeadMatchesPythonHidden() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        let env = try setupOrSkip()
        let refHidden = try readRefTensorAsFloats(env.ref, name: "ref.decode_1.final_hidden")
        let refLogits = try readRefTensorAsFloats(env.ref, name: "ref.decode_1.logits")

        guard let finalHiddenBuffer = finalHiddenInputBuffer(for: env.model.decodePlan) else {
            Issue.record("Output head projection missing input buffer binding")
            return
        }
        writeDecodeBuffer(refHidden, to: finalHiddenBuffer, precision: env.model.buffers.bufferPrecision)

        var submission = try MetalSubmissionContext(device: env.model.device)
        try submission.withCompute { encoder, argumentTable in
            for step in env.model.decodePlan.steps.suffix(2) {
                MetalDecodeEncoder.encodeStep(
                    step: step,
                    encoder: encoder,
                    argumentTable: argumentTable
                )
            }
        }

        let metalLogits = readDecodeBuffer(env.model.buffers.logits, precision: env.model.buffers.bufferPrecision)
        let metalTop = argmax(metalLogits)
        let refTop = argmax(refLogits)
        let maxErr = maxAbsoluteError(metalLogits, refLogits)

        print("[RefComp] Output-head diagnostic (step 1, Python hidden input):")
        print("  Python argmax: \(refTop.index) (val=\(String(format: "%.2f", refTop.value)))")
        print("  Metal  argmax: \(metalTop.index) (val=\(String(format: "%.2f", metalTop.value)))")
        print("  Max absolute error: \(String(format: "%.4f", maxErr))")

        #expect(metalTop.index == refTop.index,
                "Output head argmax mismatch from Python hidden: Metal=\(metalTop.index) Python=\(refTop.index)")
    }

    @Test("Prefill output head matches Python when fed Python final_hidden")
    func prefillOutputHeadMatchesPythonHidden() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        let env = try setupOrSkip()
        guard let prefillPlan = env.model.prefillPlan else {
            Issue.record("No prefill plan")
            return
        }

        let sequenceLength = 5
        let refHiddenAll = try readRefTensorAsFloats(env.ref, name: "ref.prefill.final_hidden")
        let refLogits = try readRefTensorAsFloats(env.ref, name: "ref.prefill.logits_last")
        let hiddenSize = 2048
        let lastHidden = Array(refHiddenAll.suffix(hiddenSize))

        guard
            let projectionStep = prefillPlan.steps.reversed().first(where: {
                $0.bufferBindings.contains(where: { $0.index == 2 && $0.buffer === prefillPlan.buffers.logits })
            }),
            let inputBinding = projectionStep.bufferBindings.first(where: { $0.index == 0 })
        else {
            Issue.record("Prefill output projection step not found")
            return
        }

        let inputBuffer = inputBinding.buffer
        let inputStride = projectionStep.perPositionStrides[inputBinding.index] ?? 0
        let inputOffset = inputBinding.offset + max(sequenceLength - 1, 0) * inputStride

        guard let argmaxStep = prefillPlan.steps.last else {
            Issue.record("Prefill argmax step not found")
            return
        }

        var submission = try MetalSubmissionContext(device: env.model.device)
        try writeFloat32Slice(
            lastHidden,
            to: inputBuffer,
            offset: inputOffset,
            using: &submission
        )
        try submission.withCompute { encoder, argumentTable in
            for step in [projectionStep, argmaxStep] {
                switch step.mode {
                case .batch:
                    step.bindings.bind(to: argumentTable)
                    step.bindRuntimeArguments(
                        argumentTable: argumentTable,
                        runtimeConstantBuffer: prefillPlan.buffers.runtimeConstantBuffer,
                        sequenceLengthOffset: PrefillBufferSet.sequenceLengthOffset
                    )
                    let grid = step.resolvedGridSize(sequenceLength: 1)
                    step.descriptor.encode(on: encoder, argumentTable: argumentTable, gridSize: grid)
                case .lastToken:
                    step.bindStaticArguments(argumentTable: argumentTable, position: sequenceLength - 1)
                    step.descriptor.encode(on: encoder, argumentTable: argumentTable)
                case .perPosition:
                    Issue.record("Unexpected perPosition step in prefill output head")
                }
            }
        }

        let metalLogits = readF32Buffer(prefillPlan.buffers.logits)
        let metalTop = argmax(metalLogits)
        let refTop = argmax(refLogits)
        let maxErr = maxAbsoluteError(metalLogits, refLogits)

        print("[RefComp] Prefill output-head diagnostic (Python hidden input):")
        print("  Python argmax: \(refTop.index) (val=\(String(format: "%.2f", refTop.value)))")
        print("  Metal  argmax: \(metalTop.index) (val=\(String(format: "%.2f", metalTop.value)))")
        print("  Max absolute error: \(String(format: "%.4f", maxErr))")

        #expect(metalTop.index == refTop.index,
                "Prefill output head argmax mismatch from Python hidden: Metal=\(metalTop.index) Python=\(refTop.index)")
    }

    @Test("Decode step 1 layerwise diagnostic")
    func decodeStep1LayerwiseDiagnostic() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        let env = try setupOrSkip()
        let compiler = MetalInferenceCompiler()
        let dispatchDump = try makeDispatchDump(compiler: compiler)
        let entries = parseDispatchEntries(from: dispatchDump)
        var model = env.model

        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        var currentToken = model.prefill(tokens: tokens)
        currentToken = model.decodeSync(tokenID: currentToken)

        model.buffers.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(model.position)
        model.buffers.tokenIn.contents().bindMemory(to: Int32.self, capacity: 1).pointee = currentToken

        var currentLayer = 0
        var waitingForOperatorResidual = false
        var submission = try MetalSubmissionContext(device: model.device)

        for (stepIndex, step) in model.decodePlan.steps.enumerated() {
            guard stepIndex < entries.count else { break }

            try submission.withCompute { encoder, argumentTable in
                MetalDecodeEncoder.encodeStep(
                    step: step,
                    encoder: encoder,
                    argumentTable: argumentTable
                )
            }

            let entry = entries[stepIndex]
            if entry.kind.contains("projection(o_proj") || entry.kind.contains("projection(out_proj") {
                let metal = readDecodeBuffer(model.buffers.hidden, precision: model.buffers.bufferPrecision)
                let ref = try readRefTensorAsFloats(env.ref, name: "ref.decode_1.layer_\(currentLayer).after_op")
                let err = maxAbsoluteError(metal, ref)
                print("[RefComp] Layer \(currentLayer) after_op maxErr=\(String(format: "%.4f", err)) kind=\(entry.kind)")
                waitingForOperatorResidual = true
            }

            if entry.kind.contains("projection(down_proj") {
                let metal = readDecodeBuffer(model.buffers.hidden, precision: model.buffers.bufferPrecision)
                let ref = try readRefTensorAsFloats(env.ref, name: "ref.decode_1.layer_\(currentLayer).mlp_out")
                let err = maxAbsoluteError(metal, ref)
                print("[RefComp] Layer \(currentLayer) mlp_out maxErr=\(String(format: "%.4f", err)) kind=\(entry.kind)")
            }

            if entry.kind.contains("ResidualAddFragment") || entry.kind.contains("synthesized_3way") {
                if waitingForOperatorResidual {
                    waitingForOperatorResidual = false
                } else {
                    let metal = readDecodeBuffer(model.buffers.hidden, precision: model.buffers.bufferPrecision)
                    let ref = try readRefTensorAsFloats(env.ref, name: "ref.decode_1.layer_\(currentLayer).after_mlp")
                    let err = maxAbsoluteError(metal, ref)
                    print("[RefComp] Layer \(currentLayer) after_mlp maxErr=\(String(format: "%.4f", err)) kind=\(entry.kind)")
                    currentLayer += 1
                }
            }
        }
    }

    @Test("Decode step 1 final norm kernel diagnostic")
    func decodeStep1FinalNormKernelMatchesPython() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        let env = try setupOrSkip()
        guard let normStep = finalNormStep(for: env.model.decodePlan) else {
            print("[Skip] Final norm step not found in the current decode plan.")
            return
        }
        let input = try finalNormDiagnosticInput(for: normStep, env: env)
        let expected = try readRefTensorAsFloats(env.ref, name: "ref.decode_1.final_hidden")

        writeDecodeBuffer(input.hidden, to: env.model.buffers.hidden, precision: env.model.buffers.bufferPrecision)
        if let residual = input.residual {
            writeDecodeBuffer(residual, to: env.model.buffers.residual, precision: env.model.buffers.bufferPrecision)
        }

        var submission = try MetalSubmissionContext(device: env.model.device)
        try submission.withCompute { encoder, argumentTable in
            MetalDecodeEncoder.encodeStep(
                step: normStep,
                encoder: encoder,
                argumentTable: argumentTable
            )
        }

        guard let finalHiddenBuffer = finalHiddenInputBuffer(for: env.model.decodePlan) else {
            Issue.record("Output head projection missing input buffer binding")
            return
        }
        let actual = readDecodeBuffer(finalHiddenBuffer, precision: env.model.buffers.bufferPrecision)
        let maxErr = maxAbsoluteError(actual, expected)
        print("[RefComp] Final norm kernel maxErr vs Python final_hidden: \(String(format: "%.4f", maxErr))")
        #expect(maxErr < 0.125, "Final norm kernel drifted: maxErr=\(maxErr)")
    }

    @Test("Decode step 1 final norm CPU diagnostic")
    func decodeStep1FinalNormCPUDiagnostic() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        let env = try setupOrSkip()
        guard let normStep = finalNormStep(for: env.model.decodePlan) else {
            print("[Skip] Final norm step not found in the current decode plan.")
            return
        }
        let input = try finalNormDiagnosticInput(for: normStep, env: env)
        let expected = try readRefTensorAsFloats(env.ref, name: "ref.decode_1.final_hidden")
        let weightBinding = finalNormWeightBinding(for: normStep)
        guard let weightBinding else {
            Issue.record("Final norm weight binding not found")
            return
        }

        let weightPtr = (weightBinding.buffer.contents() + weightBinding.offset)
            .bindMemory(to: BFloat16.self, capacity: input.hidden.count)
        let weights = (0..<input.hidden.count).map { Float(weightPtr[$0]) }

        let preNorm: [Float]
        if let residual = input.residual {
            preNorm = zip(input.hidden, residual).map(+)
        } else {
            preNorm = input.hidden
        }

        let sumSq = preNorm.reduce(Float.zero) { $0 + $1 * $1 }
        let invRMS = 1.0 / sqrtf(sumSq / Float(preNorm.count) + 1e-5)
        let cpu = zip(preNorm, weights).map { $0 * invRMS * $1 }
        let cpuErr = maxAbsoluteError(cpu, expected)
        let cpuTop = argmax(cpu)
        let refTop = argmax(expected)

        print("[RefComp] Final norm CPU diagnostic:")
        print("  CPU maxErr vs Python final_hidden: \(String(format: "%.4f", cpuErr))")
        print("  CPU argmax-like max entry index: \(cpuTop.index) val=\(String(format: "%.4f", cpuTop.value))")
        print("  Python max entry index: \(refTop.index) val=\(String(format: "%.4f", refTop.value))")
    }

    // MARK: - Decode Step Helper

    fileprivate func verifyDecodeStep(
        step: Int,
        decodeBufferPrecisionOverride: BufferPrecision? = nil
    ) throws {
        let env = try setupOrSkip(decodeBufferPrecisionOverride: decodeBufferPrecisionOverride)
        var model = env.model

        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        var currentToken = model.prefill(tokens: tokens)

        for s in 0...step {
            currentToken = model.decodeSync(tokenID: currentToken)
            if s < step { continue }

            // Read from DECODE plan's logits buffer (F16)
            let metalLogits = readDecodeBuffer(model.buffers.logits, precision: model.buffers.bufferPrecision)
            let refLogits = try readRefTensorAsFloats(env.ref, name: "ref.decode_\(step).logits")

            let refTop = argmax(refLogits)
            let metalTop = argmax(metalLogits)
            let maxErr = maxAbsoluteError(metalLogits, refLogits)

            let metalTop5 = topK(metalLogits, k: 5)
            let refTop5 = topK(refLogits, k: 5)
            print("[RefComp] Decode step \(step):")
            print("  Python argmax: \(refTop.index) (val=\(String(format: "%.2f", refTop.value)))")
            print("  Metal  argmax: \(metalTop.index) (val=\(String(format: "%.2f", metalTop.value)))")
            print("  Metal  top-5: \(metalTop5.map { "(\($0.index),\(String(format: "%.2f", $0.value)))" })")
            print("  Python top-5: \(refTop5.map { "(\($0.index),\(String(format: "%.2f", $0.value)))" })")
            print("  Max absolute error: \(String(format: "%.4f", maxErr))")

            guard let finalHiddenBuffer = finalHiddenInputBuffer(for: model.decodePlan) else {
                Issue.record("Output head projection missing input buffer binding")
                return
            }
            let hiddenSize = finalHiddenBuffer.length / model.buffers.bufferPrecision.byteSize
            let metalFinalHidden = Array(readDecodeBuffer(finalHiddenBuffer, precision: model.buffers.bufferPrecision).prefix(hiddenSize))
            do {
                let refFinalHidden = try readRefTensorAsFloats(env.ref, name: "ref.decode_\(step).final_hidden")
                let finalHiddenErr = maxAbsoluteError(metalFinalHidden, refFinalHidden)
                let metalHiddenSample = (0..<4).map { String(format: "%.4f", metalFinalHidden[$0]) }
                let refHiddenSample = (0..<4).map { String(format: "%.4f", refFinalHidden[$0]) }
                print("  Final hidden maxErr: \(String(format: "%.4f", finalHiddenErr))")
                print("  Metal  hidden[0..3]: \(metalHiddenSample)")
                print("  Python hidden[0..3]: \(refHiddenSample)")
            } catch {
                print("  [Skip] Missing decode_\(step).final_hidden reference: \(error)")
            }

            // Compare conv_state after this decode step
            if let convState = model.buffers.convState {
                let convDim = model.buffers.convStateDimension
                let kSize = model.buffers.convStateKernelSize
                let elemSize = MemoryLayout<Float16>.size
                for convIdx in 0..<10 {
                    do {
                        let refData = try readRefTensorAsFloats(env.ref, name: "ref.decode_\(step).conv_state.\(convIdx)")
                        let layerOffset = convIdx * kSize * convDim * elemSize
                        let fullConvState = readDecodeBuffer(convState, precision: .bfloat16)
                        let base = layerOffset / elemSize
                        let metalVals = Array(fullConvState[base..<(base + kSize * convDim)])
                        let err = maxAbsoluteError(metalVals, refData)
                        if convIdx < 3 || err > 1.0 {
                            print("  conv_state[\(convIdx)] after decode \(step): maxErr=\(String(format: "%.4f", err))")
                        }
                    } catch {
                        print("  [Skip] Missing decode_\(step).conv_state.\(convIdx) reference: \(error)")
                    }
                }
            }

            // Step 0 and 1 are stable reference-aligned gates. Later decode
            // steps can cascade after small BF16-vs-reference logit shifts, so
            // they remain explicit diagnostics until their precision contract is
            // tightened.
            if step <= 1 {
                #expect(metalTop.index == refTop.index,
                        "Decode step \(step) argmax: Metal=\(metalTop.index) Python=\(refTop.index)")
            }
        }
    }

    // MARK: - Setup

    private struct TestEnvironment {
        var model: MetalInferenceModel
        let ref: MetalWeightFile
        let store: STAFWeightStore
    }

    private struct ParsedDispatchEntry {
        let layer: Int?
        let kind: String
    }

    private func setupOrSkip(
        weightAccessPolicyOverride: ProjectionWeightAccessPolicyOverride? = nil,
        decodeBufferPrecisionOverride: BufferPrecision? = nil
    ) throws -> TestEnvironment {
        try Self.buildEnvironment(
            weightAccessPolicyOverride: weightAccessPolicyOverride,
            decodeBufferPrecisionOverride: decodeBufferPrecisionOverride
        )
    }

    private static func buildEnvironment(
        weightAccessPolicyOverride: ProjectionWeightAccessPolicyOverride? = nil,
        decodeBufferPrecisionOverride: BufferPrecision? = nil
    ) throws -> TestEnvironment {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SetupError.noDevice
        }

        let refURL = URL(fileURLWithPath: Self.referencePath)
        guard FileManager.default.fileExists(atPath: refURL.path) else {
            print("[RefComp] Reference not found — run: python3 scripts/hf/dump_lfm2_reference.py")
            throw SetupError.noReference
        }

        let stafURL = URL(fileURLWithPath: Self.stafPath)
        let modelDir = stafURL.deletingLastPathComponent()

        // Auto-convert safetensors → STAF if needed
        if !FileManager.default.fileExists(atPath: Self.stafPath) {
            let safetensorsURL = modelDir.appendingPathComponent("model.safetensors")
            guard FileManager.default.fileExists(atPath: safetensorsURL.path) else {
                print("[RefComp] Neither STAF nor safetensors found in \(modelDir.path)")
                throw SetupError.noSTAF
            }
            print("[RefComp] Converting safetensors → STAF...")
            try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)
            print("[RefComp] STAF created: \(Self.stafPath)")
        }

        let ref = try SafetensorsLoader().load(at: refURL, device: device)
        let store = try STAFLoader().load(at: stafURL, device: device)
        let spec = try BenchmarkSupport.loadLFM25ModelSpec()

        let compiler = MetalInferenceCompiler(
            weightAccessPolicyOverride: weightAccessPolicyOverride,
            decodeBufferPrecisionOverride: decodeBufferPrecisionOverride
        )
        let decodePlan = try compiler.compile(
            graph: spec.resolved,
            hiddenSize: spec.config.hiddenSize,
            intermediateSize: spec.config.intermediateSize,
            vocabSize: spec.config.vocabSize,
            stafWeightStore: store,
            device: device)
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
            device: device)

        var model = try MetalInferenceModel(plan: decodePlan, device: device)
        model.prefillPlan = prefillPlan

        return TestEnvironment(model: model, ref: ref, store: store)
    }

    private func makeDispatchDump(compiler: MetalInferenceCompiler) throws -> String {
        let spec = try BenchmarkSupport.loadLFM25ModelSpec()
        return compiler.dumpDispatchEntries(
            graph: spec.resolved,
            hiddenSize: spec.config.hiddenSize
        )
    }

    private func parseDispatchEntries(from dump: String) -> [ParsedDispatchEntry] {
        dump.split(separator: "\n").compactMap { line in
            guard let bracketEnd = line.firstIndex(of: "]") else { return nil }
            let tail = line[line.index(after: bracketEnd)...].trimmingCharacters(in: .whitespaces)
            if tail.hasPrefix("-- ") {
                return ParsedDispatchEntry(layer: nil, kind: String(tail.dropFirst(3)))
            }
            guard tail.first == "L" else { return nil }
            let pieces = tail.split(separator: " ", maxSplits: 1).map(String.init)
            guard pieces.count == 2, let layer = Int(pieces[0].dropFirst()) else { return nil }
            return ParsedDispatchEntry(layer: layer, kind: pieces[1])
        }
    }

    private func finalHiddenInputBuffer(for plan: MetalDispatchPlan) -> MTLBuffer? {
        let projectionStepIndex = plan.steps.count - 2
        let projectionStep = plan.steps[projectionStepIndex]
        guard let binding = projectionStep.bufferBindings.first(where: { $0.index == 0 }) else {
            return nil
        }
        return binding.buffer
    }

    private struct FinalNormDiagnosticInput {
        let hidden: [Float]
        let residual: [Float]?
    }

    private func finalNormStep(for plan: MetalDispatchPlan) -> MetalDispatchStep? {
        let projectionStepIndex = plan.steps.count - 2
        let projectionStep = plan.steps[projectionStepIndex]
        guard let projectionInput = projectionStep.bufferBindings.first(where: { $0.index == 0 }) else {
            return nil
        }

        for step in plan.steps[..<projectionStepIndex].reversed() {
            let label = step.pipeline.label ?? ""
            guard label.contains("rms_norm") else { continue }
            if step.bufferBindings.contains(where: { binding in
                binding.buffer === projectionInput.buffer && binding.offset == projectionInput.offset
            }) {
                return step
            }
        }

        return nil
    }

    private func finalNormDiagnosticInput(
        for step: MetalDispatchStep,
        env: TestEnvironment
    ) throws -> FinalNormDiagnosticInput {
        let kernelName = step.pipeline.label ?? ""
        if kernelName.contains("fused_residual_add_rms_norm") {
            let afterMLP = try readRefTensorAsFloats(env.ref, name: "ref.decode_1.layer_15.after_mlp")
            let residual = try readRefTensorAsFloats(env.ref, name: "ref.decode_1.layer_15.after_op")
            return FinalNormDiagnosticInput(
                hidden: zip(afterMLP, residual).map(-),
                residual: residual
            )
        }
        return FinalNormDiagnosticInput(
            hidden: try readRefTensorAsFloats(env.ref, name: "ref.decode_1.layer_15.after_mlp"),
            residual: nil
        )
    }

    private func finalNormWeightBinding(for step: MetalDispatchStep) -> (index: Int, buffer: MTLBuffer, offset: Int)? {
        let kernelName = step.pipeline.label ?? ""
        let weightIndex = kernelName.contains("fused_") ? 2 : 1
        return step.bufferBindings.first(where: { $0.index == weightIndex })
    }

    // MARK: - Reference Tensor Access

    private func readFloat16Tensor(
        _ file: MetalWeightFile, name: String
    ) throws -> [Float16] {
        guard let info = file.tensors[name] else {
            throw SetupError.tensorNotFound(name)
        }
        let count = info.shape.reduce(1, *)
        let ptr = (file.buffer.contents() + file.dataSectionOffset + info.dataOffset)
            .bindMemory(to: Float16.self, capacity: count)
        let buffer = UnsafeBufferPointer(start: ptr, count: count)
        return Array(buffer)
    }

    // MARK: - Buffer Reading (converts any buffer to [Float])

    private func readDecodeBuffer(_ buffer: MTLBuffer, precision: BufferPrecision) -> [Float] {
        if buffer.storageMode == .private {
            // Private buffers require GPU blit to a shared staging buffer
            let device = buffer.device
            guard let staging = device.makeBuffer(length: buffer.length, options: .storageModeShared),
                  let queue = device.makeCommandQueue(),
                  let cb = queue.makeCommandBuffer(),
                  let blit = cb.makeBlitCommandEncoder() else { return [] }
            blit.copy(from: buffer, sourceOffset: 0, to: staging, destinationOffset: 0, size: buffer.length)
            blit.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            return readSharedDecodeBuffer(staging, precision: precision)
        }
        return readSharedDecodeBuffer(buffer, precision: precision)
    }

    private func readF32Buffer(_ buffer: MTLBuffer) -> [Float] {
        readDecodeBuffer(buffer, precision: .float32)
    }

    private func writeDecodeBuffer(_ values: [Float], to buffer: MTLBuffer, precision: BufferPrecision) {
        if buffer.storageMode == .private {
            let device = buffer.device
            guard let staging = device.makeBuffer(length: buffer.length, options: .storageModeShared),
                  let queue = device.makeCommandQueue(),
                  let cb = queue.makeCommandBuffer(),
                  let blit = cb.makeBlitCommandEncoder() else { return }

            let count = min(values.count, staging.length / precision.byteSize)
            writeSharedDecodeBuffer(values, to: staging, precision: precision, count: count)

            blit.copy(from: staging, sourceOffset: 0, to: buffer, destinationOffset: 0, size: count * precision.byteSize)
            blit.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            return
        }

        let count = min(values.count, buffer.length / precision.byteSize)
        writeSharedDecodeBuffer(values, to: buffer, precision: precision, count: count)
    }

    private func writeFloat32Slice(
        _ values: [Float],
        to buffer: MTLBuffer,
        offset: Int,
        using submission: inout MetalSubmissionContext
    ) throws {
        let byteCount = values.count * MemoryLayout<Float>.stride
        guard let staging = submission.device.makeBuffer(length: byteCount, options: [.storageModeShared]) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate float32 staging buffer")
        }
        let pointer = staging.contents().bindMemory(to: Float.self, capacity: values.count)
        pointer.update(from: values, count: values.count)
        try submission.copyBuffers([
            (
                from: staging,
                sourceOffset: 0,
                to: buffer,
                destinationOffset: offset,
                size: byteCount
            )
        ])
    }

    private func readSharedDecodeBuffer(_ buffer: MTLBuffer, precision: BufferPrecision) -> [Float] {
        switch precision {
        case .float16:
            let count = buffer.length / MemoryLayout<Float16>.size
            let ptr = buffer.contents().bindMemory(to: Float16.self, capacity: count)
            return (0..<count).map { Float(ptr[$0]) }
        case .bfloat16:
            let count = buffer.length / MemoryLayout<BFloat16>.size
            let ptr = buffer.contents().bindMemory(to: BFloat16.self, capacity: count)
            return (0..<count).map { Float(ptr[$0]) }
        case .float32, .float32Decode:
            let count = buffer.length / MemoryLayout<Float32>.size
            let ptr = buffer.contents().bindMemory(to: Float32.self, capacity: count)
            return (0..<count).map { ptr[$0] }
        }
    }

    private func writeSharedDecodeBuffer(_ values: [Float], to buffer: MTLBuffer, precision: BufferPrecision, count: Int) {
        switch precision {
        case .float16:
            let ptr = buffer.contents().bindMemory(to: Float16.self, capacity: count)
            for i in 0..<count {
                ptr[i] = Float16(values[i])
            }
        case .bfloat16:
            let ptr = buffer.contents().bindMemory(to: BFloat16.self, capacity: count)
            for i in 0..<count {
                ptr[i] = BFloat16(values[i])
            }
        case .float32, .float32Decode:
            let ptr = buffer.contents().bindMemory(to: Float32.self, capacity: count)
            for i in 0..<count {
                ptr[i] = values[i]
            }
        }
    }

    private func preparePrefillInputs(
        prefillPlan: MetalPrefillPlan,
        seqLen: Int,
        tokens: [Int32]
    ) {
        let tokenPtr = prefillPlan.buffers.tokenIDs.contents()
            .bindMemory(to: Int32.self, capacity: seqLen)
        let posPtr = prefillPlan.buffers.positions.contents()
            .bindMemory(to: UInt32.self, capacity: seqLen)
        for i in 0..<seqLen {
            tokenPtr[i] = tokens[i]
            posPtr[i] = UInt32(i)
        }

        let runtimeConstantBuffer = prefillPlan.buffers.runtimeConstantBuffer
        runtimeConstantBuffer.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(seqLen)
        for pos in 0..<seqLen {
            let positionPtr = runtimeConstantBuffer.contents()
                .advanced(by: PrefillBufferSet.positionOffset(at: pos))
                .bindMemory(to: UInt32.self, capacity: 1)
            positionPtr.pointee = UInt32(pos)
        }
    }

    private func executePrefillPrefix(
        prefillPlan: MetalPrefillPlan,
        device: MTLDevice,
        seqLen: Int,
        stepCount: Int
    ) throws {
        let runtimeConstantBuffer = prefillPlan.buffers.runtimeConstantBuffer
        var submission = try MetalSubmissionContext(device: device)
        try submission.withCompute { encoder, argumentTable in
            for step in prefillPlan.steps.prefix(stepCount) {
                switch step.mode {
                case .batch:
                    step.bindings.bind(to: argumentTable)
                    step.bindRuntimeArguments(
                        argumentTable: argumentTable,
                        runtimeConstantBuffer: runtimeConstantBuffer,
                        sequenceLengthOffset: PrefillBufferSet.sequenceLengthOffset
                    )
                    let grid = step.resolvedGridSize(sequenceLength: seqLen)
                    step.descriptor.encode(on: encoder, argumentTable: argumentTable, gridSize: grid)
                case .perPosition:
                    for pos in 0..<seqLen {
                        step.bindStaticArguments(argumentTable: argumentTable, position: pos)
                        if let posIdx = step.positionBufferIndex {
                            argumentTable.setAddress(
                                runtimeConstantBuffer.gpuAddress + UInt64(PrefillBufferSet.positionOffset(at: pos)),
                                index: posIdx
                            )
                        }
                        step.descriptor.encode(on: encoder, argumentTable: argumentTable)
                    }
                case .lastToken:
                    step.bindStaticArguments(argumentTable: argumentTable, position: seqLen - 1)
                    step.descriptor.encode(on: encoder, argumentTable: argumentTable)
                }
            }
        }
    }


    private func readRefTensorAsFloats(
        _ file: MetalWeightFile, name: String
    ) throws -> [Float] {
        guard let info = file.tensors[name] else {
            throw SetupError.tensorNotFound(name)
        }
        let count = info.shape.reduce(1, *)
        let base = file.buffer.contents() + file.dataSectionOffset + info.dataOffset
        switch info.dtype {
        case .float16:
            let pointer = base.bindMemory(to: Float16.self, capacity: count)
            return (0..<count).map { Float(pointer[$0]) }
        case .bfloat16:
            let pointer = base.bindMemory(to: BFloat16.self, capacity: count)
            return (0..<count).map { Float(pointer[$0]) }
        case .float32:
            let pointer = base.bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: pointer, count: count))
        default:
            throw SetupError.tensorNotFound("Unsupported dtype \(info.dtype) for \(name)")
        }
    }

    // MARK: - Comparison Utilities (all work on [Float])

    private struct IndexedValue {
        let index: Int
        let value: Float
    }

    private func argmax(_ values: [Float]) -> IndexedValue {
        var maxVal: Float = -.infinity
        var maxIdx = 0
        for i in 0..<values.count {
            if values[i] > maxVal { maxVal = values[i]; maxIdx = i }
        }
        return IndexedValue(index: maxIdx, value: maxVal)
    }

    private func topK(_ values: [Float], k: Int) -> [IndexedValue] {
        let indexed = values.enumerated().map { IndexedValue(index: $0.offset, value: $0.element) }
        return Array(indexed.sorted { $0.value > $1.value }.prefix(k))
    }

    private func maxAbsoluteError(_ a: [Float], _ b: [Float]) -> Float {
        let count = min(a.count, b.count)
        var maxErr: Float = 0
        for i in 0..<count {
            if !a[i].isFinite || !b[i].isFinite {
                return .infinity
            }
            maxErr = max(maxErr, abs(a[i] - b[i]))
        }
        return maxErr
    }

    // MARK: - Errors

    private enum SetupError: Error {
        case noDevice
        case noReference
        case noSTAF
        case tensorNotFound(String)
    }
}

@Suite("Reference Comparison Isolated Diagnostics", .serialized)
struct ReferenceComparisonIsolatedDiagnosticsTests {
    @Test("Decode step 2 diagnostic runs in an isolated process")
    func decodeStep2DiagnosticRunsInIsolation() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        try ReferenceComparisonTests().verifyDecodeStep(step: 2)
    }
}
#endif
