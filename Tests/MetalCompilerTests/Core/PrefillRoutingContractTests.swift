import Testing
import Metal
@testable import MetalCompiler

@Suite("Prefill Routing Contracts", .serialized)
struct PrefillRoutingContractTests {
    @Test("Prefill output head consumes final hidden source")
    func prefillOutputHeadConsumesFinalHiddenSource() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let (device, store) = try BenchmarkSupport.loadStoreOrSkip()
        let spec = try BenchmarkSupport.loadLFM25ModelSpec()
        let sequenceLength = 5

        let compiler = MetalInferenceCompiler()
        let prefillPlan = try compiler.compilePrefill(
            graph: spec.resolved,
            hiddenSize: spec.config.hiddenSize,
            intermediateSize: spec.config.intermediateSize,
            vocabSize: spec.config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store,
            device: device
        )

        let outputHeadStep = try #require(prefillPlan.steps.first(where: { step in
            step.mode == .lastToken
                && step.bindings.buffers.contains(where: { binding in
                    binding.index == 2 && binding.buffer === prefillPlan.buffers.logits
                })
        }))
        let inputBinding = try #require(outputHeadStep.bindings.buffers.first(where: { $0.index == 0 }))
        let source = prefillPlan.finalHiddenSource(sequenceLength: sequenceLength)
        let adjustedInputOffset = inputBinding.offset
            + (outputHeadStep.perPositionStrides[0] ?? 0) * max(sequenceLength - 1, 0)

        #expect(
            inputBinding.buffer === source.buffer,
            "prefill output head should read from finalHiddenSource buffer"
        )
        #expect(
            adjustedInputOffset == source.offset,
            "prefill output head offset mismatch: expected \(source.offset), got \(adjustedInputOffset)"
        )
    }
}
