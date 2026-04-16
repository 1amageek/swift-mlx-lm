import Testing
@testable import MetalCompiler

@Suite("Prefill Hidden Override Contract", .serialized)
struct PrefillHiddenOverrideContractTests {
    @Test("Hidden override replay skips only leading embedding kernels")
    func skipsOnlyLeadingEmbeddingKernels() {
        let replayStart = prefillHiddenOverrideReplayStartStepIndex(kernelNames: [
            "embedding_lookup_seq_bf16_f32",
            "copy_buffer_seq_f32",
            "copy_buffer_seq_f32",
            "rms_norm_seq_bf16_f32_inplace",
        ])

        #expect(replayStart == 1)
    }

    @Test("Hidden override replay preserves bootstrap work when there is no leading embedding kernel")
    func preservesBootstrapWithoutLeadingEmbeddingKernel() {
        let replayStart = prefillHiddenOverrideReplayStartStepIndex(kernelNames: [
            "copy_buffer_seq_f32",
            "rms_norm_seq_bf16_f32_inplace",
        ])

        #expect(replayStart == 0)
    }
}
