import Testing
import Metal
@testable import MetalCompiler

@Suite("Generated Kernel Library")
struct GeneratedLibraryTests {

    @Test("Complete generated library compiles and contains all needed functions")
    func generatedLibraryCompiles() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let source = MetalSourceGenerator.generateCompleteLibrary(weightFormat: .bfloat16)
        let options = MTLCompileOptions()
        options.languageVersion = .version3_0

        let library = try device.makeLibrary(source: source, options: options)

        // Verify critical decode kernels
        for name in ["gemv_bf16", "rms_norm_bf16", "swiglu", "argmax",
                      "fused_copy_rms_norm_bf16", "fused_residual_add_copy_rms_norm_bf16",
                      "qk_rms_norm_bf16", "rope", "conv_state_update",
                      "flash_attn_decode", "gemv_q4_g64", "sigmoid_gate"] {
            #expect(library.makeFunction(name: name) != nil, "Missing: \(name)")
        }

        // Verify critical prefill kernels
        for name in ["gemm_bf16_f32s", "rms_norm_seq_bf16_f32_inplace",
                      "swiglu_seq_f32", "embedding_lookup_seq_bf16_f32",
                      "qk_rms_norm_seq_f32", "rope_seq_f32",
                      "conv1d_causal_seq_f32", "argmax_f32",
                      "flash_attn_decode_f32"] {
            #expect(library.makeFunction(name: name) != nil, "Missing: \(name)")
        }

        print("[GenLib] Library: \(library.functionNames.count) functions, \(source.count) chars")
    }
}
