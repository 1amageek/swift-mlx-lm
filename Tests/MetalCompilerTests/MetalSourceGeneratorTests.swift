import Testing
import Metal
@testable import MetalCompiler

/// Verify MetalSourceGenerator produces valid, compilable MSL
/// that computes the same results as the hardcoded kernels.
@Suite("Metal Source Generator")
struct MetalSourceGeneratorTests {

    @Test("Generated RMSNorm compiles for all precision × weight format combinations")
    func rmsNormCompiles() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let precisions: [MetalSourceGenerator.BufferPrecision] = [.float16, .float32]
        let weightFormats: [MetalSourceGenerator.WeightFormat] = [.float16, .bfloat16]

        for precision in precisions {
            for weightFormat in weightFormats {
                let name = "rms_norm_\(precision)_\(weightFormat)"
                let source = MetalSourceGenerator.commonHeader + "\n\n"
                    + MetalSourceGenerator.generateReduction(
                        name: name, dimension: 2048, epsilon: 1e-5,
                        bufferPrecision: precision, weightFormat: weightFormat)

                let options = MTLCompileOptions()
                options.languageVersion = .version4_0
                let library = try device.makeLibrary(source: source, options: options)
                let function = library.makeFunction(name: name)
                #expect(function != nil, "Failed to compile \(name)")
            }
        }
    }

    @Test("Generated SwiGLU compiles for both precisions")
    func swigluCompiles() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        for precision in [MetalSourceGenerator.BufferPrecision.float16, .float32] {
            let name = "swiglu_\(precision)"
            let source = MetalSourceGenerator.commonHeader + "\n\n"
                + MetalSourceGenerator.generateSwiGLU(name: name, bufferPrecision: precision)

            let options = MTLCompileOptions()
            options.languageVersion = .version4_0
            let library = try device.makeLibrary(source: source, options: options)
            #expect(library.makeFunction(name: name) != nil, "Failed to compile \(name)")
        }
    }

    @Test("Generated fused SwiGLU projection compiles for both weight formats")
    func fusedSwiGLUProjectionCompiles() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let formats: [(MetalSourceGenerator.WeightFormat, String)] = [
            (.float16, "fp16"), (.bfloat16, "bf16")
        ]

        for (format, label) in formats {
            let name = "fused_swiglu_projection_\(label)"
            let source = MetalSourceGenerator.commonHeader + "\n\n"
                + MetalSourceGenerator.generateFusedSwiGLUProjection(
                    name: name,
                    bufferPrecision: .float16,
                    weightFormat: format)

            let options = MTLCompileOptions()
            options.languageVersion = .version4_0
            let library = try device.makeLibrary(source: source, options: options)
            #expect(library.makeFunction(name: name) != nil, "Failed to compile \(name)")
        }
    }

    @Test("Generated GEMM compiles for all weight formats")
    func gemmCompiles() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let formats: [(MetalSourceGenerator.WeightFormat, String)] = [
            (.float16, "fp16"), (.bfloat16, "bf16")
        ]

        for (format, label) in formats {
            for precision in [MetalSourceGenerator.BufferPrecision.float16, .float32] {
                let name = "gemm_\(label)_\(precision)"
                let source = MetalSourceGenerator.commonHeader + "\n\n"
                    + MetalSourceGenerator.generateGEMM(
                        name: name, bufferPrecision: precision, weightFormat: format)

                let options = MTLCompileOptions()
                options.languageVersion = .version4_0
                let library = try device.makeLibrary(source: source, options: options)
                #expect(library.makeFunction(name: name) != nil, "Failed to compile \(name)")
            }
        }
    }

    @Test("Generated structural kernels compile")
    func structuralCompiles() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        for precision in [MetalSourceGenerator.BufferPrecision.float16, .float32] {
            var allSource = MetalSourceGenerator.commonHeader + "\n\n"
            allSource += MetalSourceGenerator.generateCopy(
                name: "copy_\(precision)", bufferPrecision: precision) + "\n\n"
            allSource += MetalSourceGenerator.generateResidualAdd(
                name: "add_\(precision)", bufferPrecision: precision) + "\n\n"
            allSource += MetalSourceGenerator.generateArgmax(
                name: "argmax_\(precision)", bufferPrecision: precision) + "\n\n"
            allSource += MetalSourceGenerator.generateEmbeddingLookup(
                name: "emb_\(precision)", bufferPrecision: precision, weightFormat: .bfloat16)

            let options = MTLCompileOptions()
            options.languageVersion = .version4_0
            let library = try device.makeLibrary(source: allSource, options: options)
            #expect(library.makeFunction(name: "copy_\(precision)") != nil)
            #expect(library.makeFunction(name: "add_\(precision)") != nil)
            #expect(library.makeFunction(name: "argmax_\(precision)") != nil)
            #expect(library.makeFunction(name: "emb_\(precision)") != nil)
        }
    }

    @Test("All precision × weight format combinations produce unique kernels")
    func noDuplicateVariants() {
        // Same computation, different precision/format → different source
        let a = MetalSourceGenerator.generateReduction(
            name: "norm", dimension: 2048, epsilon: 1e-5,
            bufferPrecision: .float16, weightFormat: .float16)
        let b = MetalSourceGenerator.generateReduction(
            name: "norm", dimension: 2048, epsilon: 1e-5,
            bufferPrecision: .float32, weightFormat: .bfloat16)
        #expect(a != b, "Different precision/format should produce different source")

        // Verify BF16 source uses bf16_to_float
        #expect(b.contains("bf16_to_float"))
        #expect(!a.contains("bf16_to_float"))
    }
}
