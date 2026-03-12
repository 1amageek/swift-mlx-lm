import Testing
import GGUFParser
import SwiftLM
@testable import MLXLM
@testable import MLXCompiler

// MARK: - Phase 1: Architecture Detection

@Suite("GGUFArchitectureDetector")
struct ArchitectureDetectorTests {

    let detector = GGUFArchitectureDetector()

    @Test("Standard transformer tensor names → .transformer")
    func detectTransformerFallback() {
        let names: Set<String> = [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.0.ffn_norm.weight",
            "output_norm.weight",
        ]
        #expect(detector.detect(tensorNames: names) == .transformer)
    }

    @Test("QK norm + no FFN norm → .parallelAttentionMLP")
    func detectParallelAttentionMLP() {
        let names: Set<String> = [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.attn_norm.weight",
            "blk.0.attn_q_norm.weight",
            "blk.0.attn_k_norm.weight",
            "blk.0.ffn_gate.weight",
            "output_norm.weight",
        ]
        #expect(detector.detect(tensorNames: names) == .parallelAttentionMLP)
    }

    @Test("ffn_gate_inp.weight → .moe")
    func detectMoE() {
        let names: Set<String> = [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_gate_inp.weight",
            "blk.0.ffn_gate.0.weight",
            "blk.0.ffn_up.0.weight",
            "blk.0.ffn_down.0.weight",
            "output_norm.weight",
        ]
        #expect(detector.detect(tensorNames: names) == .moe)
    }

    @Test("ssm_beta.weight → .hybridDeltaNetAttention")
    func detectQwen35Hybrid() {
        let names: Set<String> = [
            "token_embd.weight",
            "blk.0.ssm_beta.weight",
            "blk.0.ssm_alpha.weight",
            "blk.0.ssm_conv1d.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_gate.weight",
            "output_norm.weight",
        ]
        #expect(detector.detect(tensorNames: names) == .hybridDeltaNetAttention)
    }

    @Test("SSM + MoE patterns → .hybridDeltaNetAttention wins (highest priority)")
    func detectPriorityHybridOverMoE() {
        let names: Set<String> = [
            "token_embd.weight",
            "blk.0.ssm_beta.weight",
            "blk.0.ffn_gate_inp.weight",
            "blk.0.ffn_gate.0.weight",
            "output_norm.weight",
        ]
        #expect(detector.detect(tensorNames: names) == .hybridDeltaNetAttention)
    }

    @Test("QK norm + no FFN norm → .parallelAttentionMLP wins over .transformer")
    func detectPriorityParallelAttentionMLPOverTransformer() {
        let names: Set<String> = [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_q_norm.weight",
            "blk.0.attn_k_norm.weight",
            "blk.0.ffn_gate.weight",
            "output_norm.weight",
        ]
        #expect(detector.detect(tensorNames: names) == .parallelAttentionMLP)
    }

    @Test("QK norm WITH FFN norm → .transformer (not parallelAttentionMLP)")
    func detectQKNormWithFFNNormIsTransformer() {
        let names: Set<String> = [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_q_norm.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_gate.weight",
            "output_norm.weight",
        ]
        #expect(detector.detect(tensorNames: names) == .transformer)
    }
}

// MARK: - Phase 2: Config Extraction

@Suite("GGUFConfigExtractor")
struct ConfigExtractorTests {

    let extractor = GGUFConfigExtractor()

    @Test("Extract transformer core fields from metadata")
    func extractTransformerCoreFields() throws {
        let file = try makeLlamaGGUF(
            embeddingLength: 2048,
            blockCount: 22,
            headCount: 16,
            headCountKV: 4,
            feedForwardLength: 5632,
            vocabSize: 32000
        )
        let config = try extractor.extract(from: file, architecture: .transformer)

        #expect(config.hiddenSize == 2048)
        #expect(config.layerCount == 22)
        #expect(config.attentionHeads == 16)
        #expect(config.kvHeads == 4)
        #expect(config.headDim == 128) // 2048 / 16
        #expect(config.intermediateSize == 5632)
        #expect(config.vocabSize == 32000)
    }

    @Test("Bias detection from tensor names")
    func extractBiasDetection() throws {
        let file = try makeLlamaGGUF(
            extraTensors: [
                "blk.0.attn_q.bias",
                "blk.0.ffn_gate.bias",
            ]
        )
        let config = try extractor.extract(from: file, architecture: .transformer)

        #expect(config.attentionBias == true)
        #expect(config.mlpBias == true)
    }

    @Test("No bias tensors → attentionBias=false, mlpBias=false")
    func extractNoBias() throws {
        let file = try makeLlamaGGUF()
        let config = try extractor.extract(from: file, architecture: .transformer)

        #expect(config.attentionBias == false)
        #expect(config.mlpBias == false)
    }

    @Test("output.weight absent → tiedEmbeddings=true")
    func extractTiedEmbeddings() throws {
        // Default makeLlamaGGUF does not add output.weight
        let file = try makeLlamaGGUF()
        let config = try extractor.extract(from: file, architecture: .transformer)

        #expect(config.tiedEmbeddings == true)
    }

    @Test("output.weight present → tiedEmbeddings=false")
    func extractUntiedEmbeddings() throws {
        let file = try makeLlamaGGUF(extraTensors: ["output.weight"])
        let config = try extractor.extract(from: file, architecture: .transformer)

        #expect(config.tiedEmbeddings == false)
    }

    @Test("MoE config extraction")
    func extractMoEConfig() throws {
        let file = try makeLlamaGGUF(
            extraMetadata: [
                ("llama.expert_count", .uint32(8)),
                ("llama.expert_used_count", .uint32(2)),
            ]
        )
        let config = try extractor.extract(from: file, architecture: .moe)

        #expect(config.expertCount == 8)
        #expect(config.expertsPerToken == 2)
    }

    @Test("Shared-norm parallel architecture → layerNorm, qkNorm=true")
    func extractParallelAttentionMLPNormKind() throws {
        let file = try makeLlamaGGUF(
            arch: "command-r",
            extraTensors: [
                "blk.0.attn_q_norm.weight",
                "blk.0.attn_k_norm.weight",
            ]
        )
        let config = try extractor.extract(from: file, architecture: .parallelAttentionMLP)

        #expect(config.normKind == .layerNorm)
        #expect(config.qkNorm == true)
    }

    @Test("Qwen 3.5 hybrid config extraction")
    func extractQwen35HybridConfig() throws {
        let file = try makeLlamaGGUF(
            arch: "qwen3moe",
            extraMetadata: [
                ("qwen3moe.full_attention_interval", .uint32(4)),
                ("qwen3moe.ssm.group_count", .uint32(16)),
                ("qwen3moe.ssm.state_size", .uint32(128)),
                ("qwen3moe.ssm.conv_kernel", .uint32(4)),
                ("qwen3moe.rope.partial_rotary_factor", .float32(0.25)),
            ]
        )
        let config = try extractor.extract(from: file, architecture: .hybridDeltaNetAttention)

        #expect(config.fullAttentionInterval == 4)
        #expect(config.linearKeyHeads == 16)
        #expect(config.linearValueHeads == 16)
        #expect(config.linearKeyHeadDim == 128)
        #expect(config.linearValueHeadDim == 128)
        #expect(config.convKernelSize == 4)
        #expect(config.partialRotaryFactor == 0.25)
    }

    @Test("Hybrid state-space / attention extraction throws when partial rotary factor is missing")
    func extractHybridDeltaNetAttentionMissingPartialRotaryFactor() throws {
        let file = try makeLlamaGGUF(
            arch: "qwen35",
            extraMetadata: [
                ("qwen35.full_attention_interval", .uint32(4)),
                ("qwen35.ssm.group_count", .uint32(16)),
                ("qwen35.ssm.state_size", .uint32(128)),
                ("qwen35.ssm.conv_kernel", .uint32(4)),
            ]
        )
        #expect(throws: GGUFGraphBuildError.self) {
            try extractor.extract(from: file, architecture: .hybridDeltaNetAttention)
        }
    }

    @Test("Hybrid state-space / attention extraction reports inferred partial rotary factor")
    func extractHybridDeltaNetAttentionMissingPartialRotaryFactorIncludesDiagnostic() throws {
        let file = try makeLlamaGGUF(
            arch: "qwen35",
            extraMetadata: [
                ("qwen35.attention.key_length", .uint32(256)),
                ("qwen35.full_attention_interval", .uint32(4)),
                ("qwen35.ssm.group_count", .uint32(16)),
                ("qwen35.ssm.state_size", .uint32(128)),
                ("qwen35.ssm.conv_kernel", .uint32(4)),
                ("qwen35.rope.dimension_count", .uint32(64)),
            ]
        )

        do {
            _ = try extractor.extract(from: file, architecture: .hybridDeltaNetAttention)
            Issue.record("Expected missing metadata error")
        } catch let error as GGUFGraphBuildError {
            let description = error.description
            #expect(description.contains("qwen35.rope.dimension_count=64"))
            #expect(description.contains("qwen35.attention.key_length=256"))
            #expect(description.contains("inferred factor would be 0.25"))
        }
    }

    @Test("RoPE scaling extraction")
    func extractRoPEScaling() throws {
        let file = try makeLlamaGGUF(
            extraMetadata: [
                ("llama.rope.scaling.type", .string("linear")),
                ("llama.rope.scaling.factor", .float32(2.0)),
                ("llama.rope.scaling.original_max_position_embeddings", .uint32(4096)),
            ]
        )
        let config = try extractor.extract(from: file, architecture: .transformer)

        #expect(config.ropeScaling != nil)
        #expect(config.ropeScaling?.kind == .linear)
        #expect(config.ropeScaling?.factor == 2.0)
        #expect(config.ropeScaling?.originalMaxPositions == 4096)
    }

    @Test("Missing required metadata throws")
    func extractMissingMetadataThrows() {
        // No embedding_length metadata
        var builder = GGUFTestBuilder()
        builder.addMetadata("general.architecture", value: .string("llama"))
        builder.addMetadata("llama.block_count", value: .uint32(4))
        // Missing llama.embedding_length
        builder.addTensor(name: "token_embd.weight", shape: [256, 100], type: .f16)
        let data = builder.build()

        #expect(throws: GGUFGraphBuildError.self) {
            let file = try GGUFFile.parse(data: data)
            _ = try extractor.extract(from: file, architecture: .transformer)
        }
    }
}

@Suite("Qwen35 Load Configuration")
struct Qwen35LoadConfigurationTests {

    @Test("Qwen35 loader requires ssm.state_size metadata")
    func qwen35LoaderRequiresStateSize() throws {
        let file = try makeLlamaGGUF(
            arch: "qwen35",
            extraMetadata: [
                ("qwen35.ssm.group_count", .uint32(16)),
                ("qwen35.ssm.conv_kernel", .uint32(4)),
                ("qwen35.full_attention_interval", .uint32(4)),
                ("qwen35.rope.partial_rotary_factor", .float32(0.25)),
            ]
        )

        #expect(throws: GGUFLoadError.self) {
            try Qwen35Configuration(from: file)
        }
    }

    @Test("Qwen35 loader requires ssm.conv_kernel metadata")
    func qwen35LoaderRequiresConvKernel() throws {
        let file = try makeLlamaGGUF(
            arch: "qwen35",
            extraMetadata: [
                ("qwen35.ssm.group_count", .uint32(16)),
                ("qwen35.ssm.state_size", .uint32(128)),
                ("qwen35.full_attention_interval", .uint32(4)),
                ("qwen35.rope.partial_rotary_factor", .float32(0.25)),
            ]
        )

        #expect(throws: GGUFLoadError.self) {
            try Qwen35Configuration(from: file)
        }
    }

    @Test("Qwen35 loader requires full_attention_interval metadata")
    func qwen35LoaderRequiresFullAttentionInterval() throws {
        let file = try makeLlamaGGUF(
            arch: "qwen35",
            extraMetadata: [
                ("qwen35.ssm.group_count", .uint32(16)),
                ("qwen35.ssm.state_size", .uint32(128)),
                ("qwen35.ssm.conv_kernel", .uint32(4)),
                ("qwen35.rope.partial_rotary_factor", .float32(0.25)),
            ]
        )

        #expect(throws: GGUFLoadError.self) {
            try Qwen35Configuration(from: file)
        }
    }

    @Test("Qwen35 loader requires rope.partial_rotary_factor metadata")
    func qwen35LoaderRequiresPartialRotaryFactor() throws {
        let file = try makeLlamaGGUF(
            arch: "qwen35",
            extraMetadata: [
                ("qwen35.ssm.group_count", .uint32(16)),
                ("qwen35.ssm.state_size", .uint32(128)),
                ("qwen35.ssm.conv_kernel", .uint32(4)),
                ("qwen35.full_attention_interval", .uint32(4)),
            ]
        )

        #expect(throws: GGUFLoadError.self) {
            try Qwen35Configuration(from: file)
        }
    }

    @Test("Qwen35 loader reports inferred partial rotary factor")
    func qwen35LoaderPartialRotaryFactorDiagnostic() throws {
        let file = try makeLlamaGGUF(
            arch: "qwen35",
            extraMetadata: [
                ("qwen35.attention.key_length", .uint32(256)),
                ("qwen35.ssm.group_count", .uint32(16)),
                ("qwen35.ssm.state_size", .uint32(128)),
                ("qwen35.ssm.conv_kernel", .uint32(4)),
                ("qwen35.full_attention_interval", .uint32(4)),
                ("qwen35.rope.dimension_count", .uint32(64)),
            ]
        )

        do {
            _ = try Qwen35Configuration(from: file)
            Issue.record("Expected missing metadata error")
        } catch let error as GGUFLoadError {
            let description = error.description
            #expect(description.contains("qwen35.rope.dimension_count=64"))
            #expect(description.contains("qwen35.attention.key_length=256"))
            #expect(description.contains("inferred factor would be 0.25"))
        }
    }
}

// MARK: - Phase 3: IR Assembly — Transformer

@Suite("IRGraphAssembler — Transformer")
struct IRAssemblerTransformerTests {

    let assembler = IRGraphAssembler()

    func makeTransformerConfig(
        layers: Int = 4,
        tied: Bool = true
    ) -> GGUFModelConfig {
        GGUFModelConfig(
            hiddenSize: 256,
            layerCount: layers,
            intermediateSize: 512,
            vocabSize: 32000,
            attentionHeads: 4,
            kvHeads: 2,
            headDim: 64,
            attentionBias: false,
            mlpBias: false,
            normEps: 1e-5,
            normKind: .rmsNorm,
            ropeTheta: 10000.0,
            ropeDimension: 64,
            ropeScaling: nil,
            tiedEmbeddings: tied,
            expertCount: nil,
            expertsPerToken: nil,
            qkNorm: false,
            fullAttentionInterval: nil,
            linearKeyHeads: nil,
            linearValueHeads: nil,
            linearKeyHeadDim: nil,
            linearValueHeadDim: nil,
            convKernelSize: nil,
            partialRotaryFactor: nil,
            slidingWindow: nil,
            mropeAxes: nil
        )
    }

    @Test("Root region: 4 ops [embed, repeating, norm, outputHead]")
    func assembleTransformerRootStructure() throws {
        let config = makeTransformerConfig()
        let graph = try assembler.assemble(config: config, architecture: .transformer)
        let root = graph.rootRegion

        #expect(root.operations.count == 4)
        #expect(root.parameters.isEmpty)
        #expect(root.results.count == 1)

        // Verify operation kinds
        if case .tokenEmbedding = root.operations[0].kind {} else {
            Issue.record("op0 should be tokenEmbedding, got \(root.operations[0].kind)")
        }
        if case .repeating(let count, _) = root.operations[1].kind {
            #expect(count == 4)
        } else {
            Issue.record("op1 should be repeating, got \(root.operations[1].kind)")
        }
        if case .rmsNorm = root.operations[2].kind {} else {
            Issue.record("op2 should be rmsNorm, got \(root.operations[2].kind)")
        }
        if case .outputHead = root.operations[3].kind {} else {
            Issue.record("op3 should be outputHead, got \(root.operations[3].kind)")
        }
    }

    @Test("Decoder body: 2 residuals [attn, mlp]")
    func assembleTransformerRepeatingBody() throws {
        let config = makeTransformerConfig()
        let graph = try assembler.assemble(config: config, architecture: .transformer)
        let root = graph.rootRegion

        guard case .repeating(_, let body) = root.operations[1].kind else {
            Issue.record("op1 should be repeating")
            return
        }

        // Decoder body has 1 param (input), 2 ops (residuals), 1 result
        #expect(body.parameters.count == 1)
        #expect(body.operations.count == 2)
        #expect(body.results.count == 1)

        // Both should be residual operations
        if case .residual(.add, _) = body.operations[0].kind {} else {
            Issue.record("body.op0 should be residual(.add), got \(body.operations[0].kind)")
        }
        if case .residual(.add, _) = body.operations[1].kind {} else {
            Issue.record("body.op1 should be residual(.add), got \(body.operations[1].kind)")
        }
    }

    @Test("Attention residual body: [rmsNorm, attention]")
    func assembleTransformerAttnResidual() throws {
        let config = makeTransformerConfig()
        let graph = try assembler.assemble(config: config, architecture: .transformer)
        let root = graph.rootRegion

        guard case .repeating(_, let decoderBody) = root.operations[1].kind,
              case .residual(_, let attnBody) = decoderBody.operations[0].kind
        else {
            Issue.record("Cannot navigate to attention residual body")
            return
        }

        #expect(attnBody.parameters.count == 1)
        #expect(attnBody.operations.count == 2)

        if case .rmsNorm = attnBody.operations[0].kind {} else {
            Issue.record("attnBody.op0 should be rmsNorm")
        }
        if case .attention = attnBody.operations[1].kind {} else {
            Issue.record("attnBody.op1 should be attention")
        }
    }

    @Test("MLP residual body: [rmsNorm, mlp]")
    func assembleTransformerMlpResidual() throws {
        let config = makeTransformerConfig()
        let graph = try assembler.assemble(config: config, architecture: .transformer)
        let root = graph.rootRegion

        guard case .repeating(_, let decoderBody) = root.operations[1].kind,
              case .residual(_, let mlpBody) = decoderBody.operations[1].kind
        else {
            Issue.record("Cannot navigate to MLP residual body")
            return
        }

        #expect(mlpBody.parameters.count == 1)
        #expect(mlpBody.operations.count == 2)

        if case .rmsNorm = mlpBody.operations[0].kind {} else {
            Issue.record("mlpBody.op0 should be rmsNorm")
        }
        if case .mlp = mlpBody.operations[1].kind {} else {
            Issue.record("mlpBody.op1 should be mlp")
        }
    }

    @Test("All ValueIDs produced exactly once, no dangling references")
    func assembleTransformerSSAWellFormed() throws {
        let config = makeTransformerConfig(layers: 2)
        let graph = try assembler.assemble(config: config, architecture: .transformer)

        // Collect all defined ValueIDs and all used ValueIDs
        var defined = Set<ValueID>()
        var used = Set<ValueID>()
        collectValues(region: graph.rootRegion, defined: &defined, used: &used)

        // Every used value must be defined somewhere
        let danglingRefs = used.subtracting(defined)
        #expect(danglingRefs.isEmpty, "Dangling value references: \(danglingRefs)")

        // No duplicate definitions (SSA property — every ID produced exactly once)
        // We check this by counting definitions
        var defCount: [ValueID: Int] = [:]
        countDefinitions(region: graph.rootRegion, counts: &defCount)
        let duplicates = defCount.filter { $0.value > 1 }
        #expect(duplicates.isEmpty, "Duplicate definitions: \(duplicates)")
    }

    @Test("Tied output head has tiedToEmbedding == true")
    func assembleTransformerTiedOutputHead() throws {
        let config = makeTransformerConfig(tied: true)
        let graph = try assembler.assemble(config: config, architecture: .transformer)
        let root = graph.rootRegion

        guard case .outputHead(let attrs) = root.operations[3].kind else {
            Issue.record("op3 should be outputHead")
            return
        }
        #expect(attrs.tiedToEmbedding == true)
    }

    @Test("Untied output head has tiedToEmbedding == false")
    func assembleTransformerUntiedOutputHead() throws {
        let config = makeTransformerConfig(tied: false)
        let graph = try assembler.assemble(config: config, architecture: .transformer)
        let root = graph.rootRegion

        guard case .outputHead(let attrs) = root.operations[3].kind else {
            Issue.record("op3 should be outputHead")
            return
        }
        #expect(attrs.tiedToEmbedding == false)
    }

    @Test("Attention attributes: headCount, kvHeads, headDim, rope, bias")
    func assembleTransformerAttentionAttrs() throws {
        let config = makeTransformerConfig()
        let graph = try assembler.assemble(config: config, architecture: .transformer)
        let root = graph.rootRegion

        guard case .repeating(_, let body) = root.operations[1].kind,
              case .residual(_, let attnBody) = body.operations[0].kind,
              case .attention(let attrs) = attnBody.operations[1].kind
        else {
            Issue.record("Cannot navigate to attention attributes")
            return
        }

        #expect(attrs.hiddenSize == 256)
        #expect(attrs.headCount == 4)
        #expect(attrs.kvHeadCount == 2)
        #expect(attrs.headDimension == 64)
        #expect(attrs.bias == false)
        #expect(attrs.causal == true)
        #expect(attrs.rope != nil)
        #expect(attrs.rope?.dimension == 64)
        #expect(attrs.rope?.base == 10000.0)
        #expect(attrs.qkNorm == nil)
    }
}

// MARK: - Phase 4: IR Assembly — Variants

@Suite("IRGraphAssembler — Variants")
struct IRAssemblerVariantTests {

    let assembler = IRGraphAssembler()

    @Test("Shared-norm parallel transformer uses layerNorm instead of rmsNorm")
    func assembleParallelAttentionMLPLayerNorm() throws {
        let config = GGUFModelConfig(
            hiddenSize: 256, layerCount: 2, intermediateSize: 512,
            vocabSize: 32000, attentionHeads: 4, kvHeads: 2, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .layerNorm, ropeTheta: 10000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil,
            qkNorm: true,
            fullAttentionInterval: nil,
            linearKeyHeads: nil, linearValueHeads: nil,
            linearKeyHeadDim: nil, linearValueHeadDim: nil,
            convKernelSize: nil, partialRotaryFactor: nil, slidingWindow: nil, mropeAxes: nil
        )
        let graph = try assembler.assemble(config: config, architecture: .parallelAttentionMLP)
        let root = graph.rootRegion

        // Final norm should be layerNorm
        if case .layerNorm = root.operations[2].kind {} else {
            Issue.record("Final norm should be layerNorm, got \(root.operations[2].kind)")
        }

        // Shared-norm decoder block: single residual → body has [layerNorm, parallel(attn, mlp)]
        guard case .repeating(_, let decoderBody) = root.operations[1].kind,
              case .residual(_, let residualBody) = decoderBody.operations[0].kind
        else {
            Issue.record("Cannot navigate to shared-norm parallel decoder block")
            return
        }

        // Single shared LayerNorm in residual body
        if case .layerNorm = residualBody.operations[0].kind {} else {
            Issue.record("Shared norm should be layerNorm, got \(residualBody.operations[0].kind)")
        }

        // Decoder body has only 1 residual (not 2 like standard transformer)
        #expect(decoderBody.operations.count == 1)
    }

    @Test("Shared-norm parallel attention has qkNorm: .layerNorm")
    func assembleParallelAttentionMLPQKNorm() throws {
        let config = GGUFModelConfig(
            hiddenSize: 256, layerCount: 2, intermediateSize: 512,
            vocabSize: 32000, attentionHeads: 4, kvHeads: 2, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .layerNorm, ropeTheta: 10000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil,
            qkNorm: true,
            fullAttentionInterval: nil,
            linearKeyHeads: nil, linearValueHeads: nil,
            linearKeyHeadDim: nil, linearValueHeadDim: nil,
            convKernelSize: nil, partialRotaryFactor: nil, slidingWindow: nil, mropeAxes: nil
        )
        let graph = try assembler.assemble(config: config, architecture: .parallelAttentionMLP)
        let root = graph.rootRegion

        // Shared-norm parallel: repeating → decoderBody (1 residual) → residualBody (norm, parallel)
        // parallel branches: [attnBranch, mlpBranch]
        guard case .repeating(_, let decoderBody) = root.operations[1].kind,
              case .residual(_, let residualBody) = decoderBody.operations[0].kind,
              case .parallel(_, let branches) = residualBody.operations[1].kind,
              case .attention(let attrs) = branches[0].operations[0].kind
        else {
            Issue.record("Cannot navigate to shared-norm parallel attention")
            return
        }
        #expect(attrs.qkNorm == .layerNorm)
    }

    @Test("MoE replaces MLP with MoE in second residual")
    func assembleMoEReplacesMLPWithMoE() throws {
        let config = GGUFModelConfig(
            hiddenSize: 256, layerCount: 2, intermediateSize: 512,
            vocabSize: 32000, attentionHeads: 4, kvHeads: 2, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 10000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: 8, expertsPerToken: 2,
            qkNorm: false,
            fullAttentionInterval: nil,
            linearKeyHeads: nil, linearValueHeads: nil,
            linearKeyHeadDim: nil, linearValueHeadDim: nil,
            convKernelSize: nil, partialRotaryFactor: nil, slidingWindow: nil, mropeAxes: nil
        )
        let graph = try assembler.assemble(config: config, architecture: .moe)
        let root = graph.rootRegion

        guard case .repeating(_, let body) = root.operations[1].kind,
              case .residual(_, let moeBody) = body.operations[1].kind
        else {
            Issue.record("Cannot navigate to MoE residual")
            return
        }

        // Second op in the MoE residual should be .moe, not .mlp
        if case .moe(let attrs) = moeBody.operations[1].kind {
            #expect(attrs.expertCount == 8)
            #expect(attrs.expertsPerToken == 2)
        } else {
            Issue.record("Expected .moe, got \(moeBody.operations[1].kind)")
        }

        // First residual is still attention
        guard case .residual(_, let attnBody) = body.operations[0].kind else {
            Issue.record("First residual should exist")
            return
        }
        if case .attention = attnBody.operations[1].kind {} else {
            Issue.record("First residual body.op1 should be attention")
        }
    }

    @Test("Qwen 3.5 uses layerStack instead of repeating")
    func assembleQwen35LayerStack() throws {
        let config = makeQwen35Config(layers: 4, interval: 4)
        let graph = try assembler.assemble(config: config, architecture: .hybridDeltaNetAttention)
        let root = graph.rootRegion

        if case .layerStack(let layers) = root.operations[1].kind {
            #expect(layers.count == 4)
        } else {
            Issue.record("op1 should be layerStack, got \(root.operations[1].kind)")
        }
    }

    @Test("Qwen 3.5 heterogeneous layers: DeltaNet has .stateSpace, full attn has .attention")
    func assembleQwen35HeterogeneousLayers() throws {
        // 4 layers, interval=4: layers 0,1,2 = DeltaNet, layer 3 = Full Attention
        let config = makeQwen35Config(layers: 4, interval: 4)
        let graph = try assembler.assemble(config: config, architecture: .hybridDeltaNetAttention)
        let root = graph.rootRegion

        guard case .layerStack(let layers) = root.operations[1].kind else {
            Issue.record("op1 should be layerStack")
            return
        }

        // Layers 0,1,2: DeltaNet (stateSpace in first residual)
        for i in 0..<3 {
            guard case .residual(_, let body) = layers[i].operations[0].kind else {
                Issue.record("Layer \(i) first op should be residual")
                continue
            }
            if case .stateSpace = body.operations[1].kind {} else {
                Issue.record("Layer \(i) should have .stateSpace, got \(body.operations[1].kind)")
            }
        }

        // Layer 3: Full Attention
        guard case .residual(_, let attnBody) = layers[3].operations[0].kind else {
            Issue.record("Layer 3 first op should be residual")
            return
        }
        if case .attention(let attrs) = attnBody.operations[1].kind {
            // Verify Qwen 3.5 specific attention attrs
            #expect(attrs.qkNorm == .rmsNorm)
            #expect(attrs.outputGate == .sigmoidPackedInQProj)
        } else {
            Issue.record("Layer 3 should have .attention, got \(attnBody.operations[1].kind)")
        }
    }

    @Test("Qwen 3.5 partial rotary factor applied to attention rope dimension")
    func assembleQwen35PartialRotary() throws {
        let config = makeQwen35Config(layers: 4, interval: 4)
        let graph = try assembler.assemble(config: config, architecture: .hybridDeltaNetAttention)
        let root = graph.rootRegion

        guard case .layerStack(let layers) = root.operations[1].kind,
              case .residual(_, let attnBody) = layers[3].operations[0].kind,
              case .attention(let attrs) = attnBody.operations[1].kind
        else {
            Issue.record("Cannot navigate to full attention layer")
            return
        }

        // headDim=64, partialRotaryFactor=0.25 → ropePartialDim=16
        #expect(attrs.rope?.dimension == 16)
    }

    // MARK: Helpers

    func makeQwen35Config(layers: Int, interval: Int) -> GGUFModelConfig {
        GGUFModelConfig(
            hiddenSize: 256, layerCount: layers, intermediateSize: 512,
            vocabSize: 32000, attentionHeads: 4, kvHeads: 2, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-6,
            normKind: .rmsNorm, ropeTheta: 10_000_000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil,
            qkNorm: false,
            fullAttentionInterval: interval,
            linearKeyHeads: 8, linearValueHeads: 8,
            linearKeyHeadDim: 64, linearValueHeadDim: 64,
            convKernelSize: 4, partialRotaryFactor: 0.25, slidingWindow: nil, mropeAxes: nil
        )
    }
}

// MARK: - Phase 5: Slot Enumeration Compatibility

@Suite("Slot Enumeration Compatibility")
struct SlotEnumerationTests {

    let assembler = IRGraphAssembler()
    let enumerator = ModelGraphSlotEnumerator()

    @Test("Transformer slot manifest contains expected weight paths")
    func slotManifestTransformerPaths() throws {
        let config = GGUFModelConfig(
            hiddenSize: 256, layerCount: 2, intermediateSize: 512,
            vocabSize: 32000, attentionHeads: 4, kvHeads: 2, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 10000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil,
            qkNorm: false,
            fullAttentionInterval: nil,
            linearKeyHeads: nil, linearValueHeads: nil,
            linearKeyHeadDim: nil, linearValueHeadDim: nil,
            convKernelSize: nil, partialRotaryFactor: nil, slidingWindow: nil, mropeAxes: nil
        )
        let graph = try assembler.assemble(config: config, architecture: .transformer)
        let manifest = enumerator.enumerate(graph)
        let paths = Set(manifest.map(\.mlxWeightPath))

        // Token embedding
        #expect(paths.contains("model.embed_tokens.weight"))

        // Final norm
        #expect(paths.contains("model.norm.weight"))

        // Output head: tied, so lm_head.weight should NOT be present
        #expect(!paths.contains("lm_head.weight"))

        // Layer 0 attention
        #expect(paths.contains("model.layers.0.self_attn.q_proj.weight"))
        #expect(paths.contains("model.layers.0.self_attn.k_proj.weight"))
        #expect(paths.contains("model.layers.0.self_attn.v_proj.weight"))
        #expect(paths.contains("model.layers.0.self_attn.o_proj.weight"))

        // Layer 0 norms
        #expect(paths.contains("model.layers.0.input_layernorm.weight"))
        #expect(paths.contains("model.layers.0.post_attention_layernorm.weight"))

        // Layer 0 MLP
        #expect(paths.contains("model.layers.0.mlp.gate_proj.weight"))
        #expect(paths.contains("model.layers.0.mlp.up_proj.weight"))
        #expect(paths.contains("model.layers.0.mlp.down_proj.weight"))

        // Layer 1 should also be present
        #expect(paths.contains("model.layers.1.self_attn.q_proj.weight"))
        #expect(paths.contains("model.layers.1.mlp.gate_proj.weight"))
    }

    @Test("Shared-norm parallel slot manifest includes QK norm weights")
    func slotManifestParallelAttentionMLPPaths() throws {
        let config = GGUFModelConfig(
            hiddenSize: 256, layerCount: 2, intermediateSize: 512,
            vocabSize: 32000, attentionHeads: 4, kvHeads: 2, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .layerNorm, ropeTheta: 10000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil,
            qkNorm: true,
            fullAttentionInterval: nil,
            linearKeyHeads: nil, linearValueHeads: nil,
            linearKeyHeadDim: nil, linearValueHeadDim: nil,
            convKernelSize: nil, partialRotaryFactor: nil, slidingWindow: nil, mropeAxes: nil
        )
        let graph = try assembler.assemble(config: config, architecture: .parallelAttentionMLP)
        let manifest = enumerator.enumerate(graph)
        let paths = Set(manifest.map(\.mlxWeightPath))

        // QK norm weights
        #expect(paths.contains("model.layers.0.self_attn.q_norm.weight"))
        #expect(paths.contains("model.layers.0.self_attn.k_norm.weight"))

        // LayerNorm bias (affine=true)
        #expect(paths.contains("model.layers.0.input_layernorm.bias"))
        #expect(paths.contains("model.norm.bias"))
    }

    @Test("MoE slot manifest includes expert slots")
    func slotManifestMoEHasExpertSlots() throws {
        let config = GGUFModelConfig(
            hiddenSize: 256, layerCount: 2, intermediateSize: 512,
            vocabSize: 32000, attentionHeads: 4, kvHeads: 2, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 10000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: 4, expertsPerToken: 2,
            qkNorm: false,
            fullAttentionInterval: nil,
            linearKeyHeads: nil, linearValueHeads: nil,
            linearKeyHeadDim: nil, linearValueHeadDim: nil,
            convKernelSize: nil, partialRotaryFactor: nil, slidingWindow: nil, mropeAxes: nil
        )
        let graph = try assembler.assemble(config: config, architecture: .moe)
        let manifest = enumerator.enumerate(graph)
        let paths = Set(manifest.map(\.mlxWeightPath))

        // Router gate
        #expect(paths.contains("model.layers.0.block_sparse_moe.gate.weight"))

        // Expert weights (4 experts)
        for e in 0..<4 {
            #expect(paths.contains("model.layers.0.block_sparse_moe.experts.\(e).gate_proj.weight"))
            #expect(paths.contains("model.layers.0.block_sparse_moe.experts.\(e).up_proj.weight"))
            #expect(paths.contains("model.layers.0.block_sparse_moe.experts.\(e).down_proj.weight"))
        }
    }

    @Test("Qwen 3.5 slot manifest has both DeltaNet and attention slots")
    func slotManifestQwen35HasMixedSlots() throws {
        let config = GGUFModelConfig(
            hiddenSize: 256, layerCount: 4, intermediateSize: 512,
            vocabSize: 32000, attentionHeads: 4, kvHeads: 2, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-6,
            normKind: .rmsNorm, ropeTheta: 10_000_000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil,
            qkNorm: false,
            fullAttentionInterval: 4,
            linearKeyHeads: 8, linearValueHeads: 8,
            linearKeyHeadDim: 64, linearValueHeadDim: 64,
            convKernelSize: 4, partialRotaryFactor: 0.25, slidingWindow: nil, mropeAxes: nil
        )
        let graph = try assembler.assemble(config: config, architecture: .hybridDeltaNetAttention)
        let manifest = enumerator.enumerate(graph)
        let paths = Set(manifest.map(\.mlxWeightPath))

        // DeltaNet layer 0 (stateSpace)
        #expect(paths.contains("model.layers.0.linear_attn.in_proj_qkv.weight"))
        #expect(paths.contains("model.layers.0.linear_attn.out_proj.weight"))
        #expect(paths.contains("model.layers.0.linear_attn.conv1d.weight"))
        #expect(paths.contains("model.layers.0.linear_attn.norm.weight"))
        #expect(paths.contains("model.layers.0.linear_attn.A_log"))

        // Full attention layer 3
        #expect(paths.contains("model.layers.3.self_attn.q_proj.weight"))
        #expect(paths.contains("model.layers.3.self_attn.k_proj.weight"))
        #expect(paths.contains("model.layers.3.self_attn.v_proj.weight"))
        #expect(paths.contains("model.layers.3.self_attn.o_proj.weight"))

        // QK norm on full attention layers
        #expect(paths.contains("model.layers.3.self_attn.q_norm.weight"))
        #expect(paths.contains("model.layers.3.self_attn.k_norm.weight"))

        // MLP on all layers
        #expect(paths.contains("model.layers.0.mlp.gate_proj.weight"))
        #expect(paths.contains("model.layers.3.mlp.gate_proj.weight"))
    }
}

// MARK: - Phase 6: GGUFGraphBuilder Integration

@Suite("GGUFGraphBuilder Integration")
struct GGUFGraphBuilderIntegrationTests {

    let builder = GGUFGraphBuilder()

    @Test("Build transformer graph from synthetic GGUF")
    func buildTransformerFromGGUF() throws {
        let file = try makeLlamaGGUF(layers: 2)
        let result = try builder.build(file: file)

        #expect(result.architecture == .transformer)
        #expect(result.config.hiddenSize == 256)
        #expect(result.config.layerCount == 2)

        // Root structure: embed, repeating, norm, outputHead
        #expect(result.graph.rootRegion.operations.count == 4)
    }

    @Test("Build shared-norm parallel graph from synthetic GGUF")
    func buildParallelAttentionMLPFromGGUF() throws {
        let file = try makeLlamaGGUF(
            arch: "command-r",
            extraTensors: [
                "blk.0.attn_q_norm.weight",
                "blk.0.attn_k_norm.weight",
                // No ffn_norm.weight to trigger Cohere detection
            ],
            removeTensors: ["blk.0.ffn_norm.weight"]
        )
        let result = try builder.build(file: file)

        #expect(result.architecture == .parallelAttentionMLP)
        #expect(result.config.normKind == .layerNorm)
        #expect(result.config.qkNorm == true)
    }

    @Test("Build MoE graph from synthetic GGUF")
    func buildMoEFromGGUF() throws {
        let file = try makeLlamaGGUF(
            extraMetadata: [
                ("llama.expert_count", .uint32(4)),
                ("llama.expert_used_count", .uint32(2)),
            ],
            extraTensors: [
                "blk.0.ffn_gate_inp.weight",
                "blk.0.ffn_gate.0.weight",
            ]
        )
        let result = try builder.build(file: file)

        #expect(result.architecture == .moe)
        #expect(result.config.expertCount == 4)
        #expect(result.config.expertsPerToken == 2)
    }

    @Test("Build Qwen 3.5 hybrid graph from synthetic GGUF")
    func buildQwen35FromGGUF() throws {
        let file = try makeLlamaGGUF(
            arch: "qwen3moe",
            layers: 4,
            extraMetadata: [
                ("qwen3moe.full_attention_interval", .uint32(4)),
                ("qwen3moe.ssm.group_count", .uint32(8)),
                ("qwen3moe.ssm.state_size", .uint32(64)),
                ("qwen3moe.ssm.conv_kernel", .uint32(4)),
                ("qwen3moe.rope.partial_rotary_factor", .float32(0.25)),
            ],
            extraTensors: [
                "blk.0.ssm_beta.weight",
                "blk.0.ssm_alpha.weight",
            ]
        )
        let result = try builder.build(file: file)

        #expect(result.architecture == .hybridDeltaNetAttention)

        guard case .layerStack(let layers) = result.graph.rootRegion.operations[1].kind else {
            Issue.record("op1 should be layerStack")
            return
        }
        #expect(layers.count == 4)
    }

    @Test("Mapper selection by architecture")
    func mapperSelection() {
        #expect(builder.mapper(for: .transformer) is TransformerTensorNameMapper)
        #expect(builder.mapper(for: .parallelAttentionMLP) is ParallelAttentionMLPTensorNameMapper)
        #expect(builder.mapper(for: .moe) is MoETensorNameMapper)
        #expect(builder.mapper(for: .hybridDeltaNetAttention) is HybridDeltaNetAttentionTensorNameMapper)
    }
}

// MARK: - SSA Verification Helpers

/// Recursively collect all defined and used ValueIDs in a region.
private func collectValues(
    region: Region,
    defined: inout Set<ValueID>,
    used: inout Set<ValueID>
) {
    // Parameters define values
    for param in region.parameters {
        defined.insert(param.id)
    }

    for op in region.operations {
        // Operands use values
        for operand in op.operands {
            used.insert(operand.value)
        }
        // Results define values
        for result in op.results {
            defined.insert(result.id)
        }
        // Recurse into nested regions
        switch op.kind {
        case .residual(_, let body):
            collectValues(region: body, defined: &defined, used: &used)
        case .parallel(_, let branches):
            for branch in branches {
                collectValues(region: branch, defined: &defined, used: &used)
            }
        case .repeating(_, let body):
            collectValues(region: body, defined: &defined, used: &used)
        case .layerStack(let layers):
            for layer in layers {
                collectValues(region: layer, defined: &defined, used: &used)
            }
        default:
            break
        }
    }

    // Results use values
    for result in region.results {
        used.insert(result.value)
    }
}

/// Count how many times each ValueID is defined.
private func countDefinitions(region: Region, counts: inout [ValueID: Int]) {
    for param in region.parameters {
        counts[param.id, default: 0] += 1
    }
    for op in region.operations {
        for result in op.results {
            counts[result.id, default: 0] += 1
        }
        switch op.kind {
        case .residual(_, let body):
            countDefinitions(region: body, counts: &counts)
        case .parallel(_, let branches):
            for branch in branches { countDefinitions(region: branch, counts: &counts) }
        case .repeating(_, let body):
            countDefinitions(region: body, counts: &counts)
        case .layerStack(let layers):
            for layer in layers { countDefinitions(region: layer, counts: &counts) }
        default:
            break
        }
    }
}

// MARK: - Synthetic GGUF Helpers

/// Create a minimal GGUF file with llama-like metadata and tensors.
///
/// This builds a valid GGUF binary and parses it, returning a GGUFFile
/// suitable for testing architecture detection and config extraction.
private func makeLlamaGGUF(
    arch: String = "llama",
    embeddingLength: UInt32 = 256,
    blockCount: UInt32 = 4,
    headCount: UInt32 = 4,
    headCountKV: UInt32 = 2,
    feedForwardLength: UInt32 = 512,
    vocabSize: UInt32 = 32000,
    layers: Int? = nil,
    extraMetadata: [(String, GGUFMetadataValue)] = [],
    extraTensors: [String] = [],
    removeTensors: [String] = []
) throws -> GGUFFile {
    let layerCount = layers.map(UInt32.init) ?? blockCount

    var builder = GGUFTestBuilder()

    // Core metadata
    builder.addMetadata("general.architecture", value: .string(arch))
    builder.addMetadata("\(arch).embedding_length", value: .uint32(embeddingLength))
    builder.addMetadata("\(arch).block_count", value: .uint32(layerCount))
    builder.addMetadata("\(arch).attention.head_count", value: .uint32(headCount))
    builder.addMetadata("\(arch).attention.head_count_kv", value: .uint32(headCountKV))
    builder.addMetadata("\(arch).feed_forward_length", value: .uint32(feedForwardLength))
    builder.addMetadata("\(arch).vocab_size", value: .uint32(vocabSize))

    // Extra metadata
    for (key, value) in extraMetadata {
        builder.addMetadata(key, value: value)
    }

    // Standard tensors
    let removeSet = Set(removeTensors)
    let standardTensors = [
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.attn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.0.ffn_norm.weight",
        "output_norm.weight",
    ]

    for name in standardTensors where !removeSet.contains(name) {
        builder.addTensor(name: name, shape: [16, 16], type: .f16)
    }

    // Extra tensors
    for name in extraTensors {
        builder.addTensor(name: name, shape: [16, 16], type: .f16)
    }

    let data = builder.build()
    return try GGUFFile.parse(data: data)
}
