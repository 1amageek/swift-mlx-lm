import LMArchitecture
import MetalCompiler
import ModelDeclarations

struct ModelGraphResolver {
    func resolveModelGraph(modelType: String, config: ModelConfig) throws -> ModelGraph {
        switch modelType.lowercased() {
        case "llama", "qwen2", "qwen3", "mistral", "gemma", "gemma2",
             "phi", "phi3", "starcoder2", "gpt_neox", "internlm2",
             "deepseek", "yi", "baichuan", "chatglm",
             "mixtral", "qwen2_moe", "deepseek_v2", "arctic", "dbrx":
            return try Transformer(config: config).makeModelGraph()
        case "gemma4", "gemma4_text":
            do {
                try Gemma4.validate(config)
            } catch let error as ModelGraphBuildError {
                throw ModelBundleLoaderError.invalidConfig(error.description)
            }
            return try Gemma4(config: config).makeModelGraph()
        case "qwen3_5", "qwen3_vl", "qwen2_5_vl", "qwen2_vl":
            do {
                try Qwen35.validate(config)
            } catch let error as ModelGraphBuildError {
                throw ModelBundleLoaderError.invalidConfig(error.description)
            }
            return try Qwen35(config: config).makeModelGraph()
        case "lfm2", "lfm2_moe":
            return try LFM2(config: config).makeModelGraph()
        case "cohere", "command-r":
            return try Cohere(config: config).makeModelGraph()
        case "nemotron_h":
            throw ModelBundleLoaderError.invalidConfig(
                "nemotron_h (Mamba-2 hybrid) is not yet supported"
            )
        default:
            throw ModelBundleLoaderError.invalidConfig(
                "Unsupported model_type: \(modelType)"
            )
        }
    }

    func namingConvention(for modelType: String) -> ParameterResolver.WeightNamingConvention {
        switch modelType.lowercased() {
        case "gemma4", "gemma4_text":
            return .gemma4Family
        case "qwen3_5", "qwen3_vl", "qwen2_5_vl", "qwen2_vl":
            return .qwen35Family
        case "lfm2", "lfm2_moe":
            return .lfm2Family
        default:
            return .llamaFamily
        }
    }
}
