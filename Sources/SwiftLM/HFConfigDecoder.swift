import Foundation
import LMArchitecture

struct HFConfigDecoder {
    func modelType(from data: Data) throws -> String {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let modelType = json["model_type"] as? String else {
            throw ModelBundleLoaderError.invalidConfig("Missing model_type in config.json")
        }
        return modelType
    }

    func decode(from data: Data) throws -> ModelConfig {
        guard let rawJson = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ModelBundleLoaderError.invalidConfig("config.json is not a JSON object")
        }

        let json: [String: Any]
        if let textConfig = rawJson["text_config"] as? [String: Any], textConfig["hidden_size"] != nil {
            var merged = rawJson
            for (key, value) in textConfig { merged[key] = value }
            json = merged
        } else {
            json = rawJson
        }

        guard let hiddenSize = json["hidden_size"] as? Int else {
            throw ModelBundleLoaderError.invalidConfig("Missing hidden_size")
        }
        guard let layerCount = json["num_hidden_layers"] as? Int else {
            throw ModelBundleLoaderError.invalidConfig("Missing num_hidden_layers")
        }

        let rawIntermediateSize = json["intermediate_size"] as? Int
            ?? json["block_ff_dim"] as? Int
            ?? hiddenSize * 4
        let autoAdjust = json["block_auto_adjust_ff_dim"] as? Bool
            ?? json["block_use_swiglu"] as? Bool
            ?? false
        let intermediateSize: Int
        if autoAdjust {
            var adjusted = rawIntermediateSize * 2 / 3
            if let multiplier = json["block_ffn_dim_multiplier"] as? Double {
                adjusted = Int(multiplier * Double(adjusted))
            }
            let multipleOf = json["block_multiple_of"] as? Int ?? 256
            adjusted = multipleOf * ((adjusted + multipleOf - 1) / multipleOf)
            intermediateSize = adjusted
        } else {
            intermediateSize = rawIntermediateSize
        }

        guard let vocabSize = json["vocab_size"] as? Int else {
            throw ModelBundleLoaderError.invalidConfig("Missing vocab_size")
        }
        let attentionHeads = json["num_attention_heads"] as? Int ?? 32
        let kvHeads = json["num_key_value_heads"] as? Int ?? attentionHeads
        let headDim = json["head_dim"] as? Int ?? (hiddenSize / attentionHeads)
        let normEps = (json["rms_norm_eps"] as? Double
            ?? json["layer_norm_eps"] as? Double
            ?? json["norm_eps"] as? Double
            ?? json["block_norm_eps"] as? Double
            ?? 1e-6)

        let ropeParams = json["rope_parameters"] as? [String: Any]
        let slidingAttentionRoPE = ropeParams?["sliding_attention"] as? [String: Any]
        let fullAttentionRoPE = ropeParams?["full_attention"] as? [String: Any]
        let ropeTheta = json["rope_theta"] as? Double
            ?? (ropeParams?["rope_theta"] as? Double)
            ?? (slidingAttentionRoPE?["rope_theta"] as? Double)
            ?? 500000.0
        let tiedEmbeddings = json["tie_word_embeddings"] as? Bool
            ?? json["tie_embedding"] as? Bool
            ?? false

        let mropeAxes: MRoPEAxes?
        if let sections = ropeParams?["mrope_section"] as? [Int], !sections.isEmpty {
            let interleaved = ropeParams?["mrope_interleaved"] as? Bool ?? false
            mropeAxes = MRoPEAxes(sections: sections, interleaved: interleaved)
        } else {
            mropeAxes = nil
        }

        return ModelConfig(
            hiddenSize: hiddenSize,
            layerCount: layerCount,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            attentionHeads: attentionHeads,
            kvHeads: kvHeads,
            headDim: headDim,
            attentionBias: json["attention_bias"] as? Bool ?? false,
            mlpBias: json["mlp_bias"] as? Bool ?? false,
            normEps: Float(normEps),
            normKind: json["model_type"] as? String == "cohere" ? .layerNorm : .rmsNorm,
            ropeTheta: Float(ropeTheta),
            ropeDimension: json["rope_dim"] as? Int ?? headDim,
            ropeScaling: nil,
            tiedEmbeddings: tiedEmbeddings,
            expertCount: json["num_local_experts"] as? Int ?? json["num_experts"] as? Int,
            expertsPerToken: json["num_experts_per_tok"] as? Int,
            moeIntermediateSize: json["moe_intermediate_size"] as? Int,
            qkNorm: json["qk_norm"] as? Bool
                ?? (["lfm2", "lfm2_moe"].contains(json["model_type"] as? String ?? "")),
            fullAttentionInterval: json["full_attention_interval"] as? Int,
            ssmNumHeads: json["ssm_num_heads"] as? Int
                ?? json["linear_num_value_heads"] as? Int,
            ssmGroupCount: json["linear_num_key_heads"] as? Int,
            ssmKeyHeadDim: json["ssm_state_size"] as? Int
                ?? json["linear_key_head_dim"] as? Int,
            ssmValueHeadDim: json["ssm_state_size"] as? Int
                ?? json["linear_value_head_dim"] as? Int,
            convKernelSize: json["conv_kernel_size"] as? Int
                ?? json["linear_conv_kernel_dim"] as? Int,
            convLCache: json["conv_L_cache"] as? Int,
            partialRotaryFactor: (json["partial_rotary_factor"] as? Double
                ?? ropeParams?["partial_rotary_factor"] as? Double).map { Float($0) },
            slidingWindow: json["sliding_window"] as? Int,
            layerTypes: {
                if let types = json["layer_types"] as? [String] { return types }
                if let attnIdxs = json["full_attn_idxs"] as? [Int] {
                    let attnSet = Set(attnIdxs)
                    return (0..<layerCount).map { attnSet.contains($0) ? "full_attention" : "conv" }
                }
                return nil
            }(),
            hiddenSizePerLayerInput: json["hidden_size_per_layer_input"] as? Int,
            vocabSizePerLayerInput: json["vocab_size_per_layer_input"] as? Int,
            globalHeadDim: json["global_head_dim"] as? Int,
            globalKVHeads: json["num_global_key_value_heads"] as? Int,
            numKVSharedLayers: json["num_kv_shared_layers"] as? Int,
            useDoubleWideMLP: json["use_double_wide_mlp"] as? Bool ?? false,
            attentionKEqualsV: json["attention_k_eq_v"] as? Bool ?? false,
            fullAttentionRopeTheta: (fullAttentionRoPE?["rope_theta"] as? Double).map { Float($0) },
            fullAttentionPartialRotaryFactor: (fullAttentionRoPE?["partial_rotary_factor"] as? Double)
                .map { Float($0) },
            fullAttentionRoPEScaling: {
                guard let ropeType = fullAttentionRoPE?["rope_type"] as? String,
                      ropeType != "default" else {
                    return nil
                }
                return RoPEScaling(kind: .custom(ropeType), factor: 1.0)
            }(),
            finalLogitSoftcapping: (json["final_logit_softcapping"] as? Double).map { Float($0) },
            numDenseLayers: json["num_dense_layers"] as? Int ?? 0,
            mropeAxes: mropeAxes
        )
    }

    func inputCapabilities(
        from configData: Data,
        preprocessorConfigData: Data? = nil,
        visionConfiguration: ModelVisionConfiguration? = nil
    ) throws -> ModelInputCapabilities {
        guard let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw ModelBundleLoaderError.invalidConfig("config.json is not a JSON object")
        }

        let modelType = (config["model_type"] as? String ?? "").lowercased()
        let resolvedVisionConfiguration: ModelVisionConfiguration
        if let visionConfiguration {
            resolvedVisionConfiguration = visionConfiguration
        } else {
            resolvedVisionConfiguration = try self.visionConfiguration(
                from: configData,
                preprocessorConfigData: preprocessorConfigData
            ) ?? ModelVisionConfiguration()
        }

        let hasVisionConfig = config["vision_config"] != nil || modelType.contains("_vl")
        let supportsImages =
            resolvedVisionConfiguration.imageTokenID != nil &&
            (
                hasVisionConfig ||
                Gemma4Support.supportsImageProcessorClass(
                    resolvedVisionConfiguration.processorClass
                ) ||
                QwenVisionSupport.supportsImageProcessorClass(
                    resolvedVisionConfiguration.processorClass
                ) ||
                QwenVisionSupport.supportsImageProcessorType(
                    resolvedVisionConfiguration.imageProcessorType
                )
            )
        let supportsVideo =
            resolvedVisionConfiguration.videoTokenID != nil &&
            (
                hasVisionConfig ||
                QwenVisionSupport.supportsVideoProcessorClass(
                    resolvedVisionConfiguration.processorClass
                ) ||
                QwenVisionSupport.supportsVideoProcessorType(
                    resolvedVisionConfiguration.videoProcessorType
                )
            )

        return ModelInputCapabilities(
            supportsText: true,
            supportsImages: supportsImages,
            supportsVideo: supportsVideo
        )
    }

    func visionConfiguration(
        from configData: Data,
        preprocessorConfigData: Data? = nil
    ) throws -> ModelVisionConfiguration? {
        guard let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw ModelBundleLoaderError.invalidConfig("config.json is not a JSON object")
        }

        let hasVisionConfig = config["vision_config"] != nil
        let visionConfig = config["vision_config"] as? [String: Any]
        let imageTokenID = config["image_token_id"] as? Int
        let videoTokenID = config["video_token_id"] as? Int
        let visionStartTokenID = config["vision_start_token_id"] as? Int
        let visionEndTokenID = config["vision_end_token_id"] as? Int

        let processorConfig: [String: Any]?
        if let preprocessorConfigData {
            guard let json = try JSONSerialization.jsonObject(with: preprocessorConfigData) as? [String: Any] else {
                throw ModelBundleLoaderError.invalidConfig(
                    "preprocessor_config.json is not a JSON object"
                )
            }
            processorConfig = json
        } else {
            processorConfig = nil
        }

        let imageProcessorType = processorConfig?["image_processor_type"] as? String
        let videoProcessorType = processorConfig?["video_processor_type"] as? String
        let processorClass = processorConfig?["processor_class"] as? String
        let patchSize = processorConfig?["patch_size"] as? Int
        let temporalPatchSize = processorConfig?["temporal_patch_size"] as? Int
        let poolingKernelSize = processorConfig?["pooling_kernel_size"] as? Int
        let mergeSize = processorConfig?["merge_size"] as? Int
        let size = processorConfig?["size"] as? [String: Any]
        let imageMean = processorConfig?["image_mean"] as? [Double] ?? []
        let imageStd = processorConfig?["image_std"] as? [Double] ?? []
        let minimumPixelCount =
            size?["shortest_edge"] as? Int
            ?? processorConfig?["min_pixels"] as? Int
        let maximumPixelCount =
            size?["longest_edge"] as? Int
            ?? processorConfig?["max_pixels"] as? Int
        let videoFramesPerSecond =
            processorConfig?["fps"] as? Double
            ?? (processorConfig?["fps"] as? Int).map(Double.init)
        let minimumFrameCount =
            processorConfig?["min_frames"] as? Int
        let maximumFrameCount =
            processorConfig?["max_frames"] as? Int

        let hasKnownImageProcessorClass = QwenVisionSupport.supportsImageProcessorClass(
            processorClass
        ) || Gemma4Support.supportsImageProcessorClass(processorClass)
        let hasKnownImageProcessorType = QwenVisionSupport.supportsImageProcessorType(
            imageProcessorType
        )
        let hasKnownVideoProcessorType = QwenVisionSupport.supportsVideoProcessorType(
            videoProcessorType
        )
        guard hasVisionConfig ||
                imageTokenID != nil ||
                videoTokenID != nil ||
                visionStartTokenID != nil ||
                visionEndTokenID != nil ||
                hasKnownImageProcessorClass ||
                hasKnownImageProcessorType ||
                hasKnownVideoProcessorType ||
                patchSize != nil ||
                temporalPatchSize != nil ||
                mergeSize != nil ||
                minimumPixelCount != nil ||
                maximumPixelCount != nil else {
            return nil
        }

        return ModelVisionConfiguration(
            hiddenSize: visionConfig?["hidden_size"] as? Int,
            depth: visionConfig?["depth"] as? Int
                ?? visionConfig?["num_hidden_layers"] as? Int,
            intermediateSize: visionConfig?["intermediate_size"] as? Int,
            outHiddenSize: visionConfig?["out_hidden_size"] as? Int
                ?? visionConfig?["output_proj_dims"] as? Int,
            headCount: visionConfig?["num_heads"] as? Int
                ?? visionConfig?["num_attention_heads"] as? Int,
            numPositionEmbeddings: visionConfig?["num_position_embeddings"] as? Int,
            inChannels: visionConfig?["in_channels"] as? Int
                ?? visionConfig?["num_channels"] as? Int,
            hiddenAct: visionConfig?["hidden_act"] as? String
                ?? visionConfig?["hidden_activation"] as? String,
            deepstackVisualIndexes: visionConfig?["deepstack_visual_indexes"] as? [Int] ?? [],
            processorClass: processorClass,
            imageTokenID: imageTokenID,
            videoTokenID: videoTokenID,
            visionStartTokenID: visionStartTokenID,
            visionEndTokenID: visionEndTokenID,
            imageProcessorType: imageProcessorType,
            videoProcessorType: videoProcessorType,
            patchSize: patchSize ?? visionConfig?["patch_size"] as? Int,
            poolingKernelSize: poolingKernelSize ?? visionConfig?["pooling_kernel_size"] as? Int,
            temporalPatchSize: temporalPatchSize ?? visionConfig?["temporal_patch_size"] as? Int,
            mergeSize: mergeSize,
            spatialMergeSize: visionConfig?["spatial_merge_size"] as? Int,
            positionEmbeddingSize: visionConfig?["position_embedding_size"] as? Int,
            defaultOutputLength: visionConfig?["default_output_length"] as? Int
                ?? config["vision_soft_tokens_per_image"] as? Int,
            standardize: visionConfig?["standardize"] as? Bool,
            minimumPixelCount: minimumPixelCount,
            maximumPixelCount: maximumPixelCount,
            videoFramesPerSecond: videoFramesPerSecond,
            minimumFrameCount: minimumFrameCount,
            maximumFrameCount: maximumFrameCount,
            imageMean: imageMean,
            imageStd: imageStd
        )
    }
}
