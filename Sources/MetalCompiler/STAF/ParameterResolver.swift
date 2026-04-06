import LMIR

/// Resolves parameter bindings for a ModelGraph by walking the IR
/// and assigning tensor names based on a naming convention.
///
/// This connects the IR's structural operations to concrete tensor names
/// in safetensors/STAF. The resolver walks the graph, tracks layer indices
/// from repeating blocks, and constructs weight paths using the model
/// family's naming convention.
///
/// ```swift
/// let graph = try Transformer(config: config).makeModelGraph()
/// let resolved = ParameterResolver.resolve(graph: graph, convention: .llamaFamily)
/// // resolved.rootRegion now has parameterBindings on every primitive operation
/// ```
public struct ParameterResolver: Sendable {

    public init() {}

    /// Resolve all parameter bindings in a ModelGraph.
    ///
    /// Returns a new ModelGraph with `parameterBindings` populated on every
    /// primitive operation that has weight requirements.
    public func resolve(
        graph: ModelGraph,
        convention: WeightNamingConvention
    ) -> ModelGraph {
        let resolvedRegion = resolveRegion(
            graph.rootRegion,
            convention: convention,
            scope: .root,
            residualIndex: 0
        )
        return ModelGraph(rootRegion: resolvedRegion)
    }

    // MARK: - Naming Convention

    /// Weight naming convention for different model families.
    public enum WeightNamingConvention: Sendable {
        /// Llama-family: self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj
        case llamaFamily
        /// Qwen3.5-family: model.language_model.* with q_norm/k_norm attention weights.
        case qwen35Family
        /// Gemma4-family: model.language_model.* with per-layer input weights.
        case gemma4Family
        /// LFM2-family: self_attn.{q,k,v}_proj, out_proj, feed_forward.{w1,w2,w3}
        case lfm2Family
    }

    // MARK: - Scope Tracking

    private enum NamingScope {
        case root
        case layer(index: Int)
    }

    // MARK: - Region Walk

    private func resolveRegion(
        _ region: Region,
        convention: WeightNamingConvention,
        scope: NamingScope,
        residualIndex: Int
    ) -> Region {
        var operations: [Operation] = []
        var currentResidualIndex = residualIndex
        // Track layer index for flat-expanded LayerStack.
        // Each decoder layer = 2 residual blocks (norm+op, norm+mlp).
        // Count residual blocks in root scope and divide by 2 to get layer index.
        var residualCount = 0

        for operation in region.operations {
            var effectiveScope = scope
            // In root scope, assign layer index from residual block count.
            // Each pair of residuals = one decoder layer.
            if case .root = scope, case .residual = operation.kind {
                effectiveScope = .layer(index: residualCount / 2)
                residualCount += 1
            }

            let resolved = resolveOperation(
                operation,
                convention: convention,
                scope: effectiveScope,
                residualIndex: &currentResidualIndex
            )
            operations.append(resolved)
        }

        return Region(
            parameters: region.parameters,
            operations: operations,
            results: region.results
        )
    }

    private func resolveOperation(
        _ operation: Operation,
        convention: WeightNamingConvention,
        scope: NamingScope,
        residualIndex: inout Int
    ) -> Operation {
        switch operation.kind {

        case .primitive(let attributes):
            let bindings = buildBindings(
                attributes: attributes,
                convention: convention,
                scope: scope,
                residualIndex: residualIndex
            )
            return Operation(
                key: operation.key,
                kind: operation.kind,
                operands: operation.operands,
                results: operation.results,
                parameterBindings: bindings
            )

        case .residual(let strategy, let body):
            let savedIndex = residualIndex
            // Each layer has 2 residual blocks: index 0 = operator/attention norm,
            // index 1 = ffn norm. Use modulo to get the intra-layer position.
            let resolvedBody = resolveRegion(
                body, convention: convention,
                scope: scope, residualIndex: savedIndex % 2)
            residualIndex = savedIndex + 1
            return Operation(
                key: operation.key,
                kind: .residual(strategy: strategy, body: resolvedBody),
                operands: operation.operands,
                results: operation.results,
                parameterBindings: []
            )

        case .repeating(let count, let body):
            // Use layer 0 as template. The compiler substitutes
            // .layers.0. → .layers.{iteration}. during unroll.
            let templateBody = resolveRegion(
                body, convention: convention,
                scope: .layer(index: 0),
                residualIndex: 0)
            return Operation(
                key: operation.key,
                kind: .repeating(count: count, body: templateBody),
                operands: operation.operands,
                results: operation.results,
                parameterBindings: []
            )

        case .conditional(let condition, let thenBody, let elseBody):
            let resolvedThen = resolveRegion(
                thenBody, convention: convention,
                scope: scope, residualIndex: 0)
            let resolvedElse = resolveRegion(
                elseBody, convention: convention,
                scope: scope, residualIndex: 0)
            return Operation(
                key: operation.key,
                kind: .conditional(condition: condition, then: resolvedThen, else: resolvedElse),
                operands: operation.operands,
                results: operation.results,
                parameterBindings: []
            )

        case .parallel(let merge, let branches):
            let resolvedBranches = branches.map {
                resolveRegion($0, convention: convention, scope: scope, residualIndex: 0)
            }
            return Operation(
                key: operation.key,
                kind: .parallel(merge: merge, branches: resolvedBranches),
                operands: operation.operands,
                results: operation.results,
                parameterBindings: []
            )
        }
    }

    // MARK: - Binding Construction

    private func buildBindings(
        attributes: any OperationAttributes,
        convention: WeightNamingConvention,
        scope: NamingScope,
        residualIndex: Int
    ) -> [ParameterBinding] {
        switch convention {
        case .llamaFamily:
            return buildLlamaFamilyBindings(
                attributes: attributes, scope: scope, residualIndex: residualIndex)
        case .qwen35Family:
            return buildQwen35FamilyBindings(
                attributes: attributes, scope: scope, residualIndex: residualIndex)
        case .gemma4Family:
            return buildGemma4FamilyBindings(
                attributes: attributes, scope: scope, residualIndex: residualIndex)
        case .lfm2Family:
            return buildLFM2FamilyBindings(
                attributes: attributes, scope: scope, residualIndex: residualIndex)
        }
    }

    // MARK: - Llama Family

    private func buildLlamaFamilyBindings(
        attributes: any OperationAttributes,
        scope: NamingScope,
        residualIndex: Int
    ) -> [ParameterBinding] {
        if let _ = attributes as? TokenEmbeddingAttributes {
            return [ParameterBinding(role: "embedding_table", tensorName: "model.embed_tokens.weight")]
        }

        if let attrs = attributes as? OutputHeadAttributes {
            if attrs.tiedToEmbedding {
                return [ParameterBinding(role: "weight", tensorName: "model.embed_tokens.weight")]
            }
            return [ParameterBinding(role: "weight", tensorName: "lm_head.weight")]
        }

        guard case .layer(let layerIndex) = scope else {
            // Root-level norm
            if attributes is RMSNormAttributes || attributes is LayerNormAttributes {
                return [ParameterBinding(role: "scale", tensorName: "model.norm.weight")]
            }
            return []
        }

        let prefix = "model.layers.\(layerIndex)"

        if attributes is RMSNormAttributes || attributes is LayerNormAttributes {
            let normName = residualIndex == 0 ? "input_layernorm" : "post_attention_layernorm"
            return [ParameterBinding(role: "scale", tensorName: "\(prefix).\(normName).weight")]
        }

        if let attrs = attributes as? AttentionAttributes {
            let attnPrefix = "\(prefix).self_attn"
            var bindings = [
                ParameterBinding(role: "q_proj", tensorName: "\(attnPrefix).q_proj.weight"),
                ParameterBinding(role: "k_proj", tensorName: "\(attnPrefix).k_proj.weight"),
                ParameterBinding(role: "v_proj", tensorName: "\(attnPrefix).v_proj.weight"),
                ParameterBinding(role: "o_proj", tensorName: "\(attnPrefix).o_proj.weight"),
            ]
            if let qkNorm = attrs.qkNorm, qkNorm != .none {
                bindings.append(ParameterBinding(role: "q_layernorm", tensorName: "\(attnPrefix).q_layernorm.weight"))
                bindings.append(ParameterBinding(role: "k_layernorm", tensorName: "\(attnPrefix).k_layernorm.weight"))
            }
            return bindings
        }

        if let _ = attributes as? MLPAttributes {
            let mlpPrefix = "\(prefix).mlp"
            return [
                ParameterBinding(role: "gate_proj", tensorName: "\(mlpPrefix).gate_proj.weight"),
                ParameterBinding(role: "up_proj", tensorName: "\(mlpPrefix).up_proj.weight"),
                ParameterBinding(role: "down_proj", tensorName: "\(mlpPrefix).down_proj.weight"),
            ]
        }

        if let _ = attributes as? MoEAttributes {
            let moePrefix = "\(prefix).block_sparse_moe"
            return [
                ParameterBinding(role: "router", tensorName: "\(moePrefix).gate.weight"),
            ]
        }

        if let _ = attributes as? StateSpaceAttributes {
            let ssPrefix = "\(prefix).linear_attn"
            return [
                ParameterBinding(role: "in_proj_qkv", tensorName: "\(ssPrefix).in_proj_qkv.weight"),
                ParameterBinding(role: "in_proj_z", tensorName: "\(ssPrefix).in_proj_z.weight"),
                ParameterBinding(role: "in_proj_b", tensorName: "\(ssPrefix).in_proj_b.weight"),
                ParameterBinding(role: "in_proj_a", tensorName: "\(ssPrefix).in_proj_a.weight"),
                ParameterBinding(role: "out_proj", tensorName: "\(ssPrefix).out_proj.weight"),
                ParameterBinding(role: "scale", tensorName: "\(ssPrefix).norm.weight"),
                ParameterBinding(role: "conv_weight", tensorName: "\(ssPrefix).conv1d.weight"),
                ParameterBinding(role: "dt_bias", tensorName: "\(ssPrefix).dt_bias"),
                ParameterBinding(role: "A_log", tensorName: "\(ssPrefix).A_log"),
            ]
        }

        if let _ = attributes as? ShortConvAttributes {
            let convPrefix = "\(prefix).conv"
            return [
                ParameterBinding(role: "in_proj", tensorName: "\(convPrefix).in_proj.weight"),
                ParameterBinding(role: "conv_weight", tensorName: "\(convPrefix).conv.weight"),
                ParameterBinding(role: "out_proj", tensorName: "\(convPrefix).out_proj.weight"),
            ]
        }

        return []
    }

    // MARK: - Qwen3.5 Family

    private func buildQwen35FamilyBindings(
        attributes: any OperationAttributes,
        scope: NamingScope,
        residualIndex: Int
    ) -> [ParameterBinding] {
        if let _ = attributes as? TokenEmbeddingAttributes {
            return [ParameterBinding(role: "embedding_table", tensorName: "model.language_model.embed_tokens.weight")]
        }

        if let attrs = attributes as? OutputHeadAttributes {
            if attrs.tiedToEmbedding {
                return [ParameterBinding(role: "weight", tensorName: "model.language_model.embed_tokens.weight")]
            }
            return [ParameterBinding(role: "weight", tensorName: "lm_head.weight")]
        }

        guard case .layer(let layerIndex) = scope else {
            if attributes is RMSNormAttributes || attributes is LayerNormAttributes {
                return [ParameterBinding(role: "scale", tensorName: "model.language_model.norm.weight")]
            }
            return []
        }

        let prefix = "model.language_model.layers.\(layerIndex)"

        if attributes is RMSNormAttributes || attributes is LayerNormAttributes {
            let normName = residualIndex == 0 ? "input_layernorm" : "post_attention_layernorm"
            return [ParameterBinding(role: "scale", tensorName: "\(prefix).\(normName).weight")]
        }

        if let attrs = attributes as? AttentionAttributes {
            let attnPrefix = "\(prefix).self_attn"
            var bindings = [
                ParameterBinding(role: "q_proj", tensorName: "\(attnPrefix).q_proj.weight"),
                ParameterBinding(role: "k_proj", tensorName: "\(attnPrefix).k_proj.weight"),
                ParameterBinding(role: "v_proj", tensorName: "\(attnPrefix).v_proj.weight"),
                ParameterBinding(role: "o_proj", tensorName: "\(attnPrefix).o_proj.weight"),
            ]
            if let qkNorm = attrs.qkNorm, qkNorm != .none {
                bindings.append(ParameterBinding(role: "q_layernorm", tensorName: "\(attnPrefix).q_norm.weight"))
                bindings.append(ParameterBinding(role: "k_layernorm", tensorName: "\(attnPrefix).k_norm.weight"))
            }
            return bindings
        }

        if let _ = attributes as? MLPAttributes {
            let mlpPrefix = "\(prefix).mlp"
            return [
                ParameterBinding(role: "gate_proj", tensorName: "\(mlpPrefix).gate_proj.weight"),
                ParameterBinding(role: "up_proj", tensorName: "\(mlpPrefix).up_proj.weight"),
                ParameterBinding(role: "down_proj", tensorName: "\(mlpPrefix).down_proj.weight"),
            ]
        }

        if let _ = attributes as? MoEAttributes {
            let moePrefix = "\(prefix).block_sparse_moe"
            return [
                ParameterBinding(role: "router", tensorName: "\(moePrefix).gate.weight"),
            ]
        }

        if let _ = attributes as? StateSpaceAttributes {
            let ssPrefix = "\(prefix).linear_attn"
            return [
                ParameterBinding(role: "in_proj_qkv", tensorName: "\(ssPrefix).in_proj_qkv.weight"),
                ParameterBinding(role: "in_proj_z", tensorName: "\(ssPrefix).in_proj_z.weight"),
                ParameterBinding(role: "in_proj_b", tensorName: "\(ssPrefix).in_proj_b.weight"),
                ParameterBinding(role: "in_proj_a", tensorName: "\(ssPrefix).in_proj_a.weight"),
                ParameterBinding(role: "out_proj", tensorName: "\(ssPrefix).out_proj.weight"),
                ParameterBinding(role: "scale", tensorName: "\(ssPrefix).norm.weight"),
                ParameterBinding(role: "conv_weight", tensorName: "\(ssPrefix).conv1d.weight"),
                ParameterBinding(role: "dt_bias", tensorName: "\(ssPrefix).dt_bias"),
                ParameterBinding(role: "A_log", tensorName: "\(ssPrefix).A_log"),
            ]
        }

        if let _ = attributes as? ShortConvAttributes {
            let convPrefix = "\(prefix).conv"
            return [
                ParameterBinding(role: "in_proj", tensorName: "\(convPrefix).in_proj.weight"),
                ParameterBinding(role: "conv_weight", tensorName: "\(convPrefix).conv.weight"),
                ParameterBinding(role: "out_proj", tensorName: "\(convPrefix).out_proj.weight"),
            ]
        }

        return []
    }

    // MARK: - LFM2 Family

    private func buildGemma4FamilyBindings(
        attributes: any OperationAttributes,
        scope: NamingScope,
        residualIndex: Int
    ) -> [ParameterBinding] {
        if let _ = attributes as? TokenEmbeddingAttributes {
            return [ParameterBinding(role: "embedding_table", tensorName: "model.language_model.embed_tokens.weight")]
        }

        if let attrs = attributes as? OutputHeadAttributes {
            if attrs.tiedToEmbedding {
                return [ParameterBinding(role: "weight", tensorName: "model.language_model.embed_tokens.weight")]
            }
            return [ParameterBinding(role: "weight", tensorName: "lm_head.weight")]
        }

        guard case .layer(let layerIndex) = scope else {
            if attributes is RMSNormAttributes || attributes is LayerNormAttributes {
                return [ParameterBinding(role: "scale", tensorName: "model.language_model.norm.weight")]
            }
            return []
        }

        let prefix = "model.language_model.layers.\(layerIndex)"

        if attributes is RMSNormAttributes || attributes is LayerNormAttributes {
            let normName: String
            switch residualIndex {
            case 0:
                normName = "input_layernorm"
            case 1:
                normName = "pre_feedforward_layernorm"
            default:
                return []
            }
            return [ParameterBinding(role: "scale", tensorName: "\(prefix).\(normName).weight")]
        }

        if let attrs = attributes as? AttentionAttributes {
            let attnPrefix = "\(prefix).self_attn"
            var bindings = [
                ParameterBinding(role: "q_proj", tensorName: "\(attnPrefix).q_proj.weight"),
                ParameterBinding(role: "k_proj", tensorName: "\(attnPrefix).k_proj.weight"),
                ParameterBinding(role: "o_proj", tensorName: "\(attnPrefix).o_proj.weight"),
            ]
            if attrs.qkNorm == .rmsNorm {
                bindings.append(ParameterBinding(role: "q_layernorm", tensorName: "\(attnPrefix).q_norm.weight"))
                bindings.append(ParameterBinding(role: "k_layernorm", tensorName: "\(attnPrefix).k_norm.weight"))
            }
            bindings.append(ParameterBinding(role: "v_proj", tensorName: "\(attnPrefix).v_proj.weight"))
            return bindings
        }

        if let _ = attributes as? MLPAttributes {
            let mlpPrefix = "\(prefix).mlp"
            return [
                ParameterBinding(role: "gate_proj", tensorName: "\(mlpPrefix).gate_proj.weight"),
                ParameterBinding(role: "up_proj", tensorName: "\(mlpPrefix).up_proj.weight"),
                ParameterBinding(role: "down_proj", tensorName: "\(mlpPrefix).down_proj.weight"),
            ]
        }

        if let _ = attributes as? PerLayerInputAttributes {
            return [
                ParameterBinding(
                    role: "per_layer_embedding_table",
                    tensorName: "model.language_model.embed_tokens_per_layer.weight"),
                ParameterBinding(
                    role: "per_layer_model_projection",
                    tensorName: "model.language_model.per_layer_model_projection.weight"),
                ParameterBinding(
                    role: "per_layer_projection_norm",
                    tensorName: "model.language_model.per_layer_projection_norm.weight"),
                ParameterBinding(
                    role: "per_layer_input_gate",
                    tensorName: "\(prefix).per_layer_input_gate.weight"),
                ParameterBinding(
                    role: "per_layer_projection",
                    tensorName: "\(prefix).per_layer_projection.weight"),
                ParameterBinding(
                    role: "post_per_layer_input_norm",
                    tensorName: "\(prefix).post_per_layer_input_norm.weight"),
            ]
        }

        return []
    }

    // MARK: - LFM2 Family

    private func buildLFM2FamilyBindings(
        attributes: any OperationAttributes,
        scope: NamingScope,
        residualIndex: Int
    ) -> [ParameterBinding] {
        if let _ = attributes as? TokenEmbeddingAttributes {
            return [ParameterBinding(role: "embedding_table", tensorName: "model.embed_tokens.weight")]
        }

        if let attrs = attributes as? OutputHeadAttributes {
            if attrs.tiedToEmbedding {
                return [ParameterBinding(role: "weight", tensorName: "model.embed_tokens.weight")]
            }
            return [ParameterBinding(role: "weight", tensorName: "lm_head.weight")]
        }

        guard case .layer(let layerIndex) = scope else {
            if attributes is RMSNormAttributes || attributes is LayerNormAttributes {
                return [ParameterBinding(role: "scale", tensorName: "model.embedding_norm.weight")]
            }
            return []
        }

        let prefix = "model.layers.\(layerIndex)"

        if attributes is RMSNormAttributes || attributes is LayerNormAttributes {
            let normName = residualIndex == 0 ? "operator_norm" : "ffn_norm"
            return [ParameterBinding(role: "scale", tensorName: "\(prefix).\(normName).weight")]
        }

        if let attrs = attributes as? AttentionAttributes {
            let attnPrefix = "\(prefix).self_attn"
            var bindings = [
                ParameterBinding(role: "q_proj", tensorName: "\(attnPrefix).q_proj.weight"),
                ParameterBinding(role: "k_proj", tensorName: "\(attnPrefix).k_proj.weight"),
                ParameterBinding(role: "v_proj", tensorName: "\(attnPrefix).v_proj.weight"),
                ParameterBinding(role: "o_proj", tensorName: "\(attnPrefix).out_proj.weight"),
            ]
            if let qkNorm = attrs.qkNorm, qkNorm != .none {
                bindings.append(ParameterBinding(role: "q_layernorm", tensorName: "\(attnPrefix).q_layernorm.weight"))
                bindings.append(ParameterBinding(role: "k_layernorm", tensorName: "\(attnPrefix).k_layernorm.weight"))
            }
            return bindings
        }

        if let _ = attributes as? MLPAttributes {
            let ffPrefix = "\(prefix).feed_forward"
            return [
                ParameterBinding(role: "gate_proj", tensorName: "\(ffPrefix).w1.weight"),
                ParameterBinding(role: "up_proj", tensorName: "\(ffPrefix).w3.weight"),
                ParameterBinding(role: "down_proj", tensorName: "\(ffPrefix).w2.weight"),
            ]
        }

        if let attrs = attributes as? MoEAttributes {
            let ffPrefix = "\(prefix).feed_forward"
            var bindings = [
                ParameterBinding(role: "router", tensorName: "\(ffPrefix).gate.weight"),
                ParameterBinding(role: "expert_bias", tensorName: "\(ffPrefix).expert_bias"),
            ]
            for i in 0..<attrs.expertCount {
                bindings.append(contentsOf: [
                    ParameterBinding(role: "expert_\(i)_gate_proj", tensorName: "\(ffPrefix).experts.\(i).w1.weight"),
                    ParameterBinding(role: "expert_\(i)_up_proj", tensorName: "\(ffPrefix).experts.\(i).w3.weight"),
                    ParameterBinding(role: "expert_\(i)_down_proj", tensorName: "\(ffPrefix).experts.\(i).w2.weight"),
                ])
            }
            return bindings
        }

        if let _ = attributes as? ShortConvAttributes {
            let convPrefix = "\(prefix).conv"
            return [
                ParameterBinding(role: "in_proj", tensorName: "\(convPrefix).in_proj.weight"),
                ParameterBinding(role: "conv_weight", tensorName: "\(convPrefix).conv.weight"),
                ParameterBinding(role: "out_proj", tensorName: "\(convPrefix).out_proj.weight"),
            ]
        }

        return []
    }
}
