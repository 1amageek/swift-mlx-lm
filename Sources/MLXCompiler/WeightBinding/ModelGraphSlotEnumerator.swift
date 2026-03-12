import SwiftLM

/// Entry in the slot manifest: a ParameterSlot paired with its MLX weight path.
public struct SlotManifestEntry: Sendable {

    /// Semantic parameter slot (StructuralPath + role).
    public let slot: ParameterSlot

    /// MLX-convention weight path (e.g., "model.layers.0.self_attn.q_proj.weight").
    public let mlxWeightPath: String

    public init(slot: ParameterSlot, mlxWeightPath: String) {
        self.slot = slot
        self.mlxWeightPath = mlxWeightPath
    }
}

/// Walks a ModelGraph to enumerate all expected weight slots.
///
/// Produces a manifest of `(ParameterSlot, mlxWeightPath)` pairs by walking
/// the graph the same way `MLXInferenceCompiler` does. Each entry maps a
/// semantic slot to the MLX-convention weight path that GGUF name mappers
/// produce.
///
/// Currently supports Llama-family naming (Llama, Qwen2, Mistral, Phi-3,
/// StarCoder2, Mixtral, Qwen3.5).
public struct ModelGraphSlotEnumerator: Sendable {

    public init() {}

    /// Enumerate all parameter slots expected by the model graph.
    public func enumerate(_ graph: ModelGraph) -> [SlotManifestEntry] {
        var entries: [SlotManifestEntry] = []
        var context = WalkContext()
        enumerateRegion(
            graph.rootRegion,
            pathComponents: [],
            namingScope: .root,
            context: &context,
            entries: &entries
        )
        return entries
    }
}

// MARK: - Internal Walk

/// Naming scope for deriving MLX weight paths.
private enum NamingScope {
    /// Root level: "model." prefix for most ops, bare "lm_head" for output head.
    case root

    /// Inside repeating block: "model.layers.{i}." prefix.
    case layer(index: Int)
}

/// Mutable walk context.
private struct WalkContext {
    /// Tracks residual block position within a layer body (0, 1, 2, ...).
    var residualIndexInLayer: Int = 0
}

private extension ModelGraphSlotEnumerator {

    func enumerateRegion(
        _ region: Region,
        pathComponents: [StructuralPathComponent],
        namingScope: NamingScope,
        context: inout WalkContext,
        entries: inout [SlotManifestEntry]
    ) {
        for (i, op) in region.operations.enumerated() {
            let opPath = pathComponents + [.operation(i)]
            let path = StructuralPath(components: opPath)

            switch op.kind {

            // MARK: Token Embedding

            case .tokenEmbedding:
                entries.append(SlotManifestEntry(
                    slot: ParameterSlot(path: path, role: .embeddingTable),
                    mlxWeightPath: "model.embed_tokens.weight"
                ))

            // MARK: Repeating Block

            case .repeating(let count, let body):
                for idx in 0..<count {
                    let iterPath = opPath + [.regionBody, .index(idx)]
                    var layerContext = WalkContext()
                    enumerateRegion(
                        body,
                        pathComponents: iterPath,
                        namingScope: .layer(index: idx),
                        context: &layerContext,
                        entries: &entries
                    )
                }

            case .layerStack(let layers):
                for (idx, layer) in layers.enumerated() {
                    let iterPath = opPath + [.regionBody, .index(idx)]
                    var layerContext = WalkContext()
                    enumerateRegion(
                        layer,
                        pathComponents: iterPath,
                        namingScope: .layer(index: idx),
                        context: &layerContext,
                        entries: &entries
                    )
                }

            // MARK: Residual

            case .residual(_, let body):
                let bodyPath = opPath + [.regionBody]
                let savedIndex = context.residualIndexInLayer
                enumerateRegion(
                    body,
                    pathComponents: bodyPath,
                    namingScope: namingScope,
                    context: &context,
                    entries: &entries
                )
                context.residualIndexInLayer = savedIndex + 1

            // MARK: RMSNorm

            case .rmsNorm:
                let mlxPath: String
                switch namingScope {
                case .root:
                    mlxPath = "model.norm.weight"
                case .layer(let idx):
                    let normName = context.residualIndexInLayer == 0
                        ? "input_layernorm" : "post_attention_layernorm"
                    mlxPath = "model.layers.\(idx).\(normName).weight"
                }
                entries.append(SlotManifestEntry(
                    slot: ParameterSlot(path: path, role: .scale),
                    mlxWeightPath: mlxPath
                ))

            // MARK: LayerNorm

            case .layerNorm(let attrs):
                let mlxPath: String
                switch namingScope {
                case .root:
                    mlxPath = "model.norm.weight"
                case .layer(let idx):
                    let normName = context.residualIndexInLayer == 0
                        ? "input_layernorm" : "post_attention_layernorm"
                    mlxPath = "model.layers.\(idx).\(normName).weight"
                }
                entries.append(SlotManifestEntry(
                    slot: ParameterSlot(path: path, role: .scale),
                    mlxWeightPath: mlxPath
                ))
                if attrs.affine {
                    let biasPath = mlxPath.replacingOccurrences(of: ".weight", with: ".bias")
                    entries.append(SlotManifestEntry(
                        slot: ParameterSlot(path: path, role: .bias),
                        mlxWeightPath: biasPath
                    ))
                }

            // MARK: Attention

            case .attention(let attrs):
                guard case .layer(let idx) = namingScope else { continue }
                let attnPrefix = "model.layers.\(idx).self_attn"

                for field in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                    let fieldPath = path.appending(.field(field))
                    entries.append(SlotManifestEntry(
                        slot: ParameterSlot(path: fieldPath, role: .weight),
                        mlxWeightPath: "\(attnPrefix).\(field).weight"
                    ))
                    if attrs.bias {
                        entries.append(SlotManifestEntry(
                            slot: ParameterSlot(path: fieldPath, role: .bias),
                            mlxWeightPath: "\(attnPrefix).\(field).bias"
                        ))
                    }
                }

                // QK normalization weights
                if attrs.qkNorm != nil {
                    for normField in ["q_norm", "k_norm"] {
                        let normPath = path.appending(.field(normField))
                        entries.append(SlotManifestEntry(
                            slot: ParameterSlot(path: normPath, role: .scale),
                            mlxWeightPath: "\(attnPrefix).\(normField).weight"
                        ))
                    }
                }

            // MARK: MLP

            case .mlp(let attrs):
                guard case .layer(let idx) = namingScope else { continue }
                let mlpPrefix = "model.layers.\(idx).mlp"

                entries.append(SlotManifestEntry(
                    slot: ParameterSlot(
                        path: path.appending(.field("gate_proj")), role: .weight),
                    mlxWeightPath: "\(mlpPrefix).gate_proj.weight"
                ))
                entries.append(SlotManifestEntry(
                    slot: ParameterSlot(
                        path: path.appending(.field("down_proj")), role: .weight),
                    mlxWeightPath: "\(mlpPrefix).down_proj.weight"
                ))

                // upProj exists when gating is enabled
                switch attrs.gating {
                case .none:
                    break
                default:
                    entries.append(SlotManifestEntry(
                        slot: ParameterSlot(
                            path: path.appending(.field("up_proj")), role: .weight),
                        mlxWeightPath: "\(mlpPrefix).up_proj.weight"
                    ))
                }

                // Biases
                if attrs.bias {
                    entries.append(SlotManifestEntry(
                        slot: ParameterSlot(
                            path: path.appending(.field("gate_proj")), role: .bias),
                        mlxWeightPath: "\(mlpPrefix).gate_proj.bias"
                    ))
                    entries.append(SlotManifestEntry(
                        slot: ParameterSlot(
                            path: path.appending(.field("down_proj")), role: .bias),
                        mlxWeightPath: "\(mlpPrefix).down_proj.bias"
                    ))
                    switch attrs.gating {
                    case .none:
                        break
                    default:
                        entries.append(SlotManifestEntry(
                            slot: ParameterSlot(
                                path: path.appending(.field("up_proj")), role: .bias),
                            mlxWeightPath: "\(mlpPrefix).up_proj.bias"
                        ))
                    }
                }

            // MARK: MoE

            case .moe(let attrs):
                guard case .layer(let idx) = namingScope else { continue }
                let moePrefix = "model.layers.\(idx).block_sparse_moe"

                // Router
                entries.append(SlotManifestEntry(
                    slot: ParameterSlot(
                        path: path.appending(.field("router")), role: .weight),
                    mlxWeightPath: "\(moePrefix).gate.weight"
                ))

                // Experts
                for e in 0..<attrs.expertCount {
                    let expertPath = path.appending(.field("experts")).appending(.index(e))
                    let expertPrefix = "\(moePrefix).experts.\(e)"

                    for field in ["gate_proj", "up_proj", "down_proj"] {
                        entries.append(SlotManifestEntry(
                            slot: ParameterSlot(
                                path: expertPath.appending(.field(field)), role: .weight),
                            mlxWeightPath: "\(expertPrefix).\(field).weight"
                        ))
                        if attrs.expertMLP.bias {
                            entries.append(SlotManifestEntry(
                                slot: ParameterSlot(
                                    path: expertPath.appending(.field(field)), role: .bias),
                                mlxWeightPath: "\(expertPrefix).\(field).bias"
                            ))
                        }
                    }
                }

            // MARK: DeltaNet (State Space)

            case .stateSpace:
                guard case .layer(let idx) = namingScope else { continue }
                let dnPrefix = "model.layers.\(idx).linear_attn"

                // Projections (weight + bias)
                for field in ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"] {
                    let fieldPath = path.appending(.field(field))
                    entries.append(SlotManifestEntry(
                        slot: ParameterSlot(path: fieldPath, role: .weight),
                        mlxWeightPath: "\(dnPrefix).\(field).weight"
                    ))
                    entries.append(SlotManifestEntry(
                        slot: ParameterSlot(path: fieldPath, role: .bias),
                        mlxWeightPath: "\(dnPrefix).\(field).bias"
                    ))
                }

                // Raw parameters (non-standard naming)
                entries.append(SlotManifestEntry(
                    slot: ParameterSlot(
                        path: path.appending(.field("conv1d")), role: .weight),
                    mlxWeightPath: "\(dnPrefix).conv1d.weight"
                ))
                entries.append(SlotManifestEntry(
                    slot: ParameterSlot(
                        path: path.appending(.field("norm")), role: .scale),
                    mlxWeightPath: "\(dnPrefix).norm.weight"
                ))
                entries.append(SlotManifestEntry(
                    slot: ParameterSlot(
                        path: path.appending(.field("dt_bias")), role: .bias),
                    mlxWeightPath: "\(dnPrefix).dt_bias"
                ))
                entries.append(SlotManifestEntry(
                    slot: ParameterSlot(
                        path: path.appending(.field("A_log")), role: .weight),
                    mlxWeightPath: "\(dnPrefix).A_log"
                ))

            // MARK: Output Head

            case .outputHead(let attrs):
                if !attrs.tiedToEmbedding {
                    entries.append(SlotManifestEntry(
                        slot: ParameterSlot(path: path, role: .outputProjection),
                        mlxWeightPath: "lm_head.weight"
                    ))
                }

            // MARK: Linear

            case .linear(let attrs):
                entries.append(SlotManifestEntry(
                    slot: ParameterSlot(path: path, role: .weight),
                    mlxWeightPath: "linear.weight"  // Generic; override for specific contexts
                ))
                if attrs.bias {
                    entries.append(SlotManifestEntry(
                        slot: ParameterSlot(path: path, role: .bias),
                        mlxWeightPath: "linear.bias"
                    ))
                }

            // MARK: Positional Embedding

            case .positionalEmbedding:
                entries.append(SlotManifestEntry(
                    slot: ParameterSlot(path: path, role: .embeddingTable),
                    mlxWeightPath: "model.positional_embedding.weight"
                ))

            // MARK: Parallel

            case .parallel(_, let branches):
                for (branchIdx, branch) in branches.enumerated() {
                    let branchPath = opPath + [.regionBranch(branchIdx)]
                    enumerateRegion(
                        branch,
                        pathComponents: branchPath,
                        namingScope: namingScope,
                        context: &context,
                        entries: &entries
                    )
                }

            // MARK: Passthrough

            case .rope, .custom:
                // No weight parameters
                break
            }
        }
    }
}
