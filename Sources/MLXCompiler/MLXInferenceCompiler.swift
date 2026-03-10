@preconcurrency import MLX
import SwiftLM

/// Compiles a SwiftLM `ModelGraph` + `BoundWeights` into a lowered inference model.
///
/// This is the third compilation path:
/// - `MLXCompiler` → tree-walking interpreter (verification baseline)
/// - `MLXModuleCompiler` → MLXNN Module tree (LoRA/quantize compat)
/// - `MLXInferenceCompiler` → lowered value types with compile-time kernel selection
///
/// Key differences from other compilers:
/// - Preserves quantization metadata from `MLXTensorStorage` through compilation
/// - Selects `matmul` vs `quantizedMatmul` at compile time (not post-hoc)
/// - Fully unrolls repeating blocks — each layer gets distinct ops and cache indices
/// - Externalizes cache state as value types for functional-style inference
/// - Produces separate `PrefillPlan` and `DecodePlan`
public struct MLXInferenceCompiler: Sendable {

    public init() {}

    /// Compile a model graph with bound weights into a lowered inference model.
    ///
    /// Accepts either `MLXTensorStorage` or `MLXArray` in `BoundWeights.tensors`.
    /// When `MLXTensorStorage.affineQuantized` is present, the corresponding
    /// `LoweredProjection` will use `quantizedMatmul` instead of `matmul`.
    public func compile(
        graph: ModelGraph, weights: BoundWeights
    ) throws -> MLXLoweredInferenceModel {
        print("[MLXInferenceCompiler] architecture:\n\(ModelGraphDumper.dump(graph))")

        let store = try InferenceWeightStore(boundWeights: weights)

        // Phase 1: Scan — discover caches, embeddings, tied output heads
        var cacheDescriptors: [CacheDescriptor] = []
        var embeddingPaths: [StructuralPath] = []
        var hasTiedOutputHead = false
        var cacheIndex = 0

        scanRegion(
            graph.rootRegion,
            pathComponents: [],
            cacheDescriptors: &cacheDescriptors,
            cacheIndex: &cacheIndex,
            embeddingPaths: &embeddingPaths,
            hasTiedOutputHead: &hasTiedOutputHead
        )

        // Validate embedding discovery
        if embeddingPaths.count > 1 && hasTiedOutputHead {
            throw CompilerError.invalidGraphStructure(
                "Found \(embeddingPaths.count) token embeddings but model has a tied output head. "
                + "Cannot determine which embedding to tie.")
        }
        if hasTiedOutputHead && embeddingPaths.isEmpty {
            throw CompilerError.invalidGraphStructure(
                "Model has a tied output head but no token embedding was found.")
        }

        let embeddingPath = embeddingPaths.first

        // Load embedding table for tied output head resolution
        var embeddingTable: MLXArray?
        if let embPath = embeddingPath {
            let slot = ParameterSlot(path: embPath, role: .embeddingTable)
            embeddingTable = store.getDense(slot)
        }

        // Build cache slot index for path-based lookup during lowering
        var cacheSlotByPath: [StructuralPath: Int] = [:]
        for desc in cacheDescriptors {
            cacheSlotByPath[desc.path] = desc.slotIndex
        }

        // Phase 2: Lower — recursive lowering of the graph
        var context = LoweringContext(
            store: store,
            cacheSlotByPath: cacheSlotByPath,
            embeddingTable: embeddingTable
        )

        let steps = try lowerRegion(
            graph.rootRegion,
            pathComponents: [],
            context: &context
        )

        // Phase 3: Assemble — build plans
        // Phase 1: prefill and decode use the same steps
        let prefillPlan = PrefillPlan(steps: steps)
        let decodePlan = DecodePlan(steps: steps)

        let metadata = InferenceMetadata(
            cacheSlotCount: cacheDescriptors.count,
            cacheDescriptors: cacheDescriptors,
            hasTiedOutputHead: hasTiedOutputHead
        )

        return MLXLoweredInferenceModel(
            prefill: prefillPlan,
            decode: decodePlan,
            metadata: metadata
        )
    }

    // MARK: - Graph Scanning

    /// Depth-first scan — replicates `MLXCompiler.scanRegion()` logic.
    private func scanRegion(
        _ region: Region,
        pathComponents: [StructuralPathComponent],
        cacheDescriptors: inout [CacheDescriptor],
        cacheIndex: inout Int,
        embeddingPaths: inout [StructuralPath],
        hasTiedOutputHead: inout Bool
    ) {
        for (opIndex, op) in region.operations.enumerated() {
            let opPath = pathComponents + [.operation(opIndex)]

            switch op.kind {
            case .tokenEmbedding:
                embeddingPaths.append(StructuralPath(components: opPath))

            case .outputHead(let attrs):
                if attrs.tiedToEmbedding {
                    hasTiedOutputHead = true
                }

            case .attention:
                let path = StructuralPath(components: opPath)
                cacheDescriptors.append(
                    CacheDescriptor(path: path, kind: .kv, slotIndex: cacheIndex))
                cacheIndex += 1

            case .stateSpace:
                let path = StructuralPath(components: opPath)
                cacheDescriptors.append(
                    CacheDescriptor(path: path, kind: .recurrent, slotIndex: cacheIndex))
                cacheIndex += 1

            case .residual(_, let body):
                scanRegion(
                    body,
                    pathComponents: opPath + [.regionBody],
                    cacheDescriptors: &cacheDescriptors,
                    cacheIndex: &cacheIndex,
                    embeddingPaths: &embeddingPaths,
                    hasTiedOutputHead: &hasTiedOutputHead
                )

            case .parallel(_, let branches):
                for (i, branch) in branches.enumerated() {
                    scanRegion(
                        branch,
                        pathComponents: opPath + [.regionBranch(i)],
                        cacheDescriptors: &cacheDescriptors,
                        cacheIndex: &cacheIndex,
                        embeddingPaths: &embeddingPaths,
                        hasTiedOutputHead: &hasTiedOutputHead
                    )
                }

            case .repeating(let count, let body):
                for i in 0..<count {
                    scanRegion(
                        body,
                        pathComponents: opPath + [.regionBody, .index(i)],
                        cacheDescriptors: &cacheDescriptors,
                        cacheIndex: &cacheIndex,
                        embeddingPaths: &embeddingPaths,
                        hasTiedOutputHead: &hasTiedOutputHead
                    )
                }

            case .layerStack(let layers):
                for (i, layer) in layers.enumerated() {
                    scanRegion(
                        layer,
                        pathComponents: opPath + [.regionBody, .index(i)],
                        cacheDescriptors: &cacheDescriptors,
                        cacheIndex: &cacheIndex,
                        embeddingPaths: &embeddingPaths,
                        hasTiedOutputHead: &hasTiedOutputHead
                    )
                }

            default:
                break
            }
        }
    }

    // MARK: - Lowering

    /// Mutable context passed through the lowering recursion.
    private struct LoweringContext {
        let store: InferenceWeightStore
        let cacheSlotByPath: [StructuralPath: Int]
        let embeddingTable: MLXArray?
    }

    /// Lower a region into a sequence of `LoweredStep`s.
    ///
    /// `.repeating` is handled here (not in `lowerOperation`) to enable
    /// inline unrolling — each iteration's steps are appended directly
    /// to the parent's step list.
    private func lowerRegion(
        _ region: Region,
        pathComponents: [StructuralPathComponent],
        context: inout LoweringContext
    ) throws -> [LoweredStep] {
        var steps: [LoweredStep] = []

        for (i, op) in region.operations.enumerated() {
            let opPath = pathComponents + [.operation(i)]

            // Handle .repeating inline for proper unrolling
            if case .repeating(let count, let body) = op.kind {
                for iter in 0..<count {
                    let iterPath = opPath + [.regionBody, .index(iter)]
                    let iterSteps = try lowerRegion(
                        body, pathComponents: iterPath, context: &context)
                    steps.append(contentsOf: iterSteps)
                }
                continue
            }

            // Handle .layerStack inline for per-layer unrolling
            if case .layerStack(let layers) = op.kind {
                for (iter, layer) in layers.enumerated() {
                    let iterPath = opPath + [.regionBody, .index(iter)]
                    let iterSteps = try lowerRegion(
                        layer, pathComponents: iterPath, context: &context)
                    steps.append(contentsOf: iterSteps)
                }
                continue
            }

            let path = StructuralPath(components: opPath)
            let lowered = try lowerOperation(
                op, pathComponents: opPath, path: path, context: &context
            )
            steps.append(lowered)
        }

        return steps
    }

    /// Lower a single operation into a `LoweredStep`.
    private func lowerOperation(
        _ op: Operation,
        pathComponents: [StructuralPathComponent],
        path: StructuralPath,
        context: inout LoweringContext
    ) throws -> LoweredStep {
        switch op.kind {

        // MARK: Primitive Operations

        case .tokenEmbedding:
            let table = try context.store.requireDense(
                ParameterSlot(path: path, role: .embeddingTable))
            return .op(.tokenEmbedding(LoweredEmbedding(table: table)))

        case .attention(let attrs):
            let attn = try lowerAttention(attrs, path: path, context: &context)
            return .op(.attention(attn))

        case .mlp(let attrs):
            let mlp = try lowerMLP(attrs, path: path, context: &context)
            return .op(.mlp(mlp))

        case .moe(let attrs):
            let moe = try lowerMoE(attrs, path: path, context: &context)
            return .op(.moe(moe))

        case .rmsNorm(let attrs):
            let weight = try context.store.requireDense(
                ParameterSlot(path: path, role: .scale))
            return .op(.norm(.rms(weight: weight, epsilon: attrs.epsilon)))

        case .layerNorm(let attrs):
            let weight = try context.store.requireDense(
                ParameterSlot(path: path, role: .scale))
            let bias = attrs.affine
                ? context.store.getDense(ParameterSlot(path: path, role: .bias))
                : nil
            return .op(.norm(.layer(weight: weight, bias: bias, epsilon: attrs.epsilon)))

        case .linear(let attrs):
            let proj = try makeProjection(
                store: context.store, path: path, field: nil, hasBias: attrs.bias)
            return .op(.linear(proj))

        case .outputHead(let attrs):
            let head = try lowerOutputHead(attrs, path: path, context: &context)
            return .op(.outputHead(head))

        case .stateSpace(let attrs):
            let dn = try lowerDeltaNet(attrs, path: path, context: &context)
            return .op(.deltaNet(dn))

        case .rope(let attrs):
            return .op(.rope(LoweredRoPE(attrs: attrs)))

        case .positionalEmbedding:
            let table = try context.store.requireDense(
                ParameterSlot(path: path, role: .embeddingTable))
            return .op(.positionalEmbedding(LoweredPositionalEmbedding(table: table)))

        // MARK: Structural Operations

        case .residual(_, let body):
            let bodyPath = pathComponents + [.regionBody]
            let bodySteps = try lowerRegion(
                body, pathComponents: bodyPath, context: &context)
            return .residual(body: bodySteps)

        case .repeating:
            // Handled in lowerRegion() for proper inline unrolling.
            // This case should never be reached.
            throw CompilerError.invalidGraphStructure(
                "Repeating should be handled in lowerRegion, not lowerOperation")

        case .layerStack:
            // Handled in lowerRegion() for per-layer inline unrolling.
            // This case should never be reached.
            throw CompilerError.invalidGraphStructure(
                "LayerStack should be handled in lowerRegion, not lowerOperation")

        case .parallel(let merge, let branches):
            var loweredBranches: [[LoweredStep]] = []
            loweredBranches.reserveCapacity(branches.count)
            for (i, branch) in branches.enumerated() {
                let branchPath = pathComponents + [.regionBranch(i)]
                let branchSteps = try lowerRegion(
                    branch, pathComponents: branchPath, context: &context)
                loweredBranches.append(branchSteps)
            }
            return .parallel(merge: merge, branches: loweredBranches)

        // MARK: Unsupported

        case .visionEncoder:
            throw CompilerError.unsupportedOperation(
                "visionEncoder is not yet implemented in MLXInferenceCompiler")

        case .custom(let attrs):
            throw CompilerError.unsupportedOperation("custom(\(attrs.domain).\(attrs.name))")
        }
    }

    // MARK: - Operation Lowering Helpers

    /// Lower an attention operation.
    private func lowerAttention(
        _ attrs: AttentionAttributes,
        path: StructuralPath,
        context: inout LoweringContext
    ) throws -> LoweredAttention {
        let qProj = try makeProjection(
            store: context.store, path: path, field: "q_proj", hasBias: attrs.bias)
        let kProj = try makeProjection(
            store: context.store, path: path, field: "k_proj", hasBias: attrs.bias)
        let vProj = try makeProjection(
            store: context.store, path: path, field: "v_proj", hasBias: attrs.bias)
        let oProj = try makeProjection(
            store: context.store, path: path, field: "o_proj", hasBias: attrs.bias)

        // QK normalization weights (dense, not quantized)
        let qNormWeight = context.store.getDense(
            ParameterSlot(path: path.appending(.field("q_norm")), role: .scale))
        let kNormWeight = context.store.getDense(
            ParameterSlot(path: path.appending(.field("k_norm")), role: .scale))
        let qNormBias = context.store.getDense(
            ParameterSlot(path: path.appending(.field("q_norm")), role: .bias))
        let kNormBias = context.store.getDense(
            ParameterSlot(path: path.appending(.field("k_norm")), role: .bias))

        guard let slotIndex = context.cacheSlotByPath[path] else {
            throw CompilerError.invalidGraphStructure(
                "No cache slot found for attention at \(path.components)")
        }

        return LoweredAttention(
            qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
            attrs: attrs,
            qNormWeight: qNormWeight, kNormWeight: kNormWeight,
            qNormBias: qNormBias, kNormBias: kNormBias,
            cacheSlotIndex: slotIndex
        )
    }

    /// Lower an MLP operation.
    private func lowerMLP(
        _ attrs: MLPAttributes,
        path: StructuralPath,
        context: inout LoweringContext
    ) throws -> LoweredMLP {
        let gateProj = try makeProjection(
            store: context.store, path: path, field: "gate_proj", hasBias: attrs.bias)
        let downProj = try makeProjection(
            store: context.store, path: path, field: "down_proj", hasBias: attrs.bias)

        // Only create upProj when gating is enabled
        let upProj: LoweredProjection?
        switch attrs.gating {
        case .none:
            upProj = nil
        default:
            upProj = try makeProjection(
                store: context.store, path: path, field: "up_proj", hasBias: attrs.bias)
        }

        return LoweredMLP(
            gateProj: gateProj, downProj: downProj,
            upProj: upProj, activation: attrs.activation
        )
    }

    /// Lower a MoE operation.
    private func lowerMoE(
        _ attrs: MoEAttributes,
        path: StructuralPath,
        context: inout LoweringContext
    ) throws -> LoweredMoE {
        let router = try makeProjection(
            store: context.store, path: path, field: "router", hasBias: false)

        var experts: [LoweredExpertMLP] = []
        experts.reserveCapacity(attrs.expertCount)
        for i in 0..<attrs.expertCount {
            let expertPath = path.appending(.field("experts")).appending(.index(i))
            let gate = try makeProjection(
                store: context.store, path: expertPath, field: "gate_proj",
                hasBias: attrs.expertMLP.bias)
            let up = try makeProjection(
                store: context.store, path: expertPath, field: "up_proj",
                hasBias: attrs.expertMLP.bias)
            let down = try makeProjection(
                store: context.store, path: expertPath, field: "down_proj",
                hasBias: attrs.expertMLP.bias)
            experts.append(LoweredExpertMLP(
                gateProj: gate, upProj: up, downProj: down,
                activation: attrs.expertMLP.activation
            ))
        }

        return LoweredMoE(
            router: router, experts: experts,
            expertsPerToken: attrs.expertsPerToken
        )
    }

    /// Lower a DeltaNet state-space operation.
    private func lowerDeltaNet(
        _ attrs: StateSpaceAttributes,
        path: StructuralPath,
        context: inout LoweringContext
    ) throws -> LoweredDeltaNet {
        // Validate variant
        let variant = attrs.variant.lowercased()
        guard variant == "deltanet" || variant == "gated_deltanet"
            || variant == "gated-deltanet"
        else {
            throw CompilerError.unsupportedVariant(attrs.variant)
        }

        let inProjQKV = try makeProjection(
            store: context.store, path: path, field: "in_proj_qkv", hasBias: true)
        let inProjZ = try makeProjection(
            store: context.store, path: path, field: "in_proj_z", hasBias: true)
        let inProjB = try makeProjection(
            store: context.store, path: path, field: "in_proj_b", hasBias: true)
        let inProjA = try makeProjection(
            store: context.store, path: path, field: "in_proj_a", hasBias: true)
        let outProj = try makeProjection(
            store: context.store, path: path, field: "out_proj", hasBias: true)

        // Raw (non-quantizable) parameters
        let convWeight = try context.store.requireDense(
            ParameterSlot(path: path.appending(.field("conv1d")), role: .weight))
        let normWeight = try context.store.requireDense(
            ParameterSlot(path: path.appending(.field("norm")), role: .scale))
        let dtBias = try context.store.requireDense(
            ParameterSlot(path: path.appending(.field("dt_bias")), role: .bias))
        let aLog = try context.store.requireDense(
            ParameterSlot(path: path.appending(.field("A_log")), role: .weight))

        guard let slotIndex = context.cacheSlotByPath[path] else {
            throw CompilerError.invalidGraphStructure(
                "No cache slot found for DeltaNet at \(path.components)")
        }

        return LoweredDeltaNet(
            inProjQKV: inProjQKV, inProjZ: inProjZ,
            inProjB: inProjB, inProjA: inProjA, outProj: outProj,
            convWeight: convWeight, normWeight: normWeight,
            dtBias: dtBias, aLog: aLog,
            attrs: attrs, cacheSlotIndex: slotIndex
        )
    }

    /// Lower an output head operation.
    private func lowerOutputHead(
        _ attrs: OutputHeadAttributes,
        path: StructuralPath,
        context: inout LoweringContext
    ) throws -> LoweredOutputHead {
        if attrs.tiedToEmbedding {
            guard let table = context.embeddingTable else {
                throw CompilerError.invalidGraphStructure(
                    "Tied output head found but no embedding table was discovered.")
            }
            return LoweredOutputHead(
                projection: LoweredProjection(weight: table, bias: nil),
                isTied: true
            )
        }

        let proj = try makeProjection(
            store: context.store, path: path, field: nil,
            hasBias: attrs.bias, role: .outputProjection)
        return LoweredOutputHead(projection: proj, isTied: false)
    }

    // MARK: - Projection Helper

    /// Create a `LoweredProjection` from the weight store.
    ///
    /// Compile-time kernel selection happens here: if the stored weight is
    /// `MLXTensorStorage.affineQuantized`, the projection will use `quantizedMatmul`.
    ///
    /// - Parameter role: The `ParameterRole` for the weight lookup. Defaults to `.weight`.
    ///   Use `.outputProjection` for untied output heads.
    private func makeProjection(
        store: InferenceWeightStore,
        path: StructuralPath,
        field: String?,
        hasBias: Bool,
        role: ParameterRole = .weight
    ) throws -> LoweredProjection {
        let weightPath = field.map { path.appending(.field($0)) } ?? path
        let storage = try store.require(
            ParameterSlot(path: weightPath, role: role))
        let bias = hasBias
            ? store.getDense(ParameterSlot(path: weightPath, role: .bias))
            : nil
        return LoweredProjection(storage: storage, bias: bias)
    }
}
