@preconcurrency import MLX
import MLXFast
import MLXNN
import SwiftLM

// MARK: - State Space Variant

/// Resolved state-space model variant for type-safe dispatch.
///
/// Converts the stringly-typed `StateSpaceAttributes.variant` into a concrete
/// enum at the executor boundary, eliminating string matching from the hot path.
enum MLXStateSpaceVariant {
    case deltaNet
    case gatedDeltaNet

    init(variant: String) throws {
        switch variant.lowercased() {
        case "deltanet", "gated_deltanet", "gated-deltanet":
            self = .deltaNet
        default:
            throw CompilerError.unsupportedVariant(variant)
        }
    }
}

// MARK: - MLXExecutor

/// Tree-walking interpreter that executes a compiled model graph on MLX.
///
/// The executor walks the `ModelGraph` in DFS order, dispatching each semantic
/// operation (attention, mlp, norm, etc.) to the appropriate MLX implementation.
/// This preserves MLX fused kernel optimizations that would be lost by lowering
/// to operator-level IR.
///
/// Cache slots are resolved by `StructuralPath` lookup (not execution-order
/// counters), making cache addressing robust against execution path changes.
///
/// ```swift
/// let executor = MLXExecutor(compiledModel: compiled)
/// let logits = try executor.forward(tokenIDs: tokens)
/// // KV caches persist across calls for autoregressive generation:
/// let nextLogits = try executor.forward(tokenIDs: nextToken)
/// ```
public final class MLXExecutor {

    private let model: MLXCompiledModel
    private var caches: [AnyObject]

    // Runtime state (reset per forward call)
    private var tokenIDs: MLXArray!
    private var pathComponents: [StructuralPathComponent] = []

    public init(compiledModel: MLXCompiledModel) {
        self.model = compiledModel
        self.caches = compiledModel.cacheDescriptors.map { desc in
            switch desc.kind {
            case .kv:
                return MLXKVCacheSimple() as AnyObject
            case .rotating:
                return MLXKVCacheSimple() as AnyObject
            case .quantized:
                return MLXKVCacheSimple() as AnyObject
            case .recurrent:
                return MLXRecurrentCache() as AnyObject
            }
        }
    }

    // MARK: - Forward API

    /// Execute a forward pass, returning logits.
    ///
    /// KV caches persist across calls, enabling autoregressive generation.
    /// Call `resetCaches()` to clear state between independent sequences.
    ///
    /// Accepts 1D `[L]` or 2D `[B, L]` token IDs. Internally adds a batch
    /// dimension if needed and squeezes it from the output for transparent usage.
    public func forward(tokenIDs: MLXArray) throws -> MLXArray {
        let needsSqueeze = tokenIDs.ndim == 1
        self.tokenIDs = needsSqueeze ? tokenIDs.expandedDimensions(axis: 0) : tokenIDs
        self.pathComponents = []

        let results = try executeRegion(model.graph.rootRegion, inputValues: [])
        guard let logits = results.first else {
            throw CompilerError.executionError("Root region produced no results")
        }
        // Only squeeze batch dim if we added it AND the output still has size 1 at axis 0.
        // Operations like parallel(merge: .stack) may reshape axis 0 to numBranches.
        if needsSqueeze && logits.dim(0) == 1 {
            return logits.squeezed(axis: 0)
        }
        return logits
    }

    /// Reset all caches (call between independent sequences).
    public func resetCaches() {
        self.caches = model.cacheDescriptors.map { desc in
            switch desc.kind {
            case .kv, .rotating, .quantized:
                return MLXKVCacheSimple() as AnyObject
            case .recurrent:
                return MLXRecurrentCache() as AnyObject
            }
        }
    }

    // MARK: - Cache Lookup

    /// Resolve the cache for the current operation by structural path.
    ///
    /// This replaces the fragile `cacheSlotCounter` pattern. The cache slot
    /// is determined at compile time and looked up by path, so cache
    /// addressing is independent of execution ordering.
    private func cacheSlot() throws -> Int {
        let path = StructuralPath(components: pathComponents)
        guard let slot = model.cacheSlotByPath[path] else {
            throw CompilerError.invalidGraphStructure(
                "No cache descriptor for path: \(pathComponents)")
        }
        return slot
    }

    // MARK: - Region Execution

    /// Execute a region by running operations in sequence, tracking SSA values.
    private func executeRegion(
        _ region: Region, inputValues: [MLXArray]
    ) throws -> [MLXArray] {
        var valueMap: [Int: MLXArray] = [:]

        // Bind region parameters to input values
        for (param, value) in zip(region.parameters, inputValues) {
            valueMap[param.id.rawValue] = value
        }

        // Execute operations in sequence
        for (opIndex, op) in region.operations.enumerated() {
            pathComponents.append(.operation(opIndex))

            let inputs = op.operands.map { valueMap[$0.value.rawValue]! }
            let outputs = try executeOperation(op, inputs: inputs)

            for (result, output) in zip(op.results, outputs) {
                valueMap[result.id.rawValue] = output
            }

            pathComponents.removeLast()
        }

        // Collect region results
        return region.results.map { valueMap[$0.value.rawValue]! }
    }

    // MARK: - Operation Dispatch

    private func executeOperation(
        _ op: Operation, inputs: [MLXArray]
    ) throws -> [MLXArray] {
        switch op.kind {
        case .tokenEmbedding(let attrs):
            return try [executeTokenEmbedding(attrs)]

        case .attention(let attrs):
            let input = try requireUnaryInput(inputs, op: "attention")
            return try [executeAttention(attrs, input: input)]

        case .mlp(let attrs):
            let input = try requireUnaryInput(inputs, op: "mlp")
            return try [executeMLP(attrs, input: input)]

        case .moe(let attrs):
            let input = try requireUnaryInput(inputs, op: "moe")
            return try [executeMoE(attrs, input: input)]

        case .rmsNorm(let attrs):
            let input = try requireUnaryInput(inputs, op: "rmsNorm")
            return try [executeRMSNorm(attrs, input: input)]

        case .layerNorm(let attrs):
            let input = try requireUnaryInput(inputs, op: "layerNorm")
            return try [executeLayerNorm(attrs, input: input)]

        case .linear(let attrs):
            let input = try requireUnaryInput(inputs, op: "linear")
            return try [executeLinear(attrs, input: input)]

        case .outputHead(let attrs):
            let input = try requireUnaryInput(inputs, op: "outputHead")
            return try [executeOutputHead(attrs, input: input)]

        case .stateSpace(let attrs):
            let input = try requireUnaryInput(inputs, op: "stateSpace")
            return try [executeStateSpace(attrs, input: input)]

        case .rope(let attrs):
            let input = try requireUnaryInput(inputs, op: "rope")
            return [executeRoPE(attrs, input: input)]

        case .positionalEmbedding(let attrs):
            let input = try requireUnaryInput(inputs, op: "positionalEmbedding")
            return try [executePositionalEmbedding(attrs, input: input)]

        case .residual(let strategy, let body):
            return try executeResidual(strategy: strategy, body: body, inputs: inputs)

        case .parallel(let merge, let branches):
            return try executeParallel(merge: merge, branches: branches, inputs: inputs)

        case .repeating(let count, let body):
            return try executeRepeating(count: count, body: body, inputs: inputs)

        case .layerStack(let layers):
            return try executeLayerStack(layers: layers, inputs: inputs)

        case .visionEncoder:
            throw CompilerError.unsupportedOperation(
                "visionEncoder is not yet implemented in MLXExecutor")

        case .custom(let attrs):
            throw CompilerError.unsupportedOperation("custom(\(attrs.domain).\(attrs.name))")
        }
    }

    /// Validate that a primitive operation received exactly one input.
    private func requireUnaryInput(_ inputs: [MLXArray], op: String) throws -> MLXArray {
        guard inputs.count == 1 else {
            throw CompilerError.invalidGraphStructure(
                "\(op) expects 1 input, got \(inputs.count)")
        }
        return inputs[0]
    }

    // MARK: - Structural Operations

    private func executeResidual(
        strategy: ResidualStrategy, body: Region, inputs: [MLXArray]
    ) throws -> [MLXArray] {
        pathComponents.append(.regionBody)
        let bodyResults = try executeRegion(body, inputValues: inputs)
        pathComponents.removeLast()

        switch strategy {
        case .add:
            return zip(inputs, bodyResults).map { $0 + $1 }
        case .weighted, .gated, .custom:
            return zip(inputs, bodyResults).map { $0 + $1 }
        }
    }

    private func executeParallel(
        merge: ParallelMergeStrategy, branches: [Region], inputs: [MLXArray]
    ) throws -> [MLXArray] {
        guard !branches.isEmpty else {
            throw CompilerError.invalidGraphStructure("parallel has no branches")
        }

        var branchResults: [[MLXArray]] = []

        for (i, branch) in branches.enumerated() {
            pathComponents.append(.regionBranch(i))
            let results = try executeRegion(branch, inputValues: inputs)
            branchResults.append(results)
            pathComponents.removeLast()
        }

        let resultCount = branchResults[0].count
        switch merge {
        case .add:
            var merged = branchResults[0]
            for branch in branchResults.dropFirst() {
                for j in 0..<resultCount {
                    merged[j] = merged[j] + branch[j]
                }
            }
            return merged
        case .concat:
            return (0..<resultCount).map { j in
                concatenated(branchResults.map { $0[j] }, axis: -1)
            }
        case .stack:
            return (0..<resultCount).map { j in
                stacked(branchResults.map { $0[j] })
            }
        case .visionMerge:
            throw CompilerError.unsupportedOperation(
                "visionMerge is not yet implemented in MLXExecutor")
        case .custom:
            throw CompilerError.unsupportedOperation("custom parallel merge")
        }
    }

    private func executeRepeating(
        count: Int, body: Region, inputs: [MLXArray]
    ) throws -> [MLXArray] {
        pathComponents.append(.regionBody)
        var current = inputs
        for i in 0..<count {
            pathComponents.append(.index(i))
            current = try executeRegion(body, inputValues: current)
            pathComponents.removeLast()
        }
        pathComponents.removeLast()
        return current
    }

    private func executeLayerStack(
        layers: [Region], inputs: [MLXArray]
    ) throws -> [MLXArray] {
        pathComponents.append(.regionBody)
        var current = inputs
        for (i, layer) in layers.enumerated() {
            pathComponents.append(.index(i))
            current = try executeRegion(layer, inputValues: current)
            pathComponents.removeLast()
        }
        pathComponents.removeLast()
        return current
    }

    // MARK: - Token Embedding

    private func executeTokenEmbedding(
        _ attrs: TokenEmbeddingAttributes
    ) throws -> MLXArray {
        let table = try weight(role: .embeddingTable)
        return table[tokenIDs]
    }

    // MARK: - Normalization

    private func executeRMSNorm(
        _ attrs: RMSNormAttributes, input: MLXArray
    ) throws -> MLXArray {
        let w = try weight(role: .scale)
        return MLXFast.rmsNorm(input, weight: 1 + w, eps: attrs.epsilon)
    }

    private func executeLayerNorm(
        _ attrs: LayerNormAttributes, input: MLXArray
    ) throws -> MLXArray {
        let w = try weight(role: .scale)
        let mean = input.mean(axis: -1, keepDims: true)
        let variance = input.variance(axis: -1, keepDims: true)
        var normalized = (input - mean) / (variance + MLXArray(attrs.epsilon)).sqrt()
        normalized = normalized * w
        if attrs.affine {
            if let b = optionalWeight(role: .bias) {
                normalized = normalized + b
            }
        }
        return normalized
    }

    // MARK: - Linear Projection

    private func executeLinear(
        _ attrs: LinearAttributes, input: MLXArray
    ) throws -> MLXArray {
        let w = try weight(role: .weight)
        var result = MLX.matmul(input, w.T)
        if attrs.bias {
            let b = try weight(role: .bias)
            result = result + b
        }
        return result
    }

    // MARK: - Output Head

    private func executeOutputHead(
        _ attrs: OutputHeadAttributes, input: MLXArray
    ) throws -> MLXArray {
        if attrs.tiedToEmbedding, let embPath = model.embeddingPath {
            let embWeight = try model.weightStore.require(
                ParameterSlot(path: embPath, role: .embeddingTable)
            )
            return MLX.matmul(input, embWeight.T)
        }

        let w = try weight(role: .outputProjection)
        var result = MLX.matmul(input, w.T)
        if attrs.bias {
            let b = try weight(role: .bias)
            result = result + b
        }
        return result
    }

    // MARK: - RoPE

    /// Execute a standalone RoPE operation.
    ///
    /// NOTE: Standalone RoPE always uses offset=0 because it has no associated KV cache
    /// to derive the position from. In practice, all models embed RoPE within the
    /// attention operation (via AttentionAttributes.rope), which correctly uses the
    /// cache offset. This standalone path exists for completeness but should not be
    /// used for autoregressive generation.
    private func executeRoPE(
        _ attrs: RoPEAttributes, input: MLXArray
    ) -> MLXArray {
        let scale: Float
        switch attrs.scaling?.kind {
        case .linear:
            scale = 1.0 / attrs.scaling!.factor
        default:
            scale = 1.0
        }
        return MLXFast.RoPE(
            input, dimensions: attrs.dimension, traditional: false,
            base: attrs.base, scale: scale, offset: 0
        )
    }

    // MARK: - Positional Embedding

    private func executePositionalEmbedding(
        _ attrs: PositionalEmbeddingAttributes, input: MLXArray
    ) throws -> MLXArray {
        let table = try weight(role: .embeddingTable)
        let seqLen = input.dim(1)
        let positions = MLXArray(0..<seqLen)
        return input + table[positions]
    }

    // MARK: - Attention

    private func executeAttention(
        _ attrs: AttentionAttributes, input: MLXArray
    ) throws -> MLXArray {
        let (B, L) = (input.dim(0), input.dim(1))
        let headDim = attrs.headDimension
        let scale = 1.0 / Float(headDim).squareRoot()

        // Project Q, K, V
        let qWeight = try weight(field: "q_proj", role: .weight)
        let kWeight = try weight(field: "k_proj", role: .weight)
        let vWeight = try weight(field: "v_proj", role: .weight)

        var queries = MLX.matmul(input, qWeight.T)
        var keys = MLX.matmul(input, kWeight.T)
        var values = MLX.matmul(input, vWeight.T)

        if attrs.bias {
            if let qb = optionalWeight(field: "q_proj", role: .bias) { queries = queries + qb }
            if let kb = optionalWeight(field: "k_proj", role: .bias) { keys = keys + kb }
            if let vb = optionalWeight(field: "v_proj", role: .bias) { values = values + vb }
        }

        // Reshape to head layout [B, H, L, D]
        queries = queries.reshaped(B, L, attrs.headCount, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, attrs.kvHeadCount, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, attrs.kvHeadCount, -1).transposed(0, 2, 1, 3)

        // QK normalization
        if let qkNorm = attrs.qkNorm {
            switch qkNorm {
            case .rmsNorm:
                if let qnw = optionalWeight(field: "q_norm", role: .scale) {
                    queries = MLXFast.rmsNorm(queries, weight: 1 + qnw, eps: 1e-6)
                }
                if let knw = optionalWeight(field: "k_norm", role: .scale) {
                    keys = MLXFast.rmsNorm(keys, weight: 1 + knw, eps: 1e-6)
                }
            case .layerNorm:
                if let qnw = optionalWeight(field: "q_norm", role: .scale) {
                    let qnb = optionalWeight(field: "q_norm", role: .bias)
                    queries = layerNormOp(queries, weight: qnw, bias: qnb)
                }
                if let knw = optionalWeight(field: "k_norm", role: .scale) {
                    let knb = optionalWeight(field: "k_norm", role: .bias)
                    keys = layerNormOp(keys, weight: knw, bias: knb)
                }
            case .none, .custom:
                break
            }
        }

        // Resolve cache by path
        let slot = try cacheSlot()

        // RoPE
        let kvCache = caches[slot] as? MLXKVCache
        if let ropeAttrs = attrs.rope {
            let offset = kvCache?.offset ?? 0
            let ropeScale: Float = {
                if let scaling = ropeAttrs.scaling, scaling.kind == .linear {
                    return 1.0 / scaling.factor
                }
                return 1.0
            }()

            let ropeDim = ropeAttrs.dimension
            if ropeDim < headDim {
                // Partial RoPE (e.g. Qwen 3.5: 64 of 256)
                let qRot = MLXFast.RoPE(
                    queries[0..., 0..., 0..., 0..<ropeDim],
                    dimensions: ropeDim, traditional: false,
                    base: ropeAttrs.base, scale: ropeScale, offset: offset
                )
                queries = concatenated(
                    [qRot, queries[0..., 0..., 0..., ropeDim...]], axis: -1)

                let kRot = MLXFast.RoPE(
                    keys[0..., 0..., 0..., 0..<ropeDim],
                    dimensions: ropeDim, traditional: false,
                    base: ropeAttrs.base, scale: ropeScale, offset: offset
                )
                keys = concatenated(
                    [kRot, keys[0..., 0..., 0..., ropeDim...]], axis: -1)
            } else {
                queries = MLXFast.RoPE(
                    queries, dimensions: ropeDim, traditional: false,
                    base: ropeAttrs.base, scale: ropeScale, offset: offset
                )
                keys = MLXFast.RoPE(
                    keys, dimensions: ropeDim, traditional: false,
                    base: ropeAttrs.base, scale: ropeScale, offset: offset
                )
            }
        }

        // KV cache update
        let cache = caches[slot] as! MLXKVCache
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        let mask = cache.makeMask(queryLength: L)

        // Scaled dot-product attention
        let attnOutput = MLXFast.scaledDotProductAttention(
            queries: queries, keys: cachedKeys, values: cachedValues,
            scale: scale, mask: mask
        )

        // Output projection
        let oWeight = try weight(field: "o_proj", role: .weight)
        var output = MLX.matmul(
            attnOutput.transposed(0, 2, 1, 3).reshaped(B, L, -1),
            oWeight.T
        )
        if attrs.bias {
            if let ob = optionalWeight(field: "o_proj", role: .bias) { output = output + ob }
        }

        return output
    }

    // MARK: - MLP

    private func executeMLP(
        _ attrs: MLPAttributes, input: MLXArray
    ) throws -> MLXArray {
        let gateWeight = try weight(field: "gate_proj", role: .weight)
        let downWeight = try weight(field: "down_proj", role: .weight)

        var gate = MLX.matmul(input, gateWeight.T)

        if attrs.bias {
            if let gb = optionalWeight(field: "gate_proj", role: .bias) { gate = gate + gb }
        }

        // Activation
        let activated: MLXArray
        switch attrs.activation {
        case .silu, .swish:
            activated = silu(gate)
        case .gelu:
            activated = gelu(gate)
        case .relu:
            activated = relu(gate)
        case .custom:
            activated = silu(gate)
        }

        // Gating
        let gated: MLXArray
        switch attrs.gating {
        case .swiglu, .geglu, .glu:
            let upWeight = try weight(field: "up_proj", role: .weight)
            var up = MLX.matmul(input, upWeight.T)
            if attrs.bias {
                if let ub = optionalWeight(field: "up_proj", role: .bias) { up = up + ub }
            }
            gated = activated * up
        case .none:
            gated = activated
        case .custom:
            let upWeight = try weight(field: "up_proj", role: .weight)
            let up = MLX.matmul(input, upWeight.T)
            gated = activated * up
        }

        var output = MLX.matmul(gated, downWeight.T)
        if attrs.bias {
            if let db = optionalWeight(field: "down_proj", role: .bias) { output = output + db }
        }

        return output
    }

    // MARK: - Mixture of Experts

    private func executeMoE(
        _ attrs: MoEAttributes, input: MLXArray
    ) throws -> MLXArray {
        let (B, L, D) = (input.dim(0), input.dim(1), input.dim(2))
        let flat = input.reshaped(-1, D)

        let routerWeight = try weight(field: "router", role: .weight)
        let gateLogits = MLX.matmul(flat, routerWeight.T)

        let topKIndices = MLX.argSort(gateLogits, axis: -1)[
            0..., (gateLogits.dim(-1) - attrs.expertsPerToken)...]
        let topKGateLogits = MLX.takeAlong(gateLogits, topKIndices, axis: -1)
        let gateWeights = softmax(topKGateLogits, axis: -1)

        var output = MLXArray.zeros(like: flat)

        for expertIdx in 0..<attrs.expertCount {
            pathComponents.append(.field("experts"))
            pathComponents.append(.index(expertIdx))

            // Accumulate combined weight across all topK slots (pure MLX, no item() sync)
            var expertWeight = MLXArray.zeros([flat.dim(0), 1])
            for k in 0..<attrs.expertsPerToken {
                let kMask = topKIndices[0..., k..<(k + 1)] .== MLXArray(Int32(expertIdx))
                let kMaskFloat = kMask.asType(.float32)
                expertWeight = expertWeight + gateWeights[0..., k..<(k + 1)] * kMaskFloat
            }

            // Run expert MLP once with combined weight (skip if no tokens routed)
            let expertGateW = try weight(field: "gate_proj", role: .weight)
            let expertUpW = try weight(field: "up_proj", role: .weight)
            let expertDownW = try weight(field: "down_proj", role: .weight)

            let activated: MLXArray
            switch attrs.expertMLP.activation {
            case .gelu: activated = gelu(MLX.matmul(flat, expertGateW.T))
            default: activated = silu(MLX.matmul(flat, expertGateW.T))
            }

            let expertOut = MLX.matmul(
                activated * MLX.matmul(flat, expertUpW.T),
                expertDownW.T
            )

            output = output + expertOut * expertWeight

            pathComponents.removeLast()
            pathComponents.removeLast()
        }

        return output.reshaped(B, L, D)
    }

    // MARK: - State Space

    private func executeStateSpace(
        _ attrs: StateSpaceAttributes, input: MLXArray
    ) throws -> MLXArray {
        let variant = try MLXStateSpaceVariant(variant: attrs.variant)
        switch variant {
        case .deltaNet, .gatedDeltaNet:
            return try executeDeltaNet(attrs, input: input)
        }
    }

    private func executeDeltaNet(
        _ attrs: StateSpaceAttributes, input: MLXArray
    ) throws -> MLXArray {
        let B = input.dim(0)
        let T = input.dim(1)

        // Load weights
        let qkvWeight = try weight(field: "in_proj_qkv", role: .weight)
        let zWeight = try weight(field: "in_proj_z", role: .weight)
        let bWeight = try weight(field: "in_proj_b", role: .weight)
        let aWeight = try weight(field: "in_proj_a", role: .weight)
        let convWeight = try weight(field: "conv1d", role: .weight)
        let outWeight = try weight(field: "out_proj", role: .weight)
        let normWeight = try weight(field: "norm", role: .scale)
        let dtBias = try weight(field: "dt_bias", role: .bias)
        let aLog = try weight(field: "A_log", role: .weight)

        // Infer dimensions from weight shapes
        // Weight layout: [output_dim, input_dim] — dim(0) is the output dimension
        let totalQKV = qkvWeight.dim(0)
        let valueDim = zWeight.dim(0)
        let keyDim = (totalQKV - valueDim) / 2
        let linearKeyHeadDim = attrs.stateSize
        let linearKeyHeads = keyDim / linearKeyHeadDim
        let linearValueHeadDim = normWeight.dim(0)
        let linearValueHeads = valueDim / linearValueHeadDim
        let convDim = totalQKV
        let convKernelSize = convWeight.dim(1)
        let scale = 1.0 / Float(linearKeyHeadDim).squareRoot()

        // Projections
        let mixedQKV = MLX.matmul(input, qkvWeight.T)
        let z = MLX.matmul(input, zWeight.T)
        let b = MLX.matmul(input, bWeight.T)
        let a = MLX.matmul(input, aWeight.T)

        // Get recurrent cache by path
        let slot = try cacheSlot()
        let cache = caches[slot] as! MLXRecurrentCache

        // Causal Conv1D
        let prefix: MLXArray
        if let existing = cache.convState {
            prefix = existing
        } else {
            prefix = MLXArray.zeros([B, convKernelSize, convDim])
        }
        let convInput = concatenated([prefix, mixedQKV], axis: 1)
        cache.convState = convInput[0..., (convInput.dim(1) - convKernelSize)..., 0...]

        // Depthwise conv1d — ensure weight is 3D [channels, kernel_size, 1]
        let convWeight3d = convWeight.ndim == 2
            ? convWeight.expandedDimensions(axis: -1)
            : convWeight
        let rawConv = conv1d(convInput, convWeight3d, stride: 1, padding: 0, groups: convDim)
        let activated = silu(rawConv[0..., 1..., 0...])  // Skip warmup token

        // Split Q, K, V
        let parts = activated.split(indices: [keyDim, 2 * keyDim], axis: -1)
        let query = parts[0].reshaped(B, T, linearKeyHeads, linearKeyHeadDim)
        let key = parts[1].reshaped(B, T, linearKeyHeads, linearKeyHeadDim)
        let value = parts[2].reshaped(B, T, linearValueHeads, linearValueHeadDim)

        // Gates
        let beta = sigmoid(b)
        let g = -MLX.exp(aLog) * softplus(a + dtBias)
        let decay = MLX.exp(g)

        // Delta rule recurrence
        let (attnOut, newState) = deltaNetRecurrence(
            query: query, key: key, value: value,
            decay: decay, beta: beta, state: cache.recurrentState,
            scale: scale
        )
        cache.recurrentState = newState
        cache.incrementOffset(by: T)

        // Gated output norm
        let dv = linearValueHeadDim
        let numHeads = linearValueHeads
        let flat = attnOut.reshaped(B * T * numHeads, dv)
        let zFlat = z.reshaped(B, T, numHeads, dv).reshaped(B * T * numHeads, dv)
        let normed = MLXFast.rmsNorm(flat, weight: 1 + normWeight, eps: 1e-6) * silu(zFlat)
        let gated = normed.reshaped(B, T, valueDim)

        return MLX.matmul(gated, outWeight.T)
    }

    /// Per-token recurrence for the DeltaNet state update.
    ///
    /// S_t = exp(g) * S_{t-1} + k_t ⊗ [β(v_t − exp(g)·S^T·k_t)]
    /// o_t = S_t^T · (q_t / √d_k)
    ///
    /// Pre-slices all timesteps before the loop to avoid repeated slice+squeeze overhead.
    private func deltaNetRecurrence(
        query: MLXArray, key: MLXArray, value: MLXArray,
        decay: MLXArray, beta: MLXArray, state: MLXArray?,
        scale: Float
    ) -> (MLXArray, MLXArray) {
        let B = query.dim(0), T = query.dim(1), H = query.dim(2)
        let dk = query.dim(3), dv = value.dim(3)

        let qN = l2Norm(query) * MLXArray(scale)
        let kN = l2Norm(key)

        // Pre-slice all timesteps: split along axis=1 returns [B, 1, H, d] arrays
        let qSlices = qN.split(parts: T, axis: 1)
        let kSlices = kN.split(parts: T, axis: 1)
        let vSlices = value.split(parts: T, axis: 1)
        let gSlices = decay.split(parts: T, axis: 1)
        let bSlices = beta.split(parts: T, axis: 1)

        var S = state ?? MLXArray.zeros([B, H, dk, dv])
        var outputs = [MLXArray]()
        outputs.reserveCapacity(T)

        for t in 0..<T {
            let qt = qSlices[t].squeezed(axis: 1)  // [B, H, dk]
            let kt = kSlices[t].squeezed(axis: 1)  // [B, H, dk]
            let vt = vSlices[t].squeezed(axis: 1)  // [B, H, dv]
            let gt = gSlices[t].squeezed(axis: 1)  // [B, H]
            let bt = bSlices[t].squeezed(axis: 1)  // [B, H]

            // Decay state
            let gE = gt.expandedDimensions(axes: [-1, -2])
            S = S * gE

            // Memory readout
            let kE = kt.expandedDimensions(axis: -1)
            let kvMem = (S * kE).sum(axis: -2)

            // State delta
            let delta = bt.expandedDimensions(axis: -1) * (vt - kvMem)
            S = S + kE * delta.expandedDimensions(axis: -2)

            // Output readout
            let qE = qt.expandedDimensions(axis: -1)
            let ot = (S * qE).sum(axis: -2)
            outputs.append(ot.expandedDimensions(axis: 1))
        }

        return (concatenated(outputs, axis: 1), S)
    }

    // MARK: - Utility

    /// L2 normalization along the last dimension.
    private func l2Norm(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
        x / MLX.sqrt((x * x).sum(axis: -1, keepDims: true) + MLXArray(eps))
    }

    // layerNormOp is shared from LoweredNorm.swift

    /// Softplus activation: log(1 + exp(x)).
    private func softplus(_ x: MLXArray) -> MLXArray {
        MLX.log(1 + MLX.exp(x))
    }

    // MARK: - Weight Lookup

    /// Look up a weight at the current structural path with a given role.
    private func weight(role: ParameterRole) throws -> MLXArray {
        let path = StructuralPath(components: pathComponents)
        let slot = ParameterSlot(path: path, role: role)
        return try model.weightStore.require(slot)
    }

    /// Look up a weight at the current path, returning nil if not found.
    private func optionalWeight(role: ParameterRole) -> MLXArray? {
        let path = StructuralPath(components: pathComponents)
        let slot = ParameterSlot(path: path, role: role)
        return model.weightStore.get(slot)
    }

    /// Look up a weight at a sub-field of the current path.
    private func weight(field: String, role: ParameterRole) throws -> MLXArray {
        let path = StructuralPath(components: pathComponents)
            .appending(.field(field))
        let slot = ParameterSlot(path: path, role: role)
        return try model.weightStore.require(slot)
    }

    /// Look up a weight at a sub-field, returning nil if not found.
    private func optionalWeight(field: String, role: ParameterRole) -> MLXArray? {
        let path = StructuralPath(components: pathComponents)
            .appending(.field(field))
        let slot = ParameterSlot(path: path, role: role)
        return model.weightStore.get(slot)
    }
}

// MARK: - Executor Protocol Conformance

/// - Note: `@unchecked Sendable` is required for `Executor` protocol conformance.
///   `MLXExecutor` uses mutable cache state and is designed for serial
///   forward calls. Concurrent access must be synchronized externally.
extension MLXExecutor: @unchecked Sendable {}

extension MLXExecutor: Executor {

    /// Execute a compiled model via the SwiftLM `Executor` protocol.
    ///
    /// Extracts `MLXCompiledModel` from `RuntimePlan.data`, converts
    /// `ModelInputs.tokenIDs` to `MLXArray`, and wraps the output logits
    /// back into `ModelOutputs`.
    public func run(
        _ model: CompiledModel,
        inputs: ModelInputs
    ) async throws -> ModelOutputs {
        guard let tokenArray = inputs.tokenIDs.storage as? MLXArray else {
            throw CompilerError.invalidWeightStorage(
                ParameterSlot(
                    path: StructuralPath(),
                    role: .custom("tokenIDs")),
                "Expected MLXArray for tokenIDs, got \(type(of: inputs.tokenIDs.storage))")
        }

        let logits = try forward(tokenIDs: tokenArray)

        // Wrap current cache state
        let cachedLength = self.caches.compactMap { $0 as? MLXKVCache }.first?.offset ?? 0
        let cacheState = KVCacheState(storage: cachedLength as Int, cachedLength: cachedLength)

        return ModelOutputs(
            logits: TensorData(
                shape: logits.shape.map { $0 },
                dtype: .float32,
                storage: logits),
            cache: cacheState
        )
    }
}
