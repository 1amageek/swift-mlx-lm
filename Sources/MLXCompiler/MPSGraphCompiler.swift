import Foundation
import Metal
import MetalPerformanceShadersGraph
@preconcurrency import MLX
import SwiftLM

/// Compiles a SwiftLM `ModelGraph` into an MPSGraph-based inference model.
///
/// Uses the same IR walk pattern as `MLXInferenceCompiler`:
/// - Phase 1: Scan (reuses `MLXInferenceCompiler.scan()`)
/// - Phase 2: Compile (recursive region/operation walk → MPSGraph ops)
/// - Phase 3: Wrap (MPSGraphCompiledModel)
///
/// Shares IR, weight binding, and cache discovery with MLX path.
/// Only the compilation target differs: MPSGraph ops instead of LoweredSteps.
public struct MPSGraphCompiler: Sendable {

    public init() {}

    /// Compile a ModelGraph with bound weights into a fused MPSGraph model.
    ///
    /// Unsupported ops (stateSpace, shortConv, moe) throw `CompilerError`.
    /// Caller should catch and fall back to MLX path.
    public func compile(
        graph modelGraph: ModelGraph, weights: BoundWeights
    ) throws -> MPSGraphCompiledModel {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CompilerError.invalidGraphStructure("No Metal device available")
        }
        guard let queue = device.makeCommandQueue() else {
            throw CompilerError.invalidGraphStructure("Failed to create Metal command queue")
        }

        let store = try InferenceWeightStore(boundWeights: weights)
        let mpsGraph = MPSGraph()

        // Phase 1: Scan — reuse MLXInferenceCompiler's scan for cache discovery
        let scanResult = MLXInferenceCompiler().scan(graph: modelGraph)

        // Phase 2: Compile — walk IR, build MPSGraph ops
        let input = mpsGraph.placeholder(shape: [1, -1], dataType: .int32, name: "token_ids")

        var context = CompilationContext(
            graph: mpsGraph, store: store, embeddingTensor: nil)

        let output = try compileRegion(
            modelGraph.rootRegion,
            pathComponents: [],
            input: input,
            context: &context)

        return MPSGraphCompiledModel(
            graph: mpsGraph, device: device, commandQueue: queue,
            inputPlaceholder: input, outputTensor: output,
            metadata: InferenceMetadata(
                cacheSlotCount: scanResult.cacheDescriptors.count,
                cacheDescriptors: scanResult.cacheDescriptors,
                hasTiedOutputHead: scanResult.hasTiedOutputHead))
    }

    // MARK: - Compilation Context

    private struct CompilationContext {
        let graph: MPSGraph
        let store: InferenceWeightStore
        /// Token embedding variable — stored for tied output head reuse.
        var embeddingTensor: MPSGraphTensor?
        /// RoPE frequency table — built once, shared across layers.
        var ropeFreqs: MPSGraphTensor?
    }

    // MARK: - Region Walk

    private func compileRegion(
        _ region: Region, pathComponents: [StructuralPathComponent],
        input: MPSGraphTensor, context: inout CompilationContext
    ) throws -> MPSGraphTensor {
        var h = input

        for (i, op) in region.operations.enumerated() {
            let opPath = pathComponents + [.operation(i)]

            // Handle .repeating inline (before switch) — same pattern as MLXInferenceCompiler
            if case .repeating(let count, let body) = op.kind {
                for iter in 0..<count {
                    let iterPath = opPath + [.regionBody, .index(iter)]
                    h = try compileRegion(body, pathComponents: iterPath,
                                           input: h, context: &context)
                }
                continue
            }

            // Handle .layerStack inline
            if case .layerStack(let layers) = op.kind {
                for (iter, layer) in layers.enumerated() {
                    let iterPath = opPath + [.regionBody, .index(iter)]
                    h = try compileRegion(layer, pathComponents: iterPath,
                                           input: h, context: &context)
                }
                continue
            }

            let path = StructuralPath(components: opPath)
            h = try compileOperation(op, pathComponents: opPath, path: path,
                                      input: h, context: &context)
        }

        return h
    }

    // MARK: - Operation Compilation

    private func compileOperation(
        _ op: SwiftLM.Operation, pathComponents: [StructuralPathComponent],
        path: StructuralPath, input: MPSGraphTensor,
        context: inout CompilationContext
    ) throws -> MPSGraphTensor {
        let g = context.graph

        switch op.kind {

        // MARK: Token Embedding

        case .tokenEmbedding(let attrs):
            let slot = ParameterSlot(path: path, role: .embeddingTable)
            let table = try resolveWeight(
                context.store, slot: slot,
                shape: [attrs.vocabSize, attrs.embeddingSize], graph: g)
            context.embeddingTensor = table
            return g.gatherAlongAxis(0, updates: table, indices: input, name: "embed")

        // MARK: Normalization

        case .rmsNorm(let attrs):
            let w = try resolveWeight(
                context.store, slot: ParameterSlot(path: path, role: .scale),
                shape: [attrs.dimension], graph: g)
            return MPSGraphOps.rmsNorm(g, input: input, weight: w,
                                        epsilon: attrs.epsilon, name: "\(path)")

        case .layerNorm(let attrs):
            let w = try resolveWeight(
                context.store, slot: ParameterSlot(path: path, role: .scale),
                shape: [attrs.dimension], graph: g)
            // Manual LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
            let mean = g.mean(of: input, axes: [-1], name: "\(path).mean")
            let centered = g.subtraction(input, mean, name: "\(path).centered")
            let variance = g.mean(of: g.multiplication(centered, centered, name: "\(path).sq"),
                                   axes: [-1], name: "\(path).var")
            let eps = g.constant(Double(attrs.epsilon), dataType: .float16)
            let inv = g.reverseSquareRoot(
                with: g.addition(variance, eps, name: "\(path).eps"),
                name: "\(path).inv")
            var result = g.multiplication(g.multiplication(centered, inv, name: "\(path).norm"),
                                           w, name: "\(path).scaled")
            if attrs.affine, let bias = try? resolveWeight(
                context.store, slot: ParameterSlot(path: path, role: .bias),
                shape: [attrs.dimension], graph: g) {
                result = g.addition(result, bias, name: "\(path).biased")
            }
            return result

        // MARK: Attention

        case .attention(let attrs):
            return try compileAttention(attrs, path: path, input: input, context: &context)

        // MARK: MLP

        case .mlp(let attrs):
            return try compileMLP(attrs, path: path, input: input, context: &context)

        // MARK: Output Head

        case .outputHead(let attrs):
            if attrs.tiedToEmbedding, let embedding = context.embeddingTensor {
                return MPSGraphOps.linear(g, input: input, weight: embedding,
                                           name: "lm_head")
            }
            let slot = ParameterSlot(path: path, role: .outputProjection)
            let w = try resolveWeight(
                context.store, slot: slot,
                shape: [attrs.vocabSize, attrs.inputSize], graph: g)
            return MPSGraphOps.linear(g, input: input, weight: w, name: "lm_head")

        // MARK: Linear

        case .linear(let attrs):
            let w = try resolveWeight(
                context.store,
                slot: ParameterSlot(path: path, role: .weight),
                shape: [attrs.outputSize, attrs.inputSize], graph: g)
            var result = MPSGraphOps.linear(g, input: input, weight: w, name: "\(path)")
            if attrs.bias, let b = try? resolveWeight(
                context.store, slot: ParameterSlot(path: path, role: .bias),
                shape: [attrs.outputSize], graph: g) {
                result = g.addition(result, b, name: "\(path).bias")
            }
            return result

        // MARK: RoPE (standalone)

        case .rope(let attrs):
            let freqs = ensureRoPEFreqs(attrs: attrs, context: &context)
            let heads = input.shape?[1].intValue ?? 1
            return MPSGraphOps.applyRoPE(
                g, input: input, frequencies: freqs,
                heads: heads, headDim: attrs.dimension,
                name: "\(path)")

        // MARK: Positional Embedding

        case .positionalEmbedding:
            let table = try resolveWeight(
                context.store, slot: ParameterSlot(path: path, role: .embeddingTable),
                shape: [], graph: g) // shape inferred from weight
            // Simple additive: h + table[positions]
            // positions = coordinate along seq axis
            let positions = g.coordinate(
                alongAxis: 1, withShapeTensor: g.shapeOf(input, name: nil),
                name: "\(path).pos")
            let posEmb = g.gatherAlongAxis(0, updates: table, indices: positions,
                                            name: "\(path).gather")
            return g.addition(input, posEmb, name: "\(path).add")

        // MARK: Structural

        case .residual(_, let body):
            let bodyPath = pathComponents + [.regionBody]
            let bodyOutput = try compileRegion(
                body, pathComponents: bodyPath, input: input, context: &context)
            return g.addition(input, bodyOutput, name: "\(path).residual")

        case .parallel(let merge, let branches):
            var results: [MPSGraphTensor] = []
            for (i, branch) in branches.enumerated() {
                let branchPath = pathComponents + [.regionBranch(i)]
                let branchOutput = try compileRegion(
                    branch, pathComponents: branchPath, input: input, context: &context)
                results.append(branchOutput)
            }
            switch merge {
            case .add:
                return results.dropFirst().reduce(results[0]) { g.addition($0, $1, name: nil) }
            case .concat:
                return g.concatTensors(results, dimension: -1, name: "\(path).concat")
            default:
                return results.dropFirst().reduce(results[0]) { g.addition($0, $1, name: nil) }
            }

        // MARK: Unsupported (fall back to MLX)

        case .stateSpace:
            throw CompilerError.unsupportedOperation("stateSpace (MPSGraph)")
        case .shortConv:
            throw CompilerError.unsupportedOperation("shortConv (MPSGraph)")
        case .moe:
            throw CompilerError.unsupportedOperation("moe (MPSGraph)")
        case .repeating:
            throw CompilerError.invalidGraphStructure("repeating handled in compileRegion")
        case .layerStack:
            throw CompilerError.invalidGraphStructure("layerStack handled in compileRegion")
        case .custom(let attrs):
            throw CompilerError.unsupportedOperation("custom(\(attrs.domain).\(attrs.name))")
        }
    }

    // MARK: - Attention Compilation

    private func compileAttention(
        _ attrs: AttentionAttributes, path: StructuralPath,
        input: MPSGraphTensor, context: inout CompilationContext
    ) throws -> MPSGraphTensor {
        let g = context.graph
        let H = attrs.headCount
        let KVH = attrs.kvHeadCount
        let hd = attrs.headDimension
        let n = "\(path)"

        let qW = try resolveProjectionWeight(context.store, path: path, field: "q_proj",
                                              shape: [H * hd, attrs.hiddenSize], graph: g)
        let kW = try resolveProjectionWeight(context.store, path: path, field: "k_proj",
                                              shape: [KVH * hd, attrs.hiddenSize], graph: g)
        let vW = try resolveProjectionWeight(context.store, path: path, field: "v_proj",
                                              shape: [KVH * hd, attrs.hiddenSize], graph: g)
        let oW = try resolveProjectionWeight(context.store, path: path, field: "o_proj",
                                              shape: [attrs.hiddenSize, H * hd], graph: g)

        var q = MPSGraphOps.toHeads(g,
            input: MPSGraphOps.linear(g, input: input, weight: qW, name: "\(n).q"),
            heads: H, headDim: hd, name: "\(n).q")
        var k = MPSGraphOps.toHeads(g,
            input: MPSGraphOps.linear(g, input: input, weight: kW, name: "\(n).k"),
            heads: KVH, headDim: hd, name: "\(n).k")
        let v = MPSGraphOps.toHeads(g,
            input: MPSGraphOps.linear(g, input: input, weight: vW, name: "\(n).v"),
            heads: KVH, headDim: hd, name: "\(n).v")

        // RoPE
        if let rope = attrs.rope {
            let freqs = ensureRoPEFreqs(attrs: rope, context: &context)
            q = MPSGraphOps.applyRoPE(g, input: q, frequencies: freqs,
                                       heads: H, headDim: hd, name: "\(n).q")
            k = MPSGraphOps.applyRoPE(g, input: k, frequencies: freqs,
                                       heads: KVH, headDim: hd, name: "\(n).k")
        }

        // GQA head repeat
        let repeatFactor = H / KVH
        let kAttn = MPSGraphOps.repeatKVHeads(g, input: k, repeatFactor: repeatFactor, name: "\(n).k")
        let vAttn = MPSGraphOps.repeatKVHeads(g, input: v, repeatFactor: repeatFactor, name: "\(n).v")

        // Causal SDPA
        let attn = MPSGraphOps.causalScaledDotProductAttention(
            g, query: q, key: kAttn, value: vAttn, headDim: hd, name: n)

        let flat = MPSGraphOps.fromHeads(g, input: attn, totalDim: H * hd, name: n)
        return MPSGraphOps.linear(g, input: flat, weight: oW, name: "\(n).o")
    }

    // MARK: - MLP Compilation

    private func compileMLP(
        _ attrs: MLPAttributes, path: StructuralPath,
        input: MPSGraphTensor, context: inout CompilationContext
    ) throws -> MPSGraphTensor {
        let g = context.graph
        let n = "\(path)"

        let gateW = try resolveProjectionWeight(context.store, path: path, field: "gate_proj",
                                                 shape: [attrs.intermediateSize, attrs.inputSize], graph: g)
        let downW = try resolveProjectionWeight(context.store, path: path, field: "down_proj",
                                                 shape: [attrs.outputSize, attrs.intermediateSize], graph: g)

        let gate = MPSGraphOps.linear(g, input: input, weight: gateW, name: "\(n).gate")

        let activated: MPSGraphTensor
        switch attrs.gating {
        case .none:
            // No gating — just activation on gate
            activated = applySiLU(g, gate, name: n)
        default:
            // Gated: silu(gate) * up
            let upW = try resolveProjectionWeight(context.store, path: path, field: "up_proj",
                                                   shape: [attrs.intermediateSize, attrs.inputSize], graph: g)
            let up = MPSGraphOps.linear(g, input: input, weight: upW, name: "\(n).up")
            activated = MPSGraphOps.siluGate(g, gate: gate, up: up, name: n)
        }

        return MPSGraphOps.linear(g, input: activated, weight: downW, name: "\(n).down")
    }

    // MARK: - Helpers

    private func applySiLU(
        _ g: MPSGraph, _ x: MPSGraphTensor, name: String
    ) -> MPSGraphTensor {
        g.multiplication(x, g.sigmoid(with: x, name: "\(name).sig"), name: "\(name).silu")
    }

    private func ensureRoPEFreqs(
        attrs: RoPEAttributes, context: inout CompilationContext
    ) -> MPSGraphTensor {
        if let existing = context.ropeFreqs { return existing }
        let freqs = MPSGraphOps.buildRoPEFrequencies(
            context.graph, headDim: attrs.dimension, theta: attrs.base)
        context.ropeFreqs = freqs
        return freqs
    }

    // MARK: - Weight Resolution

    private func resolveWeight(
        _ store: InferenceWeightStore, slot: ParameterSlot,
        shape: [Int], graph: MPSGraph
    ) throws -> MPSGraphTensor {
        let storage = try store.require(slot)
        return storageToVariable(storage, shape: shape, graph: graph, name: "\(slot)")
    }

    private func resolveProjectionWeight(
        _ store: InferenceWeightStore, path: StructuralPath,
        field: String, shape: [Int], graph: MPSGraph
    ) throws -> MPSGraphTensor {
        let fieldPath = path.appending(.field(field))
        let slot = ParameterSlot(path: fieldPath, role: .weight)
        return try resolveWeight(store, slot: slot, shape: shape, graph: graph)
    }

    private func storageToVariable(
        _ storage: MLXTensorStorage, shape: [Int], graph: MPSGraph, name: String
    ) -> MPSGraphTensor {
        let array: MLXArray
        switch storage {
        case .dense(let a):
            array = a
        case .affineQuantized(let qt):
            array = dequantized(
                qt.packedWeight, scales: qt.scales, biases: qt.zeroBiases,
                groupSize: qt.groupSize, bits: qt.bits)
        }

        let f16 = array.asType(.float16)
        eval(f16)
        let data = f16.asArray(Float16.self).withUnsafeBytes { Data($0) }
        let nsShape = shape.isEmpty
            ? f16.shape.map { $0 as NSNumber }
            : shape.map { $0 as NSNumber }
        return graph.variable(with: data, shape: nsShape, dataType: .float16, name: name)
    }
}
