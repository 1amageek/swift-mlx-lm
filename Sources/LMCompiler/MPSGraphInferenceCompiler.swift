import Foundation
import Metal
import MetalPerformanceShadersGraph
@preconcurrency import MLX
import SwiftLM

/// Compiles a SwiftLM `ModelGraph` into an MPSGraph-based inference model.
///
/// Same IR walk pattern as `MLXInferenceCompiler`:
/// Phase 1: Scan (reuses `MLXInferenceCompiler.scan()`)
/// Phase 2: Compile (recursive region/operation walk → MPSGraph ops)
/// Phase 3: Wrap (MPSGraphInferenceModel)
public struct MPSGraphInferenceCompiler: Sendable {

    public init() {}

    public func compile(
        graph modelGraph: ModelGraph, weights: BoundWeights
    ) throws -> MPSGraphInferenceModel {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CompilerError.invalidGraphStructure("No Metal device available")
        }
        guard let queue = device.makeCommandQueue() else {
            throw CompilerError.invalidGraphStructure("Failed to create Metal command queue")
        }

        let store = try InferenceWeightStore(boundWeights: weights)
        let mpsGraph = MPSGraph()
        let scanResult = MLXInferenceCompiler().scan(graph: modelGraph)

        let input = mpsGraph.placeholder(shape: [1, -1], dataType: .int32, name: "token_ids")
        let maskPh = mpsGraph.placeholder(shape: [1, 1, -1, -1], dataType: .float16, name: "causal_mask")

        var context = CompilationContext(graph: mpsGraph, store: store, maskPlaceholder: maskPh)

        let output = try compileRegion(
            modelGraph.rootRegion, pathComponents: [], input: input, context: &context)

        return MPSGraphInferenceModel(
            graph: mpsGraph, device: device, commandQueue: queue,
            inputPlaceholder: input, maskPlaceholder: maskPh,
            ropeCosPh: context.ropeCosPh, ropeSinPh: context.ropeSinPh,
            ropeHeadDim: context.ropeHeadDim,
            ropeTheta: context.ropeTheta,
            ropeScaling: context.ropeScaling,
            outputTensor: output,
            metadata: InferenceMetadata(
                cacheSlotCount: scanResult.cacheDescriptors.count,
                cacheDescriptors: scanResult.cacheDescriptors,
                hasTiedOutputHead: scanResult.hasTiedOutputHead))
    }

    // MARK: - Context

    private struct CompilationContext {
        let graph: MPSGraph
        let store: InferenceWeightStore
        var embeddingTensor: MPSGraphTensor?
        var maskPlaceholder: MPSGraphTensor?
        var ropeCosPh: MPSGraphTensor?
        var ropeSinPh: MPSGraphTensor?
        var ropeHeadDim: Int = 0
        var ropeTheta: Float = 500000.0
        var ropeScaling: RoPEScaling?
        /// Stored gate tensor for output-gated attention (sigmoidPackedInQProj).
        var pendingGate: MPSGraphTensor?
    }

    // MARK: - Region Walk

    private func compileRegion(
        _ region: Region, pathComponents: [StructuralPathComponent],
        input: MPSGraphTensor, context: inout CompilationContext
    ) throws -> MPSGraphTensor {
        var h = input
        for (i, op) in region.operations.enumerated() {
            let opPath = pathComponents + [.operation(i)]

            if case .repeating(let count, let body) = op.kind {
                for iter in 0..<count {
                    h = try compileRegion(body, pathComponents: opPath + [.regionBody, .index(iter)],
                                           input: h, context: &context)
                }
                continue
            }
            if case .layerStack(let layers) = op.kind {
                for (iter, layer) in layers.enumerated() {
                    h = try compileRegion(layer, pathComponents: opPath + [.regionBody, .index(iter)],
                                           input: h, context: &context)
                }
                continue
            }

            h = try compileOperation(op, pathComponents: opPath,
                                      path: StructuralPath(components: opPath),
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

        case .tokenEmbedding(let attrs):
            let table = try resolveWeight(context.store, slot: ParameterSlot(path: path, role: .embeddingTable),
                                           shape: [attrs.vocabSize, attrs.embeddingSize], graph: g)
            context.embeddingTensor = table
            return g.gather(withUpdatesTensor: table, indicesTensor: input, axis: 0, batchDimensions: 0, name: "embed")

        case .rmsNorm(let attrs):
            let w = try resolveWeight(context.store, slot: ParameterSlot(path: path, role: .scale),
                                       shape: [attrs.dimension], graph: g)
            return MPSGraphOps.rmsNorm(g, input: input, weight: w, epsilon: attrs.epsilon, name: "\(path)")

        case .layerNorm(let attrs):
            let w = try resolveWeight(context.store, slot: ParameterSlot(path: path, role: .scale),
                                       shape: [attrs.dimension], graph: g)
            let mean = g.mean(of: input, axes: [-1], name: "\(path).mean")
            let centered = g.subtraction(input, mean, name: "\(path).centered")
            let variance = g.mean(of: g.multiplication(centered, centered, name: "\(path).sq"),
                                   axes: [-1], name: "\(path).var")
            let eps = g.constant(Double(attrs.epsilon), dataType: .float16)
            let inv = g.reverseSquareRoot(with: g.addition(variance, eps, name: "\(path).eps"),
                                           name: "\(path).inv")
            var result = g.multiplication(g.multiplication(centered, inv, name: "\(path).norm"),
                                           w, name: "\(path).scaled")
            if attrs.affine, let bias = try? resolveWeight(
                context.store, slot: ParameterSlot(path: path, role: .bias),
                shape: [attrs.dimension], graph: g) {
                result = g.addition(result, bias, name: "\(path).biased")
            }
            return result

        case .attention(let attrs):
            return try compileAttention(attrs, path: path, input: input, context: &context)

        case .mlp(let attrs):
            return try compileMLP(attrs, path: path, input: input, context: &context)

        case .outputHead(let attrs):
            if attrs.tiedToEmbedding, let emb = context.embeddingTensor {
                return MPSGraphOps.linear(g, input: input, weight: emb, name: "lm_head")
            }
            let w = try resolveWeight(context.store, slot: ParameterSlot(path: path, role: .outputProjection),
                                       shape: [attrs.vocabSize, attrs.inputSize], graph: g)
            return MPSGraphOps.linear(g, input: input, weight: w, name: "lm_head")

        case .linear(let attrs):
            let w = try resolveWeight(context.store, slot: ParameterSlot(path: path, role: .weight),
                                       shape: [attrs.outputSize, attrs.inputSize], graph: g)
            var result = MPSGraphOps.linear(g, input: input, weight: w, name: "\(path)")
            if attrs.bias, let b = try? resolveWeight(
                context.store, slot: ParameterSlot(path: path, role: .bias),
                shape: [attrs.outputSize], graph: g) {
                result = g.addition(result, b, name: "\(path).bias")
            }
            return result

        case .rope(let attrs):
            let (cosPh, sinPh) = ensureRoPEPlaceholders(headDim: attrs.dimension, context: &context)
            context.ropeTheta = attrs.base
            context.ropeScaling = attrs.scaling
            return MPSGraphOps.applyRoPE(g, input: input, cosTable: cosPh, sinTable: sinPh,
                                          headDim: attrs.dimension, name: "\(path)")

        // Unsupported — throw (MPSGraph cannot use coordinate ops)
        case .positionalEmbedding:
            throw CompilerError.unsupportedOperation("positionalEmbedding (MPSGraph: dynamic coordinate ops crash MLIR)")

        case .residual(_, let body):
            let bodyOutput = try compileRegion(body, pathComponents: pathComponents + [.regionBody],
                                                input: input, context: &context)
            return g.addition(input, bodyOutput, name: "\(path).residual")

        case .parallel(let merge, let branches):
            var results: [MPSGraphTensor] = []
            for (i, branch) in branches.enumerated() {
                results.append(try compileRegion(branch, pathComponents: pathComponents + [.regionBranch(i)],
                                                  input: input, context: &context))
            }
            switch merge {
            case .add: return results.dropFirst().reduce(results[0]) { g.addition($0, $1, name: nil) }
            case .concat: return g.concatTensors(results, dimension: -1, name: "\(path).concat")
            default: return results.dropFirst().reduce(results[0]) { g.addition($0, $1, name: nil) }
            }

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

    // MARK: - Attention

    private func compileAttention(
        _ attrs: AttentionAttributes, path: StructuralPath,
        input: MPSGraphTensor, context: inout CompilationContext
    ) throws -> MPSGraphTensor {
        let g = context.graph
        let H = attrs.headCount
        let KVH = attrs.kvHeadCount
        let hd = attrs.headDimension
        let n = "\(path)"

        // Determine effective Q head dim (may be doubled for output gate)
        let qOutDim: Int
        let hasOutputGate: Bool
        if case .sigmoidPackedInQProj = attrs.outputGate {
            qOutDim = H * hd * 2  // gate packed in Q projection
            hasOutputGate = true
        } else {
            qOutDim = H * hd
            hasOutputGate = false
        }

        let qW = try resolveProjectionWeight(context.store, path: path, field: "q_proj",
                                              shape: [qOutDim, attrs.hiddenSize], graph: g)
        let kW = try resolveProjectionWeight(context.store, path: path, field: "k_proj",
                                              shape: [KVH * hd, attrs.hiddenSize], graph: g)
        let vW = try resolveProjectionWeight(context.store, path: path, field: "v_proj",
                                              shape: [KVH * hd, attrs.hiddenSize], graph: g)
        let oW = try resolveProjectionWeight(context.store, path: path, field: "o_proj",
                                              shape: [attrs.hiddenSize, H * hd], graph: g)

        // Q projection
        var qRaw = MPSGraphOps.linear(g, input: input, weight: qW, name: "\(n).q")

        // Extract output gate if packed in Q
        var gateValues: MPSGraphTensor?
        if hasOutputGate {
            // qRaw: [1, T, H*2*hd] → reshape per-head [1, T, H, 2*hd] → split
            let perHead = g.reshape(qRaw, shape: [1, -1, H as NSNumber, (2 * hd) as NSNumber], name: "\(n).q.ph")
            let qPart = g.sliceTensor(perHead, dimension: 3, start: 0, length: hd, name: "\(n).q.val")
            let gPart = g.sliceTensor(perHead, dimension: 3, start: hd, length: hd, name: "\(n).q.gate")
            qRaw = g.reshape(qPart, shape: [1, -1, (H * hd) as NSNumber], name: "\(n).q.flat")
            gateValues = g.reshape(gPart, shape: [1, -1, (H * hd) as NSNumber], name: "\(n).g.flat")
        }

        var q = MPSGraphOps.toHeads(g, input: qRaw, heads: H, headDim: hd, name: "\(n).q")
        var k = MPSGraphOps.toHeads(g,
            input: MPSGraphOps.linear(g, input: input, weight: kW, name: "\(n).k"),
            heads: KVH, headDim: hd, name: "\(n).k")
        let v = MPSGraphOps.toHeads(g,
            input: MPSGraphOps.linear(g, input: input, weight: vW, name: "\(n).v"),
            heads: KVH, headDim: hd, name: "\(n).v")

        // QK Normalization (#5)
        if let qkNorm = attrs.qkNorm {
            switch qkNorm {
            case .rmsNorm, .layerNorm:
                let qnW = try resolveWeight(context.store,
                    slot: ParameterSlot(path: path.appending(.field("q_norm")), role: .scale),
                    shape: [hd], graph: g)
                let knW = try resolveWeight(context.store,
                    slot: ParameterSlot(path: path.appending(.field("k_norm")), role: .scale),
                    shape: [hd], graph: g)
                q = MPSGraphOps.rmsNorm(g, input: q, weight: qnW, epsilon: 1e-6, name: "\(n).qn")
                k = MPSGraphOps.rmsNorm(g, input: k, weight: knW, epsilon: 1e-6, name: "\(n).kn")
            case .none, .custom:
                break
            }
        }

        // RoPE (#2, #3: theta and scaling extracted from IR)
        if let rope = attrs.rope {
            context.ropeTheta = rope.base
            context.ropeScaling = rope.scaling
            let (cosPh, sinPh) = ensureRoPEPlaceholders(headDim: hd, context: &context)
            q = MPSGraphOps.applyRoPE(g, input: q, cosTable: cosPh, sinTable: sinPh,
                                       headDim: hd, name: "\(n).q")
            k = MPSGraphOps.applyRoPE(g, input: k, cosTable: cosPh, sinTable: sinPh,
                                       headDim: hd, name: "\(n).k")
        }

        // GQA head repeat
        let kAttn = MPSGraphOps.repeatKVHeads(g, input: k, repeatFactor: H / KVH, name: "\(n).k")
        let vAttn = MPSGraphOps.repeatKVHeads(g, input: v, repeatFactor: H / KVH, name: "\(n).v")

        // SDPA with causal mask
        var attnOut = MPSGraphOps.scaledDotProductAttention(
            g, query: q, key: kAttn, value: vAttn,
            mask: context.maskPlaceholder, headDim: hd, name: n)

        let flat = MPSGraphOps.fromHeads(g, input: attnOut, totalDim: H * hd, name: n)

        // Output gate (#6)
        var projected = MPSGraphOps.linear(g, input: flat, weight: oW, name: "\(n).o")
        if let gate = gateValues {
            projected = g.multiplication(projected, g.sigmoid(with: gate, name: "\(n).gsig"),
                                          name: "\(n).gated")
        }

        return projected
    }

    // MARK: - MLP

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
            activated = g.multiplication(gate, g.sigmoid(with: gate, name: "\(n).sig"), name: "\(n).silu")
        default:
            let upW = try resolveProjectionWeight(context.store, path: path, field: "up_proj",
                                                   shape: [attrs.intermediateSize, attrs.inputSize], graph: g)
            let up = MPSGraphOps.linear(g, input: input, weight: upW, name: "\(n).up")
            activated = MPSGraphOps.siluGate(g, gate: gate, up: up, name: n)
        }

        return MPSGraphOps.linear(g, input: activated, weight: downW, name: "\(n).down")
    }

    // MARK: - Helpers

    private func ensureRoPEPlaceholders(
        headDim: Int, context: inout CompilationContext
    ) -> (cos: MPSGraphTensor, sin: MPSGraphTensor) {
        if let c = context.ropeCosPh, let s = context.ropeSinPh {
            // (#9) Validate headDim consistency
            precondition(context.ropeHeadDim == headDim,
                "MPSGraph RoPE: all layers must use same headDim (\(context.ropeHeadDim) vs \(headDim))")
            return (c, s)
        }
        let halfDim = headDim / 2
        let cosPh = context.graph.placeholder(
            shape: [1, 1, -1, halfDim as NSNumber], dataType: .float16, name: "rope_cos")
        let sinPh = context.graph.placeholder(
            shape: [1, 1, -1, halfDim as NSNumber], dataType: .float16, name: "rope_sin")
        context.ropeCosPh = cosPh
        context.ropeSinPh = sinPh
        context.ropeHeadDim = headDim
        return (cosPh, sinPh)
    }

    private func resolveWeight(
        _ store: InferenceWeightStore, slot: ParameterSlot, shape: [Int], graph: MPSGraph
    ) throws -> MPSGraphTensor {
        storageToVariable(try store.require(slot), shape: shape, graph: graph, name: "\(slot)")
    }

    private func resolveProjectionWeight(
        _ store: InferenceWeightStore, path: StructuralPath,
        field: String, shape: [Int], graph: MPSGraph
    ) throws -> MPSGraphTensor {
        try resolveWeight(store, slot: ParameterSlot(path: path.appending(.field(field)), role: .weight),
                           shape: shape, graph: graph)
    }

    private func storageToVariable(
        _ storage: MLXTensorStorage, shape: [Int], graph: MPSGraph, name: String
    ) -> MPSGraphTensor {
        let array: MLXArray
        switch storage {
        case .dense(let a): array = a
        case .affineQuantized(let qt):
            array = dequantized(qt.packedWeight, scales: qt.scales, biases: qt.zeroBiases,
                                 groupSize: qt.groupSize, bits: qt.bits)
        }
        let f16 = array.asType(.float16)
        eval(f16)
        let data = f16.asArray(Float16.self).withUnsafeBytes { Data($0) }
        let nsShape = shape.isEmpty ? f16.shape.map { $0 as NSNumber } : shape.map { $0 as NSNumber }
        return graph.variable(with: data, shape: nsShape, dataType: .float16, name: name)
    }
}
