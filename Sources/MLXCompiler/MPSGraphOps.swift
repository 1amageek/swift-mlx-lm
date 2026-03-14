import Foundation
import Metal
import MetalPerformanceShadersGraph

/// Primitive MPSGraph operations for transformer inference.
///
/// Static functions that build MPSGraph tensor operations matching
/// the semantics of SwiftLM IR OperationKind primitives.
/// Used by `MPSGraphInferenceCompiler` during IR-to-MPSGraph compilation.
enum MPSGraphOps {

    // MARK: - Linear Projection

    /// x @ weight^T — standard linear projection.
    static func linear(
        _ graph: MPSGraph, input x: MPSGraphTensor, weight w: MPSGraphTensor, name: String
    ) -> MPSGraphTensor {
        graph.matrixMultiplication(
            primary: x,
            secondary: graph.transposeTensor(w, dimension: 0, withDimension: 1, name: nil),
            name: name)
    }

    // MARK: - RMSNorm

    /// RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
    static func rmsNorm(
        _ graph: MPSGraph, input x: MPSGraphTensor, weight w: MPSGraphTensor,
        epsilon: Float, name: String
    ) -> MPSGraphTensor {
        let eps = graph.constant(Double(epsilon), dataType: .float16)
        let sq = graph.multiplication(x, x, name: "\(name).sq")
        let mean = graph.mean(of: sq, axes: [-1], name: "\(name).mean")
        let inv = graph.reverseSquareRoot(
            with: graph.addition(mean, eps, name: "\(name).eps"), name: "\(name).inv")
        let normed = graph.multiplication(x, inv, name: "\(name).normed")
        return graph.multiplication(normed, w, name: "\(name).out")
    }

    // MARK: - Multi-Head Reshape

    /// Reshape [B, T, heads*headDim] → [B, heads, T, headDim] for multi-head attention.
    static func toHeads(
        _ graph: MPSGraph, input x: MPSGraphTensor,
        heads: Int, headDim: Int, name: String
    ) -> MPSGraphTensor {
        let reshaped = graph.reshape(
            x, shape: [1, -1, heads as NSNumber, headDim as NSNumber], name: nil)
        return graph.transposeTensor(reshaped, dimension: 1, withDimension: 2, name: "\(name).heads")
    }

    /// Reshape [B, heads, T, headDim] → [B, T, heads*headDim] after attention.
    static func fromHeads(
        _ graph: MPSGraph, input x: MPSGraphTensor, totalDim: Int, name: String
    ) -> MPSGraphTensor {
        let transposed = graph.transposeTensor(x, dimension: 1, withDimension: 2, name: nil)
        return graph.reshape(transposed, shape: [1, -1, totalDim as NSNumber], name: "\(name).flat")
    }

    // MARK: - Scaled Dot-Product Attention with Causal Mask

    /// SDPA with causal mask: softmax((Q @ K^T / sqrt(d)) + causal_mask) @ V
    ///
    /// Causal mask ensures position i can only attend to positions ≤ i.
    /// Implemented via coordinate comparison for dynamic sequence length.
    static func causalScaledDotProductAttention(
        _ graph: MPSGraph, query q: MPSGraphTensor, key k: MPSGraphTensor,
        value v: MPSGraphTensor, headDim: Int, name: String
    ) -> MPSGraphTensor {
        let scale = graph.constant(Double(1.0 / Float(headDim).squareRoot()), dataType: .float16)

        // Q @ K^T
        let kt = graph.transposeTensor(k, dimension: 2, withDimension: 3, name: nil)
        let scores = graph.multiplication(
            graph.matrixMultiplication(primary: q, secondary: kt, name: "\(name).qk"),
            scale, name: "\(name).scaled")

        // Causal mask: -inf where col > row (future positions)
        let scoresShape = graph.shapeOf(scores, name: "\(name).shape")
        let rows = graph.coordinate(alongAxis:-2, withShapeTensor: scoresShape, name: "\(name).rows")
        let cols = graph.coordinate(alongAxis:-1, withShapeTensor: scoresShape, name: "\(name).cols")
        let isFuture = graph.greaterThan(cols, rows, name: "\(name).future")

        let negInf = graph.constant(-1e4, dataType: .float16)
        let zero = graph.constant(0.0, dataType: .float16)
        let mask = graph.select(
            predicate: isFuture, trueTensor: negInf,
            falseTensor: zero, name: "\(name).mask")
        let masked = graph.addition(scores, mask, name: "\(name).masked")

        let weights = graph.softMax(with: masked, axis: -1, name: "\(name).sm")
        return graph.matrixMultiplication(primary: weights, secondary: v, name: "\(name).attn")
    }

    // MARK: - RoPE

    /// Apply Rotary Position Embedding.
    ///
    /// Uses the non-interleaved (contiguous) layout:
    ///   x_even = x[..., :hd/2], x_odd = x[..., hd/2:]
    ///   rotated_even = x_even * cos(θ) - x_odd * sin(θ)
    ///   rotated_odd  = x_even * sin(θ) + x_odd * cos(θ)
    ///
    /// Position indices are derived dynamically from the sequence axis.
    static func applyRoPE(
        _ graph: MPSGraph, input x: MPSGraphTensor,
        frequencies: MPSGraphTensor, heads: Int, headDim: Int, name: String
    ) -> MPSGraphTensor {
        let halfDim = headDim / 2

        // Position indices from sequence axis (dim 2): [1, heads, T, hd] → T values
        let xShape = graph.shapeOf(x, name: "\(name).xsh")
        let positions = graph.cast(
            graph.coordinate(alongAxis:2, withShapeTensor: xShape, name: "\(name).pos"),
            to: .float32, name: "\(name).pos32")
        // Extract a single row of positions: [1, 1, T, 1]
        let posSlice = graph.sliceTensor(positions, dimension: 1, start: 0, length: 1, name: "\(name).ps")
        let posFlat = graph.sliceTensor(posSlice, dimension: 3, start: 0, length: 1, name: "\(name).pf")

        // angles = positions * frequencies: [1,1,T,1] * [1,hd/2] → [1,1,T,hd/2]
        let angles = graph.multiplication(posFlat, frequencies, name: "\(name).ang")
        let cosVal = graph.cast(graph.cos(with: angles, name: "\(name).cos"), to: .float16, name: "\(name).cos16")
        let sinVal = graph.cast(graph.sin(with: angles, name: "\(name).sin"), to: .float16, name: "\(name).sin16")

        // Split even/odd halves
        let xEven = graph.sliceTensor(x, dimension: 3, start: 0, length: halfDim, name: "\(name).xe")
        let xOdd = graph.sliceTensor(x, dimension: 3, start: halfDim, length: halfDim, name: "\(name).xo")

        // Rotation
        let rotEven = graph.subtraction(
            graph.multiplication(xEven, cosVal, name: "\(name).ec"),
            graph.multiplication(xOdd, sinVal, name: "\(name).os"),
            name: "\(name).re")
        let rotOdd = graph.addition(
            graph.multiplication(xEven, sinVal, name: "\(name).es"),
            graph.multiplication(xOdd, cosVal, name: "\(name).oc"),
            name: "\(name).ro")

        return graph.concatTensors([rotEven, rotOdd], dimension: 3, name: "\(name).rope")
    }

    /// Build RoPE inverse frequency table: 1 / (theta ^ (2i/d)) for i in 0..<d/2.
    static func buildRoPEFrequencies(
        _ graph: MPSGraph, headDim: Int, theta: Float
    ) -> MPSGraphTensor {
        let halfDim = headDim / 2
        var freqs = [Float](repeating: 0, count: halfDim)
        for i in 0..<halfDim {
            freqs[i] = 1.0 / pow(theta, Float(2 * i) / Float(headDim))
        }
        return graph.variable(
            with: freqs.withUnsafeBytes { Data($0) },
            shape: [1, halfDim as NSNumber],
            dataType: .float32, name: "rope_freqs")
    }

    // MARK: - SiLU Gated MLP

    /// SiLU(gate) * up — gated activation for MLP.
    static func siluGate(
        _ graph: MPSGraph, gate: MPSGraphTensor, up: MPSGraphTensor, name: String
    ) -> MPSGraphTensor {
        let silu = graph.multiplication(
            gate, graph.sigmoid(with: gate, name: "\(name).sig"), name: "\(name).silu")
        return graph.multiplication(silu, up, name: "\(name).act")
    }

    // MARK: - GQA Head Repeat

    /// Tile K/V heads for GQA: [1, KVH, T, hd] → [1, H, T, hd]
    static func repeatKVHeads(
        _ graph: MPSGraph, input: MPSGraphTensor, repeatFactor: Int, name: String
    ) -> MPSGraphTensor {
        guard repeatFactor > 1 else { return input }
        return graph.tileTensor(
            input, withMultiplier: [1, repeatFactor, 1, 1] as [NSNumber], name: "\(name).rep")
    }
}
