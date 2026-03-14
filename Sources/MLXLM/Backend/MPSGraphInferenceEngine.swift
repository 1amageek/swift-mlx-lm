import Foundation
import Metal
import MetalPerformanceShadersGraph

/// Transformer inference engine built on MPSGraph.
///
/// Compiles the full model (all layers) into a single MPSGraph execution plan
/// with kernel fusion and memory aliasing. This provides the same graph-level
/// optimizations as CoreML's MPSGraph backend, without any Python dependency.
///
/// Usage:
/// ```swift
/// let engine = try MPSGraphInferenceEngine(
///     config: .init(hiddenSize: 896, headCount: 14, kvHeadCount: 2,
///                   headDim: 64, intermediateSize: 4864, layerCount: 24))
/// let output = engine.run(input: inputData)
/// ```
public final class MPSGraphInferenceEngine: @unchecked Sendable {

    /// Model configuration.
    public struct Config: Sendable {
        public let hiddenSize: Int
        public let headCount: Int
        public let kvHeadCount: Int
        public let headDim: Int
        public let intermediateSize: Int
        public let layerCount: Int
        public let maxSeqLen: Int

        public init(hiddenSize: Int, headCount: Int, kvHeadCount: Int, headDim: Int,
                    intermediateSize: Int, layerCount: Int, maxSeqLen: Int = 512) {
            self.hiddenSize = hiddenSize
            self.headCount = headCount
            self.kvHeadCount = kvHeadCount
            self.headDim = headDim
            self.intermediateSize = intermediateSize
            self.layerCount = layerCount
            self.maxSeqLen = maxSeqLen
        }
    }

    /// Errors from the inference engine.
    public enum Error: Swift.Error, LocalizedError {
        case noMetalDevice
        case noCommandQueue

        public var errorDescription: String? {
            switch self {
            case .noMetalDevice: return "No Metal device available"
            case .noCommandQueue: return "Failed to create Metal command queue"
            }
        }
    }

    public let config: Config

    private let device: MTLDevice
    private let graph: MPSGraph
    private let commandQueue: MTLCommandQueue
    private let inputTensor: MPSGraphTensor
    private let outputTensor: MPSGraphTensor
    private let executable: MPSGraphExecutable

    /// Build, compile, and prepare for inference.
    public init(config: Config) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Error.noMetalDevice
        }
        guard let queue = device.makeCommandQueue() else {
            throw Error.noCommandQueue
        }
        self.config = config
        self.device = device
        self.commandQueue = queue
        self.graph = MPSGraph()

        // Build graph
        let (input, output) = Self.buildTransformerGraph(graph: graph, config: config)
        self.inputTensor = input
        self.outputTensor = output

        // Compile into fused execution plan
        let feeds: [MPSGraphTensor: MPSGraphShapedType] = [
            input: MPSGraphShapedType(
                shape: [1, 1, config.hiddenSize as NSNumber], dataType: .float16)
        ]
        self.executable = graph.compile(
            with: MPSGraphDevice(mtlDevice: device),
            feeds: feeds,
            targetTensors: [output],
            targetOperations: nil,
            compilationDescriptor: nil)
    }

    /// Run a single decode step.
    ///
    /// - Parameter input: Hidden state tensor data `[1, 1, hiddenSize]` in Float16.
    /// - Returns: Output tensor data `[1, 1, hiddenSize]` in Float16.
    public func run(input: [Float16]) -> [Float16] {
        let tensorData = input.withUnsafeBufferPointer { ptr in
            MPSGraphTensorData(
                device: MPSGraphDevice(mtlDevice: device),
                data: Data(buffer: ptr),
                shape: [1, 1, config.hiddenSize as NSNumber],
                dataType: .float16)
        }

        let results = graph.run(
            with: commandQueue,
            feeds: [inputTensor: tensorData],
            targetTensors: [outputTensor],
            targetOperations: nil)

        let outputData = results[outputTensor]!
        var output = [Float16](repeating: 0, count: config.hiddenSize)
        outputData.mpsndarray().readBytes(&output, strideBytes: nil)
        return output
    }

    // MARK: - Graph Construction

    /// Build a full transformer graph (all layers).
    ///
    /// Returns the (input, output) placeholder tensors for feeding and reading.
    private static func buildTransformerGraph(
        graph: MPSGraph, config: Config
    ) -> (input: MPSGraphTensor, output: MPSGraphTensor) {
        let D = config.hiddenSize
        let H = config.headCount
        let KVH = config.kvHeadCount
        let hd = config.headDim
        let I = config.intermediateSize

        let input = graph.placeholder(
            shape: [1, 1, D as NSNumber], dataType: .float16, name: "input")

        var h = input

        for i in 0..<config.layerCount {
            h = buildTransformerLayer(
                graph: graph, input: h, layerIndex: i,
                D: D, H: H, KVH: KVH, hd: hd, I: I)
        }

        // Final RMSNorm
        let finalNormW = randomWeight(graph: graph, shape: [D], name: "final_norm")
        h = rmsNorm(graph: graph, x: h, weight: finalNormW, name: "final")

        return (input, h)
    }

    /// Build a single transformer layer (attention + MLP).
    private static func buildTransformerLayer(
        graph: MPSGraph, input: MPSGraphTensor, layerIndex i: Int,
        D: Int, H: Int, KVH: Int, hd: Int, I: Int
    ) -> MPSGraphTensor {
        var h = input
        let p = "l\(i)"

        // --- Attention sublayer ---
        let n1 = randomWeight(graph: graph, shape: [D], name: "\(p).an.w")
        let norm1 = rmsNorm(graph: graph, x: h, weight: n1, name: "\(p).an")

        let wq = randomWeight(graph: graph, shape: [H * hd, D], name: "\(p).q.w")
        let wk = randomWeight(graph: graph, shape: [KVH * hd, D], name: "\(p).k.w")
        let wv = randomWeight(graph: graph, shape: [KVH * hd, D], name: "\(p).v.w")
        let wo = randomWeight(graph: graph, shape: [D, H * hd], name: "\(p).o.w")

        let q = linearProjection(graph: graph, x: norm1, weight: wq, name: "\(p).q")
        let k = linearProjection(graph: graph, x: norm1, weight: wk, name: "\(p).k")
        let v = linearProjection(graph: graph, x: norm1, weight: wv, name: "\(p).v")

        // Reshape to multi-head: [1,1,dim] → [1,heads,1,hd]
        let qH = reshapeToHeads(graph: graph, x: q, heads: H, headDim: hd, name: "\(p).q")
        let kH = reshapeToHeads(graph: graph, x: k, heads: KVH, headDim: hd, name: "\(p).k")
        let vH = reshapeToHeads(graph: graph, x: v, heads: KVH, headDim: hd, name: "\(p).v")

        // Scaled dot-product attention
        let attnOut = scaledDotProductAttention(
            graph: graph, query: qH, key: kH, value: vH, headDim: hd, name: p)

        // Reshape back: [1,H,1,hd] → [1,1,D]
        let attnFlat = graph.reshape(
            graph.transposeTensor(attnOut, dimension: 1, withDimension: 2, name: nil),
            shape: [1, 1, (H * hd) as NSNumber], name: "\(p).attn.flat")

        let proj = linearProjection(graph: graph, x: attnFlat, weight: wo, name: "\(p).o")
        h = graph.addition(h, proj, name: "\(p).attn.res")

        // --- MLP sublayer ---
        let n2 = randomWeight(graph: graph, shape: [D], name: "\(p).mn.w")
        let norm2 = rmsNorm(graph: graph, x: h, weight: n2, name: "\(p).mn")

        let wGate = randomWeight(graph: graph, shape: [I, D], name: "\(p).gate.w")
        let wUp = randomWeight(graph: graph, shape: [I, D], name: "\(p).up.w")
        let wDown = randomWeight(graph: graph, shape: [D, I], name: "\(p).down.w")

        let gate = linearProjection(graph: graph, x: norm2, weight: wGate, name: "\(p).gate")
        let up = linearProjection(graph: graph, x: norm2, weight: wUp, name: "\(p).up")

        // SiLU(gate) * up
        let silu = graph.multiplication(
            gate, graph.sigmoid(with: gate, name: "\(p).sig"), name: "\(p).silu")
        let activated = graph.multiplication(silu, up, name: "\(p).swiglu")
        let down = linearProjection(graph: graph, x: activated, weight: wDown, name: "\(p).down")
        h = graph.addition(h, down, name: "\(p).mlp.res")

        return h
    }

    // MARK: - Op Primitives

    /// x @ weight^T
    private static func linearProjection(
        graph: MPSGraph, x: MPSGraphTensor, weight: MPSGraphTensor, name: String
    ) -> MPSGraphTensor {
        graph.matrixMultiplication(
            primary: x,
            secondary: graph.transposeTensor(weight, dimension: 0, withDimension: 1, name: nil),
            name: name)
    }

    /// RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
    private static func rmsNorm(
        graph: MPSGraph, x: MPSGraphTensor, weight: MPSGraphTensor, name: String
    ) -> MPSGraphTensor {
        let eps = graph.constant(1e-5, dataType: .float16)
        let sq = graph.multiplication(x, x, name: "\(name).sq")
        let mean = graph.mean(of: sq, axes: [-1], name: "\(name).mean")
        let inv = graph.reverseSquareRoot(
            with: graph.addition(mean, eps, name: "\(name).eps"),
            name: "\(name).inv")
        let normed = graph.multiplication(x, inv, name: "\(name).normed")
        return graph.multiplication(normed, weight, name: "\(name).out")
    }

    /// Reshape [1,1,dim] → [1,heads,1,headDim] for multi-head attention.
    private static func reshapeToHeads(
        graph: MPSGraph, x: MPSGraphTensor, heads: Int, headDim: Int, name: String
    ) -> MPSGraphTensor {
        graph.transposeTensor(
            graph.reshape(x, shape: [1, 1, heads as NSNumber, headDim as NSNumber], name: nil),
            dimension: 1, withDimension: 2, name: "\(name).heads")
    }

    /// Scaled dot-product attention: softmax(Q @ K^T / sqrt(d)) @ V
    private static func scaledDotProductAttention(
        graph: MPSGraph, query: MPSGraphTensor, key: MPSGraphTensor,
        value: MPSGraphTensor, headDim: Int, name: String
    ) -> MPSGraphTensor {
        let scale = graph.constant(
            Double(1.0 / Float(headDim).squareRoot()), dataType: .float16)
        let scores = graph.multiplication(
            graph.matrixMultiplication(
                primary: query,
                secondary: graph.transposeTensor(key, dimension: 2, withDimension: 3, name: nil),
                name: "\(name).qk"),
            scale, name: "\(name).scaled")
        let weights = graph.softMax(with: scores, axis: -1, name: "\(name).sm")
        return graph.matrixMultiplication(
            primary: weights, secondary: value, name: "\(name).attn")
    }

    /// Create a random Float16 weight variable (for benchmarking).
    private static func randomWeight(
        graph: MPSGraph, shape: [Int], name: String
    ) -> MPSGraphTensor {
        let count = shape.reduce(1, *)
        var data = [Float16](repeating: 0, count: count)
        for i in 0..<count {
            data[i] = Float16(Float.random(in: -0.02...0.02))
        }
        return graph.variable(
            with: Data(bytes: &data, count: count * MemoryLayout<Float16>.size),
            shape: shape.map { $0 as NSNumber },
            dataType: .float16, name: name)
    }
}
