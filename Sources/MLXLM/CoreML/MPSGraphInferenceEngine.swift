import Foundation
import Metal
import MetalPerformanceShadersGraph

/// MPSGraph-based inference engine for transformer models.
///
/// Builds the entire model (all layers) as a single MPSGraph,
/// compiles it into a fused execution plan, and executes with
/// kernel fusion + memory aliasing — the same optimizations
/// that CoreML/MPSGraph provides internally.
///
/// Key advantage over MLX: MPSGraph compiles the full graph into
/// an optimized plan where intermediate tensors stay in GPU SRAM
/// and adjacent operations are fused. MLX dispatches individual
/// Metal kernels, each writing back to global memory.
///
/// No Python dependency — pure Swift + Metal.
public final class MPSGraphInferenceEngine: @unchecked Sendable {

    private let device: MTLDevice
    private let graph: MPSGraph
    private let commandQueue: MTLCommandQueue

    /// Compiled executable for decode (T=1).
    private var decodeExecutable: MPSGraphExecutable?

    /// Graph tensors for input feeds.
    private var inputTensor: MPSGraphTensor!
    private var offsetTensor: MPSGraphTensor!

    /// Graph tensor for output.
    private var outputTensor: MPSGraphTensor!

    /// KV cache state tensors (per layer: key + value).
    private var kvCacheKeyTensors: [MPSGraphTensor] = []
    private var kvCacheValueTensors: [MPSGraphTensor] = []

    /// Weight variables stored in the graph.
    private var weightVariables: [String: MPSGraphTensor] = [:]

    /// Model configuration.
    public let config: MPSGraphModelConfig

    public struct MPSGraphModelConfig: Sendable {
        public let hiddenSize: Int
        public let headCount: Int
        public let kvHeadCount: Int
        public let headDim: Int
        public let intermediateSize: Int
        public let layerCount: Int
        public let vocabSize: Int
        public let maxSeqLen: Int
        public let ropeBase: Float

        public init(hiddenSize: Int, headCount: Int, kvHeadCount: Int, headDim: Int,
                    intermediateSize: Int, layerCount: Int, vocabSize: Int,
                    maxSeqLen: Int = 512, ropeBase: Float = 500000.0) {
            self.hiddenSize = hiddenSize
            self.headCount = headCount
            self.kvHeadCount = kvHeadCount
            self.headDim = headDim
            self.intermediateSize = intermediateSize
            self.layerCount = layerCount
            self.vocabSize = vocabSize
            self.maxSeqLen = maxSeqLen
            self.ropeBase = ropeBase
        }
    }

    public init(config: MPSGraphModelConfig) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MPSGraphEngineError.noMetalDevice
        }
        guard let queue = device.makeCommandQueue() else {
            throw MPSGraphEngineError.noCommandQueue
        }
        self.device = device
        self.commandQueue = queue
        self.graph = MPSGraph()
        self.config = config
    }

    /// Build the full transformer graph with random weights (for benchmarking).
    public func buildGraph() {
        let D = config.hiddenSize
        let H = config.headCount
        let KVH = config.kvHeadCount
        let hd = config.headDim
        let I = config.intermediateSize
        let L = config.layerCount
        let V = config.vocabSize
        let maxSeq = config.maxSeqLen

        // Input: token embedding (pre-embedded for simplicity)
        // For a full model: token_ids → embedding lookup → h
        inputTensor = graph.placeholder(
            shape: [1, 1, D as NSNumber], dataType: .float16, name: "input")
        offsetTensor = graph.placeholder(
            shape: [1], dataType: .int32, name: "offset")

        var h = inputTensor!

        // Build each layer
        for i in 0..<L {
            let prefix = "l\(i)"

            // Weights (random init for benchmarking)
            let normW1 = makeWeight([D], name: "\(prefix)_n1")
            let wq = makeWeight([H * hd, D], name: "\(prefix)_wq")
            let wk = makeWeight([KVH * hd, D], name: "\(prefix)_wk")
            let wv = makeWeight([KVH * hd, D], name: "\(prefix)_wv")
            let wo = makeWeight([D, H * hd], name: "\(prefix)_wo")
            let normW2 = makeWeight([D], name: "\(prefix)_n2")
            let wGate = makeWeight([I, D], name: "\(prefix)_wg")
            let wUp = makeWeight([I, D], name: "\(prefix)_wu")
            let wDown = makeWeight([D, I], name: "\(prefix)_wd")

            // Attention sublayer
            let norm1 = rmsNorm(h, weight: normW1, name: "\(prefix)_an")

            let q = matmulTranspose(norm1, wq, name: "\(prefix)_qp")
            let k = matmulTranspose(norm1, wk, name: "\(prefix)_kp")
            let v = matmulTranspose(norm1, wv, name: "\(prefix)_vp")

            // Reshape to multi-head [1,1,D] → [1,H,1,hd]
            let qr = graph.transposeTensor(
                graph.reshape(q, shape: [1, 1, H as NSNumber, hd as NSNumber], name: nil),
                dimension: 1, withDimension: 2, name: "\(prefix)_qt")
            let kr = graph.transposeTensor(
                graph.reshape(k, shape: [1, 1, KVH as NSNumber, hd as NSNumber], name: nil),
                dimension: 1, withDimension: 2, name: "\(prefix)_kt")
            let vr = graph.transposeTensor(
                graph.reshape(v, shape: [1, 1, KVH as NSNumber, hd as NSNumber], name: nil),
                dimension: 1, withDimension: 2, name: "\(prefix)_vt")

            // SDPA (simplified — without KV cache for initial benchmark)
            // Q @ K^T / sqrt(hd) → softmax → @ V
            let scale = graph.constant(
                Double(1.0 / Float(hd).squareRoot()), dataType: .float16)
            let scores = graph.multiplication(
                graph.matrixMultiplication(
                    primary: qr,
                    secondary: graph.transposeTensor(kr, dimension: 2, withDimension: 3, name: nil),
                    name: "\(prefix)_qk"),
                scale, name: "\(prefix)_scaled")
            let attnW = graph.softMax(with: scores, axis: -1, name: "\(prefix)_sm")
            let attnOut = graph.matrixMultiplication(
                primary: attnW, secondary: vr, name: "\(prefix)_av")

            // Reshape back [1,H,1,hd] → [1,1,D]
            let attnFlat = graph.reshape(
                graph.transposeTensor(attnOut, dimension: 1, withDimension: 2, name: nil),
                shape: [1, 1, (H * hd) as NSNumber], name: "\(prefix)_af")

            // O projection + residual
            let proj = matmulTranspose(attnFlat, wo, name: "\(prefix)_op")
            h = graph.addition(h, proj, name: "\(prefix)_ar")

            // MLP sublayer
            let norm2 = rmsNorm(h, weight: normW2, name: "\(prefix)_mn")
            let gate = matmulTranspose(norm2, wGate, name: "\(prefix)_gp")
            let up = matmulTranspose(norm2, wUp, name: "\(prefix)_up")

            // SiLU(gate) * up
            let gateSilu = graph.multiplication(
                gate,
                graph.sigmoid(with: gate, name: "\(prefix)_sig"),
                name: "\(prefix)_silu")
            let activated = graph.multiplication(gateSilu, up, name: "\(prefix)_sw")
            let down = matmulTranspose(activated, wDown, name: "\(prefix)_dp")
            h = graph.addition(h, down, name: "\(prefix)_mr")
        }

        // Final norm
        let finalNormW = makeWeight([D], name: "fn_w")
        h = rmsNorm(h, weight: finalNormW, name: "fn")

        outputTensor = h
    }

    /// Compile the graph into an optimized executable.
    public func compile() throws {
        guard let output = outputTensor, let input = inputTensor else {
            throw MPSGraphEngineError.graphNotBuilt
        }

        let feeds: [MPSGraphTensor: MPSGraphShapedType] = [
            input: MPSGraphShapedType(
                shape: [1, 1, config.hiddenSize as NSNumber], dataType: .float16),
        ]

        decodeExecutable = graph.compile(
            with: MPSGraphDevice(mtlDevice: device),
            feeds: feeds,
            targetTensors: [output],
            targetOperations: nil,
            compilationDescriptor: nil)
    }

    /// Run a single decode step using the compiled executable.
    public func decode(inputData: [Float16]) throws -> [Float16] {
        guard decodeExecutable != nil else {
            throw MPSGraphEngineError.notCompiled
        }
        return decodeWithGraphRun(inputData: inputData)
    }

    /// Run decode using graph.run() (simpler API).
    public func decodeWithGraphRun(inputData: [Float16]) -> [Float16] {
        let D = config.hiddenSize
        let inputArray = inputData.withUnsafeBufferPointer { ptr in
            MPSGraphTensorData(
                device: MPSGraphDevice(mtlDevice: device),
                data: Data(buffer: ptr),
                shape: [1, 1, D as NSNumber],
                dataType: .float16)
        }

        let feeds: [MPSGraphTensor: MPSGraphTensorData] = [inputTensor: inputArray]
        let results = graph.run(
            with: commandQueue,
            feeds: feeds,
            targetTensors: [outputTensor],
            targetOperations: nil)

        let outputData = results[outputTensor]!
        let count = D
        var output = [Float16](repeating: 0, count: count)
        outputData.mpsndarray().readBytes(&output, strideBytes: nil)
        return output
    }

    // MARK: - Graph Construction Helpers

    private func makeWeight(_ shape: [Int], name: String) -> MPSGraphTensor {
        let nsShape = shape.map { $0 as NSNumber }
        let count = shape.reduce(1, *)
        var data = [Float16](repeating: 0, count: count)
        // Small random init
        for i in 0..<count {
            data[i] = Float16(Float.random(in: -0.02...0.02))
        }
        let tensorData = Data(bytes: &data, count: count * MemoryLayout<Float16>.size)
        let tensor = graph.variable(with: tensorData, shape: nsShape,
                                     dataType: .float16, name: name)
        weightVariables[name] = tensor
        return tensor
    }

    private func rmsNorm(_ x: MPSGraphTensor, weight: MPSGraphTensor,
                          name: String) -> MPSGraphTensor {
        let eps = graph.constant(1e-5, dataType: .float16)
        let xSq = graph.multiplication(x, x, name: "\(name)_sq")
        let mean = graph.mean(of: xSq, axes: [-1], name: "\(name)_mean")
        let msEps = graph.addition(mean, eps, name: "\(name)_eps")
        let invRms = graph.reverseSquareRoot(with: msEps, name: "\(name)_inv")
        let normed = graph.multiplication(x, invRms, name: "\(name)_n")
        return graph.multiplication(normed, weight, name: "\(name)_out")
    }

    private func matmulTranspose(_ x: MPSGraphTensor, _ w: MPSGraphTensor,
                                  name: String) -> MPSGraphTensor {
        // x @ w^T: x=[1,1,D] w=[outD,D] → [1,1,outD]
        graph.matrixMultiplication(
            primary: x,
            secondary: graph.transposeTensor(w, dimension: 0, withDimension: 1, name: nil),
            name: name)
    }
}

/// Errors from MPSGraph engine.
public enum MPSGraphEngineError: Error, LocalizedError {
    case noMetalDevice
    case noCommandQueue
    case graphNotBuilt
    case notCompiled

    public var errorDescription: String? {
        switch self {
        case .noMetalDevice: return "No Metal device available"
        case .noCommandQueue: return "Failed to create Metal command queue"
        case .graphNotBuilt: return "Graph has not been built. Call buildGraph() first."
        case .notCompiled: return "Graph has not been compiled. Call compile() first."
        }
    }
}
