import Foundation
import Metal
import MetalPerformanceShadersGraph
@preconcurrency import MLX
import SwiftLM

/// Compiled MPSGraph model ready for inference.
///
/// Causal mask and RoPE tables are generated at call time from model config,
/// avoiding MPSGraph MLIR issues with dynamic coordinate ops.
public struct MPSGraphInferenceModel: @unchecked Sendable {

    let graph: MPSGraph
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let inputPlaceholder: MPSGraphTensor
    let maskPlaceholder: MPSGraphTensor
    let ropeCosPh: MPSGraphTensor?
    let ropeSinPh: MPSGraphTensor?
    let ropeHeadDim: Int
    let ropeTheta: Float
    let ropeScaling: RoPEScaling?
    let outputTensor: MPSGraphTensor

    public let metadata: InferenceMetadata

    /// Forward pass: token IDs → logits.
    public func forward(_ tokenIDs: [Int32]) -> MLXArray {
        let T = tokenIDs.count
        let mpsDevice = MPSGraphDevice(mtlDevice: device)

        let inputData = tokenIDs.withUnsafeBytes { ptr in
            MPSGraphTensorData(device: mpsDevice, data: Data(ptr),
                                shape: [1, T as NSNumber], dataType: .int32)
        }

        // Causal mask [1, 1, T, T]
        let maskBytes = T == 1
            ? [Float16](repeating: 0, count: 1).withUnsafeBytes { Data($0) }
            : MPSGraphOps.buildCausalMask(seqLen: T)
        let maskData = MPSGraphTensorData(
            device: mpsDevice, data: maskBytes,
            shape: [1, 1, T as NSNumber, T as NSNumber], dataType: .float16)

        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [
            inputPlaceholder: inputData,
            maskPlaceholder: maskData,
        ]

        // RoPE cos/sin tables [1, 1, T, hd/2] — using model's theta and scaling
        if let cosPh = ropeCosPh, let sinPh = ropeSinPh, ropeHeadDim > 0 {
            let halfDim = ropeHeadDim / 2
            let (cosData, sinData) = MPSGraphOps.buildRoPETables(
                seqLen: T, headDim: ropeHeadDim, theta: ropeTheta, scaling: ropeScaling)
            feeds[cosPh] = MPSGraphTensorData(
                device: mpsDevice, data: cosData,
                shape: [1, 1, T as NSNumber, halfDim as NSNumber], dataType: .float16)
            feeds[sinPh] = MPSGraphTensorData(
                device: mpsDevice, data: sinData,
                shape: [1, 1, T as NSNumber, halfDim as NSNumber], dataType: .float16)
        }

        let results = graph.run(
            with: commandQueue, feeds: feeds,
            targetTensors: [outputTensor], targetOperations: nil)

        let result = results[outputTensor]!
        let shape = result.shape.map { $0.intValue }
        let count = shape.reduce(1, *)
        var buffer = [Float16](repeating: 0, count: count)
        result.mpsndarray().readBytes(&buffer, strideBytes: nil)
        return MLXArray(buffer, shape)
    }
}
