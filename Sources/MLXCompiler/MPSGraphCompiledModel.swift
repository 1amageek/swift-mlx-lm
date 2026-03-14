import Foundation
import Metal
import MetalPerformanceShadersGraph
@preconcurrency import MLX

/// Compiled MPSGraph model ready for inference.
///
/// Contains the full transformer graph (all layers) as a single MPSGraph
/// with kernel fusion and memory aliasing. Each `forward()` call executes
/// the entire model in one dispatch.
public struct MPSGraphCompiledModel: @unchecked Sendable {

    let graph: MPSGraph
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let inputPlaceholder: MPSGraphTensor
    let outputTensor: MPSGraphTensor

    /// Model metadata (cache descriptors, tied head info).
    public let metadata: InferenceMetadata

    /// Forward pass: token IDs → logits.
    ///
    /// Processes the full token sequence through all layers.
    /// - Parameter tokenIDs: Int32 token IDs.
    /// - Returns: Logits as MLXArray [1, T, vocab].
    public func forward(_ tokenIDs: [Int32]) -> MLXArray {
        let T = tokenIDs.count
        let inputBytes = tokenIDs.withUnsafeBytes { Data($0) }
        let inputData = MPSGraphTensorData(
            device: MPSGraphDevice(mtlDevice: device), data: inputBytes,
            shape: [1, T as NSNumber], dataType: .int32)

        let results = graph.run(
            with: commandQueue,
            feeds: [inputPlaceholder: inputData],
            targetTensors: [outputTensor],
            targetOperations: nil)

        let result = results[outputTensor]!
        let shape = result.shape.map { $0.intValue }
        let count = shape.reduce(1, *)
        var buffer = [Float16](repeating: 0, count: count)
        result.mpsndarray().readBytes(&buffer, strideBytes: nil)
        return MLXArray(buffer, shape)
    }
}
