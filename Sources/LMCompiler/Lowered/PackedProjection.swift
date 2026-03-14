@preconcurrency import MLX

/// Packed projection that combines multiple projections sharing the same input
/// into a single matmul dispatch + split.
///
/// This reduces Metal kernel dispatches by packing Q/K/V (or Gate/Up) weights
/// along axis=0 and performing one matmul (or quantizedMM) instead of N separate
/// dispatches. The result is split along the last axis to recover individual outputs.
///
/// Packing is only possible when all projections use the same kernel variant:
/// - All `.dense` → concat weights, single `matmul`
/// - All `.affineQuantized` with same bits/groupSize → concat packed weights/scales/biases
/// - All `.dequantizeMatmul` with same bits/groupSize → concat packed weights/scales/biases
/// - Mixed variants → packing is not possible, fallback to individual projections
public struct PackedProjection: @unchecked Sendable {

    /// The kernel used for the packed matmul.
    public let kernel: ProjectionKernel

    /// Packed bias (concatenated biases), nil if no projections have bias.
    public let packedBias: MLXArray?

    /// Cumulative split indices along the output dimension.
    /// For Q(128)/K(64)/V(64), splitIndices = [128, 192].
    /// `MLX.split(result, indices: splitIndices, axis: -1)` produces 3 tensors.
    public let splitIndices: [Int]

    /// Number of packed projections.
    public var count: Int { splitIndices.count + 1 }

    /// Apply the packed projection and split into individual outputs.
    ///
    /// Dispatches a single matmul (or quantizedMM or dequantize+matmul) and
    /// splits the result along the last axis.
    public func apply(_ x: MLXArray) -> [MLXArray] {
        var result: MLXArray
        switch kernel {
        case .dense(let w):
            result = matmul(x, w.T)
        case .affineQuantized(let q):
            result = quantizedMM(
                x, q.packedWeight,
                scales: q.scales, biases: q.zeroBiases,
                transpose: true, groupSize: q.groupSize, bits: q.bits
            )
        case .dequantizeMatmul(let q):
            let w = dequantized(
                q.packedWeight,
                scales: q.scales, biases: q.zeroBiases,
                groupSize: q.groupSize, bits: q.bits
            )
            result = matmul(x, w.T)
        }

        if let packedBias {
            result = result + packedBias
        }

        return split(result, indices: splitIndices, axis: -1)
    }

    /// Pack multiple projections into a single packed projection.
    ///
    /// Returns `nil` if packing is not possible (mixed kernel variants or
    /// incompatible quantization parameters).
    ///
    /// - Parameter projections: Projections to pack. Must share the same input dimension.
    public static func pack(_ projections: [LoweredProjection]) -> PackedProjection? {
        guard projections.count >= 2 else { return nil }

        // Compute split indices from individual output dimensions
        var splitIndices: [Int] = []
        var cumulative = 0
        for proj in projections.dropLast() {
            cumulative += outputDim(of: proj.kernel)
            splitIndices.append(cumulative)
        }

        // Pack biases if any projection has one
        let hasBias = projections.contains { $0.bias != nil }
        let packedBias: MLXArray?
        if hasBias {
            let biases = projections.map { proj -> MLXArray in
                if let b = proj.bias { return b }
                // Zero bias matching output dimension
                return MLXArray.zeros([outputDim(of: proj.kernel)])
            }
            packedBias = concatenated(biases, axis: 0)
        } else {
            packedBias = nil
        }

        // Check kernel compatibility and pack
        switch projections[0].kernel {
        case .dense:
            // All must be dense
            var weights: [MLXArray] = []
            for proj in projections {
                guard case .dense(let w) = proj.kernel else { return nil }
                weights.append(w)
            }
            // Concat along axis=0 (output dimension)
            let packed = concatenated(weights, axis: 0)
            return PackedProjection(
                kernel: .dense(weight: packed),
                packedBias: packedBias,
                splitIndices: splitIndices
            )

        case .affineQuantized(let first):
            // All must be affineQuantized with same bits/groupSize
            var packedWeights: [MLXArray] = []
            var scales: [MLXArray] = []
            var zeroBiases: [MLXArray] = []
            var totalOutFeatures = 0

            for proj in projections {
                guard case .affineQuantized(let q) = proj.kernel,
                      q.bits == first.bits,
                      q.groupSize == first.groupSize
                else { return nil }
                packedWeights.append(q.packedWeight)
                scales.append(q.scales)
                zeroBiases.append(q.zeroBiases)
                totalOutFeatures += q.logicalShape[0]
            }

            let qt = AffineQuantizedTensor(
                logicalShape: [totalOutFeatures, first.logicalShape[1]],
                packedWeight: concatenated(packedWeights, axis: 0),
                scales: concatenated(scales, axis: 0),
                zeroBiases: concatenated(zeroBiases, axis: 0),
                groupSize: first.groupSize,
                bits: first.bits,
                origin: first.origin
            )
            return PackedProjection(
                kernel: .affineQuantized(qt),
                packedBias: packedBias,
                splitIndices: splitIndices
            )

        case .dequantizeMatmul(let first):
            // All must be dequantizeMatmul with same bits/groupSize
            var packedWeights: [MLXArray] = []
            var scales: [MLXArray] = []
            var zeroBiases: [MLXArray] = []
            var totalOutFeatures = 0

            for proj in projections {
                guard case .dequantizeMatmul(let q) = proj.kernel,
                      q.bits == first.bits,
                      q.groupSize == first.groupSize
                else { return nil }
                packedWeights.append(q.packedWeight)
                scales.append(q.scales)
                zeroBiases.append(q.zeroBiases)
                totalOutFeatures += q.logicalShape[0]
            }

            let qt = AffineQuantizedTensor(
                logicalShape: [totalOutFeatures, first.logicalShape[1]],
                packedWeight: concatenated(packedWeights, axis: 0),
                scales: concatenated(scales, axis: 0),
                zeroBiases: concatenated(zeroBiases, axis: 0),
                groupSize: first.groupSize,
                bits: first.bits,
                origin: first.origin
            )
            return PackedProjection(
                kernel: .dequantizeMatmul(qt),
                packedBias: packedBias,
                splitIndices: splitIndices
            )
        }
    }

    /// Get the output dimension of a projection kernel.
    private static func outputDim(of kernel: ProjectionKernel) -> Int {
        switch kernel {
        case .dense(let w):
            return w.dim(0)
        case .affineQuantized(let q):
            return q.logicalShape[0]
        case .dequantizeMatmul(let q):
            return q.logicalShape[0]
        }
    }
}
