@preconcurrency import MLX

/// Compile-time resolved kernel for a linear projection.
///
/// The kernel variant is determined once during model compilation (not at runtime),
/// enabling zero-overhead dispatch between `matmul` and `quantizedMatmul`.
///
/// Three variants handle all weight representations:
/// - `.dense`: unquantized weight → `matmul`
/// - `.affineQuantized`: quantized weight with `groupSize >= 32` → `quantizedMM`
/// - `.dequantizeMatmul`: quantized weight with `groupSize < 32` → `dequantized` + `matmul`
///
/// The third variant exists because MLX's `quantizedMM` Metal kernels require
/// `groupSize >= 32`. GGUF types Q2_K, Q3_K, and Q6_K produce `groupSize = 16`,
/// so the packed representation is kept in memory (preserving compression) and
/// dequantized transiently on each forward pass.
public enum ProjectionKernel: @unchecked Sendable {

    /// Dense (unquantized) weight — dispatches to `matmul(x, w.T)`.
    case dense(weight: MLXArray)

    /// Affine-quantized weight with `groupSize >= 32` — dispatches to `quantizedMM`.
    case affineQuantized(AffineQuantizedTensor)

    /// Affine-quantized weight with `groupSize < 32` — dispatches to `dequantized` + `matmul`.
    ///
    /// MLX's `quantizedMM` Metal kernels do not support `groupSize < 32`.
    /// The quantized storage is preserved to minimize memory; dequantization
    /// happens transiently on each forward pass (same strategy as `DirectQuantizedLinear`
    /// in the standard path).
    case dequantizeMatmul(AffineQuantizedTensor)
}

/// Lowered linear projection with compile-time kernel selection.
///
/// This is the core innovation of the inference compiler: the kernel variant
/// (`matmul` vs `quantizedMatmul` vs `dequantize+matmul`) is resolved at compile
/// time based on `MLXTensorStorage`, not post-hoc via `quantize(model:)`.
///
/// The `bias` field is the affine bias from `Wx + b` (a learned model parameter),
/// NOT the quantization zero-point (which is inside `AffineQuantizedTensor.zeroBiases`).
public struct LoweredProjection: @unchecked Sendable {

    /// Compile-time resolved kernel.
    public let kernel: ProjectionKernel

    /// Optional affine bias (Wx + b).
    public let bias: MLXArray?

    /// Initialize from `MLXTensorStorage` — kernel selection happens here.
    ///
    /// For quantized storage, the groupSize determines the kernel:
    /// - `groupSize >= 32` → `.affineQuantized` (hardware-accelerated `quantizedMM`)
    /// - `groupSize < 32` → `.dequantizeMatmul` (software `dequantized` + `matmul`)
    public init(storage: MLXTensorStorage, bias: MLXArray? = nil) {
        switch storage {
        case .dense(let array):
            self.kernel = .dense(weight: array)
        case .affineQuantized(let qt):
            if qt.groupSize >= 32 {
                self.kernel = .affineQuantized(qt)
            } else {
                self.kernel = .dequantizeMatmul(qt)
            }
        }
        self.bias = bias
    }

    /// Initialize with a dense weight directly.
    public init(weight: MLXArray, bias: MLXArray? = nil) {
        self.kernel = .dense(weight: weight)
        self.bias = bias
    }

    /// Apply the projection: `y = Wx + b`.
    ///
    /// Dispatches to `matmul`, `quantizedMM`, or `dequantized+matmul` based on
    /// the compile-time resolved kernel variant.
    public func apply(_ x: MLXArray) -> MLXArray {
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
        if let bias {
            result = result + bias
        }
        return result
    }
}
