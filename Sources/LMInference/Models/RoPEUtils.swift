import Foundation
import MLX
import MLXFast
import MLXNN

/// Combined RoPE protocol for integer and array offsets.
typealias RoPELayer = OffsetLayer & ArrayOffsetLayer

/// Llama 3 style RoPE with frequency interpolation.
class Llama3RoPE: Module, OffsetLayer, ArrayOffsetLayer {

    let dims: Int
    let traditional: Bool
    let _freqs: MLXArray

    init(
        dims: Int,
        maxPositionEmbeddings: Int = 2048,
        traditional: Bool = false,
        base: Float = 10000,
        scalingConfig: [String: StringOrNumber]? = nil
    ) {
        self.dims = dims
        self.traditional = traditional

        guard let scalingConfig else {
            fatalError("Llama3RoPE requires scalingConfig")
        }

        let factor = scalingConfig["factor"]?.asFloat() ?? 1.0
        let lowFreqFactor = scalingConfig["low_freq_factor"]?.asFloat() ?? 1.0
        let highFreqFactor = scalingConfig["high_freq_factor"]?.asFloat() ?? 4.0
        let oldContextLen = scalingConfig["original_max_position_embeddings"]?.asFloat() ?? 8192.0

        let lowFreqWavelen = oldContextLen / lowFreqFactor
        let highFreqWavelen = oldContextLen / highFreqFactor

        let indices = MLXArray(stride(from: 0, to: dims, by: 2))
        var frequencies = MLX.pow(base, indices / Float(dims))
        let wavelens = 2 * Float.pi * frequencies

        frequencies = MLX.where(
            wavelens .> MLXArray(lowFreqWavelen),
            frequencies * factor,
            frequencies
        )

        let isMediumFreq = MLX.logicalAnd(
            wavelens .> MLXArray(highFreqWavelen),
            wavelens .< MLXArray(lowFreqWavelen)
        )

        let smoothFactors =
            (oldContextLen / wavelens - lowFreqFactor) / (highFreqFactor - lowFreqFactor)
        let smoothFreqs = frequencies / ((1 - smoothFactors) / factor + smoothFactors)

        self._freqs = MLX.where(isMediumFreq, smoothFreqs, frequencies)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(x, dimensions: dims, traditional: traditional,
                     base: nil, scale: 1.0, offset: offset, freqs: _freqs)
    }

    func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        MLXFast.RoPE(x, dimensions: dims, traditional: traditional,
                     base: nil, scale: 1.0, offset: offset, freqs: _freqs)
    }
}

/// Yarn RoPE with magnitude scaling.
class YarnRoPE: Module, OffsetLayer, ArrayOffsetLayer {

    let dimensions: Int
    let traditional: Bool
    private let _mscale: Float
    private let _freqs: MLXArray

    init(
        dimensions: Int,
        traditional: Bool = false,
        maxPositionEmbeddings: Int = 2048,
        base: Float = 10000,
        scalingFactor: Float = 1.0,
        originalMaxPositionEmbeddings: Int = 4096,
        betaFast: Float = 32,
        betaSlow: Float = 1,
        mscale: Float = 1,
        mscaleAllDim: Float = 0
    ) {
        precondition(dimensions % 2 == 0)
        self.dimensions = dimensions
        self.traditional = traditional

        func yarnFindCorrectionDim(numRotations: Float) -> Float {
            Float(dimensions)
                * log(Float(originalMaxPositionEmbeddings) / (numRotations * 2 * Float.pi))
                / (2 * log(base))
        }

        func yarnFindCorrectionRange() -> (low: Int, high: Int) {
            let low = Int(floor(yarnFindCorrectionDim(numRotations: betaFast)))
            let high = Int(ceil(yarnFindCorrectionDim(numRotations: betaSlow)))
            return (max(low, 0), min(high, dimensions - 1))
        }

        func yarnGetMscale(scale: Float, mscale: Float) -> Float {
            scale <= 1 ? 1.0 : 0.1 * mscale * log(scale) + 1.0
        }

        func yarnLinearRampMask(minVal: Float, maxVal: Float, dim: Int) -> MLXArray {
            var maxVal = maxVal
            if minVal == maxVal { maxVal += 0.001 }
            let linearFunc = (MLXArray(0..<dim).asType(.float32) - minVal) / (maxVal - minVal)
            return clip(linearFunc, min: 0, max: 1)
        }

        self._mscale =
            yarnGetMscale(scale: scalingFactor, mscale: mscale)
            / yarnGetMscale(scale: scalingFactor, mscale: mscaleAllDim)

        let freqExtra = pow(
            base,
            MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32) / dimensions)
        let freqInter =
            scalingFactor
            * pow(
                base,
                MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32) / dimensions)

        let (low, high) = yarnFindCorrectionRange()
        let freqMask =
            1.0 - yarnLinearRampMask(minVal: Float(low), maxVal: Float(high), dim: dimensions / 2)

        self._freqs = (freqInter * freqExtra) / (freqInter * freqMask + freqExtra * (1 - freqMask))
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        var x = x
        if _mscale != 1.0 {
            x = x[0..., .ellipsis]
            x[.ellipsis, 0..<dimensions] *= _mscale
        }
        return MLXFast.RoPE(x, dimensions: dimensions, traditional: traditional,
                           base: nil, scale: 1.0, offset: offset, freqs: _freqs)
    }

    func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        var x = x
        if _mscale != 1.0 {
            x = x[0..., .ellipsis]
            x[.ellipsis, 0..<dimensions] *= _mscale
        }
        return MLXFast.RoPE(x, dimensions: dimensions, traditional: traditional,
                           base: nil, scale: 1.0, offset: offset, freqs: _freqs)
    }
}

/// Su-scaled RoPE (LongRoPE) with per-dimension non-uniform scaling.
///
/// Used by Phi-3 128K context models. Applies separate short and long frequency
/// scaling factors depending on the current sequence length.
class SuScaledRoPE: Module, OffsetLayer, ArrayOffsetLayer {

    let dims: Int
    let traditional: Bool
    let maxPositionEmbeddings: Int
    let originalMaxPositionEmbeddings: Int
    let _shortFreqs: MLXArray
    let _longFreqs: MLXArray

    init(
        dims: Int,
        maxPositionEmbeddings: Int = 131072,
        originalMaxPositionEmbeddings: Int = 4096,
        traditional: Bool = false,
        base: Float = 10000,
        shortFactor: [Float],
        longFactor: [Float]
    ) {
        precondition(shortFactor.count == dims / 2, "shortFactor length must equal dims/2")
        precondition(longFactor.count == dims / 2, "longFactor length must equal dims/2")

        self.dims = dims
        self.traditional = traditional
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings

        let indices = MLXArray(stride(from: 0, to: dims, by: 2)).asType(.float32)
        let baseFreqs = MLX.pow(base, indices / Float(dims))

        self._shortFreqs = baseFreqs * MLXArray(shortFactor)
        self._longFreqs = baseFreqs * MLXArray(longFactor)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        let seqLen = x.dim(x.ndim - 2) + offset
        let freqs = seqLen > originalMaxPositionEmbeddings ? _longFreqs : _shortFreqs
        return MLXFast.RoPE(x, dimensions: dims, traditional: traditional,
                           base: nil, scale: 1.0, offset: offset, freqs: freqs)
    }

    func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        MLXFast.RoPE(x, dimensions: dims, traditional: traditional,
                     base: nil, scale: 1.0, offset: offset, freqs: _longFreqs)
    }
}

private let yarnTypes: Set = ["yarn", "deepseek_yarn", "telechat3-yarn"]

/// Create the appropriate RoPE layer based on scaling configuration.
func initializeRope(
    dims: Int,
    base: Float,
    traditional: Bool,
    scalingConfig: [String: StringOrNumber]?,
    maxPositionEmbeddings: Int?
) -> RoPELayer {
    let ropeType: String = {
        if let config = scalingConfig,
           let typeValue = config["type"] ?? config["rope_type"],
           case .string(let s) = typeValue
        {
            return s
        }
        return "default"
    }()

    if ropeType == "default" || ropeType == "linear" {
        let scale: Float
        if ropeType == "linear", let factor = scalingConfig?["factor"]?.asFloat() {
            scale = 1 / factor
        } else {
            scale = 1.0
        }
        return RoPE(dimensions: dims, traditional: traditional, base: base, scale: scale)
    } else if ropeType == "llama3" {
        return Llama3RoPE(
            dims: dims,
            maxPositionEmbeddings: maxPositionEmbeddings ?? 2048,
            traditional: traditional,
            base: base,
            scalingConfig: scalingConfig
        )
    } else if ropeType == "su" || ropeType == "longrope" {
        guard let config = scalingConfig,
              let shortFactor = config["short_factor"]?.asFloats(),
              let longFactor = config["long_factor"]?.asFloats()
        else {
            fatalError("SuScaledRoPE requires short_factor and long_factor arrays in scalingConfig")
        }
        let origMax = config["original_max_position_embeddings"]?.asInt() ?? 4096
        return SuScaledRoPE(
            dims: dims,
            maxPositionEmbeddings: maxPositionEmbeddings ?? 131072,
            originalMaxPositionEmbeddings: origMax,
            traditional: traditional,
            base: base,
            shortFactor: shortFactor,
            longFactor: longFactor
        )
    } else if yarnTypes.contains(ropeType) {
        let factor = scalingConfig?["factor"]?.asFloat() ?? 32.0
        let origMax = scalingConfig?["original_max_position_embeddings"]?.asInt() ?? 4096
        let betaFast = scalingConfig?["beta_fast"]?.asFloat() ?? 32.0
        let betaSlow = scalingConfig?["beta_slow"]?.asFloat() ?? 1.0
        let mscale = scalingConfig?["mscale"]?.asFloat() ?? 1.0
        let mscaleAllDim = scalingConfig?["mscale_all_dim"]?.asFloat() ?? 0.0

        return YarnRoPE(
            dimensions: dims,
            traditional: traditional,
            maxPositionEmbeddings: maxPositionEmbeddings ?? 2048,
            base: base,
            scalingFactor: factor,
            originalMaxPositionEmbeddings: origMax,
            betaFast: betaFast,
            betaSlow: betaSlow,
            mscale: mscale,
            mscaleAllDim: mscaleAllDim
        )
    } else {
        fatalError("Unsupported RoPE type: \(ropeType)")
    }
}
