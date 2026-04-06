import Accelerate
import Foundation

enum QwenVisionMath {
    static func linear(
        input: [Float],
        rowCount: Int,
        inputDimension: Int,
        weight: [Float],
        outputDimension: Int,
        bias: [Float]? = nil
    ) -> [Float] {
        precondition(input.count == rowCount * inputDimension)
        precondition(weight.count == outputDimension * inputDimension)
        var output = [Float](repeating: 0, count: rowCount * outputDimension)
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            Int32(rowCount),
            Int32(outputDimension),
            Int32(inputDimension),
            1.0,
            input,
            Int32(inputDimension),
            weight,
            Int32(inputDimension),
            0.0,
            &output,
            Int32(outputDimension)
        )
        if let bias {
            precondition(bias.count == outputDimension)
            for row in 0..<rowCount {
                let base = row * outputDimension
                for column in 0..<outputDimension {
                    output[base + column] += bias[column]
                }
            }
        }
        return output
    }

    static func layerNorm(
        input: [Float],
        rowCount: Int,
        dimension: Int,
        weight: [Float],
        bias: [Float],
        epsilon: Float = 1e-6
    ) -> [Float] {
        precondition(input.count == rowCount * dimension)
        precondition(weight.count == dimension)
        precondition(bias.count == dimension)

        var output = [Float](repeating: 0, count: input.count)
        for row in 0..<rowCount {
            let base = row * dimension
            var mean: Float = 0
            for column in 0..<dimension {
                mean += input[base + column]
            }
            mean /= Float(dimension)

            var variance: Float = 0
            for column in 0..<dimension {
                let centered = input[base + column] - mean
                variance += centered * centered
            }
            variance /= Float(dimension)
            let invStd = 1 / sqrtf(variance + epsilon)

            for column in 0..<dimension {
                let normalized = (input[base + column] - mean) * invStd
                output[base + column] = normalized * weight[column] + bias[column]
            }
        }
        return output
    }

    static func add(_ lhs: [Float], _ rhs: [Float]) -> [Float] {
        precondition(lhs.count == rhs.count)
        var result = [Float](repeating: 0, count: lhs.count)
        vDSP_vadd(lhs, 1, rhs, 1, &result, 1, vDSP_Length(lhs.count))
        return result
    }

    static func gelu(_ input: [Float], kind: String) -> [Float] {
        switch kind {
        case "gelu_pytorch_tanh", "gelu_new", "gelu_fast":
            return input.map { value in
                let cubic = value * value * value
                let inner = sqrtf(2 / Float.pi) * (value + 0.044715 * cubic)
                return 0.5 * value * (1 + tanh(inner))
            }
        default:
            return input.map { value in
                let scaled = Double(value) / sqrt(2.0)
                return Float(0.5 * Double(value) * (1.0 + erf(scaled)))
            }
        }
    }

    static func softmaxRows(_ input: [Float], rowCount: Int, columnCount: Int) -> [Float] {
        precondition(input.count == rowCount * columnCount)
        var output = [Float](repeating: 0, count: input.count)
        for row in 0..<rowCount {
            let base = row * columnCount
            let rowSlice = input[base..<(base + columnCount)]
            let maxValue = rowSlice.max() ?? 0
            var sum: Float = 0
            for column in 0..<columnCount {
                let value = expf(input[base + column] - maxValue)
                output[base + column] = value
                sum += value
            }
            let scale = sum == 0 ? 0 : 1 / sum
            for column in 0..<columnCount {
                output[base + column] *= scale
            }
        }
        return output
    }
}
