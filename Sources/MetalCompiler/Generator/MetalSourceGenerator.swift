/// Generates Metal Shading Language (MSL) source from kernel fragments.
///
/// The generator handles three orthogonal concerns:
/// 1. **Computation** — from the fragment (Reduction, Elementwise, etc.)
/// 2. **Weight reading** — from QuantizationFormat (BF16, FP16, Q4, etc.)
/// 3. **Buffer precision** — from the execution mode (F16 decode, F32 prefill)
///
/// No dtype/precision variants are hardcoded. All combinations are generated
/// from these three independent inputs.
public struct MetalSourceGenerator: Sendable {

    // BufferPrecision and WeightFormat are top-level types in MetalDispatchTypes.swift.
    // These typealiases preserve source compatibility within MetalSourceGenerator.
    public typealias BufferPrecision = MetalCompiler.BufferPrecision
    public typealias WeightFormat = MetalCompiler.WeightFormat

    enum SpecializedDenseInputStaging {
        case bufferPrecision
        case float

        var stagesAsFloat: Bool {
            switch self {
            case .bufferPrecision:
                return false
            case .float:
                return true
            }
        }
    }

    enum SpecializedDenseAccumulationStyle {
        case indexed
        case pointerIncrement
    }

    /// Shared MSL declarations used by all generated kernels.
    public static let commonHeader = """
    #include <metal_stdlib>
    using namespace metal;

    constant constexpr uint SIMD_WIDTH = 32;

    /// BFloat16 → Float32: BF16 is the upper 16 bits of Float32.
    /// Zero-extend to 32 bits — no precision loss, no computation.
    inline float bf16_to_float(uint16_t bf16) {
        uint32_t f32_bits = uint32_t(bf16) << 16;
        return as_type<float>(f32_bits);
    }

    inline uint16_t float_to_bf16(float value) {
        uint32_t bits = as_type<uint32_t>(value);
        uint32_t lsb = (bits >> 16) & 1;
        uint32_t roundingBias = 0x7FFF + lsb;
        uint32_t rounded = bits + roundingBias;
        return uint16_t(rounded >> 16);
    }

    inline float2 bf16x2_to_float2(ushort2 bf16) {
        return float2(bf16_to_float(bf16.x), bf16_to_float(bf16.y));
    }

    inline float4 bf16x4_to_float4(ushort4 bf16) {
        return float4(
            bf16_to_float(bf16.x),
            bf16_to_float(bf16.y),
            bf16_to_float(bf16.z),
            bf16_to_float(bf16.w)
        );
    }

    """

    public static func sequenceStorageValue(_ expression: String, weightFormat: WeightFormat) -> String {
        if weightFormat.isBFloat16 {
            return "float(bfloat(\(expression)))"
        }
        if weightFormat.isFloat32 {
            return expression
        }
        return "float(half(\(expression)))"
    }
}
