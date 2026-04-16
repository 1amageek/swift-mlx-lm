import LMArchitecture

/// Single vision transformer layer with sandwich normalization.
///
/// Gemma 4 vision layers apply pre- and post-normalization around both
/// the attention and MLP sub-blocks (sandwich norm pattern):
///
/// ```
/// norm → attention → norm → residual_add
/// norm → mlp       → norm → residual_add
/// ```
struct Gemma4VisionLayer: ModelComponent {

    let hiddenSize: Int
    let intermediateSize: Int
    let headCount: Int
    let headDimension: Int
    let ropeTheta: Float
    let hiddenAct: String

    @ModelComponentBuilder
    var body: some ModelComponent {
        Residual {
            RMSNorm(dimension: hiddenSize)
            Attention(
                hiddenSize: hiddenSize,
                headCount: headCount,
                kvHeadCount: headCount,
                headDimension: headDimension,
                causal: false,
                rope: RoPEAttributes(
                    dimension: headDimension,
                    base: ropeTheta,
                    mropeAxes: MRoPEAxes(
                        sections: [headDimension / 4, headDimension / 4],
                        interleaved: false
                    )
                ),
                qkNorm: .rmsNorm
            )
            RMSNorm(dimension: hiddenSize)
        }

        Residual {
            RMSNorm(dimension: hiddenSize)
            MLP(
                inputSize: hiddenSize,
                intermediateSize: intermediateSize,
                activation: .custom(hiddenAct),
                gating: .geglu
            )
            RMSNorm(dimension: hiddenSize)
        }
    }
}
