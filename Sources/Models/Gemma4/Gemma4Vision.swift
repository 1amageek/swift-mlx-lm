import LMArchitecture

/// Gemma 4 vision encoder declaration.
///
/// Processes image patches through a vision transformer and projects
/// the output embeddings into the text decoder's embedding space.
///
/// ```
/// PatchEmbedding → PositionEmbedding → N × VisionLayer(sandwich norm)
///   → Pooling → [Standardize] → RMSNorm(no scale) → Linear
/// ```
///
/// Grid dimensions (`gridWidth`) are image-specific. The vision encoder
/// is compiled per unique grid configuration.
public struct Gemma4Vision: ModelComponent {

    public let hiddenSize: Int
    public let intermediateSize: Int
    public let headCount: Int
    public let layerCount: Int
    public let patchSize: Int
    public let inChannels: Int
    public let poolingKernelSize: Int
    public let positionEmbeddingSize: Int
    public let gridWidth: Int
    public let ropeTheta: Float
    public let hiddenAct: String
    public let textHiddenSize: Int
    public let standardize: Bool

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        headCount: Int,
        layerCount: Int,
        patchSize: Int,
        inChannels: Int,
        poolingKernelSize: Int,
        positionEmbeddingSize: Int = 48,
        gridWidth: Int = 1,
        ropeTheta: Float,
        hiddenAct: String,
        textHiddenSize: Int,
        standardize: Bool = false
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.headCount = headCount
        self.layerCount = layerCount
        self.patchSize = patchSize
        self.inChannels = inChannels
        self.poolingKernelSize = poolingKernelSize
        self.positionEmbeddingSize = positionEmbeddingSize
        self.gridWidth = gridWidth
        self.ropeTheta = ropeTheta
        self.hiddenAct = hiddenAct
        self.textHiddenSize = textHiddenSize
        self.standardize = standardize
    }

    private var headDimension: Int { hiddenSize / headCount }
    private var patchPixelDimension: Int { patchSize * patchSize * inChannels }

    @ModelComponentBuilder
    public var body: some ModelComponent {
        PatchEmbedding(
            patchPixelDimension: patchPixelDimension,
            hiddenSize: hiddenSize
        )

        PositionEmbedding(
            hiddenSize: hiddenSize,
            tableSize: positionEmbeddingSize,
            gridWidth: gridWidth
        )

        ForEach(0..<layerCount) { _ in
            Gemma4VisionLayer(
                hiddenSize: hiddenSize,
                intermediateSize: intermediateSize,
                headCount: headCount,
                headDimension: headDimension,
                ropeTheta: ropeTheta,
                hiddenAct: hiddenAct
            )
        }

        Pooling(
            kernelSize: poolingKernelSize,
            hiddenSize: hiddenSize,
            rescale: Float(hiddenSize).squareRoot()
        )

        if standardize {
            Standardize(dimension: hiddenSize)
        }

        RMSNorm(dimension: hiddenSize, withScale: false)

        Linear(inputSize: hiddenSize, outputSize: textHiddenSize)
    }
}
