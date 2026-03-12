/// Unified vision encoder configuration.
///
/// Covers all supported vision encoder architectures. Fields are populated
/// from mmproj GGUF metadata by ``GGUFVisionLoader``. Optional fields
/// (`windowSize`, `fullAttBlockIndexes`) distinguish architecture variants:
///
/// - **Window+Full hybrid** (e.g. Conv3d patch embed, SwiGLU MLP):
///   `windowSize != nil`, `fullAttBlockIndexes` specifies which layers use global attention.
/// - **All full attention** (e.g. Conv2d patch embed, GELU MLP):
///   `windowSize == nil`, `fullAttBlockIndexes == nil`.
struct VisionConfig: Sendable {

    var hiddenSize: Int
    var intermediateSize: Int
    var depth: Int
    var numHeads: Int
    var outHiddenSize: Int
    var patchSize: Int
    var spatialMergeSize: Int

    /// Temporal patch size for Conv3d-based patch embeddings.
    /// Set to 1 for Conv2d-based architectures.
    var temporalPatchSize: Int

    var inChannels: Int
    var normEps: Float

    /// Window size for windowed attention. `nil` means all layers use full attention.
    var windowSize: Int?

    /// Layer indices that use full (global) attention instead of windowed.
    /// `nil` means all layers use full attention.
    var fullAttBlockIndexes: [Int]?

    /// Image normalization parameters from mmproj metadata.
    var imageMean: [Float]
    var imageStd: [Float]

    /// Image dimension constraints.
    var minPixels: Int
    var maxPixels: Int

    init(
        hiddenSize: Int,
        intermediateSize: Int,
        depth: Int,
        numHeads: Int,
        outHiddenSize: Int,
        patchSize: Int,
        spatialMergeSize: Int = 2,
        temporalPatchSize: Int = 1,
        inChannels: Int = 3,
        normEps: Float = 1e-6,
        windowSize: Int? = nil,
        fullAttBlockIndexes: [Int]? = nil,
        imageMean: [Float] = [0.48145466, 0.4578275, 0.40821073],
        imageStd: [Float] = [0.26862954, 0.26130258, 0.27577711],
        minPixels: Int = 3136,
        maxPixels: Int = 12_845_056
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.depth = depth
        self.numHeads = numHeads
        self.outHiddenSize = outHiddenSize
        self.patchSize = patchSize
        self.spatialMergeSize = spatialMergeSize
        self.temporalPatchSize = temporalPatchSize
        self.inChannels = inChannels
        self.normEps = normEps
        self.windowSize = windowSize
        self.fullAttBlockIndexes = fullAttBlockIndexes
        self.imageMean = imageMean
        self.imageStd = imageStd
        self.minPixels = minPixels
        self.maxPixels = maxPixels
    }

    var headDim: Int { hiddenSize / numHeads }

    /// Factor for image dimensions (must be divisible by this).
    var imageFactor: Int { patchSize * spatialMergeSize }

    /// Whether this configuration uses windowed attention.
    var usesWindowAttention: Bool { windowSize != nil }
}
