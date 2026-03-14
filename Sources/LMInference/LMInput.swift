@preconcurrency import MLX

/// Tokenized model input.
public struct LMInput: Sendable {

    public let text: Text
    public let image: ProcessedImage?
    public let video: ProcessedVideo?

    /// Token sequence with optional mask, position IDs, and pre-computed embeddings.
    public struct Text: Sendable {
        public let tokens: MLXArray
        public let mask: MLXArray?

        /// Per-token position IDs for models that use non-sequential positioning
        /// (e.g. M-RoPE with shape `[3, B, S]` for temporal/height/width).
        public let positionIds: MLXArray?

        /// Pre-computed embeddings that bypass token embedding lookup.
        ///
        /// When set, the model skips the token embedding step and uses these
        /// directly. Shape: `[B, S, D]`. Used for VLM sequential chunk
        /// processing where vision embeddings are merged externally.
        public let embeddings: MLXArray?

        /// CPU-side token IDs cached from the tokenizer output.
        /// Avoids redundant GPU→CPU round-trips for prefix matching and
        /// repetition processor initialization.
        public let cpuTokenIDs: [Int]?

        public init(
            tokens: MLXArray, mask: MLXArray? = nil,
            positionIds: MLXArray? = nil, embeddings: MLXArray? = nil,
            cpuTokenIDs: [Int]? = nil
        ) {
            self.tokens = tokens
            self.mask = mask
            self.positionIds = positionIds
            self.embeddings = embeddings
            self.cpuTokenIDs = cpuTokenIDs
        }
    }

    /// Preprocessed image ready for model consumption.
    public struct ProcessedImage: Sendable {
        public let pixels: MLXArray
        public let frames: [THW]?

        public init(pixels: MLXArray, frames: [THW]? = nil) {
            self.pixels = pixels
            self.frames = frames
        }
    }

    /// Preprocessed video ready for model consumption.
    public struct ProcessedVideo: Sendable {
        public let pixels: MLXArray
        public let frames: [THW]?

        public init(pixels: MLXArray, frames: [THW]? = nil) {
            self.pixels = pixels
            self.frames = frames
        }
    }

    /// Temporal-Height-Width dimensions for vision features.
    public struct THW: Sendable {
        public let t: Int
        public let h: Int
        public let w: Int

        public init(t: Int, h: Int, w: Int) {
            self.t = t
            self.h = h
            self.w = w
        }

        public var values: (Int, Int, Int) { (t, h, w) }
        public var product: Int { t * h * w }
    }

    public init(tokens: MLXArray, mask: MLXArray? = nil) {
        self.text = Text(tokens: tokens, mask: mask)
        self.image = nil
        self.video = nil
    }

    public init(text: Text, image: ProcessedImage? = nil, video: ProcessedVideo? = nil) {
        self.text = text
        self.image = image
        self.video = video
    }
}
