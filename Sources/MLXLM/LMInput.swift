@preconcurrency import MLX

/// Tokenized model input.
public struct LMInput: Sendable {

    public let text: Text
    public let image: ProcessedImage?
    public let video: ProcessedVideo?

    /// Token sequence with optional mask.
    public struct Text: Sendable {
        public let tokens: MLXArray
        public let mask: MLXArray?

        public init(tokens: MLXArray, mask: MLXArray? = nil) {
            self.tokens = tokens
            self.mask = mask
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
