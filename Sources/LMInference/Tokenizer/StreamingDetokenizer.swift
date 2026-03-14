/// Incremental token-to-text decoder for streaming generation.
///
/// Accumulates tokens and emits newly decoded text on each step.
/// Handles incomplete UTF-8 sequences by buffering until complete.
public struct StreamingDetokenizer: Sendable {

    private let tokenizer: any Tokenizer
    private var segmentTokens: [Int]
    private var previousDecode: String

    /// Create a streaming detokenizer.
    public init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
        self.segmentTokens = []
        self.previousDecode = ""
    }

    /// Append a token and return the newly decoded text, if any.
    ///
    /// Returns `nil` if the token produces an incomplete UTF-8 sequence
    /// (U+FFFD replacement character), meaning more tokens are needed.
    public mutating func append(token: Int) -> String? {
        segmentTokens.append(token)
        let fullDecode = tokenizer.decode(tokens: segmentTokens)

        let newPortion: Substring
        if fullDecode.count >= previousDecode.count {
            newPortion = fullDecode.suffix(fullDecode.count - previousDecode.count)
        } else {
            // Decode shrank (shouldn't happen, but handle gracefully)
            newPortion = fullDecode[...]
        }

        // Check for incomplete UTF-8 sequence
        if newPortion.last == "\u{FFFD}" {
            return nil
        }

        // Reset segment on newlines to prevent unbounded growth
        if newPortion.hasSuffix("\n") {
            resetSegment()
        } else {
            previousDecode = fullDecode
        }

        return newPortion.isEmpty ? nil : String(newPortion)
    }

    /// Reset the segment, keeping only the last token for context.
    private mutating func resetSegment() {
        let lastToken = segmentTokens.last
        segmentTokens.removeAll()
        if let lastToken {
            segmentTokens.append(lastToken)
            previousDecode = tokenizer.decode(tokens: segmentTokens)
        } else {
            previousDecode = ""
        }
    }
}
