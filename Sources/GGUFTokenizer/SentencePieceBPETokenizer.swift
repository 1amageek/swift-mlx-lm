/// SentencePiece-style BPE tokenizer (Llama 1/2 style).
///
/// Uses vocabulary scores instead of explicit merge rules. Spaces are
/// replaced with the ▁ (U+2581) character. Unknown characters fall back
/// to individual byte tokens in `<0xXX>` format.
///
/// The algorithm:
/// 1. Replace spaces with ▁, optionally prepend ▁
/// 2. Split into individual Unicode characters
/// 3. Unknown characters are expanded to byte fallback tokens
/// 4. Iteratively merge the pair whose result has the highest vocabulary score
/// 5. Map final symbols to vocabulary IDs
struct SentencePieceBPETokenizer: Tokenizer, Sendable {

    /// Replacement character for spaces.
    static let sentencePieceSpace: Character = "\u{2581}"

    /// Token strings indexed by ID.
    let vocabulary: [String]

    /// Token string to ID mapping.
    let tokenToID: [String: Int]

    /// Per-token scores (log-probability). Higher = merge first.
    let scores: [Float]

    /// Special token configuration.
    let specialTokens: SpecialTokens

    /// Token type array.
    let tokenTypes: [Int]

    /// Whether to prepend ▁ before the text.
    let addSpacePrefix: Bool

    /// Byte token lookup: byte value -> token ID for `<0xXX>` tokens.
    let byteTokenIDs: [UInt8: Int]

    /// Control token IDs to skip during decode.
    let controlTokenIDs: Set<Int>

    var bosTokenID: Int? { specialTokens.bosTokenID }
    var eosTokenID: Int? { specialTokens.eosTokenID }
    var vocabularySize: Int { vocabulary.count }

    /// Create a SentencePiece BPE tokenizer.
    ///
    /// - Parameters:
    ///   - vocabulary: Token strings indexed by ID.
    ///   - scores: Per-token scores (same length as vocabulary).
    ///   - specialTokens: Special token configuration.
    ///   - tokenTypes: Per-token type array.
    ///   - addSpacePrefix: Whether to prepend ▁ (default true).
    init(
        vocabulary: [String],
        scores: [Float],
        specialTokens: SpecialTokens,
        tokenTypes: [Int] = [],
        addSpacePrefix: Bool = true
    ) throws {
        guard vocabulary.count == scores.count else {
            throw TokenizerError.missingScores
        }

        self.vocabulary = vocabulary
        self.scores = scores
        self.specialTokens = specialTokens
        self.tokenTypes = tokenTypes
        self.addSpacePrefix = addSpacePrefix

        // Build token-to-ID map
        var t2id: [String: Int] = [:]
        t2id.reserveCapacity(vocabulary.count)
        for (id, token) in vocabulary.enumerated() {
            t2id[token] = id
        }
        self.tokenToID = t2id

        // Build byte token lookup from token_type == 6 (BYTE)
        var byteIDs: [UInt8: Int] = [:]
        let types = tokenTypes.isEmpty ? Array(repeating: 1, count: vocabulary.count) : tokenTypes
        for (id, tokenType) in types.enumerated() where id < vocabulary.count {
            if tokenType == 6 {
                if let byte = Self.parseByteToken(vocabulary[id]) {
                    byteIDs[byte] = id
                }
            }
        }
        // Fallback: scan for <0xXX> pattern if no type info
        if byteIDs.isEmpty {
            for (id, token) in vocabulary.enumerated() {
                if let byte = Self.parseByteToken(token) {
                    byteIDs[byte] = id
                }
            }
        }
        self.byteTokenIDs = byteIDs

        // Build control token ID set
        var control = Set<Int>()
        for (id, tokenType) in types.enumerated() where id < vocabulary.count {
            if tokenType == 3 {
                control.insert(id)
            }
        }
        self.controlTokenIDs = control
    }

    // MARK: - Tokenizer

    func encode(text: String) -> [Int] {
        var result: [Int] = []

        if specialTokens.addBosToken, let bos = specialTokens.bosTokenID {
            result.append(bos)
        }

        guard !text.isEmpty else {
            if specialTokens.addEosToken, let eos = specialTokens.eosTokenID {
                result.append(eos)
            }
            return result
        }

        // Pre-process: replace spaces with ▁, optionally prepend ▁
        var processed = text.replacingOccurrences(
            of: " ",
            with: String(Self.sentencePieceSpace)
        )
        if addSpacePrefix {
            processed = String(Self.sentencePieceSpace) + processed
        }

        // Split into initial symbols (Unicode characters)
        var symbols = expandToSymbols(processed)

        // BPE merge loop
        while symbols.count > 1 {
            var bestScore: Float = -.infinity
            var bestIndex = -1

            for i in 0..<(symbols.count - 1) {
                let merged = symbols[i] + symbols[i + 1]
                if let id = tokenToID[merged] {
                    let score = scores[id]
                    if score > bestScore {
                        bestScore = score
                        bestIndex = i
                    }
                }
            }

            guard bestIndex >= 0 else { break }

            // Merge all occurrences of this pair
            var newSymbols: [String] = []
            newSymbols.reserveCapacity(symbols.count)
            let first = symbols[bestIndex]
            let second = symbols[bestIndex + 1]
            var i = 0
            while i < symbols.count {
                if i < symbols.count - 1 && symbols[i] == first && symbols[i + 1] == second {
                    // Verify the merged token is actually in vocabulary
                    let candidate = symbols[i] + symbols[i + 1]
                    if tokenToID[candidate] != nil {
                        newSymbols.append(candidate)
                        i += 2
                        continue
                    }
                }
                newSymbols.append(symbols[i])
                i += 1
            }
            symbols = newSymbols
        }

        // Convert symbols to IDs
        for symbol in symbols {
            if let id = tokenToID[symbol] {
                result.append(id)
            } else if let unknownID = specialTokens.unknownTokenID {
                result.append(unknownID)
            }
        }

        if specialTokens.addEosToken, let eos = specialTokens.eosTokenID {
            result.append(eos)
        }

        return result
    }

    func decode(tokens: [Int]) -> String {
        var parts: [String] = []

        var pendingBytes: [UInt8] = []

        for id in tokens {
            guard id >= 0, id < vocabulary.count else { continue }
            if controlTokenIDs.contains(id) { continue }

            let tokenString = vocabulary[id]

            // Check if this is a byte token
            if let byte = Self.parseByteToken(tokenString) {
                pendingBytes.append(byte)
                continue
            }

            // Flush pending bytes first
            if !pendingBytes.isEmpty {
                parts.append(String(decoding: pendingBytes, as: UTF8.self))
                pendingBytes.removeAll()
            }

            parts.append(tokenString)
        }

        // Flush remaining bytes
        if !pendingBytes.isEmpty {
            parts.append(String(decoding: pendingBytes, as: UTF8.self))
        }

        var result = parts.joined()

        // Replace ▁ back to space
        result = result.replacingOccurrences(
            of: String(Self.sentencePieceSpace),
            with: " "
        )

        // Remove leading space (from the prepended ▁)
        if addSpacePrefix, result.hasPrefix(" ") {
            result.removeFirst()
        }

        return result
    }

    func tokenToString(_ id: Int) -> String? {
        guard id >= 0, id < vocabulary.count else { return nil }
        return vocabulary[id]
    }

    // MARK: - Helpers

    /// Expand a preprocessed string into initial BPE symbols.
    ///
    /// Each Unicode character becomes a symbol. Characters not in the
    /// vocabulary are expanded to byte fallback tokens (`<0xXX>`).
    private func expandToSymbols(_ text: String) -> [String] {
        var symbols: [String] = []
        for char in text {
            let charString = String(char)
            if tokenToID[charString] != nil {
                symbols.append(charString)
            } else {
                // Byte fallback: split character into UTF-8 bytes
                for byte in charString.utf8 {
                    let byteToken = String(format: "<0x%02X>", byte)
                    symbols.append(byteToken)
                }
            }
        }
        return symbols
    }

    /// Parse a byte fallback token like `<0xAB>` into its byte value.
    static func parseByteToken(_ token: String) -> UInt8? {
        guard token.count == 6,
              token.hasPrefix("<0x"),
              token.hasSuffix(">")
        else { return nil }
        let hex = token.dropFirst(3).dropLast(1)
        return UInt8(hex, radix: 16)
    }
}
