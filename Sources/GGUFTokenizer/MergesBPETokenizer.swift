/// Merges-based BPE tokenizer (GPT-2 / Llama 3 / Qwen2 style).
///
/// Uses an ordered merge list and byte-level encoding to tokenize text.
/// The algorithm:
/// 1. Pre-tokenize text into words using a regex pattern
/// 2. Encode each word to byte-level unicode (GPT-2 mapping)
/// 3. Run BPE merge loop using merge priority ranks
/// 4. Map resulting tokens to vocabulary IDs
struct MergesBPETokenizer: Tokenizer, Sendable {

    /// Token strings indexed by ID.
    let vocabulary: [String]

    /// Token string to ID mapping.
    let tokenToID: [String: Int]

    /// Merge pair to rank mapping. Key is "tokenA tokenB", value is rank (lower = higher priority).
    let mergeRanks: [String: Int]

    /// Pre-tokenizer for splitting text into words.
    let preTokenizer: PreTokenizer

    /// Special token configuration.
    let specialTokens: SpecialTokens

    /// Token type array (1=normal, 2=unknown, 3=control, 4=user_defined, 5=unused, 6=byte).
    let tokenTypes: [Int]

    /// Set of control/added token strings for special handling.
    let controlTokens: Set<String>

    var bosTokenID: Int? { specialTokens.bosTokenID }
    var eosTokenID: Int? { specialTokens.eosTokenID }
    var vocabularySize: Int { vocabulary.count }

    /// Create a merges-based BPE tokenizer.
    ///
    /// - Parameters:
    ///   - vocabulary: Token strings indexed by ID.
    ///   - merges: Merge rules as "tokenA tokenB" strings, in priority order.
    ///   - preTokenizer: Pre-tokenizer to use for splitting text.
    ///   - specialTokens: Special token configuration.
    ///   - tokenTypes: Per-token type array (optional, defaults to all normal).
    init(
        vocabulary: [String],
        merges: [String],
        preTokenizer: PreTokenizer,
        specialTokens: SpecialTokens,
        tokenTypes: [Int] = []
    ) throws {
        self.vocabulary = vocabulary
        self.preTokenizer = preTokenizer
        self.specialTokens = specialTokens
        self.tokenTypes = tokenTypes

        // Build token-to-ID map
        var t2id: [String: Int] = [:]
        t2id.reserveCapacity(vocabulary.count)
        for (id, token) in vocabulary.enumerated() {
            t2id[token] = id
        }
        self.tokenToID = t2id

        // Build merge rank map
        var ranks: [String: Int] = [:]
        ranks.reserveCapacity(merges.count)
        for (rank, merge) in merges.enumerated() {
            let parts = merge.split(separator: " ", maxSplits: 1)
            guard parts.count == 2 else {
                throw TokenizerError.invalidMergeFormat(merge)
            }
            ranks[merge] = rank
        }
        self.mergeRanks = ranks

        // Build control token set
        var control = Set<String>()
        let types = tokenTypes.isEmpty ? Array(repeating: 1, count: vocabulary.count) : tokenTypes
        for (id, tokenType) in types.enumerated() where id < vocabulary.count {
            if tokenType == 3 || tokenType == 4 {
                control.insert(vocabulary[id])
            }
        }
        self.controlTokens = control
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

        let words = preTokenizer.split(text: text)
        for word in words {
            let byteEncoded = ByteEncoder.encode(Array(word.utf8))
            let bpeTokens = bpe(token: byteEncoded)
            for token in bpeTokens {
                if let id = tokenToID[token] {
                    result.append(id)
                }
                // If token not found, it is silently dropped.
                // This matches GPT-2 behavior for unknown sub-tokens.
            }
        }

        if specialTokens.addEosToken, let eos = specialTokens.eosTokenID {
            result.append(eos)
        }

        return result
    }

    func decode(tokens: [Int]) -> String {
        var byteEncodedParts: [String] = []
        for id in tokens {
            guard id >= 0, id < vocabulary.count else { continue }
            let tokenString = vocabulary[id]
            // Skip control tokens in output
            if controlTokens.contains(tokenString) { continue }
            byteEncodedParts.append(tokenString)
        }
        let joined = byteEncodedParts.joined()
        let bytes = ByteEncoder.decode(joined)
        return String(decoding: bytes, as: UTF8.self)
    }

    func tokenToString(_ id: Int) -> String? {
        guard id >= 0, id < vocabulary.count else { return nil }
        return vocabulary[id]
    }

    // MARK: - BPE

    /// Run BPE on a byte-encoded token string.
    ///
    /// Iteratively finds the pair with the lowest merge rank and merges
    /// all occurrences until no more merges are possible.
    func bpe(token: String) -> [String] {
        guard token.count > 1 else {
            return token.isEmpty ? [] : [token]
        }

        var word = token.map { String($0) }

        while word.count > 1 {
            // Find pair with lowest merge rank
            var bestRank = Int.max
            var bestFirst = ""
            var bestSecond = ""

            for i in 0..<(word.count - 1) {
                let key = "\(word[i]) \(word[i + 1])"
                if let rank = mergeRanks[key], rank < bestRank {
                    bestRank = rank
                    bestFirst = word[i]
                    bestSecond = word[i + 1]
                }
            }

            guard bestRank < Int.max else { break }

            // Merge all occurrences of the best pair
            let merged = bestFirst + bestSecond
            var newWord: [String] = []
            newWord.reserveCapacity(word.count)
            var i = 0
            while i < word.count {
                if i < word.count - 1 && word[i] == bestFirst && word[i + 1] == bestSecond {
                    newWord.append(merged)
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            word = newWord
        }

        return word
    }
}
