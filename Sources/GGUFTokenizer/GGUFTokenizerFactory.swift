import GGUFParser

/// Factory for creating tokenizers from GGUF file metadata.
///
/// Reads `tokenizer.ggml.*` metadata keys and constructs the appropriate
/// tokenizer type (merges-based BPE or SentencePiece BPE).
public enum GGUFTokenizerFactory {

    /// Create a tokenizer from a parsed GGUF file.
    ///
    /// Inspects `tokenizer.ggml.model` to determine the tokenizer type:
    /// - `"gpt2"`: Merges-based BPE (Llama 3, Qwen2, GPT-2)
    /// - `"llama"`: SentencePiece BPE (Llama 1/2)
    public static func create(from file: GGUFFile) throws -> any Tokenizer {
        guard let vocabulary = file.tokens else {
            throw TokenizerError.missingVocabulary
        }

        let specialTokens = SpecialTokens(
            bosTokenID: file.bosTokenID,
            eosTokenID: file.eosTokenID,
            unknownTokenID: findUnknownTokenID(vocabulary: vocabulary, tokenTypes: file.tokenTypes),
            paddingTokenID: file.paddingTokenID,
            addBosToken: file.addBosToken ?? false,
            addEosToken: file.addEosToken ?? false
        )

        let tokenTypes = file.tokenTypes ?? []

        let model = file.tokenizerModel ?? "gpt2"
        switch model {
        case "gpt2":
            return try createMergesBPE(
                vocabulary: vocabulary,
                file: file,
                specialTokens: specialTokens,
                tokenTypes: tokenTypes
            )
        case "llama":
            return try createSentencePieceBPE(
                vocabulary: vocabulary,
                file: file,
                specialTokens: specialTokens,
                tokenTypes: tokenTypes
            )
        default:
            throw TokenizerError.unsupportedModel(model)
        }
    }

    // MARK: - Private

    private static func createMergesBPE(
        vocabulary: [String],
        file: GGUFFile,
        specialTokens: SpecialTokens,
        tokenTypes: [Int]
    ) throws -> MergesBPETokenizer {
        guard let merges = file.merges else {
            throw TokenizerError.missingMerges
        }

        let preTokenizer = PreTokenizer.forType(file.preTokenizer)

        return try MergesBPETokenizer(
            vocabulary: vocabulary,
            merges: merges,
            preTokenizer: preTokenizer,
            specialTokens: specialTokens,
            tokenTypes: tokenTypes
        )
    }

    private static func createSentencePieceBPE(
        vocabulary: [String],
        file: GGUFFile,
        specialTokens: SpecialTokens,
        tokenTypes: [Int]
    ) throws -> SentencePieceBPETokenizer {
        guard let scores = file.tokenScores else {
            throw TokenizerError.missingScores
        }

        return try SentencePieceBPETokenizer(
            vocabulary: vocabulary,
            scores: scores,
            specialTokens: specialTokens,
            tokenTypes: tokenTypes,
            addSpacePrefix: true
        )
    }

    private static func findUnknownTokenID(vocabulary: [String], tokenTypes: [Int]?) -> Int? {
        // First check token_type == 2 (UNKNOWN)
        if let types = tokenTypes {
            for (id, tokenType) in types.enumerated() {
                if tokenType == 2 { return id }
            }
        }
        // Fallback: look for common unknown token strings
        let unknownTokenStrings = ["<unk>", "<|unk|>", "[UNK]"]
        for unk in unknownTokenStrings {
            if let id = vocabulary.firstIndex(of: unk) {
                return id
            }
        }
        return nil
    }
}
