import Foundation
import GGUFParser
import Synchronization

/// Factory for creating tokenizers from GGUF file metadata.
///
/// Reads `tokenizer.ggml.*` metadata keys and constructs the appropriate
/// tokenizer type (merges-based BPE or SentencePiece BPE).
///
/// When deferred arrays are available (large GGUF files), reads them
/// concurrently from different mmap byte regions to exploit I/O parallelism.
public enum GGUFTokenizerFactory {

    /// Create a tokenizer from a parsed GGUF file.
    ///
    /// Inspects `tokenizer.ggml.model` to determine the tokenizer type:
    /// - `"gpt2"`: Merges-based BPE (Llama 3, Qwen2, GPT-2)
    /// - `"llama"`: SentencePiece BPE (Llama 1/2)
    public static func create(from file: GGUFFile) throws -> any Tokenizer {
        let model = file.tokenizerModel ?? "gpt2"

        switch model {
        case "gpt2":
            return try createMergesBPE(file: file)
        case "llama":
            return try createSentencePieceBPE(file: file)
        default:
            throw TokenizerError.unsupportedModel(model)
        }
    }

    // MARK: - Private

    private static func createMergesBPE(file: GGUFFile) throws -> MergesBPETokenizer {
        let preTokenizer = PreTokenizer.forType(file.preTokenizer)

        // Fast path: read deferred arrays concurrently from different mmap regions.
        let hasVocab = file.deferredArrayCount("tokenizer.ggml.tokens") != nil
        let hasMerges = file.deferredArrayCount("tokenizer.ggml.merges") != nil

        if hasVocab, hasMerges {
            // Read vocab, merges, and token_types concurrently.
            // Each reads from a disjoint byte range in the mmap buffer.
            let vocabResult = Mutex<(vocabulary: [String], tokenToID: [String: Int])?>(nil)
            let mergesResult = Mutex<[String: Int]?>(nil)
            let typesResult = Mutex<[Int]?>(nil)

            DispatchQueue.concurrentPerform(iterations: 3) { index in
                switch index {
                case 0:
                    let v = file.readDeferredVocabulary("tokenizer.ggml.tokens")
                    vocabResult.withLock { $0 = v }
                case 1:
                    let m = file.readDeferredStringDictionary("tokenizer.ggml.merges")
                    mergesResult.withLock { $0 = m }
                case 2:
                    let t = file.readDeferredInt32Array("tokenizer.ggml.token_type")
                    typesResult.withLock { $0 = t }
                default:
                    break
                }
            }

            if let vocabData = vocabResult.withLock({ $0 }),
               let mergeRanks = mergesResult.withLock({ $0 }) {
                let tokenTypes = typesResult.withLock { $0 } ?? []
                let specialTokens = makeSpecialTokens(
                    file: file, vocabulary: vocabData.vocabulary, tokenTypes: tokenTypes)

                return MergesBPETokenizer(
                    vocabulary: vocabData.vocabulary,
                    tokenToID: vocabData.tokenToID,
                    mergeRanks: mergeRanks,
                    preTokenizer: preTokenizer,
                    specialTokens: specialTokens,
                    tokenTypes: tokenTypes
                )
            }
        }

        // Fallback: standard path (non-deferred files, e.g. tests with inline Data)
        guard let vocabulary = file.tokens else {
            throw TokenizerError.missingVocabulary
        }
        guard let merges = file.merges else {
            throw TokenizerError.missingMerges
        }
        let tokenTypes = file.tokenTypes ?? []
        let specialTokens = makeSpecialTokens(
            file: file, vocabulary: vocabulary, tokenTypes: tokenTypes)

        return try MergesBPETokenizer(
            vocabulary: vocabulary,
            merges: merges,
            preTokenizer: preTokenizer,
            specialTokens: specialTokens,
            tokenTypes: tokenTypes
        )
    }

    private static func createSentencePieceBPE(file: GGUFFile) throws -> SentencePieceBPETokenizer {
        // Fast path: read deferred arrays concurrently
        let hasVocab = file.deferredArrayCount("tokenizer.ggml.tokens") != nil
        let hasScores = file.deferredArrayCount("tokenizer.ggml.scores") != nil

        if hasVocab, hasScores {
            let vocabResult = Mutex<(vocabulary: [String], tokenToID: [String: Int])?>(nil)
            let scoresResult = Mutex<[Float]?>(nil)
            let typesResult = Mutex<[Int]?>(nil)

            DispatchQueue.concurrentPerform(iterations: 3) { index in
                switch index {
                case 0:
                    let v = file.readDeferredVocabulary("tokenizer.ggml.tokens")
                    vocabResult.withLock { $0 = v }
                case 1:
                    let s = file.readDeferredFloat32Array("tokenizer.ggml.scores")
                    scoresResult.withLock { $0 = s }
                case 2:
                    let t = file.readDeferredInt32Array("tokenizer.ggml.token_type")
                    typesResult.withLock { $0 = t }
                default:
                    break
                }
            }

            if let vocabData = vocabResult.withLock({ $0 }),
               let scores = scoresResult.withLock({ $0 }) {
                let tokenTypes = typesResult.withLock { $0 } ?? []
                let specialTokens = makeSpecialTokens(
                    file: file, vocabulary: vocabData.vocabulary, tokenTypes: tokenTypes)

                return try SentencePieceBPETokenizer(
                    vocabulary: vocabData.vocabulary,
                    scores: scores,
                    specialTokens: specialTokens,
                    tokenTypes: tokenTypes,
                    addSpacePrefix: true
                )
            }
        }

        // Fallback
        guard let vocabulary = file.tokens else {
            throw TokenizerError.missingVocabulary
        }
        guard let scores = file.tokenScores else {
            throw TokenizerError.missingScores
        }
        let tokenTypes = file.tokenTypes ?? []
        let specialTokens = makeSpecialTokens(
            file: file, vocabulary: vocabulary, tokenTypes: tokenTypes)

        return try SentencePieceBPETokenizer(
            vocabulary: vocabulary,
            scores: scores,
            specialTokens: specialTokens,
            tokenTypes: tokenTypes,
            addSpacePrefix: true
        )
    }

    private static func makeSpecialTokens(
        file: GGUFFile, vocabulary: [String], tokenTypes: [Int]
    ) -> SpecialTokens {
        SpecialTokens(
            bosTokenID: file.bosTokenID,
            eosTokenID: file.eosTokenID,
            unknownTokenID: findUnknownTokenID(vocabulary: vocabulary, tokenTypes: tokenTypes),
            paddingTokenID: file.paddingTokenID,
            addBosToken: file.addBosToken ?? false,
            addEosToken: file.addEosToken ?? false
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
