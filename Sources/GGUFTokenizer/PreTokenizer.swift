import Foundation

/// Regex-based pre-tokenizer that splits text into word-level segments
/// before BPE is applied.
///
/// Different model families use different regex patterns to determine
/// word boundaries, contraction handling, and number grouping.
struct PreTokenizer: Sendable {

    // Regex is immutable after initialization, safe to share across threads.
    nonisolated(unsafe) private let regex: Regex<AnyRegexOutput>

    /// Create a pre-tokenizer with the given regex pattern.
    init(pattern: String) throws {
        self.regex = try Regex(pattern)
    }

    /// Split text into segments using the pre-tokenizer regex.
    func split(text: String) -> [String] {
        guard !text.isEmpty else { return [] }
        return text.matches(of: regex).map { String(text[$0.range]) }
    }

    // MARK: - Pre-defined Patterns

    // GPT-2 pre-tokenizer pattern.
    // Handles contractions, words, numbers, punctuation, whitespace.
    static let gpt2Pattern =
        #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#

    // Llama 3 BPE pre-tokenizer pattern.
    // Case-insensitive contractions, numbers grouped up to 3 digits.
    static let llamaBPEPattern =
        #"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    // Qwen2 pre-tokenizer pattern.
    // Same as Llama 3 but numbers are split per-digit (\p{N} vs \p{N}{1,3}).
    static let qwen2Pattern =
        #"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    /// GPT-2 pre-tokenizer.
    static let gpt2: PreTokenizer = try! PreTokenizer(pattern: gpt2Pattern)

    /// Llama 3 BPE pre-tokenizer.
    static let llamaBPE: PreTokenizer = try! PreTokenizer(pattern: llamaBPEPattern)

    /// Qwen2 pre-tokenizer.
    static let qwen2: PreTokenizer = try! PreTokenizer(pattern: qwen2Pattern)

    /// Select pre-tokenizer based on the `tokenizer.ggml.pre` metadata value.
    static func forType(_ type: String?) -> PreTokenizer {
        switch type {
        case "llama-bpe", "llama3", "llama-v3":
            return .llamaBPE
        case "qwen2", "stablelm2", "deepseek-r1-qwen":
            return .qwen2
        case "gpt2", "phi-2", "roberta-bpe":
            return .gpt2
        case "command-r", "starcoder", "smollm":
            return .gpt2
        case .none, "default":
            return .gpt2
        default:
            return .gpt2
        }
    }
}
