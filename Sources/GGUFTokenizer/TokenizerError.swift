/// Errors that can occur during tokenizer construction or operation.
public enum TokenizerError: Error, Sendable {
    case missingVocabulary
    case missingMerges
    case missingScores
    case unsupportedModel(String)
    case invalidMergeFormat(String)
}
