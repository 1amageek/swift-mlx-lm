import Foundation

/// Model metadata extracted from GGUF and user overrides.
public struct ModelConfiguration: Sendable {

    /// Display name (typically derived from GGUF filename).
    public var name: String

    /// EOS token IDs that signal generation termination.
    public var eosTokenIds: Set<Int>

    /// Additional EOS tokens specified by string (resolved to IDs at load time).
    public var extraEOSTokens: Set<String>

    /// Tool call output format (nil = no tool calling).
    public var toolCallFormat: ToolCallFormat?

    public init(
        name: String,
        eosTokenIds: Set<Int> = [],
        extraEOSTokens: Set<String> = [],
        toolCallFormat: ToolCallFormat? = nil
    ) {
        self.name = name
        self.eosTokenIds = eosTokenIds
        self.extraEOSTokens = extraEOSTokens
        self.toolCallFormat = toolCallFormat
    }
}
