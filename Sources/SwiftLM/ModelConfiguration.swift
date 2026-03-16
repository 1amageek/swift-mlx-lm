import Foundation

/// Identity and configuration of a loaded model.
public struct ModelConfiguration: Sendable {
    /// Model name (from config.json or directory name).
    public var name: String
    /// EOS token IDs for stopping generation.
    public var eosTokenIds: Set<Int>

    public init(name: String = "model", eosTokenIds: Set<Int> = []) {
        self.name = name
        self.eosTokenIds = eosTokenIds
    }
}
