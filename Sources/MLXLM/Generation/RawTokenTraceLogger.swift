import Foundation

/// Logs prompt and sampled token traces for tokenizer/debugging investigations.
///
/// Enable with `SWIFT_MLX_LM_LOG_RAW_TOKENS=1`.
struct RawTokenTraceLogger: Sendable {

    private let tokenizer: any Tokenizer
    private let enabled: Bool
    private let promptTokenLimit: Int
    private let outputTokenLimit: Int?

    init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer

        let environment = ProcessInfo.processInfo.environment
        self.enabled = Self.resolveEnabled(environment)
        self.promptTokenLimit = Int(environment["SWIFT_MLX_LM_LOG_PROMPT_LIMIT"] ?? "") ?? 128

        if let rawLimit = environment["SWIFT_MLX_LM_LOG_OUTPUT_LIMIT"],
           let limit = Int(rawLimit),
           limit >= 0 {
            self.outputTokenLimit = limit
        } else {
            self.outputTokenLimit = 128
        }
    }

    func logPrompt(prompt: String, tokens: [Int]) {
        guard enabled else { return }

        print("[raw-token][prompt] chars=\(prompt.count) tokens=\(tokens.count)")
        for (index, token) in tokens.prefix(promptTokenLimit).enumerated() {
            let piece = tokenizer.tokenToString(token) ?? ""
            let decoded = tokenizer.decode(tokens: [token])
            print(
                "[raw-token][prompt] index=\(index) id=\(token) piece=\(Self.quote(piece)) decoded=\(Self.quote(decoded))"
            )
        }

        if tokens.count > promptTokenLimit {
            print("[raw-token][prompt] truncated=\(tokens.count - promptTokenLimit)")
        }
    }

    func logOutputToken(step: Int, token: Int, chunk: String?) {
        guard enabled else { return }

        if let outputTokenLimit {
            guard step <= outputTokenLimit else {
                if step == outputTokenLimit + 1 {
                    print("[raw-token][output] truncated=true remaining=unknown")
                }
                return
            }
        }

        let piece = tokenizer.tokenToString(token) ?? ""
        let decoded = tokenizer.decode(tokens: [token])
        let chunkValue = chunk ?? ""

        print(
            "[raw-token][output] step=\(step) id=\(token) piece=\(Self.quote(piece)) decoded=\(Self.quote(decoded)) chunk=\(Self.quote(chunkValue))"
        )
    }

    private static func parseFlag(_ value: String?) -> Bool {
        guard let value else { return false }
        switch value.lowercased() {
        case "1", "true", "yes", "on":
            return true
        default:
            return false
        }
    }

    private static func resolveEnabled(_ environment: [String: String]) -> Bool {
        if environment["XCTestConfigurationFilePath"] != nil {
            return parseFlag(environment["SWIFT_MLX_LM_LOG_RAW_TOKENS"])
        }

        if environment["SWIFT_MLX_LM_LOG_RAW_TOKENS"] != nil {
            return parseFlag(environment["SWIFT_MLX_LM_LOG_RAW_TOKENS"])
        }

        #if DEBUG
        return true
        #else
        return false
        #endif
    }

    private static func quote(_ string: String) -> String {
        var escaped = "\""
        for scalar in string.unicodeScalars {
            switch scalar.value {
            case 0x0A:
                escaped += "\\n"
            case 0x0D:
                escaped += "\\r"
            case 0x09:
                escaped += "\\t"
            case 0x22:
                escaped += "\\\""
            case 0x5C:
                escaped += "\\\\"
            default:
                escaped.unicodeScalars.append(scalar)
            }
        }
        escaped += "\""
        return escaped
    }
}
