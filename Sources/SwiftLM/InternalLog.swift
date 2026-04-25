import Foundation

/// Internal logging helpers for SwiftLM.
///
/// - `info`: telemetry output (load times, throughput). Silent by default;
///   enable with `SWIFTLM_VERBOSE=1`.
/// - `error`: failure diagnostics. Always written to stderr so callers that
///   ignore `throws` still see the reason (no silent fallback).
enum InternalLog {
    private static let verboseEnabled: Bool = {
        ProcessInfo.processInfo.environment["SWIFTLM_VERBOSE"] == "1"
    }()

    static let generationDebugEnabled: Bool = {
        ProcessInfo.processInfo.environment["SWIFTLM_DEBUG_GENERATION"] == "1"
    }()

    static let generationDebugTokenLimit: Int = {
        guard let value = ProcessInfo.processInfo.environment["SWIFTLM_DEBUG_GENERATION_TOKEN_LIMIT"] else {
            return 128
        }
        guard let parsed = Int(value), parsed >= 0 else {
            return 128
        }
        return parsed
    }()

    static func info(_ message: @autoclosure () -> String) {
        guard verboseEnabled else { return }
        print(message())
    }

    static func generationDebug(_ message: @autoclosure () -> String) {
        guard generationDebugEnabled else { return }
        let line = message() + "\n"
        if let data = line.data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    }

    static func error(_ message: @autoclosure () -> String) {
        let line = message() + "\n"
        if let data = line.data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    }
}
