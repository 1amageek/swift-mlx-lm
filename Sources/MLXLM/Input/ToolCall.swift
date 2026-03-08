/// A parsed tool/function call from model output.
public struct ToolCall: Hashable, Codable, Sendable {

    public struct Function: Hashable, Codable, Sendable {
        public let name: String
        /// Raw JSON string of the arguments.
        public let arguments: String

        public init(name: String, arguments: String) {
            self.name = name
            self.arguments = arguments
        }
    }

    public let function: Function

    public init(function: Function) {
        self.function = function
    }
}

/// Tool specification as a JSON-compatible dictionary.
public typealias ToolSpec = [String: any Sendable]

/// Identifier for tool call output format.
public struct ToolCallFormat: RawRepresentable, Hashable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    public static let json = ToolCallFormat(rawValue: "json")
    public static let xmlFunction = ToolCallFormat(rawValue: "xmlFunction")
    public static let lfm2 = ToolCallFormat(rawValue: "lfm2")
    public static let glm4 = ToolCallFormat(rawValue: "glm4")
    public static let gemma = ToolCallFormat(rawValue: "gemma")
}
