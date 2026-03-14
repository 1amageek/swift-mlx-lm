/// A flexible JSON value that can be a string, number, boolean, or array thereof.
///
/// Used for model configuration values like RoPE scaling parameters that
/// can be different types in different model configs.
enum StringOrNumber: Codable, Equatable, Sendable {
    case string(String)
    case int(Int)
    case float(Float)
    case ints([Int])
    case floats([Float])
    case bool(Bool)

    init(from decoder: Decoder) throws {
        let values = try decoder.singleValueContainer()
        if let v = try? values.decode(Int.self) {
            self = .int(v)
        } else if let v = try? values.decode(Float.self) {
            self = .float(v)
        } else if let v = try? values.decode([Int].self) {
            self = .ints(v)
        } else if let v = try? values.decode([Float].self) {
            self = .floats(v)
        } else if let v = try? values.decode(Bool.self) {
            self = .bool(v)
        } else {
            let v = try values.decode(String.self)
            self = .string(v)
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let v): try container.encode(v)
        case .int(let v): try container.encode(v)
        case .float(let v): try container.encode(v)
        case .ints(let v): try container.encode(v)
        case .floats(let v): try container.encode(v)
        case .bool(let v): try container.encode(v)
        }
    }

    func asFloat() -> Float? {
        switch self {
        case .float(let v): return v
        case .int(let v): return Float(v)
        default: return nil
        }
    }

    func asInt() -> Int? {
        switch self {
        case .int(let v): return v
        default: return nil
        }
    }

    func asInts() -> [Int]? {
        switch self {
        case .ints(let v): return v
        case .int(let v): return [v]
        default: return nil
        }
    }

    func asFloats() -> [Float]? {
        switch self {
        case .floats(let v): return v
        case .float(let v): return [v]
        default: return nil
        }
    }
}
