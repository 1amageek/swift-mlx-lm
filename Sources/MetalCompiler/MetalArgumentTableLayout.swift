public struct MetalArgumentTableLayout: Sendable, Hashable {
    public let id: Int
    public let indices: [Int]

    public init(id: Int, indices: [Int]) {
        self.id = id
        self.indices = indices
    }
}
