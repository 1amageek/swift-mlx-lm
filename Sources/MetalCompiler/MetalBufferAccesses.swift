import Metal

public struct MetalBufferAccesses: @unchecked Sendable {
    public let reads: Set<ObjectIdentifier>
    public let writes: Set<ObjectIdentifier>

    public init(reads: Set<ObjectIdentifier>, writes: Set<ObjectIdentifier>) {
        self.reads = reads
        self.writes = writes
    }

    public init(readBuffers: [MTLBuffer], writeBuffers: [MTLBuffer]) {
        self.reads = Set(readBuffers.map { ObjectIdentifier($0) })
        self.writes = Set(writeBuffers.map { ObjectIdentifier($0) })
    }

    public static func conservative(_ bindings: [MetalBufferBinding]) -> Self {
        let identifiers = Set(bindings.map { ObjectIdentifier($0.buffer) })
        return Self(reads: identifiers, writes: identifiers)
    }

    public func requiresBarrier(after pendingWrites: Set<ObjectIdentifier>) -> Bool {
        !pendingWrites.isDisjoint(with: reads.union(writes))
    }
}
