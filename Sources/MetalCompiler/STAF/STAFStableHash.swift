import Foundation

struct STAFStableHash64 {
    private static let offsetBasis: UInt64 = 0xcbf29ce484222325
    private static let prime: UInt64 = 0x00000100000001B3

    private var state: UInt64 = offsetBasis

    mutating func update(_ data: Data) {
        data.withUnsafeBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                return
            }
            update(bytes: UnsafeBufferPointer(start: baseAddress, count: rawBuffer.count))
        }
    }

    mutating func update(_ string: String) {
        update(Data(string.utf8))
    }

    mutating func update(uint64 value: UInt64) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                return
            }
            update(bytes: UnsafeBufferPointer(start: baseAddress, count: rawBuffer.count))
        }
    }

    func finalize() -> UInt64 {
        state
    }

    private mutating func update(bytes: UnsafeBufferPointer<UInt8>) {
        for byte in bytes {
            state ^= UInt64(byte)
            state &*= Self.prime
        }
    }
}
