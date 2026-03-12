import Testing
import TestHeartbeat
import Foundation
import GGUFParser
@testable import MLXLM

@Suite("Pack Function Benchmarks", .tags(.performance), .heartbeat)
struct PackBenchmarkTests {

    private let bridge = GGUFTensorBridge()

    // 1024 rows × 256 cols = 262,144 elements per tensor (1024 super-blocks)
    private static let rows = 1024
    private static let cols = 256
    private static let totalElements = rows * cols
    private static let iterations = 5

    private func makeTensor(
        qtype: GGUFQuantizationType, bytesPerSuperBlock: Int
    ) -> (GGUFTensorInfo, Data) {
        let superBlockCount = Self.totalElements / 256
        let dataSize = superBlockCount * bytesPerSuperBlock
        var data = Data(count: dataSize)
        // Fill with non-zero pattern
        for i in 0..<dataSize {
            data[i] = UInt8(truncatingIfNeeded: i &* 7 &+ 13)
        }
        // Write valid f16 scale values (d) at the start of each super-block
        for sb in 0..<superBlockCount {
            let offset = sb * bytesPerSuperBlock
            let d = Float16(0.01)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset..<offset + 2, with: $0)
            }
        }
        let tensor = GGUFTensorInfo(
            name: "bench",
            dimensions: [Self.cols, Self.rows],
            quantizationType: qtype,
            offset: 0
        )
        return (tensor, data)
    }

    private func benchmark(
        label: String, qtype: GGUFQuantizationType, bytesPerSuperBlock: Int
    ) throws {
        let (tensor, data) = makeTensor(qtype: qtype, bytesPerSuperBlock: bytesPerSuperBlock)

        // Warmup
        _ = try bridge.convertDirect(tensor: tensor, data: data)

        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<Self.iterations {
            _ = try bridge.convertDirect(tensor: tensor, data: data)
        }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        let perCall = elapsed / Double(Self.iterations)
        print("[\(label)] \(Self.iterations) calls, total=\(String(format: "%.1f", elapsed))ms, per-call=\(String(format: "%.2f", perCall))ms")
    }

    @Test("Q4_K pack speed")
    func benchQ4_K() throws {
        // Q4_K: 144 bytes per 256-element super-block
        try benchmark(label: "Q4_K", qtype: .q4_K, bytesPerSuperBlock: 144)
    }

    @Test("Q5_K pack speed")
    func benchQ5_K() throws {
        // Q5_K: 176 bytes per 256-element super-block
        try benchmark(label: "Q5_K", qtype: .q5_K, bytesPerSuperBlock: 176)
    }

    @Test("Q6_K pack speed")
    func benchQ6_K() throws {
        // Q6_K: 210 bytes per 256-element super-block
        try benchmark(label: "Q6_K", qtype: .q6_K, bytesPerSuperBlock: 210)
    }

    @Test("Q5_0 pack speed")
    func benchQ5_0() throws {
        // Q5_0: 22 bytes per 32-element block
        let blockCount = Self.totalElements / 32
        let dataSize = blockCount * 22
        var data = Data(count: dataSize)
        for i in 0..<dataSize { data[i] = UInt8(truncatingIfNeeded: i &* 7 &+ 13) }
        for b in 0..<blockCount {
            let offset = b * 22
            let d = Float16(0.01)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset..<offset + 2, with: $0)
            }
        }
        let tensor = GGUFTensorInfo(
            name: "bench", dimensions: [Self.cols, Self.rows],
            quantizationType: .q5_0, offset: 0)

        _ = try bridge.convertDirect(tensor: tensor, data: data)
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<Self.iterations {
            _ = try bridge.convertDirect(tensor: tensor, data: data)
        }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        let perCall = elapsed / Double(Self.iterations)
        print("[Q5_0] \(Self.iterations) calls, total=\(String(format: "%.1f", elapsed))ms, per-call=\(String(format: "%.2f", perCall))ms")
    }

    @Test("Q6_K large tensor (embedding-sized)")
    func benchQ6_K_large() throws {
        // Simulate embedding tensor: 1536 × 8192 = 12.6M elements
        let largeRows = 8192
        let largeCols = 1536
        let total = largeRows * largeCols
        let superBlockCount = total / 256
        let dataSize = superBlockCount * 210
        var data = Data(count: dataSize)
        for i in stride(from: 0, to: dataSize, by: 4) {
            data[i] = UInt8(truncatingIfNeeded: i &* 7 &+ 13)
        }
        for sb in 0..<superBlockCount {
            let offset = sb * 210
            let d = Float16(0.01)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset + 208..<offset + 210, with: $0)
            }
        }
        let tensor = GGUFTensorInfo(
            name: "bench", dimensions: [largeCols, largeRows],
            quantizationType: .q6_K, offset: 0)

        // Warmup
        _ = try bridge.convertDirect(tensor: tensor, data: data)

        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<3 {
            _ = try bridge.convertDirect(tensor: tensor, data: data)
        }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        print("[Q6_K-large] 3 calls, total=\(String(format: "%.0f", elapsed))ms, per-call=\(String(format: "%.0f", elapsed / 3))ms (\(total) elements)")
    }

    @Test("Q4_K large tensor (embedding-sized)")
    func benchQ4_K_large() throws {
        let largeRows = 8192
        let largeCols = 1536
        let total = largeRows * largeCols
        let superBlockCount = total / 256
        let dataSize = superBlockCount * 144
        var data = Data(count: dataSize)
        for i in stride(from: 0, to: dataSize, by: 4) {
            data[i] = UInt8(truncatingIfNeeded: i &* 7 &+ 13)
        }
        for sb in 0..<superBlockCount {
            let offset = sb * 144
            let d = Float16(0.01)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset..<offset + 2, with: $0)
            }
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset + 2..<offset + 4, with: $0)
            }
        }
        let tensor = GGUFTensorInfo(
            name: "bench", dimensions: [largeCols, largeRows],
            quantizationType: .q4_K, offset: 0)

        _ = try bridge.convertDirect(tensor: tensor, data: data)
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<3 {
            _ = try bridge.convertDirect(tensor: tensor, data: data)
        }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        print("[Q4_K-large] 3 calls, total=\(String(format: "%.0f", elapsed))ms, per-call=\(String(format: "%.0f", elapsed / 3))ms (\(total) elements)")
    }

    @Test("Q5_K large tensor (embedding-sized)")
    func benchQ5_K_large() throws {
        let largeRows = 8192
        let largeCols = 1536
        let total = largeRows * largeCols
        let superBlockCount = total / 256
        let dataSize = superBlockCount * 176
        var data = Data(count: dataSize)
        for i in stride(from: 0, to: dataSize, by: 4) {
            data[i] = UInt8(truncatingIfNeeded: i &* 7 &+ 13)
        }
        for sb in 0..<superBlockCount {
            let offset = sb * 176
            let d = Float16(0.01)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset..<offset + 2, with: $0)
            }
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset + 2..<offset + 4, with: $0)
            }
        }
        let tensor = GGUFTensorInfo(
            name: "bench", dimensions: [largeCols, largeRows],
            quantizationType: .q5_K, offset: 0)

        _ = try bridge.convertDirect(tensor: tensor, data: data)
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<3 {
            _ = try bridge.convertDirect(tensor: tensor, data: data)
        }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        print("[Q5_K-large] 3 calls, total=\(String(format: "%.0f", elapsed))ms, per-call=\(String(format: "%.0f", elapsed / 3))ms (\(total) elements)")
    }

    @Test("Q8_0 pack speed (baseline)")
    func benchQ8_0() throws {
        // Q8_0: 34 bytes per 32-element block
        let blockCount = Self.totalElements / 32
        let dataSize = blockCount * 34
        var data = Data(count: dataSize)
        for i in 0..<dataSize { data[i] = UInt8(truncatingIfNeeded: i &* 7 &+ 13) }
        for b in 0..<blockCount {
            let offset = b * 34
            let d = Float16(0.01)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                data.replaceSubrange(offset..<offset + 2, with: $0)
            }
        }
        let tensor = GGUFTensorInfo(
            name: "bench", dimensions: [Self.cols, Self.rows],
            quantizationType: .q8_0, offset: 0)

        _ = try bridge.convertDirect(tensor: tensor, data: data)
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<Self.iterations {
            _ = try bridge.convertDirect(tensor: tensor, data: data)
        }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        let perCall = elapsed / Double(Self.iterations)
        print("[Q8_0] \(Self.iterations) calls, total=\(String(format: "%.1f", elapsed))ms, per-call=\(String(format: "%.2f", perCall))ms")
    }
}
