import Testing
import Foundation
import GGUFParser
@testable import MLXLM

@Suite("Pack Function Benchmarks", .tags(.performance))
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
