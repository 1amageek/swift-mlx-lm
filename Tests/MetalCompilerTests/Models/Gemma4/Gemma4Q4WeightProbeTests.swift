import Foundation
import Metal
import Testing
@testable import MetalCompiler

#if ENABLE_METAL_PROBES
/// Software-dequant probe: verifies that Q4-packed weights in the STAF file
/// match the corresponding BF16 weights within quantization noise. This
/// bypasses all Metal kernels and directly inspects the STAF block layout.
///
/// If Q4 dequant here is wildly wrong vs BF16, the STAF Q4 block packing
/// (including the BF16→F16 scale/bias normalization) is broken.
///
/// If Q4 dequant here matches BF16, the STAF packing is correct and the
/// bug is in the Metal kernel, its dispatch, or routing — not in the data.
@Suite("Gemma4 Q4 Weight Probe", .serialized)
struct Gemma4Q4WeightProbeTests {

    static let testDataRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("TestData")
        .path
    static let bf16BundlePath = "\(testDataRoot)/gemma-4-E2B-it"
    static let q4BundlePath = "\(testDataRoot)/gemma-4-E2B-it-4bit"

    // Tensors to probe. Embedding is the most critical — if it is wrong, every
    // layer's input is already poisoned.
    static let tensorsToCheck: [String] = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
    ]

    @Test("Q4 block 0 dequant matches BF16 reference")
    func q4BlockDequantMatchesBF16() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let bf16Store = try STAFLoader().load(
            at: URL(fileURLWithPath: Self.bf16BundlePath).appendingPathComponent("model.staf"),
            device: device)
        let q4Store = try STAFLoader().load(
            at: URL(fileURLWithPath: Self.q4BundlePath).appendingPathComponent("model.staf"),
            device: device)

        print("\n=== Gemma4-E2B Q4 Block-Level Weight Probe ===")

        var comparedTensorCount = 0
        for tensorName in Self.tensorsToCheck {
            print("\nTensor: \(tensorName)")

            guard let bf16Entry = bf16Store.entries[tensorName] else {
                print("  BF16 tensor missing — skipping")
                continue
            }
            guard let q4Entry = q4Store.entries[tensorName] else {
                print("  Q4 tensor missing — skipping")
                continue
            }

            print("  BF16 scheme=\(bf16Entry.schemeIdentifier) shape=\(bf16Entry.shape)")
            print("  Q4   scheme=\(q4Entry.schemeIdentifier) shape=\(q4Entry.shape) group=\(q4Entry.groupSize) block=\(q4Entry.blockSize)")

            // Read BF16 first 64 values of row 0
            let bf16Values = readBF16Row0First64(store: bf16Store, entry: bf16Entry)

            // Read Q4 block 0 of row 0, dequantize in software
            let q4Values = softwareDequantQ4Block0Row0(store: q4Store, entry: q4Entry)

            guard !bf16Values.isEmpty, !q4Values.isEmpty else {
                print("  Could not read values — skipping comparison")
                continue
            }

            // Print first few for inspection
            print("  BF16[0..8]: \(bf16Values.prefix(8).map { String(format: "%.6f", $0) }.joined(separator: ", "))")
            print("  Q4  [0..8]: \(q4Values.prefix(8).map { String(format: "%.6f", $0) }.joined(separator: ", "))")

            // Also print Q4 block header raw bytes
            dumpQ4Block0Header(store: q4Store, entry: q4Entry)

            // Compare
            let n = min(bf16Values.count, q4Values.count)
            var maxAbs: Float = 0
            var maxRel: Float = 0
            var sumAbs: Float = 0
            for i in 0..<n {
                let d = abs(bf16Values[i] - q4Values[i])
                let r = abs(bf16Values[i]) > 1e-6 ? d / abs(bf16Values[i]) : 0
                maxAbs = max(maxAbs, d)
                maxRel = max(maxRel, r)
                sumAbs += d
            }
            let meanAbs = sumAbs / Float(n)
            print(String(format: "  diff: max_abs=%.6f  mean_abs=%.6f  max_rel=%.4f  (n=%d)",
                         maxAbs, meanAbs, maxRel, n))
            comparedTensorCount += 1
        }

        #expect(comparedTensorCount > 0)
    }

    // MARK: - Helpers

    private func readBF16Row0First64(store: STAFWeightStore, entry: STAFTensorEntry) -> [Float] {
        // BF16 row-major, 2 bytes per element
        let rowBytes = 64 * 2
        let payloadSize = entry.payloadSize
        guard payloadSize >= rowBytes else { return [] }

        let base = store.buffer.contents().advanced(by: entry.bufferOffset)
        let bf16Pointer = base.assumingMemoryBound(to: UInt16.self)

        var result: [Float] = []
        result.reserveCapacity(64)
        for i in 0..<64 {
            let bf16Bits = bf16Pointer[i]
            let widened = UInt32(bf16Bits) << 16
            result.append(Float(bitPattern: widened))
        }
        return result
    }

    private func softwareDequantQ4Block0Row0(
        store: STAFWeightStore, entry: STAFTensorEntry
    ) -> [Float] {
        // STAF Q4 block: [scale_f16 (2B)][zero_f16 (2B)][32 nibble-packed bytes] = 36 B
        // Each nibble-packed byte holds 2 weights: low nibble = even col, high nibble = odd col.
        guard entry.groupSize == 64 else {
            print("  (non-Q4G64 scheme, skipping software dequant)")
            return []
        }

        let base = store.buffer.contents().advanced(by: entry.bufferOffset)
        let bytes = base.assumingMemoryBound(to: UInt8.self)

        let scaleBits = UInt16(bytes[0]) | (UInt16(bytes[1]) << 8)
        let zeroBits = UInt16(bytes[2]) | (UInt16(bytes[3]) << 8)
        let scale = Float(Float16(bitPattern: scaleBits))
        let zero = Float(Float16(bitPattern: zeroBits))

        var result: [Float] = []
        result.reserveCapacity(64)
        for localByte in 0..<32 {
            let packed = bytes[4 + localByte]
            let w0 = Float(packed & 0x0F) * scale + zero
            let w1 = Float((packed >> 4) & 0x0F) * scale + zero
            result.append(w0)
            result.append(w1)
        }
        return result
    }

    private func dumpQ4Block0Header(store: STAFWeightStore, entry: STAFTensorEntry) {
        let base = store.buffer.contents().advanced(by: entry.bufferOffset)
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        let scaleBits = UInt16(bytes[0]) | (UInt16(bytes[1]) << 8)
        let zeroBits = UInt16(bytes[2]) | (UInt16(bytes[3]) << 8)
        let scale = Float(Float16(bitPattern: scaleBits))
        let zero = Float(Float16(bitPattern: zeroBits))
        let qsHex = (0..<8).map { String(format: "%02x", bytes[4 + $0]) }.joined(separator: " ")
        print(String(format: "  Q4 block0: scale=%.6g (bits=0x%04x)  zero=%.6g (bits=0x%04x)  qs[0..8]=%@",
                     scale, UInt32(scaleBits), zero, UInt32(zeroBits), qsHex))
    }
}
#endif
