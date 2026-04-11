import Testing
import Metal
import Foundation
@testable import MetalCompiler
import LMArchitecture
import ModelDeclarations

/// Isolated suite for Metal 4 decode token quality.
/// Runs separately from BenchmarkDiagnosticsTests to avoid GPU cache interference
/// from accumulated hazardTrackingModeUntracked models in large suites.
@Suite("Metal 4 Token Quality", .serialized)
struct Metal4TokenQualityTests {

    @Test("Metal 4 decode produces valid tokens")
    func metal4DecodeProducesValidTokens() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let (model, _) = try BenchmarkSupport.setupOrSkip(optimizer: AggressiveOptimizer())
        var m = model
        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        let decodeSteps = 30

        // Metal 4 path: decodeSync (argument table + barrier)
        var tokens: [Int32] = []
        var tok = m.prefill(tokens: promptTokens)
        for _ in 0..<decodeSteps {
            tok = m.decodeSync(tokenID: tok)
            tokens.append(tok)
        }

        // Determinism: reset and decode again, should produce identical tokens
        m.resetState()
        var tokens2: [Int32] = []
        tok = m.prefill(tokens: promptTokens)
        for _ in 0..<decodeSteps {
            tok = m.decodeSync(tokenID: tok)
            tokens2.append(tok)
        }

        let nonZeroCount = tokens.filter { $0 != 0 }.count
        let nonZeroRatio = Double(nonZeroCount) / Double(decodeSteps)
        let deterministic = tokens == tokens2

        print("\n=== Metal 4 Decode Token Quality ===")
        print("  Tokens: \(decodeSteps)")
        print("  Non-zero: \(nonZeroCount) / \(decodeSteps) (\(String(format: "%.0f", nonZeroRatio * 100))%)")
        print("  Deterministic: \(deterministic)")
        print("  First 10: \(tokens.prefix(10).map(String.init).joined(separator: ", "))")

        #expect(nonZeroRatio > 0.5, "Most tokens should be non-zero, got \(nonZeroCount)/\(decodeSteps)")
        #expect(deterministic, "Metal 4 decode should be deterministic across resets")
    }
}
