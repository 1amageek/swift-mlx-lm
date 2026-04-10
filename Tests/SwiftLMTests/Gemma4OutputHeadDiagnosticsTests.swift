import Foundation
import Metal
import Testing
@testable import MetalCompiler
@testable import SwiftLM

@Suite("Gemma4 Output Head Diagnostics", .serialized)
struct Gemma4OutputHeadDiagnosticsTests {
    @Test("Manual embedding logits from transferred final hidden are inspectable", .timeLimit(.minutes(10)))
    func manualEmbeddingLogitsFromTransferredFinalHidden() async throws {
        guard let container = try await Gemma4TestSupport.realGemma4Container() else {
            print("[Skip] No local official Gemma4 E2B snapshot found")
            return
        }

        container.resetCaches()
        let prepared = try await container.prepare(
            input: ModelInput(prompt: RealOutputAssertionSupport.strictCapitalPrompt)
        )
        let prompt = try container.makeExecutablePrompt(from: prepared)
        let diagnostics = try container.debugPrefillOutputHeadDiagnostics(prompt: prompt, topK: 10)

        guard let directory = try Gemma4TestSupport.optionalRealGemma4Directory(),
              let device = MTLCreateSystemDefaultDevice()
        else {
            Issue.record("Gemma4 diagnostics require a local model directory and Metal device")
            return
        }

        let safetensorURLs = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        )
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
        let weights = try SafetensorsLoader().loadAll(urls: safetensorURLs, device: device)
        guard let embeddingTensor = weights.tensor(for: "model.language_model.embed_tokens.weight") else {
            Issue.record("Missing Gemma4 embedding tensor")
            return
        }

        var candidateTokenIDs = diagnostics.topLogits.map(\.tokenID)
        candidateTokenIDs.append(contentsOf: container.tokenizer.encode(text: "Tokyo", addSpecialTokens: false))
        candidateTokenIDs.append(contentsOf: container.tokenizer.encode(text: " Tokyo", addSpecialTokens: false))
        candidateTokenIDs = Array(Set(candidateTokenIDs)).sorted()

        print("[Gemma4 GPU top logits]")
        for entry in diagnostics.topLogits {
            print("  id=\(entry.tokenID) logit=\(String(format: "%.4f", entry.logit)) token=\(String(reflecting: entry.decoded))")
        }

        print("[Gemma4 manual logits from transferred final hidden]")
        var ranked: [(tokenID: Int, logit: Float, decoded: String)] = []
        for tokenID in candidateTokenIDs {
            let logit = dotEmbeddingRow(
                tokenID: tokenID,
                hidden: diagnostics.transferDestination,
                tensor: embeddingTensor
            )
            let decoded = container.tokenizer.decode(tokens: [tokenID], skipSpecialTokens: false)
            ranked.append((tokenID: tokenID, logit: logit, decoded: decoded))
            print("  id=\(tokenID) logit=\(String(format: "%.4f", logit)) token=\(String(reflecting: decoded))")
        }

        ranked.sort {
            if $0.logit == $1.logit {
                return $0.tokenID < $1.tokenID
            }
            return $0.logit > $1.logit
        }
        print("[Gemma4 manual candidate ranking]")
        for entry in ranked.prefix(10) {
            print("  id=\(entry.tokenID) logit=\(String(format: "%.4f", entry.logit)) token=\(String(reflecting: entry.decoded))")
        }
    }

    private func dotEmbeddingRow(
        tokenID: Int,
        hidden: [Float],
        tensor: MetalTensor
    ) -> Float {
        guard tensor.shape.count >= 2 else { return .zero }
        let hiddenSize = hidden.count
        let basePointer = tensor.buffer.contents().advanced(by: tensor.offset)

        switch tensor.dtype {
        case .float16:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: tensor.shape.reduce(1, *))
            let start = tokenID * hiddenSize
            return zip(0..<hiddenSize, hidden).reduce(Float.zero) { partial, pair in
                partial + Float(Float16(bitPattern: pointer[start + pair.0])) * pair.1
            }
        case .bfloat16:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: tensor.shape.reduce(1, *))
            let start = tokenID * hiddenSize
            return zip(0..<hiddenSize, hidden).reduce(Float.zero) { partial, pair in
                partial + Float(bitPattern: UInt32(pointer[start + pair.0]) << 16) * pair.1
            }
        case .float32:
            let pointer = basePointer.bindMemory(to: Float.self, capacity: tensor.shape.reduce(1, *))
            let start = tokenID * hiddenSize
            return zip(0..<hiddenSize, hidden).reduce(Float.zero) { partial, pair in
                partial + pointer[start + pair.0] * pair.1
            }
        case .quantized:
            return .zero
        }
    }
}
