import Testing
import TestHeartbeat
import Foundation
import MLX
import GGUFParser
@testable import MLXLM

/// Integration tests using a real Qwen3.5-0.8B GGUF model.
///
/// These tests download the model on first run (~600MB, cached locally).
/// Requires network access and Metal GPU.
@Suite("Qwen3.5-0.8B", .tags(.diagnostic), .heartbeat)
struct Qwen3_5Tests {

    private static let repo = "unsloth/Qwen3.5-0.8B-GGUF"
    private static let filename = "Qwen3.5-0.8B-Q4_K_M.gguf"

    /// Download (or use cached) model file.
    private func downloadModel() async throws -> URL {
        let downloader = HuggingFaceDownloader()
        return try await downloader.download(
            repo: Self.repo,
            filename: Self.filename
        )
    }

    @Test("Inspect GGUF metadata")
    func inspectMetadata() async throws {
        let url = try await downloadModel()
        let file = try GGUFFile.parse(url: url)

        print("Architecture: \(file.architecture ?? "nil")")
        print("Embedding: \(file.embeddingLength ?? -1)")
        print("BlockCount: \(file.blockCount ?? -1)")
        print("HeadCount: \(file.headCount ?? -1)")
        print("HeadCountKV: \(file.headCountKV ?? -1)")
        print("HeadDimension: \(file.headDimension ?? -1)")
        print("AttentionKeyLength: \(file.attentionKeyLength ?? -1)")
        print("AttentionValueLength: \(file.attentionValueLength ?? -1)")
        print("FFN: \(file.feedForwardLength ?? -1)")

        for (key, value) in file.metadata where key.hasPrefix("qwen") {
            print("  \(key) = \(value)")
        }

        // Print all tensor names for layer 0 and 3 (DeltaNet vs FullAttn)
        for tensor in file.tensors where tensor.name.contains("blk.0.") || tensor.name.contains("blk.3.") {
            print("  Tensor: \(tensor.name) shape=\(tensor.dimensions) type=\(tensor.quantizationType)")
        }
        // Global tensors
        for tensor in file.tensors where !tensor.name.hasPrefix("blk.") {
            print("  Global: \(tensor.name) shape=\(tensor.dimensions) type=\(tensor.quantizationType)")
        }

        #expect(file.architecture != nil)
    }

    @Test("Download and load model")
    func loadModel() async throws {
        let url = try await downloadModel()
        let loader = GGUFModelLoader()
        let container = try loader.load(url: url)

        let config = await container.configuration
        #expect(!config.name.isEmpty)
    }

    @Test("Tokenize and decode round-trip")
    func tokenizeRoundTrip() async throws {
        let url = try await downloadModel()
        let loader = GGUFModelLoader()
        let container = try loader.load(url: url)

        let text = "Hello, world!"
        let tokens = await container.encode(text)
        #expect(tokens.count > 0)

        let decoded = await container.decode(tokens: tokens)
        #expect(decoded == text)
    }

    @Test("Prepare user input")
    func prepareInput() async throws {
        let url = try await downloadModel()
        let loader = GGUFModelLoader()
        let container = try loader.load(url: url)

        let input = UserInput(prompt: "Hello")
        let lmInput = try await container.prepare(input: input)
        #expect(lmInput.text.tokens.dim(1) > 0)
        #expect(lmInput.image == nil)
        #expect(lmInput.video == nil)
    }

    @Test("Generate short response")
    func generateResponse() async throws {
        let url = try await downloadModel()
        let loader = GGUFModelLoader()
        let container = try loader.load(url: url)

        let input = UserInput(chat: [
            .system("You are a helpful assistant. Reply briefly."),
            .user("What is 1+1?"),
        ])
        let lmInput = try await container.prepare(input: input)

        var params = GenerateParameters()
        params.temperature = 0
        params.maxTokens = 32

        let stream = try await container.generate(input: lmInput, parameters: params)

        var output = ""
        for await generation in stream {
            switch generation {
            case .chunk(let text):
                output += text
            case .info(let info):
                #expect(info.promptTokenCount > 0)
                #expect(info.generationTokenCount > 0)
                #expect(info.generationTokenCount <= 32)
            default:
                break
            }
        }

        #expect(!output.isEmpty)
    }

    @Test("Chat with additionalContext")
    func additionalContext() async throws {
        let url = try await downloadModel()
        let loader = GGUFModelLoader()
        let container = try loader.load(url: url)

        let input = UserInput(
            chat: [.user("Say hello")],
            additionalContext: ["enable_thinking": false]
        )
        let lmInput = try await container.prepare(input: input)
        #expect(lmInput.text.tokens.dim(1) > 0)
    }
}
