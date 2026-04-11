import Foundation
import Jinja
import Testing
@testable import SwiftLM

@Suite("Chat Template Rendering", .serialized)
struct ChatTemplateRenderingTests {
    private static let lfmDirectory = URL(
        fileURLWithPath: "/Users/1amageek/Desktop/swift-lm/TestData/LFM2.5-1.2B-Thinking"
    )
    private static let gemmaDirectory = URL(
        fileURLWithPath: "/Users/1amageek/Desktop/swift-lm/TestData/gemma-4-E2B-it"
    )

    @Test("Synthesized Gemma4 template preserves official turn markers and thinking control")
    func synthesizedGemma4TemplatePreservesOfficialMarkers() throws {
        let inspector = ModelBundleInspector()
        let directory = try makeTemporaryDirectory()
        let tokenizerConfigURL = directory.appendingPathComponent("tokenizer_config.json")
        let tokenizerConfig = """
        {
          "bos_token": "<bos>"
        }
        """
        try tokenizerConfig.write(to: tokenizerConfigURL, atomically: true, encoding: .utf8)

        let result = try inspector.loadChatTemplate(from: directory, modelType: "gemma4")
        let template = try #require(result.template)
        let source = try #require(result.source)

        #expect(source.contains("<|think|>"))
        #expect(source.contains("<|channel>thought"))

        let renderedWithoutThinking = try template.render([
            "messages": .array([
                .object([
                    "role": .string("user"),
                    "content": .string("What is the capital of Japan?"),
                ])
            ]),
            "add_generation_prompt": .boolean(true),
            "enable_thinking": .boolean(false),
            "bos_token": .string("<bos>"),
            "eos_token": .string(""),
        ])
        let renderedWithThinking = try template.render([
            "messages": .array([
                .object([
                    "role": .string("user"),
                    "content": .string("What is the capital of Japan?"),
                ])
            ]),
            "add_generation_prompt": .boolean(true),
            "enable_thinking": .boolean(true),
            "bos_token": .string("<bos>"),
            "eos_token": .string(""),
        ])

        #expect(
            renderedWithoutThinking == "<bos><|turn>user\n"
                + "What is the capital of Japan?<turn|>\n"
                + "<|turn>model\n"
        )
        #expect(
            renderedWithThinking == "<bos><|turn>system\n"
                + "<|think|>\n"
                + "<turn|>\n"
                + "<|turn>user\n"
                + "What is the capital of Japan?<turn|>\n"
                + "<|turn>model\n"
        )
    }

    @Test("Official Gemma4 template injects think control in the first system turn")
    func officialGemma4TemplateInjectsThinkControl() throws {
        let templateURL = Self.gemmaDirectory.appendingPathComponent("chat_template.jinja")
        guard FileManager.default.fileExists(atPath: templateURL.path) else {
            print("[Skip] No local official Gemma4 chat_template.jinja found")
            return
        }

        let templateSource = try String(contentsOf: templateURL, encoding: .utf8)
        let template = try Template(templateSource)
        let rendered = try template.render([
            "messages": .array([
                .object([
                    "role": .string("user"),
                    "content": .string("What is the capital of Japan?"),
                ])
            ]),
            "add_generation_prompt": .boolean(true),
            "enable_thinking": .boolean(true),
            "bos_token": .string("<bos>"),
            "eos_token": .string(""),
        ])

        #expect(rendered.contains("<|turn>system\n<|think|>\n"))
        #expect(rendered.contains("<|turn>user\nWhat is the capital of Japan?<turn|>\n"))
        #expect(rendered.hasSuffix("<|turn>model\n"))
    }

    @Test("prepare rejects conflicting thinking controls between prompt options and template variables", .timeLimit(.minutes(2)))
    func prepareRejectsConflictingThinkingControls() async throws {
        let configURL = Self.gemmaDirectory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            print("[Skip] No local Gemma4 bundle found at \(Self.gemmaDirectory.path)")
            return
        }

        let loader = ModelBundleLoader()
        let session = try await loader.load(directory: Self.gemmaDirectory)

        do {
            _ = try await session.prepare(
                ModelInput(
                    chat: [.user("What is the capital of Japan?")],
                    promptOptions: PromptPreparationOptions(
                        isThinkingEnabled: false,
                        templateVariables: ["enable_thinking": .boolean(true)]
                    )
                )
            )
            Issue.record("Expected conflicting thinking controls to fail")
        } catch let error as InferenceSessionError {
            guard case .conflictingPromptThinkingConfiguration = error else {
                Issue.record("Unexpected error: \(error)")
                return
            }
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @Test("LFM Jinja chat template renders plain text content, not JSON payloads", .timeLimit(.minutes(2)))
    func lfmJinjaTemplateRendersPlainTextContent() async throws {
        let configURL = Self.lfmDirectory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            print("[Skip] No local LFM bundle found at \(Self.lfmDirectory.path)")
            return
        }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: Self.lfmDirectory)
        let prepared = try await container.prepare( ModelInput(chat: [
                .user([.text("What is the capital of Japan? Answer with exactly one word.")])
            ])
        )

        print("[LFM rendered chat prompt]")
        print(prepared.renderedText)

        #expect(prepared.renderedText.contains("What is the capital of Japan?"))
        #expect(!prepared.renderedText.contains("\"type\":\"text\""))
        #expect(!prepared.renderedText.contains(".\"}]"))
    }

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}
