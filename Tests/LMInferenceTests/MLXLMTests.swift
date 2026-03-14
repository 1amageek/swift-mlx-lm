import Testing
import Foundation
import MLX
@testable import LMInference

// MARK: - LMInput Tests

@Suite("LMInput", .tags(.unit))
struct LMInputTests {

    @Test("Create with tokens only")
    func tokensOnly() {
        let tokens = MLXArray([1, 2, 3]).reshaped([1, 3])
        let input = LMInput(tokens: tokens)
        #expect(input.text.tokens.shape == [1, 3])
        #expect(input.text.mask == nil)
    }

    @Test("Create with tokens and mask")
    func tokensAndMask() {
        let tokens = MLXArray([1, 2, 3]).reshaped([1, 3])
        let mask = MLXArray([true, true, false]).reshaped([1, 3])
        let input = LMInput(tokens: tokens, mask: mask)
        #expect(input.text.tokens.shape == [1, 3])
        #expect(input.text.mask != nil)
        #expect(input.text.mask!.shape == [1, 3])
    }

    @Test("Create from Text struct")
    func fromText() {
        let text = LMInput.Text(tokens: MLXArray([10, 20]).reshaped([1, 2]))
        let input = LMInput(text: text)
        #expect(input.text.tokens.shape == [1, 2])
    }
}

// MARK: - ChatMessage Tests

@Suite("ChatMessage", .tags(.unit))
struct ChatMessageTests {

    @Test("Role raw values")
    func roleValues() {
        #expect(Chat.Message.Role.user.rawValue == "user")
        #expect(Chat.Message.Role.assistant.rawValue == "assistant")
        #expect(Chat.Message.Role.system.rawValue == "system")
        #expect(Chat.Message.Role.tool.rawValue == "tool")
    }

    @Test("Static constructors")
    func staticConstructors() {
        let sys = Chat.Message.system("Hello")
        #expect(sys.role == .system)
        #expect(sys.content == "Hello")
        #expect(sys.images.isEmpty)
        #expect(sys.videos.isEmpty)

        let usr = Chat.Message.user("World")
        #expect(usr.role == .user)
        #expect(usr.content == "World")
        #expect(usr.images.isEmpty)

        let asst = Chat.Message.assistant("Hi")
        #expect(asst.role == .assistant)

        let tool = Chat.Message.tool("Result")
        #expect(tool.role == .tool)
        #expect(tool.images.isEmpty)
    }

    @Test("User message with images")
    func userWithImages() {
        let url = URL(fileURLWithPath: "/tmp/test.png")
        let msg = Chat.Message.user("Describe", images: [.url(url)])
        #expect(msg.images.count == 1)
        #expect(msg.videos.isEmpty)
    }
}

// MARK: - UserInput Tests

@Suite("UserInput", .tags(.unit))
struct UserInputTests {

    @Test("Prompt convenience initializer")
    func promptInit() {
        let input = UserInput(prompt: "Hello")
        #expect(input.chat.count == 1)
        #expect(input.chat[0].role == .user)
        #expect(input.chat[0].content == "Hello")
        #expect(input.tools == nil)
        #expect(input.additionalContext == nil)
        #expect(input.images.isEmpty)
        #expect(input.videos.isEmpty)
    }

    @Test("Multi-message input")
    func multiMessage() {
        let input = UserInput(chat: [
            .system("Be helpful"),
            .user("Hello"),
        ])
        #expect(input.chat.count == 2)
    }

    @Test("With additional context")
    func withAdditionalContext() {
        let input = UserInput(
            chat: [.user("Hello")],
            additionalContext: ["enable_thinking": false]
        )
        #expect(input.additionalContext != nil)
    }

    @Test("Images collected from messages")
    func imagesFromMessages() {
        let url = URL(fileURLWithPath: "/tmp/test.png")
        let input = UserInput(chat: [
            .system("Be helpful"),
            .user("Describe", images: [.url(url)]),
        ])
        #expect(input.images.count == 1)
        #expect(input.videos.isEmpty)
    }
}

// MARK: - GenerateParameters Tests

@Suite("GenerateParameters", .tags(.unit))
struct GenerateParametersTests {

    @Test("Default values")
    func defaults() {
        let params = GenerateParameters()
        #expect(params.temperature == 0.6)
        #expect(params.topP == 1.0)
        #expect(params.maxTokens == nil)
        #expect(params.repetitionPenalty == nil)
    }

    @Test("ArgMax sampler at zero temperature")
    func argMaxSampler() throws {
        var params = GenerateParameters()
        params.temperature = 0
        let sampler = params.sampler()
        let logits = MLXArray([Float(0.1), Float(0.9), Float(0.0)])
        let result = sampler.sample(logits: logits.reshaped([1, 3]))
        eval(result)
        let token: Int32 = result.item()
        #expect(token == 1)
    }

    @Test("Repetition processor creation")
    func repetitionProcessor() {
        var params = GenerateParameters()
        params.repetitionPenalty = 1.2
        params.repetitionContextSize = 20
        let proc = params.processor()
        #expect(proc != nil)
    }

    @Test("No processor without penalty")
    func noProcessor() {
        let params = GenerateParameters()
        let proc = params.processor()
        #expect(proc == nil)
    }
}

// MARK: - ModelConfiguration Tests

@Suite("ModelConfiguration", .tags(.unit))
struct ModelConfigurationTests {

    @Test("Basic creation")
    func basicCreation() {
        let config = ModelConfiguration(name: "test-model")
        #expect(config.name == "test-model")
        #expect(config.eosTokenIds.isEmpty)
        #expect(config.toolCallFormat == nil)
    }

    @Test("With EOS tokens")
    func withEosTokens() {
        let config = ModelConfiguration(
            name: "test",
            eosTokenIds: [128001, 128009]
        )
        #expect(config.eosTokenIds.contains(128001))
        #expect(config.eosTokenIds.contains(128009))
    }
}

// MARK: - Generation Types Tests

@Suite("Generation", .tags(.unit))
struct GenerationTests {

    @Test("Completion info")
    func completionInfo() {
        let info = GenerateCompletionInfo(
            promptTokenCount: 10,
            generationTokenCount: 50,
            promptTime: 0.5,
            generateTime: 2.0,
            stopReason: .stop
        )
        #expect(info.promptTokenCount == 10)
        #expect(info.generationTokenCount == 50)
        #expect(info.promptTokensPerSecond > 0)
        #expect(info.tokensPerSecond > 0)
    }

    @Test("Stop reasons")
    func stopReasons() {
        let reasons: [GenerateStopReason] = [.stop, .length, .cancelled]
        #expect(reasons.count == 3)
    }
}

// MARK: - StringOrNumber Tests

@Suite("StringOrNumber", .tags(.unit))
struct StringOrNumberTests {

    @Test("String variant")
    func stringVariant() {
        let val = StringOrNumber.string("llama3")
        if case .string(let s) = val {
            #expect(s == "llama3")
        } else {
            Issue.record("Expected .string")
        }
        #expect(val.asFloat() == nil)
        #expect(val.asInt() == nil)
    }

    @Test("Int variant")
    func intVariant() {
        let val = StringOrNumber.int(42)
        #expect(val.asInt() == 42)
        #expect(val.asFloat() == 42.0)
    }

    @Test("Float variant")
    func floatVariant() {
        let val = StringOrNumber.float(3.14)
        #expect(val.asFloat() == 3.14)
        #expect(val.asInt() == nil)
    }

    @Test("Bool variant")
    func boolVariant() {
        let val = StringOrNumber.bool(true)
        if case .bool(let b) = val {
            #expect(b == true)
        } else {
            Issue.record("Expected .bool")
        }
    }
}
