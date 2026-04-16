import Testing
@testable import SwiftLM

@Suite("GenerationEvent Thinking Options")
struct GenerationThinkingOptionsTests {
    private let policy = ThinkingTagPolicy(
        openTag: "<think>",
        closeTag: "</think>",
        openTagTokenID: nil,
        closeTagTokenID: nil
    )
    private let gemmaChannelPolicy = ThinkingTagPolicy(
        openTag: "<|channel>thought\n",
        closeTag: "<channel|>",
        openTagTokenID: nil,
        closeTagTokenID: nil
    )

    @Test("separate mode emits reasoning chunks and strips them from answer")
    func separateModeSplitsReasoning() {
        var state = GenerationVisibilityState(policy: policy, emitsReasoning: true)

        let emitted = state.append(decodedText: "<think>plan</think>answer")
        let trailing = state.finalize()

        #expect(emitted.reasoning == "plan")
        #expect(emitted.answer == "answer")
        #expect(trailing.reasoning.isEmpty)
        #expect(trailing.answer.isEmpty)
    }

    @Test("hidden mode suppresses reasoning from visible output")
    func hiddenModeSuppressesReasoning() {
        var state = GenerationVisibilityState(policy: policy, emitsReasoning: false)

        let emitted = state.append(decodedText: "<think>plan</think>answer")
        let trailing = state.finalize()

        #expect(emitted.reasoning.isEmpty)
        #expect(emitted.answer == "answer")
        #expect(trailing.reasoning.isEmpty)
        #expect(trailing.answer.isEmpty)
    }

    @Test("visible mode keeps think tags inline when no separation is requested")
    func visibleModeKeepsInlineThinking() {
        var state = GenerationVisibilityState(policy: nil, emitsReasoning: false)

        let emitted = state.append(decodedText: "<think>plan</think>answer")
        let trailing = state.finalize()

        #expect(emitted.reasoning.isEmpty)
        #expect(emitted.answer == "<think>plan</think>answer")
        #expect(trailing.reasoning.isEmpty)
        #expect(trailing.answer.isEmpty)
    }

    @Test("Gemma channel mode emits reasoning chunks and strips them from answer")
    func gemmaChannelSeparateModeSplitsReasoning() {
        var state = GenerationVisibilityState(policy: gemmaChannelPolicy, emitsReasoning: true)

        let emitted = state.append(decodedText: "<|channel>thought\nplan\n<channel|>Tokyo")
        let trailing = state.finalize()

        #expect(emitted.reasoning == "plan\n")
        #expect(emitted.answer == "Tokyo")
        #expect(trailing.reasoning.isEmpty)
        #expect(trailing.answer.isEmpty)
    }

    @Test("Gemma channel hidden mode suppresses reasoning from visible output")
    func gemmaChannelHiddenModeSuppressesReasoning() {
        var state = GenerationVisibilityState(policy: gemmaChannelPolicy, emitsReasoning: false)

        let emitted = state.append(decodedText: "<|channel>thought\nplan\n<channel|>Tokyo")
        let trailing = state.finalize()

        #expect(emitted.reasoning.isEmpty)
        #expect(emitted.answer == "Tokyo")
        #expect(trailing.reasoning.isEmpty)
        #expect(trailing.answer.isEmpty)
    }
}
