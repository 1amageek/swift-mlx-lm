import Testing
@testable import SwiftLM

@Suite("Generation Visibility State", .serialized)
struct GenerationVisibilityStateTests {
    @Test("reasoning tags are suppressed across split fragments")
    func suppressesThinkingSpan() {
        var state = GenerationVisibilityState(
            policy: ThinkingTagPolicy(
                openTag: "<think>",
                closeTag: "</think>",
                openTagTokenID: nil,
                closeTagTokenID: nil
            ),
            emitsReasoning: false
        )

        #expect(state.append(decodedText: "Hello ").answer == "Hello ")
        #expect(state.append(decodedText: "<").answer == "")
        #expect(state.append(decodedText: "think").answer == "")
        #expect(state.append(decodedText: ">hidden").answer == "")
        #expect(state.suppressingReasoning)
        #expect(state.append(decodedText: " text</").answer == "")
        #expect(state.append(decodedText: "think> tail").answer == " tail")
        #expect(!state.suppressingReasoning)
        #expect(state.didSuppressReasoning)
    }

    @Test("unterminated prefixes are flushed when reasoning is inactive")
    func flushesPartialPrefixAtEnd() {
        var state = GenerationVisibilityState(
            policy: ThinkingTagPolicy(
                openTag: "<think>",
                closeTag: "</think>",
                openTagTokenID: nil,
                closeTagTokenID: nil
            ),
            emitsReasoning: false
        )

        #expect(state.append(decodedText: "Visible <th").answer == "Visible ")
        #expect(state.finalize().answer == "<th")
        #expect(!state.suppressingReasoning)
    }

    @Test("visibility state is a no-op without a policy")
    func noOpWithoutPolicy() {
        var state = GenerationVisibilityState(policy: nil, emitsReasoning: false)

        #expect(state.append(decodedText: "<think>hello</think>").answer == "<think>hello</think>")
        #expect(state.finalize().answer.isEmpty)
        #expect(!state.suppressingReasoning)
    }
}
