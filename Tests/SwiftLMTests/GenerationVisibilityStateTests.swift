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
            )
        )

        #expect(state.append(decodedText: "Hello ") == "Hello ")
        #expect(state.append(decodedText: "<") == "")
        #expect(state.append(decodedText: "think") == "")
        #expect(state.append(decodedText: ">hidden") == "")
        #expect(state.suppressingReasoning)
        #expect(state.append(decodedText: " text</") == "")
        #expect(state.append(decodedText: "think> tail") == " tail")
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
            )
        )

        #expect(state.append(decodedText: "Visible <th") == "Visible ")
        #expect(state.finalize() == "<th")
        #expect(!state.suppressingReasoning)
    }

    @Test("visibility state is a no-op without a policy")
    func noOpWithoutPolicy() {
        var state = GenerationVisibilityState(policy: nil)

        #expect(state.append(decodedText: "<think>hello</think>") == "<think>hello</think>")
        #expect(state.finalize().isEmpty)
        #expect(!state.suppressingReasoning)
    }
}
