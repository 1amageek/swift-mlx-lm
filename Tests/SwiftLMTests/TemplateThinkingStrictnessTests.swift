import Testing
@testable import SwiftLM

@Suite("Template Thinking Strictness")
struct TemplateThinkingStrictnessTests {
    @Test("template without reasoning tags does not treat think tags specially")
    func templateWithoutReasoningTagsDoesNotSplitThinkBlocks() {
        let template = """
        {%- for message in messages -%}
        {{- '<|turn|>' -}}
        {{- message.content -}}
        {%- endfor -%}
        """

        let policy = TemplateThinkingTagPolicyExtractor.extract(from: template)
        #expect(policy == nil)

        var state = GenerationVisibilityState(policy: policy.map {
            ThinkingTagPolicy(
                openTag: $0.openTag,
                closeTag: $0.closeTag,
                openTagTokenID: nil,
                closeTagTokenID: nil
            )
        }, emitsReasoning: true)

        let emitted = state.append(decodedText: "<think>plan</think>answer")
        let trailing = state.finalize()

        #expect(emitted.reasoning.isEmpty)
        #expect(emitted.answer == "<think>plan</think>answer")
        #expect(trailing.reasoning.isEmpty)
        #expect(trailing.answer.isEmpty)
    }

    @Test("template with custom reasoning tags splits only those tags")
    func templateWithCustomReasoningTagsSplitsOnlyCustomTags() throws {
        let template = """
        {%- if enable_thinking -%}
        {{- '<reasoning>' -}}
        {%- endif -%}
        {{- message.content -}}
        {{- '</reasoning>' -}}
        """

        let extracted = try #require(TemplateThinkingTagPolicyExtractor.extract(from: template))
        #expect(extracted.openTag == "<reasoning>")
        #expect(extracted.closeTag == "</reasoning>")

        var state = GenerationVisibilityState(
            policy: ThinkingTagPolicy(
                openTag: extracted.openTag,
                closeTag: extracted.closeTag,
                openTagTokenID: nil,
                closeTagTokenID: nil
            ),
            emitsReasoning: true
        )

        let emitted = state.append(decodedText: "<think>plan</think><reasoning>chain</reasoning>answer")
        let trailing = state.finalize()

        #expect(emitted.reasoning == "chain")
        #expect(emitted.answer == "<think>plan</think>answer")
        #expect(trailing.reasoning.isEmpty)
        #expect(trailing.answer.isEmpty)
    }

    @Test("template with think tags splits think blocks")
    func templateWithThinkTagsSplitsThinkBlocks() throws {
        let template = """
        {%- if enable_thinking -%}
        {{- '<think>' -}}
        {%- endif -%}
        {{- message.content -}}
        {{- '</think>' -}}
        """

        let extracted = try #require(TemplateThinkingTagPolicyExtractor.extract(from: template))
        var state = GenerationVisibilityState(
            policy: ThinkingTagPolicy(
                openTag: extracted.openTag,
                closeTag: extracted.closeTag,
                openTagTokenID: nil,
                closeTagTokenID: nil
            ),
            emitsReasoning: true
        )

        let emitted = state.append(decodedText: "<think>plan</think>answer")
        let trailing = state.finalize()

        #expect(emitted.reasoning == "plan")
        #expect(emitted.answer == "answer")
        #expect(trailing.reasoning.isEmpty)
        #expect(trailing.answer.isEmpty)
    }

    @Test("LFM-style template semantics split think blocks")
    func lfmStyleTemplateSemanticsSplitThinkBlocks() throws {
        let template = """
        {{- bos_token -}}
        {%- set keep_past_thinking = keep_past_thinking | default(false) -%}
        {%- for message in messages -%}
            {%- set content = message["content"] -%}
            {%- if message["role"] == "assistant" and not keep_past_thinking -%}
                {%- if "</think>" in content -%}
                    {%- set content = content.split("</think>")[-1] | trim -%}
                {%- endif -%}
            {%- endif -%}
            {{- content -}}
        {%- endfor -%}
        """

        let extracted = try #require(TemplateThinkingTagPolicyExtractor.extract(from: template))
        var state = GenerationVisibilityState(
            policy: ThinkingTagPolicy(
                openTag: extracted.openTag,
                closeTag: extracted.closeTag,
                openTagTokenID: nil,
                closeTagTokenID: nil
            ),
            emitsReasoning: true
        )

        let emitted = state.append(decodedText: "<think>plan</think>answer")
        let trailing = state.finalize()

        #expect(emitted.reasoning == "plan")
        #expect(emitted.answer == "answer")
        #expect(trailing.reasoning.isEmpty)
        #expect(trailing.answer.isEmpty)
    }
}
