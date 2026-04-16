import Testing
@testable import SwiftLM

@Suite("Template Thinking Tag Policy Extractor")
struct TemplateThinkingTagPolicyExtractorTests {
    @Test("extracts think tags from template source")
    func extractsThinkTags() {
        let template = """
        {%- if enable_thinking -%}
        {{- '<think>' -}}
        {%- endif -%}
        {{- message.content -}}
        {{- '</think>' -}}
        """

        let policy = TemplateThinkingTagPolicyExtractor.extract(from: template)

        #expect(policy?.openTag == "<think>")
        #expect(policy?.closeTag == "</think>")
    }

    @Test("extracts custom reasoning tags from template source")
    func extractsReasoningTags() {
        let template = """
        {{- '<reasoning>' -}}
        {{- message.content -}}
        {{- '</reasoning>' -}}
        """

        let policy = TemplateThinkingTagPolicyExtractor.extract(from: template)

        #expect(policy?.openTag == "<reasoning>")
        #expect(policy?.closeTag == "</reasoning>")
    }

    @Test("extracts Gemma channel reasoning tags from template source")
    func extractsGemmaChannelReasoningTags() {
        let template = """
        {%- set thinking_text = message.get('reasoning') or message.get('reasoning_content') -%}
        {%- if thinking_text -%}
        {{- '<|channel>thought\\n' + thinking_text + '\\n<channel|>' -}}
        {%- endif -%}
        """

        let policy = TemplateThinkingTagPolicyExtractor.extract(from: template)

        #expect(policy?.openTag == "<|channel>thought\n")
        #expect(policy?.closeTag == "<channel|>")
    }

    @Test("ignores templates without reasoning tags")
    func ignoresTemplatesWithoutReasoningTags() {
        let template = """
        {%- for message in messages -%}
        {{- '<|turn|>' -}}
        {{- message.content -}}
        {%- endfor -%}
        """

        #expect(TemplateThinkingTagPolicyExtractor.extract(from: template) == nil)
    }

    @Test("infers think tags from LFM-style keep_past_thinking template")
    func infersThinkTagsFromKeepPastThinkingTemplate() throws {
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

        let policy = try #require(TemplateThinkingTagPolicyExtractor.extract(from: template))

        #expect(policy.openTag == "<think>")
        #expect(policy.closeTag == "</think>")
    }
}
