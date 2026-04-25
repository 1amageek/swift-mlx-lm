import Foundation
import Jinja

/// Default Jinja chat template for Gemma4 bundles that do not ship one.
///
/// Mirrors the upstream prompt shape: turn-delimited dialogue with
/// `<|think|>` system header injection when `enable_thinking` is truthy.
/// `bos_token` comes from the render context, matching upstream
/// `chat_template.jinja`.
enum Gemma4DefaultChatTemplate {
    static func synthesizedSource() -> String {
        """
        {%- macro render_content(content) -%}
            {%- if content is string -%}
                {{- content -}}
            {%- elif content is iterable and content is not mapping -%}
                {%- for item in content -%}
                    {%- if item.type == 'text' or 'text' in item -%}
                        {{- item.text -}}
                    {%- elif item.type == 'image' -%}
                        {{- '\n\n<|image|>\n\n' -}}
                    {%- elif item.type == 'video' -%}
                        {{- '\n\n<|video|>\n\n' -}}
                    {%- endif -%}
                {%- endfor -%}
            {%- elif content is none or content is undefined -%}
                {{- '' -}}
            {%- endif -%}
        {%- endmacro -%}
        {%- macro strip_thinking(text) -%}
            {%- set ns = namespace(result='') -%}
            {%- for part in text.split('<channel|>') -%}
                {%- if '<|channel>' in part -%}
                    {%- set ns.result = ns.result + part.split('<|channel>')[0] -%}
                {%- else -%}
                    {%- set ns.result = ns.result + part -%}
                {%- endif -%}
            {%- endfor -%}
            {{- ns.result | trim -}}
        {%- endmacro -%}
        {%- set loop_messages = messages -%}
        {{- bos_token -}}
        {%- if (enable_thinking is defined and enable_thinking) or messages[0]['role'] in ['system', 'developer'] -%}
            {{- '<|turn>system\n' -}}
            {%- if enable_thinking is defined and enable_thinking -%}
                {{- '<|think|>\n' -}}
            {%- endif -%}
            {%- if messages[0]['role'] in ['system', 'developer'] -%}
                {{- render_content(messages[0]['content']) | trim -}}
                {%- set loop_messages = messages[1:] -%}
            {%- endif -%}
            {{- '<turn|>\n' -}}
        {%- endif -%}
        {%- for message in loop_messages -%}
            {%- set role = 'model' if message.role == 'assistant' else message.role -%}
            {{- '<|turn>' + role + '\n' -}}
            {%- set thinking_text = message.get('reasoning') or message.get('reasoning_content') -%}
            {%- if role == 'model' and thinking_text -%}
                {{- '<|channel>thought\n' + thinking_text + '\n<channel|>' -}}
            {%- endif -%}
            {%- if role == 'model' and message['content'] is string -%}
                {{- strip_thinking(message['content']) -}}
            {%- else -%}
                {{- render_content(message.content)|trim -}}
            {%- endif -%}
            {{- '<turn|>\n' -}}
        {%- endfor -%}
        {%- if add_generation_prompt -%}
            {{- '<|turn>model\n' -}}
        {%- endif -%}
        """
    }

    static func template() throws -> Template {
        try Template(synthesizedSource())
    }
}
