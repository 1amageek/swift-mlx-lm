import Foundation
import Jinja
import LMArchitecture

struct ModelBundleInspector {
    func inspect(directory: URL) throws -> ModelBundleResources {
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let decoder = HFConfigDecoder()
        let config = try decoder.decode(from: configData)
        let modelType = try decoder.modelType(from: configData)
        let safetensorsURLs = try findSafetensorsFiles(in: directory)
        let modulesData = try loadOptionalData(from: directory.appendingPathComponent("modules.json"))
        let sentenceTransformersConfigData = try loadOptionalData(
            from: directory.appendingPathComponent("config_sentence_transformers.json")
        )
        let chatTemplate = try loadChatTemplate(from: directory, modelType: modelType)
        let preprocessorConfigData = try loadProcessorConfigData(from: directory)
        let visionConfiguration = try decoder.visionConfiguration(
            from: configData,
            preprocessorConfigData: preprocessorConfigData
        )
        let inputCapabilities = try decoder.inputCapabilities(
            from: configData,
            preprocessorConfigData: preprocessorConfigData,
            visionConfiguration: visionConfiguration
        )
        return ModelBundleResources(
            directory: directory,
            configData: configData,
            config: config,
            modelType: modelType,
            safetensorsURLs: safetensorsURLs,
            modulesData: modulesData,
            sentenceTransformersConfigData: sentenceTransformersConfigData,
            chatTemplate: chatTemplate.template,
            chatTemplateSource: chatTemplate.source,
            preprocessorConfigData: preprocessorConfigData,
            inputCapabilities: inputCapabilities,
            visionConfiguration: visionConfiguration
        )
    }

    func findSafetensorsFiles(in directory: URL) throws -> [URL] {
        let contents = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
        let files = contents
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else {
            throw ModelBundleLoaderError.noSafetensorsFiles(directory.path)
        }
        return files
    }

    func loadChatTemplate(from directory: URL, modelType: String) throws -> (template: Template?, source: String?) {
        let jinjaURL = directory.appendingPathComponent("chat_template.jinja")
        if FileManager.default.fileExists(atPath: jinjaURL.path) {
            let templateString = try String(contentsOf: jinjaURL, encoding: .utf8)
            do {
                return (try Template(templateString), templateString)
            } catch {
                throw ModelBundleLoaderError.invalidConfig(
                    "Invalid chat_template.jinja: \(error)"
                )
            }
        }

        let tokenizerConfigURL = directory.appendingPathComponent("tokenizer_config.json")
        if FileManager.default.fileExists(atPath: tokenizerConfigURL.path) {
            let data = try Data(contentsOf: tokenizerConfigURL)
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw ModelBundleLoaderError.invalidConfig(
                    "tokenizer_config.json is not a JSON object"
                )
            }
            if let templateString = json["chat_template"] as? String {
                do {
                    return (try Template(templateString), templateString)
                } catch {
                    throw ModelBundleLoaderError.invalidConfig(
                        "Invalid tokenizer_config.json chat_template: \(error)"
                    )
                }
            }
            if let templateString = synthesizedChatTemplate(
                tokenizerConfigJSON: json,
                modelType: modelType
            ) {
                do {
                    return (try Template(templateString), templateString)
                } catch {
                    throw ModelBundleLoaderError.invalidConfig(
                        "Invalid synthesized chat template: \(error)"
                    )
                }
            }
        }

        return (nil, nil)
    }

    private func synthesizedChatTemplate(
        tokenizerConfigJSON json: [String: Any],
        modelType: String
    ) -> String? {
        guard modelType.lowercased() == "gemma4" || modelType.lowercased() == "gemma4_text" else {
            return nil
        }
        let bosToken = (json["bos_token"] as? String) ?? "<bos>"

        // Gemma 4 instruction bundles use turn-delimited prompts with optional
        // thinking control. Some local bundles omit `chat_template.jinja`, so
        // synthesize the upstream shape instead of falling back to a raw
        // role-prefixed transcript.
        return """
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
        {{- '\(bosToken)' -}}
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

    func loadOptionalData(from url: URL) throws -> Data? {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }
        return try Data(contentsOf: url)
    }

    func loadProcessorConfigData(from directory: URL) throws -> Data? {
        let candidates = [
            directory.appendingPathComponent("preprocessor_config.json"),
            directory.appendingPathComponent("processor_config.json"),
            directory.appendingPathComponent("tokenizer_config.json"),
        ]
        for candidate in candidates {
            if let data = try loadOptionalData(from: candidate) {
                return data
            }
        }
        return nil
    }
}
