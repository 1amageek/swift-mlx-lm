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
        tokenizerConfigJSON _: [String: Any],
        modelType: String
    ) -> String? {
        guard modelType.lowercased() == "gemma4" || modelType.lowercased() == "gemma4_text" else {
            return nil
        }
        return Gemma4DefaultChatTemplate.synthesizedSource()
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
