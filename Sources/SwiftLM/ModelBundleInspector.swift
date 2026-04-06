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
        let chatTemplate = try loadChatTemplate(from: directory)
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
            chatTemplate: chatTemplate,
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

    func loadChatTemplate(from directory: URL) throws -> Template? {
        let jinjaURL = directory.appendingPathComponent("chat_template.jinja")
        if FileManager.default.fileExists(atPath: jinjaURL.path) {
            let templateString = try String(contentsOf: jinjaURL, encoding: .utf8)
            do {
                return try Template(templateString)
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
                    return try Template(templateString)
                } catch {
                    throw ModelBundleLoaderError.invalidConfig(
                        "Invalid tokenizer_config.json chat_template: \(error)"
                    )
                }
            }
        }

        return nil
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
        ]
        for candidate in candidates {
            if let data = try loadOptionalData(from: candidate) {
                return data
            }
        }
        return nil
    }
}
