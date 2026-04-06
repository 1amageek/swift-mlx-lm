import Testing
import Foundation
import Metal
@testable import MetalCompiler
@testable import SwiftLM
@testable import LMArchitecture
@testable import ModelDeclarations

@Suite("Model Loading Pipeline")
struct LoadTests {

    // MARK: - Step 1: Safetensors Header Parse

    @Test("Parse safetensors header")
    func parseSafetensorsHeader() throws {
        guard let directory = try findModelDirectoryOrSkip() else { return }
        let safetensorsURLs = try findSafetensorsFiles(in: directory)

        let loader = SafetensorsLoader()
        for url in safetensorsURLs {
            let tensors = try loader.parseHeader(at: url)
            #expect(tensors.count > 0, "No tensors in \(url.lastPathComponent)")
            print("[\(url.lastPathComponent)] \(tensors.count) tensors")
            for tensor in tensors.prefix(3) {
                print("  \(tensor.name): \(tensor.dtype) \(tensor.shape) offset=\(tensor.dataOffset) bytes=\(tensor.byteCount)")
            }
        }
    }

    // MARK: - Step 2: STAF Conversion

    @Test("Convert safetensors to STAF")
    func convertToSTAF() throws {
        guard let directory = try findModelDirectoryOrSkip() else { return }
        let safetensorsURLs = try findSafetensorsFiles(in: directory)
        guard let stafURL = try makeTemporarySTAFURLOrSkip(testName: "convert") else { return }
        defer {
            do {
                try removeIfExists(stafURL)
            } catch {
                Issue.record("Failed to clean up temporary STAF: \(error)")
            }
        }

        let converter = STAFConverter()
        try converter.convert(safetensorsURLs: safetensorsURLs, outputURL: stafURL)

        let attributes = try FileManager.default.attributesOfItem(atPath: stafURL.path)
        let fileSize = attributes[.size] as? Int ?? 0
        #expect(fileSize > 0, "STAF file is empty")
        print("[STAF] file size: \(fileSize) bytes")

        // Verify header
        let data = try Data(contentsOf: stafURL, options: .mappedIfSafe)
        let magic = data.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
        #expect(magic == STAF.magic, "Bad magic: \(magic)")

        let sectionCount = data.withUnsafeBytes {
            ($0.baseAddress! + 40).loadUnaligned(as: UInt32.self)
        }
        print("[STAF] section count: \(sectionCount)")
        #expect(sectionCount > 0, "No sections")

        let sectionTableOffset = data.withUnsafeBytes {
            ($0.baseAddress! + 44).loadUnaligned(as: UInt32.self)
        }
        #expect(sectionTableOffset == 64, "Section table should start at 64, got \(sectionTableOffset)")

        // Verify each section entry
        for i in 0..<Int(sectionCount) {
            let entryBase = 64 + i * 128
            let payloadOffset = data.withUnsafeBytes {
                ($0.baseAddress! + entryBase + 44).loadUnaligned(as: UInt64.self)
            }
            let payloadSize = data.withUnsafeBytes {
                ($0.baseAddress! + entryBase + 52).loadUnaligned(as: UInt64.self)
            }
            let ndim = data[entryBase + 11]

            if payloadOffset > UInt64(fileSize) || payloadSize > UInt64(fileSize) {
                let nameOffset = data.withUnsafeBytes {
                    ($0.baseAddress! + entryBase).loadUnaligned(as: UInt32.self)
                }
                let nameLength = data.withUnsafeBytes {
                    ($0.baseAddress! + entryBase + 4).loadUnaligned(as: UInt32.self)
                }
                Issue.record("Section \(i): payloadOffset=\(payloadOffset) payloadSize=\(payloadSize) > fileSize=\(fileSize), ndim=\(ndim), nameOffset=\(nameOffset) nameLen=\(nameLength)")
            }
        }

    }

    // MARK: - Step 3: STAF Load

    @Test("Load STAF with zero-copy mmap")
    func loadSTAF() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("[Skip] No Metal device")
            return
        }

        guard let directory = try findModelDirectoryOrSkip() else { return }
        let safetensorsURLs = try findSafetensorsFiles(in: directory)
        guard let stafURL = try makeTemporarySTAFURLOrSkip(testName: "load") else { return }
        defer {
            do {
                try removeIfExists(stafURL)
            } catch {
                Issue.record("Failed to clean up temporary STAF: \(error)")
            }
        }

        let converter = STAFConverter()
        try converter.convert(safetensorsURLs: safetensorsURLs, outputURL: stafURL)

        let store = try STAFLoader().load(at: stafURL, device: device)
        #expect(store.entries.count > 0, "No entries in weight store")
        print("[STAF Load] \(store.entries.count) tensors loaded")

        for (name, entry) in store.entries.prefix(5) {
            print("  \(name): offset=\(entry.bufferOffset) size=\(entry.payloadSize) scheme=\(entry.schemeIdentifier)")
        }
    }

    // MARK: - Step 4: Config Parse

    @Test("Parse config.json")
    func parseConfig() throws {
        guard let directory = try findModelDirectoryOrSkip() else { return }
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try HFConfigDecoder().decode(from: configData)
        let modelType = try HFConfigDecoder().modelType(from: configData)

        print("[Config] type=\(modelType) hidden=\(config.hiddenSize) layers=\(config.layerCount) vocab=\(config.vocabSize)")
        #expect(config.hiddenSize > 0)
        #expect(config.layerCount > 0)
        #expect(config.vocabSize > 0)
    }

    @Test("Parse Gemma4 nested text_config")
    func parseGemma4Config() throws {
        let configJSON = """
        {
          "model_type": "gemma4",
          "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 1536,
            "num_hidden_layers": 35,
            "intermediate_size": 6144,
            "vocab_size": 262144,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "rms_norm_eps": 1.0e-6,
            "tie_word_embeddings": true,
            "attention_bias": false,
            "mlp_bias": false,
            "sliding_window": 512,
            "layer_types": ["sliding_attention", "full_attention"],
            "hidden_size_per_layer_input": 256,
            "vocab_size_per_layer_input": 262144,
            "global_head_dim": 512,
            "num_kv_shared_layers": 20,
            "use_double_wide_mlp": true,
            "attention_k_eq_v": false,
            "rope_parameters": {
              "sliding_attention": {
                "rope_theta": 10000.0,
                "rope_type": "default"
              },
              "full_attention": {
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
                "rope_type": "proportional"
              }
            }
          }
        }
        """

        let configData = try #require(configJSON.data(using: .utf8))
        let config = try HFConfigDecoder().decode(from: configData)
        let modelType = try HFConfigDecoder().modelType(from: configData)

        #expect(modelType == "gemma4")
        #expect(config.hiddenSize == 1536)
        #expect(config.layerCount == 35)
        #expect(config.hiddenSizePerLayerInput == 256)
        #expect(config.vocabSizePerLayerInput == 262144)
        #expect(config.globalHeadDim == 512)
        #expect(config.numKVSharedLayers == 20)
        #expect(config.useDoubleWideMLP == true)
        #expect(config.fullAttentionRopeTheta == 1_000_000.0)
        #expect(config.fullAttentionPartialRotaryFactor == 0.25)
        #expect(config.fullAttentionRoPEScaling?.kind == .custom("proportional"))
    }

    @Test("Resolve Gemma4 parameter bindings")
    func resolveGemma4Parameters() throws {
        let config = ModelConfig(
            hiddenSize: 1536, layerCount: 2, intermediateSize: 6144, vocabSize: 262144,
            attentionHeads: 8, kvHeads: 1, headDim: 256,
            attentionBias: false, mlpBias: false,
            normEps: 1e-6, normKind: .rmsNorm,
            ropeTheta: 10_000.0, ropeDimension: 256, ropeScaling: nil,
            tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil,
            qkNorm: true,
            fullAttentionInterval: nil,
            ssmNumHeads: nil, ssmKeyHeadDim: nil, ssmValueHeadDim: nil,
            convKernelSize: nil, partialRotaryFactor: nil, slidingWindow: 512,
            layerTypes: ["sliding_attention", "full_attention"],
            hiddenSizePerLayerInput: 256,
            vocabSizePerLayerInput: 262144,
            globalHeadDim: 512,
            globalKVHeads: nil,
            numKVSharedLayers: 1,
            useDoubleWideMLP: true,
            attentionKEqualsV: false,
            fullAttentionRopeTheta: 1_000_000.0,
            fullAttentionPartialRotaryFactor: 0.25,
            fullAttentionRoPEScaling: RoPEScaling(kind: .custom("proportional"), factor: 1.0)
        )

        let graph = try Gemma4(config: config).makeModelGraph()
        let resolved = ParameterResolver().resolve(graph: graph, convention: .gemma4Family)

        let repeatingLayers = resolved.rootRegion.operations.compactMap { operation -> Region? in
            guard case .repeating(count: 1, let body) = operation.kind else { return nil }
            return body
        }
        #expect(repeatingLayers.count == 2)

        let firstLayerBindings = collectTensorNames(in: repeatingLayers[0])
        #expect(firstLayerBindings.contains("model.language_model.layers.0.input_layernorm.weight"))
        #expect(firstLayerBindings.contains("model.language_model.layers.0.pre_feedforward_layernorm.weight"))
        #expect(firstLayerBindings.contains("model.language_model.layers.0.self_attn.q_proj.weight"))
        #expect(firstLayerBindings.contains("model.language_model.layers.0.self_attn.k_norm.weight"))
        #expect(firstLayerBindings.contains("model.language_model.layers.0.mlp.gate_proj.weight"))
        #expect(firstLayerBindings.contains("model.language_model.layers.0.per_layer_input_gate.weight"))
        #expect(firstLayerBindings.contains("model.language_model.embed_tokens_per_layer.weight"))
        #expect(firstLayerBindings.contains("model.language_model.per_layer_model_projection.weight"))
        #expect(firstLayerBindings.contains("model.language_model.per_layer_projection_norm.weight"))
    }

    // MARK: - Step 5: IR Build

    @Test("Build ModelGraph from config")
    func buildIR() throws {
        guard let directory = try findModelDirectoryOrSkip() else { return }
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try HFConfigDecoder().decode(from: configData)
        let modelType = try HFConfigDecoder().modelType(from: configData)

        let graph: ModelGraph
        switch modelType.lowercased() {
        case "lfm2":
            graph = try LFM2(config: config).makeModelGraph()
        case "qwen3_5", "qwen3_vl", "qwen2_5_vl", "qwen2_vl":
            graph = try Qwen35(config: config).makeModelGraph()
        default:
            graph = try Transformer(config: config).makeModelGraph()
        }

        let opCount = countOperations(graph.rootRegion)
        print("[IR] \(opCount) operations, model_type=\(modelType)")
        #expect(opCount > 0)
    }

    // MARK: - Step 6: Parameter Resolve

    @Test("Resolve parameter bindings")
    func resolveParameters() throws {
        guard let directory = try findModelDirectoryOrSkip() else { return }
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try HFConfigDecoder().decode(from: configData)
        let modelType = try HFConfigDecoder().modelType(from: configData)

        let graph = try Transformer(config: config).makeModelGraph()
        let convention: ParameterResolver.WeightNamingConvention = modelType == "lfm2" ? .lfm2Family : .llamaFamily
        let resolved = ParameterResolver().resolve(graph: graph, convention: convention)

        // Check that primitive operations have parameterBindings
        var bindingsCount = 0
        checkBindings(resolved.rootRegion, count: &bindingsCount)
        print("[Resolve] \(bindingsCount) operations with parameterBindings")
        #expect(bindingsCount > 0)
    }

    // MARK: - Step 7: Compile

    @Test("Compile IR to MetalCompiledModel")
    func compileIR() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("[Skip] No Metal device")
            return
        }

        guard let directory = try findModelDirectoryOrSkip() else { return }
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try HFConfigDecoder().decode(from: configData)
        let modelType = try HFConfigDecoder().modelType(from: configData)

        let graph = try Transformer(config: config).makeModelGraph()
        let convention: ParameterResolver.WeightNamingConvention = modelType == "lfm2" ? .lfm2Family : .llamaFamily
        let resolved = ParameterResolver().resolve(graph: graph, convention: convention)

        let compiler = MetalInferenceCompiler()
        let compiledModel = try compiler.compile(
            graph: resolved,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            device: device
        )

        print("[Compile] \(compiledModel.unfusedEntryCount) → \(compiledModel.fusedEntryCount) dispatches")
        #expect(compiledModel.fusedEntryCount > 0)
    }

    // MARK: - Helpers

    private func findModelDirectoryOrSkip() throws -> URL? {
        guard let directory = try findModelDirectory() else {
            print("[Skip] No HuggingFace model snapshot found in the local cache")
            return nil
        }
        return directory
    }

    private func findModelDirectory() throws -> URL? {
        let directCandidates = [
            "/Users/1amageek/Desktop/swift-lm/TestData/LFM2.5-1.2B-Thinking",
        ]

        for candidate in directCandidates {
            let directory = URL(fileURLWithPath: candidate)
            let configPath = directory.appendingPathComponent("config.json")
            let tokenizerPath = directory.appendingPathComponent("tokenizer.json")
            guard FileManager.default.fileExists(atPath: configPath.path),
                  FileManager.default.fileExists(atPath: tokenizerPath.path) else {
                continue
            }

            let contents: [URL]
            do {
                contents = try FileManager.default.contentsOfDirectory(
                    at: directory,
                    includingPropertiesForKeys: nil
                )
            } catch {
                print("[LoadTests] Skip inaccessible local test bundle candidate: \(candidate) (\(error))")
                continue
            }
            let hasSafetensors = contents.contains { $0.pathExtension == "safetensors" }
            guard hasSafetensors else { continue }

            print("[Model] Using local test bundle: \(candidate)")
            return directory
        }

        // Try common model cache locations
        let candidates = [
            "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct",
            "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B",
            "~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Thinking",
        ]

        for candidate in candidates {
            let expanded = NSString(string: candidate).expandingTildeInPath
            let baseURL = URL(fileURLWithPath: expanded)
            let snapshotsURL = baseURL.appendingPathComponent("snapshots")
            let snapshots: [URL]
            do {
                snapshots = try FileManager.default.contentsOfDirectory(
                    at: snapshotsURL,
                    includingPropertiesForKeys: nil
                )
            } catch {
                continue
            }

            guard let snapshot = snapshots.first else { continue }
            let configPath = snapshot.appendingPathComponent("config.json")
            guard FileManager.default.fileExists(atPath: configPath.path) else { continue }

            let contents: [URL]
            do {
                contents = try FileManager.default.contentsOfDirectory(
                    at: snapshot,
                    includingPropertiesForKeys: nil
                )
            } catch {
                continue
            }
            let hasSafetensors = contents.contains { $0.pathExtension == "safetensors" }
            guard hasSafetensors else { continue }

            print("[Model] Using: \(candidate)")
            return snapshot
        }

        return nil
    }

    private func findSafetensorsFiles(in directory: URL) throws -> [URL] {
        let contents = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
        return contents
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private func countOperations(_ region: Region) -> Int {
        var count = 0
        for op in region.operations {
            count += 1
            switch op.kind {
            case .residual(_, let body): count += countOperations(body)
            case .repeating(_, let body): count += countOperations(body)
            case .conditional(_, let t, let e): count += countOperations(t) + countOperations(e)
            case .parallel(_, let branches): for b in branches { count += countOperations(b) }
            case .primitive: break
            }
        }
        return count
    }

    private func checkBindings(_ region: Region, count: inout Int) {
        for op in region.operations {
            if case .primitive = op.kind, !op.parameterBindings.isEmpty {
                count += 1
            }
            switch op.kind {
            case .residual(_, let body): checkBindings(body, count: &count)
            case .repeating(_, let body): checkBindings(body, count: &count)
            case .conditional(_, let t, let e):
                checkBindings(t, count: &count)
                checkBindings(e, count: &count)
            case .parallel(_, let branches): for b in branches { checkBindings(b, count: &count) }
            case .primitive: break
            }
        }
    }

    private func makeTemporarySTAFURLOrSkip(testName: String) throws -> URL? {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("swift-lm-load-tests", isDirectory: true)
        if !hasSufficientTemporaryDiskSpace(for: directory, minimumBytes: 4 * 1024 * 1024 * 1024) {
            print("[Skip] Not enough temporary disk space for STAF conversion test")
            return nil
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
            .appendingPathComponent("\(testName)-\(UUID().uuidString)")
            .appendingPathExtension("staf")
    }

    private func hasSufficientTemporaryDiskSpace(for directory: URL, minimumBytes: Int64) -> Bool {
        do {
            let resourceValues = try directory.resourceValues(forKeys: [
                .volumeAvailableCapacityForImportantUsageKey,
                .volumeAvailableCapacityKey,
            ])
            if let available = resourceValues.volumeAvailableCapacityForImportantUsage {
                return available >= minimumBytes
            }
            if let available = resourceValues.volumeAvailableCapacity {
                return Int64(available) >= minimumBytes
            }
            return true
        } catch {
            return true
        }
    }

    private func removeIfExists(_ url: URL) throws {
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }
}

private func collectTensorNames(in region: Region) -> Set<String> {
    var tensorNames = Set<String>()
    for operation in region.operations {
        for binding in operation.parameterBindings {
            tensorNames.insert(binding.tensorName)
        }
        switch operation.kind {
        case .residual(_, let body), .repeating(_, let body):
            tensorNames.formUnion(collectTensorNames(in: body))
        case .conditional(_, let thenBody, let elseBody):
            tensorNames.formUnion(collectTensorNames(in: thenBody))
            tensorNames.formUnion(collectTensorNames(in: elseBody))
        case .parallel(_, let branches):
            for branch in branches {
                tensorNames.formUnion(collectTensorNames(in: branch))
            }
        case .primitive:
            break
        }
    }
    return tensorNames
}
