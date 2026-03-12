import Foundation
import Testing
import MLX
import MLXNN
@testable import MLXLM

// MARK: - LoRALinear Tests

@Suite("LoRALinear", .serialized, .tags(.unit))
struct LoRALinearTests {

    @Test("Wrap Linear in LoRALinear")
    func wrapLinear() {
        let linear = Linear(8, 16)
        eval(linear)
        let lora = LoRALinear.from(linear: linear, rank: 4, scale: 10.0)
        #expect(lora is LoRALinear)
    }

    @Test("Wrap QuantizedLinear in QLoRALinear")
    func wrapQuantizedLinear() {
        let weight = MLXRandom.normal([64, 32])
        let quantized = QuantizedLinear(weight: weight, bias: nil, groupSize: 32, bits: 4)
        eval(quantized)
        let lora = LoRALinear.from(linear: quantized, rank: 4, scale: 10.0)
        #expect(lora is QLoRALinear)
    }

    @Test("LoRALinear forward produces output")
    func loraForward() {
        let linear = Linear(8, 16)
        eval(linear)
        let lora = LoRALinear.from(linear: linear, rank: 4, scale: 10.0)
        let x = MLXRandom.normal([1, 8])
        eval(x)
        let output = (lora as! LoRALinear)(x)
        eval(output)
        #expect(output.shape == [1, 16])
    }

    @Test("fused() merges LoRA weights into base")
    func fuseLoRA() {
        let linear = Linear(8, 16)
        eval(linear)
        let loraLayer = LoRALinear.from(linear: linear, rank: 4, scale: 10.0) as! LoRALinear

        // Set loraB to non-zero via update() to verify fuse effect
        let newB = MLXRandom.normal([4, 16])
        eval(newB)
        loraLayer.update(parameters: ModuleParameters.unflattened(["lora_b": newB]))
        eval(loraLayer)

        let fused = loraLayer.fused()
        #expect(fused is Linear)
        #expect(!(fused is LoRALinear))

        // Verify fused weight includes LoRA contribution
        let fusedLinear = fused as! Linear
        let expectedWeight = loraLayer.weight + matmul(
            (loraLayer.scale * loraLayer.loraB.T).asType(loraLayer.weight.dtype),
            loraLayer.loraA.T.asType(loraLayer.weight.dtype)
        )
        eval(expectedWeight)

        let diff = abs(fusedLinear.weight - expectedWeight).sum()
        eval(diff)
        #expect(diff.item(Float.self) < 1e-4)
    }

    @Test("reverted() restores original Linear")
    func revertLoRA() {
        let linear = Linear(8, 16)
        eval(linear)
        let originalWeight = linear.weight
        let loraLayer = LoRALinear.from(linear: linear, rank: 4) as! LoRALinear

        let reverted = loraLayer.reverted()
        #expect(reverted is Linear)
        #expect(!(reverted is LoRALinear))

        let revertedLinear = reverted as! Linear
        let diff = abs(revertedLinear.weight - originalWeight).sum()
        eval(diff)
        #expect(diff.item(Float.self) == 0)
    }
}

// MARK: - DoRALinear Tests

@Suite("DoRALinear", .serialized, .tags(.unit))
struct DoRALinearTests {

    @Test("Wrap Linear in DoRALinear")
    func wrapLinear() {
        let linear = Linear(8, 16)
        eval(linear)
        let dora = DoRALinear.from(linear: linear, rank: 4, scale: 10.0)
        #expect(dora is DoRALinear)
    }

    @Test("Wrap QuantizedLinear in QDoRALinear")
    func wrapQuantizedLinear() {
        let weight = MLXRandom.normal([64, 32])
        let quantized = QuantizedLinear(weight: weight, bias: nil, groupSize: 32, bits: 4)
        eval(quantized)
        let dora = DoRALinear.from(linear: quantized, rank: 4, scale: 10.0)
        #expect(dora is QDoRALinear)
    }

    @Test("DoRALinear forward produces output")
    func doraForward() {
        let linear = Linear(8, 16)
        eval(linear)
        let dora = DoRALinear.from(linear: linear, rank: 4, scale: 10.0)
        let x = MLXRandom.normal([1, 8])
        eval(x)
        let output = (dora as! DoRALinear)(x)
        eval(output)
        #expect(output.shape == [1, 16])
    }

    @Test("fused() produces plain Linear")
    func fuseDora() {
        let linear = Linear(8, 16)
        eval(linear)
        let doraLayer = DoRALinear.from(linear: linear, rank: 4, scale: 10.0) as! DoRALinear
        eval(doraLayer)

        let fused = doraLayer.fused()
        #expect(fused is Linear)
        #expect(!(fused is DoRALinear))
    }
}

// MARK: - LoRAConfiguration Tests

@Suite("LoRAConfiguration", .tags(.unit))
struct LoRAConfigurationTests {

    @Test("Decode adapter_config.json format")
    func decodeJSON() throws {
        let json = """
        {
          "fine_tune_type": "lora",
          "num_layers": 28,
          "lora_parameters": {
            "rank": 16,
            "scale": 20.0
          }
        }
        """.data(using: .utf8)!

        let config = try JSONDecoder().decode(LoRAConfiguration.self, from: json)
        #expect(config.numLayers == 28)
        #expect(config.fineTuneType == LoRAConfiguration.FineTuneType.lora)
        #expect(config.loraParameters.rank == 16)
        #expect(config.loraParameters.scale == 20.0)
        #expect(config.loraParameters.keys == nil)
    }

    @Test("Decode DoRA config with keys")
    func decodeDoRAWithKeys() throws {
        let json = """
        {
          "fine_tune_type": "dora",
          "num_layers": 8,
          "lora_parameters": {
            "rank": 8,
            "scale": 10.0,
            "keys": ["self_attn.q_proj", "self_attn.v_proj"]
          }
        }
        """.data(using: .utf8)!

        let config = try JSONDecoder().decode(LoRAConfiguration.self, from: json)
        #expect(config.fineTuneType == LoRAConfiguration.FineTuneType.dora)
        #expect(config.loraParameters.keys?.count == 2)
    }
}

// MARK: - GGUFTensorNameMapper LoRA Tests

@Suite("GGUFTensorNameMapper LoRA", .tags(.unit))
struct TensorNameMapperLoRATests {

    @Test("Map LoRA A tensor names")
    func mapLoRAA() {
        let mapper = TransformerTensorNameMapper()
        #expect(mapper.mlxName(for: "blk.0.attn_q.loraA.weight") == "model.layers.0.self_attn.q_proj.lora_a")
        #expect(mapper.mlxName(for: "blk.3.attn_k.loraA.weight") == "model.layers.3.self_attn.k_proj.lora_a")
        #expect(mapper.mlxName(for: "blk.1.attn_v.loraA.weight") == "model.layers.1.self_attn.v_proj.lora_a")
        #expect(mapper.mlxName(for: "blk.0.attn_output.loraA.weight") == "model.layers.0.self_attn.o_proj.lora_a")
        #expect(mapper.mlxName(for: "blk.0.ffn_gate.loraA.weight") == "model.layers.0.mlp.gate_proj.lora_a")
        #expect(mapper.mlxName(for: "blk.0.ffn_up.loraA.weight") == "model.layers.0.mlp.up_proj.lora_a")
        #expect(mapper.mlxName(for: "blk.2.ffn_down.loraA.weight") == "model.layers.2.mlp.down_proj.lora_a")
    }

    @Test("Map LoRA B tensor names")
    func mapLoRAB() {
        let mapper = TransformerTensorNameMapper()
        #expect(mapper.mlxName(for: "blk.0.attn_q.loraB.weight") == "model.layers.0.self_attn.q_proj.lora_b")
        #expect(mapper.mlxName(for: "blk.5.ffn_down.loraB.weight") == "model.layers.5.mlp.down_proj.lora_b")
    }
}

// MARK: - TransformerModel LoRAModel Conformance Tests

@Suite("TransformerModel LoRAModel", .serialized, .tags(.unit))
struct TransformerModelLoRATests {

    @Test("loraLayers returns transformer layers")
    func loraLayersCount() {
        let config = TransformerConfiguration(
            hiddenSize: 32, hiddenLayers: 2, intermediateSize: 64,
            attentionHeads: 2, vocabularySize: 16, kvHeads: 2
        )
        let model = TransformerModel(config)
        eval(model)
        #expect(model.loraLayers.count == 2)
    }

    @Test("loraDefaultKeys contains Linear layer keys")
    func loraDefaultKeysContainsLinear() {
        let config = TransformerConfiguration(
            hiddenSize: 32, hiddenLayers: 1, intermediateSize: 64,
            attentionHeads: 2, vocabularySize: 16, kvHeads: 2
        )
        let model = TransformerModel(config)
        eval(model)
        let keys = model.loraDefaultKeys
        // Should contain attention projections and MLP projections
        #expect(keys.contains("self_attn.q_proj"))
        #expect(keys.contains("self_attn.k_proj"))
        #expect(keys.contains("self_attn.v_proj"))
        #expect(keys.contains("self_attn.o_proj"))
        #expect(keys.contains("mlp.gate_proj"))
        #expect(keys.contains("mlp.up_proj"))
        #expect(keys.contains("mlp.down_proj"))
    }
}

// MARK: - LoRAContainer Lifecycle Tests

@Suite("LoRAContainer Lifecycle", .serialized, .tags(.unit))
struct LoRAContainerLifecycleTests {

    @Test("Load and fuse LoRA into model")
    func loadAndFuse() throws {
        let config = TransformerConfiguration(
            hiddenSize: 32, hiddenLayers: 2, intermediateSize: 64,
            attentionHeads: 2, vocabularySize: 16, kvHeads: 2
        )
        let model = TransformerModel(config)
        eval(model)

        // Create a LoRA container from the model
        let loraConfig = LoRAConfiguration(
            numLayers: 2,
            fineTuneType: .lora,
            loraParameters: .init(rank: 4, scale: 10.0)
        )
        let container = try LoRAContainer.from(model: model, configuration: loraConfig)

        // Verify LoRA layers were applied
        let firstLayer = model.loraLayers[0]
        let modules = firstLayer.namedModules()
        let hasLoRA = modules.contains { _, module in module is LoRALinear || module is QLoRALinear }
        #expect(hasLoRA)

        // Fuse and verify layers are back to plain Linear
        try container.fuse(with: model)
        let modulesAfterFuse = firstLayer.namedModules()
        let hasLoRAAfterFuse = modulesAfterFuse.contains { _, module in module is LoRALinear }
        #expect(!hasLoRAAfterFuse)
    }

    @Test("Unload removes LoRA layers")
    func unload() throws {
        let config = TransformerConfiguration(
            hiddenSize: 32, hiddenLayers: 1, intermediateSize: 64,
            attentionHeads: 2, vocabularySize: 16, kvHeads: 2
        )
        let model = TransformerModel(config)
        eval(model)

        let loraConfig = LoRAConfiguration(
            numLayers: 1,
            fineTuneType: .lora,
            loraParameters: .init(rank: 4, scale: 10.0)
        )
        let container = try LoRAContainer.from(model: model, configuration: loraConfig)

        // Verify LoRA was applied
        let layer = model.loraLayers[0]
        let hasLoRA = layer.namedModules().contains { _, m in m is LoRALinear || m is QLoRALinear }
        #expect(hasLoRA)

        // Unload
        container.unload(from: model)
        let hasLoRAAfter = layer.namedModules().contains { _, m in m is LoRALinear || m is QLoRALinear }
        #expect(!hasLoRAAfter)
    }
}
