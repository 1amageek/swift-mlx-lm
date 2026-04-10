import Foundation

struct STAFConversionPlanner: Sendable {

    func plan(safetensorsURLs: [URL]) throws -> STAFConversionPlan {
        let sortedURLs = safetensorsURLs.sorted { $0.lastPathComponent < $1.lastPathComponent }

        let loader = SafetensorsLoader()
        var allTensors: [(name: String, info: SafetensorsTensorInfo, shardIndex: Int, shardURL: URL)] = []

        for (shardIndex, url) in sortedURLs.enumerated() {
            let tensors = try loader.parseHeader(at: url)
            for tensor in tensors {
                allTensors.append((name: tensor.name, info: tensor, shardIndex: shardIndex, shardURL: url))
            }
        }

        let consumedCompanions = consumedCompanions(in: allTensors)
        var entries: [STAFConversionEntry] = []
        entries.reserveCapacity(allTensors.count)

        for (name, info, shardIndex, shardURL) in allTensors {
            if consumedCompanions.contains(name) {
                continue
            }

            entries.append(
                STAFConversionEntry(
                    name: name,
                    info: info,
                    shardIndex: shardIndex,
                    shardURL: shardURL,
                    schemeIdentifier: determineScheme(name: name, info: info, allTensors: allTensors),
                    semanticRole: inferSemanticRole(name: name),
                    originalDType: mapOriginalDType(info.dtype)
                )
            )
        }

        return STAFConversionPlan(sortedURLs: sortedURLs, entries: entries)
    }

    private func consumedCompanions(
        in allTensors: [(name: String, info: SafetensorsTensorInfo, shardIndex: Int, shardURL: URL)]
    ) -> Set<String> {
        var consumed = Set<String>()
        for (name, _, _, _) in allTensors where name.hasSuffix(".weight") {
            let modulePath = String(name.dropLast(".weight".count))
            let scalesName = modulePath + ".scales"
            let biasesName = modulePath + ".biases"
            if allTensors.contains(where: { $0.name == scalesName }),
               allTensors.contains(where: { $0.name == biasesName }) {
                consumed.insert(scalesName)
                consumed.insert(biasesName)
            }
        }
        return consumed
    }

    private func determineScheme(
        name: String,
        info: SafetensorsTensorInfo,
        allTensors: [(name: String, info: SafetensorsTensorInfo, shardIndex: Int, shardURL: URL)]
    ) -> QuantizationSchemeIdentifier {
        if name.hasSuffix(".weight") {
            let modulePath = String(name.dropLast(".weight".count))
            let hasScales = allTensors.contains { $0.name == modulePath + ".scales" }
            let hasBiases = allTensors.contains { $0.name == modulePath + ".biases" }

            if hasScales && hasBiases {
                if let scalesInfo = allTensors.first(where: { $0.name == modulePath + ".scales" })?.info {
                    let groupSize = estimateGroupSize(
                        weightShape: info.shape,
                        scalesShape: scalesInfo.shape,
                        bits: 4
                    )
                    switch groupSize {
                    case 64: return .q4Group64ScaleF16
                    case 128: return .q4Group128ScaleF16
                    default: return .q4Group64ScaleF16
                    }
                }
                return .q4Group64ScaleF16
            }
        }

        if name.hasSuffix(".scales") || name.hasSuffix(".biases") {
            return .passthrough
        }

        switch info.dtype {
        case .float16: return .fp16RowMajor
        case .bfloat16: return .bf16RowMajor
        case .float32: return .fp32RowMajor
        default: return .passthrough
        }
    }

    private func estimateGroupSize(weightShape: [Int], scalesShape: [Int], bits: Int) -> Int {
        guard weightShape.count >= 2, scalesShape.count >= 2 else {
            return 64
        }
        let packedDimension = weightShape[weightShape.count - 1]
        let numberOfGroups = scalesShape[scalesShape.count - 1]
        let elementsPerUInt32 = 32 / bits
        let inputDimension = packedDimension * elementsPerUInt32
        return numberOfGroups > 0 ? inputDimension / numberOfGroups : 64
    }

    private func inferSemanticRole(name: String) -> SemanticRole {
        if name.contains("embed_tokens") || name.contains("token_embd") {
            return .tokenEmbedding
        }
        if name.contains("q_proj") { return .attentionQuery }
        if name.contains("k_proj") { return .attentionKey }
        if name.contains("v_proj") { return .attentionValue }
        if name.contains("o_proj") || name.contains("out_proj") { return .attentionOutput }
        if name.contains("gate_proj") || name.contains(".w1.") { return .mlpGate }
        if name.contains("up_proj") || name.contains(".w3.") { return .mlpUp }
        if name.contains("down_proj") || name.contains(".w2.") { return .mlpDown }
        if name.contains("layernorm") || name.contains("norm") && name.hasSuffix(".weight") {
            return .normWeight
        }
        if name.contains("lm_head") { return .languageModelHead }
        if name.contains("experts") && name.contains("gate") { return .moeExpertGate }
        if name.contains("experts") && name.contains("up") { return .moeExpertUp }
        if name.contains("experts") && name.contains("down") { return .moeExpertDown }
        if name.contains("router") || name.contains("gate.weight") && name.contains("moe") {
            return .moeRouter
        }
        return .unknown
    }

    private func mapOriginalDType(_ dtype: SafetensorsDType) -> OriginalDType {
        switch dtype {
        case .float32: return .float32
        case .float16: return .float16
        case .bfloat16: return .bfloat16
        case .int32: return .int32
        case .int16: return .int16
        case .int8: return .int8
        default: return .unknown
        }
    }
}
