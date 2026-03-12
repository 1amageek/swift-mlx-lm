import Testing
import TestHeartbeat
import Foundation
import MLX
import GGUFParser
@testable import MLXLM

/// Benchmark weight packing speed on real GGUF model.
@Suite("Packing Benchmark", .tags(.diagnostic), .heartbeat)
struct PackingBenchmarkTests {

    private static let cachedModelPath: String = {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport
            .appendingPathComponent("swift-mlx-lm/huggingface/unsloth--Qwen3.5-0.8B-GGUF/main/Qwen3.5-0.8B-Q4_K_M.gguf")
            .path
    }()

    @Test("Weight packing throughput")
    func weightPackingThroughput() throws {
        let url = URL(fileURLWithPath: Self.cachedModelPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            print("[SKIP] cached model not found at \(url.path)")
            return
        }

        let file = try GGUFFile.parse(url: url)
        let bridge = GGUFTensorBridge()

        let weightTensors = file.tensors.filter { t in
            t.name.hasSuffix(".weight") && t.dimensions.count >= 2
        }

        // Preload tensor data (exclude I/O from measurement)
        var tensorDataMap: [(GGUFTensorInfo, Data)] = []
        for tensor in weightTensors {
            let data = try file.tensorData(for: tensor)
            tensorDataMap.append((tensor, data))
        }

        // Warm up Metal kernel compilation
        if let first = tensorDataMap.first {
            _ = try bridge.convertDirect(tensor: first.0, data: first.1)
            eval()
        }

        // Measure
        let start = CFAbsoluteTimeGetCurrent()
        var totalElements = 0
        for (tensor, data) in tensorDataMap {
            let result = try bridge.convertDirect(tensor: tensor, data: data)
            switch result {
            case .float16(let arr): eval(arr)
            case .quantized(let w, let s, let b, _, _): eval(w, s, b)
            }
            totalElements += tensor.dimensions.reduce(1, *)
        }
        let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000.0

        let megaElements = Double(totalElements) / 1_000_000.0
        let throughput = megaElements / (ms / 1000.0)

        print("[bench] Model: Qwen3.5-0.8B-Q4_K_M (508MB)")
        print("[bench] Weight tensors: \(weightTensors.count)")
        print("[bench] Total elements: \(String(format: "%.1fM", megaElements))")
        print("[bench] Pack time: \(String(format: "%.0f", ms)) ms")
        print("[bench] Throughput: \(String(format: "%.0f", throughput)) M elements/s")
    }
}
