import Testing
import Foundation
import GGUFParser
import GGUFTokenizer
@testable import MLXLM

/// Profile the full model loading pipeline to identify bottlenecks.
///
/// Uses the built-in timing instrumentation in GGUFModelLoader to capture
/// per-stage durations. Results are printed to stdout for analysis.
@Suite("GGUFModelLoader Profiling", .tags(.diagnostic), .serialized)
struct LoaderProfilingTests {

    private static let cachedModelURL: URL = {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport
            .appendingPathComponent("swift-mlx-lm/huggingface/unsloth--Qwen3.5-0.8B-GGUF/main/Qwen3.5-0.8B-Q4_K_M.gguf")
    }()

    /// Profile standard loading path (MLXNN Module tree).
    ///
    /// Timing breakdown is printed by GGUFModelLoader's internal instrumentation.
    /// Look for `[GGUFModelLoader]` lines in output.
    @Test("Standard path load profile")
    func standardPathProfile() throws {
        let url = Self.cachedModelURL
        guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
            print("[SKIP] cached model not found at \(url.path(percentEncoded: false))")
            return
        }

        print("\n[profile] ==========================================")
        print("[profile] Standard Path: loadContext()")
        print("[profile] Model: Qwen3.5-0.8B-Q4_K_M (508MB)")
        print("[profile] ==========================================")

        let loader = GGUFModelLoader()
        let totalStart = CFAbsoluteTimeGetCurrent()
        let context = try loader.loadContext(url: url)
        let totalMs = (CFAbsoluteTimeGetCurrent() - totalStart) * 1000.0

        print("[profile] ------------------------------------------")
        print("[profile] TOTAL: \(String(format: "%.0f", totalMs)) ms")
        print("[profile] Model: \(context.configuration.name)")
        print("[profile] ==========================================\n")
    }

    /// Profile compiled loading path (IR → lowered inference engine).
    @Test("Compiled path load profile")
    func compiledPathProfile() throws {
        let url = Self.cachedModelURL
        guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
            print("[SKIP] cached model not found at \(url.path(percentEncoded: false))")
            return
        }

        print("\n[profile] ==========================================")
        print("[profile] Compiled Path: loadCompiledContext()")
        print("[profile] Model: Qwen3.5-0.8B-Q4_K_M (508MB)")
        print("[profile] ==========================================")

        let loader = GGUFModelLoader()
        let totalStart = CFAbsoluteTimeGetCurrent()
        let context = try loader.loadCompiledContext(url: url)
        let totalMs = (CFAbsoluteTimeGetCurrent() - totalStart) * 1000.0

        print("[profile] ------------------------------------------")
        print("[profile] TOTAL: \(String(format: "%.0f", totalMs)) ms")
        print("[profile] Model: \(context.configuration.name)")
        print("[profile] ==========================================\n")
    }

    /// Profile individual stages of the loading pipeline separately.
    ///
    /// Measures GGUF parse and tokenizer creation in isolation to precisely
    /// quantify the effect of deferred array optimization.
    @Test("Stage-by-stage profile")
    func stageByStageProfile() throws {
        let url = Self.cachedModelURL
        guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
            print("[SKIP] cached model not found at \(url.path(percentEncoded: false))")
            return
        }

        print("\n[profile] ==========================================")
        print("[profile] Stage-by-Stage Profile")
        print("[profile] Model: Qwen3.5-0.8B-Q4_K_M (508MB)")
        print("[profile] ==========================================")

        // Stage 1: GGUF parse
        var t0 = CFAbsoluteTimeGetCurrent()
        let file = try GGUFFile.parse(url: url)
        let parseMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
        print("[profile] GGUF parse:          \(String(format: "%6.0f", parseMs)) ms  (tensors=\(file.tensors.count), deferred=\(file.deferredArrays.count))")

        // Stage 2a: readDeferredVocabulary (single-pass vocab + tokenToID)
        t0 = CFAbsoluteTimeGetCurrent()
        let vocabData = file.readDeferredVocabulary("tokenizer.ggml.tokens")
        let vocabReadMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
        print("[profile] readDeferredVocab:   \(String(format: "%6.0f", vocabReadMs)) ms  (count=\(vocabData?.vocabulary.count ?? 0))")

        // Stage 2b: readDeferredStringDictionary (merges → rank dict)
        t0 = CFAbsoluteTimeGetCurrent()
        let mergeRanks = file.readDeferredStringDictionary("tokenizer.ggml.merges")
        let mergesMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
        print("[profile] readDeferredMerges:  \(String(format: "%6.0f", mergesMs)) ms  (count=\(mergeRanks?.count ?? 0))")

        // Stage 2c: readDeferredInt32Array (token types)
        t0 = CFAbsoluteTimeGetCurrent()
        let tokenTypes = file.readDeferredInt32Array("tokenizer.ggml.token_type")
        let typesMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
        print("[profile] readDeferredTypes:   \(String(format: "%6.0f", typesMs)) ms  (count=\(tokenTypes?.count ?? 0))")

        // Stage 2d: Full tokenizer creation
        t0 = CFAbsoluteTimeGetCurrent()
        let tokenizer = try GGUFTokenizerFactory.create(from: file)
        let tokenizerMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
        print("[profile] Tokenizer creation:  \(String(format: "%6.0f", tokenizerMs)) ms  (vocab=\(tokenizer.vocabularySize))")

        // Stage 3: vocabSize (check if deferred lookup is fast)
        t0 = CFAbsoluteTimeGetCurrent()
        let vocabSize = file.vocabularySize
        let vocabMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
        print("[profile] vocabularySize:      \(String(format: "%6.1f", vocabMs)) ms  (size=\(vocabSize ?? 0))")

        let totalMs = parseMs + vocabReadMs + mergesMs + typesMs
        print("[profile] ------------------------------------------")
        print("[profile] Parse + Deferred:    \(String(format: "%6.0f", totalMs)) ms")
        print("[profile] (Tokenizer reread):  \(String(format: "%6.0f", tokenizerMs)) ms  (reads again from mmap)")
        print("[profile] ==========================================\n")
    }
}
