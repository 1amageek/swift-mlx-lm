import Foundation
import Testing
@testable import SwiftLM

@Suite("Prompt State Trace Diagnostics", .serialized)
struct PromptStateTraceDiagnosticTests {
    @Test("Inspect LFM prompt-state trace", .timeLimit(.minutes(10)))
    func inspectLFMPromptStateTrace() async throws {
        let localModelDirectory = URL(
            fileURLWithPath: "/Users/1amageek/Desktop/swift-lm/TestData/LFM2.5-1.2B-Thinking"
        )
        let configURL = localModelDirectory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            print("[Skip] No local LFM bundle found at \(localModelDirectory.path)")
            return
        }

        let container = try await ModelBundleLoader().load(directory: localModelDirectory)
        let prepared = try await container.prepare( ModelInput(prompt: RealOutputAssertionSupport.strictCapitalPrompt)
        )
        let prompt = try container.makeExecutablePrompt(from: prepared)
        let trace = try container.debugPromptStateGenerationTrace(
            prompt: prompt,
            parameters: RealOutputAssertionSupport.greedyParameters()
        )

        printBoundary(trace.directBoundary, label: "LFM direct boundary")
        printBoundary(trace.restoredBoundary, label: "LFM restored boundary")
        printStepDiffs(
            directSteps: trace.directSteps,
            restoredSteps: trace.restoredSteps,
            label: "LFM"
        )
    }

    @Test("Inspect Qwen prompt-state trace", .timeLimit(.minutes(10)))
    func inspectQwenPromptStateTrace() async throws {
        guard let directory = try QwenVisionTestSupport.optionalRealQwen3VLDirectory() else {
            print("[Skip] No local Qwen3.5 snapshot found")
            return
        }

        let container = try await ModelBundleLoader().load(directory: directory)
        let prepared = try await container.prepare( ModelInput(prompt: RealOutputAssertionSupport.strictCapitalPrompt)
        )
        let prompt = try container.makeExecutablePrompt(from: prepared)
        let trace = try container.debugPromptStateGenerationTrace(
            prompt: prompt,
            parameters: RealOutputAssertionSupport.greedyParameters()
        )

        printBoundary(trace.directBoundary, label: "Qwen direct boundary")
        printBoundary(trace.restoredBoundary, label: "Qwen restored boundary")
        printStepDiffs(
            directSteps: trace.directSteps,
            restoredSteps: trace.restoredSteps,
            label: "Qwen"
        )
    }

    @Test("Inspect LFM repeated greedy determinism", .timeLimit(.minutes(10)))
    func inspectLFMRepeatedGreedyDeterminism() async throws {
        let localModelDirectory = URL(
            fileURLWithPath: "/Users/1amageek/Desktop/swift-lm/TestData/LFM2.5-1.2B-Thinking"
        )
        let configURL = localModelDirectory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            print("[Skip] No local LFM bundle found at \(localModelDirectory.path)")
            return
        }

        let loader = ModelBundleLoader()
        let firstContainer = try await loader.load(directory: localModelDirectory)
        let prepared = try await firstContainer.prepare( ModelInput(prompt: RealOutputAssertionSupport.strictCapitalPrompt)
        )
        let prompt = try firstContainer.makeExecutablePrompt(from: prepared)

        firstContainer.resetState()
        let sameContainerFirst = try firstContainer.debugGeneratedTokenIDs(
            prompt: prompt,
            parameters: RealOutputAssertionSupport.greedyParameters()
        )
        firstContainer.resetState()
        let sameContainerSecond = try firstContainer.debugGeneratedTokenIDs(
            prompt: prompt,
            parameters: RealOutputAssertionSupport.greedyParameters()
        )

        let secondContainer = try await loader.load(directory: localModelDirectory)
        let secondPrepared = try await secondContainer.prepare( ModelInput(prompt: RealOutputAssertionSupport.strictCapitalPrompt)
        )
        let secondPrompt = try secondContainer.makeExecutablePrompt(from: secondPrepared)
        secondContainer.resetState()
        let freshContainerRun = try secondContainer.debugGeneratedTokenIDs(
            prompt: secondPrompt,
            parameters: RealOutputAssertionSupport.greedyParameters()
        )

        print("[LFM determinism same-container first] \(sameContainerFirst)")
        print("[LFM determinism same-container second] \(sameContainerSecond)")
        print("[LFM determinism fresh-container] \(freshContainerRun)")
        print("[LFM determinism same-container first text] \(firstContainer.decode( sameContainerFirst))")
        print("[LFM determinism same-container second text] \(firstContainer.decode( sameContainerSecond))")
        print("[LFM determinism fresh-container text] \(secondContainer.decode( freshContainerRun))")
    }

    private func printBoundary(_ boundary: DebugGenerationBoundaryState, label: String) {
        print("[\(label)] firstToken=\(boundary.firstToken) position=\(boundary.position) positionBuffer=\(boundary.positionBufferValue)")
        print("[\(label)] ropeAxes=\(boundary.ropePositionAxes) tokenIn=\(boundary.tokenIn) tokenOut=\(boundary.tokenOut)")
        print("[\(label)] recent=\(boundary.recentTokenIDs)")
        print("[\(label)] logitsFingerprint=\(boundary.processedLogitsFingerprint) top=\(boundary.topLogits.map(\.tokenID))")
    }

    private func printStepDiffs(
        directSteps: [DebugGenerationStepTrace],
        restoredSteps: [DebugGenerationStepTrace],
        label: String
    ) {
        let count = max(directSteps.count, restoredSteps.count)
        for index in 0..<count {
            let direct = directSteps.indices.contains(index) ? directSteps[index] : nil
            let restored = restoredSteps.indices.contains(index) ? restoredSteps[index] : nil
            print("[\(label) step \(index)] direct=\(describe(direct))")
            print("[\(label) step \(index)] restored=\(describe(restored))")
        }
    }

    private func describe(_ step: DebugGenerationStepTrace?) -> String {
        guard let step else { return "nil" }
        return [
            "input=\(step.inputTokenID)",
            "position=\(step.positionBefore)",
            "tokenInBefore=\(step.tokenInBefore)",
            "tokenOutBefore=\(step.tokenOutBefore)",
            "argmax=\(step.argmaxTokenID)",
            "sampled=\(step.sampledTokenID)",
            "tokenOutAfter=\(step.tokenOutAfter)",
            "rng=\(step.rngStateBefore)",
            String(format: "random=%.8f", step.randomValue),
            "fingerprint=\(step.processedLogitsFingerprint)",
            "top=\(step.topLogits.map(\.tokenID))",
            "recent=\(step.recentTokenIDs)",
        ].joined(separator: " ")
    }
}
