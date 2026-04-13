import Foundation
import Metal
import MetalCompiler
import Tokenizers

/// Immutable, shareable container for a compiled text-embedding bundle.
///
/// A container owns the loaded embedding assets, tokenizer, and prefill plan.
/// This is the primary public entry point for most embedding use cases.
/// Initialize ``TextEmbeddingContext`` with it only when you need isolated
/// mutable runtime state for repeated embedding work.
public final class TextEmbeddingContainer: @unchecked Sendable {
    let prefillPlan: MetalPrefillPlan
    let device: MTLDevice
    let tokenizer: any Tokenizer
    let runtime: SentenceTransformerTextEmbeddingRuntime
    let modelConfiguration: ModelConfiguration
    let postProcessor: MetalEmbeddingPostProcessor?

    init(
        prefillPlan: MetalPrefillPlan,
        device: MTLDevice,
        tokenizer: any Tokenizer,
        runtime: SentenceTransformerTextEmbeddingRuntime,
        configuration: ModelConfiguration,
        postProcessor: MetalEmbeddingPostProcessor? = nil
    ) {
        self.prefillPlan = prefillPlan
        self.device = device
        self.tokenizer = tokenizer
        self.runtime = runtime
        self.modelConfiguration = configuration
        self.postProcessor = postProcessor
    }

    public var configuration: ModelConfiguration {
        modelConfiguration
    }

    public var availablePromptNames: [String] {
        runtime.availablePromptNames
    }

    public var defaultPromptName: String? {
        runtime.defaultPromptName
    }

    /// Convenience one-shot embedding.
    ///
    /// Internally creates a fresh ``TextEmbeddingContext`` so repeated requests
    /// do not share mutable runtime state. This is the recommended high-level
    /// entry point for most embedding applications.
    public func embed(_ input: TextEmbeddingInput) throws -> [Float] {
        let context = try TextEmbeddingContext(self)
        return try context.embed(input)
    }

    /// Convenience one-shot embedding.
    ///
    /// Prefer ``embed(_:)`` for new code so the request can be passed as a
    /// first-class value.
    public func embed(
        _ text: String,
        promptName: String? = nil
    ) throws -> [Float] {
        try embed(TextEmbeddingInput(text, promptName: promptName))
    }

    internal var debugPrefillPlan: MetalPrefillPlan {
        prefillPlan
    }
}

/// Mutable execution context for text-embedding inference.
///
/// A context owns the isolated prefill runtime used to compute final hidden
/// states before pooling and dense projection.
///
/// Most applications should call ``TextEmbeddingContainer/embed(_:)``
/// and let the container create a fresh context internally. Use
/// `TextEmbeddingContext` when you want explicit ownership of reusable mutable
/// embedding state.
public final class TextEmbeddingContext: @unchecked Sendable {
    private var prefillModel: MetalPrefillModel
    private let tokenizer: any Tokenizer
    private let runtime: SentenceTransformerTextEmbeddingRuntime
    private let modelConfiguration: ModelConfiguration
    private let workspace: MetalEmbeddingWorkspace?

    public convenience init(_ container: TextEmbeddingContainer) throws {
        let isolatedPlan = try container.prefillPlan.makeRuntimeIsolatedCopy(device: container.device)
        let prefillModel = try MetalPrefillModel(plan: isolatedPlan, device: container.device)
        let workspace = try container.postProcessor?.makeWorkspace(device: container.device)
        self.init(
            prefillModel: prefillModel,
            tokenizer: container.tokenizer,
            runtime: container.runtime,
            configuration: container.modelConfiguration,
            workspace: workspace
        )
    }

    init(
        prefillModel: MetalPrefillModel,
        tokenizer: any Tokenizer,
        runtime: SentenceTransformerTextEmbeddingRuntime,
        configuration: ModelConfiguration,
        workspace: MetalEmbeddingWorkspace? = nil
    ) {
        self.prefillModel = prefillModel
        self.tokenizer = tokenizer
        self.runtime = runtime
        self.modelConfiguration = configuration
        self.workspace = workspace
    }

    public var configuration: ModelConfiguration {
        modelConfiguration
    }

    /// Embed a single text input using the configured sentence-transformers
    /// prompt and pooling pipeline.
    ///
    /// Prefer ``TextEmbeddingContainer/embed(_:)`` unless you need
    /// explicit context ownership.
    public func embed(_ input: TextEmbeddingInput) throws -> [Float] {
        let prepared = try runtime.prepare(
            text: input.text,
            promptName: input.promptName,
            tokenizer: tokenizer
        )
        let tokenIDs = prepared.tokenIDs.map(Int32.init)

        // GPU path: prefill + pool + dense + L2 in a single command buffer
        if let workspace {
            return try prefillModel.captureEmbeddingVector(
                tokens: tokenIDs,
                workspace: workspace,
                promptTokenCount: prepared.promptTokenCount
            )
        }

        // CPU fallback: read hidden states, process on CPU
        let hiddenStates = try prefillModel.finalHiddenStates(tokens: tokenIDs)
        return try runtime.embed(
            hiddenStates: hiddenStates,
            promptTokenCount: prepared.promptTokenCount
        )
    }

    /// Embed a single text input using the configured sentence-transformers
    /// prompt and pooling pipeline.
    ///
    /// Prefer ``embed(_:)`` for new code so the request can be passed as a
    /// first-class value.
    public func embed(
        _ text: String,
        promptName: String? = nil
    ) throws -> [Float] {
        try embed(TextEmbeddingInput(text, promptName: promptName))
    }

    internal var debugPrefillPlan: MetalPrefillPlan {
        prefillModel.prefillPlan
    }

    internal var debugWorkspace: MetalEmbeddingWorkspace? {
        workspace
    }
}
