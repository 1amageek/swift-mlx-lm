import Testing
import Metal
@testable import SwiftLM
@testable import MetalCompiler

@Suite("EmbeddingGemma Real Bundle", .serialized)
struct EmbeddingGemmaRealBundleTests {
    @Test("Real EmbeddingGemma bundle returns normalized text embeddings", .timeLimit(.minutes(10)))
    func realBundleEmbeddings() async throws {
        guard let container = try await EmbeddingGemmaTestSupport.realEmbeddingGemmaContainer() else {
            print("[Skip] No local or configured EmbeddingGemma snapshot found")
            return
        }

        #expect(container.configuration.executionCapabilities.supportsTextEmbeddings)
        #expect(container.configuration.executionCapabilities.supportsTextGeneration == false)
        #expect(Set(container.availablePromptNames).isSuperset(of: ["document", "query"]))
        let embeddingKernelNames = container.prefillPlan.steps.compactMap(\.pipeline.label)
            .filter { $0.contains("embedding_lookup") }
        #expect(embeddingKernelNames.isEmpty == false)
        let quantizationSummary = container.prefillPlan.quantizationSummary(limit: 6)
        print("[EmbeddingGemma] quantization summary: \(quantizationSummary.replacingOccurrences(of: "\n", with: " | "))")
        let usesQuantizedEmbeddingLookup = embeddingKernelNames.contains { name in
            name.contains("_q4_") || name.contains("_q8_")
        }
        print(
            "[EmbeddingGemma] embedding lookup kernels: "
                + (usesQuantizedEmbeddingLookup ? "quantized" : "dense")
        )

        let isolatedPlan = try container.prefillPlan.makeRuntimeIsolatedCopy(device: container.device)
        var prefillModel = try MetalPrefillModel(plan: isolatedPlan, device: container.device)
        let preparedQuery = try container.runtime.prepare(
            text: "swift metal inference",
            promptName: "query",
            tokenizer: container.tokenizer
        )
        if let directory = try EmbeddingGemmaTestSupport.optionalRealEmbeddingGemmaDirectory() {
            let resources = try ModelBundleInspector().inspect(directory: directory)
            let weightStore = try STAFCacheLoader().load(resources: resources, device: container.device)
            let cpuWeights = CPUWeightStore(weights: weightStore)
            let embeddingTensor = try cpuWeights.floatTensor(named: "model.embed_tokens.weight")
            let embeddingShape = try cpuWeights.shape(named: "model.embed_tokens.weight")
            let lastTokenID = try #require(preparedQuery.tokenIDs.last)
            let expectedEmbedding = embeddingRow(
                values: embeddingTensor,
                shape: embeddingShape,
                tokenID: Int(lastTokenID),
                scale: Float(resources.config.hiddenSize).squareRoot()
            )
            #expect(l2Norm(expectedEmbedding) > 0.001)

            let embeddingStepIndex = try #require(
                container.prefillPlan.steps.firstIndex { step in
                    (step.pipeline.label ?? "").contains("embedding_lookup")
                }
            )
            let embeddingStep = container.prefillPlan.steps[embeddingStepIndex]
            let weightBinding = try #require(
                embeddingStep.bindings.buffers.first { $0.index == 1 }
            )
            let embeddingEntry = try #require(weightStore.entries["model.embed_tokens.weight"])
            #expect(weightBinding.offset == embeddingEntry.bufferOffset)
            #expect(embeddingStep.bindings.constantPolicy == .residentConstantBuffer)
            let boundPrefix = readBytes(
                from: weightBinding.buffer,
                offset: weightBinding.offset,
                count: 16
            )
            let expectedPrefix = readBytes(
                from: weightStore.buffer,
                offset: embeddingEntry.bufferOffset,
                count: 16
            )
            #expect(boundPrefix == expectedPrefix)
            let dimensionConstant = try #require(
                embeddingStep.bindings.constants.first { $0.index == 3 }
            )
            let scaleConstant = try #require(
                embeddingStep.bindings.constants.first { $0.index == 5 }
            )
            let boundEmbeddingDimension = readUInt32(from: dimensionConstant)
            let boundEmbeddingScale = readFloat(from: scaleConstant)
            #expect(boundEmbeddingDimension == UInt32(resources.config.hiddenSize))
            #expect(abs(boundEmbeddingScale - Float(resources.config.hiddenSize).squareRoot()) < 0.001)
            let directEmbedding = try runDirectEmbeddingLookup(
                pipeline: embeddingStep.pipeline,
                tokenID: Int32(lastTokenID),
                weightBuffer: weightBinding.buffer,
                weightOffset: weightBinding.offset,
                embeddingDimension: resources.config.hiddenSize,
                scale: boundEmbeddingScale
            )
            #expect(l2Norm(directEmbedding) > 0.001)
            #expect(cosineSimilarity(expectedEmbedding, directEmbedding) > 0.99)
            let prefillResidency = try MetalResidencyLease.combined(
                label: "swift-lm.embeddinggemma.prefill",
                leases: [
                    MetalResidencyLease.required(
                        device: container.device,
                        label: "swift-lm.embeddinggemma.runtime",
                        buffers: isolatedPlan.buffers.runtimeResidencyBuffers
                    ),
                    MetalResidencyLease.required(
                        device: container.device,
                        label: "swift-lm.embeddinggemma.weights",
                        buffers: isolatedPlan.buffers.weightResidencyBuffers
                    ),
                    MetalResidencyLease.required(
                        device: container.device,
                        label: "swift-lm.embeddinggemma.supplemental",
                        buffers: isolatedPlan.supplementalResidencyBuffers
                    ),
                ]
            )
            var diagnosticSubmission = try MetalSubmissionContext(device: container.device)
            let embeddingSnapshots = try MetalPrefillExecutor().captureLastTokenHiddenSnapshots(
                prefillPlan: isolatedPlan,
                submission: &diagnosticSubmission,
                position: 0,
                tokens: preparedQuery.tokenIDs.map(Int32.init),
                stepIndices: [embeddingStepIndex],
                ephemeralResidency: prefillResidency
            )
            let embeddingSnapshot = try #require(embeddingSnapshots[embeddingStepIndex])
            #expect(l2Norm(embeddingSnapshot) > 0.001)
            #expect(cosineSimilarity(expectedEmbedding, embeddingSnapshot) > 0.99)
            let lowLevelHiddenStates = try MetalPrefillExecutor().captureFinalHiddenRows(
                prefillPlan: isolatedPlan,
                submission: &diagnosticSubmission,
                position: 0,
                tokens: preparedQuery.tokenIDs.map(Int32.init),
                ephemeralResidency: prefillResidency
            )
            #expect(lowLevelHiddenStates.contains { l2Norm($0) > 0.001 })
        }
        let hiddenStates = try prefillModel.finalHiddenStates(tokens: preparedQuery.tokenIDs.map(Int32.init))
        #expect(hiddenStates.isEmpty == false)
        let hiddenNorms = hiddenStates.map(l2Norm)
        #expect(hiddenNorms.contains { $0 > 0.001 })
        if let directory = try EmbeddingGemmaTestSupport.optionalRealEmbeddingGemmaDirectory() {
            let resources = try ModelBundleInspector().inspect(directory: directory)
            let metadata = try SentenceTransformerMetadata.load(from: resources)
            let pooled = try pool(
                hiddenStates: hiddenStates,
                pooling: metadata.pooling,
                promptTokenCount: preparedQuery.promptTokenCount
            )
            #expect(l2Norm(pooled) > 0.001)
            let runtimeQuery = try container.runtime.embed(
                hiddenStates: hiddenStates,
                promptTokenCount: preparedQuery.promptTokenCount
            )
            #expect(l2Norm(runtimeQuery) > 0.001)
        }

        let query = try container.embed("swift metal inference", promptName: "query")
        let relevant = try container.embed(
            "SwiftLM performs Metal inference on Apple Silicon.",
            promptName: "document"
        )
        let unrelated = try container.embed(
            "A ripe banana is yellow and curved.",
            promptName: "document"
        )

        #expect(query.count == 768)
        #expect(relevant.count == 768)
        #expect(unrelated.count == 768)

        let queryNorm = l2Norm(query)
        let relevantNorm = l2Norm(relevant)
        let unrelatedNorm = l2Norm(unrelated)
        #expect(abs(queryNorm - 1) < 0.01)
        #expect(abs(relevantNorm - 1) < 0.01)
        #expect(abs(unrelatedNorm - 1) < 0.01)

        let relevantScore = cosineSimilarity(query, relevant)
        let unrelatedScore = cosineSimilarity(query, unrelated)
        #expect(relevantScore.isFinite)
        #expect(unrelatedScore.isFinite)
        #expect(relevantScore > unrelatedScore)
    }

    private func l2Norm(_ values: [Float]) -> Float {
        values.reduce(into: Float(0)) { partial, value in
            partial += value * value
        }.squareRoot()
    }

    private func cosineSimilarity(_ lhs: [Float], _ rhs: [Float]) -> Float {
        precondition(lhs.count == rhs.count)
        return zip(lhs, rhs).reduce(into: Float(0)) { partial, pair in
            partial += pair.0 * pair.1
        }
    }

    private func embeddingRow(
        values: [Float],
        shape: [Int],
        tokenID: Int,
        scale: Float
    ) -> [Float] {
        precondition(shape.count == 2)
        let dimension = shape[1]
        let startIndex = tokenID * dimension
        return Array(values[startIndex..<(startIndex + dimension)]).map { $0 * scale }
    }

    private func readBytes(
        from buffer: MTLBuffer,
        offset: Int,
        count: Int
    ) -> [UInt8] {
        let pointer = buffer.contents().advanced(by: offset).assumingMemoryBound(to: UInt8.self)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func readUInt32(from binding: MetalConstantBinding) -> UInt32 {
        guard case .buffer(let bufferBinding) = binding else {
            fatalError("Expected resident constant buffer binding")
        }
        return bufferBinding.buffer.contents()
            .advanced(by: bufferBinding.offset)
            .load(as: UInt32.self)
    }

    private func readFloat(from binding: MetalConstantBinding) -> Float {
        guard case .buffer(let bufferBinding) = binding else {
            fatalError("Expected resident constant buffer binding")
        }
        return bufferBinding.buffer.contents()
            .advanced(by: bufferBinding.offset)
            .load(as: Float.self)
    }

    private func runDirectEmbeddingLookup(
        pipeline: MTLComputePipelineState,
        tokenID: Int32,
        weightBuffer: MTLBuffer,
        weightOffset: Int,
        embeddingDimension: Int,
        scale: Float
    ) throws -> [Float] {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }
        var tokenID = tokenID
        let tokenBuffer = try #require(
            device.makeBuffer(
                bytes: &tokenID,
                length: MemoryLayout<Int32>.stride,
                options: .storageModeShared
            )
        )
        let outputBuffer = try #require(
            device.makeBuffer(
                length: embeddingDimension * MemoryLayout<Float>.stride,
                options: .storageModeShared
            )
        )
        memset(outputBuffer.contents(), 0, outputBuffer.length)

        var dimension = UInt32(embeddingDimension)
        var sequenceLength: UInt32 = 1
        var scale = scale

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(tokenBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: weightOffset, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&sequenceLength, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&scale, length: MemoryLayout<Float>.stride, index: 5)
        let threadCount = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let gridWidth = (embeddingDimension + threadCount - 1) / threadCount
        encoder.dispatchThreadgroups(
            MTLSize(width: gridWidth, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: embeddingDimension)
        return Array(UnsafeBufferPointer(start: pointer, count: embeddingDimension))
    }

    private func pool(
        hiddenStates: [[Float]],
        pooling: SentenceTransformerMetadata.Pooling,
        promptTokenCount: Int
    ) throws -> [Float] {
        let startIndex = pooling.includePrompt ? 0 : promptTokenCount
        #expect(startIndex < hiddenStates.count)
        let selected = Array(hiddenStates[startIndex...])
        let dimension = try #require(selected.first?.count)
        switch pooling.strategy {
        case .mean:
            var sums = [Float](repeating: 0, count: dimension)
            for row in selected {
                for index in 0..<dimension {
                    sums[index] += row[index]
                }
            }
            let count = Float(selected.count)
            return sums.map { $0 / count }
        case .cls:
            return selected[0]
        case .max:
            var output = selected[0]
            for row in selected.dropFirst() {
                for index in 0..<dimension {
                    output[index] = max(output[index], row[index])
                }
            }
            return output
        case .lastToken:
            return try #require(selected.last)
        }
    }
}
