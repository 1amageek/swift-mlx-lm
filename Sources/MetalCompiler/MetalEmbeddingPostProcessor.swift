@preconcurrency import Metal

// MARK: - Public Types

/// Pooling strategy for sentence embeddings.
public enum EmbeddingPoolingStrategy: Sendable {
    case mean
    case cls
    case max
    case lastToken
}

/// Activation function for dense projection layers.
public enum EmbeddingDenseActivation: Sendable {
    case identity
    case tanh
    case relu
    case gelu
}

/// Descriptor for a single dense projection layer with GPU-resident weights.
public struct EmbeddingDenseLayerDescriptor: @unchecked Sendable {
    public let weightBuffer: MTLBuffer
    public let biasBuffer: MTLBuffer?
    public let inputDimension: Int
    public let outputDimension: Int
    public let activation: EmbeddingDenseActivation

    public init(
        weightBuffer: MTLBuffer,
        biasBuffer: MTLBuffer?,
        inputDimension: Int,
        outputDimension: Int,
        activation: EmbeddingDenseActivation
    ) {
        self.weightBuffer = weightBuffer
        self.biasBuffer = biasBuffer
        self.inputDimension = inputDimension
        self.outputDimension = outputDimension
        self.activation = activation
    }
}

// MARK: - MSL Kernel Source

private let embeddingPostProcessingMSL = """
#include <metal_stdlib>
using namespace metal;

// MARK: - Mean Pooling

// Sums rows [startRow..sequenceLength), divides by count.
// Grid: (hiddenDimension, 1, 1), threadgroup: (min(hiddenDim, maxThreads), 1, 1)
// Argument table:
//   [0] output (float*)       — pooled vector, length = hiddenDimension
//   [1] hidden (const float*) — row-major [seqLen x hiddenDim], base-offset pre-applied
//   [2] constants (const uint*) — [sequenceLength, hiddenDimension, startRow, rowStrideFloats]
kernel void embedding_mean_pool(
    device float* output          [[buffer(0)]],
    const device float* hidden    [[buffer(1)]],
    const device uint* constants  [[buffer(2)]],
    uint tid                      [[thread_position_in_grid]]
) {
    const uint sequenceLength = constants[0];
    const uint hiddenDimension = constants[1];
    const uint startRow = constants[2];
    const uint rowStrideFloats = constants[3];
    if (tid >= hiddenDimension) return;

    const uint count = sequenceLength - startRow;
    if (count == 0) { output[tid] = 0.0f; return; }

    float sum = 0.0f;
    for (uint row = startRow; row < sequenceLength; row++) {
        sum += hidden[row * rowStrideFloats + tid];
    }
    output[tid] = sum / float(count);
}

// MARK: - CLS Pooling

// Copies the first non-prompt row.
// Grid: (hiddenDimension, 1, 1)
kernel void embedding_cls_pool(
    device float* output          [[buffer(0)]],
    const device float* hidden    [[buffer(1)]],
    const device uint* constants  [[buffer(2)]],
    uint tid                      [[thread_position_in_grid]]
) {
    const uint hiddenDimension = constants[1];
    const uint startRow = constants[2];
    const uint rowStrideFloats = constants[3];
    if (tid >= hiddenDimension) return;

    output[tid] = hidden[startRow * rowStrideFloats + tid];
}

// MARK: - Max Pooling

// Takes element-wise max across rows [startRow..sequenceLength).
// Grid: (hiddenDimension, 1, 1)
kernel void embedding_max_pool(
    device float* output          [[buffer(0)]],
    const device float* hidden    [[buffer(1)]],
    const device uint* constants  [[buffer(2)]],
    uint tid                      [[thread_position_in_grid]]
) {
    const uint sequenceLength = constants[0];
    const uint hiddenDimension = constants[1];
    const uint startRow = constants[2];
    const uint rowStrideFloats = constants[3];
    if (tid >= hiddenDimension) return;

    float maxVal = hidden[startRow * rowStrideFloats + tid];
    for (uint row = startRow + 1; row < sequenceLength; row++) {
        maxVal = max(maxVal, hidden[row * rowStrideFloats + tid]);
    }
    output[tid] = maxVal;
}

// MARK: - Last Token Pooling

// Copies the last row (sequenceLength - 1).
// Grid: (hiddenDimension, 1, 1)
kernel void embedding_last_token_pool(
    device float* output          [[buffer(0)]],
    const device float* hidden    [[buffer(1)]],
    const device uint* constants  [[buffer(2)]],
    uint tid                      [[thread_position_in_grid]]
) {
    const uint sequenceLength = constants[0];
    const uint hiddenDimension = constants[1];
    const uint rowStrideFloats = constants[3];
    if (tid >= hiddenDimension) return;

    const uint lastRow = sequenceLength - 1;
    output[tid] = hidden[lastRow * rowStrideFloats + tid];
}

// MARK: - Dense GEMV

// Matrix-vector multiply: output = W * input + bias, then activation.
// W is row-major [outDim x inDim].
// Grid: (outDim, 1, 1), threadgroup: (256, 1, 1)
//
// Argument table:
//   [0] output (float*)
//   [1] input (const float*)
//   [2] weights (const float*)  — row-major [outDim x inDim]
//   [3] bias (const float*)     — length outDim, or nullptr
//   [4] constants (const uint*) — [inDim, outDim, activationType, hasBias]
kernel void embedding_dense_gemv(
    device float* output              [[buffer(0)]],
    const device float* input         [[buffer(1)]],
    const device float* weights       [[buffer(2)]],
    const device float* bias          [[buffer(3)]],
    const device uint* constants      [[buffer(4)]],
    uint tid                          [[thread_position_in_grid]],
    uint simd_lane                    [[thread_index_in_simdgroup]],
    uint simd_group                   [[simdgroup_index_in_threadgroup]],
    uint tg_size                      [[threads_per_threadgroup]]
) {
    const uint inDim = constants[0];
    const uint outDim = constants[1];
    const uint activationType = constants[2];
    const uint hasBias = constants[3];
    if (tid >= outDim) return;

    // Each thread computes one output element via dot product
    const device float* row = weights + tid * inDim;
    float dot = 0.0f;

    // Vectorized accumulation (float4)
    const uint vecCount = inDim / 4;
    const uint remainder = inDim % 4;
    const device float4* rowVec = (const device float4*)row;
    const device float4* inVec = (const device float4*)input;

    for (uint i = 0; i < vecCount; i++) {
        float4 w = rowVec[i];
        float4 x = inVec[i];
        dot += w.x * x.x + w.y * x.y + w.z * x.z + w.w * x.w;
    }
    for (uint i = vecCount * 4; i < inDim; i++) {
        dot += row[i] * input[i];
    }

    if (hasBias) {
        dot += bias[tid];
    }

    // Activation
    // 0 = identity, 1 = tanh, 2 = relu, 3 = gelu
    if (activationType == 1) {
        dot = tanh(dot);
    } else if (activationType == 2) {
        dot = max(0.0f, dot);
    } else if (activationType == 3) {
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float coeff = 0.79788456f; // sqrt(2/pi)
        const float cubic = 0.044715f;
        float inner = coeff * (dot + cubic * dot * dot * dot);
        dot = 0.5f * dot * (1.0f + tanh(inner));
    }

    output[tid] = dot;
}

// MARK: - L2 Normalize

// In-place L2 normalization using cooperative threadgroup reduction.
// Grid: (1, 1, 1), threadgroup: (256, 1, 1)
//
// Argument table:
//   [0] data (float*)           — vector to normalize in-place
//   [1] constants (const uint*) — [dimension]
kernel void embedding_l2_normalize(
    device float* data               [[buffer(0)]],
    const device uint* constants     [[buffer(1)]],
    uint tid                         [[thread_index_in_threadgroup]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_group                  [[simdgroup_index_in_threadgroup]],
    uint tg_size                     [[threads_per_threadgroup]]
) {
    const uint dimension = constants[0];
    const uint simdGroupCount = (tg_size + 31) / 32;

    threadgroup float shared_sums[32]; // max 32 simd groups in a threadgroup

    // Phase 1: each thread accumulates partial squared sum
    float partialSum = 0.0f;
    for (uint i = tid; i < dimension; i += tg_size) {
        float val = data[i];
        partialSum += val * val;
    }

    // Phase 2: simd-level reduction
    partialSum = simd_sum(partialSum);
    if (simd_lane == 0) {
        shared_sums[simd_group] = partialSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: first simd group reduces across simd groups
    if (simd_group == 0) {
        float val = (simd_lane < simdGroupCount) ? shared_sums[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            shared_sums[0] = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float squaredNorm = shared_sums[0];
    if (squaredNorm <= 0.0f) return;

    float scale = rsqrt(squaredNorm);

    // Phase 4: scale in-place
    for (uint i = tid; i < dimension; i += tg_size) {
        data[i] *= scale;
    }
}
"""

// MARK: - MetalEmbeddingPostProcessor

/// Shared, immutable GPU post-processing infrastructure for sentence embeddings.
///
/// Owns compiled pipeline states and weight buffers. Thread-safe for concurrent
/// reads. Create one per model load, then call ``makeWorkspace(device:)`` to get
/// per-context mutable workspace for actual dispatch.
public struct MetalEmbeddingPostProcessor: @unchecked Sendable {
    let poolingPipeline: MTLComputePipelineState
    let poolingStrategy: EmbeddingPoolingStrategy
    let densePipelines: [MTLComputePipelineState]
    let denseDescriptors: [EmbeddingDenseLayerDescriptor]
    let l2NormalizePipeline: MTLComputePipelineState?
    let l2NormalizeEnabled: Bool
    let finalOutputDimension: Int

    /// All weight buffers that must remain resident during dispatch.
    public var residencyBuffers: [MTLBuffer] {
        var buffers: [MTLBuffer] = []
        for descriptor in denseDescriptors {
            buffers.append(descriptor.weightBuffer)
            if let bias = descriptor.biasBuffer {
                buffers.append(bias)
            }
        }
        return buffers
    }

    /// Compile all post-processing pipelines from MSL source.
    public static func compile(
        device: MTLDevice,
        poolingStrategy: EmbeddingPoolingStrategy,
        denseLayers: [EmbeddingDenseLayerDescriptor],
        l2NormalizeEnabled: Bool
    ) throws -> MetalEmbeddingPostProcessor {
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: embeddingPostProcessingMSL, options: options)
        } catch {
            throw MetalCompilerError.deviceSetupFailed(
                "Failed to compile embedding post-processing MSL: \(error)"
            )
        }

        // Pooling pipeline
        let poolingKernelName: String
        switch poolingStrategy {
        case .mean: poolingKernelName = "embedding_mean_pool"
        case .cls: poolingKernelName = "embedding_cls_pool"
        case .max: poolingKernelName = "embedding_max_pool"
        case .lastToken: poolingKernelName = "embedding_last_token_pool"
        }
        let poolingPipeline = try Self.makePipeline(
            library: library, device: device, functionName: poolingKernelName
        )

        // Dense GEMV pipelines (one per layer, same kernel)
        var densePipelines: [MTLComputePipelineState] = []
        if !denseLayers.isEmpty {
            let gemvPipeline = try Self.makePipeline(
                library: library, device: device, functionName: "embedding_dense_gemv"
            )
            densePipelines = Array(repeating: gemvPipeline, count: denseLayers.count)
        }

        // L2 normalize pipeline
        let l2Pipeline: MTLComputePipelineState?
        if l2NormalizeEnabled {
            l2Pipeline = try Self.makePipeline(
                library: library, device: device, functionName: "embedding_l2_normalize"
            )
        } else {
            l2Pipeline = nil
        }

        // Determine final output dimension
        let finalDimension: Int
        if let lastDense = denseLayers.last {
            finalDimension = lastDense.outputDimension
        } else {
            // No dense layers — output dimension equals hidden dimension (determined at encode time)
            finalDimension = 0
        }

        return MetalEmbeddingPostProcessor(
            poolingPipeline: poolingPipeline,
            poolingStrategy: poolingStrategy,
            densePipelines: densePipelines,
            denseDescriptors: denseLayers,
            l2NormalizePipeline: l2Pipeline,
            l2NormalizeEnabled: l2NormalizeEnabled,
            finalOutputDimension: finalDimension
        )
    }

    /// Create a per-context mutable workspace for dispatch.
    public func makeWorkspace(device: MTLDevice) throws -> MetalEmbeddingWorkspace {
        try MetalEmbeddingWorkspace(postProcessor: self, device: device)
    }

    private static func makePipeline(
        library: MTLLibrary,
        device: MTLDevice,
        functionName: String
    ) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: functionName) else {
            throw MetalCompilerError.deviceSetupFailed(
                "Embedding MSL function not found: \(functionName)"
            )
        }
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = false
        do {
            return try device.makeComputePipelineState(
                descriptor: descriptor,
                options: [],
                reflection: nil
            )
        } catch {
            throw MetalCompilerError.deviceSetupFailed(
                "Failed to create pipeline for \(functionName): \(error)"
            )
        }
    }
}

// MARK: - MetalEmbeddingWorkspace

/// Per-context mutable workspace for embedding post-processing dispatch.
///
/// Owns intermediate and output buffers. Not thread-safe — each
/// ``TextEmbeddingContext`` should hold its own workspace.
///
/// **Constant buffer layout**: All dispatch constants are packed into a single
/// shared buffer at distinct offsets. This prevents the GPU from reading stale
/// values when multiple dispatches share a command buffer — the GPU reads all
/// constants at execution time (after commit), so each stage must have its own
/// non-overlapping region.
///
/// Layout (stride = 16 bytes per slot):
///   slot 0: pooling   [sequenceLength, hiddenDimension, startRow, rowStrideFloats]
///   slot 1: dense[0]  [inDim, outDim, activationType, hasBias]
///   slot 2: dense[1]  [inDim, outDim, activationType, hasBias]
///   ...
///   slot N: L2         [dimension, 0, 0, 0]
public struct MetalEmbeddingWorkspace: @unchecked Sendable {
    private let postProcessor: MetalEmbeddingPostProcessor
    private let intermediateA: MTLBuffer
    private let intermediateB: MTLBuffer
    private let outputBuffer: MTLBuffer
    private let constantBuffer: MTLBuffer

    private static let constantSlotStride = 4 * MemoryLayout<UInt32>.stride  // 16 bytes

    /// All workspace buffers that must remain resident during dispatch.
    public var residencyBuffers: [MTLBuffer] {
        [intermediateA, intermediateB, outputBuffer, constantBuffer]
    }

    /// All buffers (workspace + post-processor weights) needed for full residency.
    public var allResidencyBuffers: [MTLBuffer] {
        residencyBuffers + postProcessor.residencyBuffers
    }

    init(postProcessor: MetalEmbeddingPostProcessor, device: MTLDevice) throws {
        self.postProcessor = postProcessor

        // Determine maximum intermediate dimension across all dense layers
        var maxDimension = 0
        for descriptor in postProcessor.denseDescriptors {
            maxDimension = max(maxDimension, descriptor.inputDimension)
            maxDimension = max(maxDimension, descriptor.outputDimension)
        }
        // Ensure at least enough for hidden dimension (set at encode time via pool output)
        // Use a generous minimum — actual hidden dim is typically 768
        let bufferSize = max(maxDimension, 4096) * MemoryLayout<Float>.stride

        guard let intermediateA = device.makeBuffer(length: bufferSize, options: .storageModeShared),
              let intermediateB = device.makeBuffer(length: bufferSize, options: .storageModeShared)
        else {
            throw MetalCompilerError.deviceSetupFailed(
                "Failed to allocate embedding intermediate buffers"
            )
        }
        intermediateA.label = "swift-lm.embedding.intermediateA"
        intermediateB.label = "swift-lm.embedding.intermediateB"
        self.intermediateA = intermediateA
        self.intermediateB = intermediateB

        // Output buffer — final embedding vector
        let outputDimension = postProcessor.finalOutputDimension > 0
            ? postProcessor.finalOutputDimension
            : maxDimension
        let outputSize = max(outputDimension, 768) * MemoryLayout<Float>.stride
        guard let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            throw MetalCompilerError.deviceSetupFailed(
                "Failed to allocate embedding output buffer"
            )
        }
        outputBuffer.label = "swift-lm.embedding.output"
        self.outputBuffer = outputBuffer

        // Packed constant buffer: 1 pool + N dense + 1 L2 (optional)
        let denseCount = postProcessor.denseDescriptors.count
        let l2SlotCount = postProcessor.l2NormalizeEnabled ? 1 : 0
        let totalSlots = 1 + denseCount + l2SlotCount
        let constantSize = totalSlots * Self.constantSlotStride
        guard let constantBuffer = device.makeBuffer(length: constantSize, options: .storageModeShared) else {
            throw MetalCompilerError.deviceSetupFailed(
                "Failed to allocate embedding constant buffer"
            )
        }
        constantBuffer.label = "swift-lm.embedding.constants"
        self.constantBuffer = constantBuffer
    }

    /// Encode all post-processing dispatches (pool + dense + L2) onto an existing encoder.
    ///
    /// Call this after prefill steps have been encoded in the same command buffer.
    /// The hidden buffer must contain Float32 hidden states from prefill.
    ///
    /// All constants are written upfront into non-overlapping slots before any
    /// dispatch is encoded, so the GPU reads correct values for each stage.
    public func encode(
        encoder: MTL4ComputeCommandEncoder,
        argumentTable: MTL4ArgumentTable,
        hiddenBuffer: MTLBuffer,
        hiddenBaseOffset: Int,
        hiddenRowStride: Int,
        hiddenDimension: Int,
        sequenceLength: Int,
        promptTokenCount: Int
    ) {
        let startRow = promptTokenCount
        let rowStrideFloats = hiddenRowStride / MemoryLayout<Float>.stride
        let basePtr = constantBuffer.contents()

        // --- Write ALL constants upfront before any dispatch ---

        // Slot 0: pooling constants
        let poolConstPtr = (basePtr + poolingConstantOffset())
            .bindMemory(to: UInt32.self, capacity: 4)
        poolConstPtr[0] = UInt32(sequenceLength)
        poolConstPtr[1] = UInt32(hiddenDimension)
        poolConstPtr[2] = UInt32(startRow)
        poolConstPtr[3] = UInt32(rowStrideFloats)

        // Slots 1..N: dense layer constants
        for (layerIndex, descriptor) in postProcessor.denseDescriptors.enumerated() {
            let activationType: UInt32
            switch descriptor.activation {
            case .identity: activationType = 0
            case .tanh: activationType = 1
            case .relu: activationType = 2
            case .gelu: activationType = 3
            }

            let densePtr = (basePtr + denseConstantOffset(layerIndex: layerIndex))
                .bindMemory(to: UInt32.self, capacity: 4)
            densePtr[0] = UInt32(descriptor.inputDimension)
            densePtr[1] = UInt32(descriptor.outputDimension)
            densePtr[2] = activationType
            densePtr[3] = descriptor.biasBuffer != nil ? 1 : 0
        }

        // Slot N+1: L2 constants
        if postProcessor.l2NormalizeEnabled {
            let dimension: Int
            if let lastDense = postProcessor.denseDescriptors.last {
                dimension = lastDense.outputDimension
            } else {
                dimension = hiddenDimension
            }
            let l2Ptr = (basePtr + l2ConstantOffset())
                .bindMemory(to: UInt32.self, capacity: 4)
            l2Ptr[0] = UInt32(dimension)
            l2Ptr[1] = 0
            l2Ptr[2] = 0
            l2Ptr[3] = 0
        }

        // --- Now encode all dispatches ---

        // Determine which buffer receives pooling output
        let poolOutput: MTLBuffer
        if postProcessor.denseDescriptors.isEmpty && !postProcessor.l2NormalizeEnabled {
            poolOutput = outputBuffer
        } else {
            poolOutput = intermediateA
        }

        // Step 1: Pooling
        encoder.barrier(
            afterEncoderStages: .dispatch,
            beforeEncoderStages: .dispatch,
            visibilityOptions: .device
        )
        argumentTable.setAddress(poolOutput.gpuAddress, index: 0)
        argumentTable.setAddress(
            hiddenBuffer.gpuAddress + UInt64(hiddenBaseOffset),
            index: 1
        )
        argumentTable.setAddress(
            constantBuffer.gpuAddress + UInt64(poolingConstantOffset()),
            index: 2
        )
        encoder.setArgumentTable(argumentTable)
        encoder.setComputePipelineState(postProcessor.poolingPipeline)

        let poolThreads = min(
            hiddenDimension,
            postProcessor.poolingPipeline.maxTotalThreadsPerThreadgroup
        )
        let poolGroups = (hiddenDimension + poolThreads - 1) / poolThreads
        encoder.dispatchThreadgroups(
            threadgroupsPerGrid: MTLSize(width: poolGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: poolThreads, height: 1, depth: 1)
        )

        // Step 2: Dense layers (ping-pong between intermediateA and intermediateB)
        var currentInput = poolOutput
        for (layerIndex, descriptor) in postProcessor.denseDescriptors.enumerated() {
            let isLastDense = layerIndex == postProcessor.denseDescriptors.count - 1
            let currentOutput: MTLBuffer
            if isLastDense && !postProcessor.l2NormalizeEnabled {
                currentOutput = outputBuffer
            } else if currentInput === intermediateA {
                currentOutput = intermediateB
            } else {
                currentOutput = intermediateA
            }

            encoder.barrier(
                afterEncoderStages: .dispatch,
                beforeEncoderStages: .dispatch,
                visibilityOptions: .device
            )

            argumentTable.setAddress(currentOutput.gpuAddress, index: 0)
            argumentTable.setAddress(currentInput.gpuAddress, index: 1)
            argumentTable.setAddress(descriptor.weightBuffer.gpuAddress, index: 2)
            if let biasBuffer = descriptor.biasBuffer {
                argumentTable.setAddress(biasBuffer.gpuAddress, index: 3)
            } else {
                argumentTable.setAddress(descriptor.weightBuffer.gpuAddress, index: 3)
            }
            argumentTable.setAddress(
                constantBuffer.gpuAddress + UInt64(denseConstantOffset(layerIndex: layerIndex)),
                index: 4
            )
            encoder.setArgumentTable(argumentTable)
            encoder.setComputePipelineState(postProcessor.densePipelines[layerIndex])

            let outDim = descriptor.outputDimension
            let denseThreads = min(
                outDim,
                postProcessor.densePipelines[layerIndex].maxTotalThreadsPerThreadgroup
            )
            let denseGroups = (outDim + denseThreads - 1) / denseThreads
            encoder.dispatchThreadgroups(
                threadgroupsPerGrid: MTLSize(width: denseGroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: denseThreads, height: 1, depth: 1)
            )

            currentInput = currentOutput
        }

        // Step 3: L2 Normalize (in-place on the last output)
        if let l2Pipeline = postProcessor.l2NormalizePipeline {
            let normalizeTarget: MTLBuffer
            if postProcessor.denseDescriptors.isEmpty {
                normalizeTarget = poolOutput
            } else {
                normalizeTarget = currentInput
            }

            encoder.barrier(
                afterEncoderStages: .dispatch,
                beforeEncoderStages: .dispatch,
                visibilityOptions: .device
            )

            argumentTable.setAddress(normalizeTarget.gpuAddress, index: 0)
            argumentTable.setAddress(
                constantBuffer.gpuAddress + UInt64(l2ConstantOffset()),
                index: 1
            )
            encoder.setArgumentTable(argumentTable)
            encoder.setComputePipelineState(l2Pipeline)

            let l2Threads = min(256, l2Pipeline.maxTotalThreadsPerThreadgroup)
            encoder.dispatchThreadgroups(
                threadgroupsPerGrid: MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: l2Threads, height: 1, depth: 1)
            )
        }
    }

    /// Read the final embedding vector after command buffer completion.
    public func readResult(hiddenDimension: Int) -> [Float] {
        let dimension: Int
        if let lastDense = postProcessor.denseDescriptors.last {
            dimension = lastDense.outputDimension
        } else {
            dimension = hiddenDimension
        }

        let source = resultBuffer(hiddenDimension: hiddenDimension)
        let pointer = source.contents().bindMemory(to: Float.self, capacity: dimension)
        return Array(UnsafeBufferPointer(start: pointer, count: dimension))
    }

    // MARK: - Debug

    /// Dump diagnostic info about all workspace buffers.
    /// Call after command buffer completion.
    public func debugDumpBuffers(hiddenDimension: Int) -> String {
        let dim = postProcessor.denseDescriptors.last?.outputDimension ?? hiddenDimension
        let checkDim = min(dim, 8)

        func head(_ buffer: MTLBuffer, _ label: String) -> String {
            let ptr = buffer.contents().bindMemory(to: Float.self, capacity: checkDim)
            let vals = (0..<checkDim).map { String(format: "%.4f", ptr[$0]) }.joined(separator: ", ")
            let norm = (0..<min(dim, buffer.length / MemoryLayout<Float>.stride)).reduce(Float(0)) {
                $0 + ptr[$1] * ptr[$1]
            }
            return "[\(label)] norm=\(String(format: "%.4f", sqrt(norm))) head=[\(vals)]"
        }

        var lines: [String] = []
        lines.append(head(intermediateA, "intermediateA"))
        lines.append(head(intermediateB, "intermediateB"))
        lines.append(head(outputBuffer, "outputBuffer"))

        // Also check constant buffer
        let constPtr = constantBuffer.contents().bindMemory(to: UInt32.self, capacity: 4)
        lines.append("[constants] slot0=[\(constPtr[0]), \(constPtr[1]), \(constPtr[2]), \(constPtr[3])]")

        return lines.joined(separator: "\n")
    }

    // MARK: - Constant Buffer Offsets

    private func poolingConstantOffset() -> Int {
        0
    }

    private func denseConstantOffset(layerIndex: Int) -> Int {
        (1 + layerIndex) * Self.constantSlotStride
    }

    private func l2ConstantOffset() -> Int {
        (1 + postProcessor.denseDescriptors.count) * Self.constantSlotStride
    }

    // MARK: - Result Buffer Routing

    private func resultBuffer(hiddenDimension: Int) -> MTLBuffer {
        let hasDense = !postProcessor.denseDescriptors.isEmpty
        let hasL2 = postProcessor.l2NormalizeEnabled

        if hasDense && hasL2 {
            return lastDenseOutput()
        } else if hasDense {
            return outputBuffer
        } else if hasL2 {
            return intermediateA
        } else {
            return outputBuffer
        }
    }

    private func lastDenseOutput() -> MTLBuffer {
        let layerCount = postProcessor.denseDescriptors.count
        guard layerCount > 0 else { return intermediateA }

        // Pool writes to intermediateA. Dense layers ping-pong:
        // layer 0: A→B, layer 1: B→A, ...
        // With l2NormalizeEnabled, last dense does NOT write to outputBuffer.
        if layerCount % 2 == 1 {
            return intermediateB
        } else {
            return intermediateA
        }
    }
}
