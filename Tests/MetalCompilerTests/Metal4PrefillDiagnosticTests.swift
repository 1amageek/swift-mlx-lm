import Testing
import Metal
import Foundation
@testable import MetalCompiler
import LMArchitecture
import ModelDeclarations
import LMIR

/// Regression tests for Metal 4 buffer residency requirements.
///
/// Metal 4 on Apple Silicon requires explicit `MTLResidencySet` for page-aligned
/// allocations (>= 16 KB). Without residency, GPU dispatches silently fail to
/// read/write these buffers. These tests verify the fix in `MetalBufferAllocator`.
@Suite("Metal 4 Residency", .serialized)
struct Metal4ResidencyTests {

    @Test("Buffers below 16 KB page boundary work without residency")
    func subPageBufferWorks() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void write_test(device float* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
            if (gid < 64) out[gid] = float(gid + 1);
        }
        """
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "write_test")!)

        // 16383 bytes: sub-page allocation, should work without residency
        let buffer = device.makeBuffer(length: 16383, options: .storageModeShared)!
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: 64)
        pointer[0] = -999.0

        var submission = try MetalSubmissionContext(device: device)
        try submission.withCompute { encoder, argumentTable in
            argumentTable.setAddress(buffer.gpuAddress, index: 0)
            encoder.setArgumentTable(argumentTable)
            encoder.setComputePipelineState(pipeline)
            encoder.dispatchThreadgroups(
                threadgroupsPerGrid: MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
        }

        #expect(pointer[0] == 1.0, "Sub-page buffer should work without explicit residency")
    }

    @Test("Buffers at 16 KB page boundary require residency")
    func pageAlignedBufferRequiresResidency() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void write_test(device float* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
            if (gid < 64) out[gid] = float(gid + 1);
        }
        """
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "write_test")!)

        // 16384 bytes: page-aligned allocation, requires residency
        let buffer = device.makeBuffer(length: 16384, options: .storageModeShared)!
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: 4096)
        pointer[0] = -999.0

        // Make resident
        let descriptor = MTLResidencySetDescriptor()
        let residencySet = try device.makeResidencySet(descriptor: descriptor)
        residencySet.addAllocation(buffer)
        residencySet.commit()
        residencySet.requestResidency()

        var submission = try MetalSubmissionContext(device: device)
        try submission.withCompute { encoder, argumentTable in
            argumentTable.setAddress(buffer.gpuAddress, index: 0)
            encoder.setArgumentTable(argumentTable)
            encoder.setComputePipelineState(pipeline)
            encoder.dispatchThreadgroups(
                threadgroupsPerGrid: MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
        }

        #expect(pointer[0] == 1.0, "Page-aligned buffer with residency should work")
    }

    @Test("Compiler-allocated prefill buffers are resident and writable")
    func prefillBuffersAreResident() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let hiddenSize = 128
        let intermediateSize = 512
        let vocabSize = 100
        let attentionHeads = 4
        let kvHeads = 2
        let headDim = 32
        let convKernelSize = 3

        let tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("residency_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDirectory) }

        try createSyntheticWeights(
            hiddenSize: hiddenSize, intermediateSize: intermediateSize,
            vocabSize: vocabSize, attentionHeads: attentionHeads,
            kvHeads: kvHeads, headDim: headDim,
            convKernelSize: convKernelSize, to: tempDirectory
        )

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        let store = try STAFLoader().load(at: stafURL, device: device)
        let config = ModelConfig(
            hiddenSize: hiddenSize, layerCount: 2, intermediateSize: intermediateSize,
            vocabSize: vocabSize, attentionHeads: attentionHeads, kvHeads: kvHeads, headDim: headDim,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 10000, ropeDimension: headDim,
            ropeScaling: nil, tiedEmbeddings: false,
            expertCount: nil, expertsPerToken: nil, qkNorm: true,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: convKernelSize, convLCache: convKernelSize,
            partialRotaryFactor: nil, slidingWindow: nil,
            layerTypes: ["conv", "attention"]
        )
        let graph = try LFM2(config: config).makeModelGraph()
        let resolved = ParameterResolver().resolve(graph: graph, convention: .lfm2Family)

        let compiler = MetalInferenceCompiler()
        let prefillPlan = try compiler.compilePrefill(
            graph: resolved, hiddenSize: hiddenSize, intermediateSize: intermediateSize,
            vocabSize: vocabSize, inferencePolicy: InferencePolicy(maximumSequenceLength: 32),
            stafWeightStore: store, device: device)

        // Verify the hidden buffer (>= 16 KB) is writable via Metal 4
        let hiddenLength = prefillPlan.buffers.hidden.length
        #expect(hiddenLength >= 16384, "Hidden buffer should be >= 16 KB (page-aligned)")

        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void write_test(device float* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
            if (gid < 64) out[gid] = float(gid + 1);
        }
        """
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "write_test")!)

        let hiddenPointer = prefillPlan.buffers.hidden.contents().bindMemory(to: Float.self, capacity: 64)
        hiddenPointer[0] = -999.0

        var submission = try MetalSubmissionContext(device: device)
        try submission.withCompute { encoder, argumentTable in
            argumentTable.setAddress(prefillPlan.buffers.hidden.gpuAddress, index: 0)
            encoder.setArgumentTable(argumentTable)
            encoder.setComputePipelineState(pipeline)
            encoder.dispatchThreadgroups(
                threadgroupsPerGrid: MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
        }

        #expect(
            hiddenPointer[0] == 1.0,
            "Plan-allocated hidden buffer must be GPU-resident and writable via Metal 4"
        )
    }
}

// MARK: - Synthetic Weight Generator

private func createSyntheticWeights(
    hiddenSize: Int, intermediateSize: Int, vocabSize: Int,
    attentionHeads: Int, kvHeads: Int, headDim: Int,
    convKernelSize: Int, to directory: URL
) throws {
    struct Tensor {
        let name: String; let dtype: String; let shape: [Int]; let data: Data
    }
    var tensors: [Tensor] = []

    func bf16(_ name: String, _ shape: [Int]) {
        let count = shape.reduce(1, *)
        var data = [BFloat16](repeating: .zero, count: count)
        for i in 0..<count {
            data[i] = BFloat16(Float(((i * 7 + 13) % 200)) * 0.0001 - 0.01)
        }
        tensors.append(Tensor(
            name: name, dtype: "BF16", shape: shape,
            data: Data(bytes: &data, count: count * MemoryLayout<BFloat16>.size)))
    }

    bf16("model.embed_tokens.weight", [vocabSize, hiddenSize])
    bf16("model.embedding_norm.weight", [hiddenSize])

    let p0 = "model.layers.0"
    bf16("\(p0).operator_norm.weight", [hiddenSize])
    bf16("\(p0).conv.in_proj.weight", [hiddenSize * 3, hiddenSize])
    bf16("\(p0).conv.conv.weight", [hiddenSize, convKernelSize])
    bf16("\(p0).conv.out_proj.weight", [hiddenSize, hiddenSize])
    bf16("\(p0).ffn_norm.weight", [hiddenSize])
    bf16("\(p0).feed_forward.w1.weight", [intermediateSize, hiddenSize])
    bf16("\(p0).feed_forward.w3.weight", [intermediateSize, hiddenSize])
    bf16("\(p0).feed_forward.w2.weight", [hiddenSize, intermediateSize])

    let p1 = "model.layers.1"
    bf16("\(p1).operator_norm.weight", [hiddenSize])
    bf16("\(p1).self_attn.q_proj.weight", [attentionHeads * headDim, hiddenSize])
    bf16("\(p1).self_attn.k_proj.weight", [kvHeads * headDim, hiddenSize])
    bf16("\(p1).self_attn.v_proj.weight", [kvHeads * headDim, hiddenSize])
    bf16("\(p1).self_attn.out_proj.weight", [hiddenSize, attentionHeads * headDim])
    bf16("\(p1).self_attn.q_layernorm.weight", [headDim])
    bf16("\(p1).self_attn.k_layernorm.weight", [headDim])
    bf16("\(p1).ffn_norm.weight", [hiddenSize])
    bf16("\(p1).feed_forward.w1.weight", [intermediateSize, hiddenSize])
    bf16("\(p1).feed_forward.w3.weight", [intermediateSize, hiddenSize])
    bf16("\(p1).feed_forward.w2.weight", [hiddenSize, intermediateSize])

    bf16("lm_head.weight", [vocabSize, hiddenSize])

    // Write safetensors
    var dataSection = Data()
    var offsets: [(begin: Int, end: Int)] = []
    for t in tensors {
        offsets.append((begin: dataSection.count, end: dataSection.count + t.data.count))
        dataSection.append(t.data)
    }
    var headerObject: [String: Any] = [:]
    for (i, t) in tensors.enumerated() {
        headerObject[t.name] = [
            "dtype": t.dtype, "shape": t.shape,
            "data_offsets": [offsets[i].begin, offsets[i].end]
        ] as [String: Any]
    }
    let headerJSON = try JSONSerialization.data(withJSONObject: headerObject, options: .sortedKeys)
    var fileData = Data()
    var headerSizeLE = UInt64(headerJSON.count).littleEndian
    fileData.append(Data(bytes: &headerSizeLE, count: 8))
    fileData.append(headerJSON)
    fileData.append(dataSection)

    let safetensorsURL = directory.appendingPathComponent("model.safetensors")
    try fileData.write(to: safetensorsURL)

    let stafURL = directory.appendingPathComponent("model.staf")
    try STAFConverter().convert(safetensorsURLs: [safetensorsURL], outputURL: stafURL)
}
