import Testing
@testable import MetalCompiler

@Suite("KernelScaffold")
struct KernelScaffoldTests {

    // MARK: - Scaffold via generate() — Sequence Mode

    @Test("Scaffold sequence kernel has correct buffer binding indices")
    func scaffoldSequenceBindings() throws {
        let reduction = Reduction(dimension: 896, epsilon: 1e-6, weightRole: "scale", weightBias: 0)
        let source = reduction.kernelSource(
            name: "rms_norm_test_seq",
            bufferPrecision: .float32,
            weightFormat: .float16
        )

        // Buffer ports: data_base(0), weight(1), output_base(2)
        #expect(source.contains("data_base [[buffer(0)]]"))
        #expect(source.contains("weight [[buffer(1)]]"))
        #expect(source.contains("output_base [[buffer(2)]]"))

        // Scalar layout: dimension(3), epsilon(4), weightBias(5), sequenceLength(6)
        #expect(source.contains("dimension        [[buffer(3)]]"))
        #expect(source.contains("epsilon       [[buffer(4)]]"))
        #expect(source.contains("weightBias       [[buffer(5)]]"))
        #expect(source.contains("sequenceLength   [[buffer(6)]]"))

        // Kernel function name
        #expect(source.contains("kernel void rms_norm_test_seq"))

        // Sequence mode: seqPos bounds check and row pointer computation
        #expect(source.contains("if (seqPos >= sequenceLength) return;"))
        #expect(source.contains("data_base + seqPos * dimension"))
        #expect(source.contains("output_base + seqPos * dimension"))
    }

    @Test("Scaffold sequence kernel has correct computation body")
    func scaffoldSequenceBody() throws {
        let reduction = Reduction(dimension: 896, epsilon: 1e-6, weightRole: "scale", weightBias: 0)
        let source = reduction.kernelSource(
            name: "rms_norm_test_seq",
            bufferPrecision: .float32,
            weightFormat: .float16
        )

        // SIMD reduction pattern
        #expect(source.contains("simd_sum(sumSquared)"))
        #expect(source.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"))
        #expect(source.contains("threadgroup float shared[32]"))

        // RMS normalization
        #expect(source.contains("rsqrt(total / float(dimension) + epsilon)"))

        // Weight read
        #expect(source.contains("float(weight[i])"))
    }

    // MARK: - Scaffold via generate() — Decode Mode

    @Test("Scaffold decode kernel has correct buffer binding indices")
    func scaffoldDecodeBindings() throws {
        let reduction = Reduction(dimension: 896, epsilon: 1e-6, weightRole: "scale", weightBias: 0)
        let source = reduction.kernelSource(
            name: "rms_norm_test",
            bufferPrecision: .float16,
            weightFormat: .float16
        )

        // Decode: no _base suffix, no seqPos
        #expect(source.contains("device const half* data [[buffer(0)]]"))
        #expect(source.contains("device half* output [[buffer(2)]]"))

        // Scalar layout: dimension(3), epsilon(4), weightBias(5)
        #expect(source.contains("dimension        [[buffer(3)]]"))
        #expect(source.contains("epsilon       [[buffer(4)]]"))
        #expect(source.contains("weightBias       [[buffer(5)]]"))

        // No sequence-related parameters
        #expect(!source.contains("sequenceLength"))
        #expect(!source.contains("seqPos"))
    }

    // MARK: - BFloat16 Weight Format

    @Test("Scaffold handles bfloat16 weight format")
    func scaffoldBFloat16Weight() throws {
        let reduction = Reduction(dimension: 2048, epsilon: 1e-5, weightRole: "scale", weightBias: 1.0)
        let source = reduction.kernelSource(
            name: "rms_norm_bf16_test",
            bufferPrecision: .float32,
            weightFormat: .bfloat16
        )

        // BFloat16 weight: uint16_t type, bf16_to_float read
        #expect(source.contains("uint16_t* weight"))
        #expect(source.contains("bf16_to_float(weight[i])"))

        // weightBias is a runtime constant (not embedded literal)
        #expect(source.contains("weightBias"))
        #expect(source.contains("constant float& weightBias"))
    }

    // MARK: - Binding Compatibility with Existing generateReduction

    @Test("Scaffold binding indices match existing generateReduction layout")
    func bindingCompatibility() throws {
        let reduction = Reduction(dimension: 896, epsilon: 1e-6, weightRole: "scale", weightBias: 0)

        let scaffold = reduction.kernelSource(
            name: "scaffold_test",
            bufferPrecision: .float32,
            weightFormat: .float16
        )
        let existing = MetalSourceGenerator.generateReduction(
            name: "existing_test",
            dimension: 0,
            epsilon: 0,
            bufferPrecision: .float32,
            weightFormat: .float16
        )

        // Both use identical buffer index assignments
        // index 0: input buffer
        #expect(scaffold.contains("[[buffer(0)]]"))
        #expect(existing.contains("[[buffer(0)]]"))

        // index 1: weight buffer
        #expect(scaffold.contains("weight [[buffer(1)]]"))
        #expect(existing.contains("weight      [[buffer(1)]]"))

        // index 3: dimension
        #expect(scaffold.contains("dimension        [[buffer(3)]]"))
        #expect(existing.contains("dimension        [[buffer(3)]]"))

        // index 4: epsilon
        #expect(scaffold.contains("epsilon       [[buffer(4)]]"))
        #expect(existing.contains("epsilon         [[buffer(4)]]"))

        // index 5: weightBias
        #expect(scaffold.contains("weightBias       [[buffer(5)]]"))
        #expect(existing.contains("weightBias      [[buffer(5)]]"))

        // index 6: sequenceLength
        #expect(scaffold.contains("sequenceLength   [[buffer(6)]]"))
        #expect(existing.contains("sequenceLength   [[buffer(6)]]"))

        // Both have same computation patterns
        #expect(scaffold.contains("simd_sum(sumSquared)"))
        #expect(existing.contains("simd_sum(sumSquared)"))
        #expect(scaffold.contains("rsqrt(total / float(dimension)"))
        #expect(existing.contains("rsqrt(total / float(dimension)"))
    }

    // MARK: - FusionContract

    @Test("Reduction fusionContract has correct ports and scalar constants")
    func reductionContract() throws {
        let reduction = Reduction(dimension: 896, epsilon: 1e-6, weightRole: "scale")
        let contract = try #require(reduction.fusionContract)

        #expect(contract.ports.count == 3)
        #expect(contract.scalarConstants.count == 2)
        #expect(contract.parallelism == .perRow(dimension: 896))
        #expect(contract.threadgroupMemoryBytes == 128)  // 32 * 4
        #expect(contract.requiresSIMDReduction == true)

        // Input: data (buffer, multiPass)
        let dataPort = try #require(contract.ports.first { $0.name == "data" })
        #expect(dataPort.direction == .input)
        #expect(dataPort.accessPattern == .multiPass)

        // Input: weight
        let weightPort = try #require(contract.ports.first { $0.name == "weight" })
        #expect(weightPort.direction == .input)

        // Output: output
        let outputPort = try #require(contract.ports.first { $0.name == "output" })
        #expect(outputPort.direction == .output)

        // Scalar constants
        #expect(contract.scalarConstants[0].name == "epsilon")
        #expect(contract.scalarConstants[0].metalType == "float")
        #expect(contract.scalarConstants[1].name == "weightBias")
        #expect(contract.scalarConstants[1].metalType == "float")
    }

    // MARK: - Parallelism Compatibility

    @Test("Parallelism compatibility rules")
    func parallelismCompatibility() {
        let perRow896 = KernelParallelism.perRow(dimension: 896)
        let perElement896 = KernelParallelism.perElement(count: 896)
        let perRow2048 = KernelParallelism.perRow(dimension: 2048)
        let perElement2048 = KernelParallelism.perElement(count: 2048)

        // Same type, same dimension
        #expect(perRow896.isCompatible(with: perRow896))
        #expect(perElement896.isCompatible(with: perElement896))

        // Cross-type, same dimension
        #expect(perRow896.isCompatible(with: perElement896))
        #expect(perElement896.isCompatible(with: perRow896))

        // Different dimensions
        #expect(!perRow896.isCompatible(with: perRow2048))
        #expect(!perElement896.isCompatible(with: perElement2048))
        #expect(!perRow896.isCompatible(with: perElement2048))
    }

    @Test("Parallelism resolution prefers perRow")
    func parallelismResolution() {
        let perRow = KernelParallelism.perRow(dimension: 896)
        let perElement = KernelParallelism.perElement(count: 896)

        // perRow wins over perElement
        #expect(perRow.resolved(with: perElement) == perRow)
        #expect(perElement.resolved(with: perRow) == perRow)

        // Same type preserves
        #expect(perElement.resolved(with: perElement) == perElement)
    }

    // MARK: - Intermediate Storage

    @Test("Intermediate storage: multiPass consumer requires threadgroup memory")
    func intermediateStorageMultiPass() {
        let producer = FusionContract(
            ports: [
                FusionPort(name: "result", direction: .output, role: .buffer)
            ],
            parallelism: .perElement(count: 896)
        )
        let consumer = FusionContract(
            ports: [
                FusionPort(name: "data", direction: .input, role: .buffer, accessPattern: .multiPass),
                FusionPort(name: "output", direction: .output, role: .buffer)
            ],
            parallelism: .perRow(dimension: 896),
            threadgroupMemoryBytes: 128,
            requiresSIMDReduction: true
        )

        let storage = producer.intermediateStorage(to: consumer)
        #expect(storage == .threadgroupMemory(dimension: 896))
    }

    @Test("Intermediate storage: singlePass perElement consumers use register")
    func intermediateStorageRegister() {
        let producer = FusionContract(
            ports: [
                FusionPort(name: "result", direction: .output, role: .buffer)
            ],
            parallelism: .perElement(count: 896)
        )
        let consumer = FusionContract(
            ports: [
                FusionPort(name: "data", direction: .input, role: .buffer, accessPattern: .singlePass),
                FusionPort(name: "output", direction: .output, role: .buffer)
            ],
            parallelism: .perElement(count: 896)
        )

        let storage = producer.intermediateStorage(to: consumer)
        #expect(storage == .register)
    }
}
