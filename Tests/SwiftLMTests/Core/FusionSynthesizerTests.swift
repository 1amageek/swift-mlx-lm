import Testing
@testable import MetalCompiler

@Suite("FusionSynthesizer")
struct FusionSynthesizerTests {

    // MARK: - Register Intermediate (perElement + perElement)

    @Test("Two perElement fragments fuse with register intermediate")
    func registerFusion() throws {
        let fragA = ScalarMultiplyFragment(count: 896, weightRole: "scale_a")
        let contractA = try #require(fragA.fusionContract)
        let bodyA = try #require(fragA.kernelBody(bufferPrecision: .float16, weightFormat: .float16))

        let fragB = ScalarMultiplyFragment(count: 896, weightRole: "scale_b")
        let contractB = try #require(fragB.fusionContract)
        let bodyB = try #require(fragB.kernelBody(bufferPrecision: .float16, weightFormat: .float16))

        // Verify intermediate storage is register
        let storage = contractA.intermediateStorage(to: contractB)
        #expect(storage == .register)

        let result = try FusionSynthesizer.synthesize([
            .init(contract: contractA, body: bodyA, weightFormats: ["scale_a": .float16]),
            .init(contract: contractB, body: bodyB, weightFormats: ["scale_b": .float16]),
        ])

        // Register variable declared
        #expect(result.body.contains("float _fused_0;"))

        // Producer output subscript stripped: output[idx] → _fused_0
        #expect(result.body.contains("_fused_0 ="))
        #expect(!result.body.contains("_fused_0["))

        // Consumer input subscript stripped: data[idx] → _fused_0
        // Final output still writes to output[idx]
        #expect(result.body.contains("output[idx]"))
        #expect(result.body.contains("_fused_0 *"))
    }

    @Test("Register fusion merged contract eliminates internal ports")
    func registerFusionContract() throws {
        let fragA = ScalarMultiplyFragment(count: 896, weightRole: "scale_a")
        let contractA = try #require(fragA.fusionContract)
        let bodyA = try #require(fragA.kernelBody(bufferPrecision: .float16, weightFormat: .float16))

        let fragB = ScalarMultiplyFragment(count: 896, weightRole: "scale_b")
        let contractB = try #require(fragB.fusionContract)
        let bodyB = try #require(fragB.kernelBody(bufferPrecision: .float16, weightFormat: .float16))

        let result = try FusionSynthesizer.synthesize([
            .init(contract: contractA, body: bodyA, weightFormats: ["scale_a": .float16]),
            .init(contract: contractB, body: bodyB, weightFormats: ["scale_b": .float16]),
        ])

        let mc = result.contract

        // A has 3 ports (data, weight, output), B has 3 ports (data, weight, output)
        // Internal: A.output → B.data (eliminated)
        // External: A.data (input), A.weight (input), B.weight (input), B.output (output)
        #expect(mc.ports.count == 4)

        // Inputs: A's data, A's weight, B's weight
        let inputs = mc.ports.filter { $0.direction == .input }
        #expect(inputs.count == 3)

        // Output: B's output only
        let outputs = mc.ports.filter { $0.direction == .output }
        #expect(outputs.count == 1)
        #expect(outputs[0].name == "output")

        // Parallelism preserved
        #expect(mc.parallelism == .perElement(count: 896))

        // No SIMD reduction needed
        #expect(mc.requiresSIMDReduction == false)

        // No threadgroup memory needed (register intermediate)
        #expect(mc.threadgroupMemoryBytes == 0)
    }

    @Test("Register fusion produces valid MSL via KernelScaffold")
    func registerFusionFullKernel() throws {
        let fragA = ScalarMultiplyFragment(count: 896, weightRole: "scale_a")
        let contractA = try #require(fragA.fusionContract)
        let bodyA = try #require(fragA.kernelBody(bufferPrecision: .float16, weightFormat: .float16))

        let fragB = ScalarMultiplyFragment(count: 896, weightRole: "scale_b")
        let contractB = try #require(fragB.fusionContract)
        let bodyB = try #require(fragB.kernelBody(bufferPrecision: .float16, weightFormat: .float16))

        let result = try FusionSynthesizer.synthesize([
            .init(contract: contractA, body: bodyA, weightFormats: ["scale_a": .float16]),
            .init(contract: contractB, body: bodyB, weightFormats: ["scale_b": .float16]),
        ])

        // Generate complete MSL kernel
        let msl = KernelScaffold.generate(
            name: "fused_scalar_multiply_test",
            body: result.body,
            contract: result.contract,
            bufferPrecision: .float16,
            weightFormats: result.weightFormats,
            isSequence: false
        )

        // Kernel function declaration
        #expect(msl.contains("kernel void fused_scalar_multiply_test"))

        // Buffer bindings for external ports
        #expect(msl.contains("[[buffer(0)]]"))  // A.data
        #expect(msl.contains("[[buffer(1)]]"))  // A.weight (scale_a)
        #expect(msl.contains("[[buffer(2)]]"))  // B.weight (scale_b)
        #expect(msl.contains("[[buffer(3)]]"))  // B.output

        // Dimension parameter
        #expect(msl.contains("dimension"))

        // Register intermediate variable
        #expect(msl.contains("float _fused_0;"))

        // No threadgroup barrier (register intermediate)
        #expect(!msl.contains("threadgroup_barrier"))
    }

    // MARK: - Threadgroup Memory Intermediate (perRow + perElement)

    @Test("Reduction + ScalarMultiply intermediate is threadgroup memory")
    func threadgroupIntermediateStorage() throws {
        let reduction = Reduction(dimension: 896, epsilon: 1e-6, weightRole: "scale")
        let contractA = try #require(reduction.fusionContract)

        let scalar = ScalarMultiplyFragment(count: 896, weightRole: "layer_scalar")
        let contractB = try #require(scalar.fusionContract)

        // perRow + perElement with singlePass → threadgroup memory
        // (perRow resolved, singlePass but perRow uses loop — TG memory needed)
        let storage = contractA.intermediateStorage(to: contractB)
        if case .threadgroupMemory(let dim) = storage {
            #expect(dim == 896)
        } else {
            Issue.record("Expected threadgroupMemory, got \(storage)")
        }

        // Resolved parallelism: perRow wins
        let resolved = contractA.parallelism.resolved(with: contractB.parallelism)
        #expect(resolved == .perRow(dimension: 896))
    }

    @Test("Reduction + ScalarMultiply fuse into single kernel with TG intermediate")
    func perRowPerElementFusion() throws {
        let reduction = Reduction(dimension: 896, epsilon: 1e-6, weightRole: "norm_scale", weightBias: 1.0)
        let contractA = try #require(reduction.fusionContract)
        let bodyA = try #require(reduction.kernelBody(bufferPrecision: .float32, weightFormat: .float16))

        let scalar = ScalarMultiplyFragment(count: 896, weightRole: "layer_scalar")
        let contractB = try #require(scalar.fusionContract)
        let bodyB = try #require(scalar.kernelBody(bufferPrecision: .float32, weightFormat: .float16))

        let result = try FusionSynthesizer.synthesize([
            .init(contract: contractA, body: bodyA, weightFormats: ["norm_scale": .float16]),
            .init(contract: contractB, body: bodyB, weightFormats: ["layer_scalar": .float16]),
        ])

        // Resolved parallelism: perRow wins
        #expect(result.contract.parallelism == .perRow(dimension: 896))

        // SIMD reduction from Reduction
        #expect(result.contract.requiresSIMDReduction == true)

        // Threadgroup memory: base 128 (32*4 from Reduction) + intermediate 896*4 = 3712
        #expect(result.contract.threadgroupMemoryBytes == 128 + 896 * MemoryLayout<Float>.size)

        // TG intermediate declared
        #expect(result.body.contains("threadgroup float _tg_fused_0[896];"))

        // Barrier between Reduction output and ScalarMultiply input
        #expect(result.body.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"))

        // Reduction body writes to TG intermediate instead of output
        #expect(result.body.contains("_tg_fused_0[i]"))

        // ScalarMultiply body wrapped in perRow loop
        #expect(result.body.contains("for (uint i = tid; i < dimension; i += threadgroupSize)"))

        // ScalarMultiply reads from TG intermediate
        // (data renamed to _tg_fused_0, original body: data[idx] → _tg_fused_0[i] after wrap+rename)
        #expect(result.body.contains("_tg_fused_0[i]"))

        // Scalar constants from both fragments merged
        #expect(result.contract.scalarConstants.count == 2)  // epsilon, weightBias from Reduction
        #expect(result.contract.scalarConstants[0].name == "epsilon")
        #expect(result.contract.scalarConstants[1].name == "weightBias")
    }

    @Test("Reduction + ScalarMultiply produces valid MSL kernel")
    func perRowPerElementFullKernel() throws {
        let reduction = Reduction(dimension: 256, epsilon: 1e-5, weightRole: "norm_scale")
        let contractA = try #require(reduction.fusionContract)
        let bodyA = try #require(reduction.kernelBody(bufferPrecision: .float32, weightFormat: .float16))

        let scalar = ScalarMultiplyFragment(count: 256, weightRole: "layer_scalar")
        let contractB = try #require(scalar.fusionContract)
        let bodyB = try #require(scalar.kernelBody(bufferPrecision: .float32, weightFormat: .float16))

        let result = try FusionSynthesizer.synthesize([
            .init(contract: contractA, body: bodyA, weightFormats: ["norm_scale": .float16]),
            .init(contract: contractB, body: bodyB, weightFormats: ["layer_scalar": .float16]),
        ])

        let msl = KernelScaffold.generate(
            name: "fused_norm_layerscale",
            body: result.body,
            contract: result.contract,
            bufferPrecision: .float32,
            weightFormats: result.weightFormats,
            isSequence: true
        )

        // Kernel function
        #expect(msl.contains("kernel void fused_norm_layerscale"))

        // perRow scaffold markers
        #expect(msl.contains("tid"))
        #expect(msl.contains("threadgroupSize"))
        #expect(msl.contains("threadgroup float shared[32]"))

        // External buffer ports: data(input), norm_weight(input), scalar_weight(input), output
        // + dimension + scalar constants + sequenceLength
        #expect(msl.contains("data_base"))
        #expect(msl.contains("output_base"))
        #expect(msl.contains("sequenceLength"))

        // RMS norm computation from Reduction body
        #expect(msl.contains("simd_sum(sumSquared)"))
        #expect(msl.contains("rsqrt(total / float(dimension) + epsilon)"))

        // TG barrier between phases
        #expect(msl.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"))

        // ScalarMultiply in perRow loop
        #expect(msl.contains("for (uint i = tid; i < dimension; i += threadgroupSize)"))
    }

    @Test("wrapPerElementBodyForPerRow adds cooperative loop")
    func wrapPerElementBody() {
        let body = """
        output[idx] = data[idx] * scale;
        """
        let wrapped = FusionSynthesizer.wrapPerElementBodyForPerRow(body)

        #expect(wrapped.contains("for (uint i = tid; i < dimension; i += threadgroupSize)"))
        #expect(wrapped.contains("uint idx = i;"))
        #expect(wrapped.contains("output[idx] = data[idx] * scale;"))
    }

    // MARK: - Merged Contract Properties

    @Test("Three-way fusion merges contracts correctly")
    func threeWayFusionContract() throws {
        let fragA = ScalarMultiplyFragment(count: 512, weightRole: "w_a")
        let fragB = ScalarMultiplyFragment(count: 512, weightRole: "w_b")
        let fragC = ScalarMultiplyFragment(count: 512, weightRole: "w_c")

        let contractA = try #require(fragA.fusionContract)
        let contractB = try #require(fragB.fusionContract)
        let contractC = try #require(fragC.fusionContract)

        let bodyA = try #require(fragA.kernelBody(bufferPrecision: .float32, weightFormat: .float16))
        let bodyB = try #require(fragB.kernelBody(bufferPrecision: .float32, weightFormat: .float16))
        let bodyC = try #require(fragC.kernelBody(bufferPrecision: .float32, weightFormat: .float16))

        let result = try FusionSynthesizer.synthesize([
            .init(contract: contractA, body: bodyA, weightFormats: ["w_a": .float16]),
            .init(contract: contractB, body: bodyB, weightFormats: ["w_b": .float16]),
            .init(contract: contractC, body: bodyC, weightFormats: ["w_c": .float16]),
        ])

        let mc = result.contract

        // A(data,weight_a,output) → B(data,weight_b,output) → C(data,weight_c,output)
        // Internal: A.output→B.data, B.output→C.data
        // External: A.data, A.weight, B.weight, C.weight, C.output
        #expect(mc.ports.count == 5)

        let inputs = mc.ports.filter { $0.direction == .input }
        #expect(inputs.count == 4)  // A.data + 3 weights

        let outputs = mc.ports.filter { $0.direction == .output }
        #expect(outputs.count == 1)

        // Two register intermediates
        #expect(result.body.contains("float _fused_0;"))
        #expect(result.body.contains("float _fused_1;"))

        // Combined weight formats
        #expect(result.weightFormats.count == 3)
    }

    // MARK: - Weight Format Preservation

    @Test("Weight formats from all entries are preserved in result")
    func weightFormatPreservation() throws {
        let fragA = ScalarMultiplyFragment(count: 896, weightRole: "scale_a")
        let contractA = try #require(fragA.fusionContract)
        let bodyA = try #require(fragA.kernelBody(bufferPrecision: .float32, weightFormat: .float16))

        let fragB = ScalarMultiplyFragment(count: 896, weightRole: "scale_b")
        let contractB = try #require(fragB.fusionContract)
        let bodyB = try #require(fragB.kernelBody(bufferPrecision: .float32, weightFormat: .bfloat16))

        let result = try FusionSynthesizer.synthesize([
            .init(contract: contractA, body: bodyA, weightFormats: ["scale_a": .float16]),
            .init(contract: contractB, body: bodyB, weightFormats: ["scale_b": .bfloat16]),
        ])

        #expect(result.weightFormats["scale_a"] == .float16)
        #expect(result.weightFormats["scale_b"] == .bfloat16)
    }

    // MARK: - Error Cases

    @Test("Insufficient entries throws error")
    func insufficientEntries() throws {
        let frag = ScalarMultiplyFragment(count: 896, weightRole: "scale")
        let contract = try #require(frag.fusionContract)
        let body = try #require(frag.kernelBody(bufferPrecision: .float16, weightFormat: .float16))

        #expect(throws: FusionSynthesizer.SynthesisError.self) {
            _ = try FusionSynthesizer.synthesize([
                .init(contract: contract, body: body)
            ])
        }
    }

    @Test("Incompatible parallelism throws error")
    func incompatibleParallelism() throws {
        let frag896 = ScalarMultiplyFragment(count: 896, weightRole: "scale_a")
        let frag2048 = ScalarMultiplyFragment(count: 2048, weightRole: "scale_b")

        let contractA = try #require(frag896.fusionContract)
        let contractB = try #require(frag2048.fusionContract)
        let bodyA = try #require(frag896.kernelBody(bufferPrecision: .float16, weightFormat: .float16))
        let bodyB = try #require(frag2048.kernelBody(bufferPrecision: .float16, weightFormat: .float16))

        #expect(throws: FusionSynthesizer.SynthesisError.self) {
            _ = try FusionSynthesizer.synthesize([
                .init(contract: contractA, body: bodyA),
                .init(contract: contractB, body: bodyB),
            ])
        }
    }

    // MARK: - Variable Renaming Correctness

    @Test("replaceArrayAccessWithScalar strips subscripts correctly")
    func arrayAccessReplacement() {
        // Basic subscript stripping
        let result1 = FusionSynthesizer.replaceArrayAccessWithScalar(
            in: "output[idx] = data[idx] * scale;",
            arrayName: "output",
            scalarName: "_fused_0"
        )
        #expect(result1 == "_fused_0 = data[idx] * scale;")

        // Multiple occurrences
        let result2 = FusionSynthesizer.replaceArrayAccessWithScalar(
            in: "x = output[i] + output[j];",
            arrayName: "output",
            scalarName: "_v"
        )
        #expect(result2 == "x = _v + _v;")

        // Does not replace substrings
        let result3 = FusionSynthesizer.replaceArrayAccessWithScalar(
            in: "output_base[idx] = 0;",
            arrayName: "output",
            scalarName: "_fused"
        )
        #expect(result3 == "output_base[idx] = 0;")

        // Bare name without subscript also replaced
        let result4 = FusionSynthesizer.replaceArrayAccessWithScalar(
            in: "float v = data;",
            arrayName: "data",
            scalarName: "_fused"
        )
        #expect(result4 == "float v = _fused;")
    }

    // MARK: - BF16 Weight Format Fusion

    @Test("BF16 weight format produces bf16_to_float read expression in fused body")
    func bfloat16WeightFusionBody() throws {
        let reduction = Reduction(dimension: 896, epsilon: 1e-6, weightRole: "scale")
        let contractA = try #require(reduction.fusionContract)
        let bodyA = try #require(reduction.kernelBody(bufferPrecision: .float32, weightFormat: .bfloat16))

        // BF16 read expression should use bf16_to_float
        #expect(bodyA.contains("bf16_to_float"))

        let scalar = ScalarMultiplyFragment(count: 896, weightRole: "layer_scalar")
        let contractB = try #require(scalar.fusionContract)
        let bodyB = try #require(scalar.kernelBody(bufferPrecision: .float32, weightFormat: .bfloat16))

        #expect(bodyB.contains("bf16_to_float"))

        let result = try FusionSynthesizer.synthesize([
            .init(contract: contractA, body: bodyA, weightFormats: ["scale": .bfloat16]),
            .init(contract: contractB, body: bodyB, weightFormats: ["layer_scalar": .bfloat16]),
        ])

        // BF16 weight formats preserved in result
        #expect(result.weightFormats["scale"] == .bfloat16)
        #expect(result.weightFormats["layer_scalar"] == .bfloat16)

        // Fused body retains bf16_to_float calls
        #expect(result.body.contains("bf16_to_float"))
    }

    @Test("BF16 weight scaffold generates uint16_t buffer declarations")
    func bfloat16WeightScaffold() throws {
        let reduction = Reduction(dimension: 256, epsilon: 1e-5, weightRole: "norm_weight")
        let contract = try #require(reduction.fusionContract)
        let body = try #require(reduction.kernelBody(bufferPrecision: .float32, weightFormat: .bfloat16))

        let scalar = ScalarMultiplyFragment(count: 256, weightRole: "layer_scalar")
        let contractB = try #require(scalar.fusionContract)
        let bodyB = try #require(scalar.kernelBody(bufferPrecision: .float32, weightFormat: .bfloat16))

        let result = try FusionSynthesizer.synthesize([
            .init(contract: contract, body: body, weightFormats: ["norm_weight": .bfloat16]),
            .init(contract: contractB, body: bodyB, weightFormats: ["layer_scalar": .bfloat16]),
        ])

        let msl = KernelScaffold.generate(
            name: "fused_bf16_test",
            body: result.body,
            contract: result.contract,
            bufferPrecision: .float32,
            weightFormats: result.weightFormats,
            isSequence: true
        )

        // BF16 weight ports should use uint16_t buffer type
        #expect(msl.contains("uint16_t*"))
        // Data buffers should use float (F32 prefill)
        #expect(msl.contains("float*"))
    }

    @Test("Mixed F16 and BF16 weight formats are preserved per port")
    func mixedWeightFormats() throws {
        let reduction = Reduction(dimension: 256, epsilon: 1e-5, weightRole: "norm_weight")
        let contract = try #require(reduction.fusionContract)
        let bodyA = try #require(reduction.kernelBody(bufferPrecision: .float32, weightFormat: .float16))

        let scalar = ScalarMultiplyFragment(count: 256, weightRole: "layer_scalar")
        let contractB = try #require(scalar.fusionContract)
        let bodyB = try #require(scalar.kernelBody(bufferPrecision: .float32, weightFormat: .bfloat16))

        let result = try FusionSynthesizer.synthesize([
            .init(contract: contract, body: bodyA, weightFormats: ["norm_weight": .float16]),
            .init(contract: contractB, body: bodyB, weightFormats: ["layer_scalar": .bfloat16]),
        ])

        // Both weight formats preserved independently
        #expect(result.weightFormats["norm_weight"] == .float16)
        #expect(result.weightFormats["layer_scalar"] == .bfloat16)

        let msl = KernelScaffold.generate(
            name: "fused_mixed_weights",
            body: result.body,
            contract: result.contract,
            bufferPrecision: .float32,
            weightFormats: result.weightFormats,
            isSequence: true
        )

        // Both half and uint16_t declarations should appear
        #expect(msl.contains("half*"))
        #expect(msl.contains("uint16_t*"))
    }

    // MARK: - ElementwiseFragment Fusion

    @Test("ElementwiseFragment provides fusionContract with gate/up/output ports")
    func elementwiseFusionContract() throws {
        let frag = ElementwiseFragment(count: 2048, kind: .swiglu)
        let contract = try #require(frag.fusionContract)

        #expect(contract.ports.count == 3)
        #expect(contract.ports[0].name == "gate")
        #expect(contract.ports[0].direction == .input)
        #expect(contract.ports[1].name == "up")
        #expect(contract.ports[1].direction == .input)
        #expect(contract.ports[2].name == "output")
        #expect(contract.ports[2].direction == .output)
        #expect(contract.parallelism == .perElement(count: 2048))
        #expect(contract.requiresSIMDReduction == false)
        #expect(contract.threadgroupMemoryBytes == 0)
    }

    @Test("SwiGLU kernelBody contains SiLU activation")
    func swigluKernelBody() throws {
        let frag = ElementwiseFragment(count: 1024, kind: .swiglu)
        let body = try #require(frag.kernelBody(bufferPrecision: .float16, weightFormat: .float16))

        #expect(body.contains("gate[idx]"))
        #expect(body.contains("up[idx]"))
        #expect(body.contains("output[idx]"))
        #expect(body.contains("exp(-g)"))
    }

    @Test("GEGLU kernelBody contains GELU tanh activation")
    func geluGatedKernelBody() throws {
        let frag = ElementwiseFragment(count: 1024, kind: .geluGated)
        let body = try #require(frag.kernelBody(bufferPrecision: .float16, weightFormat: .float16))

        #expect(body.contains("precise::tanh"))
        #expect(body.contains("0.7978845608f"))
        #expect(body.contains("gate[idx]"))
        #expect(body.contains("up[idx]"))
    }

    @Test("ElementwiseFragment produces valid MSL via KernelScaffold")
    func elementwiseScaffoldMSL() throws {
        let frag = ElementwiseFragment(count: 512, kind: .swiglu)
        let contract = try #require(frag.fusionContract)
        let body = try #require(frag.kernelBody(bufferPrecision: .float32, weightFormat: .float16))

        let msl = KernelScaffold.generate(
            name: "swiglu_scaffold_test",
            body: body,
            contract: contract,
            bufferPrecision: .float32,
            weightFormats: [:],
            isSequence: true
        )

        #expect(msl.contains("kernel void swiglu_scaffold_test"))
        #expect(msl.contains("gate"))
        #expect(msl.contains("up"))
        #expect(msl.contains("output"))
        #expect(msl.contains("sequenceLength"))
        #expect(msl.contains("dimension"))
    }

    // MARK: - QKNormFragment Fusion

    @Test("QKNormFragment provides fusionContract with perHead parallelism")
    func qkNormFusionContract() throws {
        let frag = QKNormFragment(
            headCount: 8, headDimension: 64,
            epsilon: 1e-6, weightRole: "q_layernorm"
        )
        let contract = try #require(frag.fusionContract)

        #expect(contract.ports.count == 3)
        #expect(contract.ports[0].name == "data")
        #expect(contract.ports[0].direction == .input)
        #expect(contract.ports[0].accessPattern == .multiPass)
        #expect(contract.ports[1].name == "weight")
        #expect(contract.ports[2].name == "output")
        #expect(contract.ports[2].direction == .output)
        #expect(contract.parallelism == .perHead(headCount: 8, headDimension: 64))
        #expect(contract.requiresSIMDReduction == true)
        #expect(contract.threadgroupMemoryBytes == 32 * MemoryLayout<Float>.size)
        #expect(contract.scalarConstants.count == 2)
        #expect(contract.scalarConstants[0].name == "epsilon")
        #expect(contract.scalarConstants[1].name == "weightBias")
    }

    @Test("QKNormFragment kernelBody contains RMS norm pattern")
    func qkNormKernelBody() throws {
        let frag = QKNormFragment(
            headCount: 8, headDimension: 64,
            epsilon: 1e-6, weightRole: "q_layernorm"
        )
        let body = try #require(frag.kernelBody(bufferPrecision: .float16, weightFormat: .float16))

        // SIMD reduction pattern
        #expect(body.contains("simd_sum(sumSq)"))
        #expect(body.contains("shared[simdIndex]"))
        #expect(body.contains("threadgroup_barrier"))
        #expect(body.contains("rsqrt(total / float(dimension) + epsilon)"))

        // Uses head-local pointers (no offset computation in body)
        #expect(body.contains("data[i]"))
        #expect(body.contains("output[i]"))

        // Weight read
        #expect(body.contains("weight[i]"))
    }

    @Test("QKNormFragment scalarConstantValues match contract")
    func qkNormScalarConstants() {
        let frag = QKNormFragment(
            headCount: 8, headDimension: 64,
            epsilon: 1e-5, weightRole: "k_layernorm",
            weightBias: 1.0
        )
        let values = frag.scalarConstantValues

        if case .float(let eps) = values["epsilon"] {
            #expect(abs(eps - 1e-5) < 1e-10)
        } else {
            Issue.record("epsilon should be .float")
        }
        if case .float(let bias) = values["weightBias"] {
            #expect(bias == 1.0)
        } else {
            Issue.record("weightBias should be .float")
        }
    }

    @Test("QKNormFragment BF16 kernelBody uses bf16_to_float")
    func qkNormBF16KernelBody() throws {
        let frag = QKNormFragment(
            headCount: 8, headDimension: 64,
            epsilon: 1e-6, weightRole: "q_layernorm"
        )
        let body = try #require(frag.kernelBody(bufferPrecision: .float16, weightFormat: .bfloat16))

        #expect(body.contains("bf16_to_float"))
    }

    // MARK: - SigmoidGateFragment Fusion

    @Test("SigmoidGateFragment provides fusionContract with input/gate/output ports")
    func sigmoidGateFusionContract() throws {
        let frag = SigmoidGateFragment(dimension: 1024)
        let contract = try #require(frag.fusionContract)

        #expect(contract.ports.count == 3)
        #expect(contract.ports[0].name == "input")
        #expect(contract.ports[0].direction == .input)
        #expect(contract.ports[1].name == "gate")
        #expect(contract.ports[1].direction == .input)
        #expect(contract.ports[2].name == "output")
        #expect(contract.ports[2].direction == .output)

        if case .perElement(let count) = contract.parallelism {
            #expect(count == 1024)
        } else {
            Issue.record("Expected perElement parallelism")
        }
    }

    @Test("SigmoidGateFragment kernelBody applies sigmoid to gate signal")
    func sigmoidGateFragmentKernelBody() throws {
        let frag = SigmoidGateFragment(dimension: 512)
        let body = try #require(frag.kernelBody(bufferPrecision: .float16, weightFormat: .float16))

        #expect(body.contains("gate[idx]"))
        #expect(body.contains("input[idx]"))
        #expect(body.contains("output[idx]"))
        #expect(body.contains("exp(-g)"))
    }

    // MARK: - Per-Head Scaffold

    @Test("Per-head scaffold generates decode mode MSL with head-offset pointers")
    func perHeadScaffoldDecode() throws {
        let frag = QKNormFragment(
            headCount: 4, headDimension: 32,
            epsilon: 1e-6, weightRole: "test_norm"
        )
        let contract = try #require(frag.fusionContract)
        let body = try #require(frag.kernelBody(bufferPrecision: .float16, weightFormat: .float16))

        let msl = KernelScaffold.generate(
            name: "qknorm_decode_test",
            body: body,
            contract: contract,
            bufferPrecision: .float16,
            weightFormats: ["test_norm": .float16],
            isSequence: false
        )

        #expect(msl.contains("kernel void qknorm_decode_test"))
        // Head-offset pointer computation
        #expect(msl.contains("data_base"))
        #expect(msl.contains("output_base"))
        #expect(msl.contains("headIndex * dimension"))
        // Grid attribute
        #expect(msl.contains("threadgroup_position_in_grid"))
        // Bounds check with embedded headCount
        #expect(msl.contains("headIndex >= 4"))
        // Shared memory for SIMD reduction
        #expect(msl.contains("threadgroup float shared[32]"))
        // Weight port has no _base suffix
        #expect(msl.contains("device const half* weight"))
        #expect(!msl.contains("weight_base"))
    }

    @Test("Per-head scaffold generates sequence mode MSL")
    func perHeadScaffoldSequence() throws {
        let frag = QKNormFragment(
            headCount: 8, headDimension: 64,
            epsilon: 1e-6, weightRole: "test_norm"
        )
        let contract = try #require(frag.fusionContract)
        let body = try #require(frag.kernelBody(bufferPrecision: .float32, weightFormat: .float16))

        let msl = KernelScaffold.generate(
            name: "qknorm_seq_test",
            body: body,
            contract: contract,
            bufferPrecision: .float32,
            weightFormats: ["test_norm": .float16],
            isSequence: true
        )

        #expect(msl.contains("kernel void qknorm_seq_test"))
        // Sequence mode markers
        #expect(msl.contains("sequenceLength"))
        #expect(msl.contains("seqPos"))
        #expect(msl.contains("uint2 gid"))
        #expect(msl.contains("headIndex = gid.x"))
        #expect(msl.contains("seqPos = gid.y"))
        // Head+seq offset computation
        #expect(msl.contains("seqPos * 8 * dimension + headIndex * dimension"))
        // Bounds check
        #expect(msl.contains("headIndex >= 8"))
    }

    // MARK: - Sequence Mode Fusion

    @Test("Register fusion works in sequence mode (F32 prefill)")
    func registerFusionSequenceMode() throws {
        let fragA = ScalarMultiplyFragment(count: 896, weightRole: "scale_a")
        let contractA = try #require(fragA.fusionContract)
        let bodyA = try #require(fragA.kernelBody(bufferPrecision: .float32, weightFormat: .float16))

        let fragB = ScalarMultiplyFragment(count: 896, weightRole: "scale_b")
        let contractB = try #require(fragB.fusionContract)
        let bodyB = try #require(fragB.kernelBody(bufferPrecision: .float32, weightFormat: .float16))

        let result = try FusionSynthesizer.synthesize([
            .init(contract: contractA, body: bodyA, weightFormats: ["scale_a": .float16]),
            .init(contract: contractB, body: bodyB, weightFormats: ["scale_b": .float16]),
        ])

        let msl = KernelScaffold.generate(
            name: "fused_seq_test",
            body: result.body,
            contract: result.contract,
            bufferPrecision: .float32,
            weightFormats: result.weightFormats,
            isSequence: true
        )

        // Sequence mode markers
        #expect(msl.contains("sequenceLength"))
        #expect(msl.contains("seqPos"))
        #expect(msl.contains("if (i >= dimension || seqPos >= sequenceLength) return;"))

        // Kernel function
        #expect(msl.contains("kernel void fused_seq_test"))
    }

    // MARK: - 4-Way Sandwich Norm Fusion (3 LoopGroups)

    @Test("Reduction+ResidualAdd+Copy+Reduction produces correct 3-LoopGroup body")
    func sandwichNormFourWayFusion() throws {
        // Exact pattern from Gemma4 sandwich norms:
        // Group 0: Reduction (post_attn_norm) → TG → barrier
        // Group 1: ResidualAdd + CopyFragment → TG → barrier
        // Group 2: Reduction (pre_ff_norm)
        let dim = 1536

        let norm1 = Reduction(dimension: dim, epsilon: 1e-6, weightRole: "norm1_weight")
        let residualAdd = ResidualAddFragment(dimension: dim)
        let copy = CopyFragment(dimension: dim)
        let norm2 = Reduction(dimension: dim, epsilon: 1e-6, weightRole: "norm2_weight")

        let contractNorm1 = try #require(norm1.fusionContract)
        let contractResAdd = try #require(residualAdd.fusionContract)
        let contractCopy = try #require(copy.fusionContract)
        let contractNorm2 = try #require(norm2.fusionContract)

        let bodyNorm1 = try #require(norm1.kernelBody(bufferPrecision: .float32, weightFormat: .bfloat16))
        let bodyResAdd = try #require(residualAdd.kernelBody(bufferPrecision: .float32, weightFormat: .bfloat16))
        let bodyCopy = try #require(copy.kernelBody(bufferPrecision: .float32, weightFormat: .bfloat16))
        let bodyNorm2 = try #require(norm2.kernelBody(bufferPrecision: .float32, weightFormat: .bfloat16))

        // Verify intermediate storage types
        let storage01 = contractNorm1.intermediateStorage(to: contractResAdd)
        #expect(storage01 == .threadgroupMemory(dimension: dim), "Reduction→ResidualAdd should be threadgroup")

        let storage12 = contractResAdd.intermediateStorage(to: contractCopy)
        #expect(storage12 == .register, "ResidualAdd→Copy should be register")

        let storage23 = contractCopy.intermediateStorage(to: contractNorm2)
        #expect(storage23 == .threadgroupMemory(dimension: dim), "Copy→Reduction should be threadgroup")

        let result = try FusionSynthesizer.synthesize([
            .init(contract: contractNorm1, body: bodyNorm1, weightFormats: ["norm1_weight": .bfloat16]),
            .init(contract: contractResAdd, body: bodyResAdd),
            .init(contract: contractCopy, body: bodyCopy),
            .init(contract: contractNorm2, body: bodyNorm2, weightFormats: ["norm2_weight": .bfloat16]),
        ])

        let body = result.body

        // --- Structural assertions ---

        // Two threadgroup intermediates declared
        #expect(body.contains("threadgroup float _tg_fused_0[\(dim)];"), "Group 0→1 TG intermediate")
        #expect(body.contains("threadgroup float _tg_fused_1[\(dim)];"), "Group 1→2 TG intermediate")

        // Two threadgroup barriers (between 3 groups)
        let barrierCount = body.components(separatedBy: "threadgroup_barrier(mem_flags::mem_threadgroup)").count - 1
        // Note: Reduction bodies contain internal barriers too, so at least 2 inter-group barriers
        #expect(barrierCount >= 2, "At least 2 inter-group barriers expected, got \(barrierCount)")

        // Register intermediate in Group 1
        #expect(body.contains("float _fused_1;"), "Register intermediate between ResidualAdd and Copy")

        // --- Group 0: Reduction writes to _tg_fused_0 ---
        #expect(body.contains("_tg_fused_0[i]"), "Group 0 Reduction writes to _tg_fused_0")
        // Group 0 should NOT reference _tg_fused_1
        let group0EndMarker = "_tg_fused_0[i]"
        let firstTG0 = body.range(of: group0EndMarker)
        #expect(firstTG0 != nil, "Group 0 must write to _tg_fused_0")

        // --- Group 1: ResidualAdd reads _tg_fused_0, Copy writes to _tg_fused_1 ---
        // ResidualAdd should read from _tg_fused_0 (not data, not _tg_fused_1)
        #expect(body.contains("float(_tg_fused_0[idx])"), "ResidualAdd reads from _tg_fused_0")
        // CopyFragment should write to _tg_fused_1 (renamed from output)
        #expect(body.contains("_tg_fused_1[idx]"), "Copy writes to _tg_fused_1")

        // --- Group 2: Reduction reads from _tg_fused_1 ---
        // The second Reduction must read _tg_fused_1, NOT _tg_fused_0
        #expect(body.contains("float(_tg_fused_1[i])"), "Group 2 Reduction reads from _tg_fused_1")

        // Group 2 must NOT read from _tg_fused_0
        // Find the second Reduction's body (after the second inter-group barrier)
        // Split by barriers and check the last group
        let interGroupBarrierPattern = "threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        let sections = body.components(separatedBy: interGroupBarrierPattern)
        // The body has internal Reduction barriers too, but the last section should be Group 2
        if let lastSection = sections.last {
            // Last section should be Group 2 (second Reduction)
            // It should read from _tg_fused_1, NOT _tg_fused_0
            if lastSection.contains("float v = float(") {
                #expect(lastSection.contains("_tg_fused_1[i]"), "Group 2 reads _tg_fused_1")
                #expect(!lastSection.contains("_tg_fused_0"), "Group 2 must NOT reference _tg_fused_0")
            }
        }

        // --- Merged contract ---
        let mc = result.contract
        #expect(mc.parallelism == .perRow(dimension: dim))
        #expect(mc.requiresSIMDReduction == true)

        // External ports: data(in), residual(merged out), norm1_weight(in), norm2_weight(in), output(out)
        let inputs = mc.ports.filter { $0.direction == .input }
        let outputs = mc.ports.filter { $0.direction == .output }
        #expect(inputs.count >= 2, "At least data + 2 weights = 3 inputs")
        #expect(outputs.count >= 1, "At least output port")

        // Print body for manual inspection in test output
        print("=== 4-Way Fused Body ===")
        print(body)
        print("========================")
    }
}
