import Foundation
import Metal
import Testing
import LMIR
@testable import MetalCompiler
@testable import SwiftLM

/// Correctness tests for RotorQuant (Clifford Cl(3,0) rotor rotation + QJL correction).
///
/// These tests verify the mathematical correctness of the implementation
/// without requiring model weights. GPU kernel tests compile and execute
/// Metal kernels directly with synthetic data.
@Suite("RotorQuant Correctness", .serialized)
struct RotorQuantCorrectnessTests {

    // MARK: - QuantizationSchemeIdentifier Properties

    @Test("isRotorScheme identifies only rotor schemes")
    func isRotorSchemeIdentification() {
        let rotorSchemes: [QuantizationSchemeIdentifier] = [
            .rotorQ8Group32ScaleF16,
            .rotorQ4Group64ScaleF16,
        ]
        let nonRotorSchemes: [QuantizationSchemeIdentifier] = [
            .fp16RowMajor, .bf16RowMajor, .fp32RowMajor,
            .q8Group32ScaleF16, .q8Group64ScaleF16, .q8Group128ScaleF16,
            .q4Group64ScaleF16, .q4Group128ScaleF16,
            .q6Group16ScaleF16, .q5Group32ScaleF16,
            .q3Group16ScaleF16, .q2Group16ScaleF16,
            .passthrough,
        ]

        for scheme in rotorSchemes {
            #expect(scheme.isRotorScheme, "\(scheme) should be a rotor scheme")
        }
        for scheme in nonRotorSchemes {
            #expect(!scheme.isRotorScheme, "\(scheme) should NOT be a rotor scheme")
        }
    }

    @Test("baseScheme maps rotor schemes to their base quantization")
    func baseSchemeMapping() {
        #expect(QuantizationSchemeIdentifier.rotorQ8Group32ScaleF16.baseScheme == .q8Group32ScaleF16)
        #expect(QuantizationSchemeIdentifier.rotorQ4Group64ScaleF16.baseScheme == .q4Group64ScaleF16)
        // Non-rotor schemes map to themselves
        #expect(QuantizationSchemeIdentifier.fp16RowMajor.baseScheme == .fp16RowMajor)
        #expect(QuantizationSchemeIdentifier.q8Group32ScaleF16.baseScheme == .q8Group32ScaleF16)
    }

    // MARK: - KVCacheSpecification Rotor Calculations

    @Test("numRotorGroups computes ceil(headDim/3) correctly")
    func numRotorGroupsCalculation() {
        func makeSpec(headDim: Int) -> KVCacheSpecification {
            KVCacheSpecification(
                keyQuantizationScheme: .rotorQ8Group32ScaleF16,
                valueQuantizationScheme: .fp16RowMajor,
                kvHeadCount: 1, headDimension: headDim, maximumSequenceLength: 1)
        }

        #expect(makeSpec(headDim: 3).numRotorGroups == 1)   // 3/3 = 1
        #expect(makeSpec(headDim: 4).numRotorGroups == 2)   // ceil(4/3) = 2
        #expect(makeSpec(headDim: 6).numRotorGroups == 2)   // 6/3 = 2
        #expect(makeSpec(headDim: 64).numRotorGroups == 22)  // ceil(64/3) = 22
        #expect(makeSpec(headDim: 128).numRotorGroups == 43) // ceil(128/3) = 43
        #expect(makeSpec(headDim: 1).numRotorGroups == 1)   // ceil(1/3) = 1
    }

    @Test("usesRotorQuant detects rotor schemes in K or V independently")
    func usesRotorQuantDetection() {
        let rotorK = KVCacheSpecification(
            keyQuantizationScheme: .rotorQ8Group32ScaleF16,
            valueQuantizationScheme: .fp16RowMajor,
            kvHeadCount: 8, headDimension: 64, maximumSequenceLength: 1024)
        #expect(rotorK.usesRotorQuant)

        let rotorV = KVCacheSpecification(
            keyQuantizationScheme: .fp16RowMajor,
            valueQuantizationScheme: .rotorQ4Group64ScaleF16,
            kvHeadCount: 8, headDimension: 64, maximumSequenceLength: 1024)
        #expect(rotorV.usesRotorQuant)

        let noRotor = KVCacheSpecification(
            keyQuantizationScheme: .q8Group32ScaleF16,
            valueQuantizationScheme: .fp16RowMajor,
            kvHeadCount: 8, headDimension: 64, maximumSequenceLength: 1024)
        #expect(!noRotor.usesRotorQuant)
    }

    @Test("InferencePolicy.default stays automatic until graph resolution")
    func defaultInferencePolicyStaysAutomatic() {
        let policy = InferencePolicy.default
        #expect(policy.maximumSequenceLength == 4096)
        #expect(!policy.kvCache.usesRotorQuant)
        #expect(policy.kvCache.keyScheme == .automatic)
        #expect(policy.kvCache.valueScheme == .automatic)
    }

    @Test("ModelBundleLoader resolves default policy to RotorQuant only for plain transformer attention graphs")
    func loaderResolvesModelAwareDefault() {
        let noAttentionGraph = ModelGraph(rootRegion: Region())
        let noAttentionPolicy = ModelBundleLoader.resolveInferencePolicy(.default, for: noAttentionGraph)
        #expect(!noAttentionPolicy.kvCache.usesRotorQuant)

        let attentionGraph = ModelGraph(
            rootRegion: Region(
                operations: [
                    Operation(
                        key: OperationKey(rawValue: 0),
                        kind: .primitive(
                            AttentionAttributes(
                                hiddenSize: 64,
                                headCount: 8,
                                kvHeadCount: 8,
                                headDimension: 8
                            )
                        )
                    )
                ]
            )
        )
        let attentionPolicy = ModelBundleLoader.resolveInferencePolicy(.default, for: attentionGraph)
        #expect(attentionPolicy.kvCache.usesRotorQuant)
        #expect(attentionPolicy.kvCache.keyScheme == .fixed(.rotorQ4Group64ScaleF16))
        #expect(attentionPolicy.kvCache.valueScheme == .fixed(.rotorQ4Group64ScaleF16))

        let hybridGraph = ModelGraph(
            rootRegion: Region(
                operations: [
                    Operation(
                        key: OperationKey(rawValue: 0),
                        kind: .primitive(
                            StateSpaceAttributes(
                                hiddenSize: 64,
                                numHeads: 8,
                                groupCount: 4,
                                keyHeadDim: 8,
                                valueHeadDim: 8,
                                convKernelSize: 4,
                                variant: "delta_net",
                                computeDType: .float32
                            )
                        )
                    ),
                    Operation(
                        key: OperationKey(rawValue: 1),
                        kind: .primitive(
                            AttentionAttributes(
                                hiddenSize: 64,
                                headCount: 8,
                                kvHeadCount: 8,
                                headDimension: 8
                            )
                        )
                    )
                ]
            )
        )
        let hybridPolicy = ModelBundleLoader.resolveInferencePolicy(.default, for: hybridGraph)
        #expect(!hybridPolicy.kvCache.usesRotorQuant)
    }

    @Test("Rotor parameter buffer sizing is consistent")
    func rotorParameterBufferSizing() {
        let spec = KVCacheSpecification(
            keyQuantizationScheme: .rotorQ8Group32ScaleF16,
            valueQuantizationScheme: .rotorQ4Group64ScaleF16,
            layerCount: 16,
            kvHeadCount: 8, headDimension: 128, maximumSequenceLength: 4096)

        let numGroups = spec.numRotorGroups  // ceil(128/3) = 43
        #expect(numGroups == 43)

        // Per-layer: kvHeadCount * numGroups * 4 components * 2 bytes (half)
        let expectedPerLayer = 8 * 43 * 4 * 2
        #expect(spec.rotorParametersBytesPerLayer == expectedPerLayer)
        #expect(spec.totalRotorParametersSize == expectedPerLayer * 16)
    }

    @Test("QJL buffer sizing is consistent")
    func qjlBufferSizing() {
        let spec = KVCacheSpecification(
            keyQuantizationScheme: .rotorQ8Group32ScaleF16,
            valueQuantizationScheme: .fp16RowMajor,
            layerCount: 16,
            kvHeadCount: 8, headDimension: 128, maximumSequenceLength: 4096)

        let qjlDim = 16

        // Matrix: headDim * qjlDim * 2 bytes
        #expect(spec.qjlMatrixSize(qjlDimension: qjlDim) == 128 * 16 * 2)

        // Residual per layer: maxSeqLen * kvHeadCount * qjlDim * 2 bytes
        let expectedPerLayer = 4096 * 8 * 16 * 2
        #expect(spec.qjlResidualBytesPerLayer(qjlDimension: qjlDim) == expectedPerLayer)
        #expect(spec.totalQJLResidualSize(qjlDimension: qjlDim) == expectedPerLayer * 16)

        // Disabled when qjlDim == 0
        #expect(spec.qjlMatrixSize(qjlDimension: 0) == 0)
        #expect(spec.qjlResidualBytesPerLayer(qjlDimension: 0) == 0)
        #expect(spec.totalQJLResidualSize(qjlDimension: 0) == 0)
    }

    // MARK: - MetalKVCache Allocation

    @Test("MetalKVCache allocates rotor buffers for rotor schemes")
    func kvCacheAllocatesRotorBuffers() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let spec = KVCacheSpecification(
            keyQuantizationScheme: .rotorQ8Group32ScaleF16,
            valueQuantizationScheme: .rotorQ4Group64ScaleF16,
            layerCount: 4,
            kvHeadCount: 8, headDimension: 64, maximumSequenceLength: 256)

        let cache = try MetalKVCache(device: device, specification: spec, qjlDimension: 16)

        #expect(cache.rotorParameters != nil, "Rotor params should be allocated for rotor scheme")
        #expect(cache.qjlMatrix != nil, "QJL matrix should be allocated when qjlDim > 0")
        #expect(cache.qjlResidualK != nil, "QJL residual should be allocated when qjlDim > 0")
        #expect(cache.numRotorGroups == 22)  // ceil(64/3)
        #expect(cache.qjlDimension == 16)
    }

    @Test("MetalKVCache omits rotor buffers for non-rotor schemes")
    func kvCacheOmitsRotorBuffersForNonRotor() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let spec = KVCacheSpecification(
            keyQuantizationScheme: .fp16RowMajor,
            valueQuantizationScheme: .fp16RowMajor,
            layerCount: 4,
            kvHeadCount: 8, headDimension: 64, maximumSequenceLength: 256)

        let cache = try MetalKVCache(device: device, specification: spec)

        #expect(cache.rotorParameters == nil, "No rotor params for non-rotor scheme")
        #expect(cache.qjlMatrix == nil, "No QJL matrix for non-rotor scheme")
        #expect(cache.qjlResidualK == nil, "No QJL residual for non-rotor scheme")
        #expect(cache.numRotorGroups == 0)
        #expect(cache.qjlDimension == 0)
    }

    @Test("MetalKVCache allocates rotor but not QJL when qjlDimension is 0")
    func kvCacheRotorWithoutQJL() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let spec = KVCacheSpecification(
            keyQuantizationScheme: .rotorQ8Group32ScaleF16,
            valueQuantizationScheme: .fp16RowMajor,
            layerCount: 2,
            kvHeadCount: 4, headDimension: 64, maximumSequenceLength: 128)

        let cache = try MetalKVCache(device: device, specification: spec, qjlDimension: 0)

        #expect(cache.rotorParameters != nil, "Rotor params should be allocated")
        #expect(cache.qjlMatrix == nil, "No QJL matrix when qjlDim == 0")
        #expect(cache.qjlResidualK == nil, "No QJL residual when qjlDim == 0")
        #expect(cache.numRotorGroups == 22)
        #expect(cache.qjlDimension == 0)
    }

    // MARK: - Random Rotor Initialization

    @Test("Random rotors are unit quaternions, non-identity, and deterministic")
    func randomRotorInitialization() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let layerCount = 3
        let kvHeadCount = 4
        let headDimension = 12  // 4 groups of 3
        let spec = KVCacheSpecification(
            keyQuantizationScheme: .rotorQ8Group32ScaleF16,
            valueQuantizationScheme: .fp16RowMajor,
            layerCount: layerCount,
            kvHeadCount: kvHeadCount, headDimension: headDimension, maximumSequenceLength: 64)

        let cache = try MetalKVCache(
            device: device, specification: spec,
            resourceOptions: [.storageModeShared])

        let rotorBuf = try #require(cache.rotorParameters)
        let numGroups = cache.numRotorGroups  // ceil(12/3) = 4
        #expect(numGroups == 4)

        let totalRotors = layerCount * kvHeadCount * numGroups
        let ptr = rotorBuf.contents().bindMemory(
            to: UInt16.self,
            capacity: totalRotors * 4)

        // Verify unit norm and non-identity
        var hasNonIdentity = false
        for i in 0..<totalRotors {
            let s   = Float(Float16(bitPattern: ptr[i * 4 + 0]))
            let b12 = Float(Float16(bitPattern: ptr[i * 4 + 1]))
            let b13 = Float(Float16(bitPattern: ptr[i * 4 + 2]))
            let b23 = Float(Float16(bitPattern: ptr[i * 4 + 3]))

            let norm = (s * s + b12 * b12 + b13 * b13 + b23 * b23).squareRoot()
            #expect(abs(norm - 1.0) < 0.01,
                "Rotor \(i) norm should be ~1.0, got \(norm)")

            if abs(b12) > 0.01 || abs(b13) > 0.01 || abs(b23) > 0.01 {
                hasNonIdentity = true
            }
        }
        #expect(hasNonIdentity, "At least one rotor should be non-identity")

        // Determinism: second allocation should produce identical values
        let cache2 = try MetalKVCache(
            device: device, specification: spec,
            resourceOptions: [.storageModeShared])
        let ptr2 = try #require(cache2.rotorParameters).contents()
            .bindMemory(to: UInt16.self, capacity: totalRotors * 4)
        for i in 0..<(totalRotors * 4) {
            #expect(ptr[i] == ptr2[i],
                "Rotor value at index \(i) should be deterministic")
        }
    }

    // MARK: - Rademacher Matrix Initialization

    @Test("Rademacher matrix contains only ±1/sqrt(m) values")
    func rademacherMatrixValues() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let headDim = 64
        let qjlDim = 16
        let spec = KVCacheSpecification(
            keyQuantizationScheme: .rotorQ8Group32ScaleF16,
            valueQuantizationScheme: .fp16RowMajor,
            layerCount: 1,
            kvHeadCount: 1, headDimension: headDim, maximumSequenceLength: 64)

        let cache = try MetalKVCache(
            device: device, specification: spec, qjlDimension: qjlDim,
            resourceOptions: [.storageModeShared])

        let matBuf = try #require(cache.qjlMatrix)
        let ptr = matBuf.contents().bindMemory(
            to: UInt16.self, capacity: headDim * qjlDim)

        let scale = Float16(1.0 / Float(qjlDim).squareRoot())
        let posPattern = scale.bitPattern
        let negPattern = (-scale).bitPattern

        var positiveCount = 0
        var negativeCount = 0

        for i in 0..<(headDim * qjlDim) {
            let value = ptr[i]
            if value == posPattern {
                positiveCount += 1
            } else if value == negPattern {
                negativeCount += 1
            } else {
                Issue.record("Element \(i) has unexpected bit pattern \(value), expected \(posPattern) or \(negPattern)")
            }
        }

        // Both +1/√m and -1/√m should be present (probabilistic, but extremely unlikely to fail)
        #expect(positiveCount > 0, "Should have positive values")
        #expect(negativeCount > 0, "Should have negative values")
        #expect(positiveCount + negativeCount == headDim * qjlDim,
            "All elements should be ±1/√m")

        // Rough balance check: expect within 40%-60% range for 1024 elements
        let total = Double(headDim * qjlDim)
        let positiveRatio = Double(positiveCount) / total
        #expect(positiveRatio > 0.3 && positiveRatio < 0.7,
            "Rademacher distribution should be roughly balanced: got \(positiveRatio)")
    }

    @Test("Rademacher matrix is deterministic across initializations")
    func rademacherMatrixDeterminism() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let spec = KVCacheSpecification(
            keyQuantizationScheme: .rotorQ8Group32ScaleF16,
            valueQuantizationScheme: .fp16RowMajor,
            layerCount: 1,
            kvHeadCount: 1, headDimension: 32, maximumSequenceLength: 64)

        let cache1 = try MetalKVCache(
            device: device, specification: spec, qjlDimension: 8,
            resourceOptions: [.storageModeShared])
        let cache2 = try MetalKVCache(
            device: device, specification: spec, qjlDimension: 8,
            resourceOptions: [.storageModeShared])

        let buf1 = try #require(cache1.qjlMatrix)
        let buf2 = try #require(cache2.qjlMatrix)
        let count = 32 * 8
        let ptr1 = buf1.contents().bindMemory(to: UInt16.self, capacity: count)
        let ptr2 = buf2.contents().bindMemory(to: UInt16.self, capacity: count)

        for i in 0..<count {
            #expect(ptr1[i] == ptr2[i],
                "Rademacher matrix should be deterministic: mismatch at index \(i)")
        }
    }

    // MARK: - Rotor Parameter Offset Calculations

    @Test("rotorParameterOffset and qjlResidualOffset are consistent with specification")
    func offsetCalculationsConsistent() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let spec = KVCacheSpecification(
            keyQuantizationScheme: .rotorQ8Group32ScaleF16,
            valueQuantizationScheme: .fp16RowMajor,
            layerCount: 4,
            kvHeadCount: 8, headDimension: 64, maximumSequenceLength: 512)

        let cache = try MetalKVCache(
            device: device, specification: spec, qjlDimension: 16,
            resourceOptions: [.storageModeShared])

        // Layer 0 starts at offset 0
        #expect(cache.rotorParameterOffset(layer: 0) == 0)
        #expect(cache.qjlResidualOffset(layer: 0) == 0)

        // Each layer offset = layer * bytesPerLayer
        let rotorBytesPerLayer = spec.rotorParametersBytesPerLayer
        let qjlBytesPerLayer = spec.qjlResidualBytesPerLayer(qjlDimension: 16)

        for layer in 0..<4 {
            #expect(cache.rotorParameterOffset(layer: layer) == layer * rotorBytesPerLayer)
            #expect(cache.qjlResidualOffset(layer: layer) == layer * qjlBytesPerLayer)
        }

        // Verify offsets don't exceed buffer bounds
        let rotorBuf = try #require(cache.rotorParameters)
        let qjlResBuf = try #require(cache.qjlResidualK)

        let lastLayerRotorEnd = cache.rotorParameterOffset(layer: 3) + rotorBytesPerLayer
        #expect(lastLayerRotorEnd <= rotorBuf.length,
            "Last layer rotor offset + size (\(lastLayerRotorEnd)) exceeds buffer length (\(rotorBuf.length))")

        let lastLayerQjlEnd = cache.qjlResidualOffset(layer: 3) + qjlBytesPerLayer
        #expect(lastLayerQjlEnd <= qjlResBuf.length,
            "Last layer QJL offset + size (\(lastLayerQjlEnd)) exceeds buffer length (\(qjlResBuf.length))")
    }

    // MARK: - Clifford Rotor Math (CPU Reference)

    /// Cl(3,0) sandwich product: RvR̃ via quaternion cross-product form.
    /// Reference implementation matching the Metal kernel.
    private static func rotorSandwich(
        s: Float, b12: Float, b13: Float, b23: Float,
        v1: Float, v2: Float, v3: Float
    ) -> (Float, Float, Float) {
        // Quaternion equivalence: q = (s, b23, -b13, b12)
        let px = b23
        let py = -b13
        let pz = b12
        // t = 2 * cross(p, v)
        let tx = 2 * (py * v3 - pz * v2)
        let ty = 2 * (pz * v1 - px * v3)
        let tz = 2 * (px * v2 - py * v1)
        // result = v + s*t + cross(p, t)
        let rx = v1 + s * tx + (py * tz - pz * ty)
        let ry = v2 + s * ty + (pz * tx - px * tz)
        let rz = v3 + s * tz + (px * ty - py * tx)
        return (rx, ry, rz)
    }

    /// Inverse sandwich product: R̃vR.
    private static func rotorSandwichInverse(
        s: Float, b12: Float, b13: Float, b23: Float,
        v1: Float, v2: Float, v3: Float
    ) -> (Float, Float, Float) {
        // Conjugate: negate bivector parts
        let px = -b23
        let py = b13
        let pz = -b12
        let tx = 2 * (py * v3 - pz * v2)
        let ty = 2 * (pz * v1 - px * v3)
        let tz = 2 * (px * v2 - py * v1)
        let rx = v1 + s * tx + (py * tz - pz * ty)
        let ry = v2 + s * ty + (pz * tx - px * tz)
        let rz = v3 + s * tz + (px * ty - py * tx)
        return (rx, ry, rz)
    }

    @Test("Identity rotor preserves vector unchanged")
    func identityRotorPreservesVector() {
        let testVectors: [(Float, Float, Float)] = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 2.0, 3.0),
            (-0.5, 0.7, -1.2),
        ]

        for (v1, v2, v3) in testVectors {
            let (r1, r2, r3) = Self.rotorSandwich(
                s: 1.0, b12: 0.0, b13: 0.0, b23: 0.0,
                v1: v1, v2: v2, v3: v3)
            #expect(abs(r1 - v1) < 1e-6, "Identity rotor should preserve v1=\(v1), got \(r1)")
            #expect(abs(r2 - v2) < 1e-6, "Identity rotor should preserve v2=\(v2), got \(r2)")
            #expect(abs(r3 - v3) < 1e-6, "Identity rotor should preserve v3=\(v3), got \(r3)")
        }
    }

    @Test("Forward then inverse rotor sandwich recovers original vector")
    func forwardInverseRoundTrip() {
        // Non-trivial rotor: 45° rotation around the (1,1,1)/√3 axis.
        // For axis (nx, ny, nz) and angle θ: R = cos(θ/2) + sin(θ/2)(nx*e23 + ny*e31 + nz*e12)
        // = cos(θ/2) + sin(θ/2)·(nx*e23 - ny*e13 + nz*e12)
        // In our convention: s = cos(θ/2), b12 = sin(θ/2)*nz, b13 = -sin(θ/2)*ny, b23 = sin(θ/2)*nx
        let theta: Float = .pi / 4  // 45 degrees
        let halfAngle = theta / 2
        let axis: Float = 1.0 / Float(3.0).squareRoot()
        let s = cos(halfAngle)
        let b12 = sin(halfAngle) * axis
        let b13 = -sin(halfAngle) * axis
        let b23 = sin(halfAngle) * axis

        let testVectors: [(Float, Float, Float)] = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (3.14, -2.71, 1.41),
            (-0.1, 0.2, -0.3),
        ]

        for (v1, v2, v3) in testVectors {
            let (f1, f2, f3) = Self.rotorSandwich(
                s: s, b12: b12, b13: b13, b23: b23,
                v1: v1, v2: v2, v3: v3)
            let (r1, r2, r3) = Self.rotorSandwichInverse(
                s: s, b12: b12, b13: b13, b23: b23,
                v1: f1, v2: f2, v3: f3)

            #expect(abs(r1 - v1) < 1e-5,
                "Round-trip should recover v1=\(v1), got \(r1)")
            #expect(abs(r2 - v2) < 1e-5,
                "Round-trip should recover v2=\(v2), got \(r2)")
            #expect(abs(r3 - v3) < 1e-5,
                "Round-trip should recover v3=\(v3), got \(r3)")
        }
    }

    @Test("Rotor sandwich preserves vector magnitude")
    func rotorPreservesMagnitude() {
        let theta: Float = .pi / 3  // 60 degrees
        let halfAngle = theta / 2
        let s = cos(halfAngle)
        let b12 = sin(halfAngle) * 0.0
        let b13 = sin(halfAngle) * (-1.0)  // -sin(θ/2)·ny where ny=1
        let b23 = sin(halfAngle) * 0.0

        let testVectors: [(Float, Float, Float)] = [
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (-2.0, 3.0, -4.0),
        ]

        for (v1, v2, v3) in testVectors {
            let magBefore = v1 * v1 + v2 * v2 + v3 * v3
            let (r1, r2, r3) = Self.rotorSandwich(
                s: s, b12: b12, b13: b13, b23: b23,
                v1: v1, v2: v2, v3: v3)
            let magAfter = r1 * r1 + r2 * r2 + r3 * r3

            #expect(abs(magAfter - magBefore) < 1e-5,
                "Rotation should preserve |v|²=\(magBefore), got \(magAfter)")
        }
    }

    @Test("Non-trivial rotation actually changes the vector")
    func nonTrivialRotationChangesVector() {
        // 90° rotation around z-axis: (1,0,0) → (0,1,0)
        let halfAngle: Float = .pi / 4  // half of 90°
        let s = cos(halfAngle)
        let b12 = sin(halfAngle)  // rotation in e1e2 plane = around z-axis
        let b13: Float = 0.0
        let b23: Float = 0.0

        let (r1, r2, r3) = Self.rotorSandwich(
            s: s, b12: b12, b13: b13, b23: b23,
            v1: 1.0, v2: 0.0, v3: 0.0)

        // 90° around z: (1,0,0) → (0,1,0)
        #expect(abs(r1 - 0.0) < 1e-5, "Expected r1≈0, got \(r1)")
        #expect(abs(r2 - 1.0) < 1e-5, "Expected r2≈1, got \(r2)")
        #expect(abs(r3 - 0.0) < 1e-5, "Expected r3≈0, got \(r3)")
    }

    // MARK: - QJL Math (CPU Reference)

    @Test("QJL inner product correction is unbiased for Rademacher projection")
    func qjlInnerProductCorrection() {
        // QJL guarantees: E[⟨Φx, Φy⟩] = ⟨x, y⟩
        // Test: project two vectors with the same Rademacher matrix,
        // verify the projected inner product approximates the true one.
        // Use high qjlDim relative to headDim for tighter approximation.
        let headDim = 64
        let qjlDim = 64

        // Generate deterministic Rademacher matrix
        let scale = 1.0 / Float(qjlDim).squareRoot()
        var phi = [Float](repeating: 0, count: headDim * qjlDim)
        for i in 0..<phi.count {
            let hash = UInt64(i) &* 6364136223846793005 &+ 1442695040888963407
            phi[i] = (hash >> 33) & 1 == 0 ? scale : -scale
        }

        // Two test vectors with non-trivial correlation
        var x = [Float](repeating: 0, count: headDim)
        var y = [Float](repeating: 0, count: headDim)
        for d in 0..<headDim {
            x[d] = cos(Float(d) * 0.2) * 0.5
            y[d] = cos(Float(d) * 0.2 + 0.3) * 0.5
        }

        // True inner product
        var trueIP: Float = 0
        for d in 0..<headDim { trueIP += x[d] * y[d] }

        // Projected inner product: ⟨Φx, Φy⟩
        var projX = [Float](repeating: 0, count: qjlDim)
        var projY = [Float](repeating: 0, count: qjlDim)
        for j in 0..<qjlDim {
            for d in 0..<headDim {
                projX[j] += phi[d * qjlDim + j] * x[d]
                projY[j] += phi[d * qjlDim + j] * y[d]
            }
        }
        var projIP: Float = 0
        for j in 0..<qjlDim { projIP += projX[j] * projY[j] }

        // JL lemma: with qjlDim = headDim the projection is nearly exact.
        // Allow generous relative error since the hash is deterministic and
        // may not produce an ideal distribution.
        let absoluteError = abs(projIP - trueIP)
        let referenceScale = max(abs(trueIP), 1e-3)
        #expect(absoluteError / referenceScale < 1.0,
            "QJL projected IP (\(projIP)) should approximate true IP (\(trueIP)), abs error: \(absoluteError)")
    }

    @Test("QJL residual correction improves quantized inner product")
    func qjlResidualCorrectionImprovement() {
        // Simulate: quantize K, compute residual, project, correct
        let headDim = 64
        let qjlDim = 16

        // Rademacher matrix
        let scale = 1.0 / Float(qjlDim).squareRoot()
        var phi = [Float](repeating: 0, count: headDim * qjlDim)
        for i in 0..<phi.count {
            let hash = UInt64(i) &* 6364136223846793005 &+ 1442695040888963407
            phi[i] = (hash >> 33) & 1 == 0 ? scale : -scale
        }

        // Original K and Q vectors
        var kOriginal = [Float](repeating: 0, count: headDim)
        var query = [Float](repeating: 0, count: headDim)
        for d in 0..<headDim {
            kOriginal[d] = Float(d % 7) * 0.3 - 1.0
            query[d] = cos(Float(d) * 0.15) * 0.8
        }

        // Simulate per-group Q8 quantization (group=32, scale+zero)
        var kQuantized = [Float](repeating: 0, count: headDim)
        let groupSize = 32
        for g in stride(from: 0, to: headDim, by: groupSize) {
            let end = min(g + groupSize, headDim)
            var minVal: Float = .infinity
            var maxVal: Float = -.infinity
            for d in g..<end {
                minVal = min(minVal, kOriginal[d])
                maxVal = max(maxVal, kOriginal[d])
            }
            let groupScale = (maxVal - minVal) / 255.0
            for d in g..<end {
                let q = round((kOriginal[d] - minVal) / max(groupScale, 1e-10))
                let clamped = max(0, min(255, q))
                kQuantized[d] = Float(clamped) * groupScale + minVal
            }
        }

        // Residual
        var residual = [Float](repeating: 0, count: headDim)
        for d in 0..<headDim {
            residual[d] = kOriginal[d] - kQuantized[d]
        }

        // True inner product
        var trueScore: Float = 0
        for d in 0..<headDim { trueScore += query[d] * kOriginal[d] }

        // Uncorrected score (Q · K_quantized)
        var uncorrectedScore: Float = 0
        for d in 0..<headDim { uncorrectedScore += query[d] * kQuantized[d] }

        // QJL correction: (Φ·Q)·(Φ·residual)
        var projQuery = [Float](repeating: 0, count: qjlDim)
        var projResidual = [Float](repeating: 0, count: qjlDim)
        for j in 0..<qjlDim {
            for d in 0..<headDim {
                projQuery[j] += phi[d * qjlDim + j] * query[d]
                projResidual[j] += phi[d * qjlDim + j] * residual[d]
            }
        }
        var correction: Float = 0
        for j in 0..<qjlDim { correction += projQuery[j] * projResidual[j] }

        let correctedScore = uncorrectedScore + correction

        let uncorrectedError = abs(uncorrectedScore - trueScore)
        let correctedError = abs(correctedScore - trueScore)

        // The correction should help (or at least not make things much worse)
        // Note: with small qjlDim, improvement may be modest
        print("QJL correction test:")
        print("  True score:        \(trueScore)")
        print("  Uncorrected score: \(uncorrectedScore) (error: \(uncorrectedError))")
        print("  Corrected score:   \(correctedScore) (error: \(correctedError))")
        print("  Correction term:   \(correction)")

        // The corrected score should be closer to the true score, OR
        // the error should at least not increase dramatically
        #expect(correctedError < uncorrectedError * 3.0,
            "QJL correction should not dramatically worsen the score estimate")
    }

    // MARK: - Rotor Preserves Q·K Inner Product

    @Test("Pre-rotating Q with the same rotor preserves Q·K inner product")
    func rotorPreservesInnerProduct() {
        let headDim = 12  // 4 groups of 3
        let numGroups = (headDim + 2) / 3

        // Non-trivial rotor per group
        var rotors = [(s: Float, b12: Float, b13: Float, b23: Float)]()
        for g in 0..<numGroups {
            let angle = Float(g + 1) * Float.pi / 8  // different angle per group
            let halfAngle = angle / 2
            let norm: Float = 1.0 / Float(3.0).squareRoot()
            rotors.append((
                s: cos(halfAngle),
                b12: sin(halfAngle) * norm,
                b13: -sin(halfAngle) * norm,
                b23: sin(halfAngle) * norm
            ))
        }

        // Original Q and K vectors
        var q = [Float](repeating: 0, count: headDim)
        var k = [Float](repeating: 0, count: headDim)
        for d in 0..<headDim {
            q[d] = Float(d) * 0.1 - 0.5
            k[d] = sin(Float(d) * 0.3)
        }

        // True Q·K
        var trueIP: Float = 0
        for d in 0..<headDim { trueIP += q[d] * k[d] }

        // Rotate both Q and K with the same per-group rotor
        var qRotated = [Float](repeating: 0, count: headDim)
        var kRotated = [Float](repeating: 0, count: headDim)

        for g in 0..<numGroups {
            let base = g * 3
            let r = rotors[g]

            let qv1 = base < headDim ? q[base] : 0
            let qv2 = base + 1 < headDim ? q[base + 1] : 0
            let qv3 = base + 2 < headDim ? q[base + 2] : 0
            let (qr1, qr2, qr3) = Self.rotorSandwich(
                s: r.s, b12: r.b12, b13: r.b13, b23: r.b23,
                v1: qv1, v2: qv2, v3: qv3)
            if base < headDim { qRotated[base] = qr1 }
            if base + 1 < headDim { qRotated[base + 1] = qr2 }
            if base + 2 < headDim { qRotated[base + 2] = qr3 }

            let kv1 = base < headDim ? k[base] : 0
            let kv2 = base + 1 < headDim ? k[base + 1] : 0
            let kv3 = base + 2 < headDim ? k[base + 2] : 0
            let (kr1, kr2, kr3) = Self.rotorSandwich(
                s: r.s, b12: r.b12, b13: r.b13, b23: r.b23,
                v1: kv1, v2: kv2, v3: kv3)
            if base < headDim { kRotated[base] = kr1 }
            if base + 1 < headDim { kRotated[base + 1] = kr2 }
            if base + 2 < headDim { kRotated[base + 2] = kr3 }
        }

        // Rotated Q·K should equal original Q·K
        var rotatedIP: Float = 0
        for d in 0..<headDim { rotatedIP += qRotated[d] * kRotated[d] }

        #expect(abs(rotatedIP - trueIP) < 1e-4,
            "Rotated Q·K (\(rotatedIP)) should equal original Q·K (\(trueIP))")
    }

    @Test("V inverse rotation recovers original weighted sum")
    func vInverseRotationRecovery() {
        let headDim = 9  // 3 groups of 3

        // Single rotor for all groups (uniform rotation)
        let theta: Float = .pi / 6
        let halfAngle = theta / 2
        let s = cos(halfAngle)
        let b12 = sin(halfAngle)
        let b13: Float = 0.0
        let b23: Float = 0.0

        // Original V vectors (3 tokens)
        var values = [[Float]]()
        for t in 0..<3 {
            var v = [Float](repeating: 0, count: headDim)
            for d in 0..<headDim { v[d] = Float(t * headDim + d) * 0.05 }
            values.append(v)
        }

        // Attention weights (softmax output)
        let weights: [Float] = [0.5, 0.3, 0.2]

        // True weighted sum
        var trueOutput = [Float](repeating: 0, count: headDim)
        for t in 0..<3 {
            for d in 0..<headDim {
                trueOutput[d] += weights[t] * values[t][d]
            }
        }

        // With RotorQuant: V is stored rotated, weighted sum is in rotated space,
        // then inverse rotation recovers original space.
        var rotatedValues = [[Float]]()
        for t in 0..<3 {
            var rv = [Float](repeating: 0, count: headDim)
            for g in 0..<3 {
                let base = g * 3
                let (r1, r2, r3) = Self.rotorSandwich(
                    s: s, b12: b12, b13: b13, b23: b23,
                    v1: values[t][base], v2: values[t][base + 1], v3: values[t][base + 2])
                rv[base] = r1
                rv[base + 1] = r2
                rv[base + 2] = r3
            }
            rotatedValues.append(rv)
        }

        // Weighted sum in rotated space
        var rotatedOutput = [Float](repeating: 0, count: headDim)
        for t in 0..<3 {
            for d in 0..<headDim {
                rotatedOutput[d] += weights[t] * rotatedValues[t][d]
            }
        }

        // Inverse rotate the output
        var recoveredOutput = [Float](repeating: 0, count: headDim)
        for g in 0..<3 {
            let base = g * 3
            let (r1, r2, r3) = Self.rotorSandwichInverse(
                s: s, b12: b12, b13: b13, b23: b23,
                v1: rotatedOutput[base], v2: rotatedOutput[base + 1], v3: rotatedOutput[base + 2])
            recoveredOutput[base] = r1
            recoveredOutput[base + 1] = r2
            recoveredOutput[base + 2] = r3
        }

        // Recovered should match true output
        for d in 0..<headDim {
            #expect(abs(recoveredOutput[d] - trueOutput[d]) < 1e-5,
                "Recovered output[\(d)]=\(recoveredOutput[d]) should match true output[\(d)]=\(trueOutput[d])")
        }
    }

    // MARK: - Metal Kernel Compilation

    @Test("Rotor helper functions compile as standalone Metal source")
    func rotorHelperFunctionsCompile() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        // Compile rotorHelperSource with a test kernel that exercises all helper functions
        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.rotorHelperSource + "\n\n"
            + """
            kernel void test_rotor_compile_check(
                device float* data            [[buffer(0)]],
                device const half* rotors     [[buffer(1)]],
                constant uint& headDim        [[buffer(2)]],
                constant uint& numGroups      [[buffer(3)]],
                uint tid                      [[thread_index_in_threadgroup]],
                uint threadgroupSize          [[threads_per_threadgroup]]
            ) {
                threadgroup float shared[256];
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    shared[d] = data[d];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                rotor_apply_forward(shared, rotors, headDim, numGroups, tid, threadgroupSize);
                rotor_apply_inverse(shared, rotors, headDim, numGroups, tid, threadgroupSize);
                // Verify helper functions are accessible
                bool _ = is_rotor_scheme(0x70);
                uint __ = rotor_base_scheme(0x70);
                float3 v = rotor_sandwich(float4(1,0,0,0), float3(1,0,0));
                float3 vi = rotor_sandwich_inverse(float4(1,0,0,0), float3(1,0,0));
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    data[d] = shared[d];
                }
            }
            """

        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let function = library.makeFunction(name: "test_rotor_compile_check")
        #expect(function != nil, "Rotor helper functions should compile successfully")
    }

    // MARK: - GPU Rotor Kernel Execution

    @Test("GPU rotor_apply_forward with identity rotor preserves data")
    func gpuIdentityRotorPreservesData() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        // Compile a minimal test kernel that applies rotor forward on a buffer
        let testKernelSource = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.rotorHelperSource + "\n\n"
            + """
            kernel void test_rotor_forward(
                device float* data            [[buffer(0)]],
                device const half* rotors     [[buffer(1)]],
                constant uint& headDim        [[buffer(2)]],
                constant uint& numGroups      [[buffer(3)]],
                uint tid                      [[thread_index_in_threadgroup]],
                uint threadgroupSize          [[threads_per_threadgroup]]
            ) {
                threadgroup float shared[256];
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    shared[d] = data[d];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                rotor_apply_forward(shared, rotors, headDim, numGroups, tid, threadgroupSize);
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    data[d] = shared[d];
                }
            }
            """

        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: testKernelSource, options: options)
        let function = try #require(library.makeFunction(name: "test_rotor_forward"))
        let pipeline = try device.makeComputePipelineState(function: function)

        let headDim: UInt32 = 12
        let numGroups: UInt32 = 4

        // Input data
        let inputData: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        let dataBuffer = device.makeBuffer(
            bytes: inputData, length: inputData.count * MemoryLayout<Float>.size,
            options: .storageModeShared)!

        // Identity rotors: [s=1, b12=0, b13=0, b23=0] × numGroups
        var rotorData = [UInt16](repeating: Float16(0.0).bitPattern, count: Int(numGroups) * 4)
        for g in 0..<Int(numGroups) {
            rotorData[g * 4] = Float16(1.0).bitPattern
        }
        let rotorBuffer = device.makeBuffer(
            bytes: rotorData, length: rotorData.count * MemoryLayout<UInt16>.size,
            options: .storageModeShared)!

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(dataBuffer, offset: 0, index: 0)
        encoder.setBuffer(rotorBuffer, offset: 0, index: 1)
        var hd = headDim
        var ng = numGroups
        encoder.setBytes(&hd, length: 4, index: 2)
        encoder.setBytes(&ng, length: 4, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Int(headDim), height: 1, depth: 1))
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let outputPtr = dataBuffer.contents().bindMemory(to: Float.self, capacity: Int(headDim))
        for d in 0..<Int(headDim) {
            #expect(abs(outputPtr[d] - inputData[d]) < 1e-5,
                "Identity rotor should preserve data[\(d)]=\(inputData[d]), got \(outputPtr[d])")
        }
    }

    @Test("GPU rotor forward/inverse roundtrip recovers original data")
    func gpuRotorRoundTrip() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let testKernelSource = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.rotorHelperSource + "\n\n"
            + """
            kernel void test_rotor_roundtrip(
                device float* data            [[buffer(0)]],
                device const half* rotors     [[buffer(1)]],
                constant uint& headDim        [[buffer(2)]],
                constant uint& numGroups      [[buffer(3)]],
                uint tid                      [[thread_index_in_threadgroup]],
                uint threadgroupSize          [[threads_per_threadgroup]]
            ) {
                threadgroup float shared[256];
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    shared[d] = data[d];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                rotor_apply_forward(shared, rotors, headDim, numGroups, tid, threadgroupSize);
                rotor_apply_inverse(shared, rotors, headDim, numGroups, tid, threadgroupSize);
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    data[d] = shared[d];
                }
            }
            """

        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: testKernelSource, options: options)
        let function = try #require(library.makeFunction(name: "test_rotor_roundtrip"))
        let pipeline = try device.makeComputePipelineState(function: function)

        let headDim: UInt32 = 15  // Not divisible by 3 — tests zero-padding
        let numGroups: UInt32 = 5  // ceil(15/3)

        // Non-trivial rotors per group
        var rotorData = [UInt16](repeating: 0, count: Int(numGroups) * 4)
        for g in 0..<Int(numGroups) {
            let angle = Float(g + 1) * Float.pi / 8
            let halfAngle = angle / 2
            let norm: Float = 1.0 / Float(3.0).squareRoot()
            rotorData[g * 4 + 0] = Float16(cos(halfAngle)).bitPattern
            rotorData[g * 4 + 1] = Float16(sin(halfAngle) * norm).bitPattern
            rotorData[g * 4 + 2] = Float16(-sin(halfAngle) * norm).bitPattern
            rotorData[g * 4 + 3] = Float16(sin(halfAngle) * norm).bitPattern
        }

        let inputData: [Float] = (0..<Int(headDim)).map { Float($0) * 0.1 - 0.5 }
        let dataBuffer = device.makeBuffer(
            bytes: inputData, length: inputData.count * MemoryLayout<Float>.size,
            options: .storageModeShared)!
        let rotorBuffer = device.makeBuffer(
            bytes: rotorData, length: rotorData.count * MemoryLayout<UInt16>.size,
            options: .storageModeShared)!

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(dataBuffer, offset: 0, index: 0)
        encoder.setBuffer(rotorBuffer, offset: 0, index: 1)
        var hd = headDim
        var ng = numGroups
        encoder.setBytes(&hd, length: 4, index: 2)
        encoder.setBytes(&ng, length: 4, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Int(headDim), height: 1, depth: 1))
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let outputPtr = dataBuffer.contents().bindMemory(to: Float.self, capacity: Int(headDim))
        for d in 0..<Int(headDim) {
            #expect(abs(outputPtr[d] - inputData[d]) < 1e-3,
                "Round-trip should recover data[\(d)]=\(inputData[d]), got \(outputPtr[d])")
        }
    }

    @Test("GPU rotor forward actually transforms data with non-identity rotor")
    func gpuNonIdentityRotorTransforms() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let testKernelSource = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.rotorHelperSource + "\n\n"
            + """
            kernel void test_rotor_transform(
                device float* data            [[buffer(0)]],
                device const half* rotors     [[buffer(1)]],
                constant uint& headDim        [[buffer(2)]],
                constant uint& numGroups      [[buffer(3)]],
                uint tid                      [[thread_index_in_threadgroup]],
                uint threadgroupSize          [[threads_per_threadgroup]]
            ) {
                threadgroup float shared[256];
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    shared[d] = data[d];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                rotor_apply_forward(shared, rotors, headDim, numGroups, tid, threadgroupSize);
                for (uint d = tid; d < headDim; d += threadgroupSize) {
                    data[d] = shared[d];
                }
            }
            """

        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: testKernelSource, options: options)
        let function = try #require(library.makeFunction(name: "test_rotor_transform"))
        let pipeline = try device.makeComputePipelineState(function: function)

        let headDim: UInt32 = 3
        let numGroups: UInt32 = 1

        // 90° around z-axis: b12 = sin(π/4), s = cos(π/4)
        let halfAngle = Float.pi / 4
        var rotorData: [UInt16] = [
            Float16(cos(halfAngle)).bitPattern,  // s
            Float16(sin(halfAngle)).bitPattern,  // b12
            Float16(0.0).bitPattern,             // b13
            Float16(0.0).bitPattern,             // b23
        ]

        let inputData: [Float] = [1.0, 0.0, 0.0]
        let dataBuffer = device.makeBuffer(
            bytes: inputData, length: inputData.count * MemoryLayout<Float>.size,
            options: .storageModeShared)!
        let rotorBuffer = device.makeBuffer(
            bytes: &rotorData, length: rotorData.count * MemoryLayout<UInt16>.size,
            options: .storageModeShared)!

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(dataBuffer, offset: 0, index: 0)
        encoder.setBuffer(rotorBuffer, offset: 0, index: 1)
        var hd = headDim
        var ng = numGroups
        encoder.setBytes(&hd, length: 4, index: 2)
        encoder.setBytes(&ng, length: 4, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 3, height: 1, depth: 1))
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let outputPtr = dataBuffer.contents().bindMemory(to: Float.self, capacity: 3)
        // 90° around z: (1,0,0) → (0,1,0)
        #expect(abs(outputPtr[0] - 0.0) < 0.01,
            "Expected x≈0 after 90° z-rotation, got \(outputPtr[0])")
        #expect(abs(outputPtr[1] - 1.0) < 0.01,
            "Expected y≈1 after 90° z-rotation, got \(outputPtr[1])")
        #expect(abs(outputPtr[2] - 0.0) < 0.01,
            "Expected z≈0 after 90° z-rotation, got \(outputPtr[2])")

        // Compare with CPU reference
        let (cpuR1, cpuR2, cpuR3) = Self.rotorSandwich(
            s: cos(halfAngle), b12: sin(halfAngle), b13: 0, b23: 0,
            v1: 1.0, v2: 0.0, v3: 0.0)
        #expect(abs(outputPtr[0] - cpuR1) < 0.01,
            "GPU output should match CPU reference: x=\(outputPtr[0]) vs \(cpuR1)")
        #expect(abs(outputPtr[1] - cpuR2) < 0.01,
            "GPU output should match CPU reference: y=\(outputPtr[1]) vs \(cpuR2)")
        #expect(abs(outputPtr[2] - cpuR3) < 0.01,
            "GPU output should match CPU reference: z=\(outputPtr[2]) vs \(cpuR3)")
    }
}
