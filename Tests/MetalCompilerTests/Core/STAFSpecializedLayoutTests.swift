import Testing
import Metal
@testable import MetalCompiler
import LMArchitecture
import ModelDeclarations
import LMIR

#if ENABLE_METAL_PROBES
@Suite("STAF Specialized Layout", .serialized)
struct STAFSpecializedLayoutTests {
    private static let realModelSTAFPath = BenchmarkSupport.stafPath

    private static func requireRealModelStore() throws -> STAFWeightStore {
        guard let resources = try RealModelTestSupport.loadOrSkip(
            skipMessage: "Missing STAF fixture at \(Self.realModelSTAFPath)"
        ) else {
            throw MetalCompilerError.deviceSetupFailed("Missing STAF fixture at \(Self.realModelSTAFPath)")
        }
        defer { resources.release() }
        return resources.store
    }

    @Test("blockedRows8Tiles128 packs row-major weights into 8x128 tiles")
    func blockedRows8Tiles128PackingMatchesExpectedOrder() throws {
        let fixture = try makeBlockedRowsFixture()
        let packed = fixture.packedElements

        var packedIndex = 0
        for rowBlock in stride(from: 0, to: fixture.outputDimension, by: fixture.rowsPerBlock) {
            for base in stride(from: 0, to: fixture.inputDimension, by: fixture.tileElements) {
                for rowInBlock in 0..<fixture.rowsPerBlock {
                    let row = rowBlock + rowInBlock
                    for column in 0..<fixture.tileElements {
                        let expected = fixture.sourceElements[row * fixture.inputDimension + base + column]
                        #expect(
                            packed[packedIndex] == expected,
                            "Packed element mismatch at packed[\(packedIndex)] row=\(row) col=\(base + column)"
                        )
                        packedIndex += 1
                    }
                }
            }
        }
    }

    @Test("blockedRows8Tiles128 shader pairwise indexing matches packed buffer")
    func blockedRows8Tiles128PairwiseShaderAddressingMatchesPackedData() throws {
        let fixture = try makeBlockedRowsFixture()
        let packed = fixture.packedElements
        let tilePairs = fixture.tileElements / 2
        let pairCount = 2
        let pairsPerRowBlock = fixture.inputDimension * fixture.rowsPerBlock / 2

        for gid in 0..<(fixture.outputDimension / fixture.rowsPerBlock) {
            for tile in 0..<(fixture.inputDimension / fixture.tileElements) {
                for rowInBlock in 0..<fixture.rowsPerBlock {
                    let row = gid * fixture.rowsPerBlock + rowInBlock
                    for tiisg in 0..<32 {
                        for pair in 0..<pairCount {
                            let packedPairIndex =
                                gid * pairsPerRowBlock +
                                tile * fixture.rowsPerBlock * tilePairs +
                                rowInBlock * tilePairs +
                                tiisg * pairCount +
                                pair
                            let packedElementIndex = packedPairIndex * 2
                            let column = tile * fixture.tileElements + (tiisg * pairCount + pair) * 2

                            #expect(
                                packed[packedElementIndex] == fixture.sourceElements[row * fixture.inputDimension + column],
                                "Packed low element mismatch gid=\(gid) tile=\(tile) row=\(row) col=\(column)"
                            )
                            #expect(
                                packed[packedElementIndex + 1] == fixture.sourceElements[row * fixture.inputDimension + column + 1],
                                "Packed high element mismatch gid=\(gid) tile=\(tile) row=\(row) col=\(column + 1)"
                            )
                        }
                    }
                }
            }
        }
    }

    @Test("blockedRows8Tiles128 BF16 kernel matches row-major on synthetic tensor")
    func blockedRows8Tiles128BFloatKernelMatchesRowMajorOnSyntheticTensor() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }
        guard let queue = device.makeCommandQueue() else {
            throw MetalCompilerError.deviceSetupFailed("No command queue")
        }

        let outputDimension = 6_144
        let inputDimension = 2_048
        let tensorName = "synthetic.weight"
        let weightBits = (0..<(outputDimension * inputDimension)).map { index in
            encodeBFloat16(Float((index % 37) - 18) / 32.0)
        }
        var mutableWeights = weightBits
        guard let weightBuffer = device.makeBuffer(
            bytes: &mutableWeights,
            length: weightBits.count * MemoryLayout<UInt16>.size,
            options: .storageModeShared
        ) else {
            throw MetalCompilerError.deviceSetupFailed("Failed to allocate synthetic weight buffer")
        }
        requestResidencyIfAvailable(device: device, buffer: weightBuffer)

        let entry = STAFTensorEntry(
            name: tensorName,
            payloadOffset: 0,
            payloadSize: weightBits.count * MemoryLayout<UInt16>.size,
            schemeIdentifier: .bf16RowMajor,
            semanticRole: .unknown,
            shape: [outputDimension, inputDimension],
            blockSize: 0,
            groupSize: 0,
            bufferOffset: 0
        )
        let store = STAFWeightStore(
            buffer: weightBuffer,
            entries: [tensorName: entry],
            metadata: .empty,
            specializedBufferAccesses: [:]
        )
        let builder = STAFSpecializedWeightStoreBuilder(device: device)
        let rowMajorAccess = try requireAccess(store.bufferAccess(for: tensorName, layout: .rowMajor), message: "Missing synthetic row-major access")
        let blockedAccess = try builder.makeBlockedRows8Tiles128Access(for: tensorName, store: store)

        let source = MetalSourceGenerator.commonHeader + "\n" + [
            MetalSourceGenerator.generateInput2048GEMVArgumentTableVariant(
                name: "test_rowmajor_synthetic_bf16_argbuf",
                argumentBufferIndex: 30,
                bufferPrecision: .float16,
                weightFormat: .bfloat16,
                fixedOutputDimension: outputDimension,
                includesDimensionBindings: false,
                fixedRowsPerThreadgroup: 8,
                stagesInputAsFloat: false,
                weightLayoutPolicy: .rowMajor,
                bf16ArgumentReadPolicy: .pairwisePointerInput,
                unrollFactor: 4
            ),
            MetalSourceGenerator.generateInput2048GEMVArgumentTableVariant(
                name: "test_blocked_synthetic_bf16_argbuf",
                argumentBufferIndex: 30,
                bufferPrecision: .float16,
                weightFormat: .bfloat16,
                fixedOutputDimension: outputDimension,
                includesDimensionBindings: false,
                fixedRowsPerThreadgroup: 8,
                stagesInputAsFloat: false,
                weightLayoutPolicy: .blockedRows8Tiles128,
                bf16ArgumentReadPolicy: .pairwisePointerInput,
                unrollFactor: 4
            ),
        ].joined(separator: "\n")

        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let rowMajorFunction = try requireAccess(library.makeFunction(name: "test_rowmajor_synthetic_bf16_argbuf"), message: "Missing synthetic row-major kernel")
        let blockedFunction = try requireAccess(library.makeFunction(name: "test_blocked_synthetic_bf16_argbuf"), message: "Missing synthetic blocked kernel")
        let rowMajorPipeline = try device.makeComputePipelineState(function: rowMajorFunction)
        let blockedPipeline = try device.makeComputePipelineState(function: blockedFunction)

        let inputValues = (0..<inputDimension).map { index in
            Float16(Float((index % 29) - 14) / 16.0)
        }
        var mutableInput = inputValues
        guard let inputBuffer = device.makeBuffer(
            bytes: &mutableInput,
            length: inputValues.count * MemoryLayout<Float16>.size,
            options: .storageModeShared
        ) else {
            throw MetalCompilerError.deviceSetupFailed("Failed to allocate synthetic input buffer")
        }
        guard let rowMajorOutput = device.makeBuffer(
            length: outputDimension * MemoryLayout<Float16>.size,
            options: .storageModeShared
        ), let blockedOutput = device.makeBuffer(
            length: outputDimension * MemoryLayout<Float16>.size,
            options: .storageModeShared
        ) else {
            throw MetalCompilerError.deviceSetupFailed("Failed to allocate synthetic output buffers")
        }

        try encodeAndRunArgumentKernel(
            pipeline: rowMajorPipeline,
            function: rowMajorFunction,
            queue: queue,
            groups: outputDimension / 8,
            inputBuffer: inputBuffer,
            weightAccess: rowMajorAccess,
            outputBuffer: rowMajorOutput
        )
        try encodeAndRunArgumentKernel(
            pipeline: blockedPipeline,
            function: blockedFunction,
            queue: queue,
            groups: outputDimension / 8,
            inputBuffer: inputBuffer,
            weightAccess: blockedAccess,
            outputBuffer: blockedOutput
        )

        let rowMajorValues = Array(UnsafeBufferPointer(
            start: rowMajorOutput.contents().bindMemory(to: Float16.self, capacity: outputDimension),
            count: outputDimension
        ))
        let blockedValues = Array(UnsafeBufferPointer(
            start: blockedOutput.contents().bindMemory(to: Float16.self, capacity: outputDimension),
            count: outputDimension
        ))

        for index in rowMajorValues.indices {
            #expect(
                rowMajorValues[index] == blockedValues[index],
                "Synthetic row-major/blocked mismatch at output[\(index)]"
            )
        }
    }

    @Test("blockedRows8Tiles128 packs real 2048->6144 BF16 tensor bytes as expected")
    func blockedRows8Tiles128PacksRealTensorBytesAsExpected() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }

        let stafURL = URL(fileURLWithPath: Self.realModelSTAFPath)
        guard FileManager.default.fileExists(atPath: stafURL.path) else {
            Issue.record("Missing STAF fixture at \(stafURL.path)")
            return
        }

        let store = try Self.requireRealModelStore()
        guard let tensorName = store.entries.values
            .filter({ $0.shape == [6_144, 2_048] && $0.schemeIdentifier == .bf16RowMajor })
            .map(\.name)
            .sorted()
            .first else {
            Issue.record("No BF16 2048->6144 tensor found in STAF fixture")
            return
        }

        guard let rowMajorAccess = store.bufferAccess(for: tensorName, layout: .rowMajor) else {
            Issue.record("Missing row-major access for \(tensorName)")
            return
        }

        let builder = STAFSpecializedWeightStoreBuilder(device: device)
        let blockedAccess = try builder.makeBlockedRows8Tiles128Access(for: tensorName, store: store)
        let rowMajorElements = Array(UnsafeBufferPointer(
            start: rowMajorAccess.buffer.contents()
                .advanced(by: rowMajorAccess.offset)
                .assumingMemoryBound(to: UInt16.self),
            count: 6_144 * 2_048
        ))
        let blockedElements = Array(UnsafeBufferPointer(
            start: blockedAccess.buffer.contents().assumingMemoryBound(to: UInt16.self),
            count: 6_144 * 2_048
        ))

        var packedIndex = 0
        for rowBlock in stride(from: 0, to: 6_144, by: 8) {
            for base in stride(from: 0, to: 2_048, by: 128) {
                for rowInBlock in 0..<8 {
                    let row = rowBlock + rowInBlock
                    for column in 0..<128 {
                        let expected = rowMajorElements[row * 2_048 + base + column]
                        #expect(
                            blockedElements[packedIndex] == expected,
                            "Packed real tensor mismatch at packed[\(packedIndex)] tensor=\(tensorName)"
                        )
                        packedIndex += 1
                    }
                }
            }
        }
    }

    @Test("blockedRows8Tiles128 BF16 kernel matches row-major on real hot tensor")
    func blockedRows8Tiles128BFloatKernelMatchesRowMajorOnRealHotTensor() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }
        guard let queue = device.makeCommandQueue() else {
            throw MetalCompilerError.deviceSetupFailed("No command queue")
        }

        let stafURL = URL(fileURLWithPath: Self.realModelSTAFPath)
        guard FileManager.default.fileExists(atPath: stafURL.path) else {
            Issue.record("Missing STAF fixture at \(stafURL.path)")
            return
        }

        let tensorName = "model.layers.0.conv.in_proj.weight"
        let outputDimension = 6_144
        let inputDimension = 2_048

        let store = try Self.requireRealModelStore()
        let builder = STAFSpecializedWeightStoreBuilder(device: device)
        let rowMajorAccess = try requireAccess(
            store.bufferAccess(for: tensorName, layout: .rowMajor),
            message: "Missing row-major access for \(tensorName)"
        )
        let blockedAccess = try builder.makeBlockedRows8Tiles128Access(for: tensorName, store: store)

        let source = MetalSourceGenerator.commonHeader + "\n" + [
            MetalSourceGenerator.generateInput2048GEMVArgumentTableVariant(
                name: "test_rowmajor_real_bf16_argbuf",
                argumentBufferIndex: 30,
                bufferPrecision: .float16,
                weightFormat: .bfloat16,
                fixedOutputDimension: outputDimension,
                includesDimensionBindings: false,
                fixedRowsPerThreadgroup: 8,
                stagesInputAsFloat: false,
                weightLayoutPolicy: .rowMajor,
                bf16ArgumentReadPolicy: .pairwisePointerInput,
                unrollFactor: 4
            ),
            MetalSourceGenerator.generateInput2048GEMVArgumentTableVariant(
                name: "test_blocked_real_bf16_argbuf",
                argumentBufferIndex: 30,
                bufferPrecision: .float16,
                weightFormat: .bfloat16,
                fixedOutputDimension: outputDimension,
                includesDimensionBindings: false,
                fixedRowsPerThreadgroup: 8,
                stagesInputAsFloat: false,
                weightLayoutPolicy: .blockedRows8Tiles128,
                bf16ArgumentReadPolicy: .pairwisePointerInput,
                unrollFactor: 4
            ),
        ].joined(separator: "\n")

        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let rowMajorFunction = try requireAccess(
            library.makeFunction(name: "test_rowmajor_real_bf16_argbuf"),
            message: "Missing real row-major kernel"
        )
        let blockedFunction = try requireAccess(
            library.makeFunction(name: "test_blocked_real_bf16_argbuf"),
            message: "Missing real blocked kernel"
        )
        let rowMajorPipeline = try device.makeComputePipelineState(function: rowMajorFunction)
        let blockedPipeline = try device.makeComputePipelineState(function: blockedFunction)

        let inputValues = (0..<inputDimension).map { index in
            Float16(Float((index % 29) - 14) / 16.0)
        }
        var mutableInput = inputValues
        guard let inputBuffer = device.makeBuffer(
            bytes: &mutableInput,
            length: inputValues.count * MemoryLayout<Float16>.size,
            options: .storageModeShared
        ) else {
            throw MetalCompilerError.deviceSetupFailed("Failed to allocate real input buffer")
        }
        guard let rowMajorOutput = device.makeBuffer(
            length: outputDimension * MemoryLayout<Float16>.size,
            options: .storageModeShared
        ), let blockedOutput = device.makeBuffer(
            length: outputDimension * MemoryLayout<Float16>.size,
            options: .storageModeShared
        ) else {
            throw MetalCompilerError.deviceSetupFailed("Failed to allocate real output buffers")
        }

        try encodeAndRunArgumentKernel(
            pipeline: rowMajorPipeline,
            function: rowMajorFunction,
            queue: queue,
            groups: outputDimension / 8,
            inputBuffer: inputBuffer,
            weightAccess: rowMajorAccess,
            outputBuffer: rowMajorOutput
        )
        try encodeAndRunArgumentKernel(
            pipeline: blockedPipeline,
            function: blockedFunction,
            queue: queue,
            groups: outputDimension / 8,
            inputBuffer: inputBuffer,
            weightAccess: blockedAccess,
            outputBuffer: blockedOutput
        )

        let rowMajorValues = Array(UnsafeBufferPointer(
            start: rowMajorOutput.contents().bindMemory(to: Float16.self, capacity: outputDimension),
            count: outputDimension
        ))
        let blockedValues = Array(UnsafeBufferPointer(
            start: blockedOutput.contents().bindMemory(to: Float16.self, capacity: outputDimension),
            count: outputDimension
        ))

        for index in rowMajorValues.indices {
            #expect(
                rowMajorValues[index] == blockedValues[index],
                "Real row-major/blocked mismatch at output[\(index)] tensor=\(tensorName)"
            )
        }
    }

    @Test("blockedRows8Tiles128 conv update chain matches row-major on all real hot tensors")
    func blockedRows8Tiles128ConvUpdateChainMatchesRowMajorOnAllRealHotTensors() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }
        guard let queue = device.makeCommandQueue() else {
            throw MetalCompilerError.deviceSetupFailed("No command queue")
        }

        let stafURL = URL(fileURLWithPath: Self.realModelSTAFPath)
        guard FileManager.default.fileExists(atPath: stafURL.path) else {
            Issue.record("Missing STAF fixture at \(stafURL.path)")
            return
        }

        let inputDimension = 2_048
        let outputDimension = 6_144

        let store = try Self.requireRealModelStore()
        let builder = STAFSpecializedWeightStoreBuilder(device: device)

        let source = MetalSourceGenerator.commonHeader + "\n" + [
            MetalSourceGenerator.generateInput2048GEMVArgumentTableVariant(
                name: "test_rowmajor_chain_bf16_argbuf",
                argumentBufferIndex: 30,
                bufferPrecision: .float16,
                weightFormat: .bfloat16,
                fixedOutputDimension: outputDimension,
                includesDimensionBindings: false,
                fixedRowsPerThreadgroup: 8,
                stagesInputAsFloat: false,
                weightLayoutPolicy: .rowMajor,
                bf16ArgumentReadPolicy: .pairwisePointerInput,
                unrollFactor: 4
            ),
            MetalSourceGenerator.generateInput2048GEMVArgumentTableVariant(
                name: "test_blocked_chain_bf16_argbuf",
                argumentBufferIndex: 30,
                bufferPrecision: .float16,
                weightFormat: .bfloat16,
                fixedOutputDimension: outputDimension,
                includesDimensionBindings: false,
                fixedRowsPerThreadgroup: 8,
                stagesInputAsFloat: false,
                weightLayoutPolicy: .blockedRows8Tiles128,
                bf16ArgumentReadPolicy: .pairwisePointerInput,
                unrollFactor: 4
            ),
            MetalSourceGenerator.generateConvStateUpdateArgumentTableVariant(
                name: "test_conv_state_update_bf16_argbuf",
                argumentBufferIndex: 30,
                bufferPrecision: .float16,
                weightFormat: .bfloat16
            ),
        ].joined(separator: "\n")

        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let rowMajorFunction = try requireAccess(
            library.makeFunction(name: "test_rowmajor_chain_bf16_argbuf"),
            message: "Missing chain row-major kernel"
        )
        let blockedFunction = try requireAccess(
            library.makeFunction(name: "test_blocked_chain_bf16_argbuf"),
            message: "Missing chain blocked kernel"
        )
        let convFunction = try requireAccess(
            library.makeFunction(name: "test_conv_state_update_bf16_argbuf"),
            message: "Missing conv-state-update chain kernel"
        )
        let rowMajorPipeline = try device.makeComputePipelineState(function: rowMajorFunction)
        let blockedPipeline = try device.makeComputePipelineState(function: blockedFunction)
        let convPipeline = try device.makeComputePipelineState(function: convFunction)

        let inputValues = (0..<inputDimension).map { index in
            Float16(Float((index % 29) - 14) / 16.0)
        }
        var mutableInput = inputValues
        guard let inputBuffer = device.makeBuffer(
            bytes: &mutableInput,
            length: inputValues.count * MemoryLayout<Float16>.size,
            options: .storageModeShared
        ) else {
            throw MetalCompilerError.deviceSetupFailed("Failed to allocate chain input buffer")
        }

        for tensorName in Self.realHot2048To6144TensorNames {
            let convWeightName = tensorName.replacingOccurrences(of: "in_proj.weight", with: "conv.weight")
            let rowMajorAccess = try requireAccess(
                store.bufferAccess(for: tensorName, layout: .rowMajor),
                message: "Missing row-major access for \(tensorName)"
            )
            let blockedAccess = try builder.makeBlockedRows8Tiles128Access(for: tensorName, store: store)
            let convWeightAccess = try requireAccess(
                store.bufferAccess(for: convWeightName, layout: .rowMajor),
                message: "Missing conv weight access for \(convWeightName)"
            )
            let convEntry = try requireAccess(
                store.entries[convWeightName],
                message: "Missing conv entry for \(convWeightName)"
            )
            guard convEntry.shape.count >= 2 else {
                throw MetalCompilerError.deviceSetupFailed("Unexpected conv shape for \(convWeightName): \(convEntry.shape)")
            }
            let convDimension = convEntry.shape[0]
            let kernelSize = convEntry.shape[1]
            let convStateCount = convDimension * kernelSize
            let convStateValues = (0..<convStateCount).map { index in
                Float16(Float((index % 17) - 8) / 32.0)
            }
            var mutableRowMajorConvState = convStateValues
            var mutableBlockedConvState = convStateValues

            guard
                let rowMajorGEMVOutput = device.makeBuffer(
                    length: outputDimension * MemoryLayout<Float16>.size,
                    options: .storageModeShared
                ),
                let blockedGEMVOutput = device.makeBuffer(
                    length: outputDimension * MemoryLayout<Float16>.size,
                    options: .storageModeShared
                ),
                let rowMajorConvState = device.makeBuffer(
                    bytes: &mutableRowMajorConvState,
                    length: convStateCount * MemoryLayout<Float16>.size,
                    options: .storageModeShared
                ),
                let blockedConvState = device.makeBuffer(
                    bytes: &mutableBlockedConvState,
                    length: convStateCount * MemoryLayout<Float16>.size,
                    options: .storageModeShared
                ),
                let rowMajorOutput = device.makeBuffer(
                    length: convDimension * MemoryLayout<Float16>.size,
                    options: .storageModeShared
                ),
                let blockedOutput = device.makeBuffer(
                    length: convDimension * MemoryLayout<Float16>.size,
                    options: .storageModeShared
                )
            else {
                throw MetalCompilerError.deviceSetupFailed("Failed to allocate chain buffers for \(tensorName)")
            }

            try encodeAndRunArgumentKernel(
                pipeline: rowMajorPipeline,
                function: rowMajorFunction,
                queue: queue,
                groups: outputDimension / 8,
                inputBuffer: inputBuffer,
                weightAccess: rowMajorAccess,
                outputBuffer: rowMajorGEMVOutput
            )
            try encodeAndRunArgumentKernel(
                pipeline: blockedPipeline,
                function: blockedFunction,
                queue: queue,
                groups: outputDimension / 8,
                inputBuffer: inputBuffer,
                weightAccess: blockedAccess,
                outputBuffer: blockedGEMVOutput
            )

            try encodeAndRunConvStateUpdateArgumentKernel(
                pipeline: convPipeline,
                function: convFunction,
                queue: queue,
                convStateBuffer: rowMajorConvState,
                inProjOutputBuffer: rowMajorGEMVOutput,
                weightAccess: convWeightAccess,
                outputBuffer: rowMajorOutput,
                dimension: convDimension,
                kernelSize: kernelSize
            )
            try encodeAndRunConvStateUpdateArgumentKernel(
                pipeline: convPipeline,
                function: convFunction,
                queue: queue,
                convStateBuffer: blockedConvState,
                inProjOutputBuffer: blockedGEMVOutput,
                weightAccess: convWeightAccess,
                outputBuffer: blockedOutput,
                dimension: convDimension,
                kernelSize: kernelSize
            )

            let rowMajorOutputValues = Array(UnsafeBufferPointer(
                start: rowMajorOutput.contents().bindMemory(to: Float16.self, capacity: convDimension),
                count: convDimension
            ))
            let blockedOutputValues = Array(UnsafeBufferPointer(
                start: blockedOutput.contents().bindMemory(to: Float16.self, capacity: convDimension),
                count: convDimension
            ))
            let rowMajorConvStateValues = Array(UnsafeBufferPointer(
                start: rowMajorConvState.contents().bindMemory(to: Float16.self, capacity: convStateCount),
                count: convStateCount
            ))
            let blockedConvStateValues = Array(UnsafeBufferPointer(
                start: blockedConvState.contents().bindMemory(to: Float16.self, capacity: convStateCount),
                count: convStateCount
            ))

            for index in rowMajorOutputValues.indices {
                #expect(
                    rowMajorOutputValues[index] == blockedOutputValues[index],
                    "Conv-update output mismatch at output[\(index)] tensor=\(tensorName)"
                )
            }
            for index in rowMajorConvStateValues.indices {
                #expect(
                    rowMajorConvStateValues[index] == blockedConvStateValues[index],
                    "Conv-state mismatch at state[\(index)] tensor=\(tensorName)"
                )
            }
        }
    }

    @Test("blockedRows8Tiles128 full conv operator chain matches row-major on all real hot tensors")
    func blockedRows8Tiles128FullConvOperatorChainMatchesRowMajorOnAllRealHotTensors() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }
        guard let queue = device.makeCommandQueue() else {
            throw MetalCompilerError.deviceSetupFailed("No command queue")
        }

        let stafURL = URL(fileURLWithPath: Self.realModelSTAFPath)
        guard FileManager.default.fileExists(atPath: stafURL.path) else {
            Issue.record("Missing STAF fixture at \(stafURL.path)")
            return
        }

        let inputDimension = 2_048
        let inProjOutputDimension = 6_144
        let outProjOutputDimension = 2_048
        let outProjPolicy = Input2048GEMVSourcePolicy.square(weightFormat: .bfloat16)

        let store = try Self.requireRealModelStore()
        let builder = STAFSpecializedWeightStoreBuilder(device: device)

        let source = MetalSourceGenerator.commonHeader + "\n" + [
            MetalSourceGenerator.generateInput2048GEMVArgumentTableVariant(
                name: "test_rowmajor_full_conv_bf16_argbuf",
                argumentBufferIndex: 30,
                bufferPrecision: .float16,
                weightFormat: .bfloat16,
                fixedOutputDimension: inProjOutputDimension,
                includesDimensionBindings: false,
                fixedRowsPerThreadgroup: 8,
                stagesInputAsFloat: false,
                weightLayoutPolicy: .rowMajor,
                bf16ArgumentReadPolicy: .pairwisePointerInput,
                unrollFactor: 4
            ),
            MetalSourceGenerator.generateInput2048GEMVArgumentTableVariant(
                name: "test_blocked_full_conv_bf16_argbuf",
                argumentBufferIndex: 30,
                bufferPrecision: .float16,
                weightFormat: .bfloat16,
                fixedOutputDimension: inProjOutputDimension,
                includesDimensionBindings: false,
                fixedRowsPerThreadgroup: 8,
                stagesInputAsFloat: false,
                weightLayoutPolicy: .blockedRows8Tiles128,
                bf16ArgumentReadPolicy: .pairwisePointerInput,
                unrollFactor: 4
            ),
            MetalSourceGenerator.generateConvStateUpdateArgumentTableVariant(
                name: "test_full_conv_state_update_bf16_argbuf",
                argumentBufferIndex: 30,
                bufferPrecision: .float16,
                weightFormat: .bfloat16
            ),
            MetalSourceGenerator.generateInput2048GEMVArgumentTableVariant(
                name: "test_full_conv_out_proj_bf16_argbuf",
                argumentBufferIndex: 30,
                bufferPrecision: .float16,
                weightFormat: .bfloat16,
                fixedOutputDimension: outProjOutputDimension,
                includesDimensionBindings: false,
                fixedRowsPerThreadgroup: outProjPolicy.fixedRowsPerThreadgroup,
                stagesInputAsFloat: outProjPolicy.stagesInputAsFloat,
                weightLayoutPolicy: .rowMajor,
                bf16ArgumentReadPolicy: outProjPolicy.bf16ArgumentReadPolicy,
                unrollFactor: outProjPolicy.unrollFactor
            ),
        ].joined(separator: "\n")

        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let rowMajorFunction = try requireAccess(
            library.makeFunction(name: "test_rowmajor_full_conv_bf16_argbuf"),
            message: "Missing full-conv row-major kernel"
        )
        let blockedFunction = try requireAccess(
            library.makeFunction(name: "test_blocked_full_conv_bf16_argbuf"),
            message: "Missing full-conv blocked kernel"
        )
        let convFunction = try requireAccess(
            library.makeFunction(name: "test_full_conv_state_update_bf16_argbuf"),
            message: "Missing full-conv conv-update kernel"
        )
        let outProjFunction = try requireAccess(
            library.makeFunction(name: "test_full_conv_out_proj_bf16_argbuf"),
            message: "Missing full-conv out-proj kernel"
        )
        let rowMajorPipeline = try device.makeComputePipelineState(function: rowMajorFunction)
        let blockedPipeline = try device.makeComputePipelineState(function: blockedFunction)
        let convPipeline = try device.makeComputePipelineState(function: convFunction)
        let outProjPipeline = try device.makeComputePipelineState(function: outProjFunction)

        let inputValues = (0..<inputDimension).map { index in
            Float16(Float((index % 29) - 14) / 16.0)
        }
        var mutableInput = inputValues
        guard let inputBuffer = device.makeBuffer(
            bytes: &mutableInput,
            length: inputValues.count * MemoryLayout<Float16>.size,
            options: .storageModeShared
        ) else {
            throw MetalCompilerError.deviceSetupFailed("Failed to allocate full-conv input buffer")
        }

        for tensorName in Self.realHot2048To6144TensorNames {
            let convWeightName = tensorName.replacingOccurrences(of: "in_proj.weight", with: "conv.weight")
            let outProjWeightName = tensorName.replacingOccurrences(of: "in_proj.weight", with: "out_proj.weight")
            let rowMajorAccess = try requireAccess(
                store.bufferAccess(for: tensorName, layout: .rowMajor),
                message: "Missing row-major access for \(tensorName)"
            )
            let blockedAccess = try builder.makeBlockedRows8Tiles128Access(for: tensorName, store: store)
            let convWeightAccess = try requireAccess(
                store.bufferAccess(for: convWeightName, layout: .rowMajor),
                message: "Missing conv weight access for \(convWeightName)"
            )
            let outProjWeightAccess = try requireAccess(
                store.bufferAccess(for: outProjWeightName, layout: .rowMajor),
                message: "Missing out-proj weight access for \(outProjWeightName)"
            )
            let convEntry = try requireAccess(
                store.entries[convWeightName],
                message: "Missing conv entry for \(convWeightName)"
            )
            guard convEntry.shape.count >= 2 else {
                throw MetalCompilerError.deviceSetupFailed("Unexpected conv shape for \(convWeightName): \(convEntry.shape)")
            }
            let convDimension = convEntry.shape[0]
            let kernelSize = convEntry.shape[1]
            let convStateCount = convDimension * kernelSize
            let convStateValues = (0..<convStateCount).map { index in
                Float16(Float((index % 17) - 8) / 32.0)
            }
            var mutableRowMajorConvState = convStateValues
            var mutableBlockedConvState = convStateValues

            guard
                let rowMajorGEMVOutput = device.makeBuffer(
                    length: inProjOutputDimension * MemoryLayout<Float16>.size,
                    options: .storageModeShared
                ),
                let blockedGEMVOutput = device.makeBuffer(
                    length: inProjOutputDimension * MemoryLayout<Float16>.size,
                    options: .storageModeShared
                ),
                let rowMajorConvState = device.makeBuffer(
                    bytes: &mutableRowMajorConvState,
                    length: convStateCount * MemoryLayout<Float16>.size,
                    options: .storageModeShared
                ),
                let blockedConvState = device.makeBuffer(
                    bytes: &mutableBlockedConvState,
                    length: convStateCount * MemoryLayout<Float16>.size,
                    options: .storageModeShared
                ),
                let rowMajorConvOutput = device.makeBuffer(
                    length: convDimension * MemoryLayout<Float16>.size,
                    options: .storageModeShared
                ),
                let blockedConvOutput = device.makeBuffer(
                    length: convDimension * MemoryLayout<Float16>.size,
                    options: .storageModeShared
                ),
                let rowMajorFinalOutput = device.makeBuffer(
                    length: outProjOutputDimension * MemoryLayout<Float16>.size,
                    options: .storageModeShared
                ),
                let blockedFinalOutput = device.makeBuffer(
                    length: outProjOutputDimension * MemoryLayout<Float16>.size,
                    options: .storageModeShared
                )
            else {
                throw MetalCompilerError.deviceSetupFailed("Failed to allocate full-conv buffers for \(tensorName)")
            }

            try encodeAndRunArgumentKernel(
                pipeline: rowMajorPipeline,
                function: rowMajorFunction,
                queue: queue,
                groups: inProjOutputDimension / 8,
                inputBuffer: inputBuffer,
                weightAccess: rowMajorAccess,
                outputBuffer: rowMajorGEMVOutput
            )
            try encodeAndRunArgumentKernel(
                pipeline: blockedPipeline,
                function: blockedFunction,
                queue: queue,
                groups: inProjOutputDimension / 8,
                inputBuffer: inputBuffer,
                weightAccess: blockedAccess,
                outputBuffer: blockedGEMVOutput
            )

            try encodeAndRunConvStateUpdateArgumentKernel(
                pipeline: convPipeline,
                function: convFunction,
                queue: queue,
                convStateBuffer: rowMajorConvState,
                inProjOutputBuffer: rowMajorGEMVOutput,
                weightAccess: convWeightAccess,
                outputBuffer: rowMajorConvOutput,
                dimension: convDimension,
                kernelSize: kernelSize
            )
            try encodeAndRunConvStateUpdateArgumentKernel(
                pipeline: convPipeline,
                function: convFunction,
                queue: queue,
                convStateBuffer: blockedConvState,
                inProjOutputBuffer: blockedGEMVOutput,
                weightAccess: convWeightAccess,
                outputBuffer: blockedConvOutput,
                dimension: convDimension,
                kernelSize: kernelSize
            )

            try encodeAndRunArgumentKernel(
                pipeline: outProjPipeline,
                function: outProjFunction,
                queue: queue,
                groups: outProjOutputDimension / 8,
                inputBuffer: rowMajorConvOutput,
                weightAccess: outProjWeightAccess,
                outputBuffer: rowMajorFinalOutput
            )
            try encodeAndRunArgumentKernel(
                pipeline: outProjPipeline,
                function: outProjFunction,
                queue: queue,
                groups: outProjOutputDimension / 8,
                inputBuffer: blockedConvOutput,
                weightAccess: outProjWeightAccess,
                outputBuffer: blockedFinalOutput
            )

            let rowMajorFinalValues = Array(UnsafeBufferPointer(
                start: rowMajorFinalOutput.contents().bindMemory(to: Float16.self, capacity: outProjOutputDimension),
                count: outProjOutputDimension
            ))
            let blockedFinalValues = Array(UnsafeBufferPointer(
                start: blockedFinalOutput.contents().bindMemory(to: Float16.self, capacity: outProjOutputDimension),
                count: outProjOutputDimension
            ))

            for index in rowMajorFinalValues.indices {
                #expect(
                    rowMajorFinalValues[index] == blockedFinalValues[index],
                    "Full conv chain mismatch at output[\(index)] tensor=\(tensorName)"
                )
            }
        }
    }

    @Test("specialized decode store keeps full conv operator chain matching row-major")
    func specializedDecodeStoreKeepsFullConvOperatorChainMatchingRowMajor() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }
        guard let queue = device.makeCommandQueue() else {
            throw MetalCompilerError.deviceSetupFailed("No command queue")
        }

        let stafURL = URL(fileURLWithPath: Self.realModelSTAFPath)
        guard FileManager.default.fileExists(atPath: stafURL.path) else {
            Issue.record("Missing STAF fixture at \(stafURL.path)")
            return
        }

        let inputDimension = 2_048
        let inProjOutputDimension = 6_144
        let outProjOutputDimension = 2_048
        let outProjPolicy = Input2048GEMVSourcePolicy.square(weightFormat: .bfloat16)

        let baseStore = try Self.requireRealModelStore()
        let builder = STAFSpecializedWeightStoreBuilder(device: device)
        let specializedStore = try Self.realHot2048To6144TensorNames.reduce(baseStore) { store, tensorName in
            let request = STAFWeightAccessRequest(
                tensorName: tensorName,
                executionPhase: .decode,
                layoutPreference: .optimized(.blockedRows8Tiles128)
            )
            let access = try builder.makeSpecializedAccess(for: request, store: store)
            return store.registeringSpecializedBufferAccess(access, for: request)
        }

        let source = MetalSourceGenerator.commonHeader + "\n" + [
            MetalSourceGenerator.generateInput2048GEMVArgumentTableVariant(
                name: "test_specialized_store_rowmajor_bf16_argbuf",
                argumentBufferIndex: 30,
                bufferPrecision: .float16,
                weightFormat: .bfloat16,
                fixedOutputDimension: inProjOutputDimension,
                includesDimensionBindings: false,
                fixedRowsPerThreadgroup: 8,
                stagesInputAsFloat: false,
                weightLayoutPolicy: .rowMajor,
                bf16ArgumentReadPolicy: .pairwisePointerInput,
                unrollFactor: 4
            ),
            MetalSourceGenerator.generateInput2048GEMVArgumentTableVariant(
                name: "test_specialized_store_blocked_bf16_argbuf",
                argumentBufferIndex: 30,
                bufferPrecision: .float16,
                weightFormat: .bfloat16,
                fixedOutputDimension: inProjOutputDimension,
                includesDimensionBindings: false,
                fixedRowsPerThreadgroup: 8,
                stagesInputAsFloat: false,
                weightLayoutPolicy: .blockedRows8Tiles128,
                bf16ArgumentReadPolicy: .pairwisePointerInput,
                unrollFactor: 4
            ),
            MetalSourceGenerator.generateConvStateUpdateArgumentTableVariant(
                name: "test_specialized_store_conv_state_update_bf16_argbuf",
                argumentBufferIndex: 30,
                bufferPrecision: .float16,
                weightFormat: .bfloat16
            ),
            MetalSourceGenerator.generateInput2048GEMVArgumentTableVariant(
                name: "test_specialized_store_out_proj_bf16_argbuf",
                argumentBufferIndex: 30,
                bufferPrecision: .float16,
                weightFormat: .bfloat16,
                fixedOutputDimension: outProjOutputDimension,
                includesDimensionBindings: false,
                fixedRowsPerThreadgroup: outProjPolicy.fixedRowsPerThreadgroup,
                stagesInputAsFloat: outProjPolicy.stagesInputAsFloat,
                weightLayoutPolicy: .rowMajor,
                bf16ArgumentReadPolicy: outProjPolicy.bf16ArgumentReadPolicy,
                unrollFactor: outProjPolicy.unrollFactor
            ),
        ].joined(separator: "\n")

        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let rowMajorFunction = try requireAccess(
            library.makeFunction(name: "test_specialized_store_rowmajor_bf16_argbuf"),
            message: "Missing specialized-store row-major kernel"
        )
        let blockedFunction = try requireAccess(
            library.makeFunction(name: "test_specialized_store_blocked_bf16_argbuf"),
            message: "Missing specialized-store blocked kernel"
        )
        let convFunction = try requireAccess(
            library.makeFunction(name: "test_specialized_store_conv_state_update_bf16_argbuf"),
            message: "Missing specialized-store conv-update kernel"
        )
        let outProjFunction = try requireAccess(
            library.makeFunction(name: "test_specialized_store_out_proj_bf16_argbuf"),
            message: "Missing specialized-store out-proj kernel"
        )
        let rowMajorPipeline = try device.makeComputePipelineState(function: rowMajorFunction)
        let blockedPipeline = try device.makeComputePipelineState(function: blockedFunction)
        let convPipeline = try device.makeComputePipelineState(function: convFunction)
        let outProjPipeline = try device.makeComputePipelineState(function: outProjFunction)

        let inputValues = (0..<inputDimension).map { index in
            Float16(Float((index % 29) - 14) / 16.0)
        }
        var mutableInput = inputValues
        guard let inputBuffer = device.makeBuffer(
            bytes: &mutableInput,
            length: inputValues.count * MemoryLayout<Float16>.size,
            options: .storageModeShared
        ) else {
            throw MetalCompilerError.deviceSetupFailed("Failed to allocate specialized-store input buffer")
        }

        for tensorName in Self.realHot2048To6144TensorNames {
            let convWeightName = tensorName.replacingOccurrences(of: "in_proj.weight", with: "conv.weight")
            let outProjWeightName = tensorName.replacingOccurrences(of: "in_proj.weight", with: "out_proj.weight")
            let rowMajorAccess = try requireAccess(
                baseStore.bufferAccess(for: tensorName, layout: .rowMajor),
                message: "Missing base row-major access for \(tensorName)"
            )
            let blockedAccess = try requireAccess(
                specializedStore.resolvedBufferAccess(for: STAFWeightAccessRequest(
                    tensorName: tensorName,
                    executionPhase: .decode,
                    layoutPreference: .optimized(.blockedRows8Tiles128)
                )),
                message: "Missing specialized decode access for \(tensorName)"
            )
            let convWeightAccess = try requireAccess(
                specializedStore.bufferAccess(for: convWeightName, layout: .rowMajor),
                message: "Missing specialized-store conv access for \(convWeightName)"
            )
            let outProjWeightAccess = try requireAccess(
                specializedStore.bufferAccess(for: outProjWeightName, layout: .rowMajor),
                message: "Missing specialized-store out-proj access for \(outProjWeightName)"
            )
            let convEntry = try requireAccess(
                specializedStore.entries[convWeightName],
                message: "Missing specialized-store conv entry for \(convWeightName)"
            )
            guard convEntry.shape.count >= 2 else {
                throw MetalCompilerError.deviceSetupFailed("Unexpected conv shape for \(convWeightName): \(convEntry.shape)")
            }
            let convDimension = convEntry.shape[0]
            let kernelSize = convEntry.shape[1]
            let convStateCount = convDimension * kernelSize
            let convStateValues = (0..<convStateCount).map { index in
                Float16(Float((index % 17) - 8) / 32.0)
            }
            var mutableRowMajorConvState = convStateValues
            var mutableBlockedConvState = convStateValues

            guard
                let rowMajorGEMVOutput = device.makeBuffer(length: inProjOutputDimension * MemoryLayout<Float16>.size, options: .storageModeShared),
                let blockedGEMVOutput = device.makeBuffer(length: inProjOutputDimension * MemoryLayout<Float16>.size, options: .storageModeShared),
                let rowMajorConvState = device.makeBuffer(bytes: &mutableRowMajorConvState, length: convStateCount * MemoryLayout<Float16>.size, options: .storageModeShared),
                let blockedConvState = device.makeBuffer(bytes: &mutableBlockedConvState, length: convStateCount * MemoryLayout<Float16>.size, options: .storageModeShared),
                let rowMajorConvOutput = device.makeBuffer(length: convDimension * MemoryLayout<Float16>.size, options: .storageModeShared),
                let blockedConvOutput = device.makeBuffer(length: convDimension * MemoryLayout<Float16>.size, options: .storageModeShared),
                let rowMajorFinalOutput = device.makeBuffer(length: outProjOutputDimension * MemoryLayout<Float16>.size, options: .storageModeShared),
                let blockedFinalOutput = device.makeBuffer(length: outProjOutputDimension * MemoryLayout<Float16>.size, options: .storageModeShared)
            else {
                throw MetalCompilerError.deviceSetupFailed("Failed to allocate specialized-store chain buffers for \(tensorName)")
            }

            try encodeAndRunArgumentKernel(
                pipeline: rowMajorPipeline,
                function: rowMajorFunction,
                queue: queue,
                groups: inProjOutputDimension / 8,
                inputBuffer: inputBuffer,
                weightAccess: rowMajorAccess,
                outputBuffer: rowMajorGEMVOutput
            )
            try encodeAndRunArgumentKernel(
                pipeline: blockedPipeline,
                function: blockedFunction,
                queue: queue,
                groups: inProjOutputDimension / 8,
                inputBuffer: inputBuffer,
                weightAccess: blockedAccess,
                outputBuffer: blockedGEMVOutput
            )

            try encodeAndRunConvStateUpdateArgumentKernel(
                pipeline: convPipeline,
                function: convFunction,
                queue: queue,
                convStateBuffer: rowMajorConvState,
                inProjOutputBuffer: rowMajorGEMVOutput,
                weightAccess: convWeightAccess,
                outputBuffer: rowMajorConvOutput,
                dimension: convDimension,
                kernelSize: kernelSize
            )
            try encodeAndRunConvStateUpdateArgumentKernel(
                pipeline: convPipeline,
                function: convFunction,
                queue: queue,
                convStateBuffer: blockedConvState,
                inProjOutputBuffer: blockedGEMVOutput,
                weightAccess: convWeightAccess,
                outputBuffer: blockedConvOutput,
                dimension: convDimension,
                kernelSize: kernelSize
            )

            try encodeAndRunArgumentKernel(
                pipeline: outProjPipeline,
                function: outProjFunction,
                queue: queue,
                groups: outProjOutputDimension / 8,
                inputBuffer: rowMajorConvOutput,
                weightAccess: outProjWeightAccess,
                outputBuffer: rowMajorFinalOutput
            )
            try encodeAndRunArgumentKernel(
                pipeline: outProjPipeline,
                function: outProjFunction,
                queue: queue,
                groups: outProjOutputDimension / 8,
                inputBuffer: blockedConvOutput,
                weightAccess: outProjWeightAccess,
                outputBuffer: blockedFinalOutput
            )

            let rowMajorFinalValues = Array(UnsafeBufferPointer(
                start: rowMajorFinalOutput.contents().bindMemory(to: Float16.self, capacity: outProjOutputDimension),
                count: outProjOutputDimension
            ))
            let blockedFinalValues = Array(UnsafeBufferPointer(
                start: blockedFinalOutput.contents().bindMemory(to: Float16.self, capacity: outProjOutputDimension),
                count: outProjOutputDimension
            ))

            for index in rowMajorFinalValues.indices {
                #expect(
                    rowMajorFinalValues[index] == blockedFinalValues[index],
                    "Specialized-store full chain mismatch at output[\(index)] tensor=\(tensorName)"
                )
            }
        }
    }

    @Test("blockedRows8Tiles128 GPU direct BF16 reads match packed real tensor")
    func blockedRows8Tiles128GPUDirectBFloatReadsMatchPackedRealTensor() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }
        guard let queue = device.makeCommandQueue() else {
            throw MetalCompilerError.deviceSetupFailed("No command queue")
        }

        let stafURL = URL(fileURLWithPath: Self.realModelSTAFPath)
        guard FileManager.default.fileExists(atPath: stafURL.path) else {
            Issue.record("Missing STAF fixture at \(stafURL.path)")
            return
        }

        let tensorName = "model.layers.0.conv.in_proj.weight"
        let store = try Self.requireRealModelStore()
        let builder = STAFSpecializedWeightStoreBuilder(device: device)
        let blockedAccess = try builder.makeBlockedRows8Tiles128Access(for: tensorName, store: store)
        let probePairCount = 256
        let probeElementCount = probePairCount * 2

        let source = MetalSourceGenerator.commonHeader + "\n" + """
        struct test_blocked_real_bf16_probe_args {
            device const half* input [[id(0)]];
            device const uint16_t* weight [[id(1)]];
            device half* output [[id(2)]];
        };

        kernel void test_blocked_real_bf16_probe(
            constant test_blocked_real_bf16_probe_args& args [[buffer(30)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= \(probePairCount)u) return;
            device const ushort2* packed = (device const ushort2*)args.weight;
            float2 value = bf16x2_to_float2(packed[gid]);
            args.output[gid * 2u] = half(value.x);
            args.output[gid * 2u + 1u] = half(value.y);
        }
        """

        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let function = try requireAccess(
            library.makeFunction(name: "test_blocked_real_bf16_probe"),
            message: "Missing blocked probe kernel"
        )
        let pipeline = try device.makeComputePipelineState(function: function)

        var dummyInput = [Float16](repeating: 0, count: 1)
        guard let inputBuffer = device.makeBuffer(
            bytes: &dummyInput,
            length: MemoryLayout<Float16>.size,
            options: .storageModeShared
        ), let outputBuffer = device.makeBuffer(
            length: probeElementCount * MemoryLayout<Float16>.size,
            options: .storageModeShared
        ) else {
            throw MetalCompilerError.deviceSetupFailed("Failed to allocate blocked probe buffers")
        }

        let argumentEncoder = function.makeArgumentEncoder(bufferIndex: 30)
        guard let argumentBuffer = device.makeBuffer(
            length: argumentEncoder.encodedLength,
            options: .storageModeShared
        ) else {
            throw MetalCompilerError.deviceSetupFailed("Failed to allocate blocked probe argument buffer")
        }

        argumentEncoder.setArgumentBuffer(argumentBuffer, offset: 0)
        argumentEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
        argumentEncoder.setBuffer(blockedAccess.buffer, offset: blockedAccess.offset, index: 1)
        argumentEncoder.setBuffer(outputBuffer, offset: 0, index: 2)

        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalCompilerError.deviceSetupFailed("Failed to encode blocked probe command")
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(argumentBuffer, offset: 0, index: 30)
        encoder.useResource(argumentBuffer, usage: .read)
        encoder.useResource(inputBuffer, usage: .read)
        encoder.useResource(blockedAccess.buffer, usage: .read)
        encoder.useResource(outputBuffer, usage: .write)
        encoder.dispatchThreads(
            MTLSize(width: probePairCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(probePairCount, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw MetalCompilerError.deviceSetupFailed("Blocked probe GPU error: \(error)")
        }

        let blockedBits = Array(UnsafeBufferPointer(
            start: blockedAccess.buffer.contents().assumingMemoryBound(to: UInt16.self),
            count: probeElementCount
        ))
        let gpuValues = Array(UnsafeBufferPointer(
            start: outputBuffer.contents().bindMemory(to: Float16.self, capacity: probeElementCount),
            count: probeElementCount
        ))

        for index in 0..<probeElementCount {
            #expect(
                gpuValues[index] == decodeBFloat16ToFloat16(blockedBits[index]),
                "Blocked probe mismatch at packed[\(index)] tensor=\(tensorName)"
            )
        }
    }

    @Test("blockedRows8Tiles128 direct buffer binding reads packed real tensor")
    func blockedRows8Tiles128DirectBufferBindingReadsPackedRealTensor() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }
        guard let queue = device.makeCommandQueue() else {
            throw MetalCompilerError.deviceSetupFailed("No command queue")
        }

        let stafURL = URL(fileURLWithPath: Self.realModelSTAFPath)
        guard FileManager.default.fileExists(atPath: stafURL.path) else {
            Issue.record("Missing STAF fixture at \(stafURL.path)")
            return
        }

        let tensorName = "model.layers.0.conv.in_proj.weight"
        let store = try Self.requireRealModelStore()
        let builder = STAFSpecializedWeightStoreBuilder(device: device)
        let blockedAccess = try builder.makeBlockedRows8Tiles128Access(for: tensorName, store: store)
        let probePairCount = 256
        let probeElementCount = probePairCount * 2

        let source = MetalSourceGenerator.commonHeader + "\n" + """
        kernel void test_blocked_real_bf16_direct(
            device const uint16_t* weight [[buffer(0)]],
            device half* output [[buffer(1)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= \(probePairCount)u) return;
            device const ushort2* packed = (device const ushort2*)weight;
            float2 value = bf16x2_to_float2(packed[gid]);
            output[gid * 2u] = half(value.x);
            output[gid * 2u + 1u] = half(value.y);
        }
        """

        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let function = try requireAccess(
            library.makeFunction(name: "test_blocked_real_bf16_direct"),
            message: "Missing blocked direct kernel"
        )
        let pipeline = try device.makeComputePipelineState(function: function)

        guard let outputBuffer = device.makeBuffer(
            length: probeElementCount * MemoryLayout<Float16>.size,
            options: .storageModeShared
        ), let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalCompilerError.deviceSetupFailed("Failed to allocate blocked direct probe resources")
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(blockedAccess.buffer, offset: blockedAccess.offset, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.dispatchThreads(
            MTLSize(width: probePairCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(probePairCount, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw MetalCompilerError.deviceSetupFailed("Blocked direct GPU error: \(error)")
        }

        let blockedBits = Array(UnsafeBufferPointer(
            start: blockedAccess.buffer.contents().assumingMemoryBound(to: UInt16.self),
            count: probeElementCount
        ))
        let gpuValues = Array(UnsafeBufferPointer(
            start: outputBuffer.contents().bindMemory(to: Float16.self, capacity: probeElementCount),
            count: probeElementCount
        ))

        for index in 0..<probeElementCount {
            #expect(
                gpuValues[index] == decodeBFloat16ToFloat16(blockedBits[index]),
                "Blocked direct-binding mismatch at packed[\(index)] tensor=\(tensorName)"
            )
        }
    }

    @Test("prefill resolves canonical row-major access even when decode-specialized layout exists")
    func prefillResolvesCanonicalRowMajorAccess() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }

        let stafURL = URL(fileURLWithPath: Self.realModelSTAFPath)
        guard FileManager.default.fileExists(atPath: stafURL.path) else {
            Issue.record("Missing STAF fixture at \(stafURL.path)")
            return
        }

        let tensorName = "model.layers.0.conv.in_proj.weight"
        let store = try Self.requireRealModelStore()
        let builder = STAFSpecializedWeightStoreBuilder(device: device)
        let blockedAccess = try builder.makeBlockedRows8Tiles128Access(for: tensorName, store: store)
        let specializedStore = store.registeringSpecializedBufferAccess(
            blockedAccess,
            for: STAFWeightAccessRequest(
                tensorName: tensorName,
                executionPhase: .decode,
                layoutPreference: .optimized(.blockedRows8Tiles128)
            )
        )

        let decodeAccess = try requireAccess(
            specializedStore.resolvedBufferAccess(for: STAFWeightAccessRequest(
                tensorName: tensorName,
                executionPhase: .decode,
                layoutPreference: .optimized(.blockedRows8Tiles128)
            )),
            message: "Missing decode access"
        )
        let prefillAccess = try requireAccess(
            specializedStore.resolvedBufferAccess(for: STAFWeightAccessRequest(
                tensorName: tensorName,
                executionPhase: .prefill,
                layoutPreference: .optimized(.blockedRows8Tiles128)
            )),
            message: "Missing prefill access"
        )

        #expect(decodeAccess.layout == .blockedRows8Tiles128)
        #expect(prefillAccess.layout == .rowMajor)
        #expect(prefillAccess.buffer === specializedStore.buffer)
    }

    @Test("real hot 2048->6144 decode tensors are conv.in_proj weights")
    func realHot2048To6144DecodeTensorsAreConvInProjWeights() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }

        let compiler = MetalInferenceCompiler()
        let graph = try makeResolvedLFM2Graph()
        let store = try Self.requireRealModelStore()
        let summaries = try compiler.summarizeCompiledDecodeWeightBindings(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            stafWeightStore: store,
            device: device
        )

        let targetSummaries = summaries.filter {
            $0.inputDimension == 2_048 &&
            $0.outputDimension == 6_144 &&
            $0.roles == ["in_proj"]
        }
        let tensorNames = Array(Set(targetSummaries.flatMap(\.tensorNames))).sorted()

        #expect(!tensorNames.isEmpty, "No real hot 2048->6144 decode tensors found")
        print("[STAF layout] hot 2048->6144 tensors: \(tensorNames.joined(separator: ", "))")
        #expect(tensorNames == Self.realHot2048To6144TensorNames)
    }

    @Test("real hot 2048->2048 decode tensors are square projection weights")
    func realHot2048To2048DecodeTensorsAreSquareProjectionWeights() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }

        let compiler = MetalInferenceCompiler()
        let graph = try makeResolvedLFM2Graph()
        let store = try Self.requireRealModelStore()
        let summaries = try compiler.summarizeCompiledDecodeWeightBindings(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            stafWeightStore: store,
            device: device
        )

        let targetSummaries = summaries.filter {
            let roles = Set($0.roles)
            return $0.inputDimension == 2_048 &&
                $0.outputDimension == 2_048 &&
                (roles == ["out_proj"] || roles == ["q_proj"] || roles == ["o_proj"])
        }
        let tensorNames = Array(Set(targetSummaries.flatMap(\.tensorNames))).sorted()

        #expect(!tensorNames.isEmpty, "No real hot 2048->2048 decode tensors found")
        print("[STAF layout] hot 2048->2048 tensors: \(tensorNames.joined(separator: ", "))")
        #expect(tensorNames == Self.realHot2048To2048SummaryTensorNames)
    }

    @Test("real hot 2048->6144 decode tensors resolve the current layout policy")
    func realHot2048To6144DecodeTensorsResolveCurrentLayoutPolicy() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }

        let stafURL = URL(fileURLWithPath: Self.realModelSTAFPath)
        guard FileManager.default.fileExists(atPath: stafURL.path) else {
            Issue.record("Missing STAF fixture at \(stafURL.path)")
            return
        }

        let store = try Self.requireRealModelStore()
        let compiler = MetalInferenceCompiler()
        let graph = try makeResolvedLFM2Graph()
        let summaries = try compiler.summarizeCompiledDecodeWeightBindings(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            stafWeightStore: store,
            device: device
        )

        let targetSummaries = summaries.filter {
            $0.inputDimension == 2_048 &&
            $0.outputDimension == 6_144 &&
            $0.roles == ["in_proj"]
        }
        #expect(!targetSummaries.isEmpty, "No resolved 2048->6144 decode summaries found")

        let expectedLayout = Input2048GEMVSourcePolicy
            .expanded6144(weightFormat: .bfloat16)
            .weightLayoutPolicy
            .stafWeightLayout

        for summary in targetSummaries {
            #expect(summary.preferredLayouts == [expectedLayout])
            #expect(summary.resolvedLayouts == [expectedLayout])
            if expectedLayout == .blockedRows8Tiles128 {
                #expect(
                    summary.resolvedBufferLabels.allSatisfy { $0.contains("::blockedRows8Tiles128") },
                    "Expected blocked buffer labels for step \(summary.stepIndex), got \(summary.resolvedBufferLabels)"
                )
            }
        }
    }

    @Test("single decode tensor override targets only the requested hot tensor")
    func singleDecodeTensorOverrideTargetsOnlyRequestedHotTensor() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }

        let stafURL = URL(fileURLWithPath: Self.realModelSTAFPath)
        guard FileManager.default.fileExists(atPath: stafURL.path) else {
            Issue.record("Missing STAF fixture at \(stafURL.path)")
            return
        }

        let store = try Self.requireRealModelStore()
        let targetTensor = try requireAccess(
            Self.realHot2048To6144TensorNames.first,
            message: "Missing hot tensor name"
        )
        let override = ProjectionWeightAccessPolicyOverride.prefer(
            .optimized(.blockedRows8Tiles128),
            forTensorNames: [targetTensor]
        )
        let compiler = MetalInferenceCompiler(weightAccessPolicyOverride: override)
        let graph = try makeResolvedLFM2Graph()
        let summaries = try compiler.summarizeCompiledDecodeWeightBindings(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            stafWeightStore: store,
            device: device
        )

        let targetSummaries = summaries.filter {
            $0.inputDimension == 2_048 &&
            $0.outputDimension == 6_144 &&
            $0.roles == ["in_proj"] &&
            $0.tensorNames == [targetTensor]
        }
        #expect(!targetSummaries.isEmpty, "No summaries found for override target tensor")
        for summary in targetSummaries {
            #expect(summary.preferredLayouts == [.blockedRows8Tiles128])
            #expect(summary.resolvedLayouts == [.blockedRows8Tiles128])
            #expect(summary.resolvedBufferLabels.allSatisfy { $0.contains("::blockedRows8Tiles128") })
            #expect(summary.kernelName.contains("blocked8x128"))
        }

        let untargetedSummaries = summaries.filter {
            $0.inputDimension == 2_048 &&
            $0.outputDimension == 6_144 &&
            $0.roles == ["in_proj"] &&
            $0.tensorNames.count == 1 &&
            $0.tensorNames[0] != targetTensor
        }
        #expect(!untargetedSummaries.isEmpty, "No untargeted hot summaries found")
        for summary in untargetedSummaries {
            #expect(summary.preferredLayouts == [.rowMajor])
            #expect(summary.resolvedLayouts == [.rowMajor])
        }
    }

    @Test("single decode tensor blocked8x128 override changes first decode token")
    func singleDecodeTensorBlocked8OverrideChangesFirstDecodeToken() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }

        let stafURL = URL(fileURLWithPath: Self.realModelSTAFPath)
        guard FileManager.default.fileExists(atPath: stafURL.path) else {
            Issue.record("Missing STAF fixture at \(stafURL.path)")
            return
        }

        let store = try Self.requireRealModelStore()
        let graph = try makeResolvedLFM2Graph()
        let targetTensor = try requireAccess(
            Self.realHot2048To6144TensorNames.first,
            message: "Missing hot tensor name"
        )

        let baselineCompiler = MetalInferenceCompiler()
        let overrideCompiler = MetalInferenceCompiler(
            weightAccessPolicyOverride: .prefer(
                .optimized(.blockedRows8Tiles128),
                forTensorNames: [targetTensor]
            )
        )

        let baselineDecodePlan = try baselineCompiler.compile(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            stafWeightStore: store,
            device: device
        )
        let baselinePrefillPlan = try baselineCompiler.compilePrefill(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store,
            device: device
        )
        let overrideDecodePlan = try overrideCompiler.compile(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            stafWeightStore: store,
            device: device
        )
        let overridePrefillPlan = try overrideCompiler.compilePrefill(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store,
            device: device
        )

        var baselineModel = try MetalInferenceModel(plan: baselineDecodePlan, device: device)
        baselineModel.prefillPlan = baselinePrefillPlan
        var overrideModel = try MetalInferenceModel(plan: overrideDecodePlan, device: device)
        overrideModel.prefillPlan = overridePrefillPlan

        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        let baselineFirst = baselineModel.prefill(tokens: tokens)
        let overrideFirst = overrideModel.prefill(tokens: tokens)
        #expect(baselineFirst == overrideFirst)

        let baselineSecond = baselineModel.decodeSync(tokenID: baselineFirst)
        let overrideSecond = overrideModel.decodeSync(tokenID: overrideFirst)
        #expect(baselineSecond != overrideSecond)
    }

    @Test("single square q_proj blocked8x128 override preserves first decode token")
    func singleSquareQProjBlocked8OverridePreservesFirstDecodeToken() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }

        let stafURL = URL(fileURLWithPath: Self.realModelSTAFPath)
        guard FileManager.default.fileExists(atPath: stafURL.path) else {
            Issue.record("Missing STAF fixture at \(stafURL.path)")
            return
        }

        let store = try Self.requireRealModelStore()
        let graph = try makeResolvedLFM2Graph()
        let targetTensor = try requireAccess(
            Self.realHot2048To2048TensorNames.first(where: { $0.contains(".self_attn.q_proj.weight") }),
            message: "Missing hot square q_proj tensor"
        )

        let baselineCompiler = MetalInferenceCompiler()
        let overrideCompiler = MetalInferenceCompiler(
            weightAccessPolicyOverride: .prefer(
                .optimized(.blockedRows8Tiles128),
                forTensorNames: [targetTensor]
            )
        )

        let baselineDecodePlan = try baselineCompiler.compile(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            stafWeightStore: store,
            device: device
        )
        let baselinePrefillPlan = try baselineCompiler.compilePrefill(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store,
            device: device
        )
        let overrideDecodePlan = try overrideCompiler.compile(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            stafWeightStore: store,
            device: device
        )
        let overridePrefillPlan = try overrideCompiler.compilePrefill(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store,
            device: device
        )

        var baselineModel = try MetalInferenceModel(plan: baselineDecodePlan, device: device)
        baselineModel.prefillPlan = baselinePrefillPlan
        var overrideModel = try MetalInferenceModel(plan: overrideDecodePlan, device: device)
        overrideModel.prefillPlan = overridePrefillPlan

        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        let baselineFirst = baselineModel.prefill(tokens: tokens)
        let overrideFirst = overrideModel.prefill(tokens: tokens)
        #expect(baselineFirst == overrideFirst)

        let baselineSecond = baselineModel.decodeSync(tokenID: baselineFirst)
        let overrideSecond = overrideModel.decodeSync(tokenID: overrideFirst)
        #expect(baselineSecond == overrideSecond)
    }

    @Test("single square q_proj blocked8x128 override preserves later decode tokens")
    func singleSquareQProjBlocked8OverridePreservesLaterDecodeTokens() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }

        let stafURL = URL(fileURLWithPath: Self.realModelSTAFPath)
        guard FileManager.default.fileExists(atPath: stafURL.path) else {
            Issue.record("Missing STAF fixture at \(stafURL.path)")
            return
        }

        let store = try Self.requireRealModelStore()
        let graph = try makeResolvedLFM2Graph()
        let targetTensor = try requireAccess(
            Self.realHot2048To2048TensorNames.first(where: { $0.contains(".self_attn.q_proj.weight") }),
            message: "Missing hot square q_proj tensor"
        )

        let baselineCompiler = MetalInferenceCompiler()
        let overrideCompiler = MetalInferenceCompiler(
            weightAccessPolicyOverride: .prefer(
                .optimized(.blockedRows8Tiles128),
                forTensorNames: [targetTensor]
            )
        )

        let baselineDecodePlan = try baselineCompiler.compile(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            stafWeightStore: store,
            device: device
        )
        let baselinePrefillPlan = try baselineCompiler.compilePrefill(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store,
            device: device
        )
        let overrideDecodePlan = try overrideCompiler.compile(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            stafWeightStore: store,
            device: device
        )
        let overridePrefillPlan = try overrideCompiler.compilePrefill(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store,
            device: device
        )

        var baselineModel = try MetalInferenceModel(plan: baselineDecodePlan, device: device)
        baselineModel.prefillPlan = baselinePrefillPlan
        var overrideModel = try MetalInferenceModel(plan: overrideDecodePlan, device: device)
        overrideModel.prefillPlan = overridePrefillPlan

        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        let baselineFirst = baselineModel.prefill(tokens: tokens)
        let overrideFirst = overrideModel.prefill(tokens: tokens)
        #expect(baselineFirst == overrideFirst)

        var baselineToken = baselineFirst
        var overrideToken = overrideFirst
        for step in 0..<3 {
            baselineToken = baselineModel.decodeSync(tokenID: baselineToken)
            overrideToken = overrideModel.decodeSync(tokenID: overrideToken)
            #expect(baselineToken == overrideToken, "q_proj override diverged at decode step \(step)")
        }
    }

    @Test("single square self_attn.out_proj blocked8x128 override changes first decode token")
    func singleSquareSelfAttentionOutProjBlocked8OverrideChangesFirstDecodeToken() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }

        let stafURL = URL(fileURLWithPath: Self.realModelSTAFPath)
        guard FileManager.default.fileExists(atPath: stafURL.path) else {
            Issue.record("Missing STAF fixture at \(stafURL.path)")
            return
        }

        let store = try Self.requireRealModelStore()
        let graph = try makeResolvedLFM2Graph()
        let targetTensor = try requireAccess(
            Self.realHot2048To2048TensorNames.first(where: { $0.contains(".self_attn.out_proj.weight") }),
            message: "Missing hot square self_attn.out_proj tensor"
        )

        let baselineCompiler = MetalInferenceCompiler()
        let overrideCompiler = MetalInferenceCompiler(
            weightAccessPolicyOverride: .prefer(
                .optimized(.blockedRows8Tiles128),
                forTensorNames: [targetTensor]
            )
        )

        let baselineDecodePlan = try baselineCompiler.compile(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            stafWeightStore: store,
            device: device
        )
        let baselinePrefillPlan = try baselineCompiler.compilePrefill(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store,
            device: device
        )
        let overrideDecodePlan = try overrideCompiler.compile(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            stafWeightStore: store,
            device: device
        )
        let overridePrefillPlan = try overrideCompiler.compilePrefill(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store,
            device: device
        )

        var baselineModel = try MetalInferenceModel(plan: baselineDecodePlan, device: device)
        baselineModel.prefillPlan = baselinePrefillPlan
        var overrideModel = try MetalInferenceModel(plan: overrideDecodePlan, device: device)
        overrideModel.prefillPlan = overridePrefillPlan

        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        let baselineFirst = baselineModel.prefill(tokens: tokens)
        let overrideFirst = overrideModel.prefill(tokens: tokens)
        #expect(baselineFirst == overrideFirst)

        let baselineSecond = baselineModel.decodeSync(tokenID: baselineFirst)
        let overrideSecond = overrideModel.decodeSync(tokenID: overrideFirst)
        #expect(baselineSecond != overrideSecond)
    }

    @Test("single square conv.out_proj blocked8x128 override changes first decode token")
    func singleSquareConvOutProjBlocked8OverrideChangesFirstDecodeToken() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }

        let stafURL = URL(fileURLWithPath: Self.realModelSTAFPath)
        guard FileManager.default.fileExists(atPath: stafURL.path) else {
            Issue.record("Missing STAF fixture at \(stafURL.path)")
            return
        }

        let store = try Self.requireRealModelStore()
        let graph = try makeResolvedLFM2Graph()
        let targetTensor = try requireAccess(
            Self.realHot2048To2048TensorNames.first(where: { $0.contains(".conv.out_proj.weight") }),
            message: "Missing hot square conv.out_proj tensor"
        )

        let baselineCompiler = MetalInferenceCompiler()
        let overrideCompiler = MetalInferenceCompiler(
            weightAccessPolicyOverride: .prefer(
                .optimized(.blockedRows8Tiles128),
                forTensorNames: [targetTensor]
            )
        )

        let baselineDecodePlan = try baselineCompiler.compile(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            stafWeightStore: store,
            device: device
        )
        let baselinePrefillPlan = try baselineCompiler.compilePrefill(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store,
            device: device
        )
        let overrideDecodePlan = try overrideCompiler.compile(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            stafWeightStore: store,
            device: device
        )
        let overridePrefillPlan = try overrideCompiler.compilePrefill(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store,
            device: device
        )

        var baselineModel = try MetalInferenceModel(plan: baselineDecodePlan, device: device)
        baselineModel.prefillPlan = baselinePrefillPlan
        var overrideModel = try MetalInferenceModel(plan: overrideDecodePlan, device: device)
        overrideModel.prefillPlan = overridePrefillPlan

        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        let baselineFirst = baselineModel.prefill(tokens: tokens)
        let overrideFirst = overrideModel.prefill(tokens: tokens)
        #expect(baselineFirst == overrideFirst)

        let baselineSecond = baselineModel.decodeSync(tokenID: baselineFirst)
        let overrideSecond = overrideModel.decodeSync(tokenID: overrideFirst)
        #expect(baselineSecond != overrideSecond)
    }

    @Test("all square q_proj blocked8x128 override preserves first decode token")
    func allSquareQProjBlocked8OverridePreservesFirstDecodeToken() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }

        let stafURL = URL(fileURLWithPath: Self.realModelSTAFPath)
        guard FileManager.default.fileExists(atPath: stafURL.path) else {
            Issue.record("Missing STAF fixture at \(stafURL.path)")
            return
        }

        let store = try Self.requireRealModelStore()
        let graph = try makeResolvedLFM2Graph()
        let targetTensors = Set(Self.realHot2048To2048TensorNames.filter { $0.contains(".self_attn.q_proj.weight") })
        #expect(!targetTensors.isEmpty)

        let baselineCompiler = MetalInferenceCompiler()
        let overrideCompiler = MetalInferenceCompiler(
            weightAccessPolicyOverride: .prefer(
                .optimized(.blockedRows8Tiles128),
                forTensorNames: targetTensors
            )
        )

        let baselineDecodePlan = try baselineCompiler.compile(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            stafWeightStore: store,
            device: device
        )
        let baselinePrefillPlan = try baselineCompiler.compilePrefill(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store,
            device: device
        )
        let overrideDecodePlan = try overrideCompiler.compile(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            stafWeightStore: store,
            device: device
        )
        let overridePrefillPlan = try overrideCompiler.compilePrefill(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store,
            device: device
        )

        var baselineModel = try MetalInferenceModel(plan: baselineDecodePlan, device: device)
        baselineModel.prefillPlan = baselinePrefillPlan
        var overrideModel = try MetalInferenceModel(plan: overrideDecodePlan, device: device)
        overrideModel.prefillPlan = overridePrefillPlan

        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        let baselineFirst = baselineModel.prefill(tokens: tokens)
        let overrideFirst = overrideModel.prefill(tokens: tokens)
        #expect(baselineFirst == overrideFirst)

        let baselineSecond = baselineModel.decodeSync(tokenID: baselineFirst)
        let overrideSecond = overrideModel.decodeSync(tokenID: overrideFirst)
        #expect(baselineSecond == overrideSecond)
    }

    @Test("square q_proj prefix-2 blocked8x128 override preserves later decode tokens")
    func squareQProjPrefix2Blocked8OverridePreservesLaterDecodeTokens() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }

        let stafURL = URL(fileURLWithPath: Self.realModelSTAFPath)
        guard FileManager.default.fileExists(atPath: stafURL.path) else {
            Issue.record("Missing STAF fixture at \(stafURL.path)")
            return
        }

        let store = try Self.requireRealModelStore()
        let graph = try makeResolvedLFM2Graph()
        let targetTensors = Set(
            Self.realHot2048To2048TensorNames
                .filter { $0.contains(".self_attn.q_proj.weight") }
                .sorted()
                .prefix(2)
        )
        #expect(targetTensors.count == 2)

        let baselineCompiler = MetalInferenceCompiler()
        let overrideCompiler = MetalInferenceCompiler(
            weightAccessPolicyOverride: .prefer(
                .optimized(.blockedRows8Tiles128),
                forTensorNames: targetTensors
            )
        )

        let baselineDecodePlan = try baselineCompiler.compile(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            stafWeightStore: store,
            device: device
        )
        let baselinePrefillPlan = try baselineCompiler.compilePrefill(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store,
            device: device
        )
        let overrideDecodePlan = try overrideCompiler.compile(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            stafWeightStore: store,
            device: device
        )
        let overridePrefillPlan = try overrideCompiler.compilePrefill(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store,
            device: device
        )

        var baselineModel = try MetalInferenceModel(plan: baselineDecodePlan, device: device)
        baselineModel.prefillPlan = baselinePrefillPlan
        var overrideModel = try MetalInferenceModel(plan: overrideDecodePlan, device: device)
        overrideModel.prefillPlan = overridePrefillPlan

        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        let baselineFirst = baselineModel.prefill(tokens: tokens)
        let overrideFirst = overrideModel.prefill(tokens: tokens)
        #expect(baselineFirst == overrideFirst)

        var baselineToken = baselineFirst
        var overrideToken = overrideFirst
        for step in 0..<3 {
            baselineToken = baselineModel.decodeSync(tokenID: baselineToken)
            overrideToken = overrideModel.decodeSync(tokenID: overrideToken)
            #expect(baselineToken == overrideToken, "q_proj prefix override diverged at decode step \(step)")
        }
    }

    @Test("square q_proj blocked8x128 prefix boundary preserves decode token")
    func squareQProjBlocked8PrefixBoundaryPreservesDecodeToken() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }

        let stafURL = URL(fileURLWithPath: Self.realModelSTAFPath)
        guard FileManager.default.fileExists(atPath: stafURL.path) else {
            Issue.record("Missing STAF fixture at \(stafURL.path)")
            return
        }

        let store = try Self.requireRealModelStore()
        let graph = try makeResolvedLFM2Graph()
        let qProjTensors = Self.realHot2048To2048TensorNames
            .filter { $0.contains(".self_attn.q_proj.weight") }
            .sorted()
        #expect(!qProjTensors.isEmpty)

        let baselineCompiler = MetalInferenceCompiler()
        let baselineDecodePlan = try baselineCompiler.compile(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            stafWeightStore: store,
            device: device
        )
        let baselinePrefillPlan = try baselineCompiler.compilePrefill(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store,
            device: device
        )
        var baselineModel = try MetalInferenceModel(plan: baselineDecodePlan, device: device)
        baselineModel.prefillPlan = baselinePrefillPlan

        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        let baselineFirst = baselineModel.prefill(tokens: tokens)
        let baselineSecond = baselineModel.decodeSync(tokenID: baselineFirst)

        var firstFailingPrefixSize: Int?
        for prefixSize in 1...qProjTensors.count {
            let overrideCompiler = MetalInferenceCompiler(
                weightAccessPolicyOverride: .prefer(
                    .optimized(.blockedRows8Tiles128),
                    forTensorNames: Set(qProjTensors.prefix(prefixSize))
                )
            )
            let overrideDecodePlan = try overrideCompiler.compile(
                graph: graph,
                hiddenSize: 2_048,
                intermediateSize: 8_192,
                vocabSize: 65_536,
                stafWeightStore: store,
                device: device
            )
            let overridePrefillPlan = try overrideCompiler.compilePrefill(
                graph: graph,
                hiddenSize: 2_048,
                intermediateSize: 8_192,
                vocabSize: 65_536,
                inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
                stafWeightStore: store,
                device: device
            )
            var overrideModel = try MetalInferenceModel(plan: overrideDecodePlan, device: device)
            overrideModel.prefillPlan = overridePrefillPlan

            let overrideFirst = overrideModel.prefill(tokens: tokens)
            #expect(overrideFirst == baselineFirst)
            let overrideSecond = overrideModel.decodeSync(tokenID: overrideFirst)
            print("[STAF layout] q_proj prefix \(prefixSize): second token = \(overrideSecond) tensors=\(Array(qProjTensors.prefix(prefixSize)))")
            if overrideSecond != baselineSecond, firstFailingPrefixSize == nil {
                firstFailingPrefixSize = prefixSize
            }
        }

        #expect(firstFailingPrefixSize == nil)
    }

    @Test("all hot decode tensors blocked4x128 override changes second decode token")
    func allHotDecodeTensorsBlocked4OverrideChangesSecondDecodeToken() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }

        let stafURL = URL(fileURLWithPath: Self.realModelSTAFPath)
        guard FileManager.default.fileExists(atPath: stafURL.path) else {
            Issue.record("Missing STAF fixture at \(stafURL.path)")
            return
        }

        let store = try Self.requireRealModelStore()
        let graph = try makeResolvedLFM2Graph()
        let hotTensors = Set(Self.realHot2048To6144TensorNames)

        let baselineCompiler = MetalInferenceCompiler()
        let overrideCompiler = MetalInferenceCompiler(
            weightAccessPolicyOverride: .prefer(
                .optimized(.blockedRows4Tiles128),
                forTensorNames: hotTensors
            )
        )

        let baselineDecodePlan = try baselineCompiler.compile(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            stafWeightStore: store,
            device: device
        )
        let baselinePrefillPlan = try baselineCompiler.compilePrefill(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store,
            device: device
        )
        let overrideDecodePlan = try overrideCompiler.compile(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            stafWeightStore: store,
            device: device
        )
        let overridePrefillPlan = try overrideCompiler.compilePrefill(
            graph: graph,
            hiddenSize: 2_048,
            intermediateSize: 8_192,
            vocabSize: 65_536,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store,
            device: device
        )

        var baselineModel = try MetalInferenceModel(plan: baselineDecodePlan, device: device)
        baselineModel.prefillPlan = baselinePrefillPlan
        var overrideModel = try MetalInferenceModel(plan: overrideDecodePlan, device: device)
        overrideModel.prefillPlan = overridePrefillPlan

        let tokens: [Int32] = [1, 1, 6, 6423, 708]
        let baselineFirst = baselineModel.prefill(tokens: tokens)
        let overrideFirst = overrideModel.prefill(tokens: tokens)
        #expect(baselineFirst == overrideFirst)

        let baselineSecond = baselineModel.decodeSync(tokenID: baselineFirst)
        let overrideSecond = overrideModel.decodeSync(tokenID: overrideFirst)
        #expect(baselineSecond != overrideSecond)
    }

    private func makeBlockedRowsFixture() throws -> BlockedRowsFixture {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }

        let outputDimension = 16
        let inputDimension = 2_048
        let rowsPerBlock = 8
        let tileElements = 128
        let tensorName = "test.weight"
        let sourceElements = (0..<(outputDimension * inputDimension)).map { UInt16($0) }
        let bufferLength = sourceElements.count * MemoryLayout<UInt16>.size

        var mutableSource = sourceElements
        guard let sourceBuffer = device.makeBuffer(
            bytes: &mutableSource,
            length: bufferLength,
            options: .storageModeShared
        ) else {
            throw MetalCompilerError.deviceSetupFailed("Failed to allocate source buffer")
        }

        let entry = STAFTensorEntry(
            name: tensorName,
            payloadOffset: 0,
            payloadSize: bufferLength,
            schemeIdentifier: .passthrough,
            semanticRole: .unknown,
            shape: [outputDimension, inputDimension],
            blockSize: 0,
            groupSize: 0,
            bufferOffset: 0
        )
        let store = STAFWeightStore(
            buffer: sourceBuffer,
            entries: [tensorName: entry],
            metadata: .empty,
            specializedBufferAccesses: [:]
        )
        let builder = STAFSpecializedWeightStoreBuilder(device: device)
        let access = try builder.makeBlockedRows8Tiles128Access(for: tensorName, store: store)
        let packedElements = Array(UnsafeBufferPointer(
            start: access.buffer.contents().assumingMemoryBound(to: UInt16.self),
            count: sourceElements.count
        ))

        return BlockedRowsFixture(
            outputDimension: outputDimension,
            inputDimension: inputDimension,
            rowsPerBlock: rowsPerBlock,
            tileElements: tileElements,
            sourceElements: sourceElements,
            packedElements: packedElements
        )
    }

    private func makeResolvedLFM2Graph() throws -> ModelGraph {
        let config = ModelConfig(
            hiddenSize: 2_048, layerCount: 16, intermediateSize: 8_192,
            vocabSize: 65_536, attentionHeads: 32, kvHeads: 8, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 1_000_000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: true,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: nil, convLCache: 3,
            partialRotaryFactor: nil, slidingWindow: nil,
            layerTypes: ["conv", "conv", "full_attention", "conv", "conv", "full_attention",
                         "conv", "conv", "full_attention", "conv", "full_attention", "conv",
                         "full_attention", "conv", "full_attention", "conv"]
        )
        let graph = try ModelGraph(LFM2(config: config))
        return ParameterResolver().resolve(graph: graph, convention: .lfm2Family)
    }

    private func encodeAndRunArgumentKernel(
        pipeline: MTLComputePipelineState,
        function: MTLFunction,
        queue: MTLCommandQueue,
        groups: Int,
        inputBuffer: MTLBuffer,
        weightAccess: STAFWeightBufferAccess,
        outputBuffer: MTLBuffer
    ) throws {
        let argumentEncoder = function.makeArgumentEncoder(bufferIndex: 30)
        guard let argumentBuffer = pipeline.device.makeBuffer(
            length: argumentEncoder.encodedLength,
            options: .storageModeShared
        ) else {
            throw MetalCompilerError.deviceSetupFailed("Failed to allocate argument buffer")
        }

        argumentEncoder.setArgumentBuffer(argumentBuffer, offset: 0)
        argumentEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
        argumentEncoder.setBuffer(weightAccess.buffer, offset: weightAccess.offset, index: 1)
        argumentEncoder.setBuffer(outputBuffer, offset: 0, index: 2)

        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalCompilerError.deviceSetupFailed("Failed to encode compute command")
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(argumentBuffer, offset: 0, index: 30)
        encoder.useResource(argumentBuffer, usage: .read)
        encoder.useResource(inputBuffer, usage: .read)
        encoder.useResource(weightAccess.buffer, usage: .read)
        encoder.useResource(outputBuffer, usage: .write)
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw MetalCompilerError.deviceSetupFailed("Argument-kernel GPU error: \(error)")
        }
    }

    private func encodeAndRunConvStateUpdateArgumentKernel(
        pipeline: MTLComputePipelineState,
        function: MTLFunction,
        queue: MTLCommandQueue,
        convStateBuffer: MTLBuffer,
        inProjOutputBuffer: MTLBuffer,
        weightAccess: STAFWeightBufferAccess,
        outputBuffer: MTLBuffer,
        dimension: Int,
        kernelSize: Int
    ) throws {
        let argumentEncoder = function.makeArgumentEncoder(bufferIndex: 30)
        guard let argumentBuffer = pipeline.device.makeBuffer(
            length: argumentEncoder.encodedLength,
            options: .storageModeShared
        ) else {
            throw MetalCompilerError.deviceSetupFailed("Failed to allocate conv argument buffer")
        }

        var mutableDimension = UInt32(dimension)
        var mutableKernelSize = UInt32(kernelSize)

        argumentEncoder.setArgumentBuffer(argumentBuffer, offset: 0)
        argumentEncoder.setBuffer(convStateBuffer, offset: 0, index: 0)
        argumentEncoder.setBuffer(inProjOutputBuffer, offset: 0, index: 1)
        argumentEncoder.setBuffer(weightAccess.buffer, offset: weightAccess.offset, index: 2)
        argumentEncoder.setBuffer(outputBuffer, offset: 0, index: 3)

        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalCompilerError.deviceSetupFailed("Failed to encode conv command")
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(argumentBuffer, offset: 0, index: 30)
        encoder.setBytes(&mutableDimension, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&mutableKernelSize, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.useResource(argumentBuffer, usage: .read)
        encoder.useResource(convStateBuffer, usage: .read)
        encoder.useResource(inProjOutputBuffer, usage: .read)
        encoder.useResource(weightAccess.buffer, usage: .read)
        encoder.useResource(outputBuffer, usage: .write)
        encoder.dispatchThreads(
            MTLSize(width: dimension, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(dimension, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw MetalCompilerError.deviceSetupFailed("Conv argument-kernel GPU error: \(error)")
        }
    }

    private func encodeBFloat16(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let lsb = (bits >> 16) & 1
        let roundingBias: UInt32 = 0x7FFF + lsb
        return UInt16((bits + roundingBias) >> 16)
    }

    private func decodeBFloat16ToFloat16(_ bits: UInt16) -> Float16 {
        Float16(Float(bitPattern: UInt32(bits) << 16))
    }

    private func requestResidencyIfAvailable(device: MTLDevice, buffer: MTLBuffer) {
        if #available(macOS 15.0, iOS 18.0, *) {
            do {
                let descriptor = MTLResidencySetDescriptor()
                let residencySet = try device.makeResidencySet(descriptor: descriptor)
                residencySet.addAllocation(buffer)
                residencySet.commit()
                residencySet.requestResidency()
            } catch {
                // Residency is optional in tests.
            }
        }
    }

    private func requireAccess<T>(_ value: T?, message: String) throws -> T {
        guard let value else {
            throw MetalCompilerError.deviceSetupFailed(message)
        }
        return value
    }
}

private struct BlockedRowsFixture {
    let outputDimension: Int
    let inputDimension: Int
    let rowsPerBlock: Int
    let tileElements: Int
    let sourceElements: [UInt16]
    let packedElements: [UInt16]
}

private extension STAFSpecializedLayoutTests {
    static let realHot2048To2048SummaryTensorNames = [
        "model.layers.0.conv.out_proj.weight",
        "model.layers.1.conv.out_proj.weight",
        "model.layers.10.self_attn.out_proj.weight",
        "model.layers.11.conv.out_proj.weight",
        "model.layers.12.self_attn.out_proj.weight",
        "model.layers.13.conv.out_proj.weight",
        "model.layers.14.self_attn.out_proj.weight",
        "model.layers.15.conv.out_proj.weight",
        "model.layers.2.self_attn.out_proj.weight",
        "model.layers.3.conv.out_proj.weight",
        "model.layers.4.conv.out_proj.weight",
        "model.layers.5.self_attn.out_proj.weight",
        "model.layers.6.conv.out_proj.weight",
        "model.layers.7.conv.out_proj.weight",
        "model.layers.8.self_attn.out_proj.weight",
        "model.layers.9.conv.out_proj.weight",
    ]

    static let realHot2048To2048TensorNames = [
        "model.layers.0.conv.out_proj.weight",
        "model.layers.1.conv.out_proj.weight",
        "model.layers.10.self_attn.out_proj.weight",
        "model.layers.10.self_attn.q_proj.weight",
        "model.layers.11.conv.out_proj.weight",
        "model.layers.12.self_attn.out_proj.weight",
        "model.layers.12.self_attn.q_proj.weight",
        "model.layers.13.conv.out_proj.weight",
        "model.layers.14.self_attn.out_proj.weight",
        "model.layers.14.self_attn.q_proj.weight",
        "model.layers.15.conv.out_proj.weight",
        "model.layers.2.self_attn.out_proj.weight",
        "model.layers.2.self_attn.q_proj.weight",
        "model.layers.3.conv.out_proj.weight",
        "model.layers.4.conv.out_proj.weight",
        "model.layers.5.self_attn.out_proj.weight",
        "model.layers.5.self_attn.q_proj.weight",
        "model.layers.6.conv.out_proj.weight",
        "model.layers.7.conv.out_proj.weight",
        "model.layers.8.self_attn.out_proj.weight",
        "model.layers.8.self_attn.q_proj.weight",
        "model.layers.9.conv.out_proj.weight",
    ]

    static let realHot2048To6144TensorNames = [
        "model.layers.0.conv.in_proj.weight",
        "model.layers.1.conv.in_proj.weight",
        "model.layers.11.conv.in_proj.weight",
        "model.layers.13.conv.in_proj.weight",
        "model.layers.15.conv.in_proj.weight",
        "model.layers.3.conv.in_proj.weight",
        "model.layers.4.conv.in_proj.weight",
        "model.layers.6.conv.in_proj.weight",
        "model.layers.7.conv.in_proj.weight",
        "model.layers.9.conv.in_proj.weight",
    ]
}
#endif
