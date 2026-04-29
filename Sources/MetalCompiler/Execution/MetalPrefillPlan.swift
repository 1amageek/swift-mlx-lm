import Metal

// MARK: - Prefill Dispatch Plan

/// Execution mode for a prefill step.
public enum PrefillStepMode: Sendable {
    case batch
    case perPosition
    case lastToken
}

/// How a prefill step consumes the runtime sequence length.
public enum PrefillSequenceLengthPolicy: Sendable {
    case none
    case bind(index: Int)
    case bindAndAdjustGridHeight(index: Int)
    /// Bind sequence length and adjust grid height with tiling: ceil(seqLen / tileHeight).
    case bindAndAdjustGridHeightTiled(index: Int, tileHeight: Int)

    public var bindingIndex: Int? {
        switch self {
        case .none:
            return nil
        case .bind(let index), .bindAndAdjustGridHeight(let index),
             .bindAndAdjustGridHeightTiled(let index, _):
            return index
        }
    }

    public var adjustsGridHeightToSequenceLength: Bool {
        switch self {
        case .bindAndAdjustGridHeight, .bindAndAdjustGridHeightTiled:
            return true
        case .none, .bind:
            return false
        }
    }

    public var tileHeight: Int? {
        switch self {
        case .bindAndAdjustGridHeightTiled(_, let tileHeight):
            return tileHeight
        default:
            return nil
        }
    }
}

/// Tile-size variant for MPP GEMM prefill steps.
///
/// A single prefill step can compile multiple pipelines differing only in
/// the matmul2d M-tile constant. At dispatch time the runtime picks the
/// smallest M-tile whose padded grid still utilises the hardware well, which
/// reduces padding waste for short sequences without regressing long-sequence
/// throughput.
public struct PrefillTileVariant: @unchecked Sendable {
    public let tileHeight: Int
    public let descriptor: MetalDispatchDescriptor

    public init(tileHeight: Int, descriptor: MetalDispatchDescriptor) {
        self.tileHeight = tileHeight
        self.descriptor = descriptor
    }
}

/// A single step in the prefill sequence graph.
public struct MetalPrefillStep: @unchecked Sendable {
    public let descriptor: MetalDispatchDescriptor
    public let bindings: MetalBindingTable
    public let mode: PrefillStepMode
    public let sequenceLengthPolicy: PrefillSequenceLengthPolicy
    public let positionBufferIndex: Int?
    public let perPositionStrides: [Int: Int]
    public let metadata: MetalDispatchStepMetadata
    /// MPP GEMM tile-size variants sorted ascending by `tileHeight`. Empty for
    /// non-MPP steps; in that case the base `descriptor` is used unconditionally.
    public let tileVariants: [PrefillTileVariant]

    public var pipeline: MTLComputePipelineState { descriptor.pipeline }
    public var gridSize: MTLSize { descriptor.gridSize }
    public var threadgroupSize: MTLSize { descriptor.threadgroupSize }
    public var threadgroupMemoryLength: Int { descriptor.threadgroupMemoryLength }
    public var sync: SynchronizationKind { descriptor.sync }
    public var barrierPolicy: MetalBarrierPolicy { descriptor.barrierPolicy }
    public var bufferBindings: [(index: Int, buffer: MTLBuffer, offset: Int)] {
        bindings.buffers.map { (index: $0.index, buffer: $0.buffer, offset: $0.offset) }
    }
    public var bytesBindings: [(index: Int, value: [UInt8])] {
        bindings.constantBindings.inlineBindings.map { (index: $0.index, value: $0.value) }
    }

    public init(
        pipeline: MTLComputePipelineState,
        gridSize: MTLSize,
        threadgroupSize: MTLSize,
        bufferBindings: [(index: Int, buffer: MTLBuffer, offset: Int)],
        bytesBindings: [(index: Int, value: [UInt8])],
        threadgroupMemoryLength: Int,
        sync: SynchronizationKind,
        mode: PrefillStepMode,
        sequenceLengthPolicy: PrefillSequenceLengthPolicy,
        positionBufferIndex: Int?,
        perPositionStrides: [Int : Int],
        metadata: MetalDispatchStepMetadata = .init(),
        tileVariants: [PrefillTileVariant] = []
    ) {
        self.descriptor = MetalDispatchDescriptor(
            pipeline: pipeline,
            gridSize: gridSize,
            threadgroupSize: threadgroupSize,
            threadgroupMemoryLength: threadgroupMemoryLength,
            barrierPolicy: MetalBarrierPolicy(sync))
        self.bindings = MetalBindingTable(
            bufferBindings: bufferBindings,
            bytesBindings: bytesBindings)
        self.mode = mode
        self.sequenceLengthPolicy = sequenceLengthPolicy
        self.positionBufferIndex = positionBufferIndex
        self.perPositionStrides = perPositionStrides
        self.metadata = metadata
        self.tileVariants = tileVariants.sorted { $0.tileHeight < $1.tileHeight }
    }

    public init(
        descriptor: MetalDispatchDescriptor,
        bindings: MetalBindingTable,
        mode: PrefillStepMode,
        sequenceLengthPolicy: PrefillSequenceLengthPolicy,
        positionBufferIndex: Int?,
        perPositionStrides: [Int: Int],
        metadata: MetalDispatchStepMetadata = .init(),
        tileVariants: [PrefillTileVariant] = []
    ) {
        self.descriptor = descriptor
        self.bindings = bindings
        self.mode = mode
        self.sequenceLengthPolicy = sequenceLengthPolicy
        self.positionBufferIndex = positionBufferIndex
        self.perPositionStrides = perPositionStrides
        self.metadata = metadata
        self.tileVariants = tileVariants.sorted { $0.tileHeight < $1.tileHeight }
    }

    public func bindRuntimeArguments(
        argumentTable: MTL4ArgumentTable,
        runtimeConstantBuffer: MTLBuffer,
        sequenceLengthOffset: Int
    ) {
        guard let bindingIndex = sequenceLengthPolicy.bindingIndex else { return }
        argumentTable.setAddress(
            runtimeConstantBuffer.gpuAddress + UInt64(sequenceLengthOffset),
            index: bindingIndex
        )
    }

    public func bindStaticArguments(
        argumentTable: MTL4ArgumentTable,
        position: Int? = nil
    ) {
        guard let position else {
            bindings.bind(to: argumentTable)
            return
        }
        var adjustedOffsets: [Int: Int] = [:]
        adjustedOffsets.reserveCapacity(bindings.buffers.count)
        for binding in bindings.buffers {
            adjustedOffsets[binding.index] = binding.offset + position * (perPositionStrides[binding.index] ?? 0)
        }
        bindings.bind(to: argumentTable, adjustedBufferOffsets: adjustedOffsets)
    }

    public func resolvedGridSize(sequenceLength: Int) -> MTLSize {
        let baseGridSize = resolvedDescriptor(sequenceLength: sequenceLength).gridSize
        guard sequenceLengthPolicy.adjustsGridHeightToSequenceLength, baseGridSize.height > 1 else {
            return baseGridSize
        }
        if let tileHeight = resolvedTileHeight(sequenceLength: sequenceLength) {
            let tiledHeight = (sequenceLength + tileHeight - 1) / tileHeight
            return MTLSize(width: baseGridSize.width, height: tiledHeight, depth: baseGridSize.depth)
        }
        return MTLSize(width: baseGridSize.width, height: sequenceLength, depth: baseGridSize.depth)
    }

    /// Pick a dispatch descriptor for the given runtime sequence length.
    ///
    /// If tile variants are available, selects the smallest `tileHeight` that
    /// still produces ≥ 1 tile (i.e. `tileHeight >= sequenceLength`) when the
    /// sequence is short enough to fit; otherwise falls back to the largest
    /// available variant so long sequences keep the high-utilization tile.
    public func resolvedDescriptor(sequenceLength: Int) -> MetalDispatchDescriptor {
        guard !tileVariants.isEmpty else { return descriptor }
        // tileVariants is sorted ascending by tileHeight (init guarantee).
        for variant in tileVariants where variant.tileHeight >= sequenceLength {
            return variant.descriptor
        }
        return tileVariants.last!.descriptor
    }

    /// Effective tile height to use when computing the tiled grid dimension.
    /// Returns nil when the policy is not tiled.
    private func resolvedTileHeight(sequenceLength: Int) -> Int? {
        // Explicit tile height in the policy takes precedence but may be
        // overridden by the selected tile variant.
        guard sequenceLengthPolicy.tileHeight != nil else { return nil }
        guard !tileVariants.isEmpty else { return sequenceLengthPolicy.tileHeight }
        for variant in tileVariants where variant.tileHeight >= sequenceLength {
            return variant.tileHeight
        }
        return tileVariants.last!.tileHeight
    }
}

/// Sequence-aware execution plan for prefill.
public struct MetalPrefillPlan: @unchecked Sendable {
    public let steps: [MetalPrefillStep]
    public let buffers: PrefillBufferSet
    public let slotDimension: Int
    public let maximumSequenceLength: Int
    public let stepCount: Int
    public let usesMPP: Bool
    let quantizationPlan: MetalQuantizationPlan
    let finalHiddenBuffer: MTLBuffer
    let finalHiddenBaseOffset: Int
    let finalHiddenRowStride: Int
    let supplementalResidencyBuffers: [MTLBuffer]

    init(
        steps: [MetalPrefillStep],
        buffers: PrefillBufferSet,
        slotDimension: Int,
        maximumSequenceLength: Int,
        stepCount: Int,
        usesMPP: Bool,
        quantizationPlan: MetalQuantizationPlan = .empty,
        finalHiddenBuffer: MTLBuffer,
        finalHiddenBaseOffset: Int,
        finalHiddenRowStride: Int,
        supplementalResidencyBuffers: [MTLBuffer]
    ) {
        self.steps = steps
        self.buffers = buffers
        self.slotDimension = slotDimension
        self.maximumSequenceLength = maximumSequenceLength
        self.stepCount = stepCount
        self.usesMPP = usesMPP
        self.quantizationPlan = quantizationPlan
        self.finalHiddenBuffer = finalHiddenBuffer
        self.finalHiddenBaseOffset = finalHiddenBaseOffset
        self.finalHiddenRowStride = finalHiddenRowStride
        self.supplementalResidencyBuffers = supplementalResidencyBuffers
    }

    package func finalHiddenSource(sequenceLength: Int) -> (buffer: MTLBuffer, offset: Int) {
        let positionOffset = max(sequenceLength - 1, 0) * finalHiddenRowStride
        return (buffer: finalHiddenBuffer, offset: finalHiddenBaseOffset + positionOffset)
    }

    package func quantizationSummary(limit: Int = 8) -> String {
        quantizationPlan.summarizedLines(limit: limit).joined(separator: "\n")
    }

    package func quantizationKernelFamilies(path: String? = nil) -> [String] {
        quantizationPlan.entries
            .filter { entry in
                guard let path else { return true }
                return entry.path.rawValue == path
            }
            .map { $0.kernelFamily.description }
    }

    package enum SequencePrefillFallbackReason: Sendable, Equatable, CustomStringConvertible {
        case unsupportedQ3Quantization

        package var description: String {
            switch self {
            case .unsupportedQ3Quantization:
                return "Q3 prefill projection or embedding lookup is not supported by sequence prefill"
            }
        }
    }

    package var sequencePrefillFallbackReason: SequencePrefillFallbackReason? {
        if quantizationPlan.entries.contains(where: { entry in
            guard entry.path == .prefillProjection || entry.path == .embeddingLookup else {
                return false
            }
            switch entry.schemeIdentifier.baseScheme {
            case .q3Group16ScaleF16, .q3Group32ScaleF16, .q3Group64ScaleF16:
                return true
            default:
                return false
            }
        }) {
            return .unsupportedQ3Quantization
        }

        return nil
    }

    package var requiresSequentialPromptIngestion: Bool {
        sequencePrefillFallbackReason != nil
    }
}

/// Prefill buffer set ([maxSeqLen × dim] layout).
public struct PrefillBufferSet: @unchecked Sendable {
    public let bufferPrecision: BufferPrecision
    public let hidden: MTLBuffer
    public let residual: MTLBuffer
    public let scratch: MTLBuffer
    public let weights: [MTLBuffer]
    public let kvCache: MetalKVCache?
    public let convState: MTLBuffer?
    public let recurrentState: MTLBuffer?
    public let convStateDimension: Int
    public let convStateKernelSize: Int
    public let recurrentStateBytesPerLayer: Int
    public let perLayerInputs: MTLBuffer?
    public let perLayerInputDimension: Int
    public let perLayerInputLayerCount: Int
    public let logits: MTLBuffer
    public let tokenIDs: MTLBuffer
    public let positions: MTLBuffer
    public let ropePositionAxes: MTLBuffer
    public let tokenOut: MTLBuffer

    /// Scratch buffer for dequantized Q4→BF16 weights consumed by AMX matmul2d.
    /// Sized for the largest projection weight matrix: outputDim × inputDim × sizeof(bfloat16).
    /// Nil when no Q4 weight dequantization is needed (e.g., BF16 or FP16 models).
    public let dequantScratch: MTLBuffer?

    /// Shared buffer for runtime constants (sequenceLength, positions, etc.)
    /// that replace `setBytes()` calls in Metal 4 prefill encoding.
    ///
    /// Layout:
    /// - Offset 0: sequenceLength (UInt32)
    /// - Offset 4: hiddenConversionCount (UInt32)
    /// - Offset 8..<(8 + 4 * maxSeqLen): per-position values (UInt32 each)
    public let runtimeConstantBuffer: MTLBuffer

    public static let sequenceLengthOffset = 0
    public static let hiddenConversionCountOffset = 4
    public static let positionBaseOffset = 8

    public static func positionOffset(at index: Int) -> Int {
        positionBaseOffset + index * MemoryLayout<UInt32>.stride
    }

    public static func runtimeConstantBufferSize(maximumSequenceLength: Int) -> Int {
        positionBaseOffset + maximumSequenceLength * MemoryLayout<UInt32>.stride
    }

    var runtimeResidencyBuffers: [MTLBuffer] {
        var buffers: [MTLBuffer] = [
            hidden, residual, scratch, logits,
            tokenIDs, positions, ropePositionAxes, tokenOut, runtimeConstantBuffer,
        ]
        if let dequantScratch { buffers.append(dequantScratch) }
        if let convState { buffers.append(convState) }
        if let recurrentState { buffers.append(recurrentState) }
        if let perLayerInputs { buffers.append(perLayerInputs) }
        if let kvCache {
            buffers.append(kvCache.keys)
            buffers.append(kvCache.values)
            if let rotors = kvCache.rotorParameters { buffers.append(rotors) }
            if let qjl = kvCache.qjlMatrix { buffers.append(qjl) }
            if let qjlRes = kvCache.qjlResidualK { buffers.append(qjlRes) }
        }
        return buffers
    }

    var weightResidencyBuffers: [MTLBuffer] { weights }
}
