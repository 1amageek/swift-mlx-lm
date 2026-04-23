import Metal

// MARK: - KV Cache Specification

/// Runtime specification for KV cache memory layout.
///
/// K and V are quantized independently — K tolerates aggressive quantization
/// (used for dot product), V requires conservative quantization (used for
/// weighted sum where outliers affect output quality).
///
/// The entire KV cache for all layers is consolidated into a single K buffer
/// and a single V buffer. Individual layers are accessed via offset.
///
/// See `STAF/KVCacheSpec.md` for the full specification.
public struct KVCacheSpecification: Sendable {

    /// Quantization scheme for K cache.
    public let keyQuantizationScheme: QuantizationSchemeIdentifier

    /// Quantization scheme for V cache.
    public let valueQuantizationScheme: QuantizationSchemeIdentifier

    /// Memory layout mode. Must match the flash attention kernel implementation.
    public let layoutMode: KVCacheLayoutMode

    /// Number of decoder layers.
    public let layerCount: Int

    /// Number of KV heads per layer (may differ from query heads in GQA/MQA).
    public let kvHeadCount: Int

    /// Dimension per head.
    public let headDimension: Int

    /// Maximum sequence length to allocate.
    public let maximumSequenceLength: Int

    public init(
        keyQuantizationScheme: QuantizationSchemeIdentifier = .fp16RowMajor,
        valueQuantizationScheme: QuantizationSchemeIdentifier = .fp16RowMajor,
        layoutMode: KVCacheLayoutMode = .sequenceMajor,
        layerCount: Int = 1,
        kvHeadCount: Int,
        headDimension: Int,
        maximumSequenceLength: Int
    ) {
        precondition(
            keyQuantizationScheme.isSupportedForKVCache,
            "KVCacheSpecification: keyQuantizationScheme \(keyQuantizationScheme) has no matching FlashAttention KV cache kernel path. Supported: fp16/bf16/fp32, q4g{64,128,128zero}, q8g{32,64,128}, rotorQ4g64, rotorQ8g32. Sizing would succeed but write/read would corrupt the cache."
        )
        precondition(
            valueQuantizationScheme.isSupportedForKVCache,
            "KVCacheSpecification: valueQuantizationScheme \(valueQuantizationScheme) has no matching FlashAttention KV cache kernel path. Supported: fp16/bf16/fp32, q4g{64,128,128zero}, q8g{32,64,128}, rotorQ4g64, rotorQ8g32. Sizing would succeed but write/read would corrupt the cache."
        )
        self.keyQuantizationScheme = keyQuantizationScheme
        self.valueQuantizationScheme = valueQuantizationScheme
        self.layoutMode = layoutMode
        self.layerCount = layerCount
        self.kvHeadCount = kvHeadCount
        self.headDimension = headDimension
        self.maximumSequenceLength = maximumSequenceLength
    }

    /// Resolve a scheme identifier to its block-layout `QuantizationFormat`.
    ///
    /// RotorQuant schemes share the block layout of their base scheme
    /// (`q8Group32ScaleF16` / `q4Group64ScaleF16`); rotation is a kernel-level
    /// concern and does not alter storage sizing. Unknown schemes terminate
    /// immediately rather than falling back to FP16 — a silent fallback here
    /// would hand out undersized buffers and cause out-of-bounds KV writes.
    private func resolveBlockLayout(
        for scheme: QuantizationSchemeIdentifier
    ) -> any QuantizationFormat {
        let base = scheme.baseScheme
        guard let format = QuantizationFormatRegistry.format(for: base) else {
            fatalError("KVCacheSpecification: no QuantizationFormat registered for scheme \(scheme) (base: \(base)). Register the format in QuantizationFormatRegistry before referencing it from an InferencePolicy.")
        }
        return format
    }

    /// Token slot alignment policy, expressed directly over the block format.
    ///
    /// Dense formats use 256B so that common `head_dim = 128` slots land on a
    /// cache-line-friendly natural boundary. Quantized block-packed formats use
    /// 64B — the block payloads (36B or 68B per group) fit comfortably within
    /// one cache line at this alignment.
    ///
    /// This is the single source of truth; the scheme-keyed overload and the
    /// byte-sizing path both route through here so the two stay in lockstep.
    private func tokenSlotAlignment(for format: any QuantizationFormat) -> Int {
        format.isQuantized ? 64 : 256
    }

    /// Token slot alignment in bytes for a given scheme.
    public func tokenSlotAlignment(scheme: QuantizationSchemeIdentifier) -> Int {
        tokenSlotAlignment(for: resolveBlockLayout(for: scheme))
    }

    /// Byte size of one head's data for one token in a given scheme.
    ///
    /// Derived directly from the format's block layout: each head stores
    /// `ceil(headDimension / weightsPerBlock)` blocks of `bytesPerBlock` bytes.
    /// For dense formats `weightsPerBlock == 1` and `bytesPerBlock` equals the
    /// element stride, so the expression collapses to `headDimension * stride`.
    /// For quantized formats the scale/zero header is already accounted for in
    /// `bytesPerBlock`. Result is rounded up to the token slot alignment for
    /// the same format — a single resolve so the alignment and byte math can
    /// never disagree.
    public func bytesPerHeadSlot(scheme: QuantizationSchemeIdentifier) -> Int {
        let format = resolveBlockLayout(for: scheme)
        let blockCount = (headDimension + format.weightsPerBlock - 1) / format.weightsPerBlock
        let rawBytes = blockCount * format.bytesPerBlock
        return alignUp(rawBytes, to: tokenSlotAlignment(for: format))
    }

    /// Byte size of one token slot (one token, all KV heads).
    public func bytesPerTokenSlot(scheme: QuantizationSchemeIdentifier) -> Int {
        kvHeadCount * bytesPerHeadSlot(scheme: scheme)
    }

    /// Total buffer size for K or V across all layers.
    public func totalBufferSize(scheme: QuantizationSchemeIdentifier) -> Int {
        layerCount * maximumSequenceLength * bytesPerTokenSlot(scheme: scheme)
    }

    /// Byte size of one layer's K or V cache.
    public func bytesPerLayer(scheme: QuantizationSchemeIdentifier) -> Int {
        maximumSequenceLength * bytesPerTokenSlot(scheme: scheme)
    }

    /// Byte offset of a specific layer within the consolidated buffer.
    public func layerOffset(layer: Int, scheme: QuantizationSchemeIdentifier) -> Int {
        layer * bytesPerLayer(scheme: scheme)
    }

    /// Byte offset for a specific (layer, head, position) in sequence-major layout.
    ///
    /// Layout: [layer][head][seq][dim]
    /// Each head has its own contiguous sequence of token slots.
    public func sequenceMajorOffset(
        layer: Int, head: Int, position: Int,
        scheme: QuantizationSchemeIdentifier
    ) -> Int {
        let headSlotSize = bytesPerHeadSlot(scheme: scheme)
        let headStride = maximumSequenceLength * headSlotSize        // 1 head's full sequence
        let layerStride = kvHeadCount * headStride                   // all heads in 1 layer
        return layer * layerStride + head * headStride + position * headSlotSize
    }

    /// Byte offset for a specific (layer, position, head) in head-major layout.
    ///
    /// Layout: [layer][seq][head][dim]
    /// All heads for one token are contiguous.
    public func headMajorOffset(
        layer: Int, position: Int, head: Int,
        scheme: QuantizationSchemeIdentifier
    ) -> Int {
        let headSlotSize = bytesPerHeadSlot(scheme: scheme)
        let tokenStride = kvHeadCount * headSlotSize                 // all heads for 1 token
        let layerStride = maximumSequenceLength * tokenStride        // all tokens in 1 layer
        return layer * layerStride + position * tokenStride + head * headSlotSize
    }

    /// Byte offset for a given (layer, head, position) using the configured layout mode.
    public func offset(
        layer: Int, head: Int, position: Int,
        scheme: QuantizationSchemeIdentifier
    ) -> Int {
        switch layoutMode {
        case .sequenceMajor:
            return sequenceMajorOffset(layer: layer, head: head, position: position, scheme: scheme)
        case .headMajor:
            return headMajorOffset(layer: layer, position: position, head: head, scheme: scheme)
        }
    }

    /// Number of Clifford Cl(3,0) rotor groups for RotorQuant.
    /// Each group covers 3 dimensions. The last group may be zero-padded.
    public var numRotorGroups: Int {
        (headDimension + 2) / 3
    }

    /// Whether either K or V uses a RotorQuant scheme.
    public var usesRotorQuant: Bool {
        keyQuantizationScheme.isRotorScheme || valueQuantizationScheme.isRotorScheme
    }

    /// Byte size of rotor parameters for one layer.
    /// Layout: [kvHeadCount × numRotorGroups × 4] half values.
    /// Each rotor has 4 components: [s, b₁₂, b₁₃, b₂₃].
    public var rotorParametersBytesPerLayer: Int {
        kvHeadCount * numRotorGroups * 4 * MemoryLayout<UInt16>.size
    }

    /// Total byte size of rotor parameters across all layers.
    public var totalRotorParametersSize: Int {
        layerCount * rotorParametersBytesPerLayer
    }

    /// Byte size of QJL projected residuals for one layer.
    /// Layout: [maxSeqLen × kvHeadCount × qjlDim] half values.
    public func qjlResidualBytesPerLayer(qjlDimension: Int) -> Int {
        guard qjlDimension > 0 else { return 0 }
        return maximumSequenceLength * kvHeadCount * qjlDimension * MemoryLayout<UInt16>.size
    }

    /// Total byte size of QJL projected residuals across all layers.
    public func totalQJLResidualSize(qjlDimension: Int) -> Int {
        layerCount * qjlResidualBytesPerLayer(qjlDimension: qjlDimension)
    }

    /// Byte size of the shared QJL random projection matrix.
    /// Layout: [headDim × qjlDim] half values.
    public func qjlMatrixSize(qjlDimension: Int) -> Int {
        guard qjlDimension > 0 else { return 0 }
        return headDimension * qjlDimension * MemoryLayout<UInt16>.size
    }

    private func alignUp(_ value: Int, to alignment: Int) -> Int {
        let remainder = value % alignment
        return remainder == 0 ? value : value + (alignment - remainder)
    }
}

/// KV cache memory layout mode.
///
/// Must match the flash attention kernel implementation.
/// Changing layout_mode requires a paired kernel change.
public enum KVCacheLayoutMode: UInt8, Sendable {
    /// [layer][head][seq][dim] — decode append is contiguous write.
    case sequenceMajor = 0x00
    /// [layer][seq][head][dim] — GQA/MQA parallel head read.
    case headMajor = 0x01
}

// MARK: - KV Cache

/// Consolidated KV cache: single K buffer + single V buffer for all layers.
///
/// The entire cache is pre-allocated. Layers are accessed via offset.
/// K and V are separate buffers to allow independent quantization.
///
/// For RotorQuant schemes, additional buffers store per-group Clifford Cl(3,0)
/// rotor parameters and (optionally) QJL projected residuals for inner product
/// correction.
public struct MetalKVCache: @unchecked Sendable {
    /// K cache buffer (all layers consolidated).
    public let keys: MTLBuffer
    /// V cache buffer (all layers consolidated).
    public let values: MTLBuffer
    /// Specification that governs layout, quantization, and sizing.
    public let specification: KVCacheSpecification
    /// Current number of tokens in cache.
    public var length: Int

    // MARK: - RotorQuant State

    /// Per-layer per-head Clifford rotor parameters.
    /// Layout: [layer × kvHead × numGroups × 4] half values.
    /// Each rotor: [s, b₁₂, b₁₃, b₂₃] in Cl(3,0).
    /// nil when neither K nor V uses a RotorQuant scheme.
    public let rotorParameters: MTLBuffer?

    /// Shared QJL random projection matrix (Rademacher ±1/√m).
    /// Layout: [headDim × qjlDim] half values.
    /// nil when qjlDimension == 0.
    public let qjlMatrix: MTLBuffer?

    /// Per-layer per-token projected K quantization residuals.
    /// Layout: [layer × maxSeqLen × kvHead × qjlDim] half values.
    /// nil when qjlDimension == 0.
    public let qjlResidualK: MTLBuffer?

    /// Number of Cl(3,0) rotor groups per head (⌈headDim/3⌉). 0 for non-rotor.
    public let numRotorGroups: Int

    /// QJL projection dimension. 0 = disabled.
    public let qjlDimension: Int

    public init(
        device: MTLDevice,
        specification: KVCacheSpecification,
        qjlDimension: Int = 0,
        resourceOptions: MTLResourceOptions = [.storageModePrivate, .hazardTrackingModeUntracked]
    ) throws {
        let keySize = specification.totalBufferSize(scheme: specification.keyQuantizationScheme)
        let valueSize = specification.totalBufferSize(scheme: specification.valueQuantizationScheme)

        guard let k = device.makeBuffer(length: keySize, options: resourceOptions),
              let v = device.makeBuffer(length: valueSize, options: resourceOptions)
        else {
            throw MetalCompilerError.bufferAllocationFailed("Cannot allocate KV cache")
        }
        self.keys = k
        self.values = v
        self.specification = specification
        self.length = 0
        self.qjlDimension = qjlDimension

        // Allocate RotorQuant buffers if needed
        if specification.usesRotorQuant {
            let numGroups = specification.numRotorGroups
            self.numRotorGroups = numGroups

            // Rotor parameters: CPU-writable for initialization, GPU-readable
            let rotorSize = specification.totalRotorParametersSize
            guard let rotorBuf = device.makeBuffer(length: max(rotorSize, 16), options: .storageModeShared) else {
                throw MetalCompilerError.bufferAllocationFailed("Cannot allocate rotor parameters")
            }
            // Initialize to random unit rotors for PolarQuant outlier spreading
            Self.initializeRandomUnitRotors(
                buffer: rotorBuf,
                layerCount: specification.layerCount,
                kvHeadCount: specification.kvHeadCount,
                numGroups: numGroups
            )
            self.rotorParameters = rotorBuf

            // QJL buffers
            if qjlDimension > 0 {
                let matrixSize = specification.qjlMatrixSize(qjlDimension: qjlDimension)
                guard let matBuf = device.makeBuffer(length: max(matrixSize, 16), options: .storageModeShared) else {
                    throw MetalCompilerError.bufferAllocationFailed("Cannot allocate QJL matrix")
                }
                Self.initializeRademacherMatrix(
                    buffer: matBuf,
                    headDimension: specification.headDimension,
                    qjlDimension: qjlDimension
                )
                self.qjlMatrix = matBuf

                let residualSize = specification.totalQJLResidualSize(qjlDimension: qjlDimension)
                guard let resBuf = device.makeBuffer(length: max(residualSize, 16), options: resourceOptions) else {
                    throw MetalCompilerError.bufferAllocationFailed("Cannot allocate QJL residual buffer")
                }
                self.qjlResidualK = resBuf
            } else {
                self.qjlMatrix = nil
                self.qjlResidualK = nil
            }
        } else {
            self.numRotorGroups = 0
            self.rotorParameters = nil
            self.qjlMatrix = nil
            self.qjlResidualK = nil
        }
    }

    /// Initialize rotor parameters to random unit quaternions using deterministic LCG hash.
    ///
    /// Each rotor [s, b₁₂, b₁₃, b₂₃] satisfies s² + b₁₂² + b₁₃² + b₂₃² = 1.
    /// The same (layerCount, kvHeadCount, numGroups) always produces the same rotors.
    private static func initializeRandomUnitRotors(
        buffer: MTLBuffer,
        layerCount: Int,
        kvHeadCount: Int,
        numGroups: Int
    ) {
        let ptr = buffer.contents().bindMemory(
            to: UInt16.self,
            capacity: layerCount * kvHeadCount * numGroups * 4
        )
        let totalRotors = layerCount * kvHeadCount * numGroups
        for i in 0..<totalRotors {
            // 4 sequential LCG hashes for 4 components
            var hash = UInt64(i) &* 6364136223846793005 &+ 1442695040888963407
            let raw0 = hash
            hash = hash &* 6364136223846793005 &+ 1442695040888963407
            let raw1 = hash
            hash = hash &* 6364136223846793005 &+ 1442695040888963407
            let raw2 = hash
            hash = hash &* 6364136223846793005 &+ 1442695040888963407
            let raw3 = hash

            // Map to [-1, 1] in Float32
            let scale: Float = 1.0 / Float(1 << 30)
            var s   = Float(Int64(bitPattern: raw0 >> 33) - (1 << 30)) * scale
            var b12 = Float(Int64(bitPattern: raw1 >> 33) - (1 << 30)) * scale
            var b13 = Float(Int64(bitPattern: raw2 >> 33) - (1 << 30)) * scale
            var b23 = Float(Int64(bitPattern: raw3 >> 33) - (1 << 30)) * scale

            // Normalize to unit quaternion
            let norm = (s * s + b12 * b12 + b13 * b13 + b23 * b23).squareRoot()
            if norm < 1e-8 {
                s = 1.0; b12 = 0.0; b13 = 0.0; b23 = 0.0
            } else {
                let invNorm = 1.0 / norm
                s *= invNorm; b12 *= invNorm; b13 *= invNorm; b23 *= invNorm
            }

            ptr[i * 4 + 0] = Float16(s).bitPattern
            ptr[i * 4 + 1] = Float16(b12).bitPattern
            ptr[i * 4 + 2] = Float16(b13).bitPattern
            ptr[i * 4 + 3] = Float16(b23).bitPattern
        }
    }

    /// Initialize QJL matrix with Rademacher distribution: ±1/√m.
    private static func initializeRademacherMatrix(
        buffer: MTLBuffer,
        headDimension: Int,
        qjlDimension: Int
    ) {
        let ptr = buffer.contents().bindMemory(
            to: UInt16.self,
            capacity: headDimension * qjlDimension
        )
        let scale = 1.0 / Float(qjlDimension).squareRoot()
        let posVal = Float16(scale).bitPattern
        let negVal = Float16(-scale).bitPattern
        // Deterministic pseudo-random via simple hash
        for i in 0..<(headDimension * qjlDimension) {
            let hash = UInt64(i) &* 6364136223846793005 &+ 1442695040888963407
            ptr[i] = (hash >> 33) & 1 == 0 ? posVal : negVal
        }
    }

    /// K cache byte offset for a given (layer, head, position).
    public func keyOffset(layer: Int, head: Int, position: Int) -> Int {
        specification.offset(
            layer: layer, head: head, position: position,
            scheme: specification.keyQuantizationScheme)
    }

    /// V cache byte offset for a given (layer, head, position).
    public func valueOffset(layer: Int, head: Int, position: Int) -> Int {
        specification.offset(
            layer: layer, head: head, position: position,
            scheme: specification.valueQuantizationScheme)
    }

    /// Rotor parameter byte offset for a given layer.
    public func rotorParameterOffset(layer: Int) -> Int {
        layer * specification.rotorParametersBytesPerLayer
    }

    /// QJL residual byte offset for a given layer.
    public func qjlResidualOffset(layer: Int) -> Int {
        layer * specification.qjlResidualBytesPerLayer(qjlDimension: qjlDimension)
    }
}

// MARK: - Errors

public enum MetalCompilerError: Error, CustomStringConvertible {
    case deviceSetupFailed(String)
    case kernelNotFound(String)
    case bufferAllocationFailed(String)
    case compilationFailed(String)
    case unsupportedOperation(String)
    case weightNotFound(String)

    public var description: String {
        switch self {
        case .deviceSetupFailed(let message):
            return "MetalCompilerError.deviceSetupFailed: \(message)"
        case .kernelNotFound(let message):
            return "MetalCompilerError.kernelNotFound: \(message)"
        case .bufferAllocationFailed(let message):
            return "MetalCompilerError.bufferAllocationFailed: \(message)"
        case .compilationFailed(let message):
            return "MetalCompilerError.compilationFailed: \(message)"
        case .unsupportedOperation(let message):
            return "MetalCompilerError.unsupportedOperation: \(message)"
        case .weightNotFound(let message):
            return "MetalCompilerError.weightNotFound: \(message)"
        }
    }
}
