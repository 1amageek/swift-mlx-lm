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
        self.keyQuantizationScheme = keyQuantizationScheme
        self.valueQuantizationScheme = valueQuantizationScheme
        self.layoutMode = layoutMode
        self.layerCount = layerCount
        self.kvHeadCount = kvHeadCount
        self.headDimension = headDimension
        self.maximumSequenceLength = maximumSequenceLength
    }

    /// Token slot alignment in bytes, determined by quantization scheme.
    ///
    /// FP16: 256B (head_dim=128 → 256B natural)
    /// Q8:    64B (block fits in 64B)
    /// Q4:    64B (block fits in 64B)
    public func tokenSlotAlignment(scheme: QuantizationSchemeIdentifier) -> Int {
        switch scheme {
        case .fp16RowMajor, .bf16RowMajor, .fp32RowMajor:
            return 256
        default:
            return 64  // quantized blocks fit in 64B alignment
        }
    }

    /// Byte size of one head's data for one token in a given scheme.
    ///
    /// This is the per-head slot size. Includes scale/zero overhead for quantized schemes.
    /// For a full token (all heads), multiply by kvHeadCount.
    public func bytesPerHeadSlot(scheme: QuantizationSchemeIdentifier) -> Int {
        let rawBytes: Int
        switch scheme {
        case .fp16RowMajor:
            rawBytes = headDimension * 2
        case .bf16RowMajor:
            rawBytes = headDimension * 2
        case .fp32RowMajor:
            rawBytes = headDimension * 4
        case .q8Group32ScaleF16:
            // Per group: 4B header (scale+zero) + 32B int8 = 36B per 32 elements
            let groups = (headDimension + 31) / 32
            rawBytes = groups * 36
        case .q8Group64ScaleF16:
            let groups = (headDimension + 63) / 64
            rawBytes = groups * 68
        case .q4Group64ScaleF16:
            // Per group: 4B header (scale+zero) + 32B packed 4-bit = 36B per 64 elements
            let groups = (headDimension + 63) / 64
            rawBytes = groups * 36
        case .q4Group128ScaleF16Zero:
            let groups = (headDimension + 127) / 128
            rawBytes = groups * 68
        default:
            rawBytes = headDimension * 2  // fallback FP16
        }
        let alignment = tokenSlotAlignment(scheme: scheme)
        return alignUp(rawBytes, to: alignment)
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
public struct MetalKVCache: @unchecked Sendable {
    /// K cache buffer (all layers consolidated).
    public let keys: MTLBuffer
    /// V cache buffer (all layers consolidated).
    public let values: MTLBuffer
    /// Specification that governs layout, quantization, and sizing.
    public let specification: KVCacheSpecification
    /// Current number of tokens in cache.
    public var length: Int

    public init(
        device: MTLDevice,
        specification: KVCacheSpecification,
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

    /// Whether K cache uses quantization (not FP16).
    public var isKeyQuantized: Bool {
        specification.keyQuantizationScheme != .fp16RowMajor
    }

    /// Whether V cache uses quantization (not FP16).
    public var isValueQuantized: Bool {
        specification.valueQuantizationScheme != .fp16RowMajor
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
