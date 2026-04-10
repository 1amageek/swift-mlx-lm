import Foundation
import Metal
import LMIR

// MARK: - Safetensors Format Types

/// Parsed tensor metadata from safetensors JSON header.
public struct SafetensorsTensorInfo: Sendable {
    public let name: String
    public let dtype: SafetensorsDType
    public let shape: [Int]
    /// Byte offset from the start of the tensor data section.
    public let dataOffset: Int
    /// Byte count of the tensor data.
    public let byteCount: Int
}

/// Data types supported by safetensors format.
public enum SafetensorsDType: String, Sendable {
    case float16 = "F16"
    case bfloat16 = "BF16"
    case float32 = "F32"
    case float64 = "F64"
    case int8 = "I8"
    case int16 = "I16"
    case int32 = "I32"
    case int64 = "I64"
    case uint8 = "U8"
    case uint16 = "U16"
    case uint32 = "U32"

    /// Bytes per element.
    public var elementSize: Int {
        switch self {
        case .float16, .bfloat16, .int16, .uint16: return 2
        case .float32, .int32, .uint32: return 4
        case .float64, .int64: return 8
        case .int8, .uint8: return 1
        }
    }

    /// Convert to SwiftLM DTypeHint.
    public var dtypeHint: DTypeHint {
        switch self {
        case .float16: return .float16
        case .bfloat16: return .bfloat16
        case .float32: return .float32
        default: return .float16
        }
    }
}

// MARK: - Safetensors Parser

/// Parses safetensors files and loads tensor data directly into MTLBuffers.
///
/// safetensors format:
/// ```
/// [8 bytes: header_size as u64 LE]
/// [header_size bytes: JSON header]
/// [tensor data bytes]
/// ```
///
/// The JSON header maps tensor names to `{dtype, shape, data_offsets: [begin, end]}`.
public struct SafetensorsLoader: Sendable {

    public init() {}

    private func resolvedFileURL(for url: URL) -> URL {
        url.resolvingSymlinksInPath()
    }

    /// Parse the safetensors header without loading tensor data.
    ///
    /// - Parameter url: Path to the safetensors file.
    /// - Returns: Array of tensor metadata entries.
    public func parseHeader(at url: URL) throws -> [SafetensorsTensorInfo] {
        let resolvedURL = resolvedFileURL(for: url)
        let fileHandle = try FileHandle(forReadingFrom: resolvedURL)
        defer { fileHandle.closeFile() }

        // Read 8-byte header size (little-endian u64)
        guard let sizeData = try fileHandle.read(upToCount: 8), sizeData.count == 8 else {
            throw SafetensorsError.invalidHeader("Cannot read header size")
        }
        let headerSizeRaw = sizeData.withUnsafeBytes { $0.load(as: UInt64.self) }
        let headerSizeInt = Int(UInt64(littleEndian: headerSizeRaw))

        guard headerSizeInt > 0, headerSizeInt < 100_000_000 else {
            throw SafetensorsError.invalidHeader("Header size \(headerSizeInt) out of range")
        }

        // Read JSON header
        guard let headerData = try fileHandle.read(upToCount: headerSizeInt),
              headerData.count == headerSizeInt else {
            throw SafetensorsError.invalidHeader("Cannot read header data")
        }

        guard let json = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw SafetensorsError.invalidHeader("Header is not a JSON object")
        }

        var tensors: [SafetensorsTensorInfo] = []

        for (name, value) in json {
            // Skip metadata entry
            if name == "__metadata__" { continue }

            guard let tensorDict = value as? [String: Any] else {
                throw SafetensorsError.invalidTensorEntry(name, "Not a dictionary")
            }

            guard let dtypeStr = tensorDict["dtype"] as? String,
                  let dtype = SafetensorsDType(rawValue: dtypeStr) else {
                throw SafetensorsError.invalidTensorEntry(name, "Missing or unknown dtype")
            }

            guard let shapeArray = tensorDict["shape"] as? [Int] else {
                throw SafetensorsError.invalidTensorEntry(name, "Missing shape")
            }

            guard let offsets = tensorDict["data_offsets"] as? [Int],
                  offsets.count == 2 else {
                throw SafetensorsError.invalidTensorEntry(name, "Missing data_offsets")
            }

            let byteCount = offsets[1] - offsets[0]
            tensors.append(SafetensorsTensorInfo(
                name: name,
                dtype: dtype,
                shape: shapeArray,
                dataOffset: offsets[0],
                byteCount: byteCount
            ))
        }

        return tensors
    }

    /// Load a safetensors file into a single MTLBuffer via zero-copy mmap.
    ///
    /// Uses `makeBuffer(bytesNoCopy:)` to wrap the mmap'd file directly as a
    /// Metal buffer — no memcpy. The GPU reads weight data directly from the
    /// mmap'd region. This follows llama.cpp's approach.
    ///
    /// Page alignment: `bytesNoCopy` requires both pointer and length to be
    /// page-aligned. mmap guarantees page-aligned start. The data section
    /// may not start on a page boundary, so we align the pointer back to
    /// the page boundary and store the alignment offset. Tensor offsets are
    /// adjusted by this amount.
    ///
    /// - Parameters:
    ///   - url: Path to the safetensors file.
    ///   - device: Metal device for buffer creation.
    /// - Returns: File-backed buffer and tensor metadata.
    public func load(
        at url: URL, device: MTLDevice
    ) throws -> MetalWeightFile {
        let tensors = try parseHeader(at: url)
        let resolvedURL = resolvedFileURL(for: url)

        // Compute data section offset (8 bytes header size + header JSON)
        let fileHandle = try FileHandle(forReadingFrom: resolvedURL)
        defer { fileHandle.closeFile() }

        guard let sizeData2 = try fileHandle.read(upToCount: 8), sizeData2.count == 8 else {
            throw SafetensorsError.invalidHeader("Cannot read header size")
        }
        let headerSize = sizeData2.withUnsafeBytes { Int(UInt64(littleEndian: $0.load(as: UInt64.self))) }
        let dataSectionOffset = 8 + headerSize

        // Get file size
        let fileAttributes = try FileManager.default.attributesOfItem(atPath: resolvedURL.path)
        guard let fileSize = fileAttributes[.size] as? Int else {
            throw SafetensorsError.invalidHeader("Cannot determine file size")
        }
        let dataSize = fileSize - dataSectionOffset

        guard dataSize > 0 else {
            throw SafetensorsError.invalidHeader("No tensor data in file")
        }

        // mmap the entire file (read-only, private mapping)
        let fileDescriptor = open(resolvedURL.path, O_RDONLY)
        guard fileDescriptor >= 0 else {
            throw SafetensorsError.mmapFailed("Cannot open file: \(resolvedURL.path)")
        }

        let mmapPointer = mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, fileDescriptor, 0)
        close(fileDescriptor)

        guard let validPointer = mmapPointer, validPointer != MAP_FAILED else {
            throw SafetensorsError.mmapFailed("mmap failed for: \(url.path)")
        }

        // Page-align the data section pointer (llama.cpp approach)
        let pageSize = Int(sysconf(_SC_PAGESIZE))
        let dataPointer = validPointer + dataSectionOffset
        let alignmentOffset = Int(bitPattern: dataPointer) % pageSize
        let alignedPointer = dataPointer - alignmentOffset
        let alignedSize = pageAlign(dataSize + alignmentOffset, pageSize: pageSize)

        // Zero-copy MTLBuffer: GPU reads directly from mmap'd memory.
        // storageModeShared is required for bytesNoCopy on Apple Silicon.
        // The deallocator unmaps the entire file when the buffer is released.
        let mmapSize = fileSize  // capture for deallocator
        guard let buffer = device.makeBuffer(
            bytesNoCopy: alignedPointer,
            length: alignedSize,
            options: .storageModeShared,
            deallocator: { pointer, _ in
                munmap(pointer, mmapSize)
            }
        ) else {
            munmap(validPointer, fileSize)
            throw SafetensorsError.bufferCreationFailed("MTLBuffer bytesNoCopy failed")
        }

        // Build tensor lookup — adjust offsets by alignment
        var tensorMap: [String: SafetensorsTensorInfo] = [:]
        for tensor in tensors {
            tensorMap[tensor.name] = tensor
        }

        return MetalWeightFile(
            buffer: buffer,
            tensors: tensorMap,
            dataSectionOffset: alignmentOffset  // tensor offset += alignmentOffset
        )
    }

    /// Round up to the nearest page boundary.
    private func pageAlign(_ size: Int, pageSize: Int) -> Int {
        let remainder = size % pageSize
        return remainder == 0 ? size : size + (pageSize - remainder)
    }

    /// Load multiple safetensors files into a MetalWeightStore.
    ///
    /// - Parameters:
    ///   - urls: Paths to safetensors files.
    ///   - device: Metal device for buffer creation.
    /// - Returns: Unified weight store with all tensors accessible by name.
    public func loadAll(
        urls: [URL], device: MTLDevice
    ) throws -> MetalWeightStore {
        var files: [MetalWeightFile] = []
        var globalTensorMap: [String: (fileIndex: Int, info: SafetensorsTensorInfo)] = [:]

        for (index, url) in urls.enumerated() {
            let file = try load(at: url, device: device)
            for (name, info) in file.tensors {
                globalTensorMap[name] = (fileIndex: index, info: info)
            }
            files.append(file)
        }

        return MetalWeightStore(files: files, tensorMap: globalTensorMap)
    }
}

// MARK: - Weight Storage Types

/// A single safetensors file loaded into a MTLBuffer.
public struct MetalWeightFile: @unchecked Sendable {
    /// MTLBuffer containing all tensor data from this file.
    public let buffer: MTLBuffer
    /// Tensor metadata indexed by name.
    public let tensors: [String: SafetensorsTensorInfo]
    /// Page alignment offset added to all tensor data offsets.
    /// With zero-copy mmap, the buffer may start before the data section
    /// to satisfy page alignment. This offset must be added to each
    /// tensor's dataOffset for correct addressing.
    public let dataSectionOffset: Int
}

/// Unified weight store across multiple safetensors files.
public struct MetalWeightStore: @unchecked Sendable {
    /// Loaded safetensors files.
    public let files: [MetalWeightFile]
    /// Global tensor lookup: name → (file index, tensor info).
    public let tensorMap: [String: (fileIndex: Int, info: SafetensorsTensorInfo)]

    var residencyCandidateBuffers: [MTLBuffer] {
        files.map(\.buffer)
    }

    /// Get the MTLBuffer and offset for a named tensor.
    ///
    /// - Parameter name: Tensor name from safetensors.
    /// - Returns: Buffer and byte offset, or nil if not found.
    public func buffer(for name: String) -> (buffer: MTLBuffer, offset: Int)? {
        guard let entry = tensorMap[name] else { return nil }
        let file = files[entry.fileIndex]
        return (buffer: file.buffer, offset: file.dataSectionOffset + entry.info.dataOffset)
    }

    /// Get tensor info by name.
    public func tensorInfo(for name: String) -> SafetensorsTensorInfo? {
        tensorMap[name]?.info
    }

    /// Get a MetalTensor for a named weight.
    ///
    /// For dense weights, returns a MetalTensor with the appropriate dtype.
    /// For quantized weights (detected by presence of companion .scales/.biases tensors),
    /// returns a MetalTensor with QuantizationDescriptor.
    ///
    /// - Parameter name: Tensor name (e.g., "model.layers.0.self_attn.q_proj.weight")
    /// - Returns: MetalTensor or nil if not found.
    public func tensor(for name: String) -> MetalTensor? {
        guard let entry = tensorMap[name] else { return nil }
        let file = files[entry.fileIndex]
        let info = entry.info

        // Check for quantization: look for companion .scales and .biases tensors
        let modulePath = name.hasSuffix(".weight")
            ? String(name.dropLast(".weight".count))
            : name
        let scalesName = modulePath + ".scales"
        let biasesName = modulePath + ".biases"

        if let scalesEntry = tensorMap[scalesName],
           let biasesEntry = tensorMap[biasesName] {
            // Quantized weight
            let scalesFile = files[scalesEntry.fileIndex]
            let biasesFile = files[biasesEntry.fileIndex]

            // Determine bits from element packing
            // MLX packs into uint32: elements_per_int = 32 / bits
            // packed shape [N, inFeatures / elements_per_int]
            // scales shape [N, numGroups] where numGroups = inFeatures / groupSize
            let scalesInfo = scalesEntry.info
            let groupSize: Int
            if scalesInfo.shape.count >= 2 && info.shape.count >= 2 {
                let numGroups = scalesInfo.shape[scalesInfo.shape.count - 1]
                // inFeatures = packed_dim * elements_per_int
                // groupSize = inFeatures / numGroups
                // For now, estimate: common group sizes are 32, 64, 128
                let packedDim = info.shape[info.shape.count - 1]
                // Try bits = 4 first (most common)
                let elementsPerInt = 8  // 32 bits / 4 bits
                let inFeatures = packedDim * elementsPerInt
                groupSize = numGroups > 0 ? inFeatures / numGroups : 64
            } else {
                groupSize = 64
            }

            let desc = QuantizationDescriptor(
                bits: 4,  // TODO: detect from tensor shape ratio
                groupSize: groupSize,
                scales: MetalTensorRef(buffer: scalesFile.buffer, offset: scalesFile.dataSectionOffset + scalesInfo.dataOffset),
                zeros: MetalTensorRef(buffer: biasesFile.buffer, offset: biasesFile.dataSectionOffset + biasesEntry.info.dataOffset)
            )

            return MetalTensor(
                buffer: file.buffer,
                offset: file.dataSectionOffset + info.dataOffset,
                shape: info.shape,
                dtype: .quantized(desc)
            )
        }

        // Dense weight
        let dtype: MetalTensorDType
        switch info.dtype {
        case .float16: dtype = .float16
        case .bfloat16: dtype = .bfloat16
        case .float32: dtype = .float32
        default: dtype = .float16
        }

        return MetalTensor(
            buffer: file.buffer,
            offset: file.dataSectionOffset + info.dataOffset,
            shape: info.shape,
            dtype: dtype
        )
    }
}

// MARK: - Errors

public enum SafetensorsError: Error, CustomStringConvertible {
    case invalidHeader(String)
    case invalidTensorEntry(String, String)
    case mmapFailed(String)
    case bufferCreationFailed(String)
    case tensorNotFound(String)

    public var description: String {
        switch self {
        case .invalidHeader(let msg): return "SafetensorsError.invalidHeader: \(msg)"
        case .invalidTensorEntry(let name, let msg): return "SafetensorsError.invalidTensorEntry(\(name)): \(msg)"
        case .mmapFailed(let msg): return "SafetensorsError.mmapFailed: \(msg)"
        case .bufferCreationFailed(let msg): return "SafetensorsError.bufferCreationFailed: \(msg)"
        case .tensorNotFound(let name): return "SafetensorsError.tensorNotFound: \(name)"
        }
    }
}
