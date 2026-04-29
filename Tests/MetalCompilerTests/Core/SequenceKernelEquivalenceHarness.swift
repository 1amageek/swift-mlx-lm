import Metal
import Testing
@testable import MetalCompiler

enum SequenceKernelEquivalenceHarnessError: Error, CustomStringConvertible {
    case missingDevice
    case missingCommandQueue
    case missingCommandBuffer
    case missingEncoder
    case emptyBufferInput
    case commandBufferFailed(String)

    var description: String {
        switch self {
        case .missingDevice:
            "No Metal device is available"
        case .missingCommandQueue:
            "Unable to create Metal command queue"
        case .missingCommandBuffer:
            "Unable to create Metal command buffer"
        case .missingEncoder:
            "Unable to create Metal command encoder"
        case .emptyBufferInput:
            "Cannot create a typed buffer from an empty value array"
        case .commandBufferFailed(let message):
            "Metal command buffer failed: \(message)"
        }
    }
}

struct SequenceKernelEquivalenceHarness {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary

    init(device: MTLDevice) throws {
        try self.init(
            device: device,
            source: MetalSourceGenerator.generateCompleteLibrary(weightFormat: .bfloat16)
        )
    }

    init(device: MTLDevice, source: String) throws {
        self.device = device
        self.queue = try #require(device.makeCommandQueue())
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        self.library = try device.makeLibrary(
            source: source,
            options: options
        )
    }

    func pipeline(named name: String) throws -> MTLComputePipelineState {
        try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: name))
        )
    }

    func makeSharedBuffer<T>(values: [T]) throws -> MTLBuffer {
        guard !values.isEmpty else {
            throw SequenceKernelEquivalenceHarnessError.emptyBufferInput
        }
        var copy = values
        let length = copy.count * MemoryLayout<T>.stride
        return try copy.withUnsafeMutableBytes { rawBuffer in
            let base = try #require(rawBuffer.baseAddress)
            return try #require(
                device.makeBuffer(bytes: base, length: length, options: .storageModeShared)
            )
        }
    }

    func makeZeroedSharedBuffer(byteLength: Int) throws -> MTLBuffer {
        let buffer = try #require(
            device.makeBuffer(length: byteLength, options: .storageModeShared)
        )
        memset(buffer.contents(), 0, byteLength)
        return buffer
    }

    func makeCommandEncoder() throws -> (MTLCommandBuffer, MTLComputeCommandEncoder) {
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        return (commandBuffer, encoder)
    }

    func complete(_ commandBuffer: MTLCommandBuffer) throws {
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw SequenceKernelEquivalenceHarnessError.commandBufferFailed(error.localizedDescription)
        }
    }

    func readFloat32(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return (0..<count).map { pointer[$0] }
    }

    func readFloat16AsFloat(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(pointer[$0]) }
    }

    func readBFloat16AsFloat(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().bindMemory(to: BFloat16.self, capacity: count)
        return (0..<count).map { Float(pointer[$0]) }
    }

    func readBFloat16Bits(_ buffer: MTLBuffer, count: Int) -> [UInt16] {
        let pointer = buffer.contents().bindMemory(to: BFloat16.self, capacity: count)
        return (0..<count).map { pointer[$0].bitPattern }
    }

    func firstMismatch(
        expected: [Float],
        actual: [Float],
        tolerance: Float
    ) -> (index: Int, expected: Float, actual: Float, delta: Float)? {
        let count = min(expected.count, actual.count)
        for index in 0..<count {
            let delta = abs(expected[index] - actual[index])
            if delta > tolerance {
                return (index, expected[index], actual[index], delta)
            }
        }
        if expected.count != actual.count {
            return (count, Float(expected.count), Float(actual.count), .infinity)
        }
        return nil
    }

    func maxAbsoluteError(expected: [Float], actual: [Float]) -> Float {
        zip(expected, actual).reduce(Float.zero) { partial, pair in
            max(partial, abs(pair.0 - pair.1))
        }
    }
}
