import Metal

struct MetalSubmissionTransaction: @unchecked Sendable {
    let commandBuffer: MTLCommandBuffer

    func makeComputeEncoder() throws -> MTLComputeCommandEncoder {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create compute encoder")
        }
        return encoder
    }

    func makeBlitEncoder() throws -> MTLBlitCommandEncoder {
        guard let encoder = commandBuffer.makeBlitCommandEncoder() else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create blit encoder")
        }
        return encoder
    }

    func withComputeEncoder(
        _ encode: (MTLComputeCommandEncoder) throws -> Void
    ) throws {
        let encoder = try makeComputeEncoder()
        defer { encoder.endEncoding() }
        try encode(encoder)
    }

    func withBlitEncoder(
        _ encode: (MTLBlitCommandEncoder) throws -> Void
    ) throws {
        let encoder = try makeBlitEncoder()
        defer { encoder.endEncoding() }
        try encode(encoder)
    }
}
