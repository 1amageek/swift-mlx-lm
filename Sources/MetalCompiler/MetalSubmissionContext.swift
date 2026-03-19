import Metal

struct MetalSubmissionContext: Sendable {
    let commandQueue: MTLCommandQueue

    var device: MTLDevice {
        commandQueue.device
    }

    func makeCommandBuffer(label: String? = nil) throws -> MTLCommandBuffer {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw MetalCompilerError.deviceSetupFailed("Cannot create command buffer")
        }
        commandBuffer.label = label
        return commandBuffer
    }

    func makeTransaction(label: String? = nil) throws -> MetalSubmissionTransaction {
        MetalSubmissionTransaction(commandBuffer: try makeCommandBuffer(label: label))
    }

    func commit(_ commandBuffer: MTLCommandBuffer, waitUntilCompleted shouldWaitUntilCompleted: Bool) throws {
        commandBuffer.commit()
        guard shouldWaitUntilCompleted else { return }

        try waitUntilCompleted(commandBuffer)
    }

    func waitUntilCompleted(_ commandBuffer: MTLCommandBuffer) throws {
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            let labelSuffix: String
            if let label = commandBuffer.label, !label.isEmpty {
                labelSuffix = " [\(label)]"
            } else {
                labelSuffix = ""
            }
            throw MetalCompilerError.deviceSetupFailed("GPU submission failed\(labelSuffix): \(error.localizedDescription)")
        }
    }

    func withBlit(
        label: String? = nil,
        waitUntilCompleted: Bool = true,
        _ encode: (MTLBlitCommandEncoder) throws -> Void
    ) throws {
        let transaction = try makeTransaction(label: label)
        try transaction.withBlitEncoder(encode)
        try commit(transaction.commandBuffer, waitUntilCompleted: waitUntilCompleted)
    }

    func withCompute(
        label: String? = nil,
        waitUntilCompleted: Bool = true,
        _ encode: (MTLComputeCommandEncoder) throws -> Void
    ) throws -> MTLCommandBuffer {
        let transaction = try makeTransaction(label: label)
        try transaction.withComputeEncoder(encode)
        try commit(transaction.commandBuffer, waitUntilCompleted: waitUntilCompleted)
        return transaction.commandBuffer
    }

    func withTransaction(
        label: String? = nil,
        waitUntilCompleted: Bool = true,
        _ encode: (MetalSubmissionTransaction) throws -> Void
    ) throws -> MTLCommandBuffer {
        let transaction = try makeTransaction(label: label)
        try encode(transaction)
        try commit(transaction.commandBuffer, waitUntilCompleted: waitUntilCompleted)
        return transaction.commandBuffer
    }
}
