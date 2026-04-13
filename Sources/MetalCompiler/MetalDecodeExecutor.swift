import Metal

struct MetalDecodeExecutor: Sendable {

    // MARK: - Decode (Metal 4)

    func decodeSync(
        plan: MetalDispatchPlan,
        submission: inout MetalSubmissionContext,
        position: inout Int,
        tokenID: Int32,
        ropePositionAxes: (UInt32, UInt32, UInt32)? = nil
    ) -> Int32 {
        let buffers = plan.buffers
        buffers.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(position)
        writeRoPEPositionAxes(
            buffers: buffers,
            position: UInt32(position),
            ropePositionAxes: ropePositionAxes
        )
        buffers.tokenIn.contents().bindMemory(to: Int32.self, capacity: 1).pointee = tokenID

        do {
            try submission.withCompute { encoder, argumentTable in
                MetalDecodeEncoder.encodeSteps(
                    plan: plan,
                    encoder: encoder,
                    argumentTable: argumentTable
                )
            }
            position += 1
            return buffers.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
        } catch {
            print("[MetalInference] GPU error: \(error)")
            return -1
        }
    }

    /// Decode with GPU timing feedback for profiling.
    func decodeSyncTimed(
        plan: MetalDispatchPlan,
        submission: inout MetalSubmissionContext,
        position: inout Int,
        tokenID: Int32,
        ropePositionAxes: (UInt32, UInt32, UInt32)? = nil
    ) -> (token: Int32, gpuStartTime: CFTimeInterval, gpuEndTime: CFTimeInterval) {
        let buffers = plan.buffers
        buffers.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(position)
        writeRoPEPositionAxes(
            buffers: buffers,
            position: UInt32(position),
            ropePositionAxes: ropePositionAxes
        )
        buffers.tokenIn.contents().bindMemory(to: Int32.self, capacity: 1).pointee = tokenID

        do {
            let timing = try submission.withComputeTimed { encoder, argumentTable in
                MetalDecodeEncoder.encodeSteps(
                    plan: plan,
                    encoder: encoder,
                    argumentTable: argumentTable
                )
            }
            position += 1
            let token = buffers.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
            return (token, timing.gpuStartTime, timing.gpuEndTime)
        } catch {
            print("[MetalInference] GPU error: \(error)")
            return (-1, 0, 0)
        }
    }

    private func writeRoPEPositionAxes(
        buffers: MetalBufferSet,
        position: UInt32,
        ropePositionAxes: (UInt32, UInt32, UInt32)?
    ) {
        let values = ropePositionAxes ?? (position, position, position)
        let pointer = buffers.ropePositionAxes.contents().bindMemory(to: UInt32.self, capacity: 3)
        pointer[0] = values.0
        pointer[1] = values.1
        pointer[2] = values.2
    }
}
