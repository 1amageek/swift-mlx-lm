import Metal
import Foundation

struct MetalDecodeExecutor: Sendable {
    private static let debugLogURL = URL(fileURLWithPath: "/tmp/swiftlm-decode-steps.log")

    private var shouldDebugDecodeSteps: Bool {
        false
    }

    func decode(
        plan: MetalDispatchPlan,
        submission: MetalSubmissionContext,
        position: inout Int,
        pendingCommandBuffer: inout MTLCommandBuffer?,
        hasPendingResult: inout Bool,
        tokenID: Int32,
        ropePositionAxes: (UInt32, UInt32, UInt32)? = nil
    ) -> Int32 {
        let buffers = plan.buffers
        var result: Int32 = -1

        if hasPendingResult {
            result = consumePendingDecodeResult(
                plan: plan,
                submission: submission,
                pendingCommandBuffer: pendingCommandBuffer
            )
        }

        buffers.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(position)
        writeRoPEPositionAxes(
            buffers: buffers,
            position: UInt32(position),
            ropePositionAxes: ropePositionAxes
        )
        buffers.tokenIn.contents().bindMemory(to: Int32.self, capacity: 1).pointee = tokenID

        do {
            let commandBuffer = try submission.withCompute(label: "decode", waitUntilCompleted: false) { encoder in
                encodeSteps(plan: plan, on: encoder)
            }
            pendingCommandBuffer = commandBuffer
            hasPendingResult = true
            position += 1
        } catch {
            print("[MetalInference] Failed to submit decode: \(error)")
        }

        return result
    }

    func decodeSync(
        plan: MetalDispatchPlan,
        submission: MetalSubmissionContext,
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

        if shouldDebugDecodeSteps {
            return debugDecodeSync(
                plan: plan,
                submission: submission,
                position: &position,
                tokenID: tokenID
            )
        }

        do {
            _ = try submission.withCompute(label: "decode.sync") { encoder in
                encodeSteps(plan: plan, on: encoder)
            }
            position += 1
            return buffers.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
        } catch {
            print("[MetalInference] GPU error: \(error)")
            return -1
        }
    }

    private func debugDecodeSync(
        plan: MetalDispatchPlan,
        submission: MetalSubmissionContext,
        position: inout Int,
        tokenID: Int32
    ) -> Int32 {
        for (stepIndex, step) in plan.steps.enumerated() {
            let kernelName = step.metadata.kernelName ?? "(unknown)"
            let message = "[MetalInference][debug] decode step \(stepIndex): \(kernelName)"
            logDebugMessage(message)
            if kernelName == "ssm_recurrence" || kernelName == "ssm_recurrence_f32" {
                let bindingSummary = step.bufferBindings
                    .map { "\($0.index)=\($0.offset)" }
                    .joined(separator: ",")
                logDebugMessage("[MetalInference][debug] decode buffers \(stepIndex): \(bindingSummary)")
            }
            do {
                _ = try submission.withCompute(label: "decode.step.\(stepIndex).\(kernelName)") { encoder in
                    step.bindings.bind(to: encoder)
                    step.descriptor.encode(on: encoder)
                }
            } catch {
                logDebugMessage("[MetalInference][debug] decode step \(stepIndex) failed: \(error)")
                return -1
            }
        }

        position += 1
        return plan.buffers.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
    }

    private func logDebugMessage(_ message: String) {
        print(message)
        let data = Data((message + "\n").utf8)
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: Self.debugLogURL.path) {
            do {
                let handle = try FileHandle(forWritingTo: Self.debugLogURL)
                defer {
                    do {
                        try handle.close()
                    } catch {
                        print("[MetalInference][debug] failed to close log handle: \(error)")
                    }
                }
                try handle.seekToEnd()
                try handle.write(contentsOf: data)
            } catch {
                print("[MetalInference][debug] failed to write log: \(error)")
            }
        } else {
            do {
                try data.write(to: Self.debugLogURL)
            } catch {
                print("[MetalInference][debug] failed to create log: \(error)")
            }
        }
    }

    func flush(
        plan: MetalDispatchPlan,
        submission: MetalSubmissionContext,
        pendingCommandBuffer: inout MTLCommandBuffer?,
        hasPendingResult: inout Bool
    ) -> Int32 {
        guard hasPendingResult else {
            return -1
        }
        let result = consumePendingDecodeResult(
            plan: plan,
            submission: submission,
            pendingCommandBuffer: pendingCommandBuffer
        )
        hasPendingResult = false
        pendingCommandBuffer = nil
        return result
    }

    private func encodeSteps(plan: MetalDispatchPlan, on encoder: MTLComputeCommandEncoder) {
        for step in plan.steps {
            step.bindings.bind(to: encoder)
            step.descriptor.encode(on: encoder)
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

    private func consumePendingDecodeResult(
        plan: MetalDispatchPlan,
        submission: MetalSubmissionContext,
        pendingCommandBuffer: MTLCommandBuffer?
    ) -> Int32 {
        guard let pendingCommandBuffer else {
            return -1
        }
        do {
            try submission.waitUntilCompleted(pendingCommandBuffer)
            return plan.buffers.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
        } catch {
            print("[MetalInference] Pending decode failed: \(error)")
            return -1
        }
    }

    func logDebugStep(_ message: String) {
        guard shouldDebugDecodeSteps else { return }
        logDebugMessage(message)
    }
}
