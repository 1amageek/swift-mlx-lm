import Metal

struct MetalPromptStateStore: Sendable {

    func makePromptState(
        plan: MetalDispatchPlan,
        submission: MetalSubmissionContext,
        position: Int,
        firstToken: Int32
    ) throws -> MetalPromptState {
        let device = submission.device
        let snapshotHidden = try makePrivateBuffer(length: plan.buffers.hidden.length, device: device)
        let snapshotResidual = try makePrivateBuffer(length: plan.buffers.residual.length, device: device)
        let snapshotScratch = try makePrivateBuffer(length: plan.buffers.scratch.length, device: device)
        let snapshotLogits = try makePrivateBuffer(length: plan.buffers.logits.length, device: device)
        let snapshotPosition = try makePrivateBuffer(length: plan.buffers.position.length, device: device)
        let snapshotRoPEPositionAxes = try makePrivateBuffer(length: plan.buffers.ropePositionAxes.length, device: device)
        let snapshotTokenIn = try makePrivateBuffer(length: plan.buffers.tokenIn.length, device: device)
        let snapshotTokenOut = try makePrivateBuffer(length: plan.buffers.tokenOut.length, device: device)
        let snapshotKVKeys = try plan.buffers.kvCache.map {
            try makePrivateBuffer(length: $0.keys.length, device: device)
        }
        let snapshotKVValues = try plan.buffers.kvCache.map {
            try makePrivateBuffer(length: $0.values.length, device: device)
        }
        let snapshotConvState = try plan.buffers.convState.map {
            try makePrivateBuffer(length: $0.length, device: device)
        }
        let snapshotRecurrentState = try plan.buffers.recurrentState.map {
            try makePrivateBuffer(length: $0.length, device: device)
        }
        let snapshotPerLayerInputs = try plan.buffers.perLayerInputs.map {
            try makePrivateBuffer(length: $0.length, device: device)
        }

        _ = try submission.withTransaction(label: "prompt.snapshot") { transaction in
            try transaction.withBlitEncoder { blit in
                blit.copy(from: plan.buffers.hidden, sourceOffset: 0, to: snapshotHidden, destinationOffset: 0, size: plan.buffers.hidden.length)
                blit.copy(from: plan.buffers.residual, sourceOffset: 0, to: snapshotResidual, destinationOffset: 0, size: plan.buffers.residual.length)
                blit.copy(from: plan.buffers.scratch, sourceOffset: 0, to: snapshotScratch, destinationOffset: 0, size: plan.buffers.scratch.length)
                blit.copy(from: plan.buffers.logits, sourceOffset: 0, to: snapshotLogits, destinationOffset: 0, size: plan.buffers.logits.length)
                blit.copy(from: plan.buffers.position, sourceOffset: 0, to: snapshotPosition, destinationOffset: 0, size: plan.buffers.position.length)
                blit.copy(from: plan.buffers.ropePositionAxes, sourceOffset: 0, to: snapshotRoPEPositionAxes, destinationOffset: 0, size: plan.buffers.ropePositionAxes.length)
                blit.copy(from: plan.buffers.tokenIn, sourceOffset: 0, to: snapshotTokenIn, destinationOffset: 0, size: plan.buffers.tokenIn.length)
                blit.copy(from: plan.buffers.tokenOut, sourceOffset: 0, to: snapshotTokenOut, destinationOffset: 0, size: plan.buffers.tokenOut.length)
                if let liveKV = plan.buffers.kvCache,
                   let snapshotKVKeys,
                   let snapshotKVValues {
                    blit.copy(from: liveKV.keys, sourceOffset: 0, to: snapshotKVKeys, destinationOffset: 0, size: liveKV.keys.length)
                    blit.copy(from: liveKV.values, sourceOffset: 0, to: snapshotKVValues, destinationOffset: 0, size: liveKV.values.length)
                }
                if let liveConvState = plan.buffers.convState,
                   let snapshotConvState {
                    blit.copy(from: liveConvState, sourceOffset: 0, to: snapshotConvState, destinationOffset: 0, size: liveConvState.length)
                }
                if let liveRecurrentState = plan.buffers.recurrentState,
                   let snapshotRecurrentState {
                    blit.copy(from: liveRecurrentState, sourceOffset: 0, to: snapshotRecurrentState, destinationOffset: 0, size: liveRecurrentState.length)
                }
                if let livePerLayerInputs = plan.buffers.perLayerInputs,
                   let snapshotPerLayerInputs {
                    blit.copy(from: livePerLayerInputs, sourceOffset: 0, to: snapshotPerLayerInputs, destinationOffset: 0, size: livePerLayerInputs.length)
                }
            }
        }

        return MetalPromptState(
            position: position,
            firstToken: firstToken,
            hidden: snapshotHidden,
            residual: snapshotResidual,
            scratch: snapshotScratch,
            logits: snapshotLogits,
            positionBuffer: snapshotPosition,
            ropePositionAxes: snapshotRoPEPositionAxes,
            tokenIn: snapshotTokenIn,
            tokenOut: snapshotTokenOut,
            kvKeys: snapshotKVKeys,
            kvValues: snapshotKVValues,
            convState: snapshotConvState,
            recurrentState: snapshotRecurrentState,
            perLayerInputs: snapshotPerLayerInputs
        )
    }

    func restore(
        plan: MetalDispatchPlan,
        submission: MetalSubmissionContext,
        promptState: MetalPromptState
    ) throws {
        _ = try submission.withTransaction(label: "prompt.restore") { transaction in
            try transaction.withBlitEncoder { blit in
                blit.copy(from: promptState.hidden, sourceOffset: 0, to: plan.buffers.hidden, destinationOffset: 0, size: plan.buffers.hidden.length)
                blit.copy(from: promptState.residual, sourceOffset: 0, to: plan.buffers.residual, destinationOffset: 0, size: plan.buffers.residual.length)
                blit.copy(from: promptState.scratch, sourceOffset: 0, to: plan.buffers.scratch, destinationOffset: 0, size: plan.buffers.scratch.length)
                blit.copy(from: promptState.logits, sourceOffset: 0, to: plan.buffers.logits, destinationOffset: 0, size: plan.buffers.logits.length)
                blit.copy(from: promptState.positionBuffer, sourceOffset: 0, to: plan.buffers.position, destinationOffset: 0, size: plan.buffers.position.length)
                blit.copy(from: promptState.ropePositionAxes, sourceOffset: 0, to: plan.buffers.ropePositionAxes, destinationOffset: 0, size: plan.buffers.ropePositionAxes.length)
                blit.copy(from: promptState.tokenIn, sourceOffset: 0, to: plan.buffers.tokenIn, destinationOffset: 0, size: plan.buffers.tokenIn.length)
                blit.copy(from: promptState.tokenOut, sourceOffset: 0, to: plan.buffers.tokenOut, destinationOffset: 0, size: plan.buffers.tokenOut.length)

                if let liveKV = plan.buffers.kvCache,
                   let snapshotKVKeys = promptState.kvKeys,
                   let snapshotKVValues = promptState.kvValues {
                    blit.copy(from: snapshotKVKeys, sourceOffset: 0, to: liveKV.keys, destinationOffset: 0, size: liveKV.keys.length)
                    blit.copy(from: snapshotKVValues, sourceOffset: 0, to: liveKV.values, destinationOffset: 0, size: liveKV.values.length)
                }
                if let liveConvState = plan.buffers.convState,
                   let snapshotConvState = promptState.convState {
                    blit.copy(from: snapshotConvState, sourceOffset: 0, to: liveConvState, destinationOffset: 0, size: liveConvState.length)
                }
                if let liveRecurrentState = plan.buffers.recurrentState,
                   let snapshotRecurrentState = promptState.recurrentState {
                    blit.copy(from: snapshotRecurrentState, sourceOffset: 0, to: liveRecurrentState, destinationOffset: 0, size: liveRecurrentState.length)
                }
                if let livePerLayerInputs = plan.buffers.perLayerInputs,
                   let snapshotPerLayerInputs = promptState.perLayerInputs {
                    blit.copy(from: snapshotPerLayerInputs, sourceOffset: 0, to: livePerLayerInputs, destinationOffset: 0, size: livePerLayerInputs.length)
                }
            }
        }
    }

    private func makePrivateBuffer(length: Int, device: MTLDevice) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: length, options: .storageModePrivate) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate prompt state buffer")
        }
        return buffer
    }
}
