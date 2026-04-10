import Metal

struct MetalPromptStateStore: Sendable {

    func makePromptState(
        plan: MetalDispatchPlan,
        submission: inout MetalSubmissionContext,
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
        let residencyLease = try makeResidencyLease(
            device: device,
            buffers: [
                snapshotHidden,
                snapshotResidual,
                snapshotScratch,
                snapshotLogits,
                snapshotPosition,
                snapshotRoPEPositionAxes,
                snapshotTokenIn,
                snapshotTokenOut,
                snapshotKVKeys,
                snapshotKVValues,
                snapshotConvState,
                snapshotRecurrentState,
                snapshotPerLayerInputs,
            ].compactMap { $0 }
        )

        try submission.withCompute(ephemeralResidency: residencyLease) { encoder, _ in
            encoder.copy(sourceBuffer: plan.buffers.hidden, sourceOffset: 0, destinationBuffer: snapshotHidden, destinationOffset: 0, size: plan.buffers.hidden.length)
            encoder.copy(sourceBuffer: plan.buffers.residual, sourceOffset: 0, destinationBuffer: snapshotResidual, destinationOffset: 0, size: plan.buffers.residual.length)
            encoder.copy(sourceBuffer: plan.buffers.scratch, sourceOffset: 0, destinationBuffer: snapshotScratch, destinationOffset: 0, size: plan.buffers.scratch.length)
            encoder.copy(sourceBuffer: plan.buffers.logits, sourceOffset: 0, destinationBuffer: snapshotLogits, destinationOffset: 0, size: plan.buffers.logits.length)
            encoder.copy(sourceBuffer: plan.buffers.position, sourceOffset: 0, destinationBuffer: snapshotPosition, destinationOffset: 0, size: plan.buffers.position.length)
            encoder.copy(sourceBuffer: plan.buffers.ropePositionAxes, sourceOffset: 0, destinationBuffer: snapshotRoPEPositionAxes, destinationOffset: 0, size: plan.buffers.ropePositionAxes.length)
            encoder.copy(sourceBuffer: plan.buffers.tokenIn, sourceOffset: 0, destinationBuffer: snapshotTokenIn, destinationOffset: 0, size: plan.buffers.tokenIn.length)
            encoder.copy(sourceBuffer: plan.buffers.tokenOut, sourceOffset: 0, destinationBuffer: snapshotTokenOut, destinationOffset: 0, size: plan.buffers.tokenOut.length)
            if let liveKV = plan.buffers.kvCache,
               let snapshotKVKeys,
               let snapshotKVValues {
                encoder.copy(sourceBuffer: liveKV.keys, sourceOffset: 0, destinationBuffer: snapshotKVKeys, destinationOffset: 0, size: liveKV.keys.length)
                encoder.copy(sourceBuffer: liveKV.values, sourceOffset: 0, destinationBuffer: snapshotKVValues, destinationOffset: 0, size: liveKV.values.length)
            }
            if let liveConvState = plan.buffers.convState,
               let snapshotConvState {
                encoder.copy(sourceBuffer: liveConvState, sourceOffset: 0, destinationBuffer: snapshotConvState, destinationOffset: 0, size: liveConvState.length)
            }
            if let liveRecurrentState = plan.buffers.recurrentState,
               let snapshotRecurrentState {
                encoder.copy(sourceBuffer: liveRecurrentState, sourceOffset: 0, destinationBuffer: snapshotRecurrentState, destinationOffset: 0, size: liveRecurrentState.length)
            }
            if let livePerLayerInputs = plan.buffers.perLayerInputs,
               let snapshotPerLayerInputs {
                encoder.copy(sourceBuffer: livePerLayerInputs, sourceOffset: 0, destinationBuffer: snapshotPerLayerInputs, destinationOffset: 0, size: livePerLayerInputs.length)
            }
        }

        return MetalPromptState(
            position: position,
            firstToken: firstToken,
            residencyLease: residencyLease,
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
        submission: inout MetalSubmissionContext,
        promptState: MetalPromptState
    ) throws {
        try submission.withCompute(ephemeralResidency: promptState.residencyLease) { encoder, _ in
            encoder.copy(sourceBuffer: promptState.hidden, sourceOffset: 0, destinationBuffer: plan.buffers.hidden, destinationOffset: 0, size: plan.buffers.hidden.length)
            encoder.copy(sourceBuffer: promptState.residual, sourceOffset: 0, destinationBuffer: plan.buffers.residual, destinationOffset: 0, size: plan.buffers.residual.length)
            encoder.copy(sourceBuffer: promptState.scratch, sourceOffset: 0, destinationBuffer: plan.buffers.scratch, destinationOffset: 0, size: plan.buffers.scratch.length)
            encoder.copy(sourceBuffer: promptState.logits, sourceOffset: 0, destinationBuffer: plan.buffers.logits, destinationOffset: 0, size: plan.buffers.logits.length)
            encoder.copy(sourceBuffer: promptState.positionBuffer, sourceOffset: 0, destinationBuffer: plan.buffers.position, destinationOffset: 0, size: plan.buffers.position.length)
            encoder.copy(sourceBuffer: promptState.ropePositionAxes, sourceOffset: 0, destinationBuffer: plan.buffers.ropePositionAxes, destinationOffset: 0, size: plan.buffers.ropePositionAxes.length)
            encoder.copy(sourceBuffer: promptState.tokenIn, sourceOffset: 0, destinationBuffer: plan.buffers.tokenIn, destinationOffset: 0, size: plan.buffers.tokenIn.length)
            encoder.copy(sourceBuffer: promptState.tokenOut, sourceOffset: 0, destinationBuffer: plan.buffers.tokenOut, destinationOffset: 0, size: plan.buffers.tokenOut.length)

            if let liveKV = plan.buffers.kvCache,
               let snapshotKVKeys = promptState.kvKeys,
               let snapshotKVValues = promptState.kvValues {
                encoder.copy(sourceBuffer: snapshotKVKeys, sourceOffset: 0, destinationBuffer: liveKV.keys, destinationOffset: 0, size: liveKV.keys.length)
                encoder.copy(sourceBuffer: snapshotKVValues, sourceOffset: 0, destinationBuffer: liveKV.values, destinationOffset: 0, size: liveKV.values.length)
            }
            if let liveConvState = plan.buffers.convState,
               let snapshotConvState = promptState.convState {
                encoder.copy(sourceBuffer: snapshotConvState, sourceOffset: 0, destinationBuffer: liveConvState, destinationOffset: 0, size: liveConvState.length)
            }
            if let liveRecurrentState = plan.buffers.recurrentState,
               let snapshotRecurrentState = promptState.recurrentState {
                encoder.copy(sourceBuffer: snapshotRecurrentState, sourceOffset: 0, destinationBuffer: liveRecurrentState, destinationOffset: 0, size: liveRecurrentState.length)
            }
            if let livePerLayerInputs = plan.buffers.perLayerInputs,
               let snapshotPerLayerInputs = promptState.perLayerInputs {
                encoder.copy(sourceBuffer: snapshotPerLayerInputs, sourceOffset: 0, destinationBuffer: livePerLayerInputs, destinationOffset: 0, size: livePerLayerInputs.length)
            }
        }
    }

    private func makePrivateBuffer(length: Int, device: MTLDevice) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: length, options: .storageModePrivate) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate prompt state buffer")
        }
        return buffer
    }

    private func makeResidencyLease(device: MTLDevice, buffers: [MTLBuffer]) throws -> MetalResidencyLease {
        try MetalResidencyLease.required(
            device: device,
            label: "swift-lm.prompt-state",
            buffers: buffers
        )
    }
}
