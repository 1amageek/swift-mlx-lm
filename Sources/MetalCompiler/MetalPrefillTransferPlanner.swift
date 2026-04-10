import Metal

struct MetalPrefillTransferPlanner: Sendable {
    func makeTransferPlan(
        prefillPlan: MetalPrefillPlan,
        decodePlan: MetalDispatchPlan,
        sequenceLength: Int
    ) -> PostPrefillTransferPlan {
        let decodeElementSize = decodePlan.buffers.bufferPrecision.byteSize
        let decodeHiddenSize = decodePlan.buffers.hidden.length / decodeElementSize
        let hiddenSource = prefillPlan.finalHiddenSource(sequenceLength: sequenceLength)
        let hiddenDestination = decodePlan.outputHeadInputBinding()

        // GPU-side F32→decode precision conversion element count.
        // When decode precision is already F32, a blit copy suffices.
        let hiddenConversionElementCount: Int
        let hiddenBlitCopySize: Int
        if decodePlan.buffers.bufferPrecision == .float32 {
            hiddenConversionElementCount = 0
            hiddenBlitCopySize = decodeHiddenSize * decodeElementSize
        } else {
            hiddenConversionElementCount = decodeHiddenSize
            hiddenBlitCopySize = 0
        }

        let kvCopySize: Int
        let kvTransformSequenceLength: Int
        if let prefillKV = prefillPlan.buffers.kvCache,
           let decodeKV = decodePlan.buffers.kvCache,
           prefillKV.keys !== decodeKV.keys {
            if requiresKVTransform(prefillKV: prefillKV, decodeKV: decodeKV) {
                kvTransformSequenceLength = sequenceLength
                kvCopySize = 0
            } else {
                kvTransformSequenceLength = 0
                kvCopySize = min(decodeKV.keys.length, prefillKV.keys.length)
            }
        } else {
            kvTransformSequenceLength = 0
            kvCopySize = 0
        }

        let valueCopySize: Int
        if let prefillKV = prefillPlan.buffers.kvCache,
           let decodeKV = decodePlan.buffers.kvCache,
           kvTransformSequenceLength == 0,
           prefillKV.values !== decodeKV.values {
            valueCopySize = min(decodeKV.values.length, prefillKV.values.length)
        } else {
            valueCopySize = 0
        }

        let qjlResidualCopySize: Int
        if let prefillKV = prefillPlan.buffers.kvCache,
           let decodeKV = decodePlan.buffers.kvCache,
           let prefillQJLResidual = prefillKV.qjlResidualK,
           let decodeQJLResidual = decodeKV.qjlResidualK,
           kvTransformSequenceLength == 0,
           prefillQJLResidual !== decodeQJLResidual {
            qjlResidualCopySize = min(decodeQJLResidual.length, prefillQJLResidual.length)
        } else {
            qjlResidualCopySize = 0
        }

        let convCopySize: Int
        if let prefillConvState = prefillPlan.buffers.convState,
           let decodeConvState = decodePlan.buffers.convState,
           prefillConvState !== decodeConvState {
            convCopySize = min(decodeConvState.length, prefillConvState.length)
        } else {
            convCopySize = 0
        }

        let recurrentCopySize: Int
        if let prefillRecurrentState = prefillPlan.buffers.recurrentState,
           let decodeRecurrentState = decodePlan.buffers.recurrentState,
           prefillRecurrentState !== decodeRecurrentState {
            recurrentCopySize = min(decodeRecurrentState.length, prefillRecurrentState.length)
        } else {
            recurrentCopySize = 0
        }

        return PostPrefillTransferPlan(
            hiddenSourceBuffer: hiddenSource.buffer,
            hiddenSourceOffset: hiddenSource.offset,
            hiddenDestinationBuffer: hiddenDestination.buffer,
            hiddenDestinationOffset: hiddenDestination.offset,
            hiddenConversionElementCount: hiddenConversionElementCount,
            hiddenBlitCopySize: hiddenBlitCopySize,
            kvTransformSequenceLength: kvTransformSequenceLength,
            kvCopySize: kvCopySize,
            valueCopySize: valueCopySize,
            qjlResidualCopySize: qjlResidualCopySize,
            convCopySize: convCopySize,
            recurrentCopySize: recurrentCopySize
        )
    }

    private func requiresKVTransform(prefillKV: MetalKVCache, decodeKV: MetalKVCache) -> Bool {
        prefillKV.specification.keyQuantizationScheme != decodeKV.specification.keyQuantizationScheme
            || prefillKV.specification.valueQuantizationScheme != decodeKV.specification.valueQuantizationScheme
            || prefillKV.specification.layoutMode != decodeKV.specification.layoutMode
            || prefillKV.qjlDimension != decodeKV.qjlDimension
    }
}

struct PostPrefillTransferPlan: @unchecked Sendable {
    let hiddenSourceBuffer: MTLBuffer
    let hiddenSourceOffset: Int
    let hiddenDestinationBuffer: MTLBuffer
    let hiddenDestinationOffset: Int
    /// Element count for GPU-side F32→F16/BF16 hidden conversion.
    /// Zero when decode precision is F32 (blit copy used instead).
    let hiddenConversionElementCount: Int
    /// Byte count for F32→F32 hidden blit copy (only when decode is F32).
    let hiddenBlitCopySize: Int
    /// Non-zero when post-prefill KV requires GPU-side format conversion.
    let kvTransformSequenceLength: Int
    let kvCopySize: Int
    let valueCopySize: Int
    let qjlResidualCopySize: Int
    let convCopySize: Int
    let recurrentCopySize: Int
}
