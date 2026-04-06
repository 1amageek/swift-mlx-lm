import Metal

struct MetalPrefillTransferPlanner: Sendable {
    func makeTransferPlan(
        prefillPlan: MetalPrefillPlan,
        decodePlan: MetalDispatchPlan,
        sequenceLength: Int
    ) -> PostPrefillTransferPlan {
        let decodeElementSize = decodePlan.buffers.bufferPrecision.byteSize
        let decodeHiddenSize = decodePlan.buffers.hidden.length / decodeElementSize
        let prefillHiddenStride = decodeHiddenSize * MemoryLayout<Float32>.size
        let hiddenSourceOffset = (sequenceLength - 1) * prefillHiddenStride
        let requiresCPUHiddenConversion =
            decodePlan.buffers.hidden.storageMode == .private &&
            decodePlan.buffers.bufferPrecision != .float32
        let usesSharedDecodeHidden = decodePlan.buffers.hidden.storageMode != .private

        let hiddenCopySize: Int
        if hiddenSourceOffset + prefillHiddenStride <= prefillPlan.buffers.hidden.length,
           decodePlan.buffers.hidden.storageMode == .private {
            hiddenCopySize = decodeHiddenSize * decodeElementSize
        } else {
            hiddenCopySize = 0
        }

        let kvCopySize: Int
        if let prefillKV = prefillPlan.buffers.kvCache,
           let decodeKV = decodePlan.buffers.kvCache,
           prefillKV.keys !== decodeKV.keys {
            kvCopySize = min(decodeKV.keys.length, prefillKV.keys.length)
        } else {
            kvCopySize = 0
        }

        let valueCopySize: Int
        if let prefillKV = prefillPlan.buffers.kvCache,
           let decodeKV = decodePlan.buffers.kvCache,
           prefillKV.values !== decodeKV.values {
            valueCopySize = min(decodeKV.values.length, prefillKV.values.length)
        } else {
            valueCopySize = 0
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
            hiddenSourceOffset: hiddenSourceOffset,
            hiddenCopySize: hiddenCopySize,
            kvCopySize: kvCopySize,
            valueCopySize: valueCopySize,
            convCopySize: convCopySize,
            recurrentCopySize: recurrentCopySize,
            requiresCPUHiddenConversion: requiresCPUHiddenConversion,
            usesSharedDecodeHidden: usesSharedDecodeHidden
        )
    }
}

struct PostPrefillTransferPlan: Sendable {
    let hiddenSourceOffset: Int
    let hiddenCopySize: Int
    let kvCopySize: Int
    let valueCopySize: Int
    let convCopySize: Int
    let recurrentCopySize: Int
    let requiresCPUHiddenConversion: Bool
    let usesSharedDecodeHidden: Bool

    var needsBlit: Bool {
        hiddenCopySize > 0 || kvCopySize > 0 || valueCopySize > 0 || convCopySize > 0 || recurrentCopySize > 0
    }

    var canEncodeInSameTransaction: Bool {
        !requiresCPUHiddenConversion && needsBlit
    }

    var shouldStageHiddenOnCPU: Bool {
        requiresCPUHiddenConversion && hiddenCopySize > 0
    }

    var needsStandaloneBlit: Bool {
        needsBlit && !canEncodeInSameTransaction
    }

    func afterInlineBlit() -> Self {
        Self(
            hiddenSourceOffset: hiddenSourceOffset,
            hiddenCopySize: 0,
            kvCopySize: 0,
            valueCopySize: 0,
            convCopySize: 0,
            recurrentCopySize: 0,
            requiresCPUHiddenConversion: requiresCPUHiddenConversion,
            usesSharedDecodeHidden: usesSharedDecodeHidden
        )
    }

    func withoutHiddenCopy() -> Self {
        Self(
            hiddenSourceOffset: hiddenSourceOffset,
            hiddenCopySize: 0,
            kvCopySize: kvCopySize,
            valueCopySize: valueCopySize,
            convCopySize: convCopySize,
            recurrentCopySize: recurrentCopySize,
            requiresCPUHiddenConversion: false,
            usesSharedDecodeHidden: false
        )
    }
}
