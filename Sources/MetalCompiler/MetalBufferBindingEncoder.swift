import Metal

struct MetalBufferBindingEncoder {
    func bind(
        _ bufferBindings: MetalBufferBindingSet,
        to encoder: MTLComputeCommandEncoder,
        adjustedBufferOffsets: [Int: Int]
    ) {
        switch bufferBindings {
        case .inline(let bindings):
            bindFallbackBindings(bindings, to: encoder, adjustedBufferOffsets: adjustedBufferOffsets)
        case .argumentTable(let table):
            if adjustedBufferOffsets.isEmpty {
                switch table.encodingState {
                case .encoded(let buffer, let index, let offset):
                    encoder.setBuffer(buffer, offset: offset, index: index)
                case .planned:
                    bindFallbackBindings(table.bindings, to: encoder, adjustedBufferOffsets: adjustedBufferOffsets)
                case .prepared:
                    bindFallbackBindings(table.bindings, to: encoder, adjustedBufferOffsets: adjustedBufferOffsets)
                }
            } else {
                bindFallbackBindings(table.bindings, to: encoder, adjustedBufferOffsets: adjustedBufferOffsets)
            }
        }
    }

    private func bindFallbackBindings(
        _ bindings: [MetalBufferBinding],
        to encoder: MTLComputeCommandEncoder,
        adjustedBufferOffsets: [Int: Int]
    ) {
        for binding in bindings {
            let offset = adjustedBufferOffsets[binding.index] ?? binding.offset
            encoder.setBuffer(binding.buffer, offset: offset, index: binding.index)
        }
    }
}
