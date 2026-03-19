import Metal

struct MetalConstantBindingEncoder {
    func bind(
        _ constantBindings: MetalConstantBindingSet,
        to encoder: MTLComputeCommandEncoder
    ) {
        switch constantBindings {
        case .inline(let bindings):
            for binding in bindings {
                binding.value.withUnsafeBufferPointer {
                    encoder.setBytes($0.baseAddress!, length: $0.count, index: binding.index)
                }
            }
        case .resident(let resident):
            for binding in resident.bindings {
                encoder.setBuffer(binding.buffer, offset: binding.offset, index: binding.index)
            }
        case .mixed(let bindings):
            for constant in bindings {
                bind(constant, to: encoder)
            }
        }
    }

    private func bind(
        _ constant: MetalConstantBinding,
        to encoder: MTLComputeCommandEncoder
    ) {
        switch constant {
        case .inline(let binding):
            binding.value.withUnsafeBufferPointer {
                encoder.setBytes($0.baseAddress!, length: $0.count, index: binding.index)
            }
        case .buffer(let binding):
            encoder.setBuffer(binding.buffer, offset: binding.offset, index: binding.index)
        }
    }
}
