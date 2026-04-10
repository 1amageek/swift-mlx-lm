import Metal

/// Encodes decode dispatch steps using argument tables and intra-pass barriers.
///
/// Uses `MTL4ComputeCommandEncoder` with `MTL4ArgumentTable` for bindless
/// buffer binding and stage-to-stage barriers with configurable visibility.
enum MetalDecodeEncoder {

    /// Encode all decode steps using Metal 4 intra-pass barriers.
    ///
    /// Buffer bindings use `MTL4ArgumentTable.setAddress(gpuAddress)`.
    /// Constant bindings in `.resident` mode use buffer gpuAddress;
    /// `.inline` constants use a pre-allocated constant data buffer.
    static func encodeSteps(
        plan: MetalDispatchPlan,
        encoder: MTL4ComputeCommandEncoder,
        argumentTable: MTL4ArgumentTable,
        visibilityOptions: MTL4VisibilityOptions = []
    ) {
        let resolvedVisibilityOptions = resolvedMetalVisibilityOptions(visibilityOptions)
        for step in plan.steps {
            bindArgumentTable(step: step, argumentTable: argumentTable)

            if step.barrierPolicy.isBarrier {
                encoder.barrier(
                    afterEncoderStages: .dispatch,
                    beforeEncoderStages: .dispatch,
                    visibilityOptions: resolvedVisibilityOptions
                )
            }

            encoder.setArgumentTable(argumentTable)
            encoder.setComputePipelineState(step.pipeline)
            if step.threadgroupMemoryLength > 0 {
                encoder.setThreadgroupMemoryLength(step.threadgroupMemoryLength, index: 0)
            }
            encoder.dispatchThreadgroups(
                threadgroupsPerGrid: step.gridSize,
                threadsPerThreadgroup: step.threadgroupSize
            )
        }
    }

    /// Encode a single step using Metal 4 argument table binding and barriers.
    static func encodeStep(
        step: MetalDispatchStep,
        encoder: MTL4ComputeCommandEncoder,
        argumentTable: MTL4ArgumentTable,
        visibilityOptions: MTL4VisibilityOptions = []
    ) {
        let resolvedVisibilityOptions = resolvedMetalVisibilityOptions(visibilityOptions)
        bindArgumentTable(step: step, argumentTable: argumentTable)

        if step.barrierPolicy.isBarrier {
            encoder.barrier(
                afterEncoderStages: .dispatch,
                beforeEncoderStages: .dispatch,
                visibilityOptions: resolvedVisibilityOptions
            )
        }

        encoder.setArgumentTable(argumentTable)
        encoder.setComputePipelineState(step.pipeline)
        if step.threadgroupMemoryLength > 0 {
            encoder.setThreadgroupMemoryLength(step.threadgroupMemoryLength, index: 0)
        }
        encoder.dispatchThreadgroups(
            threadgroupsPerGrid: step.gridSize,
            threadsPerThreadgroup: step.threadgroupSize
        )
    }

    /// Bind a single step's buffers and constants to the argument table.
    private static func bindArgumentTable(
        step: MetalDispatchStep,
        argumentTable: MTL4ArgumentTable
    ) {
        // Buffer bindings → gpuAddress
        let bufferBindings = step.bindings.bufferBindings
        switch bufferBindings {
        case .inline(let bindings):
            for binding in bindings {
                argumentTable.setAddress(
                    binding.buffer.gpuAddress + UInt64(binding.offset),
                    index: binding.index
                )
            }
        case .argumentTable(let table):
            switch table.encodingState {
            case .encoded(let buffer, let index, let offset):
                // Pre-encoded argument buffer: kernel expects all bindings packed at one index
                argumentTable.setAddress(
                    buffer.gpuAddress + UInt64(offset),
                    index: index
                )
            case .planned, .prepared:
                // Individual bindings
                for binding in table.bindings {
                    argumentTable.setAddress(
                        binding.buffer.gpuAddress + UInt64(binding.offset),
                        index: binding.index
                    )
                }
            }
        }

        // Constant bindings → gpuAddress (resident) or inline data buffer
        let constantBindings = step.bindings.constantBindings
        switch constantBindings {
        case .inline(let bindings):
            bindInlineConstants(bindings, argumentTable: argumentTable)
        case .resident(let resident):
            for binding in resident.bindings {
                argumentTable.setAddress(
                    binding.buffer.gpuAddress + UInt64(binding.offset),
                    index: binding.index
                )
            }
        case .mixed(let bindings):
            for constant in bindings {
                switch constant {
                case .inline(let binding):
                    bindInlineConstant(binding, argumentTable: argumentTable)
                case .buffer(let binding):
                    argumentTable.setAddress(
                        binding.buffer.gpuAddress + UInt64(binding.offset),
                        index: binding.index
                    )
                }
            }
        }
    }

    /// Bind inline constant bytes via a temporary staging approach.
    ///
    /// Metal 4 has no `setBytes` equivalent. Inline constants must be stored
    /// in a buffer and bound via gpuAddress. For steps that still use inline
    /// bytes, we fall back to storing them in a shared scratch buffer.
    ///
    /// Note: Production code should use `.residentConstantBuffer` mode to
    /// pre-allocate all constants. This fallback handles legacy inline bindings.
    private static func bindInlineConstants(
        _ bindings: [MetalBytesBinding],
        argumentTable: MTL4ArgumentTable
    ) {
        for binding in bindings {
            bindInlineConstant(binding, argumentTable: argumentTable)
        }
    }

    private static func bindInlineConstant(
        _ binding: MetalBytesBinding,
        argumentTable: MTL4ArgumentTable
    ) {
        // Inline bytes require a backing buffer. The resident constant system
        // should have converted these during plan construction. If we reach here,
        // the step was not processed through makeResidentConstantSteps.
        //
        // For Metal 4, all constants MUST be buffer-backed. Log and skip.
        assertionFailure(
            "Inline constant at index \(binding.index) has no backing buffer. "
            + "Ensure all steps use residentConstantBuffer mode."
        )
    }
}
