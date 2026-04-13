import Metal
import LMIR

public enum MetalArgumentBindingPolicy: Sendable, Equatable {
    case inlineBindings
    case argumentTable
}

public enum MetalConstantBindingPolicy: Sendable, Equatable {
    case inlineBytes
    case residentConstantBuffer
}

public enum MetalBarrierPolicy: Sendable, Equatable {
    /// No barrier needed before this dispatch.
    case none
    /// Stage-to-stage barrier with explicit visibility options.
    ///
    /// - `.device`: Full cache flush — required when writes target shared-mode buffers.
    /// - `[]` (empty): Execution ordering only — sufficient for private-mode buffers
    ///   where GPU caches remain coherent across dispatches within the same encoder.
    case barrier(visibility: MTL4VisibilityOptions)

    public init(_ synchronizationKind: SynchronizationKind) {
        switch synchronizationKind {
        case .none:
            self = .none
        case .bufferBarrier:
            self = .barrier(visibility: .device)
        }
    }

    public var synchronizationKind: SynchronizationKind {
        switch self {
        case .none:
            return .none
        case .barrier:
            return .bufferBarrier
        }
    }

    public var isBarrier: Bool {
        switch self {
        case .none: return false
        case .barrier: return true
        }
    }

    /// Convenience for the legacy `.bufferBarrier` pattern — barrier with `.device` visibility.
    public static var bufferBarrier: MetalBarrierPolicy {
        .barrier(visibility: .device)
    }
}

public struct MetalBufferBinding: @unchecked Sendable {
    public let index: Int
    public let buffer: MTLBuffer
    public let offset: Int
}

public enum MetalArgumentTableEncodingState: @unchecked Sendable {
    case planned
    case prepared(buffer: MTLBuffer, index: Int, offset: Int)
    case encoded(buffer: MTLBuffer, index: Int, offset: Int)

    public var isEncoded: Bool {
        switch self {
        case .planned:
            return false
        case .prepared:
            return false
        case .encoded:
            return true
        }
    }
}

public struct MetalArgumentTableBindings: @unchecked Sendable {
    public let layout: MetalArgumentTableLayout
    public let bindings: [MetalBufferBinding]
    public let encodingState: MetalArgumentTableEncodingState

    public init(
        layout: MetalArgumentTableLayout,
        bindings: [MetalBufferBinding],
        encodingState: MetalArgumentTableEncodingState = .planned
    ) {
        self.layout = layout
        self.bindings = bindings
        self.encodingState = encodingState
    }

    public var hasEncodedArgumentBuffer: Bool {
        encodingState.isEncoded
    }
}

public enum MetalBufferBindingSet: @unchecked Sendable {
    case inline([MetalBufferBinding])
    case argumentTable(MetalArgumentTableBindings)

    public var policy: MetalArgumentBindingPolicy {
        switch self {
        case .inline:
            return .inlineBindings
        case .argumentTable:
            return .argumentTable
        }
    }

    public var bindings: [MetalBufferBinding] {
        switch self {
        case .inline(let bindings):
            return bindings
        case .argumentTable(let table):
            return table.bindings
        }
    }
}

public struct MetalBytesBinding: Sendable {
    public let index: Int
    public let value: [UInt8]
}

public struct MetalConstantBufferBinding: @unchecked Sendable {
    public let index: Int
    public let buffer: MTLBuffer
    public let offset: Int
    public let length: Int
}

public struct MetalResidentConstantBindings: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let bindings: [MetalConstantBufferBinding]
}

public enum MetalConstantBinding: @unchecked Sendable {
    case inline(MetalBytesBinding)
    case buffer(MetalConstantBufferBinding)

    public var index: Int {
        switch self {
        case .inline(let binding):
            return binding.index
        case .buffer(let binding):
            return binding.index
        }
    }
}

public enum MetalConstantBindingSet: @unchecked Sendable {
    case inline([MetalBytesBinding])
    case resident(MetalResidentConstantBindings)
    case mixed([MetalConstantBinding])

    public var policy: MetalConstantBindingPolicy {
        switch self {
        case .inline:
            return .inlineBytes
        case .resident:
            return .residentConstantBuffer
        case .mixed(let bindings):
            if bindings.allSatisfy({
                if case .buffer = $0 { return true }
                return false
            }) {
                return .residentConstantBuffer
            }
            return .inlineBytes
        }
    }

    public var bindings: [MetalConstantBinding] {
        switch self {
        case .inline(let bindings):
            return bindings.map(MetalConstantBinding.inline)
        case .resident(let resident):
            return resident.bindings.map(MetalConstantBinding.buffer)
        case .mixed(let bindings):
            return bindings
        }
    }

    public var inlineBindings: [MetalBytesBinding] {
        bindings.compactMap { binding in
            guard case .inline(let bytes) = binding else { return nil }
            return bytes
        }
    }
}

public struct MetalBindingTable: @unchecked Sendable {
    public let bufferBindings: MetalBufferBindingSet
    public let constantBindings: MetalConstantBindingSet
    public var argumentPolicy: MetalArgumentBindingPolicy {
        bufferBindings.policy
    }
    public var buffers: [MetalBufferBinding] {
        bufferBindings.bindings
    }
    public var constantPolicy: MetalConstantBindingPolicy {
        constantBindings.policy
    }
    public var constants: [MetalConstantBinding] {
        constantBindings.bindings
    }

    public init(
        buffers: [MetalBufferBinding] = [],
        constants: [MetalConstantBinding] = [],
        argumentPolicy: MetalArgumentBindingPolicy = .inlineBindings,
        constantPolicy: MetalConstantBindingPolicy = .inlineBytes
    ) {
        switch argumentPolicy {
        case .inlineBindings:
            self.bufferBindings = .inline(buffers)
        case .argumentTable:
            self.bufferBindings = .argumentTable(MetalArgumentTableBindings(
                layout: MetalArgumentTableLayout(id: 0, indices: buffers.map(\.index)),
                bindings: buffers))
        }
        switch constantPolicy {
        case .inlineBytes:
            self.constantBindings = .mixed(constants)
        case .residentConstantBuffer:
            self.constantBindings = .mixed(constants)
        }
    }

    public init(
        buffers: [MetalBufferBinding] = [],
        bufferBindings: MetalBufferBindingSet,
        constantBindings: MetalConstantBindingSet,
        argumentPolicy: MetalArgumentBindingPolicy = .inlineBindings,
        constantPolicy: MetalConstantBindingPolicy? = nil
    ) {
        _ = buffers
        self.bufferBindings = bufferBindings
        self.constantBindings = constantBindings
        _ = argumentPolicy
        _ = constantPolicy
    }

    public init(
        bufferBindings: [(index: Int, buffer: MTLBuffer, offset: Int)],
        bytesBindings: [(index: Int, value: [UInt8])],
        argumentPolicy: MetalArgumentBindingPolicy = .inlineBindings,
        constantPolicy: MetalConstantBindingPolicy = .inlineBytes
    ) {
        let mappedBuffers = bufferBindings.map { MetalBufferBinding(index: $0.index, buffer: $0.buffer, offset: $0.offset) }
        switch argumentPolicy {
        case .inlineBindings:
            self.bufferBindings = .inline(mappedBuffers)
        case .argumentTable:
            self.bufferBindings = .argumentTable(MetalArgumentTableBindings(
                layout: MetalArgumentTableLayout(id: 0, indices: mappedBuffers.map(\.index)),
                bindings: mappedBuffers))
        }
        let inlineBindings = bytesBindings.map { MetalBytesBinding(index: $0.index, value: $0.value) }
        switch constantPolicy {
        case .inlineBytes:
            self.constantBindings = .inline(inlineBindings)
        case .residentConstantBuffer:
            self.constantBindings = .mixed(inlineBindings.map(MetalConstantBinding.inline))
        }
    }

}

// MARK: - Metal 4 Barrier Encoding

extension MetalBarrierPolicy {
    /// Encode a barrier on a Metal 4 compute encoder.
    ///
    /// The visibility option is embedded in the policy itself — the barrier optimizer
    /// determines both WHETHER a barrier is needed and WHAT visibility is required
    /// based on buffer storage modes (private vs shared).
    func encode(on encoder: MTL4ComputeCommandEncoder) {
        switch self {
        case .none:
            return
        case .barrier(let visibility):
            encoder.barrier(
                afterEncoderStages: .dispatch,
                beforeEncoderStages: .dispatch,
                visibilityOptions: visibility
            )
        }
    }
}

// MARK: - Metal 4 Argument Table Binding

extension MetalBindingTable {
    func bind(to argumentTable: MTL4ArgumentTable) {
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
                argumentTable.setAddress(
                    buffer.gpuAddress + UInt64(offset),
                    index: index
                )
            case .planned, .prepared:
                for binding in table.bindings {
                    argumentTable.setAddress(
                        binding.buffer.gpuAddress + UInt64(binding.offset),
                        index: binding.index
                    )
                }
            }
        }
        bindConstants(to: argumentTable)
    }

    func bind(to argumentTable: MTL4ArgumentTable, adjustedBufferOffsets: [Int: Int]) {
        switch bufferBindings {
        case .inline(let bindings):
            for binding in bindings {
                let offset = adjustedBufferOffsets[binding.index] ?? binding.offset
                argumentTable.setAddress(
                    binding.buffer.gpuAddress + UInt64(offset),
                    index: binding.index
                )
            }
        case .argumentTable(let table):
            switch table.encodingState {
            case .encoded(let buffer, let index, let offset):
                argumentTable.setAddress(
                    buffer.gpuAddress + UInt64(offset),
                    index: index
                )
            case .planned, .prepared:
                for binding in table.bindings {
                    let offset = adjustedBufferOffsets[binding.index] ?? binding.offset
                    argumentTable.setAddress(
                        binding.buffer.gpuAddress + UInt64(offset),
                        index: binding.index
                    )
                }
            }
        }
        bindConstants(to: argumentTable)
    }

    var ownedResidencyBuffers: [MTLBuffer] {
        var buffers: [MTLBuffer] = []
        switch bufferBindings {
        case .inline:
            break
        case .argumentTable(let table):
            switch table.encodingState {
            case .planned:
                break
            case .prepared(let buffer, _, _), .encoded(let buffer, _, _):
                buffers.append(buffer)
            }
        }

        switch constantBindings {
        case .inline:
            break
        case .resident(let resident):
            buffers.append(resident.buffer)
        case .mixed(let bindings):
            for binding in bindings {
                guard case .buffer(let bufferBinding) = binding else { continue }
                buffers.append(bufferBinding.buffer)
            }
        }
        return buffers
    }

    private func bindConstants(to argumentTable: MTL4ArgumentTable) {
        switch constantBindings {
        case .inline(let bindings):
            for binding in bindings {
                assertionFailure(
                    "Inline constant at index \(binding.index) has no backing buffer. "
                    + "Ensure all steps use residentConstantBuffer mode."
                )
            }
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
                    assertionFailure(
                        "Inline constant at index \(binding.index) has no backing buffer. "
                        + "Ensure all steps use residentConstantBuffer mode."
                    )
                case .buffer(let binding):
                    argumentTable.setAddress(
                        binding.buffer.gpuAddress + UInt64(binding.offset),
                        index: binding.index
                    )
                }
            }
        }
    }
}

public struct MetalDispatchDescriptor: @unchecked Sendable {
    public let pipeline: MTLComputePipelineState
    public let gridSize: MTLSize
    public let threadgroupSize: MTLSize
    public let threadgroupMemoryLength: Int
    public let barrierPolicy: MetalBarrierPolicy

    public var sync: SynchronizationKind {
        barrierPolicy.synchronizationKind
    }

    public func encode(
        on encoder: MTL4ComputeCommandEncoder,
        argumentTable: MTL4ArgumentTable,
        gridSize overrideGridSize: MTLSize? = nil
    ) {
        barrierPolicy.encode(on: encoder)
        encoder.setArgumentTable(argumentTable)
        encoder.setComputePipelineState(pipeline)
        if threadgroupMemoryLength > 0 {
            encoder.setThreadgroupMemoryLength(threadgroupMemoryLength, index: 0)
        }
        encoder.dispatchThreadgroups(
            threadgroupsPerGrid: overrideGridSize ?? gridSize,
            threadsPerThreadgroup: threadgroupSize
        )
    }
}
