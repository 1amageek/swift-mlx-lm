import Metal

struct MetalConstantBindingAllocator: Sendable {
    let device: MTLDevice
    let alignment: Int
    let resourceOptions: MTLResourceOptions

    init(
        device: MTLDevice,
        alignment: Int = 16,
        resourceOptions: MTLResourceOptions = [.storageModeShared]
    ) {
        self.device = device
        self.alignment = alignment
        self.resourceOptions = resourceOptions
    }

    func makeBindingTable(
        bufferBindings: [(index: Int, buffer: MTLBuffer, offset: Int)],
        bytesBindings: [(index: Int, value: [UInt8])],
        argumentPolicy: MetalArgumentBindingPolicy = .inlineBindings
    ) throws -> MetalBindingTable {
        let mappedBuffers = bufferBindings.map { MetalBufferBinding(index: $0.index, buffer: $0.buffer, offset: $0.offset) }
        guard !bytesBindings.isEmpty else {
            return MetalBindingTable(
                buffers: mappedBuffers,
                constants: [],
                argumentPolicy: argumentPolicy,
                constantPolicy: .inlineBytes)
        }

        let plan = makeAllocationPlan(bytesBindings: bytesBindings)
        guard let buffer = device.makeBuffer(length: plan.totalLength, options: resourceOptions) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate resident constant buffer")
        }

        let baseAddress = buffer.contents()
        var residentBindings: [MetalConstantBufferBinding] = []
        residentBindings.reserveCapacity(bytesBindings.count)

        for item in plan.items {
            let destination = baseAddress.advanced(by: item.offset)
            _ = item.value.withUnsafeBytes { src in
                memcpy(destination, src.baseAddress!, src.count)
            }
            residentBindings.append(MetalConstantBufferBinding(
                index: item.index,
                buffer: buffer,
                offset: item.offset,
                length: item.value.count))
        }

        return MetalBindingTable(
            bufferBindings: .inline(mappedBuffers),
            constantBindings: .resident(MetalResidentConstantBindings(
                buffer: buffer,
                bindings: residentBindings)),
            argumentPolicy: argumentPolicy)
    }

    func makeBindingTables(
        from tables: [MetalBindingTable]
    ) throws -> [MetalBindingTable] {
        let plan = makeArenaPlan(tables: tables)
        guard let arena = plan else {
            return tables
        }
        guard let buffer = device.makeBuffer(length: arena.totalLength, options: resourceOptions) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate resident constant arena")
        }

        let baseAddress = buffer.contents()
        for item in arena.items {
            let destination = baseAddress.advanced(by: item.offset)
            _ = item.value.withUnsafeBytes { src in
                memcpy(destination, src.baseAddress!, src.count)
            }
        }

        var tablesWithResidentConstants: [MetalBindingTable] = []
        tablesWithResidentConstants.reserveCapacity(tables.count)

        for (tableIndex, table) in tables.enumerated() {
            let residentBindings: [MetalConstantBufferBinding] = table.constants.compactMap { constant in
                switch constant {
                case .buffer(let binding):
                    return binding
                case .inline(let bytes):
                    guard let allocation = arena.allocations[tableIndex, default: [:]][bytes.index] else {
                        return nil
                    }
                    return MetalConstantBufferBinding(
                        index: bytes.index,
                        buffer: buffer,
                        offset: allocation.offset,
                        length: allocation.length)
                }
            }
            tablesWithResidentConstants.append(MetalBindingTable(
                bufferBindings: table.bufferBindings,
                constantBindings: residentBindings.isEmpty
                    ? table.constantBindings
                    : .resident(MetalResidentConstantBindings(
                        buffer: buffer,
                        bindings: residentBindings)),
                argumentPolicy: table.argumentPolicy))
        }

        return tablesWithResidentConstants
    }

    private func makeAllocationPlan(
        bytesBindings: [(index: Int, value: [UInt8])]
    ) -> (items: [(index: Int, offset: Int, value: [UInt8])], totalLength: Int) {
        var items: [(index: Int, offset: Int, value: [UInt8])] = []
        items.reserveCapacity(bytesBindings.count)
        var cursor = 0

        for binding in bytesBindings {
            let offset = aligned(cursor, to: alignment)
            items.append((index: binding.index, offset: offset, value: binding.value))
            cursor = offset + binding.value.count
        }

        return (items, totalLength: aligned(cursor, to: alignment))
    }

    private func makeArenaPlan(
        tables: [MetalBindingTable]
    ) -> (
        items: [(tableIndex: Int, index: Int, offset: Int, value: [UInt8])],
        allocations: [Int: [Int: (offset: Int, length: Int)]],
        totalLength: Int
    )? {
        var items: [(tableIndex: Int, index: Int, offset: Int, value: [UInt8])] = []
        var allocations: [Int: [Int: (offset: Int, length: Int)]] = [:]
        var cursor = 0

        for (tableIndex, table) in tables.enumerated() {
            for constant in table.constants {
                guard case .inline(let bytes) = constant else {
                    continue
                }
                let offset = aligned(cursor, to: alignment)
                items.append((tableIndex: tableIndex, index: bytes.index, offset: offset, value: bytes.value))
                allocations[tableIndex, default: [:]][bytes.index] = (offset: offset, length: bytes.value.count)
                cursor = offset + bytes.value.count
            }
        }

        guard !items.isEmpty else {
            return nil
        }

        return (
            items: items,
            allocations: allocations,
            totalLength: aligned(cursor, to: alignment))
    }

    private func aligned(_ value: Int, to alignment: Int) -> Int {
        let mask = alignment - 1
        return (value + mask) & ~mask
    }
}
