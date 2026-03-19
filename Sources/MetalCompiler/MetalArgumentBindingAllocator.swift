import Metal

struct MetalArgumentTableLayoutUsage: Sendable, Hashable {
    let layout: MetalArgumentTableLayout
    let bindingCount: Int
    let useCount: Int
}

struct MetalArgumentBindingAllocator: Sendable {
    let minimumBindingCount: Int

    init(minimumBindingCount: Int = 3) {
        self.minimumBindingCount = minimumBindingCount
    }

    func makeBindingTables(
        from tables: [MetalBindingTable]
    ) -> [MetalBindingTable] {
        var layoutIDs: [[Int]: Int] = [:]
        var nextLayoutID = 0

        return tables.map { table in
            makeBindingTable(
                from: table,
                layoutIDs: &layoutIDs,
                nextLayoutID: &nextLayoutID)
        }
    }

    func summarizeUsage(
        in tables: [MetalBindingTable]
    ) -> [MetalArgumentTableLayoutUsage] {
        struct Accumulator {
            var layout: MetalArgumentTableLayout
            var bindingCount: Int
            var useCount: Int
        }

        var usage: [Int: Accumulator] = [:]
        for table in tables {
            guard case .argumentTable(let argumentTable) = table.bufferBindings else {
                continue
            }
            let layoutID = argumentTable.layout.id
            usage[layoutID, default: Accumulator(
                layout: argumentTable.layout,
                bindingCount: argumentTable.bindings.count,
                useCount: 0
            )].useCount += 1
        }

        return usage.values
            .map { value in
                MetalArgumentTableLayoutUsage(
                    layout: value.layout,
                    bindingCount: value.bindingCount,
                    useCount: value.useCount)
            }
            .sorted {
                if $0.useCount != $1.useCount {
                    return $0.useCount > $1.useCount
                }
                if $0.bindingCount != $1.bindingCount {
                    return $0.bindingCount > $1.bindingCount
                }
                return $0.layout.id < $1.layout.id
            }
    }

    private func makeBindingTable(
        from table: MetalBindingTable,
        layoutIDs: inout [[Int]: Int],
        nextLayoutID: inout Int
    ) -> MetalBindingTable {
        guard shouldUseArgumentTable(for: table) else {
            return table
        }
        let indices = table.buffers.map(\.index)
        let layoutID: Int
        if let existing = layoutIDs[indices] {
            layoutID = existing
        } else {
            layoutID = nextLayoutID
            layoutIDs[indices] = layoutID
            nextLayoutID += 1
        }
        return MetalBindingTable(
            bufferBindings: .argumentTable(MetalArgumentTableBindings(
                layout: MetalArgumentTableLayout(id: layoutID, indices: indices),
                bindings: table.buffers)),
            constantBindings: table.constantBindings)
    }

    private func shouldUseArgumentTable(
        for table: MetalBindingTable
    ) -> Bool {
        guard table.argumentPolicy == .inlineBindings else {
            return false
        }
        guard table.buffers.count >= minimumBindingCount || shouldUseTwoBufferArgumentTable(for: table) else {
            return false
        }
        let uniqueIndices = Set(table.buffers.map(\.index))
        return uniqueIndices.count == table.buffers.count
    }

    private func shouldUseTwoBufferArgumentTable(
        for table: MetalBindingTable
    ) -> Bool {
        guard table.buffers.count == 2 else {
            return false
        }
        return !table.constants.isEmpty
    }
}
