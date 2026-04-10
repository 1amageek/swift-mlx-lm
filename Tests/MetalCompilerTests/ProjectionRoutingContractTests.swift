import Testing
@testable import MetalCompiler

@Suite("Projection Routing Contracts", .serialized)
struct ProjectionRoutingContractTests {
    @Test("Attention sibling projections share a common prefill input for all optimizers")
    func attentionSiblingProjectionsShareCommonPrefillInput() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let optimizers: [(String, any DispatchOptimizer)] = [
            ("none", NoOptimizer()),
            ("standard", StandardOptimizer()),
            ("aggressive", AggressiveOptimizer()),
        ]

        for (name, optimizer) in optimizers {
            let setup = try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(optimizer: optimizer)
            let plan = try #require(setup.model.prefillPlan)
            let group = try #require(firstAttentionProjectionGroup(in: setup.collected.fusedEntries))
            try assertCommonPrefillInput(
                group: group,
                plan: plan,
                optimizerName: name,
                groupLabel: "attention"
            )
        }
    }

    @Test("Attention sibling projections share a common decode input for all optimizers")
    func attentionSiblingProjectionsShareCommonDecodeInput() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let optimizers: [(String, any DispatchOptimizer)] = [
            ("none", NoOptimizer()),
            ("standard", StandardOptimizer()),
            ("aggressive", AggressiveOptimizer()),
        ]

        for (name, optimizer) in optimizers {
            let setup = try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(optimizer: optimizer)
            let group = try #require(firstAttentionProjectionGroup(in: setup.collected.fusedEntries))
            try assertCommonDecodeInput(
                group: group,
                plan: setup.model.decodePlan,
                optimizerName: name,
                groupLabel: "attention"
            )
        }
    }

    @Test("MLP sibling projections share a common prefill input for all optimizers")
    func mlpSiblingProjectionsShareCommonPrefillInput() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let optimizers: [(String, any DispatchOptimizer)] = [
            ("none", NoOptimizer()),
            ("standard", StandardOptimizer()),
            ("aggressive", AggressiveOptimizer()),
        ]

        for (name, optimizer) in optimizers {
            let setup = try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(optimizer: optimizer)
            let plan = try #require(setup.model.prefillPlan)
            let group = try #require(firstMLPProjectionGroup(in: setup.collected.fusedEntries))
            try assertCommonPrefillInput(
                group: group,
                plan: plan,
                optimizerName: name,
                groupLabel: "mlp"
            )
        }
    }

    @Test("MLP sibling projections share a common decode input for all optimizers")
    func mlpSiblingProjectionsShareCommonDecodeInput() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let optimizers: [(String, any DispatchOptimizer)] = [
            ("none", NoOptimizer()),
            ("standard", StandardOptimizer()),
            ("aggressive", AggressiveOptimizer()),
        ]

        for (name, optimizer) in optimizers {
            let setup = try BenchmarkSupport.setupWithCollectedPrefillEntriesOrSkip(optimizer: optimizer)
            let group = try #require(firstMLPProjectionGroup(in: setup.collected.fusedEntries))
            try assertCommonDecodeInput(
                group: group,
                plan: setup.model.decodePlan,
                optimizerName: name,
                groupLabel: "mlp"
            )
        }
    }

    private func firstAttentionProjectionGroup(
        in entries: [DispatchEntry]
    ) -> ProjectionGroup? {
        var index = 0
        while index < entries.count {
            let compositeID = entries[index].compositeID
            var endIndex = index + 1
            while endIndex < entries.count, entries[endIndex].compositeID == compositeID {
                endIndex += 1
            }

            let group = Array(entries[index..<endIndex])
            if let batchedEntry = group.first(where: { entry in
                guard case .batchedProjection(let batched) = entry.kind else { return false }
                let roles = Set(batched.projections.map(\.field))
                return roles.isSuperset(of: ["q_proj", "k_proj", "v_proj"])
            }) {
                return .batched(entryIndex: batchedEntry.index)
            }

            let projectionEntries = group.compactMap { entry -> DispatchEntry? in
                guard case .projection(let projection, let isOutput) = entry.kind, !isOutput else {
                    return nil
                }
                return projection.field == "q_proj" || projection.field == "k_proj" || projection.field == "v_proj"
                    ? entry
                    : nil
            }

            let roles = Set(projectionEntries.compactMap { entry -> String? in
                guard case .projection(let projection, _) = entry.kind else { return nil }
                return projection.field
            })
            if roles.isSuperset(of: ["q_proj", "k_proj", "v_proj"]) {
                return .single(entryIndices: projectionEntries.map(\.index))
            }

            index = endIndex
        }

        return nil
    }

    private func firstMLPProjectionGroup(
        in entries: [DispatchEntry]
    ) -> ProjectionGroup? {
        var index = 0
        while index < entries.count {
            let compositeID = entries[index].compositeID
            var endIndex = index + 1
            while endIndex < entries.count, entries[endIndex].compositeID == compositeID {
                endIndex += 1
            }

            let group = Array(entries[index..<endIndex])
            if let fusedEntry = group.first(where: { entry in
                if case .fusedSwiGLUProjection = entry.kind { return true }
                return false
            }) {
                return .fused(entryIndex: fusedEntry.index)
            }

            let projectionEntries = group.compactMap { entry -> DispatchEntry? in
                guard case .projection(let projection, let isOutput) = entry.kind, !isOutput else {
                    return nil
                }
                return projection.field == "gate_proj" || projection.field == "up_proj"
                    ? entry
                    : nil
            }

            let roles = Set(projectionEntries.compactMap { entry -> String? in
                guard case .projection(let projection, _) = entry.kind else { return nil }
                return projection.field
            })
            if roles.isSuperset(of: ["gate_proj", "up_proj"]) {
                return .single(entryIndices: projectionEntries.map(\.index))
            }

            index = endIndex
        }

        return nil
    }

    private func assertCommonPrefillInput(
        group: ProjectionGroup,
        plan: MetalPrefillPlan,
        optimizerName: String,
        groupLabel: String
    ) throws {
        switch group {
        case .single(let entryIndices):
            let inputBindings = try entryIndices.map { entryIndex in
                let step = try #require(plan.steps.first(where: { $0.metadata.entryIndex == entryIndex }))
                return try #require(step.bindings.buffers.first(where: { $0.index == 0 }))
            }
            try assertCommonInputBindings(
                inputBindings,
                optimizerName: optimizerName,
                groupLabel: groupLabel,
                phase: "prefill"
            )
        case .batched(let entryIndex), .fused(let entryIndex):
            let step = try #require(plan.steps.first(where: { $0.metadata.entryIndex == entryIndex }))
            _ = try #require(step.bindings.buffers.first(where: { $0.index == 0 }))
        }
    }

    private func assertCommonDecodeInput(
        group: ProjectionGroup,
        plan: MetalDispatchPlan,
        optimizerName: String,
        groupLabel: String
    ) throws {
        switch group {
        case .single(let entryIndices):
            let inputBindings = try entryIndices.map { entryIndex in
                let step = try #require(plan.steps.first(where: { $0.metadata.entryIndex == entryIndex }))
                return try #require(step.bindings.buffers.first(where: { $0.index == 0 }))
            }
            try assertCommonInputBindings(
                inputBindings,
                optimizerName: optimizerName,
                groupLabel: groupLabel,
                phase: "decode"
            )
        case .batched(let entryIndex), .fused(let entryIndex):
            let step = try #require(plan.steps.first(where: { $0.metadata.entryIndex == entryIndex }))
            _ = try #require(step.bindings.buffers.first(where: { $0.index == 0 }))
        }
    }

    private func assertCommonInputBindings(
        _ inputBindings: [MetalBufferBinding],
        optimizerName: String,
        groupLabel: String,
        phase: String
    ) throws {
        let first = try #require(inputBindings.first)
        for binding in inputBindings.dropFirst() {
            #expect(
                binding.buffer === first.buffer,
                "\(optimizerName) optimizer \(groupLabel) sibling projections must read the same \(phase) source buffer"
            )
            #expect(
                binding.offset == first.offset,
                "\(optimizerName) optimizer \(groupLabel) sibling projections must read the same \(phase) source offset"
            )
        }
    }
}

private enum ProjectionGroup {
    case single(entryIndices: [Int])
    case batched(entryIndex: Int)
    case fused(entryIndex: Int)
}
