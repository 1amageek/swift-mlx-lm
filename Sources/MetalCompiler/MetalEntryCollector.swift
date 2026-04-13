import LMIR

struct MetalEntryCollector {
    let optimizer: any DispatchOptimizer

    func collect(
        using context: CompileContext,
        kernelContext: KernelContext
    ) -> (walkContext: WalkContext, unfusedCount: Int, fusedEntries: [DispatchEntry]) {
        let fusionContext = FusionContext(
            device: context.device,
            hiddenSize: context.hiddenSize,
            maximumSequenceLength: context.maximumSequenceLength
        )
        var walkContext = WalkContext()
        walkRegion(
            context.graph.rootRegion,
            pathComponents: [],
            layerIndex: nil,
            hiddenSize: context.hiddenSize,
            context: &walkContext,
            kernelContext: kernelContext,
            fusionContext: fusionContext
        )
        let unfusedCount = walkContext.entries.count
        let fusedEntries = optimizer.optimizeGraph(walkContext.entries, context: fusionContext)
        return (walkContext, unfusedCount, fusedEntries)
    }

    private func walkRegion(
        _ region: Region,
        pathComponents: [StructuralPathComponent],
        layerIndex: Int?,
        hiddenSize: Int,
        context: inout WalkContext,
        kernelContext: KernelContext,
        fusionContext: FusionContext
    ) {
        var implicitLayerIndex = 0
        for (operationIndex, operation) in region.operations.enumerated() {
            let operationPath = pathComponents + [.operation(operationIndex)]
            let _ = StructuralPath(components: operationPath)

            switch operation.kind {
            case .residual(_, let body):
                context.emit(.structuralCopy(dimension: hiddenSize), layerIndex: layerIndex)
                walkRegion(
                    body,
                    pathComponents: operationPath + [.regionBody],
                    layerIndex: layerIndex,
                    hiddenSize: hiddenSize,
                    context: &context,
                    kernelContext: kernelContext,
                    fusionContext: fusionContext
                )
                context.emit(.structuralAdd(dimension: hiddenSize), layerIndex: layerIndex)

            case .repeating(let count, let body):
                let baseLayerIndex = layerIndex == nil ? implicitLayerIndex : nil
                for iteration in 0..<count {
                    walkRegion(
                        body,
                        pathComponents: operationPath + [.regionBody, .index(iteration)],
                        layerIndex: baseLayerIndex.map { $0 + iteration } ?? iteration,
                        hiddenSize: hiddenSize,
                        context: &context,
                        kernelContext: kernelContext,
                        fusionContext: fusionContext
                    )
                }
                if layerIndex == nil {
                    implicitLayerIndex += count
                }

            case .conditional(let condition, let thenBody, let elseBody):
                if let currentLayer = layerIndex, case .layerIndices(let indices) = condition {
                    let selectedBody = indices.contains(currentLayer) ? thenBody : elseBody
                    walkRegion(
                        selectedBody,
                        pathComponents: operationPath + [.regionBody],
                        layerIndex: currentLayer,
                        hiddenSize: hiddenSize,
                        context: &context,
                        kernelContext: kernelContext,
                        fusionContext: fusionContext
                    )
                } else {
                    walkRegion(
                        thenBody,
                        pathComponents: operationPath + [.regionBody],
                        layerIndex: layerIndex,
                        hiddenSize: hiddenSize,
                        context: &context,
                        kernelContext: kernelContext,
                        fusionContext: fusionContext
                    )
                }

            case .parallel(_, let branches):
                for (branchIndex, branch) in branches.enumerated() {
                    walkRegion(
                        branch,
                        pathComponents: operationPath + [.regionBranch(branchIndex)],
                        layerIndex: layerIndex,
                        hiddenSize: hiddenSize,
                        context: &context,
                        kernelContext: kernelContext,
                        fusionContext: fusionContext
                    )
                }

            case .primitive(let attributes):
                let bindings: [ParameterBinding]
                if let currentLayerIndex = layerIndex {
                    bindings = operation.parameterBindings.map { binding in
                        let resolved = binding.tensorName.replacingOccurrences(
                            of: ".layers.0.",
                            with: ".layers.\(currentLayerIndex)."
                        )
                        return ParameterBinding(role: binding.role, tensorName: resolved)
                    }
                } else {
                    bindings = operation.parameterBindings
                }

                guard let fragment = attributes as? (any MetalKernelFragment) else {
                    continue
                }

                var primitives: [CollectedPrimitive] = []
                collectPrimitives(
                    fragment,
                    bindings: bindings,
                    layerIndex: layerIndex,
                    primitives: &primitives,
                    context: &context,
                    kernelContext: kernelContext
                )
                let optimized = optimizer.optimizeFragment(primitives, context: fusionContext)
                let startIndex = context.entries.count
                let compositeID = context.nextCompositeID
                context.nextCompositeID += 1
                for entry in optimized {
                    context.emitOptimized(entry, compositeID: compositeID)
                }
                markLastProjectionAsOutput(entries: &context.entries, from: startIndex)
            }
        }
    }

    private func collectPrimitives(
        _ fragment: any MetalKernelFragment,
        bindings: [ParameterBinding],
        layerIndex: Int?,
        primitives: inout [CollectedPrimitive],
        context: inout WalkContext,
        kernelContext: KernelContext
    ) {
        if let primitive = fragment as? any PrimitiveMetalKernelFragment {
            for slot in primitive.cacheSlots where slot.kind == .kv {
                context.cacheSlots.append(
                    CacheSlotInfo(
                        kvHeadCount: slot.kvHeadCount,
                        headDimension: slot.headDimension
                    )
                )
            }
            primitives.append(
                CollectedPrimitive(
                    fragment: primitive,
                    parameterBindings: bindings,
                    layerIndex: layerIndex
                )
            )
            return
        }

        if let tuple = fragment as? any _TupleFragmentProtocol {
            tuple._visitChildren { child in
                collectPrimitives(
                    child,
                    bindings: bindings,
                    layerIndex: layerIndex,
                    primitives: &primitives,
                    context: &context,
                    kernelContext: kernelContext
                )
            }
            return
        }
        if let optional = fragment as? any _OptionalFragmentProtocol {
            optional._visitContent { child in
                collectPrimitives(
                    child,
                    bindings: bindings,
                    layerIndex: layerIndex,
                    primitives: &primitives,
                    context: &context,
                    kernelContext: kernelContext
                )
            }
            return
        }
        if let conditional = fragment as? any _ConditionalFragmentProtocol {
            conditional._visitActive { child in
                collectPrimitives(
                    child,
                    bindings: bindings,
                    layerIndex: layerIndex,
                    primitives: &primitives,
                    context: &context,
                    kernelContext: kernelContext
                )
            }
            return
        }
        if let bodyAccessor = fragment as? any _FragmentBodyAccessor {
            bodyAccessor._visitBody(context: kernelContext) { child in
                collectPrimitives(
                    child,
                    bindings: bindings,
                    layerIndex: layerIndex,
                    primitives: &primitives,
                    context: &context,
                    kernelContext: kernelContext
                )
            }
        }
    }

    private func markLastProjectionAsOutput(entries: inout [DispatchEntry], from startIndex: Int) {
        for index in stride(from: entries.count - 1, through: startIndex, by: -1) {
            if case .projection(let projection, _) = entries[index].kind {
                entries[index] = DispatchEntry(
                    index: entries[index].index,
                    kind: .projection(projection, isOutput: true),
                    parameterBindings: entries[index].parameterBindings,
                    layerIndex: entries[index].layerIndex,
                    compositeID: entries[index].compositeID
                )
                break
            }
        }
    }
}
