import LMIR

struct MetalEntryCollector {

    func collect(
        using context: CompileContext,
        kernelContext: KernelContext
    ) -> (walkContext: WalkContext, unfusedCount: Int, fusedEntries: [DispatchEntry]) {
        let hasOutputHead = context.graph.rootRegion.operations.contains { operation in
            if case .primitive(let attributes) = operation.kind {
                return attributes is OutputHeadAttributes
            }
            return false
        }
        let fusionContext = FusionContext(
            device: context.device,
            hiddenSize: context.hiddenSize,
            maximumSequenceLength: context.maximumSequenceLength,
            hasOutputHead: hasOutputHead
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
        let fusedEntries = fuseCrossComponent(walkContext.entries, context: fusionContext)
        let finalEntries = decomposeQuantizedBatchedProjectionsForDecode(
            fusedEntries,
            stafWeightStore: context.stafWeightStore,
            kernelContext: kernelContext
        )
        return (walkContext, unfusedCount, finalEntries)
    }

    // MARK: - Quantized BatchedProjection Decomposition

    /// For decode, decompose `BatchedProjection` into per-projection `LinearFragment`
    /// entries when any projection has a quantized weight format.
    ///
    /// Reason: the batched GEMV kernel (`batched_gemv2/3/4`) is generated assuming
    /// dense weight layout (`weightRow[base + j]`). With Q4/Q8 block-packed weights
    /// this emits `dequantize(...)` without a matching helper — Metal compilation
    /// fails. Dense BF16/FP16 decode keeps `BatchedProjection` intact and benefits
    /// from multi-output GEMV.
    ///
    /// Prefill is unaffected: the catalog already decomposes for source generation
    /// (`sourceGenerationEntries` in `MetalKernelSourceCatalog`) and prefill step
    /// building routes through `buildBatchedProjectionPrefillSteps`.
    private func decomposeQuantizedBatchedProjectionsForDecode(
        _ entries: [DispatchEntry],
        stafWeightStore: STAFWeightStore?,
        kernelContext: KernelContext
    ) -> [DispatchEntry] {
        // Only decode is affected (prefill has a separate path).
        guard kernelContext.bufferPrecision != .float32 else { return entries }
        guard let stafWeightStore else { return entries }

        return entries.flatMap { entry -> [DispatchEntry] in
            guard let batched = entry.fragment as? BatchedProjection else {
                return [entry]
            }
            let hasQuantized = batched.projections.contains { projection in
                guard let binding = entry.parameterBindings.first(where: { $0.role == projection.field }),
                      let info = stafWeightStore.tensor(for: binding.tensorName) else {
                    return false
                }
                // Any sub-16-bit weight format uses block-packed dequantization
                // and therefore cannot flow through the dense batched GEMV path.
                return info.format.bits < 16
            }
            guard hasQuantized else { return [entry] }

            return batched.projections.map { projection in
                DispatchEntry(
                    index: entry.index,
                    fragment: LinearFragment(
                        field: projection.field,
                        inputDimension: projection.inputDimension,
                        outputDimension: projection.outputDimension,
                        isOutput: false
                    ),
                    parameterBindings: entry.parameterBindings,
                    layerIndex: entry.layerIndex,
                    compositeID: entry.compositeID
                )
            }
        }
    }

    // MARK: - Cross-Component Optimization

    /// Fixed cross-component fusion pass.
    ///
    /// Uses FusionContract compatibility to fuse adjacent fragment entries into
    /// SynthesizedFragment kernels. Also enables direct-scratch attention when
    /// no OutputHead is present (embedding-only models skip KV cache).
    private func fuseCrossComponent(
        _ entries: [DispatchEntry],
        context: FusionContext
    ) -> [DispatchEntry] {
        var result = entries

        // Direct-scratch attention: when no OutputHead, FlashAttention skips KV cache
        // and reads K/V directly from scratch buffers. This is a cross-component
        // optimization (depends on graph-level knowledge of OutputHead presence).
        if !context.hasOutputHead {
            for i in 0..<result.count {
                guard let attn = result[i].fragment as? FlashAttentionFragment,
                      !attn.directScratchMode else { continue }
                let modified = FlashAttentionFragment(
                    headCount: attn.headCount,
                    kvHeadCount: attn.kvHeadCount,
                    headDimension: attn.headDimension,
                    attentionScale: attn.attentionScale,
                    ropeDimension: attn.ropeDimension,
                    ropeBase: attn.ropeBase,
                    ropeScaling: attn.ropeScaling,
                    mropeAxes: attn.mropeAxes,
                    querySlotIndex: attn.querySlotIndex,
                    causal: attn.causal,
                    windowLeft: attn.windowLeft,
                    windowRight: attn.windowRight,
                    sharedKVSourceLayerIndex: attn.sharedKVSourceLayerIndex,
                    directScratchMode: true,
                    suppressPrefillRoPE: attn.suppressPrefillRoPE
                )
                result[i] = DispatchEntry(
                    index: result[i].index,
                    fragment: modified,
                    parameterBindings: result[i].parameterBindings,
                    layerIndex: result[i].layerIndex,
                    compositeID: result[i].compositeID
                )
            }
        }

        // Generic fusion: adjacent .fragment entries with compatible FusionContracts.
        // Uses FusionContract only — no concrete type inspection.
        //
        // When extending an existing SynthesizedFragment, flatten the leaf fragments
        // rather than nesting. SynthesizedFragment always holds a flat list of leaf
        // PrimitiveMetalKernelFragments — this ensures synthesize() processes all
        // fragments in a single pass with consistent variable naming.
        var changed = true

        while changed {
            changed = false
            var index = 0

            while index < result.count {
                if index + 1 < result.count,
                   let contractA = result[index].fragment.fusionContract,
                   let contractB = result[index + 1].fragment.fusionContract,
                   contractA.parallelism.isCompatible(with: contractB.parallelism),
                   contractA.primaryOutput != nil,
                   contractB.primaryInput != nil {
                    // Verify combined threadgroup memory fits
                    let storage = contractA.intermediateStorage(to: contractB)
                    var tgMemory = max(contractA.threadgroupMemoryBytes, contractB.threadgroupMemoryBytes)
                    if case .threadgroupMemory(let dim) = storage {
                        tgMemory += dim * MemoryLayout<Float>.size
                    }
                    if tgMemory <= context.threadgroupMemoryLimit {
                        // Flatten: unpack existing SynthesizedFragments into leaf fragments
                        let leftFragments: [any PrimitiveMetalKernelFragment]
                        if let synth = result[index].fragment as? SynthesizedFragment {
                            leftFragments = synth.fragments
                        } else {
                            leftFragments = [result[index].fragment]
                        }
                        let rightFragments: [any PrimitiveMetalKernelFragment]
                        if let synth = result[index + 1].fragment as? SynthesizedFragment {
                            rightFragments = synth.fragments
                        } else {
                            rightFragments = [result[index + 1].fragment]
                        }
                        let flatFragments = leftFragments + rightFragments

                        // Build entries from leaf contracts and compute resolved parallelism
                        let flatEntries = flatFragments.map { frag in
                            FusionSynthesizer.Entry(contract: frag.fusionContract!, body: "")
                        }
                        var resolvedParallelism = flatEntries[0].contract.parallelism
                        for i in 1..<flatEntries.count {
                            resolvedParallelism = resolvedParallelism.resolved(with: flatEntries[i].contract.parallelism)
                        }

                        let mergedContract = FusionSynthesizer.mergeContracts(
                            entries: flatEntries,
                            resolvedParallelism: resolvedParallelism
                        )
                        let synthesized = SynthesizedFragment(
                            fragments: flatFragments,
                            mergedContract: mergedContract
                        )
                        let mergedBindings = result[index].parameterBindings + result[index + 1].parameterBindings
                        let fused = DispatchEntry(
                            index: result[index].index,
                            fragment: synthesized,
                            parameterBindings: mergedBindings,
                            layerIndex: result[index].layerIndex,
                            compositeID: nil
                        )
                        result.replaceSubrange(index...index + 1, with: [fused])
                        changed = true
                        continue
                    }
                }

                index += 1
            }
        }

        return result
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
                context.emit(CopyFragment(dimension: hiddenSize), layerIndex: layerIndex)
                walkRegion(
                    body,
                    pathComponents: operationPath + [.regionBody],
                    layerIndex: layerIndex,
                    hiddenSize: hiddenSize,
                    context: &context,
                    kernelContext: kernelContext,
                    fusionContext: fusionContext
                )
                context.emit(ResidualAddFragment(dimension: hiddenSize), layerIndex: layerIndex)

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

                guard let compilable = attributes as? any MetalCompilable else {
                    fatalError("OperationAttributes type \(type(of: attributes)) does not conform to MetalCompilable")
                }
                let fragment = compilable.fragment(context: kernelContext)

                // MetalCompilable returns already-optimized fragments.
                // Collect primitives and emit directly — no optimizer intervention.
                var primitives: [CollectedPrimitive] = []
                collectPrimitives(
                    fragment,
                    bindings: bindings,
                    layerIndex: layerIndex,
                    primitives: &primitives,
                    context: &context,
                    kernelContext: kernelContext
                )
                let startIndex = context.entries.count
                let compositeID = context.nextCompositeID
                context.nextCompositeID += 1
                for primitive in primitives {
                    let disambiguated = disambiguateWeightRoles(
                        primitive,
                        emissionIndex: context.nextIndex
                    )
                    context.emitPrimitive(disambiguated, compositeID: compositeID)
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

    /// Disambiguate weight roles on Reduction fragments that default to "scale".
    ///
    /// Multiple RMSNorm operations (e.g., `input_layernorm`, `post_attention_layernorm`,
    /// `pre_feedforward_layernorm`, `post_feedforward_layernorm` in Gemma sandwich norms)
    /// all have ParameterBindings with `role: "scale"`. When cross-component fusion
    /// concatenates bindings from two such operations into a single DispatchEntry,
    /// `WeightResolver.resolve(role: "scale")` returns the first match both times,
    /// causing both norms in the fused kernel to read the same weight buffer.
    ///
    /// Rewrite each Reduction's `weightRole` and its matching binding's role to a
    /// globally unique token keyed on the emission index so that post-fusion
    /// resolution remains unambiguous.
    private func disambiguateWeightRoles(
        _ primitive: CollectedPrimitive,
        emissionIndex: Int
    ) -> CollectedPrimitive {
        guard let reduction = primitive.fragment as? Reduction,
              reduction.withScale else {
            return primitive
        }
        let originalRole = reduction.weightRole
        let uniqueRole = "\(originalRole)#e\(emissionIndex)"
        let renamedFragment = Reduction(
            dimension: reduction.dimension,
            epsilon: reduction.epsilon,
            weightRole: uniqueRole,
            weightBias: reduction.weightBias,
            withScale: reduction.withScale
        )
        var didRenameBinding = false
        let renamedBindings = primitive.parameterBindings.map { binding -> ParameterBinding in
            if !didRenameBinding, binding.role == originalRole {
                didRenameBinding = true
                return ParameterBinding(role: uniqueRole, tensorName: binding.tensorName)
            }
            return binding
        }
        return CollectedPrimitive(
            fragment: renamedFragment,
            parameterBindings: renamedBindings,
            layerIndex: primitive.layerIndex
        )
    }

    private func markLastProjectionAsOutput(entries: inout [DispatchEntry], from startIndex: Int) {
        for index in stride(from: entries.count - 1, through: startIndex, by: -1) {
            if let projection = entries[index].fragment as? ProjectionDescribing {
                let updated = projection.withOutputProjectionEnabled()
                entries[index] = DispatchEntry(
                    index: entries[index].index,
                    fragment: updated,
                    parameterBindings: entries[index].parameterBindings,
                    layerIndex: entries[index].layerIndex,
                    compositeID: entries[index].compositeID
                )
                break
            }
        }
    }
}
