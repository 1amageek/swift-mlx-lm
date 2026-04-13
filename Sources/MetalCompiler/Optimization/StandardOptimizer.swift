/// Standard optimizer — norm fusion plus exact-shape MLP front-half fusion.
///
/// Fragment-level:
/// - Rule 1: Fuse gate_proj + up_proj + SwiGLU into one dispatch
/// - Rule 2: Batch consecutive non-output projections with same inputDimension
///
/// Graph-level:
/// Graph-level: fuses structuralAdd/Copy + Reduction patterns into single dispatches.
///
/// Patterns:
/// 1. gate_proj + up_proj + SwiGLU → fusedSwiGLUProjection (3→1)
/// 2. Consecutive .gemv projections with same inputDimension → batchedProjection (N→1)
/// 3. structuralAdd + structuralCopy + Reduction → fusedResidualAddCopyNorm (3→1)
/// 4. structuralCopy + Reduction → fusedCopyNorm (2→1)
/// 5. structuralAdd + Reduction → fusedResidualAddNorm (2→1)
public struct StandardOptimizer: DispatchOptimizer {
    public let name = "standard"

    public init() {}

    public func optimizeFragment(_ primitives: [CollectedPrimitive], context: FusionContext) -> [OptimizedEntry] {
        var result: [OptimizedEntry] = []
        var index = 0

        while index < primitives.count {
            // Rule 1: Fuse gate_proj + up_proj + SwiGLU
            if let fused = FusedSwiGLUProjectionRule.match(at: index, primitives: primitives) {
                result.append(fused)
                index += 3
                continue
            }

            // Rule 2: Batch consecutive projections (.gemv) with same inputDimension.
            // Stops before the last projection in the composite — it is likely the
            // output projection (o_proj, down_proj) which must remain individual
            // for markLastProjectionAsOutput().
            if case .gemv(_, let inputDim) = primitives[index].fragment.dispatchDimension {
                var batch: [CollectedPrimitive] = [primitives[index]]
                var j = index + 1
                while j < primitives.count,
                      case .gemv(_, let nextInputDim) = primitives[j].fragment.dispatchDimension,
                      nextInputDim == inputDim {
                    let isLastProjection = !primitives[(j+1)...].contains {
                        if case .gemv = $0.fragment.dispatchDimension { return true }
                        return false
                    }
                    if isLastProjection { break }
                    batch.append(primitives[j])
                    j += 1
                }

                if batch.count >= 2 {
                    let projections = batch.map { p in
                        guard case .gemv(let outDim, let inDim) = p.fragment.dispatchDimension else {
                            fatalError("Expected .gemv dispatchDimension in batched projection")
                        }
                        let field = p.fragment.weightSlots.first?.field ?? "weight"
                        return BatchedProjection.Entry(
                            field: field,
                            inputDimension: inDim,
                            outputDimension: outDim)
                    }
                    let mergedBindings = batch.flatMap { $0.parameterBindings }
                    result.append(.batchedProjection(
                        BatchedProjection(projections: projections),
                        parameterBindings: mergedBindings,
                        layerIndex: primitives[index].layerIndex))
                    index = j
                    continue
                }
            }

            // Rule 3: Batch consecutive QKNorm fragments into a single dispatch.
            // QKNorm Q + QKNorm K share the same headDimension and epsilon,
            // so they can be batched into one kernel with combined grid.
            if primitives[index].fragment is QKNormFragment,
               index + 1 < primitives.count,
               primitives[index + 1].fragment is QKNormFragment {
                let batch = [primitives[index], primitives[index + 1]]
                let combinedHeadCount = batch.reduce(0) { total, p in
                    if case .perHead(let count) = p.fragment.dispatchDimension { return total + count }
                    return total
                }
                let mergedBindings = batch.flatMap { $0.parameterBindings }
                result.append(.batchedFragment(
                    BatchedFragment(
                        fragments: batch.map { $0.fragment },
                        dispatchDimension: .perHead(headCount: combinedHeadCount)),
                    parameterBindings: mergedBindings,
                    layerIndex: primitives[index].layerIndex))
                index += 2
                continue
            }

            result.append(.single(primitives[index]))
            index += 1
        }

        return result
    }

    public func optimizeGraph(_ entries: [DispatchEntry], context: FusionContext) -> [DispatchEntry] {
        var result = entries
        var changed = true

        while changed {
            changed = false
            var index = 0

            while index < result.count {
                // Check if the fragment at index is a fusable reduction
                // Uses protocol properties only — no concrete type checks
                func isFusableReduction(at i: Int) -> (dimension: Int, epsilon: Float, weightBias: Float)? {
                    guard case .fragment(let frag) = result[i].kind,
                          frag.isFusable,
                          case .reduction(let dim) = frag.dispatchDimension,
                          let epsilon = frag.normEpsilon else {
                        return nil
                    }
                    return (dim, epsilon, frag.normWeightBias ?? 0)
                }

                // Pattern 0a: fusable reduction + structuralAdd + structuralCopy + fusable reduction → 4→1
                // Absorbs preceding post-norm into fused residual operation.
                // Checked first (largest pattern) per greedy largest-first algorithm.
                if index + 3 < result.count,
                   let preNormReduction = isFusableReduction(at: index),
                   case .structuralAdd(let addDimension) = result[index + 1].kind,
                   case .structuralCopy = result[index + 2].kind,
                   let outputReduction = isFusableReduction(at: index + 3) {
                    // Feasibility: fused kernel needs shared memory for sequential reductions
                    let reductionThreads = min(context.hiddenSize, context.maxThreadsPerThreadgroup)
                    let sharedMemoryNeeded = reductionThreads * MemoryLayout<Float>.size
                    if sharedMemoryNeeded <= context.threadgroupMemoryLimit {
                        let preNorm = FusedResidualAddCopyNorm.PreNorm(
                            epsilon: preNormReduction.epsilon,
                            weightBias: preNormReduction.weightBias,
                            parameterBindings: result[index].parameterBindings)
                        let fused = DispatchEntry(
                            index: result[index].index,
                            kind: .fusedResidualAddCopyNorm(FusedResidualAddCopyNorm(
                                dimension: addDimension, epsilon: outputReduction.epsilon,
                                weightBias: outputReduction.weightBias, preNorm: preNorm)),
                            parameterBindings: result[index + 3].parameterBindings,
                            layerIndex: result[index].layerIndex,
                            compositeID: nil)
                        result.replaceSubrange(index...index + 3, with: [fused])
                        changed = true
                        continue
                    }
                }

                // Pattern 0b: fusable reduction + structuralAdd + fusable reduction → 3→1 (no copy)
                // Entry at index is a reduction (not structuralAdd), so no overlap with Pattern 1.
                if index + 2 < result.count,
                   let preNormReduction = isFusableReduction(at: index),
                   case .structuralAdd(let addDimension) = result[index + 1].kind,
                   let outputReduction = isFusableReduction(at: index + 2) {
                    let reductionThreads = min(context.hiddenSize, context.maxThreadsPerThreadgroup)
                    let sharedMemoryNeeded = reductionThreads * MemoryLayout<Float>.size
                    if sharedMemoryNeeded <= context.threadgroupMemoryLimit {
                        let preNorm = FusedResidualAddCopyNorm.PreNorm(
                            epsilon: preNormReduction.epsilon,
                            weightBias: preNormReduction.weightBias,
                            parameterBindings: result[index].parameterBindings)
                        let fused = DispatchEntry(
                            index: result[index].index,
                            kind: .fusedResidualAddNorm(FusedResidualAddNorm(
                                dimension: addDimension, epsilon: outputReduction.epsilon,
                                weightBias: outputReduction.weightBias, preNorm: preNorm)),
                            parameterBindings: result[index + 2].parameterBindings,
                            layerIndex: result[index].layerIndex,
                            compositeID: nil)
                        result.replaceSubrange(index...index + 2, with: [fused])
                        changed = true
                        continue
                    }
                }

                // Pattern 1: structuralAdd + structuralCopy + fusable reduction → 3→1
                if index + 2 < result.count,
                   case .structuralAdd(let addDimension) = result[index].kind,
                   case .structuralCopy = result[index + 1].kind,
                   let reduction = isFusableReduction(at: index + 2) {
                    let fused = DispatchEntry(
                        index: result[index].index,
                        kind: .fusedResidualAddCopyNorm(FusedResidualAddCopyNorm(
                            dimension: addDimension, epsilon: reduction.epsilon,
                            weightBias: reduction.weightBias)),
                        parameterBindings: result[index + 2].parameterBindings,
                        layerIndex: result[index].layerIndex,
                        compositeID: nil)
                    result.replaceSubrange(index...index + 2, with: [fused])
                    changed = true
                    continue
                }

                // Pattern 2: structuralCopy + fusable reduction → 2→1
                if index + 1 < result.count,
                   case .structuralCopy = result[index].kind,
                   let reduction = isFusableReduction(at: index + 1) {
                    let fused = DispatchEntry(
                        index: result[index].index,
                        kind: .fusedCopyNorm(FusedCopyNorm(
                            dimension: reduction.dimension, epsilon: reduction.epsilon,
                            weightBias: reduction.weightBias)),
                        parameterBindings: result[index + 1].parameterBindings,
                        layerIndex: result[index].layerIndex,
                        compositeID: nil)
                    result.replaceSubrange(index...index + 1, with: [fused])
                    changed = true
                    continue
                }

                // Pattern 3: structuralAdd + fusable reduction → 2→1 (no copy)
                if index + 1 < result.count,
                   case .structuralAdd(let addDimension) = result[index].kind,
                   let reduction = isFusableReduction(at: index + 1) {
                    let fused = DispatchEntry(
                        index: result[index].index,
                        kind: .fusedResidualAddNorm(FusedResidualAddNorm(
                            dimension: addDimension, epsilon: reduction.epsilon,
                            weightBias: reduction.weightBias)),
                        parameterBindings: result[index + 1].parameterBindings,
                        layerIndex: result[index].layerIndex,
                        compositeID: nil)
                    result.replaceSubrange(index...index + 1, with: [fused])
                    changed = true
                    continue
                }

                index += 1
            }
        }

        return result
    }
}
