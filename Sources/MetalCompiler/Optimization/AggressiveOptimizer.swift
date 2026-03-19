/// Aggressive optimizer — full optimization with projection batching and per-head batching.
///
/// Fragment-level optimization (during IR walk):
/// - Rule 1: Batch consecutive non-output projections with same inputDimension
/// - Rule 2: Batch consecutive in-place, fusable fragments with same dispatchDimension
///
/// Graph-level optimization (post IR walk):
/// - Pattern 1: structuralAdd + structuralCopy + fusable reduction → 3→1
/// - Pattern 2: structuralCopy + fusable reduction → 2→1
/// - Pattern 3: structuralAdd + fusable reduction → 2→1 (no copy)
///
/// All rules use fragment properties only — no concrete type checks.
public struct AggressiveOptimizer: DispatchOptimizer {
    public let name = "aggressive"

    public init() {}

    // MARK: - Fragment-Level Optimization

    public func optimizeFragment(_ primitives: [CollectedPrimitive]) -> [OptimizedEntry] {
        var result: [OptimizedEntry] = []
        var i = 0

        while i < primitives.count {
            if let fused = FusedSwiGLUProjectionRule.match(at: i, primitives: primitives) {
                result.append(fused)
                i += 3
                continue
            }

            // Rule 1: Batch consecutive projections (.gemv dispatch dimension)
            // Only batch non-output projections. The last projection in the
            // composite is likely the output projection (o_proj, down_proj) and
            // must remain as .projection for markLastProjectionAsOutput().
            if case .gemv(let outputDim, let inputDim) = primitives[i].fragment.dispatchDimension {
                var batch: [CollectedPrimitive] = [primitives[i]]
                var j = i + 1
                while j < primitives.count,
                      case .gemv(_, let nextInputDim) = primitives[j].fragment.dispatchDimension,
                      nextInputDim == inputDim {
                    // Stop before the last projection in the composite —
                    // it might be the output projection.
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
                        layerIndex: primitives[i].layerIndex))
                    i = j
                    continue
                }
            }

            // Rule 2: Batch consecutive in-place, fusable fragments with same dispatchDimension
            if primitives[i].fragment.isFusable && primitives[i].fragment.isInPlace {
                let baseDimension = primitives[i].fragment.dispatchDimension
                var batch: [CollectedPrimitive] = [primitives[i]]
                var j = i + 1
                while j < primitives.count,
                      primitives[j].fragment.isFusable,
                      primitives[j].fragment.isInPlace,
                      isSameDimensionKind(baseDimension, primitives[j].fragment.dispatchDimension) {
                    batch.append(primitives[j])
                    j += 1
                }

                if batch.count >= 2 {
                    // Compute combined dispatch dimension
                    let combinedDimension = combineDimensions(batch.map { $0.fragment.dispatchDimension })
                    let mergedBindings = batch.flatMap { $0.parameterBindings }
                    result.append(.batchedFragment(
                        BatchedFragment(
                            fragments: batch.map { $0.fragment },
                            dispatchDimension: combinedDimension),
                        parameterBindings: mergedBindings,
                        layerIndex: primitives[i].layerIndex))
                    i = j
                    continue
                }
            }

            // Default: emit as-is
            result.append(.single(primitives[i]))
            i += 1
        }

        return result
    }

    // MARK: - Graph-Level Optimization

    public func optimizeGraph(_ entries: [DispatchEntry]) -> [DispatchEntry] {
        var result = entries
        var changed = true

        while changed {
            changed = false
            var index = 0

            while index < result.count {
                // Check if the fragment at an index is a fusable reduction
                func isFusableReduction(at i: Int) -> (dimension: Int, epsilon: Float)? {
                    guard case .fragment(let frag) = result[i].kind,
                          frag.isFusable,
                          case .reduction(let dim) = frag.dispatchDimension,
                          let epsilon = frag.normEpsilon else {
                        return nil
                    }
                    return (dim, epsilon)
                }

                // Pattern 1: structuralAdd + structuralCopy + fusable reduction → 3→1
                if index + 2 < result.count,
                   case .structuralAdd(let addDimension) = result[index].kind,
                   case .structuralCopy = result[index + 1].kind,
                   let reduction = isFusableReduction(at: index + 2) {
                    let fused = DispatchEntry(
                        index: result[index].index,
                        kind: .fusedResidualAddCopyNorm(FusedResidualAddCopyNorm(
                            dimension: addDimension, epsilon: reduction.epsilon)),
                        parameterBindings: result[index + 2].parameterBindings,
                        layerIndex: result[index].layerIndex)
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
                            dimension: reduction.dimension, epsilon: reduction.epsilon)),
                        parameterBindings: result[index + 1].parameterBindings,
                        layerIndex: result[index].layerIndex)
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
                            dimension: addDimension, epsilon: reduction.epsilon)),
                        parameterBindings: result[index + 1].parameterBindings,
                        layerIndex: result[index].layerIndex)
                    result.replaceSubrange(index...index + 1, with: [fused])
                    changed = true
                    continue
                }

                index += 1
            }
        }

        return result
    }
    // MARK: - Helpers

    /// Check if two dispatch dimensions are the same KIND (ignoring values).
    private func isSameDimensionKind(
        _ a: MetalDispatchDimension,
        _ b: MetalDispatchDimension
    ) -> Bool {
        switch (a, b) {
        case (.reduction, .reduction): return true
        case (.elementwise, .elementwise): return true
        case (.perHead, .perHead): return true
        case (.gather, .gather): return true
        case (.gemv, .gemv): return true
        default: return false
        }
    }

    /// Combine dispatch dimensions for batched fragments.
    /// For perHead: sum the head counts.
    /// For elementwise: sum the element counts.
    private func combineDimensions(_ dimensions: [MetalDispatchDimension]) -> MetalDispatchDimension {
        guard let first = dimensions.first else { return .elementwise(count: 0) }
        switch first {
        case .perHead:
            let totalHeads = dimensions.reduce(0) { total, dim in
                if case .perHead(let count) = dim { return total + count }
                return total
            }
            return .perHead(headCount: totalHeads)
        case .elementwise:
            let totalCount = dimensions.reduce(0) { total, dim in
                if case .elementwise(let count) = dim { return total + count }
                return total
            }
            return .elementwise(count: totalCount)
        case .reduction:
            let totalDim = dimensions.reduce(0) { total, dim in
                if case .reduction(let d) = dim { return total + d }
                return total
            }
            return .reduction(dimension: totalDim)
        default:
            return first
        }
    }
}
