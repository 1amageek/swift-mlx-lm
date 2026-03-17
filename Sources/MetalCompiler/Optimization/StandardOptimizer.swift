/// Standard optimizer — norm fusion only (matches previous fusionPass behavior).
///
/// Fragment-level: no optimization (all primitives emitted individually).
/// Graph-level: fuses structuralAdd/Copy + Reduction patterns into single dispatches.
///
/// Patterns:
/// 1. structuralAdd + structuralCopy + Reduction → fusedResidualAddCopyNorm (3→1)
/// 2. structuralCopy + Reduction → fusedCopyNorm (2→1)
public struct StandardOptimizer: DispatchOptimizer {
    public let name = "standard"

    public init() {}

    public func optimizeFragment(_ primitives: [CollectedPrimitive]) -> [OptimizedEntry] {
        primitives.map { .single($0) }
    }

    public func optimizeGraph(_ entries: [DispatchEntry]) -> [DispatchEntry] {
        var result = entries
        var changed = true

        while changed {
            changed = false
            var index = 0

            while index < result.count {
                // Check if the fragment at index is a fusable reduction
                // Uses protocol properties only — no concrete type checks
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

                index += 1
            }
        }

        return result
    }
}
