/// Strategy for combining residual connections.
public enum ResidualStrategy: Codable, Equatable, Sendable {
    case add
    case weighted
    case gated
    case custom(String)
}

/// Strategy for merging parallel branch outputs.
///
/// All concrete strategies describe **tensor-level** combination.
/// They do NOT affect value-flow arity at the IR level — all branches
/// must produce equal result arity, and `operation.results.count`
/// equals that arity.
///
/// The difference between strategies is in tensor semantics
/// (handled by the compiler/executor), not in IR value flow:
///
/// - `.add`: element-wise addition of corresponding branch tensors.
/// - `.concat`: concatenation along a tensor dimension (shape changes,
///   value count unchanged).
/// - `.stack`: stacking along a new tensor dimension.
/// - `.custom`: user-defined combination; no arity constraint at IR level.
///
/// ## Arity Contract (enforced by `GraphValidator`)
///
/// For `.add`, `.concat`, `.stack`:
///   `∀ i, j: branches[i].results.count == branches[j].results.count`
///   `operation.results.count == branches[0].results.count`
///
/// For `.custom(String)`: no constraint.
public enum ParallelMergeStrategy: Codable, Equatable, Sendable {
    case add
    case concat
    case stack
    case custom(String)
}
