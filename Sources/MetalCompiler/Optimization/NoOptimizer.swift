/// Baseline optimizer — no optimization applied.
///
/// Emits all primitives as individual dispatches and returns the graph unchanged.
/// Use as a baseline for benchmarking other optimization strategies.
public struct NoOptimizer: DispatchOptimizer {
    public let name = "none"

    public init() {}

    public func optimizeFragment(_ primitives: [CollectedPrimitive], context: FusionContext) -> [OptimizedEntry] {
        primitives.map { .single($0) }
    }

    public func optimizeGraph(_ entries: [DispatchEntry], context: FusionContext) -> [DispatchEntry] {
        entries
    }
}
