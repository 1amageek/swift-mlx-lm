import LMIR

// MARK: - Optimizer Protocol

/// Pluggable optimization strategy for dispatch entry generation.
///
/// The compiler calls the optimizer at two phases:
/// 1. **Fragment-level** (`optimizeFragment`): called per composite fragment during IR walk.
///    Primitives within a composite fragment are implicitly independent.
/// 2. **Graph-level** (`optimizeGraph`): called on the complete dispatch list after IR walk.
///    Handles cross-fragment fusion (structural operations + reductions).
///
/// Different implementations allow benchmarking and comparing strategies:
/// - `NoOptimizer` — baseline, no optimization
/// - `StandardOptimizer` — norm fusion only (current behavior)
/// - `AggressiveOptimizer` — full: projection batch + per-head batch + structural fusion
public protocol DispatchOptimizer: Sendable {
    /// Strategy name for benchmark reports.
    var name: String { get }

    /// Optimize primitives collected from a single composite fragment.
    ///
    /// Called once per IR primitive operation (Attention, MLP, ShortConv, etc.).
    /// Primitives within a composite fragment belong to the same operation —
    /// they operate on disjoint data and are implicitly independent.
    ///
    /// The optimizer may:
    /// - Batch consecutive non-output projections into a single dispatch
    /// - Batch consecutive in-place fragments with same dispatchDimension
    /// - Return primitives unmodified
    func optimizeFragment(
        _ primitives: [CollectedPrimitive]
    ) -> [OptimizedEntry]

    /// Optimize the complete dispatch list after IR walk.
    ///
    /// Handles cross-fragment fusion that requires seeing the full sequence:
    /// - structuralAdd + structuralCopy + fusable reduction → 3-to-1
    /// - structuralCopy + fusable reduction → 2-to-1
    /// - structuralAdd + fusable reduction → 2-to-1
    func optimizeGraph(
        _ entries: [DispatchEntry]
    ) -> [DispatchEntry]
}

// MARK: - Collected Primitive

/// A primitive fragment collected during fragment tree walk, before emission.
///
/// The compiler collects all primitives from a composite fragment's tree,
/// passes them to the optimizer, then emits the optimized result.
public struct CollectedPrimitive: Sendable {
    /// The primitive fragment (carries dispatchDimension, isFusable, isInPlace, etc.).
    public let fragment: any PrimitiveMetalKernelFragment
    /// Parameter bindings for weight resolution (layer-resolved).
    public let parameterBindings: [ParameterBinding]
    /// Layer index for this operation (nil if not in a repeating block).
    public let layerIndex: Int?

    public init(fragment: any PrimitiveMetalKernelFragment,
                parameterBindings: [ParameterBinding],
                layerIndex: Int?) {
        self.fragment = fragment
        self.parameterBindings = parameterBindings
        self.layerIndex = layerIndex
    }
}

// MARK: - Optimized Entry

/// Result of fragment-level optimization.
///
/// The optimizer returns a sequence of these, which the compiler
/// translates into `DispatchEntry` values.
public enum OptimizedEntry: Sendable {
    /// Single dispatch — no optimization applied.
    case single(CollectedPrimitive)
    /// Batched projections — N non-output GEMV in one dispatch.
    case batchedProjection(BatchedProjection, parameterBindings: [ParameterBinding], layerIndex: Int?)
    /// Batched same-dimension fragments — N in-place operations in one dispatch.
    case batchedFragment(BatchedFragment, parameterBindings: [ParameterBinding], layerIndex: Int?)
}

// MARK: - Optimization Report

/// Diagnostic report from an optimization pass.
///
/// Used for benchmarking: compare dispatch counts and pattern breakdowns
/// across different optimizer implementations.
public struct OptimizationReport: Sendable {
    public let optimizerName: String
    public let unfusedCount: Int
    public let optimizedCount: Int
    public let patterns: [PatternMatch]
    public var totalSaved: Int { unfusedCount - optimizedCount }

    public struct PatternMatch: Sendable {
        public let name: String
        public let count: Int
        public let savedDispatches: Int

        public init(name: String, count: Int, savedDispatches: Int) {
            self.name = name
            self.count = count
            self.savedDispatches = savedDispatches
        }
    }

    public init(optimizerName: String, unfusedCount: Int, optimizedCount: Int, patterns: [PatternMatch]) {
        self.optimizerName = optimizerName
        self.unfusedCount = unfusedCount
        self.optimizedCount = optimizedCount
        self.patterns = patterns
    }

    /// Print a formatted report to stdout.
    public func printReport() {
        print("[Optimizer: \(optimizerName)] \(unfusedCount) → \(optimizedCount) dispatches (saved \(totalSaved))")
        for p in patterns {
            print("  \(p.name): \(p.count)× saves \(p.savedDispatches)")
        }
    }
}
