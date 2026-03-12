@preconcurrency import MLX
import MLXNN
import SwiftLM

// MARK: - GraphSequential

/// Sequential execution container compiled from ModelGraph.
///
/// Passes output from each child module as input to the next.
/// All child modules must conform to `UnaryLayer`.
final class GraphSequential: Module, UnaryLayer {

    @ModuleInfo var modules: [Module]

    init(_ modules: [Module]) {
        self._modules.wrappedValue = modules
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for m in modules {
            h = (m as! UnaryLayer)(h)
        }
        return h
    }
}

// MARK: - GraphResidual

/// Residual connection: output = input + body(input).
///
/// The body is a `GraphSequential` containing the sub-operations.
final class GraphResidual: Module, UnaryLayer {

    @ModuleInfo var body: GraphSequential

    let strategy: ResidualStrategy

    init(body: GraphSequential, strategy: ResidualStrategy = .add) {
        self.strategy = strategy
        self._body.wrappedValue = body
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + body(x)
    }
}

// MARK: - GraphRepeating

/// Repeated block: applies N copies of a sub-block sequentially.
///
/// Each iteration has its own module instance with distinct weights,
/// corresponding to different `StructuralPath` indices.
final class GraphRepeating: Module, UnaryLayer {

    @ModuleInfo var iterations: [Module]

    init(iterations: [Module]) {
        self._iterations.wrappedValue = iterations
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for iter in iterations {
            h = (iter as! UnaryLayer)(h)
        }
        return h
    }
}

// MARK: - GraphParallel

/// Parallel branches with merge strategy.
///
/// Each branch receives the same input; outputs are merged
/// according to the configured strategy (add, concat, stack).
final class GraphParallel: Module, UnaryLayer {

    @ModuleInfo var branches: [Module]

    let merge: ParallelMergeStrategy

    init(branches: [Module], merge: ParallelMergeStrategy) {
        self.merge = merge
        self._branches.wrappedValue = branches
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let results = branches.map { ($0 as! UnaryLayer)(x) }

        switch merge {
        case .add:
            return results.dropFirst().reduce(results[0]) { $0 + $1 }
        case .concat:
            return concatenated(results, axis: -1)
        case .stack:
            return stacked(results)
        case .custom:
            fatalError("Unsupported parallel merge strategy: \(merge)")
        }
    }
}
