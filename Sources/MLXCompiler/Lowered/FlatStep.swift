@preconcurrency import MLX
import SwiftLM

/// Flattened execution step — no recursion needed.
///
/// Residual connections use save/add markers instead of recursive nesting.
/// Parallel branches are preserved as-is (rare, used only in MoE/hybrid models).
public enum FlatStep: @unchecked Sendable {

    /// A leaf operation.
    case op(LoweredInferenceOp)

    /// Save current hidden state for later residual addition.
    case saveResidual

    /// Add saved residual to current hidden state.
    case addResidual

    /// Parallel branches (kept recursive — rare in practice).
    case parallel(merge: ParallelMergeStrategy, branches: [[FlatStep]])
}

// MARK: - Flattening

/// Flatten recursive LoweredSteps into linear FlatSteps.
///
/// Converts `.residual(body:)` into `[.saveResidual, ...body..., .addResidual]`.
/// `.parallel` branches are recursively flattened but kept as a single step.
func flattenSteps(_ steps: [LoweredStep]) -> [FlatStep] {
    var result: [FlatStep] = []
    for step in steps {
        switch step {
        case .op(let op):
            result.append(.op(op))

        case .residual(let body):
            result.append(.saveResidual)
            result.append(contentsOf: flattenSteps(body))
            result.append(.addResidual)

        case .parallel(let merge, let branches):
            let flatBranches = branches.map { flattenSteps($0) }
            result.append(.parallel(merge: merge, branches: flatBranches))
        }
    }
    return result
}

// MARK: - Flat Execution Engine

/// Execute flattened steps with a residual stack.
///
/// Eliminates recursive dispatch overhead for single-token decode.
/// The residual stack replaces nested `.residual(body:)` with push/pop markers.
func executeFlatSteps(
    _ steps: [FlatStep], input: MLXArray, state: inout InferenceState
) -> MLXArray {
    var h = input
    var residualStack: [MLXArray] = []

    for step in steps {
        switch step {
        case .op(let op):
            h = executeOp(op, input: h, state: &state)

        case .saveResidual:
            residualStack.append(h)

        case .addResidual:
            h = residualStack.removeLast() + h

        case .parallel(let merge, let branches):
            let results = branches.map { branch in
                var branchState = state
                let result = executeFlatSteps(branch, input: h, state: &branchState)
                state = branchState
                return result
            }
            h = mergeResults(results, strategy: merge)
        }
    }
    return h
}
