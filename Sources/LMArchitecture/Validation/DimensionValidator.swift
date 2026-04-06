/// Validates dimensional consistency of a `ModelGraph`.
///
/// `DimensionValidator` performs two levels of validation:
///
/// 1. **Attribute invariants**: each operation's attributes satisfy internal
///    dimensional constraints (e.g., `headCount * headDimension == hiddenSize`).
///
/// 2. **Hidden dimension propagation**: tracks the "current hidden dimension"
///    through the graph and verifies that each operation's expected input
///    dimension matches what the preceding operation produces.
///
/// These checks catch misconfigurations that `GraphValidator` (structural
/// arity) cannot detect — e.g., a residual body whose norm dimension
/// doesn't match the attention hidden size, or a stateSpace block where
/// `groupCount > numHeads`.
public enum DimensionValidator {

    /// Validate dimensional consistency of a model graph.
    ///
    /// - Throws: `DimensionValidationError` if any dimensional invariant is violated.
    public static func validate(_ graph: ModelGraph) throws {
        _ = try validateRegion(graph.rootRegion, inputDim: nil)
    }
}

// MARK: - Errors

/// Errors from dimensional validation.
public enum DimensionValidationError: Error, Sendable, CustomStringConvertible {

    /// A positive integer was expected but got zero or negative.
    case nonPositiveDimension(field: String, value: Int, operationKey: OperationKey?)

    /// An internal attribute invariant is violated.
    case attributeInvariant(message: String, operationKey: OperationKey?)

    /// An operation's expected input dimension doesn't match the current hidden dimension.
    case dimensionMismatch(
        expected: Int,
        actual: Int,
        field: String,
        operationKey: OperationKey?
    )

    /// A structural operation's body produces a different dimension than required.
    case bodyDimensionMismatch(
        expected: Int,
        actual: Int,
        context: String,
        operationKey: OperationKey?
    )

    public var description: String {
        switch self {
        case .nonPositiveDimension(let field, let value, let key):
            return "Non-positive dimension: \(field) = \(value) (operation \(keyDesc(key)))"
        case .attributeInvariant(let message, let key):
            return "Attribute invariant: \(message) (operation \(keyDesc(key)))"
        case .dimensionMismatch(let expected, let actual, let field, let key):
            return "Dimension mismatch: \(field) expected \(expected) but got \(actual) (operation \(keyDesc(key)))"
        case .bodyDimensionMismatch(let expected, let actual, let context, let key):
            return "Body dimension mismatch: \(context) expected \(expected) but got \(actual) (operation \(keyDesc(key)))"
        }
    }
}

private func keyDesc(_ key: OperationKey?) -> String {
    key.map { "#\($0.rawValue)" } ?? "root"
}

// MARK: - Attribute Invariant Validation

private func validateAttentionInvariants(
    _ attrs: AttentionAttributes,
    key: OperationKey
) throws {
    try requirePositive("attention.hiddenSize", attrs.hiddenSize, key: key)
    try requirePositive("attention.headCount", attrs.headCount, key: key)
    try requirePositive("attention.kvHeadCount", attrs.kvHeadCount, key: key)
    try requirePositive("attention.headDimension", attrs.headDimension, key: key)

    // Output projection is [headCount * headDimension] → [hiddenSize] matmul.
    // Some architectures (e.g. Qwen3.5-4B: headCount=16, headDim=256, hiddenSize=2560)
    // use a rectangular output projection, so headCount * headDimension != hiddenSize is valid.

    // GQA: kvHeadCount must divide headCount evenly
    if attrs.kvHeadCount > attrs.headCount {
        throw DimensionValidationError.attributeInvariant(
            message: "kvHeadCount(\(attrs.kvHeadCount)) > headCount(\(attrs.headCount))",
            operationKey: key
        )
    }
    if attrs.headCount % attrs.kvHeadCount != 0 {
        throw DimensionValidationError.attributeInvariant(
            message: "headCount(\(attrs.headCount)) not divisible by kvHeadCount(\(attrs.kvHeadCount))",
            operationKey: key
        )
    }

    // RoPE dimension constraints
    if let rope = attrs.rope {
        try validateRoPEInvariants(rope, headDimension: attrs.headDimension, key: key)
    }
}

private func validateStateSpaceInvariants(
    _ attrs: StateSpaceAttributes,
    key: OperationKey
) throws {
    try requirePositive("stateSpace.hiddenSize", attrs.hiddenSize, key: key)
    try requirePositive("stateSpace.numHeads", attrs.numHeads, key: key)
    try requirePositive("stateSpace.groupCount", attrs.groupCount, key: key)
    try requirePositive("stateSpace.keyHeadDim", attrs.keyHeadDim, key: key)
    try requirePositive("stateSpace.valueHeadDim", attrs.valueHeadDim, key: key)

    // DeltaNet: output projection is numHeads * valueHeadDim → hiddenSize (matmul).
    // Asymmetric variants (e.g. Qwen3.5-4B: numHeads=32, valueHeadDim=128, hiddenSize=2560)
    // use a rectangular output projection, so numHeads * valueHeadDim != hiddenSize is valid.
    // No invariant between numHeads * valueHeadDim and hiddenSize is enforced.

    // DeltaNet: groupCount is the key/query head count, expanded to match numHeads.
    // groupCount must divide numHeads evenly for clean expansion.
    let isDeltaNet = attrs.variant.contains("deltanet") || attrs.variant.contains("delta_net")
    if isDeltaNet {
        if attrs.groupCount > attrs.numHeads {
            throw DimensionValidationError.attributeInvariant(
                message: "groupCount(\(attrs.groupCount)) > numHeads(\(attrs.numHeads))",
                operationKey: key
            )
        }
        if attrs.numHeads % attrs.groupCount != 0 {
            throw DimensionValidationError.attributeInvariant(
                message: "numHeads(\(attrs.numHeads)) not divisible by groupCount(\(attrs.groupCount))",
                operationKey: key
            )
        }
    }
}

private func validateShortConvInvariants(
    _ attrs: ShortConvAttributes,
    key: OperationKey
) throws {
    try requirePositive("shortConv.hiddenSize", attrs.hiddenSize, key: key)
    try requirePositive("shortConv.kernelSize", attrs.kernelSize, key: key)
}

private func validateMLPInvariants(
    _ attrs: MLPAttributes,
    key: OperationKey
) throws {
    try requirePositive("mlp.inputSize", attrs.inputSize, key: key)
    try requirePositive("mlp.outputSize", attrs.outputSize, key: key)
    try requirePositive("mlp.intermediateSize", attrs.intermediateSize, key: key)
}

private func validateMoEInvariants(
    _ attrs: MoEAttributes,
    key: OperationKey
) throws {
    try requirePositive("moe.expertCount", attrs.expertCount, key: key)
    try requirePositive("moe.expertsPerToken", attrs.expertsPerToken, key: key)

    if attrs.expertsPerToken > attrs.expertCount {
        throw DimensionValidationError.attributeInvariant(
            message: "expertsPerToken(\(attrs.expertsPerToken)) > expertCount(\(attrs.expertCount))",
            operationKey: key
        )
    }

    try validateMLPInvariants(attrs.expertMLP, key: key)
}

private func validateRoPEInvariants(
    _ attrs: RoPEAttributes,
    headDimension: Int?,
    key: OperationKey
) throws {
    try requirePositive("rope.dimension", attrs.dimension, key: key)

    // RoPE operates on pairs of dimensions
    if attrs.dimension % 2 != 0 {
        throw DimensionValidationError.attributeInvariant(
            message: "rope.dimension(\(attrs.dimension)) must be even (rotation applied to pairs)",
            operationKey: key
        )
    }

    // RoPE dimension must not exceed head dimension (partial RoPE allowed)
    if let headDim = headDimension, attrs.dimension > headDim {
        throw DimensionValidationError.attributeInvariant(
            message: "rope.dimension(\(attrs.dimension)) > headDimension(\(headDim))",
            operationKey: key
        )
    }

    if attrs.base <= 0 {
        throw DimensionValidationError.attributeInvariant(
            message: "rope.base(\(attrs.base)) must be positive",
            operationKey: key
        )
    }
}

private func validateNormInvariants(
    dimension: Int,
    epsilon: Float,
    label: String,
    key: OperationKey
) throws {
    try requirePositive("\(label).dimension", dimension, key: key)
    if epsilon <= 0 {
        throw DimensionValidationError.attributeInvariant(
            message: "\(label).epsilon(\(epsilon)) must be positive",
            operationKey: key
        )
    }
}

private func requirePositive(
    _ field: String,
    _ value: Int,
    key: OperationKey
) throws {
    if value <= 0 {
        throw DimensionValidationError.nonPositiveDimension(
            field: field, value: value, operationKey: key
        )
    }
}

// MARK: - Hidden Dimension Propagation

/// Validate a region and return the output hidden dimension.
///
/// - Parameter inputDim: The hidden dimension flowing into this region
///   (nil for root region, which starts from a source operation).
/// - Returns: The hidden dimension produced by this region.
@discardableResult
private func validateRegion(
    _ region: Region,
    inputDim: Int?
) throws -> Int? {
    var currentDim = inputDim

    for op in region.operations {
        currentDim = try validateOperation(op, currentDim: currentDim)
    }

    return currentDim
}

/// Validate a single operation and return the output hidden dimension.
private func validateOperation(
    _ op: Operation,
    currentDim: Int?
) throws -> Int? {
    switch op.kind {

    // MARK: Primitive operations

    case .primitive(let attrs):
        return try validatePrimitiveAttributes(attrs, currentDim: currentDim, key: op.key)

    // MARK: Structural operations

    case .residual(let strategy, let body):
        let bodyOutputDim = try validateRegion(body, inputDim: currentDim)
        // For `.add` strategy, body output must match input dimension
        if case .add = strategy, let inDim = currentDim, let outDim = bodyOutputDim {
            if inDim != outDim {
                throw DimensionValidationError.bodyDimensionMismatch(
                    expected: inDim,
                    actual: outDim,
                    context: "residual(.add) body output",
                    operationKey: op.key
                )
            }
        }
        return currentDim

    case .parallel(let merge, let branches):
        var branchDims: [Int] = []
        for branch in branches {
            if let dim = try validateRegion(branch, inputDim: currentDim) {
                branchDims.append(dim)
            }
        }

        switch merge {
        case .add:
            // All branches must produce the same dimension as input
            if let inDim = currentDim {
                for (i, dim) in branchDims.enumerated() {
                    if dim != inDim {
                        throw DimensionValidationError.bodyDimensionMismatch(
                            expected: inDim,
                            actual: dim,
                            context: "parallel(.add) branch \(i)",
                            operationKey: op.key
                        )
                    }
                }
            }
            return currentDim

        case .concat:
            // Output is concatenation of all branch dimensions
            return branchDims.isEmpty ? currentDim : branchDims.reduce(0, +)

        case .stack:
            // Stack adds a new axis; hidden dimension is preserved
            return currentDim

        case .custom:
            // No dimensional constraint for custom merge
            return currentDim
        }

    case .repeating(_, let body):
        let bodyOutputDim = try validateRegion(body, inputDim: currentDim)
        // Loop-carried: body output must match input (feeds back as next iteration input)
        if let inDim = currentDim, let outDim = bodyOutputDim {
            if inDim != outDim {
                throw DimensionValidationError.bodyDimensionMismatch(
                    expected: inDim,
                    actual: outDim,
                    context: "repeating body output",
                    operationKey: op.key
                )
            }
        }
        return currentDim

    case .conditional(_, let thenRegion, let elseRegion):
        let thenDim = try validateRegion(thenRegion, inputDim: currentDim)
        let elseDim = try validateRegion(elseRegion, inputDim: currentDim)
        // Both branches must produce the same output dimension
        if let td = thenDim, let ed = elseDim, td != ed {
            throw DimensionValidationError.bodyDimensionMismatch(
                expected: td,
                actual: ed,
                context: "conditional then/else dimension mismatch",
                operationKey: op.key
            )
        }
        return thenDim ?? elseDim ?? currentDim
    }
}

/// Validate primitive attributes and return the output hidden dimension.
private func validatePrimitiveAttributes(
    _ attrs: any OperationAttributes,
    currentDim: Int?,
    key: OperationKey
) throws -> Int? {
    switch attrs {
    // Source operations (produce initial dimension)
    case let a as TokenEmbeddingAttributes:
        try requirePositive("tokenEmbedding.vocabSize", a.vocabSize, key: key)
        try requirePositive("tokenEmbedding.embeddingSize", a.embeddingSize, key: key)
        return a.embeddingSize

    // Dimension-preserving primitives
    case let a as AttentionAttributes:
        try validateAttentionInvariants(a, key: key)
        if let dim = currentDim {
            try checkDimensionMatch(expected: dim, actual: a.hiddenSize, field: "attention.hiddenSize", key: key)
        }
        return a.hiddenSize

    case let a as StateSpaceAttributes:
        try validateStateSpaceInvariants(a, key: key)
        if let dim = currentDim {
            try checkDimensionMatch(expected: dim, actual: a.hiddenSize, field: "stateSpace.hiddenSize", key: key)
        }
        return a.hiddenSize

    case let a as ShortConvAttributes:
        try validateShortConvInvariants(a, key: key)
        if let dim = currentDim {
            try checkDimensionMatch(expected: dim, actual: a.hiddenSize, field: "shortConv.hiddenSize", key: key)
        }
        return a.hiddenSize

    case let a as RMSNormAttributes:
        try validateNormInvariants(dimension: a.dimension, epsilon: a.epsilon, label: "rmsNorm", key: key)
        if let dim = currentDim {
            try checkDimensionMatch(expected: dim, actual: a.dimension, field: "rmsNorm.dimension", key: key)
        }
        return a.dimension

    case let a as LayerNormAttributes:
        try validateNormInvariants(dimension: a.dimension, epsilon: a.epsilon, label: "layerNorm", key: key)
        if let dim = currentDim {
            try checkDimensionMatch(expected: dim, actual: a.dimension, field: "layerNorm.dimension", key: key)
        }
        return a.dimension

    // Dimension-transforming primitives
    case let a as MLPAttributes:
        try validateMLPInvariants(a, key: key)
        if let dim = currentDim {
            try checkDimensionMatch(expected: dim, actual: a.inputSize, field: "mlp.inputSize", key: key)
        }
        return a.outputSize

    case let a as MoEAttributes:
        try validateMoEInvariants(a, key: key)
        if let dim = currentDim {
            try checkDimensionMatch(expected: dim, actual: a.expertMLP.inputSize, field: "moe.expertMLP.inputSize", key: key)
        }
        return a.expertMLP.outputSize

    case let a as LinearAttributes:
        try requirePositive("linear.inputSize", a.inputSize, key: key)
        try requirePositive("linear.outputSize", a.outputSize, key: key)
        if let dim = currentDim {
            try checkDimensionMatch(expected: dim, actual: a.inputSize, field: "linear.inputSize", key: key)
        }
        return a.outputSize

    case let a as PerLayerInputAttributes:
        try requirePositive("perLayerInput.hiddenSize", a.hiddenSize, key: key)
        try requirePositive("perLayerInput.perLayerInputSize", a.perLayerInputSize, key: key)
        try requirePositive("perLayerInput.vocabSize", a.vocabSize, key: key)
        if let dim = currentDim {
            try checkDimensionMatch(
                expected: dim,
                actual: a.hiddenSize,
                field: "perLayerInput.hiddenSize",
                key: key
            )
        }
        return a.hiddenSize

    case let a as OutputHeadAttributes:
        try requirePositive("outputHead.inputSize", a.inputSize, key: key)
        try requirePositive("outputHead.vocabSize", a.vocabSize, key: key)
        if let dim = currentDim {
            try checkDimensionMatch(expected: dim, actual: a.inputSize, field: "outputHead.inputSize", key: key)
        }
        return a.vocabSize

    // Pass-through primitives
    case let a as RoPEAttributes:
        try validateRoPEInvariants(a, headDimension: nil, key: key)
        return currentDim

    case is PositionalEmbeddingAttributes:
        return currentDim

    case is CustomNodeAttributes:
        // Cannot validate custom operations dimensionally
        return currentDim

    default:
        return currentDim
    }
}

private func checkDimensionMatch(
    expected: Int,
    actual: Int,
    field: String,
    key: OperationKey
) throws {
    if expected != actual {
        throw DimensionValidationError.dimensionMismatch(
            expected: expected,
            actual: actual,
            field: field,
            operationKey: key
        )
    }
}
