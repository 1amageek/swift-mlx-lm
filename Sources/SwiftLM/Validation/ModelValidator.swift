/// Validates model graph structure and weight compatibility.
///
/// Validation occurs before compilation. The validator checks:
/// - Structural consistency (node attributes, references, topology)
/// - Weight completeness (required tensors, shape/dtype compatibility)
/// - Cross-validation (structure ↔ weights consistency)
public protocol ModelValidator: Sendable {

    /// Validate a model graph, optionally against raw weights.
    func validate(graph: ModelGraph, rawWeights: RawWeights?) throws

    /// Validate a model graph against bound weights.
    func validate(graph: ModelGraph, boundWeights: BoundWeights?) throws
}
