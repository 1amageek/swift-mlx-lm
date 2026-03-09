/// Resolves a `WeightsDeclaration` into concrete `RawWeights`.
///
/// This is the boundary where declaration becomes I/O.
/// The resolver handles file reading, network fetching, or random generation
/// depending on the weight declaration.
public protocol WeightsResolver: Sendable {

    /// Resolve a weight declaration into raw tensor data.
    func resolve(_ declaration: WeightsDeclaration) async throws -> RawWeights
}
