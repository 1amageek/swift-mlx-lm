/// Execution backend identifier.
///
/// Phase 1 uses `.metal` exclusively. Phase 2+ introduces `.ane` for
/// linear/FFN operations and `.cpu` for sampling/tokenization.
enum ComputeBackend: Sendable {
    case metal
    case ane
    case cpu
}
