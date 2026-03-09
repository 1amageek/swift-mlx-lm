/// Raw parameter tensors as loaded from a checkpoint.
///
/// Tensor names follow the checkpoint's naming convention (e.g., GGUF tensor names).
/// These have not yet been bound to semantic parameter slots in the model graph.
public struct RawWeights: Sendable {

    /// Tensors keyed by their checkpoint-native names.
    public let tensors: [String: TensorData]

    public init(tensors: [String: TensorData]) {
        self.tensors = tensors
    }
}

/// Opaque tensor data placeholder.
///
/// The actual tensor representation depends on the runtime backend (MLX, etc.).
/// SwiftLM core defines only the schema; concrete implementations provide
/// backend-specific tensor types.
public struct TensorData: Sendable {

    /// Shape of the tensor.
    public let shape: [Int]

    /// Data type of the tensor elements.
    public let dtype: DTypeHint

    /// Opaque storage for the raw data.
    ///
    /// The concrete type depends on the backend.
    /// Typed access is provided by backend-specific extensions.
    public let storage: any Sendable

    public init(shape: [Int], dtype: DTypeHint, storage: any Sendable) {
        self.shape = shape
        self.dtype = dtype
        self.storage = storage
    }
}
