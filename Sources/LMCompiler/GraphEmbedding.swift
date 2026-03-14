@preconcurrency import MLX
import MLXFast
import MLXNN
import SwiftLM

// MARK: - GraphEmbedding

/// Token embedding module compiled from ModelGraph.
///
/// Uses MLXNN `Embedding` which supports quantization via `QuantizedEmbedding`.
/// When quantized, `asLinear()` uses `quantizedMM` for tied output heads.
final class GraphEmbedding: Module, UnaryLayer {

    @ModuleInfo(key: "embedding") var embedding: Embedding

    init(attrs: TokenEmbeddingAttributes, store: MLXWeightStore, path: StructuralPath) throws {
        let table = try store.require(ParameterSlot(path: path, role: .embeddingTable))
        self._embedding.wrappedValue = Embedding(weight: table)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        embedding(x)
    }
}

// MARK: - GraphOutputHead

/// Output projection (language model head) compiled from ModelGraph.
///
/// For tied output heads, creates a `Linear` initialized with the embedding weight.
/// Both tied and untied variants are quantizable through `@ModuleInfo`.
///
/// When the embedding is a `QuantizedEmbedding`, the tied output head uses
/// `embedding.asLinear()` which calls `quantizedMM` internally.
final class GraphOutputHead: Module, UnaryLayer {

    @ModuleInfo(key: "projection") var projection: MLXNN.Linear

    let isTied: Bool

    /// Reference to the embedding module for tied output (uses `asLinear()`).
    /// Stored as unowned to avoid retain cycle.
    private weak var tiedEmbeddingModule: GraphEmbedding?

    /// Initialize with tied embedding weight.
    init(tiedTo embedding: GraphEmbedding) {
        self.isTied = true
        self.tiedEmbeddingModule = embedding
        // Create a Linear with the embedding weight for quantization discovery
        self._projection.wrappedValue = MLXNN.Linear(weight: embedding.embedding.weight)
    }

    /// Initialize with own output projection weight.
    init(attrs: OutputHeadAttributes, store: MLXWeightStore, path: StructuralPath) throws {
        self.isTied = false
        self.tiedEmbeddingModule = nil
        self._projection.wrappedValue = MLXNN.Linear(
            weight: try store.require(ParameterSlot(path: path, role: .outputProjection)),
            bias: store.get(ParameterSlot(path: path, role: .bias))
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if isTied, let emb = tiedEmbeddingModule {
            // Use embedding's asLinear() which handles quantized path automatically
            return emb.embedding.asLinear(x)
        }
        return projection(x)
    }
}

// MARK: - GraphPositionalEmbedding

/// Positional embedding module compiled from ModelGraph.
///
/// Adds learned position embeddings to the input tensor.
final class GraphPositionalEmbedding: Module, UnaryLayer {

    let table: MLXArray

    init(attrs: PositionalEmbeddingAttributes, store: MLXWeightStore, path: StructuralPath) throws {
        self.table = try store.require(ParameterSlot(path: path, role: .embeddingTable))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let seqLen = x.dim(1)
        let positions = MLXArray(0..<seqLen)
        return x + table[positions]
    }
}

// MARK: - GraphLinear

/// Standalone linear projection compiled from ModelGraph.
///
/// Uses `@ModuleInfo` for quantization support.
final class GraphLinear: Module, UnaryLayer {

    @ModuleInfo(key: "projection") var projection: MLXNN.Linear

    init(attrs: LinearAttributes, store: MLXWeightStore, path: StructuralPath) throws {
        self._projection.wrappedValue = MLXNN.Linear(
            weight: try store.require(ParameterSlot(path: path, role: .weight)),
            bias: attrs.bias ? store.get(ParameterSlot(path: path, role: .bias)) : nil
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        projection(x)
    }
}

// MARK: - GraphRoPE

/// Standalone RoPE module compiled from ModelGraph.
///
/// Uses offset=0 (no cache association). In practice, RoPE is usually
/// embedded within attention operations.
final class GraphRoPE: Module, UnaryLayer {

    let attrs: RoPEAttributes

    init(attrs: RoPEAttributes) {
        self.attrs = attrs
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let scale: Float
        switch attrs.scaling?.kind {
        case .linear:
            scale = 1.0 / attrs.scaling!.factor
        default:
            scale = 1.0
        }
        return MLXFast.RoPE(
            x, dimensions: attrs.dimension, traditional: false,
            base: attrs.base, scale: scale, offset: 0
        )
    }
}
