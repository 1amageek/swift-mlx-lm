import LMIR

extension RoPEAttributes: MetalKernelFragment {
    public var fragment: Never { fatalError("RoPE is inlined within Attention fragment") }
    public var isFusable: Bool { false }
}

extension PositionalEmbeddingAttributes: MetalKernelFragment {
    public var fragment: Never { fatalError("PositionalEmbedding not yet supported") }
    public var isFusable: Bool { false }
}
