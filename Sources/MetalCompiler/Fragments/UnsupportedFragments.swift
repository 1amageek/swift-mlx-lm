import LMIR

extension RoPEAttributes: MetalKernelFragment {
    public func fragment(context: KernelContext) -> Never { fatalError("RoPE is inlined within Attention fragment") }
    public var isFusable: Bool { false }
}

extension PositionalEmbeddingAttributes: MetalKernelFragment {
    public func fragment(context: KernelContext) -> Never { fatalError("PositionalEmbedding not yet supported") }
    public var isFusable: Bool { false }
}
