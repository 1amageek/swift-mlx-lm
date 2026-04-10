/// Multi-head attention component.
///
/// Represents the full attention operation as a single semantic unit.
///
/// ```swift
/// Attention(
///     hiddenSize: 4096,
///     headCount: 32,
///     kvHeadCount: 8,
///     headDimension: 128
/// )
/// ```
public struct Attention: ModelComponent {

    public typealias Body = Never

    public let hiddenSize: Int
    public let headCount: Int
    public let kvHeadCount: Int
    public let headDimension: Int
    public let attentionScale: Float?
    public let bias: Bool
    public let causal: Bool
    public let rope: RoPEAttributes?
    public let qkNorm: QKNormKind?
    public let valueNorm: AttentionValueNormKind?
    public let valueProjectionSource: AttentionValueProjectionSource
    public let window: AttentionWindow?
    public let implementationHint: AttentionImplementationHint?
    public let outputGate: AttentionGateKind?
    public let sharedKeyValueSourceLayerIndex: Int?

    public init(
        hiddenSize: Int,
        headCount: Int,
        kvHeadCount: Int,
        headDimension: Int? = nil,
        attentionScale: Float? = nil,
        bias: Bool = false,
        causal: Bool = true,
        rope: RoPEAttributes? = nil,
        qkNorm: QKNormKind? = nil,
        valueNorm: AttentionValueNormKind? = nil,
        valueProjectionSource: AttentionValueProjectionSource = .dedicatedProjection,
        window: AttentionWindow? = nil,
        implementationHint: AttentionImplementationHint? = nil,
        outputGate: AttentionGateKind? = nil,
        sharedKeyValueSourceLayerIndex: Int? = nil
    ) {
        precondition(hiddenSize > 0, "hiddenSize must be positive")
        precondition(headCount > 0, "headCount must be positive")
        precondition(kvHeadCount > 0, "kvHeadCount must be positive")
        precondition(kvHeadCount <= headCount, "kvHeadCount must not exceed headCount")
        if let headDimension {
            precondition(headDimension > 0, "headDimension must be positive")
        } else {
            precondition(hiddenSize % headCount == 0,
                "hiddenSize must be divisible by headCount when headDimension is not specified")
        }
        self.hiddenSize = hiddenSize
        self.headCount = headCount
        self.kvHeadCount = kvHeadCount
        self.headDimension = headDimension ?? (hiddenSize / headCount)
        self.attentionScale = attentionScale
        self.bias = bias
        self.causal = causal
        self.rope = rope
        self.qkNorm = qkNorm
        self.valueNorm = valueNorm
        self.valueProjectionSource = valueProjectionSource
        self.window = window
        self.implementationHint = implementationHint
        self.outputGate = outputGate
        self.sharedKeyValueSourceLayerIndex = sharedKeyValueSourceLayerIndex
    }
}

extension Attention: PrimitiveComponent {

    package var operationKind: OperationKind {
        .primitive(AttentionAttributes(
            hiddenSize: hiddenSize,
            headCount: headCount,
            kvHeadCount: kvHeadCount,
            headDimension: headDimension,
            attentionScale: attentionScale,
            bias: bias,
            causal: causal,
            rope: rope,
            qkNorm: qkNorm,
            valueNorm: valueNorm,
            valueProjectionSource: valueProjectionSource,
            window: window,
            implementationHint: implementationHint,
            outputGate: outputGate,
            sharedKeyValueSourceLayerIndex: sharedKeyValueSourceLayerIndex
        ))
    }

    package var operationSignature: OperationSignature {
        OperationSignature(operandArity: .exact(1), resultArity: .exact(1))
    }
}
