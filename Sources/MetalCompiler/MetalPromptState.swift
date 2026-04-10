import Metal

public struct MetalPromptState: @unchecked Sendable {
    public let position: Int
    public let firstToken: Int32
    let residencyLease: MetalResidencyLease
    let hidden: MTLBuffer
    let residual: MTLBuffer
    let scratch: MTLBuffer
    let logits: MTLBuffer
    let positionBuffer: MTLBuffer
    let ropePositionAxes: MTLBuffer
    let tokenIn: MTLBuffer
    let tokenOut: MTLBuffer
    let kvKeys: MTLBuffer?
    let kvValues: MTLBuffer?
    let convState: MTLBuffer?
    let recurrentState: MTLBuffer?
    let perLayerInputs: MTLBuffer?

    public var logitsBuffer: MTLBuffer { logits }

    init(
        position: Int,
        firstToken: Int32,
        residencyLease: MetalResidencyLease,
        hidden: MTLBuffer,
        residual: MTLBuffer,
        scratch: MTLBuffer,
        logits: MTLBuffer,
        positionBuffer: MTLBuffer,
        ropePositionAxes: MTLBuffer,
        tokenIn: MTLBuffer,
        tokenOut: MTLBuffer,
        kvKeys: MTLBuffer?,
        kvValues: MTLBuffer?,
        convState: MTLBuffer?,
        recurrentState: MTLBuffer?,
        perLayerInputs: MTLBuffer?
    ) {
        self.position = position
        self.firstToken = firstToken
        self.residencyLease = residencyLease
        self.hidden = hidden
        self.residual = residual
        self.scratch = scratch
        self.logits = logits
        self.positionBuffer = positionBuffer
        self.ropePositionAxes = ropePositionAxes
        self.tokenIn = tokenIn
        self.tokenOut = tokenOut
        self.kvKeys = kvKeys
        self.kvValues = kvValues
        self.convState = convState
        self.recurrentState = recurrentState
        self.perLayerInputs = perLayerInputs
    }
}
