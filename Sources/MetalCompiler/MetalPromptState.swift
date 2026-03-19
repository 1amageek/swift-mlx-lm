import Metal

public struct MetalPromptState: @unchecked Sendable {
    public let position: Int
    public let firstToken: Int32
    let kvKeys: MTLBuffer?
    let kvValues: MTLBuffer?
    let convState: MTLBuffer?

    init(
        position: Int,
        firstToken: Int32,
        kvKeys: MTLBuffer?,
        kvValues: MTLBuffer?,
        convState: MTLBuffer?
    ) {
        self.position = position
        self.firstToken = firstToken
        self.kvKeys = kvKeys
        self.kvValues = kvValues
        self.convState = convState
    }
}
