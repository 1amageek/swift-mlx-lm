/// Manages special token IDs and auto-insertion behavior.
struct SpecialTokens: Sendable {
    let bosTokenID: Int?
    let eosTokenID: Int?
    let unknownTokenID: Int?
    let paddingTokenID: Int?
    let addBosToken: Bool
    let addEosToken: Bool

    init(
        bosTokenID: Int? = nil,
        eosTokenID: Int? = nil,
        unknownTokenID: Int? = nil,
        paddingTokenID: Int? = nil,
        addBosToken: Bool = false,
        addEosToken: Bool = false
    ) {
        self.bosTokenID = bosTokenID
        self.eosTokenID = eosTokenID
        self.unknownTokenID = unknownTokenID
        self.paddingTokenID = paddingTokenID
        self.addBosToken = addBosToken
        self.addEosToken = addEosToken
    }
}
