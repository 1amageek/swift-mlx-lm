import Testing
import Foundation
@testable import GGUFTokenizer
@testable import GGUFParser

// MARK: - ByteEncoder Tests

@Suite("ByteEncoder")
struct ByteEncoderTests {

    @Test("All 256 bytes are mapped")
    func allBytesMapped() {
        #expect(ByteEncoder.byteToUnicode.count == 256)
    }

    @Test("All mappings are unique")
    func allMappingsUnique() {
        let scalars = Set(ByteEncoder.byteToUnicode.values)
        #expect(scalars.count == 256)
    }

    @Test("Reverse mapping has same size")
    func reverseMappingSize() {
        #expect(ByteEncoder.unicodeToByte.count == 256)
    }

    @Test("Printable ASCII maps to itself")
    func printableASCII() {
        // 'A' = 0x41 -> U+0041
        #expect(ByteEncoder.byteToUnicode[0x41] == Unicode.Scalar(0x41))
        // '!' = 0x21 -> U+0021
        #expect(ByteEncoder.byteToUnicode[0x21] == Unicode.Scalar(0x21))
        // '~' = 0x7E -> U+007E
        #expect(ByteEncoder.byteToUnicode[0x7E] == Unicode.Scalar(0x7E))
    }

    @Test("Non-printable bytes map to U+0100 range")
    func nonPrintableMapping() {
        // byte 0x00 -> first non-printable -> U+0100
        #expect(ByteEncoder.byteToUnicode[0x00] == Unicode.Scalar(0x0100))
        // byte 0x01 -> U+0101
        #expect(ByteEncoder.byteToUnicode[0x01] == Unicode.Scalar(0x0101))
        // byte 0x20 (space) -> maps to U+0100 range
        let spaceScalar = ByteEncoder.byteToUnicode[0x20]!
        #expect(spaceScalar.value >= 0x0100)
    }

    @Test("Latin-1 Supplement maps to itself")
    func latin1Supplement() {
        // 0xA1 '¡' -> U+00A1
        #expect(ByteEncoder.byteToUnicode[0xA1] == Unicode.Scalar(0xA1))
        // 0xAE '®' -> U+00AE
        #expect(ByteEncoder.byteToUnicode[0xAE] == Unicode.Scalar(0xAE))
        // 0xFF 'ÿ' -> U+00FF
        #expect(ByteEncoder.byteToUnicode[0xFF] == Unicode.Scalar(0xFF))
    }

    @Test("0xAD (soft hyphen) maps to U+0100 range")
    func softHyphenMapping() {
        let scalar = ByteEncoder.byteToUnicode[0xAD]!
        #expect(scalar.value >= 0x0100)
    }

    @Test("Roundtrip encode-decode")
    func roundtrip() {
        let original: [UInt8] = [0x00, 0x41, 0x7F, 0xAD, 0xFF, 0x20]
        let encoded = ByteEncoder.encode(original)
        let decoded = ByteEncoder.decode(encoded)
        #expect(decoded == original)
    }

    @Test("ASCII text roundtrip")
    func asciiRoundtrip() {
        let text = "Hello, World!"
        let bytes = Array(text.utf8)
        let encoded = ByteEncoder.encode(bytes)
        let decoded = ByteEncoder.decode(encoded)
        #expect(String(decoding: decoded, as: UTF8.self) == text)
    }

    @Test("UTF-8 multibyte roundtrip")
    func utf8MultbyteRoundtrip() {
        let text = "日本語"
        let bytes = Array(text.utf8)
        let encoded = ByteEncoder.encode(bytes)
        let decoded = ByteEncoder.decode(encoded)
        #expect(String(decoding: decoded, as: UTF8.self) == text)
    }

    @Test("Exactly 68 non-printable bytes mapped to U+0100 range")
    func nonPrintableCount() {
        let highMappings = ByteEncoder.byteToUnicode.values.filter { $0.value >= 0x0100 }
        #expect(highMappings.count == 68)
    }
}

// MARK: - PreTokenizer Tests

@Suite("PreTokenizer")
struct PreTokenizerTests {

    @Test("Empty text returns empty array")
    func emptyText() {
        let result = PreTokenizer.gpt2.split(text: "")
        #expect(result.isEmpty)
    }

    @Test("GPT-2 pattern splits words")
    func gpt2Words() {
        let result = PreTokenizer.gpt2.split(text: "Hello World")
        #expect(result == ["Hello", " World"])
    }

    @Test("GPT-2 pattern handles contractions")
    func gpt2Contractions() {
        let result = PreTokenizer.gpt2.split(text: "I'm don't")
        #expect(result.contains("'m"))
        #expect(result.contains("'t"))
    }

    @Test("GPT-2 pattern handles numbers")
    func gpt2Numbers() {
        let result = PreTokenizer.gpt2.split(text: "test 123 abc")
        #expect(result.contains(" 123"))
    }

    @Test("GPT-2 pattern handles punctuation")
    func gpt2Punctuation() {
        let result = PreTokenizer.gpt2.split(text: "a.b")
        #expect(result.count >= 2)
    }

    @Test("Llama BPE case-insensitive contractions")
    func llamaContractions() {
        let result = PreTokenizer.llamaBPE.split(text: "I'M DON'T")
        #expect(result.contains("'M"))
        #expect(result.contains("'T"))
    }

    @Test("Llama BPE groups numbers up to 3 digits")
    func llamaNumberGrouping() {
        let result = PreTokenizer.llamaBPE.split(text: "12345")
        #expect(result == ["123", "45"])
    }

    @Test("Qwen2 splits numbers per-digit")
    func qwen2NumberSplit() {
        let result = PreTokenizer.qwen2.split(text: "123")
        #expect(result == ["1", "2", "3"])
    }

    @Test("forType selects correct pre-tokenizer")
    func forTypeSelection() {
        let llama = PreTokenizer.forType("llama-bpe")
        let qwen = PreTokenizer.forType("qwen2")
        let gpt2 = PreTokenizer.forType("gpt2")
        let defaultPre = PreTokenizer.forType(nil)

        // Verify they produce different results for numbers
        let llamaResult = llama.split(text: "12345")
        let qwenResult = qwen.split(text: "12345")
        #expect(llamaResult == ["123", "45"])
        #expect(qwenResult == ["1", "2", "3", "4", "5"])

        // Default falls back to GPT-2
        let gpt2Result = gpt2.split(text: "Hello World")
        let defaultResult = defaultPre.split(text: "Hello World")
        #expect(gpt2Result == defaultResult)
    }

    @Test("Whitespace handling")
    func whitespace() {
        let result = PreTokenizer.gpt2.split(text: "a  b")
        // Double space should be handled
        #expect(!result.isEmpty)
        let joined = result.joined()
        #expect(joined == "a  b")
    }

    @Test("Newlines in Llama BPE")
    func llamaNewlines() {
        let result = PreTokenizer.llamaBPE.split(text: "hello\nworld")
        let joined = result.joined()
        #expect(joined == "hello\nworld")
    }
}

// MARK: - MergesBPETokenizer Tests

@Suite("MergesBPETokenizer")
struct MergesBPETokenizerTests {

    // Build a small test vocabulary and merges for testing
    static func makeTestTokenizer() throws -> MergesBPETokenizer {
        // Vocabulary: individual byte-encoded characters + some merged tokens
        // For ASCII: byte 0x48='H' -> Unicode 'H', etc.
        var vocab: [String] = []

        // First, build all single byte-encoded characters (256 entries)
        for byte: UInt8 in 0...255 {
            vocab.append(String(Character(ByteEncoder.byteToUnicode[byte]!)))
        }

        // Add merged tokens
        // "He" as byte-encoded
        let hChar = String(Character(ByteEncoder.byteToUnicode[0x48]!)) // H
        let eChar = String(Character(ByteEncoder.byteToUnicode[0x65]!)) // e
        let lChar = String(Character(ByteEncoder.byteToUnicode[0x6C]!)) // l
        let oChar = String(Character(ByteEncoder.byteToUnicode[0x6F]!)) // o

        vocab.append(hChar + eChar)            // 256: "He"
        vocab.append(lChar + lChar)            // 257: "ll"
        vocab.append(hChar + eChar + lChar + lChar) // 258: "Hell"
        vocab.append(oChar)                    // already at index for 'o'

        let merges = [
            "\(hChar) \(eChar)",                                // rank 0: H + e -> He
            "\(lChar) \(lChar)",                                // rank 1: l + l -> ll
            "\(hChar + eChar) \(lChar + lChar)",                // rank 2: He + ll -> Hell
        ]

        return try MergesBPETokenizer(
            vocabulary: vocab,
            merges: merges,
            preTokenizer: .gpt2,
            specialTokens: SpecialTokens(
                bosTokenID: nil,
                eosTokenID: nil,
                addBosToken: false,
                addEosToken: false
            )
        )
    }

    @Test("BPE merges basic pair")
    func bpeMergesBasic() throws {
        let tokenizer = try MergesBPETokenizerTests.makeTestTokenizer()

        let hChar = String(Character(ByteEncoder.byteToUnicode[0x48]!))
        let eChar = String(Character(ByteEncoder.byteToUnicode[0x65]!))

        let result = tokenizer.bpe(token: hChar + eChar)
        #expect(result == [hChar + eChar])
    }

    @Test("BPE merges chain")
    func bpeMergesChain() throws {
        let tokenizer = try MergesBPETokenizerTests.makeTestTokenizer()

        let hChar = String(Character(ByteEncoder.byteToUnicode[0x48]!))
        let eChar = String(Character(ByteEncoder.byteToUnicode[0x65]!))
        let lChar = String(Character(ByteEncoder.byteToUnicode[0x6C]!))

        // "Hell" = H + e + l + l
        // Step 1: H + e -> He, l + l -> ll
        // Step 2: He + ll -> Hell
        let result = tokenizer.bpe(token: hChar + eChar + lChar + lChar)
        #expect(result == [hChar + eChar + lChar + lChar])
    }

    @Test("BPE single character")
    func bpeSingleChar() throws {
        let tokenizer = try MergesBPETokenizerTests.makeTestTokenizer()
        let result = tokenizer.bpe(token: "A")
        #expect(result == ["A"])
    }

    @Test("BPE empty string")
    func bpeEmpty() throws {
        let tokenizer = try MergesBPETokenizerTests.makeTestTokenizer()
        let result = tokenizer.bpe(token: "")
        #expect(result.isEmpty)
    }

    @Test("BPE no matching merges")
    func bpeNoMerges() throws {
        let tokenizer = try MergesBPETokenizerTests.makeTestTokenizer()
        let xChar = String(Character(ByteEncoder.byteToUnicode[0x78]!)) // x
        let yChar = String(Character(ByteEncoder.byteToUnicode[0x79]!)) // y
        let result = tokenizer.bpe(token: xChar + yChar)
        #expect(result == [xChar, yChar])
    }

    @Test("Encode and decode roundtrip for simple ASCII")
    func encodeDecodeRoundtrip() throws {
        let tokenizer = try MergesBPETokenizerTests.makeTestTokenizer()
        let text = "ab"
        let tokens = tokenizer.encode(text: text)
        let decoded = tokenizer.decode(tokens: tokens)
        #expect(decoded == text)
    }

    @Test("Vocabulary size")
    func vocabularySize() throws {
        let tokenizer = try MergesBPETokenizerTests.makeTestTokenizer()
        #expect(tokenizer.vocabularySize >= 256)
    }

    @Test("Invalid merge format throws")
    func invalidMergeFormat() {
        #expect(throws: TokenizerError.self) {
            _ = try MergesBPETokenizer(
                vocabulary: ["a", "b"],
                merges: ["invalid_no_space"],
                preTokenizer: .gpt2,
                specialTokens: SpecialTokens()
            )
        }
    }

    @Test("BOS/EOS tokens are added")
    func bosEosTokens() throws {
        var vocab: [String] = []
        for byte: UInt8 in 0...255 {
            vocab.append(String(Character(ByteEncoder.byteToUnicode[byte]!)))
        }
        vocab.append("<s>")    // 256
        vocab.append("</s>")   // 257

        let tokenizer = try MergesBPETokenizer(
            vocabulary: vocab,
            merges: [],
            preTokenizer: .gpt2,
            specialTokens: SpecialTokens(
                bosTokenID: 256,
                eosTokenID: 257,
                addBosToken: true,
                addEosToken: true
            )
        )

        let tokens = tokenizer.encode(text: "a")
        #expect(tokens.first == 256)
        #expect(tokens.last == 257)
    }

    @Test("tokenToString returns correct string")
    func tokenToStringTest() throws {
        let tokenizer = try MergesBPETokenizerTests.makeTestTokenizer()
        let aChar = String(Character(ByteEncoder.byteToUnicode[0x61]!))
        #expect(tokenizer.tokenToString(0x61) == aChar)
        #expect(tokenizer.tokenToString(-1) == nil)
        #expect(tokenizer.tokenToString(999999) == nil)
    }

    @Test("Merges all occurrences of best pair")
    func mergesAllOccurrences() throws {
        // Create a tokenizer where the same pair appears multiple times
        var vocab: [String] = []
        for byte: UInt8 in 0...255 {
            vocab.append(String(Character(ByteEncoder.byteToUnicode[byte]!)))
        }
        let aChar = String(Character(ByteEncoder.byteToUnicode[0x61]!))
        let bChar = String(Character(ByteEncoder.byteToUnicode[0x62]!))
        vocab.append(aChar + bChar) // 256: "ab"

        let merges = ["\(aChar) \(bChar)"]

        let tokenizer = try MergesBPETokenizer(
            vocabulary: vocab,
            merges: merges,
            preTokenizer: .gpt2,
            specialTokens: SpecialTokens()
        )

        // "abab" should merge both occurrences of "ab"
        let result = tokenizer.bpe(token: aChar + bChar + aChar + bChar)
        #expect(result == [aChar + bChar, aChar + bChar])
    }
}

// MARK: - SentencePieceBPETokenizer Tests

@Suite("SentencePieceBPETokenizer")
struct SentencePieceBPETokenizerTests {

    static func makeTestTokenizer() throws -> SentencePieceBPETokenizer {
        // Build a small SentencePiece vocabulary
        var vocab: [String] = []
        var scores: [Float] = []
        var types: [Int] = []

        // Token 0: <unk>
        vocab.append("<unk>")
        scores.append(-1000.0)
        types.append(2)

        // Token 1: <s>
        vocab.append("<s>")
        scores.append(0.0)
        types.append(3)

        // Token 2: </s>
        vocab.append("</s>")
        scores.append(0.0)
        types.append(3)

        // Byte tokens (3-258): <0x00> through <0xFF>
        for byte in 0...255 {
            vocab.append(String(format: "<0x%02X>", byte))
            scores.append(-1000.0)
            types.append(6)
        }

        // Normal tokens starting at 259
        let normalTokens: [(String, Float)] = [
            ("\u{2581}", -1.0),          // 259: ▁
            ("a", -2.0),                  // 260
            ("b", -2.0),                  // 261
            ("c", -2.0),                  // 262
            ("\u{2581}a", -0.5),         // 263: ▁a (merge ▁ + a)
            ("ab", -0.3),                // 264: ab (merge a + b)
            ("\u{2581}ab", -0.1),        // 265: ▁ab (merge ▁a + b)
            ("\u{2581}abc", 0.0),        // 266: ▁abc (merge ▁ab + c)
        ]

        for (token, score) in normalTokens {
            vocab.append(token)
            scores.append(score)
            types.append(1)
        }

        return try SentencePieceBPETokenizer(
            vocabulary: vocab,
            scores: scores,
            specialTokens: SpecialTokens(
                bosTokenID: 1,
                eosTokenID: 2,
                unknownTokenID: 0,
                addBosToken: true,
                addEosToken: false
            ),
            tokenTypes: types,
            addSpacePrefix: true
        )
    }

    @Test("parseByteToken valid")
    func parseByteTokenValid() {
        #expect(SentencePieceBPETokenizer.parseByteToken("<0x00>") == 0x00)
        #expect(SentencePieceBPETokenizer.parseByteToken("<0xFF>") == 0xFF)
        #expect(SentencePieceBPETokenizer.parseByteToken("<0xAB>") == 0xAB)
    }

    @Test("parseByteToken invalid")
    func parseByteTokenInvalid() {
        #expect(SentencePieceBPETokenizer.parseByteToken("hello") == nil)
        #expect(SentencePieceBPETokenizer.parseByteToken("<0x>") == nil)
        #expect(SentencePieceBPETokenizer.parseByteToken("<0xGG>") == nil)
        #expect(SentencePieceBPETokenizer.parseByteToken("") == nil)
    }

    @Test("Encode simple text produces BOS")
    func encodeWithBOS() throws {
        let tokenizer = try SentencePieceBPETokenizerTests.makeTestTokenizer()
        let tokens = tokenizer.encode(text: "abc")
        #expect(tokens.first == 1) // BOS token
    }

    @Test("Encode text with merges")
    func encodeWithMerges() throws {
        let tokenizer = try SentencePieceBPETokenizerTests.makeTestTokenizer()
        let tokens = tokenizer.encode(text: "abc")
        // Should produce: BOS + ▁abc (merged from ▁ + a + b + c)
        #expect(tokens.contains(266)) // ▁abc token
    }

    @Test("Decode reverses encode for simple text")
    func decodeRoundtrip() throws {
        let tokenizer = try SentencePieceBPETokenizerTests.makeTestTokenizer()
        let text = "abc"
        let tokens = tokenizer.encode(text: text)
        // Remove BOS for decode comparison
        let withoutBOS = Array(tokens.dropFirst())
        let decoded = tokenizer.decode(tokens: withoutBOS)
        #expect(decoded == text)
    }

    @Test("Decode skips control tokens")
    func decodeSkipsControl() throws {
        let tokenizer = try SentencePieceBPETokenizerTests.makeTestTokenizer()
        let decoded = tokenizer.decode(tokens: [1, 266, 2]) // BOS + ▁abc + EOS
        #expect(decoded == "abc")
    }

    @Test("Byte fallback for unknown characters")
    func byteFallback() throws {
        let tokenizer = try SentencePieceBPETokenizerTests.makeTestTokenizer()
        // 'x' is not in our normal vocabulary, should fall back to bytes
        let tokens = tokenizer.encode(text: "x")
        // Should contain byte token for 'x' (0x78)
        // Check that it contains the ▁ token or byte tokens
        #expect(tokens.count >= 2) // At least BOS + something
    }

    @Test("Vocabulary size")
    func vocabSize() throws {
        let tokenizer = try SentencePieceBPETokenizerTests.makeTestTokenizer()
        #expect(tokenizer.vocabularySize == 267)
    }

    @Test("Mismatched vocabulary and scores throws")
    func mismatchedScores() {
        #expect(throws: TokenizerError.self) {
            _ = try SentencePieceBPETokenizer(
                vocabulary: ["a", "b", "c"],
                scores: [1.0, 2.0], // Wrong count
                specialTokens: SpecialTokens()
            )
        }
    }

    @Test("Space replacement in encode")
    func spaceReplacement() throws {
        let tokenizer = try SentencePieceBPETokenizerTests.makeTestTokenizer()
        // Input "a b" should become "▁a▁b" after preprocessing
        let tokens = tokenizer.encode(text: "a b")
        // Verify decode gives back original
        let withoutBOS = Array(tokens.dropFirst())
        let decoded = tokenizer.decode(tokens: withoutBOS)
        #expect(decoded == "a b")
    }

    @Test("tokenToString returns correct string")
    func tokenToStringTest() throws {
        let tokenizer = try SentencePieceBPETokenizerTests.makeTestTokenizer()
        #expect(tokenizer.tokenToString(0) == "<unk>")
        #expect(tokenizer.tokenToString(1) == "<s>")
        #expect(tokenizer.tokenToString(266) == "\u{2581}abc")
        #expect(tokenizer.tokenToString(-1) == nil)
    }
}

// MARK: - StreamingDetokenizer Tests

@Suite("StreamingDetokenizer")
struct StreamingDetokenizerTests {

    static func makeSimpleTokenizer() throws -> MergesBPETokenizer {
        var vocab: [String] = []
        for byte: UInt8 in 0...255 {
            vocab.append(String(Character(ByteEncoder.byteToUnicode[byte]!)))
        }
        return try MergesBPETokenizer(
            vocabulary: vocab,
            merges: [],
            preTokenizer: .gpt2,
            specialTokens: SpecialTokens()
        )
    }

    @Test("Single token produces output")
    func singleToken() throws {
        let tokenizer = try StreamingDetokenizerTests.makeSimpleTokenizer()
        var detok = StreamingDetokenizer(tokenizer: tokenizer)

        // Token for 'H' (0x48)
        let result = detok.append(token: 0x48)
        #expect(result == "H")
    }

    @Test("Multiple tokens produce incremental output")
    func multipleTokens() throws {
        let tokenizer = try StreamingDetokenizerTests.makeSimpleTokenizer()
        var detok = StreamingDetokenizer(tokenizer: tokenizer)

        let r1 = detok.append(token: 0x48) // H
        let r2 = detok.append(token: 0x69) // i
        #expect(r1 == "H")
        #expect(r2 == "i")
    }

    @Test("Full streaming matches batch decode")
    func streamingMatchesBatch() throws {
        let tokenizer = try StreamingDetokenizerTests.makeSimpleTokenizer()
        var detok = StreamingDetokenizer(tokenizer: tokenizer)

        let tokenIDs: [Int] = [0x48, 0x65, 0x6C, 0x6C, 0x6F] // "Hello"
        var streamed = ""
        for id in tokenIDs {
            if let text = detok.append(token: id) {
                streamed += text
            }
        }

        let batchDecoded = tokenizer.decode(tokens: tokenIDs)
        #expect(streamed == batchDecoded)
    }
}

// MARK: - GGUFTokenizerFactory Tests

@Suite("GGUFTokenizerFactory")
struct GGUFTokenizerFactoryTests {

    @Test("Create merges-based tokenizer from GGUF")
    func createMergesBPE() throws {
        let data = buildGGUFWithTokenizer(model: "gpt2")
        let file = try GGUFFile.parse(data: data)
        let tokenizer = try GGUFTokenizerFactory.create(from: file)

        #expect(tokenizer.vocabularySize == 4)
        #expect(tokenizer.bosTokenID == 2)
        #expect(tokenizer.eosTokenID == 3)
    }

    @Test("Create SentencePiece tokenizer from GGUF")
    func createSentencePieceBPE() throws {
        let data = buildGGUFWithTokenizer(model: "llama")
        let file = try GGUFFile.parse(data: data)
        let tokenizer = try GGUFTokenizerFactory.create(from: file)

        #expect(tokenizer.vocabularySize == 4)
    }

    @Test("Missing vocabulary throws")
    func missingVocabulary() throws {
        let data = buildMinimalGGUF()
        let file = try GGUFFile.parse(data: data)

        #expect(throws: TokenizerError.self) {
            _ = try GGUFTokenizerFactory.create(from: file)
        }
    }

    @Test("Unsupported model throws")
    func unsupportedModel() throws {
        let data = buildGGUFWithTokenizer(model: "unknown_model")
        let file = try GGUFFile.parse(data: data)

        #expect(throws: TokenizerError.self) {
            _ = try GGUFTokenizerFactory.create(from: file)
        }
    }
}

// MARK: - BPE Algorithm Correctness Tests

@Suite("BPE Algorithm")
struct BPEAlgorithmTests {

    @Test("Merge priority order is respected")
    func mergePriorityOrder() throws {
        // Create vocabulary where we have two possible merges
        // but one has higher priority (lower rank)
        var vocab: [String] = []
        for byte: UInt8 in 0...255 {
            vocab.append(String(Character(ByteEncoder.byteToUnicode[byte]!)))
        }
        let aChar = String(Character(ByteEncoder.byteToUnicode[0x61]!))
        let bChar = String(Character(ByteEncoder.byteToUnicode[0x62]!))
        let cChar = String(Character(ByteEncoder.byteToUnicode[0x63]!))
        vocab.append(aChar + bChar) // 256: "ab"
        vocab.append(bChar + cChar) // 257: "bc"

        // "ab" merge has higher priority (rank 0) than "bc" (rank 1)
        let merges = [
            "\(aChar) \(bChar)",
            "\(bChar) \(cChar)",
        ]

        let tokenizer = try MergesBPETokenizer(
            vocabulary: vocab,
            merges: merges,
            preTokenizer: .gpt2,
            specialTokens: SpecialTokens()
        )

        // "abc" should first merge "ab", then "c" remains separate
        let result = tokenizer.bpe(token: aChar + bChar + cChar)
        #expect(result == [aChar + bChar, cChar])
    }

    @Test("SentencePiece score-based merge selects highest score")
    func sentencePieceScorePriority() throws {
        var vocab: [String] = []
        var scores: [Float] = []
        var types: [Int] = []

        // Build minimal vocabulary
        // 0: <unk>
        vocab.append("<unk>"); scores.append(-1000); types.append(2)

        // Individual characters
        vocab.append("a"); scores.append(-5.0); types.append(1)  // 1
        vocab.append("b"); scores.append(-5.0); types.append(1)  // 2
        vocab.append("c"); scores.append(-5.0); types.append(1)  // 3

        // Merged tokens with different scores
        vocab.append("ab"); scores.append(-1.0); types.append(1) // 4: ab (high score)
        vocab.append("bc"); scores.append(-2.0); types.append(1) // 5: bc (lower score)

        let tokenizer = try SentencePieceBPETokenizer(
            vocabulary: vocab,
            scores: scores,
            specialTokens: SpecialTokens(unknownTokenID: 0),
            tokenTypes: types,
            addSpacePrefix: false // Disable for this test
        )

        // "abc": both "ab" and "bc" merges are possible
        // "ab" has higher score (-1.0 > -2.0), so should merge first
        let tokens = tokenizer.encode(text: "abc")
        // Should produce: "ab" + "c" (not "a" + "bc")
        #expect(tokens.contains(4)) // ab token
        #expect(tokens.contains(3)) // c token
        #expect(!tokens.contains(5)) // bc should NOT be present
    }
}

// MARK: - Test Helpers

private func buildMinimalGGUF() -> Data {
    var data = Data()
    appendUInt32(&data, 0x4655_4747) // Magic
    appendUInt32(&data, 3)            // Version
    appendUInt64(&data, 0)            // Tensor count
    appendUInt64(&data, 1)            // Metadata KV count

    appendString(&data, "general.architecture")
    appendUInt32(&data, GGUFMetadataValueType.string.rawValue)
    appendString(&data, "llama")

    return data
}

private func buildGGUFWithTokenizer(model: String) -> Data {
    var data = Data()
    appendUInt32(&data, 0x4655_4747) // Magic
    appendUInt32(&data, 3)            // Version
    appendUInt64(&data, 0)            // Tensor count

    // Count metadata entries
    var kvCount: UInt64 = 6 // architecture, model, tokens, bos, eos, token_type
    if model == "gpt2" {
        kvCount += 1 // merges
    }
    if model == "llama" {
        kvCount += 1 // scores
    }
    appendUInt64(&data, kvCount)

    // general.architecture
    appendString(&data, "general.architecture")
    appendUInt32(&data, GGUFMetadataValueType.string.rawValue)
    appendString(&data, "llama")

    // tokenizer.ggml.model
    appendString(&data, "tokenizer.ggml.model")
    appendUInt32(&data, GGUFMetadataValueType.string.rawValue)
    appendString(&data, model)

    // tokenizer.ggml.tokens: ["a", "b", "<s>", "</s>"]
    appendString(&data, "tokenizer.ggml.tokens")
    appendUInt32(&data, GGUFMetadataValueType.array.rawValue)
    appendUInt32(&data, GGUFMetadataValueType.string.rawValue) // element type
    appendUInt64(&data, 4) // count
    appendString(&data, "a")
    appendString(&data, "b")
    appendString(&data, "<s>")
    appendString(&data, "</s>")

    // tokenizer.ggml.token_type: [1, 1, 3, 3]
    appendString(&data, "tokenizer.ggml.token_type")
    appendUInt32(&data, GGUFMetadataValueType.array.rawValue)
    appendUInt32(&data, GGUFMetadataValueType.int32.rawValue) // element type
    appendUInt64(&data, 4) // count
    appendInt32(&data, 1)
    appendInt32(&data, 1)
    appendInt32(&data, 3)
    appendInt32(&data, 3)

    // tokenizer.ggml.bos_token_id
    appendString(&data, "tokenizer.ggml.bos_token_id")
    appendUInt32(&data, GGUFMetadataValueType.uint32.rawValue)
    appendUInt32(&data, 2)

    // tokenizer.ggml.eos_token_id
    appendString(&data, "tokenizer.ggml.eos_token_id")
    appendUInt32(&data, GGUFMetadataValueType.uint32.rawValue)
    appendUInt32(&data, 3)

    if model == "gpt2" {
        // tokenizer.ggml.merges: ["a b"]
        appendString(&data, "tokenizer.ggml.merges")
        appendUInt32(&data, GGUFMetadataValueType.array.rawValue)
        appendUInt32(&data, GGUFMetadataValueType.string.rawValue)
        appendUInt64(&data, 1)
        appendString(&data, "a b")
    }

    if model == "llama" {
        // tokenizer.ggml.scores: [-1.0, -1.0, 0.0, 0.0]
        appendString(&data, "tokenizer.ggml.scores")
        appendUInt32(&data, GGUFMetadataValueType.array.rawValue)
        appendUInt32(&data, GGUFMetadataValueType.float32.rawValue)
        appendUInt64(&data, 4)
        appendFloat32(&data, -1.0)
        appendFloat32(&data, -1.0)
        appendFloat32(&data, 0.0)
        appendFloat32(&data, 0.0)
    }

    return data
}

private func appendUInt32(_ data: inout Data, _ value: UInt32) {
    var v = value.littleEndian
    data.append(Data(bytes: &v, count: 4))
}

private func appendInt32(_ data: inout Data, _ value: Int32) {
    var v = value.littleEndian
    data.append(Data(bytes: &v, count: 4))
}

private func appendUInt64(_ data: inout Data, _ value: UInt64) {
    var v = value.littleEndian
    data.append(Data(bytes: &v, count: 8))
}

private func appendFloat32(_ data: inout Data, _ value: Float) {
    var v = value.bitPattern.littleEndian
    data.append(Data(bytes: &v, count: 4))
}

private func appendString(_ data: inout Data, _ string: String) {
    let utf8 = string.utf8
    appendUInt64(&data, UInt64(utf8.count))
    data.append(contentsOf: utf8)
}
