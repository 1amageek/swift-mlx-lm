import Testing
import Foundation
@testable import GGUFTokenizer
@testable import GGUFParser

// MARK: - ByteEncoder Edge Cases

@Suite("ByteEncoder Edge Cases")
struct ByteEncoderEdgeCaseTests {

    @Test("Empty input encode")
    func emptyEncode() {
        let result = ByteEncoder.encode([UInt8]())
        #expect(result == "")
    }

    @Test("Empty string decode")
    func emptyDecode() {
        let result = ByteEncoder.decode("")
        #expect(result.isEmpty)
    }

    @Test("Single byte boundary values")
    func singleByteBoundaries() {
        // Test all boundary bytes between mapped ranges
        let boundaries: [UInt8] = [0x00, 0x20, 0x21, 0x7E, 0x7F, 0xA0, 0xA1, 0xAC, 0xAD, 0xAE, 0xFF]
        for byte in boundaries {
            let encoded = ByteEncoder.encode([byte])
            let decoded = ByteEncoder.decode(encoded)
            #expect(decoded == [byte], "Roundtrip failed for byte 0x\(String(format: "%02X", byte))")
        }
    }

    @Test("Decode string with unknown scalars drops them")
    func decodeUnknownScalars() {
        // U+1F600 (😀) is not in the byte encoder mapping
        let result = ByteEncoder.decode("A\u{1F600}B")
        // Only A (0x41) and B (0x42) should survive
        #expect(result == [0x41, 0x42])
    }

    @Test("Full 256-byte roundtrip")
    func full256Roundtrip() {
        let allBytes: [UInt8] = (0...255).map { UInt8($0) }
        let encoded = ByteEncoder.encode(allBytes)
        let decoded = ByteEncoder.decode(encoded)
        #expect(decoded == allBytes)
    }

    @Test("Repeated same byte")
    func repeatedByte() {
        let bytes: [UInt8] = Array(repeating: 0x41, count: 1000)
        let encoded = ByteEncoder.encode(bytes)
        let decoded = ByteEncoder.decode(encoded)
        #expect(decoded == bytes)
    }
}

// MARK: - PreTokenizer Edge Cases

@Suite("PreTokenizer Edge Cases")
struct PreTokenizerEdgeCaseTests {

    @Test("Only whitespace")
    func onlyWhitespace() {
        let result = PreTokenizer.gpt2.split(text: "   ")
        #expect(!result.isEmpty)
        #expect(result.joined() == "   ")
    }

    @Test("Only punctuation")
    func onlyPunctuation() {
        let result = PreTokenizer.gpt2.split(text: "...")
        #expect(!result.isEmpty)
    }

    @Test("Only numbers")
    func onlyNumbers() {
        let result = PreTokenizer.gpt2.split(text: "42")
        #expect(!result.isEmpty)
        #expect(result.joined() == "42")
    }

    @Test("CJK characters")
    func cjkCharacters() {
        let result = PreTokenizer.llamaBPE.split(text: "日本語テスト")
        let joined = result.joined()
        #expect(joined == "日本語テスト")
    }

    @Test("Emoji")
    func emoji() {
        let result = PreTokenizer.gpt2.split(text: "Hello 👋 World")
        let joined = result.joined()
        #expect(joined == "Hello 👋 World")
    }

    @Test("Mixed scripts")
    func mixedScripts() {
        let result = PreTokenizer.llamaBPE.split(text: "Hello世界مرحبا")
        let joined = result.joined()
        #expect(joined == "Hello世界مرحبا")
    }

    @Test("Tab and carriage return")
    func tabAndCR() {
        let result = PreTokenizer.gpt2.split(text: "a\tb\rc")
        let joined = result.joined()
        #expect(joined == "a\tb\rc")
    }

    @Test("Single character")
    func singleChar() {
        let result = PreTokenizer.gpt2.split(text: "a")
        #expect(result == ["a"])
    }

    @Test("Very long input preserves all content")
    func veryLongInput() {
        let long = String(repeating: "Hello World! ", count: 1000)
        let result = PreTokenizer.gpt2.split(text: long)
        let joined = result.joined()
        #expect(joined == long)
    }

    @Test("Llama BPE with leading non-letter")
    func llamaLeadingNonLetter() {
        // Pattern: [^\r\n\p{L}\p{N}]?\p{L}+ matches optional non-letter + letters
        let result = PreTokenizer.llamaBPE.split(text: " Hello")
        #expect(result.contains(" Hello") || result.contains("Hello"))
    }

    @Test("Consecutive newlines in Llama BPE")
    func consecutiveNewlines() {
        let result = PreTokenizer.llamaBPE.split(text: "a\n\nb")
        let joined = result.joined()
        #expect(joined == "a\n\nb")
    }

    @Test("Numbers at Qwen2 vs Llama boundary")
    func numberBoundary() {
        // "1234" → Llama groups [123, 4], Qwen splits [1, 2, 3, 4]
        let llama = PreTokenizer.llamaBPE.split(text: "1234")
        let qwen = PreTokenizer.qwen2.split(text: "1234")
        #expect(llama == ["123", "4"])
        #expect(qwen == ["1", "2", "3", "4"])
    }
}

// MARK: - MergesBPETokenizer Edge Cases

@Suite("MergesBPE Edge Cases")
struct MergesBPEEdgeCaseTests {

    static func makeFullTokenizer(
        extraVocab: [(String, Int)] = [],
        extraMerges: [String] = [],
        specialTokens: SpecialTokens = SpecialTokens(),
        tokenTypes: [Int] = []
    ) throws -> MergesBPETokenizer {
        var vocab: [String] = []
        for byte: UInt8 in 0...255 {
            vocab.append(String(Character(ByteEncoder.byteToUnicode[byte]!)))
        }
        for (token, _) in extraVocab {
            vocab.append(token)
        }
        var types = tokenTypes
        if types.isEmpty {
            types = Array(repeating: 1, count: vocab.count)
        }
        return try MergesBPETokenizer(
            vocabulary: vocab,
            merges: extraMerges,
            preTokenizer: .gpt2,
            specialTokens: specialTokens,
            tokenTypes: types
        )
    }

    @Test("Empty string encode returns only BOS/EOS")
    func emptyStringEncode() throws {
        let tokenizer = try MergesBPEEdgeCaseTests.makeFullTokenizer(
            specialTokens: SpecialTokens(bosTokenID: 0, eosTokenID: 1, addBosToken: true, addEosToken: true)
        )
        let tokens = tokenizer.encode(text: "")
        #expect(tokens == [0, 1])
    }

    @Test("Empty string encode without BOS/EOS returns empty")
    func emptyStringEncodeNoBOS() throws {
        let tokenizer = try MergesBPEEdgeCaseTests.makeFullTokenizer()
        let tokens = tokenizer.encode(text: "")
        #expect(tokens.isEmpty)
    }

    @Test("Decode empty token list returns empty string")
    func decodeEmptyTokens() throws {
        let tokenizer = try MergesBPEEdgeCaseTests.makeFullTokenizer()
        let result = tokenizer.decode(tokens: [])
        #expect(result == "")
    }

    @Test("Decode with negative token IDs skips them")
    func decodeNegativeIDs() throws {
        let tokenizer = try MergesBPEEdgeCaseTests.makeFullTokenizer()
        let result = tokenizer.decode(tokens: [-1, -100, Int.min])
        #expect(result == "")
    }

    @Test("Decode with out-of-range token IDs skips them")
    func decodeOutOfRange() throws {
        let tokenizer = try MergesBPEEdgeCaseTests.makeFullTokenizer()
        let result = tokenizer.decode(tokens: [999999, Int.max])
        #expect(result == "")
    }

    @Test("Decode with only control tokens returns empty")
    func decodeOnlyControlTokens() throws {
        var types = Array(repeating: 1, count: 256)
        types[0] = 3 // Make token 0 a control token
        types[1] = 3 // Make token 1 a control token
        let tokenizer = try MergesBPEEdgeCaseTests.makeFullTokenizer(tokenTypes: types)
        let result = tokenizer.decode(tokens: [0, 1])
        #expect(result == "")
    }

    @Test("Unicode roundtrip through full pipeline - CJK")
    func unicodeRoundtripCJK() throws {
        let tokenizer = try MergesBPEEdgeCaseTests.makeFullTokenizer()
        let text = "日本語"
        let tokens = tokenizer.encode(text: text)
        let decoded = tokenizer.decode(tokens: tokens)
        #expect(decoded == text)
    }

    @Test("Unicode roundtrip - Emoji")
    func unicodeRoundtripEmoji() throws {
        let tokenizer = try MergesBPEEdgeCaseTests.makeFullTokenizer()
        let text = "Hello 👋🏽"
        let tokens = tokenizer.encode(text: text)
        let decoded = tokenizer.decode(tokens: tokens)
        #expect(decoded == text)
    }

    @Test("Unicode roundtrip - Arabic")
    func unicodeRoundtripArabic() throws {
        let tokenizer = try MergesBPEEdgeCaseTests.makeFullTokenizer()
        let text = "مرحبا"
        let tokens = tokenizer.encode(text: text)
        let decoded = tokenizer.decode(tokens: tokens)
        #expect(decoded == text)
    }

    @Test("Repeated character with merge - aaaa")
    func repeatedCharWithMerge() throws {
        let aChar = String(Character(ByteEncoder.byteToUnicode[0x61]!))
        var vocab: [String] = []
        for byte: UInt8 in 0...255 {
            vocab.append(String(Character(ByteEncoder.byteToUnicode[byte]!)))
        }
        vocab.append(aChar + aChar) // 256: "aa"

        let tokenizer = try MergesBPETokenizer(
            vocabulary: vocab,
            merges: ["\(aChar) \(aChar)"],
            preTokenizer: .gpt2,
            specialTokens: SpecialTokens()
        )

        // "aaaa" → should merge pairs: [aa, aa]
        let result = tokenizer.bpe(token: aChar + aChar + aChar + aChar)
        #expect(result == [aChar + aChar, aChar + aChar])

        // "aaa" → [aa, a]
        let result3 = tokenizer.bpe(token: aChar + aChar + aChar)
        #expect(result3 == [aChar + aChar, aChar])

        // "aaaaa" → [aa, aa, a]
        let result5 = tokenizer.bpe(
            token: aChar + aChar + aChar + aChar + aChar
        )
        #expect(result5 == [aChar + aChar, aChar + aChar, aChar])
    }

    @Test("No merges - each character stays separate")
    func noMerges() throws {
        let tokenizer = try MergesBPEEdgeCaseTests.makeFullTokenizer()
        let text = "ab"
        let tokens = tokenizer.encode(text: text)
        let decoded = tokenizer.decode(tokens: tokens)
        #expect(decoded == text)
        #expect(tokens.count == 2) // Each char is a separate token
    }

    @Test("Mixed valid and invalid token IDs in decode")
    func mixedValidInvalidDecode() throws {
        let tokenizer = try MergesBPEEdgeCaseTests.makeFullTokenizer()
        // 0x48 = 'H', 0x69 = 'i'
        let result = tokenizer.decode(tokens: [-1, 0x48, 999999, 0x69, -5])
        #expect(result == "Hi")
    }

    @Test("Encode-decode roundtrip with spaces and punctuation")
    func roundtripSpacesPunctuation() throws {
        let tokenizer = try MergesBPEEdgeCaseTests.makeFullTokenizer()
        let text = "Hello, World! How's it going?"
        let tokens = tokenizer.encode(text: text)
        let decoded = tokenizer.decode(tokens: tokens)
        #expect(decoded == text)
    }

    @Test("Encode-decode roundtrip with newlines")
    func roundtripNewlines() throws {
        let tokenizer = try MergesBPEEdgeCaseTests.makeFullTokenizer()
        let text = "line1\nline2\nline3"
        let tokens = tokenizer.encode(text: text)
        let decoded = tokenizer.decode(tokens: tokens)
        #expect(decoded == text)
    }

    @Test("Duplicate token in vocabulary - last ID wins in tokenToID")
    func duplicateVocab() throws {
        let vocab = ["a", "b", "a"] // "a" at ID 0 and 2
        let tokenizer = try MergesBPETokenizer(
            vocabulary: vocab,
            merges: [],
            preTokenizer: .gpt2,
            specialTokens: SpecialTokens()
        )
        // tokenToID["a"] should be 2 (last occurrence)
        #expect(tokenizer.tokenToID["a"] == 2)
        // But decode token 0 should still give "a"
        #expect(tokenizer.decode(tokens: [0]) == "a")
    }

    @Test("Token types array shorter than vocabulary is handled")
    func shortTokenTypes() throws {
        var vocab: [String] = []
        for byte: UInt8 in 0...255 {
            vocab.append(String(Character(ByteEncoder.byteToUnicode[byte]!)))
        }
        // Only 10 types for 256 tokens - should not crash
        let tokenizer = try MergesBPETokenizer(
            vocabulary: vocab,
            merges: [],
            preTokenizer: .gpt2,
            specialTokens: SpecialTokens(),
            tokenTypes: Array(repeating: 1, count: 10)
        )
        #expect(tokenizer.vocabularySize == 256)
    }

    @Test("Chained merges (A+B → AB, AB+C → ABC)")
    func chainedMerges() throws {
        let aChar = String(Character(ByteEncoder.byteToUnicode[0x61]!))
        let bChar = String(Character(ByteEncoder.byteToUnicode[0x62]!))
        let cChar = String(Character(ByteEncoder.byteToUnicode[0x63]!))

        var vocab: [String] = []
        for byte: UInt8 in 0...255 {
            vocab.append(String(Character(ByteEncoder.byteToUnicode[byte]!)))
        }
        vocab.append(aChar + bChar)                    // 256: ab
        vocab.append(aChar + bChar + cChar)            // 257: abc

        let merges = [
            "\(aChar) \(bChar)",                       // rank 0: a + b → ab
            "\(aChar + bChar) \(cChar)",               // rank 1: ab + c → abc
        ]

        let tokenizer = try MergesBPETokenizer(
            vocabulary: vocab,
            merges: merges,
            preTokenizer: .gpt2,
            specialTokens: SpecialTokens()
        )

        let result = tokenizer.bpe(token: aChar + bChar + cChar)
        #expect(result == [aChar + bChar + cChar])
    }

    @Test("Competing merges - lower rank wins")
    func competingMerges() throws {
        let aChar = String(Character(ByteEncoder.byteToUnicode[0x61]!))
        let bChar = String(Character(ByteEncoder.byteToUnicode[0x62]!))
        let cChar = String(Character(ByteEncoder.byteToUnicode[0x63]!))

        var vocab: [String] = []
        for byte: UInt8 in 0...255 {
            vocab.append(String(Character(ByteEncoder.byteToUnicode[byte]!)))
        }
        vocab.append(bChar + cChar)   // 256: bc
        vocab.append(aChar + bChar)   // 257: ab

        // bc has higher priority (rank 0) than ab (rank 1)
        let merges = [
            "\(bChar) \(cChar)",   // rank 0
            "\(aChar) \(bChar)",   // rank 1
        ]

        let tokenizer = try MergesBPETokenizer(
            vocabulary: vocab,
            merges: merges,
            preTokenizer: .gpt2,
            specialTokens: SpecialTokens()
        )

        // "abc": (b,c) rank 0 beats (a,b) rank 1 → [a, bc]
        let result = tokenizer.bpe(token: aChar + bChar + cChar)
        #expect(result == [aChar, bChar + cChar])
    }
}

// MARK: - SentencePieceBPE Edge Cases

@Suite("SentencePieceBPE Edge Cases")
struct SentencePieceBPEEdgeCaseTests {

    static func makeTokenizer(
        extraTokens: [(String, Float, Int)] = [],
        specialTokens: SpecialTokens = SpecialTokens(
            bosTokenID: 1, eosTokenID: 2, unknownTokenID: 0,
            addBosToken: true, addEosToken: false
        ),
        addSpacePrefix: Bool = true
    ) throws -> SentencePieceBPETokenizer {
        var vocab: [String] = []
        var scores: [Float] = []
        var types: [Int] = []

        // 0: <unk>
        vocab.append("<unk>"); scores.append(-1000.0); types.append(2)
        // 1: <s>
        vocab.append("<s>"); scores.append(0.0); types.append(3)
        // 2: </s>
        vocab.append("</s>"); scores.append(0.0); types.append(3)

        // Byte tokens 3-258
        for byte in 0...255 {
            vocab.append(String(format: "<0x%02X>", byte))
            scores.append(-1000.0)
            types.append(6)
        }

        // Normal tokens from 259
        let baseTokens: [(String, Float, Int)] = [
            ("\u{2581}", -1.0, 1),     // 259: ▁
            ("a", -2.0, 1),             // 260
            ("b", -2.0, 1),             // 261
            ("c", -2.0, 1),             // 262
            ("\u{2581}a", -0.5, 1),    // 263
            ("ab", -0.3, 1),           // 264
            ("\u{2581}ab", -0.1, 1),   // 265
            ("\u{2581}abc", 0.0, 1),   // 266
        ]

        for (token, score, tokenType) in baseTokens + extraTokens {
            vocab.append(token)
            scores.append(score)
            types.append(tokenType)
        }

        return try SentencePieceBPETokenizer(
            vocabulary: vocab,
            scores: scores,
            specialTokens: specialTokens,
            tokenTypes: types,
            addSpacePrefix: addSpacePrefix
        )
    }

    @Test("Empty string encode with BOS returns only BOS")
    func emptyStringEncode() throws {
        let tokenizer = try SentencePieceBPEEdgeCaseTests.makeTokenizer()
        let tokens = tokenizer.encode(text: "")
        #expect(tokens == [1]) // Only BOS
    }

    @Test("Empty string encode with BOS+EOS returns only BOS+EOS")
    func emptyStringEncodeWithEOS() throws {
        let tokenizer = try SentencePieceBPEEdgeCaseTests.makeTokenizer(
            specialTokens: SpecialTokens(
                bosTokenID: 1, eosTokenID: 2, unknownTokenID: 0,
                addBosToken: true, addEosToken: true
            )
        )
        let tokens = tokenizer.encode(text: "")
        #expect(tokens == [1, 2])
    }

    @Test("Empty string encode without special tokens returns empty")
    func emptyStringNoSpecial() throws {
        let tokenizer = try SentencePieceBPEEdgeCaseTests.makeTokenizer(
            specialTokens: SpecialTokens(unknownTokenID: 0)
        )
        let tokens = tokenizer.encode(text: "")
        #expect(tokens.isEmpty)
    }

    @Test("Decode empty token list returns empty")
    func decodeEmpty() throws {
        let tokenizer = try SentencePieceBPEEdgeCaseTests.makeTokenizer()
        #expect(tokenizer.decode(tokens: []) == "")
    }

    @Test("Decode with negative and out-of-range IDs")
    func decodeInvalidIDs() throws {
        let tokenizer = try SentencePieceBPEEdgeCaseTests.makeTokenizer()
        #expect(tokenizer.decode(tokens: [-1, 999999, Int.max, Int.min]) == "")
    }

    @Test("addSpacePrefix=false does not prepend ▁")
    func noSpacePrefix() throws {
        let tokenizer = try SentencePieceBPEEdgeCaseTests.makeTokenizer(addSpacePrefix: false)
        let tokens = tokenizer.encode(text: "abc")
        // Without prefix, "abc" → symbols: [a, b, c] → merge → [abc] or similar
        // Token 266 is "▁abc" which won't match "abc" without prefix
        // So tokens should NOT contain 266
        #expect(!tokens.contains(266))
    }

    @Test("Text with only spaces")
    func onlySpaces() throws {
        let tokenizer = try SentencePieceBPEEdgeCaseTests.makeTokenizer()
        let tokens = tokenizer.encode(text: "   ")
        let decoded = tokenizer.decode(tokens: Array(tokens.dropFirst())) // drop BOS
        #expect(decoded == "   ")
    }

    @Test("All byte fallback for unknown characters")
    func allByteFallback() throws {
        let tokenizer = try SentencePieceBPEEdgeCaseTests.makeTokenizer()
        // 'x' is not in normal vocabulary, should use byte fallback
        let tokens = tokenizer.encode(text: "x")
        // Should have BOS + ▁ byte tokens + x byte token
        #expect(tokens.first == 1) // BOS
        // Should be decodable
        let decoded = tokenizer.decode(tokens: Array(tokens.dropFirst()))
        #expect(decoded == "x")
    }

    @Test("Multi-byte UTF-8 byte fallback - CJK")
    func cjkByteFallback() throws {
        let tokenizer = try SentencePieceBPEEdgeCaseTests.makeTokenizer()
        let text = "日" // 3-byte UTF-8: E6 97 A5
        let tokens = tokenizer.encode(text: text)
        let decoded = tokenizer.decode(tokens: Array(tokens.dropFirst()))
        #expect(decoded == text)
    }

    @Test("Emoji byte fallback")
    func emojiByteFallback() throws {
        let tokenizer = try SentencePieceBPEEdgeCaseTests.makeTokenizer()
        let text = "😀" // 4-byte UTF-8: F0 9F 98 80
        let tokens = tokenizer.encode(text: text)
        let decoded = tokenizer.decode(tokens: Array(tokens.dropFirst()))
        #expect(decoded == text)
    }

    @Test("Interleaved byte and non-byte tokens in decode")
    func interleavedByteDecode() throws {
        let tokenizer = try SentencePieceBPEEdgeCaseTests.makeTokenizer()
        // Manually construct: "a" token + byte tokens for 'x' (0x78) + "b" token
        let aID = 260
        let bID = 261
        let xByteID = 3 + 0x78 // byte token for 'x'
        let decoded = tokenizer.decode(tokens: [aID, xByteID, bID])
        #expect(decoded == "axb")
    }

    @Test("Consecutive byte tokens form valid UTF-8")
    func consecutiveByteTokensDecode() throws {
        let tokenizer = try SentencePieceBPEEdgeCaseTests.makeTokenizer()
        // UTF-8 for "日": E6 97 A5
        let byte1 = 3 + 0xE6
        let byte2 = 3 + 0x97
        let byte3 = 3 + 0xA5
        let decoded = tokenizer.decode(tokens: [byte1, byte2, byte3])
        #expect(decoded == "日")
    }

    @Test("Invalid byte sequence in decode produces replacement character")
    func invalidByteSequenceDecode() throws {
        let tokenizer = try SentencePieceBPEEdgeCaseTests.makeTokenizer()
        // 0xFF 0xFE is not valid UTF-8
        let decoded = tokenizer.decode(tokens: [3 + 0xFF, 3 + 0xFE])
        // Should produce replacement characters, not crash
        #expect(decoded.contains("\u{FFFD}"))
    }

    @Test("No possible merges - symbols stay individual")
    func noPossibleMerges() throws {
        // Create tokenizer with no merged tokens
        var vocab: [String] = ["<unk>"]
        var scores: [Float] = [-1000.0]
        var types: [Int] = [2]

        for byte in 0...255 {
            vocab.append(String(format: "<0x%02X>", byte))
            scores.append(-1000.0)
            types.append(6)
        }
        vocab.append("a"); scores.append(-2.0); types.append(1)
        vocab.append("b"); scores.append(-2.0); types.append(1)

        let tokenizer = try SentencePieceBPETokenizer(
            vocabulary: vocab,
            scores: scores,
            specialTokens: SpecialTokens(unknownTokenID: 0),
            tokenTypes: types,
            addSpacePrefix: false
        )

        // "ab" with no "ab" in vocabulary → no merges → [a, b]
        let tokens = tokenizer.encode(text: "ab")
        // Should contain individual a and b tokens
        let aID = vocab.firstIndex(of: "a")!
        let bID = vocab.firstIndex(of: "b")!
        #expect(tokens.contains(aID))
        #expect(tokens.contains(bID))
    }

    @Test("Score tie breaks by leftmost position")
    func scoreTieBreak() throws {
        var vocab: [String] = ["<unk>"]
        var scores: [Float] = [-1000.0]
        var types: [Int] = [2]

        vocab.append("a"); scores.append(-5.0); types.append(1)  // 1
        vocab.append("b"); scores.append(-5.0); types.append(1)  // 2
        vocab.append("c"); scores.append(-5.0); types.append(1)  // 3
        vocab.append("ab"); scores.append(-1.0); types.append(1) // 4
        vocab.append("bc"); scores.append(-1.0); types.append(1) // 5

        let tokenizer = try SentencePieceBPETokenizer(
            vocabulary: vocab,
            scores: scores,
            specialTokens: SpecialTokens(unknownTokenID: 0),
            tokenTypes: types,
            addSpacePrefix: false
        )

        // "abc": ab and bc both have score -1.0
        // Tie: leftmost pair (ab at index 0) should win
        let tokens = tokenizer.encode(text: "abc")
        #expect(tokens.contains(4)) // ab
        #expect(tokens.contains(3)) // c
        #expect(!tokens.contains(5)) // NOT bc
    }

    @Test("parseByteToken edge cases")
    func parseByteTokenEdges() {
        // Lowercase hex is valid
        #expect(SentencePieceBPETokenizer.parseByteToken("<0xff>") == 0xFF)

        // Wrong length
        #expect(SentencePieceBPETokenizer.parseByteToken("<0x0>") == nil)
        #expect(SentencePieceBPETokenizer.parseByteToken("<0x000>") == nil)

        // Missing brackets
        #expect(SentencePieceBPETokenizer.parseByteToken("0x00>") == nil)
        #expect(SentencePieceBPETokenizer.parseByteToken("<0x00") == nil)

        // Non-hex characters
        #expect(SentencePieceBPETokenizer.parseByteToken("<0xZZ>") == nil)
    }
}

// MARK: - StreamingDetokenizer Edge Cases

@Suite("StreamingDetokenizer Edge Cases")
struct StreamingDetokenizerEdgeCaseTests {

    static func makeTokenizer() throws -> MergesBPETokenizer {
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

    @Test("Newline resets segment")
    func newlineResetsSegment() throws {
        let tokenizer = try StreamingDetokenizerEdgeCaseTests.makeTokenizer()
        var detok = StreamingDetokenizer(tokenizer: tokenizer)

        // Send "a\n" then "b"
        let r1 = detok.append(token: 0x61) // a
        #expect(r1 == "a")
        let r2 = detok.append(token: 0x0A) // \n
        #expect(r2 == "\n")
        let r3 = detok.append(token: 0x62) // b
        #expect(r3 == "b")
    }

    @Test("Consecutive newlines")
    func consecutiveNewlines() throws {
        let tokenizer = try StreamingDetokenizerEdgeCaseTests.makeTokenizer()
        var detok = StreamingDetokenizer(tokenizer: tokenizer)

        let r1 = detok.append(token: 0x0A) // \n
        #expect(r1 == "\n")
        let r2 = detok.append(token: 0x0A) // \n
        #expect(r2 == "\n")
    }

    @Test("Incomplete UTF-8 returns nil then completes")
    func incompleteUTF8() throws {
        // Create a SentencePiece tokenizer where byte tokens produce incomplete UTF-8
        var vocab: [String] = ["<unk>"]
        var scores: [Float] = [-1000.0]
        var types: [Int] = [2]

        for byte in 0...255 {
            vocab.append(String(format: "<0x%02X>", byte))
            scores.append(-1000.0)
            types.append(6)
        }

        let tokenizer = try SentencePieceBPETokenizer(
            vocabulary: vocab,
            scores: scores,
            specialTokens: SpecialTokens(unknownTokenID: 0),
            tokenTypes: types,
            addSpacePrefix: false
        )

        var detok = StreamingDetokenizer(tokenizer: tokenizer)

        // Send first byte of 'é' (0xC3) - incomplete UTF-8
        let r1 = detok.append(token: 1 + 0xC3) // byte token at index 1+0xC3
        // Should return nil because decode produces U+FFFD
        #expect(r1 == nil)

        // Send second byte (0xA9) - completes 'é'
        let r2 = detok.append(token: 1 + 0xA9)
        #expect(r2 == "é")
    }

    @Test("Long stream without newlines accumulates correctly")
    func longStreamNoNewlines() throws {
        let tokenizer = try StreamingDetokenizerEdgeCaseTests.makeTokenizer()
        var detok = StreamingDetokenizer(tokenizer: tokenizer)

        var accumulated = ""
        for i in 0..<100 {
            let byte = UInt8(0x61 + (i % 26)) // a-z cycling
            if let text = detok.append(token: Int(byte)) {
                accumulated += text
            }
        }

        // Verify accumulated output matches batch decode
        let tokens = (0..<100).map { Int(UInt8(0x61 + ($0 % 26))) }
        let batchDecoded = tokenizer.decode(tokens: tokens)
        #expect(accumulated == batchDecoded)
    }

    @Test("Out-of-range token produces empty decode")
    func outOfRangeToken() throws {
        let tokenizer = try StreamingDetokenizerEdgeCaseTests.makeTokenizer()
        var detok = StreamingDetokenizer(tokenizer: tokenizer)

        // Valid token first
        let r1 = detok.append(token: 0x61)
        #expect(r1 == "a")

        // Out-of-range token - decode should still work (token is skipped)
        let r2 = detok.append(token: 999999)
        // The full decode of [0x61, 999999] is "a" (999999 skipped)
        // Previous decode was "a", so new portion is empty
        #expect(r2 == nil)
    }
}

// MARK: - BPE Algorithm Edge Cases

@Suite("BPE Algorithm Edge Cases")
struct BPEAlgorithmEdgeCaseTests {

    @Test("Overlapping merge candidates at same position")
    func overlappingMergeCandidates() throws {
        // "abcd" with merges: ab (rank 0), cd (rank 1)
        // Both can proceed independently since they don't overlap
        let aChar = String(Character(ByteEncoder.byteToUnicode[0x61]!))
        let bChar = String(Character(ByteEncoder.byteToUnicode[0x62]!))
        let cChar = String(Character(ByteEncoder.byteToUnicode[0x63]!))
        let dChar = String(Character(ByteEncoder.byteToUnicode[0x64]!))

        var vocab: [String] = []
        for byte: UInt8 in 0...255 {
            vocab.append(String(Character(ByteEncoder.byteToUnicode[byte]!)))
        }
        vocab.append(aChar + bChar)  // 256: ab
        vocab.append(cChar + dChar)  // 257: cd

        let merges = [
            "\(aChar) \(bChar)",  // rank 0
            "\(cChar) \(dChar)",  // rank 1
        ]

        let tokenizer = try MergesBPETokenizer(
            vocabulary: vocab,
            merges: merges,
            preTokenizer: .gpt2,
            specialTokens: SpecialTokens()
        )

        // "abcd" → merge ab (rank 0) → [ab, c, d] → merge cd (rank 1) → [ab, cd]
        let result = tokenizer.bpe(
            token: aChar + bChar + cChar + dChar
        )
        #expect(result == [aChar + bChar, cChar + dChar])
    }

    @Test("BPE termination with no applicable merges")
    func bpeTerminationNoMerges() throws {
        var vocab: [String] = []
        for byte: UInt8 in 0...255 {
            vocab.append(String(Character(ByteEncoder.byteToUnicode[byte]!)))
        }
        // Add merges that don't apply to our input
        let xChar = String(Character(ByteEncoder.byteToUnicode[0x78]!))
        let yChar = String(Character(ByteEncoder.byteToUnicode[0x79]!))
        vocab.append(xChar + yChar)

        let tokenizer = try MergesBPETokenizer(
            vocabulary: vocab,
            merges: ["\(xChar) \(yChar)"],
            preTokenizer: .gpt2,
            specialTokens: SpecialTokens()
        )

        let aChar = String(Character(ByteEncoder.byteToUnicode[0x61]!))
        let bChar = String(Character(ByteEncoder.byteToUnicode[0x62]!))

        // "ab" has no matching merges → stays [a, b]
        let result = tokenizer.bpe(token: aChar + bChar)
        #expect(result == [aChar, bChar])
    }

    @Test("SentencePiece merge creates new higher-scoring pair")
    func sentencePieceCascadeMerge() throws {
        var vocab: [String] = ["<unk>"]
        var scores: [Float] = [-1000.0]
        var types: [Int] = [2]

        vocab.append("a"); scores.append(-5.0); types.append(1) // 1
        vocab.append("b"); scores.append(-5.0); types.append(1) // 2
        vocab.append("c"); scores.append(-5.0); types.append(1) // 3
        vocab.append("ab"); scores.append(-2.0); types.append(1) // 4
        vocab.append("abc"); scores.append(-0.5); types.append(1) // 5

        let tokenizer = try SentencePieceBPETokenizer(
            vocabulary: vocab,
            scores: scores,
            specialTokens: SpecialTokens(unknownTokenID: 0),
            tokenTypes: types,
            addSpacePrefix: false
        )

        // "abc":
        // Pairs: (a,b) → "ab" score -2.0, (b,c) → "bc" not in vocab
        // Merge (a,b) → [ab, c]
        // Now pair (ab,c) → "abc" score -0.5
        // Merge → [abc]
        let tokens = tokenizer.encode(text: "abc")
        #expect(tokens == [5]) // abc token
    }

    @Test("SentencePiece with all characters unknown - full byte fallback")
    func allCharactersUnknown() throws {
        var vocab: [String] = ["<unk>"]
        var scores: [Float] = [-1000.0]
        var types: [Int] = [2]

        for byte in 0...255 {
            vocab.append(String(format: "<0x%02X>", byte))
            scores.append(-1000.0)
            types.append(6)
        }

        let tokenizer = try SentencePieceBPETokenizer(
            vocabulary: vocab,
            scores: scores,
            specialTokens: SpecialTokens(unknownTokenID: 0),
            tokenTypes: types,
            addSpacePrefix: false
        )

        // "Hi" - neither H nor i is in vocabulary as normal token
        // Should fall back to bytes: 0x48, 0x69
        let tokens = tokenizer.encode(text: "Hi")
        let decoded = tokenizer.decode(tokens: tokens)
        #expect(decoded == "Hi")
    }

    @Test("SentencePiece byte fallback decode handles 4-byte emoji")
    func fourByteEmojiByteFallback() throws {
        var vocab: [String] = ["<unk>"]
        var scores: [Float] = [-1000.0]
        var types: [Int] = [2]

        for byte in 0...255 {
            vocab.append(String(format: "<0x%02X>", byte))
            scores.append(-1000.0)
            types.append(6)
        }

        let tokenizer = try SentencePieceBPETokenizer(
            vocabulary: vocab,
            scores: scores,
            specialTokens: SpecialTokens(unknownTokenID: 0),
            tokenTypes: types,
            addSpacePrefix: false
        )

        let text = "🎉" // F0 9F 8E 89
        let tokens = tokenizer.encode(text: text)
        #expect(tokens.count == 4) // 4 byte tokens
        let decoded = tokenizer.decode(tokens: tokens)
        #expect(decoded == text)
    }
}

// MARK: - GGUFTokenizerFactory Edge Cases

@Suite("GGUFTokenizerFactory Edge Cases")
struct GGUFTokenizerFactoryEdgeCaseTests {

    @Test("Missing merges for gpt2 model throws")
    func missingMergesForGPT2() throws {
        let data = buildGGUFWithPartialTokenizer(model: "gpt2", includeMerges: false, includeScores: false)
        let file = try GGUFFile.parse(data: data)
        #expect(throws: TokenizerError.self) {
            _ = try GGUFTokenizerFactory.create(from: file)
        }
    }

    @Test("Missing scores for llama model throws")
    func missingScoresForLlama() throws {
        let data = buildGGUFWithPartialTokenizer(model: "llama", includeMerges: false, includeScores: false)
        let file = try GGUFFile.parse(data: data)
        #expect(throws: TokenizerError.self) {
            _ = try GGUFTokenizerFactory.create(from: file)
        }
    }

    @Test("Unknown token found by type")
    func unknownTokenByType() throws {
        let data = buildGGUFWithTokenizer(model: "gpt2")
        let file = try GGUFFile.parse(data: data)
        let tokenizer = try GGUFTokenizerFactory.create(from: file)
        // The factory should have found an unknown token (if token_type has type 2)
        // In our test data, no type 2 exists, so unknownTokenID should be nil
        // This verifies the factory doesn't crash when no unknown token exists
        #expect(tokenizer.vocabularySize > 0)
    }
}

// MARK: - Test Helpers

private func buildGGUFWithPartialTokenizer(model: String, includeMerges: Bool, includeScores: Bool) -> Data {
    var data = Data()
    appendUInt32(&data, 0x4655_4747) // Magic
    appendUInt32(&data, 3)            // Version
    appendUInt64(&data, 0)            // Tensor count

    var kvCount: UInt64 = 4 // architecture, model, tokens, token_type
    if includeMerges { kvCount += 1 }
    if includeScores { kvCount += 1 }
    appendUInt64(&data, kvCount)

    // general.architecture
    appendString(&data, "general.architecture")
    appendUInt32(&data, GGUFMetadataValueType.string.rawValue)
    appendString(&data, "llama")

    // tokenizer.ggml.model
    appendString(&data, "tokenizer.ggml.model")
    appendUInt32(&data, GGUFMetadataValueType.string.rawValue)
    appendString(&data, model)

    // tokenizer.ggml.tokens: ["a", "b"]
    appendString(&data, "tokenizer.ggml.tokens")
    appendUInt32(&data, GGUFMetadataValueType.array.rawValue)
    appendUInt32(&data, GGUFMetadataValueType.string.rawValue)
    appendUInt64(&data, 2)
    appendString(&data, "a")
    appendString(&data, "b")

    // tokenizer.ggml.token_type: [1, 1]
    appendString(&data, "tokenizer.ggml.token_type")
    appendUInt32(&data, GGUFMetadataValueType.array.rawValue)
    appendUInt32(&data, GGUFMetadataValueType.int32.rawValue)
    appendUInt64(&data, 2)
    appendInt32(&data, 1)
    appendInt32(&data, 1)

    if includeMerges {
        appendString(&data, "tokenizer.ggml.merges")
        appendUInt32(&data, GGUFMetadataValueType.array.rawValue)
        appendUInt32(&data, GGUFMetadataValueType.string.rawValue)
        appendUInt64(&data, 1)
        appendString(&data, "a b")
    }

    if includeScores {
        appendString(&data, "tokenizer.ggml.scores")
        appendUInt32(&data, GGUFMetadataValueType.array.rawValue)
        appendUInt32(&data, GGUFMetadataValueType.float32.rawValue)
        appendUInt64(&data, 2)
        appendFloat32(&data, -1.0)
        appendFloat32(&data, -1.0)
    }

    return data
}

private func buildGGUFWithTokenizer(model: String) -> Data {
    var data = Data()
    appendUInt32(&data, 0x4655_4747)
    appendUInt32(&data, 3)
    appendUInt64(&data, 0)

    var kvCount: UInt64 = 6
    if model == "gpt2" { kvCount += 1 }
    if model == "llama" { kvCount += 1 }
    appendUInt64(&data, kvCount)

    appendString(&data, "general.architecture")
    appendUInt32(&data, GGUFMetadataValueType.string.rawValue)
    appendString(&data, "llama")

    appendString(&data, "tokenizer.ggml.model")
    appendUInt32(&data, GGUFMetadataValueType.string.rawValue)
    appendString(&data, model)

    appendString(&data, "tokenizer.ggml.tokens")
    appendUInt32(&data, GGUFMetadataValueType.array.rawValue)
    appendUInt32(&data, GGUFMetadataValueType.string.rawValue)
    appendUInt64(&data, 4)
    appendString(&data, "a")
    appendString(&data, "b")
    appendString(&data, "<s>")
    appendString(&data, "</s>")

    appendString(&data, "tokenizer.ggml.token_type")
    appendUInt32(&data, GGUFMetadataValueType.array.rawValue)
    appendUInt32(&data, GGUFMetadataValueType.int32.rawValue)
    appendUInt64(&data, 4)
    appendInt32(&data, 1); appendInt32(&data, 1)
    appendInt32(&data, 3); appendInt32(&data, 3)

    appendString(&data, "tokenizer.ggml.bos_token_id")
    appendUInt32(&data, GGUFMetadataValueType.uint32.rawValue)
    appendUInt32(&data, 2)

    appendString(&data, "tokenizer.ggml.eos_token_id")
    appendUInt32(&data, GGUFMetadataValueType.uint32.rawValue)
    appendUInt32(&data, 3)

    if model == "gpt2" {
        appendString(&data, "tokenizer.ggml.merges")
        appendUInt32(&data, GGUFMetadataValueType.array.rawValue)
        appendUInt32(&data, GGUFMetadataValueType.string.rawValue)
        appendUInt64(&data, 1)
        appendString(&data, "a b")
    }
    if model == "llama" {
        appendString(&data, "tokenizer.ggml.scores")
        appendUInt32(&data, GGUFMetadataValueType.array.rawValue)
        appendUInt32(&data, GGUFMetadataValueType.float32.rawValue)
        appendUInt64(&data, 4)
        appendFloat32(&data, -1.0); appendFloat32(&data, -1.0)
        appendFloat32(&data, 0.0); appendFloat32(&data, 0.0)
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
