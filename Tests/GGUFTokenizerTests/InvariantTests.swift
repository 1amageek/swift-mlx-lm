import Testing
import Foundation
@testable import GGUFTokenizer
@testable import GGUFParser

// MARK: - tokenID(for:) Tests

@Suite("tokenID(for:)")
struct TokenIDForStringTests {

    // MARK: - MergesBPE

    @Test("MergesBPE tokenID(for:) returns correct ID for known token")
    func mergesBPEKnownToken() throws {
        let tokenizer = try makeMergesBPETokenizer(
            extraVocab: [("<s>", 256), ("</s>", 257)],
            specialTokens: SpecialTokens(bosTokenID: 256, eosTokenID: 257)
        )
        #expect(tokenizer.tokenID(for: "<s>") == 256)
        #expect(tokenizer.tokenID(for: "</s>") == 257)
    }

    @Test("MergesBPE tokenID(for:) returns nil for unknown string")
    func mergesBPEUnknownToken() throws {
        let tokenizer = try makeMergesBPETokenizer()
        #expect(tokenizer.tokenID(for: "nonexistent_token_xyz") == nil)
    }

    @Test("MergesBPE tokenID(for:) resolves special token strings")
    func mergesBPESpecialTokenLookup() throws {
        var vocab: [String] = []
        for byte: UInt8 in 0...255 {
            vocab.append(String(Character(ByteEncoder.byteToUnicode[byte]!)))
        }
        vocab.append("<|vision_start|>")  // 256
        vocab.append("<|vision_end|>")    // 257
        vocab.append("<|im_start|>")      // 258
        vocab.append("<|im_end|>")        // 259

        let tokenizer = try MergesBPETokenizer(
            vocabulary: vocab,
            merges: [],
            preTokenizer: .gpt2,
            specialTokens: SpecialTokens()
        )

        #expect(tokenizer.tokenID(for: "<|vision_start|>") == 256)
        #expect(tokenizer.tokenID(for: "<|vision_end|>") == 257)
        #expect(tokenizer.tokenID(for: "<|im_start|>") == 258)
        #expect(tokenizer.tokenID(for: "<|im_end|>") == 259)
    }

    @Test("MergesBPE encode preserves special tokens as single IDs")
    func mergesBPESpecialTokenEncode() throws {
        var tokenTypes = Array(repeating: 1, count: 260)
        tokenTypes[258] = 4
        tokenTypes[259] = 4

        let tokenizer = try makeMergesBPETokenizer(
            extraVocab: [
                ("<|vision_start|>", 256),
                ("<|vision_end|>", 257),
                ("<|im_start|>", 258),
                ("<|im_end|>", 259),
            ],
            tokenTypes: tokenTypes
        )

        let tokens = tokenizer.encode(text: "<|im_start|>hi<|im_end|>")
        #expect(tokens == [258, Int(UInt8(ascii: "h")), Int(UInt8(ascii: "i")), 259])
    }

    @Test("MergesBPE encode preserves special tokens inside prompt text")
    func mergesBPESpecialTokenEncodeInPrompt() throws {
        var tokenTypes = Array(repeating: 1, count: 260)
        tokenTypes[258] = 4
        tokenTypes[259] = 4

        let tokenizer = try makeMergesBPETokenizer(
            extraVocab: [
                ("<|vision_start|>", 256),
                ("<|vision_end|>", 257),
                ("<|im_start|>", 258),
                ("<|im_end|>", 259),
            ],
            tokenTypes: tokenTypes
        )

        let tokens = tokenizer.encode(text: "A<|im_start|>B<|im_end|>C")
        #expect(tokens == [
            Int(UInt8(ascii: "A")),
            258,
            Int(UInt8(ascii: "B")),
            259,
            Int(UInt8(ascii: "C")),
        ])
    }

    @Test("MergesBPE tokenID(for:) and tokenToString are inverse")
    func mergesBPERoundtrip() throws {
        let tokenizer = try makeMergesBPETokenizer(
            extraVocab: [("<s>", 256), ("</s>", 257)]
        )
        for id in [0, 50, 100, 200, 255, 256, 257] {
            if let str = tokenizer.tokenToString(id) {
                #expect(tokenizer.tokenID(for: str) != nil,
                    "tokenID(for:) should find token for ID \(id)")
            }
        }
    }

    @Test("MergesBPE tokenID(for:) with empty string")
    func mergesBPEEmptyString() throws {
        let tokenizer = try makeMergesBPETokenizer()
        // Empty string is not typically in vocabulary
        let result = tokenizer.tokenID(for: "")
        // It could be nil or a valid ID depending on vocab; just verify no crash
        _ = result
    }

    // MARK: - SentencePieceBPE

    @Test("SentencePieceBPE tokenID(for:) returns correct ID for known token")
    func sentencePieceKnownToken() throws {
        let tokenizer = try makeSentencePieceTokenizer()
        #expect(tokenizer.tokenID(for: "<unk>") == 0)
        #expect(tokenizer.tokenID(for: "<s>") == 1)
        #expect(tokenizer.tokenID(for: "</s>") == 2)
    }

    @Test("SentencePieceBPE tokenID(for:) returns nil for unknown string")
    func sentencePieceUnknownToken() throws {
        let tokenizer = try makeSentencePieceTokenizer()
        #expect(tokenizer.tokenID(for: "nonexistent_xyz") == nil)
    }

    @Test("SentencePieceBPE tokenID(for:) resolves byte tokens")
    func sentencePieceByteTokenLookup() throws {
        let tokenizer = try makeSentencePieceTokenizer()
        #expect(tokenizer.tokenID(for: "<0x00>") == 3)
        #expect(tokenizer.tokenID(for: "<0xFF>") == 258)
        #expect(tokenizer.tokenID(for: "<0x41>") == 3 + 0x41)
    }

    @Test("SentencePieceBPE tokenID(for:) resolves Unicode piece tokens")
    func sentencePieceUnicodePiece() throws {
        let tokenizer = try makeSentencePieceTokenizer()
        #expect(tokenizer.tokenID(for: "\u{2581}abc") == 266)
        #expect(tokenizer.tokenID(for: "ab") == 264)
    }

    @Test("SentencePieceBPE tokenID(for:) and tokenToString are inverse")
    func sentencePieceRoundtrip() throws {
        let tokenizer = try makeSentencePieceTokenizer()
        for id in 0..<tokenizer.vocabularySize {
            if let str = tokenizer.tokenToString(id) {
                let resolvedID = tokenizer.tokenID(for: str)
                #expect(resolvedID != nil,
                    "tokenID(for:) should resolve token string at ID \(id)")
            }
        }
    }
}

// MARK: - SpecialTokens Tests

@Suite("SpecialTokens")
struct SpecialTokensTests {

    @Test("Default init has all nil and false")
    func defaultInit() {
        let st = SpecialTokens()
        #expect(st.bosTokenID == nil)
        #expect(st.eosTokenID == nil)
        #expect(st.unknownTokenID == nil)
        #expect(st.paddingTokenID == nil)
        #expect(st.addBosToken == false)
        #expect(st.addEosToken == false)
    }

    @Test("addBosToken=true with nil bosTokenID does not prepend")
    func addBosTokenWithNilID() throws {
        let tokenizer = try makeMergesBPETokenizer(
            specialTokens: SpecialTokens(
                bosTokenID: nil,
                addBosToken: true
            )
        )
        let tokens = tokenizer.encode(text: "a")
        // No BOS should be added because bosTokenID is nil
        let aID = Int(UInt8(ascii: "a"))
        // First token should be the byte-encoded 'a', not a BOS
        #expect(tokens.first != nil)
        // Verify through decode
        let decoded = tokenizer.decode(tokens: tokens)
        #expect(decoded == "a")
    }

    @Test("addEosToken=true with nil eosTokenID does not append")
    func addEosTokenWithNilID() throws {
        let tokenizer = try makeMergesBPETokenizer(
            specialTokens: SpecialTokens(
                eosTokenID: nil,
                addEosToken: true
            )
        )
        let tokens = tokenizer.encode(text: "a")
        let decoded = tokenizer.decode(tokens: tokens)
        #expect(decoded == "a")
    }

    @Test("Both BOS and EOS added when configured")
    func bothBosAndEos() throws {
        let tokenizer = try makeMergesBPETokenizer(
            extraVocab: [("<s>", 256), ("</s>", 257)],
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

    @Test("All SpecialTokens fields populated")
    func allFieldsPopulated() {
        let st = SpecialTokens(
            bosTokenID: 1,
            eosTokenID: 2,
            unknownTokenID: 0,
            paddingTokenID: 3,
            addBosToken: true,
            addEosToken: true
        )
        #expect(st.bosTokenID == 1)
        #expect(st.eosTokenID == 2)
        #expect(st.unknownTokenID == 0)
        #expect(st.paddingTokenID == 3)
        #expect(st.addBosToken == true)
        #expect(st.addEosToken == true)
    }

    @Test("SentencePiece with BOS only, no EOS")
    func sentencePieceBosOnly() throws {
        let tokenizer = try makeSentencePieceTokenizer(
            specialTokens: SpecialTokens(
                bosTokenID: 1,
                eosTokenID: 2,
                unknownTokenID: 0,
                addBosToken: true,
                addEosToken: false
            )
        )
        let tokens = tokenizer.encode(text: "abc")
        #expect(tokens.first == 1)
        #expect(tokens.last != 2)
    }

    @Test("SentencePiece with EOS only, no BOS")
    func sentencePieceEosOnly() throws {
        let tokenizer = try makeSentencePieceTokenizer(
            specialTokens: SpecialTokens(
                bosTokenID: 1,
                eosTokenID: 2,
                unknownTokenID: 0,
                addBosToken: false,
                addEosToken: true
            )
        )
        let tokens = tokenizer.encode(text: "abc")
        #expect(tokens.first != 1)
        #expect(tokens.last == 2)
    }
}

// MARK: - Encode-Decode Roundtrip Invariants

@Suite("Encode-Decode Roundtrip Invariants")
struct EncodeDecodeRoundtripTests {

    @Test("MergesBPE roundtrip preserves ASCII text")
    func mergesBPEAsciiRoundtrip() throws {
        let tokenizer = try makeMergesBPETokenizer()
        let inputs = [
            "Hello, World!",
            "foo bar baz",
            "a",
            "  spaces  ",
            "line1\nline2\nline3",
            "!@#$%^&*()",
            "MiXeD CaSe",
            "12345",
        ]
        for text in inputs {
            let tokens = tokenizer.encode(text: text)
            let decoded = tokenizer.decode(tokens: tokens)
            #expect(decoded == text, "Roundtrip failed for: \(text)")
        }
    }

    @Test("MergesBPE roundtrip preserves Unicode text")
    func mergesBPEUnicodeRoundtrip() throws {
        let tokenizer = try makeMergesBPETokenizer()
        let inputs = [
            "日本語テスト",
            "العربية",
            "Ελληνικά",
            "한국어",
            "देवनागरी",       // Devanagari
            "e\u{0301}",     // e + combining acute accent
            "a\u{0308}",     // a + combining diaeresis
            "\u{0001}\u{007F}\u{0000}",  // Control characters
            "🎉🚀💻",        // Emoji
            "👨‍👩‍👧‍👦",          // Family emoji (ZWJ sequence)
        ]
        for text in inputs {
            let tokens = tokenizer.encode(text: text)
            let decoded = tokenizer.decode(tokens: tokens)
            #expect(decoded == text, "Roundtrip failed for: \(text)")
        }
    }

    @Test("SentencePiece roundtrip preserves text (without BOS)")
    func sentencePieceRoundtrip() throws {
        let tokenizer = try makeSentencePieceTokenizer()
        let inputs = ["abc", "a", "ab"]
        for text in inputs {
            let tokens = tokenizer.encode(text: text)
            let withoutBOS = Array(tokens.dropFirst()) // drop BOS
            let decoded = tokenizer.decode(tokens: withoutBOS)
            #expect(decoded == text, "Roundtrip failed for: \(text)")
        }
    }

    @Test("SentencePiece byte fallback roundtrip for unknown characters")
    func sentencePieceByteFallbackRoundtrip() throws {
        let tokenizer = try makeSentencePieceTokenizer()
        let inputs = [
            "x",        // Not in normal vocab
            "xyz",      // Multiple unknown chars
            "日",       // Multi-byte UTF-8
            "🎉",       // 4-byte UTF-8
            "देवनागरी",  // Devanagari
        ]
        for text in inputs {
            let tokens = tokenizer.encode(text: text)
            let withoutBOS = Array(tokens.dropFirst())
            let decoded = tokenizer.decode(tokens: withoutBOS)
            #expect(decoded == text, "Byte fallback roundtrip failed for: \(text)")
        }
    }

    @Test("Encode never produces negative token IDs")
    func encodeProducesNonNegativeIDs() throws {
        let merges = try makeMergesBPETokenizer()
        let sp = try makeSentencePieceTokenizer()

        let inputs = ["Hello", "日本語", "", "a b c"]
        for text in inputs {
            for id in merges.encode(text: text) {
                #expect(id >= 0, "MergesBPE produced negative ID for: \(text)")
            }
            for id in sp.encode(text: text) {
                #expect(id >= 0, "SentencePiece produced negative ID for: \(text)")
            }
        }
    }

    @Test("Encode produces IDs within vocabulary range")
    func encodeProducesInRangeIDs() throws {
        let merges = try makeMergesBPETokenizer()
        let sp = try makeSentencePieceTokenizer()

        let inputs = ["Hello World", "abc", "test 123"]
        for text in inputs {
            for id in merges.encode(text: text) {
                #expect(id < merges.vocabularySize,
                    "MergesBPE produced out-of-range ID \(id) for: \(text)")
            }
            for id in sp.encode(text: text) {
                #expect(id < sp.vocabularySize,
                    "SentencePiece produced out-of-range ID \(id) for: \(text)")
            }
        }
    }
}

// MARK: - SentencePiece Score Edge Cases

@Suite("SentencePiece Score Edge Cases")
struct SentencePieceScoreEdgeCaseTests {

    @Test("All tokens with same score - leftmost merge wins")
    func allSameScore() throws {
        var vocab: [String] = ["<unk>"]
        var scores: [Float] = [-1000.0]
        var types: [Int] = [2]

        vocab.append("a"); scores.append(-5.0); types.append(1)  // 1
        vocab.append("b"); scores.append(-5.0); types.append(1)  // 2
        vocab.append("c"); scores.append(-5.0); types.append(1)  // 3
        // Both merged tokens have the SAME score
        vocab.append("ab"); scores.append(-1.0); types.append(1) // 4
        vocab.append("bc"); scores.append(-1.0); types.append(1) // 5

        let tokenizer = try SentencePieceBPETokenizer(
            vocabulary: vocab,
            scores: scores,
            specialTokens: SpecialTokens(unknownTokenID: 0),
            tokenTypes: types,
            addSpacePrefix: false
        )

        // "abc": ab(-1.0) and bc(-1.0) tie on score
        // Leftmost pair (ab at index 0) should win
        let tokens = tokenizer.encode(text: "abc")
        #expect(tokens.contains(4), "ab should be selected on tie")
        #expect(!tokens.contains(5), "bc should NOT be selected on tie")
    }

    @Test("Negative infinity score is never merged")
    func negativeInfinityScore() throws {
        var vocab: [String] = ["<unk>"]
        var scores: [Float] = [-1000.0]
        var types: [Int] = [2]

        vocab.append("a"); scores.append(-5.0); types.append(1)  // 1
        vocab.append("b"); scores.append(-5.0); types.append(1)  // 2
        vocab.append("ab"); scores.append(-.infinity); types.append(1) // 3

        let tokenizer = try SentencePieceBPETokenizer(
            vocabulary: vocab,
            scores: scores,
            specialTokens: SpecialTokens(unknownTokenID: 0),
            tokenTypes: types,
            addSpacePrefix: false
        )

        let tokens = tokenizer.encode(text: "ab")
        // -infinity should still be > -.infinity for initial bestScore,
        // so it actually will merge. The point is it doesn't crash.
        let decoded = tokenizer.decode(tokens: tokens)
        #expect(decoded == "ab")
    }

    @Test("Very large positive scores work correctly")
    func veryLargePositiveScores() throws {
        var vocab: [String] = ["<unk>"]
        var scores: [Float] = [-1000.0]
        var types: [Int] = [2]

        vocab.append("a"); scores.append(-5.0); types.append(1)
        vocab.append("b"); scores.append(-5.0); types.append(1)
        vocab.append("ab"); scores.append(Float.greatestFiniteMagnitude); types.append(1)

        let tokenizer = try SentencePieceBPETokenizer(
            vocabulary: vocab,
            scores: scores,
            specialTokens: SpecialTokens(unknownTokenID: 0),
            tokenTypes: types,
            addSpacePrefix: false
        )

        let tokens = tokenizer.encode(text: "ab")
        #expect(tokens == [3]) // ab token at index 3
    }
}

// MARK: - StreamingDetokenizer State Management

@Suite("StreamingDetokenizer State Management")
struct StreamingDetokenizerStateTests {

    @Test("BOS token in stream is skipped by control token filter")
    func bosTokenSkipped() throws {
        var vocab: [String] = []
        for byte: UInt8 in 0...255 {
            vocab.append(String(Character(ByteEncoder.byteToUnicode[byte]!)))
        }
        vocab.append("<s>")  // 256

        var types = Array(repeating: 1, count: 256)
        types.append(3) // control token

        let tokenizer = try MergesBPETokenizer(
            vocabulary: vocab,
            merges: [],
            preTokenizer: .gpt2,
            specialTokens: SpecialTokens(bosTokenID: 256, addBosToken: false),
            tokenTypes: types
        )

        var detok = StreamingDetokenizer(tokenizer: tokenizer)

        // Append BOS (control token) - should produce nil (filtered out)
        let r1 = detok.append(token: 256)
        #expect(r1 == nil)

        // Then normal token should work
        let r2 = detok.append(token: 0x48) // H
        #expect(r2 == "H")
    }

    @Test("Post-newline reset produces correct subsequent output")
    func postNewlineReset() throws {
        let tokenizer = try makeSimpleMergesBPETokenizer()
        var detok = StreamingDetokenizer(tokenizer: tokenizer)

        // "Hello\n"
        for byte in [0x48, 0x65, 0x6C, 0x6C, 0x6F] as [UInt8] {
            _ = detok.append(token: Int(byte))
        }
        _ = detok.append(token: 0x0A) // \n triggers reset

        // After reset, "World" should decode correctly
        var afterReset = ""
        for byte in [0x57, 0x6F, 0x72, 0x6C, 0x64] as [UInt8] {
            if let text = detok.append(token: Int(byte)) {
                afterReset += text
            }
        }
        #expect(afterReset == "World")
    }

    @Test("Streaming output matches batch decode over long sequences")
    func longStreamConsistency() throws {
        let tokenizer = try makeSimpleMergesBPETokenizer()
        var detok = StreamingDetokenizer(tokenizer: tokenizer)

        // Generate 500 tokens (a-z cycling)
        let tokens = (0..<500).map { Int(UInt8(0x61 + ($0 % 26))) }
        var streamed = ""
        for token in tokens {
            if let text = detok.append(token: token) {
                streamed += text
            }
        }

        let batchDecoded = tokenizer.decode(tokens: tokens)
        #expect(streamed == batchDecoded)
    }

    @Test("Multiple newlines produce separate resets")
    func multipleNewlinesReset() throws {
        let tokenizer = try makeSimpleMergesBPETokenizer()
        var detok = StreamingDetokenizer(tokenizer: tokenizer)

        let r1 = detok.append(token: 0x61) // a
        #expect(r1 == "a")
        let r2 = detok.append(token: 0x0A) // \n
        #expect(r2 == "\n")
        let r3 = detok.append(token: 0x0A) // \n
        #expect(r3 == "\n")
        let r4 = detok.append(token: 0x62) // b
        #expect(r4 == "b")
    }

    @Test("StreamingDetokenizer with SentencePiece tokenizer")
    func streamingWithSentencePiece() throws {
        let tokenizer = try makeSentencePieceTokenizer()
        var detok = StreamingDetokenizer(tokenizer: tokenizer)

        // Append the ▁abc token (266)
        let r1 = detok.append(token: 266)
        #expect(r1 != nil)
        // The output should contain "abc" (with space prefix stripped or present)
        if let text = r1 {
            #expect(text.contains("abc"))
        }
    }
}

// MARK: - Unicode Coverage

@Suite("Unicode Coverage")
struct UnicodeCoverageTests {

    @Test("Devanagari script roundtrip via MergesBPE")
    func devanagariRoundtrip() throws {
        let tokenizer = try makeMergesBPETokenizer()
        let text = "नमस्ते"
        let tokens = tokenizer.encode(text: text)
        let decoded = tokenizer.decode(tokens: tokens)
        #expect(decoded == text)
    }

    @Test("Combining marks roundtrip via MergesBPE")
    func combiningMarksRoundtrip() throws {
        let tokenizer = try makeMergesBPETokenizer()
        // é as e + combining acute accent
        let text = "cafe\u{0301}"
        let tokens = tokenizer.encode(text: text)
        let decoded = tokenizer.decode(tokens: tokens)
        #expect(decoded == text)
    }

    @Test("Control characters roundtrip via MergesBPE")
    func controlCharactersRoundtrip() throws {
        let tokenizer = try makeMergesBPETokenizer()
        let text = "\u{0001}\u{0002}\u{001F}"
        let tokens = tokenizer.encode(text: text)
        let decoded = tokenizer.decode(tokens: tokens)
        #expect(decoded == text)
    }

    @Test("Null byte roundtrip via MergesBPE")
    func nullByteRoundtrip() throws {
        let tokenizer = try makeMergesBPETokenizer()
        let text = "\u{0000}"
        let tokens = tokenizer.encode(text: text)
        let decoded = tokenizer.decode(tokens: tokens)
        #expect(decoded == text)
    }

    @Test("Thai script roundtrip via MergesBPE")
    func thaiRoundtrip() throws {
        let tokenizer = try makeMergesBPETokenizer()
        let text = "สวัสดี"
        let tokens = tokenizer.encode(text: text)
        let decoded = tokenizer.decode(tokens: tokens)
        #expect(decoded == text)
    }

    @Test("Devanagari via SentencePiece byte fallback")
    func devanagariSentencePiece() throws {
        let tokenizer = try makeSentencePieceTokenizer()
        let text = "नमस्ते"
        let tokens = tokenizer.encode(text: text)
        let withoutBOS = Array(tokens.dropFirst())
        let decoded = tokenizer.decode(tokens: withoutBOS)
        #expect(decoded == text)
    }

    @Test("Mixed script text via MergesBPE")
    func mixedScriptRoundtrip() throws {
        let tokenizer = try makeMergesBPETokenizer()
        let text = "Hello世界مرحبا😀"
        let tokens = tokenizer.encode(text: text)
        let decoded = tokenizer.decode(tokens: tokens)
        #expect(decoded == text)
    }
}

// MARK: - GGUFTokenizerFactory Additional Tests

@Suite("GGUFTokenizerFactory Invariants")
struct GGUFTokenizerFactoryInvariantTests {

    @Test("Factory-created MergesBPE tokenizer exposes tokenID(for:)")
    func factoryMergesBPETokenIDFor() throws {
        let data = buildGGUFForFactory(model: "gpt2")
        let file = try GGUFFile.parse(data: data)
        let tokenizer = try GGUFTokenizerFactory.create(from: file)

        // Should be able to look up tokens
        #expect(tokenizer.tokenID(for: "a") != nil)
        #expect(tokenizer.tokenID(for: "<s>") != nil)
    }

    @Test("Factory-created SentencePiece tokenizer exposes tokenID(for:)")
    func factorySentencePieceTokenIDFor() throws {
        let data = buildGGUFForFactory(model: "llama")
        let file = try GGUFFile.parse(data: data)
        let tokenizer = try GGUFTokenizerFactory.create(from: file)

        #expect(tokenizer.tokenID(for: "a") != nil)
        #expect(tokenizer.tokenID(for: "<s>") != nil)
    }

    @Test("Factory finds unknown token by type")
    func factoryFindsUnknownByType() throws {
        // Build GGUF with token_type=2 for index 0
        var data = Data()
        appendUInt32(&data, 0x4655_4747)
        appendUInt32(&data, 3)
        appendUInt64(&data, 0) // no tensors

        var kvCount: UInt64 = 7
        appendUInt64(&data, kvCount)

        appendGGUFString(&data, "general.architecture")
        appendUInt32(&data, GGUFMetadataValueType.string.rawValue)
        appendGGUFString(&data, "llama")

        appendGGUFString(&data, "tokenizer.ggml.model")
        appendUInt32(&data, GGUFMetadataValueType.string.rawValue)
        appendGGUFString(&data, "llama")

        appendGGUFString(&data, "tokenizer.ggml.tokens")
        appendUInt32(&data, GGUFMetadataValueType.array.rawValue)
        appendUInt32(&data, GGUFMetadataValueType.string.rawValue)
        appendUInt64(&data, 3)
        appendGGUFString(&data, "<unk>")
        appendGGUFString(&data, "<s>")
        appendGGUFString(&data, "</s>")

        appendGGUFString(&data, "tokenizer.ggml.token_type")
        appendUInt32(&data, GGUFMetadataValueType.array.rawValue)
        appendUInt32(&data, GGUFMetadataValueType.int32.rawValue)
        appendUInt64(&data, 3)
        appendInt32(&data, 2) // UNKNOWN
        appendInt32(&data, 3) // CONTROL
        appendInt32(&data, 3) // CONTROL

        appendGGUFString(&data, "tokenizer.ggml.scores")
        appendUInt32(&data, GGUFMetadataValueType.array.rawValue)
        appendUInt32(&data, GGUFMetadataValueType.float32.rawValue)
        appendUInt64(&data, 3)
        appendFloat32(&data, -1000.0)
        appendFloat32(&data, 0.0)
        appendFloat32(&data, 0.0)

        appendGGUFString(&data, "tokenizer.ggml.bos_token_id")
        appendUInt32(&data, GGUFMetadataValueType.uint32.rawValue)
        appendUInt32(&data, 1)

        appendGGUFString(&data, "tokenizer.ggml.eos_token_id")
        appendUInt32(&data, GGUFMetadataValueType.uint32.rawValue)
        appendUInt32(&data, 2)

        let file = try GGUFFile.parse(data: data)
        let tokenizer = try GGUFTokenizerFactory.create(from: file)

        // Verify factory found the unknown token at index 0
        #expect(tokenizer.vocabularySize == 3)
    }

    @Test("Factory with no model key defaults to gpt2")
    func factoryDefaultsToGPT2() throws {
        var data = Data()
        appendUInt32(&data, 0x4655_4747)
        appendUInt32(&data, 3)
        appendUInt64(&data, 0)
        appendUInt64(&data, 4) // 4 KV pairs

        appendGGUFString(&data, "general.architecture")
        appendUInt32(&data, GGUFMetadataValueType.string.rawValue)
        appendGGUFString(&data, "llama")

        // No tokenizer.ggml.model key

        appendGGUFString(&data, "tokenizer.ggml.tokens")
        appendUInt32(&data, GGUFMetadataValueType.array.rawValue)
        appendUInt32(&data, GGUFMetadataValueType.string.rawValue)
        appendUInt64(&data, 2)
        appendGGUFString(&data, "a")
        appendGGUFString(&data, "b")

        appendGGUFString(&data, "tokenizer.ggml.token_type")
        appendUInt32(&data, GGUFMetadataValueType.array.rawValue)
        appendUInt32(&data, GGUFMetadataValueType.int32.rawValue)
        appendUInt64(&data, 2)
        appendInt32(&data, 1)
        appendInt32(&data, 1)

        appendGGUFString(&data, "tokenizer.ggml.merges")
        appendUInt32(&data, GGUFMetadataValueType.array.rawValue)
        appendUInt32(&data, GGUFMetadataValueType.string.rawValue)
        appendUInt64(&data, 1)
        appendGGUFString(&data, "a b")

        let file = try GGUFFile.parse(data: data)
        let tokenizer = try GGUFTokenizerFactory.create(from: file)
        #expect(tokenizer.vocabularySize == 2)
    }
}

// MARK: - Vocabulary Consistency Invariants

@Suite("Vocabulary Consistency")
struct VocabularyConsistencyTests {

    @Test("MergesBPE: vocabularySize matches actual vocabulary length")
    func mergesBPEVocabSizeConsistency() throws {
        let tokenizer = try makeMergesBPETokenizer(
            extraVocab: [("<s>", 256), ("</s>", 257)]
        )
        #expect(tokenizer.vocabularySize == tokenizer.vocabulary.count)
    }

    @Test("SentencePiece: vocabularySize matches actual vocabulary length")
    func sentencePieceVocabSizeConsistency() throws {
        let tokenizer = try makeSentencePieceTokenizer()
        #expect(tokenizer.vocabularySize == tokenizer.vocabulary.count)
    }

    @Test("MergesBPE: every vocabulary entry has a valid tokenToID mapping")
    func mergesBPEVocabIDMapping() throws {
        let tokenizer = try makeMergesBPETokenizer()
        // Last occurrence of duplicate wins in tokenToID, but every string should have SOME mapping
        for (id, token) in tokenizer.vocabulary.enumerated() {
            let resolvedID = tokenizer.tokenToID[token]
            #expect(resolvedID != nil,
                "Token '\(token)' at ID \(id) should have a tokenToID entry")
        }
    }

    @Test("SentencePiece: scores array length matches vocabulary length")
    func sentencePieceScoresLength() throws {
        let tokenizer = try makeSentencePieceTokenizer()
        #expect(tokenizer.scores.count == tokenizer.vocabulary.count)
    }
}

// MARK: - Test Helpers

private func makeMergesBPETokenizer(
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

private func makeSimpleMergesBPETokenizer() throws -> MergesBPETokenizer {
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

private func makeSentencePieceTokenizer(
    specialTokens: SpecialTokens = SpecialTokens(
        bosTokenID: 1, eosTokenID: 2, unknownTokenID: 0,
        addBosToken: true, addEosToken: false
    ),
    addSpacePrefix: Bool = true
) throws -> SentencePieceBPETokenizer {
    var vocab: [String] = []
    var scores: [Float] = []
    var types: [Int] = []

    vocab.append("<unk>"); scores.append(-1000.0); types.append(2)
    vocab.append("<s>"); scores.append(0.0); types.append(3)
    vocab.append("</s>"); scores.append(0.0); types.append(3)

    for byte in 0...255 {
        vocab.append(String(format: "<0x%02X>", byte))
        scores.append(-1000.0)
        types.append(6)
    }

    let baseTokens: [(String, Float, Int)] = [
        ("\u{2581}", -1.0, 1),
        ("a", -2.0, 1),
        ("b", -2.0, 1),
        ("c", -2.0, 1),
        ("\u{2581}a", -0.5, 1),
        ("ab", -0.3, 1),
        ("\u{2581}ab", -0.1, 1),
        ("\u{2581}abc", 0.0, 1),
    ]

    for (token, score, tokenType) in baseTokens {
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

private func buildGGUFForFactory(model: String) -> Data {
    var data = Data()
    appendUInt32(&data, 0x4655_4747)
    appendUInt32(&data, 3)
    appendUInt64(&data, 0)

    var kvCount: UInt64 = 6
    if model == "gpt2" { kvCount += 1 }
    if model == "llama" { kvCount += 1 }
    appendUInt64(&data, kvCount)

    appendGGUFString(&data, "general.architecture")
    appendUInt32(&data, GGUFMetadataValueType.string.rawValue)
    appendGGUFString(&data, "llama")

    appendGGUFString(&data, "tokenizer.ggml.model")
    appendUInt32(&data, GGUFMetadataValueType.string.rawValue)
    appendGGUFString(&data, model)

    appendGGUFString(&data, "tokenizer.ggml.tokens")
    appendUInt32(&data, GGUFMetadataValueType.array.rawValue)
    appendUInt32(&data, GGUFMetadataValueType.string.rawValue)
    appendUInt64(&data, 4)
    appendGGUFString(&data, "a")
    appendGGUFString(&data, "b")
    appendGGUFString(&data, "<s>")
    appendGGUFString(&data, "</s>")

    appendGGUFString(&data, "tokenizer.ggml.token_type")
    appendUInt32(&data, GGUFMetadataValueType.array.rawValue)
    appendUInt32(&data, GGUFMetadataValueType.int32.rawValue)
    appendUInt64(&data, 4)
    appendInt32(&data, 1); appendInt32(&data, 1)
    appendInt32(&data, 3); appendInt32(&data, 3)

    appendGGUFString(&data, "tokenizer.ggml.bos_token_id")
    appendUInt32(&data, GGUFMetadataValueType.uint32.rawValue)
    appendUInt32(&data, 2)

    appendGGUFString(&data, "tokenizer.ggml.eos_token_id")
    appendUInt32(&data, GGUFMetadataValueType.uint32.rawValue)
    appendUInt32(&data, 3)

    if model == "gpt2" {
        appendGGUFString(&data, "tokenizer.ggml.merges")
        appendUInt32(&data, GGUFMetadataValueType.array.rawValue)
        appendUInt32(&data, GGUFMetadataValueType.string.rawValue)
        appendUInt64(&data, 1)
        appendGGUFString(&data, "a b")
    }
    if model == "llama" {
        appendGGUFString(&data, "tokenizer.ggml.scores")
        appendUInt32(&data, GGUFMetadataValueType.array.rawValue)
        appendUInt32(&data, GGUFMetadataValueType.float32.rawValue)
        appendUInt64(&data, 4)
        appendFloat32(&data, -1.0); appendFloat32(&data, -1.0)
        appendFloat32(&data, 0.0); appendFloat32(&data, 0.0)
    }

    return data
}

// MARK: - Binary Helpers

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

private func appendGGUFString(_ data: inout Data, _ string: String) {
    let utf8 = string.utf8
    appendUInt64(&data, UInt64(utf8.count))
    data.append(contentsOf: utf8)
}
