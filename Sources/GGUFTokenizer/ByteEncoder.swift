/// GPT-2 style byte-level encoding.
///
/// Maps each byte value (0-255) to a unique Unicode character, ensuring
/// all byte sequences can be represented as valid Unicode strings.
/// This is used by merges-based BPE tokenizers (GPT-2, Llama 3, Qwen2).
///
/// The mapping:
/// - Bytes 0x21-0x7E (printable ASCII): map to same codepoint
/// - Bytes 0xA1-0xAC, 0xAE-0xFF (Latin-1 Supplement): map to same codepoint
/// - Remaining 68 bytes (0x00-0x20, 0x7F-0xA0, 0xAD): map to U+0100 onwards
enum ByteEncoder: Sendable {

    /// Byte (0-255) to Unicode scalar mapping. All 256 entries are unique.
    static let byteToUnicode: [UInt8: Unicode.Scalar] = {
        var map: [UInt8: Unicode.Scalar] = [:]
        map.reserveCapacity(256)

        // Printable ASCII: 0x21 '!' through 0x7E '~'
        for byte: UInt8 in 0x21...0x7E {
            map[byte] = Unicode.Scalar(UInt32(byte))!
        }
        // Latin-1 Supplement: 0xA1 through 0xAC
        for byte: UInt8 in 0xA1...0xAC {
            map[byte] = Unicode.Scalar(UInt32(byte))!
        }
        // Latin-1 Supplement: 0xAE through 0xFF
        for byte: UInt8 in 0xAE...0xFF {
            map[byte] = Unicode.Scalar(UInt32(byte))!
        }

        // Remaining bytes -> U+0100 onwards
        var n: UInt32 = 0
        for i in 0...255 {
            let byte = UInt8(i)
            if map[byte] == nil {
                map[byte] = Unicode.Scalar(256 + n)!
                n += 1
            }
        }
        return map
    }()

    /// Unicode scalar to byte mapping (reverse of byteToUnicode).
    static let unicodeToByte: [Unicode.Scalar: UInt8] = {
        var map: [Unicode.Scalar: UInt8] = [:]
        map.reserveCapacity(256)
        for (byte, scalar) in byteToUnicode {
            map[scalar] = byte
        }
        return map
    }()

    /// Encode raw bytes to GPT-2 unicode string.
    static func encode(_ bytes: some Sequence<UInt8>) -> String {
        String(bytes.map { Character(byteToUnicode[$0]!) })
    }

    /// Decode GPT-2 unicode string back to raw bytes.
    /// Scalars not in the mapping are dropped.
    static func decode(_ string: String) -> [UInt8] {
        string.unicodeScalars.compactMap { unicodeToByte[$0] }
    }
}
