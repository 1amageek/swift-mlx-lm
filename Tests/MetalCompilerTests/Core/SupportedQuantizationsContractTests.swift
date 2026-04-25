import Testing
@testable import MetalCompiler

/// Pins the contract defined in `docs/design/supported-quantizations.md`:
/// which `QuantizationSchemeIdentifier` values resolve to a concrete
/// `QuantizationFormat`, and which are declared-only placeholders.
///
/// If `QuantizationFormatRegistry.format(for:)` gains or loses entries,
/// this suite fails — the expectation is that the design document is
/// updated in the same change.
@Suite("Supported Quantizations Contract")
struct SupportedQuantizationsContractTests {

    /// Schemes that must resolve to a concrete `QuantizationFormat`.
    /// Keep in sync with the "Weight Formats" / "KV Cache Schemes" tables in
    /// `docs/design/supported-quantizations.md`.
    static let registeredSchemes: Set<QuantizationSchemeIdentifier> = [
        .fp16RowMajor,
        .bf16RowMajor,
        .fp32RowMajor,
        .q2Group16ScaleF16,
        .q2Group32ScaleF16,
        .q3Group16ScaleF16,
        .q3Group32ScaleF16,
        .q3Group64ScaleF16,
        .q4Group64ScaleF16,
        .q4Group128ScaleF16,
        .q4Group128ScaleF16Zero,
        .q5Group32ScaleF16,
        .q5Group64ScaleF16,
        .q6Group16ScaleF16,
        .q6Group32ScaleF16,
        .q8Group32ScaleF16,
        .q8Group64ScaleF16,
        .q8Group128ScaleF16,
        .passthrough,
    ]

    /// Schemes that are declared in the enum but intentionally have no
    /// runtime mapping today. Attempting to load one must fail explicitly
    /// rather than silently falling back.
    static let declaredButUnsupportedSchemes: Set<QuantizationSchemeIdentifier> = []

    /// KV-cache-only schemes. These are weight formats too in principle but
    /// the registry intentionally does not map them to GEMV kernels — rotor
    /// schemes only apply to KV cache entries.
    static let kvCacheOnlySchemes: Set<QuantizationSchemeIdentifier> = [
        .rotorQ8Group32ScaleF16,
        .rotorQ4Group64ScaleF16,
    ]

    @Test("Every scheme identifier is classified exactly once")
    func everySchemeClassifiedOnce() {
        let all = Set(QuantizationSchemeIdentifier.allCases)
        let registered = Self.registeredSchemes
        let declared = Self.declaredButUnsupportedSchemes
        let kvOnly = Self.kvCacheOnlySchemes

        let union = registered.union(declared).union(kvOnly)
        #expect(union == all,
            "Every scheme must be classified. Missing: \(all.subtracting(union))")

        #expect(registered.isDisjoint(with: declared),
            "registered and declared-but-unsupported must not overlap")
        #expect(registered.isDisjoint(with: kvOnly),
            "registered (weight) and kvCacheOnly must not overlap")
        #expect(declared.isDisjoint(with: kvOnly),
            "declared-but-unsupported and kvCacheOnly must not overlap")
    }

    @Test("Registered schemes resolve to a concrete QuantizationFormat")
    func registeredSchemesResolve() {
        for scheme in Self.registeredSchemes {
            #expect(QuantizationFormatRegistry.format(for: scheme) != nil,
                "Scheme \(scheme) must resolve to a QuantizationFormat")
        }
    }

    @Test("Declared-but-unsupported schemes do not resolve")
    func declaredSchemesDoNotResolve() {
        for scheme in Self.declaredButUnsupportedSchemes {
            #expect(QuantizationFormatRegistry.format(for: scheme) == nil,
                "Scheme \(scheme) must not resolve — it is declared-only. If you registered it, update docs/design/supported-quantizations.md and move it to registeredSchemes in this test.")
        }
    }

    @Test("KV-cache-only schemes do not resolve as weight formats")
    func kvCacheSchemesDoNotResolveAsWeight() {
        for scheme in Self.kvCacheOnlySchemes {
            #expect(QuantizationFormatRegistry.format(for: scheme) == nil,
                "Rotor scheme \(scheme) is KV-cache-only and must not have a weight GEMV kernel")
        }
    }

    @Test("Registered schemes advertise consistent block geometry")
    func registeredSchemesAdvertiseBlockGeometry() throws {
        for scheme in Self.registeredSchemes {
            let format = try #require(QuantizationFormatRegistry.format(for: scheme))
            #expect(format.bits >= 2,
                "Registered scheme \(scheme) has bits=\(format.bits); sub-2-bit formats are not supported")
            if format.bits >= 16 {
                // Dense formats: groupSize is not meaningful; block struct is empty
                #expect(format.blockStructName.isEmpty,
                    "Dense scheme \(scheme) must not declare a block struct name")
            } else {
                // Block-quantized: group size must be positive
                #expect(format.groupSize > 0,
                    "Quantized scheme \(scheme) must declare positive groupSize")
                #expect(!format.blockStructName.isEmpty,
                    "Quantized scheme \(scheme) must declare a block struct name")
            }
        }
    }

    @Test("Rotor schemes are marked as rotor-bearing")
    func rotorSchemesFlagged() {
        for scheme in Self.kvCacheOnlySchemes {
            #expect(scheme.isRotorScheme,
                "Scheme \(scheme) must report isRotorScheme == true")
        }
        for scheme in Self.registeredSchemes {
            #expect(!scheme.isRotorScheme,
                "Weight-format scheme \(scheme) must not report isRotorScheme == true")
        }
    }
}
