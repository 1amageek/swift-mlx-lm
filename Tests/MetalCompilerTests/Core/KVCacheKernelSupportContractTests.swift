import Testing
@testable import MetalCompiler

/// Pins the set of `QuantizationSchemeIdentifier` values that the FlashAttention
/// KV cache MSL kernels actually read/write, and pins the precondition that
/// protects sizing-capable-but-kernel-unsupported schemes from silently
/// corrupting the cache.
///
/// `KVCacheSpecification` can size a buffer for any registered format, but
/// only a subset has matching kernel paths in
/// `MetalSourceGenerator+Attention`. Mis-classifying this set is a
/// silent-fallback bug: sizing succeeds, the FlashAttention kernel reads a
/// scheme it was never taught, and the cache fills with garbage.
@Suite("KV Cache Kernel Support Contract")
struct KVCacheKernelSupportContractTests {

    /// Schemes whose write+read MSL paths are implemented today.
    /// Expanding this set requires corresponding kernel work in
    /// `MetalSourceGenerator+Attention` (dispatcher + block-layout helpers).
    static let supportedSchemes: Set<QuantizationSchemeIdentifier> = [
        .fp16RowMajor, .bf16RowMajor, .fp32RowMajor,
        .q4Group64ScaleF16, .q4Group128ScaleF16, .q4Group128ScaleF16Zero,
        .q8Group32ScaleF16, .q8Group64ScaleF16, .q8Group128ScaleF16,
        .rotorQ4Group64ScaleF16, .rotorQ8Group32ScaleF16,
    ]

    /// Schemes registered as `QuantizationFormat` (so sizing works) but with
    /// no matching KV kernel path. Referencing these from an `InferencePolicy`
    /// must trap at `KVCacheSpecification.init` rather than silently bleeding
    /// into the cache.
    static let sizableButUnsupportedSchemes: Set<QuantizationSchemeIdentifier> = [
        .q2Group16ScaleF16, .q2Group32ScaleF16,
        .q3Group16ScaleF16, .q3Group32ScaleF16, .q3Group64ScaleF16,
        .q5Group32ScaleF16, .q5Group64ScaleF16,
        .q6Group16ScaleF16, .q6Group32ScaleF16,
        .passthrough,
    ]

    // MARK: - Flag agreement

    @Test("isSupportedForKVCache matches the pinned supported set")
    func flagAgreesWithPinnedSet() {
        for scheme in QuantizationSchemeIdentifier.allCases {
            let expected = Self.supportedSchemes.contains(scheme)
            #expect(
                scheme.isSupportedForKVCache == expected,
                "Scheme \(scheme): isSupportedForKVCache=\(scheme.isSupportedForKVCache) but pinned=\(expected). If you changed kernel support, update the pinned set in this test and the 'Supported' list in isSupportedForKVCache docs in lockstep."
            )
        }
    }

    @Test("Every scheme falls in exactly one kernel-support bucket")
    func classificationCoversAllSchemes() {
        let all = Set(QuantizationSchemeIdentifier.allCases)
        let supported = Self.supportedSchemes
        let unsupported = Self.sizableButUnsupportedSchemes
        let union = supported.union(unsupported)
        #expect(
            union == all,
            "Every scheme must be classified as kernel-supported or not. Missing: \(all.subtracting(union))"
        )
        #expect(
            supported.isDisjoint(with: unsupported),
            "Supported and sizable-but-unsupported must not overlap"
        )
    }

    // MARK: - Sizing vs kernel-support separation

    @Test("Sizable-but-unsupported schemes still produce a valid head-slot size")
    func unsupportedSchemesStillSize() {
        // Use a dense-only spec to avoid tripping the KV kernel precondition,
        // then query sizing for the unsupported scheme through the scheme-keyed
        // accessor. Sizing is protocol-driven and must work for every
        // registered scheme — that is what makes the kernel-support flag
        // load-bearing.
        let spec = KVCacheSpecification(
            keyQuantizationScheme: .fp16RowMajor,
            valueQuantizationScheme: .fp16RowMajor,
            layoutMode: .sequenceMajor,
            layerCount: 1,
            kvHeadCount: 1,
            headDimension: 128,
            maximumSequenceLength: 1
        )
        for scheme in Self.sizableButUnsupportedSchemes where scheme != .passthrough {
            let bytes = spec.bytesPerHeadSlot(scheme: scheme)
            #expect(
                bytes > 0,
                "Sizable scheme \(scheme) must produce a positive head-slot size even though the KV kernel does not implement it"
            )
        }
    }

    // MARK: - Rotor schemes delegate to base

    @Test("Rotor schemes are kernel-supported iff their base scheme is")
    func rotorSupportTracksBase() {
        for scheme in QuantizationSchemeIdentifier.allCases where scheme.isRotorScheme {
            #expect(
                scheme.isSupportedForKVCache == scheme.baseScheme.isSupportedForKVCache,
                "Rotor scheme \(scheme) must track its base scheme \(scheme.baseScheme) for KV kernel support (rotor is a pre-rotation layer over the base block layout)"
            )
        }
    }

    // MARK: - Construction guard

    @Test("KVCacheSpecification accepts every kernel-supported scheme")
    func specAcceptsSupportedSchemes() {
        for scheme in Self.supportedSchemes {
            let spec = KVCacheSpecification(
                keyQuantizationScheme: scheme,
                valueQuantizationScheme: scheme,
                layoutMode: .sequenceMajor,
                layerCount: 1,
                kvHeadCount: 1,
                headDimension: 128,
                maximumSequenceLength: 1
            )
            #expect(
                spec.bytesPerHeadSlot(scheme: scheme) > 0,
                "Spec built from supported scheme \(scheme) must report positive head-slot bytes"
            )
        }
    }
}
