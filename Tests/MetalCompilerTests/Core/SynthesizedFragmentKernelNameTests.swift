import Testing
import Foundation
@testable import MetalCompiler

/// Regression tests for `SynthesizedFragment.kernelName`.
///
/// Background (2026-04-17 LFM2 regression):
/// Before the composition-tag fix, two distinct fusions with the same fragment count
/// and port count collapsed to identical kernel names — e.g. `CopyFragment + Reduction`
/// and `ResidualAddFragment + Reduction` both hashed to
/// `synthesized_2way_4p_<parallelism>_<precision>_<weight>`. The pipeline cache
/// (`SharedPipelineCache`) and kernel source catalog (`MetalKernelSourceCatalog.generatedNames`)
/// deduplicate by name, so only the first MSL body actually got compiled. Every later
/// dispatch that *should* have run the second composition instead executed the first
/// one's source, silently corrupting decode output.
///
/// These tests pin the invariant that **distinct fragment compositions must produce
/// distinct kernel names** so the cache cannot alias two different MSL bodies.
@Suite("SynthesizedFragment Kernel Name")
struct SynthesizedFragmentKernelNameTests {

    // MARK: - Helpers

    /// Build a SynthesizedFragment for a two-fragment composition using real
    /// FusionSynthesizer contract merging. This mirrors what
    /// `MetalEntryCollector.fuseCrossComponent` does at compile time.
    private func makeSynthesized(
        _ a: any PrimitiveMetalKernelFragment,
        _ b: any PrimitiveMetalKernelFragment,
        bufferPrecision: BufferPrecision = .float32,
        weightFormat: WeightFormat = .bfloat16
    ) throws -> SynthesizedFragment {
        let aContract = try #require(a.fusionContract)
        let bContract = try #require(b.fusionContract)
        let aBody = try #require(a.kernelBody(bufferPrecision: bufferPrecision, weightFormat: weightFormat))
        let bBody = try #require(b.kernelBody(bufferPrecision: bufferPrecision, weightFormat: weightFormat))

        let entries: [FusionSynthesizer.Entry] = [
            .init(contract: aContract, body: aBody),
            .init(contract: bContract, body: bBody),
        ]
        let resolvedParallelism = aContract.parallelism.resolved(with: bContract.parallelism)
        let merged = FusionSynthesizer.mergeContracts(
            entries: entries,
            resolvedParallelism: resolvedParallelism
        )
        return SynthesizedFragment(fragments: [a, b], mergedContract: merged)
    }

    private static let dimension = 2048
    private static let prefillContext = KernelContext(
        bufferPrecision: .float32,
        weightFormat: .bfloat16
    )

    // MARK: - Core Regression

    @Test("Copy+Reduction and ResidualAdd+Reduction produce distinct kernel names")
    func copyReductionAndResidualAddReductionDiffer() throws {
        let copyPlusNorm = try makeSynthesized(
            CopyFragment(dimension: Self.dimension),
            Reduction(dimension: Self.dimension, epsilon: 1e-6)
        )
        let residualAddPlusNorm = try makeSynthesized(
            ResidualAddFragment(dimension: Self.dimension),
            Reduction(dimension: Self.dimension, epsilon: 1e-6)
        )

        let nameA = copyPlusNorm.kernelName(context: Self.prefillContext)
        let nameB = residualAddPlusNorm.kernelName(context: Self.prefillContext)

        #expect(nameA != nameB,
            "Distinct compositions must produce distinct kernel names: nameA=\(nameA) nameB=\(nameB)")
    }

    @Test("Kernel name encodes fragment composition identifiers")
    func kernelNameEncodesCompositionIdentifiers() throws {
        let copyPlusNorm = try makeSynthesized(
            CopyFragment(dimension: Self.dimension),
            Reduction(dimension: Self.dimension, epsilon: 1e-6)
        )
        let residualAddPlusNorm = try makeSynthesized(
            ResidualAddFragment(dimension: Self.dimension),
            Reduction(dimension: Self.dimension, epsilon: 1e-6)
        )

        let copyName = copyPlusNorm.kernelName(context: Self.prefillContext)
        let residualName = residualAddPlusNorm.kernelName(context: Self.prefillContext)

        #expect(copyName.contains("copy"),
            "Copy composition name should reference 'copy': \(copyName)")
        #expect(copyName.contains("reduction"),
            "Copy composition name should reference 'reduction': \(copyName)")
        #expect(residualName.contains("residualadd"),
            "ResidualAdd composition name should reference 'residualadd': \(residualName)")
        #expect(residualName.contains("reduction"),
            "ResidualAdd composition name should reference 'reduction': \(residualName)")
    }

    @Test("Composition ordering changes the kernel name")
    func compositionOrderingChangesName() throws {
        // Reduction → Copy and Copy → Reduction are semantically different
        // compositions even if both are two-way; the kernel name must reflect order.
        let copyThenNorm = try makeSynthesized(
            CopyFragment(dimension: Self.dimension),
            Reduction(dimension: Self.dimension, epsilon: 1e-6)
        )
        // Note: this ordering is not what the compiler emits in production,
        // but SynthesizedFragment.kernelName must still disambiguate based on
        // fragment order so that any future rearrangement cannot alias.
        let copyName = copyThenNorm.kernelName(context: Self.prefillContext)
        let reversed = SynthesizedFragment(
            fragments: [
                Reduction(dimension: Self.dimension, epsilon: 1e-6),
                CopyFragment(dimension: Self.dimension),
            ],
            mergedContract: copyThenNorm.mergedContract
        )
        let reversedName = reversed.kernelName(context: Self.prefillContext)

        #expect(copyName != reversedName,
            "Fragment order must be reflected in the kernel name: forward=\(copyName) reversed=\(reversedName)")
    }

    // MARK: - Body Divergence

    @Test("Distinct compositions produce distinct MSL bodies")
    func distinctCompositionsProduceDistinctBodies() throws {
        let copyPlusNorm = try makeSynthesized(
            CopyFragment(dimension: Self.dimension),
            Reduction(dimension: Self.dimension, epsilon: 1e-6)
        )
        let residualAddPlusNorm = try makeSynthesized(
            ResidualAddFragment(dimension: Self.dimension),
            Reduction(dimension: Self.dimension, epsilon: 1e-6)
        )

        let copySource = copyPlusNorm.kernelSource(
            name: copyPlusNorm.kernelName(context: Self.prefillContext),
            bufferPrecision: .float32,
            weightFormat: .bfloat16
        )
        let residualSource = residualAddPlusNorm.kernelSource(
            name: residualAddPlusNorm.kernelName(context: Self.prefillContext),
            bufferPrecision: .float32,
            weightFormat: .bfloat16
        )

        #expect(copySource != residualSource,
            "MSL bodies for Copy+Reduction and ResidualAdd+Reduction must differ")
        // If bodies differ but names collide, MetalKernelSourceCatalog.generatedNames
        // silently drops the second source. The next check guards against that.
        #expect(copyPlusNorm.kernelName(context: Self.prefillContext)
                != residualAddPlusNorm.kernelName(context: Self.prefillContext),
            "Names must not collide when MSL bodies differ — pipeline cache would alias")
    }

    // MARK: - Context Tagging Still Works

    @Test("Kernel name still varies across precision and weight format")
    func kernelNameVariesAcrossContext() throws {
        let fragment = try makeSynthesized(
            CopyFragment(dimension: Self.dimension),
            Reduction(dimension: Self.dimension, epsilon: 1e-6)
        )
        let prefillBF16 = fragment.kernelName(
            context: KernelContext(bufferPrecision: .float32, weightFormat: .bfloat16)
        )
        let prefillF16 = fragment.kernelName(
            context: KernelContext(bufferPrecision: .float32, weightFormat: .float16)
        )
        let decodeBF16 = fragment.kernelName(
            context: KernelContext(bufferPrecision: .float16, weightFormat: .bfloat16)
        )

        #expect(prefillBF16 != prefillF16, "Weight format must vary kernel name")
        #expect(prefillBF16 != decodeBF16, "Buffer precision must vary kernel name")
    }
}
