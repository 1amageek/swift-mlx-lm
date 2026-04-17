import Metal

/// A fused kernel fragment produced by the compiler's automatic fusion pass.
///
/// SynthesizedFragment wraps multiple adjacent fusable fragments into a single
/// dispatch unit. The compiler creates this during graph-level optimization
/// when adjacent fragments have compatible `FusionContract`s.
///
/// The kernel source is generated lazily: `kernelSource()` calls
/// `FusionSynthesizer.synthesize()` to concatenate fragment bodies, then
/// wraps the result with `KernelScaffold.generate()`.
///
/// Buffer bindings are determined by `BufferIntent` on each port in the
/// merged contract:
/// - `.dataFlow` input → current input buffer (hidden or scratch per routing state)
/// - `.dataFlow` output → hidden buffer
/// - `.residual` → residual buffer
/// - `.weight(field)` → STAF weight resolution
///
/// This type is the bridge between the optimizer (which decides WHAT to fuse)
/// and the source generator (which produces HOW to fuse). Fragment-agnostic:
/// the compiler never inspects concrete fragment types.
public struct SynthesizedFragment: PrimitiveMetalKernelFragment {

    /// Original fragments in execution order.
    public let fragments: [any PrimitiveMetalKernelFragment]

    /// Merged fusion contract (external ports only, internal junctions eliminated).
    public let mergedContract: FusionContract

    public init(fragments: [any PrimitiveMetalKernelFragment], mergedContract: FusionContract) {
        self.fragments = fragments
        self.mergedContract = mergedContract
    }

    // MARK: - PrimitiveMetalKernelFragment

    public var isFusable: Bool { false }

    public var fusionContract: FusionContract? { mergedContract }

    public var dispatchDimension: MetalDispatchDimension {
        switch mergedContract.parallelism {
        case .perRow(let dim): return .reduction(dimension: dim)
        case .perElement(let count): return .elementwise(count: count)
        case .perHead(let headCount, _): return .perHead(headCount: headCount)
        }
    }

    public var weightSlots: [MetalWeightSlot] {
        fragments.flatMap { $0.weightSlots }
    }

    public func kernelName(context: KernelContext) -> String {
        let parallelismTag: String
        switch mergedContract.parallelism {
        case .perRow(let d): parallelismTag = "row\(d)"
        case .perElement(let c): parallelismTag = "elem\(c)"
        case .perHead(let h, let d): parallelismTag = "head\(h)x\(d)"
        }
        let precisionTag = context.bufferPrecision == .float32 ? "f32" : "f16"
        let weightTag: String
        switch context.weightFormat {
        case .float16: weightTag = "wf16"
        case .bfloat16: weightTag = "wbf16"
        case .float32: weightTag = "wf32"
        case .quantized4Bit(let gs): weightTag = "wq4g\(gs)"
        case .quantized8Bit(let gs): weightTag = "wq8g\(gs)"
        }
        // Composition tag: distinct compositions must produce distinct kernel names
        // because the MSL body differs per composition. Using fragments.count alone
        // causes aliasing (e.g., `CopyFragment+Reduction` vs `ResidualAddFragment+Reduction`
        // both resolve to the same 2-way/4-port name), which lets the pipeline cache
        // reuse the first-registered kernel source for a later, semantically different
        // fusion and corrupts inference output (LFM2 regression, 2026-04-17).
        let compositionTag = fragments
            .map { Self.fragmentShortName($0) }
            .joined(separator: "_")
        let portCount = mergedContract.ports.count
        return "synthesized_\(fragments.count)way_\(compositionTag)_\(portCount)p_\(parallelismTag)_\(precisionTag)_\(weightTag)"
    }

    /// Short, stable identifier for a fragment type, used to disambiguate
    /// synthesized kernel names by composition.
    private static func fragmentShortName(_ fragment: any PrimitiveMetalKernelFragment) -> String {
        let raw = String(describing: type(of: fragment))
        let trimmed = raw.replacingOccurrences(of: "Fragment", with: "")
        return trimmed.lowercased()
    }

    public func kernelBody(bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String? {
        do {
            let result = try synthesize(bufferPrecision: bufferPrecision, weightFormat: weightFormat)
            return result.body
        } catch {
            return nil
        }
    }

    public func kernelSource(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        do {
            let result = try synthesize(bufferPrecision: bufferPrecision, weightFormat: weightFormat)
            return KernelScaffold.generate(
                name: name,
                body: result.body,
                contract: result.contract,
                bufferPrecision: bufferPrecision,
                weightFormats: result.weightFormats,
                isSequence: bufferPrecision == .float32
            )
        } catch {
            fatalError("[SynthesizedFragment] Fusion synthesis failed: \(error)")
        }
    }

    // MARK: - Scalar Constant Values

    /// Collect scalar constant values from all constituent fragments, deduplicating
    /// names to match the merged contract's scalar constants.
    ///
    /// When multiple fragments have the same scalar constant name (e.g., two
    /// Reduction fragments both have "epsilon"), the merged contract renames
    /// the second occurrence to "epsilon_N". This property rebuilds the value map
    /// using the same deduplication logic so that lookup by renamed name succeeds.
    public var scalarConstantValues: [String: ScalarConstantValue] {
        var values: [String: ScalarConstantValue] = [:]
        var nameCounts: [String: Int] = [:]

        // Count occurrences of each scalar name across fragments
        for fragment in fragments {
            for key in fragment.scalarConstantValues.keys {
                nameCounts[key, default: 0] += 1
            }
        }

        let collidingNames = Set(nameCounts.filter { $0.value > 1 }.map(\.key))
        var seenCounts: [String: Int] = [:]

        for (fragmentIndex, fragment) in fragments.enumerated() {
            for (key, value) in fragment.scalarConstantValues {
                if collidingNames.contains(key) {
                    let seen = seenCounts[key, default: 0]
                    seenCounts[key] = seen + 1
                    if seen > 0 {
                        values["\(key)_\(fragmentIndex)"] = value
                    } else {
                        values[key] = value
                    }
                } else {
                    values[key] = value
                }
            }
        }
        return values
    }

    // MARK: - Decode Bindings

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let ports = mergedContract.ports
        var bufferBindings: [(index: Int, buffer: MTLBuffer, offset: Int)] = []
        var writeIndices = Set<Int>()

        for (index, port) in ports.enumerated() {
            switch port.role {
            case .weight(let field):
                let (weightBuffer, weightOffset) = context.resolveWeight(field)
                bufferBindings.append((index, weightBuffer, weightOffset))

            case .buffer:
                switch port.bufferIntent {
                case .residual:
                    bufferBindings.append((index, context.bufferSet.residual, 0))
                    if port.direction == .output {
                        writeIndices.insert(index)
                    }

                case .dataFlow:
                    if port.direction == .input {
                        bufferBindings.append((index, context.currentInputBuffer, context.currentInputOffset))
                    } else {
                        // Output dataFlow port → hidden buffer
                        bufferBindings.append((index, context.bufferSet.hidden, 0))
                        writeIndices.insert(index)
                    }
                }
            }
        }

        // Dimension binding (after buffer ports)
        let dimensionIndex = ports.count
        var bytesBindings: [(index: Int, value: [UInt8])] = [
            uint32Binding(dimensionIndex, UInt32(mergedContract.parallelism.dimension)),
        ]

        // Scalar constant bindings
        let scalarValues = self.scalarConstantValues
        for (offset, sc) in mergedContract.scalarConstants.enumerated() {
            let bindingIndex = dimensionIndex + 1 + offset
            guard let value = scalarValues[sc.name] else {
                fatalError("[SynthesizedFragment] Missing scalar constant value for '\(sc.name)'")
            }
            bytesBindings.append((index: bindingIndex, value: value.bytes))
        }

        // Routing state: derived from the merged contract's output structure.
        // The last output port determines whether the result goes to hidden.
        // resetsProjectionIndex is true when any constituent fragment resets it
        // (e.g., Reduction always resets projection index).
        let hasDataFlowOutput = ports.contains { port in
            port.direction == .output && port.bufferIntent == .dataFlow
        }

        return FragmentBindings(
            buffers: bufferBindings,
            bytes: bytesBindings,
            outputIsHidden: hasDataFlowOutput,
            resetsProjectionIndex: hasDataFlowOutput,
            writeBufferIndices: writeIndices
        )
    }

    // MARK: - Prefill Steps

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let kernelName = kernelName(context: context.kernelContext)
        let pipeline = try context.getPipeline(kernelName)

        let ports = mergedContract.ports
        var bufferBindings: [(Int, MTLBuffer, Int)] = []

        for (index, port) in ports.enumerated() {
            switch port.role {
            case .weight(let field):
                let (weightBuffer, weightOffset) = context.resolveWeight(field)
                bufferBindings.append((index, weightBuffer, weightOffset))

            case .buffer:
                switch port.bufferIntent {
                case .residual:
                    bufferBindings.append((index, context.buffers.residual, 0))

                case .dataFlow:
                    if port.direction == .input {
                        bufferBindings.append((index, context.currentInputBuffer, context.currentInputOffset))
                    } else {
                        // Output dataFlow port → hidden buffer
                        bufferBindings.append((index, context.buffers.hidden, 0))
                    }
                }
            }
        }

        // Dimension binding
        let dimensionIndex = ports.count
        var bytesBindings: [(Int, [UInt8])] = [
            uint32Binding(dimensionIndex, UInt32(mergedContract.parallelism.dimension)),
        ]

        // Scalar constant bindings
        let scalarValues = self.scalarConstantValues
        for (offset, sc) in mergedContract.scalarConstants.enumerated() {
            let bindingIndex = dimensionIndex + 1 + offset
            guard let value = scalarValues[sc.name] else {
                fatalError("[SynthesizedFragment] Missing scalar constant value for '\(sc.name)'")
            }
            bytesBindings.append((bindingIndex, value.bytes))
        }

        // Sequence length binding (after scalar constants)
        let seqLenIndex = dimensionIndex + 1 + mergedContract.scalarConstants.count
        bytesBindings.append(uint32Binding(seqLenIndex, UInt32(context.maximumSequenceLength)))

        // Grid/threadgroup sizing based on parallelism
        let gridSize: MTLSize
        let threadgroupSize: MTLSize
        let threadgroupMemoryLength: Int

        switch mergedContract.parallelism {
        case .perRow:
            let simdWidth = pipeline.threadExecutionWidth
            let dim = mergedContract.parallelism.dimension
            let clamped = min(max(dim, 1), 1024)
            let rounded = ((clamped + simdWidth - 1) / simdWidth) * simdWidth
            let threads = min(rounded, pipeline.maxTotalThreadsPerThreadgroup)
            gridSize = MTLSize(width: context.maximumSequenceLength, height: 1, depth: 1)
            threadgroupSize = MTLSize(width: threads, height: 1, depth: 1)
            threadgroupMemoryLength = 0

        case .perElement(let count):
            let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
            let gridX = (count + tgSize - 1) / tgSize
            gridSize = MTLSize(width: gridX, height: context.maximumSequenceLength, depth: 1)
            threadgroupSize = MTLSize(width: tgSize, height: 1, depth: 1)
            threadgroupMemoryLength = 0

        case .perHead(let headCount, _):
            let threads = min(32, pipeline.maxTotalThreadsPerThreadgroup)
            gridSize = MTLSize(width: headCount, height: context.maximumSequenceLength, depth: 1)
            threadgroupSize = MTLSize(width: threads, height: 1, depth: 1)
            threadgroupMemoryLength = 0
        }

        // Routing state: derived from merged contract output structure
        let hasDataFlowOutput = ports.contains { port in
            port.direction == .output && port.bufferIntent == .dataFlow
        }

        let sequenceLengthPolicy: PrefillSequenceLengthPolicy
        switch mergedContract.parallelism {
        case .perRow:
            sequenceLengthPolicy = .bind(index: seqLenIndex)
        case .perElement:
            sequenceLengthPolicy = .bindAndAdjustGridHeight(index: seqLenIndex)
        case .perHead:
            sequenceLengthPolicy = .bindAndAdjustGridHeight(index: seqLenIndex)
        }

        return FragmentPrefillSteps(
            steps: [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: gridSize,
                threadgroupSize: threadgroupSize,
                bufferBindings: bufferBindings,
                bytesBindings: bytesBindings,
                threadgroupMemoryLength: threadgroupMemoryLength,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: sequenceLengthPolicy,
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )],
            outputIsHidden: hasDataFlowOutput,
            resetsProjectionIndex: hasDataFlowOutput
        )
    }

    // MARK: - Internal

    private func synthesize(
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) throws -> FusionSynthesizer.SynthesisResult {
        var entries: [FusionSynthesizer.Entry] = []

        for frag in fragments {
            guard let contract = frag.fusionContract,
                  let body = frag.kernelBody(bufferPrecision: bufferPrecision, weightFormat: weightFormat) else {
                fatalError("[SynthesizedFragment] Fragment \(type(of: frag)) missing fusionContract or kernelBody")
            }
            var weightFormats: [String: WeightFormat] = [:]
            for port in contract.ports {
                if case .weight(let field) = port.role {
                    weightFormats[port.name] = weightFormat
                    weightFormats[field] = weightFormat
                }
            }
            entries.append(.init(contract: contract, body: body, weightFormats: weightFormats))
        }

        return try FusionSynthesizer.synthesize(entries)
    }
}
