import Metal

private enum SharedPipelineCache {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var pipelines: [String: MTLComputePipelineState] = [:]

    static func pipeline(named name: String) -> MTLComputePipelineState? {
        lock.lock()
        defer { lock.unlock() }
        return pipelines[name]
    }

    static func store(_ pipeline: MTLComputePipelineState, named name: String) {
        lock.lock()
        pipelines[name] = pipeline
        lock.unlock()
    }
}

struct MetalPipelineCompiler {
    let device: MTLDevice

    func compile(_ generated: GeneratedKernelSources) throws -> (
        pipelines: [String: MTLComputePipelineState],
        argumentEncoders: [String: MTLArgumentEncoder],
        usesMPP: Bool
    ) {
        let baseLibrary = try makeLibrary(source: generated.baseSource, options: baseCompileOptions())
        emitKernelDiagnosticsIfRequested(
            generated: generated,
            library: baseLibrary
        )
        var pipelineCache = try makeBasePipelineCache(
            from: baseLibrary,
            mppKernelNames: generated.mppKernelNames)
        var argumentEncoderCache = makeArgumentEncoderCache(from: baseLibrary)

        guard !generated.mppSources.isEmpty else {
            return (pipelineCache, argumentEncoderCache, false)
        }

        do {
            let mppLibrary = try makeLibrary(
                source: generated.mppSources.joined(separator: "\n\n"),
                options: mppCompileOptions())
            try mergeMPPipelines(from: mppLibrary, into: &pipelineCache)
            argumentEncoderCache.merge(
                makeArgumentEncoderCache(from: mppLibrary),
                uniquingKeysWith: { existing, _ in existing })
            return (pipelineCache, argumentEncoderCache, true)
        } catch {
            return (pipelineCache, argumentEncoderCache, false)
        }
    }

    private func makeLibrary(source: String, options: MTLCompileOptions) throws -> MTLLibrary {
        try device.makeLibrary(source: source, options: options)
    }

    private func emitKernelDiagnosticsIfRequested(
        generated: GeneratedKernelSources,
        library: MTLLibrary
    ) {
        guard ProcessInfo.processInfo.environment["SWIFTLM_DEBUG_KERNELS"] == "1" else {
            return
        }
        let interestingNames = [
            "embedding_lookup",
            "embedding_lookup_bf16",
            "embedding_lookup_argbuf",
            "embedding_lookup_bf16_argbuf",
        ]
        let available = Set(library.functionNames)
        let emitted = interestingNames.filter { generated.baseSource.contains("kernel void \($0)(") }
        let compiled = interestingNames.filter { available.contains($0) }
        print("[MetalPipelineCompiler] emitted embedding kernels: \(emitted)")
        print("[MetalPipelineCompiler] compiled embedding kernels: \(compiled)")
    }

    private func baseCompileOptions() -> MTLCompileOptions {
        let options = MTLCompileOptions()
        options.mathMode = .safe
        options.languageVersion = .version4_0
        return options
    }

    private func mppCompileOptions() -> MTLCompileOptions {
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        return options
    }

    private func makeBasePipelineCache(
        from library: MTLLibrary,
        mppKernelNames: Set<String>
    ) throws -> [String: MTLComputePipelineState] {
        var pipelineCache: [String: MTLComputePipelineState] = [:]
        for name in library.functionNames {
            if mppKernelNames.contains(name),
               let cachedMPP = SharedPipelineCache.pipeline(named: "mpp::\(name)") {
                pipelineCache[name] = cachedMPP
                continue
            }
            if let cached = SharedPipelineCache.pipeline(named: name) {
                pipelineCache[name] = cached
                continue
            }
            guard let function = library.makeFunction(name: name) else {
                continue
            }
            let pipeline = try makePipeline(function: function, label: name)
            pipelineCache[name] = pipeline
            if !mppKernelNames.contains(name) {
                SharedPipelineCache.store(pipeline, named: name)
            }
        }
        return pipelineCache
    }

    private func mergeMPPipelines(
        from library: MTLLibrary,
        into pipelineCache: inout [String: MTLComputePipelineState]
    ) throws {
        for name in library.functionNames {
            let cacheKey = "mpp::\(name)"
            if let cached = SharedPipelineCache.pipeline(named: cacheKey) {
                pipelineCache[name] = cached
                continue
            }
            guard let function = library.makeFunction(name: name) else {
                continue
            }
            let pipeline = try makePipeline(function: function, label: name)
            pipelineCache[name] = pipeline
            SharedPipelineCache.store(pipeline, named: cacheKey)
        }
    }

    private func makeArgumentEncoderCache(
        from library: MTLLibrary
    ) -> [String: MTLArgumentEncoder] {
        var cache: [String: MTLArgumentEncoder] = [:]
        for name in library.functionNames where name.hasSuffix("_argbuf") {
            guard let function = library.makeFunction(name: name) else {
                continue
            }
            cache[name] = function.makeArgumentEncoder(
                bufferIndex: MetalInferenceCompiler.argumentTableBindingIndex)
        }
        return cache
    }

    private func makePipeline(
        function: MTLFunction,
        label: String
    ) throws -> MTLComputePipelineState {
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function
        descriptor.label = label
        return try device.makeComputePipelineState(
            descriptor: descriptor,
            options: [],
            reflection: nil)
    }
}
