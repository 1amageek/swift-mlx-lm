/// Pretty-prints a `ModelGraph` as a human-readable architecture summary.
///
/// Output resembles MLIR-style textual IR with indented regions,
/// value flow annotations, and compact attribute summaries.
///
/// ```swift
/// let graph = try model.makeModelGraph()
/// print(ModelGraphDumper.dump(graph))
/// ```
public enum ModelGraphDumper {

    /// Dump a `ModelGraph` as a readable string.
    ///
    /// - Parameters:
    ///   - graph: The model graph to dump.
    ///   - metadata: Optional metadata for labels.
    ///   - verbose: If true, show all attribute details. If false, show compact summary.
    /// - Returns: Multi-line string representation.
    public static func dump(
        _ graph: ModelGraph,
        metadata: ModelGraphMetadata? = nil,
        verbose: Bool = false
    ) -> String {
        var lines: [String] = []
        lines.append("ModelGraph {")
        dumpRegion(graph.rootRegion, metadata: metadata, verbose: verbose, indent: 1, lines: &lines)
        lines.append("}")
        return lines.joined(separator: "\n")
    }

    /// Dump a `NormalizedModel` (graph + metadata).
    public static func dump(
        _ model: NormalizedModel,
        verbose: Bool = false
    ) -> String {
        dump(model.graph, metadata: model.metadata, verbose: verbose)
    }

    // MARK: - Region

    private static func dumpRegion(
        _ region: Region,
        metadata: ModelGraphMetadata?,
        verbose: Bool,
        indent: Int,
        lines: inout [String],
        path: StructuralPath = StructuralPath()
    ) {
        let pad = String(repeating: "  ", count: indent)

        // Region parameters
        if !region.parameters.isEmpty {
            let params = region.parameters.map { "%\($0.id.rawValue)" }.joined(separator: ", ")
            lines.append("\(pad)params: (\(params))")
        }

        // Operations
        for (i, op) in region.operations.enumerated() {
            let opPath = path.appending(.operation(i))
            let label = metadata?.annotation(for: opPath)?.label
            dumpOperation(op, label: label, metadata: metadata, verbose: verbose,
                         indent: indent, lines: &lines, path: opPath)
        }

        // Region results
        if !region.results.isEmpty {
            let results = region.results.map { "%\($0.value.rawValue)" }.joined(separator: ", ")
            lines.append("\(pad)yield (\(results))")
        }
    }

    // MARK: - Operation

    private static func dumpOperation(
        _ op: Operation,
        label: String?,
        metadata: ModelGraphMetadata?,
        verbose: Bool,
        indent: Int,
        lines: inout [String],
        path: StructuralPath
    ) {
        let pad = String(repeating: "  ", count: indent)

        // Result values
        let resultStr: String
        if op.results.isEmpty {
            resultStr = ""
        } else {
            let vals = op.results.map { "%\($0.id.rawValue)" }.joined(separator: ", ")
            resultStr = "\(vals) = "
        }

        // Operand values
        let operandStr: String
        if op.operands.isEmpty {
            operandStr = ""
        } else {
            let vals = op.operands.map { "%\($0.value.rawValue)" }.joined(separator: ", ")
            operandStr = "(\(vals))"
        }

        // Label suffix
        let labelStr = label.map { "  // \($0)" } ?? ""

        switch op.kind {
        // Primitive operations
        case .primitive(let attrs):
            let kindStr = primitiveAttributesSummary(attrs, verbose: verbose)
            lines.append("\(pad)\(resultStr)\(kindStr)\(operandStr)\(labelStr)")

        // Structural operations with regions
        case .residual(let strategy, let body):
            lines.append("\(pad)\(resultStr)residual(\(strategy))\(operandStr) {\(labelStr)")
            dumpRegion(body, metadata: metadata, verbose: verbose, indent: indent + 1,
                      lines: &lines, path: path.appending(.regionBody))
            lines.append("\(pad)}")

        case .parallel(let merge, let branches):
            lines.append("\(pad)\(resultStr)parallel(\(mergeStr(merge)))\(operandStr) {\(labelStr)")
            for (i, branch) in branches.enumerated() {
                lines.append("\(pad)  branch[\(i)] {")
                dumpRegion(branch, metadata: metadata, verbose: verbose, indent: indent + 2,
                          lines: &lines, path: path.appending(.regionBranch(i)))
                lines.append("\(pad)  }")
            }
            lines.append("\(pad)}")

        case .repeating(let count, let body):
            lines.append("\(pad)\(resultStr)repeating(\(count)x)\(operandStr) {\(labelStr)")
            dumpRegion(body, metadata: metadata, verbose: verbose, indent: indent + 1,
                      lines: &lines, path: path.appending(.regionBody))
            lines.append("\(pad)}")

        case .conditional(let condition, let thenRegion, let elseRegion):
            lines.append("\(pad)\(resultStr)conditional(\(condition))\(operandStr) {\(labelStr)")
            lines.append("\(pad)  then {")
            dumpRegion(thenRegion, metadata: metadata, verbose: verbose, indent: indent + 2,
                      lines: &lines, path: path.appending(.regionBody))
            lines.append("\(pad)  }")
            lines.append("\(pad)  else {")
            dumpRegion(elseRegion, metadata: metadata, verbose: verbose, indent: indent + 2,
                      lines: &lines, path: path.appending(.regionBranch(1)))
            lines.append("\(pad)  }")
            lines.append("\(pad)}")
        }
    }

    // MARK: - Primitive Attributes Summary

    private static func primitiveAttributesSummary(_ attrs: any OperationAttributes, verbose: Bool) -> String {
        switch attrs {
        case let a as TokenEmbeddingAttributes:
            if verbose {
                return "tokenEmbedding(vocab=\(a.vocabSize), dim=\(a.embeddingSize)\(dtypeStr(a.dtypeHint)))"
            }
            return "tokenEmbedding(vocab=\(a.vocabSize), dim=\(a.embeddingSize))"

        case let a as PositionalEmbeddingAttributes:
            return "positionalEmbedding(\(a.kind), dim=\(a.embeddingSize), maxPos=\(a.maxPositions))"

        case let a as RoPEAttributes:
            var parts = ["dim=\(a.dimension)", "base=\(formatFloat(a.base))"]
            if let s = a.scaling { parts.append("scaling=\(s.kind)(\(formatFloat(s.factor)))") }
            if a.mropeAxes != nil { parts.append("mrope") }
            return "rope(\(parts.joined(separator: ", ")))"

        case let a as AttentionAttributes:
            var parts = [
                "hidden=\(a.hiddenSize)",
                "heads=\(a.headCount)",
                "kvHeads=\(a.kvHeadCount)",
                "headDim=\(a.headDimension)"
            ]
            if a.bias { parts.append("bias") }
            if let rope = a.rope {
                parts.append("rope(dim=\(rope.dimension), base=\(formatFloat(rope.base)))")
            }
            if let qk = a.qkNorm { parts.append("qkNorm=\(qk)") }
            if let w = a.window { parts.append("window=\(windowStr(w))") }
            if let g = a.outputGate { parts.append("gate=\(g)") }
            if verbose {
                if let hint = a.implementationHint { parts.append("hint=\(hint)") }
            }
            return "attention(\(parts.joined(separator: ", ")))"

        case let a as MLPAttributes:
            var parts = [
                "in=\(a.inputSize)",
                "out=\(a.outputSize)",
                "inter=\(a.intermediateSize)",
                "\(a.activation)",
                "\(a.gating)"
            ]
            if a.bias { parts.append("bias") }
            return "mlp(\(parts.joined(separator: ", ")))"

        case let a as MoEAttributes:
            let expert = a.expertMLP
            return "moe(experts=\(a.expertCount), topK=\(a.expertsPerToken), " +
                "gate=\(a.gateKind), mlp(in=\(expert.inputSize), inter=\(expert.intermediateSize), " +
                "\(expert.activation), \(expert.gating)))"

        case let a as RMSNormAttributes:
            return "rmsNorm(dim=\(a.dimension), eps=\(formatFloat(a.epsilon)))"

        case let a as LayerNormAttributes:
            var s = "layerNorm(dim=\(a.dimension), eps=\(formatFloat(a.epsilon))"
            if !a.affine { s += ", affine=false" }
            return s + ")"

        case let a as LinearAttributes:
            var s = "linear(in=\(a.inputSize), out=\(a.outputSize)"
            if a.bias { s += ", bias" }
            return s + ")"

        case let a as OutputHeadAttributes:
            var s = "outputHead(in=\(a.inputSize), vocab=\(a.vocabSize)"
            if a.tiedToEmbedding { s += ", tied" }
            if a.bias { s += ", bias" }
            return s + ")"

        case let a as StateSpaceAttributes:
            return "stateSpace(hidden=\(a.hiddenSize), heads=\(a.numHeads), dk=\(a.keyHeadDim), dv=\(a.valueHeadDim), variant=\(a.variant))"

        case let a as ShortConvAttributes:
            return "shortConv(hidden=\(a.hiddenSize), kernel=\(a.kernelSize))"

        case let a as CustomNodeAttributes:
            return "custom(\(a.domain).\(a.name))"

        default:
            return "primitive(\(type(of: attrs)))"
        }
    }

    // MARK: - Helpers

    private static func formatFloat(_ v: Float) -> String {
        if v == Float(Int(v)) {
            return String(Int(v))
        }
        let s = String(format: "%.6g", v)
        return s
    }

    private static func dtypeStr(_ hint: DTypeHint?) -> String {
        guard let h = hint else { return "" }
        return ", dtype=\(h)"
    }

    private static func mergeStr(_ merge: ParallelMergeStrategy) -> String {
        switch merge {
        case .add: return "add"
        case .concat: return "concat"
        case .stack: return "stack"
        case .custom(let s): return "custom(\(s))"
        }
    }

    private static func windowStr(_ w: AttentionWindow) -> String {
        var parts: [String] = []
        if let l = w.left { parts.append("L=\(l)") }
        if let r = w.right { parts.append("R=\(r)") }
        return parts.joined(separator: ",")
    }
}

// MARK: - Convenience Extensions

extension ModelGraph {

    /// Pretty-print this graph as a readable architecture summary.
    public func dump(metadata: ModelGraphMetadata? = nil, verbose: Bool = false) -> String {
        ModelGraphDumper.dump(self, metadata: metadata, verbose: verbose)
    }
}

extension NormalizedModel {

    /// Pretty-print this model as a readable architecture summary.
    public func dump(verbose: Bool = false) -> String {
        ModelGraphDumper.dump(self, verbose: verbose)
    }
}
