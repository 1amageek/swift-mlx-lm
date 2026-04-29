/// Wraps fragment `kernelBody()` into complete MSL kernel source.
///
/// The scaffold is responsible for everything OUTSIDE the computation body:
/// - Kernel function signature and buffer declarations
/// - Sequence iteration (seqPos bounds check, row pointer computation)
/// - Threadgroup shared memory allocation
/// - Output precision cast (applied only at the final output of a fused group)
///
/// The scaffold does NOT know which fragment it wraps. It operates entirely
/// on `FusionContract` (ports, parallelism, TG memory) and the body string.
public struct KernelScaffold {

    /// Generate a complete MSL kernel from a single fragment's body and contract.
    ///
    /// - Parameters:
    ///   - name: Kernel function name.
    ///   - body: MSL computation body from `kernelBody()`.
    ///   - contract: Fragment's fusion contract.
    ///   - bufferPrecision: Buffer element precision (F16/BF16/F32).
    ///   - weightFormats: Resolved weight format per weight port name.
    ///   - isSequence: `true` for prefill (multi-position), `false` for decode (single-position).
    public static func generate(
        name: String,
        body: String,
        contract: FusionContract,
        bufferPrecision: BufferPrecision,
        weightFormats: [String: WeightFormat],
        isSequence: Bool
    ) -> String {
        switch contract.parallelism {
        case .perRow(let dimension):
            return generatePerRow(
                name: name,
                body: body,
                contract: contract,
                dimension: dimension,
                bufferPrecision: bufferPrecision,
                weightFormats: weightFormats,
                isSequence: isSequence
            )
        case .perElement(let count):
            return generatePerElement(
                name: name,
                body: body,
                contract: contract,
                count: count,
                bufferPrecision: bufferPrecision,
                weightFormats: weightFormats,
                isSequence: isSequence
            )
        case .perHead(let headCount, let headDimension):
            return generatePerHead(
                name: name,
                body: body,
                contract: contract,
                headCount: headCount,
                headDimension: headDimension,
                bufferPrecision: bufferPrecision,
                weightFormats: weightFormats,
                isSequence: isSequence
            )
        }
    }

    // MARK: - Per-Row Scaffold

    private static func generatePerRow(
        name: String,
        body: String,
        contract: FusionContract,
        dimension: Int,
        bufferPrecision: BufferPrecision,
        weightFormats: [String: WeightFormat],
        isSequence: Bool
    ) -> String {
        let bt = bufferPrecision.metalType
        let sharedDecl = contract.requiresSIMDReduction ? "threadgroup float shared[32];" : ""

        let indentedBody = body.split(separator: "\n", omittingEmptySubsequences: false)
            .map { "        \($0)" }
            .joined(separator: "\n")

        if isSequence {
            // Sequence mode: buffer ports get _base suffix, row pointers computed from seqPos
            var bufferDecls: [String] = []
            var bufferIndex = 0
            var rowPointerLines: [String] = []
            var rowStrideDecls: [String] = []

            for port in contract.ports {
                switch port.role {
                case .buffer:
                    let constQualifier = port.direction == .input ? "const " : ""
                    bufferDecls.append("    device \(constQualifier)\(bt)* \(port.name)_base [[buffer(\(bufferIndex))]]")
                    rowPointerLines.append("        device \(constQualifier)\(bt)* \(port.name) = \(port.name)_base + seqPos * \(port.name)RowStride;")
                    bufferIndex += 1
                case .weight(let field):
                    let wf = weightFormats[port.name] ?? weightFormats[field] ?? WeightFormats.float16
                    bufferDecls.append("    device const \(wf.bufferType)* \(port.name) [[buffer(\(bufferIndex))]]")
                    bufferIndex += 1
                }
            }

            let dimensionIndex = bufferIndex
            var nextIndex = dimensionIndex + 1
            var scalarDecls: [String] = []
            for sc in contract.scalarConstants {
                scalarDecls.append("    constant \(sc.metalType)& \(sc.name)       [[buffer(\(nextIndex))]]")
                nextIndex += 1
            }
            let seqLenIndex = nextIndex
            nextIndex += 1
            for port in contract.ports where port.role.isBuffer {
                rowStrideDecls.append("    constant uint& \(port.name)RowStride [[buffer(\(nextIndex))]]")
                nextIndex += 1
            }

            let allDecls = bufferDecls + [
                "    constant uint& dimension        [[buffer(\(dimensionIndex))]]"
            ] + scalarDecls + [
                "    constant uint& sequenceLength   [[buffer(\(seqLenIndex))]]"
            ] + rowStrideDecls

            return """
            kernel void \(name)(
            \(allDecls.joined(separator: ",\n")),
                uint gid_x                      [[threadgroup_position_in_grid]],
                uint tid                        [[thread_index_in_threadgroup]],
                uint threadgroupSize            [[threads_per_threadgroup]]
            ) {
                uint seqPos = gid_x;
                if (seqPos >= sequenceLength) return;

            \(rowPointerLines.joined(separator: "\n"))
                \(sharedDecl)

            \(indentedBody)
            }
            """
        } else {
            // Decode mode: buffer ports use port name directly (no row offset)
            var bufferDecls: [String] = []
            var bufferIndex = 0

            for port in contract.ports {
                switch port.role {
                case .buffer:
                    let constQualifier = port.direction == .input ? "const " : ""
                    bufferDecls.append("    device \(constQualifier)\(bt)* \(port.name) [[buffer(\(bufferIndex))]]")
                    bufferIndex += 1
                case .weight(let field):
                    let wf = weightFormats[port.name] ?? weightFormats[field] ?? WeightFormats.float16
                    bufferDecls.append("    device const \(wf.bufferType)* \(port.name) [[buffer(\(bufferIndex))]]")
                    bufferIndex += 1
                }
            }

            let dimensionIndex = bufferIndex
            var nextIndex = dimensionIndex + 1
            var scalarDecls: [String] = []
            for sc in contract.scalarConstants {
                scalarDecls.append("    constant \(sc.metalType)& \(sc.name)       [[buffer(\(nextIndex))]]")
                nextIndex += 1
            }

            let allDecls = bufferDecls + [
                "    constant uint& dimension        [[buffer(\(dimensionIndex))]]"
            ] + scalarDecls

            return """
            kernel void \(name)(
            \(allDecls.joined(separator: ",\n")),
                uint tid                        [[thread_index_in_threadgroup]],
                uint threadgroupSize            [[threads_per_threadgroup]]
            ) {
                \(sharedDecl)

            \(indentedBody)
            }
            """
        }
    }

    // MARK: - Per-Element Scaffold

    private static func generatePerElement(
        name: String,
        body: String,
        contract: FusionContract,
        count: Int,
        bufferPrecision: BufferPrecision,
        weightFormats: [String: WeightFormat],
        isSequence: Bool
    ) -> String {
        let bt = bufferPrecision.metalType
        let params = bufferParameters(contract: contract, bufferPrecision: bufferPrecision, weightFormats: weightFormats)

        // perElement body uses `i` as the element index within the current row
        // and `idx` as the flat index into the buffer (seqPos * count + i)
        let indentedBody = body.split(separator: "\n", omittingEmptySubsequences: false)
            .map { "        \($0)" }
            .joined(separator: "\n")

        // Build scalar constant declarations
        let scalarDecls = scalarConstantDeclarations(contract: contract, startIndex: params.count + 1)

        if isSequence {
            var bufferDecls: [String] = []
            var rowPointerLines: [String] = []
            var bufferPortNames: [String] = []
            for (index, port) in contract.ports.enumerated() {
                switch port.role {
                case .buffer:
                    let constQualifier = port.direction == .input ? "const " : ""
                    bufferDecls.append("    device \(constQualifier)\(bt)* \(port.name)_base [[buffer(\(index))]]")
                    rowPointerLines.append("        device \(constQualifier)\(bt)* \(port.name) = \(port.name)_base + seqPos * \(port.name)RowStride;")
                    bufferPortNames.append(port.name)
                case .weight(let field):
                    let wf = weightFormats[port.name] ?? weightFormats[field] ?? WeightFormats.float16
                    bufferDecls.append("    device const \(wf.bufferType)* \(port.name) [[buffer(\(index))]]")
                }
            }

            var paramIndex = contract.ports.count
            let countIndex = paramIndex; paramIndex += 1
            paramIndex += contract.scalarConstants.count  // skip scalar constants
            let seqLenIndex = paramIndex
            paramIndex += 1
            let rowStrideDecls = bufferPortNames.enumerated().map { offset, name in
                "    constant uint& \(name)RowStride [[buffer(\(paramIndex + offset))]]"
            }

            return """
            kernel void \(name)(
            \(bufferDecls.joined(separator: ",\n")),
                constant uint& dimension         [[buffer(\(countIndex))]],
            \(scalarDecls)\(scalarDecls.isEmpty ? "" : "\n")    constant uint& sequenceLength    [[buffer(\(seqLenIndex))]],
            \(rowStrideDecls.joined(separator: ",\n")),
                uint2 gid                        [[thread_position_in_grid]]
            ) {
                uint i = gid.x;
                uint seqPos = gid.y;
                if (i >= dimension || seqPos >= sequenceLength) return;

            \(rowPointerLines.joined(separator: "\n"))
                uint idx = i;

            \(indentedBody)

            }
            """
        } else {
            var paramIndex = params.count
            let countIndex = paramIndex; paramIndex += 1
            paramIndex += contract.scalarConstants.count  // skip scalar constants

            return """
            kernel void \(name)(
            \(params.enumerated().map { "    \($0.element.declaration) [[buffer(\($0.offset))]]" }.joined(separator: ",\n")),
                constant uint& dimension         [[buffer(\(countIndex))]],
            \(scalarDecls)\(scalarDecls.isEmpty ? "" : "\n")    uint gid                         [[thread_position_in_grid]]
            ) {
                if (gid >= dimension) return;
                uint i = gid;
                uint idx = gid;

            \(indentedBody)

            }
            """
        }
    }

    // MARK: - Per-Head Scaffold

    private static func generatePerHead(
        name: String,
        body: String,
        contract: FusionContract,
        headCount: Int,
        headDimension: Int,
        bufferPrecision: BufferPrecision,
        weightFormats: [String: WeightFormat],
        isSequence: Bool
    ) -> String {
        let bt = bufferPrecision.metalType
        let sharedDecl = contract.requiresSIMDReduction ? "threadgroup float shared[32];" : ""

        let indentedBody = body.split(separator: "\n", omittingEmptySubsequences: false)
            .map { "        \($0)" }
            .joined(separator: "\n")

        if isSequence {
            // Sequence mode: grid = (headCount, seqLen), one threadgroup per (head, position)
            var bufferDecls: [String] = []
            var bufferIndex = 0
            var pointerLines: [String] = []
            var rowStrideDecls: [String] = []

            for port in contract.ports {
                switch port.role {
                case .buffer:
                    let constQualifier = port.direction == .input ? "const " : ""
                    bufferDecls.append("    device \(constQualifier)\(bt)* \(port.name)_base [[buffer(\(bufferIndex))]]")
                    pointerLines.append("        device \(constQualifier)\(bt)* \(port.name) = \(port.name)_base + seqPos * \(port.name)RowStride + headIndex * dimension;")
                    bufferIndex += 1
                case .weight(let field):
                    let wf = weightFormats[port.name] ?? weightFormats[field] ?? WeightFormats.float16
                    bufferDecls.append("    device const \(wf.bufferType)* \(port.name) [[buffer(\(bufferIndex))]]")
                    bufferIndex += 1
                }
            }

            let dimensionIndex = bufferIndex
            var nextIndex = dimensionIndex + 1
            var scalarDecls: [String] = []
            for sc in contract.scalarConstants {
                scalarDecls.append("    constant \(sc.metalType)& \(sc.name)       [[buffer(\(nextIndex))]]")
                nextIndex += 1
            }
            let seqLenIndex = nextIndex
            nextIndex += 1
            for port in contract.ports where port.role.isBuffer {
                rowStrideDecls.append("    constant uint& \(port.name)RowStride [[buffer(\(nextIndex))]]")
                nextIndex += 1
            }

            let allDecls = bufferDecls + [
                "    constant uint& dimension        [[buffer(\(dimensionIndex))]]"
            ] + scalarDecls + [
                "    constant uint& sequenceLength   [[buffer(\(seqLenIndex))]]"
            ] + rowStrideDecls

            return """
            kernel void \(name)(
            \(allDecls.joined(separator: ",\n")),
                uint2 gid                       [[threadgroup_position_in_grid]],
                uint tid                        [[thread_index_in_threadgroup]],
                uint threadgroupSize            [[threads_per_threadgroup]]
            ) {
                uint headIndex = gid.x;
                uint seqPos = gid.y;
                if (headIndex >= \(headCount) || seqPos >= sequenceLength) return;

            \(pointerLines.joined(separator: "\n"))
                \(sharedDecl)

            \(indentedBody)
            }
            """
        } else {
            // Decode mode: grid = (headCount), one threadgroup per head
            var bufferDecls: [String] = []
            var bufferIndex = 0
            var pointerLines: [String] = []

            for port in contract.ports {
                switch port.role {
                case .buffer:
                    let constQualifier = port.direction == .input ? "const " : ""
                    bufferDecls.append("    device \(constQualifier)\(bt)* \(port.name)_base [[buffer(\(bufferIndex))]]")
                    pointerLines.append("        device \(constQualifier)\(bt)* \(port.name) = \(port.name)_base + headIndex * dimension;")
                    bufferIndex += 1
                case .weight(let field):
                    let wf = weightFormats[port.name] ?? weightFormats[field] ?? WeightFormats.float16
                    bufferDecls.append("    device const \(wf.bufferType)* \(port.name) [[buffer(\(bufferIndex))]]")
                    bufferIndex += 1
                }
            }

            let dimensionIndex = bufferIndex
            var nextIndex = dimensionIndex + 1
            var scalarDecls: [String] = []
            for sc in contract.scalarConstants {
                scalarDecls.append("    constant \(sc.metalType)& \(sc.name)       [[buffer(\(nextIndex))]]")
                nextIndex += 1
            }

            let allDecls = bufferDecls + [
                "    constant uint& dimension        [[buffer(\(dimensionIndex))]]"
            ] + scalarDecls

            return """
            kernel void \(name)(
            \(allDecls.joined(separator: ",\n")),
                uint headIndex                  [[threadgroup_position_in_grid]],
                uint tid                        [[thread_index_in_threadgroup]],
                uint threadgroupSize            [[threads_per_threadgroup]]
            ) {
                if (headIndex >= \(headCount)) return;

            \(pointerLines.joined(separator: "\n"))
                \(sharedDecl)

            \(indentedBody)
            }
            """
        }
    }

    // MARK: - Buffer Parameter Generation

    private struct BufferParam {
        let declaration: String
        let portName: String
    }

    /// Generate buffer parameter declarations from contract ports.
    private static func bufferParameters(
        contract: FusionContract,
        bufferPrecision: BufferPrecision,
        weightFormats: [String: WeightFormat]
    ) -> [BufferParam] {
        let bt = bufferPrecision.metalType
        var params: [BufferParam] = []

        for port in contract.ports {
            let constQualifier: String
            let metalType: String

            switch port.role {
            case .buffer:
                metalType = bt
                switch port.direction {
                case .input:
                    constQualifier = "const "
                case .output:
                    constQualifier = ""
                }

            case .weight(let field):
                let wf = weightFormats[port.name] ?? weightFormats[field] ?? WeightFormats.float16
                metalType = wf.bufferType
                constQualifier = "const "
            }

            let decl = "device \(constQualifier)\(metalType)* \(port.name)"
            params.append(BufferParam(declaration: decl, portName: port.name))
        }

        return params
    }

    // MARK: - Scalar Constant Declarations

    /// Generate scalar constant parameter declarations for the kernel signature.
    ///
    /// Scalar constants are placed after dimension (buffer index = startIndex)
    /// and before sequenceLength.
    private static func scalarConstantDeclarations(
        contract: FusionContract,
        startIndex: Int
    ) -> String {
        guard !contract.scalarConstants.isEmpty else { return "" }
        return contract.scalarConstants.enumerated().map { offset, sc in
            "    constant \(sc.metalType)& \(sc.name)       [[buffer(\(startIndex + offset))]]"
        }.joined(separator: ",\n") + ","
    }

}
