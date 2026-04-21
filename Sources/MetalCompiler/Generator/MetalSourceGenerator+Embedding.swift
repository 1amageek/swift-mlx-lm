extension MetalSourceGenerator {
    /// Generate MSL source for embedding lookup.
    public static func generateEmbeddingLookup(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        isSequence: Bool = true,
        embeddingScale: Float? = nil
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let scaleExpr = embeddingScale != nil ? " * scale" : ""

        if isSequence {
            let scaleParam = embeddingScale != nil ? "\n            constant float& scale            [[buffer(5)]]," : ""
            return """
            kernel void \(name)(
                device const int* tokenIDs       [[buffer(0)]],
                device const \(wt)* table        [[buffer(1)]],
                device \(bt)* output             [[buffer(2)]],
                constant uint& embeddingDim      [[buffer(3)]],
                constant uint& sequenceLength    [[buffer(4)]],\(scaleParam)
                uint2 gid                        [[thread_position_in_grid]]
            ) {
                uint dim = gid.x;
                uint seqPos = gid.y;
                if (dim >= embeddingDim || seqPos >= sequenceLength) return;
                int tokenID = tokenIDs[seqPos];
                output[seqPos * embeddingDim + dim] = \(bt)(float(\(readWeight("table[tokenID * embeddingDim + dim]")))\(scaleExpr));
            }
            """
        } else {
            let scaleParam = embeddingScale != nil ? "\n            constant float& scale            [[buffer(4)]]," : ""
            return """
            kernel void \(name)(
                device const int* tokenID        [[buffer(0)]],
                device const \(wt)* table        [[buffer(1)]],
                device \(bt)* output             [[buffer(2)]],
                constant uint& embeddingDim      [[buffer(3)]],\(scaleParam)
                uint gid                         [[thread_position_in_grid]]
            ) {
                if (gid >= embeddingDim) return;
                output[gid] = \(bt)(float(\(readWeight("table[tokenID[0] * embeddingDim + gid]")))\(scaleExpr));
            }
            """
        }
    }

    public static func generateEmbeddingLookupArgumentTableVariant(
        name: String,
        argumentBufferIndex: Int,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let inputStructName = "\(name)_args"

        return """
        struct \(inputStructName) {
            device const int* tokenID [[id(0)]];
            device const \(wt)* table [[id(1)]];
            device \(bt)* output [[id(2)]];
        };

        kernel void \(name)(
            constant \(inputStructName)& args         [[buffer(\(argumentBufferIndex))]],
            constant uint& embeddingDim               [[buffer(3)]],
            uint gid                                  [[thread_position_in_grid]]
        ) {
            if (gid >= embeddingDim) return;
            args.output[gid] = \(bt)(\(readWeight("args.table[args.tokenID[0] * embeddingDim + gid]")));
        }
        """
    }

    public static func generateQuantizedEmbeddingLookupQ4(
        name: String,
        bufferPrecision: BufferPrecision,
        groupSize: Int,
        isSequence: Bool = true,
        embeddingScale: Float? = nil
    ) -> String {
        let bt = bufferPrecision.metalType
        let bytesPerBlock = 4 + groupSize / 2
        let scaleParam = embeddingScale != nil
            ? (isSequence
                ? "\n            constant float& scale            [[buffer(5)]],"
                : "\n            constant float& scale            [[buffer(4)]],")
            : ""
        let scaleExpr = embeddingScale != nil ? " * scale" : ""

        if isSequence {
            return """
            kernel void \(name)(
                device const int* tokenIDs       [[buffer(0)]],
                device const uchar* table        [[buffer(1)]],
                device \(bt)* output             [[buffer(2)]],
                constant uint& embeddingDim      [[buffer(3)]],
                constant uint& sequenceLength    [[buffer(4)]],\(scaleParam)
                uint2 gid                        [[thread_position_in_grid]]
            ) {
                const uint GROUP_SIZE = \(groupSize);
                const uint BYTES_PER_BLOCK = \(bytesPerBlock);
                uint dim = gid.x;
                uint seqPos = gid.y;
                if (dim >= embeddingDim || seqPos >= sequenceLength) return;

                uint tokenID = uint(tokenIDs[seqPos]);
                uint blocksPerRow = embeddingDim / GROUP_SIZE;
                device const uchar* rowBase = table + tokenID * blocksPerRow * BYTES_PER_BLOCK;
                uint groupIndex = dim / GROUP_SIZE;
                uint indexInGroup = dim % GROUP_SIZE;
                device const uchar* block = rowBase + groupIndex * BYTES_PER_BLOCK;
                float blockScale = float(*(device const half*)(block));
                float blockZero = float(*(device const half*)(block + 2));
                device const uchar* nibbles = block + 4;
                uchar packed = nibbles[indexInGroup / 2];
                float quantized = float((indexInGroup & 1u) == 0 ? (packed & 0x0F) : (packed >> 4));
                float value = quantized * blockScale + blockZero;
                output[seqPos * embeddingDim + dim] = \(bt)(value\(scaleExpr));
            }
            """
        }

        return """
        kernel void \(name)(
            device const int* tokenID        [[buffer(0)]],
            device const uchar* table        [[buffer(1)]],
            device \(bt)* output             [[buffer(2)]],
            constant uint& embeddingDim      [[buffer(3)]],\(scaleParam)
            uint gid                         [[thread_position_in_grid]]
        ) {
            const uint GROUP_SIZE = \(groupSize);
            const uint BYTES_PER_BLOCK = \(bytesPerBlock);
            if (gid >= embeddingDim) return;

            uint blocksPerRow = embeddingDim / GROUP_SIZE;
            device const uchar* rowBase = table + uint(tokenID[0]) * blocksPerRow * BYTES_PER_BLOCK;
            uint groupIndex = gid / GROUP_SIZE;
            uint indexInGroup = gid % GROUP_SIZE;
            device const uchar* block = rowBase + groupIndex * BYTES_PER_BLOCK;
            float blockScale = float(*(device const half*)(block));
            float blockZero = float(*(device const half*)(block + 2));
            device const uchar* nibbles = block + 4;
            uchar packed = nibbles[indexInGroup / 2];
            float quantized = float((indexInGroup & 1u) == 0 ? (packed & 0x0F) : (packed >> 4));
            float value = quantized * blockScale + blockZero;
            output[gid] = \(bt)(value\(scaleExpr));
        }
        """
    }

    public static func generateQuantizedEmbeddingLookupQ8(
        name: String,
        bufferPrecision: BufferPrecision,
        groupSize: Int,
        isSequence: Bool = true,
        embeddingScale: Float? = nil
    ) -> String {
        let bt = bufferPrecision.metalType
        let bytesPerBlock = 4 + groupSize
        let scaleParam = embeddingScale != nil
            ? (isSequence
                ? "\n            constant float& scale            [[buffer(5)]],"
                : "\n            constant float& scale            [[buffer(4)]],")
            : ""
        let scaleExpr = embeddingScale != nil ? " * scale" : ""

        if isSequence {
            return """
            kernel void \(name)(
                device const int* tokenIDs       [[buffer(0)]],
                device const uchar* table        [[buffer(1)]],
                device \(bt)* output             [[buffer(2)]],
                constant uint& embeddingDim      [[buffer(3)]],
                constant uint& sequenceLength    [[buffer(4)]],\(scaleParam)
                uint2 gid                        [[thread_position_in_grid]]
            ) {
                const uint GROUP_SIZE = \(groupSize);
                const uint BYTES_PER_BLOCK = \(bytesPerBlock);
                uint dim = gid.x;
                uint seqPos = gid.y;
                if (dim >= embeddingDim || seqPos >= sequenceLength) return;

                uint tokenID = uint(tokenIDs[seqPos]);
                uint blocksPerRow = embeddingDim / GROUP_SIZE;
                device const uchar* rowBase = table + tokenID * blocksPerRow * BYTES_PER_BLOCK;
                uint groupIndex = dim / GROUP_SIZE;
                uint indexInGroup = dim % GROUP_SIZE;
                device const uchar* block = rowBase + groupIndex * BYTES_PER_BLOCK;
                float blockScale = float(*(device const half*)(block));
                float blockZero = float(*(device const half*)(block + 2));
                device const uchar* quantized = block + 4;
                float value = float(quantized[indexInGroup]) * blockScale + blockZero;
                output[seqPos * embeddingDim + dim] = \(bt)(value\(scaleExpr));
            }
            """
        }

        return """
        kernel void \(name)(
            device const int* tokenID        [[buffer(0)]],
            device const uchar* table        [[buffer(1)]],
            device \(bt)* output             [[buffer(2)]],
            constant uint& embeddingDim      [[buffer(3)]],\(scaleParam)
            uint gid                         [[thread_position_in_grid]]
        ) {
            const uint GROUP_SIZE = \(groupSize);
            const uint BYTES_PER_BLOCK = \(bytesPerBlock);
            if (gid >= embeddingDim) return;

            uint blocksPerRow = embeddingDim / GROUP_SIZE;
            device const uchar* rowBase = table + uint(tokenID[0]) * blocksPerRow * BYTES_PER_BLOCK;
            uint groupIndex = gid / GROUP_SIZE;
            uint indexInGroup = gid % GROUP_SIZE;
            device const uchar* block = rowBase + groupIndex * BYTES_PER_BLOCK;
            float blockScale = float(*(device const half*)(block));
            float blockZero = float(*(device const half*)(block + 2));
            device const uchar* quantized = block + 4;
            float value = float(quantized[indexInGroup]) * blockScale + blockZero;
            output[gid] = \(bt)(value\(scaleExpr));
        }
        """
    }

    /// Protocol-driven quantized embedding lookup kernel.
    ///
    /// Works for any `QuantizationFormat` — aligned (Q2/Q4/Q8) uses
    /// `perWeightReadExpression`; non-aligned (Q6) expands the full group via
    /// `emitGroupDequant` into a thread-local float array then indexes by
    /// `indexInGroup`. Q4 and Q8 still have hand-written generators above
    /// (`generateQuantizedEmbeddingLookupQ4/Q8`) — this unified generator is
    /// the sole path for Q2/Q6.
    public static func generateUnifiedQuantizedEmbeddingLookup(
        name: String,
        format: any QuantizationFormat,
        bufferPrecision: BufferPrecision,
        isSequence: Bool = true,
        embeddingScale: Float? = nil
    ) -> String {
        let bt = bufferPrecision.metalType
        let groupSize = format.groupSize
        let bytesPerBlock = format.bytesPerBlock
        let blockStructName = format.blockStructName
        let scaleParam = embeddingScale != nil
            ? (isSequence
                ? "\n            constant float& scale            [[buffer(5)]],"
                : "\n            constant float& scale            [[buffer(4)]],")
            : ""
        let scaleExpr = embeddingScale != nil ? " * scale" : ""

        let blockDecl = format.mslDeclarations

        let dequantBody: String
        if format.isAligned, let perWeight = format.perWeightReadExpression(
            blocksVar: "block->qs", weightIndexVar: "indexInGroup"
        ) {
            dequantBody = """
                device const \(blockStructName)* block = reinterpret_cast<device const \(blockStructName)*>(rowBase + groupIndex * BYTES_PER_BLOCK);
                float scale = float(block->scale);
                float zero  = float(block->zero);
                float value = \(perWeight);
            """
        } else if let groupDequant = format.emitGroupDequant(
            blocksVar: "block->qs", blockIndexVar: "groupIndex", outputArrayVar: "weights_f32"
        ) {
            dequantBody = """
                device const \(blockStructName)* block = reinterpret_cast<device const \(blockStructName)*>(rowBase + groupIndex * BYTES_PER_BLOCK);
                float scale = float(block->scale);
                float zero  = float(block->zero);
                float weights_f32[\(groupSize)];
                \(groupDequant)
                float value = weights_f32[indexInGroup];
            """
        } else {
            fatalError(
                "generateUnifiedQuantizedEmbeddingLookup: format \(format.schemeIdentifier) " +
                "provides neither perWeightReadExpression nor emitGroupDequant")
        }

        if isSequence {
            return """
            \(blockDecl)

            kernel void \(name)(
                device const int* tokenIDs       [[buffer(0)]],
                device const uchar* table        [[buffer(1)]],
                device \(bt)* output             [[buffer(2)]],
                constant uint& embeddingDim      [[buffer(3)]],
                constant uint& sequenceLength    [[buffer(4)]],\(scaleParam)
                uint2 gid                        [[thread_position_in_grid]]
            ) {
                const uint GROUP_SIZE = \(groupSize);
                const uint BYTES_PER_BLOCK = \(bytesPerBlock);
                uint dim = gid.x;
                uint seqPos = gid.y;
                if (dim >= embeddingDim || seqPos >= sequenceLength) return;

                uint tokenID = uint(tokenIDs[seqPos]);
                uint blocksPerRow = embeddingDim / GROUP_SIZE;
                device const uchar* rowBase = table + tokenID * blocksPerRow * BYTES_PER_BLOCK;
                uint groupIndex = dim / GROUP_SIZE;
                uint indexInGroup = dim % GROUP_SIZE;
            \(dequantBody)
                output[seqPos * embeddingDim + dim] = \(bt)(value\(scaleExpr));
            }
            """
        }

        return """
        \(blockDecl)

        kernel void \(name)(
            device const int* tokenID        [[buffer(0)]],
            device const uchar* table        [[buffer(1)]],
            device \(bt)* output             [[buffer(2)]],
            constant uint& embeddingDim      [[buffer(3)]],\(scaleParam)
            uint gid                         [[thread_position_in_grid]]
        ) {
            const uint GROUP_SIZE = \(groupSize);
            const uint BYTES_PER_BLOCK = \(bytesPerBlock);
            if (gid >= embeddingDim) return;

            uint blocksPerRow = embeddingDim / GROUP_SIZE;
            device const uchar* rowBase = table + uint(tokenID[0]) * blocksPerRow * BYTES_PER_BLOCK;
            uint groupIndex = gid / GROUP_SIZE;
            uint indexInGroup = gid % GROUP_SIZE;
        \(dequantBody)
            output[gid] = \(bt)(value\(scaleExpr));
        }
        """
    }
}
