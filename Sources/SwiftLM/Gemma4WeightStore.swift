import Foundation
import MetalCompiler

final class Gemma4WeightStore {
    struct DenseTensor {
        let values: [Float]
        let shape: [Int]
    }

    private let weights: MetalWeightStore?
    private let stafWeights: STAFWeightStore?
    private let denseTensors: [String: DenseTensor]
    private var floatTensorCache: [String: [Float]] = [:]

    init(weights: MetalWeightStore) {
        self.weights = weights
        self.stafWeights = nil
        self.denseTensors = [:]
    }

    init(weights: STAFWeightStore) {
        self.weights = nil
        self.stafWeights = weights
        self.denseTensors = [:]
    }

    init(denseTensors: [String: DenseTensor]) {
        self.weights = nil
        self.stafWeights = nil
        self.denseTensors = denseTensors
    }

    func floatTensor(named name: String) throws -> [Float] {
        if let dense = denseTensors[name] {
            return dense.values
        }
        if let cached = floatTensorCache[name] {
            return cached
        }
        if let stafWeights {
            guard let entry = stafWeights.entries[name] else {
                throw ModelBundleLoaderError.invalidConfig("Missing Gemma4 tensor: \(name)")
            }
            let elementCount = entry.shape.reduce(1, *)
            let basePointer = stafWeights.buffer.contents().advanced(by: entry.bufferOffset)
            let decoded: [Float]
            switch entry.schemeIdentifier {
            case .fp16RowMajor:
                let values = basePointer.bindMemory(to: UInt16.self, capacity: elementCount)
                decoded = (0..<elementCount).map { Float(Float16(bitPattern: values[$0])) }
            case .bf16RowMajor:
                let values = basePointer.bindMemory(to: UInt16.self, capacity: elementCount)
                decoded = (0..<elementCount).map { Float(bitPattern: UInt32(values[$0]) << 16) }
            default:
                throw ModelBundleLoaderError.invalidConfig(
                    "Quantized Gemma4 tensor is not supported: \(name)"
                )
            }
            floatTensorCache[name] = decoded
            return decoded
        }
        guard let weights else {
            throw ModelBundleLoaderError.invalidConfig("Missing Gemma4 tensor: \(name)")
        }
        guard let tensor = weights.tensor(for: name) else {
            throw ModelBundleLoaderError.invalidConfig("Missing Gemma4 tensor: \(name)")
        }
        let elementCount = tensor.shape.reduce(1, *)
        let basePointer = tensor.buffer.contents().advanced(by: tensor.offset)
        let decoded: [Float]
        switch tensor.dtype {
        case .float16:
            let values = basePointer.bindMemory(to: UInt16.self, capacity: elementCount)
            decoded = (0..<elementCount).map { Float(Float16(bitPattern: values[$0])) }
        case .bfloat16:
            let values = basePointer.bindMemory(to: UInt16.self, capacity: elementCount)
            decoded = (0..<elementCount).map { Float(bitPattern: UInt32(values[$0]) << 16) }
        case .float32:
            let values = basePointer.bindMemory(to: Float.self, capacity: elementCount)
            decoded = Array(UnsafeBufferPointer(start: values, count: elementCount))
        case .quantized:
            throw ModelBundleLoaderError.invalidConfig(
                "Quantized Gemma4 tensor is not supported: \(name)"
            )
        }

        floatTensorCache[name] = decoded
        return decoded
    }

    func optionalFloatTensor(named name: String) throws -> [Float]? {
        if let dense = denseTensors[name] {
            return dense.values
        }
        if let cached = floatTensorCache[name] {
            return cached
        }
        if let stafWeights {
            guard let entry = stafWeights.entries[name] else {
                return nil
            }
            let elementCount = entry.shape.reduce(1, *)
            let basePointer = stafWeights.buffer.contents().advanced(by: entry.bufferOffset)
            let decoded: [Float]
            switch entry.schemeIdentifier {
            case .fp16RowMajor:
                let values = basePointer.bindMemory(to: UInt16.self, capacity: elementCount)
                decoded = (0..<elementCount).map { Float(Float16(bitPattern: values[$0])) }
            case .bf16RowMajor:
                let values = basePointer.bindMemory(to: UInt16.self, capacity: elementCount)
                decoded = (0..<elementCount).map { Float(bitPattern: UInt32(values[$0]) << 16) }
            default:
                throw ModelBundleLoaderError.invalidConfig(
                    "Quantized Gemma4 tensor is not supported: \(name)"
                )
            }
            floatTensorCache[name] = decoded
            return decoded
        }
        guard let weights else {
            return nil
        }
        guard let tensor = weights.tensor(for: name) else {
            return nil
        }
        let elementCount = tensor.shape.reduce(1, *)
        let basePointer = tensor.buffer.contents().advanced(by: tensor.offset)
        let decoded: [Float]
        switch tensor.dtype {
        case .float16:
            let values = basePointer.bindMemory(to: UInt16.self, capacity: elementCount)
            decoded = (0..<elementCount).map { Float(Float16(bitPattern: values[$0])) }
        case .bfloat16:
            let values = basePointer.bindMemory(to: UInt16.self, capacity: elementCount)
            decoded = (0..<elementCount).map { Float(bitPattern: UInt32(values[$0]) << 16) }
        case .float32:
            let values = basePointer.bindMemory(to: Float.self, capacity: elementCount)
            decoded = Array(UnsafeBufferPointer(start: values, count: elementCount))
        case .quantized:
            throw ModelBundleLoaderError.invalidConfig(
                "Quantized Gemma4 tensor is not supported: \(name)"
            )
        }
        floatTensorCache[name] = decoded
        return decoded
    }

    func shape(named name: String) throws -> [Int] {
        if let dense = denseTensors[name] {
            return dense.shape
        }
        if let stafWeights {
            guard let entry = stafWeights.entries[name] else {
                throw ModelBundleLoaderError.invalidConfig("Missing Gemma4 tensor shape: \(name)")
            }
            return entry.shape
        }
        guard let weights else {
            throw ModelBundleLoaderError.invalidConfig("Missing Gemma4 tensor shape: \(name)")
        }
        guard let info = weights.tensorInfo(for: name) else {
            throw ModelBundleLoaderError.invalidConfig("Missing Gemma4 tensor shape: \(name)")
        }
        return info.shape
    }
}
