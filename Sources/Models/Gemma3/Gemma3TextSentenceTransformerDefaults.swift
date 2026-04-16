import Foundation

public enum SentenceTransformerModuleDefaultsResolver {
    public enum PoolingStrategy: Sendable {
        case mean
        case cls
        case max
        case lastToken
    }

    public struct PoolingDefaults: Sendable {
        public let strategy: PoolingStrategy
        public let includePrompt: Bool

        public init(
            strategy: PoolingStrategy,
            includePrompt: Bool
        ) {
            self.strategy = strategy
            self.includePrompt = includePrompt
        }
    }

    public struct DenseDefaults: Sendable {
        public let activationFunctionName: String

        public init(activationFunctionName: String) {
            self.activationFunctionName = activationFunctionName
        }
    }

    public static func poolingDefaults(
        modelType: String,
        modulePath: String
    ) -> PoolingDefaults? {
        switch modelType {
        case "gemma3_text":
            guard modulePath == "1_Pooling" else { return nil }
            return PoolingDefaults(
                strategy: .mean,
                includePrompt: true
            )
        default:
            return nil
        }
    }

    public static func denseDefaults(
        modelType: String,
        modulePath: String,
        denseIndex: Int
    ) -> DenseDefaults? {
        switch modelType {
        case "gemma3_text":
            guard denseIndex == 0 || denseIndex == 1 else { return nil }
            guard modulePath == "2_Dense" || modulePath == "3_Dense" else { return nil }
            return DenseDefaults(
                activationFunctionName: "torch.nn.modules.linear.Identity"
            )
        default:
            return nil
        }
    }
}
