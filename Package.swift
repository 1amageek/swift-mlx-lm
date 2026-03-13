// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "swift-mlx-lm",
    platforms: [.macOS(.v15), .iOS(.v18), .visionOS(.v2)],
    products: [
        .library(name: "SwiftLM", targets: ["SwiftLM"]),
        .library(name: "ModelDeclarations", targets: ["ModelDeclarations"]),
        .library(name: "MLXLM", targets: ["MLXLM"]),
        .library(name: "GGUFParser", targets: ["GGUFParser"]),
        .library(name: "GGUFTokenizer", targets: ["GGUFTokenizer"]),
        .library(name: "GGUFToolingCore", targets: ["GGUFToolingCore"]),
        .library(name: "GGUFValidation", targets: ["GGUFValidation"]),
        .executable(name: "gguf-tool", targets: ["gguf-tool"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.30.6")),
        .package(url: "https://github.com/apple/swift-collections.git", from: "1.1.0"),
        .package(url: "https://github.com/huggingface/swift-jinja", from: "2.3.2"),
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.1.9")),
        .package(url: "https://github.com/mattt/JSONSchema", from: "1.3.1"),
        .package(url: "https://github.com/1amageek/swift-testing-heartbeat", from: "0.1.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "SwiftLM",
            dependencies: [
                .product(name: "JSONSchema", package: "JSONSchema"),
            ]
        ),
        .target(name: "ModelDeclarations", dependencies: ["SwiftLM"], path: "Sources/Models"),
        .target(name: "GGUFParser"),
        .target(name: "GGUFTokenizer", dependencies: ["GGUFParser"]),
        .target(name: "GGUFToolingCore", dependencies: ["GGUFParser"]),
        .target(
            name: "GGUFValidation",
            dependencies: ["GGUFParser", "MLXLM", "GGUFToolingCore"]
        ),
        .target(
            name: "MLXLM",
            dependencies: [
                "MLXCompiler",
                .product(name: "Jinja", package: "swift-jinja"),
                .product(name: "OrderedCollections", package: "swift-collections"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "gguf-tool",
            dependencies: [
                "GGUFParser",
                "GGUFToolingCore",
                "GGUFValidation",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .target(
            name: "MLXCompiler",
            dependencies: [
                "SwiftLM",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
            ]
        ),
        .testTarget(name: "MLXCompilerTests", dependencies: [
            "MLXCompiler", "SwiftLM", "ModelDeclarations",
            .product(name: "TestHeartbeat", package: "swift-testing-heartbeat"),
        ]),
        .testTarget(name: "ModelsTests", dependencies: [
            "ModelDeclarations", "SwiftLM",
            .product(name: "TestHeartbeat", package: "swift-testing-heartbeat"),
        ]),
        .testTarget(name: "SwiftLMTests", dependencies: [
            "SwiftLM",
            .product(name: "TestHeartbeat", package: "swift-testing-heartbeat"),
        ]),
        .testTarget(name: "GGUFParserTests", dependencies: [
            "GGUFParser",
            .product(name: "TestHeartbeat", package: "swift-testing-heartbeat"),
        ]),
        .testTarget(name: "GGUFTokenizerTests", dependencies: [
            "GGUFTokenizer",
            .product(name: "TestHeartbeat", package: "swift-testing-heartbeat"),
        ]),
        .testTarget(name: "GGUFToolingCoreTests", dependencies: [
            "GGUFParser",
            "GGUFToolingCore",
            .product(name: "TestHeartbeat", package: "swift-testing-heartbeat"),
        ]),
        .testTarget(name: "GGUFValidationTests", dependencies: [
            "GGUFParser",
            "GGUFValidation",
            "GGUFToolingCore",
            .product(name: "TestHeartbeat", package: "swift-testing-heartbeat"),
        ]),
        .testTarget(name: "MLXLMTests", dependencies: [
            "MLXLM",
            .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "TestHeartbeat", package: "swift-testing-heartbeat"),
        ]),
        .testTarget(name: "MLXLMDiagnosticTests", dependencies: [
            "MLXLM",
            "GGUFToolingCore",
            "GGUFValidation",
            .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "TestHeartbeat", package: "swift-testing-heartbeat"),
        ]),
    ]
)
