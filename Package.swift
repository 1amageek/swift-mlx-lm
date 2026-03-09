// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "swift-mlx-lm",
    platforms: [.macOS(.v15), .iOS(.v18), .visionOS(.v2)],
    products: [
        .library(name: "SwiftLM", targets: ["SwiftLM"]),
        .library(name: "MLXLM", targets: ["MLXLM"]),
        .library(name: "GGUFParser", targets: ["GGUFParser"]),
        .library(name: "GGUFTokenizer", targets: ["GGUFTokenizer"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.30.6")),
        .package(url: "https://github.com/huggingface/swift-jinja", from: "2.3.2"),
        .package(url: "https://github.com/mattt/JSONSchema", from: "1.3.1"),
    ],
    targets: [
        .target(
            name: "SwiftLM",
            dependencies: [
                .product(name: "JSONSchema", package: "JSONSchema"),
            ]
        ),
        .target(name: "GGUFParser"),
        .target(name: "GGUFTokenizer", dependencies: ["GGUFParser"]),
        .target(
            name: "MLXLM",
            dependencies: [
                "GGUFParser",
                "GGUFTokenizer",
                .product(name: "Jinja", package: "swift-jinja"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
            ]
        ),
        .testTarget(name: "SwiftLMTests", dependencies: ["SwiftLM"]),
        .testTarget(name: "GGUFParserTests", dependencies: ["GGUFParser"]),
        .testTarget(name: "GGUFTokenizerTests", dependencies: ["GGUFTokenizer"]),
        .testTarget(name: "MLXLMTests", dependencies: ["MLXLM"]),
    ]
)
