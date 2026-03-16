// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "swift-lm",
    platforms: [.macOS(.v15), .iOS(.v18), .visionOS(.v2)],
    products: [
        .library(name: "LMIR", targets: ["LMIR"]),
        .library(name: "LMArchitecture", targets: ["LMArchitecture"]),
        .library(name: "ModelDeclarations", targets: ["ModelDeclarations"]),
        .library(name: "MetalCompiler", targets: ["MetalCompiler"]),
        .library(name: "SwiftLM", targets: ["SwiftLM"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-collections.git", from: "1.1.0"),
        .package(url: "https://github.com/huggingface/swift-jinja", from: "2.3.2"),
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.1.9")),
        .package(url: "https://github.com/mattt/JSONSchema", from: "1.3.1"),
        .package(url: "https://github.com/1amageek/swift-testing-heartbeat", from: "0.1.0"),
    ],
    targets: [
        .target(
            name: "LMIR"
        ),
        .target(
            name: "LMArchitecture",
            dependencies: [
                "LMIR",
                .product(name: "JSONSchema", package: "JSONSchema"),
            ]
        ),
        .target(name: "ModelDeclarations", dependencies: ["LMArchitecture"], path: "Sources/Models"),
        .target(
            name: "MetalCompiler",
            dependencies: ["LMIR"]
        ),
        .target(
            name: "SwiftLM",
            dependencies: [
                "LMArchitecture",
                "MetalCompiler",
                "ModelDeclarations",
                .product(name: "Jinja", package: "swift-jinja"),
                .product(name: "OrderedCollections", package: "swift-collections"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ]
        ),
        .testTarget(name: "MetalCompilerTests", dependencies: [
            "MetalCompiler", "LMArchitecture", "ModelDeclarations", "SwiftLM",
            .product(name: "TestHeartbeat", package: "swift-testing-heartbeat"),
            .product(name: "Tokenizers", package: "swift-transformers"),
            .product(name: "Jinja", package: "swift-jinja"),
            .product(name: "OrderedCollections", package: "swift-collections"),
        ]),
        .testTarget(name: "ModelsTests", dependencies: [
            "ModelDeclarations", "LMArchitecture",
            .product(name: "TestHeartbeat", package: "swift-testing-heartbeat"),
        ], exclude: ["ModelDeclarationTests.swift"]),
        .testTarget(name: "SwiftLMTests", dependencies: [
            "SwiftLM",
            .product(name: "TestHeartbeat", package: "swift-testing-heartbeat"),
        ], exclude: ["DSLLoweringTests.swift", "IRInvariantTests.swift", "PerformanceTests.swift", "ModelGraphTests.swift", "DimensionValidatorTests.swift"]),
    ]
)
