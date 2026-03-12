import ArgumentParser
import Foundation
import GGUFParser
import GGUFToolingCore
import GGUFValidation

@main
struct GGUFTool: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "gguf-tool",
        abstract: "Validate and repair GGUF metadata without changing tensor payloads.",
        subcommands: [Validate.self, RepairPlan.self, Repair.self]
    )
}

struct Validate: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Validate a GGUF file and report missing or invalid metadata."
    )

    @Argument(help: "Path to the GGUF file to inspect.")
    var file: String

    @Flag(help: "Emit JSON instead of human-readable output.")
    var json = false

    mutating func run() throws {
        let url = URL(fileURLWithPath: file)
        let gguf = try GGUFFile.parse(url: url)
        let report = GGUFValidationRegistry.default.validate(file: gguf)
        if json {
            print(try JSONOutput.render(report: report))
        } else {
            print(TextOutput.render(report: report))
        }
        if report.hasErrors {
            throw ExitCode.failure
        }
    }
}

struct RepairPlan: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "repair-plan",
        abstract: "Show the metadata patch plan without writing a new GGUF file."
    )

    @Argument(help: "Path to the GGUF file to inspect.")
    var file: String

    @Flag(help: "Emit JSON instead of human-readable output.")
    var json = false

    mutating func run() throws {
        let url = URL(fileURLWithPath: file)
        let gguf = try GGUFFile.parse(url: url)
        let plan = GGUFValidationRegistry.default.makeRepairPlan(
            file: gguf,
            mode: .includeInferredRepairs,
            sourceURL: url
        )
        if json {
            print(try JSONOutput.render(plan: plan))
        } else {
            print(TextOutput.render(plan: plan))
        }
    }
}

struct Repair: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Write a repaired GGUF file with metadata-only changes."
    )

    @Argument(help: "Path to the GGUF file to repair.")
    var file: String

    @Option(name: .long, help: "Destination path for the repaired GGUF file.")
    var output: String

    @Flag(help: "Compatibility flag. Deterministic inferred repairs are applied in v1 regardless.")
    var applyInferred = false

    mutating func run() throws {
        let inputURL = URL(fileURLWithPath: file)
        let outputURL = URL(fileURLWithPath: output)
        let gguf = try GGUFFile.parse(url: inputURL)
        let plan = GGUFValidationRegistry.default.makeRepairPlan(
            file: gguf,
            mode: .includeInferredRepairs,
            sourceURL: inputURL
        )
        guard !plan.actions.isEmpty else {
            throw ValidationError("No deterministic metadata repairs are available for this GGUF file.")
        }

        let patch = GGUFMetadataPatch(actions: plan.actions)
        try GGUFFileRewriter().applying(patch, to: inputURL, outputURL: outputURL)
        print(TextOutput.render(repairPlan: plan, outputURL: outputURL))
    }
}
