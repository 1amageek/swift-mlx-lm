import GGUFParser
import MLXLM

public struct ValidationContext: Sendable {
    public let file: GGUFFile
    public let architecture: String?
    public let family: DetectedArchitecture

    public init(file: GGUFFile, family: DetectedArchitecture) {
        self.file = file
        self.architecture = file.architecture
        self.family = family
    }
}
