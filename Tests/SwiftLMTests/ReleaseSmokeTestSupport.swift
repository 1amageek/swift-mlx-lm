import Foundation

enum ReleaseSmokeTestSupport {
    static let localModelDirectory = URL(
        fileURLWithPath: "/Users/1amageek/Desktop/swift-lm/TestData/LFM2.5-1.2B-Thinking"
    )

    static func readableLocalModelDirectoryOrSkip() -> URL? {
        let configURL = localModelDirectory.appendingPathComponent("config.json")
        do {
            _ = try Data(contentsOf: configURL)
            return localModelDirectory
        } catch {
            print("[Skip] Local release smoke bundle is not readable at \(localModelDirectory.path): \(error)")
            return nil
        }
    }
}
