import Foundation

enum GPUTestExclusion {
    static func acquire() throws -> LockHandle {
        try LockHandle()
    }

    final class LockHandle {
        private let fileDescriptor: Int32
        private var released = false

        init() throws {
            let lockPath = FileManager.default.temporaryDirectory
                .appendingPathComponent("swift-lm-metal-tests.lock")
                .path
            let descriptor = open(lockPath, O_CREAT | O_RDWR, 0o666)
            guard descriptor >= 0 else {
                throw GPUTestExclusionError.openFailed(path: lockPath, errno: errno)
            }
            guard flock(descriptor, LOCK_EX) == 0 else {
                let lockErrno = errno
                close(descriptor)
                throw GPUTestExclusionError.lockFailed(path: lockPath, errno: lockErrno)
            }
            self.fileDescriptor = descriptor
        }

        func release() {
            guard !released else { return }
            released = true
            flock(fileDescriptor, LOCK_UN)
            close(fileDescriptor)
        }

        deinit {
            release()
        }
    }
}

private enum GPUTestExclusionError: Error, CustomStringConvertible {
    case openFailed(path: String, errno: Int32)
    case lockFailed(path: String, errno: Int32)

    var description: String {
        switch self {
        case .openFailed(let path, let errnoValue):
            return "Failed to open GPU test lock file at \(path): errno=\(errnoValue)"
        case .lockFailed(let path, let errnoValue):
            return "Failed to acquire GPU test lock at \(path): errno=\(errnoValue)"
        }
    }
}
