import Foundation
import Metal
@testable import MetalCompiler

enum RealModelTestSupport {
    private struct SharedResources {
        let device: MTLDevice
        let store: STAFWeightStore
    }

    struct Resources {
        let gpuLock: GPUTestExclusion.LockHandle
        let device: MTLDevice
        let store: STAFWeightStore

        func release() {
            gpuLock.release()
            RealModelTestSupport.inProcessLock.unlock()
        }
    }

    private static let inProcessLock = NSLock()
    private static let sharedResourcesLock = NSLock()
    nonisolated(unsafe) private static var sharedResources: SharedResources?

    static func loadOrSkip(skipMessage: String) throws -> Resources? {
        inProcessLock.lock()
        let gpuLock = try GPUTestExclusion.acquire()

        sharedResourcesLock.lock()
        defer { sharedResourcesLock.unlock() }

        if let sharedResources {
            return Resources(gpuLock: gpuLock, device: sharedResources.device, store: sharedResources.store)
        }

        do {
            let (device, store) = try BenchmarkSupport.loadStoreOrSkip()
            sharedResources = SharedResources(device: device, store: store)
            return Resources(gpuLock: gpuLock, device: device, store: store)
        } catch BenchmarkSupport.BenchError.noModel {
            gpuLock.release()
            inProcessLock.unlock()
            print(skipMessage)
            return nil
        } catch {
            gpuLock.release()
            inProcessLock.unlock()
            throw error
        }
    }
}
