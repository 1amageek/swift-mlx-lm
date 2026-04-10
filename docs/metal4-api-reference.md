# Metal 4 API Reference for swift-lm

Metal 4 (`macOS 26.0+`, `iOS 26.0+`) — swift-lm の primary target API.

## Object Creation

All Metal 4 objects are created from `MTLDevice`:

```swift
let device = MTLCreateSystemDefaultDevice()!

let queue: MTL4CommandQueue       = device.makeMTL4CommandQueue()!
let allocator: MTL4CommandAllocator = device.makeCommandAllocator()!
let commandBuffer: MTL4CommandBuffer = device.makeCommandBuffer()!  // returns MTL4CommandBuffer

let atDesc = MTL4ArgumentTableDescriptor()
atDesc.maxBufferBindCount = 31
let argumentTable: MTL4ArgumentTable = try device.makeArgumentTable(descriptor: atDesc)
```

Runtime detection:
```swift
if let queue = device.makeMTL4CommandQueue() {
    // Metal 4 available
}
```

---

## MTL4CommandAllocator

Command encoding memory manager. NOT thread-safe — use separate allocators for separate threads.

```swift
protocol MTL4CommandAllocator: NSObjectProtocol {
    var device: MTLDevice { get }
    var label: String? { get set }
    func allocatedSize() -> UInt64
    func reset()  // reclaim memory after GPU finishes
}
```

Triple-buffered pattern:
```swift
let kMaxInFlight = 3
let allocators = (0..<kMaxInFlight).map { _ in device.makeCommandAllocator()! }
let commandBuffer = device.makeCommandBuffer()!  // single reusable buffer

func decodeToken(frameIndex: Int) {
    let allocator = allocators[frameIndex % kMaxInFlight]
    allocator.reset()
    commandBuffer.beginCommandBuffer(allocator: allocator)
    // ... encode ...
    commandBuffer.endCommandBuffer()
    queue.commit([commandBuffer])
}
```

---

## MTL4CommandBuffer

Reusable command buffer. Created once from device, begin/end cycle repeats per submission.

```swift
protocol MTL4CommandBuffer: NSObjectProtocol {
    var device: MTLDevice { get }
    var label: String? { get set }

    func beginCommandBuffer(allocator: MTL4CommandAllocator)
    func beginCommandBuffer(allocator: MTL4CommandAllocator, options: MTL4CommandBufferOptions)
    func endCommandBuffer()

    func makeComputeCommandEncoder() -> MTL4ComputeCommandEncoder?
    func makeRenderCommandEncoder(descriptor: MTL4RenderPassDescriptor, options: MTL4RenderEncoderOptions) -> MTL4RenderCommandEncoder?
    func makeMachineLearningCommandEncoder() -> MTL4MachineLearningCommandEncoder?

    func useResidencySet(_ residencySet: MTLResidencySet)
    func useResidencySets(_ residencySets: [MTLResidencySet])

    func writeTimestamp(counterHeap: MTL4CounterHeap, index: Int)
    func resolveCounterHeap(_ heap: MTL4CounterHeap, range: Range<Int>,
                            buffer: MTL4BufferRange,
                            fenceToWait: MTLFence?, fenceToUpdate: MTLFence?)

    func pushDebugGroup(_ string: String)
    func popDebugGroup()
}
```

Key differences from `MTLCommandBuffer`:
- Created from device, not queue
- Reusable (long-lived object)
- Does NOT retain resources
- Submitted via queue's `commit()`, not its own `commit()`

---

## MTL4CommandQueue

```swift
protocol MTL4CommandQueue: NSObjectProtocol, Sendable {
    var device: MTLDevice { get }
    var label: String? { get }

    func commit(_ commandBuffers: [MTL4CommandBuffer], options: MTL4CommitOptions?)

    // Cross-queue synchronization
    func signalEvent(_ event: MTLEvent, value: UInt64)
    func waitForEvent(_ event: MTLEvent, value: UInt64)

    // Residency management (up to 32 per queue)
    func addResidencySet(_ set: MTLResidencySet)
    func addResidencySets(_ sets: [MTLResidencySet])
    func removeResidencySet(_ set: MTLResidencySet)
    func removeResidencySets(_ sets: [MTLResidencySet])
}
```

Feedback:
```swift
let options = MTL4CommitOptions()
options.addFeedbackHandler { feedback in
    if let error = feedback.error { print("GPU error: \(error)") }
    let gpuTime = feedback.gpuEndTime - feedback.gpuStartTime
    print("GPU time: \(gpuTime * 1000) ms")
}
queue.commit([commandBuffer], options: options)
```

---

## MTL4CommandEncoder (Base Protocol)

Inherited by `MTL4ComputeCommandEncoder`, `MTL4RenderCommandEncoder`, `MTL4MachineLearningCommandEncoder`.

### Three Barrier Types

```swift
protocol MTL4CommandEncoder: NSObjectProtocol {
    var label: String? { get set }
    var commandBuffer: MTL4CommandBuffer? { get }

    // 1. Intra-pass barrier (within same encoder)
    func barrier(afterEncoderStages: MTLStages,
                 beforeEncoderStages: MTLStages,
                 visibilityOptions: MTL4VisibilityOptions)

    // 2. Consumer barrier (wait for previous passes)
    //    afterQueueStages: stages in PREVIOUS passes to wait for
    //    beforeStages: stages in THIS and ALL SUBSEQUENT passes to block
    func barrier(afterQueueStages: MTLStages,
                 beforeStages: MTLStages,
                 visibilityOptions: MTL4VisibilityOptions)

    // 3. Producer barrier (notify subsequent passes)
    //    afterStages: stages in THIS and ALL PREVIOUS passes that must complete
    //    beforeQueueStages: stages in SUBSEQUENT passes to block
    func barrier(afterStages: MTLStages,
                 beforeQueueStages: MTLStages,
                 visibilityOptions: MTL4VisibilityOptions)

    // Fence operations
    func updateFence(_ fence: MTLFence, afterEncoderStages: MTLStages)
    func waitForFence(_ fence: MTLFence, beforeEncoderStages: MTLStages)

    func endEncoding()
    func insertDebugSignpost(_ string: String)
    func pushDebugGroup(_ string: String)
    func popDebugGroup()
}
```

### Default Visibility

Swift overlays default `visibilityOptions` to `.device`:
```swift
// These are equivalent:
encoder.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch)
encoder.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch, visibilityOptions: .device)
```

To use execution-only barrier (no cache flush):
```swift
encoder.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch, visibilityOptions: [])
```

---

## MTL4VisibilityOptions

```swift
struct MTL4VisibilityOptions: OptionSet, Sendable {
    static var device: MTL4VisibilityOptions        // Flush caches to GPU memory coherence point
    static var resourceAlias: MTL4VisibilityOptions  // Flush for aliased virtual addresses
    // [] (empty) = execution ordering only, NO cache flush
}
```

| Value | Behavior | Use Case |
|---|---|---|
| `[]` (none) | Execution ordering only | `private` GPU-only buffers on unified memory |
| `.device` | + cache flush to device memory | `shared` buffers, CPU-visible data |
| `.resourceAlias` | + aliased address flush | Overlapping virtual memory mappings |
| `[.device, .resourceAlias]` | Both | Aliased shared buffers |

**For swift-lm decode path**: All intermediate buffers (`hidden`, `scratch`, `residual`) are `private` + `hazardTrackingModeUntracked`, accessed only by GPU. `visibilityOptions: []` should be sufficient.

---

## MTLStages

```swift
struct MTLStages: OptionSet, Sendable {
    static var vertex: MTLStages                  // 1 << 0  — render
    static var fragment: MTLStages                // 1 << 1  — render
    static var tile: MTLStages                    // 1 << 2  — render
    static var object: MTLStages                  // 1 << 3  — render
    static var mesh: MTLStages                    // 1 << 4  — render
    static var resourceState: MTLStages           // 1 << 26 — sparse mapping
    static var dispatch: MTLStages                // 1 << 27 — compute dispatches
    static var blit: MTLStages                    // 1 << 28 — copy/fill operations
    static var accelerationStructure: MTLStages   // 1 << 29 — ray tracing
    static var machineLearning: MTLStages         // 1 << 30 — ML network dispatches
    static var all: MTLStages                     // all bits
}
```

**For compute-only LLM inference**: `.dispatch` and `.blit` are the relevant stages.

---

## MTL4ComputeCommandEncoder

Unified compute + blit encoder. No separate `MTLBlitCommandEncoder` in Metal 4.

```swift
protocol MTL4ComputeCommandEncoder: MTL4CommandEncoder {
    func stages() -> MTLStages

    // Configuration
    func setComputePipelineState(_ state: MTLComputePipelineState)
    func setArgumentTable(_ table: MTL4ArgumentTable?)
    func setThreadgroupMemoryLength(_ length: Int, index: Int)
    func setImageblockSize(width: Int, height: Int)

    // Dispatch (stage: .dispatch)
    func dispatchThreads(threadsPerGrid: MTLSize, threadsPerThreadgroup: MTLSize)
    func dispatchThreads(indirectBuffer: MTLGPUAddress)
    func dispatchThreadgroups(threadgroupsPerGrid: MTLSize, threadsPerThreadgroup: MTLSize)
    func dispatchThreadgroups(indirectBuffer: MTLGPUAddress, threadsPerThreadgroup: MTLSize)

    // Copy (stage: .blit)
    func copy(sourceBuffer: MTLGPUAddress, sourceOffset: Int,
              destinationBuffer: MTLGPUAddress, destinationOffset: Int, size: Int)
    func fill(buffer: MTL4BufferRange, value: UInt8)
    func generateMipmaps(texture: MTLResourceID)
    // ... additional copy variants for textures, tensors

    // Timestamps
    func writeTimestamp(granularity: MTL4TimestampGranularity,
                        counterHeap: MTL4CounterHeap, index: Int)

    // ICB operations
    func executeCommandsInBuffer(_ buffer: MTLIndirectCommandBuffer, range: Range<Int>)
    func resetCommandsInBuffer(_ buffer: MTLIndirectCommandBuffer, range: Range<Int>)
}
```

### Key Differences from MTLComputeCommandEncoder

| Feature | Metal 3 | Metal 4 |
|---|---|---|
| Buffer binding | `setBuffer(buf, offset, index)` | `argumentTable.setAddress(gpuAddress, index)` |
| Bytes binding | `setBytes(ptr, length, index)` | Pre-allocate in buffer, bind via address |
| Barrier | `memoryBarrier(scope:)` / `memoryBarrier(resources:)` | `barrier(afterEncoderStages:beforeEncoderStages:visibilityOptions:)` |
| Blit operations | Separate `MTLBlitCommandEncoder` | Unified in compute encoder |
| Dispatch mode | Serial by default | **Concurrent by default** |
| Resource tracking | `hazardTrackingMode` respected | **Ignored** — all synchronization explicit |

### No Individual Resource Binding

Metal 4 compute encoder has NO `setBuffer`, `setBytes`, `setTexture` methods. All resources go through `MTL4ArgumentTable`.

---

## MTL4ArgumentTable

Bindless argument binding. Uses `MTLGPUAddress` and `MTLResourceID`.

```swift
protocol MTL4ArgumentTable: NSObjectProtocol {
    var device: MTLDevice { get }
    var label: String? { get }

    func setAddress(_ address: MTLGPUAddress, index: Int)
    func setAddress(_ address: MTLGPUAddress, attributeStride: Int, index: Int)
    func setResource(_ resourceID: MTLResourceID, bufferIndex: Int)
    func setTexture(_ textureID: MTLResourceID, index: Int)
    func setSamplerState(_ samplerID: MTLResourceID, index: Int)
}
```

Descriptor:
```swift
let desc = MTL4ArgumentTableDescriptor()
desc.maxBufferBindCount = 31        // max 31
desc.maxTextureBindCount = 128      // max 128
desc.maxSamplerStateBindCount = 16  // max 16
desc.initializeBindings = false     // true to zero-init
desc.supportAttributeStrides = false
desc.label = "decode_args"

let table = try device.makeArgumentTable(descriptor: desc)
```

### Usage Pattern

Argument table state is **snapshotted at dispatch time**. Use one table and re-set addresses between dispatches:

```swift
let table = try device.makeArgumentTable(descriptor: desc)

// Dispatch 1
table.setAddress(weightBuffer.gpuAddress, index: 0)
table.setAddress(hiddenBuffer.gpuAddress, index: 1)
table.setAddress(outputBuffer.gpuAddress, index: 2)
encoder.setArgumentTable(table)
encoder.setComputePipelineState(kernel1)
encoder.dispatchThreadgroups(grid1, threadsPerThreadgroup: tg1)

// Barrier
encoder.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch, visibilityOptions: [])

// Dispatch 2 — update only changed bindings
table.setAddress(outputBuffer.gpuAddress, index: 1)  // output is now input
table.setAddress(scratch.gpuAddress, index: 2)
encoder.setArgumentTable(table)
encoder.setComputePipelineState(kernel2)
encoder.dispatchThreadgroups(grid2, threadsPerThreadgroup: tg2)
```

### Buffer Address Binding

```swift
// Bind buffer at offset
table.setAddress(buffer.gpuAddress + UInt64(offset), index: bindingIndex)

// Bind constant values via pre-allocated buffer
let constantBuffer: MTLBuffer = ...  // pre-filled with UInt32, Float, etc.
table.setAddress(constantBuffer.gpuAddress + UInt64(constantOffset), index: bindingIndex)
```

---

## MTL4BufferRange

```swift
struct MTL4BufferRange {
    var bufferAddress: MTLGPUAddress
    var length: UInt64  // UInt64.max = from address to end of buffer
}
```

---

## MTLTensor

First-class tensor resource type. Can wrap existing `MTLBuffer`.

```swift
protocol MTLTensor: MTLResource {
    var gpuResourceID: MTLResourceID { get }
    var buffer: MTLBuffer? { get }
    var bufferOffset: Int { get }
    var strides: MTLTensorExtents? { get }
    var dimensions: MTLTensorExtents { get }
    var dataType: MTLTensorDataType { get }
    var usage: MTLTensorUsage { get }
}
```

Data types:
```swift
enum MTLTensorDataType {
    case float32, float16, bfloat16
    case int8, uint8, int16, uint16, int32, uint32
    case int4, uint4  // macOS 26.4+
}
```

Max rank: 16. Used with cooperative tensors in MSL 4.0.

---

## MTL4MachineLearningCommandEncoder

Runs CoreML/MPP networks on the GPU timeline.

```swift
protocol MTL4MachineLearningCommandEncoder: MTL4CommandEncoder {
    func setPipelineState(_ state: MTL4MachineLearningPipelineState)
    func setArgumentTable(_ table: MTL4ArgumentTable?)
    func dispatchNetwork(intermediatesHeap: MTLHeap)
}
```

---

## MTL4CommitFeedback

GPU timing via commit feedback (replaces command buffer `gpuStartTime`/`gpuEndTime`):

```swift
protocol MTL4CommitFeedback: NSObjectProtocol {
    var error: NSError? { get }
    var gpuStartTime: CFTimeInterval { get }
    var gpuEndTime: CFTimeInterval { get }
}
```

Usage:
```swift
let options = MTL4CommitOptions()
options.addFeedbackHandler { (feedback: MTL4CommitFeedback) in
    let gpuMs = (feedback.gpuEndTime - feedback.gpuStartTime) * 1000
    print("GPU: \(gpuMs) ms")
}
queue.commit([commandBuffer], options: options)
```

---

## MSL 4.0 (Metal Shading Language)

`__METAL_VERSION__ == 400`

Key additions:
- `cooperative_tensor<ElementType, Extents, Layout>` — cooperative tensor operations
- `tensor_inline` / `tensor_handle` — wrap buffers as tensors
- `__HAVE_INT4B_FORMAT_TYPE__` — 4-bit integer support
- `__HAVE_EXECUTION_UNIT__` — execution unit intrinsics
- `__HAVE_FUNCTION_HANDLES__` — function handle support

---

## Synchronization Hierarchy

From finest to coarsest scope:

| Mechanism | Scope | Use Case |
|---|---|---|
| Intra-pass barrier | Within same encoder | Sequential dispatches in decode chain |
| Fence | Across passes in same queue | Blit → compute dependencies |
| Consumer/Producer barrier | Coarse cross-pass | Multi-pass pipelines |
| MTLEvent | Across queues on same device | Parallel compute + render |
| MTLSharedEvent | Across devices/processes | Multi-GPU, CPU↔GPU |

**Rule**: Use the smallest scope that resolves the dependency.

---

## Lifecycle Example: LLM Decode

```swift
// === Setup (once) ===
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeMTL4CommandQueue()!
let commandBuffer = device.makeCommandBuffer()!

let kMaxInFlight = 2
let allocators = (0..<kMaxInFlight).map { _ in device.makeCommandAllocator()! }

let atDesc = MTL4ArgumentTableDescriptor()
atDesc.maxBufferBindCount = 31
let argumentTable = try device.makeArgumentTable(descriptor: atDesc)

// === Per-token decode ===
func decodeToken(frameIndex: Int, tokenID: Int32) {
    let allocator = allocators[frameIndex % kMaxInFlight]
    allocator.reset()

    commandBuffer.beginCommandBuffer(allocator: allocator)
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    // In production, use MetalDecodeEncoder.encodeSteps(plan:encoder:argumentTable:)
    // which handles binding, barriers, and dispatch. The loop below shows the
    // conceptual pattern for illustration.
    for step in decodePlan.steps {
        // Bind buffers: step.bindings.bufferBindings → argumentTable.setAddress(gpuAddress, index:)
        // Bind constants: step.bindings.constantBindings → argumentTable.setAddress(gpuAddress, index:)
        // (See MetalDecodeEncoder.bindArgumentTable for the full implementation)
        for binding in step.bindings.buffers {
            argumentTable.setAddress(
                binding.buffer.gpuAddress + UInt64(binding.offset),
                index: binding.index
            )
        }

        // Barrier (if needed) — all barriers are full stage-to-stage in Metal 4;
        // no resource-scoped variant exists.
        if step.barrierPolicy.isBarrier {
            encoder.barrier(
                afterEncoderStages: .dispatch,
                beforeEncoderStages: .dispatch,
                visibilityOptions: []  // private buffers: execution order only
            )
        }

        // Dispatch
        encoder.setArgumentTable(argumentTable)
        encoder.setComputePipelineState(step.pipeline)
        if step.threadgroupMemoryLength > 0 {
            encoder.setThreadgroupMemoryLength(step.threadgroupMemoryLength, index: 0)
        }
        encoder.dispatchThreadgroups(
            threadgroupsPerGrid: step.gridSize,
            threadsPerThreadgroup: step.threadgroupSize
        )
    }

    encoder.endEncoding()
    commandBuffer.endCommandBuffer()
    queue.commit([commandBuffer])
}
```

---

## Migration Checklist (Metal 3 → Metal 4)

| Metal 3 | Metal 4 | Notes |
|---|---|---|
| `device.makeCommandQueue()` | `device.makeMTL4CommandQueue()` | Queue from device |
| `queue.makeCommandBuffer()` | `device.makeCommandBuffer()` | Buffer from device, reusable |
| N/A | `device.makeCommandAllocator()` | Memory management |
| `cb.makeComputeCommandEncoder()` | `cb.makeComputeCommandEncoder()` | Returns `MTL4ComputeCommandEncoder` |
| `cb.makeBlitCommandEncoder()` | Merged into compute encoder | `copy()`, `fill()` on compute encoder |
| `encoder.setBuffer(buf, offset, idx)` | `argTable.setAddress(buf.gpuAddress + offset, index: idx)` | Bindless |
| `encoder.setBytes(ptr, len, idx)` | Pre-allocate in buffer, bind address | No inline bytes |
| `encoder.memoryBarrier(scope: .buffers)` | `encoder.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch, visibilityOptions: .device)` | Stage-level |
| `encoder.memoryBarrier(resources: [...])` | `encoder.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch, visibilityOptions: [])` | No resource-level variant in Metal 4. All barriers are full stage-to-stage barriers; `visibilityOptions` controls cache semantics, not resource scope |
| `cb.commit()` + `cb.waitUntilCompleted()` | `queue.commit([cb])` + feedback handler | Batch commit |
| `cb.gpuStartTime` / `cb.gpuEndTime` | `MTL4CommitFeedback.gpuStartTime/gpuEndTime` | Via feedback handler |
| `hazardTrackingModeUntracked` | Ignored on MTL4CommandQueue | All sync is explicit |
