import Foundation

struct STAFMetadataTableEntry: Sendable {
    let keyOffset: UInt32
    let keyLength: UInt32
    let valueType: STAFMetadataValueType
    let payload0: UInt64
    let payload1: UInt64
}
