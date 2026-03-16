/// Diagnostic metadata for a `ModelGraph`, stored as a sidecar.
///
/// `ModelGraphMetadata` contains non-semantic information (labels, source
/// locations) that is useful for diagnostics but does NOT affect canonical
/// identity. It is separated from `ModelGraph` so that equivalence
/// comparison operates on pure semantic structure.
public struct ModelGraphMetadata: Codable, Equatable, Sendable {

    /// Annotations keyed by structural path.
    public let annotations: [AnnotationEntry]

    public init(annotations: [AnnotationEntry] = []) {
        self.annotations = annotations
    }

    /// Look up the annotation for a given structural path.
    public func annotation(for path: StructuralPath) -> OperationAnnotation? {
        annotations.first { $0.path == path }?.annotation
    }
}

/// A single annotation entry associating a structural path with metadata.
public struct AnnotationEntry: Codable, Equatable, Sendable {

    /// Structural path to the annotated operation.
    public let path: StructuralPath

    /// The annotation.
    public let annotation: OperationAnnotation

    public init(path: StructuralPath, annotation: OperationAnnotation) {
        self.path = path
        self.annotation = annotation
    }
}

/// Diagnostic annotation for an operation.
///
/// Contains non-semantic metadata that does not affect canonical identity.
public struct OperationAnnotation: Codable, Equatable, Sendable {

    /// Human-readable label for debugging and diagnostics.
    public let label: String?

    public init(label: String? = nil) {
        self.label = label
    }
}
