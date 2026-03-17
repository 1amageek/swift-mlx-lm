/// Result builder for composing `MetalKernelFragment` values.
///
/// Mirrors `ModelComponentBuilder` — enables declarative fragment composition:
/// ```swift
/// func fragment(context: KernelContext) -> some MetalKernelFragment {
///     Linear(field: "gate_proj", input: 2048, output: 8192)
///     Linear(field: "up_proj", input: 2048, output: 8192)
///     SwiGLU(dimension: 8192)
///     Linear(field: "down_proj", input: 8192, output: 2048)
/// }
/// ```
@resultBuilder
public enum MetalKernelFragmentBuilder {

    public static func buildBlock<C: MetalKernelFragment>(_ component: C) -> C {
        component
    }

    public static func buildBlock<each C: MetalKernelFragment>(
        _ components: repeat each C
    ) -> TupleFragment<repeat each C> {
        TupleFragment(repeat each components)
    }

    public static func buildOptional<C: MetalKernelFragment>(_ component: C?) -> OptionalFragment<C> {
        OptionalFragment(component)
    }

    public static func buildEither<First: MetalKernelFragment, Second: MetalKernelFragment>(
        first component: First
    ) -> ConditionalFragment<First, Second> {
        .first(component)
    }

    public static func buildEither<First: MetalKernelFragment, Second: MetalKernelFragment>(
        second component: Second
    ) -> ConditionalFragment<First, Second> {
        .second(component)
    }
}

/// Optional fragment — nil when the condition is false.
public struct OptionalFragment<Content: MetalKernelFragment>: MetalKernelFragment, _OptionalFragmentProtocol {
    public let content: Content?

    public init(_ content: Content?) {
        self.content = content
    }

    public func fragment(context: KernelContext) -> Never { fatalError() }
    public var isFusable: Bool { content?.isFusable ?? false }

    public func _visitContent(_ visitor: (any MetalKernelFragment) -> Void) {
        if let content { visitor(content) }
    }
}

/// Conditional fragment — either first or second branch.
public enum ConditionalFragment<First: MetalKernelFragment, Second: MetalKernelFragment>: MetalKernelFragment, _ConditionalFragmentProtocol {
    case first(First)
    case second(Second)

    public func fragment(context: KernelContext) -> Never { fatalError() }
    public var isFusable: Bool {
        switch self {
        case .first(let f): return f.isFusable
        case .second(let s): return s.isFusable
        }
    }

    public func _visitActive(_ visitor: (any MetalKernelFragment) -> Void) {
        switch self {
        case .first(let f): visitor(f)
        case .second(let s): visitor(s)
        }
    }
}

// MARK: - Internal Visitor Protocols

/// Internal protocol for accessing a fragment's body (generic fragment → child).
public protocol _FragmentBodyAccessor {
    func _visitBody(context: KernelContext, _ visitor: (any MetalKernelFragment) -> Void)
}

/// Internal protocol for visiting TupleFragment children.
public protocol _TupleFragmentProtocol {
    func _visitChildren(_ visitor: (any MetalKernelFragment) -> Void)
}

/// Internal protocol for visiting OptionalFragment content.
public protocol _OptionalFragmentProtocol {
    func _visitContent(_ visitor: (any MetalKernelFragment) -> Void)
}

/// Internal protocol for visiting ConditionalFragment active branch.
public protocol _ConditionalFragmentProtocol {
    func _visitActive(_ visitor: (any MetalKernelFragment) -> Void)
}

/// A tuple of fragments produced by the result builder.
public struct TupleFragment<each Child: MetalKernelFragment>: MetalKernelFragment, _TupleFragmentProtocol {

    public let children: (repeat each Child)

    public init(_ children: repeat each Child) {
        self.children = (repeat each children)
    }

    public func fragment(context: KernelContext) -> Never { fatalError() }
    public var isFusable: Bool { false }

    public func _visitChildren(_ visitor: (any MetalKernelFragment) -> Void) {
        repeat visitor(each children)
    }
}
