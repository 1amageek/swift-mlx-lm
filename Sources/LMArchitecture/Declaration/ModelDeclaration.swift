// This file previously contained `ModelDeclaration` and `PrimitiveDeclaration`.
// Those enums have been eliminated — all declarations are now `ModelComponent` types.
//
// - Sequential composition: `TupleComponent` (see TupleComponent.swift)
// - Control flow: `OptionalComponent`, `ConditionalComponent`
// - Primitives: MLP, Attention, RMSNorm, etc. (see Components/)
// - Structural: Residual, Parallel, Repeat, Group (see Components/)
//
// The `SemanticNormalizer` dispatches via `_BuiltinComponent` protocol
// (file-private in SemanticNormalizer.swift) instead of enum case matching.
