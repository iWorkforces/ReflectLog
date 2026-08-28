"""ReflectLog Core Package.

This package contains protocol definitions and abstractions that define the
interface contracts for the application layer. All components depend on
these protocols rather than concrete implementations, enabling:

- Runtime component substitution
- Compile-time type checking
- Dependency injection
- Testability through mock implementations

Modules:
    config: Configuration protocols
    memory: Memory operation protocols
    types: Canonical search/memory types (ISemanticSearchEngine)
    reranking: Reranker protocols
    tools: Tool registration protocols
    logging: Logging protocols
    prompts: LLM prompt constants
"""
