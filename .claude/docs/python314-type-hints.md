# Python 3.14 Type Hints Best Practices

## Native Union Syntax

**Use `|` instead of `Union` or `Optional`:**

```python
# ✅ Python 3.14
def process(query: str | None) -> list[str] | None:
    pass

# ❌ Legacy
from typing import Optional, List
def process(query: Optional[str]) -> Optional[List[str]]:
    pass
```

## Built-in Generics

**Use `list[T]`, `dict[K,V]`, `tuple[T,...]` instead of `List[T]`, `Dict[K,V]`, `Tuple`:**

```python
# ✅ Python 3.14
def search(query: str) -> list[tuple[str, float]]:
    pass

# ❌ Legacy
from typing import List, Tuple
def search(query: str) -> List[Tuple[str, float]]:
    pass
```

## Keep These Imports

Some typing imports remain valid in Python 3.14:

```python
from typing import (
    TYPE_CHECKING,  # For circular imports
    Protocol,       # Structural subtyping
    runtime_checkable,  # Runtime protocol checks
    TypeVar,        # Generic type variables
    Generic,        # Generic base classes
    TypeAlias,      # Explicit type aliases
    Annotated,      # Metadata
    Literal,        # Literal types
    TypedDict,      # Typed dicts (not dataclasses)
    Callable,       # Callable types
)
```

## Still Required: TYPE_CHECKING Guard

For circular imports, use `TYPE_CHECKING` for type-only imports:

```python
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from reflectlog.application.memory import MemoryManager

class IMemoryManager(Protocol):
    def get_all(self) -> list[str]: ...
```

## Avoid These

- ❌ `Optional[...]` → use `... | None`
- ❌ `Union[...]` → use `... | ...`
- ❌ `List[...]` → use `list[...]`
- ❌ `Dict[...]` → use `dict[...]`
- ❌ `Tuple[...]` → use `tuple[...]`
- ❌ `Set[...]` → use `set[...]`

## Key Benefits

1. **No runtime overhead** - Deferred evaluation means no `from __future__ import annotations`
2. **Cleaner code** - Less import boilerplate
3. **Better IDE support** - Native syntax recognized by all tools
