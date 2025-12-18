# stubs/

This directory contains type stub files (.pyi) for external libraries that lack type annotations.

## Structure

```
stubs/
├── fastmcp/               # Type stubs for fastmcp library
│   ├── __init__.pyi
│   ├── client/            # Client stubs
│   │   ├── __init__.pyi
│   │   └── client.pyi
│   └── utilities/         # Utility stubs
│       ├── __init__.pyi
│       └── logging.pyi
├── libsql/                # Type stubs for libsql library
│   └── __init__.pyi
├── numba/                 # Type stubs for numba JIT compiler
│   ├── __init__.pyi
│   └── core/
│       ├── __init__.pyi
│       └── errors.pyi
├── ranx/                  # Type stubs for ranx library
│   └── __init__.pyi
└── usearch/               # Type stubs for usearch library
    ├── __init__.pyi
    └── index.pyi          # Index, Match, Matches classes
```

## Purpose

Type stubs provide type information for libraries that:
1. Don't include type annotations in their source
2. Don't provide `py.typed` marker
3. Don't have stubs published to typeshed or PyPI

This enables:
- Full type checking with ty strict rules
- Better IDE autocomplete and type hints
- Early detection of API misuse
- Documentation of library interfaces

## Why Stubs?

Several external libraries used by this project don't include comprehensive type annotations. This project uses ty with strict rules, which requires type information for all dependencies.

Without stubs, ty would report errors about missing type information for external libraries.

**Libraries requiring stubs:**
- `usearch`: HNSW vector search library
- `ranx`: RRF fusion ranking library
- `fastmcp`: MCP server framework
- `libsql`: High-performance SQLite fork with MVCC
- `numba`: JIT compiler for Python numerical code

## Configuring ty

The `pyproject.toml` tells ty to look for stubs via `extra-paths`:

```toml
[tool.ty.environment]
extra-paths = ["stubs"]
```

This configuration is in `pyproject.toml` under `[tool.ty.environment]`.

## Creating Type Stubs

### When to Create Stubs

Create stubs when:
- Library lacks type annotations
- ty reports missing type information
- You need type checking for library APIs you use

### Stub File Conventions

1. **File structure mirrors imports**:
   - Import: `from usearch.index import Index`
   - Stub: `stubs/usearch/index.pyi`

2. **Stub syntax**:
```python
# stubs/package/__init__.pyi
from typing import Any

class SomeClass:
    def method(self, arg: str) -> int: ...

def function(param: Any) -> None: ...
```

3. **Use `...` for method bodies**:
```python
def method(self) -> str: ...  # Not 'pass'
```

4. **Be precise or use `Any`**:
```python
# If you know the type
def get_data(self) -> list[str]: ...

# If you don't know
def get_data(self) -> Any: ...
```

### Stub Completeness

You only need to stub:
- **APIs you actually use** in your code
- **Public interfaces** (not internal implementation)
- **Common use cases** for the library

You don't need to stub:
- Internal/private APIs (`_internal_method`)
- Unused parts of the library
- Complex type relationships (use `Any` if too complex)

## usearch Stubs

The `stubs/usearch/` directory provides types for:

1. **Index class** (`stubs/usearch/index.pyi`):
   - `__init__(ndim, metric, dtype, connectivity)`: Initialize HNSW index
   - `add(key: int, vector: ndarray)`: Add vector with key
   - `search(vector: ndarray, count: int) -> Matches`: Search for similar vectors
   - `save(path: str)`: Save index to disk
   - `restore(path: str) -> Index`: Load index from disk
   - `remove(key: int)`: Remove vector by key

2. **Matches class**: Search result container
   - `.keys`: Array of matching key IDs
   - `.distances`: Array of distances
   - `.count`: Number of results

3. **Match class**: Single search result
   - `.key`: Key ID of matching vector
   - `.distance`: Distance to query vector

## ranx Stubs

The `stubs/ranx/__init__.pyi` file provides types for:

1. **Run class**: Ranked result list for fusion
   - `__init__(run: dict[str, dict[str, float]])`: Initialize from doc scores
   - Result format: `{query_id: {doc_id: score}}`

2. **fuse function**: RRF fusion algorithm
   - `fuse(runs: list[Run], method: str, k: int) -> Run`
   - Supports methods: "rrf", "sum", "mnz", "max", "bordafuse"

## fastmcp Stubs

The `stubs/fastmcp/` directory provides types for the MCP server framework:

1. **FastMCP class**: MCP server implementation
2. **Client class**: MCP client for testing
3. **Logging utilities**: Structured logging helpers

## libsql Stubs

The `stubs/libsql/__init__.pyi` file provides types for the libSQL database client:

1. **connect function**: Create database connection
   - `connect(path: str) -> Connection`: Connect to local database

2. **Connection class**: Database connection
   - `execute(sql: str, params: tuple) -> Cursor`: Execute SQL
   - `commit()`: Commit transaction
   - `close()`: Close connection

3. **Cursor class**: Query result cursor
   - `fetchone() -> tuple | None`: Fetch single row
   - `fetchall() -> list[tuple]`: Fetch all rows
   - `lastrowid: int`: Last inserted row ID

## numba Stubs

The `stubs/numba/` directory provides types for the Numba JIT compiler:

1. **jit decorator** (`__init__.pyi`):
   - `jit(nopython: bool, cache: bool, fastmath: bool, parallel: bool)`: JIT compile a function
   - Supports various compilation modes

2. **prange function** (`__init__.pyi`):
   - `prange(start: int, stop: int, step: int)`: Parallel range for loops
   - Used with `parallel=True` in jit decorator

3. **core.errors** (`core/errors.pyi`):
   - `TypingError`: Type inference errors
   - `NumbaError`: Base Numba exception

## Testing Stubs

### Verify ty recognizes stubs
```bash
# Should pass with no errors
./start-type-check.sh

# Check specific file
uv run ty check openmemories/infrastructure/usearch_engine.py
```

### Verify stubs match actual API
```python
# Test script: test_usearch_stubs.py
from usearch.index import Index
import numpy as np

# Should match stub signatures
index = Index(ndim=128, metric="cos", dtype="f32", connectivity=16)
vector = np.random.rand(128).astype(np.float32)
index.add(1, vector)

# Search
results = index.search(vector, 5)
for i in range(results.count):
    key = results.keys[i]       # Should have .keys
    dist = results.distances[i]  # Should have .distances
```

## Updating Stubs

When to update stubs:
1. **Library is updated**: New version adds/changes APIs
2. **Type errors**: Type checking fails due to incorrect stubs
3. **Using new APIs**: You start using previously unstubbed features

How to update:
1. Check library documentation for API signatures
2. Update `.pyi` file with correct types
3. Run type checking: `./start-type-check.sh`
4. Commit updated stubs

## Alternative to Manual Stubs

### stubgen (Automatic Stub Generation)
```bash
# Generate stubs automatically
uv run stubgen -p usearch -o stubs/

# Review and edit generated stubs
# May need manual refinement
```

### Third-party Stubs
```bash
# Check if community stubs exist
uv pip install types-usearch

# If available, you may not need manual stubs
```

### py.typed Marker
If library adds `py.typed` marker in future versions, you can remove manual stubs:
```bash
# Check if library has py.typed
python -c "import usearch; print(usearch.__path__)"
ls <usearch_path>/py.typed

# If present, remove stubs/usearch/
rm -rf stubs/usearch/
```

## Best Practices

1. **Start minimal**: Only stub what you use
2. **Be accurate**: Check library docs for correct signatures
3. **Use Any when uncertain**: Better than incorrect types
4. **Document assumptions**: Add comments for complex types
5. **Keep updated**: Update when library changes
6. **Test thoroughly**: Ensure stubs match runtime behavior
7. **Version control**: Commit stubs to repository
8. **Share with community**: Consider contributing to typeshed

## Troubleshooting

### ty still reports missing stubs
```bash
# Check extra-paths in pyproject.toml
grep -A2 "\[tool.ty.environment\]" pyproject.toml

# Verify stubs directory structure
ls -R stubs/

# Ensure __init__.pyi exists
ls stubs/usearch/__init__.pyi
ls stubs/ranx/__init__.pyi
```

### Type errors from stubs
```bash
# Check stub signatures match actual API
python3 -c "from usearch.index import Index; help(Index)"

# Update stubs to match
# Run type checking
./start-type-check.sh
```

### IDE not recognizing stubs
```bash
# Restart language server
# In VS Code: Cmd+Shift+P → "Reload Window"

# Check IDE uses ty settings
# Point to pyproject.toml
```

## Resources

- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [PEP 561 - Distributing Type Information](https://peps.python.org/pep-0561/)
- [ty - Astral Type Checker](https://docs.astral.sh/ty/)
- [typeshed - Repository of stubs](https://github.com/python/typeshed)
