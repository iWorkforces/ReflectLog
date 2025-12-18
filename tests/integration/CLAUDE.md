# tests/integration/

This directory contains integration tests that verify the interaction between components and external systems.

## Structure

```
integration/
├── __init__.py
├── test_mcp_workflows.py             # Full MCP tool workflows
├── test_memory_manager_usearch.py    # USearch engine integration
└── test_qwen_embeddings_integration.py # Qwen embeddings integration
```

## Purpose

Integration tests verify:
- End-to-end flows through multiple components
- Real USearch vector store operations
- Real Tantivy full-text search operations
- Real or stubbed OpenRouter API interactions
- Data persistence across operations
- Full MCP tool workflows

## Integration vs Unit Tests

| Aspect | Unit Tests | Integration Tests |
|--------|-----------|-------------------|
| Speed | < 1 second | Several seconds |
| Isolation | Mocked dependencies | Real or stubbed services |
| Scope | Single component | Multiple components |
| Failure diagnosis | Easy (single point) | Harder (multiple points) |
| Purpose | Verify logic | Verify integration |

## Current Test Files

| File | Tests For |
|------|-----------|
| `test_mcp_workflows.py` | End-to-end MCP tool workflows |
| `test_memory_manager_usearch.py` | USearch engine integration, persistence |
| `test_qwen_embeddings_integration.py` | Qwen embedding provider integration |

## Test Environment Setup

### Environment Variables
```python
import os
import pytest

@pytest.fixture(scope="module")
def integration_env():
    """Set up environment for integration tests."""
    env = {
        'PROJECT_ID': 'test-integration-project',
        'OPENROUTER_API_KEY': os.getenv('OPENROUTER_API_KEY', 'sk-test-key'),
        'SEARCH_LIMIT': '10',
        'RERANKER_ENGINE': 'none',  # Disable for faster tests
    }
    with patch.dict(os.environ, env):
        yield

@pytest.fixture(scope="function")
def clean_usearch_index():
    """Clean up USearch index before and after each test."""
    index_path = "indexes/test-integration-project/usearch"

    # Clean before
    if os.path.exists(index_path):
        shutil.rmtree(index_path)

    yield

    # Clean after
    if os.path.exists(index_path):
        shutil.rmtree(index_path)
```

### OpenAI API Mocking (Optional)
```python
# Option 1: Use real API (slower, costs money)
# Set OPENAI_API_KEY in environment

# Option 2: Use VCR.py for recording/replaying requests
import vcr

@pytest.fixture
def vcr_cassette():
    with vcr.use_cassette('fixtures/vcr_cassettes/test_name.yaml'):
        yield

# Option 3: Mock OpenAI client
@pytest.fixture
def mock_openai():
    with patch('openai.OpenAI') as mock:
        # Configure mock responses
        yield mock
```

## Test Scenarios

### End-to-End Workflows

**Test: Full memory lifecycle**
```python
@pytest.mark.integration
def test_full_memory_lifecycle(integration_env, clean_usearch_index):
    """Test add → get_all → search → remove workflow."""
    server = FastMCPServer()

    # Add messages
    messages = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Go is great for system programming"
    ]
    server.add(messages)

    # Retrieve all
    all_messages = server.get_all()
    assert len(all_messages) == 3
    assert set(all_messages) == set(messages)

    # Search
    results = server.search("Python")
    assert len(results) == 1
    assert "Python" in results[0]

    # Remove
    server.remove([messages[0]])

    # Verify removal
    remaining = server.get_all()
    assert len(remaining) == 2
    assert messages[0] not in remaining
```

**Test: Persistence across server restarts**
```python
@pytest.mark.integration
@pytest.mark.slow
def test_persistence_across_restarts(integration_env, clean_usearch_index):
    """Data should persist in USearch index across server instances."""
    messages = ["Persistent message 1", "Persistent message 2"]

    # First server instance - add data
    server1 = FastMCPServer()
    server1.add(messages)
    del server1  # Simulate server shutdown

    # Second server instance - should load existing data
    server2 = FastMCPServer()
    results = server2.get_all()

    assert len(results) == 2
    assert set(results) == set(messages)
```

### Search Behavior

**Test: Semantic similarity**
```python
@pytest.mark.integration
def test_semantic_search_finds_similar_concepts():
    """Search should find semantically similar messages."""
    server = FastMCPServer()

    messages = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Neural networks are inspired by biological brains"
    ]
    server.add(messages)

    # Search for AI-related concept
    results = server.search("AI")

    # Should find the AI-related messages (semantic similarity)
    # Note: Results depend on embeddings and may vary
    assert len(results) >= 1
```

**Test: Hybrid search (semantic + substring)**
```python
@pytest.mark.integration
def test_hybrid_search_semantic_and_substring():
    """Search combines semantic similarity AND substring matching."""
    server = FastMCPServer()

    messages = [
        "Python programming tutorial",
        "Advanced Python techniques",
        "Java programming basics",
    ]
    server.add(messages)

    # Search for "Python" - should find semantically similar AND containing substring
    results = server.search("Python")

    # Should only return messages containing "Python"
    assert len(results) == 2
    assert all("Python" in result for result in results)
```

### Edge Cases

**Test: Large message batches**
```python
@pytest.mark.integration
@pytest.mark.slow
def test_large_batch_add():
    """Test adding large number of messages."""
    server = FastMCPServer()

    # Create 1000 messages
    messages = [f"Message number {i}" for i in range(1000)]

    server.add(messages)

    all_messages = server.get_all()
    assert len(all_messages) == 1000
```

**Test: Unicode and special characters**
```python
@pytest.mark.integration
def test_unicode_and_special_characters():
    """Test messages with unicode and special characters."""
    server = FastMCPServer()

    messages = [
        "Hello 世界 🌍",
        "Καλημέρα κόσμε",
        "Message with emoji 🚀🎉",
        "Special chars: @#$%^&*()",
    ]

    server.add(messages)

    all_messages = server.get_all()
    assert len(all_messages) == 4

    # Search should work with unicode
    results = server.search("世界")
    assert len(results) == 1
```

### Performance Optimizations

**Test: Phased parallel add**
```python
@pytest.mark.integration
async def test_phased_parallel_add():
    """Test 3-phase parallel add processing."""
    manager = MemoryManager(config, logger)

    # Add messages that trigger parallel processing
    messages = [f"Message {i}" for i in range(10)]
    results = await manager.add_messages_async(messages)

    # Verify all messages added with parallel processing
    all_messages = manager.get_all()
    assert len(all_messages) == 10
```

**Test: Tantivy soft-delete integration**
```python
@pytest.mark.integration
def test_tantivy_soft_delete():
    """Test soft-delete with tombstone marking."""
    manager = MemoryManager(config, logger)

    # Add and then delete messages
    manager.add_messages(["Message to delete"])
    manager.remove(["Message to delete"])

    # Verify soft-delete (message should not appear in search)
    results = manager.search("Message to delete")
    assert len(results) == 0
```

**Test: Adaptive overfetch**
```python
@pytest.mark.integration
def test_adaptive_overfetch_small_index():
    """Test adaptive overfetch with small index."""
    manager = MemoryManager(config, logger)

    # Small index should use higher overfetch multiplier
    messages = [f"Message {i}" for i in range(10)]
    manager.add_messages(messages)

    # Verify search uses adaptive overfetch
    results = manager.search("Message")
    assert len(results) <= config.search_limit
```

### Error Handling

**Test: Invalid configuration**
```python
@pytest.mark.integration
def test_invalid_openai_key_raises_error():
    """Invalid OpenAI key should raise error during operations."""
    with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'invalid-key'}):
        server = FastMCPServer()

        with pytest.raises(RuntimeError):
            server.add(["test message"])
```

## Running Integration Tests

```bash
# All integration tests
uv run pytest tests/integration/ -v

# With marker
uv run pytest -m integration -v

# Exclude slow tests
uv run pytest tests/integration/ -v -m "integration and not slow"

# Specific test file
uv run pytest tests/integration/test_mcp_server_e2e.py -v

# With output (see prints)
uv run pytest tests/integration/ -v -s

# Stop on first failure
uv run pytest tests/integration/ -v -x
```

## Test Markers

```python
@pytest.mark.integration  # Mark as integration test
@pytest.mark.slow         # Mark as slow test (> 5 seconds)

# Example:
@pytest.mark.integration
@pytest.mark.slow
def test_large_dataset():
    pass
```

## Performance Considerations

- **Test duration**: Integration tests may take 30+ seconds total
- **OpenRouter API costs**: Using real API incurs costs (~$0.0001 per request)
- **USearch index size**: Large datasets create larger index files
- **Cleanup**: Always clean up USearch indices in teardown

## Best Practices

1. **Use fixtures for setup/teardown**: Clean USearch indices, set environment
2. **Stub expensive operations**: Mock OpenAI calls when possible
3. **Test realistic scenarios**: Use real-world data patterns
4. **Keep tests independent**: Each test should clean up after itself
5. **Mark slow tests**: Use `@pytest.mark.slow` for tests > 5 seconds
6. **Use descriptive names**: `test_persistence_across_restarts` not `test_persistence`
7. **Document assumptions**: Note when real API keys are required

## Debugging Integration Tests

```bash
# Verbose output with prints
uv run pytest tests/integration/test_mcp_server_e2e.py -v -s

# Leave USearch index for inspection (comment out cleanup)
# Then inspect: ls -la indexes/test-integration-project/usearch/

# Use debugger
uv run pytest tests/integration/ -v --pdb

# Show local variables on failure
uv run pytest tests/integration/ -v -l
```

## CI/CD Considerations

Integration tests in CI:
- Require OpenRouter API key (set as secret)
- May be slow (consider parallel execution)
- Should run on every PR
- Consider caching USearch indices for speed

Example GitHub Actions:
```yaml
- name: Run integration tests
  env:
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
    PROJECT_ID: test-integration
  run: |
    uv run pytest tests/integration/ -v -m integration
```
