# Utility Unit Tests

**Generated:** 2026-02-22
**Commit:** 6c2d6fa
**Branch:** develop

## OVERVIEW

Unit tests for platform-specific credential retrieval utilities. Mocked subprocess calls.

## STRUCTURE

```
tests/unit/utility/
└── test_utility.py         # Credential parsing, platform factory
```

## WHERE TO LOOK

| Test | Purpose |
|------|---------|
| test_utility.py | `get_platform_retriever()`, credential format parsing |

## KEY PATTERNS

### Platform Factory Mock
```python
@patch('platform.system')
def test_darwin_factory(mock_system):
    mock_system.return_value = "Darwin"
    retriever = get_platform_retriever()
    assert isinstance(retriever, DarwinCredentialRetriever)
```

### Credential Format Parsing
```python
def test_parse_oauth_json():
    raw = '{"claudeAiOauth": {"accessToken": "sk-ant-test"}}'
    result = retriever.parse_credential(raw)
    assert result == "sk-ant-test"
```

## ANTI-PATTERNS

- Never call real `security` or `secret-tool` commands
- Never log credential values in tests

## NOTES

- **Subprocess mocked**: All platform commands return test data
- **Graceful degradation**: Tests verify `None` returned on errors
