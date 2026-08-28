# Security Tests

**Generated:** 2026-08-26  **Commit:** 95567fa  **Branch:** develop

## OVERVIEW

Dedicated security tests for SQL injection, path traversal, and input validation. Critical for production safety.

## STRUCTURE

```
tests/security/
└── test_security.py        # SQL injection, path traversal, malicious input
```

## WHERE TO LOOK

| Test | Purpose |
|------|---------|
| test_sql_injection_* | SQL injection pattern detection |
| test_path_traversal_* | WORKSPACE_ID path traversal prevention |
| test_malicious_input | Edge cases, unicode attacks |

## KEY PATTERNS

### SQL Injection Patterns
```python
SQL_INJECTION_PATTERNS = [
    "'; DROP TABLE memories; --",
    "' OR '1' = '1",
    "1; DELETE FROM messages WHERE 1=1",
    "' UNION SELECT * FROM users --",
    "admin'--",
]
```

### Path Traversal Prevention
```python
PATH_TRAVERSAL_INPUTS = [
    "../../../etc/passwd",
    "..\\..\\..\\windows\\system32",
    "....//....//etc/shadow",
    "%2e%2e%2f%2e%2e%2f",
]
```

### Validation Test
```python
@pytest.mark.parametrize("malicious", SQL_INJECTION_PATTERNS)
def test_sql_injection_blocked(malicious):
    with pytest.raises(ValidationError, match="SQL injection"):
        validate_messages([malicious])
```

## ANTI-PATTERNS

- Never skip security tests before release
- Never test with sanitized inputs only

## NOTES

- **Critical**: These tests must pass 100%
- **Regex patterns**: `WORKSPACE_ID_PATTERN = r"^[A-Za-z0-9_.-]{1,64}$"`
