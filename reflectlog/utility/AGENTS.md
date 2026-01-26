# Agent Guidelines for reflectlog/utility/

This directory contains cross-cutting utilities that extend beyond the application layer. It provides platform-specific functionality and general utility functions that can be used across the entire project.

## Directory Structure

```
utility/
├── __init__.py          # Package exports and public API
├── types.py             # Type definitions for utilities
├── utility.py           # General utility functions
└── platforms/           # Platform-specific implementations
    ├── __init__.py      # Factory function and exports
    ├── base.py          # CredentialRetriever abstract base class
    ├── darwin.py        # macOS credential retrieval
    ├── linux.py         # Linux credential retrieval
    └── windows.py       # Windows credential retrieval
```

## Core Responsibilities

### General Utilities

The `utility.py` module provides helper functions used throughout the project:

- Type conversion utilities
- String manipulation helpers
- File system operations
- Date and time utilities

### Platform Abstraction

The `platforms/` subpackage provides OS-specific implementations for retrieving credentials from secure system credential stores:

```python
# Factory function for platform-specific retrieval
def get_platform_retriever() -> CredentialRetriever:
    '''Get the appropriate credential retriever for the current platform.'''
    if sys.platform == "darwin":
        return DarwinCredentialRetriever()
    elif sys.platform == "linux":
        return LinuxCredentialRetriever()
    elif sys.platform == "win32":
        return WindowsCredentialRetriever()
    else:
        raise NotImplementedError(f"Unsupported platform: {sys.platform}")
```

## Key Components

### Credential Retrieval

The credential retrieval system provides secure access to API keys stored in the operating system's credential manager:

```python
class CredentialRetriever(ABC):
    '''Abstract base class for platform-specific credential retrieval.'''

    service_name: str = "Claude Code-credentials"
    token_prefix: str = "sk-ant-"

    @abstractmethod
    def get_credential(self) -> str | None:
        '''Retrieve credential from platform-specific storage.'''
        ...

    def parse_credential(self, raw: str) -> str | None:
        '''Parse credential from various formats.'''
        import json

        # Format 1: OAuth JSON
        try:
            data = json.loads(raw)
            if "claudeAiOauth" in data:
                return data["claudeAiOauth"]["accessToken"]
        except json.JSONDecodeError:
            pass

        # Format 2: Legacy JSON
        try:
            data = json.loads(raw)
            if "apiKey" in data:
                return data["apiKey"]
        except json.JSONDecodeError:
            pass

        # Format 3: Raw token
        if raw.startswith(self.token_prefix):
            return raw

        return None
```

### Platform Implementations

#### macOS (darwin.py)

Uses the `security` command-line tool:

```python
class DarwinCredentialRetriever(CredentialRetriever):
    '''Retrieve credentials from macOS Keychain.'''

    def get_credential(self) -> str | None:
        import subprocess

        try:
            result = subprocess.run(
                ["security", "find-generic-password", "-s", self.service_name, "-w"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return self.parse_credential(result.stdout.strip())
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return None
```

#### Linux (linux.py)

Uses `secret-tool` or falls back to environment variables:

```python
class LinuxCredentialRetriever(CredentialRetriever):
    '''Retrieve credentials from Linux secret service.'''

    def get_credential(self) -> str | None:
        import subprocess

        try:
            result = subprocess.run(
                ["secret-tool", "get", "service", self.service_name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return self.parse_credential(result.stdout.strip())
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Fallback to environment variable
        return os.getenv("ANTHROPIC_API_KEY")
```

#### Windows (windows.py)

Uses the Windows Credential Manager via winreg or pywin32:

```python
class WindowsCredentialRetriever(CredentialRetriever):
    '''Retrieve credentials from Windows Credential Manager.'''

    def get_credential(self) -> str | None:
        import winreg

        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\SystemCertificates\My",
            )
            # Query credential from registry
            # ...
        except WindowsError:
            pass

        return None
```

## Key Patterns

### Platform Detection

Use `sys.platform` for OS detection:

```python
import sys

def get_platform_retriever() -> CredentialRetriever:
    if sys.platform == "darwin":
        return DarwinCredentialRetriever()
    elif sys.platform == "linux":
        return LinuxCredentialRetriever()
    elif sys.platform == "win32":
        return WindowsCredentialRetriever()
    else:
        raise NotImplementedError(f"Platform not supported: {sys.platform}")
```

### Credential Parsing

Handle multiple credential formats:

```python
def parse_credential(raw: str) -> str | None:
    '''Parse credential from various formats.'''
    import json

    # Try JSON formats first
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            if "accessToken" in data:
                return data["accessToken"]
            if "apiKey" in data:
                return data["apiKey"]
    except json.JSONDecodeError:
        pass

    # Check for raw token
    if raw.startswith("sk-ant-"):
        return raw

    return None
```

### Utility Functions

Common utility patterns:

```python
def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    '''Truncate string to maximum length.'''
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def ensure_directory_exists(path: str) -> None:
    '''Ensure the directory for the given path exists.'''
    os.makedirs(os.path.dirname(path), exist_ok=True)


def get_file_size(path: str) -> int:
    '''Get file size in bytes.'''
    return os.path.getsize(path) if os.path.exists(path) else 0


def sanitize_filename(filename: str) -> str:
    '''Sanitize filename for safe file system usage.'''
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
```

## Error Handling

### Credential Retrieval Errors

Handle platform-specific errors gracefully:

```python
def get_credential(self) -> str | None:
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", self.service_name, "-w"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return self.parse_credential(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        self.logger.warning(
            "Failed to retrieve credential from Keychain",
            extra={"error": str(e)}
        )

    return None
```

### Utility Function Errors

Provide meaningful error messages:

```python
def read_json_file(path: str) -> dict:
    '''Read and parse JSON file.'''
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e
```

## Testing Guidelines

### Unit Tests

- Test credential parsing with various formats
- Test utility functions with edge cases
- Mock platform-specific behavior
- Test error handling paths

### Platform Tests

- Test platform detection logic
- Test credential retrieval with mocks
- Verify format parsing correctness

### Test Cases

```python
def test_parse_oauth_json():
    '''Should parse OAuth JSON format.'''
    raw = '{"claudeAiOauth": {"accessToken": "sk-ant-xyz"}}'
    result = parse_credential(raw)
    assert result == "sk-ant-xyz"

def test_parse_legacy_json():
    '''Should parse legacy JSON format.'''
    raw = '{"apiKey": "sk-ant-legacy"}}'
    result = parse_credential(raw)
    assert result == "sk-ant-legacy"

def test_parse_raw_token():
    '''Should parse raw token format.'''
    raw = "sk-ant-raw-token"
    result = parse_credential(raw)
    assert result == "sk-ant-raw-token"

def test_parse_invalid():
    '''Should return None for invalid format.'''
    raw = "invalid-credential"
    result = parse_credential(raw)
    assert result is None

def test_truncate_string():
    '''Should truncate long strings.'''
    text = "This is a very long string"
    result = truncate_string(text, max_length=15, suffix="...")
    assert result == "This is a very..."
    assert len(result) == 15
```

## Dependencies

### Internal Dependencies

- `application/utils/logging.py`: StructuredLogger for logging
- `application/config/`: Configuration values

### External Dependencies

- `subprocess`: For executing platform commands
- `json`: For credential format parsing
- `os`: For file system operations
- `sys`: For platform detection
- `winreg` (Windows): For Windows Credential Manager access

## Important Notes

### Security

- Never log credential values
- Use platform-specific secure storage when available
- Fall back to environment variables with warnings
- Implement proper credential parsing to avoid injection

### Platform Compatibility

- Test on all supported platforms (macOS, Linux, Windows)
- Handle missing commands gracefully
- Provide useful error messages for unsupported platforms
- Use conditional imports for platform-specific modules

### Performance

- Credential retrieval may involve subprocess calls
- Cache credentials if retrieved frequently
- Set appropriate timeouts for external commands
