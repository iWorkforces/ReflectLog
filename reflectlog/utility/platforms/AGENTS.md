# Agent Guidelines for reflectlog/utility/platforms/

This subpackage provides OS-specific implementations for retrieving Claude Code credentials from secure system credential stores. Each platform has its own retrieval mechanism using the native operating system's credential management API.

## Directory Structure

```
platforms/
├── __init__.py      # Factory function and exports
├── base.py          # CredentialRetriever abstract base class
├── darwin.py        # macOS credential retrieval using Keychain
├── linux.py         # Linux credential retrieval using secret-tool
└── windows.py       # Windows credential retrieval using Credential Manager
```

## Core Responsibilities

### Platform-Specific Credential Retrieval

Each platform implementation provides secure credential retrieval from the native credential store:

- **macOS**: Uses the `security` command-line tool to access Keychain
- **Linux**: Uses `secret-tool` (GNOME Keyring) or falls back to environment variables
- **Windows**: Uses Windows Credential Manager via registry or pywin32

### Credential Format Handling

The system handles three credential formats:

1. **OAuth JSON**: `{"claudeAiOauth": {"accessToken": "sk-ant-..."}}`
2. **Legacy JSON**: `{"apiKey": "sk-ant-..."}`
3. **Raw Token**: `"sk-ant-..."`

## Key Components

### CredentialRetriever Base Class (base.py)

Abstract base class defining the credential retrieval interface:

```python
class CredentialRetriever(ABC):
    '''Abstract base class for platform-specific credential retrieval.'''

    service_name: str = "Claude Code-credentials"
    token_prefix: str = "sk-ant-"

    @abstractmethod
    def get_credential(self) -> str | None:
        '''Retrieve credential from platform-specific storage.

        Returns:
            Parsed credential token or None if not found.
        '''
        ...

    def parse_credential(self, raw: str) -> str | None:
        '''Parse credential from various formats.

        Args:
            raw: Raw credential string from storage.

        Returns:
            Parsed token or None if invalid format.
        '''
        import json

        # Try OAuth JSON format
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                if "claudeAiOauth" in data:
                    return data["claudeAiOauth"]["accessToken"]
                if "apiKey" in data:
                    return data["apiKey"]
        except json.JSONDecodeError:
            pass

        # Try raw token format
        if raw.startswith(self.token_prefix):
            return raw

        return None
```

### macOS Implementation (darwin.py)

Uses the macOS `security` command-line tool:

```python
class DarwinCredentialRetriever(CredentialRetriever):
    '''Retrieve credentials from macOS Keychain.'''

    def get_credential(self) -> str | None:
        '''Query macOS Keychain for credentials.

        Returns:
            Parsed credential token or None if not found.
        '''
        import subprocess

        try:
            result = subprocess.run(
                [
                    "security",
                    "find-generic-password",
                    "-s", self.service_name,
                    "-w"
                ],
                capture_output=True,
                text=True,
                timeout=10,  # 10 second timeout
            )

            if result.returncode == 0:
                return self.parse_credential(result.stdout.strip())

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            # Log warning but don't raise - credential may not exist
            pass

        return None
```

### Linux Implementation (linux.py)

Uses `secret-tool` (GNOME Keyring) with environment variable fallback:

```python
class LinuxCredentialRetriever(CredentialRetriever):
    '''Retrieve credentials from Linux secret service.'''

    def get_credential(self) -> str | None:
        '''Query GNOME Keyring via secret-tool.

        Returns:
            Parsed credential token or None if not found.
        '''
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

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

        # Fallback to environment variable
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            return self.parse_credential(api_key)

        return None
```

### Windows Implementation (windows.py)

Uses Windows Credential Manager:

```python
class WindowsCredentialRetriever(CredentialRetriever):
    '''Retrieve credentials from Windows Credential Manager.'''

    def get_credential(self) -> str | None:
        '''Query Windows Credential Manager.

        Returns:
            Parsed credential token or None if not found.
        '''
        import winreg

        try:
            # Open the registry key for stored credentials
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\SystemCertificates\My",
            )

            # Query the credential value
            # Implementation details depend on how credentials are stored

        except WindowsError:
            pass

        return None
```

## Factory Function (__init__.py)

The package exports a factory function to create the appropriate retriever:

```python
def get_platform_retriever() -> CredentialRetriever:
    '''Get the appropriate credential retriever for the current platform.

    Returns:
        CredentialRetriever instance for the current platform.

    Raises:
        NotImplementedError: If the platform is not supported.
    '''
    import sys

    if sys.platform == "darwin":
        return DarwinCredentialRetriever()
    elif sys.platform == "linux":
        return LinuxCredentialRetriever()
    elif sys.platform == "win32":
        return WindowsCredentialRetriever()
    else:
        raise NotImplementedError(
            f"Platform not supported: {sys.platform}"
        )
```

## Usage Examples

### Basic Usage

```python
from reflectlog.utility.platforms import get_platform_retriever

# Get the appropriate retriever for the current platform
retriever = get_platform_retriever()

# Retrieve credential
credential = retriever.get_credential()

if credential:
    print(f"Found credential: {credential[:10]}...")
else:
    print("No credential found")
```

### Credential Parsing

```python
# The parse_credential method handles multiple formats
retriever = get_platform_retriever()

# OAuth JSON format
raw = '{"claudeAiOauth": {"accessToken": "sk-ant-oauth-token"}}'
parsed = retriever.parse_credential(raw)
# Returns: "sk-ant-oauth-token"

# Legacy JSON format
raw = '{"apiKey": "sk-ant-legacy-token"}}'
parsed = retriever.parse_credential(raw)
# Returns: "sk-ant-legacy-token"

# Raw token format
raw = "sk-ant-raw-token"
parsed = retriever.parse_credential(raw)
# Returns: "sk-ant-raw-token"
```

### Integration with Configuration

```python
def get_api_key() -> str | None:
    '''Get API key from platform storage or environment.'''
    # Try platform-specific storage first
    try:
        retriever = get_platform_retriever()
        credential = retriever.get_credential()
        if credential:
            return credential
    except NotImplementedError:
        pass

    # Fallback to environment variable
    return os.getenv("ANTHROPIC_API_KEY")
```

## Error Handling

### Graceful Degradation

All implementations handle errors gracefully:

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
        # Command not available or failed - not an error
        pass

    return None
```

### Timeout Handling

All external commands use timeouts to prevent hanging:

```python
result = subprocess.run(
    ["security", "find-generic-password", "-s", self.service_name, "-w"],
    capture_output=True,
    text=True,
    timeout=10,  # 10 second maximum
)
```

## Testing Guidelines

### Unit Tests

- Test credential parsing with various formats
- Test factory function platform detection
- Mock platform-specific behavior
- Test error handling paths

### Platform-Specific Tests

```python
import pytest
from unittest.mock import MagicMock, patch
from reflectlog.utility.platforms import get_platform_retriever
from reflectlog.utility.platforms.darwin import DarwinCredentialRetriever

def test_parse_oauth_json():
    '''Should parse OAuth JSON format correctly.'''
    retriever = DarwinCredentialRetriever()
    raw = '{"claudeAiOauth": {"accessToken": "sk-ant-test"}}'
    result = retriever.parse_credential(raw)
    assert result == "sk-ant-test"

def test_parse_legacy_json():
    '''Should parse legacy JSON format correctly.'''
    retriever = DarwinCredentialRetriever()
    raw = '{"apiKey": "sk-ant-legacy"}}'
    result = retriever.parse_credential(raw)
    assert result == "sk-ant-legacy"

def test_parse_raw_token():
    '''Should parse raw token format correctly.'''
    retriever = DarwinCredentialRetriever()
    raw = "sk-ant-raw"
    result = retriever.parse_credential(raw)
    assert result == "sk-ant-raw"

def test_parse_invalid():
    '''Should return None for invalid format.'''
    retriever = DarwinCredentialRetriever()
    raw = "invalid-credential"
    result = retriever.parse_credential(raw)
    assert result is None

@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_darwin_retriever_integration():
    '''Integration test for macOS credential retrieval.'''
    retriever = DarwinCredentialRetriever()
    # May return None if no credential stored
    result = retriever.get_credential()
    assert result is None or result.startswith("sk-ant-")

@patch('subprocess.run')
def test_darwin_retriever_with_mock(mock_run):
    '''Test macOS retriever with mocked subprocess.'''
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout='{"claudeAiOauth": {"accessToken": "sk-ant-mock"}}',
    )

    retriever = DarwinCredentialRetriever()
    result = retriever.get_credential()

    assert result == "sk-ant-mock"
    mock_run.assert_called_once()
```

## Dependencies

### Internal Dependencies

- `utility/types.py`: Type definitions (if used)

### External Dependencies

- `subprocess`: For executing platform commands (macOS, Linux)
- `os`: For environment variable access (Linux fallback)
- `winreg`: For Windows Credential Manager access (Windows)
- `json`: For credential format parsing

### Platform-Specific Tools

| Platform | Tool Required | Package |
|----------|---------------|---------|
| macOS | `security` | Built-in (macOS) |
| Linux | `secret-tool` | `libsecret-tools` (Debian/Ubuntu) |
| Windows | `winreg` | Built-in (Python) |

## Important Notes

### Security

- Never log or expose credential values
- Use platform-native secure storage when available
- Credentials should be retrieved only when needed
- Implement proper parsing to avoid injection attacks

### Platform Compatibility

- Test on all target platforms (macOS, Linux, Windows)
- Provide graceful fallbacks for missing tools
- Handle missing permissions or access denied scenarios
- Consider multiple Linux distributions (GNOME Keyring vs KWallet)

### Performance

- Credential retrieval involves external process execution
- Consider caching credentials in memory if retrieved frequently
- Set appropriate timeouts to prevent hanging
- Background refresh may be appropriate for long-running processes

### Credential Format Evolution

The system handles multiple formats for backward compatibility:

- New OAuth format (preferred)
- Legacy API key format (deprecated but supported)
- Raw token format (for direct token storage)

This allows for gradual migration without breaking existing installations.
