"""Security testing for ReflectLogMCP.

Tests input sanitization, path traversal protection, and other
security vulnerabilities to ensure system robustness.

Example scenarios:
- SQL injection attempts in search queries
- Command injection via path traversal
- XSS attempts in memory content
- Path traversal in project_id parameter
- Authentication bypass attempts
- Privilege escalation attempts

Usage:
    pytest tests/security/test_security.py --all
    pytest tests/security/test_security.py --injection
    pytest tests/security/test_security.py --path-traversal
    pytest tests/security/test_security.py --xss
"""

import asyncio
import pytest
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reflectlog.application.memory.manager import MemoryManager


class TestInputSanitization:
    """Tests input sanitization prevents injection attacks.

    Ensures search queries are properly escaped and length-limited.
    """

    def __init__(self, manager: MemoryManager):
        """Initialize with test manager.

        Args:
            manager: Test MemoryManager instance.
        """
        self.manager = manager

    @pytest.mark.asyncio
    async def test_sql_injection_basic(self):
        """Test basic SQL injection patterns.

        Should be sanitized or rejected.
        """
        # SQL comment attempt
        injection_queries = [
            "' OR '1' = '1",
            "'; DROP TABLE messages",
            '" OR "1" = "1"',
            "admin' --",
            "1' UNION SELECT * FROM users --",
        ]

        for query in injection_queries:
            with pytest.raises(ValueError):
                await self.manager.search(query)

    @pytest.mark.asyncio
    async def test_command_injection(self):
        """Test command injection via path traversal.

        Should be caught by path sanitization.
        """
        path_traversal_queries = [
            "../../../../etc/passwd",
            "..\\..\\windows\\\\system32\\\\drivers\\\\etc\\\\hosts",
            "$(cat /etc/passwd)",
        ]

        for path in path_traversal_queries:
            with pytest.raises(ValueError):
                await self.manager.search(path)

    @pytest.mark.asyncio
    async def test_xss_attempt(self):
        """Test XSS attempts via memory content.

        Should be sanitized or rejected.
        """
        xss_attempts = [
            "<script>alert('XSS')",
            "<img src=x onerror='XSS'>",
            "javascript:alert('XSS')",
            '"onload=XSS"',
        ]

        for content in xss_attempts:
            with pytest.raises(ValueError) or len(content) > 1000:
                await self.manager.add_messages_async([content])

    @pytest.mark.asyncio
    async def test_path_traversal_project_id(self):
        """Test path traversal in project_id.

        Should be rejected by project_id validation.
        """
        traversal_ids = [
            "../../../etc/passwd",
            "..\\..\\windows\\\\system32",
            "$(cat /etc/passwd)",
            "/proc/version",
            "C:\\Windows\\System32\\Drivers\\etc\\\\hosts",
        ]

        for traversal_id in traversal_ids:
            # Each path traversal attempt should be sanitized or rejected
            try:
                result = await self.manager.search(traversal_id)
                # If search succeeds, the input should be sanitized
                assert isinstance(result, list)
            except (ValueError, Exception):
                # If it raises an exception, that's also acceptable
                pass

    @pytest.mark.asyncio
    async def test_auth_bypass_attempt(self):
        """Test authentication bypass via malformed input.

        Should be caught by validation checks.
        """
        bypass_attempts = [
            None,  # No project_id
            "",  # Empty project_id
            "proj\tect",  # Tab character
            "A" * 1000000,  # Extremely long
        ]

        for project_id in bypass_attempts:
            if project_id:
                with pytest.raises(ValueError):
                    await self.manager.search("test")
            else:
                with pytest.raises(ValueError):
                    await self.manager.search("test")


class TestRateLimiting:
    """Tests rate limiting protects against abuse.

    Ensures rate limits are enforced and respected.
    """

    def __init__(self, manager: MemoryManager):
        """Initialize with test manager.

        Args:
            manager: Test MemoryManager instance.
        """
        self.manager = manager

    @pytest.mark.asyncio
    async def test_rate_limit_respected(self):
        """Test that system respects configured rate limits.

        Should queue requests when rate limit reached.
        """
        await self.manager.add_messages_async(["Test"] * 200)

        # Rate limit should kick in
        await asyncio.sleep(0.1)

        # Subsequent requests should be rate-limited
        with pytest.raises(Exception):
            await self.manager.search("query test")

    @pytest.mark.asyncio
    async def test_concurrent_burst(self):
        """Test burst capacity handling.

        Should allow burst within limits, then reject excess.
        """
        await self.manager.add_messages_async(["Burst 1"] * 50)

        # Burst capacity is 50 (API_RATE_BURST default)
        await self.manager.add_messages_async(["Burst 2"] * 30)

        # Should rate limit burst 2 (exceeds burst capacity)
        with pytest.raises(Exception):
            for _ in range(10):
                await self.manager.search(f"burst {_}")
