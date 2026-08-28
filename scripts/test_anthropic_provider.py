#!/usr/bin/env python3
'''Test script for AnthropicReplacementProvider.

This script tests the Anthropic provider for SmartReplacer by:
1. Initializing credentials from the keychain via init_credentials()
2. Creating a SmartReplacer with provider=anthropic
3. Running a real replacement check

Usage:
    WORKSPACE_ID=test-project uv run python scripts/test_anthropic_provider.py
'''

import asyncio
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    '''Test the AnthropicReplacementProvider.'''
    # Set environment variables for the test
    os.environ.setdefault("WORKSPACE_ID", "test-project")
    os.environ["LLM_PROVIDER"] = "anthropic"
    os.environ["LLM_MODEL"] = "claude-sonnet-4-5"

    print("=" * 60)
    print("Testing AnthropicReplacementProvider")
    print("=" * 60)
    print(f"Provider: {os.environ.get('LLM_PROVIDER')}")
    print(f"Model: {os.environ.get('LLM_MODEL')}")
    print()

    # Import after setting env vars
    from reflectlog.infrastructure.smart_replacer import (
        SmartReplacer,
        SmartReplacerConfig,
    )

    # Create config with Anthropic provider
    config = SmartReplacerConfig(
        api_key="not-used-for-anthropic",
        base_url="not-used-for-anthropic",
        model=os.environ.get("LLM_MODEL", "claude-sonnet-4-5"),
        threshold=0.7,
        enabled=True,
        provider="anthropic",
    )

    print(f"Config: {config}")
    print()

    # Create SmartReplacer
    replacer = SmartReplacer(config=config, logger=None)

    # Test case 1: Should replace (preference change)
    print("-" * 60)
    print("Test 1: Preference change (should replace)")
    print("-" * 60)
    new_memory = "I don't like cats anymore, I prefer dogs now"
    existing_memory = "I like cats"
    print(f"New memory: {new_memory}")
    print(f"Existing memory: {existing_memory}")
    print()

    try:
        should_replace, confidence, reason = await replacer.check_replacement(
            new_memory=new_memory,
            existing_memory=existing_memory,
        )
        print("Result:")
        print(f"  should_replace: {should_replace}")
        print(f"  confidence: {confidence:.2f}")
        print(f"  reason: {reason}")
        print()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Test case 2: Should NOT replace (unrelated memories)
    print("-" * 60)
    print("Test 2: Unrelated memories (should NOT replace)")
    print("-" * 60)
    new_memory2 = "I enjoy playing chess on weekends"
    existing_memory2 = "My favorite programming language is Python"
    print(f"New memory: {new_memory2}")
    print(f"Existing memory: {existing_memory2}")
    print()

    try:
        should_replace2, confidence2, reason2 = await replacer.check_replacement(
            new_memory=new_memory2,
            existing_memory=existing_memory2,
        )
        print("Result:")
        print(f"  should_replace: {should_replace2}")
        print(f"  confidence: {confidence2:.2f}")
        print(f"  reason: {reason2}")
        print()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Test case 3: Location update (should replace)
    print("-" * 60)
    print("Test 3: Location update (should replace)")
    print("-" * 60)
    new_memory3 = "I moved to San Francisco last month"
    existing_memory3 = "I live in New York City"
    print(f"New memory: {new_memory3}")
    print(f"Existing memory: {existing_memory3}")
    print()

    try:
        should_replace3, confidence3, reason3 = await replacer.check_replacement(
            new_memory=new_memory3,
            existing_memory=existing_memory3,
        )
        print("Result:")
        print(f"  should_replace: {should_replace3}")
        print(f"  confidence: {confidence3:.2f}")
        print(f"  reason: {reason3}")
        print()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
