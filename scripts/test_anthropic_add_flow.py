#!/usr/bin/env python3
"""Test AnthropicReplacementProvider through the add tool flow.

This script tests the full smart replacement flow by:
1. Creating a MemoryManager with Anthropic provider
2. Adding first message: "I like to program in Python"
3. Adding second message: "I don't like Python anymore, I'm now TypeScript developer."
4. Verifying the second message replaced the first

Usage:
    WORKSPACE_ID=test-anthropic-add uv run python scripts/test_anthropic_add_flow.py
"""

import asyncio
import os
import shutil
import sys

from dotenv import load_dotenv

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load .env file if it exists
load_dotenv(os.path.join(project_root, ".env"))


async def main():
    """Test the add flow with AnthropicReplacementProvider."""
    # Use a unique workspace ID to avoid conflicts
    workspace_id = "test-anthropic-add"
    index_path = f"indexes/{workspace_id}"

    # Clean up any existing index
    if os.path.exists(index_path):
        print(f"Cleaning up existing index at {index_path}")
        shutil.rmtree(index_path)

    # Set environment variables for the test
    os.environ["WORKSPACE_ID"] = workspace_id
    os.environ["LLM_PROVIDER"] = "anthropic"
    os.environ["LLM_MODEL"] = "claude-sonnet-4-5"
    os.environ["ENABLE_SMART_REPLACE"] = "true"
    os.environ["SMART_REPLACE_THRESHOLD"] = "0.7"
    os.environ["SMART_REPLACE_MIN_SIMILARITY"] = (
        "0.3"  # Lower threshold to catch the match
    )
    # Note: OPENROUTER_API_KEY must be set for embeddings (via .env or environment)
    os.environ["LOG_LEVEL"] = "INFO"

    print("=" * 70)
    print("Testing AnthropicReplacementProvider - Add Tool Flow")
    print("=" * 70)
    print(f"Workspace ID: {workspace_id}")
    print(f"Provider: {os.environ.get('LLM_PROVIDER')}")
    print(f"Model: {os.environ.get('LLM_MODEL')}")
    print()

    # Import after setting env vars
    from reflectlog.application.config.settings import Config
    from reflectlog.application.memory.manager import MemoryManager
    from reflectlog.application.utils.logging import create_logger

    # Create config and logger
    config = Config.from_environment()
    logger = create_logger(config.workspace_id, config.log_level)

    print(f"Config llm_provider: {config.llm_provider}")
    print()

    # Create MemoryManager
    print("Initializing MemoryManager...")
    manager = MemoryManager(config=config, logger=logger)
    print("MemoryManager initialized.")
    print()

    # Test messages
    first_message = "I like to program in Python"
    second_message = "I don't like Python anymore, I'm now TypeScript developer."

    try:
        # Step 1: Add first message
        print("-" * 70)
        print("Step 1: Adding first message")
        print("-" * 70)
        print(f"Message: {first_message}")
        print()

        added_count = await manager.add_messages_async([first_message])
        print(f"Added {added_count} message(s)")

        # Verify first message is stored
        all_messages = manager.get_all()
        print(f"Current messages ({len(all_messages)}):")
        for i, msg in enumerate(all_messages, 1):
            print(f"  {i}. {msg}")
        print()

        # Step 2: Add second message (should trigger replacement)
        print("-" * 70)
        print("Step 2: Adding second message (should replace first)")
        print("-" * 70)
        print(f"Message: {second_message}")
        print()

        added_count = await manager.add_messages_async([second_message])
        print(f"Added {added_count} message(s)")

        # Verify replacement
        all_messages_after = manager.get_all()
        print(f"Current messages ({len(all_messages_after)}):")
        for i, msg in enumerate(all_messages_after, 1):
            print(f"  {i}. {msg}")
        print()

        # Step 3: Verify results
        print("-" * 70)
        print("Verification")
        print("-" * 70)

        if len(all_messages_after) == 1 and second_message in all_messages_after:
            print("✅ SUCCESS: Second message replaced the first message!")
            print(f"   Only message remaining: {all_messages_after[0]}")
            result = 0
        elif len(all_messages_after) == 2:
            print(
                "⚠️  PARTIAL: Both messages exist (replacement may not have triggered)"
            )
            print("   This could happen if similarity threshold wasn't met")
            result = 1
        else:
            print(f"❌ UNEXPECTED: Got {len(all_messages_after)} messages")
            result = 1

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        result = 1

    finally:
        # Cleanup
        print()
        print("-" * 70)
        print("Cleanup")
        print("-" * 70)
        manager.close()
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
            print(f"Removed index at {index_path}")

    print()
    print("=" * 70)
    print("Test completed!")
    print("=" * 70)
    return result


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
