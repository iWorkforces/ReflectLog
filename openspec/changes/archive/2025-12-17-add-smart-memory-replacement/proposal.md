# Change: Add Smart Memory Replacement

## Why

When users add new memories that semantically update or replace existing ones (e.g., "I like cats" becomes "I don't like cats anymore, I like dogs"), the old outdated memory persists alongside the new one. This creates inconsistent memory states where contradictory information coexists, degrading the quality of AI agent memory retrieval.

Smart memory replacement automatically detects when a new memory should replace an existing one, removes the old memory, and adds the new one - keeping the memory store consistent and up-to-date.

## What Changes

- Add **smart replacement detection** during the `add` operation using LLM
- Use the configured `LLM_MODEL` to determine if a new memory replaces an existing one
- Automatically remove the old memory before adding the new one when replacement is detected
- Provide detailed logging of replacement decisions (old/new memory preview, confidence, reason)
- Add configurable threshold via `SMART_REPLACE_THRESHOLD` environment variable
- Feature enabled by default, can be disabled with `ENABLE_SMART_REPLACE=false`

## Impact

- **Affected specs**: `add-tool` (new capability: Smart Memory Replacement)
- **Affected code**:
  - `openmemories/application/config/settings.py` - New config options
  - `openmemories/application/config/prompts.py` - New prompt template
  - `openmemories/infrastructure/smart_replacer.py` - New component (create)
  - `openmemories/infrastructure/__init__.py` - Export new component
  - `openmemories/application/memory/manager.py` - Integration with add flow
  - `tests/unit/infrastructure/test_smart_replacer.py` - Unit tests (create)
  - `CLAUDE.md` - Documentation update
