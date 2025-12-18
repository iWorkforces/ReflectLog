# Tasks: Add Smart Memory Replacement

## Status: COMPLETED

All tasks for the smart memory replacement feature have been implemented and tested.

## 1. Configuration

- [x] 1.1 Add `enable_smart_replace: bool = True` to Config dataclass in `settings.py`
- [x] 1.2 Add `smart_replace_threshold: float = 0.7` to Config dataclass
- [x] 1.3 Add environment variable parsing for `ENABLE_SMART_REPLACE` and `SMART_REPLACE_THRESHOLD` in `from_environment()`

## 2. Prompt Template

- [x] 2.1 Add `REPLACEMENT_DETECTION_PROMPT` constant to `prompts.py`
- [x] 2.2 Export `REPLACEMENT_DETECTION_PROMPT` in `config/__init__.py`

## 3. Infrastructure Component

- [x] 3.1 Create `openmemories/infrastructure/smart_replacer.py` with:
  - `ReplacementDecision` Pydantic schema
  - `SmartReplacerConfig` frozen dataclass with `from_app_config()` factory
  - `SmartReplacer` class with `check_replacement()` async method
- [x] 3.2 Export `SmartReplacer`, `SmartReplacerConfig`, `ReplacementDecision` from `infrastructure/__init__.py`

## 4. Memory Manager Integration

- [x] 4.1 Initialize `SmartReplacer` in `MemoryManager.__init__` when `enable_smart_replace=True`
- [x] 4.2 Add `_check_for_replacement()` async method to find replacement candidates
- [x] 4.3 Update `add_messages_async()` to call replacement check before adding each message
- [x] 4.4 Implement graceful degradation for replacement detection failures

## 5. Testing

- [x] 5.1 Create `tests/unit/infrastructure/test_smart_replacer.py` with 28 unit tests:
  - Test `ReplacementDecision` schema validation
  - Test `SmartReplacerConfig` defaults and factory method
  - Test `SmartReplacer` initialization (enabled/disabled)
  - Test `check_replacement()` with mocked LLM response
  - Test threshold filtering logic
  - Test error handling and graceful degradation
  - Test structured output parsing with fallback
- [x] 5.2 All tests passing (28 tests for SmartReplacer)

## 6. Documentation

- [x] 6.1 Update `CLAUDE.md` with:
  - New environment variables (`ENABLE_SMART_REPLACE`, `SMART_REPLACE_THRESHOLD`)
  - Smart replacement feature description in Architecture section
  - Add to "Tuning" section for configuration guidance
  - Add SmartReplacer to project structure
- [x] 6.2 Update `openmemories/infrastructure/CLAUDE.md` with:
  - Full SmartReplacer documentation
  - Configuration, usage examples, and design rationale
- [x] 6.3 Update `tests/unit/infrastructure/CLAUDE.md` with:
  - New test file documentation
  - Test scenarios and mocking patterns

## Implementation Summary

### Files Modified
- `openmemories/application/config/settings.py` - Added config fields
- `openmemories/application/config/prompts.py` - Added prompt template
- `openmemories/application/config/__init__.py` - Updated exports
- `openmemories/infrastructure/smart_replacer.py` - NEW: SmartReplacer component
- `openmemories/infrastructure/__init__.py` - Updated exports
- `openmemories/application/memory/manager.py` - Integration with MemoryManager
- `tests/unit/infrastructure/test_smart_replacer.py` - NEW: 28 unit tests
- `CLAUDE.md` - Documentation update
- `openmemories/infrastructure/CLAUDE.md` - Documentation update
- `tests/unit/infrastructure/CLAUDE.md` - Documentation update

### Key Features Implemented
1. **LLM-based replacement detection** using structured JSON output
2. **Configurable confidence threshold** (default: 0.7)
3. **Graceful degradation** - adds normally if LLM fails
4. **Detailed logging** of replacement decisions
5. **HTTP/2 support** via DefaultAioHttpClient
6. **Comprehensive test coverage** with 28 unit tests
