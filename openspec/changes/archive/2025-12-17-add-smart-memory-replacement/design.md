# Design: Add Smart Memory Replacement

## Context

The `add` tool currently supports exact deduplication (via `DEDUPLICATE_MESSAGES`) but lacks semantic replacement detection. When a user updates their preference (e.g., "I like cats" to "I don't like cats anymore"), both contradictory memories persist.

This design introduces LLM-based semantic replacement detection following existing patterns in the codebase.

## Goals / Non-Goals

**Goals:**
- Detect when new memories semantically replace existing ones
- Automatically remove outdated memories during add operation
- Provide detailed logging for transparency
- Follow existing `LLMReranker` pattern for LLM integration
- Enable graceful degradation on failures

**Non-Goals:**
- Batch replacement detection across multiple existing memories
- User confirmation before replacement (automatic by design)
- Offline/local-only replacement detection (requires LLM)

## Decisions

### Decision 1: New Infrastructure Component

**What:** Create `SmartReplacer` in `openmemories/infrastructure/smart_replacer.py`

**Why:**
- Follows the Dependency Inversion Principle (infrastructure implements protocols)
- Matches existing `LLMReranker` and `CrossEncoderReranker` patterns
- Keeps LLM API calls in the infrastructure layer
- Enables easy mocking for unit tests

**Alternatives considered:**
- Inline in MemoryManager: Would mix concerns and make testing harder
- Extend LLMReranker: Different purpose, would conflate responsibilities

### Decision 2: Check Top 1 Similar Memory Only

**What:** Only check the single most semantically similar existing memory for replacement.

**Why:**
- User requirement: minimize LLM API calls per add operation
- Most replacements occur between highly similar memories
- Cost-effective for the common case

**Alternatives considered:**
- Top 3-5 candidates: More thorough but 3-5x more LLM calls
- All memories above threshold: Could be expensive for large memory stores

### Decision 3: Configurable Threshold with Sensible Default

**What:** `SMART_REPLACE_THRESHOLD=0.7` (default), configurable via environment variable.

**Why:**
- 0.7 balances false positives (replacing unrelated memories) and false negatives (missing replacements)
- User-configurable allows tuning for specific use cases
- Environment variable follows existing configuration pattern

### Decision 4: Graceful Degradation on Failures

**What:** If LLM call fails, log warning and proceed with normal add (no replacement).

**Why:**
- The `add` operation must not fail due to smart replacement
- Better to have duplicate memories than failed adds
- Follows existing `LLMReranker` pattern for error handling

## Architecture

### Flow Diagram

```
AddTool.add(messages)
    │
    ▼
MemoryManager.add_messages_async()
    │
    ▼ (for each message)
┌─────────────────────────────────────┐
│ Step 1: Semantic Search             │
│   USearchEngine.search(message, 1)  │
│   → Find top 1 most similar memory  │
└─────────────────────────────────────┘
    │
    ▼ (if similar memory found)
┌─────────────────────────────────────┐
│ Step 2: LLM Replacement Check       │
│   SmartReplacer.check_replacement() │
│   → (should_replace, confidence,    │
│      reason)                        │
└─────────────────────────────────────┘
    │
    ▼ (if should_replace && confidence >= threshold)
┌─────────────────────────────────────┐
│ Step 3: Remove Old Memory           │
│   delete_by_message(old_memory)     │
│   + Detailed logging                │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Step 4: Add New Memory              │
│   Normal _add_message() flow        │
│   (dedup check + dual engine store) │
└─────────────────────────────────────┘
```

### Component Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│  AddTool                                                     │
│    │                                                         │
│    ▼                                                         │
│  MemoryManager ───────────────────────────────────────────┐ │
│    │                                                       │ │
│    │ (orchestrates)                                        │ │
│    │                                                       │ │
│    ├──▶ _check_for_replacement() ──┐                      │ │
│    │                                │                      │ │
│    │                                ▼                      │ │
│    │                    ┌─────────────────────┐            │ │
│    │                    │ Infrastructure      │            │ │
│    │                    │                     │            │ │
│    │                    │ SmartReplacer       │◀──────────┐│ │
│    │                    │   (LLM API calls)   │           ││ │
│    │                    └─────────────────────┘           ││ │
│    │                                │                      ││ │
│    │                                ▼                      ││ │
│    │                    ┌─────────────────────┐           ││ │
│    │                    │ USearchEngine       │           ││ │
│    │                    │   (semantic search) │───────────┘│ │
│    │                    └─────────────────────┘            │ │
│    │                                                       │ │
│    └──▶ _add_message(replacement_target)                   │ │
│           │                                                │ │
│           ├──▶ delete_by_message() (if replacement)        │ │
│           └──▶ add to USearch + Tantivy                    │ │
└─────────────────────────────────────────────────────────────┘
```

### Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_SMART_REPLACE` | bool | `true` | Enable/disable smart replacement |
| `SMART_REPLACE_THRESHOLD` | float | `0.7` | Min LLM confidence to trigger replacement |

### LLM Prompt Template

```python
REPLACEMENT_DETECTION_PROMPT = """You are a memory replacement detection system...

OUTPUT FORMAT:
{{"should_replace": true, "confidence": 0.85, "reason": "Same topic with updated preference"}}

REPLACEMENT CRITERIA:
- Same topic with updated information
- Contradictory statements about the same thing
- New preference replacing old preference
- Updated facts about the same entity

DO NOT REPLACE IF:
- Memories are about different topics
- New memory adds information without contradicting old
- Memories are complementary, not contradictory

Existing Memory: "{old_memory}"
New Memory: "{new_memory}"
"""
```

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| LLM API latency adds to add operation | Check only top 1 similar; async call |
| LLM API costs per add | Single LLM call per message; configurable disable |
| False positive replacements | Configurable threshold (default 0.7); detailed logging |
| False negative (missed replacements) | Users can lower threshold; manual cleanup still works |
| LLM unavailable | Graceful degradation: proceed with normal add |

## Migration Plan

No migration needed - this is a new feature that is additive:
- Existing memories remain unchanged
- Feature is enabled by default but can be disabled
- No schema changes to storage engines

## Open Questions

None - all clarifications resolved with user:
- Threshold: Configurable via `SMART_REPLACE_THRESHOLD`
- Default: Enabled by default
- Candidates: Top 1 only
- Logging: Detailed (old/new preview, confidence, reason)
