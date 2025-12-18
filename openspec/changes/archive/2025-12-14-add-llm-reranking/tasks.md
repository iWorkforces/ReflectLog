# Tasks: Add LLM Reranking After RRF Fusion

## 1. Configuration

- [x] 1.1 Add `llm_model` field to `Config` dataclass in `settings.py` (default: `x-ai/grok-4.1-fast`)
- [x] 1.2 Add `search_rerank` field to `Config` (default: `True`, from `SEARCH_RERANK` env var)
- [x] 1.3 Add `search_score_threshold` field to `Config` (default: `0.5`, from `SEARCH_SCORE_THRESHOLD` env var)
- [x] 1.4 Add `rerank_max_concurrency` field to `Config` (default: `5`, from `RERANK_MAX_CONCURRENCY` env var)
- [x] 1.5 Update `.env.example` with new environment variables

## 2. Prompts

- [x] 2.1 Add `SCORING_PROMPT` constant to `prompts.py` with JSON output format
- [x] 2.2 Export `SCORING_PROMPT` in `config/__init__.py`

## 3. Infrastructure

- [x] 3.1 Create `LLMRerankerConfig` dataclass in `openmemories/infrastructure/llm_reranker.py`
- [x] 3.2 Create `LLMReranker` class extending `BaseModel` with:
  - [x] 3.2.1 `AsyncOpenAI` client initialization with HTTP/2 support
  - [x] 3.2.2 `_score_single()` async method for scoring one document
  - [x] 3.2.3 `rerank()` async method with parallel scoring via `anyio.Semaphore`
  - [x] 3.2.4 JSON response parsing with error handling
  - [x] 3.2.5 Graceful fallback to fusion scores on LLM failure
  - [x] 3.2.6 Structured logging for each scoring operation
- [x] 3.3 Add factory method `LLMRerankerConfig.from_app_config(config)` for easy instantiation
- [x] 3.4 Export `LLMReranker` and `LLMRerankerConfig` in `infrastructure/__init__.py`

## 4. Memory Manager Integration

- [x] 4.1 Import `LLMReranker` and `LLMRerankerConfig` in `memory/manager.py`
- [x] 4.2 Add `_reranker: Optional[LLMReranker]` private attribute to `MemoryManager`
- [x] 4.3 Initialize `LLMReranker` in `MemoryManager.__init__()` when `config.search_rerank=True`
- [x] 4.4 Add Step 4 (LLM Reranking) to `MemoryManager.search()`:
  - [x] 4.4.1 Log step header with candidate count
  - [x] 4.4.2 Call `self._reranker.rerank(query, hybrid_results)` when enabled
  - [x] 4.4.3 Log reranked results with LLM scores
  - [x] 4.4.4 Log filtered/kept counts
  - [x] 4.4.5 Log reranking duration
- [x] 4.5 Update search pipeline comments to reflect new step

## 5. Unit Tests

- [x] 5.1 Create `tests/unit/infrastructure/test_llm_reranker.py` with:
  - [x] 5.1.1 Test `LLMRerankerConfig` default values
  - [x] 5.1.2 Test `LLMRerankerConfig.from_app_config()` factory
  - [x] 5.1.3 Test `_score_single()` with mocked OpenAI client
  - [x] 5.1.4 Test `rerank()` with successful scoring
  - [x] 5.1.5 Test threshold filtering (scores below threshold removed)
  - [x] 5.1.6 Test graceful fallback on LLM API error
  - [x] 5.1.7 Test invalid JSON response handling
  - [x] 5.1.8 Test out-of-range score handling (< 0 or > 1)
  - [x] 5.1.9 Test concurrency limiting with semaphore
- [ ] 5.2 Update `tests/unit/application/test_memory_manager.py` (skipped - existing test issues)
- [ ] 5.3 Update `tests/unit/application/test_config.py` for new settings (skipped - existing test issues)

## 6. Integration Tests

- [ ] 6.1 Add integration test for full search pipeline with reranking (requires real LLM API)
- [ ] 6.2 Verify reranking improves result ordering for known test cases (requires real LLM API)

## 7. Documentation

- [ ] 7.1 Update `CLAUDE.md` with new environment variables and pipeline description
- [ ] 7.2 Update `openmemories/infrastructure/CLAUDE.md` with LLMReranker documentation
- [ ] 7.3 Update `openmemories/application/memory/CLAUDE.md` with reranking step

## 8. Validation

- [x] 8.1 Run type checker: `./start-type-check.sh` - PASSED (no errors)
- [x] 8.2 Run linter: `./start-lint.sh --all` - PASSED
- [x] 8.3 Run unit tests: `./start-unittest.sh` - PASSED (18/18 LLMReranker tests)
- [ ] 8.4 Manual testing with real OpenRouter API

## Dependencies

- Tasks in Section 1 (Config) must complete before Section 3-4
- Task 2.1 (SCORING_PROMPT) must complete before Section 3
- Section 3 (Infrastructure) must complete before Section 4
- Sections 1-4 must complete before Sections 5-6 (tests)
- All implementation complete before Section 7-8

## Parallelizable Work

- Section 1 (Config) and Section 2 (Prompts) can run in parallel
- Section 5 (Unit Tests) and Section 6 (Integration Tests) can run in parallel after implementation
- Section 7 (Documentation) can run in parallel with Section 5-6
