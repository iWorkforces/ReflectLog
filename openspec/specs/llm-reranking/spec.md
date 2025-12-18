# llm-reranking Specification

## Purpose
TBD - created by archiving change add-llm-reranking. Update Purpose after archive.
## Requirements
### Requirement: LLM Reranker Configuration

The system SHALL support configuration of reranking via environment variables:
- `RERANKER_ENGINE`: Reranking engine: `llm`, `cross_encoder`, or `none` (default: `llm`)
- `LLM_MODEL`: LLM model identifier for reranking (default: `x-ai/grok-4.1-fast`)
- `SEARCH_SCORE_THRESHOLD`: Minimum LLM relevance score to keep results (default: `0.5`)
- `RERANK_MAX_CONCURRENCY`: Maximum parallel LLM calls (default: `5`)

#### Scenario: Default configuration
- **GIVEN** no LLM reranking environment variables are set
- **WHEN** the server initializes
- **THEN** reranking is enabled with model `x-ai/grok-4.1-fast`, threshold `0.5`, and max concurrency `5`

#### Scenario: Disable reranking
- **GIVEN** `RERANKER_ENGINE=none` is set
- **WHEN** a search is executed
- **THEN** the LLM reranking step is skipped and fusion results are returned directly

#### Scenario: Custom threshold
- **GIVEN** `SEARCH_SCORE_THRESHOLD=0.7` is set
- **WHEN** search results are reranked
- **THEN** only results with LLM score >= 0.7 are returned

---

### Requirement: LLM Reranker Infrastructure

The system SHALL provide an `LLMReranker` infrastructure class that:
- Uses `AsyncOpenAI` client to call OpenRouter LLM API
- Accepts a query and list of candidate (message, score) tuples
- Scores each candidate's relevance to the query using the configured LLM model
- Returns candidates sorted by LLM relevance score, filtered by threshold
- Supports parallel scoring with configurable concurrency via `anyio.Semaphore`

#### Scenario: Successful reranking
- **GIVEN** a query "Python tutorials" and 3 candidate messages
- **WHEN** the reranker is called
- **THEN** each candidate receives an LLM relevance score (0.0-1.0)
- **AND** results are sorted by score descending
- **AND** results below threshold are filtered out

#### Scenario: LLM API failure graceful fallback
- **GIVEN** the LLM API is unavailable or returns an error
- **WHEN** reranking is attempted
- **THEN** the system logs a warning
- **AND** returns candidates with their original fusion scores
- **AND** the search operation does not fail

#### Scenario: Parallel scoring with concurrency limit
- **GIVEN** 10 candidates to rerank and max concurrency of 5
- **WHEN** reranking is executed
- **THEN** at most 5 LLM calls run concurrently
- **AND** all candidates are eventually scored

---

### Requirement: Scoring Prompt Template

The system SHALL use a structured scoring prompt (`SCORING_PROMPT`) that:
- Instructs the LLM to evaluate document relevance to a query
- Requests JSON output with a single `score` field (0.0-1.0)
- Provides clear scoring guidelines (1.0 = perfect match, 0.0 = no relevance)

#### Scenario: Prompt produces valid JSON score
- **GIVEN** a query and document are provided to the prompt
- **WHEN** the LLM responds
- **THEN** the response is valid JSON with format `{"score": <float>}`
- **AND** the score is between 0.0 and 1.0 inclusive

#### Scenario: Invalid LLM response handling
- **GIVEN** the LLM returns invalid JSON or out-of-range score
- **WHEN** the response is parsed
- **THEN** the system treats it as a scoring failure
- **AND** falls back to the candidate's fusion score

---

### Requirement: Search Pipeline Integration

The system SHALL integrate LLM reranking into the search pipeline as Step 4:
1. Parallel search (USearch + Tantivy)
2. RRF Fusion (combine rankings)
3. Fusion threshold filtering (`FUSION_RANKING_THRESHOLD`)
4. **LLM Reranking** (when `RERANKER_ENGINE=llm`)
5. Return top `limit` results

#### Scenario: Full hybrid search with reranking
- **GIVEN** hybrid search is enabled and `RERANKER_ENGINE=llm`
- **WHEN** a search query is executed
- **THEN** results pass through all 5 pipeline stages
- **AND** final results are ordered by LLM relevance scores

#### Scenario: Two-stage threshold filtering
- **GIVEN** `FUSION_RANKING_THRESHOLD=0.5` and `SEARCH_SCORE_THRESHOLD=0.6`
- **WHEN** a search produces fusion results with scores 0.4, 0.55, 0.65, 0.8
- **THEN** fusion filtering removes the 0.4 result (below 0.5)
- **AND** LLM reranking is applied to remaining 3 results
- **AND** only results with LLM score >= 0.6 are returned

---

### Requirement: Reranking Logging

The system SHALL log reranking operations with structured logging:
- Log step header "STEP 4: LLM Reranking..."
- Log number of candidates being reranked
- Log LLM scores for each candidate (with message preview)
- Log filtered/kept counts after threshold application
- Log total reranking duration in milliseconds

#### Scenario: Reranking logs visibility
- **GIVEN** reranking is enabled and debug logging is on
- **WHEN** a search with reranking is executed
- **THEN** logs include rerank step with candidate count
- **AND** logs include individual LLM scores
- **AND** logs include duration metrics

