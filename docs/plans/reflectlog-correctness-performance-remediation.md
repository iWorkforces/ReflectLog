# reflectlog-correctness-performance-remediation - Work Plan

## TL;DR (For humans)
**What you'll get:** A storage and search service that safely coordinates multiple writer processes, survives crashes without silently losing indexed memories, reports search outages correctly, avoids the verified query/retry/cache costs, and has reproducible release checks across all supported operating systems.

**Why this approach:** Correctness is established at the shared persistence boundary first, using one portable workspace lock and recoverable generation publication. Search and performance fixes then build on that stable state before broad lint and release cleanup.

**What it will NOT do:** It will not introduce distributed locking, new providers or tools, authentication changes, database migrations, or unrelated architecture refactors. It will not call paid services during verification or claim atomicity across independent storage engines.

**Effort:** XL
**Risk:** High - coordinated native indexes, process death, cross-platform file locking, and strict backward compatibility must all hold simultaneously.
**Decisions to sanity-check:** Multi-process writers remain supported; Portalocker is the local lock dependency; ranx remains mandatory but lazy; the current FastMCP and Pydantic beta versions are pinned intentionally; strict test-first development is required.

Your next move: wait for the required dual plan review, then execute the approved plan in an isolated worker worktree. Full execution detail follows below.

---

> TL;DR (machine): XL/high-risk, 35 TDD increments plus 4 final gates covering multiprocess persistence, search correctness/performance, frozen tooling, real MCP QA, and three-platform proof.

## Scope
### Must have
- Coordinated multi-process writes for every workspace across SQLite, USearch, and Tantivy, using a direct `portalocker==4.3.0` dependency and stable sidecars at `indexes/<workspace_id>/.reflectlog.writer.lock` and `indexes/<workspace_id>/.reflectlog.storage-generation`.
- Lock order: Portalocker workspace lease -> `MemoryManager._write_lock` -> `MemoryManager._lock` -> engine-local locks -> SQLite connection lock. Embedding, smart-replacement provider calls, fusion, and reranking stay outside the exclusive lease.
- Legacy stores open as generation 0 without a schema or path migration. SQLite remains the identity/source of truth; journal kinds and later-write-wins recovery remain unchanged.
- USearch refreshes before every coordinated mutation and publishes only validated same-directory temporary snapshots via fsync + `os.replace()` + parent-directory fsync where supported. Failures leave either the previous valid index or the complete new index.
- Tantivy readers and writers are operation-scoped under coordinator leases. No live object may retain a replaceable directory after lease release; request-path hard deletion and compaction must use safe logical delete/commit/merge behavior rather than destructive directory replacement.
- Long-lived managers observe external completed writes on the next coordinated refresh. Storage generation is published only after SQLite/USearch/Tantivy converge and before the durable intent is completed.
- Search-triggered recovery remains logically before search but runs off the event loop. Semantic failure plus no live FTS fallback raises `SearchError`.
- Timestamp hydration becomes one typed bulk lookup; incomplete/invalid timestamp coverage disables recency for the whole post-threshold candidate batch.
- OpenAI SDK retries are the only OpenAI retry owner. Sync and async embedding caches single-flight identical misses without serializing distinct keys or caching failures.
- Tombstone cache invalidation is workspace-scoped for local writes and conservative on external-generation reload, with positive capacity validation.
- `reflectlog --version` derives from installed distribution metadata and exits before FastMCP/search/Numba/ranx imports. HTTP client first-call overrides are honored without changing singleton ownership.
- Manifest intent explicitly pins `fastmcp==4.0.0b5`, `pydantic==2.14.0b1`, and `ranx==0.3.21`; ranx stays mandatory but is imported only for ranx-backed fusion methods. Its third-party Python 3.14 warning is isolated narrowly without weakening unrelated warnings-as-errors.
- Malformed numeric environment settings raise field-specific `ConfigurationError` without exposing values or secrets.
- Validation wrappers run checks through `uv run --frozen --no-sync`, never install/upgrade/fix in check mode, cover configured source/test/script scope, and enforce Ruff, format, ty, Pyright, pytest, warnings, and coverage >=90%.
- All 1,522 configured Ruff findings are resolved mechanically in bounded package waves. Any behavior-affecting lint change leaves the mechanical wave and receives its own RED/GREEN task.
- Agent-executed real-surface QA covers installed CLI, real FastMCP client/tool calls, real SQLite/USearch/Tantivy, deterministic loopback embeddings, spawned writer processes, process death, shutdown, and 10,000-record characterization. Graceful shutdown uses POSIX `SIGINT`/`SIGTERM` and Windows `SIGBREAK` delivered by `CTRL_BREAK_EVENT` to a new process group; forced termination remains a separate abrupt-death case.
- Existing macOS, Linux, and Windows support is proven through focused platform jobs with no required scenario skipped; unsupported NFS client-local locking and SMB/CIFS `nobrl` mounts are documented rather than guessed at runtime.

### Must NOT have (guardrails, anti-slop, scope boundaries)
- No Redis/distributed lock, provider migration, FlagEmbedding work, plugin policy, new MCP tool, authentication redesign, SQLite schema migration, persisted-path change, or public result-schema change.
- No cross-backend ACID claim. Pending intents remain the convergence mechanism for crashes between independent stores.
- No provider credentials, paid network calls, real user memories, or secrets in required tests/evidence.
- No `sleep()`-driven concurrency tests, grep-only acceptance, self-reported success, skipped required scenarios, weakened tests, broad warning ignores, type suppressions, or unsafe Ruff fixes.
- No exclusive writer lease during embedding/LLM calls, fusion, reranking, or other non-persistence work.
- No lock-file deletion as stale-owner recovery; OS lock ownership, not pathname existence, determines ownership.
- No repository-wide module split or unrelated architecture cleanup. Extraction is allowed only where the coordinator/search seams need a typed owner and must preserve layer rules.
- Do not touch, stage, delete, or commit the pre-existing untracked `proposal-local-flagembedding-engine.md` and `repository_context.md`.
- No `git add -A`, hand-edited `uv.lock`, committed `.omgb/evidence` artifacts, or omnibus end-of-run commit.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: strict RED -> GREEN -> real-surface TDD with pytest; characterization-first for dependency/fusion/persisted-data contracts; formatting-only Ruff cleanup is exempt from a new behavioral test.
- Evidence: <attemptDir>/task-<N>-reflectlog-correctness-performance-remediation.<ext> (attemptDir = currentAttemptDir from 'omo-agent-toolkit ulw-loop status --json', .omgb/evidence/ulw/<session>/<goalId>/a<attempt>; outside ulw-loop use .omgb/evidence/)
- Every behavior todo records the exact RED command and expected assertion, the GREEN command, and an artifact containing the actual postcondition. Implementation and direct tests are one todo and one commit.
- Concurrency uses `multiprocessing.get_context("spawn")`, barriers/events/queues, unique temporary roots, ephemeral ports, and a third fresh process for final persistence inspection. Never use timing sleeps as synchronization.
- Mandatory assertions inspect SQLite rows, USearch vector/search results, Tantivy live results, archive rows, pending-intent count, and storage generation. Call-count-only mocks are supplemental.
- Required network behavior uses a loopback OpenAI-compatible fake returning deterministic 1024-dimensional vectors. No external provider call is permitted.
- Performance hard gates are query count, provider call count, import boundary, absence of lost state, and event-loop progress. Same-machine cold/warm median and p95 are recorded but are not release SLOs.
- Before and after each check wrapper, record `git status --short`, tracked diff, and relevant tool versions to prove the gate did not mutate source or the environment.
- All changed Python files receive clean LSP diagnostics; related focused tests run before the configured full suite and coverage gate.

## Execution strategy
### Parallel execution waves
> Target 5-8 todos per wave. Fewer than 3 (except the final) means you under-split.
- Wave 1, contracts/tooling (1-7): freeze dependencies and legacy behavior, make wrappers deterministic, lazy-load ranx, and lock CLI/HTTP/config contracts. Tasks 1, 2, 3, 6, and 7 can start in parallel; 4 and 5 follow dependency pinning.
- Wave 2, core correctness seams (8-12): build the coordinator and independently repair async recovery, backend failure semantics, recency policy, and retry ownership.
- Wave 3, engines/performance primitives (13-17): single-flight cache, bulk timestamp store API, atomic USearch publication, safe Tantivy lifecycle, and reusable platform gate.
- Wave 4, application integration (18-22): thread coordination through manager/add/recovery/cache/shutdown. Tasks sharing manager/recovery locks are serialized as shown in the matrix.
- Wave 5, real process/surface proof (23-29): independently exercise write races, delete/replace races, crash boundaries, Tantivy leases, shutdown, MCP tools, and bounded measurements.
- Wave 6, release cleanup/documentation (30-35): run the four bounded Ruff partitions in parallel, then execute the platform matrix against the clean integrated tree, then publish evidence-backed support documentation.
- Final wave F1-F4 runs only after all implementation todos and all commits are complete. Every lane must approve the same final tree.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | - | 4, 5, 8, 12, 17 | 2, 3, 6, 7 |
| 2 | - | 8-12, 15, 16, 18-20 | 1, 3, 6, 7 |
| 3 | - | 17, 34 | 1, 2, 6, 7 |
| 4 | 1 | 28, 29 | 5-7, 8-12 |
| 5 | 1 | 28, 29 | 4, 6-12 |
| 6 | - | 28, 29 | 1-5, 7-12 |
| 7 | - | 28 | 1-6, 8-12 |
| 8 | 1, 2 | 15-22 | 9-12 |
| 9 | 2 | 20 | 8, 10-12 |
| 10 | 2 | 14, 28 | 8, 9, 11, 12 |
| 11 | 2 | 14 | 8-10, 12 |
| 12 | 1, 2 | 13, 28, 29 | 8-11 |
| 13 | 12 | 29 | 14-17 |
| 14 | 10, 11 | 29 | 13, 15-17 |
| 15 | 8, 2 | 18-20, 23-25 | 13, 14, 16, 17 |
| 16 | 8, 2 | 18-22, 23-27 | 13-15, 17 |
| 17 | 1, 3, 8 | 30 | 13-16 |
| 18 | 8, 15, 16 | 22-29 | 21 |
| 19 | 8, 15, 16, 18 | 23-29 | 21 |
| 20 | 9, 15, 16, 18 | 22-29 | 21 |
| 21 | 16 | 23-29 | 18-20 |
| 22 | 18, 20 | 27-29 | 21 |
| 23 | 18-20 | 29, 30 | 24-28 |
| 24 | 18-20 | 29, 30 | 23, 25-28 |
| 25 | 15, 18-20 | 29, 30 | 23, 24, 26-28 |
| 26 | 16, 18, 21 | 29, 30 | 23-25, 27, 28 |
| 27 | 22-26 | 29, 30 | 28 |
| 28 | 4-7, 18-22 | 29, 30 | 23-27 |
| 29 | 13-28 | 30-35 | - |
| 30 | 17, 23-29, 31-34 | 35, F1-F4 | - |
| 31 | 29 | 30, F1-F4 | 32-34 |
| 32 | 29 | 30, F1-F4 | 31, 33, 34 |
| 33 | 29 | 30, F1-F4 | 31, 32, 34 |
| 34 | 3, 29 | 30, F1-F4 | 31-33 |
| 35 | 29, 30 | F1-F4 | - |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->
- [x] 1. Pin the approved runtime contract and add Portalocker
  What to do / Must NOT do: Update `pyproject.toml` to declare `fastmcp==4.0.0b5`, `pydantic==2.14.0b1`, compatible `pydantic-core==2.48.0`, `ranx==0.3.21`, and `portalocker==4.3.0`; regenerate `uv.lock` with uv while preserving all unrelated resolutions. Add a focused runtime-contract test that exercises FastMCP construction, Pydantic protocol validation, ranx metadata, and a real Portalocker timeout/release cycle. Never hand-edit the lock or upgrade unrelated packages.
  Parallelization: Wave 1 | Blocked by: none | Blocks: 4, 5, 8, 12, 17
  References (executor has NO interview context - be exhaustive): `pyproject.toml:20-55,73-82`; `uv.lock:1028-1055,2704-2717,3010-3031`; `reflectlog/core/logging.py:164-210`; pre-existing untracked files are out of scope.
  Acceptance criteria (agent-executable): RED `uv run --frozen --no-sync pytest -q tests/unit/test_runtime_contract.py` fails because the manifest does not declare the exact approved contract; GREEN passes, `uv lock --check` passes, the explicitly separate preparation command `uv sync --frozen` exits 0, and runtime metadata reports the exact versions. A second spawned process holding the same Portalocker path makes the tested acquisition hit its injected timeout, then acquire successfully after owner exit.
  QA scenarios (name the exact tool + invocation): Happy: `uv run --frozen --no-sync python -c 'from importlib.metadata import version; print(version("fastmcp"), version("pydantic"), version("ranx"), version("portalocker"))'` plus the focused pytest. Failure: temporarily request a conflicting Portalocker lease in the test and assert the typed timeout without deleting the lock file. Evidence `<attemptDir>/task-1-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `unspecified-high` - manifest, lock, runtime, and process behavior must remain one compatibility unit.
  Commit: Y | `build(deps): pin supported beta runtime and add portalocker`

- [x] 2. Characterize legacy persisted data and public contracts
  What to do / Must NOT do: Before storage edits, add generated-at-test-time legacy fixtures for SQLite rows/transitions, empty legacy timestamps, current USearch HNSW, and current Tantivy documents. Pin MCP tool signatures/results, workspace identity, index paths, journal kinds, later-write-wins, and refusal to load a missing/empty/unreadable HNSW when SQLite has rows. Do not add a schema migration or copy real user data.
  Parallelization: Wave 1 | Blocked by: none | Blocks: 8-12, 15, 16, 18-20
  References (executor has NO interview context - be exhaustive): `reflectlog/infrastructure/memory_store.py:194-329,955-1129`; `reflectlog/infrastructure/usearch_engine.py:220-365`; `reflectlog/infrastructure/tantivy_engine.py:292-340`; `reflectlog/application/memory/replacement_recovery.py:93-385`; `tests/integration/test_replacement_recovery.py`; `tests/integration/test_memory_manager_usearch.py`.
  Acceptance criteria (agent-executable): Characterization tests are GREEN before production edits; the same fixtures are listed as required gates for tasks 15, 16, 18-20, and F4. A populated SQLite database with a corrupt/missing HNSW fails closed rather than creating an empty index.
  QA scenarios (name the exact tool + invocation): Happy: `uv run --frozen --no-sync pytest -q tests/integration/test_legacy_storage_compatibility.py`. Failure: the test replaces the HNSW with invalid bytes and asserts the existing typed initialization failure and unchanged SQLite rows. Evidence `<attemptDir>/task-2-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `deep` - real native engines and restart semantics must be characterized together.
  Commit: Y | `test(storage): characterize legacy persistence contracts`

- [x] 3. Make validation wrappers frozen and non-mutating
  What to do / Must NOT do: Rewrite check paths in `start-lint.sh`, `start-type-check.sh`, and `start-unittest.sh` to call locked `uv run --frozen --no-sync` commands, honor configured pytest paths, run coverage once, and never install, upgrade, create test trees, format, fix, or rewrite in check mode. `--frozen` prevents lock updates; `--no-sync` prevents environment synchronization/installation. Keep explicitly named mutation commands only if they are clearly separate from verification.
  Parallelization: Wave 1 | Blocked by: none | Blocks: 17, 34
  References (executor has NO interview context - be exhaustive): `start-lint.sh:68-123,126-292,454-590`; `start-type-check.sh:66-110,265-383`; `start-unittest.sh:44-70,147-243,404-586`; `pyproject.toml:73-124,200-259`; README development commands at `README.md:133-163`.
  Acceptance criteria (agent-executable): RED shell-level regression tests fail because wrappers mutate/install or omit configured scope; GREEN passes. Running all wrappers in a disposable worktree preserves the pre-run tracked diff and installed lock versions. Injected lint/type/test defects each cause nonzero exit and remain unmodified.
  QA scenarios (name the exact tool + invocation): Happy: `./start-lint.sh --check && ./start-type-check.sh && ./start-unittest.sh --coverage` in a disposable worktree, with pre/post `git status --short` and `uv tree --depth 1`. Failure: introduce one temporary Ruff violation, type error, and failing test separately; each wrapper fails without fixing it, then scrub the disposable tree. Evidence `<attemptDir>/task-3-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `unspecified-high` - three interdependent shell gates must agree with one locked toolchain.
  Commit: Y | `build(checks): make validation wrappers frozen and read only`

- [x] 4. Keep ranx mandatory but remove it from default imports
  What to do / Must NOT do: Move ranx imports behind the non-RRF fusion path while preserving `sum`, `mnz`, `max`, and `bordafuse` outputs and errors. Default RRF and one-list behavior must remain local/current. Isolate only ranx's known invalid-escape `SyntaxWarning` at its lazy import boundary; unrelated warnings must still fail. Do not make ranx optional, rename methods, normalize raw RRF, or add eager warmup.
  Parallelization: Wave 1 | Blocked by: 1 | Blocks: 28, 29
  References (executor has NO interview context - be exhaustive): `reflectlog/application/memory/fusion/__init__.py:8-47`; `reflectlog/application/memory/fusion/ranx_fusion.py:10-18,139-251,253-357`; `reflectlog/application/memory/fusion/AGENTS.md:23-41`; `tests/unit/application/test_ranx_fusion.py`; `pyproject.toml:102-108`.
  Acceptance criteria (agent-executable): RED subprocess test fails because importing CLI/default RRF loads ranx; GREEN proves `ranx` absent from `sys.modules` for `reflectlog --version`, help, and local RRF, then present and behaviorally correct after each ranx-backed method. Fresh-cache pytest collection succeeds on Python 3.14, while an unrelated emitted warning still fails.
  QA scenarios (name the exact tool + invocation): Happy: `uv run --frozen --no-sync pytest -q tests/unit/application/test_ranx_fusion.py tests/unit/test_server.py -k 'fusion or version or help'`. Failure: a test emits an unrelated `RuntimeWarning` and must fail under the global policy; a ranx method with invalid weights retains its existing typed/runtime failure. Evidence `<attemptDir>/task-4-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `deep` - import isolation must preserve five subtly different ranking contracts.
  Commit: Y | `perf(fusion): lazy load mandatory ranx methods`

- [x] 5. Make CLI metadata exits lightweight and truthful
  What to do / Must NOT do: Use installed distribution metadata as the sole CLI/module version source and parse help/version before importing FastMCPServer, scoring, fusion, or engines. Preserve the installed entry point, argparse spelling, transport behavior, and normal startup path. Do not introduce a new CLI framework.
  Parallelization: Wave 1 | Blocked by: 1 | Blocks: 28, 29
  References (executor has NO interview context - be exhaustive): `pyproject.toml:1-4,63-65`; `reflectlog/version.py:1-3`; `reflectlog/server.py:1-29,96-203`; `tests/unit/test_server.py:177-199`.
  Acceptance criteria (agent-executable): RED test fails because CLI reports `0.1.7` and imports heavy modules; GREEN `uv run --frozen --no-sync reflectlog --version` exactly equals `importlib.metadata.version("reflectlog")`, help/version exit 0, and subprocess `sys.modules` excludes FastMCP/search/Numba/ranx. Normal startup still constructs one server after parsing.
  QA scenarios (name the exact tool + invocation): Happy: `/usr/bin/time -p uv run --frozen --no-sync reflectlog --version` and `uv run --frozen --no-sync reflectlog --help`, plus `uv run --frozen --no-sync pytest -q tests/unit/test_server.py`. Failure: `uv run --frozen --no-sync reflectlog --definitely-invalid` exits nonzero with argparse behavior and still does not construct the server. Evidence `<attemptDir>/task-5-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `quick` - one entry-point seam and direct tests.
  Commit: Y | `fix(cli): use package metadata before runtime imports`

- [x] 6. Honor HTTP client first-call overrides
  What to do / Must NOT do: Apply explicit non-`None` limits/timeouts/http2 on first sync/async HTTPX client creation; otherwise use environment defaults. Preserve singleton first-call-wins, follow-redirect behavior, async/sync ownership, and aiohttp's documented lack of HTTP/2. Use `is not None`, not truthiness; do not rebuild an existing singleton on a later conflicting call.
  Parallelization: Wave 1 | Blocked by: none | Blocks: 28, 29
  References (executor has NO interview context - be exhaustive): `reflectlog/utility/http.py:35-115,131-255`; `tests/unit/utility/test_http.py:18-35,108-197`; OpenAI `http_client` injection remains outside this task.
  Acceptance criteria (agent-executable): RED constructor-capture tests fail because overrides are discarded; GREEN exact client kwargs match first-call values, `None` selects environment defaults, `http2=False` is preserved, and second-call conflicts return the original singleton. Close/reset leaves no unclosed-client warnings.
  QA scenarios (name the exact tool + invocation): Happy: `uv run --frozen --no-sync pytest -q tests/unit/utility/test_http.py` and a loopback delayed HTTP endpoint proving the configured timeout. Failure: a first call with a deliberately short timeout fails at the expected boundary; a second longer timeout does not mutate the existing client. Evidence `<attemptDir>/task-6-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `quick` - contained factory contract with existing reset fixture.
  Commit: Y | `fix(http): honor first client configuration`

- [x] 7. Normalize malformed numeric configuration errors
  What to do / Must NOT do: Parse integer/float environment fields through typed helpers that reject empty, malformed, NaN/infinite, negative, and out-of-range values according to each existing field contract; raise `ConfigurationError` naming the field without echoing its value or secrets. Preserve defaults, profiles, environment names, and CLI precedence, especially `MCP_PORT`.
  Parallelization: Wave 1 | Blocked by: none | Blocks: 28
  References (executor has NO interview context - be exhaustive): `reflectlog/application/config/settings.py:306-585,623-698`; `reflectlog/application/config/validation.py`; `tests/unit/application/config/test_settings.py`; `tests/unit/application/config/test_validation.py`.
  Acceptance criteria (agent-executable): RED parameterized tests expose raw `ValueError`; GREEN all malformed classes raise field-specific `ConfigurationError`, valid boundary values parse, and secrets/invalid raw values are absent from exception/log captures. Existing precedence tests remain green.
  QA scenarios (name the exact tool + invocation): Happy: `uv run --frozen --no-sync pytest -q tests/unit/application/config/test_settings.py tests/unit/application/config/test_validation.py`. Failure: launch config parsing with `MCP_PORT=nan`, empty batch sizes, negative concurrency, and infinite timeouts; assert typed failure before engine construction. Evidence `<attemptDir>/task-7-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `unspecified-low` - repetitive but bounded configuration-boundary repair.
  Commit: Y | `fix(config): report typed numeric parsing failures`

- [x] 8. Introduce the workspace storage coordinator protocol
  What to do / Must NOT do: Add a core protocol/value types and an infrastructure Portalocker implementation providing stable per-workspace paths, process-local reentrancy, shared/exclusive leases, generation read/publish, 30-second production timeout with injected test timeout, and typed acquisition/generation errors. Missing generation is 0. Lock files remain after release; corrupt generation fails closed. This task builds the seam but does not thread every manager path yet.
  Parallelization: Wave 2 | Blocked by: 1, 2 | Blocks: 15-22
  References (executor has NO interview context - be exhaustive): new `reflectlog/core/storage_coordination.py`; new `reflectlog/infrastructure/storage_coordinator.py`; layer constraints `.sentrux/rules.toml:7-42`; current local locks `manager.py:138-153`, `usearch_engine.py:192-198`, `tantivy_engine.py:120-136`, `memory_store.py:115-118`; workspace paths in config/adapters.
  Acceptance criteria (agent-executable): RED new unit/integration tests fail because no cross-process coordinator exists; GREEN proves reentrancy, separate-workspace parallelism, same-workspace contention, typed timeout, exception release, owner-process kill release without lock-file deletion, atomic generation publication, legacy generation 0, and corrupt-generation fail-closed behavior.
  QA scenarios (name the exact tool + invocation): Happy: `uv run --frozen --no-sync pytest -q tests/unit/infrastructure/test_storage_coordinator.py tests/integration/test_storage_coordinator_processes.py`. Failure: child acquires exclusive lease then is killed; parent acquires and publishes a new generation while the sidecar pathname persists. Evidence `<attemptDir>/task-8-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `ultrabrain` - this is the one cohesive hard protocol that all storage correctness depends on.
  Commit: Y | `feat(storage): coordinate workspace access across processes`

- [x] 9. Offload pre-search recovery without weakening ordering
  What to do / Must NOT do: In `MemoryManager.search()`, await `asyncify(self.reconcile_pending_replacements)()` before backend search and index sizing. Preserve successful recovery-before-search, `InitializationError` propagation, existing treatment of recoverable non-initialization failures, and native-thread cancellation semantics. Do not move recovery into the search pipeline or run backends before it completes.
  Parallelization: Wave 2 | Blocked by: 2 | Blocks: 20
  References (executor has NO interview context - be exhaustive): `reflectlog/application/memory/manager.py:310-322,746-809`; `reflectlog/application/memory/replacement_recovery.py:27-90`; `tests/unit/application/memory/test_search_pipeline.py:1254-1371`; `tests/unit/application/memory/test_replacement_recovery.py`.
  Acceptance criteria (agent-executable): RED event-loop ticker test stalls on synchronous recovery; GREEN ticker advances while recovery blocks in a worker, backend calls remain absent until release, success invokes search once, initialization failure aborts search, and recoverable failure follows the pinned contract.
  QA scenarios (name the exact tool + invocation): Happy: `uv run --frozen --no-sync pytest -q tests/unit/application/memory/test_search_pipeline.py -k recovery_offload`. Failure: blocking fake raises `InitializationError`; assert exception identity/cause and zero backend calls. Evidence `<attemptDir>/task-9-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `quick` - narrow async boundary with an existing pattern.
  Commit: Y | `fix(search): offload pending recovery before retrieval`

- [x] 10. Enforce the hybrid backend outage matrix
  What to do / Must NOT do: Change the canonical `search_strategies.SearchPipeline` so semantic error plus no live FTS result raises `SearchError`; semantic error plus live FTS falls back; Tantivy error plus semantic hits falls back; both successful empty returns `[]`; both failed raises. Decide after SQLite live-hit filtering so stale/tombstoned raw FTS hits count as empty. Preserve error chaining and public result shape.
  Parallelization: Wave 2 | Blocked by: 2 | Blocks: 14, 28
  References (executor has NO interview context - be exhaustive): `reflectlog/application/memory/search_strategies.py:131-166,212-335,806-827`; `reflectlog/application/memory/AGENTS.md:39`; contradictory test `tests/unit/application/memory/test_search_pipeline.py:684-696` and adjacent matrix tests `:596-776`.
  Acceptance criteria (agent-executable): RED first changes the stale empty-success test to expect `SearchError` and fails against current code; GREEN parameterized matrix passes for live, empty, stale-only, disabled, and both-failed backends, including exact cause preservation.
  QA scenarios (name the exact tool + invocation): Happy: `uv run --frozen --no-sync pytest -q tests/unit/application/memory/test_search_pipeline.py -k 'backend or semantic_error or tantivy'`. Failure: semantic raises and FTS returns only content absent from SQLite; assert `SearchError` rather than empty success. Evidence `<attemptDir>/task-10-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `quick` - one canonical decision seam with dense existing tests.
  Commit: Y | `fix(search): surface semantic outage with empty fts`

- [x] 11. Make recency all-or-nothing per candidate batch
  What to do / Must NOT do: Apply recency only after CE thresholding and only when every candidate has a valid timestamp. Missing, empty, malformed, or unavailable metadata disables decay for the whole batch and preserves post-CE order/scores. Do not change the low-level scoring contract for unrelated callers unless references prove it is exclusive to this pipeline.
  Parallelization: Wave 2 | Blocked by: 2 | Blocks: 14
  References (executor has NO interview context - be exhaustive): `reflectlog/infrastructure/cross_encoder_reranker.py:311-328`; `reflectlog/utility/scoring.py:462-514`; `tests/unit/application/memory/reranking/test_normalization.py:482-505`; `tests/unit/infrastructure/test_cross_encoder_reranker.py:756-942`.
  Acceptance criteria (agent-executable): RED current partial-timestamp test demonstrates rank reversal; GREEN complete maps decay normally while every incomplete/invalid/failure case returns exact pre-recency scores and order. One-result/empty-result paths remain unchanged.
  QA scenarios (name the exact tool + invocation): Happy: `uv run --frozen --no-sync pytest -q tests/unit/application/memory/reranking/test_normalization.py tests/unit/infrastructure/test_cross_encoder_reranker.py -k recency`. Failure: two candidates where only the stronger old hit has a timestamp must no longer reverse. Evidence `<attemptDir>/task-11-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `quick` - focused ranking policy change with exact numeric tests.
  Commit: Y | `fix(search): disable recency for incomplete timestamp batches`

- [x] 12. Make the OpenAI SDK the sole retry owner
  What to do / Must NOT do: Remove outer OpenAI retry loops from sync/async Qwen embeddings and smart replacement while preserving SDK `max_retries=2`, typed exhaustion, cancellation, and non-OpenAI provider behavior. Recovery must not trigger a second implicit embedding sequence under write locks; failed precompute leaves the intent pending for a later recovery. Do not catch all exceptions as retryable.
  Parallelization: Wave 2 | Blocked by: 1, 2 | Blocks: 13, 28, 29
  References (executor has NO interview context - be exhaustive): `reflectlog/infrastructure/embeddings/qwen3_embedding.py:102,122-176,212-258`; `reflectlog/infrastructure/smart_replacer.py:175-208`; `reflectlog/application/memory/replacement_recovery.py:316-412,510-575`; `tests/unit/infrastructure/test_qwen3_embedding.py:175-737`; `tests/unit/application/memory/test_replacement_recovery.py`.
  Acceptance criteria (agent-executable): RED loopback-wire tests observe outer amplification; GREEN 500,500,200 produces exactly three HTTP requests, 400 produces one, exhaustion surfaces the approved typed result/cause, no provider call occurs under `_write_lock`, and unresolved recovery remains pending without a second sequence.
  QA scenarios (name the exact tool + invocation): Happy: `uv run --frozen --no-sync pytest -q tests/unit/infrastructure/test_qwen3_embedding.py tests/unit/application/memory/test_replacement_recovery.py -k retry`. Failure: loopback returns permanent 400 and assertion requires one wire request, no sleeps, no completed intent, and no cached vector. Evidence `<attemptDir>/task-12-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `deep` - provider, recovery, sync/async, and lock ownership must converge on one retry contract.
  Commit: Y | `fix(embed): use one OpenAI retry policy`

- [x] 13. Single-flight synchronous and asynchronous cache misses
  What to do / Must NOT do: Add per-key in-flight coordination to `CachedEmbeddings.embed_query()` and `aembed_query()`: one leader computes, same-key waiters share the exact value or exception, different keys proceed independently, failures/empty vectors are not cached, and waiter cancellation does not cancel the leader or poison later calls. Preserve key normalization, LRU ordering/capacity, batch behavior, and existing hit/miss semantics; add a separate coalesced counter only if it is typed and tested.
  Parallelization: Wave 3 | Blocked by: 12 | Blocks: 29
  References (executor has NO interview context - be exhaustive): `reflectlog/infrastructure/embeddings/cached_embeddings.py:13-156,158-198,200-286`; `reflectlog/core/types.py:147-219`; `tests/unit/infrastructure/test_cached_embeddings.py:152-340`.
  Acceptance criteria (agent-executable): RED barrier tests make eight same-key calls and observe eight provider calls; GREEN sync and async variants observe one provider call, identical results, one cache entry, and no global serialization for two keys. Leader failure reaches every waiter, is absent from cache, and the next call invokes the provider once.
  QA scenarios (name the exact tool + exact invocation): Happy: `uv run --frozen --no-sync pytest -q tests/unit/infrastructure/test_cached_embeddings.py -k 'single_flight or concurrent'`. Failure: cancel one async waiter and force one leader exception; remaining waiters receive the correct outcome and a subsequent call succeeds. Evidence `<attemptDir>/task-13-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `deep` - sync/async coordination and cancellation require one shared invariant.
  Commit: Y | `perf(embed): single flight concurrent cache misses`

- [x] 14. Bulk-load candidate records for timestamp hydration
  What to do / Must NOT do: Add `MemoryStore.get_records_by_contents(workspace_id, contents) -> list[IStoredMemory]`, delegate it through `USearchEngine`, and expose it on `ISemanticSearchEngine`. Use bounded SQL chunks and workspace filtering; preserve unique content identity and existing scalar methods. Replace `_complete_timestamp_map()`'s two queries per candidate. A missing/invalid timestamp disables recency per task 11; a storage exception raises `SearchError` rather than returning a partial map.
  Parallelization: Wave 3 | Blocked by: 10, 11 | Blocks: 29
  References (executor has NO interview context - be exhaustive): `reflectlog/core/types.py:222-326`; `reflectlog/infrastructure/memory_store.py:467-557,779-811`; `reflectlog/infrastructure/usearch_engine.py`; `reflectlog/application/memory/search_strategies.py:773-804`; tests `tests/unit/infrastructure/test_memory_store.py`, `test_usearch_engine.py`, and `tests/unit/application/memory/test_search_strategies.py:668-743`.
  Acceptance criteria (agent-executable): RED SQLite trace for five FTS-only candidates records ten SELECTs; GREEN records one bounded SELECT, returns only requested workspace records, keeps existing semantic timestamps without requery, and handles duplicate/missing inputs deterministically. Database failure raises `SearchError` with cause.
  QA scenarios (name the exact tool + exact invocation): Happy: `uv run --frozen --no-sync pytest -q tests/unit/infrastructure/test_memory_store.py tests/unit/infrastructure/test_usearch_engine.py tests/unit/application/memory/test_search_strategies.py -k 'records_by_contents or timestamp'`. Failure: inject SQLite operational failure and assert no partial recency map or reordered output. Evidence `<attemptDir>/task-14-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `unspecified-high` - one typed protocol crosses core, storage, engine, and search layers.
  Commit: Y | `perf(search): bulk load candidate timestamps`

- [ ] 15. Publish refreshed USearch snapshots atomically
  What to do / Must NOT do: Give `USearchEngine` the coordinator; under an exclusive lease reload when generation/file identity changed, then mutate the refreshed index. Save to a unique same-directory temp path, reopen/validate the temp index, fsync it, atomically `os.replace()` the live file, fsync the parent where supported, and clear dirty state only after HNSW publication succeeds. Coordinated startup removes orphan temp files. `USearchEngine` must not publish the workspace generation or complete durable intents because it cannot prove Tantivy/cross-store convergence. Never save a stale in-memory snapshot or create an empty live HNSW over populated SQLite.
  Parallelization: Wave 3 | Blocked by: 8, 2 | Blocks: 18-20, 23-25
  References (executor has NO interview context - be exhaustive): `reflectlog/infrastructure/usearch_engine.py:158-365,392-620,859-956,958-1028`; SQLite SoT rules in `reflectlog/infrastructure/AGENTS.md`; tests `tests/unit/infrastructure/test_usearch_engine.py`, `tests/integration/test_memory_manager_usearch.py`, and task 2 fixture.
  Acceptance criteria (agent-executable): RED two-engine stale-snapshot test ends with two SQLite rows and one vector; GREEN reopens with all rows/vectors. Engine-owned failpoints before save, after temp save/validation, after file fsync, and before/after replace yield either the old valid or complete new HNSW, never partial/corrupt/empty state. Tests assert this layer does not advance generation or complete an intent; legacy fixture remains exact.
  QA scenarios (name the exact tool + exact invocation): Happy: `uv run --frozen --no-sync pytest -q tests/unit/infrastructure/test_usearch_engine.py tests/integration/test_memory_manager_usearch.py -k 'atomic or external or stale'`. Failure: terminate a spawned writer at each HNSW publication barrier; a fresh process validates SQLite rows, vector count/search, unchanged orchestration-owned generation/intent state, and temp cleanup. Evidence `<attemptDir>/task-15-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `deep` - native index publication, crash boundaries, and external refresh are inseparable.
  Commit: Y | `fix(usearch): publish refreshed indexes atomically`

- [ ] 16. Scope Tantivy readers and writers to coordinator leases
  What to do / Must NOT do: Give `TantivyEngine` the coordinator; open/reload/search under a shared lease and create/commit/wait/relinquish each writer under one exclusive lease. Ensure no reader/writer references replaceable directory state after release. Replace `_delete_via_rebuild()` and `compact()` request-path directory replacement with Tantivy logical delete/commit/merge or defer physical reclamation while preserving logical deletion. Preserve default soft-delete, tombstone counts, no compact-on-delete, read-after-write reload, and backup compatibility for legacy startup only.
  Parallelization: Wave 3 | Blocked by: 8, 2 | Blocks: 18-22, 23-27
  References (executor has NO interview context - be exhaustive): `reflectlog/infrastructure/tantivy_engine.py:120-158,292-340,491-706,740-823,903-993,1037-1260,1430-1579`; `tests/unit/infrastructure/test_tantivy_engine.py:698-827,1042-1133,1623-1771,2218-2339`; task 2 legacy fixture.
  Acceptance criteria (agent-executable): RED deterministic add-versus-rebuild loses a committed document; GREEN shared readers and exclusive maintenance cannot overlap, add-before-maintenance and maintenance-before-add both preserve all active documents, writer ownership is released after each batch/death, hard-delete content stays absent after restart, and no request path depends on destructive directory replacement.
  QA scenarios (name the exact tool + exact invocation): Happy: `uv run --frozen --no-sync pytest -q tests/unit/infrastructure/test_tantivy_engine.py -k 'writer or reader or compact or rebuild or external'`. Failure: kill a process holding the writer and verify another process commits/reloads without manual directory/lock cleanup; active reader behavior is correctness-first on every supported platform. Evidence `<attemptDir>/task-16-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `deep` - native writer ownership and maintenance semantics span the whole engine.
  Commit: Y | `fix(tantivy): scope index ownership to coordinated operations`

- [ ] 17. Add a reusable cross-platform focused gate runner
  What to do / Must NOT do: Add a PEP 723 `scripts/run_platform_gates.py` that creates unique temp roots/ephemeral ports, runs dependency/coordinator/storage/CLI focused checks from the frozen environment, emits machine-readable JSON, and scrubs processes/temp resources. It must accept an output path and failure-injection flag. Do not install dependencies, mutate tracked files, use fixed ports, or call paid/external services.
  Parallelization: Wave 3 | Blocked by: 1, 3, 8 | Blocks: 30
  References (executor has NO interview context - be exhaustive): `scripts/`; wrapper contract from task 3; `tests/AGENTS.md`; Portalocker and platform modules under `reflectlog/utility/platforms/`; evidence naming in Verification strategy.
  Acceptance criteria (agent-executable): Happy invocation returns 0 and JSON with OS/Python/tool versions, commands, exits, teardown receipts, and artifact paths. Failure injection returns nonzero with the failed scenario, still tears down all children/temp roots, and leaves tracked state unchanged.
  QA scenarios (name the exact tool + exact invocation): Happy: `uv run --frozen --no-sync scripts/run_platform_gates.py --focused --output <attemptDir>/task-17-reflectlog-correctness-performance-remediation.json`. Failure: add `--inject-failure coordinator-timeout`; assert nonzero JSON and no surviving PID/port/temp root. Evidence is the JSON itself.
  Recommended task executor category: `unspecified-high` - cross-platform orchestration must be deterministic and self-cleaning.
  Commit: Y | `test(platform): add focused release gate runner`

- [ ] 18. Integrate coordination into MemoryManager lifecycle and direct operations
  What to do / Must NOT do: Construct one coordinator before persistent engines, pass it to USearch/Tantivy/factory paths, and make startup, direct add/delete, pre-search refresh, `get_all`/count consistency reads, and close obey the coordinator lease and existing `_write_lock` then `_lock`. Revalidate duplicate/delete decisions after exclusive acquisition. For every direct write, validate SQLite, USearch, and enabled Tantivy convergence, then publish the next generation, then complete the durable intent; add test-only failpoints immediately before/after generation publication and intent completion. Shared leases remain through active backend reads where required by task 16. Do not hold exclusive leases during embedding/reranking or change public method signatures.
  Parallelization: Wave 4 | Blocked by: 8, 15, 16 | Blocks: 22-29
  References (executor has NO interview context - be exhaustive): `reflectlog/application/memory/manager.py:96-203,280-322,553-809,895-1077,1222-1321`; `reflectlog/application/memory/engine_factory.py:70-169`; `reflectlog/application/mcp_server.py:106`; manager lock guidance `reflectlog/application/AGENTS.md:30-31`; tests `tests/unit/application/memory/test_manager.py`, `tests/unit/application/test_memory_manager.py`.
  Acceptance criteria (agent-executable): RED two long-lived managers fail external visibility/overwrite tests; GREEN managers opened before either write observe each completed generation, same-content identity remains unique, generation advances only after all enabled stores validate and before the intent completes, lock recorder proves coordinator outermost, no exclusive lease spans embedder/fusion/reranker calls, close relinquishes engine/OS ownership, and post-close calls still fail.
  QA scenarios (name the exact tool + exact invocation): Happy: `uv run --frozen --no-sync pytest -q tests/unit/application/memory/test_manager.py tests/unit/application/test_memory_manager.py tests/integration/test_memory_manager_usearch.py -k 'coordinator or external or close'`. Failure: force lease timeout and assert typed operation failure, unchanged pending intent, and later successful recovery. Evidence `<attemptDir>/task-18-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `deep` - manager composition and lock ownership are the cross-engine control plane.
  Commit: Y | `fix(storage): coordinate manager lifecycle and direct writes`

- [ ] 19. Revalidate phased adds under the exclusive workspace lease
  What to do / Must NOT do: Thread the coordinator into `StoragePhase`; keep embedding and LLM replacement detection outside the lease, then under exclusive ownership refresh engines and recheck duplicates/replacements/deletes before journal/storage mutation. Batch one request per lease and preserve journal-before-mutation, NEW-before-OLD, add results, dry-run, and disabled-hybrid behavior. After SQLite, USearch, and enabled Tantivy validate, publish the next generation and only then complete the intent; expose the orchestration failpoints from task 18 here as dependency-injected test seams. Do not trust pre-lock dedup/replacement decisions.
  Parallelization: Wave 4 | Blocked by: 8, 15, 16, 18 | Blocks: 23-29
  References (executor has NO interview context - be exhaustive): `reflectlog/application/memory/add_phases.py:626-757,689-946,918-1012`; `reflectlog/application/memory/AGENTS.md:27-36`; tests `tests/unit/application/memory/test_add_phases.py` and integration add/recovery suites.
  Acceptance criteria (agent-executable): RED barrier lets another process change dedup/replacement state between phase 2 and storage and current code makes a stale decision; GREEN revalidation yields one active identity, correct archive/journal, all index convergence, generation publication before intent completion, and proves provider calls happen without the exclusive lease.
  QA scenarios (name the exact tool + exact invocation): Happy: `uv run --frozen --no-sync pytest -q tests/unit/application/memory/test_add_phases.py tests/integration/test_replacement_recovery.py -k 'coordinator or stale or revalidate'`. Failure: competing writer inserts the same/replacement content before lease acquisition; assert no duplicate, stale deletion, or completed divergent intent. Evidence `<attemptDir>/task-19-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `deep` - phased async decisions must be revalidated at one persistence seam.
  Commit: Y | `fix(storage): revalidate phased writes under coordination`

- [ ] 20. Coordinate journal replay and crash convergence
  What to do / Must NOT do: Acquire the coordinator before recovery locks, refresh both indexes, precompute embeddings outside exclusive ownership, and apply pending transitions once under later-write-wins rules. Validate SQLite, USearch, and enabled Tantivy convergence, publish generation, then complete the intent, using the same dependency-injected before/after generation and intent failpoints as direct/phased writes. Prevent two processes from replaying the same transition concurrently; exhausted embeddings leave it pending without provider retry under locks.
  Parallelization: Wave 4 | Blocked by: 9, 15, 16, 18 | Blocks: 22-29
  References (executor has NO interview context - be exhaustive): `reflectlog/application/memory/replacement_recovery.py:27-90,93-220,253-385,411-581`; `reflectlog/infrastructure/memory_store.py:955-1129`; unit/integration recovery tests including `tests/integration/test_replacement_recovery.py`.
  Acceptance criteria (agent-executable): RED spawned processes both replay one pending intent or diverge; GREEN only one coordinated application occurs, fresh-process inspection finds SQLite/USearch/Tantivy/archive/generation converged and zero pending rows only after success, later DELETE/ADD is not resurrected, and a failed precompute remains retryable.
  QA scenarios (name the exact tool + exact invocation): Happy: `uv run --frozen --no-sync pytest -q tests/unit/application/memory/test_replacement_recovery.py tests/integration/test_replacement_recovery.py -k 'multi or coordinator or later_write'`. Failure: kill replay after each backend/generation boundary and verify a third process converges idempotently. Evidence `<attemptDir>/task-20-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `deep` - durable state machine and interprocess ownership must be proved together.
  Commit: Y | `fix(recovery): coordinate journal replay across processes`

- [ ] 21. Make tombstone cache invalidation workspace-aware
  What to do / Must NOT do: For local writes invalidate only the touched workspace; on external storage-generation reload conservatively invalidate affected/current cache entries. Validate capacity is positive and keep typed immutable cache state/LRU. Preserve tombstone/live duplicate semantics, search limits, restart behavior, and no compact-on-delete. Do not claim sublinear cold rebuild without changing the index design.
  Parallelization: Wave 4 | Blocked by: 16 | Blocks: 23-29
  References (executor has NO interview context - be exhaustive): `reflectlog/infrastructure/tantivy_engine.py:526-706,740-823`; tests `tests/unit/infrastructure/test_tantivy_engine.py:698-827,1623-1771`; config/validation ownership for cache capacity.
  Acceptance criteria (agent-executable): RED commit clears every workspace and next lookup scans all requested documents; GREEN local commit preserves unrelated workspace entries, external generation forces safe refresh, cache hits avoid scans, LRU/positive capacity hold, and search never leaks deleted content. Record scan/query counts and real results.
  QA scenarios (name the exact tool + exact invocation): Happy: `uv run --frozen --no-sync pytest -q tests/unit/infrastructure/test_tantivy_engine.py -k 'tombstone_cache or generation or workspace'`. Failure: concurrent external commit/search plus invalid capacity; assert safe reload or typed config error, not stale live results. Evidence `<attemptDir>/task-21-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `unspecified-high` - cache policy is bounded but coupled to external generation.
  Commit: Y | `perf(tantivy): invalidate tombstones by workspace generation`

- [ ] 22. Preserve coordinated shutdown and signal behavior
  What to do / Must NOT do: Keep the coordinator outermost while `MemoryManager.close()` flushes/relinquishes Tantivy, atomically persists USearch, closes stores, and marks closed. Thread this through FastMCP lifespan and installed CLI platform-native control handling: register POSIX `SIGINT`/`SIGTERM`, and on Windows register `SIGBREAK` while launching the test/server subprocess with `CREATE_NEW_PROCESS_GROUP` so the harness can deliver `CTRL_BREAK_EVENT`. Preserve retryable close failure and second-control-event prompt termination; test forced termination separately as abrupt death. Do not release the workspace lease while an engine resource remains open or treat Windows `terminate()` as graceful signaling.
  Parallelization: Wave 4 | Blocked by: 18, 20 | Blocks: 27-29
  References (executor has NO interview context - be exhaustive): `reflectlog/application/memory/manager.py:1222-1321`; `reflectlog/application/mcp_server.py:336-358`; `reflectlog/server.py:161-203,254-323`; existing server/shutdown tests; new `tests/integration/test_server_shutdown_recovery.py`, expanded by task 27.
  Acceptance criteria (agent-executable): RED contended shutdown can hand off before persistence/resource release; GREEN the native graceful event exits only after durable close, another process then acquires/writes, a forced persistence error leaves manager retryable and intent pending, and a second native control event terminates within a bounded test event without corrupting the prior generation. POSIX proves both `SIGINT` and `SIGTERM`; Windows proves `SIGBREAK` via `CTRL_BREAK_EVENT`; neither platform path is skipped.
  QA scenarios (name the exact tool + exact invocation): Happy/failure: `uv run --frozen --no-sync pytest -q tests/integration/test_server_shutdown_recovery.py -k 'graceful_signal or lease_handoff or second_signal'`. The harness starts the installed CLI, writes synthetic memory, sends POSIX `SIGTERM`/`SIGINT` or Windows `CTRL_BREAK_EVENT` to a `CREATE_NEW_PROCESS_GROUP` process, requires exit 0 after persisted stores/generation validate, proves a second process acquires the released lease, injects persistence/lease failure, sends the second platform-native event, requires bounded termination, exercises forced termination only in the abrupt-death case, and always records PID/port/temp teardown. Evidence `<attemptDir>/task-22-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `deep` - OS signals, FastMCP lifecycle, and three-store finalization form one boundary.
  Commit: Y | `fix(server): coordinate durable shutdown handoff`

- [ ] 23. Prove duplicate and disjoint multiprocess writes
  What to do / Must NOT do: Add real spawned-process scenarios where four writers open before writing, add 1,000 synthetic records with disjoint and shared-duplicate sets, and a fifth fresh process inspects persistence. Use barriers/events/queues only; no sleeps, mock engines, or same-process substitute.
  Parallelization: Wave 5 | Blocked by: 18-20 | Blocks: 29, 30
  References (executor has NO interview context - be exhaustive): new `tests/integration/test_multiprocess_writers.py`; `tests/integration/test_concurrent_operations.py:24-27,83-251`; `tests/integration/test_thread_safety.py`; task 15/16/18 contracts.
  Acceptance criteria (agent-executable): All child exits are 0; inspector asserts exact SQLite unique rows, USearch vector count and sampled semantic results, Tantivy live results, generation progression, zero pending intents, no lost disjoint write, and one identity per shared duplicate.
  QA scenarios (name the exact tool + exact invocation): Happy: `RUN_USEARCH_CONCURRENCY_TESTS=1 uv run --frozen --no-sync pytest -q tests/integration/test_multiprocess_writers.py -k 'duplicate or disjoint'`. Failure: inject one lease timeout/writer crash; surviving committed batches remain exact and recovery converges the pending batch. Evidence `<attemptDir>/task-23-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `deep` - real native multiprocess evidence is the product contract.
  Commit: Y | `test(storage): prove coordinated multiprocess writes`

- [ ] 24. Prove add/delete/re-add/replacement race semantics
  What to do / Must NOT do: Add deterministic spawned interleavings for add versus delete, delete versus re-add, concurrent replacements, and later DELETE/ADD journal order. Inspect from a fresh third process. Preserve one active successor, exact archive audit, idempotent replay, and default soft-delete behavior.
  Parallelization: Wave 5 | Blocked by: 18-20 | Blocks: 29, 30
  References (executor has NO interview context - be exhaustive): new/extended `tests/integration/test_multiprocess_writers.py`; `tests/integration/test_replacement_recovery.py`; `reflectlog/application/memory/replacement_recovery.py:93-385`; `add_phases.py:689-946`.
  Acceptance criteria (agent-executable): Every enumerated barrier order produces the documented later-write-wins state, one active row/vector/live FTS copy, correct archive rows, zero stale resurrection, correct generation, zero pending intents after successful replay, and identical state after a second recovery.
  QA scenarios (name the exact tool + exact invocation): Happy: `RUN_USEARCH_CONCURRENCY_TESTS=1 uv run --frozen --no-sync pytest -q tests/integration/test_multiprocess_writers.py -k 'delete or replacement or later_write'`. Failure: kill one actor after intent but before an index commit; fresh recovery must converge to the order encoded in durable transitions. Evidence `<attemptDir>/task-24-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `deep` - ordering semantics span journal, identity, and two indexes.
  Commit: Y | `test(storage): cover cross process mutation races`

- [ ] 25. Prove every USearch crash-publication boundary
  What to do / Must NOT do: Compose the engine-owned failpoints from task 15 (temp save, validation, file fsync, replace, directory fsync) with the orchestration-owned failpoints from tasks 18-20 (generation publication and intent completion). Spawn a writer, terminate it at each barrier, then inspect with a fresh process. Production has no global debug flag or sleep; failpoints stay dependency-injected in tests.
  Parallelization: Wave 5 | Blocked by: 15, 18-20 | Blocks: 29, 30
  References (executor has NO interview context - be exhaustive): `reflectlog/infrastructure/usearch_engine.py:859-956`; coordinator/generation task 8; recovery task 20; `tests/integration/test_replacement_recovery.py:221-250`; new `tests/integration/test_usearch_atomic_recovery.py`.
  Acceptance criteria (agent-executable): At every kill point, the fresh inspector opens either the exact previous generation or the complete new generation, never partial/corrupt/silent-empty HNSW. SQLite rows, vector/search results, generation, temp files, and pending intent match the crash table; repeated recovery is idempotent.
  QA scenarios (name the exact tool + exact invocation): Happy/failure matrix: `RUN_USEARCH_CONCURRENCY_TESTS=1 uv run --frozen --no-sync pytest -q tests/integration/test_usearch_atomic_recovery.py`; the test itself kills the process at every named failpoint and records child/inspector receipts. Evidence `<attemptDir>/task-25-reflectlog-correctness-performance-remediation.json`.
  Recommended task executor category: `deep` - process death and native publication require one end-to-end harness.
  Commit: Y | `test(usearch): verify crash atomic publication`

- [ ] 26. Prove Tantivy reader/writer/death behavior across processes
  What to do / Must NOT do: Add real-process tests for shared active reads versus exclusive mutation/maintenance, writer death, delete/merge, external commit visibility, and restart. Do not depend on old destructive `.rebuild-bak` recovery for new request paths; retain only the legacy-open characterization from task 2.
  Parallelization: Wave 5 | Blocked by: 16, 18, 21 | Blocks: 29, 30
  References (executor has NO interview context - be exhaustive): `reflectlog/infrastructure/tantivy_engine.py:491-595,611-823,903-1260,1430-1579`; `tests/unit/infrastructure/test_tantivy_engine.py`; new/extended `tests/integration/test_multiprocess_tantivy.py`.
  Acceptance criteria (agent-executable): Active reader and conflicting exclusive operation follow the supported Portalocker semantics; no unsafe overlap occurs. Killing a writer releases ownership, a new process commits/reloads, deleted content remains absent, added content remains present, caches reflect generation, and no live object references a replaced directory after lease release.
  QA scenarios (name the exact tool + exact invocation): Happy: `uv run --frozen --no-sync pytest -q tests/integration/test_multiprocess_tantivy.py`. Failure: kill a writer between mutation and commit and a reader during external commit; fresh process reports exact committed snapshot and recoverable pending state. Evidence `<attemptDir>/task-26-reflectlog-correctness-performance-remediation.json`.
  Recommended task executor category: `deep` - native searcher/writer lifecycle must be proved in independent processes.
  Commit: Y | `test(tantivy): verify process scoped index ownership`

- [ ] 27. Prove graceful shutdown, contention, and abrupt-death recovery
  What to do / Must NOT do: Drive the installed server and managers through POSIX `SIGTERM`/`SIGINT`, Windows `SIGBREAK` delivered by `CTRL_BREAK_EVENT` to a `CREATE_NEW_PROCESS_GROUP` child, lease contention, first close failure/retry, a second native control event, and a separate forced abrupt kill/termination. Use unique ports/temp roots and complete teardown. Assert durable state through a fresh process; never infer success from log text or surviving lock-file absence, and never classify Windows forced termination as graceful.
  Parallelization: Wave 5 | Blocked by: 22-26 | Blocks: 29, 30
  References (executor has NO interview context - be exhaustive): `reflectlog/server.py:254-323`; `reflectlog/application/mcp_server.py:336-358`; `reflectlog/application/memory/manager.py:1222-1321`; tests `tests/unit/test_server.py` and new `tests/integration/test_server_shutdown_recovery.py`.
  Acceptance criteria (agent-executable): Each platform-native graceful event exits 0 only after all stores and generation are durable; another process acquires immediately. Forced persistence failure leaves retryable close/pending intent; the second native control event exits within the event-bounded scenario; forced abrupt death requires no lock-file deletion and converges on restart. POSIX and Windows matrix lanes execute their required path without skips, and no PID/port/temp root survives.
  QA scenarios (name the exact tool + exact invocation): `uv run --frozen --no-sync pytest -q tests/integration/test_server_shutdown_recovery.py` executes POSIX `SIGINT`/`SIGTERM`, Windows new-process-group `CTRL_BREAK_EVENT`/`SIGBREAK`, second-event, and separate forced-death scenarios as selected by the current OS, writes teardown receipts, and treats any skipped native scenario as failure. Evidence `<attemptDir>/task-27-reflectlog-correctness-performance-remediation.json`.
  Recommended task executor category: `deep` - signals, lock handoff, and crash recovery are one real-surface boundary.
  Commit: Y | `test(server): cover coordinated shutdown recovery`

- [ ] 28. Exercise the installed FastMCP surface without external credentials
  What to do / Must NOT do: Add an integration scenario that starts `uv run --frozen --no-sync reflectlog` on loopback with synthetic auth/config and a deterministic local OpenAI-compatible embedding server, then uses the real FastMCP 4.0.0b5 client. Exercise exactly the five registered tools through public schemas: add, search, get_all, remove, and health. Include malformed input and shutdown. Do not call internal engines directly or require user action/provider keys.
  Parallelization: Wave 5 | Blocked by: 4-7, 18-22 | Blocks: 29, 30
  References (executor has NO interview context - be exhaustive): `reflectlog/application/mcp_server.py`; tools under `reflectlog/application/tools/`; current `tests/integration/test_mcp_workflows.py`; opt-in live tests `tests/test_real_server_fastmcp.py`, `test_real_server_mcp.py`, `test_real_tools_direct.py` are not acceptable substitutes.
  Acceptance criteria (agent-executable): Installed server starts on an ephemeral loopback port, client lists five tools, adds two synthetic memories, searches exact expected result, gets all, removes one, health reports expected counts, and malformed `memories="not-a-list"` produces Pydantic schema failure. Shutdown exits cleanly and fresh reopen preserves expected state.
  QA scenarios (name the exact tool + exact invocation): Happy/failure: `uv run --frozen --no-sync pytest -q tests/integration/test_installed_mcp_surface.py`; test owns fake server, MCP process, client actions, port/PID/temp teardown, and response artifacts. Evidence `<attemptDir>/task-28-reflectlog-correctness-performance-remediation.json`.
  Recommended task executor category: `deep` - installed protocol, native storage, auth/config, and loopback provider must run together.
  Commit: Y | `test(mcp): exercise installed memory workflows`

- [ ] 29. Lock performance contracts and characterize supported capacity
  What to do / Must NOT do: Add hard assertions for one bounded timestamp query, one provider call per same key, no outer retry amplification, event-loop progress, workspace-scoped tombstone invalidation, and no heavy modules on CLI metadata paths. Run 1,000-record multiprocess functional stress and 10,000-record single-workspace correctness/capacity characterization. Record environment, cold/warm median and p95; do not invent wall-clock SLOs or retain unsupported speedup claims.
  Parallelization: Wave 5 | Blocked by: 13-28 | Blocks: 30-35
  References (executor has NO interview context - be exhaustive): hard-gate tests from tasks 4, 5, 9, 13, 14, 21, 23; `README.md:165-170`; `tests/load/AGENTS.md`; new `scripts/benchmark_remediation.py` with PEP 723 and machine-readable output.
  Acceptance criteria (agent-executable): All hard counts/import/correctness assertions pass. At 10,000 records every store count agrees and sampled search/delete/restart scenarios are correct. Benchmark JSON includes platform, Python, dependency versions, data size, repetitions, median/p95, query/provider counts, and no claimed ratio absent stable evidence.
  QA scenarios (name the exact tool + exact invocation): Happy: `uv run --frozen --no-sync scripts/benchmark_remediation.py --records 10000 --repetitions 5 --output <attemptDir>/task-29-reflectlog-correctness-performance-remediation.json`. Failure: `--inject-regression timestamp-n-plus-one` exits nonzero and identifies the violated hard gate, then tears down. Evidence is the JSON.
  Recommended task executor category: `unspecified-high` - combines deterministic contracts with bounded measurement, not algorithm redesign.
  Commit: Y | `perf(test): lock remediation performance contracts`

- [ ] 30. Run the focused storage gates on macOS, Linux, and Windows
  What to do / Must NOT do: Add a focused GitHub Actions matrix (or the repository's eventual equivalent) that installs from the lock and runs task 17 plus dependency, coordinator, USearch, Tantivy, multiprocess, CLI, shutdown, and configured full gates on macOS, Ubuntu, and Windows. POSIX jobs must execute `SIGINT`/`SIGTERM`; Windows must execute the `CREATE_NEW_PROCESS_GROUP` plus `CTRL_BREAK_EVENT`/`SIGBREAK` graceful path, with forced termination retained as a separate abrupt-death scenario. Upload machine-readable evidence. Do not skip a required native signal case, self-certify an OS that did not execute, add deployment/release automation, or include secrets.
  Parallelization: Wave 6 after Ruff integration | Blocked by: 17, 23-29, 31-34 | Blocks: 35, F1-F4
  References (executor has NO interview context - be exhaustive): new `.github/workflows/platform-storage.yml`; `scripts/run_platform_gates.py`; `pyproject.toml` Python requirement; Windows/macOS/Linux platform modules; no existing hosted CI was found.
  Acceptance criteria (agent-executable): All three matrix jobs install once with frozen uv, execute every check through `uv run --frozen --no-sync`, run focused and configured full gates against the integrated Ruff-clean tree, prove their required native graceful-signal path plus forced-death recovery with zero required skips, upload JSON/JUnit/coverage artifacts, and report green URLs/IDs. Any unavailable, skipped, or failed OS remains an explicit blocker rather than a documented support claim.
  QA scenarios (name the exact tool + exact invocation): Happy: trigger the workflow for the task branch/PR with `gh workflow run platform-storage.yml --ref <branch>` or consume the PR run, then `gh run watch <id> --exit-status` and download artifacts; inspect each job's shutdown JSON for the expected POSIX or Windows event and `skipped_required=0`. Failure: a matrix job with an injected coordinator or native-signal test failure must report nonzero and still upload diagnostics. Evidence `<attemptDir>/task-30-reflectlog-correctness-performance-remediation.txt` plus downloaded artifacts.
  Recommended task executor category: `unspecified-high` - focused cross-platform release proof with no product behavior change.
  Commit: Y | `ci(storage): verify coordinated indexes across platforms`

- [ ] 31. Resolve configured Ruff findings in core, utility, server, config, and stubs
  What to do / Must NOT do: Apply only the existing configured safe Ruff/format fixes to `reflectlog/core`, `reflectlog/utility`, `reflectlog/server.py`, `reflectlog/application/config`, and `stubs`; manually resolve remaining naming/import/typing findings without behavior change. No new rules, unsafe fixes, broad refactor, or behavior mixed into this mechanical commit.
  Parallelization: Wave 6 | Blocked by: 29 | Blocks: 30, F1-F4
  References (executor has NO interview context - be exhaustive): `pyproject.toml:200-259`; direct baseline finding of 14 production/stub issues; focused tests for tasks 5-7 and core adapters.
  Acceptance criteria (agent-executable): `uv run --frozen --no-sync ruff check <listed paths>` and `uv run --frozen --no-sync ruff format --check <listed paths>` pass; related config/HTTP/server/core tests and both type checkers pass; diff review confirms no behavior beyond formatting/naming/import modernization. Formatting-only TDD exemption is recorded.
  QA scenarios (name the exact tool + exact invocation): Happy: run targeted Ruff/format/type/tests and save diff. Failure: if a fix changes an assertion/result, revert that hunk and create a new behavior RED/GREEN task rather than weakening the test. Evidence `<attemptDir>/task-31-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `quick` - bounded mechanical cleanup with explicit paths.
  Commit: Y | `style(core): satisfy configured Ruff rules`

- [ ] 32. Resolve configured Ruff findings in application code
  What to do / Must NOT do: Apply existing safe Ruff/format fixes to `reflectlog/application` excluding paths committed by task 31 as needed; preserve manager/search/recovery/fusion behavior and avoid module decomposition. Remaining behavior-sensitive findings are removed from this task and replanned separately.
  Parallelization: Wave 6 | Blocked by: 29 | Blocks: 30, F1-F4
  References (executor has NO interview context - be exhaustive): `pyproject.toml:200-259`; application AGENTS guidance; all focused tests from tasks 4, 9-12, 14, 18-22.
  Acceptance criteria (agent-executable): Targeted Ruff/format pass, all application unit/integration tests pass, type checkers remain clean, and only mechanical diffs are staged. No public environment/tool/search behavior changes.
  QA scenarios (name the exact tool + exact invocation): Happy: `uv run --frozen --no-sync ruff check reflectlog/application && uv run --frozen --no-sync ruff format --check reflectlog/application` plus application tests. Failure: compare behavior characterization before/after; any delta blocks and moves to its own todo. Evidence `<attemptDir>/task-32-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `quick` - mechanical package-scoped cleanup.
  Commit: Y | `style(application): satisfy configured Ruff rules`

- [ ] 33. Resolve configured Ruff findings in infrastructure and plugins
  What to do / Must NOT do: Apply existing safe Ruff/format fixes to `reflectlog/infrastructure` and `reflectlog/plugins`, preserving native engine, coordinator, retry, cache, and plugin contracts. No unsafe automatic fixes or unrelated storage refactor.
  Parallelization: Wave 6 | Blocked by: 29 | Blocks: 30, F1-F4
  References (executor has NO interview context - be exhaustive): `pyproject.toml:200-259`; infrastructure/plugin AGENTS files; tasks 1, 8, 12-16, 21 and their direct tests.
  Acceptance criteria (agent-executable): Targeted Ruff/format pass; focused storage/embed/cache/HTTP/plugin tests and real engine compatibility fixture pass; type checkers clean; staged diff is mechanical only.
  QA scenarios (name the exact tool + exact invocation): Happy: `uv run --frozen --no-sync ruff check reflectlog/infrastructure reflectlog/plugins && uv run --frozen --no-sync ruff format --check reflectlog/infrastructure reflectlog/plugins` plus direct tests. Failure: any persisted-result or provider-call delta blocks and is removed/replanned. Evidence `<attemptDir>/task-33-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `quick` - mechanical cleanup around already-tested native seams.
  Commit: Y | `style(infrastructure): satisfy configured Ruff rules`

- [ ] 34. Resolve configured Ruff findings in tests and scripts
  What to do / Must NOT do: Apply configured quote/import/format fixes across `tests` and `scripts`, including the approximately 1,500 existing quote findings; preserve test names, assertions, fixtures, collection scope, and shell behavior. Make pre-push/check hooks check-only. Do not run the legacy quote-replacement shell path, delete tests, or change expected behavior to pass.
  Parallelization: Wave 6 | Blocked by: 3, 29 | Blocks: 30, F1-F4
  References (executor has NO interview context - be exhaustive): `pyproject.toml:200-259`; baseline 1,501 test findings; `start-lint.sh:230-292`; `scripts/git-hooks/`; all tests introduced by tasks 1-30.
  Acceptance criteria (agent-executable): Ruff check/format pass for tests/scripts; pytest collection count is explained and stable; full suite/coverage pass; running hooks/wrappers creates no tracked diff. Any behavior-changing test edit is rejected.
  QA scenarios (name the exact tool + exact invocation): Happy: `uv run --frozen --no-sync ruff check tests scripts && uv run --frozen --no-sync ruff format --check tests scripts && uv run --frozen --no-sync pytest --collect-only -q`. Failure: introduce a temporary single-quoted docstring and failing assertion in a disposable copy; checks fail and do not rewrite them. Evidence `<attemptDir>/task-34-reflectlog-correctness-performance-remediation.txt`.
  Recommended task executor category: `quick` - high-volume but deterministic mechanical cleanup.
  Commit: Y | `style(tests): satisfy configured Ruff rules`

- [ ] 35. Document the supported coordination and release contract
  What to do / Must NOT do: Update README and focused storage/release documentation with local-filesystem lock behavior, sidecars, operation-scoped Tantivy lifecycle, crash/recovery guarantees, rollback, mandatory pinned betas/ranx, supported OS evidence, POSIX `SIGINT`/`SIGTERM` versus Windows new-process-group `CTRL_BREAK_EVENT`/`SIGBREAK` graceful shutdown, separate forced-death recovery, NFS client-local and SMB/CIFS `nobrl` exclusions, no distributed lock, no ACID claim, `<10K` capacity context, and actual benchmark results. Extend task 17's gate runner with a documentation mode that executes the documented setup/version/focused-check commands in a disposable copy and validates structured support claims against task 23-30 artifacts without pinning prose. Remove or qualify unsupported `5-8x` claims. Do not expose secrets, internal memory text, or claim unexecuted platform support.
  Parallelization: Wave 6 | Blocked by: 29, 30 | Blocks: F1-F4
  References (executor has NO interview context - be exhaustive): `README.md:133-196`; new `docs/storage-coordination.md`; `.env.example`; `scripts/run_platform_gates.py` from task 17; artifacts from tasks 1, 23-30; scope guardrails in this plan.
  Acceptance criteria (agent-executable): Documentation mode returns 0 only when disposable setup/version/focused-check commands pass and every structured support/performance/platform-signal claim names a successful artifact from the matching OS; legacy fixture reopens unchanged; unsupported filesystems/distributed cases are explicit; installation commands reproduce the pinned runtime. Injecting an unsupported claim or a skipped/mismatched native-signal artifact returns nonzero with a machine-readable rejection reason. No grep-only or natural-language prose test is added.
  QA scenarios (name the exact tool + exact invocation): Happy: `uv run --frozen --no-sync scripts/run_platform_gates.py --docs README.md docs/storage-coordination.md --artifacts <attemptDir> --output <attemptDir>/task-35-reflectlog-correctness-performance-remediation.json` runs documented commands in a disposable copy and validates artifact-backed claims. Failure: rerun with `--inject-failure unsupported-doc-claim`; require nonzero exit, `status: rejected`, the unsupported claim/artifact ID in JSON, complete teardown, and no tracked diff. Evidence is the JSON.
  Recommended task executor category: `writing` - evidence-backed technical contract and rollback guidance.
  Commit: Y | `docs(storage): define coordinated persistence support`

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [ ] F1. Plan compliance audit
  Verify every numbered todo has a matching commit/evidence receipt, every Must Have and fixed owner decision is implemented, every Must NOT Have is absent, all required RED/GREEN transitions are genuine, and no skipped/missing evidence is counted. Compare against the reviewed plan digest and final tree. Output `<attemptDir>/final-F1-plan-compliance.json` with APPROVE/REJECT and cited blockers.
  Recommended task executor category: `unspecified-high` - independent goal/constraint verification.

- [ ] F2. Code quality and frozen release gates
  Run `uv sync --frozen` once, then execute full Ruff check/format, ty, Pyright, and pytest with warnings-as-errors and coverage XML/JSON at >=90% through `uv run --frozen --no-sync`; also run LSP diagnostics for changed Python files and `sentrux-mcp_check_rules`. Prove an unrelated warning fails and no paid credentials/network are needed. Output `<attemptDir>/final-F2-quality.txt` plus coverage artifacts with APPROVE/REJECT.
  Recommended task executor category: `unspecified-high` - independent code/release review.

- [ ] F3. Agent-executed real-surface QA
  Execute installed CLI help/version, the real FastMCP tool workflow, multiprocess disjoint/duplicate/race cases, process-death lock release and journal convergence, USearch failpoint matrix, Tantivy reader/writer lifecycle, POSIX `SIGINT`/`SIGTERM` and Windows `CTRL_BREAK_EVENT`/`SIGBREAK` graceful behavior on their actual platform jobs, separate forced-death recovery, and 10,000-record characterization. Capture actions, responses, process exits, teardown receipts, native-event identity, skip counts, and persisted-state inspection in `<attemptDir>/final-F3-real-surface.json`. No human action or required-scenario skip counts.
  Recommended task executor category: `deep` - hands-on full-system QA across process and protocol surfaces.

- [ ] F4. Scope, compatibility, platform, and Git audit
  Reopen the legacy fixture; inspect MCP schemas, SQLite schema/IDs/paths, journal and fusion methods; confirm no Redis/provider/plugin/auth/new-tool/content-logging/unrelated-refactor scope creep. Verify macOS/Linux/Windows artifacts, atomic commit boundaries, explicit staging, no committed evidence, and final worktree preservation of the two pre-existing untracked files. Output `<attemptDir>/final-F4-scope.json` with APPROVE/REJECT.
  Recommended task executor category: `unspecified-high` - independent scope/data/Git fidelity review.

## Commit strategy
- One atomic commit per numbered todo; final verification creates no commit. Implementation and direct tests always land together.
- Before each commit, inspect `git status`, unstaged/staged diff, recent log, and path history; use explicit `git add <paths>` only. Never stage `.omgb/evidence` or the pre-existing untracked files.
- Match existing English Conventional Commit style. The exact subjects are listed in each todo; adjust only scope wording if recent path history requires it.
- Dependency order follows the matrix. Independent tasks may commit in parallel worktrees only when their files do not overlap; integrate in wave/dependency order.
- Mechanical Ruff commits 31-34 remain separate from behavioral commits and contain no logic changes.
- If a todo's RED/GREEN, real-surface, or compatibility gate fails, fix within that todo before commit or revert that todo's work. Do not stack compensating commits on a broken increment.
- Execution should use `$ulw-execute reflectlog-correctness-performance-remediation --make-pr --worktree <absolute-path>` so the required platform workflow and review artifacts can run without touching the user's dirty main worktree.

## Success criteria
1. All 35 implementation todos pass their happy and failure QA, have evidence, and are committed atomically.
2. Manifest and lock reproducibly provide FastMCP 4.0.0b5, Pydantic 2.14.0b1, Pydantic Core 2.48.0, ranx 0.3.21, and Portalocker 4.3.0.
3. Independent processes cannot lose committed SQLite rows, USearch vectors, Tantivy documents/tombstones, archive records, journal ordering, or generation state.
4. A killed writer leaves an old valid generation or a recoverable pending intent, never a partial/corrupt/silent-empty index; lock ownership recovers without deleting the sidecar.
5. Search recovery does not block the event loop, and backend outage, timestamp/recency, retry, cache, tombstone, HTTP, numeric-config, and CLI contracts match this plan.
6. Default RRF and metadata CLI paths do not import ranx or the heavy server/search stack; every public ranx method remains behaviorally compatible.
7. Frozen Ruff/format, ty, Pyright, warnings-as-errors, full pytest, and coverage >=90% pass without mutating files or installing/upgrading tools.
8. The installed five-tool FastMCP surface passes deterministic real-client QA with real local stores and no paid/external provider call.
9. The focused macOS, Linux, and Windows jobs pass; unsupported network-filesystem semantics are documented without runtime guessing.
10. Legacy persisted data and public MCP/config/fusion identities remain compatible, no schema migration occurs, and no cross-backend ACID claim is introduced.
11. The final worktree contains only intended commits and preserves untouched `proposal-local-flagembedding-engine.md` and `repository_context.md` as pre-existing untracked files.
12. F1-F4 all return unconditional APPROVE for the same final tree; then execution surfaces results and waits for the user's explicit completion approval.
