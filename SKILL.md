---
name: curry-project
description: Maintain and extend the Curry functional database project in this repository. Use when modifying Curry's SQLite-backed runtime, versioned constants/functions/models, inference provenance, adapters, tests, examples, or documentation, especially when preserving deterministic behavior and exact-version semantics matters.
---

# Curry Project

Keep Curry deterministic. Treat versioned constants, functions, models, and inference records as immutable historical facts once written.

Validate changes with runtime checks, not only by inspection. Run:

```bash
python curry_tests.py
python curry_example.py
```

Preserve these invariants:

- Enforce constant type identity across the full lifetime of a constant ID, including retired versions.
- Deserialize constants by declared `type_signature`, not by best-effort guessing.
- Record explicit inference parameters faithfully. Do not use truthiness fallbacks for numeric fields where `0` or `0.0` is valid.
- Cache only pure function calls and key cache entries on the full effective evaluation context.
- Reject references to retired dependencies when declaring new functions or models.

When editing `call_function()`:

- Keep evaluation deterministic for a given function version, dependency versions, and arguments.
- Restrict the execution environment. Do not expose unrestricted builtins.
- Treat cache failures as secondary. Prefer returning a correct result over failing due to memoization storage.
- Preserve function-binding version locks when exposing nested function calls.

When editing constants or inference storage:

- Add or update tests for round-trip behavior of affected types.
- Prefer explicit serialization formats over implicit `str(...)` conversion.
- Keep SQLite schema fields and Python method signatures aligned.

When editing adapters:

- Verify SDK usage against official provider docs before changing call shapes.
- Keep adapter failures descriptive and surface provider errors unchanged where practical.
- Avoid introducing provider-specific defaults that override Curry's stored model parameters.
- **Claude API**: `temperature` and `top_p` are mutually exclusive. Never pass both. Prefer `temperature` when the model record has it set; fall back to `top_p` only when `temperature` is `None`.

When editing examples or tests:

- Keep console output portable across UTF-8 and non-UTF-8 terminals.
- Prefer ASCII-safe fallbacks for status markers and banners.
- Add regression tests for every bug fix that changes runtime semantics.

When editing documentation:

- Ensure README examples match actual method signatures and supported behavior.
- Do not claim functionality that is stubbed, partial, or aspirational.

## Tokenized Data Translation Best Practices

When working with tokenized data and cross-model translation:

- **Raw Text as Source of Truth**: Token IDs are ephemeral, model-specific projections. Always store and use the raw text as the immutable source of truth when translating between models.
- **Explicit Vocabulary Referencing**: If token IDs must be stored, structure them as a dictionary `{"model": "gpt-4", "token_ids": [...]}` rather than a naked array.
- **Special Token Mapping**: Control tokens (`<|endoftext|>`, `[INST]`, etc.) do not have 1:1 mappings across LLM vocabularies. Translate the intent, not the raw ID.
- **Decouple Storage from Counting**: Store `input_tokens` as raw text or structured dicts, but always rely on adapter-specific tokenizers (e.g., `tiktoken`) when precise counting is needed.

## Core Best Practices

- Favor explicit schema evolution: add migration notes whenever table columns change; keep table definitions and Python field handling in lockstep.
- Preserve deterministic semantics: no wall-clock randomness, global mutable state, or environment-dependent behavior in pure function execution.
- Improve error ergonomics: include `name@version` and dependency references in all raised runtime errors; prefer typed exceptions with stable, testable messages.
- Test strategy: one positive and one negative regression test per fix; at least one retirement-related test whenever dependency or retrieval rules change; mock provider responses for deterministic CI.
- Tokenized data: keep raw prompt/response text immutable and separate from provider tokenization artifacts; never assume token IDs are portable across model families.

---

## Two-Tier Architecture: CurrySession (Current Design)

Curry uses a two-tier database model. Understanding this split is required before editing `curry_core.py`, writing MCP tools, or changing how sessions are opened.

### Directory Layout

```
C:\AI-Local\Curry\
    curry_core.db          <- global: model registrations, shared constants, system prompts
    curry_backup.py
    backups\
        migrations\        <- pre-migration snapshots, kept indefinitely

<project-root>\
    .curry\
        curry.db           <- local: project inferences, functions, project constants
        config.json
```

### config.json Schema

```json
{
    "project": "my-project",
    "version": 1,
    "core_db": "C:/AI-Local/Curry/curry_core.db",
    "local_db": ".curry/curry.db",
    "default_model": "claude-sonnet-4-6",
    "default_model_version": 1,
    "mcp_tool_prefix": "curry_my_project"
}
```

`mcp_tool_prefix` namespaces MCP tool names so multiple project sessions can coexist in the same Claude context without tool name collision.

### What Lives Where

| Entity | Database | Rationale |
|---|---|---|
| Model registrations | core_db | Shared across all projects; locked params must be canonical |
| Shared system prompts | core_db | Global constants referenced by model versions |
| Project constants | local_db | Project-specific; may shadow core constants by name |
| Function declarations | local_db | Pipeline-specific logic |
| Inference records | local_db | Project provenance; never mixed across projects |
| Execution cache | local_db | Scoped to local function versions |

### Fallback Resolution

`Curry` accepts an optional `fallback_db` parameter. When `get_constant()`, `get_function()`, or `get_model()` raises `KeyError` locally, it retries against `fallback_db` before propagating the error. This lets `call_function()` and `get_function_lineage()` traverse dependencies that live in either tier without caller changes.

`list_constants()`, `list_functions()`, and `list_models()` merge both tiers. Local entities shadow core entities with the same name.

### CurrySession Read-Only Core Rule

**Core DB is always opened read-only within a `CurrySession`.** Sessions use SQLite URI mode:

```python
core_db = Curry(f"file:{core_db_path}?mode=ro", uri=True)
```

This is enforced at the driver level, not by convention. Any write attempt through a session raises `sqlite3.OperationalError` immediately.

`CurrySession.register_model()` raises `PermissionError` with a clear message. Admin writes always go through a direct connection:

```python
# Admin path -- always done outside a session
with Curry(r"C:\AI-Local\Curry\curry_core.db") as admin:
    admin.register_model("claude-sonnet-4-6", 1, checkpoint_hash="...", temperature=1.0, top_p=0.9, max_tokens=8192)
```

This boundary prevents MCP tool calls from modifying the global model registry autonomously. Never remove it.

### CurrySession Routing Table

| Method | Routes to | Notes |
|---|---|---|
| `get_model()`, `get_model_latest()`, `list_models()` | core_db (read) | Model params are global facts |
| `register_model()`, `retire_model()` | raises `PermissionError` | Admin-only; use direct Curry connection |
| `declare_constant()`, `get_constant()`, `list_constants()` | local_db | Falls back to core_db on KeyError |
| `declare_function()`, `call_function()`, `get_function_lineage()` | local_db | Falls back to core_db on KeyError |
| `record_inference()`, `search_inferences()`, `compare_inferences()` | local_db | Inference is always project-local |
| `backup()` | call on each db separately | `session.local_db.backup(...)` and core via admin |

---

## Backup Strategy

Backup is **not optional** for a deterministic provenance database. Every inference record, constant version, and function binding is irreplaceable historical fact. Treat each `.db` file as an append-only ledger.

### WAL Mode

Core and local DBs both open with `PRAGMA journal_mode=WAL`. WAL allows safe online backups without blocking writers and reduces corruption risk on crash. Read-only connections (core within a session) inherit WAL automatically.

### Rotation Script

`curry_backup.py` in `C:\AI-Local\Curry\` handles timestamped rotation with post-backup `PRAGMA integrity_check`. Run via Windows Task Scheduler daily at 03:00:

```
schtasks /create /tn "CurryDailyBackup" /tr "python C:\AI-Local\Curry\curry_backup.py" /sc daily /st 03:00 /f
```

Retention: 14 daily rotations for `curry_core.db`. Project local DBs should run the same script pointed at `.curry/curry.db` with their own scheduler task or pre-run hook.

### Pre-Migration Snapshot Rule

Before any schema change, column addition, or bulk `UPDATE`/`DELETE`, call `db.backup(snapshot_path)` with a migration-specific name (e.g., `curry_premigration_20260501.db`). Store in `backups\migrations\` separately from rotation backups. Never delete migration snapshots.

### Integrity Check

`db.integrity_check()` returns `"ok"` on success. Call it after every backup and as a startup guard for persistent databases. The rotation script calls it automatically and raises `RuntimeError` on failure.

---

## Implemented -- Full Audit Closure (May 2026)

All items from prior Code Review Notes and the May 2026 audit are resolved. This is the authoritative current-state record.

### Resolved: Bugs / Silent Failures
- `retire_constant()` and `retire_function()` raise `KeyError` on missing entities.
- `declare_function()` and `register_model()` enforce monotonic version ordering.
- `ClaudeAdapter.infer()` passes only one of `temperature` or `top_p` to `client.messages.create()`. Claude models (claude-sonnet-4-6 and later) reject requests that supply both; the adapter prefers `temperature` when set, falls back to `top_p` otherwise.
- `_deserialize_constant_value()` re-validates type after deserialization.

### Resolved: Design Gaps
- `evict_execution_cache()` uses `rowid NOT IN` subquery and returns evicted row count.
- `declare_function()` runs AST-based `validate_function_body()` at declaration time.
- `expected_args` stored as JSON in `functions` table; used for strict symbol validation; returned by `get_function()` and `list_functions()`.
- Adapters pass resolved `temperature_used`/`top_p_used` directly; double `get_model()` call eliminated.

### Resolved: Missing Methods
- `get_model_latest()`, `list_constants()`, `list_functions()`, `list_models()` added.
- `backup()` and `integrity_check()` added.
- `__enter__` / `__exit__` context manager added to `Curry`.

### Resolved: Execution Environment
- `_SAFE_BUILTINS` module-level dict wired into `call_function()` eval.
- AST dunder guard checks both `node.attr` and `node.value.id`.

### Resolved: Architecture
- Two-tier `CurrySession` implemented with `from_project(project_dir)` classmethod.
- `fallback_db` parameter added to `Curry.__init__()`.
- Core DB opened read-only via SQLite URI in `CurrySession.from_project()`.
- `register_model()` on `CurrySession` raises `PermissionError`.
- `test_two_tier_session()` added; 39 tests pass.

### Still Outstanding (Next Sprint)
- `get_inference_lineage(inference_id)` -- provenance graph from inference to model/prompt/data.
- `get_stale_functions()` -- detect functions bound to retired dependencies.
- `upsert_prompt()` / `render_prompt()` -- first-class prompt registry.
- `archive_inference_output(inference_id, codec)` -- compressed blob storage.
- `search_inferences()` O(n) memory -- add denormalized token-count columns for SQL-level filtering.
- `export_schema()` -- include indices, views, triggers in output.
- `TOKENS` list element validation -- `all(isinstance(t, int) for t in value)`.
- `CURRENCY` string format validation -- document and enforce expected shape.
- `declare_function()` `description` parameter -- store in `functions` table, surface in `get_function()` / `list_functions()`, and use in dynamic MCP schema generation for per-function tool descriptions and argument unit hints (see Lag Pattern 2).
- `build_dynamic_tools()` argument hint generation -- derive unit/range hints from stored `description` field; fall back to `"Value for argument '{arg}'"` only when no description is present; flag missing descriptions on non-obvious args as a schema warning at generation time.
- Benchmark correctness scoring -- add `expected_answer` field to `TASKS` and a `correct` bool to `RunResult`; update `print_summary()` to report correctness rate alongside completion rate.

---

## REU Call Success Contract

Treat REU as **Request -> Execute -> Update**. Every call path satisfies all three phases or fails loudly with typed errors.

- **Request**: Validate model/function version exists and is active. Validate payload shape. Attach deterministic call metadata (`request_id`, `seed`, effective params).
- **Execute**: Use locked parameters only. Preserve provider errors without swallowing root cause. Enforce timeout/retry policy at adapter boundary with bounded retries.
- **Update**: Persist inference record atomically after successful execution. Ensure metadata is JSON-serializable. Return durable identifiers only (`inference_id`) once commit succeeds.

Minimum REU reliability tests: success path writes exactly one inference row with complete metadata; failure path raises `RuntimeError` and writes no partial row; retryable failures obey max retry count; non-serializable metadata raises `TypeError` before write.

---

## Agent Benchmark: Dynamic vs Generic Tool Surface (May 2026)

Validated with `curry_agent_bench.py` on `claude-haiku-4-5-20251001`, 6 tasks, 1 run each.

| Metric | Generic | Dynamic | Delta |
|---|---|---|---|
| Turns per task | 3.17 | 2.17 | **-1.0 (-32%)** |
| Tokens per task | 4,289 | 3,119 | **-1,169 (-27%)** |
| Discovery calls | 1.17 | 0.00 | **eliminated** |
| First call to function | 0% | 100% | agent goes direct every time |
| Completion rate | 100% | 100% | both modes reliable |

**Why it matters**: The generic surface requires the agent to call `list_functions` first to discover argument names before it can call `call_function`. Dynamic per-function tools (named `{fn}_v{version}` with typed schemas derived from `expected_args`) eliminate that discovery turn entirely. The agent knows what to call and how from the tool schema alone.

**Design rule**: When implementing MCP tools for user-facing agents, always generate per-function dynamic tools from `expected_args`. The generic `call_function` dispatcher is appropriate for programmatic/scripted callers, not for agent tool surfaces.

---

## Agent Benchmark: Lag Pattern Analysis and Rule Refinements (3-Run Extension, 2026)

Follow-up to the May 2026 single-run benchmark. Same model (`claude-haiku-4-5-20251001`), same 6 tasks, 3 runs per task/mode pair (36 total). Raw data in `curry_bench_results.json`.

### Updated Aggregate Results

| Metric | Generic | Dynamic | Delta |
|---|---|---|---|
| Turns per task | 2.94 ±0.54 | 2.22 ±0.43 | **-0.72 (-24.5%)** |
| Input tokens/task | 3,722 | 3,069 | **-653 (-17.5%)** |
| Output tokens/task | 294 | 155 | **-139 (-47.3%)** |
| Total tokens/task | 4,015 | 3,224 | **-792 (-19.7%)** |
| Cost/task | $0.004152 | $0.003074 | **-$0.001078 (-26.0%)** |
| Discovery calls | 1.22 | 0.06 | **near-eliminated** |
| First call to function | 0% | 100% | unchanged |
| Completion rate | 100% | 100% | unchanged |

Schema overhead: generic=906 tokens, dynamic=1,255 tokens (+349, +38.5%). Break-even: 0.4 tasks — overhead recovers almost immediately.

Note: deltas are lower than the 1-run figures above because the 3-run set captured variance the single run missed, particularly the `sequential_two` failure mode in generic and the `single_multiarg` regression in dynamic. Single-run figures should be treated as optimistic; 3-run figures are the working baseline.

---

### Lag Pattern 1 — Sequential Multi-Call Token Inversion

**Observation:** For tasks requiring N≥2 *sequential* function calls (`sequential_two`), dynamic mode's per-turn input cost can exceed generic mode's. In run 1, dynamic used 4,195 input tokens vs. generic's 4,123 — the only run in 36 where generic was cheaper. Dynamic carries its full tool schema (+349 tokens) into every turn's context window. Generic pays a fixed discovery cost once, then accumulates only function response payloads.

**Rule:** Declare composed Curry functions for every recurring sequential pipeline. `final_price` (discount + tax in one call) costs half the turns of the two-step `apply_discount` → `compute_tax` approach and eliminates the per-turn schema carry cost entirely. Sequential function chains that appear more than once in agent workflows are a signal to compose them as a named function version.

---

### Lag Pattern 2 — Dynamic Argument Unit Ambiguity

**Observation:** `single_multiarg` dynamic run 3 called `compound_interest_v1(rate=6)` instead of `rate=0.06`, receiving `16,807,000` (wrong). The agent then called `session_info` (irrelevant) and produced a wrong final answer — unrecovered. The generic agent received the same wrong result but self-corrected by reasoning from the function body it had seen via `list_functions`. The dynamic agent never sees the function body; its tool description was `"Value for argument 'rate'"` with no unit hint.

**Rules:**
- When generating dynamic tool schemas from `expected_args`, the property `description` field for any argument whose unit is non-obvious from the name alone (rates, proportions, fractions, scaled values, enumerated flags) **must** include a unit hint and an example value. Use the form: `"Annual rate as a decimal fraction (e.g. 0.06 for 6%)"`, not `"Value for argument 'rate'"`.
- Add a `description` field to `declare_function()` and store it in the `functions` table. Surface it in `get_function()`, `list_functions()`, and dynamic schema generation. Do not rely on function name and body alone to communicate expected argument conventions to an agent.
- This is a schema defect, not a model failure. A missing unit hint on a non-obvious argument is treated the same as a missing required field: the declaration is incomplete.

---

### Lag Pattern 3 — Generic Tool Bypass (Hallucination Risk)

**Observation:** `sequential_two` generic run 2 made zero tool calls and returned a non-answer asking the user for discount and tax rates — values discoverable via `list_constants`. The model concluded the task was under-specified without checking its tools. This spent 935 tokens doing nothing. This failure mode does not exist in dynamic mode: the function names alone make the tool surface self-evidently applicable.

**Rule:** The system prompt for any agent operating on the generic tool surface must include an explicit tool-use directive: *"You MUST call available tools to answer. Do not attempt to calculate from memory or ask for information that may be discoverable via `list_functions` or `list_constants`."* Do not rely on the model inferring that relevant information is retrievable — state it.

---

### Lag Pattern 4 — Double-Discovery Compounding

**Observation:** Tasks where the prompt references implicit named rates or thresholds (`sequential_two`, `discovery_ambiguous`) triggered two discovery calls — both `list_constants` and `list_functions` — in every valid generic run. The marginal cost of the second discovery call is ~+364 tokens. This is 5× the cost of two extra function calls (+70 tokens). Discovery call count scales with semantic ambiguity, not function call count. Dynamic mode is immune: both schemas are pre-loaded.

**Rules:**
- Function `description` fields (Lag Pattern 2 rule) should embed the names of the constants they bind to. Example: `"Apply the standard markup (markup_rate constant) to cost"` vs. `"Apply markup to cost"`. This reduces the double-discovery rate in generic mode by making the constant relationship explicit in the function listing.
- Treat any benchmark task that consistently requires ≥2 discovery calls as a signal that function names and descriptions are insufficiently self-documenting. Refine before expanding the function set.
- When a task mentions a named threshold or rate by human label ("standard markup", "minimum order"), treat it as a constant-binding hint in documentation, not an opaque implementation detail.

---

### Lag Pattern 5 — Non-Deterministic Discovery Call Ordering (Latent)

**Observation:** For `discovery_ambiguous`, the agent called `list_constants` before `list_functions` in 2/3 generic runs and reversed the order in 1/3. Currently benign — both orderings succeed at equal token cost. But in tasks where the second discovery call's target cannot be determined without the first response, this 2:1 ordering split could produce different correctness outcomes across runs.

**Rule:** When adding tasks to the benchmark or designing agent workflows over the generic surface, flag any task where discovery call ordering is semantically constrained (i.e., the correct second discovery call cannot be determined without the first response). These require either a composed dynamic schema (eliminating the ordering dependency) or a scaffolded prompt that specifies discovery order. Non-deterministic ordering on critical tasks is a latency and correctness risk, even when current task set results are stable.

---

### Benchmark Design Rules (derived from 3-run data)

- Use `--runs 3` minimum for any benchmark run intended to inform design decisions. Single-run results suppress failure modes that appear at ~33% frequency (bypass, argument error).
- The `first_call_to_function` metric is a perfect binary discriminator of mode across all 36 runs; retain it in all future benchmark runs.
- Add a `correct_answer` field to task definitions and a correctness check to `RunResult`. Token efficiency without answer correctness is not a valid efficiency metric — `sequential_two` generic run 2 was the "cheapest" run in the benchmark and produced zero useful output.
- When a dynamic run's total tokens exceed the generic average for the same task, file it as a regression case and inspect the tool call trace for recovery loops (`session_info` calls, repeated function calls with same args).

---

## MCP Server Rules (Invariants)

Enforce these before adding any new MCP tool:

- All tools route through `CurrySession`, never directly against `core_db` for writes.
- Tool names use `config["mcp_tool_prefix"]` to avoid cross-project collision.
- Input schema for `call_function` tools derives from `get_function()["expected_args"]` -- do not hardcode schemas.
- Dynamic tool property descriptions must include unit and range hints for any argument whose expected domain is non-obvious from the name alone (rates, proportions, fractions, scaled values). A description of `"Value for argument 'rate'"` without a unit hint is a schema defect. Derive hints from the stored function `description` field once implemented; until then, annotate manually in `build_dynamic_tools()`.
- Tool errors surface Curry typed exceptions (`KeyError`, `ValueError`, `PermissionError`, `RuntimeError`) as structured MCP error responses, not raw tracebacks.
- Inference-writing tools always return `inference_id` as the durable result, never raw output text.
- No MCP tool exposes `register_model`, `retire_model`, or any direct core write.
- The MCP server opens one `CurrySession` at startup; it does not open new sessions per tool call.
- Backup runs via OS scheduler (`curry_backup.py`), not as an MCP tool.
