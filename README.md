# Curry

**A functional, versioned database for LLM operations.**

Curry stores constants, functions, models, and inference records as immutable, version-locked facts backed by SQLite. Every computation is reproducible, every inference is traceable, and every dependency is pinned to an exact version at declaration time.

```python
from curry_core import Curry, TypeSignature

with Curry("project.db") as db:
    db.declare_constant("discount_rate", 1, 0.10, TypeSignature.FLOAT64.value)
    db.declare_function(
        name="apply_discount", version=1,
        body="amount * (1 - discount_rate)",
        constant_bindings={"discount_rate": 1},
        expected_args=["amount"],
        is_pure=True,
    )
    print(db.call_function("apply_discount", 1, {"amount": 100}))  # 90.0
```

---

## Why Curry

Standard databases store current state. Curry stores **history as fact**.

- A constant declared at version 1 is retrievable forever, even after version 2 is declared.
- A function locked to `discount_rate@v1` continues to produce identical output regardless of what later versions of that constant contain.
- An inference record carries the exact model version, seed, temperature, and top_p used — not defaults, the actual values.
- The execution cache is keyed on the full evaluation context (args + all bound dependency versions), so cache hits are guaranteed to be semantically identical to a fresh run.

This makes Curry useful for reproducible ML pipelines, prompt A/B testing, fine-tuning lineage tracking, and any workflow where "what did we actually run, with what parameters, and what did it produce?" needs a definitive answer.

---

## Architecture

### Two-Tier Design

Curry separates global schema from local project provenance:

```
C:\AI-Local\Curry\
    curry_core.db       ← global: model registrations, shared constants
    curry_backup.py

<your-project>\
    .curry\
        curry.db        ← local: inferences, project functions, project constants
        config.json
```

**`curry_core.db`** holds model registrations and shared system-prompt constants. It is opened **read-only** inside any project session — model parameters are global facts, not per-project settings.

**`.curry/curry.db`** holds everything specific to a project: inference records, local constants, function declarations.

### CurrySession

`CurrySession.from_project(project_dir)` opens both databases and wires `curry.db` to fall back to `curry_core.db` for any constant, function, or model lookup that fails locally. `call_function` and `get_function_lineage` traverse dependencies across both tiers transparently.

```python
from curry_core import CurrySession

with CurrySession.from_project(r"C:\my-project") as session:
    # model lookup goes to core_db (read-only)
    model = session.get_model_latest("claude-sonnet-4-6")
    # inference write goes to local_db
    iid = session.record_inference(
        model_name="claude-sonnet-4-6",
        model_version=1,
        input_tokens="Explain backpropagation.",
        output_tokens=b"Backpropagation is...",
        seed=42,
    )
```

### `.curry/config.json`

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

---

## Core Concepts

### Constants

Typed, versioned, immutable values. Version numbers must increase monotonically. Type identity is enforced across the full lifetime of a constant ID.

```python
db.declare_constant("rate", 1, 0.10, "Float64")
db.declare_constant("rate", 2, 0.12, "Float64")   # OK — same type, higher version
db.declare_constant("rate", 3, "high", "String")  # TypeError — type changed
```

Supported types: `Float64`, `Int32`, `String`, `Blob`, `Json`, `Tokens`, `Currency`, `Bool`.

### Functions

Expressions (not statements) evaluated against locked constant and function dependencies. The body is AST-validated at declaration time.

```python
db.declare_function(
    name="final_price",
    version=1,
    body="amount * (1 - rate) * (1 + tax)",
    constant_bindings={"rate": 2, "tax": 1},  # pinned to exact versions
    expected_args=["amount"],                  # runtime args declared explicitly
    is_pure=True,                              # enables execution cache
)
```

`expected_args` is stored in the database and drives MCP tool schema generation — each function's tool input schema is derived from it at server startup.

### Models

Model versions lock all inference parameters at registration time.

```python
# Admin operation — done once, outside any project session
with Curry(r"C:\AI-Local\Curry\curry_core.db") as admin:
    admin.register_model(
        model_name="claude-sonnet-4-6",
        version=1,
        checkpoint_hash="release_20260101",
        temperature=1.0,
        top_p=0.9,
        max_tokens=8192,
    )
```

`register_model` is intentionally not exposed as an MCP tool. It is an admin operation that affects the global registry shared by all projects.

### Inferences

```python
iid = session.record_inference(
    model_name="claude-sonnet-4-6",
    model_version=1,
    input_tokens="What is entropy?",
    output_tokens=b"Entropy is a measure of...",
    seed=42,
    duration_ms=812,
    metadata={"input_tokens_count": 6, "output_tokens_count": 40},
)

# Compare two runs
diff = session.compare_inferences(iid_a, iid_b)
# same_model, same_seed, same_input_tokens, output SHA-256 hashes, param deltas

# Search with filters
results = session.search_inferences(
    model_name="claude-sonnet-4-6",
    start_timestamp="2026-05-01T00:00:00",
    limit=20,
)
```

---

## MCP Server

Curry ships a full Model Context Protocol server that exposes all project operations as typed tools. Claude can use it to declare constants and functions, call them, record inferences, search history, and inspect lineage — all scoped to a single project session.

### Start the server

```bash
pip install mcp
python curry_mcp_server.py --project C:\path\to\your\project
```

### Available tools (22)

| Group | Tools |
|---|---|
| Constants | `declare_constant`, `get_constant`, `get_constant_latest`, `list_constants`, `retire_constant` |
| Functions | `declare_function`, `get_function`, `list_functions`, `call_function`, `get_function_lineage`, `retire_function` |
| Models | `get_model`, `get_model_latest`, `list_models` |
| Inferences | `record_inference`, `get_inference`, `search_inferences`, `compare_inferences` |
| Maintenance | `create_retirement_tag`, `evict_execution_cache` |
| Introspection | `session_info`, `integrity_check` |

Tool names are prefixed with `config["mcp_tool_prefix"]` so multiple project servers can run in the same Claude context without collision. `register_model` and `retire_model` are not exposed — those are admin operations.

---

## LLM Adapters

Built-in adapters handle inference parameter locking, provider error surfacing, and automatic recording.

```python
from curry_llm_adapters import get_adapter
import os

adapter = get_adapter("claude", session, api_key=os.environ["ANTHROPIC_API_KEY"])
inference_id = adapter.infer_and_record(
    model_name="claude-sonnet-4-6",
    model_version=1,
    prompt="Summarize backpropagation in two sentences.",
    seed=42,
)
```

Supported: `"claude"` (Anthropic), `"openai"` (OpenAI), `"local"` (Ollama / llama.cpp).

---

## Backup

`Curry.backup(dest_path)` uses SQLite's online backup API — safe to call while the database is open and in use.

`curry_backup.py` is a standalone rotation script with post-backup `PRAGMA integrity_check`. Schedule it with Windows Task Scheduler:

```
schtasks /create /tn "CurryDailyBackup" /tr "python C:\AI-Local\Curry\curry_backup.py" /sc daily /st 03:00 /f
```

Pre-migration snapshots go in `backups\migrations\` and are never rotated out.

---

## Installation

**Requirements:** Python 3.10+, SQLite 3.35+ (bundled with Python 3.10).

```bash
git clone https://github.com/yourusername/curry.git
cd curry
python curry_tests.py   # 39 tests, all should pass
python curry_example.py
```

**Optional dependencies** (for LLM adapters):

```bash
pip install anthropic   # Claude
pip install openai      # OpenAI
pip install requests    # Local models via Ollama
pip install mcp         # MCP server
```

---

## File Reference

| File | Purpose |
|---|---|
| `curry_core.py` | `Curry` class, `CurrySession`, all DB operations |
| `curry_llm_adapters.py` | OpenAI, Claude, and local model adapters |
| `curry_mcp_server.py` | MCP server (22 tools, stdio transport) |
| `curry_backup.py` | Timestamped rotation backup script |
| `curry_tests.py` | Test suite (39 tests) |
| `curry_example.py` | Walkthrough of all major features |
| `test_token_translation.py` | Cross-model token translation tests |
| `SKILL.md` | Living development guide and architectural decisions |

---

## Roadmap

**Next sprint**
- `get_inference_lineage(inference_id)` — full provenance graph from inference to model, prompt, and training data
- `get_stale_functions()` — detect functions bound to retired dependencies
- `upsert_prompt()` / `render_prompt()` — first-class versioned prompt registry
- `search_inferences` SQL-level token-count filtering (currently applied in Python)
- Dynamic per-function MCP tools generated from `expected_args` at server startup

**Later**
- `archive_inference_output(inference_id, codec)` — compressed blob storage for large outputs
- Multi-project lineage queries across `core_db` and multiple local DBs
- Visualization: dependency graphs, version timelines

---

## License

MIT
