# Curry: A Functional Database for LLM Operations

## Project Vision

Curry is a deterministic, versioned substrate for machine learning workflows. It combines:
- **Immutable, versioned functions and constants** (Haskell-like semantics)
- **SQLite as the backbone** (embeddable, offline-first)
- **Direct LLM hooks** (reproducible inference, token tracking)
- **Complete provenance tracking** (audit trail from token → output)

Every computation is reproducible. Every inference is traceable. Every model state is locked in time.

---

## Core Properties

### 1. Strict Version Locking
- Functions reference exact versions of dependencies (functions and constants)
- No automatic upgrades to "latest"
- Version mismatch → function is invalid (not silently upgraded)
- Cascade retirements ensure consistency

### 2. Type Safety at Database Level
- Constants cannot change type across versions
- Function signatures are verified at declaration
- Type mismatches detected at insert time, not query time

### 3. Deterministic Execution
- Same input + same function version + same model version = identical output, always
- Perfect for caching, A/B testing, reproducible research
- Church-Rosser property: evaluation order doesn't matter

### 4. Immutable Constants with Retirement
- Constants are declared once, never mutated
- New versions are created, old versions are retired (with reason)
- Retirements can be grouped (atomically changed together)

### 5. ML-Native
- Model versions are locked at inference time
- Inference parameters (temperature, top_p, etc.) are versioned
- Training datasets can reference exact inference chains
- RAG context is versioned

---

## Architecture

### Layers

```
┌─────────────────────────────────────────┐
│     LLM Service Adapters                │
│  (OpenAI, Anthropic, Local models)      │
└────────────┬────────────────────────────┘
             │
┌────────────▼────────────────────────────┐
│  Curry Execution Engine                 │
│  (Version resolution, type checking,    │
│   deterministic execution)              │
└────────────┬────────────────────────────┘
             │
┌────────────▼────────────────────────────┐
│  SQLite Database                        │
│  (Schema: functions, constants,         │
│   inferences, model_versions, etc.)     │
└─────────────────────────────────────────┘
```

### MVP Feature Set

**Phase 1: Core Infrastructure**
- [ ] SQLite schema with all tables
- [ ] Type validation triggers
- [ ] Version locking enforcement
- [ ] Cascade retirement logic
- [ ] Retirement tag grouping

**Phase 2: Execution Engine**
- [ ] Function/constant declaration API
- [ ] Version resolution (exact match required)
- [ ] Function composition tracking
- [ ] Type checking at execution time
- [ ] Deterministic caching

**Phase 3: LLM Integration**
- [ ] Model version definitions
- [ ] Inference recording with full provenance
- [ ] Prompt versioning and composition
- [ ] Integration with one LLM API (OpenAI or Claude)
- [ ] Inference result tracking

**Phase 4: Testing & Validation**
- [ ] Reproducibility tests (run same inference twice, verify identical)
- [ ] A/B testing framework
- [ ] Cross-model comparison (same prompt on different models)
- [ ] Mobile SQLite sync simulation

---

## Schema Overview

### Core Tables

**retirement_tags**
- Group related constant/function retirements
- Provide reason and timestamp

**constants**
- `id TEXT, version INTEGER` (composite key)
- `value BLOB` (serialized)
- `type_signature TEXT` (enforced)
- `retired_at TIMESTAMP NULL` (immutable once set)
- `retirement_tag_id TEXT` (links to retirement group)

**functions**
- `name TEXT, version INTEGER` (composite key)
- `body TEXT` (function definition)
- `constant_bindings JSON` (exact versions: `{"base_rate": "v2"}`)
- `is_pure BOOLEAN` (purity marker)
- `retired_at TIMESTAMP NULL`

**function_dependencies**
- Maps exact versions of dependencies
- Validates on insert (dependency must exist and be active)

**model_versions**
- `model_name TEXT, version INTEGER` (composite key)
- `checkpoint_hash TEXT` (SHA256 of weights)
- `temperature REAL, top_p REAL, max_tokens INTEGER` (LOCKED at version time)
- `system_prompt_id TEXT, system_prompt_version INTEGER` (exact ref)
- `trained_on_data_id, trained_on_data_version` (lineage)

**inferences**
- `inference_id TEXT PRIMARY KEY` (UUID)
- `model_name TEXT, model_version INTEGER` (locked)
- `input_data_id TEXT, input_data_version INTEGER` (locked)
- `output_data_id TEXT, output_data_version INTEGER` (locked)
- `output_tokens BLOB` (result)
- `seed INTEGER` (reproducibility)
- `metadata JSON` (cost, latency, etc.)

**prompts**
- `prompt_id TEXT, version INTEGER` (composite key)
- `instruction_template TEXT` (with variable slots)
- `input_schema JSON, output_schema JSON` (shape validation)
- `system_prompt_id, system_prompt_version` (exact ref)

---

## API (Python SDK)

### Basic Operations

```python
from curry import Curry

# Initialize
db = Curry("curry.db")

# Declare a constant
db.declare_constant(
    id="base_rate",
    version=1,
    value=0.1,
    type_signature="Float64"
)

# Declare another version (old is retired atomically)
tag = db.create_retirement_tag("Q2-2025-pricing", "New pricing model")
db.retire_constant("base_rate", 1, retirement_tag=tag)
db.declare_constant(
    id="base_rate",
    version=2,
    value=0.12,
    type_signature="Float64"
)

# Declare a function
db.declare_function(
    name="apply_discount",
    version=1,
    body="lambda amount, rate: amount * (1 - rate)",
    constant_bindings={"rate": "base_rate@v1"},
    is_pure=True
)

# Call a function (exact version required)
result = db.call_function(
    name="apply_discount",
    version=1,
    args={"amount": 100},
    # Automatically uses base_rate@v1 (locked at function declaration)
)
# result: 90.0

# Register a model
db.register_model(
    model_name="gpt-4",
    version=1,
    checkpoint_hash="abc123...",
    temperature=0.7,
    top_p=0.9,
    max_tokens=2048,
    system_prompt_id="helpful-assistant",
    system_prompt_version=2
)

# Run an inference (all versions locked)
inference_id = db.infer(
    model_name="gpt-4",
    model_version=1,
    input_prompt="What is the capital of France?",
    seed=42  # deterministic
)
# Returns inference_id, stores result with full provenance

# Retrieve inference
inference = db.get_inference(inference_id)
# Returns: {
#   "model_name": "gpt-4",
#   "model_version": 1,
#   "output_tokens": "...",
#   "temperature_used": 0.7,
#   "seed": 42
# }

# Reproduce inference (same output, guaranteed)
inference_2 = db.infer(
    model_name="gpt-4",
    model_version=1,
    input_prompt="What is the capital of France?",
    seed=42
)
assert inference["output_tokens"] == inference_2["output_tokens"]
```

### Advanced: RAG + Prompts

```python
# Declare a system prompt
db.declare_constant(
    id="system-helpful-assistant",
    version=1,
    value="You are a helpful assistant. Answer concisely.",
    type_signature="String"
)

# Declare a prompt template
db.declare_prompt(
    prompt_id="customer-support",
    version=1,
    system_prompt_id="system-helpful-assistant",
    system_prompt_version=1,
    instruction_template="Customer query: {query}\nContext: {context}\nRespond:",
    input_schema={"query": "string", "context": "string"},
    output_schema={"response": "string"}
)

# Retrieve context (versioned)
context = "The product comes in three colors..."
context_tokens = db.tokenize(context, tokenizer_version=1)

# Run RAG generation
generation = db.rag_generate(
    prompt_id="customer-support",
    prompt_version=1,
    model_name="gpt-4",
    model_version=1,
    context_tokens=context_tokens,
    input_query="What colors are available?",
    seed=42
)

# Query full lineage
lineage = db.get_generation_lineage(generation["generation_id"])
# Returns: prompt@v1 → model@v1 → context@v1 → output tokens
```

---

## Testing Strategy

### 1. Reproducibility Tests
```
Run inference A → get output X
Run inference A again → assert output == X
(Same model, same inputs, same seed)
```

### 2. Determinism Tests
```
Compose f(g(x)) → get Y
Compose g(x), then pass to f → assert result == Y
(Evaluation order doesn't matter)
```

### 3. Version Locking Tests
```
Declare function@v1 using constant@v1
Retire constant@v1
Assert function@v1 is now invalid/marked corrupted
Attempt to call function@v1 → raises VersionError
```

### 4. Type Safety Tests
```
Declare constant "rate" with type Float64
Attempt to declare "rate" v2 with type String
→ raises TypeError at insert time (not query time)
```

### 5. A/B Testing
```
Run 100 inferences with model@v1, prompt@v1
Run 100 inferences with model@v1, prompt@v2
Compare output distributions (now deterministic, reproducible)
```

---

## Implementation Stack (MVP)

- **Language**: Python 3.10+ (SQLAlchemy for DB)
- **Database**: SQLite3
- **LLM Integration**: OpenAI API (or Claude if preferred)
- **Serialization**: JSON for most, BLOB for tokens
- **Tokenization**: tiktoken or transformers library
- **Testing**: pytest

---

## Success Metrics (MVP)

1. ✅ Can declare versioned constants and functions
2. ✅ Can call functions with exact version locking
3. ✅ Type mismatches detected at insert time
4. ✅ Can run LLM inferences with full provenance
5. ✅ Inference is perfectly reproducible (same seed = same output)
6. ✅ Can query complete lineage of any inference
7. ✅ Can A/B test prompts deterministically
8. ✅ Handles cascade retirement correctly

---

## Future Enhancements

- **Mobile sync**: Compress versioned data, sync only diffs
- **Distributed execution**: Federated model evaluation
- **Fine-tuning pipeline**: Auto-generate training data from inferences
- **Optimization**: Learn which prompt versions work best
- **Visualization**: Dependency graphs, version timelines
- **Multi-LLM composition**: Chain models across providers
- **Knowledge distillation**: Smaller models trained from larger ones (with provenance)

---

## Project Phases

### Phase 1 (Week 1): Schema + Core API
- SQLite schema with triggers
- Python wrapper for basic operations
- Type validation

### Phase 2 (Week 2): Execution + Caching
- Function composition engine
- Deterministic caching
- Version resolution logic

### Phase 3 (Week 3): LLM Integration
- Model registration
- Inference recording
- OpenAI/Claude API hooks

### Phase 4 (Week 4): Testing + Validation
- Reproducibility tests
- A/B testing framework
- Cross-model comparisons

---

## File Structure

```
curry/
├── curry/
│   ├── __init__.py
│   ├── core.py              # Main Curry class
│   ├── schema.py            # SQLite schema + triggers
│   ├── types.py             # Type system + validation
│   ├── execution.py         # Function execution engine
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── openai_adapter.py
│   │   └── claude_adapter.py
│   ├── cache.py             # Deterministic caching
│   └── utils.py
├── tests/
│   ├── test_versioning.py
│   ├── test_types.py
│   ├── test_execution.py
│   ├── test_llm_integration.py
│   └── test_reproducibility.py
├── examples/
│   ├── basic_usage.py
│   ├── rag_pipeline.py
│   └── model_comparison.py
├── curry.db                 # Example database
└── README.md
```

