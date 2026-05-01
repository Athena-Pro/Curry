"""
Curry Test Suite
Validates: versioning, type safety, reproducibility, and LLM integration
"""

import json
import sys
from curry_core import Curry, TypeSignature
from curry_llm_adapters import LocalModelAdapter


class TestRunner:
    """Simple test runner."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
        # Check if console supports Unicode
        try:
            "✓".encode(sys.stdout.encoding or 'utf-8')
            self.unicode_ok = True
        except (UnicodeEncodeError, AttributeError):
            self.unicode_ok = False

    def _check(self, passed: bool) -> str:
        """Return checkmark or X, handling encoding issues."""
        if self.unicode_ok:
            return "✓" if passed else "✗"
        else:
            return "[OK]" if passed else "[FAIL]"

    def test(self, name: str, fn):
        """Run a test."""
        try:
            fn()
            self.passed += 1
            self.tests.append((name, "PASS", None))
            print(f"{self._check(True)} {name}")
        except AssertionError as e:
            self.failed += 1
            self.tests.append((name, "FAIL", str(e)))
            print(f"{self._check(False)} {name}: {e}")
        except Exception as e:
            self.failed += 1
            self.tests.append((name, "ERROR", str(e)))
            print(f"{self._check(False)} {name}: ERROR - {e}")

    def summary(self):
        """Print test summary."""
        print(f"\n{'='*70}")
        print(f"Tests: {self.passed} passed, {self.failed} failed")
        print(f"Total: {self.passed + self.failed}")
        print(f"{'='*70}\n")

        if self.failed > 0:
            print("FAILURES:")
            for name, status, error in self.tests:
                if "FAIL" in status or "ERROR" in status:
                    print(f"  {name}: {error}")


def test_constant_declaration():
    """Test declaring and retrieving constants."""
    db = Curry()

    # Declare constant
    db.declare_constant("rate", 1, 0.1, TypeSignature.FLOAT64.value)

    # Retrieve
    const = db.get_constant("rate", 1)
    assert const["id"] == "rate"
    assert const["version"] == 1
    assert const["value"] == 0.1
    assert const["type_signature"] == TypeSignature.FLOAT64.value

    db.close()


def test_constant_retirement():
    """Test retiring constants."""
    db = Curry()

    # Declare v1
    db.declare_constant("param", 1, 10, TypeSignature.INT32.value)

    # Retire v1
    db.retire_constant("param", 1)

    # Should not be retrievable
    try:
        db.get_constant("param", 1)
        raise AssertionError("Should have raised ValueError for retired constant")
    except ValueError:
        pass  # Expected

    db.close()


def test_retire_invalid_id():
    """Test retiring non-existent IDs throws KeyError."""
    db = Curry()
    try:
        db.retire_constant("nonexistent", 1)
        raise AssertionError("Should have raised KeyError for non-existent constant")
    except KeyError:
        pass

    db.declare_function("f1", 1, "1")
    try:
        db.retire_function("nonexistent", 1)
        raise AssertionError("Should have raised KeyError for non-existent function")
    except KeyError:
        pass

    db.close()


def test_type_safety():
    """Test type checking at database level."""
    db = Curry()

    # Declare string constant
    db.declare_constant("msg", 1, "hello", TypeSignature.STRING.value)

    # Try to declare v2 with different type
    try:
        db.declare_constant("msg", 2, 3.14, TypeSignature.FLOAT64.value)
        raise AssertionError("Should have rejected type mismatch")
    except TypeError:
        pass  # Expected

    # Can declare v2 with same type
    db.declare_constant("msg", 2, "goodbye", TypeSignature.STRING.value)

    db.close()


def test_type_safety_after_retirement():
    """Test that retired constants still preserve family type identity."""
    db = Curry()

    db.declare_constant("x", 1, 1, TypeSignature.INT32.value)
    db.retire_constant("x", 1)

    try:
        db.declare_constant("x", 2, "oops", TypeSignature.STRING.value)
        raise AssertionError("Should have rejected type mismatch after retirement")
    except TypeError:
        pass

    db.close()


def test_constant_versions_must_increase_monotonically():
    """Test that constant versions cannot be redeclared with non-increasing versions."""
    db = Curry()
    db.declare_constant("threshold", 1, 10, TypeSignature.INT32.value)

    try:
        db.declare_constant("threshold", 1, 20, TypeSignature.INT32.value)
        raise AssertionError("Should reject redeclaring the same constant version")
    except ValueError:
        pass

    try:
        db.declare_constant("threshold", 0, 5, TypeSignature.INT32.value)
        raise AssertionError("Should reject lower version numbers")
    except ValueError:
        pass

    db.close()


def test_function_declaration():
    """Test declaring functions with locked dependencies."""
    db = Curry()

    # Setup constant
    db.declare_constant("rate", 1, 0.5, TypeSignature.FLOAT64.value)

    # Declare function using exact constant version
    db.declare_function(
        "apply_rate",
        1,
        "amount * rate",
        constant_bindings={"rate": 1},
        is_pure=True
    )

    # Retrieve
    func = db.get_function("apply_rate", 1)
    assert func["name"] == "apply_rate"
    assert func["version"] == 1
    assert func["constant_bindings"]["rate"] == 1
    assert func["is_pure"] is True

    db.close()


def test_function_execution_and_caching():
    """Test executing pure functions and hitting the cache on repeat calls."""
    db = Curry()

    db.declare_constant("rate", 1, 0.1, TypeSignature.FLOAT64.value)
    db.declare_function(
        "apply_rate",
        1,
        "amount * (1 - rate)",
        constant_bindings={"rate": 1},
        is_pure=True,
    )

    result_1 = db.call_function("apply_rate", 1, {"amount": 100})
    result_2 = db.call_function("apply_rate", 1, {"amount": 100})

    assert result_1 == 90.0
    assert result_2 == 90.0

    cursor = db.conn.cursor()
    cursor.execute(
        """SELECT hit_count FROM execution_cache
           WHERE function_name = ? AND function_version = ?""",
        ("apply_rate", 1)
    )
    row = cursor.fetchone()
    assert row is not None
    assert row["hit_count"] == 2

    db.close()


def test_function_dependency_validation():
    """Test that functions cannot use retired constants."""
    db = Curry()

    # Declare constant
    db.declare_constant("param", 1, 100, TypeSignature.INT32.value)

    # Retire it
    db.retire_constant("param", 1)

    # Try to declare function using retired constant
    try:
        db.declare_function(
            "use_param",
            1,
            "param * 2",
            constant_bindings={"param": 1},
        )
        raise AssertionError("Should have rejected function using retired constant")
    except ValueError:
        pass  # Expected

    db.close()


def test_function_composition():
    """Test function composition and lineage tracking."""
    db = Curry()

    # Setup constants
    db.declare_constant("a", 1, 10, TypeSignature.INT32.value)
    db.declare_constant("b", 1, 20, TypeSignature.INT32.value)

    # Setup functions
    db.declare_function("add_a", 1, "x + a", {"a": 1}, is_pure=True)
    db.declare_function("mult_b", 1, "x * b", {"b": 1}, is_pure=True)

    # Composite function
    db.declare_function(
        "combined",
        1,
        "True",
        constant_bindings={"a": 1, "b": 1},
        function_bindings={"add_a": 1, "mult_b": 1},
        is_pure=True
    )

    # Check lineage
    lineage = db.get_function_lineage("combined", 1)
    assert lineage["function"] == "combined@v1"
    assert len(lineage["dependencies"]["constants"]) == 2
    assert len(lineage["dependencies"]["functions"]) == 2

    db.close()


def test_function_dependency_rejects_unknown_function():
    """Test that function bindings reject unknown function references."""
    db = Curry()
    try:
        db.declare_function(
            "f1",
            1,
            "x + 1",
            function_bindings={"missing_fn": 1},
        )
        raise AssertionError("Should reject unknown function dependency")
    except ValueError:
        pass

    db.close()


def test_function_syntax_error():
    """Test that function bodies with syntax errors are rejected."""
    db = Curry()
    try:
        db.declare_function("bad_func", 1, "if True") # invalid python syntax
        raise AssertionError("Should reject syntax error")
    except ValueError:
        pass
    db.close()


def test_function_version_monotonicity():
    """Test that function versions must increase monotonically."""
    db = Curry()
    db.declare_function("func", 1, "1")
    try:
        db.declare_function("func", 1, "2")
        raise AssertionError("Should reject same version")
    except ValueError:
        pass
    db.close()


def test_model_registration():
    """Test registering models with locked parameters."""
    db = Curry()

    # Register model
    db.register_model(
        model_name="test-model",
        version=1,
        checkpoint_hash="abc123",
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
    )

    # Retrieve
    model = db.get_model("test-model", 1)
    assert model["model_name"] == "test-model"
    assert model["version"] == 1
    assert model["temperature"] == 0.7
    assert model["max_tokens"] == 1024

    db.close()


def test_model_version_monotonicity():
    """Test that model versions must increase monotonically."""
    db = Curry()
    db.register_model("m1", 1, "hash1")
    try:
        db.register_model("m1", 1, "hash2")
        raise AssertionError("Should reject same version")
    except ValueError:
        pass
    db.close()


def test_inference_recording():
    """Test recording inference results with provenance."""
    db = Curry()

    # Setup model
    db.register_model(
        "test-model", 1, "hash123",
        temperature=0.5, top_p=0.9, max_tokens=512
    )

    # Record inference
    inf_id = db.record_inference(
        model_name="test-model",
        model_version=1,
        input_tokens="test input",
        output_tokens=b"test output",
        seed=42,
        duration_ms=100,
        metadata={"custom": "data"}
    )

    # Retrieve
    inference = db.get_inference(inf_id)
    assert inference["model_name"] == "test-model"
    assert inference["model_version"] == 1
    assert inference["output_tokens"] == b"test output"
    assert inference["seed"] == 42
    assert inference["metadata"]["custom"] == "data"

    db.close()


def test_zero_inference_parameters_are_preserved():
    """Test that explicit zero values are stored rather than replaced with defaults."""
    db = Curry()

    db.register_model("zero-model", 1, "hash-zero", temperature=0.7, top_p=0.9, max_tokens=128)
    inf_id = db.record_inference(
        model_name="zero-model",
        model_version=1,
        input_tokens="test",
        output_tokens=b"output",
        temperature_used=0.0,
        top_p_used=0.0,
    )

    inference = db.get_inference(inf_id)
    assert inference["temperature_used"] == 0.0
    assert inference["top_p_used"] == 0.0

    db.close()


def test_inference_input_is_canonical_json():
    """Test that inference input is stored in canonical structured JSON."""
    db = Curry()
    db.register_model("canonical-model", 1, "hash-canon", temperature=0.1, top_p=0.5, max_tokens=64)
    inf_id = db.record_inference(
        model_name="canonical-model",
        model_version=1,
        input_tokens="hello",
        output_tokens=b"world",
    )
    inference = db.get_inference(inf_id)
    payload = json.loads(inference["input_tokens"])
    assert payload["raw_text"] == "hello"
    assert payload["token_refs"] is None
    assert payload["source_type"] == "str"
    db.close()


def test_negative_duration_is_rejected():
    """Test that negative duration values are rejected."""
    db = Curry()
    db.register_model("duration-model", 1, "hash-dur")
    try:
        db.record_inference(
            model_name="duration-model",
            model_version=1,
            input_tokens="prompt",
            output_tokens=b"output",
            duration_ms=-1,
        )
        raise AssertionError("Should reject negative duration")
    except ValueError:
        pass
    db.close()


def test_inference_reproducibility():
    """Test that same input + seed produces consistent records."""
    db = Curry()

    # Setup model
    db.register_model("model", 1, "hash", temperature=0.7, top_p=0.9, max_tokens=512)

    # Record two inferences with same input and seed
    inf1 = db.record_inference(
        model_name="model", model_version=1,
        input_tokens="prompt", output_tokens=b"output",
        seed=42
    )

    inf2 = db.record_inference(
        model_name="model", model_version=1,
        input_tokens="prompt", output_tokens=b"output",
        seed=42
    )

    # Retrieve both
    res1 = db.get_inference(inf1)
    res2 = db.get_inference(inf2)

    # Should have same model, input, output, seed
    assert res1["model_name"] == res2["model_name"]
    assert res1["model_version"] == res2["model_version"]
    assert res1["input_tokens"] == res2["input_tokens"]
    assert res1["output_tokens"] == res2["output_tokens"]
    assert res1["seed"] == res2["seed"]

    db.close()


def test_different_seeds_different_results():
    """Test that different seeds can produce different outputs."""
    db = Curry()

    db.register_model("model", 1, "hash", temperature=0.9, top_p=0.9, max_tokens=512)

    # Same input, different seed
    inf1 = db.record_inference(
        model_name="model", model_version=1,
        input_tokens="prompt", output_tokens=b"output A",
        seed=42
    )

    inf2 = db.record_inference(
        model_name="model", model_version=1,
        input_tokens="prompt", output_tokens=b"output B",  # Different
        seed=99
    )

    res1 = db.get_inference(inf1)
    res2 = db.get_inference(inf2)

    assert res1["seed"] != res2["seed"]
    assert res1["output_tokens"] != res2["output_tokens"]

    db.close()


def test_model_version_locks_parameters():
    """Test that model versions lock inference parameters."""
    db = Curry()

    # Register v1 with specific params
    db.register_model("m", 1, "h1", temperature=0.7, top_p=0.9, max_tokens=512)

    # Register v2 with different params
    db.register_model("m", 2, "h2", temperature=0.3, top_p=0.8, max_tokens=1024)

    # Get both
    v1 = db.get_model("m", 1)
    v2 = db.get_model("m", 2)

    # Parameters should differ
    assert v1["temperature"] != v2["temperature"]
    assert v1["top_p"] != v2["top_p"]
    assert v1["max_tokens"] != v2["max_tokens"]

    db.close()


def test_search_inferences_filters_and_pagination():
    """Test search filters by model/metadata and deterministic pagination."""
    db = Curry()
    db.register_model("search-model", 1, "hash-s1")
    db.register_model("search-model", 2, "hash-s2")

    inf_ids = []
    inf_ids.append(
        db.record_inference(
            model_name="search-model",
            model_version=1,
            input_tokens="a",
            output_tokens=b"a",
            seed=1,
            metadata={"api_provider": "local", "input_tokens_count": 10, "output_tokens_count": 20},
        )
    )
    inf_ids.append(
        db.record_inference(
            model_name="search-model",
            model_version=1,
            input_tokens="b",
            output_tokens=b"b",
            seed=2,
            metadata={"api_provider": "openai", "input_tokens_count": 11, "output_tokens_count": 21},
        )
    )
    inf_ids.append(
        db.record_inference(
            model_name="search-model",
            model_version=2,
            input_tokens="c",
            output_tokens=b"c",
            seed=2,
            metadata={"api_provider": "local", "input_tokens_count": 30, "output_tokens_count": 40},
        )
    )

    filtered = db.search_inferences(
        model_name="search-model",
        model_version=1,
        metadata_filters={"api_provider": "local"},
    )
    assert len(filtered) == 1
    assert filtered[0]["inference_id"] == inf_ids[0]

    ranged = db.search_inferences(
        model_name="search-model",
        min_input_tokens_count=20,
        max_output_tokens_count=45,
    )
    assert len(ranged) == 1
    assert ranged[0]["inference_id"] == inf_ids[2]

    page1 = db.search_inferences(model_name="search-model", limit=2, offset=0)
    page2 = db.search_inferences(model_name="search-model", limit=2, offset=2)
    assert len(page1) == 2
    assert len(page2) == 1
    assert page1[0]["inference_id"] != page2[0]["inference_id"]

    db.close()


def test_compare_inferences_diff():
    """Test structured inference comparison output."""
    db = Curry()
    db.register_model("cmp-model", 1, "hash-cmp")

    a_id = db.record_inference(
        model_name="cmp-model",
        model_version=1,
        input_tokens="same prompt",
        output_tokens=b"alpha output",
        seed=42,
        metadata={"api_provider": "local", "input_tokens_count": 10, "run": "a"},
    )
    b_id = db.record_inference(
        model_name="cmp-model",
        model_version=1,
        input_tokens="same prompt",
        output_tokens=b"beta output",
        seed=7,
        metadata={"api_provider": "local", "input_tokens_count": 10, "run": "b"},
    )

    diff = db.compare_inferences(a_id, b_id)
    assert diff["same_model"] is True
    assert diff["same_seed"] is False
    assert diff["same_input_tokens"] is True
    assert diff["same_output_hash"] is False
    assert "run" in diff["metadata_diff"]["changed_keys"]

    db.close()


def test_retirement_tag_grouping():
    """Test atomic grouping of related retirements."""
    db = Curry()

    # Declare constants
    db.declare_constant("a", 1, 1, TypeSignature.INT32.value)
    db.declare_constant("b", 1, 2, TypeSignature.INT32.value)
    db.declare_constant("c", 1, 3, TypeSignature.INT32.value)

    # Create tag
    tag = db.create_retirement_tag("batch-update", "Update a, b, c together")

    # Retire with tag
    db.retire_constant("a", 1, retirement_tag=tag)
    db.retire_constant("b", 1, retirement_tag=tag)
    db.retire_constant("c", 1, retirement_tag=tag)

    # Verify all are retired
    for const_id in ["a", "b", "c"]:
        try:
            db.get_constant(const_id, 1)
            raise AssertionError(f"Constant {const_id} should be retired")
        except ValueError:
            pass  # Expected

    db.close()


def test_cache_eviction():
    """Test that cache eviction limits size."""
    db = Curry()
    db.declare_function("f", 1, "x", is_pure=True)

    # Fill cache with 5 entries
    for i in range(5):
        db.call_function("f", 1, {"x": i})

    db.evict_execution_cache(max_entries=3)

    cursor = db.conn.cursor()
    cursor.execute("SELECT COUNT(*) as c FROM execution_cache")
    count = cursor.fetchone()["c"]
    assert count == 3, f"Expected 3 entries, got {count}"
    db.close()


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict, text: str = "", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._payload


class _FakeRequests:
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        if not self.sequence:
            raise RuntimeError("No fake responses left")
        item = self.sequence.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _make_fake_local_adapter(db, fake_requests, max_retries=2):
    adapter = LocalModelAdapter.__new__(LocalModelAdapter)
    adapter.db = db
    adapter.base_url = "http://fake.local"
    adapter.max_retries = max_retries
    adapter.retry_backoff_seconds = 0.0
    adapter.timeout_seconds = 5.0
    adapter.requests = fake_requests
    return adapter


def test_reu_success_path_writes_inference_row():
    """REU success path writes one complete inference row."""
    db = Curry()
    db.register_model("reu-local", 1, "reu-hash")
    fake_requests = _FakeRequests([
        _FakeResponse(
            200,
            {"response": "ok output", "prompt_eval_count": 12, "eval_count": 8, "id": "req-123"},
            headers={"x-request-id": "req-123"},
        )
    ])
    adapter = _make_fake_local_adapter(db, fake_requests)

    inf_id = adapter.infer_and_record("reu-local", 1, "prompt", seed=42)
    row = db.get_inference(inf_id)
    assert row["metadata"]["api_provider"] == "local"
    assert row["metadata"]["request_id"] == "req-123"
    assert row["metadata"]["input_tokens_count"] == 12
    assert row["metadata"]["output_tokens_count"] == 8

    db.close()


def test_reu_failure_writes_no_partial_inference():
    """REU failure raises and does not write inference rows."""
    db = Curry()
    db.register_model("reu-local", 1, "reu-hash")
    cursor = db.conn.cursor()
    cursor.execute("SELECT COUNT(*) AS count FROM inferences")
    before = cursor.fetchone()["count"]

    fake_requests = _FakeRequests([
        _FakeResponse(500, {}, text="server exploded"),
        _FakeResponse(500, {}, text="server exploded again"),
        _FakeResponse(500, {}, text="still failing"),
    ])
    adapter = _make_fake_local_adapter(db, fake_requests, max_retries=2)

    try:
        adapter.infer_and_record("reu-local", 1, "prompt", seed=42)
        raise AssertionError("Expected infer_and_record to fail")
    except RuntimeError:
        pass

    cursor.execute("SELECT COUNT(*) AS count FROM inferences")
    after = cursor.fetchone()["count"]
    assert before == after

    db.close()


def test_reu_retryable_transport_completes_successfully():
    """REU retry path eventually succeeds within retry budget."""
    db = Curry()
    db.register_model("reu-local", 1, "reu-hash")

    fake_requests = _FakeRequests([
        ConnectionError("temporary network issue 1"),
        ConnectionError("temporary network issue 2"),
        _FakeResponse(
            200,
            {"response": "eventual success", "prompt_eval_count": 5, "eval_count": 3, "id": "req-xyz"},
            headers={"x-request-id": "req-xyz"},
        ),
    ])
    adapter = _make_fake_local_adapter(db, fake_requests, max_retries=2)
    inf_id = adapter.infer_and_record("reu-local", 1, "prompt", seed=7)
    result = db.get_inference(inf_id)
    assert result["metadata"]["request_id"] == "req-xyz"
    assert fake_requests.calls == 3

    db.close()


def test_deserialize_type_validation():
    """Test that deserialization validates type signatures."""
    db = Curry()
    db.declare_constant("num", 1, 10, TypeSignature.INT32.value)

    # Manually corrupt the database type signature
    cursor = db.conn.cursor()
    cursor.execute("UPDATE constants SET type_signature = ? WHERE id = ?", (TypeSignature.STRING.value, "num"))
    db.conn.commit()

    try:
        db.get_constant("num", 1)
        raise AssertionError("Should have raised TypeError during deserialization")
    except TypeError:
        pass
    db.close()


def test_list_methods():
    """Test listing active constants, functions, and models."""
    db = Curry()

    db.declare_constant("c1", 1, 10, TypeSignature.INT32.value)
    db.declare_constant("c1", 2, 20, TypeSignature.INT32.value)
    db.declare_constant("c2", 1, 30, TypeSignature.INT32.value)
    db.retire_constant("c2", 1)

    db.declare_function("f1", 1, "c1", constant_bindings={"c1": 2})
    db.declare_function("f2", 1, "1")
    db.retire_function("f2", 1)

    db.register_model("m1", 1, "hash1", temperature=0.5)
    db.register_model("m1", 2, "hash2", temperature=0.6)
    db.register_model("m2", 1, "hash3")

    cursor = db.conn.cursor()
    cursor.execute("UPDATE model_versions SET retired_at = CURRENT_TIMESTAMP WHERE model_name = 'm2'")
    db.conn.commit()

    c_list = db.list_constants()
    f_list = db.list_functions()
    m_list = db.list_models()

    assert len(c_list) == 1
    assert c_list[0]["id"] == "c1"
    assert c_list[0]["latest_version"] == 2

    assert len(f_list) == 1
    assert f_list[0]["name"] == "f1"

    assert len(m_list) == 1
    assert m_list[0]["model_name"] == "m1"
    assert m_list[0]["latest_version"] == 2
    db.close()


def test_get_model_latest():
    """Test retrieving latest active model."""
    db = Curry()
    db.register_model("m1", 1, "hash1", temperature=0.1)
    db.register_model("m1", 2, "hash2", temperature=0.2)

    m = db.get_model_latest("m1")
    assert m["version"] == 2
    assert m["temperature"] == 0.2
    db.close()


def test_context_manager():
    """Test context manager usage."""
    with Curry() as db:
        db.declare_constant("c", 1, 1, TypeSignature.INT32.value)
        c = db.get_constant("c", 1)
        assert c["value"] == 1
    # Check if closed
    try:
        db.conn.cursor()
        raise AssertionError("Connection should be closed")
    except Exception:
        pass


def test_validate_function_body_unbound_variable():
    """Test function body validation."""
    db = Curry()
    try:
        db.declare_function("f", 1, "len(x) + a", expected_args=[])
        raise AssertionError("Should reject unbound variables")
    except ValueError:
        pass

    # Should work with expected_args
    db.declare_function("f", 2, "len(x) + a", expected_args=["x", "a"])

    # Should work with bindings
    db.declare_constant("b", 1, 10, TypeSignature.INT32.value)
    db.declare_function("g", 1, "b + sum([1, 2, 3])", constant_bindings={"b": 1})
    db.close()


def test_backup_database():
    import os
    with Curry() as db:
        db.declare_constant("c", 1, 99, TypeSignature.INT32.value)
        db.backup("test_backup.db")

    assert os.path.exists("test_backup.db")
    with Curry("test_backup.db") as backup_db:
        c = backup_db.get_constant("c", 1)
        assert c["value"] == 99

    # Clean up (ignoring windows lock errors)
    try:
        os.remove("test_backup.db")
        if os.path.exists("test_backup.db-wal"): os.remove("test_backup.db-wal")
        if os.path.exists("test_backup.db-shm"): os.remove("test_backup.db-shm")
    except OSError:
        pass


def test_json_serialization():
    """Test storing JSON constants."""
    db = Curry()

    data = {"key": "value", "nested": {"a": 1, "b": 2}}
    db.declare_constant("config", 1, data, TypeSignature.JSON_TYPE.value)

    # Retrieve
    const = db.get_constant("config", 1)
    assert const["value"] == data
    assert const["value"]["nested"]["a"] == 1

    db.close()


def test_bool_and_blob_constants():
    """Test round-tripping bool and blob constant values."""
    db = Curry()

    db.declare_constant("flag", 1, True, TypeSignature.BOOL.value)
    db.declare_constant("payload", 1, b"\x00\xff", TypeSignature.BLOB.value)

    flag = db.get_constant("flag", 1)
    payload = db.get_constant("payload", 1)

    assert flag["value"] is True
    assert payload["value"] == b"\x00\xff"

    db.close()


def test_string_constants():
    """Test storing string constants."""
    db = Curry()

    text = "This is a long string\nwith multiple\nlines"
    db.declare_constant("prompt", 1, text, TypeSignature.STRING.value)

    const = db.get_constant("prompt", 1)
    assert const["value"] == text
    assert "\n" in const["value"]

    db.close()


def test_two_tier_session():
    import tempfile
    import os
    import json
    from curry_core import CurrySession, TypeSignature

    with tempfile.TemporaryDirectory() as td:
        core_db_path = os.path.join(td, "core.db")
        project_dir = os.path.join(td, "project_a")
        os.makedirs(os.path.join(project_dir, ".curry"))

        config_path = os.path.join(project_dir, ".curry", "config.json")
        with open(config_path, "w") as f:
            json.dump({
                "project": "project-a",
                "core_db": core_db_path,
                "local_db": ".curry/curry.db"
            }, f)

        # Register stuff in core
        with Curry(core_db_path) as core:
            core.declare_constant("global_prompt", 1, "Be helpful", TypeSignature.STRING.value)
            core.register_model("claude-sonnet", 1, "hash123")

        # Use session
        with CurrySession.from_project(project_dir) as session:
            # Model operations map to core
            models = session.list_models()
            assert len(models) == 1
            assert models[0]["model_name"] == "claude-sonnet"

            # Local operations can access core constants
            session.declare_constant("local_var", 1, 42, TypeSignature.INT32.value)

            # Function depends on global_prompt
            session.declare_function(
                "test_func", 1,
                "global_prompt + '!'",
                constant_bindings={"global_prompt": 1}
            )

            res = session.call_function("test_func", 1, {})
            assert res == "Be helpful!"

            # List merges
            consts = session.list_constants()
            ids = {c["id"] for c in consts}
            assert "global_prompt" in ids
            assert "local_var" in ids


def test_function_description_fields():
    """Test that description and arg_descriptions are stored and returned correctly."""
    db = Curry()
    db.declare_function(
        name="priced_item",
        version=1,
        body="round(cost * (1 + margin), 2)",
        expected_args=["cost", "margin"],
        description="Compute selling price from cost and margin (markup_rate pattern).",
        arg_descriptions={
            "cost": "Wholesale cost in dollars (e.g. 80.00)",
            "margin": "Margin as a decimal fraction (e.g. 0.20 for 20%, NOT 20)",
        },
    )

    func = db.get_function("priced_item", 1)
    assert func["description"] == "Compute selling price from cost and margin (markup_rate pattern)."
    assert func["arg_descriptions"] is not None
    assert func["arg_descriptions"]["margin"] == "Margin as a decimal fraction (e.g. 0.20 for 20%, NOT 20)"

    listed = db.list_functions()
    assert len(listed) == 1
    assert listed[0]["description"] is not None
    assert listed[0]["arg_descriptions"]["cost"] == "Wholesale cost in dollars (e.g. 80.00)"

    # Round-trip: call the function to verify the body still works
    result = db.call_function("priced_item", 1, {"cost": 100.0, "margin": 0.20})
    assert result == 120.0

    db.close()


def test_arg_descriptions_surfaced_in_listing():
    """Test None fallback when arg_descriptions is omitted (Lag Pattern 2 guard)."""
    db = Curry()
    # Function without arg_descriptions -- should surface None, not raise
    db.declare_function(
        name="simple",
        version=1,
        body="x + 1",
        expected_args=["x"],
    )
    listed = db.list_functions()
    assert listed[0]["arg_descriptions"] is None
    assert listed[0]["description"] is None

    # Function with partial arg_descriptions -- missing keys surface None via .get()
    db.declare_function(
        name="partial",
        version=1,
        body="a + b",
        expected_args=["a", "b"],
        arg_descriptions={"a": "The first operand"},  # 'b' intentionally omitted
    )
    func = db.get_function("partial", 1)
    assert func["arg_descriptions"].get("a") == "The first operand"
    assert func["arg_descriptions"].get("b") is None  # missing key returns None

    db.close()


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "="*70)
    print("CURRY TEST SUITE")
    print("="*70 + "\n")

    runner = TestRunner()

    # Constants and Types
    print("Constants and Type Safety:")
    runner.test("Constant declaration and retrieval", test_constant_declaration)
    runner.test("Constant retirement", test_constant_retirement)
    runner.test("Constant retirement invalid ID", test_retire_invalid_id)
    runner.test("Type safety enforcement", test_type_safety)
    runner.test("Type safety after retirement", test_type_safety_after_retirement)
    runner.test("Constant version monotonicity", test_constant_versions_must_increase_monotonically)
    runner.test("JSON serialization", test_json_serialization)
    runner.test("Bool and blob constants", test_bool_and_blob_constants)
    runner.test("String constants", test_string_constants)
    runner.test("Deserialization type validation", test_deserialize_type_validation)

    print("\nFunction Versioning:")
    runner.test("Function declaration", test_function_declaration)
    runner.test("Function execution and caching", test_function_execution_and_caching)
    runner.test("Function dependency validation", test_function_dependency_validation)
    runner.test("Function composition and lineage", test_function_composition)
    runner.test("Unknown function dependency rejection", test_function_dependency_rejects_unknown_function)
    runner.test("Function syntax error detection", test_function_syntax_error)
    runner.test("Function version monotonicity", test_function_version_monotonicity)

    print("\nModel and Inference:")
    runner.test("Model registration", test_model_registration)
    runner.test("Model version monotonicity", test_model_version_monotonicity)
    runner.test("Inference recording", test_inference_recording)
    runner.test("Inference input canonical JSON", test_inference_input_is_canonical_json)
    runner.test("Negative inference duration rejection", test_negative_duration_is_rejected)
    runner.test("Zero-valued inference parameters", test_zero_inference_parameters_are_preserved)
    runner.test("Model version parameter locking", test_model_version_locks_parameters)
    runner.test("Search inferences filters and pagination", test_search_inferences_filters_and_pagination)
    runner.test("Compare inferences diff", test_compare_inferences_diff)

    print("\nReproducibility:")
    runner.test("Inference reproducibility", test_inference_reproducibility)
    runner.test("Different seeds different results", test_different_seeds_different_results)

    print("\nRetirement Management:")
    runner.test("Retirement tag grouping", test_retirement_tag_grouping)
    runner.test("Cache eviction", test_cache_eviction)

    print("\nREU Reliability:")
    runner.test("REU success writes complete inference", test_reu_success_path_writes_inference_row)
    runner.test("REU failure writes no partial rows", test_reu_failure_writes_no_partial_inference)
    runner.test("REU retryable transport eventually succeeds", test_reu_retryable_transport_completes_successfully)

    print("\nMCP Readiness:")
    runner.test("List constants, functions, models", test_list_methods)
    runner.test("Get model latest", test_get_model_latest)
    runner.test("Context manager", test_context_manager)
    runner.test("Validate function body unbound variable", test_validate_function_body_unbound_variable)
    runner.test("Backup database", test_backup_database)
    runner.test("Two-tier CurrySession architecture", test_two_tier_session)
    runner.test("Function description and arg_descriptions stored and retrieved", test_function_description_fields)
    runner.test("Dynamic tools use arg_descriptions for unit hints", test_arg_descriptions_surfaced_in_listing)

    runner.summary()

    return runner.failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
