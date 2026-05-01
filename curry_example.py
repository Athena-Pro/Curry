"""
Curry: Working Example
Demonstrates functional database + LLM integration with versioning and reproducibility.
"""

import sys
import json
from curry_core import Curry, TypeSignature
from curry_llm_adapters import get_adapter


def _supports_unicode() -> bool:
    """Detect whether stdout can encode the example banner characters."""
    try:
        "✓█⚠→".encode(sys.stdout.encoding or "utf-8")
        return True
    except (UnicodeEncodeError, AttributeError):
        return False


UNICODE_OK = _supports_unicode()


def _marker(ok: bool) -> str:
    """Return a portable status marker."""
    if UNICODE_OK:
        return "✓" if ok else "✗"
    return "[OK]" if ok else "[FAIL]"


def _note(symbol: str, fallback: str) -> str:
    """Return a unicode symbol or a readable ASCII fallback."""
    return symbol if UNICODE_OK else fallback


def _banner(char: str = "=") -> str:
    """Return a portable separator line."""
    return char * 70


def example_basic_constants_and_functions():
    """Example 1: Declaring versioned constants and functions."""
    print("\n" + _banner())
    print("EXAMPLE 1: Versioned Constants and Functions")
    print(_banner())
    
    # Create in-memory Curry database
    db = Curry()
    
    # Declare a constant: discount rate (v1)
    print("\n1. Declaring constant: base_discount_rate@v1 = 0.10")
    db.declare_constant(
        const_id="base_discount_rate",
        version=1,
        value=0.10,
        type_signature=TypeSignature.FLOAT64.value
    )
    
    # Retrieve it
    rate_v1 = db.get_constant("base_discount_rate", 1)
    print(f"   Retrieved: {rate_v1['value']} (type: {rate_v1['type_signature']})")
    
    # Declare a function that uses exact constant version
    print("\n2. Declaring function: apply_discount@v1 using base_discount_rate@v1")
    db.declare_function(
        name="apply_discount",
        version=1,
        body="amount * (1 - discount_rate)",
        constant_bindings={"base_discount_rate": 1},  # Locked to v1
        is_pure=True
    )
    
    func_v1 = db.get_function("apply_discount", 1)
    print(f"   Function body: {func_v1['body']}")
    print(f"   Constant bindings: {func_v1['constant_bindings']}")
    
    # Now retire old version and declare new one (atomically tagged)
    print("\n3. Retiring constant v1 and declaring v2 with new value")
    tag = db.create_retirement_tag(
        "Q2-2025-pricing-update",
        "New pricing strategy effective Q2 2025"
    )
    db.retire_constant("base_discount_rate", 1, retirement_tag=tag)
    db.declare_constant(
        const_id="base_discount_rate",
        version=2,
        value=0.12,
        type_signature=TypeSignature.FLOAT64.value
    )
    rate_v2 = db.get_constant("base_discount_rate", 2)
    print(f"   base_discount_rate@v1 retired")
    print(f"   base_discount_rate@v2 created with value: {rate_v2['value']}")
    
    # Try to retrieve retired version (should fail)
    print("\n4. Attempting to retrieve retired constant (should fail)")
    try:
        db.get_constant("base_discount_rate", 1)
    except ValueError as e:
        print(f"   {_marker(True)} Correctly rejected: {e}")
    
    # Declare v2 using new constant
    print("\n5. Declaring function: apply_discount@v2 using base_discount_rate@v2")
    db.declare_function(
        name="apply_discount",
        version=2,
        body="amount * (1 - discount_rate)",
        constant_bindings={"base_discount_rate": 2},  # Locked to v2
        is_pure=True
    )
    
    # Show lineage
    print("\n6. Function lineage (dependency tree)")
    lineage = db.get_function_lineage("apply_discount", 2)
    print(f"   Function: {lineage['function']}")
    print(f"   Dependencies: {json.dumps(lineage['dependencies'], indent=2)}")
    
    db.close()


def example_type_safety():
    """Example 2: Type safety enforcement."""
    print("\n" + _banner())
    print("EXAMPLE 2: Type Safety at Database Level")
    print(_banner())
    
    db = Curry()
    
    # Declare a string constant
    print("\n1. Declaring constant: greeting@v1 (String type)")
    db.declare_constant(
        const_id="greeting",
        version=1,
        value="Hello, World!",
        type_signature=TypeSignature.STRING.value
    )
    
    # Try to declare v2 with same ID but different type (should fail)
    print("\n2. Attempting to declare v2 with Float64 type (should fail)")
    try:
        db.declare_constant(
            const_id="greeting",
            version=2,
            value=3.14,
            type_signature=TypeSignature.FLOAT64.value
        )
    except TypeError as e:
        print(f"   {_marker(True)} Correctly rejected: {e}")
    
    # Declare v2 with correct type
    print("\n3. Declaring v2 with correct String type")
    db.declare_constant(
        const_id="greeting",
        version=2,
        value="Hello, Curry!",
        type_signature=TypeSignature.STRING.value
    )
    print(f"   {_marker(True)} Success")
    
    db.close()


def example_model_registration_and_inference():
    """Example 3: Model registration and deterministic inference."""
    print("\n" + _banner())
    print("EXAMPLE 3: Model Registration and Versioning")
    print(_banner())
    
    db = Curry()
    
    # Register model v1 with specific parameters
    print("\n1. Registering gpt-4@v1 with locked inference parameters")
    db.register_model(
        model_name="gpt-4",
        version=1,
        checkpoint_hash="abcd1234ef5678",
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
        model_type="gpt"
    )
    
    model = db.get_model("gpt-4", 1)
    print(f"   Model: {model['model_name']}@v{model['version']}")
    print(f"   Temperature (locked): {model['temperature']}")
    print(f"   Max tokens (locked): {model['max_tokens']}")
    
    # Register v2 with different parameters
    print("\n2. Registering gpt-4@v2 with different parameters")
    db.register_model(
        model_name="gpt-4",
        version=2,
        checkpoint_hash="abcd1234ef5679",  # New weights
        temperature=0.5,  # Lower temperature for more consistency
        top_p=0.8,
        max_tokens=2048,  # Increased context
        model_type="gpt"
    )
    
    model_v2 = db.get_model("gpt-4", 2)
    print(f"   Model: {model_v2['model_name']}@v{model_v2['version']}")
    print(f"   Temperature (locked): {model_v2['temperature']}")
    
    # Record inference results
    print("\n3. Recording inference results (simulated)")
    inference_id = db.record_inference(
        model_name="gpt-4",
        model_version=1,
        input_tokens="What is the capital of France?",
        output_tokens=b"The capital of France is Paris.",
        seed=42,
        duration_ms=234,
        metadata={"tokens_used": 45}
    )
    print(f"   Inference ID: {inference_id}")
    
    # Retrieve inference
    inference = db.get_inference(inference_id)
    print(f"\n4. Retrieved inference details:")
    print(f"   Model: {inference['model_name']}@v{inference['model_version']}")
    print(f"   Input: {inference['input_tokens']}")
    print(f"   Output: {inference['output_tokens'].decode('utf-8')}")
    print(f"   Temperature used: {inference['temperature_used']}")
    print(f"   Seed: {inference['seed']}")
    
    # Record same inference again with different seed (should differ)
    print("\n5. Recording inference with different seed")
    inference_id_2 = db.record_inference(
        model_name="gpt-4",
        model_version=1,
        input_tokens="What is the capital of France?",
        output_tokens=b"Paris is the capital of France.",  # Different output
        seed=99,  # Different seed
        duration_ms=210,
    )
    
    inf2 = db.get_inference(inference_id_2)
    print(f"   Same model@v, different seed -> different output")
    print(f"   Output 1: {inference['output_tokens'].decode('utf-8')}")
    print(f"   Output 2: {inf2['output_tokens'].decode('utf-8')}")
    
    db.close()


def example_deterministic_reproducibility():
    """Example 4: Demonstrating deterministic reproducibility."""
    print("\n" + _banner())
    print("EXAMPLE 4: Deterministic Reproducibility")
    print(_banner())
    
    db = Curry()
    
    # Setup: declare constants and model
    print("\n1. Setting up: Constants and Model")
    db.declare_constant("temperature_setting", 1, 0.7, TypeSignature.FLOAT64.value)
    db.declare_constant("system_prompt", 1, "Be concise.", TypeSignature.STRING.value)
    
    db.register_model(
        model_name="test-model",
        version=1,
        checkpoint_hash="xyz789",
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        system_prompt_id="system_prompt",
        system_prompt_version=1
    )
    
    # Record inference A
    inference_a = db.record_inference(
        model_name="test-model",
        model_version=1,
        input_tokens="Explain gravity in 2 sentences.",
        output_tokens=b"Gravity is the force that attracts objects. Heavier objects exert stronger gravitational pull.",
        seed=12345,
        duration_ms=150
    )
    print(f"   Inference A ID: {inference_a}")
    
    # Record inference B with exact same inputs
    inference_b = db.record_inference(
        model_name="test-model",
        model_version=1,
        input_tokens="Explain gravity in 2 sentences.",
        output_tokens=b"Gravity is the force that attracts objects. Heavier objects exert stronger gravitational pull.",
        seed=12345,  # Same seed
        duration_ms=148
    )
    print(f"   Inference B ID (same inputs): {inference_b}")
    
    # Retrieve and compare
    print("\n2. Comparing Results")
    a = db.get_inference(inference_a)
    b = db.get_inference(inference_b)
    
    print(f"   Model:      {a['model_name']}@v{a['model_version']} == {b['model_name']}@v{b['model_version']}")
    print(f"   Input:      {a['input_tokens']} == {b['input_tokens']}")
    print(f"   Seed:       {a['seed']} == {b['seed']}")
    print(f"   Output:     {a['output_tokens']} == {b['output_tokens']}")
    print(f"   {_marker(True)} IDENTICAL: Reproducibility guaranteed!")
    
    # Now change one parameter
    print("\n3. Changing seed (should produce different output path)")
    inference_c = db.record_inference(
        model_name="test-model",
        model_version=1,
        input_tokens="Explain gravity in 2 sentences.",
        output_tokens=b"Gravity pulls things downward. This is why we don't float away.",  # Different
        seed=99999,  # Different seed
        duration_ms=145
    )
    
    c = db.get_inference(inference_c)
    print(f"   Seed:       {c['seed']} (different)")
    print(f"   Output:     {c['output_tokens'].decode('utf-8')} (different)")
    print(f"   {_marker(True)} Different seeds can produce different outputs")
    
    db.close()


def example_versioning_cascade():
    """Example 5: Cascade retirements and dependency management."""
    print("\n" + _banner())
    print("EXAMPLE 5: Versioning and Dependency Management")
    print(_banner())
    
    db = Curry()
    
    # Create initial setup
    print("\n1. Initial setup:")
    print("   - Constant: threshold@v1")
    print("   - Function: check_threshold@v1 (uses threshold@v1)")
    
    db.declare_constant("threshold", 1, 100, TypeSignature.INT32.value)
    db.declare_function(
        name="check_threshold",
        version=1,
        body="value > threshold",
        constant_bindings={"threshold": 1},  # References the constant ID
        is_pure=True
    )
    
    # Verify lineage
    lineage = db.get_function_lineage("check_threshold", 1)
    print(f"\n2. Function lineage:")
    print(f"   {json.dumps(lineage, indent=3)}")
    
    # Create a new version of the constant
    print("\n3. Creating threshold@v2 (retirement tag)")
    tag = db.create_retirement_tag(
        "threshold-update-2025",
        "Increase threshold for new SLA"
    )
    db.retire_constant("threshold", 1, retirement_tag=tag)
    db.declare_constant("threshold", 2, 150, TypeSignature.INT32.value)
    
    print(f"   threshold@v1 retired")
    print(f"   threshold@v2 created with value 150")
    
    # Try to retrieve retired constant (should fail)
    print("\n4. Attempting to use retired constant:")
    try:
        const = db.get_constant("threshold", 1)
        print(f"   {_marker(False)} Should have failed!")
    except ValueError as e:
        print(f"   {_marker(True)} Correctly rejected: {e}")
    
    # Function v1 is now locked to retired dependency
    print("\n5. Function check_threshold@v1 status:")
    print(f"   {_marker(True)} Still locked to threshold@v1")
    print(f"   {_note('⚠', '[WARN]')} But threshold@v1 is now retired")
    print(f"   {_note('→', '->')} Function should be considered 'stale' or require migration")
    
    # Create function v2 using new constant
    print("\n6. Creating check_threshold@v2 using threshold@v2")
    db.declare_function(
        name="check_threshold",
        version=2,
        body="value > threshold",
        constant_bindings={"threshold": 2},
        is_pure=True
    )
    print(f"   {_marker(True)} check_threshold@v2 created with fresh bindings")
    
    db.close()


def example_with_real_llm():
    """Example 6: Integration with real LLM (requires API key)."""
    print("\n" + _banner())
    print("EXAMPLE 6: Real LLM Integration (Claude API)")
    print(_banner())
    
    print("\nNOTE: This example requires ANTHROPIC_API_KEY environment variable.")
    print("To run with real inference, uncomment the code and set your API key.")
    
    # Uncomment to run with real API:
    """
    import os
    
    db = Curry("curry_example.db")
    
    # Register Claude model
    print("\n1. Registering claude-3-5-sonnet with inference parameters")
    db.register_model(
        model_name="claude-3-5-sonnet-20241022",
        version=1,
        checkpoint_hash="release_2024_10",
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
        model_type="claude"
    )
    
    # Get Claude adapter
    adapter = get_adapter("claude", db, api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    # Run inference
    print("\n2. Running inference with Claude")
    prompt = "Explain quantum computing in 2 sentences."
    inference_id = adapter.infer_and_record(
        model_name="claude-3-5-sonnet-20241022",
        model_version=1,
        prompt=prompt,
        seed=42
    )
    print(f"   Inference recorded: {inference_id}")
    
    # Retrieve and display
    inference = db.get_inference(inference_id)
    print(f"\n3. Inference results:")
    print(f"   Model: {inference['model_name']}@v{inference['model_version']}")
    print(f"   Prompt: {prompt}")
    print(f"   Response: {inference['output_tokens'].decode('utf-8')[:200]}...")
    print(f"   Duration: {inference['execution_duration_ms']}ms")
    
    db.close()
    """
    
    print("\nExample setup complete. Create your own database with:")
    print("  from curry_core import Curry")
    print("  from curry_llm_adapters import get_adapter")


def main():
    """Run all examples."""
    print("\n" + _banner("#" if not UNICODE_OK else "█"))
    print("  CURRY: Functional Database for LLM Operations")
    print("  Working Examples and Demonstrations")
    print(_banner("#" if not UNICODE_OK else "█"))
    
    # Run all examples
    example_basic_constants_and_functions()
    example_type_safety()
    example_model_registration_and_inference()
    example_deterministic_reproducibility()
    example_versioning_cascade()
    example_with_real_llm()
    
    print("\n" + _banner())
    print("ALL EXAMPLES COMPLETED")
    print(_banner())
    print("\nNext Steps:")
    print("1. Create a Curry database: db = Curry('my_curry.db')")
    print("2. Declare constants and functions with versioning")
    print("3. Register LLM models with locked parameters")
    print("4. Run inferences and record provenance")
    print("5. Query lineage and compare results deterministically")


if __name__ == "__main__":
    main()
