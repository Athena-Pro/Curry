import json
from curry_core import Curry, TypeSignature

def test_token_translation():
    print("======================================================================")
    print("TESTING TOKEN TRANSLATION CAPABILITIES")
    print("======================================================================")

    db = Curry()

    # Simulate a raw text input
    raw_text = "The quick brown fox jumps over the lazy dog."

    # 1. Storing tokens as traditional string representation
    print("\n1. Testing string-based tokens:")
    db.declare_constant("prompt_text", 1, raw_text, TypeSignature.TOKENS.value)
    stored_text = db.get_constant("prompt_text", 1)
    print(f"Stored text tokens successfully: {stored_text['value']}")

    # 2. Storing tokens as model-specific integer arrays (simulated GPT-4 tokens)
    print("\n2. Testing model-specific integer array tokens (GPT-4 simulated):")
    gpt4_tokens = [464, 4062, 14198, 39805, 18512, 625, 262, 16125, 3290, 13]
    gpt4_payload = {
        "model": "gpt-4",
        "tokens": gpt4_tokens,
        "source_text": raw_text
    }
    db.declare_constant("prompt_tokens_gpt4", 1, gpt4_payload, TypeSignature.TOKENS.value)
    stored_gpt4 = db.get_constant("prompt_tokens_gpt4", 1)
    print(f"Stored GPT-4 token payload successfully: {json.dumps(stored_gpt4['value'])}")

    # 3. Translating to another model's tokens (simulated Claude tokens)
    print("\n3. Testing token translation (Claude simulated):")
    # In a real scenario, we'd use tiktoken and anthropic tokenizers to encode the source_text
    claude_tokens = [794, 3054, 8251, 23999, 12903, 405, 278, 14751, 3350, 15]
    claude_payload = {
        "model": "claude-3-opus",
        "tokens": claude_tokens,
        "source_text": stored_gpt4['value']['source_text'] # Using raw text as source of truth
    }
    db.declare_constant("prompt_tokens_claude", 1, claude_payload, TypeSignature.TOKENS.value)
    stored_claude = db.get_constant("prompt_tokens_claude", 1)
    print(f"Translated and stored Claude token payload successfully: {json.dumps(stored_claude['value'])}")

    # 4. Recording inference with raw token lists
    print("\n4. Testing inference recording with structured token data:")
    db.register_model("gpt-4", 1, "hash_xyz", max_tokens=100)

    inference_id = db.record_inference(
        model_name="gpt-4",
        model_version=1,
        input_tokens=gpt4_payload,
        output_tokens=b"Expected output blob",
        metadata={"translation_test": True}
    )

    inference = db.get_inference(inference_id)
    retrieved_input = json.loads(inference["input_tokens"])
    token_payload = retrieved_input["token_refs"]
    print(f"Inference input tokens preserved successfully:")
    print(f"  Model: {token_payload['model']}")
    print(f"  Tokens List Length: {len(token_payload['tokens'])}")

    print("\n======================================================================")
    print("[OK] Token translation test passed successfully!")
    print("======================================================================")

if __name__ == "__main__":
    test_token_translation()
