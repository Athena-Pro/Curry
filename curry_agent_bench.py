"""
curry_agent_bench.py
--------------------
Benchmark: do dynamic per-function MCP tools reduce agent turn count and token cost
compared to the generic call_function surface?

Two modes per task:

  GENERIC   -- agent has call_function(name, version, args) + list_functions().
               Discovering what to call and how costs at least one extra turn.

  DYNAMIC   -- each Curry function is exposed as its own named tool with a typed
               input schema derived from expected_args. No discovery step needed.

Both modes have identical non-function tools (list_constants, session_info).
list_functions is available in GENERIC mode but absent in DYNAMIC mode -- that
asymmetry is the whole point.

Usage:
    pip install anthropic
    python curry_agent_bench.py
    python curry_agent_bench.py --tasks simple_single chained
    python curry_agent_bench.py --model claude-haiku-4-5-20251001 --runs 3

Output:
    curry_bench_results.json  -- raw per-run data
    Prints a summary table to stdout.
"""

import argparse
import json
import math # noqa: F401
import os
import re
import statistics
import sys
import time
from copy import deepcopy # noqa: F401
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

try:
    import requests as _requests
except ImportError:
    _requests = None  # type: ignore[assignment]

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

sys.path.insert(0, os.path.dirname(__file__))
from curry_core import Curry


# ─────────────────────────────────────────────────────────────────────────────
# Pricing (USD per million tokens, May 2026)
# ─────────────────────────────────────────────────────────────────────────────

PRICING: Dict[str, Dict[str, float]] = {
    "claude-haiku-4-5-20251001":  {"input": 0.80,  "output": 4.00},
    "claude-haiku-4-5":           {"input": 0.80,  "output": 4.00},
    "claude-sonnet-4-5-20251001": {"input": 3.00,  "output": 15.00},
    "claude-sonnet-4-5":          {"input": 3.00,  "output": 15.00},
    "claude-sonnet-4-6":          {"input": 3.00,  "output": 15.00},
    "claude-opus-4-5":            {"input": 15.00, "output": 75.00},
}

def _price(model: str, input_tokens: float, output_tokens: float) -> float:
    """Estimated USD cost for a given token count at model list pricing."""
    rates = PRICING.get(model, {"input": 3.00, "output": 15.00})
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: in-memory Curry DB with benchmark functions
# ─────────────────────────────────────────────────────────────────────────────

def build_fixture() -> Curry:
    """
    Create an in-memory Curry DB with a small but realistic set of functions.
    Covers single-arg, multi-arg, chained, and composed cases.
    All functions include description and arg_descriptions for Lag Pattern 2 compliance.
    """
    db = Curry(":memory:")

    # Constants
    db.declare_constant("discount_rate", 1, 0.15, "Float64")    # 15% discount
    db.declare_constant("tax_rate",      1, 0.08, "Float64")    # 8% tax
    db.declare_constant("markup_rate",   1, 0.20, "Float64")    # 20% markup
    db.declare_constant("min_order",     1, 50,   "Int32")      # $50 minimum

    # apply_discount(amount) -> amount after 15% discount
    db.declare_function(
        name="apply_discount",
        version=1,
        body="round(amount * (1 - discount_rate), 2)",
        constant_bindings={"discount_rate": 1},
        expected_args=["amount"],
        is_pure=True,
        description="Apply the standard 15% discount (discount_rate constant) to a gross amount.",
        arg_descriptions={"amount": "Gross price in dollars (e.g. 250.00)"},
    )

    # compute_tax(amount) -> tax portion on an amount
    db.declare_function(
        name="compute_tax",
        version=1,
        body="round(amount * tax_rate, 2)",
        constant_bindings={"tax_rate": 1},
        expected_args=["amount"],
        is_pure=True,
        description="Compute the tax amount (tax_rate constant, 8%) on a given price.",
        arg_descriptions={"amount": "Price in dollars to compute tax on (e.g. 212.50)"},
    )

    # final_price(amount) -> discounted amount + tax
    db.declare_function(
        name="final_price",
        version=1,
        body="round(amount * (1 - discount_rate) * (1 + tax_rate), 2)",
        constant_bindings={"discount_rate": 1, "tax_rate": 1},
        expected_args=["amount"],
        is_pure=True,
        description="Compute final price after applying discount_rate then tax_rate. Single-step alternative to chaining apply_discount and compute_tax.",
        arg_descriptions={"amount": "Original gross price in dollars (e.g. 450.00)"},
    )

    # apply_markup(cost) -> cost with markup
    db.declare_function(
        name="apply_markup",
        version=1,
        body="round(cost * (1 + markup_rate), 2)",
        constant_bindings={"markup_rate": 1},
        expected_args=["cost"],
        is_pure=True,
        description="Apply the standard markup (markup_rate constant, 20%) to a wholesale cost to get selling price.",
        arg_descriptions={"cost": "Wholesale cost in dollars (e.g. 80.00)"},
    )

    # meets_minimum(order_total) -> bool: does order meet minimum?
    db.declare_function(
        name="meets_minimum",
        version=1,
        body="order_total >= min_order",
        constant_bindings={"min_order": 1},
        expected_args=["order_total"],
        is_pure=True,
        description="Check whether an order total meets the minimum order threshold (min_order constant, $50). Returns True or False.",
        arg_descriptions={"order_total": "Order total in dollars (e.g. 40.00)"},
    )

    # compound_interest(principal, rate, years) -> final value
    db.declare_function(
        name="compound_interest",
        version=1,
        body="round(principal * (1 + rate) ** years, 2)",
        constant_bindings={},
        expected_args=["principal", "rate", "years"],
        is_pure=True,
        description="Compute compound interest final value. Formula: principal * (1 + rate) ** years.",
        arg_descriptions={
            "principal": "Starting principal amount in dollars (e.g. 1000.00)",
            "rate": "Annual interest rate as a decimal fraction (e.g. 0.06 for 6%, NOT 6)",
            "years": "Investment duration in whole years (e.g. 5)",
        },
    )

    return db


# ─────────────────────────────────────────────────────────────────────────────
# Tool surface builders
# ─────────────────────────────────────────────────────────────────────────────

def _base_tools() -> List[Dict]:
    """Infrastructure tools present in both modes."""
    return [
        {
            "name": "list_constants",
            "description": "List all active constants and their current values.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "call_constant",
            "description": "Retrieve the current value of a named constant.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "const_id": {"type": "string", "description": "Constant name"},
                },
                "required": ["const_id"],
            },
        },
        {
            "name": "session_info",
            "description": "Return session metadata: available functions and constants.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
    ]


def build_generic_tools() -> List[Dict]:
    """
    GENERIC surface: agent must discover functions via list_functions,
    then invoke them through the single call_function dispatcher.
    """
    tools = _base_tools()
    tools += [
        {
            "name": "list_functions",
            "description": (
                "List all available Curry functions with their names, versions, "
                "argument names, and descriptions. Call this first to discover "
                "what functions are available and what arguments they need."
            ),
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "call_function",
            "description": (
                "Execute a Curry function by name and version with the given arguments. "
                "Use list_functions first to discover available functions and their argument names."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "name":    {"type": "string",  "description": "Function name"},
                    "version": {"type": "integer", "description": "Function version"},
                    "args":    {"type": "object",  "description": "Runtime arguments dict"},
                },
                "required": ["name", "version", "args"],
            },
        },
    ]
    return tools


def build_dynamic_tools(db: Curry) -> List[Dict]:
    """
    DYNAMIC surface: each function gets its own named tool with a typed schema
    derived from expected_args and arg_descriptions (Lag Pattern 2 fix).
    Non-obvious arguments carry unit hints from arg_descriptions; without a
    stored hint the fallback is a generic description which is insufficient for
    args like 'rate' or 'years'.
    """
    tools = _base_tools()
    for func in db.list_functions():
        name        = func["name"]
        version     = func["latest_version"]
        args        = func.get("expected_args") or []
        fn_desc     = func.get("description") or ""
        fn_arg_desc = func.get("arg_descriptions") or {}

        tool_description = (
            f"Call '{name}' (v{version})."
            + (f" {fn_desc}" if fn_desc else "")
            + (f" Args: {', '.join(args)}." if args else " No args.")
        )

        properties = {}
        for arg in args:
            hint = fn_arg_desc.get(arg) or f"Value for argument '{arg}'"
            properties[arg] = {"type": "number", "description": hint}

        tools.append({
            "name": f"{name}_v{version}",
            "description": tool_description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": args,
            },
        })

    return tools


# ─────────────────────────────────────────────────────────────────────────────
# Tool execution (simulates MCP dispatch against the in-memory DB)
# ─────────────────────────────────────────────────────────────────────────────

def execute_tool(
    tool_name: str,
    tool_input: Dict[str, Any],
    db: Curry,
    mode: str,
) -> Any:
    """Dispatch a tool call against the fixture DB and return a JSON-serializable result."""

    if tool_name == "session_info":
        return {
            "mode": mode,
            "functions": [
                {"name": f["name"], "version": f["latest_version"],
                 "expected_args": f.get("expected_args")}
                for f in db.list_functions()
            ],
            "constants": [
                {"id": c["id"], "version": c["latest_version"]}
                for c in db.list_constants()
            ],
        }

    if tool_name == "list_constants":
        result = []
        for c in db.list_constants():
            try:
                val = db.get_constant_latest(c["id"])
                result.append({"id": c["id"], "value": val["value"],
                                "type": val["type_signature"]})
            except Exception:
                pass
        return result

    if tool_name == "call_constant":
        const_id = tool_input["const_id"]
        val = db.get_constant_latest(const_id)
        return {"id": const_id, "value": val["value"]}

    if tool_name == "list_functions":
        return [
            {
                "name": f["name"],
                "version": f["latest_version"],
                "expected_args": f.get("expected_args"),
                "description": f"Call with: {f['name']}_v{f['latest_version']}({', '.join(f.get('expected_args') or [])})",
            }
            for f in db.list_functions()
        ]

    if tool_name == "call_function":
        name    = tool_input["name"]
        version = tool_input["version"]
        args    = tool_input.get("args", {})
        result  = db.call_function(name, version, args)
        return {"function": f"{name}@v{version}", "result": result}

    # Dynamic per-function tool: e.g. "apply_discount_v1"
    for func in db.list_functions():
        fn    = func["name"]
        ver   = func["latest_version"]
        if tool_name == f"{fn}_v{ver}":
            result = db.call_function(fn, ver, tool_input)
            return {"function": f"{fn}@v{ver}", "result": result}

    return {"error": f"Unknown tool: {tool_name}"}


# ─────────────────────────────────────────────────────────────────────────────
# Schema overhead measurement
# ─────────────────────────────────────────────────────────────────────────────

def measure_schema_overhead(
    client: anthropic.Anthropic,
    tools: List[Dict],
    model: str,
    system: str,
) -> int:
    """
    Count input tokens consumed by the tool definitions + system prompt alone,
    before any conversation history.  Uses client.messages.count_tokens() so no
    inference cost is incurred.

    Returns the token count, or -1 if the API does not support count_tokens.
    The single-word user message ("x") is intentional -- it gives a clean
    baseline for fixed overhead independent of task prompt length.
    """
    try:
        resp = client.messages.count_tokens(
            model=model,
            system=system,
            tools=tools,          # type: ignore[arg-type]
            messages=[{"role": "user", "content": "x"}],
        )
        return resp.input_tokens
    except Exception:
        return -1


# ─────────────────────────────────────────────────────────────────────────────
# Provider-agnostic helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_ollama_tools(tools: List[Dict]) -> List[Dict]:
    """Convert Anthropic tool schema format to Ollama/OpenAI function-calling format.

    Anthropic:  {"name": ..., "description": ..., "input_schema": {"type": "object", ...}}
    Ollama:     {"type": "function", "function": {"name": ..., "description": ...,
                    "parameters": {"type": "object", ...}}}
    """
    result = []
    for t in tools:
        schema = t.get("input_schema", {})
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": {
                    "type": schema.get("type", "object"),
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            },
        })
    return result


def check_correctness(answer: str, expected_values: List) -> bool:
    """Return True if *answer* contains all expected values.

    - Numeric values (int/float): accepted if the answer contains a number
      within ±0.01 of the expected value.
    - String values (e.g. "false", "true"): checked case-insensitively as a
      substring of the answer.
    """
    answer_lower = answer.lower()
    nums_in_answer: List[float] = []
    for m in re.finditer(r"-?\d+(?:\.\d+)?", answer):
        try:
            nums_in_answer.append(float(m.group()))
        except ValueError:
            pass

    for ev in expected_values:
        if isinstance(ev, bool):
            needle = "true" if ev else "false"
            if needle not in answer_lower:
                return False
        elif isinstance(ev, (int, float)):
            if not any(abs(n - float(ev)) <= 0.01 for n in nums_in_answer):
                return False
        elif isinstance(ev, str):
            if ev.lower() not in answer_lower:
                return False
    return True


TASKS = [
    {
        "id": "single_known",
        "prompt": (
            "Calculate the discounted price for an item that costs $250. "
            "Use the apply_discount function and return only the final dollar amount."
        ),
        "hint": "Direct single-function call. Generic must discover args first.",
        # apply_discount(250) = round(250 * (1 - 0.15), 2) = 212.5
        "expected_values": [212.5],
    },
    {
        "id": "single_multiarg",
        "prompt": (
            "What is the compound interest final value on a $1000 principal "
            "at 6% annual rate over 5 years? Return the numeric result only."
        ),
        "hint": "Multi-arg function. Dynamic tool schema makes args unambiguous.",
        # compound_interest(1000, 0.06, 5) = round(1000 * 1.06**5, 2) = 1338.23
        "expected_values": [1338.23],
    },
    {
        "id": "sequential_two",
        "prompt": (
            "An item costs $300. First apply the discount, then compute the tax "
            "on the discounted price. Return both intermediate values and the tax amount."
        ),
        "hint": "Two sequential function calls. Tests chaining efficiency.",
        # apply_discount(300)=255.0  compute_tax(255.0)=round(255*0.08,2)=20.4
        "expected_values": [255.0, 20.4],
    },
    {
        "id": "composed_final",
        "prompt": (
            "What is the final price (after both discount and tax) for an item "
            "originally priced at $450? Use the most direct function available."
        ),
        "hint": "final_price does it in one call. Does agent find it directly?",
        # final_price(450) = round(450 * 0.85 * 1.08, 2) = 413.1
        "expected_values": [413.1],
    },
    {
        "id": "multi_values",
        "prompt": (
            "Calculate the discounted price for three items: $100, $200, and $500. "
            "Return a list of all three discounted amounts."
        ),
        "hint": "Repeated single-function calls. Tests whether agent batches or loops.",
        # apply_discount(100)=85.0  apply_discount(200)=170.0  apply_discount(500)=425.0
        "expected_values": [85.0, 170.0, 425.0],
    },
    {
        "id": "discovery_ambiguous",
        "prompt": (
            "I have a product that costs $80 to make. What's the selling price "
            "after applying the standard markup? Also tell me if a $40 order "
            "meets the minimum order requirement."
        ),
        "hint": "Two different functions with non-obvious names. Discovery value is high.",
        # apply_markup(80)=96.0  meets_minimum(40)=False (40 < 50)
        "expected_values": [96.0, "false"],
    },
]

TASK_MAP = {t["id"]: t for t in TASKS}


# ─────────────────────────────────────────────────────────────────────────────
# Run a single task in a single mode
# ─────────────────────────────────────────────────────────────────────────────

MAX_TURNS = 10  # Safety ceiling -- real tasks should finish in 2-4

@dataclass
class RunResult:
    task_id: str
    mode: str               # "generic" | "dynamic"
    turns: int              # API calls made
    tool_calls: int         # total tool invocations across all turns
    discovery_calls: int    # calls to list_functions / list_constants
    function_calls: int     # calls that actually executed a Curry function
    input_tokens: int
    output_tokens: int
    total_tokens: int
    first_call_to_function: bool  # did turn 1's first tool go directly to a function?
    completed: bool
    final_answer: str
    elapsed_ms: int
    correct: Optional[bool] = None    # None = not graded; True/False = graded
    error: Optional[str] = None
    tool_call_trace: List[str] = field(default_factory=list)


def run_task_anthropic(
    task: Dict,
    mode: str,
    tools: List[Dict],
    db: Curry,
    client,          # anthropic.Anthropic
    model: str,
) -> RunResult:
    system = (
        "You are an agent with access to Curry, a deterministic functional database. "
        "Complete the user's request using the available tools. "
        "Be efficient -- use the minimum number of tool calls necessary. "
        "Return a concise final answer when done."
    )

    messages = [{"role": "user", "content": task["prompt"]}]

    turns            = 0
    tool_calls       = 0
    discovery_calls  = 0
    function_calls   = 0
    input_tokens     = 0
    output_tokens    = 0
    first_call_to_fn = False   # set on the very first tool call this run
    first_call_seen  = False
    call_trace: List[str] = []
    start_ms = time.monotonic()

    while turns < MAX_TURNS:
        turns += 1
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=system,
                tools=tools,
                messages=messages,
            )
        except Exception as exc:
            elapsed = int((time.monotonic() - start_ms) * 1000)
            return RunResult(
                task_id=task["id"], mode=mode, turns=turns,
                tool_calls=tool_calls, discovery_calls=discovery_calls,
                function_calls=function_calls,
                input_tokens=input_tokens, output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                first_call_to_function=first_call_to_fn,
                completed=False, final_answer="",
                elapsed_ms=elapsed, correct=False, error=str(exc),
                tool_call_trace=call_trace,
            )

        input_tokens  += response.usage.input_tokens
        output_tokens += response.usage.output_tokens

        # Collect assistant content
        assistant_content = []
        tool_use_blocks   = []

        for block in response.content:
            if block.type == "tool_use":
                tool_calls += 1
                tool_use_blocks.append(block)

                # Classify this tool call
                tname = block.name
                call_trace.append(tname)

                is_discovery = tname in ("list_functions", "list_constants", "session_info")
                is_fn_call   = (
                    tname == "call_function"
                    or any(
                        tname == f"{f['name']}_v{f['latest_version']}"
                        for f in db.list_functions()
                    )
                )

                if is_discovery:
                    discovery_calls += 1
                if is_fn_call:
                    function_calls += 1

                if not first_call_seen:
                    first_call_seen  = True
                    first_call_to_fn = is_fn_call

            assistant_content.append(block)

        messages.append({"role": "assistant", "content": assistant_content})

        # End condition: model gave a text-only response
        if response.stop_reason == "end_turn" and not tool_use_blocks:
            final_text = " ".join(
                b.text for b in response.content
                if hasattr(b, "text")
            ).strip()
            elapsed = int((time.monotonic() - start_ms) * 1000)
            ev = task.get("expected_values", [])
            return RunResult(
                task_id=task["id"], mode=mode, turns=turns,
                tool_calls=tool_calls, discovery_calls=discovery_calls,
                function_calls=function_calls,
                input_tokens=input_tokens, output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                first_call_to_function=first_call_to_fn,
                completed=True, final_answer=final_text,
                elapsed_ms=elapsed,
                correct=check_correctness(final_text, ev) if ev else None,
                error=None,
                tool_call_trace=call_trace,
            )

        # Process tool calls and build tool_result blocks
        tool_results = []
        for block in tool_use_blocks:
            try:
                result = execute_tool(block.name, dict(block.input), db, mode)
                result_text = json.dumps(result, default=str)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })
            except Exception as exc:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps({"error": str(exc)}),
                    "is_error": True,
                })

        messages.append({"role": "user", "content": tool_results})

    # Hit MAX_TURNS
    elapsed = int((time.monotonic() - start_ms) * 1000)
    return RunResult(
        task_id=task["id"], mode=mode, turns=turns,
        tool_calls=tool_calls, discovery_calls=discovery_calls,
        function_calls=function_calls,
        input_tokens=input_tokens, output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        first_call_to_function=first_call_to_fn,
        completed=False, final_answer="[MAX_TURNS exceeded]",
        elapsed_ms=elapsed, correct=False, error="MAX_TURNS",
        tool_call_trace=call_trace,
    )


def run_task_ollama(
    task: Dict,
    mode: str,
    tools: List[Dict],
    db: Curry,
    base_url: str,
    model: str,
) -> RunResult:
    """Run a single benchmark task using Ollama's /api/chat endpoint with tool calling.

    Ollama (>= 0.2) exposes an OpenAI-compatible tool-calling API.  The tool
    schema format differs from Anthropic: use _to_ollama_tools() to convert.
    Tool results are sent as role='tool' messages (one per call, in order).
    Token counts come from prompt_eval_count / eval_count in the response body.
    """
    if _requests is None:
        raise RuntimeError("requests package not found. Run: pip install requests")

    system = (
        "You are an agent with access to Curry, a deterministic functional database. "
        "Complete the user's request using the available tools. "
        "Be efficient -- use the minimum number of tool calls necessary. "
        "Return a concise final answer when done."
    )

    ollama_tools = _to_ollama_tools(tools)
    messages: List[Dict] = [{"role": "user", "content": task["prompt"]}]

    turns            = 0
    tool_calls       = 0
    discovery_calls  = 0
    function_calls   = 0
    input_tokens     = 0
    output_tokens    = 0
    first_call_to_fn = False
    first_call_seen  = False
    call_trace: List[str] = []
    start_ms = time.monotonic()

    while turns < MAX_TURNS:
        turns += 1
        try:
            resp = _requests.post(
                f"{base_url.rstrip('/')}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "system", "content": system}] + messages,
                    "tools": ollama_tools,
                    "stream": False,
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            elapsed = int((time.monotonic() - start_ms) * 1000)
            return RunResult(
                task_id=task["id"], mode=mode, turns=turns,
                tool_calls=tool_calls, discovery_calls=discovery_calls,
                function_calls=function_calls,
                input_tokens=input_tokens, output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                first_call_to_function=first_call_to_fn,
                completed=False, final_answer="",
                elapsed_ms=elapsed, correct=False, error=str(exc),
                tool_call_trace=call_trace,
            )

        input_tokens  += data.get("prompt_eval_count", 0)
        output_tokens += data.get("eval_count", 0)

        msg             = data.get("message") or {}
        tool_use_blocks = msg.get("tool_calls") or []
        content_text    = (msg.get("content") or "").strip()

        # Classify tool calls
        for tc in tool_use_blocks:
            tool_calls += 1
            tname = tc["function"]["name"]
            call_trace.append(tname)

            is_discovery = tname in ("list_functions", "list_constants", "session_info")
            is_fn_call   = (
                tname == "call_function"
                or any(
                    tname == f"{f['name']}_v{f['latest_version']}"
                    for f in db.list_functions()
                )
            )

            if is_discovery:
                discovery_calls += 1
            if is_fn_call:
                function_calls += 1

            if not first_call_seen:
                first_call_seen  = True
                first_call_to_fn = is_fn_call

        # Append assistant message verbatim (Ollama needs it for context)
        messages.append(msg)

        # End condition: model returned text with no tool calls
        if not tool_use_blocks:
            elapsed = int((time.monotonic() - start_ms) * 1000)
            ev = task.get("expected_values", [])
            return RunResult(
                task_id=task["id"], mode=mode, turns=turns,
                tool_calls=tool_calls, discovery_calls=discovery_calls,
                function_calls=function_calls,
                input_tokens=input_tokens, output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                first_call_to_function=first_call_to_fn,
                completed=True, final_answer=content_text,
                elapsed_ms=elapsed,
                correct=check_correctness(content_text, ev) if ev else None,
                error=None,
                tool_call_trace=call_trace,
            )

        # Execute each tool call and append a tool-role message for each result
        for tc in tool_use_blocks:
            tname  = tc["function"]["name"]
            tinput = tc["function"].get("arguments") or {}
            try:
                result      = execute_tool(tname, tinput, db, mode)
                result_text = json.dumps(result, default=str)
            except Exception as exc:
                result_text = json.dumps({"error": str(exc)})
            messages.append({"role": "tool", "content": result_text})

    # Hit MAX_TURNS
    elapsed = int((time.monotonic() - start_ms) * 1000)
    return RunResult(
        task_id=task["id"], mode=mode, turns=turns,
        tool_calls=tool_calls, discovery_calls=discovery_calls,
        function_calls=function_calls,
        input_tokens=input_tokens, output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        first_call_to_function=first_call_to_fn,
        completed=False, final_answer="[MAX_TURNS exceeded]",
        elapsed_ms=elapsed, correct=False, error="MAX_TURNS",
        tool_call_trace=call_trace,
    )


def run_task(
    task: Dict,
    mode: str,
    tools: List[Dict],
    db: Curry,
    client,
    model: str,
    provider: str = "anthropic",
    base_url: str = "http://localhost:11434",
) -> RunResult:
    """Dispatch to the correct provider implementation."""
    if provider == "ollama":
        return run_task_ollama(task, mode, tools, db, base_url, model)
    return run_task_anthropic(task, mode, tools, db, client, model)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _avg(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def print_summary(
    results: List[RunResult],
    schema_overheads: Optional[Dict[str, int]] = None,
    model: str = "",
) -> None:
    """
    Print a rigorous comparison table.

    schema_overheads -- {"generic": N, "dynamic": N} from measure_schema_overhead().
                        Pass None to skip that section.
    model            -- used for cost estimation; falls back to PRICING default if unknown.
    """

    def group(mode: str) -> List[RunResult]:
        return [r for r in results if r.mode == mode]

    def _std(vals: List[float]) -> float:
        return statistics.stdev(vals) if len(vals) >= 2 else 0.0

    def _pct(a: float, b: float) -> str:
        return f"({(b / a - 1) * 100:+.1f}%)" if a > 0 else "(n/a)"

    generic  = group("generic")
    dynamic  = group("dynamic")
    n_runs   = max(len({r.task_id for r in generic}), 1)
    multi    = any(
        sum(1 for r in results if r.task_id == tid and r.mode == "generic") > 1
        for tid in {r.task_id for r in results}
    )

    W = 100

    # ── Header ────────────────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("CURRY AGENT BENCHMARK  --  Generic call_function  vs  Dynamic per-function tools")
    if model:
        print(f"Model: {model}  |  Runs per task/mode: {max(1, len(generic) // max(1, n_runs))}")
    print("=" * W)

    # ── Schema overhead ───────────────────────────────────────────────────────
    if schema_overheads and any(v > 0 for v in schema_overheads.values()):
        g_sch = schema_overheads.get("generic", -1)
        d_sch = schema_overheads.get("dynamic", -1)
        print("\nSCHEMA OVERHEAD  (tool definitions + system prompt, zero-history baseline)")
        print("-" * 60)
        if g_sch > 0:
            g_cost = _price(model, g_sch, 0)
            print(f"  generic   : {g_sch:>6,} tokens   (${g_cost * 1000:.4f} per 1k calls)")
        if d_sch > 0:
            d_cost = _price(model, d_sch, 0)
            print(f"  dynamic   : {d_sch:>6,} tokens   (${d_cost * 1000:.4f} per 1k calls)")
        if g_sch > 0 and d_sch > 0:
            delta = d_sch - g_sch
            pct   = (delta / g_sch) * 100
            print(f"  delta     : {delta:>+6,} tokens   ({pct:+.1f}%  larger tool surface in dynamic mode)")
            # Break-even: extra schema cost amortized over token savings per task
            g_avg_tok = _avg([r.total_tokens for r in generic]) if generic else 0
            d_avg_tok = _avg([r.total_tokens for r in dynamic]) if dynamic else 0
            task_savings = g_avg_tok - d_avg_tok
            if task_savings > 0:
                breakeven = delta / task_savings
                print(f"  break-even: schema overhead recovered after {breakeven:.1f} tasks "
                      f"(dynamic saves ~{task_savings:.0f} tok/task on average)")

    # ── Per-task token breakdown ──────────────────────────────────────────────
    print("\nPER-TASK TOKEN BREAKDOWN  (input | output | total | est. cost USD)")
    print("-" * W)
    hdr = (f"{'Task':<24} {'Mode':<9} {'Turns':>5}  "
           f"{'In':>7}  {'Out':>6}  {'Total':>7}  {'Cost$':>8}  "
           f"{'1stFn':>5}  {'Done':>4}  {'Corr':>4}")
    if multi:
        hdr = (f"{'Task':<24} {'Mode':<9} {'Turns':>5}  "
               f"{'In(mean±σ)':>14}  {'Out(mean±σ)':>13}  {'Total':>7}  "
               f"{'Cost$':>8}  {'1stFn':>5}  {'Done':>4}  {'Corr':>4}")
    print(hdr)
    print("-" * W)

    task_ids = sorted({r.task_id for r in results})
    for tid in task_ids:
        for mode in ("generic", "dynamic"):
            runs = [r for r in results if r.task_id == tid and r.mode == mode]
            if not runs:
                continue
            avg_turns = _avg([r.turns for r in runs])
            avg_in    = _avg([r.input_tokens  for r in runs])
            avg_out   = _avg([r.output_tokens for r in runs])
            avg_tot   = _avg([r.total_tokens  for r in runs])
            std_in    = _std([r.input_tokens  for r in runs])
            std_out   = _std([r.output_tokens for r in runs])
            cost      = _price(model, avg_in, avg_out)
            first_fn  = sum(1 for r in runs if r.first_call_to_function) / len(runs)
            done      = sum(1 for r in runs if r.completed) / len(runs)
            graded    = [r for r in runs if r.correct is not None]
            corr      = (sum(1 for r in graded if r.correct) / len(graded)) if graded else None
            corr_col  = f"{corr:>4.0%}" if corr is not None else "  n/a"

            if multi:
                in_col  = f"{avg_in:>7.0f}±{std_in:>5.0f}"
                out_col = f"{avg_out:>6.0f}±{std_out:>5.0f}"
                print(f"{tid:<24} {mode:<9} {avg_turns:>5.1f}  "
                      f"{in_col:>14}  {out_col:>13}  {avg_tot:>7.0f}  "
                      f"${cost:>7.5f}  {first_fn:>4.0%}   {done:>3.0%}  {corr_col}")
            else:
                print(f"{tid:<24} {mode:<9} {avg_turns:>5.1f}  "
                      f"{avg_in:>7.0f}  {avg_out:>6.0f}  {avg_tot:>7.0f}  "
                      f"${cost:>7.5f}  {first_fn:>4.0%}   {done:>3.0%}  {corr_col}")
        print()

    # ── Aggregate comparison ──────────────────────────────────────────────────
    print("=" * W)
    print("AGGREGATE AVERAGES  (all tasks, all runs)")
    print("-" * 60)
    for mode, runs in [("generic", generic), ("dynamic", dynamic)]:
        if not runs:
            continue
        in_vals  = [r.input_tokens  for r in runs]
        out_vals = [r.output_tokens for r in runs]
        tot_vals = [r.total_tokens  for r in runs]
        trn_vals = [r.turns         for r in runs]
        disc_vals = [r.discovery_calls for r in runs]

        avg_in  = _avg(in_vals);  std_in  = _std(in_vals)
        avg_out = _avg(out_vals); std_out = _std(out_vals)
        avg_tot = _avg(tot_vals); std_tot = _std(tot_vals)
        avg_trn = _avg(trn_vals); std_trn = _std(trn_vals)
        avg_cost = _price(model, avg_in, avg_out)

        print(f"\n  {mode.upper()}")
        print(f"    Turns per task       {avg_trn:.2f}" + (f"  ±{std_trn:.2f}" if multi else ""))
        print(f"    Discovery calls      {_avg(disc_vals):.2f}" + (f"  ±{_std(disc_vals):.2f}" if multi else ""))
        print(f"    Input tokens/task    {avg_in:.0f}" + (f"  ±{std_in:.0f}" if multi else ""))
        print(f"    Output tokens/task   {avg_out:.0f}" + (f"  ±{std_out:.0f}" if multi else ""))
        print(f"    Total tokens/task    {avg_tot:.0f}" + (f"  ±{std_tot:.0f}" if multi else ""))
        print(f"    Est. cost/task       ${avg_cost:.6f}")
        if len(runs) > 0:
            rates = PRICING.get(model, {"input": 3.00, "output": 15.00})
            print(f"    Output/Input ratio   {avg_out/avg_in:.3f}  "
                  f"(output costs {rates['output']/rates['input']:.1f}x more per token)")
        print(f"    First call to fn     {sum(1 for r in runs if r.first_call_to_function)/len(runs):.0%}")
        print(f"    Completion rate      {sum(1 for r in runs if r.completed)/len(runs):.0%}")
        graded = [r for r in runs if r.correct is not None]
        if graded:
            print(f"    Correctness rate     {sum(1 for r in graded if r.correct)/len(graded):.0%}  "
                  f"({sum(1 for r in graded if r.correct)}/{len(graded)} tasks graded)")

    # ── Delta ─────────────────────────────────────────────────────────────────
    if generic and dynamic:
        g_in   = _avg([r.input_tokens  for r in generic])
        d_in   = _avg([r.input_tokens  for r in dynamic])
        g_out  = _avg([r.output_tokens for r in generic])
        d_out  = _avg([r.output_tokens for r in dynamic])
        g_tot  = _avg([r.total_tokens  for r in generic])
        d_tot  = _avg([r.total_tokens  for r in dynamic])
        g_trn  = _avg([r.turns         for r in generic])
        d_trn  = _avg([r.turns         for r in dynamic])
        g_disc = _avg([r.discovery_calls for r in generic])
        d_disc = _avg([r.discovery_calls for r in dynamic])
        g_cost = _price(model, g_in, g_out)
        d_cost = _price(model, d_in, d_out)

        print(f"\n{'=' * W}")
        print("DYNAMIC vs GENERIC DELTA")
        print(f"-" * 60)
        print(f"    Turns saved          {g_trn - d_trn:+.2f}  {_pct(g_trn,  d_trn)}")
        print(f"    Input tokens saved   {g_in  - d_in:+.0f}  {_pct(g_in,   d_in)}")
        print(f"    Output tokens saved  {g_out - d_out:+.0f}  {_pct(g_out,  d_out)}")
        print(f"    Total tokens saved   {g_tot - d_tot:+.0f}  {_pct(g_tot,  d_tot)}")
        print(f"    Cost saved/task      ${g_cost - d_cost:+.6f}  {_pct(g_cost, d_cost)}")
        print(f"    Discovery calls saved {g_disc - d_disc:+.2f}")

    print("=" * W)



# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Curry agent benchmark")
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Task IDs to run (default: all). E.g. --tasks single_known chained"
    )
    parser.add_argument(
        "--model", default="claude-haiku-4-5-20251001",
        help="Model name. For anthropic: model string. For ollama: tag pulled with 'ollama pull'. "
             "Default: claude-haiku-4-5-20251001"
    )
    parser.add_argument(
        "--provider", default="anthropic", choices=["anthropic", "ollama"],
        help="LLM provider: 'anthropic' (default) or 'ollama' for local models."
    )
    parser.add_argument(
        "--base-url", dest="base_url", default="http://localhost:11434",
        help="Base URL for the Ollama server (default: http://localhost:11434). "
             "Ignored when provider=anthropic."
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Repetitions per task/mode pair for averaging (default: 1). "
             "Use >= 3 for statistically meaningful std dev."
    )
    parser.add_argument(
        "--output", default="curry_bench_results.json",
        help="Path for raw JSON results (default: curry_bench_results.json)"
    )
    parser.add_argument(
        "--no-schema-check", dest="no_schema_check", action="store_true",
        help="Skip the count_tokens schema overhead measurement."
    )
    args = parser.parse_args()

    # ── Provider setup ────────────────────────────────────────────────────────
    if args.provider == "anthropic":
        if anthropic is None:
            print("anthropic package not found. Run: pip install anthropic", file=sys.stderr)
            sys.exit(1)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ANTHROPIC_API_KEY not set.", file=sys.stderr)
            sys.exit(1)
        client = anthropic.Anthropic(api_key=api_key)
    else:
        if _requests is None:
            print("requests package not found. Run: pip install requests", file=sys.stderr)
            sys.exit(1)
        client = None  # Ollama uses direct HTTP; no client object needed

    db = build_fixture()

    system = (
        "You are an agent with access to Curry, a deterministic functional database. "
        "Complete the user's request using the available tools. "
        "Be efficient -- use the minimum number of tool calls necessary. "
        "Return a concise final answer when done."
    )

    generic_tools = build_generic_tools()
    dynamic_tools = build_dynamic_tools(db)

    # ── Schema overhead measurement (Anthropic only) ──────────────────────────
    schema_overheads: Dict[str, int] = {}
    if not args.no_schema_check and args.provider == "anthropic":
        print("\nMeasuring schema overhead via count_tokens (no inference cost)...", end="", flush=True)
        schema_overheads["generic"] = measure_schema_overhead(client, generic_tools, args.model, system)
        schema_overheads["dynamic"] = measure_schema_overhead(client, dynamic_tools, args.model, system)
        if all(v > 0 for v in schema_overheads.values()):
            print(
                f"  generic={schema_overheads['generic']:,}  "
                f"dynamic={schema_overheads['dynamic']:,}  "
                f"delta={schema_overheads['dynamic'] - schema_overheads['generic']:+,}"
            )
        else:
            print("  (count_tokens not supported for this model, skipping)")
            schema_overheads = {}
    elif args.provider == "ollama":
        print("  (schema overhead measurement skipped -- not supported for local models)")

    task_ids = args.tasks or [t["id"] for t in TASKS]
    tasks    = [TASK_MAP[tid] for tid in task_ids if tid in TASK_MAP]
    if not tasks:
        print(f"No valid task IDs. Available: {list(TASK_MAP.keys())}", file=sys.stderr)
        sys.exit(1)

    print(f"\nBenchmark: {args.model}  |  provider={args.provider}  |  "
          f"{len(tasks)} tasks  |  {args.runs} run(s) each  |  2 modes")
    if args.runs == 1:
        print("  Tip: use --runs 3 (or more) for std dev and statistical confidence.")
    if args.provider == "ollama":
        print(f"  Ollama base URL: {args.base_url}")
        print("  Token costs will show $0.000000 (local model, no API cost).")
        print("  Correctness rate is the primary quality metric for local models.")
    print(f"Generic tools  : {len(generic_tools)}")
    print(f"Dynamic tools  : {len(dynamic_tools)}  (includes {len(dynamic_tools) - 3} per-function tools)")
    print()

    all_results: List[RunResult] = []
    total_runs = len(tasks) * 2 * args.runs
    run_idx    = 0

    for task in tasks:
        for mode, tools in [("generic", generic_tools), ("dynamic", dynamic_tools)]:
            for run_n in range(args.runs):
                run_idx += 1
                print(f"  [{run_idx:>3}/{total_runs}] task={task['id']}  mode={mode}  run={run_n+1}", end="", flush=True)
                result = run_task(
                    task, mode, tools, db, client, args.model,
                    provider=args.provider, base_url=args.base_url,
                )
                all_results.append(result)
                status = "OK" if result.completed else f"FAIL({result.error})"
                corr_tag = f"  corr={'Y' if result.correct else ('N' if result.correct is False else '?')}"
                print(
                    f"  turns={result.turns}  tools={result.tool_calls}  "
                    f"in={result.input_tokens}  out={result.output_tokens}  {status}{corr_tag}"
                )

    # Save raw results
    raw = [asdict(r) for r in all_results]
    with open(args.output, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\nRaw results saved to: {args.output}")

    print_summary(all_results, schema_overheads=schema_overheads or None, model=args.model)


if __name__ == "__main__":
    main()

