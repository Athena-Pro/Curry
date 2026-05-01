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
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

try:
    import anthropic
except ImportError:
    print("anthropic package not found. Run: pip install anthropic", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, os.path.dirname(__file__))
from curry_core import Curry


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: in-memory Curry DB with benchmark functions
# ─────────────────────────────────────────────────────────────────────────────

def build_fixture() -> Curry:
    """
    Create an in-memory Curry DB with a small but realistic set of functions.
    Covers single-arg, multi-arg, chained, and composed cases.
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
    )

    # compute_tax(amount) -> tax portion on an amount
    db.declare_function(
        name="compute_tax",
        version=1,
        body="round(amount * tax_rate, 2)",
        constant_bindings={"tax_rate": 1},
        expected_args=["amount"],
        is_pure=True,
    )

    # final_price(amount) -> discounted amount + tax
    db.declare_function(
        name="final_price",
        version=1,
        body="round(amount * (1 - discount_rate) * (1 + tax_rate), 2)",
        constant_bindings={"discount_rate": 1, "tax_rate": 1},
        expected_args=["amount"],
        is_pure=True,
    )

    # apply_markup(cost) -> cost with markup
    db.declare_function(
        name="apply_markup",
        version=1,
        body="round(cost * (1 + markup_rate), 2)",
        constant_bindings={"markup_rate": 1},
        expected_args=["cost"],
        is_pure=True,
    )

    # meets_minimum(order_total) -> bool: does order meet minimum?
    db.declare_function(
        name="meets_minimum",
        version=1,
        body="order_total >= min_order",
        constant_bindings={"min_order": 1},
        expected_args=["order_total"],
        is_pure=True,
    )

    # compound_interest(principal, rate, years) -> final value
    db.declare_function(
        name="compound_interest",
        version=1,
        body="round(principal * (1 + rate) ** years, 2)",
        constant_bindings={},
        expected_args=["principal", "rate", "years"],
        is_pure=True,
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
    derived from expected_args. No discovery needed -- the agent can call
    apply_discount_v1(amount=150) directly.
    """
    tools = _base_tools()
    for func in db.list_functions():
        name    = func["name"]
        version = func["latest_version"]
        args    = func.get("expected_args") or []

        properties = {
            arg: {
                "type": "number",
                "description": f"Value for argument '{arg}'",
            }
            for arg in args
        }

        tools.append({
            "name": f"{name}_v{version}",
            "description": (
                f"Call the '{name}' function (version {version}). "
                f"Arguments: {', '.join(args) if args else 'none'}."
            ),
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
# Benchmark task definitions
# ─────────────────────────────────────────────────────────────────────────────

TASKS = [
    {
        "id": "single_known",
        "prompt": (
            "Calculate the discounted price for an item that costs $250. "
            "Use the apply_discount function and return only the final dollar amount."
        ),
        "hint": "Direct single-function call. Generic must discover args first.",
    },
    {
        "id": "single_multiarg",
        "prompt": (
            "What is the compound interest final value on a $1000 principal "
            "at 6% annual rate over 5 years? Return the numeric result only."
        ),
        "hint": "Multi-arg function. Dynamic tool schema makes args unambiguous.",
    },
    {
        "id": "sequential_two",
        "prompt": (
            "An item costs $300. First apply the discount, then compute the tax "
            "on the discounted price. Return both intermediate values and the tax amount."
        ),
        "hint": "Two sequential function calls. Tests chaining efficiency.",
    },
    {
        "id": "composed_final",
        "prompt": (
            "What is the final price (after both discount and tax) for an item "
            "originally priced at $450? Use the most direct function available."
        ),
        "hint": "final_price does it in one call. Does agent find it directly?",
    },
    {
        "id": "multi_values",
        "prompt": (
            "Calculate the discounted price for three items: $100, $200, and $500. "
            "Return a list of all three discounted amounts."
        ),
        "hint": "Repeated single-function calls. Tests whether agent batches or loops.",
    },
    {
        "id": "discovery_ambiguous",
        "prompt": (
            "I have a product that costs $80 to make. What's the selling price "
            "after applying the standard markup? Also tell me if a $40 order "
            "meets the minimum order requirement."
        ),
        "hint": "Two different functions with non-obvious names. Discovery value is high.",
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
    error: Optional[str] = None
    tool_call_trace: List[str] = field(default_factory=list)


def run_task(
    task: Dict,
    mode: str,
    tools: List[Dict],
    db: Curry,
    client: anthropic.Anthropic,
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
                elapsed_ms=elapsed, error=str(exc),
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
            return RunResult(
                task_id=task["id"], mode=mode, turns=turns,
                tool_calls=tool_calls, discovery_calls=discovery_calls,
                function_calls=function_calls,
                input_tokens=input_tokens, output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                first_call_to_function=first_call_to_fn,
                completed=True, final_answer=final_text,
                elapsed_ms=elapsed, error=None,
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
        elapsed_ms=elapsed, error="MAX_TURNS",
        tool_call_trace=call_trace,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _avg(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def print_summary(results: List[RunResult]) -> None:
    """Print a compact comparison table to stdout."""

    def group(mode: str) -> List[RunResult]:
        return [r for r in results if r.mode == mode]

    generic = group("generic")
    dynamic = group("dynamic")

    # Per-task table
    print("\n" + "=" * 90)
    print("CURRY AGENT BENCHMARK  --  Generic call_function  vs  Dynamic per-function tools")
    print("=" * 90)

    header = f"{'Task':<24} {'Mode':<9} {'Turns':>5} {'ToolCalls':>9} {'Disc':>5} {'FnCalls':>8} {'Tokens':>7} {'1stFn':>5} {'Done':>5}"
    print(header)
    print("-" * 90)

    task_ids = sorted({r.task_id for r in results})
    for tid in task_ids:
        for mode in ("generic", "dynamic"):
            runs = [r for r in results if r.task_id == tid and r.mode == mode]
            if not runs:
                continue
            avg_turns     = _avg([r.turns for r in runs])
            avg_tc        = _avg([r.tool_calls for r in runs])
            avg_disc      = _avg([r.discovery_calls for r in runs])
            avg_fn        = _avg([r.function_calls for r in runs])
            avg_tok       = _avg([r.total_tokens for r in runs])
            first_fn_rate = sum(1 for r in runs if r.first_call_to_function) / len(runs)
            done_rate     = sum(1 for r in runs if r.completed) / len(runs)
            print(
                f"{tid:<24} {mode:<9} {avg_turns:>5.1f} {avg_tc:>9.1f} "
                f"{avg_disc:>5.1f} {avg_fn:>8.1f} {avg_tok:>7.0f} "
                f"{first_fn_rate:>4.0%}  {done_rate:>4.0%}"
            )
        print()

    # Aggregate comparison
    print("=" * 90)
    print("AGGREGATE AVERAGES")
    print("-" * 50)
    for mode, runs in [("generic", generic), ("dynamic", dynamic)]:
        if not runs:
            continue
        print(f"  {mode.upper()}")
        print(f"    Turns per task       {_avg([r.turns for r in runs]):.2f}")
        print(f"    Tool calls per task  {_avg([r.tool_calls for r in runs]):.2f}")
        print(f"    Discovery calls      {_avg([r.discovery_calls for r in runs]):.2f}")
        print(f"    Function calls       {_avg([r.function_calls for r in runs]):.2f}")
        print(f"    Total tokens         {_avg([r.total_tokens for r in runs]):.0f}")
        print(f"    First call to fn     {sum(1 for r in runs if r.first_call_to_function)/len(runs):.0%}")
        print(f"    Completion rate      {sum(1 for r in runs if r.completed)/len(runs):.0%}")
        print()

    # Delta row
    if generic and dynamic:
        g_turns  = _avg([r.turns        for r in generic])
        d_turns  = _avg([r.turns        for r in dynamic])
        g_tokens = _avg([r.total_tokens for r in generic])
        d_tokens = _avg([r.total_tokens for r in dynamic])
        g_disc   = _avg([r.discovery_calls for r in generic])
        d_disc   = _avg([r.discovery_calls for r in dynamic])
        turns_pct  = f"({(d_turns/g_turns - 1)*100:+.1f}%)"  if g_turns  > 0 else "(n/a)"
        tokens_pct = f"({(d_tokens/g_tokens - 1)*100:+.1f}%)" if g_tokens > 0 else "(n/a)"
        print("DYNAMIC vs GENERIC DELTA")
        print(f"    Turns saved          {g_turns - d_turns:+.2f}  {turns_pct}")
        print(f"    Tokens saved         {g_tokens - d_tokens:+.0f}  {tokens_pct}")
        print(f"    Discovery calls saved {g_disc - d_disc:+.2f}")
    print("=" * 90)


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
        help="Anthropic model string (default: claude-haiku-4-5-20251001)"
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Repetitions per task/mode pair for averaging (default: 1)"
    )
    parser.add_argument(
        "--output", default="curry_bench_results.json",
        help="Path for raw JSON results (default: curry_bench_results.json)"
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    db     = build_fixture()

    generic_tools = build_generic_tools()
    dynamic_tools = build_dynamic_tools(db)

    task_ids = args.tasks or [t["id"] for t in TASKS]
    tasks    = [TASK_MAP[tid] for tid in task_ids if tid in TASK_MAP]
    if not tasks:
        print(f"No valid task IDs. Available: {list(TASK_MAP.keys())}", file=sys.stderr)
        sys.exit(1)

    print(f"\nBenchmark: {args.model}  |  {len(tasks)} tasks  |  {args.runs} run(s) each  |  2 modes")
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
                result = run_task(task, mode, tools, db, client, args.model)
                all_results.append(result)
                status = "OK" if result.completed else f"FAIL({result.error})"
                print(
                    f"  turns={result.turns}  tools={result.tool_calls}  "
                    f"disc={result.discovery_calls}  tokens={result.total_tokens}  {status}"
                )

    # Save raw results
    raw = [asdict(r) for r in all_results]
    with open(args.output, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\nRaw results saved to: {args.output}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
