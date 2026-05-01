"""
curry_mcp_server.py
-------------------
MCP server for Curry: a functional database for LLM operations.

Startup (stdio transport, standard MCP convention):
    python curry_mcp_server.py --project <project-dir>

The server opens one CurrySession against <project-dir>/.curry/config.json
at startup and keeps it alive for the entire session. It does NOT open new
sessions per tool call.

Core DB is read-only within a session by design. register_model and
retire_model are NOT exposed as MCP tools -- use a direct Curry admin
connection for those operations.

Tool names are prefixed with config["mcp_tool_prefix"] so multiple project
servers can coexist in the same Claude context without collision.
"""

import argparse
import json
import sys
import traceback
from typing import Any, Dict, List

# ── MCP SDK ──────────────────────────────────────────────────────────────────
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types
except ImportError:
    print(
        "mcp package not found. Install with: pip install mcp",
        file=sys.stderr,
    )
    sys.exit(1)

# ── Curry ─────────────────────────────────────────────────────────────────────
try:
    from curry_core import CurrySession
except ImportError:
    print(
        "curry_core not found. Run this script from C:\\AI-Local\\Curry\\",
        file=sys.stderr,
    )
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Error helpers
# ─────────────────────────────────────────────────────────────────────────────

def _curry_error(exc: Exception) -> types.TextContent:
    """Format a Curry exception as a structured MCP error text block."""
    return types.TextContent(
        type="text",
        text=json.dumps(
            {
                "error": type(exc).__name__,
                "message": str(exc),
            },
            indent=2,
        ),
    )


def _ok(payload: Any) -> List[types.TextContent]:
    """Wrap a successful result as a JSON text block."""
    return [
        types.TextContent(
            type="text",
            text=json.dumps(payload, indent=2, default=str),
        )
    ]


def _err(exc: Exception) -> List[types.TextContent]:
    """Wrap a Curry exception as a JSON error block."""
    return [_curry_error(exc)]


# ─────────────────────────────────────────────────────────────────────────────
# Tool schema helpers
# ─────────────────────────────────────────────────────────────────────────────

def _str_prop(description: str) -> Dict:
    return {"type": "string", "description": description}


def _int_prop(description: str) -> Dict:
    return {"type": "integer", "description": description}


def _num_prop(description: str) -> Dict:
    return {"type": "number", "description": description}


def _bool_prop(description: str) -> Dict:
    return {"type": "boolean", "description": description}


def _obj_prop(description: str) -> Dict:
    return {"type": "object", "description": description}


# ─────────────────────────────────────────────────────────────────────────────
# Server factory
# ─────────────────────────────────────────────────────────────────────────────

def build_server(session: CurrySession) -> Server:
    """
    Construct the MCP Server instance and register all Curry tools against it.
    Tool names are namespaced by config["mcp_tool_prefix"].
    """
    prefix = session.config.get("mcp_tool_prefix", "curry")
    server = Server(f"{prefix}-server")

    # ── Tool name builder ────────────────────────────────────────────────────
    def t(name: str) -> str:
        return f"{prefix}_{name}"

    # ── Tool listing ─────────────────────────────────────────────────────────
    @server.list_tools()
    async def list_tools() -> List[types.Tool]:
        static_tools = [

            # ── Constants ────────────────────────────────────────────────────
            types.Tool(
                name=t("declare_constant"),
                description=(
                    "Declare a new version of a typed constant in the project database. "
                    "Type must be one of: Float64, Int32, String, Blob, Json, Tokens, Currency, Bool. "
                    "Version must be strictly greater than any existing version for this constant ID."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "const_id": _str_prop("Unique constant identifier"),
                        "version": _int_prop("Version number (must exceed current max)"),
                        "value": {"description": "The constant value (must match type_signature)"},
                        "type_signature": _str_prop("One of: Float64, Int32, String, Blob, Json, Tokens, Currency, Bool"),
                    },
                    "required": ["const_id", "version", "value", "type_signature"],
                },
            ),

            types.Tool(
                name=t("get_constant"),
                description="Retrieve a constant by exact ID and version.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "const_id": _str_prop("Constant identifier"),
                        "version": _int_prop("Exact version to retrieve"),
                    },
                    "required": ["const_id", "version"],
                },
            ),

            types.Tool(
                name=t("get_constant_latest"),
                description="Retrieve the most recent active version of a constant.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "const_id": _str_prop("Constant identifier"),
                    },
                    "required": ["const_id"],
                },
            ),

            types.Tool(
                name=t("list_constants"),
                description=(
                    "List all constants with their latest versions. "
                    "Returns merged results from both the project DB and the global core DB."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "active_only": _bool_prop("If true (default), exclude retired constants"),
                    },
                    "required": [],
                },
            ),

            types.Tool(
                name=t("retire_constant"),
                description="Mark a specific constant version as retired.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "const_id": _str_prop("Constant identifier"),
                        "version": _int_prop("Version to retire"),
                        "retirement_tag": _str_prop("Optional retirement tag ID"),
                    },
                    "required": ["const_id", "version"],
                },
            ),

            # ── Functions ─────────────────────────────────────────────────────
            types.Tool(
                name=t("declare_function"),
                description=(
                    "Declare a versioned function with a Python expression body. "
                    "The body is statically validated via AST at declaration time. "
                    "Provide expected_args to enable strict symbol validation and auto-generate "
                    "MCP input schemas for call_function. "
                    "Provide description (human-readable summary including which constants are bound) "
                    "and arg_descriptions (per-arg unit hints for non-obvious arguments) so that "
                    "dynamic per-function tools are fully self-documenting."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": _str_prop("Function name"),
                        "version": _int_prop("Version number (must exceed current max)"),
                        "body": _str_prop("Python expression string (single expression, no statements)"),
                        "constant_bindings": _obj_prop(
                            'Dict mapping constant ID to version, e.g. {"rate": 1}'
                        ),
                        "function_bindings": _obj_prop(
                            'Dict mapping function name to version, e.g. {"helper": 2}'
                        ),
                        "is_pure": _bool_prop("If true, results are memoized in the execution cache"),
                        "expected_args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Runtime argument names the caller must supply to call_function",
                        },
                        "description": _str_prop(
                            "Human-readable summary of what the function does and which constants "
                            "it binds to. Used as the MCP tool description for the dynamic tool. "
                            "Example: 'Apply the standard markup (markup_rate constant) to a wholesale cost.'"
                        ),
                        "arg_descriptions": _obj_prop(
                            "Per-argument hint strings. Non-obvious args (rates, proportions, enums) "
                            "MUST include a unit hint and example value. "
                            "Example: {\"rate\": \"Annual rate as a decimal fraction (e.g. 0.06 for 6%)\", "
                            "\"years\": \"Duration in whole years (e.g. 5)\"}"
                        ),
                    },
                    "required": ["name", "version", "body"],
                },
            ),

            types.Tool(
                name=t("get_function"),
                description="Retrieve a function definition by exact name and version.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": _str_prop("Function name"),
                        "version": _int_prop("Exact version to retrieve"),
                    },
                    "required": ["name", "version"],
                },
            ),

            types.Tool(
                name=t("list_functions"),
                description=(
                    "List all functions with their latest versions. "
                    "Includes expected_args so callers can build call_function invocations. "
                    "Returns merged results from project DB and global core DB."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "active_only": _bool_prop("If true (default), exclude retired functions"),
                    },
                    "required": [],
                },
            ),

            types.Tool(
                name=t("call_function"),
                description=(
                    "Execute a versioned function with the provided runtime arguments. "
                    "Locked constant and function dependencies are resolved automatically. "
                    "Pure functions use the execution cache; cache misses are transparent. "
                    "Use list_functions to discover expected_args before calling."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": _str_prop("Function name"),
                        "version": _int_prop("Exact version to execute"),
                        "args": _obj_prop("Runtime arguments as a flat dict, e.g. {\"amount\": 100}"),
                    },
                    "required": ["name", "version", "args"],
                },
            ),

            types.Tool(
                name=t("get_function_lineage"),
                description=(
                    "Return the full dependency tree for a function version: "
                    "all constants and nested functions it depends on, recursively."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": _str_prop("Function name"),
                        "version": _int_prop("Exact version"),
                    },
                    "required": ["name", "version"],
                },
            ),

            types.Tool(
                name=t("retire_function"),
                description="Mark a specific function version as retired.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": _str_prop("Function name"),
                        "version": _int_prop("Version to retire"),
                        "retirement_tag": _str_prop("Optional retirement tag ID"),
                    },
                    "required": ["name", "version"],
                },
            ),

            # ── Models ────────────────────────────────────────────────────────
            types.Tool(
                name=t("get_model"),
                description=(
                    "Retrieve a model version and its locked inference parameters "
                    "(temperature, top_p, max_tokens) from the global core DB."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_name": _str_prop("Model name"),
                        "version": _int_prop("Exact version"),
                    },
                    "required": ["model_name", "version"],
                },
            ),

            types.Tool(
                name=t("get_model_latest"),
                description="Retrieve the most recent active version of a model from the global core DB.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_name": _str_prop("Model name"),
                    },
                    "required": ["model_name"],
                },
            ),

            types.Tool(
                name=t("list_models"),
                description="List all models registered in the global core DB.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "active_only": _bool_prop("If true (default), exclude retired models"),
                    },
                    "required": [],
                },
            ),

            # ── Inferences ────────────────────────────────────────────────────
            types.Tool(
                name=t("record_inference"),
                description=(
                    "Record a completed LLM inference with full provenance. "
                    "Returns inference_id -- the durable handle for this record. "
                    "Always use inference_id to reference results, not raw output text."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_name": _str_prop("Model name (must be registered in core DB)"),
                        "model_version": _int_prop("Model version"),
                        "input_tokens": _str_prop("Prompt text or structured token reference"),
                        "output_text": _str_prop("Model output text (encoded to bytes internally)"),
                        "seed": _int_prop("RNG seed used for inference (default 42)"),
                        "temperature_used": _num_prop("Actual temperature used (defaults to model's locked value)"),
                        "top_p_used": _num_prop("Actual top_p used (defaults to model's locked value)"),
                        "duration_ms": _int_prop("Wall-clock inference duration in milliseconds"),
                        "metadata": _obj_prop("Additional metadata dict (must be JSON-serializable)"),
                    },
                    "required": ["model_name", "model_version", "input_tokens", "output_text"],
                },
            ),

            types.Tool(
                name=t("get_inference"),
                description="Retrieve a full inference record by inference_id.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "inference_id": _str_prop("UUID returned by record_inference"),
                    },
                    "required": ["inference_id"],
                },
            ),

            types.Tool(
                name=t("search_inferences"),
                description=(
                    "Search inference records with optional filters: model, version, seed, "
                    "timestamp range, metadata key-value pairs, and token count bounds. "
                    "Results are ordered deterministically by (execution_timestamp, inference_id). "
                    "Note: token count filters are applied in Python after SQL fetch -- "
                    "avoid on tables larger than ~10k rows without timestamp/model filters."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_name": _str_prop("Filter by model name"),
                        "model_version": _int_prop("Filter by model version"),
                        "seed": _int_prop("Filter by seed"),
                        "start_timestamp": _str_prop("ISO timestamp lower bound (inclusive)"),
                        "end_timestamp": _str_prop("ISO timestamp upper bound (inclusive)"),
                        "metadata_filters": _obj_prop("Dict of metadata key-value pairs to match exactly"),
                        "min_input_tokens_count": _int_prop("Minimum input token count (from metadata)"),
                        "max_input_tokens_count": _int_prop("Maximum input token count (from metadata)"),
                        "min_output_tokens_count": _int_prop("Minimum output token count (from metadata)"),
                        "max_output_tokens_count": _int_prop("Maximum output token count (from metadata)"),
                        "limit": _int_prop("Maximum results to return (default 50)"),
                        "offset": _int_prop("Pagination offset (default 0)"),
                    },
                    "required": [],
                },
            ),

            types.Tool(
                name=t("compare_inferences"),
                description=(
                    "Compare two inference records. Returns a structured diff: "
                    "same model, same seed, same input, output SHA-256 hashes, "
                    "parameter deltas, and metadata key differences."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "a_id": _str_prop("First inference_id"),
                        "b_id": _str_prop("Second inference_id"),
                    },
                    "required": ["a_id", "b_id"],
                },
            ),

            # ── Retirement tags ───────────────────────────────────────────────
            types.Tool(
                name=t("create_retirement_tag"),
                description=(
                    "Create a retirement tag to group related retirements under a single label. "
                    "Applies to the project DB only."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tag_id": _str_prop("Unique tag identifier"),
                        "reason": _str_prop("Short reason string"),
                        "description": _str_prop("Optional longer description"),
                    },
                    "required": ["tag_id", "reason"],
                },
            ),

            # ── Cache ─────────────────────────────────────────────────────────
            types.Tool(
                name=t("evict_execution_cache"),
                description=(
                    "Evict the execution cache down to max_entries, "
                    "keeping the most recently cached entries. "
                    "Returns the number of rows deleted."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "max_entries": _int_prop("Maximum cache rows to retain (default 1000)"),
                    },
                    "required": [],
                },
            ),

            # ── Introspection ─────────────────────────────────────────────────
            types.Tool(
                name=t("session_info"),
                description=(
                    "Return the current session configuration: project name, "
                    "core DB path, local DB path, default model, and tool prefix."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),

            types.Tool(
                name=t("integrity_check"),
                description=(
                    "Run PRAGMA integrity_check on both the local project DB and the core DB. "
                    "Returns 'ok' for each on success, or the error string on failure."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
        ]

        # ── Dynamic per-function tools (Lag Pattern 2 fix) ─────────────────────
        # Each Curry function declared with expected_args gets its own named
        # tool so agents can call directly without a discovery round trip.
        # arg_descriptions provides per-argument unit hints; without them the
        # tool falls back to a generic 'Value for argument X' description which
        # is insufficient for non-obvious args like rates or proportions.
        try:
            for func in session.list_functions():
                fn_name     = func["name"]
                fn_ver      = func["latest_version"]
                fn_args     = func.get("expected_args") or []
                fn_desc     = func.get("description") or ""
                fn_arg_desc = func.get("arg_descriptions") or {}

                tool_desc = (
                    f"Call Curry function '{fn_name}' (v{fn_ver})."
                    + (f" {fn_desc}" if fn_desc else "")
                    + (f" Args: {', '.join(fn_args)}." if fn_args else " No args.")
                )

                properties = {}
                for arg in fn_args:
                    hint = fn_arg_desc.get(arg) or f"Value for argument '{arg}'"
                    properties[arg] = {"type": "number", "description": hint}

                static_tools.append(
                    types.Tool(
                        name=t(f"fn_{fn_name}_v{fn_ver}"),
                        description=tool_desc,
                        inputSchema={
                            "type": "object",
                            "properties": properties,
                            "required": fn_args,
                        },
                    )
                )
        except Exception as dyn_exc:
            print(
                f"[curry-mcp] Warning: failed to build dynamic tools: {dyn_exc}",
                file=sys.stderr,
            )

        return static_tools

    # ── Tool dispatch ─────────────────────────────────────────────────────────
    @server.call_tool()
    async def call_tool(
        name: str, arguments: Dict[str, Any]
    ) -> List[types.TextContent]:

        # Strip the prefix to get the bare op name
        bare = name[len(prefix) + 1:] if name.startswith(prefix + "_") else name

        try:
            # ── Constants ────────────────────────────────────────────────────
            if bare == "declare_constant":
                session.declare_constant(
                    const_id=arguments["const_id"],
                    version=arguments["version"],
                    value=arguments["value"],
                    type_signature=arguments["type_signature"],
                )
                return _ok({"status": "declared", "const_id": arguments["const_id"], "version": arguments["version"]})

            if bare == "get_constant":
                return _ok(session.get_constant(arguments["const_id"], arguments["version"]))

            if bare == "get_constant_latest":
                return _ok(session.get_constant_latest(arguments["const_id"]))

            if bare == "list_constants":
                return _ok(session.list_constants(active_only=arguments.get("active_only", True)))

            if bare == "retire_constant":
                session.retire_constant(
                    arguments["const_id"],
                    arguments["version"],
                    retirement_tag=arguments.get("retirement_tag"),
                )
                return _ok({"status": "retired", "const_id": arguments["const_id"], "version": arguments["version"]})

            # ── Functions ─────────────────────────────────────────────────────
            if bare == "declare_function":
                session.declare_function(
                    name=arguments["name"],
                    version=arguments["version"],
                    body=arguments["body"],
                    constant_bindings=arguments.get("constant_bindings"),
                    function_bindings=arguments.get("function_bindings"),
                    is_pure=arguments.get("is_pure", False),
                    expected_args=arguments.get("expected_args"),
                    description=arguments.get("description"),
                    arg_descriptions=arguments.get("arg_descriptions"),
                )
                return _ok({"status": "declared", "name": arguments["name"], "version": arguments["version"]})

            if bare == "get_function":
                return _ok(session.get_function(arguments["name"], arguments["version"]))

            if bare == "list_functions":
                return _ok(session.list_functions(active_only=arguments.get("active_only", True)))

            if bare == "call_function":
                result = session.call_function(
                    name=arguments["name"],
                    version=arguments["version"],
                    args=arguments.get("args", {}),
                )
                return _ok({"result": result})

            if bare == "get_function_lineage":
                return _ok(session.get_function_lineage(arguments["name"], arguments["version"]))

            if bare == "retire_function":
                session.retire_function(
                    arguments["name"],
                    arguments["version"],
                    retirement_tag=arguments.get("retirement_tag"),
                )
                return _ok({"status": "retired", "name": arguments["name"], "version": arguments["version"]})

            # ── Models ────────────────────────────────────────────────────────
            if bare == "get_model":
                return _ok(session.get_model(arguments["model_name"], arguments["version"]))

            if bare == "get_model_latest":
                return _ok(session.get_model_latest(arguments["model_name"]))

            if bare == "list_models":
                return _ok(session.list_models(active_only=arguments.get("active_only", True)))

            # ── Inferences ────────────────────────────────────────────────────
            if bare == "record_inference":
                output_bytes = arguments["output_text"].encode("utf-8")
                inference_id = session.record_inference(
                    model_name=arguments["model_name"],
                    model_version=arguments["model_version"],
                    input_tokens=arguments["input_tokens"],
                    output_tokens=output_bytes,
                    seed=arguments.get("seed", 42),
                    temperature_used=arguments.get("temperature_used"),
                    top_p_used=arguments.get("top_p_used"),
                    duration_ms=arguments.get("duration_ms"),
                    metadata=arguments.get("metadata"),
                )
                return _ok({"inference_id": inference_id})

            if bare == "get_inference":
                row = session.get_inference(arguments["inference_id"])
                # output_tokens is bytes -- decode for MCP transport
                if isinstance(row.get("output_tokens"), (bytes, bytearray)):
                    row["output_tokens"] = row["output_tokens"].decode("utf-8", errors="replace")
                return _ok(row)

            if bare == "search_inferences":
                rows = session.search_inferences(
                    model_name=arguments.get("model_name"),
                    model_version=arguments.get("model_version"),
                    seed=arguments.get("seed"),
                    start_timestamp=arguments.get("start_timestamp"),
                    end_timestamp=arguments.get("end_timestamp"),
                    metadata_filters=arguments.get("metadata_filters"),
                    min_input_tokens_count=arguments.get("min_input_tokens_count"),
                    max_input_tokens_count=arguments.get("max_input_tokens_count"),
                    min_output_tokens_count=arguments.get("min_output_tokens_count"),
                    max_output_tokens_count=arguments.get("max_output_tokens_count"),
                    limit=arguments.get("limit", 50),
                    offset=arguments.get("offset", 0),
                )
                # Decode output_tokens for each row
                for row in rows:
                    if isinstance(row.get("output_tokens"), (bytes, bytearray)):
                        row["output_tokens"] = row["output_tokens"].decode("utf-8", errors="replace")
                return _ok(rows)

            if bare == "compare_inferences":
                return _ok(session.compare_inferences(arguments["a_id"], arguments["b_id"]))

            # ── Retirement tags ───────────────────────────────────────────────
            if bare == "create_retirement_tag":
                tag_id = session.local_db.create_retirement_tag(
                    tag_id=arguments["tag_id"],
                    reason=arguments["reason"],
                    description=arguments.get("description"),
                )
                return _ok({"status": "created", "tag_id": tag_id})

            # ── Cache ─────────────────────────────────────────────────────────
            if bare == "evict_execution_cache":
                deleted = session.local_db.evict_execution_cache(
                    max_entries=arguments.get("max_entries", 1000)
                )
                return _ok({"status": "evicted", "rows_deleted": deleted})

            # ── Introspection ─────────────────────────────────────────────────
            if bare == "session_info":
                return _ok(
                    {
                        "project": session.config.get("project"),
                        "core_db": session.core_db.db_path,
                        "local_db": session.local_db.db_path,
                        "default_model": session.config.get("default_model"),
                        "default_model_version": session.config.get("default_model_version"),
                        "mcp_tool_prefix": session.config.get("mcp_tool_prefix"),
                    }
                )

            if bare == "integrity_check":
                return _ok(
                    {
                        "local_db": session.local_db.integrity_check(),
                        "core_db": session.core_db.integrity_check(),
                    }
                )

            # ── Dynamic per-function dispatch ────────────────────────────────
            # Tool name format: {prefix}_fn_{func_name}_v{version}
            if bare.startswith("fn_"):
                inner = bare[3:]  # strip "fn_"
                if "_v" in inner:
                    fn_name, fn_ver_str = inner.rsplit("_v", 1)
                    try:
                        fn_ver = int(fn_ver_str)
                        result = session.call_function(fn_name, fn_ver, arguments)
                        return _ok({"function": f"{fn_name}@v{fn_ver}", "result": result})
                    except (ValueError, KeyError, RuntimeError) as exc:
                        return _err(exc)

            return _err(ValueError(f"Unknown tool: {name}"))

        except (KeyError, ValueError, TypeError, PermissionError, RuntimeError) as exc:
            return _err(exc)
        except Exception as exc:
            # Unexpected errors: include traceback in debug field
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": type(exc).__name__,
                            "message": str(exc),
                            "debug": traceback.format_exc(),
                        },
                        indent=2,
                    ),
                )
            ]

    return server


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main(project_dir: str) -> None:
    with CurrySession.from_project(project_dir) as session:
        prefix = session.config.get("mcp_tool_prefix", "curry")
        print(
            f"[curry-mcp] Starting server for project '{session.config.get('project')}' "
            f"with prefix '{prefix}'",
            file=sys.stderr,
        )
        server = build_server(session)
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )


if __name__ == "__main__":
    import asyncio

    parser = argparse.ArgumentParser(description="Curry MCP Server")
    parser.add_argument(
        "--project",
        required=True,
        help="Path to the project directory containing .curry/config.json",
    )
    args = parser.parse_args()

    asyncio.run(main(args.project))
