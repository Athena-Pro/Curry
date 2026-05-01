"""
Curry: A Functional Database for LLM Operations
Core implementation with SQLite backend, type safety, and deterministic execution.
"""

import sqlite3
import json
import hashlib
import uuid
import base64
import ast
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum


_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, 
    "enumerate": enumerate, "filter": filter, "float": float, "int": int, 
    "len": len, "list": list, "map": map, "max": max, "min": min,
    "set": set, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "round": round
}


class TypeSignature(Enum):
    """Supported type signatures for constants."""
    FLOAT64 = "Float64"
    INT32 = "Int32"
    STRING = "String"
    BLOB = "Blob"
    JSON_TYPE = "Json"
    TOKENS = "Tokens"  # Token sequences
    CURRENCY = "Currency"
    BOOL = "Bool"


@dataclass
class VersionedRef:
    """Reference to a versioned entity (constant, function, or model)."""
    name: str
    version: int
    
    def __str__(self):
        return f"{self.name}@v{self.version}"
    
    @staticmethod
    def parse(ref_str: str) -> 'VersionedRef':
        """Parse 'name@v3' format."""
        if '@v' not in ref_str:
            raise ValueError(f"Invalid versioned reference format: {ref_str}")
        name, version_str = ref_str.split('@v')
        return VersionedRef(name, int(version_str))


class Curry:
    """Main Curry database interface."""
    
    def __init__(self, db_path: str = ":memory:", fallback_db: Optional['Curry'] = None, uri: bool = False):
        """Initialize Curry with SQLite backend."""
        self.db_path = db_path
        self.fallback_db = fallback_db
        self.conn = sqlite3.connect(db_path, uri=uri)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._initialize_schema()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def _initialize_schema(self):
        """Create all tables and triggers for Curry."""
        cursor = self.conn.cursor()
        
        # Retirement tags: group related retirements
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retirement_tags (
                tag_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reason TEXT NOT NULL,
                description TEXT
            )
        """)
        
        # Constants: immutable, versioned values
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS constants (
                id TEXT NOT NULL,
                version INTEGER NOT NULL,
                value BLOB NOT NULL,
                type_signature TEXT NOT NULL,
                declared_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                retired_at TIMESTAMP,
                retirement_tag_id TEXT,
                
                PRIMARY KEY (id, version),
                FOREIGN KEY (retirement_tag_id) REFERENCES retirement_tags(tag_id)
            )
        """)
        
        # Type compatibility: ensure type consistency across versions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS type_compatibility (
                constant_id TEXT NOT NULL,
                from_version INTEGER NOT NULL,
                to_version INTEGER NOT NULL,
                is_compatible BOOLEAN DEFAULT 1,
                conversion_function TEXT,
                
                PRIMARY KEY (constant_id, from_version, to_version),
                FOREIGN KEY (constant_id, from_version) REFERENCES constants(id, version),
                FOREIGN KEY (constant_id, to_version) REFERENCES constants(id, version)
            )
        """)
        
        # Functions: composed from constants and other functions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS functions (
                name TEXT NOT NULL,
                version INTEGER NOT NULL,
                body TEXT NOT NULL,
                constant_bindings TEXT NOT NULL,  -- JSON: {"const_id": "v2", ...}
                function_bindings TEXT,  -- JSON: {"func_name": "v1", ...}
                is_pure BOOLEAN DEFAULT 0,
                declared_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                retired_at TIMESTAMP,
                retirement_tag_id TEXT,
                
                PRIMARY KEY (name, version),
                FOREIGN KEY (retirement_tag_id) REFERENCES retirement_tags(tag_id)
            )
        """)
        
        try:
            cursor.execute("ALTER TABLE functions ADD COLUMN expected_args TEXT")
        except sqlite3.OperationalError:
            pass
        
        # Function dependencies: track exact versions used
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS function_dependencies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                function_name TEXT NOT NULL,
                function_version INTEGER NOT NULL,
                depends_on_constant_id TEXT,
                depends_on_constant_version INTEGER,
                depends_on_function_name TEXT,
                depends_on_function_version INTEGER,
                
                FOREIGN KEY (function_name, function_version) REFERENCES functions(name, version),
                FOREIGN KEY (depends_on_constant_id, depends_on_constant_version) 
                    REFERENCES constants(id, version),
                FOREIGN KEY (depends_on_function_name, depends_on_function_version) 
                    REFERENCES functions(name, version)
            )
        """)
        
        # Model versions: LLM checkpoints with locked inference parameters
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                model_name TEXT NOT NULL,
                version INTEGER NOT NULL,
                checkpoint_hash TEXT NOT NULL,
                model_type TEXT,  -- 'llama', 'gpt', 'claude', etc.
                base_model_name TEXT,
                base_model_version INTEGER,
                
                -- Inference parameters (locked at version time)
                temperature REAL,
                top_p REAL,
                max_tokens INTEGER,
                
                -- System prompt reference
                system_prompt_id TEXT,
                system_prompt_version INTEGER,
                
                -- Training lineage
                trained_on_data_id TEXT,
                trained_on_data_version INTEGER,
                
                declared_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                retired_at TIMESTAMP,
                retirement_tag_id TEXT,
                
                PRIMARY KEY (model_name, version),
                FOREIGN KEY (retirement_tag_id) REFERENCES retirement_tags(tag_id),
                FOREIGN KEY (system_prompt_id, system_prompt_version) 
                    REFERENCES constants(id, version)
            )
        """)
        
        # Prompts: template compositions with input/output schemas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                prompt_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                name TEXT,
                description TEXT,
                system_prompt_id TEXT,
                system_prompt_version INTEGER,
                instruction_template TEXT NOT NULL,
                input_schema TEXT,  -- JSON
                output_schema TEXT,  -- JSON
                
                declared_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                retired_at TIMESTAMP,
                retirement_tag_id TEXT,
                
                PRIMARY KEY (prompt_id, version),
                FOREIGN KEY (retirement_tag_id) REFERENCES retirement_tags(tag_id),
                FOREIGN KEY (system_prompt_id, system_prompt_version) 
                    REFERENCES constants(id, version)
            )
        """)
        
        # Inferences: LLM inference results with full provenance
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inferences (
                inference_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_version INTEGER NOT NULL,
                input_tokens TEXT,  -- JSON or text representation
                output_tokens BLOB NOT NULL,
                
                temperature_used REAL,
                top_p_used REAL,
                seed INTEGER,
                
                execution_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                execution_duration_ms INTEGER,
                metadata TEXT,  -- JSON: cost, latency details, etc.
                
                FOREIGN KEY (model_name, model_version) REFERENCES model_versions(model_name, version)
            )
        """)
        
        # Execution cache: deterministic memoization
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_cache (
                function_name TEXT NOT NULL,
                function_version INTEGER NOT NULL,
                input_hash TEXT NOT NULL,
                output_hash TEXT NOT NULL,
                cached_result BLOB NOT NULL,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                hit_count INTEGER DEFAULT 1,
                
                PRIMARY KEY (function_name, function_version, input_hash),
                FOREIGN KEY (function_name, function_version) REFERENCES functions(name, version)
            )
        """)
        
        # Note: Validation is done in Python layer for clarity and robustness
        # Type checking happens in declare_constant() method
        # Dependency validation happens in declare_function() method
        
        self.conn.commit()

    def _validate_type_signature(self, type_signature: str) -> TypeSignature:
        """Validate and normalize a declared type signature."""
        for candidate in TypeSignature:
            if candidate.value == type_signature:
                return candidate
        raise ValueError(f"Unsupported type signature: {type_signature}")

    def _serialize_constant_value(self, value: Any, type_signature: str) -> bytes:
        """Serialize a constant value according to its declared type."""
        type_enum = self._validate_type_signature(type_signature)

        if type_enum == TypeSignature.BLOB:
            if not isinstance(value, bytes):
                raise TypeError("Blob constants must be bytes")
            return value

        if type_enum == TypeSignature.FLOAT64:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError("Float64 constants must be numeric")
            return json.dumps(float(value)).encode("utf-8")

        if type_enum == TypeSignature.INT32:
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError("Int32 constants must be integers")
            return json.dumps(value).encode("utf-8")

        if type_enum == TypeSignature.BOOL:
            if not isinstance(value, bool):
                raise TypeError("Bool constants must be booleans")
            return json.dumps(value).encode("utf-8")

        if type_enum == TypeSignature.STRING:
            if not isinstance(value, str):
                raise TypeError(f"{type_signature} constants must be strings")
            return json.dumps(value).encode("utf-8")

        if type_enum == TypeSignature.TOKENS:
            if not isinstance(value, (str, list, dict)):
                raise TypeError(f"{type_signature} constants must be strings, lists of integers, or dictionaries")
            return json.dumps(value).encode("utf-8")

        if type_enum == TypeSignature.JSON_TYPE:
            return json.dumps(value).encode("utf-8")

        if type_enum == TypeSignature.CURRENCY:
            if not isinstance(value, (int, float, str)) or isinstance(value, bool):
                raise TypeError("Currency constants must be numeric or string values")
            return json.dumps(value).encode("utf-8")

        raise ValueError(f"Unsupported type signature: {type_signature}")

    def _deserialize_constant_value(self, raw_value: bytes, type_signature: str) -> Any:
        """Deserialize a constant value according to its declared type."""
        type_enum = self._validate_type_signature(type_signature)

        if type_enum == TypeSignature.BLOB:
            return raw_value

        value = json.loads(raw_value.decode("utf-8"))
        
        # Validate that the retrieved value still matches the declared type
        if type_enum == TypeSignature.FLOAT64 and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            raise TypeError("Float64 constants must be numeric")
        elif type_enum == TypeSignature.INT32 and (not isinstance(value, int) or isinstance(value, bool)):
            raise TypeError("Int32 constants must be integers")
        elif type_enum == TypeSignature.BOOL and not isinstance(value, bool):
            raise TypeError("Bool constants must be booleans")
        elif type_enum == TypeSignature.STRING and not isinstance(value, str):
            raise TypeError(f"{type_signature} constants must be strings")
        elif type_enum == TypeSignature.TOKENS and not isinstance(value, (str, list, dict)):
            raise TypeError(f"{type_signature} constants must be strings, lists of integers, or dictionaries")
        elif type_enum == TypeSignature.CURRENCY and (not isinstance(value, (int, float, str)) or isinstance(value, bool)):
            raise TypeError("Currency constants must be numeric or string values")
            
        return value

    def _serialize_cached_result(self, result: Any) -> bytes:
        """Serialize a cached function result."""
        if isinstance(result, bytes):
            payload = {"encoding": "base64", "value": base64.b64encode(result).decode("ascii")}
        else:
            payload = {"encoding": "json", "value": result}

        try:
            return json.dumps(payload, sort_keys=True).encode("utf-8")
        except TypeError as exc:
            raise TypeError(
                "Function results must be JSON-serializable or bytes to be cached"
            ) from exc

    def _deserialize_cached_result(self, raw_value: bytes) -> Any:
        """Deserialize a cached function result."""
        payload = json.loads(raw_value.decode("utf-8"))
        if payload["encoding"] == "base64":
            return base64.b64decode(payload["value"].encode("ascii"))
        return payload["value"]

    def _canonicalize_for_hash(self, value: Any) -> Any:
        """Convert values into a deterministic, JSON-compatible structure."""
        if value is None or isinstance(value, (str, int, bool)):
            return value

        if isinstance(value, float):
            # Preserve deterministic float representation for hashing purposes.
            return {"__float__": repr(value)}

        if isinstance(value, bytes):
            return {
                "__bytes__": base64.b64encode(value).decode("ascii")
            }

        if isinstance(value, list):
            return [self._canonicalize_for_hash(item) for item in value]

        if isinstance(value, tuple):
            return {
                "__tuple__": [self._canonicalize_for_hash(item) for item in value]
            }

        if isinstance(value, dict):
            return {
                str(key): self._canonicalize_for_hash(val)
                for key, val in sorted(value.items(), key=lambda item: str(item[0]))
            }

        raise TypeError(
            f"Unsupported argument type for deterministic hashing: {type(value).__name__}"
        )

    def _canonical_json_dumps(self, value: Any) -> str:
        """Serialize value in a deterministic way suitable for hashing/storage."""
        canonical = self._canonicalize_for_hash(value)
        return json.dumps(canonical, sort_keys=True, separators=(",", ":"))

    def _normalize_inference_input(self, input_tokens: Any) -> Dict[str, Any]:
        """Normalize inference input into a canonical structure."""
        normalized = {
            "raw_text": input_tokens if isinstance(input_tokens, str) else None,
            "token_refs": input_tokens if isinstance(input_tokens, (dict, list)) else None,
            "source_type": type(input_tokens).__name__,
        }
        return normalized
    
    # ============================================================================
    # CONSTANT OPERATIONS
    # ============================================================================
    
    def declare_constant(
        self,
        const_id: str,
        version: int,
        value: Any,
        type_signature: str,
    ) -> None:
        """Declare a new version of a constant."""
        cursor = self.conn.cursor()
        
        # Validate type consistency
        cursor.execute(
            "SELECT DISTINCT type_signature FROM constants WHERE id = ? LIMIT 2",
            (const_id,)
        )
        rows = cursor.fetchall()
        if rows and rows[0]["type_signature"] != type_signature:
            raise TypeError(
                f"Type mismatch for constant {const_id}: "
                f"existing type is {rows[0]['type_signature']}, "
                f"but attempted to declare {type_signature}"
            )

        cursor.execute(
            "SELECT MAX(version) AS max_version FROM constants WHERE id = ?",
            (const_id,)
        )
        existing = cursor.fetchone()
        if existing and existing["max_version"] is not None and version <= existing["max_version"]:
            raise ValueError(
                f"Version for constant {const_id} must be greater than existing max "
                f"version {existing['max_version']}; got {version}"
            )
        
        # Serialize value
        value_blob = self._serialize_constant_value(value, type_signature)
        
        cursor.execute(
            """INSERT INTO constants (id, version, value, type_signature)
               VALUES (?, ?, ?, ?)""",
            (const_id, version, value_blob, type_signature)
        )
        self.conn.commit()
    
    def retire_constant(
        self,
        const_id: str,
        version: int,
        retirement_tag: Optional[str] = None,
    ) -> None:
        """Mark a constant version as retired."""
        cursor = self.conn.cursor()
        cursor.execute(
            """UPDATE constants 
               SET retired_at = CURRENT_TIMESTAMP, retirement_tag_id = ?
               WHERE id = ? AND version = ?""",
            (retirement_tag, const_id, version)
        )
        if cursor.rowcount == 0:
            raise KeyError(f"Constant {const_id}@v{version} not found")
        self.conn.commit()
    
    def get_constant(
        self,
        const_id: str,
        version: int,
    ) -> Dict[str, Any]:
        """Retrieve a constant by exact version."""
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT id, version, value, type_signature, declared_at, retired_at
               FROM constants
               WHERE id = ? AND version = ?""",
            (const_id, version)
        )
        row = cursor.fetchone()
        if not row:
            if self.fallback_db:
                return self.fallback_db.get_constant(const_id, version)
            raise KeyError(f"Constant {const_id}@v{version} not found")
        
        if row["retired_at"]:
            raise ValueError(f"Constant {const_id}@v{version} has been retired")
        
        # Deserialize value
        value = self._deserialize_constant_value(row["value"], row["type_signature"])
        
        return {
            "id": row["id"],
            "version": row["version"],
            "value": value,
            "type_signature": row["type_signature"],
            "declared_at": row["declared_at"],
        }
    
    def get_constant_latest(self, const_id: str) -> Dict[str, Any]:
        """Get the most recent active version of a constant."""
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT id, version, value, type_signature, declared_at
               FROM constants
               WHERE id = ? AND retired_at IS NULL
               ORDER BY version DESC
               LIMIT 1""",
            (const_id,)
        )
        row = cursor.fetchone()
        if not row:
            if self.fallback_db:
                return self.fallback_db.get_constant_latest(const_id)
            raise KeyError(f"No active version of constant {const_id} found")
        
        value = self._deserialize_constant_value(row["value"], row["type_signature"])
        
        return {
            "id": row["id"],
            "version": row["version"],
            "value": value,
            "type_signature": row["type_signature"],
            "declared_at": row["declared_at"],
        }

    def list_constants(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List all constants with their latest versions."""
        cursor = self.conn.cursor()
        query = "SELECT id, MAX(version) as latest_version, type_signature, declared_at FROM constants"
        if active_only:
            query += " WHERE retired_at IS NULL"
        query += " GROUP BY id"
        cursor.execute(query)
        results = [dict(row) for row in cursor.fetchall()]
        
        if self.fallback_db:
            fallback_results = self.fallback_db.list_constants(active_only)
            local_ids = {r["id"] for r in results}
            for fr in fallback_results:
                if fr["id"] not in local_ids:
                    results.append(fr)
                    
        return results
    
    # ============================================================================
    # FUNCTION OPERATIONS
    # ============================================================================
    
    def validate_function_body(
        self, body: str, allowed_names: Set[str], expected_args: Optional[List[str]] = None
    ) -> None:
        """Statically analyze a function body to prevent unsafe constructs and verify names."""
        try:
            tree = ast.parse(body, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"Function body has syntax error: {e}")
            
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("__"):
                    raise ValueError(f"Unsafe dunder attribute access: .{node.attr}")
                if isinstance(node.value, ast.Name) and node.value.id.startswith("__"):
                    raise ValueError(f"Unsafe access on {node.value.id}")
            
            if isinstance(node, ast.Name):
                if node.id not in allowed_names and node.id not in _SAFE_BUILTINS:
                    if expected_args is not None and node.id not in expected_args:
                        raise ValueError(f"Unbound name '{node.id}' not in bindings or expected args")

    def declare_function(
        self,
        name: str,
        version: int,
        body: str,
        constant_bindings: Optional[Dict[str, int]] = None,
        function_bindings: Optional[Dict[str, int]] = None,
        is_pure: bool = False,
        expected_args: Optional[List[str]] = None,
    ) -> None:
        """Declare a versioned function with exact dependency versions."""
        constant_bindings = constant_bindings or {}
        function_bindings = function_bindings or {}
        
        allowed_names = set(constant_bindings.keys()) | set(function_bindings.keys())
        try:
            self.validate_function_body(body, allowed_names, expected_args)
        except ValueError as e:
            raise ValueError(f"Function {name}@v{version} body is invalid: {e}")

        cursor = self.conn.cursor()
        
        cursor.execute(
            "SELECT MAX(version) AS max_version FROM functions WHERE name = ?",
            (name,)
        )
        existing = cursor.fetchone()
        if existing and existing["max_version"] is not None and version <= existing["max_version"]:
            raise ValueError(
                f"Version for function {name} must be greater than existing max "
                f"version {existing['max_version']}; got {version}"
            )
        
        
        # Validate all dependencies exist and are active
        for const_id, const_version in constant_bindings.items():
            try:
                self.get_constant(const_id, const_version)
            except KeyError:
                raise ValueError(
                    f"Function {name}@v{version} references non-existent constant {const_id}@v{const_version}"
                )
            # get_constant already checks for retired_at and raises ValueError
        
        for func_name, func_version in function_bindings.items():
            try:
                self.get_function(func_name, func_version)
            except KeyError:
                raise ValueError(
                    f"Function {name}@v{version} references non-existent function {func_name}@v{func_version}"
                )
        
        # Insert function
        cursor.execute(
            """INSERT INTO functions 
               (name, version, body, constant_bindings, function_bindings, is_pure, expected_args)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (name, version, body, json.dumps(constant_bindings), json.dumps(function_bindings), is_pure, json.dumps(expected_args) if expected_args is not None else None)
        )
        
        # Record dependencies
        for const_id, const_version in constant_bindings.items():
            cursor.execute(
                """INSERT INTO function_dependencies 
                   (function_name, function_version, depends_on_constant_id, depends_on_constant_version)
                   VALUES (?, ?, ?, ?)""",
                (name, version, const_id, const_version)
            )
        
        for func_name, func_version in function_bindings.items():
            cursor.execute(
                """INSERT INTO function_dependencies 
                   (function_name, function_version, depends_on_function_name, depends_on_function_version)
                   VALUES (?, ?, ?, ?)""",
                (name, version, func_name, func_version)
            )
        
        self.conn.commit()
    
    def get_function(self, name: str, version: int) -> Dict[str, Any]:
        """Retrieve a function by exact version."""
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT name, version, body, constant_bindings, function_bindings, is_pure, expected_args, retired_at
               FROM functions
               WHERE name = ? AND version = ?""",
            (name, version)
        )
        row = cursor.fetchone()
        if not row:
            if self.fallback_db:
                return self.fallback_db.get_function(name, version)
            raise KeyError(f"Function {name}@v{version} not found")
        
        if row["retired_at"]:
            raise ValueError(f"Function {name}@v{version} has been retired")
        
        return {
            "name": row["name"],
            "version": row["version"],
            "body": row["body"],
            "constant_bindings": json.loads(row["constant_bindings"]),
            "function_bindings": json.loads(row["function_bindings"]),
            "is_pure": bool(row["is_pure"]),
            "expected_args": json.loads(row["expected_args"]) if row["expected_args"] is not None else None,
        }
    
    def retire_function(
        self,
        name: str,
        version: int,
        retirement_tag: Optional[str] = None,
    ) -> None:
        """Mark a function version as retired."""
        cursor = self.conn.cursor()
        cursor.execute(
            """UPDATE functions 
               SET retired_at = CURRENT_TIMESTAMP, retirement_tag_id = ?
               WHERE name = ? AND version = ?""",
            (retirement_tag, name, version)
        )
        if cursor.rowcount == 0:
            raise KeyError(f"Function {name}@v{version} not found")
        self.conn.commit()

    def list_functions(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List all functions with their latest versions."""
        cursor = self.conn.cursor()
        query = "SELECT name, MAX(version) as latest_version, is_pure, expected_args, declared_at FROM functions"
        if active_only:
            query += " WHERE retired_at IS NULL"
        query += " GROUP BY name"
        cursor.execute(query)
        results = []
        for row in cursor.fetchall():
            d = dict(row)
            if "expected_args" in d and d["expected_args"] is not None:
                d["expected_args"] = json.loads(d["expected_args"])
            else:
                d["expected_args"] = None
            results.append(d)
            
        if self.fallback_db:
            fallback_results = self.fallback_db.list_functions(active_only)
            local_names = {r["name"] for r in results}
            for fr in fallback_results:
                if fr["name"] not in local_names:
                    results.append(fr)
                    
        return results
    
    # ============================================================================
    # EXECUTION AND COMPOSITION
    # ============================================================================
    
    def call_function(
        self,
        name: str,
        version: int,
        args: Dict[str, Any],
        _call_stack: Optional[Set[str]] = None,
    ) -> Any:
        """Execute a versioned function with locked dependencies."""
        call_key = f"{name}@v{version}"
        call_stack = set(_call_stack or set())
        if call_key in call_stack:
            raise RuntimeError(f"Cycle detected while executing function call stack at {call_key}")
        call_stack.add(call_key)

        func_def = self.get_function(name, version)

        context = {}
        for const_id, const_version in func_def["constant_bindings"].items():
            const = self.get_constant(const_id, const_version)
            context[const_id] = const["value"]

        for func_name, func_version in func_def["function_bindings"].items():
            context[func_name] = (
                lambda nested_args, fn=func_name, fv=func_version, cs=call_stack:
                self.call_function(fn, fv, nested_args, _call_stack=cs)
            )

        eval_context = {**context, **args}
        cacheable = bool(func_def["is_pure"])
        input_hash = None

        if cacheable:
            hash_payload = {
                "args": args,
                "constant_bindings": func_def["constant_bindings"],
                "function_bindings": func_def["function_bindings"],
            }
            input_hash = hashlib.sha256(
                self._canonical_json_dumps(hash_payload).encode("utf-8")
            ).hexdigest()
            cursor = self.conn.cursor()
            cursor.execute(
                """SELECT cached_result FROM execution_cache
                   WHERE function_name = ? AND function_version = ? AND input_hash = ?""",
                (name, version, input_hash)
            )
            cached = cursor.fetchone()
            if cached:
                cursor.execute(
                    """UPDATE execution_cache
                       SET hit_count = hit_count + 1
                       WHERE function_name = ? AND function_version = ? AND input_hash = ?""",
                    (name, version, input_hash)
                )
                self.conn.commit()
                return self._deserialize_cached_result(cached["cached_result"])

        try:
            result = eval(func_def["body"], {"__builtins__": _SAFE_BUILTINS}, eval_context)
        except Exception as exc:
            raise RuntimeError(f"Failed to execute function {name}@v{version}: {exc}") from exc

        if cacheable and input_hash is not None:
            try:
                cursor = self.conn.cursor()
                cached_result = self._serialize_cached_result(result)
                output_hash = hashlib.sha256(cached_result).hexdigest()
                cursor.execute(
                    """INSERT OR REPLACE INTO execution_cache
                       (function_name, function_version, input_hash, output_hash, cached_result, hit_count)
                       VALUES (?, ?, ?, ?, ?, COALESCE(
                           (SELECT hit_count FROM execution_cache
                            WHERE function_name = ? AND function_version = ? AND input_hash = ?), 1
                       ))""",
                    (name, version, input_hash, output_hash, cached_result, name, version, input_hash)
                )
                self.conn.commit()
            except Exception:
                pass

        return result
    
    def get_function_lineage(self, name: str, version: int) -> Dict[str, Any]:
        """Get complete dependency tree for a function."""
        cursor = self.conn.cursor()
        
        def get_dependencies(
            fn_name: str,
            fn_version: int,
            path: Optional[Set[str]] = None
        ) -> Dict:
            current_key = f"{fn_name}@v{fn_version}"
            current_path = set(path or set())
            if current_key in current_path:
                return {
                    "constants": [],
                    "functions": [],
                    "cycle_detected": True,
                    "cycle_at": current_key,
                }
            current_path.add(current_key)
            
            cursor.execute(
                """SELECT depends_on_constant_id, depends_on_constant_version,
                          depends_on_function_name, depends_on_function_version
                   FROM function_dependencies
                   WHERE function_name = ? AND function_version = ?""",
                (fn_name, fn_version)
            )
            deps = {"constants": [], "functions": []}
            
            for row in cursor.fetchall():
                if row["depends_on_constant_id"]:
                    deps["constants"].append({
                        "id": row["depends_on_constant_id"],
                        "version": row["depends_on_constant_version"],
                    })
                if row["depends_on_function_name"]:
                    deps["functions"].append({
                        "name": row["depends_on_function_name"],
                        "version": row["depends_on_function_version"],
                        "lineage": get_dependencies(
                            row["depends_on_function_name"],
                            row["depends_on_function_version"],
                            current_path
                        ),
                    })
            
            return deps
        
        return {
            "function": f"{name}@v{version}",
            "dependencies": get_dependencies(name, version),
        }
    
    # ============================================================================
    # MODEL AND INFERENCE OPERATIONS
    # ============================================================================
    
    def register_model(
        self,
        model_name: str,
        version: int,
        checkpoint_hash: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        system_prompt_id: Optional[str] = None,
        system_prompt_version: Optional[int] = None,
        model_type: Optional[str] = None,
        trained_on_data_id: Optional[str] = None,
        trained_on_data_version: Optional[int] = None,
    ) -> None:
        """Register a model version with locked inference parameters."""
        cursor = self.conn.cursor()
        
        cursor.execute(
            "SELECT MAX(version) AS max_version FROM model_versions WHERE model_name = ?",
            (model_name,)
        )
        existing = cursor.fetchone()
        if existing and existing["max_version"] is not None and version <= existing["max_version"]:
            raise ValueError(
                f"Version for model {model_name} must be greater than existing max "
                f"version {existing['max_version']}; got {version}"
            )
        
        # Validate system prompt if provided
        if system_prompt_id is not None or system_prompt_version is not None:
            if system_prompt_id is None or system_prompt_version is None:
                raise ValueError("system_prompt_id and system_prompt_version must be provided together")
            self.get_constant(system_prompt_id, system_prompt_version)

        if trained_on_data_id is not None or trained_on_data_version is not None:
            if trained_on_data_id is None or trained_on_data_version is None:
                raise ValueError("trained_on_data_id and trained_on_data_version must be provided together")
            self.get_constant(trained_on_data_id, trained_on_data_version)
        
        cursor.execute(
            """INSERT INTO model_versions 
               (model_name, version, checkpoint_hash, temperature, top_p, max_tokens, 
                system_prompt_id, system_prompt_version, model_type,
                trained_on_data_id, trained_on_data_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (model_name, version, checkpoint_hash, temperature, top_p, max_tokens,
             system_prompt_id, system_prompt_version, model_type,
             trained_on_data_id, trained_on_data_version)
        )
        self.conn.commit()
    
    def get_model(self, model_name: str, version: int) -> Dict[str, Any]:
        """Retrieve model configuration by exact version."""
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT * FROM model_versions
               WHERE model_name = ? AND version = ? AND retired_at IS NULL""",
            (model_name, version)
        )
        row = cursor.fetchone()
        if not row:
            if self.fallback_db:
                return self.fallback_db.get_model(model_name, version)
            raise KeyError(f"Model {model_name}@v{version} not found")
        
        return {
            "model_name": row["model_name"],
            "version": row["version"],
            "checkpoint_hash": row["checkpoint_hash"],
            "temperature": row["temperature"],
            "top_p": row["top_p"],
            "max_tokens": row["max_tokens"],
            "system_prompt_id": row["system_prompt_id"],
            "system_prompt_version": row["system_prompt_version"],
            "model_type": row["model_type"],
            "trained_on_data_id": row["trained_on_data_id"],
            "trained_on_data_version": row["trained_on_data_version"],
        }

    def get_model_latest(self, model_name: str) -> Dict[str, Any]:
        """Get the most recent active version of a model."""
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT *
               FROM model_versions
               WHERE model_name = ? AND retired_at IS NULL
               ORDER BY version DESC
               LIMIT 1""",
            (model_name,)
        )
        row = cursor.fetchone()
        if not row:
            if self.fallback_db:
                return self.fallback_db.get_model_latest(model_name)
            raise KeyError(f"No active version of model {model_name} found")
        return dict(row)

    def list_models(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List all models with their latest versions."""
        cursor = self.conn.cursor()
        query = "SELECT model_name, MAX(version) as latest_version, model_type, declared_at FROM model_versions"
        if active_only:
            query += " WHERE retired_at IS NULL"
        query += " GROUP BY model_name"
        cursor.execute(query)
        results = [dict(row) for row in cursor.fetchall()]
        
        if self.fallback_db:
            fallback_results = self.fallback_db.list_models(active_only)
            local_names = {r["model_name"] for r in results}
            for fr in fallback_results:
                if fr["model_name"] not in local_names:
                    results.append(fr)
                    
        return results
    
    def record_inference(
        self,
        model_name: str,
        model_version: int,
        input_tokens: Any,
        output_tokens: bytes,
        seed: int = 42,
        temperature_used: Optional[float] = None,
        top_p_used: Optional[float] = None,
        duration_ms: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Record an LLM inference result with full provenance."""
        cursor = self.conn.cursor()
        inference_id = str(uuid.uuid4())

        if duration_ms is not None and duration_ms < 0:
            raise ValueError("duration_ms must be non-negative when provided")
        
        # Get model to verify it exists
        model = self.get_model(model_name, model_version)
        if temperature_used is None:
            temperature_used = model["temperature"]
        if top_p_used is None:
            top_p_used = model["top_p"]

        input_tokens_json = self._canonical_json_dumps(
            self._normalize_inference_input(input_tokens)
        )
        try:
            metadata_json = json.dumps(metadata or {})
        except (TypeError, ValueError) as exc:
            raise TypeError(f"metadata must be JSON-serializable: {exc}") from exc

        cursor.execute(
            """INSERT INTO inferences 
               (inference_id, model_name, model_version, input_tokens, output_tokens,
                temperature_used, top_p_used, seed, execution_duration_ms, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (inference_id, model_name, model_version, input_tokens_json, output_tokens,
             temperature_used, top_p_used,
             seed, duration_ms, metadata_json)
        )
        self.conn.commit()
        
        return inference_id
    
    def get_inference(self, inference_id: str) -> Dict[str, Any]:
        """Retrieve an inference record with full provenance."""
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT * FROM inferences WHERE inference_id = ?""",
            (inference_id,)
        )
        row = cursor.fetchone()
        if not row:
            raise KeyError(f"Inference {inference_id} not found")
        
        return {
            "inference_id": row["inference_id"],
            "model_name": row["model_name"],
            "model_version": row["model_version"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "temperature_used": row["temperature_used"],
            "top_p_used": row["top_p_used"],
            "seed": row["seed"],
            "execution_timestamp": row["execution_timestamp"],
            "execution_duration_ms": row["execution_duration_ms"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
        }

    def search_inferences(
        self,
        model_name: Optional[str] = None,
        model_version: Optional[int] = None,
        seed: Optional[int] = None,
        start_timestamp: Optional[str] = None,
        end_timestamp: Optional[str] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        min_input_tokens_count: Optional[int] = None,
        max_input_tokens_count: Optional[int] = None,
        min_output_tokens_count: Optional[int] = None,
        max_output_tokens_count: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Search inference rows with deterministic ordering and pagination."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        if offset < 0:
            raise ValueError("offset must be non-negative")

        cursor = self.conn.cursor()
        where_clauses = []
        params: List[Any] = []

        if model_name is not None:
            where_clauses.append("model_name = ?")
            params.append(model_name)
        if model_version is not None:
            where_clauses.append("model_version = ?")
            params.append(model_version)
        if seed is not None:
            where_clauses.append("seed = ?")
            params.append(seed)
        if start_timestamp is not None:
            where_clauses.append("execution_timestamp >= ?")
            params.append(start_timestamp)
        if end_timestamp is not None:
            where_clauses.append("execution_timestamp <= ?")
            params.append(end_timestamp)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        cursor.execute(
            f"""SELECT * FROM inferences
                {where_sql}
                ORDER BY execution_timestamp ASC, inference_id ASC""",
            tuple(params),
        )

        rows = cursor.fetchall()
        metadata_filters = metadata_filters or {}
        filtered: List[Dict[str, Any]] = []

        for row in rows:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            input_tokens_count = metadata.get("input_tokens_count")
            output_tokens_count = metadata.get("output_tokens_count")

            metadata_match = all(metadata.get(k) == v for k, v in metadata_filters.items())
            if not metadata_match:
                continue
            if min_input_tokens_count is not None and (
                input_tokens_count is None or input_tokens_count < min_input_tokens_count
            ):
                continue
            if max_input_tokens_count is not None and (
                input_tokens_count is None or input_tokens_count > max_input_tokens_count
            ):
                continue
            if min_output_tokens_count is not None and (
                output_tokens_count is None or output_tokens_count < min_output_tokens_count
            ):
                continue
            if max_output_tokens_count is not None and (
                output_tokens_count is None or output_tokens_count > max_output_tokens_count
            ):
                continue

            filtered.append(
                {
                    "inference_id": row["inference_id"],
                    "model_name": row["model_name"],
                    "model_version": row["model_version"],
                    "input_tokens": row["input_tokens"],
                    "output_tokens": row["output_tokens"],
                    "temperature_used": row["temperature_used"],
                    "top_p_used": row["top_p_used"],
                    "seed": row["seed"],
                    "execution_timestamp": row["execution_timestamp"],
                    "execution_duration_ms": row["execution_duration_ms"],
                    "metadata": metadata,
                }
            )

        return filtered[offset:offset + limit]

    def compare_inferences(self, a_id: str, b_id: str) -> Dict[str, Any]:
        """Compare two inference records and return structured deltas."""
        a = self.get_inference(a_id)
        b = self.get_inference(b_id)
        a_output_hash = hashlib.sha256(a["output_tokens"]).hexdigest()
        b_output_hash = hashlib.sha256(b["output_tokens"]).hexdigest()

        return {
            "a_id": a_id,
            "b_id": b_id,
            "same_model": (
                a["model_name"] == b["model_name"] and
                a["model_version"] == b["model_version"]
            ),
            "same_seed": a["seed"] == b["seed"],
            "same_input_tokens": a["input_tokens"] == b["input_tokens"],
            "same_output_hash": a_output_hash == b_output_hash,
            "a_output_sha256": a_output_hash,
            "b_output_sha256": b_output_hash,
            "parameter_diff": {
                "temperature_used": [a["temperature_used"], b["temperature_used"]],
                "top_p_used": [a["top_p_used"], b["top_p_used"]],
            },
            "metadata_diff": {
                "a_only_keys": sorted(set(a["metadata"].keys()) - set(b["metadata"].keys())),
                "b_only_keys": sorted(set(b["metadata"].keys()) - set(a["metadata"].keys())),
                "changed_keys": sorted(
                    key for key in (set(a["metadata"].keys()) & set(b["metadata"].keys()))
                    if a["metadata"][key] != b["metadata"][key]
                ),
            },
        }
    
    # ============================================================================
    # RETIREMENT AND TAGGING
    # ============================================================================
    
    def create_retirement_tag(self, tag_id: str, reason: str, description: Optional[str] = None) -> str:
        """Create a retirement tag to group related retirements."""
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO retirement_tags (tag_id, reason, description)
               VALUES (?, ?, ?)""",
            (tag_id, reason, description)
        )
        self.conn.commit()
        return tag_id

    def evict_execution_cache(self, max_entries: int = 1000) -> None:
        """Evict execution cache down to max_entries using least recently cached policy."""
        cursor = self.conn.cursor()
        cursor.execute(
            """DELETE FROM execution_cache
               WHERE rowid NOT IN (
                   SELECT rowid FROM execution_cache
                   ORDER BY cached_at DESC LIMIT ?
               )""",
            (max_entries,)
        )
        self.conn.commit()
    
    def close(self):
        """Close the database connection."""
        self.conn.close()

    def backup(self, target_path: str, pages: int = -1, sleep: float = 0.250) -> None:
        """Safely backup the database to a target file."""
        with sqlite3.connect(target_path) as dst:
            self.conn.backup(dst, pages=pages, sleep=sleep)
    
    def export_schema(self) -> str:
        """Export the current schema as SQL."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        return "\n\n".join(tables)

import os

class CurrySession:
    """A two-tier session managing a global core DB and a local project DB."""
    
    def __init__(self, core_db: Curry, local_db: Curry, config: Dict[str, Any]):
        self.core_db = core_db
        self.local_db = local_db
        self.config = config

    @classmethod
    def from_project(cls, project_dir: str) -> 'CurrySession':
        config_path = os.path.join(project_dir, ".curry", "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Curry config not found at {config_path}")
            
        with open(config_path, "r") as f:
            config = json.load(f)
            
        core_db_path = config.get("core_db")
        if not core_db_path:
            raise ValueError("config.json must specify 'core_db'")
            
        # For relative paths in config, resolve them relative to project_dir
        local_db_path = config.get("local_db", ".curry/curry.db")
        if not os.path.isabs(local_db_path):
            local_db_path = os.path.join(project_dir, local_db_path)
            
        # Open core as read-only — no accidental writes from project sessions
        core_db_uri = f"file:{core_db_path.replace(chr(92), '/')}?mode=ro"
        core_db = Curry(core_db_uri, uri=True)
        
        # Ensure local db dir exists
        os.makedirs(os.path.dirname(local_db_path), exist_ok=True)
        local_db = Curry(local_db_path, fallback_db=core_db)
        
        return cls(core_db, local_db, config)

    def close(self):
        self.local_db.close()
        self.core_db.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # Model operations -> core_db
    def register_model(self, *args, **kwargs):
        raise PermissionError(
            "register_model writes to the global core DB and cannot be called from a project session. "
            "Use Curry(core_db_path) directly for model registration."
        )
        
    def get_model(self, *args, **kwargs):
        return self.core_db.get_model(*args, **kwargs)
        
    def get_model_latest(self, *args, **kwargs):
        return self.core_db.get_model_latest(*args, **kwargs)
        
    def list_models(self, *args, **kwargs):
        return self.core_db.list_models(*args, **kwargs)
        
    def retire_model(self, *args, **kwargs):
        raise PermissionError(
            "retire_model writes to the global core DB and cannot be called from a project session. "
            "Use Curry(core_db_path) directly for model registration."
        )

    # Local operations -> local_db
    def declare_constant(self, *args, **kwargs):
        return self.local_db.declare_constant(*args, **kwargs)
        
    def get_constant(self, *args, **kwargs):
        return self.local_db.get_constant(*args, **kwargs)
        
    def get_constant_latest(self, *args, **kwargs):
        return self.local_db.get_constant_latest(*args, **kwargs)
        
    def retire_constant(self, *args, **kwargs):
        return self.local_db.retire_constant(*args, **kwargs)
        
    def list_constants(self, *args, **kwargs):
        return self.local_db.list_constants(*args, **kwargs)

    def declare_function(self, *args, **kwargs):
        return self.local_db.declare_function(*args, **kwargs)
        
    def get_function(self, *args, **kwargs):
        return self.local_db.get_function(*args, **kwargs)
        
    def retire_function(self, *args, **kwargs):
        return self.local_db.retire_function(*args, **kwargs)
        
    def list_functions(self, *args, **kwargs):
        return self.local_db.list_functions(*args, **kwargs)
        
    def call_function(self, *args, **kwargs):
        return self.local_db.call_function(*args, **kwargs)
        
    def get_function_lineage(self, *args, **kwargs):
        return self.local_db.get_function_lineage(*args, **kwargs)

    def record_inference(self, *args, **kwargs):
        return self.local_db.record_inference(*args, **kwargs)
        
    def get_inference(self, *args, **kwargs):
        return self.local_db.get_inference(*args, **kwargs)
        
    def search_inferences(self, *args, **kwargs):
        return self.local_db.search_inferences(*args, **kwargs)
        
    def compare_inferences(self, *args, **kwargs):
        return self.local_db.compare_inferences(*args, **kwargs)
        
    def get_retirement_tag(self, *args, **kwargs):
        return self.local_db.get_retirement_tag(*args, **kwargs)

