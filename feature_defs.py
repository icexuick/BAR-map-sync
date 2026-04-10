"""
FeatureDef loader — evaluates BAR/Spring feature definition Lua files and
returns a flat {feature_name_lower: def_dict} mapping.

BAR feature files (e.g. features/rocks30.lua, features/enginetrees_override.lua)
are standalone Lua modules that build a table of FeatureDefs and `return` it.
They use `string.format`, for-loops, helper functions like `lowerkeys`, and
reference `customParams` (capital P) which Spring normalizes to `customparams`.

We evaluate these in a lupa sandbox with the minimum shims BAR needs:
  - lowerkeys(t)  : returns a shallow copy with lowercased top-level keys
  - Spring        : stub table (some files reference it guardedly)

Per-map feature files (inside the sd7 archive) follow the same conventions.
"""

import os
from typing import Dict, Any, Optional

import lupa
from lupa import LuaRuntime


# ──────────────────────────────────────────────────────────────────────────
# Lua helpers that BAR's feature files expect to be globally available
# ──────────────────────────────────────────────────────────────────────────

_LUA_SHIM = r"""
function lowerkeys(t)
    if type(t) ~= 'table' then return t end
    local out = {}
    for k, v in pairs(t) do
        if type(k) == 'string' then
            out[string.lower(k)] = v
        else
            out[k] = v
        end
    end
    return out
end

-- table.copy: BAR/Spring helper for deep-copying a table (used in e.g.
-- candycane.lua, tombstones.lua). Our implementation is recursive and handles
-- nested tables, which matches Spring's behavior for FeatureDef authoring.
function table.copy(t)
    if type(t) ~= 'table' then return t end
    local out = {}
    for k, v in pairs(t) do
        if type(v) == 'table' then
            out[k] = table.copy(v)
        else
            out[k] = v
        end
    end
    return out
end

-- Minimal Spring stub: some feature files reference Spring.GetGameFrame or
-- similar guardedly. Returning nil/false on any lookup is safe for static
-- FeatureDef files.
Spring = setmetatable({}, {
    __index = function(_, _) return function(...) return nil end end,
})

-- VFS stub: ignore any VFS.Include() calls made from feature files.
VFS = setmetatable({
    Include = function(...) return {} end,
    DirList = function(...) return {} end,
}, {
    __index = function(_, _) return function(...) return nil end end,
})

-- Some files reference `gadgetHandler`, `Script`, etc. — stub them all.
gadgetHandler = setmetatable({}, {__index = function() return function() end end})
Script = setmetatable({}, {__index = function() return function() end end})
"""


def _new_runtime() -> LuaRuntime:
    """Create a fresh Lua runtime with BAR feature-file shims installed."""
    lua = LuaRuntime(unpack_returned_tuples=True)
    lua.execute(_LUA_SHIM)
    return lua


def _lua_table_to_py(obj: Any) -> Any:
    """Recursively convert a lupa Lua table to plain Python dict/list/primitive."""
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return obj
    # lupa tables expose .items() for dict-like access
    if lupa.lua_type(obj) == 'table':
        # Detect pure sequence (1-indexed contiguous integer keys)
        keys = list(obj.keys())
        if keys and all(isinstance(k, int) for k in keys):
            sorted_keys = sorted(keys)
            if sorted_keys == list(range(1, len(sorted_keys) + 1)):
                return [_lua_table_to_py(obj[k]) for k in sorted_keys]
        return {k: _lua_table_to_py(v) for k, v in obj.items()}
    return obj


def load_feature_file(lua_path: str) -> Dict[str, dict]:
    """Evaluate one feature Lua file and return {name_lower: def_dict}.

    Returns an empty dict on any error (file missing, parse error, unexpected
    return type) with a printed warning — we never want one malformed feature
    file to kill the whole pipeline.
    """
    if not os.path.isfile(lua_path):
        return {}

    try:
        with open(lua_path, 'r', encoding='utf-8', errors='replace') as f:
            source = f.read()
    except Exception as e:
        print(f"      [FeatureDef] Read error {lua_path}: {e}")
        return {}

    lua = _new_runtime()

    # Feature files use `return <expr>` at top level. lupa's .execute() runs
    # statements and returns nothing; .eval() evaluates an expression. We wrap
    # the source in a function so the `return` statement works.
    wrapped = f"return (function()\n{source}\nend)()"

    try:
        result = lua.execute(wrapped)
    except Exception as e:
        print(f"      [FeatureDef] Eval error {os.path.basename(lua_path)}: {e}")
        return {}

    py = _lua_table_to_py(result)
    if not isinstance(py, dict):
        print(f"      [FeatureDef] {os.path.basename(lua_path)}: unexpected return type ({type(py).__name__})")
        return {}

    # Normalize: lowercase keys (Spring does this), and ensure each value is a dict
    out: Dict[str, dict] = {}
    for k, v in py.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            continue
        out[k.lower()] = v
    return out


def load_feature_dir(features_dir: str) -> Dict[str, dict]:
    """Load every *.lua under a features/ directory (recursively) and merge
    into one dict.

    Map-local features are often organized in subfolders (e.g.
    features/lathan/eistatue2.lua, features/smoth/harborset/radardish_1.lua),
    so we walk the tree rather than just the top level.

    If two files define the same feature name, the later one wins (Spring's
    behavior is load-order dependent; for our purposes any consistent rule
    is fine since feature names are globally unique in practice).
    """
    merged: Dict[str, dict] = {}
    if not os.path.isdir(features_dir):
        return merged

    for root, _dirs, files in os.walk(features_dir):
        for fname in sorted(files):
            if not fname.lower().endswith('.lua'):
                continue
            path = os.path.join(root, fname)
            defs = load_feature_file(path)
            if defs:
                rel = os.path.relpath(path, features_dir).replace('\\', '/')
                print(f"      [FeatureDef] {rel}: {len(defs)} definitions")
                merged.update(defs)
    return merged


# ──────────────────────────────────────────────────────────────────────────
# FeatureDef field accessors (handle customparams / customParams casing)
# ──────────────────────────────────────────────────────────────────────────

def get_object_path(feature_def: dict) -> Optional[str]:
    """Return the .s3o object path from a FeatureDef (relative to objects3d/)."""
    obj = feature_def.get('object')
    if isinstance(obj, str) and obj.strip():
        return obj.strip()
    return None


def get_custom_params(feature_def: dict) -> dict:
    """Return customparams dict (case-insensitive lookup)."""
    for key in ('customparams', 'customParams', 'CustomParams'):
        v = feature_def.get(key)
        if isinstance(v, dict):
            return v
    return {}


def get_normal_texture(feature_def: dict) -> Optional[str]:
    """Return the normaltex path from customparams, if any.

    BAR uses customparams.normaltex = "unittextures/<name>.dds".
    """
    cp = get_custom_params(feature_def)
    for key in ('normaltex', 'normalTex', 'NormalTex'):
        v = cp.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python feature_defs.py <features_dir_or_file>")
        sys.exit(1)
    target = sys.argv[1]
    if os.path.isdir(target):
        defs = load_feature_dir(target)
    else:
        defs = load_feature_file(target)
    print(f"\nLoaded {len(defs)} feature definitions.")
    # Show first 5 as a sanity check
    for name in list(defs.keys())[:5]:
        d = defs[name]
        obj = get_object_path(d)
        nrm = get_normal_texture(d)
        print(f"  {name}:")
        print(f"    object: {obj}")
        print(f"    normal: {nrm}")
