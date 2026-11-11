"""OpenAI tool schemas for the vendored tools, built the way FastMCP (mcp 1.14, used by the authors) built its MCP tool list:
name = function name, description = the raw docstring (`fn.__doc__`), inputSchema = JSON schema of a pydantic model created from the
signature (one field per parameter: annotation + default; title "<name>Arguments"). Arguments are validated / coerced with the same
model before the call (FastMCP: pre-parse JSON strings for non-str fields, model_validate, one-level dump)."""
from __future__ import annotations

import inspect
import json
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model


class _ArgModelBase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


def _arg_model(name: str, fn) -> Tuple[type, Dict[str, Any]]:
    sig = inspect.signature(fn)
    fields: Dict[str, Any] = {}
    str_params: Dict[str, bool] = {}
    for pname, param in sig.parameters.items():
        ann = param.annotation if param.annotation is not inspect.Parameter.empty else Any
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[pname] = (ann, Field(default=default))
        str_params[pname] = ann is str
    model = create_model(f"{name}Arguments", __base__=_ArgModelBase, **fields)
    return model, str_params


def build(tools: Dict[str, Any]) -> Tuple[List[dict], Dict[str, Tuple[type, Dict[str, bool]]]]:
    """-> (OpenAI `tools` list, {name: (arg model, {param: is_str})})."""
    schemas, models = [], {}
    for name, fn in tools.items():
        model, str_params = _arg_model(name, fn)
        models[name] = (model, str_params)
        schemas.append({"type": "function", "function": {"name": name, "description": fn.__doc__ or "",
                                                          "parameters": model.model_json_schema(by_alias=True)}})
    return schemas, models


def validate_args(name: str, model, str_params: Dict[str, bool], args: Dict[str, Any]) -> Dict[str, Any]:
    """FastMCP call_fn_with_arg_validation: JSON strings given for non-str fields are parsed first, then pydantic validation."""
    parsed = {}
    for k, v in args.items():
        if isinstance(v, str) and k in str_params and not str_params[k]:
            try:
                pv = json.loads(v)
                if not isinstance(pv, str):
                    v = pv
            except json.JSONDecodeError:
                pass
        parsed[k] = v
    try:
        inst = model.model_validate(parsed)
    except ValidationError as e:
        raise ValueError(str(e))
    return {k: getattr(inst, k) for k in inst.model_fields}
