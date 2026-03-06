# -*- coding: utf-8 -*-
"""模型工厂（环境变量驱动优先）。"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langextract import factory

# 侧效导入：注册本地 OpenAI 兼容 provider（支持 extra_body）。
from openai_compatible_provider import OpenAICompatibleLanguageModel  # noqa: F401

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_OUTPUT_TOKENS = 8192


@dataclass
class ModelProfile:
  """模型配置 + 运行时标志。"""

  config: factory.ModelConfig
  use_schema_constraints: bool
  label: str


def _load_env() -> None:
  """尝试从项目根 .env 加载环境变量。"""
  try:
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.is_file():
      load_dotenv(env_path)
  except Exception:
    pass


def _env_text(key: str) -> str | None:
  raw = os.environ.get(key)
  if raw is None:
    return None
  value = raw.strip()
  return value or None


def _env_bool(key: str, default: bool | None = None) -> bool | None:
  raw = _env_text(key)
  if raw is None:
    return default
  normalized = raw.lower()
  if normalized in {"1", "true", "yes", "on"}:
    return True
  if normalized in {"0", "false", "no", "off"}:
    return False
  raise RuntimeError(f"{key} 必须是布尔值（true/false/1/0）")


def _env_int(key: str, default: int | None = None) -> int | None:
  raw = _env_text(key)
  if raw is None:
    return default
  try:
    return int(raw)
  except ValueError as e:
    raise RuntimeError(f"{key} 必须是整数") from e


def _env_float(key: str, default: float | None = None) -> float | None:
  raw = _env_text(key)
  if raw is None:
    return default
  try:
    return float(raw)
  except ValueError as e:
    raise RuntimeError(f"{key} 必须是数字") from e


def _env_json_dict(key: str) -> dict[str, Any] | None:
  raw = _env_text(key)
  if raw is None:
    return None
  try:
    parsed = json.loads(raw)
  except json.JSONDecodeError as e:
    raise RuntimeError(f"{key} 不是合法 JSON: {e}") from e
  if not isinstance(parsed, dict):
    raise RuntimeError(f"{key} 必须是 JSON 对象")
  return parsed


def _safe_label(raw: str) -> str:
  label = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).strip("._-")
  return label or "custom"


def get_default_model_selector() -> str:
  """默认模型选择器：统一走 env。"""
  _load_env()
  return "env"


def _build_openai_provider_kwargs(
    api_key: str,
    base_url: str,
    default_max_output_tokens: int,
) -> dict[str, Any]:
  """构建 OpenAI 兼容 provider_kwargs（含扩展请求参数）。"""
  kwargs: dict[str, Any] = {}

  # 高级透传参数（可直接注入 provider_kwargs），例如:
  # LLM_OPENAI_KWARGS='{"top_p":0.9,"response_format":{"type":"json_object"}}'
  if (extra := _env_json_dict("LLM_OPENAI_KWARGS")):
    kwargs.update(extra)

  kwargs["api_key"] = api_key
  kwargs["base_url"] = base_url
  kwargs.setdefault("timeout", DEFAULT_TIMEOUT_SECONDS)
  kwargs.setdefault("temperature", DEFAULT_TEMPERATURE)
  kwargs.setdefault("max_output_tokens", default_max_output_tokens)

  if (v := _env_float("LLM_TIMEOUT")) is not None:
    kwargs["timeout"] = v
  if (v := _env_float("LLM_TEMPERATURE")) is not None:
    kwargs["temperature"] = v
  if (v := _env_int("LLM_MAX_OUTPUT_TOKENS")) is not None:
    kwargs["max_output_tokens"] = v
  if (v := _env_float("LLM_TOP_P")) is not None:
    kwargs["top_p"] = v
  if (v := _env_float("LLM_FREQUENCY_PENALTY")) is not None:
    kwargs["frequency_penalty"] = v
  if (v := _env_float("LLM_PRESENCE_PENALTY")) is not None:
    kwargs["presence_penalty"] = v
  if (v := _env_int("LLM_SEED")) is not None:
    kwargs["seed"] = v
  if (v := _env_text("LLM_REASONING_EFFORT")):
    kwargs["reasoning_effort"] = v
  if (v := _env_json_dict("LLM_REASONING")):
    kwargs["reasoning"] = v
  if (v := _env_json_dict("LLM_RESPONSE_FORMAT")):
    kwargs["response_format"] = v
  if (v := _env_json_dict("LLM_EXTRA_HEADERS")):
    kwargs["extra_headers"] = v
  if (v := _env_json_dict("LLM_EXTRA_QUERY")):
    kwargs["extra_query"] = v

  # 扩展请求体：适用于 OpenAI 兼容但自定义字段的网关（如 enable_thinking）。
  extra_body: dict[str, Any] = {}
  if isinstance(kwargs.get("extra_body"), dict):
    extra_body.update(kwargs["extra_body"])
  if (v := _env_json_dict("LLM_EXTRA_BODY")):
    extra_body.update(v)
  thinking_flag = _env_bool("LLM_ENABLE_THINKING")
  if thinking_flag is not None:
    extra_body["enable_thinking"] = thinking_flag
  thinking_budget = _env_int("LLM_THINKING_BUDGET")
  if thinking_budget is not None:
    if thinking_budget < 100 or thinking_budget > 60000:
      raise RuntimeError("LLM_THINKING_BUDGET 必须在 [100, 60000] 范围内")
    extra_body["thinking_budget"] = thinking_budget
  if extra_body:
    kwargs["extra_body"] = extra_body

  return kwargs


def _build_openai_profile(
    model_id: str,
    label: str,
    default_max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> ModelProfile:
  api_key = _env_text("LLM_API_KEY")
  if not api_key:
    raise RuntimeError("未设置 LLM_API_KEY")

  base_url = _env_text("LLM_BASE_URL")
  if not base_url:
    raise RuntimeError("未设置 LLM_BASE_URL")

  provider_kwargs = _build_openai_provider_kwargs(
      api_key=api_key,
      base_url=base_url,
      default_max_output_tokens=default_max_output_tokens,
  )
  use_schema_constraints = bool(_env_bool("LLM_USE_SCHEMA_CONSTRAINTS", False))

  return ModelProfile(
      config=factory.ModelConfig(
          model_id=model_id,
          provider="OpenAICompatibleLanguageModel",
          provider_kwargs=provider_kwargs,
      ),
      use_schema_constraints=use_schema_constraints,
      label=label,
  )


def _build_gemini_profile() -> ModelProfile:
  api_key = _env_text("GOOGLE_API_KEY") or _env_text("LANGEXTRACT_API_KEY")
  if not api_key:
    raise RuntimeError("未设置 GOOGLE_API_KEY 或 LANGEXTRACT_API_KEY")

  return ModelProfile(
      config=factory.ModelConfig(
          model_id=_env_text("GEMINI_MODEL") or "gemini-2.0-flash",
          provider_kwargs={"api_key": api_key},
      ),
      use_schema_constraints=True,
      label="gemini",
  )


def get_model_config(model_name: str) -> ModelProfile:
  """根据名称/模型ID返回模型配置。"""
  _load_env()
  selected = (model_name or "").strip()
  if not selected:
    raise ValueError("model_name 不能为空")

  if selected == "gemini":
    return _build_gemini_profile()

  if selected == "env":
    model_id = _env_text("LLM_MODEL")
    if not model_id:
      raise RuntimeError("使用 --model env 时必须设置 LLM_MODEL")
    label = _env_text("LLM_LABEL") or _safe_label(model_id)
    return _build_openai_profile(
        model_id=model_id,
        label=label,
        default_max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    )

  # 直接按 model_id 处理，支持任意 OpenAI 兼容模型。
  label = _env_text("LLM_LABEL") or _safe_label(selected)
  return _build_openai_profile(
      model_id=selected,
      label=label,
      default_max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
  )
