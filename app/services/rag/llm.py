"""OpenAI 兼容 LLM 客户端（httpx 直连，不依赖 openai SDK）。

只依赖 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` 三项配置，
换成 DashScope、DeepSeek、智谱等任何兼容 `/chat/completions` 的厂商时无需改代码。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.config import settings


class LLMNotConfigured(RuntimeError):
    """三项必填配置不齐全。"""


class LLMError(RuntimeError):
    """请求失败、HTTP 错误或响应不可解析。"""


def _ensure_configured() -> None:
    if not settings.llm_configured:
        raise LLMNotConfigured(
            "LLM 未配置：请在 .env 中设置 LLM_BASE_URL、LLM_API_KEY、LLM_MODEL"
        )


def _endpoint() -> str:
    return settings.llm_base_url.rstrip("/") + "/chat/completions"


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }


def _payload(
    messages: list[dict[str, str]],
    stream: bool,
    temperature: float | None,
    max_tokens: int | None,
) -> dict[str, Any]:
    return {
        "model": settings.llm_model,
        "messages": messages,
        "stream": stream,
        "temperature": (
            settings.llm_temperature if temperature is None else temperature
        ),
        "max_tokens": settings.llm_max_tokens if max_tokens is None else max_tokens,
    }


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(settings.llm_timeout_seconds, connect=10.0)


def _network_error(exc: httpx.HTTPError) -> str:
    """网络层错误常常 str(exc) 为空，补上异常类型与目标地址，便于定位。"""
    detail = str(exc).strip() or type(exc).__name__
    return f"LLM 网络请求失败：{detail}（目标 {_endpoint()}）"


def _http_error(status: int, body: str) -> LLMError:
    """把厂商错误转成可操作提示：401/403 明确指向 Key，其余保留响应片段。"""
    if status in (401, 403):
        return LLMError(
            f"LLM 认证失败（{status}）：请检查 .env 的 LLM_API_KEY 是否为有效密钥"
            f"（当前配置地址 {settings.llm_base_url}，模型 {settings.llm_model}）"
        )
    if status == 404:
        return LLMError(
            f"LLM 接口不存在（404）：请检查 LLM_BASE_URL 是否包含正确的 /v1 等路径前缀"
            f"（当前 {_endpoint()}）"
        )
    return LLMError(f"LLM 返回 {status}：{body[:300]}")


async def stream_chat(
    messages: list[dict[str, str]],
    usage_out: dict[str, Any] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AsyncIterator[str]:
    """流式调用，逐段产出增量文本；若厂商在末尾返回 usage，会写入 usage_out。"""
    _ensure_configured()
    try:
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            async with client.stream(
                "POST",
                _endpoint(),
                json=_payload(messages, True, temperature, max_tokens),
                headers=_headers(),
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "ignore")
                    raise _http_error(response.status_code, body)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if usage_out is not None and chunk.get("usage"):
                        usage_out.update(chunk["usage"])
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    text = (choices[0].get("delta") or {}).get("content")
                    if text:
                        yield text
    except httpx.HTTPError as exc:
        raise LLMError(_network_error(exc)) from exc


async def complete_chat(
    messages: list[dict[str, str]],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """非流式调用，返回 (content, usage)。"""
    _ensure_configured()
    try:
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            response = await client.post(
                _endpoint(),
                json=_payload(messages, False, temperature, max_tokens),
                headers=_headers(),
            )
    except httpx.HTTPError as exc:
        raise LLMError(_network_error(exc)) from exc

    if response.status_code >= 400:
        raise _http_error(response.status_code, response.text)

    try:
        data = response.json()
    except ValueError as exc:
        raise LLMError("LLM 响应不是合法 JSON") from exc

    choices = data.get("choices") or []
    content = ""
    if choices:
        content = (choices[0].get("message") or {}).get("content") or ""
    return content, data.get("usage") or {}
