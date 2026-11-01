"""
llm_client.py
-------------
Tüm ajanların LLM çağrıları için tek geçit noktası.

call_llm() orijinal client.messages.create() ile aynı yanıtı döndürür; ek
olarak aktif trace'e şunları yazar:
  llm_call  → purpose, model, system_sha, mesajlar, ham yanıt, token sayıları,
              stop_reason, gecikme (ms)
  llm_error → çağrı patlarsa (hata yine de yukarı fırlatılır)

Böylece ajanların "ne sorulduğu, ne cevap geldiği, ne kadar sürdüğü" sonradan
incelenebilir. İzleme hatası çağrıyı asla bozmaz.

Ortam değişkeni:
  EMLAIK_TRACE_MAX_CHARS  trace'e yazılacak metin başına üst sınır (varsayılan 4000)
"""

import hashlib
import os
import sys
import time

import tracing


def _max_chars() -> int:
    try:
        return int(os.environ.get("EMLAIK_TRACE_MAX_CHARS", "4000"))
    except ValueError:
        return 4000


def _clip(value, limit: int):
    text = value if isinstance(value, str) else str(value)
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _response_text(response) -> str:
    parts = []
    for block in getattr(response, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


def _tool_uses(response) -> list:
    uses = []
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "tool_use":
            uses.append({"name": getattr(block, "name", None), "input": getattr(block, "input", None)})
    return uses


def call_llm(client, agent: str, purpose: str, **kwargs):
    """
    client.messages.create(**kwargs) çağırır ve trace'e kaydeder.
    agent: çağıran ajanın adı, purpose: çağrının amacı (ör. "classify_intent").
    """
    system = kwargs.get("system") or ""
    system_sha = hashlib.sha1(str(system).encode("utf-8")).hexdigest()[:10]
    started = time.perf_counter()

    try:
        response = client.messages.create(**kwargs)
    except Exception as e:
        tracing.emit(agent, "llm_error", {
            "purpose": purpose,
            "model": kwargs.get("model"),
            "system_sha": system_sha,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "error": repr(e)[:500],
        })
        raise

    latency_ms = int((time.perf_counter() - started) * 1000)

    try:
        limit = _max_chars()
        messages = []
        truncated = False
        for m in kwargs.get("messages") or []:
            content, cut = _clip(m.get("content", ""), limit)
            truncated = truncated or cut
            messages.append({"role": m.get("role"), "content": content})

        text, resp_cut = _clip(_response_text(response), limit)
        usage = getattr(response, "usage", None)

        data = {
            "purpose": purpose,
            "model": kwargs.get("model"),
            "max_tokens": kwargs.get("max_tokens"),
            "system_sha": system_sha,
            "messages": messages,
            "response": text,
            "stop_reason": getattr(response, "stop_reason", None),
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "latency_ms": latency_ms,
            "truncated": truncated or resp_cut,
            "tools": [t.get("name") for t in kwargs.get("tools") or []],
            "tool_uses": _tool_uses(response),
        }
        # System prompt her çağrıda tekrar yazılmasın: trace başına ilk görüşte tam metin
        if tracing.first_time(f"system:{system_sha}"):
            data["system"], _ = _clip(system, limit)

        tracing.emit(agent, "llm_call", data)
    except Exception as e:   # izleme çağrıyı bozmasın
        print(f"[llm_client] trace kaydı yazılamadı: {e}", file=sys.stderr)

    return response
