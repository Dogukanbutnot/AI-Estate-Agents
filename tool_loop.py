"""
tool_loop.py
------------
Anthropic "tool use" döngüsü: model araç çağırır → biz çalıştırırız → sonucu modele
geri veririz → model bitirene kadar tekrarlanır.

Tasarım ilkeleri:
  - Hata modelden SAKLANMAZ. Araç hata verirse model bunu is_error=true bir tool_result
    olarak görür (retryable / retry_after_seconds dahil) ve ne yapacağına (aynı çağrıyı
    tekrar mı, başka argüman mı, vazgeçmek mi) kendisi karar verir. Bu davranış ölçülür.
  - Ama döngü SINIRSIZ dönmez. Her zaman biter:
        end_turn           model araç çağırmadan bitirdi
        max_steps          adım sınırı doldu
        too_many_errors    üst üste N araç hatası (varsayılan 3)
        llm_error          LLM çağrısı patladı (kısmi sonuç korunur)
  - Model yalnızca KAYITLI araçları çağırabilir. Bilinmeyen araç hiçbir zaman çalışmaz.
  - Argümanlar çalıştırılmadan önce araç şemasına göre doğrulanır.
  - Aynı çağrı BAŞARIYLA yapıldıysa tekrarı reddedilir (sonuç değişmez); başarısız
    çağrının tekrarı serbesttir (geçici hata düzelmiş olabilir).
  - Araç başına çağrı bütçesi (max_calls) vardır.
  - retry_after bildiren hatalarda, bir sonraki LLM çağrısından önce (üst sınırlı) beklenir.

Trace olayları: tool_call, tool_result, tool_error, tool_backoff, loop_end, loop_aborted
"""

import json
import time
from collections import Counter

import tracing
from llm_client import call_llm

_sleep = time.sleep        # testlerde / laboratuvarda değiştirilebilir


class ToolError(Exception):
    """Araç işleyicisinin BEKLENEN hatası. Modele retryable bilgisiyle iletilir."""

    def __init__(self, message: str, retryable: bool = False, retry_after=None):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


class Tool:
    def __init__(self, name, description, schema, handler, max_calls=None):
        self.name, self.description = name, description
        self.schema, self.handler, self.max_calls = schema, handler, max_calls

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.schema}


# ─── Şema doğrulama (JSON Schema'nın küçük bir alt kümesi) ────────

def _is_type(value, type_name) -> bool:
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    return True


def validate_args(schema: dict, args) -> list:
    """Hata mesajları listesi döndürür (boş = geçerli)."""
    if not isinstance(args, dict):
        return ["argümanlar bir nesne olmalı"]
    props = schema.get("properties", {})
    errors = [f"'{name}' zorunlu" for name in schema.get("required", []) if name not in args]
    if schema.get("additionalProperties") is False:
        errors += [f"bilinmeyen alan '{k}'" for k in args if k not in props]
    for key, value in args.items():
        spec = props.get(key)
        if not spec:
            continue
        if "type" in spec and not _is_type(value, spec["type"]):
            errors.append(f"'{key}' {spec['type']} olmalı")
            continue
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"'{key}' şunlardan biri olmalı: {spec['enum']}")
        if isinstance(value, str):
            if "minLength" in spec and len(value) < spec["minLength"]:
                errors.append(f"'{key}' en az {spec['minLength']} karakter olmalı")
            if "maxLength" in spec and len(value) > spec["maxLength"]:
                errors.append(f"'{key}' en fazla {spec['maxLength']} karakter olmalı")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in spec and value < spec["minimum"]:
                errors.append(f"'{key}' en az {spec['minimum']} olmalı")
            if "maximum" in spec and value > spec["maximum"]:
                errors.append(f"'{key}' en fazla {spec['maximum']} olmalı")
    return errors


# ─── Yardımcılar ──────────────────────────────────────────────────

def _block_to_dict(block):
    """SDK blok nesnesini (veya dict'i) düz dict'e çevirir; yalnızca text ve tool_use korunur."""
    if isinstance(block, dict):
        kind = block.get("type")
        data = block
    else:
        kind = getattr(block, "type", None)
        data = {"text": getattr(block, "text", ""), "id": getattr(block, "id", None),
                "name": getattr(block, "name", None), "input": getattr(block, "input", None)}
    if kind == "text":
        text = data.get("text") or ""
        return {"type": "text", "text": text} if text else None   # API boş text bloğunu reddeder
    if kind == "tool_use":
        return {"type": "tool_use", "id": data["id"], "name": data["name"], "input": dict(data.get("input") or {})}
    return None


def _fail(agent, name, kind, message, retryable=False, retry_after=None, started=None, call_id=None):
    payload = {"error": message[:300], "kind": kind, "retryable": retryable,
               "retry_after_seconds": retry_after}
    tracing.emit(agent, "tool_error", {
        "name": name, "kind": kind, "error": message[:300], "retryable": retryable, "id": call_id,
        "latency_ms": int((time.perf_counter() - started) * 1000) if started else None})
    return False, json.dumps(payload, ensure_ascii=False), kind, retry_after if retryable else None


def _execute_one(agent, use, registry, counts, succeeded, max_chars):
    """(ok, modele giden metin, hata türü, bekleme) döndürür."""
    name, args, call_id = use["name"], use.get("input") or {}, use["id"]
    tracing.emit(agent, "tool_call", {"name": name, "input": args, "id": call_id})

    tool = registry.get(name)
    if tool is None:
        return _fail(agent, name, "unknown_tool",
                     f"Bilinmeyen araç '{name}'. Kullanılabilir araçlar: {sorted(registry)}", call_id=call_id)
    errors = validate_args(tool.schema, args)
    if errors:
        return _fail(agent, name, "invalid_arguments", "; ".join(errors), call_id=call_id)
    signature = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
    if signature in succeeded:
        return _fail(agent, name, "duplicate_call",
                     "Bu çağrıyı zaten başarıyla yaptın; sonuç değişmez. Farklı bir argüman dene veya bitir.",
                     call_id=call_id)
    if tool.max_calls is not None and counts[name] >= tool.max_calls:
        return _fail(agent, name, "budget_exceeded",
                     f"'{name}' için çağrı bütçesi ({tool.max_calls}) doldu.", call_id=call_id)

    counts[name] += 1
    started = time.perf_counter()
    try:
        out = tool.handler(**args)
    except ToolError as e:
        return _fail(agent, name, "tool_error", str(e), e.retryable, e.retry_after, started, call_id)
    except Exception as e:                      # işleyici beklenmedik şekilde patladı
        return _fail(agent, name, "handler_exception", f"{type(e).__name__}: {e}", False, None, started, call_id)

    text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, default=str)
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + "…[kısaltıldı]"
    succeeded.add(signature)
    tracing.emit(agent, "tool_result", {
        "name": name, "id": call_id, "ok": True, "size": len(text), "truncated": truncated,
        "latency_ms": int((time.perf_counter() - started) * 1000)})
    return True, text, None, None


# ─── Ana döngü ────────────────────────────────────────────────────

def run_tool_loop(client, agent: str, purpose: str, *, system: str, messages: list, tools: list,
                  model: str, max_tokens: int = 1024, max_steps: int = 8,
                  max_consecutive_errors: int = 3, max_result_chars: int = 3000,
                  max_backoff: float = 10.0) -> dict:
    registry = {t.name: t for t in tools}
    specs = [t.spec() for t in tools]
    counts, succeeded = Counter(), set()
    calls, consecutive, steps = [], 0, 0
    stop, final_text = None, ""

    for step in range(max_steps):
        steps = step + 1
        tracing.set_context(tool_step=step)
        try:
            response = call_llm(client, agent, purpose, model=model, max_tokens=max_tokens,
                                system=system, messages=messages, tools=specs)
        except Exception as e:
            stop = "llm_error"
            tracing.emit(agent, "loop_aborted", {"reason": stop, "error": repr(e)[:300], "step": step})
            break

        blocks = [b for b in (_block_to_dict(x) for x in (getattr(response, "content", None) or [])) if b]
        uses = [b for b in blocks if b["type"] == "tool_use"]
        if blocks:
            messages.append({"role": "assistant", "content": blocks})
        if not uses:
            stop = "end_turn"
            final_text = "".join(b["text"] for b in blocks if b["type"] == "text")
            break

        results, wait = [], 0.0
        for use in uses:
            ok, content, kind, retry_after = _execute_one(agent, use, registry, counts, succeeded, max_result_chars)
            calls.append({"step": step, "name": use["name"], "input": use["input"], "ok": ok, "kind": kind})
            block = {"type": "tool_result", "tool_use_id": use["id"], "content": content}
            if not ok:
                block["is_error"] = True
                if retry_after:
                    wait = max(wait, float(retry_after))
            results.append(block)
            consecutive = 0 if ok else consecutive + 1
        messages.append({"role": "user", "content": results})

        if consecutive >= max_consecutive_errors:
            stop = "too_many_errors"
            tracing.emit(agent, "loop_aborted", {"reason": stop, "consecutive_errors": consecutive, "step": step})
            break
        if wait > 0:
            seconds = min(wait, max_backoff)
            tracing.emit(agent, "tool_backoff", {"requested": wait, "seconds": seconds})
            _sleep(seconds)
    else:
        stop = "max_steps"
        tracing.emit(agent, "loop_aborted", {"reason": stop, "steps": steps})

    tracing.set_context(tool_step=None)
    errors = Counter(c["kind"] for c in calls if not c["ok"])
    tracing.emit(agent, "loop_end", {"stop": stop, "steps": steps, "tool_calls": len(calls),
                                     "errors": dict(errors)})
    return {"stop": stop, "steps": steps, "tool_calls": calls, "final_text": final_text,
            "messages": messages, "executed": dict(counts)}
