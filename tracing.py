"""
tracing.py
----------
Tüm ajanlar için ortak izleme (trace) altyapısı.

Fikir:
- Bir pipeline çalıştırması = tek bir trace_id.
- Her olay traces/<trace_id>.jsonl dosyasına satır satır yazılır (her satır 1 JSON).
- Ajanlar trace_id taşımaz; _log() içinde tracing.emit() çağırırlar, aktif
  çalıştırma contextvars ile bulunur. Aktif çalıştırma yoksa emit() sessizce
  hiçbir şey yapmaz.
- İzleme asla pipeline'ı düşürmemeli: dosya açılamazsa veya yazılamazsa
  stderr'e uyarı yazılır, iş devam eder.

Kullanım:
    import tracing
    with tracing.run("görev açıklaması") as tracer:
        ...                      # ajanlar çalışır, olaylar dosyaya yazılır
    print(tracer.path)

    python tracing.py traces/<dosya>.jsonl     # zaman çizelgesi + özet

Not: Trace dosyaları tweet metni ve kullanıcı adı gibi kişisel veri içerebilir.
"""

import contextvars
import json
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager

_current = contextvars.ContextVar("emlaik_tracer", default=None)


def trace_dir() -> str:
    return os.environ.get("EMLAIK_TRACE_DIR", "traces")


class Tracer:
    """Tek bir çalıştırmanın olaylarını JSONL dosyasına yazar."""

    def __init__(self, directory: str = None):
        self.trace_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.directory = directory or trace_dir()
        self.path = None
        self._fh = None
        self._seq = 0
        self._ctx = {}
        self._lock = threading.Lock()
        self._seen = set()
        try:
            os.makedirs(self.directory, exist_ok=True)
            self.path = os.path.join(self.directory, f"{self.trace_id}.jsonl")
            self._fh = open(self.path, "a", encoding="utf-8")
        except OSError as e:
            self.path = None
            print(f"[tracing] trace dosyası açılamadı, bu çalıştırma için "
                  f"kayıt tutulmayacak: {e}", file=sys.stderr)

    def set_context(self, **fields):
        """Sonraki tüm olaylara eklenecek alanlar (ör. attempt, step). None → sil."""
        with self._lock:
            for key, value in fields.items():
                if value is None:
                    self._ctx.pop(key, None)
                else:
                    self._ctx[key] = value

    def emit(self, agent: str, event: str, data: dict = None):
        try:
            with self._lock:
                self._seq += 1
                record = {
                    "trace_id": self.trace_id,
                    "seq": self._seq,
                    "timestamp": time.time(),
                    "agent": agent,
                    "event": event,
                    "ctx": dict(self._ctx),
                    "data": data or {},
                }
                if self._fh:
                    self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    self._fh.flush()
        except Exception as e:   # izleme pipeline'ı düşürmemeli
            print(f"[tracing] olay yazılamadı ({agent}/{event}): {e}", file=sys.stderr)

    def close(self):
        if self._fh:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None


@contextmanager
def run(task: str = ""):
    """
    Yeni bir izleme çalıştırması başlatır. Zaten aktif bir çalıştırma varsa
    (ör. main.py dışarıda başlattıysa) onu yeniden kullanır; böylece
    Orchestrator ve Evaluator aynı trace_id altında toplanır.
    """
    existing = _current.get()
    if existing is not None:
        yield existing
        return

    tracer = Tracer()
    token = _current.set(tracer)
    started = time.time()
    status = "ok"
    tracer.emit("tracer", "run_start", {"task": task})
    try:
        yield tracer
    except BaseException as e:
        status = "error"
        tracer.emit("tracer", "run_error", {"error": repr(e)})
        raise
    finally:
        tracer.emit("tracer", "run_end", {
            "status": status,
            "duration_seconds": round(time.time() - started, 3),
        })
        tracer.close()
        _current.reset(token)


def emit(agent: str, event: str, data: dict = None):
    """Aktif çalıştırma varsa olayı yazar, yoksa hiçbir şey yapmaz."""
    tracer = _current.get()
    if tracer is not None:
        tracer.emit(agent, event, data)


def set_context(**fields):
    tracer = _current.get()
    if tracer is not None:
        tracer.set_context(**fields)


def current_trace_id():
    tracer = _current.get()
    return tracer.trace_id if tracer else None


def first_time(key: str) -> bool:
    """Aktif trace içinde bu anahtar ilk kez mi görülüyor? (aktif trace yoksa False)"""
    tracer = _current.get()
    if tracer is None or key in tracer._seen:
        return False
    tracer._seen.add(key)
    return True


# ─── Okuma / analiz ────────────────────────────────────────────────

def read_trace(path: str) -> list:
    """JSONL dosyasını olay listesine çevirir; bozuk satırları atlar."""
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
    return events


_ISSUE_MARKERS = ("error", "fallback", "retry", "gave_up", "rejected", "exceeded",
                  "failed", "suspected", "blocked", "conflict_detected", "aborted")


def summarize(events: list) -> dict:
    """Ajan başına olay sayısı, süre, LLM kullanımı ve dikkat çeken (hata/retry/fallback) olaylar."""
    llm = {"calls": 0, "errors": 0, "latency_ms": 0,
           "input_tokens": 0, "output_tokens": 0, "by_agent": {}}
    tools = {"calls": 0, "errors": 0, "by_tool": {}}
    if not events:
        return {"trace_id": None, "events": 0, "duration_seconds": 0,
                "events_by_agent": {}, "llm": llm, "tools": tools, "issues": []}
    by_agent, issues = {}, []
    for e in events:
        agent, name, d = e["agent"], e["event"], (e.get("data") or {})
        by_agent[agent] = by_agent.get(agent, 0) + 1
        if name == "llm_call":
            tin, tout = d.get("input_tokens") or 0, d.get("output_tokens") or 0
            llm["calls"] += 1
            llm["latency_ms"] += d.get("latency_ms") or 0
            llm["input_tokens"] += tin
            llm["output_tokens"] += tout
            row = llm["by_agent"].setdefault(agent, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            row["calls"] += 1
            row["input_tokens"] += tin
            row["output_tokens"] += tout
        elif name == "llm_error":
            llm["errors"] += 1
        elif name == "tool_call":
            tools["calls"] += 1
            tools["by_tool"][d.get("name")] = tools["by_tool"].get(d.get("name"), 0) + 1
        elif name == "tool_error":
            tools["errors"] += 1
        if any(marker in name for marker in _ISSUE_MARKERS):
            issues.append({"seq": e["seq"], "agent": agent, "event": name, "ctx": e.get("ctx", {})})
    return {
        "trace_id": events[0].get("trace_id"),
        "events": len(events),
        "duration_seconds": round(events[-1]["timestamp"] - events[0]["timestamp"], 3),
        "events_by_agent": by_agent,
        "llm": llm,
        "tools": tools,
        "issues": issues,
    }


def format_timeline(events: list) -> str:
    if not events:
        return "(boş trace)"
    t0 = events[0]["timestamp"]
    lines = []
    for e in events:
        ms = int((e["timestamp"] - t0) * 1000)
        where = ",".join(f"{k}={v}" for k, v in (e.get("ctx") or {}).items())
        data = json.dumps(e.get("data", {}), ensure_ascii=False, default=str)
        if len(data) > 80:
            data = data[:77] + "..."
        lines.append(f"{e['seq']:>3} +{ms:>5}ms {e['agent']:<13} {e['event']:<22} {where:<16} {data}")
    return "\n".join(lines)


def format_llm_calls(events: list, width: int = 300) -> str:
    """Her LLM çağrısı için: kim, hangi adımda, ne sordu, ne cevap aldı, ne kadar sürdü."""
    def clip(value):
        text = str(value).replace("\n", " ")
        return text if len(text) <= width else text[:width - 3] + "..."

    blocks = []
    for e in events:
        if e["event"] not in ("llm_call", "llm_error"):
            continue
        d = e.get("data") or {}
        where = ",".join(f"{k}={v}" for k, v in (e.get("ctx") or {}).items())
        head = f"#{e['seq']} {e['agent']}/{d.get('purpose')} [{where}]"
        if e["event"] == "llm_error":
            blocks.append(f"{head} HATA ({d.get('latency_ms')}ms): {d.get('error')}")
            continue
        msgs = d.get("messages") or []
        prompt = msgs[-1].get("content", "") if msgs else ""
        blocks.append(
            f"{head} {d.get('model')} {d.get('latency_ms')}ms "
            f"in={d.get('input_tokens')} out={d.get('output_tokens')}\n"
            f"   prompt: {clip(prompt)}\n   yanıt : {clip(d.get('response', ''))}"
        )
    return "\n".join(blocks) or "(LLM çağrısı yok)"


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        print("Kullanım: python tracing.py traces/<dosya>.jsonl [--llm]")
        sys.exit(1)
    evs = read_trace(args[0])
    if "--llm" in sys.argv:
        print(format_llm_calls(evs))
    else:
        print(format_timeline(evs))
        print("\nÖZET:", json.dumps(summarize(evs), ensure_ascii=False, indent=2))
