"""
Orchestrator Agent
------------------
Sistemin beyni. Görevi alır, alt ajanlara böler,
sonuçları toplar ve kalite kontrolü yapar.

Kendi başına iş yapmaz — neyin, ne zaman, kim tarafından
yapılacağını bilir ve koordine eder.
"""

import os
import time
import json
import anthropic
import tracing
import conflicts
from llm_client import call_llm
from llm_utils import extract_json
from agents import ScoutAgent, ClassifierAgent, ConversationAgent, EvaluatorAgent, FormAgent

client = anthropic.Anthropic()

ORCHESTRATOR_SYSTEM = """
Sen bir emlak lead toplama sisteminin orkestratör ajanısın.
Sana bir görev geldiğinde şunları yap:

1. Görevi analiz et
2. Hangi alt ajanların sırayla çalışması gerektiğine karar ver
3. Adımları JSON olarak döndür

Kullanılabilir ajanlar:
- scout: Sosyal medyada aday kullanıcı bulmak için
- classifier: Kullanıcının niyet skorunu hesaplamak için
- conversation: Qualified lead'lere DM taslağı yazmak için
- form_agent: Qualified lead'ler için kişisel, tek kullanımlık form linki üretmek için
- evaluator: Diğer ajanların çıktısını değerlendirmek için (opsiyonel)

Çıktı formatı (SADECE JSON):
{
  "plan": [
    {"step": 1, "agent": "scout", "input": "<görev açıklaması>"},
    {"step": 2, "agent": "classifier", "input": "{{step_1_output}}"},
    {"step": 3, "agent": "conversation", "input": "{{step_2_output}}"},
    {"step": 4, "agent": "form_agent", "input": "{{step_2_output}}"}
  ],
  "reasoning": "<neden bu planı seçtin>",
  "expected_output": "<ne bekliyorsun>"
}
"""

AGENT_REGISTRY = {
    "scout":        ScoutAgent(use_mock=True, mode=os.environ.get("EMLAIK_SCOUT_MODE", "legacy")),
    "classifier":   ClassifierAgent(score_threshold=60),
    "conversation": ConversationAgent(),
    "form_agent":   FormAgent(),
    "evaluator":    EvaluatorAgent(),
}


class OrchestratorAgent:
    """
    Görev: Sistemi koordine etmek.

    Öğrenme hedefleri için bu ajanı gözlemle:
    - _make_plan: LLM ile plan üretimi, prompt kalitesi
    - _execute_plan: Delegasyon ve placeholder çözümü
    - _evaluate_quality: Retry kararı, eşik mantığı
    - self.trace: Tüm kararların zaman damgalı logu
    """

    def __init__(self, max_retries=2, quality_threshold=75):
        self.max_retries = max_retries
        self.quality_threshold = quality_threshold
        self.trace = []
        self.last_conflicts = {"detected": [], "resolutions": []}

    def run(self, task: str) -> dict:
        """
        Bir pipeline çalıştırmasını ortak bir trace_id altında yürütür.
        Dışarıda zaten aktif bir tracing.run() varsa onu kullanır.
        """
        self.trace = []   # her çalıştırma temiz başlar
        self.last_conflicts = {"detected": [], "resolutions": []}
        with tracing.run(task) as tracer:
            result = self._run(task)
            result["trace_id"] = tracer.trace_id
            result["trace_file"] = tracer.path
            return result

    def _run(self, task: str) -> dict:
        """
        Ana çalışma döngüsü:
        Görev al → Plan yap → Çalıştır → Değerlendir → Retry veya bitir
        """
        self._log("task_received", {"task": task})
        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            tracing.set_context(attempt=attempt)
            # 1. Plan üret
            plan_data = self._make_plan(task)
            self._log("plan_created", {
                "attempt": attempt,
                "plan": plan_data.get("plan", []),
                "reasoning": plan_data.get("reasoning", "")
            })

            # 2. Planı çalıştır
            results = self._execute_plan(plan_data.get("plan", []))
            self._log("plan_executed", {"step_count": len(results)})

            # 2b. Ajanlar arası çakışma kontrolü — kalite puanından ÖNCE, ki durdurulan
            # DM'ler kalite hesabında "gönderilebilir" sayılmasın.
            found, resolutions = conflicts.review_pipeline(results)
            for c in found:
                self._log("conflict_detected", {
                    "id": c["id"], "rule": c["rule"], "type": c["type"], "severity": c["severity"],
                    "agents": c["agents"], "detail": c["detail"]})
            for r in resolutions:
                self._log("conflict_resolved", {
                    "id": r["conflict_id"], "winner": r["winner"], "loser": r["loser"],
                    "action": r["action"], "affected": r["affected"]})
            self.last_conflicts = {"detected": found, "resolutions": resolutions}

            # 3. Kalite değerlendir
            quality_score = self._evaluate_quality(results)
            self._log("quality_evaluated", {
                "score": quality_score,
                "threshold": self.quality_threshold,
                "attempt": attempt
            })

            if quality_score >= self.quality_threshold:
                elapsed = time.time() - start_time
                self._log("success", {
                    "elapsed_seconds": round(elapsed, 2),
                    "final_score": quality_score
                })
                return {
                    "status": "success",
                    "results": results,
                    "quality_score": quality_score,
                    "attempts": attempt + 1,
                    "elapsed": round(elapsed, 2),
                    "conflicts": self.last_conflicts,
                    "trace": self.trace
                }

            self._log("retry_decision", {
                "reason": f"Kalite skoru {quality_score} < eşik {self.quality_threshold}",
                "attempt": attempt
            })

        # Max retry aşıldı
        elapsed = time.time() - start_time
        self._log("max_retries_exceeded", {"elapsed": round(elapsed, 2)})
        return {
            "status": "failed",
            "reason": "Maksimum deneme sayısı aşıldı",
            "attempts": self.max_retries + 1,
            "elapsed": round(elapsed, 2),
            "results": results,                 # son denemenin çıktısı: kanıt kaybolmasın
            "conflicts": self.last_conflicts,
            "trace": self.trace
        }

    def _make_plan(self, task: str) -> dict:
        """
        Orchestrator'ın ana LLM çağrısı.
        Görevi adımlara böler, hangi ajan ne yapacak?
        """
        try:
            response = call_llm(
                client, "orchestrator", "make_plan",
                model="claude-opus-4-5",
                max_tokens=1024,
                system=ORCHESTRATOR_SYSTEM,
                messages=[{"role": "user", "content": task}]
            )
            raw = response.content[0].text.strip()
            data = extract_json(raw)
            if not isinstance(data, dict) or not data.get("plan"):
                raise ValueError("yanıtta plan alanı yok")
            return data
        except ValueError:
            # Varsayılan plan: tam pipeline
            self._log("plan_parse_error", {"raw": raw[:100]})
            return {
                "plan": [
                    {"step": 1, "agent": "scout", "input": task},
                    {"step": 2, "agent": "classifier", "input": "{{step_1_output}}"},
                    {"step": 3, "agent": "conversation", "input": "{{step_2_output}}"},
                    {"step": 4, "agent": "form_agent", "input": "{{step_2_output}}"}
                ],
                "reasoning": "Varsayılan plan (parse hatası)",
                "expected_output": "Qualified lead DM taslakları"
            }
        except Exception as e:
            self._log("plan_api_error", {"error": str(e)})
            return {"plan": [], "reasoning": f"Hata: {str(e)}"}

    def _execute_plan(self, plan: list) -> dict:
        """
        Her adımı ilgili ajana delege eder.
        Önceki adımın çıktısını bir sonrakine taşır.
        """
        step_outputs = {}

        for step in plan:
            step_num = step.get("step", 0)
            agent_name = step.get("agent", "")
            raw_input = step.get("input", "")
            tracing.set_context(step=step_num)

            # Placeholder'ları çöz: {{step_1_output}} → gerçek değer
            resolved_input = self._resolve_placeholders(raw_input, step_outputs)

            self._log("step_start", {
                "step": step_num,
                "agent": agent_name,
                "input_preview": str(resolved_input)[:100]
            })

            agent = AGENT_REGISTRY.get(agent_name)
            if not agent:
                self._log("unknown_agent", {"agent": agent_name})
                continue

            output = agent.run(resolved_input)
            step_outputs[f"step_{step_num}_output"] = output

            self._log("step_complete", {
                "step": step_num,
                "agent": agent_name,
                "output_keys": list(output.keys()) if isinstance(output, dict) else "non-dict"
            })

        tracing.set_context(step=None)
        return step_outputs

    def _resolve_placeholders(self, text: str, outputs: dict) -> any:
        """
        {{step_1_output}} gibi referansları gerçek değerlerle değiştirir.
        Eğer tam eşleşme varsa dict döndürür, yoksa string döndürür.
        """
        if not isinstance(text, str):
            return text

        # Tam placeholder kontrolü: sadece "{{step_X_output}}" varsa dict döndür
        stripped = text.strip()
        if stripped.startswith("{{") and stripped.endswith("}}"):
            key = stripped[2:-2]
            if key in outputs:
                return outputs[key]

        # Kısmi placeholder: string içinde değiştir
        for key, val in outputs.items():
            placeholder = "{{" + key + "}}"
            if placeholder in text:
                text = text.replace(placeholder, json.dumps(val, ensure_ascii=False))

        return text

    def _evaluate_quality(self, results: dict) -> int:
        """
        Sonuçların kalitesini basit heuristiklerle değerlendirir.
        İleride Evaluator ajanına devredilebilir.
        """
        score = 0

        # Scout çalıştı mı?
        step1 = results.get("step_1_output", {})
        if isinstance(step1, dict) and step1.get("candidates"):
            score += 30
            candidate_count = len(step1["candidates"])
            if candidate_count >= 2:
                score += 10

        # Classifier qualified lead buldu mu?
        # Yalnızca çakışma nedeniyle durdurulmamış (needs_review olmayan) lead'ler sayılır
        step2 = results.get("step_2_output", {})
        sendable_leads = conflicts.sendable_leads(step2) if isinstance(step2, dict) else []
        if sendable_leads:
            score += 30
            if len(sendable_leads) >= 1:
                score += 10

        # Conversation DM taslağı ürettir mi?
        step3 = results.get("step_3_output", {})
        if isinstance(step3, dict) and conflicts.sendable_drafts(step3):
            score += 20

        return min(score, 100)

    def reconcile_with_evaluator(self, result: dict, eval_result: dict) -> list:
        """
        Orchestrator'ın kalite puanı ile Evaluator'ın değerlendirmesini karşılaştırır.
        Karar Orchestrator'da kalır (bkz. conflicts.RULES); uyuşmazlık kayda geçer.
        """
        found = conflicts.check_quality_disagreement(
            result.get("quality_score"), eval_result, self.quality_threshold)
        resolutions = conflicts.apply_resolutions({}, found)
        for c in found:
            self._log("conflict_detected", {
                "id": c["id"], "rule": c["rule"], "type": c["type"], "severity": c["severity"],
                "agents": c["agents"], "detail": c["detail"]})
        for r in resolutions:
            self._log("conflict_resolved", {
                "id": r["conflict_id"], "winner": r["winner"], "loser": r["loser"],
                "action": r["action"], "affected": r["affected"]})
        bucket = result.setdefault("conflicts", {"detected": [], "resolutions": []})
        bucket["detected"] += found
        bucket["resolutions"] += resolutions
        return found

    def get_trace_summary(self) -> str:
        """İnsan okunabilir trace özeti."""
        lines = [f"=== ORCHESTRATOR TRACE ({len(self.trace)} olay) ==="]
        for entry in self.trace:
            ts = time.strftime("%H:%M:%S", time.localtime(entry["timestamp"]))
            lines.append(f"[{ts}] {entry['event']}: {str(entry['data'])[:80]}")
        return "\n".join(lines)

    def _log(self, event: str, data: dict):
        tracing.emit("orchestrator", event, data)
        self.trace.append({
            "agent": "orchestrator",
            "event": event,
            "timestamp": time.time(),
            "data": data
        })
