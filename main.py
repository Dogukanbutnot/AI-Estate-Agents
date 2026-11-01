"""
main.py — Emlak Lead Agent Sistemi
-----------------------------------
Çalıştırma: python main.py

Önce: pip install -r requirements.txt
      export ANTHROPIC_API_KEY=YOUR_API_KEY
"""

import json
import time

import tracing
from orchestrator import OrchestratorAgent
from agents import EvaluatorAgent


def run_pipeline(task: str, verbose: bool = True) -> dict:
    """Pipeline + değerlendirme tek bir trace_id altında çalışır."""
    print(f"\n{'='*60}")
    print(f"GÖREV: {task}")
    print(f"{'='*60}\n")

    with tracing.run(task):
        orch = OrchestratorAgent(max_retries=2, quality_threshold=75)
        result = orch.run(task)

        if result["status"] == "success":
            print(f"{'─'*60}\nEVALUATOR AJAN çalışıyor...")
            eval_result = EvaluatorAgent().run(result)
            result["evaluation"] = eval_result
            orch.reconcile_with_evaluator(result, eval_result)
            print(f"Sistem skoru: {eval_result['system_score']}/100")
            for agent_name, data in eval_result.get("agent_evaluations", {}).items():
                print(f"  {agent_name:<15}: {data.get('overall_score', 'N/A')}/100")

    if verbose:
        print(f"\nDurum    : {result['status'].upper()}")
        print(f"Girişim  : {result.get('attempts', 1)}")
        print(f"Süre     : {result.get('elapsed', 0):.2f} saniye")
        print(f"Kalite   : {result.get('quality_score', 0)}/100")

    found = (result.get("conflicts") or {}).get("detected", [])
    if found:
        by_id = {r["conflict_id"]: r for r in result["conflicts"]["resolutions"]}
        print(f"\nÇakışmalar ({len(found)}):")
        for c in found:
            r = by_id.get(c["id"], {})
            print(f"  [{c['severity']}] {c['rule']} ({c['user'] or '-'}) → {r.get('action')}"
                  f" | kazanan: {r.get('winner') or '—'}")
    else:
        print("\nÇakışma yok.")

    trace_file = result.get("trace_file")
    if trace_file:
        summary = tracing.summarize(tracing.read_trace(trace_file))
        print(f"\nTrace ID   : {result['trace_id']}")
        print(f"Trace dosya: {trace_file}  ({summary['events']} olay)")
        print(f"Olaylar    : {summary['events_by_agent']}")
        llm = summary["llm"]
        print(f"LLM        : {llm['calls']} çağrı, {llm['errors']} hata, "
              f"{llm['input_tokens']} giriş + {llm['output_tokens']} çıkış token, "
              f"toplam {llm['latency_ms'] / 1000:.1f}s gecikme")
        if summary["issues"]:
            print("Dikkat çeken olaylar:")
            for issue in summary["issues"]:
                print(f"  #{issue['seq']} {issue['agent']}/{issue['event']} {issue['ctx']}")
        print(f"Ayrıntı    : python tracing.py {trace_file}  (--llm: prompt/yanıtlar)")
    else:
        print("\n(Trace dosyası yazılamadı — stderr çıktısına bakın.)")

    return result


def main():
    task = (
        "Twitter'da son 24 saatte İstanbul'da ev almak isteyen kullanıcıları tara, "
        "niyet skorlarını hesapla ve en yüksek skorlu adaylara uygun DM taslakları yaz."
    )

    result = run_pipeline(task)

    with open("pipeline_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print("Tam sonuç → pipeline_result.json")

    drafts = result.get("results", {}).get("step_3_output", {}).get("dm_drafts", [])
    if drafts:
        first = drafts[0]
        print(f"\n{'─'*60}\nDM TASLAĞI (ilk lead)\n{'─'*60}")
        print(f"Kullanıcı : {first.get('user')}")
        print(f"Skor      : {first.get('score')}/100")
        print(f"Mesaj     :\n{first['dm_draft'].get('dm_text', '')}")


if __name__ == "__main__":
    main()
