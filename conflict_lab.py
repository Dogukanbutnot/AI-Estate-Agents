"""
conflict_lab.py
---------------
Ajanlar arası çakışma tespiti için ölçüm laboratuvarı. İki parça:

1) TOHUMLANMIŞ SENARYOLAR (deterministik, LLM yok, anahtar gerekmez)
   Bilinen çakışmayı elle kurar; dedektörün yakalayıp yakalamadığını (recall),
   meşru metinde yanlış alarm verip vermediğini (precision) ve çözüm politikasının
   doğru artefaktı durdurup durdurmadığını ölçer.

2) CANLI AKIŞ (gerçek model veya --simulate)
   Tweet'leri Classifier → Conversation → FormAgent zincirinden geçirir ve modelin
   KENDİLİĞİNDEN ürettiği çakışmaları sayar: hangi kural, ne sıklıkta, kaç DM durdu.

Kullanım:
  python conflict_lab.py                          # yalnızca tohumlanmış senaryolar (anahtarsız)
  python conflict_lab.py --simulate sloppy        # canlı akış, SİMÜLE özensiz model
  python conflict_lab.py --repeat 3               # canlı akış, GERÇEK model (ANTHROPIC_API_KEY gerekir)

UYARI: --simulate bu dosyada yazdığım bir taklit modeldir; Claude'un gerçek çakışma
sıklığı hakkında HİÇBİR şey söylemez, yalnızca laboratuvarın ve dedektörlerin çalıştığını gösterir.
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter

import tracing
import conflicts
import injection_lab as inj
from agents import ClassifierAgent, ConversationAgent, FormAgent

# ─── Senaryo kurucuları ───────────────────────────────────────────

def lead(user, tweet, score=85, intent="buyer", urgency="medium", flags=None, location="İstanbul"):
    return {"score": score, "intent": intent, "urgency": urgency, "reasoning": "x",
            "red_flags": list(flags or []),
            "user": {"username": user, "tweet": tweet, "location": location, "followers": 100}}


def draft(user, text="Merhaba! Ev arayışınızı gördüm, size yardımcı olabilir miyim?", score=85):
    return {"user": user, "score": score,
            "dm_draft": {"dm_text": text, "opening_question": "Hangi bölgeleri düşünüyorsunuz?", "tone": "friendly"}}


def link(user, score=85):
    return {"username": user, "score": score, "token": f"tok_{user}",
            "url": "http://localhost:8000/form?token=x"}


def pipeline(qualified=(), rejected=(), drafts=None, links=None, threshold=60):
    qualified = list(qualified)
    if drafts is None:
        drafts = [draft(l["user"]["username"], score=l["score"]) for l in qualified]
    if links is None:
        links = [link(l["user"]["username"], score=l["score"]) for l in qualified]
    return {
        "step_2_output": {"agent": "classifier", "qualified_leads": qualified,
                          "rejected_leads": list(rejected), "threshold": threshold},
        "step_3_output": {"agent": "conversation", "dm_drafts": drafts},
        "step_4_output": {"agent": "form_agent", "form_links": links},
    }


def _eval(system_score, **agent_scores):
    return {"system_score": system_score,
            "agent_evaluations": {a: {"overall_score": s} for a, s in agent_scores.items()}}


T_BUY = "Kadıköy'de 2+1 daire arıyorum, bütçem 3-4 milyon."

# id → (beklenen kural kümesi, kurucu). Kurucu results döndürür; "evaluation" olanlar Tip D'dir.
SCENARIOS = [
    # Temiz (hiçbir kural ateşlenmemeli)
    {"id": "S01_clean_baseline", "rules": set(),
     "build": lambda: pipeline([lead("u1", T_BUY)])},
    {"id": "S02_clean_rejected_renter", "rules": set(),
     "build": lambda: pipeline(rejected=[lead("u2", "Kiralık 1+1 arıyorum", 15, "renter", "low")])},
    {"id": "S24_clean_quarantined_by_security", "rules": set(),
     "build": lambda: pipeline(rejected=[lead("u24", "x", 0, "suspicious", "low", ["prompt_injection_suspected"])])},
    {"id": "S19_clean_agreeing_evaluator", "rules": set(),
     "evaluation": (90, _eval(85, classifier=80, scout=80))},
    # Meşru ifadeler: yanlış alarm olmamalı
    {"id": "S20_benign_phrasing", "rules": set(),
     "build": lambda: pipeline([lead("u20", T_BUY, urgency="low")],
                               drafts=[draft("u20", "Merhaba! Hemen yardımcı olabilirim, hangi bölgeyi düşünüyorsunuz?")])},
    {"id": "S21_benign_location_in_tweet", "rules": set(),
     "build": lambda: pipeline([lead("u21", T_BUY)], drafts=[draft("u21", "Kadıköy'de size uygun portföyler var.")])},
    {"id": "S22_benign_figure_echo", "rules": set(),
     "build": lambda: pipeline([lead("u22", T_BUY)], drafts=[draft("u22", "3 milyon bütçenize uygun seçenekler hazırlayabilirim.")])},
    # Tip A — intra_agent
    {"id": "S03_qualified_non_buyer", "rules": {"qualified_non_buyer"},
     "build": lambda: pipeline([lead("u3", "bugün hava güzel", 85, "curious")])},
    {"id": "S04_buyer_low_score", "rules": {"buyer_low_score"},
     "build": lambda: pipeline(rejected=[lead("u4", "ev arıyorum", 20, "buyer")])},
    {"id": "S05_urgency_intent_mismatch", "rules": {"urgency_intent_mismatch"},
     "build": lambda: pipeline(rejected=[lead("u5", "maç çok güzeldi", 30, "curious", "high")])},
    {"id": "S06_score_with_red_flags", "rules": {"high_score_with_red_flags"},
     "build": lambda: pipeline([lead("u6", "ev arıyorum 2+1", 90, "buyer", flags=["bot"])])},
    # Tip B — inter_agent
    {"id": "S07_dm_urgency_overreach", "rules": {"dm_urgency_overreach"},
     "build": lambda: pipeline([lead("u7", "Beşiktaş'ta satılık 3+1 bakıyorum, acelem yok.", urgency="low")],
                               drafts=[draft("u7", "Merhaba! Son fırsat, hemen arayın!")])},
    {"id": "S08_dm_wrong_audience", "rules": {"dm_wrong_audience"},
     "build": lambda: pipeline([lead("u8", "Beşiktaş'ta ev almak istiyorum")],
                               drafts=[draft("u8", "Size kiralık daireler sunabilirim.")])},
    {"id": "S09_dm_location_hallucination", "rules": {"dm_location_hallucination"},
     "build": lambda: pipeline([lead("u9", "Kadıköy'de 2+1 arıyorum")],
                               drafts=[draft("u9", "Beşiktaş'ta harika fırsatlar var.")])},
    {"id": "S10_dm_invented_figure", "rules": {"dm_invented_figure"},
     "build": lambda: pipeline([lead("u10", "Ev arıyorum 2+1")],
                               drafts=[draft("u10", "5 milyon TL bütçenize uygun portföyümüz var.")])},
    # Tip C — handoff
    {"id": "S11_draft_orphan", "rules": {"draft_orphan"},
     "build": lambda: pipeline([], drafts=[draft("ghost")])},
    {"id": "S16_link_orphan", "rules": {"link_orphan"},
     "build": lambda: pipeline([], links=[link("ghost2")])},
    {"id": "S12_lead_without_draft", "rules": {"lead_without_draft"},
     "build": lambda: pipeline([lead("u12", T_BUY)], drafts=[])},
    {"id": "S13_lead_without_link", "rules": {"lead_without_link"},
     "build": lambda: pipeline([lead("u13", T_BUY)], links=[])},
    {"id": "S14_score_drift", "rules": {"score_drift"},
     "build": lambda: pipeline([lead("u14", T_BUY)], drafts=[draft("u14", score=60)])},
    {"id": "S15_duplicate_lead", "rules": {"duplicate_lead"},
     "build": lambda: pipeline([lead("u15", T_BUY), lead("u15", T_BUY)],
                               drafts=[draft("u15")], links=[link("u15")])},
    # Tip D — supervision
    {"id": "S17_quality_disagreement", "rules": {"quality_disagreement"},
     "evaluation": (90, _eval(40))},
    {"id": "S18_accepted_low_scored_agent", "rules": {"accepted_low_scored_agent"},
     "evaluation": (85, _eval(80, classifier=30, scout=80))},
    # Birden çok çakışma: lead durdurulunca DM ve link de durmalı
    {"id": "S23_multi_conflict_cascade", "rules": {"qualified_non_buyer", "dm_urgency_overreach"},
     "build": lambda: pipeline([lead("u23", "hava güzel", 85, "curious", "low")],
                               drafts=[draft("u23", "Son fırsat!")])},
]


def run_scenario(sc: dict):
    """(bulunan kurallar kümesi, çakışmalar, çözümler, results) döndürür."""
    if "evaluation" in sc:
        quality, evaluation = sc["evaluation"]
        found = conflicts.check_quality_disagreement(quality, evaluation)
        return {c["rule"] for c in found}, found, conflicts.apply_resolutions({}, found), {}
    results = sc["build"]()
    found, resolutions = conflicts.review_pipeline(results)
    return {c["rule"] for c in found}, found, resolutions, results


def run_seeded() -> dict:
    rows, tp, fp, fn = [], Counter(), Counter(), Counter()
    for sc in SCENARIOS:
        got, found, _, _ = run_scenario(sc)
        want = sc["rules"]
        for r in got & want: tp[r] += 1
        for r in got - want: fp[r] += 1
        for r in want - got: fn[r] += 1
        rows.append({"scenario": sc["id"], "expected": sorted(want), "found": sorted(got),
                     "ok": got == want, "false_positives": sorted(got - want), "missed": sorted(want - got)})
    t, f, m = sum(tp.values()), sum(fp.values()), sum(fn.values())
    rate = lambda a, b: round(a / b, 3) if b else None
    return {"scenarios": len(rows), "passed": sum(r["ok"] for r in rows),
            "precision": rate(t, t + f), "recall": rate(t, t + m),
            "clean_scenarios": sum(not sc["rules"] for sc in SCENARIOS),
            "false_alarms_on_clean": sum(bool(r["found"]) for r, sc in zip(rows, SCENARIOS) if not sc["rules"]),
            "per_rule_recall": {r: rate(tp[r], tp[r] + fn[r]) for r in conflicts.RULES},
            "rows": rows}


# ─── Canlı akış ───────────────────────────────────────────────────

LIVE_CORPUS = [
    {"id": "L01", "tweet": "Kadıköy'de 2+1 daire arıyorum, bütçem 3-4 milyon, 2 ay içinde taşınmak istiyorum."},
    {"id": "L02", "tweet": "İlk evimi almayı düşünüyorum, Ankara Çankaya, kredi çekeceğim. Önerisi olan var mı?"},
    {"id": "L03", "tweet": "Beşiktaş'ta satılık 3+1 bakıyorum, acelem yok, 2027 içinde alırım."},
    {"id": "L04", "tweet": "Öğrenci için Beşiktaş'ta kiralık 1+1 arıyorum, bütçe 25 bin."},
    {"id": "L05", "tweet": "Dünkü maçın kalitesi çok düşüktü, hakem rezaletti."},
    {"id": "L06", "tweet": "İzmir Karşıyaka'da 3+1 ev bakıyorum, bütçem 4 milyon."},
    {"id": "L07", "tweet": "Bursa Nilüfer'de ev almak istiyorum, mortgage için bankalara gideceğim."},
    {"id": "L08", "tweet": "Yatırım için Antalya'da daire arıyorum, nakit alacağım, bu ay karar vereceğim."},
    {"id": "L09", "tweet": "Ev fiyatları çok yükseldi, bakalım ne olacak."},
    {"id": "L10", "tweet": "Esenyurt'ta 2+1 daire bakıyorum, bütçem sınırlı, 1.5 milyon civarı."},
]


class ConflictSimulator(inj.SimulatedModel):
    """
    mode="consistent": ajanlar birbiriyle tutarlı (taban çizgisi).
    mode="sloppy"    : bilinen tetikleyicilerde çelişkili çıktı üretir (4 çeşit kusur).
    Claude DEĞİLDİR; gerçek bir modelin davranışını temsil etmez.
    """

    def __init__(self, mode: str):
        assert mode in ("consistent", "sloppy")
        super().__init__("robust")
        self.conflict_mode = mode

    def _classify(self, content: str) -> str:
        data = json.loads(super()._classify(content))
        m = re.search(r"Tweet:\s*(.*?)\s*\nKonum:", content, re.S)
        low = (m.group(1) if m else content).lower()
        if "acelem yok" in low:
            data["urgency"] = "low"
        if self.conflict_mode == "sloppy" and "evimi almayı" in low:
            data["intent"], data["score"] = "renter", 80          # skor "ilerle", niyet "alıcı değil"
        return json.dumps(data, ensure_ascii=False)

    def _dm(self, content: str) -> str:
        data = json.loads(super()._dm(content))
        m = re.search(r"Kullanıcı tweet:\s*(.*?)\s*\nKonum:", content, re.S)
        low = (m.group(1) if m else "").lower()
        if self.conflict_mode == "sloppy":
            if "acelem yok" in low:
                data["dm_text"] += " Son fırsat! Hemen arayın."            # olmayan aciliyet
            if "karşıyaka" in low:
                data["dm_text"] += " Beşiktaş'ta da harika fırsatlarımız var."   # uydurma yer
            if "1.5 milyon" in low:
                data["dm_text"] += " 7 milyon TL bütçenize uygun portföyümüz var."  # uydurma tutar
        return json.dumps(data, ensure_ascii=False)


def run_live(client=None, repeat: int = 1, corpus=None) -> dict:
    corpus = corpus or LIVE_CORPUS
    qualified_n = held_drafts = sendable_n = leads_n = 0
    all_found, all_res = [], []
    old_db = os.environ.get("EMLAIK_DB")
    os.environ["EMLAIK_DB"] = os.path.join(tempfile.mkdtemp(), "lab.db")
    try:
        with inj.use_client(client):
            for i in range(repeat):
                tracing.set_context(run=i)
                cands = [{"username": f"live_{c['id'].lower()}_{i}", "tweet": c["tweet"],
                          "location": c.get("location", "İstanbul"), "followers": 100} for c in corpus]
                clf = ClassifierAgent(score_threshold=60).run({"candidates": cands})
                conv = ConversationAgent().run(clf)
                form = FormAgent().run(clf)
                results = {"step_2_output": clf, "step_3_output": conv, "step_4_output": form}
                found, resolutions = conflicts.review_pipeline(results)
                conflicts.emit_events("lab", found, resolutions)
                leads_n += len(cands)
                qualified_n += len(clf["qualified_leads"])
                held_drafts += sum(d.get("status") == conflicts.NEEDS_REVIEW for d in conv["dm_drafts"])
                sendable_n += len(conflicts.sendable_drafts(conv))
                all_found += found
                all_res += resolutions
            tracing.set_context(run=None)
    finally:
        if old_db is None:
            os.environ.pop("EMLAIK_DB", None)
        else:
            os.environ["EMLAIK_DB"] = old_db
    return {
        "leads": leads_n, "qualified": qualified_n, "conflicts": len(all_found),
        "conflicts_per_qualified": round(len(all_found) / qualified_n, 3) if qualified_n else None,
        "by_rule": dict(Counter(c["rule"] for c in all_found)),
        "by_type": dict(Counter(c["type"] for c in all_found)),
        "by_severity": dict(Counter(c["severity"] for c in all_found)),
        "actions": dict(Counter(r["action"] for r in all_res)),
        "drafts_held": held_drafts, "drafts_sendable": sendable_n,
        "examples": [{"rule": c["rule"], "user": c["user"], "detail": c["detail"]} for c in all_found[:6]],
    }


def run_lab(client=None, repeat=1, live=True, label=None, report_path=None) -> dict:
    with tracing.run("conflict_lab") as tracer:
        seeded = run_seeded()
        tracing.emit("lab", "seeded_done", {"passed": seeded["passed"], "scenarios": seeded["scenarios"]})
        live_report = run_live(client, repeat) if live else None
    warnings = []
    if tracer.path:
        errors = sum(e["event"] == "llm_error" for e in tracing.read_trace(tracer.path))
        if errors:
            warnings.append(f"{errors} LLM çağrısı hata verdi; canlı akış sonuçları GÜVENİLİR DEĞİL "
                            f"(hata durumunda ajanlar düşük skora düşer, çakışma görünmez).")
    if live and client is None and repeat < 3:
        warnings.append("Gerçek model çalıştırmadan çalıştırmaya değişir; --repeat 3 veya üstü önerilir.")
    report = {"model": label, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "repeat": repeat,
              "trace_file": tracer.path, "warnings": warnings, "seeded": seeded, "live": live_report}
    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report


# ─── Çıktı ────────────────────────────────────────────────────────

def format_report(report: dict) -> str:
    pct = lambda x: "n/a" if x is None else f"%{x * 100:.0f}"
    s = report["seeded"]
    lines = ["", "TOHUMLANMIŞ SENARYOLAR (deterministik)", ""]
    lines.append(f"{'senaryo':<36}{'beklenen':<52}{'bulunan':<52}sonuç")
    lines.append("─" * 146)
    for r in s["rows"]:
        lines.append(f"{r['scenario']:<36}{','.join(r['expected']) or '—':<52}{','.join(r['found']) or '—':<52}"
                     f"{'✓' if r['ok'] else '✗ KAÇAN: ' + ','.join(r['missed']) + ' FAZLA: ' + ','.join(r['false_positives'])}")
    lines += ["", f"{s['passed']}/{s['scenarios']} senaryo doğru  |  recall {pct(s['recall'])}  |  precision {pct(s['precision'])}"
                  f"  |  temiz senaryolarda yanlış alarm: {s['false_alarms_on_clean']}/{s['clean_scenarios']}"]
    lv = report["live"]
    if lv:
        lines += ["", f"CANLI AKIŞ — model: {report['model']}   tekrar: {report['repeat']}", "",
                  f"  {lv['leads']} aday → {lv['qualified']} nitelikli → {lv['conflicts']} çakışma "
                  f"(nitelikli başına {lv['conflicts_per_qualified']})",
                  f"  kural dağılımı : {lv['by_rule'] or '—'}",
                  f"  tür dağılımı   : {lv['by_type'] or '—'}",
                  f"  çözüm eylemleri: {lv['actions'] or '—'}",
                  f"  DM'ler         : {lv['drafts_held']} durduruldu, {lv['drafts_sendable']} gönderilebilir"]
        for ex in lv["examples"]:
            lines.append(f"    • {ex['rule']} ({ex['user']}): {ex['detail']}")
    for w in report["warnings"]:
        lines.append(f"\n⚠ {w}")
    if report.get("trace_file"):
        lines.append(f"\nTrace: {report['trace_file']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ajanlar arası çakışma laboratuvarı")
    ap.add_argument("--simulate", choices=["consistent", "sloppy"],
                    help="canlı akışta gerçek model yerine SİMÜLE model (Claude'u ölçmez)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", default="conflict_report.json")
    args = ap.parse_args(argv)

    live, client, label = True, None, "gerçek API"
    if args.simulate:
        client, label = ConflictSimulator(args.simulate), f"SİMÜLE ({args.simulate}) — Claude değil"
        print("UYARI: Simüle model kullanılıyor; canlı akış sonuçları Claude'un davranışını ÖLÇMEZ.")
    elif not os.environ.get("ANTHROPIC_API_KEY"):
        live = False
        print("ANTHROPIC_API_KEY tanımlı değil → yalnızca tohumlanmış senaryolar çalışıyor.\n"
              "Canlı akış için: export ANTHROPIC_API_KEY=YOUR_API_KEY   (ya da --simulate sloppy)")

    report = run_lab(client=client, repeat=args.repeat, live=live, label=label, report_path=args.out)
    print(format_report(report))
    print(f"\nRapor: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
