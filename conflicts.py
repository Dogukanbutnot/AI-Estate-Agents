"""
conflicts.py
------------
Ajanlar arası çakışma tespiti ve çözümü. LLM KULLANMAZ: tüm kurallar deterministiktir.

Dört çakışma türü:
  intra_agent  Tek bir ajanın kendi çıktısı çelişiyor (Classifier: skor "ilerle" diyor,
               niyet "alıcı değil" diyor)
  inter_agent  İki ajanın kararı çelişiyor (Classifier aciliyeti "düşük" derken
               Conversation "son fırsat, hemen arayın" yazıyor)
  handoff      Adımlar arası devirde veri bozuluyor (DM'i olan ama nitelikli olmayan
               lead, skorun adımlar arasında değişmesi)
  supervision  Denetçi ile hakem ayrışıyor (Orchestrator "kalite 90" diyor,
               Evaluator "sistem 40" diyor)

"Çakışırsa kim kazanır?" sorusunun cevabı RULES tablosundaki winner/loser/action/policy
alanlarıdır. Üç ilke:
  1. Olgu (intent, urgency, skor) → tweete en yakın uzman ajan (Classifier) kazanır;
     DM üslubu bu olguları aşamaz.
  2. Hakem yoksa (ajan kendiyle çelişiyor) → güvenli taraf: gönderim durdurulur,
     insan incelemesine alınır ("fail closed").
  3. LLM yargıcı (Evaluator) danışmandır; geçti/kaldı kararı deterministik kanıta
     dayanan Orchestrator'da kalır, ama uyuşmazlık kayda geçer.

Çözüm artefakt silmez, işaretler: status="needs_review" + conflicts=[id...].
Gönderilecek DM'ler için sendable_drafts() kullanılmalıdır.
"""

import re

import tracing

# ─── Kural tablosu: tespit + çözüm politikası tek yerde ───────────

RULES = {
    # Tip A — intra_agent
    "qualified_non_buyer": dict(
        type="intra_agent", severity="high", agents=["classifier"],
        winner=None, loser=None, action="hold_lead",
        policy="Skor 'ilerle' derken niyet 'alıcı değil' diyor; arada hakem yok. Güvenli taraf: "
               "lead ve ona bağlı DM/link gönderilmeden insan incelemesine alınır."),
    "high_score_with_red_flags": dict(
        type="intra_agent", severity="high", agents=["classifier"],
        winner=None, loser=None, action="hold_lead",
        policy="Yüksek skor ile bot/spam/enjeksiyon işareti birlikte olamaz; hakem yok. "
               "Güvenli taraf: lead insan incelemesine alınır."),
    "buyer_low_score": dict(
        type="intra_agent", severity="medium", agents=["classifier"],
        winner=None, loser=None, action="flag_review",
        policy="Niyet 'alıcı' ama skor çok düşük: lead zaten elendiği için gönderim riski yok; "
               "kaçırılmış alıcı olabilir, incelemeye işaretlenir."),
    "urgency_intent_mismatch": dict(
        type="intra_agent", severity="medium", agents=["classifier"],
        winner=None, loser=None, action="flag_review",
        policy="Aciliyet 'yüksek' ama niyet 'alıcı değil'; gönderim kararını etkilemez "
               "(nitelik kontrolü ayrıdır), incelemeye işaretlenir."),
    # Tip B — inter_agent (Classifier olguyu belirler, Conversation üslubu)
    "dm_urgency_overreach": dict(
        type="inter_agent", severity="medium", agents=["classifier", "conversation"],
        winner="classifier", loser="conversation", action="quarantine_draft",
        policy="Aciliyet olgusunu tweete en yakın uzman (Classifier) belirler; DM, olmayan "
               "bir aciliyeti baskı diliyle yaratamaz. Taslak gönderilmez."),
    "dm_wrong_audience": dict(
        type="inter_agent", severity="high", agents=["classifier", "conversation"],
        winner="classifier", loser="conversation", action="quarantine_draft",
        policy="Niyet (alıcı/kiracı) Classifier'ın olgusudur; DM yanlış kitleye yazılmış. "
               "Taslak gönderilmez."),
    "dm_location_hallucination": dict(
        type="inter_agent", severity="medium", agents=["conversation"],
        winner="source_tweet", loser="conversation", action="quarantine_draft",
        policy="DM, tweette ve konum alanında geçmeyen bir yerden söz ediyor. Kaynak "
               "(tweet) kanıttır, uydurma yer kazanamaz. Taslak gönderilmez."),
    "dm_invented_figure": dict(
        type="inter_agent", severity="medium", agents=["conversation"],
        winner="source_tweet", loser="conversation", action="quarantine_draft",
        policy="DM, tweette olmayan bir tutar/rakam öne sürüyor. Kaynak (tweet) kazanır. "
               "Taslak gönderilmez."),
    # Tip C — handoff
    "draft_orphan": dict(
        type="handoff", severity="high", agents=["classifier", "conversation"],
        winner="classifier", loser="conversation", action="hold_artifact",
        policy="Nitelikli listede olmayan biri için DM üretilmiş; nitelik kararı Classifier'ındır, "
               "yetim taslak gönderilmez."),
    "link_orphan": dict(
        type="handoff", severity="high", agents=["classifier", "form_agent"],
        winner="classifier", loser="form_agent", action="hold_artifact",
        policy="Nitelikli listede olmayan biri için form linki üretilmiş; yetim link gönderilmez."),
    "lead_without_draft": dict(
        type="handoff", severity="medium", agents=["classifier", "conversation"],
        winner=None, loser=None, action="flag_review",
        policy="Nitelikli lead için DM taslağı üretilmemiş; gönderilecek bir şey yok, "
               "devir kaybı incelemeye işaretlenir."),
    "lead_without_link": dict(
        type="handoff", severity="medium", agents=["classifier", "form_agent"],
        winner=None, loser=None, action="flag_review",
        policy="Nitelikli lead için form linki üretilmemiş; devir kaybı incelemeye işaretlenir."),
    "score_drift": dict(
        type="handoff", severity="high", agents=["classifier"],
        winner="classifier", loser="downstream", action="hold_artifact",
        policy="Skor adımlar arasında değişmiş. Skor kaynağı Classifier'dır; değişen aşağı akış "
               "çıktısı (DM veya link) gönderilmez."),
    "duplicate_lead": dict(
        type="handoff", severity="low", agents=["classifier"],
        winner=None, loser=None, action="flag_review",
        policy="Aynı kullanıcı nitelikli listede birden çok kez; tekrar eden gönderim riski "
               "incelemeye işaretlenir."),
    # Tip D — supervision
    "quality_disagreement": dict(
        type="supervision", severity="medium", agents=["orchestrator", "evaluator"],
        winner="orchestrator", loser="evaluator", action="flag_review",
        policy="Geçti/kaldı kararı deterministik kanıta dayanan Orchestrator'da kalır; LLM yargıcı "
               "(Evaluator) danışmandır. Uyuşmazlık kayda geçer ve incelemeye işaretlenir."),
    "accepted_low_scored_agent": dict(
        type="supervision", severity="medium", agents=["orchestrator", "evaluator"],
        winner="orchestrator", loser="evaluator", action="flag_review",
        policy="Orchestrator'ın kabul ettiği bir ajanı Evaluator düşük puanlamış; karar Orchestrator'da "
               "kalır, ajan incelemeye işaretlenir."),
}

NEEDS_REVIEW = "needs_review"

# ─── Yardımcılar ──────────────────────────────────────────────────

def _fold(text) -> str:
    """Türkçe harf farkını eşitle (İ,ı,I → i) ve küçült; ç,ş,ğ,ö,ü korunur."""
    return ("" if text is None else str(text)).replace("İ", "i").replace("ı", "i").lower()


def _make(rule, user=None, artifact=None, detail="", severity=None, **evidence):
    meta = RULES[rule]
    suffix = f":{artifact}" if rule == "score_drift" else ""
    return {
        "id": f"{rule}:{user or '-'}{suffix}",
        "rule": rule,
        "type": meta["type"],
        "severity": severity or meta["severity"],
        "agents": list(meta["agents"]),
        "user": user,
        "artifact": artifact,
        "detail": detail,
        "evidence": evidence,
    }


def _outputs(results: dict) -> dict:
    """Orchestrator sonuç sözlüğünden {ajan adı: o ajanın çıktısı}."""
    by_agent = {}
    for value in (results or {}).values():
        if isinstance(value, dict) and value.get("agent"):
            by_agent[value["agent"]] = value
    return by_agent


def _username(lead: dict):
    return (lead.get("user") or {}).get("username")


# ─── Tip A: tek ajanın iç tutarlılığı ─────────────────────────────

_RED_FLAGS = {"bot", "spam", "prompt_injection_suspected", "api_error", "parse_error"}


def check_lead(lead: dict, threshold: int = 60) -> list:
    intent, urgency = lead.get("intent"), lead.get("urgency")
    try:
        score = int(lead.get("score", 0) or 0)
    except (TypeError, ValueError):
        return []
    if intent in ("error", "unknown"):      # hata yedekleri kendi içinde tutarlıdır
        return []
    user, flags, found = _username(lead), set(lead.get("red_flags") or []), []

    if score >= threshold and intent != "buyer":
        found.append(_make("qualified_non_buyer", user, "lead",
                           f"skor {score} ≥ eşik {threshold} ama niyet '{intent}'",
                           score=score, intent=intent, threshold=threshold))
    if intent == "buyer" and score < 40:
        found.append(_make("buyer_low_score", user, "lead",
                           f"niyet 'buyer' ama skor yalnızca {score}", score=score))
    if urgency == "high" and intent != "buyer":
        found.append(_make("urgency_intent_mismatch", user, "lead",
                           f"aciliyet 'high' ama niyet '{intent}'", intent=intent))
    if score >= 70 and (flags & _RED_FLAGS or intent == "bot"):
        found.append(_make("high_score_with_red_flags", user, "lead",
                           f"skor {score} iken işaretler: {sorted(flags & _RED_FLAGS) or ['intent=bot']}",
                           score=score, flags=sorted(flags)))
    return found


# ─── Tip B: Classifier ↔ Conversation ─────────────────────────────

# (katlanmış biçim, orijinal biçim) çiftleri: aramada katlanmış, raporda orijinal gösterilir
_pairs = lambda *terms: [(_fold(t), t) for t in terms]
_PRESSURE = _pairs("son fırsat", "kaçırmayın", "kaçırma", "acil", "tükenmeden", "sınırlı sayıda",
                   "hemen karar", "hemen arayın", "bugün arayın")
_BUYER_TERMS = _pairs("satılık", "satın al", "ev sahibi", "mortgage", "konut kredisi")
_RENTER_TERMS = _pairs("kiralık", "kiracı")
_PLACES = [(_fold(p), p) for p in (
    "İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Kadıköy", "Beşiktaş", "Şişli", "Üsküdar",
    "Ümraniye", "Maltepe", "Bağcılar", "Beylikdüzü", "Esenyurt", "Sarıyer", "Çankaya",
    "Karşıyaka", "Bornova", "Nilüfer", "Mamak", "Keçiören")]
_FIGURE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:milyon|bin|tl|₺|m\b)", re.I)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def check_dm(lead: dict, draft: dict) -> list:
    user = _username(lead) or draft.get("user")
    dm = draft.get("dm_draft") or {}
    dm_text = f"{dm.get('dm_text') or ''} {dm.get('opening_question') or ''}"
    dm_f = _fold(dm_text)
    info = lead.get("user") or {}
    tweet_f = _fold(info.get("tweet"))
    source_f = _fold(f"{info.get('tweet') or ''} {info.get('location') or ''}")
    intent, urgency, found = lead.get("intent"), lead.get("urgency"), []

    if urgency == "low" or (intent and intent != "buyer"):
        hits = [orig for fold, orig in _PRESSURE if fold in dm_f and fold not in tweet_f]
        if hits:
            found.append(_make("dm_urgency_overreach", user, "draft",
                               f"aciliyet '{urgency}' iken DM baskı dili kullanıyor: {hits}",
                               urgency=urgency, phrases=hits))

    if intent == "renter":
        hits = [orig for fold, orig in _BUYER_TERMS if fold in dm_f and fold not in tweet_f]
    elif intent == "buyer":
        hits = [orig for fold, orig in _RENTER_TERMS if fold in dm_f and fold not in tweet_f]
    else:
        hits = []
    if hits:
        found.append(_make("dm_wrong_audience", user, "draft",
                           f"niyet '{intent}' iken DM karşı kitle dilini kullanıyor: {hits}",
                           intent=intent, terms=hits))

    invented = [orig for folded, orig in _PLACES if folded in dm_f and folded not in source_f]
    if invented:
        found.append(_make("dm_location_hallucination", user, "draft",
                           f"DM tweette olmayan yerleri anıyor: {invented}", places=invented))

    tweet_numbers = {n.replace(",", ".") for n in _NUMBER.findall(info.get("tweet") or "")}
    bad = [m.replace(",", ".") for m in (g[0] if isinstance(g, tuple) else g
                                         for g in _FIGURE.findall(dm_text))
           if m.replace(",", ".") not in tweet_numbers]
    if bad:
        found.append(_make("dm_invented_figure", user, "draft",
                           f"DM tweette olmayan tutarlar içeriyor: {bad}", figures=bad))
    return found


# ─── Tip C: devir (handoff) bütünlüğü ─────────────────────────────

def check_handoffs(agents: dict) -> list:
    clf, conv, form = agents.get("classifier"), agents.get("conversation"), agents.get("form_agent")
    if not clf:
        return []
    qualified = clf.get("qualified_leads", [])
    by_user, found, seen = {}, [], set()
    for lead in qualified:
        user = _username(lead)
        if user in seen:
            found.append(_make("duplicate_lead", user, "lead", f"'{user}' nitelikli listede birden çok kez"))
        seen.add(user)
        by_user.setdefault(user, lead)

    def drift(items, key, artifact, owner):
        for item in items:
            user = item.get(key)
            lead = by_user.get(user)
            if lead is not None and item.get("score") != lead.get("score"):
                found.append(_make("score_drift", user, artifact,
                                   f"{owner} skoru {item.get('score')}, Classifier skoru {lead.get('score')}",
                                   downstream=item.get("score"), source=lead.get("score")))

    if conv is not None:
        drafts = conv.get("dm_drafts", [])
        drafted = {d.get("user") for d in drafts}
        for d in drafts:
            if d.get("user") not in by_user:
                found.append(_make("draft_orphan", d.get("user"), "draft",
                                   f"'{d.get('user')}' nitelikli değil ama DM taslağı var"))
        for user in by_user:
            if user not in drafted:
                found.append(_make("lead_without_draft", user, "lead", f"'{user}' için DM taslağı yok"))
        drift(drafts, "user", "draft", "DM taslağı")

    if form is not None:
        links = form.get("form_links", [])
        linked = {l.get("username") for l in links}
        for l in links:
            if l.get("username") not in by_user:
                found.append(_make("link_orphan", l.get("username"), "link",
                                   f"'{l.get('username')}' nitelikli değil ama form linki var"))
        for user in by_user:
            if user not in linked:
                found.append(_make("lead_without_link", user, "lead", f"'{user}' için form linki yok"))
        drift(links, "username", "link", "Form linki")
    return found


# ─── Tip D: Orchestrator ↔ Evaluator ──────────────────────────────

def check_quality_disagreement(quality_score, evaluation: dict, quality_threshold: int = 75,
                               delta: int = 25) -> list:
    if quality_score is None or not isinstance(evaluation, dict):
        return []
    found = []
    system_score = evaluation.get("system_score")
    if system_score is not None and abs(quality_score - system_score) >= delta:
        passed = quality_score >= quality_threshold
        severity = "high" if (passed and system_score < 50) else "medium"
        found.append(_make("quality_disagreement", None, None,
                           f"Orchestrator kalite {quality_score}, Evaluator sistem skoru {system_score}",
                           severity=severity, orchestrator=quality_score, evaluator=system_score,
                           orchestrator_passed=passed))
    if quality_score >= quality_threshold:
        for agent, ev in (evaluation.get("agent_evaluations") or {}).items():
            score = (ev or {}).get("overall_score")
            if score is not None and score < 50:
                found.append(_make("accepted_low_scored_agent", agent, None,
                                   f"Orchestrator kabul etti ama Evaluator '{agent}' ajanını {score} puanladı",
                                   agent=agent, evaluator_score=score))
    return found


# ─── Çözüm ────────────────────────────────────────────────────────

def _find(agents: dict, kind: str, user):
    """('lead'|'draft'|'link', kullanıcı) → ilgili artefaktlar listesi."""
    if kind == "lead":
        pool = (agents.get("classifier") or {}).get("qualified_leads", [])
        return [x for x in pool if _username(x) == user]
    if kind == "draft":
        pool = (agents.get("conversation") or {}).get("dm_drafts", [])
        return [x for x in pool if x.get("user") == user]
    pool = (agents.get("form_agent") or {}).get("form_links", [])
    return [x for x in pool if x.get("username") == user]


def _hold(item: dict, conflict_id: str):
    item["status"] = NEEDS_REVIEW
    ids = item.setdefault("conflicts", [])
    if conflict_id not in ids:
        ids.append(conflict_id)


def apply_resolutions(agents: dict, found: list) -> list:
    """Politikayı uygular: artefaktları işaretler (silmez). Çözüm kayıtlarını döndürür."""
    resolutions = []
    for c in found:
        meta, affected = RULES[c["rule"]], []
        action, user = meta["action"], c["user"]
        if action == "hold_lead":
            targets = [("lead", user), ("draft", user), ("link", user)]
        elif action == "quarantine_draft":
            targets = [("draft", user)]
        elif action == "hold_artifact":
            targets = [(c["artifact"], user)]
        else:
            targets = []
        for kind, who in targets:
            for item in _find(agents, kind, who):
                _hold(item, c["id"])
                affected.append(f"{kind}:{who}")
        loser = meta["loser"]
        if loser == "downstream":
            loser = "form_agent" if c["artifact"] == "link" else "conversation"
        resolutions.append({
            "conflict_id": c["id"], "rule": c["rule"], "winner": meta["winner"], "loser": loser,
            "action": action, "policy": meta["policy"], "affected": affected,
        })
    return resolutions


def review_pipeline(results: dict):
    """Orchestrator sonuçlarını tarar → (çakışmalar, çözümler). Artefaktları yerinde işaretler."""
    agents = _outputs(results)
    clf, conv = agents.get("classifier"), agents.get("conversation")
    found = []
    if clf:
        threshold = clf.get("threshold", 60)
        for lead in clf.get("qualified_leads", []) + clf.get("rejected_leads", []):
            found += check_lead(lead, threshold)
        if conv:
            leads = {_username(l): l for l in clf.get("qualified_leads", [])}
            for draft in conv.get("dm_drafts", []):
                lead = leads.get(draft.get("user"))
                if lead is not None:
                    found += check_dm(lead, draft)
    found += check_handoffs(agents)

    unique, ids = [], set()
    for c in found:
        if c["id"] not in ids:
            ids.add(c["id"])
            unique.append(c)
    return unique, apply_resolutions(agents, unique)


def sendable_drafts(conversation_output: dict) -> list:
    """Çakışma nedeniyle durdurulmamış DM taslakları — gönderim yalnızca bunlardan yapılmalı."""
    return [d for d in (conversation_output or {}).get("dm_drafts", [])
            if d.get("status") != NEEDS_REVIEW]


def sendable_leads(classifier_output: dict) -> list:
    return [l for l in (classifier_output or {}).get("qualified_leads", [])
            if l.get("status") != NEEDS_REVIEW]


def emit_events(agent: str, found: list, resolutions: list):
    """Çakışma ve çözümleri aktif trace'e yazar."""
    for c in found:
        tracing.emit(agent, "conflict_detected", {
            "id": c["id"], "rule": c["rule"], "type": c["type"], "severity": c["severity"],
            "agents": c["agents"], "detail": c["detail"]})
    for r in resolutions:
        tracing.emit(agent, "conflict_resolved", {
            "id": r["conflict_id"], "winner": r["winner"], "loser": r["loser"],
            "action": r["action"], "affected": r["affected"]})
