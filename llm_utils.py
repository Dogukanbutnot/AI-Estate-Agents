"""
llm_utils.py
------------
LLM çıktılarını güvenle JSON'a çeviren ortak yardımcı.

Model bazen JSON'u ```json ... ``` bloğu içinde, bazen de önüne düşünce
metni ekleyerek döndürür. Düz json.loads bunların hepsinde patlar.
"""

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(raw: str):
    """
    Sırayla dener:
      1. Yanıtın tamamı JSON mu?
      2. ``` kod bloğu içinde JSON var mı?
      3. Metnin içine gömülü ilk geçerli JSON nesnesi / dizisi var mı?
    Hiçbiri olmazsa ValueError fırlatır.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("boş yanıt")

    text = raw.strip()

    candidates = [text] + [m.strip() for m in _FENCE.findall(text)]
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except ValueError:
            continue

    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj
            except ValueError:
                continue

    raise ValueError("yanıtta geçerli JSON bulunamadı")
