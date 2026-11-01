"""
form_server.py — FastAPI form endpoint
----------------------------------------
Çalıştırma: uvicorn form_server:app --reload --port 8000

  GET  /form?token=...        → formu gösterir (token lead_store'da kayıtlı olmalı)
  POST /form/submit           → formu kaydeder (tek kullanımlık, KVKK zorunlu)
  GET  /api/lead/{token}      → token durumu
  GET  /api/leads/completed   → tamamlanan lead'ler (emlakçı paneli)

Token'ları FormAgent üretir ve lead_store'a yazar; bu sunucu aynı depoyu okur.
"""

import time
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import lead_store

app = FastAPI(title="Emlak Lead Form API")


class FormSubmission(BaseModel):
    token: str
    name: str
    phone: str
    email: str
    budget: str
    location: str
    property_type: str
    timeline: str
    kvkk_consent: bool


FORM_HTML = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Emlak Danışmanlığı — Bilgi Formu</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #f5f5f5;
         display: flex; justify-content: center; padding: 2rem 1rem; }
  .card { background: #fff; border-radius: 12px; padding: 2rem;
           max-width: 520px; width: 100%;
           box-shadow: 0 1px 4px rgba(0,0,0,.08); }
  h1 { font-size: 1.3rem; font-weight: 600; margin-bottom: .25rem; color: #1a1a1a; }
  .sub { font-size: .875rem; color: #666; margin-bottom: 1.5rem; }
  label { display: block; font-size: .875rem; font-weight: 500;
           color: #333; margin-bottom: .25rem; margin-top: 1rem; }
  input, select, textarea {
    width: 100%; padding: .6rem .8rem; border: 1px solid #ddd;
    border-radius: 8px; font-size: .9rem; outline: none; color: #1a1a1a;
    background: #fafafa; transition: border-color .15s;
  }
  input:focus, select:focus { border-color: #534AB7; background: #fff; }
  .kvkk { font-size: .8rem; color: #555; margin-top: 1.25rem;
           padding: .75rem; background: #f0eeff;
           border-radius: 8px; border-left: 3px solid #534AB7; }
  .kvkk label { font-size: .8rem; color: #333; margin: 0;
                 display: flex; align-items: flex-start; gap: .5rem; cursor: pointer; }
  .kvkk input[type=checkbox] { width: auto; margin-top: 2px; }
  button { margin-top: 1.5rem; width: 100%; padding: .8rem;
            background: #534AB7; color: #fff; border: none;
            border-radius: 8px; font-size: 1rem; font-weight: 500;
            cursor: pointer; transition: background .15s; }
  button:hover { background: #3C3489; }
  button:disabled { background: #aaa; cursor: not-allowed; }
  .success { display: none; text-align: center; padding: 2rem 0; }
  .success h2 { color: #1D9E75; font-size: 1.2rem; margin-bottom: .5rem; }
  .success p { color: #555; font-size: .9rem; }
  .err { color: #c0392b; font-size: .8rem; margin-top: .5rem; display: none; }
</style>
</head>
<body>
<div class="card">
  <div id="form-section">
    <h1>Emlak Danışmanlığı</h1>
    <p class="sub">Bilgilerinizi doldurarak size en uygun portföyleri alalım.</p>

    <label>Ad Soyad</label>
    <input id="name" type="text" placeholder="Adınız Soyadınız">

    <label>Telefon</label>
    <input id="phone" type="tel" placeholder="05xx xxx xx xx">

    <label>E-posta</label>
    <input id="email" type="email" placeholder="ornek@mail.com">

    <label>Bütçe aralığı</label>
    <select id="budget">
      <option value="">Seçiniz</option>
      <option value="1m_alt">1.000.000 TL altı</option>
      <option value="1m_2m">1.000.000 – 2.000.000 TL</option>
      <option value="2m_4m">2.000.000 – 4.000.000 TL</option>
      <option value="4m_ust">4.000.000 TL üzeri</option>
    </select>

    <label>Tercih ettiğiniz semt / ilçe</label>
    <input id="location" type="text" placeholder="Kadıköy, Beşiktaş...">

    <label>Mülk tipi</label>
    <select id="property_type">
      <option value="">Seçiniz</option>
      <option value="daire">Daire</option>
      <option value="mustakil">Müstakil ev</option>
      <option value="villa">Villa</option>
      <option value="diger">Diğer</option>
    </select>

    <label>Ne zaman taşınmayı planlıyorsunuz?</label>
    <select id="timeline">
      <option value="">Seçiniz</option>
      <option value="1ay">1 ay içinde</option>
      <option value="3ay">1-3 ay</option>
      <option value="6ay">3-6 ay</option>
      <option value="belirsiz">Henüz belirli değil</option>
    </select>

    <div class="kvkk">
      <label>
        <input type="checkbox" id="kvkk">
        Kişisel verilerimin 6698 sayılı KVKK kapsamında emlak danışmanlığı amacıyla
        işlenmesini ve bu amaçla kayıt altına alınmasını kabul ediyorum.
      </label>
    </div>

    <p class="err" id="err">Lütfen tüm alanları doldurun ve KVKK onayı verin.</p>
    <button onclick="submitForm()">Gönder</button>
  </div>

  <div class="success" id="success-section">
    <h2>Teşekkürler!</h2>
    <p>Bilgileriniz alındı. En kısa sürede size uygun portföyleri paylaşacağız.</p>
  </div>
</div>

<script>
const TOKEN = new URLSearchParams(location.search).get('token') || '';
const REF   = new URLSearchParams(location.search).get('ref')   || '';

async function submitForm() {
  const fields = ['name','phone','email','budget','location','property_type','timeline'];
  const vals = {};
  for (const f of fields) {
    vals[f] = document.getElementById(f).value.trim();
    if (!vals[f]) { showErr(); return; }
  }
  if (!document.getElementById('kvkk').checked) { showErr(); return; }

  document.querySelector('button').disabled = true;

  const res = await fetch('/form/submit', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ token: TOKEN, kvkk_consent: true, ...vals })
  });

  if (res.ok) {
    document.getElementById('form-section').style.display = 'none';
    document.getElementById('success-section').style.display = 'block';
  } else {
    const data = await res.json();
    document.getElementById('err').textContent = data.detail || 'Bir hata oluştu.';
    document.getElementById('err').style.display = 'block';
    document.querySelector('button').disabled = false;
  }
}

function showErr() {
  document.getElementById('err').style.display = 'block';
}
</script>
</body>
</html>"""


@app.get("/form", response_class=HTMLResponse)
async def form_page(token: str = Query("")):
    """Kullanıcıya formu gösterir; bilinmeyen veya kullanılmış token'ı reddeder."""
    record = lead_store.get_token(token) if token else None
    if not record:
        raise HTTPException(status_code=404, detail="Geçersiz form linki.")
    if record["completed_at"]:
        raise HTTPException(status_code=409, detail="Bu form zaten doldurulmuş.")
    return HTMLResponse(content=FORM_HTML)


@app.post("/form/submit")
async def form_submit(data: FormSubmission):
    """Formu kaydeder. Kullanıcı adı token'dan çözülür, istemciden alınmaz."""
    try:
        username = lead_store.save_submission(data.token, data.model_dump())
    except lead_store.UnknownToken:
        raise HTTPException(status_code=404, detail="Geçersiz form linki.")
    except lead_store.AlreadySubmitted:
        raise HTTPException(status_code=409, detail="Bu form zaten gönderilmiş.")
    except lead_store.ConsentRequired:
        raise HTTPException(status_code=422, detail="KVKK onayı olmadan kayıt yapılamaz.")
    return {"status": "ok", "username": username}


@app.get("/api/lead/{token}")
async def get_lead_status(token: str):
    record = lead_store.get_token(token)
    if not record:
        return {"found": False, "completed": False}
    return {
        "found": True,
        "completed": record["completed_at"] is not None,
        "username": record["username"],
    }


@app.get("/api/leads/completed")
async def get_completed_leads():
    return {"leads": lead_store.list_completed()}


@app.get("/health")
async def health():
    return {"status": "ok", "time": time.time()}
