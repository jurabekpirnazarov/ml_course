# 🚦 Ko'cha kesishuvi monitoringi — YOLO26s + Streamlit

YouTube jonli streamni o'qib, **odam va mashinalarni** YOLO26s bilan real vaqtda aniqlaydi,
siz chizgan **liniyadan kesib o'tganlarni** SQLite DB ga yozadi, **grafiklar** va
**sodda statistika chati** (AI emas) bilan ko'rsatadi. Hammasi lokal ishlaydi.

## O'rnatish

```bash
python -m venv venv
# Windows:  venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
```

> Birinchi ishga tushirishda `yolo26s.pt` (~19 MB) avtomatik yuklab olinadi.
> NVIDIA GPU bo'lsa, PyTorch CUDA versiyasini o'rnating — ancha tezroq ishlaydi.

## Ishga tushirish

```bash
streamlit run app.py
```

Brauzerda `http://localhost:8501` ochiladi.

## Foydalanish tartibi

1. **Chap panel** — YouTube link allaqachon kiritilgan (`...WSm_r0eNl1E`), **▶️ Boshlash** bosing.
2. **📹 Jonli video** tabida annotatsiyalangan stream va jonli hisoblagichlar ko'rinadi.
3. **✏️ Liniyalar** tabida:
   - **📸 Snapshot olish** bosing,
   - rasm ustida **ikki nuqta** bosing → yangi sanash liniyasi qo'shiladi (bir nechta liniya mumkin),
   - yoki koordinatalarni qo'lda kiriting.
4. Obyekt liniyani kesib o'tganda DB ga yoziladi: `timestamp, type (person/car), direction (N->S, S->N, E->W, W->E), line_name`.
5. **📊 Analitika** — har 3 soniyada yangilanadi: bugungi soatlik traffic, yo'nalishlar, 14 kunlik dinamika, oxirgi yozuvlar.
6. **💬 Chat** — sodda savollar:
   - "Bugun qancha odam o'tdi?"
   - "Jami qancha mashina?"
   - "Oxirgi soatda qancha?"
   - "Qaysi yo'nalishda ko'p?"
   - "Eng band soat?"
   - "Liniyalar bo'yicha statistika"

## Yengil ishlashi uchun maslahatlar

| Sozlama | Tavsiya |
|---|---|
| Model | `yolo26n.pt` — eng yengil (CPU uchun), `yolo26s.pt` — aniqroq |
| Inference o'lchami | 320–480 (CPU), 640 (GPU) |
| Kadr tashlash | 2–4 (real vaqtda qolish uchun) |

## Demo ma'lumotlar (ixtiyoriy)

Grafiklarni darhol sinash uchun:

```bash
python seed_db.py
```

## Fayllar

- `app.py` — asosiy ilova (detektsiya, DB, UI, chat)
- `seed_db.py` — ixtiyoriy demo ma'lumot generatori
- `street_crossing.db` — avtomatik yaratiladi
- `requirements.txt`

## Eslatmalar

- YouTube live URL muddati tugasa, ilova **avtomatik qayta ulanadi**.
- Webcam uchun manba sifatida `0` kiriting; RTSP/MP4 ham ishlaydi.
- `lapx` paketi ByteTrack tracking uchun kerak.
