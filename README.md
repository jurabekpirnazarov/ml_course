# 🚦 Street Crossing Monitor — YOLO26s + Streamlit

Reads a YouTube live stream, detects **people and cars** in real time with YOLO26s,
logs every object that crosses your **user-drawn line(s)** into a SQLite database,
and shows **live charts** plus a **simple rule-based stats chat** (no AI).
Everything runs locally.

## Installation

```bash
python -m venv venv
# Windows:  venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
```

> On first run, `yolo26s.pt` (~19 MB) is downloaded automatically.
> If you have an NVIDIA GPU, install the CUDA build of PyTorch — it runs much faster.

## Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## How to use

1. **Left sidebar** — the YouTube link is pre-filled (`...WSm_r0eNl1E`). Click **▶️ Boshlash (Start)**.
2. **📹 Live video** tab shows the annotated stream with live counters and FPS.
3. **✏️ Lines** tab:
   - Click **📸 Take snapshot**,
   - then click **two points** on the image → a new counting line is added (multiple lines supported),
   - or enter coordinates manually in the expander.
4. When an object crosses a line, a row is written to the DB:
   `timestamp, type (person/car), direction (N->S, S->N, E->W, W->E), line_name`.
5. **📊 Analytics** — auto-refreshes every 3 seconds: today's hourly traffic, direction breakdown, 14-day trend, latest DB rows.
6. **💬 Chat** — simple questions (rule-based, not AI):
   - "Bugun qancha odam o'tdi?" (How many people today?)
   - "Jami qancha mashina?" (Total cars?)
   - "Oxirgi soatda qancha?" (Last hour?)
   - "Qaysi yo'nalishda ko'p?" (Which direction is busiest?)
   - "Eng band soat?" (Busiest hour?)
   - "Liniyalar bo'yicha statistika" (Stats per line)

## Tips for lightweight performance

| Setting | Recommendation |
|---|---|
| Model | `yolo26n.pt` — lightest (good for CPU), `yolo26s.pt` — more accurate |
| Inference size | 320–480 (CPU), 640 (GPU) |
| Frame skip | 2–4 (to stay real-time) |

## Demo data (optional)

To see the charts populated right away:

```bash
python seed_db.py
```

## Files

- `app.py` — main app (detection, DB, UI, chat)
- `seed_db.py` — optional demo data generator
- `street_crossing.db` — created automatically
- `requirements.txt`

## Notes

- If the YouTube live URL expires, the app **reconnects automatically**.
- For a webcam, enter `0` as the source; RTSP and local MP4 files also work.
- The `lapx` package is required for ByteTrack tracking.
