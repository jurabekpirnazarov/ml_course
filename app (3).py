# -*- coding: utf-8 -*-
"""
Ko'cha kesishuvi monitoringi — YOLO26s + Streamlit (lokal)
==========================================================
- YouTube live stream (yoki istalgan video URL / webcam) o'qiydi
- YOLO26s bilan odam va mashinalarni real vaqtda aniqlaydi va kuzatadi (tracking)
- Foydalanuvchi chizgan liniya(lar)dan kesib o'tganlarni SQLite DB ga yozadi
- Real-time analitika (grafiklar) + sodda savol-javob chati (AI emas, qoidaga asoslangan)

Ishga tushirish:  streamlit run app.py
"""

import os
import time
import threading
import sqlite3
from datetime import datetime, timedelta

import cv2
import numpy as np
import pandas as pd
import streamlit as st

DB_PATH = "street_crossing.db"
PROC_WIDTH = 960          # ishlov berish uchun kadr kengligi (yengil ishlashi uchun)
DEFAULT_YT = "https://www.youtube.com/watch?v=WSm_r0eNl1E"

# Aniqlash uchun COCO klasslari: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
DETECT_CLASSES = [0, 2, 3, 5, 7]
CLS_TO_TYPE = {0: "person", 2: "car", 3: "car", 5: "car", 7: "car"}

# =========================================================
# 1. DATABASE
# =========================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS crossings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            type TEXT,        -- 'person' yoki 'car'
            direction TEXT,   -- N->S, S->N, E->W, W->E
            line_name TEXT    -- qaysi liniyadan kesib o'tdi
        )
    """)
    conn.commit()
    conn.close()


def query_df(sql, params=()):
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()
    return df


# =========================================================
# 2. GEOMETRIYA — liniya kesishuvi
# =========================================================
def _ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1, p2, p3, p4):
    """(p1,p2) kesma (p3,p4) kesma bilan kesishadimi"""
    return (_ccw(p1, p3, p4) != _ccw(p2, p3, p4)) and (_ccw(p1, p2, p3) != _ccw(p1, p2, p4))


def movement_direction(prev, cur):
    """Harakat vektoridan kompas yo'nalishi (tasvirda y pastga qarab o'sadi)"""
    dx = cur[0] - prev[0]
    dy = cur[1] - prev[1]
    if abs(dy) >= abs(dx):
        return "N->S" if dy > 0 else "S->N"
    return "W->E" if dx > 0 else "E->W"


# =========================================================
# 3. DETECTION ENGINE (orqa fonda ishlaydigan thread)
# =========================================================
class CrossingEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.running = False
        self.status = "To'xtagan"
        self.error = None
        self.fps = 0.0

        self.lines = []          # [{"name": str, "p1": (x,y), "p2": (x,y)}]
        self.latest_jpeg = None  # annotatsiyalangan kadr (bytes)
        self.latest_raw = None   # toza kadr (numpy, liniya chizish uchun snapshot)
        self.frame_size = (PROC_WIDTH, 540)

        self.session_counts = {"person": 0, "car": 0}
        self.line_counts = {}    # line_name -> {"person": n, "car": n}
        self.last_events = []    # oxirgi 30 ta hodisa (chat/log uchun)

    # ---------- boshqaruv ----------
    def set_lines(self, lines):
        with self.lock:
            self.lines = list(lines)
            for ln in lines:
                self.line_counts.setdefault(ln["name"], {"person": 0, "car": 0})

    def start(self, source, model_name="yolo26s.pt", conf=0.35, imgsz=640, skip=2):
        if self.running:
            return
        self.running = True
        self.error = None
        self.status = "Ishga tushmoqda..."
        self.thread = threading.Thread(
            target=self._run, args=(source, model_name, conf, imgsz, skip), daemon=True
        )
        self.thread.start()

    def stop(self):
        self.running = False
        self.status = "To'xtagan"

    # ---------- manbani ochish ----------
    def _resolve_source(self, source):
        src = source.strip()
        if src.isdigit():
            return int(src)  # webcam
        if "youtube.com" in src or "youtu.be" in src:
            import yt_dlp
            opts = {
                "quiet": True,
                "no_warnings": True,
                # yengil ishlashi uchun 480p atrofidagi oqimni olamiz
                "format": "best[height<=720][protocol^=m3u8]/best[height<=720]/best",
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(src, download=False)
            return info["url"]
        return src

    # ---------- asosiy sikl ----------
    def _run(self, source, model_name, conf, imgsz, skip):
        try:
            from ultralytics import YOLO
            self.status = f"{model_name} yuklanmoqda..."
            model = YOLO(model_name)  # birinchi marta avtomatik yuklab oladi

            self.status = "Stream ochilmoqda..."
            url = self._resolve_source(source)
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                raise RuntimeError("Stream ochilmadi. URL ni tekshiring.")

            conn = sqlite3.connect(DB_PATH)
            prev_pos = {}            # track_id -> (x, y)
            counted = {}             # (track_id, line_name) -> oxirgi sanash vaqti
            fail, t0, nframes = 0, time.time(), 0
            self.status = "Ishlamoqda ✅"

            while self.running:
                ok, frame = cap.read()
                if not ok:
                    fail += 1
                    if fail > 50:  # YouTube URL muddati tugagan bo'lishi mumkin — qayta ulanish
                        cap.release()
                        try:
                            url = self._resolve_source(source)
                            cap = cv2.VideoCapture(url)
                            fail = 0
                            self.status = "Qayta ulanmoqda..."
                        except Exception:
                            time.sleep(3)
                    time.sleep(0.05)
                    continue
                fail = 0
                self.status = "Ishlamoqda ✅"

                # real vaqtda qolish uchun ortiqcha kadrlarni tashlab yuborish
                for _ in range(skip):
                    cap.grab()

                # kadrni kichraytirish (yengillik)
                h, w = frame.shape[:2]
                scale = PROC_WIDTH / w
                frame = cv2.resize(frame, (PROC_WIDTH, int(h * scale)))
                self.frame_size = (frame.shape[1], frame.shape[0])
                with self.lock:
                    self.latest_raw = frame.copy()
                    lines = list(self.lines)

                # --- YOLO26 tracking ---
                results = model.track(
                    frame, persist=True, conf=conf, imgsz=imgsz,
                    classes=DETECT_CLASSES, verbose=False, tracker="bytetrack.yaml",
                )
                r = results[0]
                annotated = frame

                now = time.time()
                if r.boxes is not None and r.boxes.id is not None:
                    ids = r.boxes.id.int().tolist()
                    clss = r.boxes.cls.int().tolist()
                    xyxy = r.boxes.xyxy.cpu().numpy()

                    for tid, cls_id, box in zip(ids, clss, xyxy):
                        obj_type = CLS_TO_TYPE.get(cls_id, "car")
                        x1, y1, x2, y2 = box.astype(int)
                        cx, cy = int((x1 + x2) / 2), int(y2)  # pastki markaz (oyoq/g'ildirak)

                        color = (60, 200, 60) if obj_type == "person" else (255, 140, 0)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, f"{obj_type} #{tid}", (x1, y1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                        cv2.circle(annotated, (cx, cy), 3, color, -1)

                        # --- liniya kesishuvini tekshirish ---
                        if tid in prev_pos:
                            prev = prev_pos[tid]
                            for ln in lines:
                                key = (tid, ln["name"])
                                # bitta obyekt bitta liniyada 2s ichida qayta sanalmaydi
                                if key in counted and now - counted[key] < 2.0:
                                    continue
                                if segments_intersect(prev, (cx, cy), ln["p1"], ln["p2"]):
                                    direction = movement_direction(prev, (cx, cy))
                                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    conn.execute(
                                        "INSERT INTO crossings (timestamp, type, direction, line_name) "
                                        "VALUES (?, ?, ?, ?)",
                                        (ts, obj_type, direction, ln["name"]),
                                    )
                                    conn.commit()
                                    counted[key] = now
                                    with self.lock:
                                        self.session_counts[obj_type] += 1
                                        lc = self.line_counts.setdefault(
                                            ln["name"], {"person": 0, "car": 0})
                                        lc[obj_type] += 1
                                        self.last_events.insert(
                                            0, f"{ts.split()[1]}  {obj_type}  {direction}  ({ln['name']})")
                                        self.last_events = self.last_events[:30]
                        prev_pos[tid] = (cx, cy)

                # --- liniyalarni chizish ---
                with self.lock:
                    for ln in lines:
                        cv2.line(annotated, ln["p1"], ln["p2"], (0, 230, 255), 3)
                        mx = (ln["p1"][0] + ln["p2"][0]) // 2
                        my = (ln["p1"][1] + ln["p2"][1]) // 2
                        lc = self.line_counts.get(ln["name"], {"person": 0, "car": 0})
                        label = f"{ln['name']}: P {lc['person']} | C {lc['car']}"
                        cv2.putText(annotated, label, (mx - 70, my - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 2, cv2.LINE_AA)

                # FPS
                nframes += 1
                if now - t0 >= 1.0:
                    self.fps = nframes / (now - t0)
                    t0, nframes = now, 0
                cv2.putText(annotated, f"FPS: {self.fps:.1f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                ok_jpg, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok_jpg:
                    with self.lock:
                        self.latest_jpeg = buf.tobytes()

            cap.release()
            conn.close()
        except Exception as e:
            self.error = str(e)
            self.status = f"Xato ❌: {e}"
            self.running = False


@st.cache_resource
def get_engine():
    init_db()
    return CrossingEngine()


# =========================================================
# 4. CHAT — qoidaga asoslangan savol-javob (AI emas)
# =========================================================
def answer_question(q: str) -> str:
    ql = q.lower().replace("'", "'")
    today = datetime.now().strftime("%Y-%m-%d")
    hour_ago = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")

    def count(where="", params=()):
        df = query_df(f"SELECT type, COUNT(*) c FROM crossings {where} GROUP BY type", params)
        d = dict(zip(df["type"], df["c"])) if not df.empty else {}
        return d.get("person", 0), d.get("car", 0)

    wants_person = any(w in ql for w in ["odam", "piyoda", "person", "люди", "people"])
    wants_car = any(w in ql for w in ["mashina", "avtomobil", "car", "moshina"])

    # Yo'nalish bo'yicha
    if any(w in ql for w in ["yo'nalish", "yonalish", "tomonga", "qayoq", "direction", "kelayotgan", "ketayotgan"]):
        df = query_df("SELECT direction, COUNT(*) c FROM crossings GROUP BY direction ORDER BY c DESC")
        if df.empty:
            return "Hozircha DB da ma'lumot yo'q."
        lines = [f"• {r.direction}: {r.c} ta" for r in df.itertuples()]
        top = df.iloc[0]
        return ("Yo'nalishlar bo'yicha jami kesib o'tishlar:\n" + "\n".join(lines) +
                f"\n\nEng faol yo'nalish: **{top.direction}** ({top.c} ta).")

    # Eng band soat
    if "eng" in ql and ("soat" in ql or "band" in ql or "ko'p" in ql or "kop" in ql):
        df = query_df(
            "SELECT substr(timestamp,12,2) h, COUNT(*) c FROM crossings "
            "WHERE substr(timestamp,1,10)=? GROUP BY h ORDER BY c DESC LIMIT 1", (today,))
        if df.empty:
            return "Bugun uchun hali ma'lumot yo'q."
        return f"Bugungi eng band soat: **{df.iloc[0].h}:00** — {df.iloc[0].c} ta kesib o'tish."

    # Oxirgi soat
    if any(w in ql for w in ["oxirgi soat", "songgi soat", "so'nggi soat", "last hour"]):
        p, c = count("WHERE timestamp >= ?", (hour_ago,))
        return f"Oxirgi 1 soatda: 🚶 {p} ta odam, 🚗 {c} ta mashina kesib o'tdi."

    # Bugun
    if "bugun" in ql or "today" in ql:
        p, c = count("WHERE substr(timestamp,1,10)=?", (today,))
        if wants_person and not wants_car:
            return f"Bugun 🚶 **{p} ta odam** kesib o'tdi."
        if wants_car and not wants_person:
            return f"Bugun 🚗 **{c} ta mashina** kesib o'tdi."
        return f"Bugun: 🚶 {p} ta odam, 🚗 {c} ta mashina (jami {p + c})."

    # Jami / umumiy
    if any(w in ql for w in ["jami", "umumiy", "hammasi", "total", "qancha", "nechta", "necha"]):
        p, c = count()
        if wants_person and not wants_car:
            return f"Jami 🚶 **{p} ta odam** kesib o'tgan."
        if wants_car and not wants_person:
            return f"Jami 🚗 **{c} ta mashina** kesib o'tgan."
        return f"Jami DB da: 🚶 {p} ta odam, 🚗 {c} ta mashina ({p + c} ta yozuv)."

    # Liniyalar bo'yicha
    if "liniya" in ql or "line" in ql or "chiziq" in ql:
        df = query_df("SELECT line_name, type, COUNT(*) c FROM crossings "
                      "WHERE line_name IS NOT NULL GROUP BY line_name, type")
        if df.empty:
            return "Liniyalar bo'yicha hali yozuv yo'q."
        out = "Liniyalar bo'yicha:\n"
        for name, g in df.groupby("line_name"):
            d = dict(zip(g["type"], g["c"]))
            out += f"• {name}: 🚶 {d.get('person', 0)}, 🚗 {d.get('car', 0)}\n"
        return out

    return ("Men sodda statistika botiman 🤖 (AI emas). Quyidagicha so'rang:\n"
            "• *Bugun qancha odam o'tdi?*\n"
            "• *Jami qancha mashina?*\n"
            "• *Oxirgi soatda qancha?*\n"
            "• *Qaysi yo'nalishda ko'p?*\n"
            "• *Eng band soat qaysi?*\n"
            "• *Liniyalar bo'yicha statistika*")


# =========================================================
# 5. STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Ko'cha monitoringi — YOLO26s", page_icon="🚦", layout="wide")
engine = get_engine()

if "lines" not in st.session_state:
    st.session_state.lines = []
if "pending_point" not in st.session_state:
    st.session_state.pending_point = None
if "chat" not in st.session_state:
    st.session_state.chat = []
if "snapshot" not in st.session_state:
    st.session_state.snapshot = None

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🚦 Sozlamalar")
    source = st.text_input("Stream URL (YouTube / RTSP / fayl / 0=webcam)", value=DEFAULT_YT)
    model_name = st.selectbox("Model", ["yolo26s.pt", "yolo26n.pt", "yolo11s.pt"], index=0,
                              help="yolo26n — eng yengil; yolo26s — aniqroq")
    imgsz = st.select_slider("Inference o'lchami (kichik = tezroq)", [320, 480, 640], value=480)
    conf = st.slider("Ishonch chegarasi (conf)", 0.2, 0.7, 0.35, 0.05)
    skip = st.slider("Kadr tashlash (yengillik)", 0, 5, 2,
                     help="Har o'qilgan kadrdan keyin nechta kadr tashlab yuboriladi")

    c1, c2 = st.columns(2)
    if c1.button("▶️ Boshlash", use_container_width=True, type="primary"):
        engine.set_lines(st.session_state.lines)
        engine.start(source, model_name, conf, imgsz, skip)
    if c2.button("⏹ To'xtatish", use_container_width=True):
        engine.stop()

    st.caption(f"Holat: **{engine.status}**")
    if engine.error:
        st.error(engine.error)

    st.divider()
    st.caption("💡 Birinchi ishga tushirishda model (~19 MB) avtomatik yuklab olinadi. "
               "GPU bo'lsa avtomatik ishlatiladi, bo'lmasa CPU.")

# ---------------- Tabs ----------------
tab_live, tab_lines, tab_stats, tab_chat = st.tabs(
    ["📹 Jonli video", "✏️ Liniyalar", "📊 Analitika", "💬 Chat"])

# ========== 📹 LIVE ==========
with tab_live:
    @st.fragment(run_every=0.4)
    def live_view():
        with engine.lock:
            jpeg = engine.latest_jpeg
            sc = dict(engine.session_counts)
            events = list(engine.last_events[:8])
        m1, m2, m3 = st.columns(3)
        m1.metric("🚶 Odamlar (sessiya)", sc["person"])
        m2.metric("🚗 Mashinalar (sessiya)", sc["car"])
        m3.metric("⚡ FPS", f"{engine.fps:.1f}")
        if jpeg:
            st.image(jpeg, channels="BGR", use_container_width=True)
        else:
            st.info("Video hali yo'q. Chap paneldan ▶️ Boshlash tugmasini bosing.")
        if events:
            st.caption("Oxirgi hodisalar: " + "  |  ".join(events))

    live_view()
    st.caption("Asl stream: " + source)
    with st.expander("▶️ YouTube playerda ko'rish (taqqoslash uchun)"):
        if "youtube.com" in source or "youtu.be" in source:
            st.video(source)

# ========== ✏️ LINES ==========
with tab_lines:
    st.subheader("Sanash liniyalarini chizish")
    st.write("1) **Snapshot olish** tugmasini bosing → 2) rasm ustida **ikki nuqta** bosing — "
             "ular orasidagi chiziq yangi sanash liniyasi bo'ladi.")

    cA, cB = st.columns([1, 1])
    if cA.button("📸 Snapshot olish (jonli kadrdan)"):
        with engine.lock:
            st.session_state.snapshot = None if engine.latest_raw is None else engine.latest_raw.copy()
        if st.session_state.snapshot is None:
            st.warning("Hali kadr yo'q — avval streamni ishga tushiring.")
    if cB.button("🗑 Barcha liniyalarni o'chirish"):
        st.session_state.lines = []
        st.session_state.pending_point = None
        engine.set_lines([])

    snap = st.session_state.snapshot
    if snap is not None:
        disp = snap.copy()
        for ln in st.session_state.lines:
            cv2.line(disp, ln["p1"], ln["p2"], (0, 230, 255), 3)
        if st.session_state.pending_point:
            cv2.circle(disp, st.session_state.pending_point, 6, (0, 0, 255), -1)

        disp_w = 760
        scale = snap.shape[1] / disp_w
        try:
            from streamlit_image_coordinates import streamlit_image_coordinates
            from PIL import Image
            pil = Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
            click = streamlit_image_coordinates(pil, width=disp_w, key="line_canvas")
            if click is not None:
                pt = (int(click["x"] * scale), int(click["y"] * scale))
                last = st.session_state.get("_last_click")
                if pt != last:  # rerunlarda takror sanamaslik
                    st.session_state._last_click = pt
                    if st.session_state.pending_point is None:
                        st.session_state.pending_point = pt
                        st.rerun()
                    else:
                        name = f"Liniya-{len(st.session_state.lines) + 1}"
                        st.session_state.lines.append(
                            {"name": name, "p1": st.session_state.pending_point, "p2": pt})
                        st.session_state.pending_point = None
                        engine.set_lines(st.session_state.lines)
                        st.rerun()
        except ImportError:
            st.warning("`pip install streamlit-image-coordinates` o'rnating — "
                       "yoki pastdagi qo'lda kiritishdan foydalaning.")
            st.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), width=disp_w)
    else:
        st.info("Snapshot yo'q. Stream ishga tushgach 📸 tugmasini bosing.")

    with st.expander("✍️ Liniyani qo'lda kiritish (koordinatalar bilan)"):
        w, h = engine.frame_size
        c1, c2, c3, c4 = st.columns(4)
        x1 = c1.number_input("x1", 0, w, int(w * 0.2))
        y1 = c2.number_input("y1", 0, h, int(h * 0.6))
        x2 = c3.number_input("x2", 0, w, int(w * 0.8))
        y2 = c4.number_input("y2", 0, h, int(h * 0.6))
        if st.button("➕ Liniya qo'shish"):
            name = f"Liniya-{len(st.session_state.lines) + 1}"
            st.session_state.lines.append({"name": name, "p1": (int(x1), int(y1)),
                                           "p2": (int(x2), int(y2))})
            engine.set_lines(st.session_state.lines)
            st.success(f"{name} qo'shildi")

    if st.session_state.lines:
        st.write("**Faol liniyalar:**")
        for i, ln in enumerate(st.session_state.lines):
            c1, c2 = st.columns([4, 1])
            c1.write(f"🟡 {ln['name']}: {ln['p1']} → {ln['p2']}")
            if c2.button("O'chirish", key=f"del_{i}"):
                st.session_state.lines.pop(i)
                engine.set_lines(st.session_state.lines)
                st.rerun()

# ========== 📊 STATS ==========
with tab_stats:
    @st.fragment(run_every=3)
    def stats_view():
        today = datetime.now().strftime("%Y-%m-%d")
        total = query_df("SELECT type, COUNT(*) c FROM crossings GROUP BY type")
        tdict = dict(zip(total["type"], total["c"])) if not total.empty else {}
        tp, tc = tdict.get("person", 0), tdict.get("car", 0)

        td = query_df("SELECT type, COUNT(*) c FROM crossings WHERE substr(timestamp,1,10)=? GROUP BY type",
                      (today,))
        d = dict(zip(td["type"], td["c"])) if not td.empty else {}

        m = st.columns(4)
        m[0].metric("🚶 Bugun odamlar", d.get("person", 0))
        m[1].metric("🚗 Bugun mashinalar", d.get("car", 0))
        m[2].metric("Jami odamlar (DB)", tp)
        m[3].metric("Jami mashinalar (DB)", tc)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Bugungi soatlik traffic**")
            hourly = query_df(
                "SELECT substr(timestamp,12,2) soat, type, COUNT(*) c FROM crossings "
                "WHERE substr(timestamp,1,10)=? GROUP BY soat, type ORDER BY soat", (today,))
            if hourly.empty:
                st.caption("Bugun uchun hali ma'lumot yo'q.")
            else:
                st.bar_chart(hourly.pivot_table(index="soat", columns="type",
                                                values="c", fill_value=0))
        with c2:
            st.markdown("**Yo'nalishlar bo'yicha (jami)**")
            dirs = query_df("SELECT direction, COUNT(*) c FROM crossings GROUP BY direction")
            if dirs.empty:
                st.caption("Ma'lumot yo'q.")
            else:
                st.bar_chart(dirs.set_index("direction"))

        st.markdown("**Kunlik dinamika (oxirgi 14 kun)**")
        daily = query_df(
            "SELECT substr(timestamp,1,10) kun, type, COUNT(*) c FROM crossings "
            "GROUP BY kun, type ORDER BY kun DESC LIMIT 60")
        if daily.empty:
            st.caption("Ma'lumot yo'q.")
        else:
            piv = daily.pivot_table(index="kun", columns="type", values="c",
                                    fill_value=0).sort_index().tail(14)
            st.line_chart(piv)

        with st.expander("Oxirgi 20 ta yozuv (DB)"):
            st.dataframe(query_df(
                "SELECT timestamp, type, direction, line_name FROM crossings "
                "ORDER BY id DESC LIMIT 20"), use_container_width=True)

    stats_view()

# ========== 💬 CHAT ==========
with tab_chat:
    st.subheader("Statistika chati (qoidaga asoslangan, AI emas)")
    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(msg)
    q = st.chat_input("Masalan: Bugun qancha odam o'tdi?")
    if q:
        st.session_state.chat.append(("user", q))
        st.session_state.chat.append(("assistant", answer_question(q)))
        st.rerun()
