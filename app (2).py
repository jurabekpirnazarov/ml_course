import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os
from datetime import datetime, timedelta
from analytics import CrowdAnalytics, generate_ai_insights
import plotly.express as px
import plotly.graph_objects as go
from urllib import error, request
from streamlit.runtime.scriptrunner import get_script_run_ctx

if get_script_run_ctx(suppress_warning=True) is None:
    print("This app must be run with: streamlit run app.py")
    raise SystemExit(0)

st.set_page_config(page_title="ASSBI Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
        font-family: 'Outfit', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #6366f1, #38bdf8, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }

    .sub-header {
        color: #9ca3af;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }

    div[data-testid="metric-container"] {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 20px 24px;
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.3);
    }

    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 800;
        color: #38bdf8;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.95rem;
        color: #9ca3af;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTextInput>div>div>input {
        background-color: #1e293b;
        color: #ffffff;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stButton>button {
        background: linear-gradient(135deg, #6366f1, #38bdf8);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 20px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        transition: all 0.2s;
    }

    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
    }

    .alert-high {
        background-color: rgba(231, 76, 60, 0.1);
        border-left: 4px solid #e74c3c;
    }

    .alert-medium {
        background-color: rgba(241, 196, 15, 0.1);
        border-left: 4px solid #f1c40f;
    }

    .alert-normal {
        background-color: rgba(46, 204, 113, 0.1);
        border-left: 4px solid #2ecc71;
    }
</style>
""", unsafe_allow_html=True)


def _default_stats():
    return {
        'total': 0,
        'avg': 0,
        'max': 0,
        'min': 0,
        'std': 0,
        'count': 0,
        'peak_hour': 'N/A',
        'trend': 0,
    }


def _get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
        if secret_key:
            api_key = str(secret_key).strip()
    except Exception:
        pass

    typed_key = st.session_state.get("openai_api_key", "").strip()
    if typed_key:
        api_key = typed_key

    return api_key


def _build_ai_context(location, df, stats, current, density, peak, current_level, risk_score, anomaly_score, anomalies, hourly):
    recent_rows = []
    if not df.empty:
        recent_rows = [
            {
                "Timestamp": row["Timestamp"].strftime("%Y-%m-%d %H:%M:%S") if pd.notna(row["Timestamp"]) else None,
                "People": int(row["People"]),
            }
            for _, row in df.tail(15).iterrows()
        ]

    hourly_rows = []
    if hourly is not None and not hourly.empty:
        hourly_rows = hourly.sort_values("Hour").head(12).to_dict(orient="records")

    anomaly_rows = []
    if anomalies is not None and not anomalies.empty:
        anomaly_rows = anomalies[['Timestamp', 'People']].tail(10).copy()
        anomaly_rows['Timestamp'] = anomaly_rows['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        anomaly_rows = anomaly_rows.to_dict(orient="records")

    context = {
        "location": location,
        "current": int(current),
        "density": float(density),
        "peak": int(peak),
        "current_level": current_level,
        "risk_score": float(risk_score),
        "anomaly_score": float(anomaly_score),
        "statistics": {
            "total": int(stats.get("total", 0)),
            "avg": float(stats.get("avg", 0)),
            "max": int(stats.get("max", 0)),
            "min": int(stats.get("min", 0)),
            "std": float(stats.get("std", 0)),
            "count": int(stats.get("count", 0)),
            "peak_hour": stats.get("peak_hour", "N/A"),
            "trend": float(stats.get("trend", 0)),
        },
        "recent_records": recent_rows,
        "hourly_distribution": hourly_rows,
        "anomalies": anomaly_rows,
    }

    return json.dumps(context, ensure_ascii=False, indent=2, default=str)


def _call_openai_chat(api_key, model, messages):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 600,
    }

    req = request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI request failed: {exc.code} {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenAI connection failed: {exc.reason}") from exc

    choices = result.get("choices", [])
    if not choices:
        raise RuntimeError("OpenAI returned an empty response")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))

    return str(content).strip()


def _generate_ai_reply(prompt_text, context_text, history, model):
    system_message = (
        "Siz ASSBI dashboard uchun AI analitikasiz. "
        "Foydalanuvchining savollariga faqat berilgan dashboard konteksti va chat tarixiga tayangan holda javob bering. "
        "Javobni o'zbek tilida, aniq, amaliy va qisqa yozing. "
        "Agar ma'lumot yetarli bo'lmasa, buni ochiq ayting va nimalar kerakligini ko'rsating."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "system", "content": f"Joriy dashboard ma'lumotlari:\n{context_text}"},
    ]

    for item in history[-8:]:
        if item.get("role") in {"user", "assistant"} and item.get("content"):
            messages.append({"role": item["role"], "content": item["content"]})

    messages.append({"role": "user", "content": prompt_text})
    return _call_openai_chat(_get_openai_api_key(), model, messages)

st.markdown('<h1 class="main-header">🛡️ AI-Powered Smart Surveillance & BI</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.title("⚙️ ASSBI Analytics Settings")
    location = st.selectbox("📍 Lokatsiya", ["Walworth Road, London (Live)", "Universities", "Public Streets"])

    refresh_rate = st.slider("🔄 Yangilash tezligi (soniya)", 5, 60, 10)

    st.markdown("### 🤖 AI Chat Sozlamalari")
    st.text_input(
        "OpenAI API key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        key="openai_api_key",
        help="Bu kalit faqat shu sessiyada ishlatiladi va faylga yozilmaydi.",
    )
    ai_model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4.1-mini"], index=0)

    tab_nav = st.radio("📋 Bo'lim tanlang:",
                       ["🏠 Dashboard", "📊 Analytics", "⚠️ Anomalies", "🔮 Forecast", "📄 Reports"])

analytics = CrowdAnalytics("data_log.csv")

try:
    if os.path.exists("data_log.csv") and os.path.getsize("data_log.csv") > 20:
        df = pd.read_csv("data_log.csv")
        if 'Vehicles' in df.columns and 'People' not in df.columns:
            df = df.rename(columns={'Vehicles': 'People'})
        if not df.empty:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            stats = analytics.get_statistics()
            current, density, peak = analytics.get_crowd_density()
            current_level, level_color = analytics.get_crowd_level(current)
            status = "🟢 Online"
        else:
            df = pd.DataFrame(columns=["Timestamp", "People"])
            current, density, peak = 0, 0, 0
            current_level = "N/A"
            level_color = "#95a5a6"
            stats = _default_stats()
            status = "🟢 Online"
    else:
        df = pd.DataFrame(columns=["Timestamp", "People"])
        current, density, peak = 0, 0, 0
        current_level = "N/A"
        level_color = "#95a5a6"
        stats = _default_stats()
        status = "🔴 Offline"
except Exception as e:
    df = pd.DataFrame(columns=["Timestamp", "People"])
    current, density, peak = 0, 0, 0
    current_level = "N/A"
    level_color = "#95a5a6"
    stats = _default_stats()
    status = "🔴 Error"

if tab_nav == "🏠 Dashboard":
    st.markdown(f'<p class="sub-header">📡 Real-time monitoring for: <b>{location}</b></p>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🚗 Joriy Transport Soni", int(current), f"{current_level}")
    col2.metric("📊 O'rtacha Oqim", f"{density:.1f}", "vehicle/10min")
    col3.metric("📈 Maksimal Pik", int(peak), f"{stats['max']} bugun")
    col4.metric("🎯 Jami Transport", int(stats['total']), f"Trend: {stats['trend']:+.1f}%")
    col5.metric("🛡️ Kamera", status, "Live")

    st.divider()

    col_main, col_side = st.columns([3, 1])

    with col_main:
        st.subheader("📈 Real-time Traffic Trend")
        if not df.empty:
            df_plot = df.set_index('Timestamp')[['People']]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot['People'],
                mode='lines+markers',
                name='Vehicle Count',
                line=dict(color='#38bdf8', width=3),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor='rgba(56, 189, 248, 0.2)'
            ))

            if len(df_plot) > 2:
                mean_val = df_plot['People'].mean()
                fig.add_hline(y=mean_val, line_dash="dash", line_color="orange",
                            annotation_text="O'rtacha", annotation_position="right")

            fig.update_layout(
                template="plotly_dark",
                hovermode="x unified",
                height=350,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("📊 Hozircha ma'lumot yo'q. `python live_monitoring.py` dasturini ishga tushiring.")

    with col_side:
        st.subheader("🎯 KPI Holati")

        risk_score = analytics.get_risk_score()
        anomaly_score = analytics.get_anomaly_score()

        if risk_score > 70:
            st.markdown(f'<div class="alert-high">⚠️ RISK: {risk_score:.1f}%</div>', unsafe_allow_html=True)
        elif risk_score > 40:
            st.markdown(f'<div class="alert-medium">⚡ RISK: {risk_score:.1f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-normal">✅ RISK: {risk_score:.1f}%</div>', unsafe_allow_html=True)

            st.markdown(f"**Anomaliya skori:** {anomaly_score:.1f}/5")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🚨 Anomaly Detection Log")
        anomalies = analytics.detect_anomalies(sensitivity=2.0)
        if not anomalies.empty:
            anomalies_display = anomalies[['Timestamp', 'People']].tail(5).copy()
            anomalies_display['Timestamp'] = anomalies_display['Timestamp'].dt.strftime('%H:%M:%S')
            st.dataframe(anomalies_display, width="stretch")
        else:
            st.info("✅ Anomaliya aniqlangan yo'q")

    with col_right:
        st.subheader("📊 Soati bo'yicha tarqatilishi")
        hourly = analytics.get_hourly_distribution()
        if not hourly.empty:
            fig_hourly = px.bar(hourly, x='Hour', y='Average',
                               title='Soatning o\'rtacha transport oqimi',
                               labels={'Hour': 'Soat', 'Average': 'O\'rtacha transport'},
                               color='Average',
                               color_continuous_scale='Viridis')
            fig_hourly.update_layout(template="plotly_dark", height=300, showlegend=False)
            st.plotly_chart(fig_hourly, width="stretch")

    st.divider()
    st.subheader("🤖 AI Data Chat")
    st.caption("CSV ichidagi ma'lumotlar asosida savol bering yoki qisqa xulosa so'rang.")

    if "dashboard_chat" not in st.session_state:
        st.session_state.dashboard_chat = []

    chat_clear_col, _ = st.columns([1, 5])
    with chat_clear_col:
        if st.button("🧹 Chatni tozalash", use_container_width=True):
            st.session_state.dashboard_chat = []
            st.rerun()

    for item in st.session_state.dashboard_chat:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    chat_prompt = st.chat_input("Masalan: oxirgi trendni tahlil qil, anomaliyalarni ayt yoki keyingi 1 soatni bahola")

    if chat_prompt:
        st.session_state.dashboard_chat.append({"role": "user", "content": chat_prompt})

        api_key = _get_openai_api_key()
        if not api_key:
            assistant_reply = "OpenAI API key topilmadi. Sidebar'da kalitni kiriting yoki OPENAI_API_KEY muhit o'zgaruvchisini sozlang."
        else:
            context_text = _build_ai_context(
                location,
                df,
                stats,
                current,
                density,
                peak,
                current_level,
                analytics.get_risk_score(),
                analytics.get_anomaly_score(),
                analytics.detect_anomalies(),
                hourly,
            )
            try:
                assistant_reply = _generate_ai_reply(
                    chat_prompt,
                    context_text,
                    st.session_state.dashboard_chat[:-1],
                    ai_model,
                )
            except Exception as exc:
                assistant_reply = f"AI javobini olishda xatolik yuz berdi: {exc}"

        st.session_state.dashboard_chat.append({"role": "assistant", "content": assistant_reply})
        st.rerun()

elif tab_nav == "📊 Analytics":
    st.markdown(f'<p class="sub-header">Murakkab tahlil va statistika</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Jami Transport", f"{stats['total']:,.0f}")
    col2.metric("O'rtacha", f"{stats['avg']:.1f}", f"Std: {stats['std']:.1f}")
    col3.metric("Pik soati", stats['peak_hour'])

    st.divider()

    if not df.empty:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=df['People'], nbinsx=20,
            name='Distribution',
            marker_color='#6366f1'
        ))
        fig_dist.update_layout(
            template="plotly_dark",
            title="Transport vositalari tarqatilishi",
            xaxis_title="Transport vositasi soni",
            yaxis_title="Chastota",
            height=400
        )
        st.plotly_chart(fig_dist, width="stretch")

    st.subheader("🔍 Arifmetik Tahlil")
    insights = generate_ai_insights(analytics)
    for insight in insights:
        st.write(insight)

elif tab_nav == "⚠️ Anomalies":
    st.markdown(f'<p class="sub-header">Onormal holatlarni aniqlash va tahlil</p>', unsafe_allow_html=True)

    anomaly_sensitivity = st.slider("Anomaliya sezgirlik darajasi", 1.0, 3.0, 2.0)
    anomalies = analytics.detect_anomalies(sensitivity=anomaly_sensitivity)

    col_info1, col_info2 = st.columns(2)
    col_info1.metric("Aniqlangan Anomaliyalar", len(anomalies))
    col_info2.metric("Anomaliya Skori", f"{analytics.get_anomaly_score():.2f}/5")

    if not anomalies.empty:
        st.dataframe(anomalies[['Timestamp', 'People']], width="stretch")
    else:
        st.success("✅ Anomaliya aniqlangan yo'q")

elif tab_nav == "🔮 Forecast":
    st.markdown(f'<p class="sub-header">Kelgusi soatlarning bashoratlari</p>', unsafe_allow_html=True)

    hours_to_forecast = st.slider("Bashorat davomi (soat)", 1, 6, 2)

    forecast_df = analytics.forecast_crowd(hours_ahead=hours_to_forecast)

    if forecast_df is not None and not forecast_df.empty:
        fig_forecast = go.Figure()

        if not df.empty:
            fig_forecast.add_trace(go.Scatter(
                x=df['Timestamp'], y=df['People'],
                mode='lines', name='Actual',
                line=dict(color='#38bdf8', width=2)
            ))

        fig_forecast.add_trace(go.Scatter(
            x=forecast_df['Hour'], y=forecast_df['Forecast'],
            mode='lines', name='Forecast',
            line=dict(color='#f1c40f', width=2, dash='dash')
        ))

        fig_forecast.add_trace(go.Scatter(
            x=forecast_df['Hour'], y=forecast_df['Upper'],
            fill=None, mode='lines', name='Upper Bound',
            line=dict(width=0), showlegend=False
        ))

        fig_forecast.add_trace(go.Scatter(
            x=forecast_df['Hour'], y=forecast_df['Lower'],
            fill='tonexty', mode='lines', name='Lower Bound',
            line=dict(width=0), showlegend=False,
            fillcolor='rgba(241, 196, 15, 0.2)'
        ))

        fig_forecast.update_layout(
            template="plotly_dark", height=400,
            title="Transport oqimi bashoratlari",
            hovermode="x unified"
        )
        st.plotly_chart(fig_forecast, width="stretch")
    else:
        st.info("📊 Bashorat uchun yetarli ma'lumot yo'q")

elif tab_nav == "📄 Reports":
    st.markdown(f'<p class="sub-header">Avtomatik hisobotlar va xulosa</p>', unsafe_allow_html=True)

    report_type = st.selectbox("Hisobot turi", ["📅 Kunlik", "📊 Haftalik", "📈 Oylik"])

    st.subheader("📋 Xulosa")

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    summary_col1.metric("Jami Transport", f"{stats['total']:,.0f}")
    summary_col2.metric("O'rtacha/Interval", f"{stats['avg']:.1f}")
    summary_col3.metric("Maksimal", f"{stats['max']}")
    summary_col4.metric("Trend", f"{stats['trend']:+.1f}%")

    st.subheader("🔍 Asosiy Natijalar")
    insights = generate_ai_insights(analytics)
    for insight in insights:
        st.write(f"• {insight}")

    st.subheader("⚠️ Tavsiyalar")
    if stats['trend'] > 25:
        st.warning("⬆️ Transport oqimi tezda o'smoqda - qo'shimcha kuzatuv tavsiya etiladi")
    if analytics.get_risk_score() > 60:
        st.error("🚨 Xavf darajasi yuqori - qo'shimcha monitoring talab etiladi")
    if not analytics.detect_anomalies().empty:
        st.warning("⚠️ Onormal faoliyat aniqlandi - diqqat bilan kuzatilsin")

    if st.button("📥 Hisobotni Yuklab Olish (JSON)"):
        report_data = {
            "Generated": datetime.now().isoformat(),
            "Location": location,
            "Statistics": stats,
            "CurrentStatus": {"Vehicles": int(current), "Level": current_level},
            "RiskScore": float(analytics.get_risk_score()),
            "Anomalies": len(analytics.detect_anomalies()),
            "Insights": insights
        }
        st.download_button(
            label="📥 JSON Hisobotini Yuklab Olish",
            data=str(report_data),
            file_name=f"assbi_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

st.sidebar.divider()
st.sidebar.info(f"🔄 Oxirgi yangilanish: {datetime.now().strftime('%H:%M:%S')}")
