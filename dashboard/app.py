"""
SurgIQ — Streamlit Dashboard
=============================
Real-time surgical training dashboard that reads pipeline results
from Redis and displays:

  - Live instrument activity label + confidence
  - Active track count
  - LLM coaching feedback
  - Instrument activity timeline (rolling chart)
  - Per-class frame distribution (pie chart)
  - Session summary stats

Usage:
    # Make sure the pipeline is running first:
    python main.py --source video.mp4

    # Then launch the dashboard:
    streamlit run dashboard/app.py
"""

import sys
import json
import time
from pathlib import Path
from collections import deque, defaultdict

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from pipeline.redis_consumer import RedisConsumer


# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "SurgIQ — Surgical Training Coach",
    page_icon  = "🔬",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #7c3aed;
    }
    .feedback-box {
        background: #0f3460;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #00d4ff;
        font-style: italic;
        color: #e0f0ff;
        margin-top: 0.5rem;
    }
    .label-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .stMetric label { color: #aaaacc !important; }
</style>
""", unsafe_allow_html=True)

# ── Label colours ─────────────────────────────────────────────────────────────

LABEL_COLORS = {
    "no_instrument"   : "#6b7280",
    "grasper_only"    : "#a855f7",
    "hook_only"       : "#f59e0b",
    "both_instruments": "#10b981",
}

LABEL_ICONS = {
    "no_instrument"   : "⚫",
    "grasper_only"    : "🟣",
    "hook_only"       : "🟡",
    "both_instruments": "🟢",
}

# ── Session State ─────────────────────────────────────────────────────────────

HISTORY_LEN = 200   # rolling window of frames to display

if "consumer"      not in st.session_state:
    st.session_state.consumer      = None
if "history"       not in st.session_state:
    st.session_state.history       = deque(maxlen=HISTORY_LEN)
if "label_counts"  not in st.session_state:
    st.session_state.label_counts  = defaultdict(int)
if "total_frames"  not in st.session_state:
    st.session_state.total_frames  = 0
if "last_feedback" not in st.session_state:
    st.session_state.last_feedback = "Waiting for pipeline..."
if "session_id"    not in st.session_state:
    st.session_state.session_id    = "demo"
if "spoken_feedback" not in st.session_state:
    st.session_state.spoken_feedback = ""
if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = True


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/surgery.png", width=60)
    st.title("SurgIQ")
    st.caption("Real-Time Surgical Training Coach")
    st.divider()

    session_id = st.text_input(
        "Session ID", value=st.session_state.session_id,
        help="Must match --session-id in main.py"
    )

    if session_id != st.session_state.session_id:
        st.session_state.session_id   = session_id
        st.session_state.consumer     = None
        st.session_state.history      = deque(maxlen=HISTORY_LEN)
        st.session_state.label_counts = defaultdict(int)
        st.session_state.total_frames = 0

    refresh_rate = st.slider(
        "Refresh rate (seconds)", 0.5, 3.0, cfg.DASHBOARD_REFRESH_S, 0.1
    )

    st.divider()
    st.markdown("**Pipeline status**")
    pipeline_placeholder = st.empty()

    st.divider()
    st.session_state.voice_enabled = st.toggle(
        "🔊 Voice coaching", value=st.session_state.voice_enabled,
        help="Speak coaching feedback aloud when it changes"
    )

    st.divider()
    if st.button("🗑️ Reset session", use_container_width=True):
        st.session_state.history      = deque(maxlen=HISTORY_LEN)
        st.session_state.label_counts = defaultdict(int)
        st.session_state.total_frames = 0
        st.session_state.last_feedback = "Session reset."
        st.rerun()

    st.divider()
    st.markdown("**How to start the pipeline:**")
    st.code("python main.py --source video.mp4", language="bash")
    st.code("streamlit run dashboard/app.py", language="bash")


# ── Connect to Redis ──────────────────────────────────────────────────────────

if st.session_state.consumer is None:
    try:
        st.session_state.consumer = RedisConsumer(session_id=st.session_state.session_id)
        st.session_state.consumer.reset()   # read from beginning
        pipeline_placeholder.success("Redis connected ✅")
    except Exception as e:
        pipeline_placeholder.error(f"Redis error: {e}")
        st.stop()


# ── Pull new messages ─────────────────────────────────────────────────────────

consumer = st.session_state.consumer
new_msgs  = consumer.read_all(count=500)

for msg in new_msgs:
    st.session_state.history.append(msg)
    st.session_state.label_counts[msg["classifier_label"]] += 1
    st.session_state.total_frames += 1
    if msg["feedback"]:
        st.session_state.last_feedback = msg["feedback"]

history      = list(st.session_state.history)
label_counts = dict(st.session_state.label_counts)
total_frames = st.session_state.total_frames


# ── Header ────────────────────────────────────────────────────────────────────

st.title("🔬 SurgIQ — Surgical Training Dashboard")
st.caption(f"Session: `{st.session_state.session_id}`  •  "
           f"Frames processed: {total_frames}")
st.divider()


# ── Current State ─────────────────────────────────────────────────────────────

latest = history[-1] if history else None

col1, col2, col3, col4 = st.columns(4)

with col1:
    label = latest["classifier_label"] if latest else "—"
    conf  = latest["classifier_conf"]  if latest else 0.0
    icon  = LABEL_ICONS.get(label, "⚪")
    color = LABEL_COLORS.get(label, "#888")
    st.markdown(f"**Current Activity**")
    st.markdown(
        f'<span class="label-badge" style="background:{color}22; '
        f'border:2px solid {color}; color:{color}">'
        f'{icon} {label.replace("_", " ").title()}</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"Confidence: {conf:.1%}")

with col2:
    tracks     = latest["tracks"] if latest else []
    num_tracks = len(tracks)
    st.metric("Active Tracks", num_tracks,
              help="Number of instruments being tracked this frame")

with col3:
    frame_idx = latest["frame_idx"] if latest else 0
    st.metric("Frame", frame_idx)

with col4:
    if history and len(history) > 1:
        t0 = history[0]["timestamp"]
        t1 = history[-1]["timestamp"]
        dur = t1 - t0
        fps = len(history) / dur if dur > 0 else 0
        st.metric("Pipeline FPS", f"{fps:.1f}")
    else:
        st.metric("Pipeline FPS", "—")


# ── LLM Feedback ──────────────────────────────────────────────────────────────

st.markdown("### 💬 Coaching Feedback")
st.markdown(
    f'<div class="feedback-box">{st.session_state.last_feedback}</div>',
    unsafe_allow_html=True,
)

# ── Voice Feedback (browser Web Speech API) ───────────────────────────────────

current_feedback = st.session_state.last_feedback
should_speak = (
    st.session_state.voice_enabled
    and current_feedback != st.session_state.spoken_feedback
    and current_feedback != "Waiting for pipeline..."
)

if should_speak:
    st.session_state.spoken_feedback = current_feedback
    # Take first 2 sentences to keep it short
    sentences = current_feedback.replace("  ", " ").split(". ")
    short_text = ". ".join(sentences[:2])
    if not short_text.endswith("."):
        short_text += "."
    # Escape for JS string
    safe_text = short_text.replace("'", "\\'").replace("\n", " ")

    components.html(f"""
        <script>
            window.speechSynthesis.cancel();
            var msg = new SpeechSynthesisUtterance('{safe_text}');
            msg.rate  = 0.95;
            msg.pitch = 1.0;
            msg.volume = 1.0;
            // Use a clear voice if available
            var voices = window.speechSynthesis.getVoices();
            var preferred = voices.find(v =>
                v.name.includes('Samantha') ||
                v.name.includes('Google US') ||
                v.name.includes('Karen')
            );
            if (preferred) msg.voice = preferred;
            window.speechSynthesis.speak(msg);
        </script>
    """, height=0)


# ── Charts ────────────────────────────────────────────────────────────────────

st.divider()
chart_col1, chart_col2 = st.columns([2, 1])

with chart_col1:
    st.markdown("### 📈 Instrument Activity Timeline")
    if len(history) > 1:
        df = pd.DataFrame(history)
        df["frame_idx"] = df["frame_idx"].astype(int)

        label_to_int = {
            "no_instrument"   : 0,
            "grasper_only"    : 1,
            "hook_only"       : 2,
            "both_instruments": 3,
        }
        df["label_int"] = df["classifier_label"].map(label_to_int)

        fig = go.Figure()
        for label, val in label_to_int.items():
            mask = df["classifier_label"] == label
            fig.add_trace(go.Scatter(
                x    = df[mask]["frame_idx"],
                y    = df[mask]["classifier_conf"],
                mode = "markers",
                name = label.replace("_", " ").title(),
                marker = dict(
                    color = LABEL_COLORS.get(label, "#888"),
                    size  = 6,
                ),
            ))

        fig.update_layout(
            height     = 300,
            margin     = dict(l=0, r=0, t=10, b=0),
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            legend     = dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis_title = "Frame",
            yaxis_title = "Confidence",
            yaxis       = dict(range=[0, 1]),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waiting for pipeline data...")


with chart_col2:
    st.markdown("### 🥧 Activity Distribution")
    if label_counts:
        labels = [k.replace("_", " ").title() for k in label_counts]
        values = list(label_counts.values())
        colors = [LABEL_COLORS.get(k, "#888") for k in label_counts]

        fig2 = go.Figure(go.Pie(
            labels    = labels,
            values    = values,
            marker    = dict(colors=colors),
            hole      = 0.4,
            textinfo  = "percent+label",
            textfont  = dict(size=11),
        ))
        fig2.update_layout(
            height        = 300,
            margin        = dict(l=0, r=0, t=10, b=0),
            paper_bgcolor = "rgba(0,0,0,0)",
            showlegend    = False,
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No data yet.")


# ── Active Tracks Table ───────────────────────────────────────────────────────

if latest and latest["tracks"]:
    st.divider()
    st.markdown("### 🎯 Active Instrument Tracks")
    track_df = pd.DataFrame(latest["tracks"])
    track_df["confidence"] = track_df["confidence"].map("{:.1%}".format)
    track_df.columns = [c.replace("_", " ").title() for c in track_df.columns]
    st.dataframe(track_df, use_container_width=True, hide_index=True)


# ── Session Summary ───────────────────────────────────────────────────────────

if total_frames > 0:
    st.divider()
    st.markdown("### 📊 Session Summary")
    summary_cols = st.columns(len(label_counts) or 1)
    for i, (lbl, cnt) in enumerate(label_counts.items()):
        pct   = cnt / total_frames * 100
        color = LABEL_COLORS.get(lbl, "#888")
        with summary_cols[i % len(summary_cols)]:
            st.markdown(
                f'<div class="metric-card" style="border-left-color:{color}">'
                f'<div style="color:{color};font-weight:bold">'
                f'{LABEL_ICONS.get(lbl,"⚪")} {lbl.replace("_"," ").title()}</div>'
                f'<div style="font-size:1.8rem;font-weight:bold">{pct:.1f}%</div>'
                f'<div style="color:#888">{cnt} frames</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Auto-refresh ──────────────────────────────────────────────────────────────

time.sleep(refresh_rate)
st.rerun()
