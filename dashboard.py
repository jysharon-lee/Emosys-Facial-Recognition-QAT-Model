import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import time
import os

st.set_page_config(page_title="EmoSys Live Monitor", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Minimal SaaS Look ---
st.markdown("""
    <style>
        /* Base styles */
        .stApp {
            background-color: #F7F7F7;
        }
        /* Hide main menu and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Metric cards */
        div[data-testid="metric-container"] {
            background-color: #FFFFFF;
            border: 1px solid #E8E8E8;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        /* Hero Card Override */
        .hero-metric div[data-testid="metric-container"] {
            background-color: #2D3282;
            color: white !important;
        }
        .hero-metric label, .hero-metric div {
            color: white !important;
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #E8E8E8;
        }
    </style>
""", unsafe_allow_html=True)

# Auto refresh every 1000ms
st_autorefresh(interval=1000, limit=None, key="data_refresh")

CSV_FILE = "live/live_data.csv"

# -----------------
# SIDEBAR
# -----------------
with st.sidebar:
    st.markdown("## **EmoSys**")
    st.markdown("<small style='color:gray;'>MENU</small>", unsafe_allow_html=True)
    st.markdown("<span style='color:#2D3282; font-weight:bold;'>🔹 Live Monitor</span>", unsafe_allow_html=True)
    st.markdown("🔸 Session History")
    st.markdown("🔸 Settings")
    st.markdown("---")
    st.markdown("<small style='color:gray;'>Session Active</small>", unsafe_allow_html=True)


# -----------------
# DATA LOADING
# -----------------
if not os.path.exists(CSV_FILE):
    st.warning("Awaiting live data from Raspberry Pi inference script... Make sure qat_student_tflite_pi.py is running.")
    st.stop()

try:
    df = pd.read_csv(CSV_FILE)
except Exception as e:
    st.error(f"Error reading data: {e}")
    st.stop()

if df.empty:
    st.info("No data yet. Waiting for script to write to CSV...")
    st.stop()

# -----------------
# TOP BAR & TABS
# -----------------
col1, col2 = st.columns([2, 1])
with col1:
    st.title("Live Monitor")
    st.markdown("<p style='color:gray; margin-top:-15px;'>Real-time emotional and environmental analysis</p>", unsafe_allow_html=True)

with col2:
    st.markdown(f"<div style='text-align:right;'><span style='color:green;'>● Live</span> &nbsp; {time.strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)
    
    # Person selector
    people = df[df['person_id'] != 'None']['person_id'].unique()
    if len(people) > 0:
        selected_person = st.selectbox("Select Person", people, index=0, label_visibility="collapsed")
        df_person = df[df['person_id'] == selected_person]
    else:
        st.write("No faces detected recently.")
        df_person = df

# -----------------
# CURRENT METRICS (ROW 1)
# -----------------
st.write("---")
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

if not df_person.empty:
    latest = df_person.iloc[-1]
    
    # 1. Dominant Emotion
    with col_m1:
        st.markdown("<div class='hero-metric'>", unsafe_allow_html=True)
        if latest['person_id'] != 'None':
            val = str(latest['dominant_emotion']).upper()
        else:
            val = "NONE"
        st.metric(label="Dominant Emotion", value=val)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # 2. Posture Status
    with col_m2:
        st.metric(label="Posture Status", value=latest['posture_label'], delta=f"Score: {latest['posture_score']:.2f}", delta_color="off")
        
    # 3. Env. Discomfort
    with col_m3:
        disc = latest['discomfort']
        disc_lbl = "Comfortable" if disc < 50 else ("Moderate" if disc < 80 else "High")
        st.metric(label="Env. Discomfort", value=f"{disc:.0f}%", delta=disc_lbl, delta_color="inverse")
        
    # 4. Avg Confidence
    with col_m4:
        if latest['person_id'] != 'None':
            avg_conf = df_person[latest['dominant_emotion']].mean() * 100
            st.metric(label="Avg Confidence", value=f"{avg_conf:.1f}%", delta="This session", delta_color="off")
        else:
            st.metric(label="Avg Confidence", value="N/A", delta="This session", delta_color="off")

# -----------------
# EMOTION CHARTS (ROW 2)
# -----------------
st.write("")
col_c1, col_c2 = st.columns([2, 1])

with col_c1:
    st.markdown("**Emotion Confidence** <small style='color:gray;'>(Full Session)</small>", unsafe_allow_html=True)
    if not df_person.empty and latest['person_id'] != 'None':
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        fig = go.Figure()
        colors = ['#2D3282', '#6b7280', '#f59e0b', '#10b981', '#3b82f6', '#ef4444', '#8b5cf6']
        for i, emo in enumerate(emotions):
            fig.add_trace(go.Scatter(
                x=df_person['timestamp'] - df['timestamp'].min(), 
                y=df_person[emo] * 100, 
                mode='lines',
                name=emo.capitalize(),
                line=dict(color=colors[i], width=2)
            ))
            
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='white',
            paper_bgcolor='white',
            yaxis=dict(range=[0, 100], gridcolor='#E8E8E8', title="Confidence %"),
            xaxis=dict(gridcolor='#E8E8E8', title="Time (s)"),
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No face data to display emotion chart.")

with col_c2:
    st.markdown("**Emotion Distribution**", unsafe_allow_html=True)
    if not df_person.empty and latest['person_id'] != 'None':
        counts = df_person['dominant_emotion'].value_counts()
        fig_bar = px.bar(
            x=counts.values,
            y=counts.index,
            orientation='h',
            color_discrete_sequence=['#2D3282']
        )
        fig_bar.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(gridcolor='#E8E8E8', title="Frames Detected"),
            yaxis=dict(title=""),
            height=300
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No face data.")

# -----------------
# POSTURE & ENVIRONMENT (ROW 3)
# -----------------
st.write("")
col_c3, col_c4 = st.columns(2)

with col_c3:
    st.markdown("**Posture Tension**", unsafe_allow_html=True)
    if not df_person.empty and latest['person_id'] != 'None':
        fig_pos = go.Figure()
        fig_pos.add_trace(go.Scatter(
            x=df_person['timestamp'] - df['timestamp'].min(), 
            y=df_person['posture_score'],
            mode='lines',
            name="Tension",
            line=dict(color='#2D3282', width=2)
        ))
        # Thresholds
        fig_pos.add_hline(y=0.5, line_dash="dash", line_color="#f59e0b", annotation_text="Slightly Tense")
        fig_pos.add_hline(y=1.0, line_dash="dash", line_color="#ef4444", annotation_text="Tense")
        
        fig_pos.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='white',
            paper_bgcolor='white',
            yaxis=dict(range=[-0.1, 1.2], gridcolor='#E8E8E8', title="Tension Score"),
            xaxis=dict(gridcolor='#E8E8E8', title="Time (s)"),
            height=300
        )
        st.plotly_chart(fig_pos, use_container_width=True)
    else:
        st.info("No face data to display posture.")

with col_c4:
    st.markdown("**Environment Sensors**", unsafe_allow_html=True)
    if not df.empty:
        fig_env = go.Figure()
        # Create secondary y-axis logic for temp/hum
        from plotly.subplots import make_subplots
        fig_env = make_subplots(specs=[[{"secondary_y": True}]])
        
        t_hist = df['timestamp'] - df['timestamp'].min()
        fig_env.add_trace(go.Scatter(x=t_hist, y=df['co2'], name="CO2 (ppm)", line=dict(color="#3b82f6")), secondary_y=False)
        fig_env.add_trace(go.Scatter(x=t_hist, y=df['voc'], name="VOC", line=dict(color="#8b5cf6")), secondary_y=False)
        fig_env.add_trace(go.Scatter(x=t_hist, y=df['pm'], name="PM", line=dict(color="#f59e0b")), secondary_y=False)
        
        fig_env.add_trace(go.Scatter(x=t_hist, y=df['temp'], name="Temp (C)", line=dict(color="#ef4444", dash="dash")), secondary_y=True)
        fig_env.add_trace(go.Scatter(x=t_hist, y=df['humidity'], name="Hum (%)", line=dict(color="#10b981", dash="dash")), secondary_y=True)
        
        fig_env.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(gridcolor='#E8E8E8', title="Time (s)"),
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_env.update_yaxes(title_text="CO2 / VOC / PM", secondary_y=False, gridcolor='#E8E8E8')
        fig_env.update_yaxes(title_text="Temp / Humidity", secondary_y=True, showgrid=False)
        
        st.plotly_chart(fig_env, use_container_width=True)

# -----------------
# DOWNLOAD
# -----------------
st.write("---")
with open(CSV_FILE, "r") as f:
    st.download_button(
        label="↓ Download Session Report",
        data=f,
        file_name=f"emosys_session_{time.strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
