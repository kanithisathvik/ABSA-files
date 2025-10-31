import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import requests

# -------------------------------------------------------------
#  Streamlit setup
# -------------------------------------------------------------
st.set_page_config(page_title="💬 Smart ABSA Dashboard", layout="wide", page_icon="🌈")
st.title("Aspect-Based Sentiment Analyzer")
st.caption("Analyze **any user-defined aspects** with colorful insights, charts, and summaries.")

# -------------------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------------------
st.sidebar.header("⚙️ Settings")
API_URL = "http://127.0.0.1:8000"  # FastAPI backend URL

# -------------------------------------------------------------
# Visuals
# -------------------------------------------------------------
def plot_bar(df):
    fig = px.bar(
        df, x="Aspect", y="Score (1-10)", color="Sentiment",
        text="Score (1-10)",
        color_discrete_map={"Positive":"#00CC96","Neutral":"#636EFA","Negative":"#EF553B"},
        title="Aspect-wise Sentiment Scores"
    )
    fig.update_traces(textposition="outside", hovertemplate="<b>%{x}</b><br>Score: %{y}<extra></extra>")
    fig.update_layout(template="seaborn", yaxis=dict(range=[0,10]))
    st.plotly_chart(fig, use_container_width=True)

def plot_pie(df):
    counts = df["Sentiment"].value_counts().reset_index()
    counts.columns = ["Sentiment","Count"]
    fig = px.pie(counts, names="Sentiment", values="Count", hole=0.4,
                 color="Sentiment",
                 color_discrete_map={"Positive":"#00CC96","Neutral":"#636EFA","Negative":"#EF553B"},
                 title="Overall Sentiment Breakdown")
    fig.update_traces(textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

def plot_radar(df):
    if len(df) < 3: return
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=df["Score (1-10)"], theta=df["Aspect"], fill="toself",
        line_color="#1482E9", name="Aspect Score"
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,10])),
        template="seaborn", title="Aspect Strength Radar"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_insights(df):
    avg = round(df["Score (1-10)"].mean(), 2)
    pos = df[df["Sentiment"]=="Positive"]["Aspect"].tolist()
    neg = df[df["Sentiment"]=="Negative"]["Aspect"].tolist()
    
    st.markdown(f"### 🌟 Average Sentiment Score: `{avg}/10`")
    if pos: st.success(f"✅ Strengths: {', '.join(pos)}")
    if neg: st.error(f"⚠️ Weak Points: {', '.join(neg)}")

# -------------------------------------------------------------
# Check API Status
# -------------------------------------------------------------
try:
    response = requests.get(f"{API_URL}/status")
    if response.ok:
        st.sidebar.success("✅ Backend API Connected")
        model_status = response.json()
        if model_status.get("model_loaded"):
            st.sidebar.success("✅ Model Ready")
        else:
            st.sidebar.warning("⚠️ Model Loading...")
    else:
        st.sidebar.error("❌ API Error")
except Exception as e:
    st.sidebar.error(f"❌ Cannot connect to API: {str(e)}")

# -------------------------------------------------------------
# Main Interface
# -------------------------------------------------------------
st.sidebar.header("🧠 Define Your Aspects")
custom_input = st.sidebar.text_area(
    "Enter aspects (comma separated):",
    "battery, camera, display, performance, price",
    height=100
)
aspects = [x.strip() for x in custom_input.split(",") if x.strip()]

st.subheader("📝 Single Review Analysis")
review = st.text_area(
    "Paste a review here:",
    "The battery life is amazing, camera is decent, but the phone is overpriced and performance is average.",
    height=150
)

if st.button("🔍 Analyze"):
    if not review.strip():
        st.warning("Please enter a review first.")
    elif not aspects:
        st.warning("Please define at least one aspect.")
    else:
        with st.spinner("Analyzing aspects..."):
            try:
                response = requests.post(
                    f"{API_URL}/analyze",
                    json={"review": review, "aspects": custom_input}
                )
                if response.ok:
                    result = response.json()
                    if "result" in result:
                        # Convert API results to DataFrame
                        df = pd.DataFrame(result["result"])
                        
                        # Display results
                        st.success("✅ Analysis Complete")
                        st.dataframe(df)
                        
                        # Show visualizations
                        col1, col2 = st.columns(2)
                        with col1:
                            plot_pie(df)
                        with col2:
                            plot_bar(df)
                            
                        plot_radar(df)
                        show_insights(df)
                    else:
                        st.error("No results in API response")
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("💡 Tip: You can analyze any aspects relevant to your domain (e.g., 'delivery', 'packaging', 'durability').")