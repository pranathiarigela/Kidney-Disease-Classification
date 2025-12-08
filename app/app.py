# app/app.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from PIL import Image
import io
import time
import json
import numpy as np
import torch
import torch.nn.functional as F

from src.pipeline.prediction_pipeline import PredictionPipeline

# Page
st.set_page_config(page_title="Kidney Classifier", layout="wide", initial_sidebar_state="auto")

# Minimal CSS (no white bars)
st.markdown("""
<style>
:root{
  --accent:#0b6fa6;
  --muted:#7b8794;
  --bg:#0b0f14;
  --card-bg: rgba(255,255,255,0.02);
  --radius:10px;
  --shadow: 0 6px 18px rgba(11,111,166,0.06);
}
body { background-color: #0b0f14; color: #e6eef8; }
.card { background: var(--card-bg); border-radius: var(--radius); padding:14px; box-shadow: var(--shadow); margin-bottom:12px; }
.header { font-weight:700; color:var(--accent); font-size:20px; }
.sub { color:var(--muted); font-size:13px; margin-bottom:8px; }
.placeholder {
  height:220px; display:flex; align-items:center; justify-content:center;
  border-radius:8px; background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.02));
  color: #9aa4b2;
}
.small { color:var(--muted); font-size:12px; }
.bar { height:10px; border-radius:8px; background: rgba(255,255,255,0.04); overflow:hidden; }
.fill { height:100%; background: linear-gradient(90deg,var(--accent), #2a8bd6); min-width:2px; }
.pred-class { font-size:22px; font-weight:800; color: #0b6fa6; margin-top:8px; }
</style>
""", unsafe_allow_html=True)

# Sidebar: only auto and cpu; removed models text
with st.sidebar:
    st.markdown("### Settings")
    device_choice = st.selectbox("Device", ["auto", "cpu"], index=0, help="Use CPU if GPU gives issues.")
    st.markdown("---")
    st.markdown("Small tips")
    st.markdown("- Upload clear CT/US images.\n- Use CPU if GPU memory is limited.")
    st.markdown("---")
    st.caption("Demo only — not a medical diagnostic tool.")

# Header
col1, _ = st.columns([6,1])
with col1:
    st.markdown("<div class='header'>Kidney Disease Classifier</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub'>Classify kidney scans into: Normal • Cyst • Stone • Tumor</div>", unsafe_allow_html=True)

# Load predictor (cached); show friendly message if fails
device = None if device_choice == "auto" else "cpu"
predictor = None
predictor_error = None
try:
    predictor = PredictionPipeline(device=device)
except Exception as e:
    predictor_error = str(e)

# Layout: left = upload & predict & immediate class; right = detailed results
left, right = st.columns([2, 1])

# LEFT: upload, predict, immediate simple class output
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### Upload image")
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])
    st.markdown("<div class='small'>Images are resized to 224×224 before prediction.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
            # reduced preview size
            st.image(image, caption="Preview", width=420)
        except Exception as e:
            st.error("Unable to read uploaded image: " + str(e))
            image = None
    else:
        st.markdown("<div class='placeholder'>No image uploaded — choose an image to predict</div>", unsafe_allow_html=True)
        image = None
    st.markdown("</div>", unsafe_allow_html=True)

    # compact model-load error
    if predictor_error:
        st.error("Model failed to initialize. Check your models folder and label_map.json.")
        st.caption(predictor_error)

    # Predict button
    prediction = None
    probs = None
    elapsed = None
    if predictor and image is not None:
        if st.button("Predict", key="predict"):
            with st.spinner("Running prediction..."):
                t0 = time.time()
                try:
                    prediction = predictor.predict(image)
                    # attempt to get full probs for details panel
                    try:
                        x = predictor._preprocess(image)
                        with torch.no_grad():
                            logits = predictor.model(x)
                            probs = F.softmax(logits, dim=1).cpu().numpy().squeeze()
                    except Exception:
                        probs = None
                    elapsed = time.time() - t0
                except Exception as e:
                    st.error("Prediction failed: " + str(e))
                    prediction = None
                    probs = None

    # Show ONLY the predicted class here (immediately under button) when available
    if prediction is not None:
        pred_cls = prediction.get("class", "unknown")
        st.markdown(f"<div class='pred-class'>{pred_cls}</div>", unsafe_allow_html=True)
    else:
        # guidance shown until prediction
        if not predictor_error:
            st.markdown("<div class='small'>Upload an image and press Predict. The predicted class will appear here.</div>", unsafe_allow_html=True)

# RIGHT: detailed results (render only after a prediction)
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### Results")
    if prediction is None:
        st.markdown("<div class='small'>No prediction yet. Detailed results will appear here after you predict.</div>", unsafe_allow_html=True)
    else:
        # class (repeat, smaller) and confidence bar
        st.markdown(f"<div style='font-weight:700;color:#0b6fa6'>{pred_cls}</div>", unsafe_allow_html=True)
        if elapsed is not None:
            st.markdown(f"<div class='small'>Time: {elapsed:.2f}s</div>", unsafe_allow_html=True)
        conf = float(prediction.get("confidence", 0.0))
        pct = max(0.0, min(1.0, conf))
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bar'><div class='fill' style='width:{int(pct*100)}%'></div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small'>Confidence: {conf:.3f}</div>", unsafe_allow_html=True)

        # top-3 probabilities if available
        if probs is not None:
            st.markdown("Top predictions")
            idxs = np.argsort(probs)[::-1][:3]
            for i in idxs:
                name = predictor.inv_label_map.get(int(i), str(i))
                p = float(probs[int(i)])
                st.markdown(f"<div style='display:flex;justify-content:space-between'><div>{name}</div><div class='small'>{p:.3f}</div></div>", unsafe_allow_html=True)

        # download JSON
        payload = {"predicted_class": pred_cls, "confidence": conf, "time_s": round(elapsed if elapsed else 0.0, 3)}
        buf = io.BytesIO()
        buf.write(json.dumps(payload, indent=2).encode("utf-8"))
        buf.seek(0)
        st.download_button("Download result (JSON)", buf, file_name="prediction.json", mime="application/json")

    st.markdown("</div>", unsafe_allow_html=True)

# Footer note
st.markdown("<div style='text-align:center;margin-top:12px;color:#9aa4b2'>Demo app — not a clinical diagnostic tool.</div>", unsafe_allow_html=True)
