import os
import sys
import time
import numpy as np
from PIL import Image
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="PatternScan — BOM Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp { background-color: #F1F5F9; }

/* Hide Streamlit chrome */
#MainMenu, header, footer { visibility: hidden; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.75rem 1rem 1rem 1rem !important;
}

/* ── Main layout ── */
.main .block-container {
    padding: 2.25rem 2.5rem !important;
    max-width: 1440px;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.1rem 1.4rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.04);
}
div[data-testid="metric-container"] > label {
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #94A3B8 !important;
}
div[data-testid="stMetricValue"] > div {
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    color: #0F172A !important;
    letter-spacing: -0.03em !important;
}

/* ── Buttons ── */
div[data-testid="stButton"] button[kind="primary"] {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.7rem 0 !important;
    transition: background 0.15s, box-shadow 0.15s !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: #1D4ED8 !important;
    box-shadow: 0 6px 18px rgba(37,99,235,.35) !important;
}
div[data-testid="stButton"] button[kind="secondary"] {
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    color: #475569 !important;
    font-weight: 500 !important;
    background: white !important;
}

/* ── File uploader ── */
div[data-testid="stFileUploader"] > section {
    border: 2px dashed #CBD5E1 !important;
    border-radius: 12px !important;
    background: #F8FAFC !important;
    transition: border-color 0.2s !important;
}
div[data-testid="stFileUploader"] > section:hover {
    border-color: #2563EB !important;
}

/* ── Images ── */
div[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid #E2E8F0;
}

/* ── Sliders ── */
div[data-baseweb="slider"] [role="slider"] { background: #2563EB !important; }
div[data-baseweb="slider"] div[data-testid="stThumbValue"] { color: #2563EB !important; }

/* ── Expander ── */
div[data-testid="stExpander"] {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
}

/* ── Divider ── */
hr { border-color: #E2E8F0 !important; }

/* ── Custom component helpers ── */
.section-label {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 0.6rem;
    display: block;
}
.badge {
    display: inline-block;
    padding: 0.15rem 0.65rem;
    border-radius: 999px;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-blue  { background:#EFF6FF; color:#2563EB; }
.badge-green { background:#F0FDF4; color:#15803D; }
.badge-amber { background:#FFFBEB; color:#B45309; }
.badge-red   { background:#FEF2F2; color:#DC2626; }

.det-card {
    background: #FAFBFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.55rem;
}
.det-card-title { font-weight: 600; font-size: 0.85rem; color: #0F172A; }
.det-card-sub   { font-size: 0.72rem; color: #94A3B8; margin-top: 0.1rem; }

.nav-item {
    display: flex; align-items: center; gap: 0.55rem;
    padding: 0.52rem 0.75rem;
    border-radius: 8px;
    font-size: 0.82rem; font-weight: 500;
    color: #64748B;
    margin-bottom: 0.2rem;
    cursor: default;
}
.nav-item.active {
    background: #EFF6FF;
    color: #2563EB;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ── Pipeline loader ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading DINOv2 ViT-S/14 …")
def _load_pipeline():
    from src.pipeline import PatternDetectionPipeline
    return PatternDetectionPipeline()

try:
    pipeline = _load_pipeline()
    _model_ok = True
except Exception as _e:
    _model_ok = False
    _model_err = str(_e)


# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="padding: 0 0 1.25rem 0;">
        <div style="font-weight:800;font-size:1.2rem;color:#0F172A;letter-spacing:-0.03em;">
            ⚡ PatternScan
        </div>
        <div style="font-size:0.7rem;color:#94A3B8;margin-top:0.2rem;">
            BOM Engineering Drawings
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Navigation
    st.markdown("""
    <div class="nav-item active">📐&nbsp; Detection</div>
    <div class="nav-item">📊&nbsp; History</div>
    <div class="nav-item">⚙&nbsp; Settings</div>
    """, unsafe_allow_html=True)

    st.divider()

    # Algorithm settings
    st.markdown('<span class="section-label">Algorithm</span>', unsafe_allow_html=True)
    manual_mode = st.toggle("Manual mode", value=False,
                            help="Override automatic threshold selection")

    if manual_mode:
        ncc_thr  = st.slider("NCC Threshold",    0.15, 0.9,  0.28, 0.01,
                             help="Lower → higher recall, more false positives")
        dino_thr = st.slider("DINOv2 Threshold", 0.50, 0.95, 0.84, 0.01,
                             help="Higher → fewer false positives")
        dilate   = st.slider("Stroke Dilation",  0,    9,    0,    1,
                             help="Thicken template strokes for bold drawings")
    else:
        ncc_thr, dino_thr, dilate = 0.28, 0.84, 0

    st.divider()

    # Model status
    if _model_ok:
        st.markdown('<span class="badge badge-green">● Model Ready</span>', unsafe_allow_html=True)
        st.caption("DINOv2 ViT-S/14 · NCC multi-scale")
    else:
        st.markdown('<span class="badge badge-red">● Error</span>', unsafe_allow_html=True)
        st.caption(_model_err[:80])

    st.divider()
    st.caption("Zero-shot · No training required")
    st.caption("NCC + DINOv2 + Containment NMS")


# ============================================================
#  MAIN CONTENT
# ============================================================

# Page header
st.markdown("""
<h1 style="font-size:1.65rem;font-weight:800;color:#0F172A;
           letter-spacing:-0.03em;margin-bottom:0.2rem;">
    Zero-Shot Pattern Detection
</h1>
<p style="color:#64748B;font-size:0.88rem;margin-bottom:1.75rem;">
    Locate component symbols in BOM engineering drawings &mdash;
    no training data required. Upload a pattern and drawing, then click&nbsp;<b>Run</b>.
</p>
""", unsafe_allow_html=True)


# ── Upload row ───────────────────────────────────────────────────────────────
up1, up2 = st.columns(2, gap="large")

with up1:
    st.markdown('<span class="section-label">Pattern Symbol</span>', unsafe_allow_html=True)
    pat_file = st.file_uploader(
        "pattern", type=["png", "jpg", "jpeg", "tif", "tiff"],
        label_visibility="collapsed", key="pat_up",
    )
    if pat_file:
        pil_p = Image.open(pat_file).convert("RGB")
        st.image(pil_p, caption=f"{pil_p.width} × {pil_p.height} px",
                 use_container_width=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#94A3B8;font-size:0.82rem;
                    background:#F8FAFC;border-radius:10px;border:2px dashed #CBD5E1;">
            📂&nbsp; PNG · JPG · TIFF
        </div>""", unsafe_allow_html=True)

with up2:
    st.markdown('<span class="section-label">Engineering Drawing</span>', unsafe_allow_html=True)
    drw_file = st.file_uploader(
        "drawing", type=["png", "jpg", "jpeg", "tif", "tiff"],
        label_visibility="collapsed", key="drw_up",
    )
    if drw_file:
        pil_d = Image.open(drw_file).convert("RGB")
        st.image(pil_d, caption=f"{pil_d.width} × {pil_d.height} px",
                 use_container_width=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#94A3B8;font-size:0.82rem;
                    background:#F8FAFC;border-radius:10px;border:2px dashed #CBD5E1;">
            📂&nbsp; PNG · JPG · TIFF
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Run button ───────────────────────────────────────────────────────────────
btn_col, _ = st.columns([1, 2])
with btn_col:
    run_clicked = st.button(
        "⚡  Run Detection",
        type="primary",
        use_container_width=True,
        disabled=not _model_ok,
    )


# ── Detection logic ──────────────────────────────────────────────────────────
if run_clicked:
    if pat_file is None or drw_file is None:
        st.warning("⚠ Upload both a **pattern image** and an **engineering drawing** first.")
    else:
        with st.spinner("Running detection pipeline …"):
            try:
                pat_arr = np.array(Image.open(pat_file).convert("RGB"))
                drw_arr = np.array(Image.open(drw_file).convert("RGB"))

                t0 = time.time()
                if manual_mode:
                    pipeline.update_thresholds(
                        ncc_threshold=ncc_thr,
                        cosine_threshold=dino_thr,
                    )
                    pipeline.dilate_pattern = dilate
                    result = pipeline.detect(pat_arr, drw_arr, return_visualization=True)
                else:
                    result = pipeline.detect_auto(pat_arr, drw_arr, return_visualization=True)
                elapsed = time.time() - t0

                vis   = result.pop("visualization", None)
                dets  = result.get("detections", [])
                n     = result["total_detections"]

                st.session_state.update({
                    "result": result, "vis": vis, "elapsed": elapsed,
                    "n": n, "dets": dets,
                })
            except Exception as ex:
                st.error(f"Detection failed: {ex}")
                st.session_state["result"] = None


# ── Results section ───────────────────────────────────────────────────────────
if st.session_state.get("result") is not None:
    result  = st.session_state["result"]
    vis     = st.session_state["vis"]
    elapsed = st.session_state["elapsed"]
    n       = st.session_state["n"]
    dets    = st.session_state["dets"]

    st.markdown("---")

    # ── Stat cards ──
    st.markdown('<span class="section-label">Detection Summary</span>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4, gap="medium")
    with s1:
        status_lbl = "Detected" if n > 0 else "Not found"
        st.metric("Instances Found", n)
    with s2:
        avg_c = round(sum(d["confidence"] for d in dets) / n, 2) if n else 0.0
        st.metric("Avg Confidence", f"{avg_c:.2f}")
    with s3:
        best_d = round(max(d["dino_score"] for d in dets), 4) if n else 0.0
        st.metric("Best DINO Score", f"{best_d:.4f}")
    with s4:
        st.metric("Processing Time", f"{elapsed:.1f} s")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Output image + detection list ──
    out1, out2 = st.columns([3, 1], gap="large")

    with out1:
        st.markdown('<span class="section-label">Annotated Output</span>', unsafe_allow_html=True)
        if vis is not None:
            st.image(vis, use_container_width=True,
                     caption=f"{n} instance(s) detected · {elapsed:.1f} s · "
                             f"NCC + DINOv2 ViT-S/14")
        else:
            st.info("No visualization available.")

    with out2:
        st.markdown('<span class="section-label">Detections</span>', unsafe_allow_html=True)

        if not dets:
            st.markdown("""
            <div style="text-align:center;padding:2.5rem 1rem;color:#94A3B8;
                        font-size:0.82rem;background:#F8FAFC;border-radius:10px;
                        border:1px dashed #CBD5E1;">
                No patterns detected
            </div>""", unsafe_allow_html=True)
        else:
            for i, det in enumerate(dets):
                bb   = det["bbox"]
                conf = float(det["confidence"])
                if conf >= 0.70:
                    ccolor, cbadge = "#15803D", "badge-green"
                elif conf >= 0.55:
                    ccolor, cbadge = "#B45309", "badge-amber"
                else:
                    ccolor, cbadge = "#DC2626", "badge-red"

                st.markdown(f"""
                <div class="det-card">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                        <div>
                            <div class="det-card-title">Detection &nbsp;#{i + 1}</div>
                            <div class="det-card-sub">
                                ({bb['x']}, {bb['y']}) &ensp; {bb['w']} × {bb['h']} px
                            </div>
                        </div>
                        <div style="font-weight:800;font-size:1.1rem;color:{ccolor};">
                            {conf:.2f}
                        </div>
                    </div>
                    <div style="margin-top:0.6rem;display:flex;gap:1rem;flex-wrap:wrap;">
                        <span style="font-size:0.7rem;color:#64748B;">
                            NCC &nbsp;<b>{det['ncc_score']:.3f}</b>
                        </span>
                        <span style="font-size:0.7rem;color:#64748B;">
                            DINO &nbsp;<b>{det['dino_score']:.4f}</b>
                        </span>
                        <span style="font-size:0.7rem;color:#64748B;">
                            Scale &nbsp;<b>{det['scale']:.2f}×</b>
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Raw JSON
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📄 Raw JSON output"):
            st.json(result)
