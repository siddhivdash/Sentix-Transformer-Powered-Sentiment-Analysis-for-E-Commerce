import streamlit as st
import pandas as pd
import os

# --- 1. PAGE SETTINGS ---
st.set_page_config(page_title="Sentix: Transformer-Powered Sentiment Analysis for E-Commerce", page_icon="🎯", layout="wide")

# --- 2. IMPROVED UI CSS ---
st.markdown("""
    <style>
    @import url('https://api.fontshare.com/v2/css?f[]=satoshi@400,500,700,900&f[]=cabinet-grotesk@400,500,700,800&display=swap');

    /* ── ROOT TOKENS ── */
    :root {
        --bg:          #0F0F0F;
        --surface:     #1A1A1A;
        --surface-2:   #242424;
        --surface-3:   #2E2E2E;
        --border:      rgba(255,255,255,0.10);
        --border-bold: rgba(255,255,255,0.22);
        --text:        #F0EEE9;
        --text-muted:  #9A9893;
        --accent:      #E8622A;
        --accent-dim:  rgba(232,98,42,0.15);
        --accent-glow: rgba(232,98,42,0.35);
        --success:     #3DAA72;
        --success-dim: rgba(61,170,114,0.15);
        --danger:      #E84040;
        --danger-dim:  rgba(232,64,64,0.15);
        --radius:      10px;
        --radius-lg:   16px;
    }

    /* ── BASE ── */
    .stApp, .stApp > div { background-color: var(--bg) !important; color: var(--text) !important; }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 2.5rem 2.5rem 3rem !important; max-width: 900px !important; }

    /* ── TYPOGRAPHY ── */
    h1, h2, h3, h4, p, span, label, div,
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Satoshi', 'Inter', sans-serif !important;
        color: var(--text) !important;
    }
    h1 { font-size: 2rem !important; font-weight: 800 !important; letter-spacing: -0.5px !important; }
    h2 { font-size: 1.4rem !important; font-weight: 700 !important; letter-spacing: -0.3px !important; }
    h3 { font-size: 1.1rem !important; font-weight: 600 !important; }
    p, span, label { font-size: 0.95rem !important; font-weight: 400 !important; line-height: 1.6 !important; }

    /* ── APP HEADER BAND ── */
    .brand-header {
        background: linear-gradient(135deg, var(--surface) 0%, var(--surface-2) 100%);
        border: 1px solid var(--border-bold);
        border-radius: var(--radius-lg);
        padding: 28px 32px;
        margin-bottom: 28px;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .brand-icon { font-size: 2.4rem; line-height: 1; }
    .brand-title {
        font-family: 'Cabinet Grotesk', 'Satoshi', sans-serif !important;
        font-size: 1.9rem !important;
        font-weight: 800 !important;
        color: var(--text) !important;
        letter-spacing: -0.6px;
        margin: 0 !important;
    }
    .brand-sub {
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 2px !important;
    }
    .accent-dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: var(--accent);
        box-shadow: 0 0 10px var(--accent-glow);
        display: inline-block;
        margin-right: 6px;
    }

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface) !important;
        border-radius: var(--radius) !important;
        padding: 4px !important;
        gap: 4px !important;
        border: 1px solid var(--border) !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 7px !important;
        color: var(--text-muted) !important;
        font-family: 'Satoshi', sans-serif !important;
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
        border: none !important;
        transition: all 0.18s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent) !important;
        color: #FFFFFF !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }

    /* ── SECTION LABEL ── */
    .section-label {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        color: var(--text-muted) !important;
        margin-bottom: 10px !important;
        margin-top: 4px !important;
    }

    /* ── TEXTAREA ── */
    .stTextArea label {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: var(--text) !important;
        margin-bottom: 6px !important;
    }
    .stTextArea textarea {
        background: var(--surface) !important;
        border: 1.5px solid var(--border-bold) !important;
        border-radius: var(--radius) !important;
        color: var(--text) !important;
        font-family: 'Satoshi', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 400 !important;
        padding: 14px 16px !important;
        transition: border-color 0.18s ease !important;
        caret-color: var(--accent) !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-dim) !important;
        outline: none !important;
    }
    .stTextArea textarea::placeholder { color: var(--text-muted) !important; opacity: 0.7 !important; }

    /* ── PRIMARY BUTTON ── */
    .stButton > button {
        background: var(--accent) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: var(--radius) !important;
        padding: 10px 28px !important;
        font-family: 'Satoshi', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.2px !important;
        height: auto !important;
        transition: background 0.18s ease, box-shadow 0.18s ease, transform 0.12s ease !important;
        box-shadow: 0 4px 16px var(--accent-glow) !important;
    }
    .stButton > button:hover {
        background: #D4521E !important;
        box-shadow: 0 6px 24px var(--accent-glow) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active { transform: translateY(0px) !important; }

    /* ── RESULT CARDS ── */
    .result-card {
        padding: 32px 28px;
        border-radius: var(--radius-lg);
        text-align: center;
        margin: 18px 0;
    }
    .pos-card {
        background: var(--success-dim);
        border: 1.5px solid rgba(61,170,114,0.40);
    }
    .neg-card {
        background: var(--danger-dim);
        border: 1.5px solid rgba(232,64,64,0.40);
    }
    .result-card .verdict {
        font-family: 'Cabinet Grotesk', 'Satoshi', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.8px;
        margin-bottom: 6px;
    }
    .result-card .confidence {
        font-size: 0.85rem;
        font-weight: 500;
        opacity: 0.75;
        text-transform: uppercase;
        letter-spacing: 1.4px;
    }
    .pos-card .verdict { color: #4EC989; }
    .pos-card .confidence { color: #4EC989; }
    .neg-card .verdict { color: #F06060; }
    .neg-card .confidence { color: #F06060; }

    /* ── FILE UPLOADER ── */
    [data-testid="stFileUploader"] {
        background: var(--surface) !important;
        border: 1.5px dashed var(--border-bold) !important;
        border-radius: var(--radius-lg) !important;
        padding: 20px !important;
        color: var(--text) !important;
    }
    [data-testid="stFileUploader"] label {
        color: var(--text) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] small { color: var(--text-muted) !important; }

    /* ── SELECTBOX ── */
    [data-testid="stSelectbox"] label {
        color: var(--text) !important;
        font-size: 0.875rem !important;
        font-weight: 600 !important;
    }
    .stSelectbox > div > div {
        background: var(--surface) !important;
        border: 1.5px solid var(--border-bold) !important;
        border-radius: var(--radius) !important;
        color: var(--text) !important;
    }

    /* ── ALERT / WARNING / SUCCESS ── */
    .stAlert {
        border-radius: var(--radius) !important;
        border-left: 4px solid !important;
        font-family: 'Satoshi', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    .stAlert[data-baseweb="notification"] { color: var(--text) !important; }

    /* ── METRICS / DATA ── */
    [data-testid="metric-container"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 16px 20px !important;
    }
    [data-testid="metric-container"] label { color: var(--text-muted) !important; }
    [data-testid="metric-container"] [data-testid="metric-value"] { color: var(--text) !important; }

    /* ── DIVIDER ── */
    hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 24px 0 !important; }

    /* ── SPINNER ── */
    .stSpinner { color: var(--accent) !important; }
    .stSpinner > div { border-top-color: var(--accent) !important; }

    /* ── DATAFRAME / TABLE ── */
    .stDataFrame { background: var(--surface) !important; border-radius: var(--radius) !important; }

    /* ── DOWNLOAD BUTTON ── */
    [data-testid="stDownloadButton"] button {
        background: var(--surface-2) !important;
        color: var(--text) !important;
        border: 1.5px solid var(--border-bold) !important;
        border-radius: var(--radius) !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        box-shadow: none !important;
        transition: background 0.15s ease, border-color 0.15s ease !important;
    }
    [data-testid="stDownloadButton"] button:hover {
        background: var(--surface-3) !important;
        border-color: var(--accent) !important;
        transform: none !important;
    }

    /* ── FOOTER ── */
    .app-footer {
        text-align: center;
        padding: 20px 0 0;
        font-size: 0.75rem !important;
        color: var(--text-muted) !important;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-weight: 500 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOAD THE TRANSFORMER ENGINE ---
@st.cache_resource
def load_nlp():
    # Import inside the function to speed up initial app boot
    from transformers import pipeline
    
    # We load directly from Hugging Face for deployment stability
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    
    try:
        # This will download the model to the Streamlit server on first run
        return pipeline("sentiment-analysis", model=model_name)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

nlp_engine = load_nlp()

# --- 4. BRAND HEADER ---
st.markdown("""
    <div class="brand-header">
        <div class="brand-icon">🎯</div>
        <div>
            <div class="brand-title">Sentix: Transformer-Powered Sentiment Analysis for E-Commerce</div>
            <div class="brand-sub"><span class="accent-dot"></span>High-Contrast Contextual Analysis Engine</div>
        </div>
    </div>
""", unsafe_allow_html=True)

if not nlp_engine:
    st.error("⚠️ ENGINE OFFLINE — The model failed to initialize from the Hugging Face Hub. Check your internet connection or deployment logs.")
    st.stop()

# --- 5. TABS ---
tab1, tab2 = st.tabs(["🔍  Single Analysis", "📁  Batch Processing"])

# ── TAB 1: SINGLE ──
with tab1:
    st.markdown('<div class="section-label">📝 &nbsp; Enter Text for Analysis</div>', unsafe_allow_html=True)
    user_text = st.text_area(
        "Review Content",
        placeholder="Paste a review here… sarcasm and nuance are supported.",
        height=180,
        label_visibility="visible"
    )

    if st.button("Run AI Analysis ✨"):
        if user_text.strip():
            with st.spinner("Analyzing…"):
                result = nlp_engine(user_text)[0]
                label  = result['label']
                score  = result['score'] * 100

                if label == "POSITIVE":
                    st.markdown(f"""
                        <div class="result-card pos-card">
                            <div class="verdict">✓ &nbsp; Positive</div>
                            <div class="confidence">Confidence &nbsp;·&nbsp; {score:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="result-card neg-card">
                            <div class="verdict">✕ &nbsp; Negative</div>
                            <div class="confidence">Confidence &nbsp;·&nbsp; {score:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("Please enter some text before running analysis.")

# ── TAB 2: BATCH ──
with tab2:
    st.markdown('<div class="section-label">📁 &nbsp; Upload Dataset (CSV)</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Try standard UTF-8 first
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
        # If UTF-8 fails, try latin-1, which handles a wider range of characters
            uploaded_file.seek(0) # Reset file pointer to the beginning
            df = pd.read_csv(uploaded_file, encoding='latin-1')
        st.markdown('<div class="section-label">🗂 &nbsp; Select Column to Analyze</div>', unsafe_allow_html=True)
        text_column = st.selectbox("Text Column", df.columns, label_visibility="visible")

        if st.button("🚀 Execute Batch Analysis"):
            with st.spinner("Processing rows…"):
                def get_sentiment(text):
                    if not isinstance(text, str) or len(text.strip()) == 0:
                        return "NEUTRAL"
                    res = nlp_engine(text[:512])[0]
                    return res['label']

                df['AI_SENTIMENT'] = df[text_column].apply(get_sentiment)
                st.success("✅ Analysis complete!")

                counts = df['AI_SENTIMENT'].value_counts()
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="section-label">📊 &nbsp; Distribution</div>', unsafe_allow_html=True)
                    st.dataframe(counts, use_container_width=True)
                with c2:
                    st.bar_chart(counts)

                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Tagged Report",
                    data=csv_data,
                    file_name="sentiment_analysis.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# --- FOOTER ---
st.markdown('<div class="app-footer">Sentix Engineering &nbsp;·&nbsp; v2.0 &nbsp;·&nbsp; 2026</div>', unsafe_allow_html=True)
