import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import time

st.set_page_config(
    page_title="UC3 - Keyword Mapping",
    
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background-color: #0d1117; color: #e6edf3; }

[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #21262d; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebar"] label {
    color: #8b949e !important; font-size: 0.75rem !important;
    font-weight: 500 !important; text-transform: uppercase !important; letter-spacing: 0.06em !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: #0d1117; border: 1px solid #30363d; color: #e6edf3;
}
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background-color: #0d1117; border: 1px solid #30363d; color: #e6edf3;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem;
}

h1 { font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 600 !important; color: #e6edf3 !important; letter-spacing: -0.02em !important; }
h2, h3 { font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 500 !important; color: #c9d1d9 !important; }

[data-testid="stFileUploader"] {
    background-color: #161b22; border: 1px dashed #30363d; border-radius: 6px; padding: 12px;
}
[data-testid="stFileUploader"] label {
    color: #8b949e !important; font-size: 0.75rem !important;
    text-transform: uppercase !important; letter-spacing: 0.06em !important;
}

.stButton > button {
    background-color: #9b59b6 !important; color: #fff !important;
    border: none !important; border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 500 !important;
    padding: 0.5rem 1.25rem !important;
}
.stButton > button:hover { background-color: #b36fd1 !important; }
.stDownloadButton > button {
    background-color: #238636 !important; color: #fff !important;
    border: none !important; border-radius: 6px !important;
}

[data-testid="stMetric"] {
    background-color: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.75rem !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 1.6rem !important; font-weight: 600 !important; }

[data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 6px; }

.stAlert { border-radius: 6px !important; border-left-width: 3px !important; }
[data-testid="stInfo"] { background-color: #1a0a2e !important; border-left-color: #9b59b6 !important; color: #cf9fff !important; }
[data-testid="stSuccess"] { background-color: #0a2217 !important; border-left-color: #238636 !important; color: #3fb950 !important; }
[data-testid="stWarning"] { background-color: #231c00 !important; border-left-color: #e3b341 !important; color: #e3b341 !important; }
[data-testid="stError"] { background-color: #280d11 !important; border-left-color: #da3633 !important; color: #f85149 !important; }

hr { border-color: #21262d; }

.streamlit-expanderHeader {
    background-color: #161b22 !important; border: 1px solid #21262d !important;
    border-radius: 6px !important; color: #c9d1d9 !important; font-size: 0.85rem !important;
}
.streamlit-expanderContent {
    background-color: #0d1117 !important; border: 1px solid #21262d !important; border-top: none !important;
}

code {
    font-family: 'IBM Plex Mono', monospace !important;
    background-color: #161b22 !important; color: #cf9fff !important;
    padding: 2px 6px !important; border-radius: 3px !important; font-size: 0.82em !important;
}

.section-bar {
    background: linear-gradient(90deg, #9b59b622, transparent);
    border-left: 3px solid #9b59b6;
    padding: 8px 16px; border-radius: 0 6px 6px 0;
    margin: 20px 0 12px 0;
    font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; color: #cf9fff;
}

.tier-strong { color: #3fb950; font-weight: 600; }
.tier-moderate { color: #e3b341; font-weight: 600; }
.tier-weak { color: #f85149; font-weight: 600; }

.cost-box {
    background: #1a0a2e; border: 1px solid #9b59b633;
    border-radius: 6px; padding: 10px 16px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; color: #cf9fff;
    margin: 8px 0;
}

.stProgress > div > div { background-color: #9b59b6 !important; }
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ────────────────────────────────────────────────────────

def norm_url(url):
    if not isinstance(url, str): return ''
    return re.sub(r'[?#].*$', '', url.strip().lower().rstrip('/'))

def load_file(f):
    if f is None: return None
    try:
        if f.name.lower().endswith(('.xlsx', '.xls')):
            return pd.read_excel(f)
        return pd.read_csv(f)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None

def detect_emb_col(df):
    for col in df.columns:
        if col.lower() in ['embedding', 'embeddings']:
            return col
    for col in df.columns:
        if col == 'Address': continue
        sample = df[col].dropna()
        if len(sample) > 0 and isinstance(sample.iloc[0], str) and sample.iloc[0].count(',') > 50:
            return col
    return None

def parse_vec(s):
    try:
        return np.array([float(x) for x in str(s).strip().strip('[]').split(',')], dtype=np.float32)
    except:
        return None

def section(title):
    st.markdown(f'<div class="section-bar">{title}</div>', unsafe_allow_html=True)


# ─── SIDEBAR ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Configuration")
    st.markdown("---")

    provider = st.selectbox("Embedding Provider", ["Gemini", "OpenAI"],
        help="Must match the provider used in your Screaming Frog crawl")
    api_key = st.text_input("API Key", type="password",
        help="Used only to vectorise your keywords. Never stored.")

    st.markdown("---")
    st.markdown("**Relevance Thresholds**")
    high_thresh = st.slider("Strong match (≥)", 0.60, 0.90, 0.75, 0.05)
    low_thresh  = st.slider("Weak match (<)", 0.30, 0.70, 0.55, 0.05)

    st.markdown("---")
    st.markdown("**Column Names in Your Keyword CSV**")
    kw_col = st.text_input("Keyword column", value="Keyword")
    url_col = st.text_input("Landing URL column", value="Landing URL")

    batch_size = st.slider("API batch size", 5, 50, 20, 5,
        help="Reduce to 10 if you hit rate limit errors")

    st.markdown("---")
    if api_key:
        n_est = st.session_state.get('kw_count', 100)
        cost_oai = n_est * 0.00002
        cost_gem = n_est * 0.000025
        st.markdown(f"""<div class="cost-box">
Est. cost for {n_est} keywords<br>
OpenAI  : ${cost_oai:.4f}<br>
Gemini  : ${cost_gem:.4f}
</div>""", unsafe_allow_html=True)


# ─── MAIN ───────────────────────────────────────────────────────────

st.markdown("# Keyword Mapping & Relevance Scoring")
st.markdown("---")

section("01 - Upload Files")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**SF Embeddings Export** `required`")
    st.caption("Screaming Frog → Bulk Export → AI Tab → Export")
    emb_file = st.file_uploader("Drop embeddings file", type=["csv", "xlsx", "xls"], key="emb", label_visibility="collapsed")

with col2:
    st.markdown("**Keyword CSV** `required`")
    st.caption(f"Must have `{kw_col}` and `{url_col}` columns (or update names in sidebar)")
    kw_file = st.file_uploader("Drop keyword file", type=["csv", "xlsx", "xls"], key="kw", label_visibility="collapsed")


emb_df = load_file(emb_file)
kw_df  = load_file(kw_file)

col_s1, col_s2 = st.columns(2)
with col_s1:
    if emb_df is not None:
        st.success(f"✓ Embeddings - {len(emb_df):,} pages")
    else:
        st.error("✗ Embeddings - required")
with col_s2:
    if kw_df is not None:
        st.session_state['kw_count'] = len(kw_df)
        st.success(f"✓ Keywords - {len(kw_df):,} rows")
    else:
        st.error("✗ Keywords - required")

emb_col = None
if emb_df is not None:
    if 'Address' not in emb_df.columns:
        st.error("No 'Address' column in embeddings file.")
    else:
        emb_col = detect_emb_col(emb_df)
        if emb_col is None:
            st.error("Could not detect the embedding vector column.")
        else:
            emb_df['_url_norm'] = emb_df['Address'].apply(norm_url)

if kw_df is not None and kw_col not in kw_df.columns:
    st.error(f"Column '{kw_col}' not found in keyword file. Found: {list(kw_df.columns)}")
    kw_df = None

if kw_df is not None and url_col not in kw_df.columns:
    st.warning(f"Column '{url_col}' not found - mismatch detection will be skipped.")


# ── RUN ──────────────────────────────────────────────────────────────
section("02 - Vectorise Keywords & Score")

ready = (emb_df is not None and emb_col is not None and kw_df is not None and api_key)
if not api_key:
    st.info("Add your API key in the sidebar to enable keyword vectorisation.")

run_btn = st.button("▶  Run Analysis", disabled=not ready)
resume_btn = st.button("⏩  Resume from checkpoint", disabled=not (ready and "kw_cache" in st.session_state and len(st.session_state.get("kw_cache", {})) > 0))

if "kw_cache" in st.session_state and st.session_state.kw_cache:
    cached_count = len(st.session_state.kw_cache)
    st.caption(f"Checkpoint: {cached_count} keywords already vectorised. Hit Resume to continue.")

if run_btn:
    st.session_state.kw_cache = {}  # fresh run clears cache

if run_btn or resume_btn:
    keywords = kw_df[kw_col].dropna().tolist()
    n = len(keywords)

    if "kw_cache" not in st.session_state:
        st.session_state.kw_cache = {}

    cache = st.session_state.kw_cache

    # ── Vectorise keywords
    prog = st.progress(len(cache) / n if n > 0 else 0)
    status = st.empty()

    try:
        if provider == "OpenAI":
            import openai
            client = openai.OpenAI(api_key=api_key)
            for i in range(0, n, batch_size):
                batch_kws = keywords[i:i+batch_size]
                # skip already cached
                to_fetch = [kw for kw in batch_kws if kw not in cache]
                if not to_fetch:
                    prog.progress(min((i + batch_size) / n, 1.0))
                    continue
                status.caption(f"Vectorising keywords {min(i+batch_size,n)}/{n}...")
                resp = client.embeddings.create(input=to_fetch, model="text-embedding-3-small")
                for kw, item in zip(to_fetch, resp.data):
                    cache[kw] = np.array(item.embedding, dtype=np.float32)
                    st.session_state.kw_cache = cache
                prog.progress(min((i + batch_size) / n, 1.0))
                time.sleep(0.05)

        else:  # Gemini
            import requests, re
            headers = {"Content-Type": "application/json"}
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={api_key}"
            for i, kw in enumerate(keywords):
                if kw in cache:
                    prog.progress((i + 1) / n)
                    continue
                status.caption(f"Vectorising keywords {i+1}/{n}...")
                body = {
                    "model": "models/gemini-embedding-001",
                    "content": {"parts": [{"text": kw}]},
                    "taskType": "SEMANTIC_SIMILARITY"
                }
                for attempt in range(5):
                    resp = requests.post(url, json=body, headers=headers)
                    if resp.status_code == 200:
                        break
                    elif resp.status_code == 429:
                        retry_match = re.search(r'retry in (\d+)', resp.text)
                        wait = int(retry_match.group(1)) + 2 if retry_match else 60
                        status.caption(f"Rate limit hit - waiting {wait}s... ({i+1}/{n})")
                        time.sleep(wait)
                    else:
                        st.error(f"API error: {resp.status_code} {resp.text}")
                        st.stop()
                else:
                    st.error("Failed after 5 retries. Hit Resume to continue from this point.")
                    st.stop()
                cache[kw] = np.array(resp.json()["embedding"]["values"], dtype=np.float32)
                st.session_state.kw_cache = cache
                prog.progress((i + 1) / n)
                time.sleep(0.65)  # ~90 req/min to stay under free tier limit

    except Exception as e:
        st.error(f"API error: {e}")
        st.stop()

    prog.empty()
    status.empty()

    # Build ordered embeddings list from cache
    kw_embeddings = [cache[kw] for kw in keywords if kw in cache]
    keywords = [kw for kw in keywords if kw in cache]
    n = len(keywords)

    if len(kw_embeddings) != n:
        st.error(f"Embedding count mismatch: got {len(kw_embeddings)}, expected {n}")
        st.stop()

    kw_df = kw_df.copy()
    kw_df['_embedding'] = kw_embeddings[:len(kw_df)]

    # ── Parse page vectors
    with st.spinner("Parsing page vectors..."):
        emb_df['_vec'] = emb_df[emb_col].apply(parse_vec)
        emb_df = emb_df[emb_df['_vec'].notna()].reset_index(drop=True)

        page_urls   = emb_df['Address'].tolist()
        page_norms_ = emb_df['_url_norm'].tolist()
        page_vecs   = np.stack(emb_df['_vec'].values)
        pv = np.linalg.norm(page_vecs, axis=1, keepdims=True)
        pv[pv == 0] = 1.0
        page_unit = page_vecs / pv

        kw_vecs = np.stack(kw_df['_embedding'].values)
        kv = np.linalg.norm(kw_vecs, axis=1, keepdims=True)
        kv[kv == 0] = 1.0
        kw_unit = kw_vecs / kv

        sim_matrix = kw_unit @ page_unit.T

    with st.spinner("Scoring keyword-page relevance..."):
        results = []
        has_url_col = url_col in kw_df.columns

        for i, row in kw_df.iterrows():
            kw = row[kw_col]
            cur_url  = row[url_col] if has_url_col else ''
            cur_norm = norm_url(str(cur_url)) if has_url_col else ''

            scores   = sim_matrix[i]
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            best_url   = page_urls[best_idx]
            best_norm  = page_norms_[best_idx]

            cur_relevance = None
            if cur_norm and cur_norm in page_norms_:
                cur_idx = page_norms_.index(cur_norm)
                cur_relevance = float(scores[cur_idx])

            mismatch = (cur_norm != best_norm and cur_norm != '') if cur_norm else None

            if cur_relevance is None:
                tier = 'Unknown'
            elif cur_relevance >= high_thresh:
                tier = '🟢 Strong'
            elif cur_relevance >= low_thresh:
                tier = '🟡 Moderate'
            else:
                tier = '🔴 Weak'

            results.append({
                'Keyword': kw,
                'Current Landing URL': cur_url if has_url_col else 'N/A',
                'Best Matched Page': best_url,
                'Best Match Score': round(best_score, 4),
                'Current Page Relevance': round(cur_relevance, 4) if cur_relevance is not None else None,
                'Relevance Tier': tier,
                'Mismatch': mismatch,
            })

        out_df = pd.DataFrame(results)
        st.session_state['uc3_results'] = out_df


# ── RESULTS ──────────────────────────────────────────────────────────
if 'uc3_results' in st.session_state:
    out_df = st.session_state['uc3_results']
    section("03 - Results")

    strong   = (out_df['Relevance Tier'] == '🟢 Strong').sum()
    moderate = (out_df['Relevance Tier'] == '🟡 Moderate').sum()
    weak     = (out_df['Relevance Tier'] == '🔴 Weak').sum()
    mismatches = out_df['Mismatch'].sum() if out_df['Mismatch'].dtype == bool else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🟢 Strong", f"{strong:,}")
    m2.metric("🟡 Moderate", f"{moderate:,}")
    m3.metric("🔴 Weak", f"{weak:,}")
    m4.metric("⚡ Mismatches", f"{mismatches:,}")

    st.markdown("")

    tab1, tab2, tab3 = st.tabs(["All Keywords", "🔴 Weak - Fix First", "⚡ Mismatches"])

    with tab1:
        st.dataframe(
            out_df.sort_values('Current Page Relevance', ascending=True),
            use_container_width=True, hide_index=True,
            column_config={
                'Best Match Score': st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
                'Current Page Relevance': st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
            }
        )

    with tab2:
        weak_df = out_df[out_df['Relevance Tier'] == '🔴 Weak'].sort_values('Current Page Relevance')
        if len(weak_df):
            st.dataframe(weak_df, use_container_width=True, hide_index=True,
                column_config={'Best Match Score': st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
                               'Current Page Relevance': st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f")})
        else:
            st.success("No weak matches found.")

    with tab3:
        mis_df = out_df[out_df['Mismatch'] == True] if 'Mismatch' in out_df.columns else pd.DataFrame()
        if len(mis_df):
            st.caption("The page currently ranking for these keywords is NOT the best semantic match on your site.")
            st.dataframe(mis_df, use_container_width=True, hide_index=True)
        else:
            st.success("No mismatches detected.")

    section("04 - Export")
    csv = out_df.to_csv(index=False).encode()
    st.download_button(
        "⬇  Download keyword_relevance_scores.csv",
        data=csv, file_name="keyword_relevance_scores.csv", mime="text/csv",
    )
    with st.expander("How to action this data"):
        st.markdown("""
- **🔴 Weak + Mismatch = True** → wrong page ranking. Create a dedicated page or consolidate content.
- **🔴 Weak + Mismatch = False** → right page, but needs significant content depth added.
- **🟡 Moderate** → review content. Consider expanding coverage or tightening focus.
- **🟢 Strong + poor rankings** → content is fine. Investigate links, CWV, or technical issues.
""")
