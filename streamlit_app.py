from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz

ROOT             = Path(__file__).resolve().parent
CSV_PATH         = ROOT / "songs_clustered.csv"
BEST_MODEL_PATH  = ROOT / "best_model.pkl"
EMBED_MODEL_PATH = ROOT / "embedding_model.pkl"

def _safe_text(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)

def _normalize(s: str) -> str:
    return _safe_text(s).lower().strip()

def _clean_query(q: str) -> str:
    return " ".join(_normalize(q).split())

def _fmt_int(x: Any) -> str:
    try:
        n = float(x)
        if not np.isfinite(n):
            return "—"
        v = int(n)
        if v >= 1_000_000:
            return f"{v/1_000_000:.1f}M"
        if v >= 1_000:
            return f"{v/1_000:.1f}K"
        return f"{v:,}"
    except Exception:
        return "—"

def _load_sentence_transformer_fallback():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_models() -> tuple[Any, Any]:
    clf = joblib.load(BEST_MODEL_PATH)
    try:
        embed = joblib.load(EMBED_MODEL_PATH)
    except Exception:
        embed = _load_sentence_transformer_fallback()
    if not hasattr(embed, "encode"):
        embed = _load_sentence_transformer_fallback()
    return clf, embed

@st.cache_data(show_spinner=False)
def load_df() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    for col in ["song", "artist", "youtube_link", "text", "better_text", "cleaned_text", "cluster", "label"]:
        if col not in df.columns:
            df[col] = ""

    for col in ["views", "likes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    df["_song_n"]    = df["song"].map(_normalize)
    df["_artist_n"]  = df["artist"].map(_normalize)
    df["_label_n"]   = df["label"].map(_normalize)
    df["_cluster_n"] = df["cluster"].map(_normalize)

    df["_search"] = (
        df["_song_n"] + " " + df["_artist_n"] + " " +
        df["_label_n"] + " " + df["_cluster_n"] + " " +
        df["cleaned_text"].map(_normalize)
    )

    nlp = (
        df["song"].map(_safe_text)
        + " — " + df["artist"].map(_safe_text)
        + " | label: " + df["label"].map(_safe_text)
        + " | cluster: " + df["cluster"].map(_safe_text)
        + " | " + df["better_text"].map(_safe_text)
    )
    df["_nlp"] = nlp.str.slice(0, 512)

    if "id" not in df.columns:
        df["id"] = df.index.astype(str)
    else:
        df["id"] = df["id"].map(_safe_text)

    return df

def _score_row(qn: str, song: str, artist: str, label: str, cluster: str,
               search_fields: list[str]) -> float:
    scores = []
    if "song" in search_fields and song:
        scores.append(max(fuzz.token_set_ratio(qn, song), fuzz.partial_ratio(qn, song)) * 1.40)
    if "artist" in search_fields and artist:
        scores.append(max(fuzz.token_set_ratio(qn, artist), fuzz.partial_ratio(qn, artist)) * 1.25)
    if "label" in search_fields and label:
        scores.append(max(fuzz.token_set_ratio(qn, label), fuzz.partial_ratio(qn, label)) * 1.10)
    if "cluster" in search_fields and cluster:
        scores.append(max(fuzz.token_set_ratio(qn, cluster), fuzz.partial_ratio(qn, cluster)) * 1.00)
    combined = f"{song} {artist} {label} {cluster}"
    scores.append(fuzz.token_set_ratio(qn, combined) * 0.80)
    return max(scores) if scores else 0.0


def fuzzy_search(df: pd.DataFrame, q: str, search_fields: list[str], threshold: int = 40) -> pd.DataFrame:
    qn = _clean_query(q)
    if not qn:
        return df

    tokens = [t for t in qn.split() if len(t) >= 2]
    field_col_map = {
        "song": "_song_n", "artist": "_artist_n",
        "label": "_label_n", "cluster": "_cluster_n",
    }

    if tokens:
        mask = pd.Series(False, index=df.index)
        for field in search_fields:
            col = field_col_map.get(field, "_search")
            for t in tokens[:6]:
                mask |= df[col].str.contains(t, na=False, regex=False)
        for t in tokens[:6]:
            mask |= df["_search"].str.contains(t, na=False, regex=False)
        cand = df[mask]
        if len(cand) < 50:
            cand = df
    else:
        cand = df

    if len(cand) > 12_000:
        cand = cand.sort_values("views", ascending=False).head(12_000)

    songs    = cand["_song_n"].tolist()
    artists  = cand["_artist_n"].tolist()
    labels   = cand["_label_n"].tolist()
    clusters = cand["_cluster_n"].tolist()

    scores = [
        _score_row(qn, s, a, l, c, search_fields)
        for s, a, l, c in zip(songs, artists, labels, clusters)
    ]

    cand = cand.copy()
    cand["_score"] = scores
    cand = cand[cand["_score"] >= threshold].sort_values(["_score", "views"], ascending=[False, False])
    return cand.drop(columns=["_score"], errors="ignore")


@st.cache_resource(show_spinner=False)
def build_nlp_index() -> tuple[Any, np.ndarray]:
    _clf, embed = load_models()
    df    = load_df()
    texts = df["_nlp"].fillna("").astype(str).tolist()
    emb   = embed.encode(texts, batch_size=96, show_progress_bar=False)
    vecs  = np.asarray(emb, dtype=np.float32)
    vecs  = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    return embed, vecs


def nlp_search(df: pd.DataFrame, q: str, search_fields: list[str], top_k: int) -> pd.DataFrame:
    qn = _clean_query(q)
    if not qn:
        return df

    embed, vecs = build_nlp_index()
    full_df     = load_df()

    qv   = np.asarray(embed.encode([qn], batch_size=1, show_progress_bar=False), dtype=np.float32)
    qv   = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)
    sims = (vecs @ qv[0]).astype(np.float32)

    if len(df) == len(full_df):
        k   = min(max(300, int(top_k)), len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        picked = df.iloc[top].copy()
        picked["_sim"] = sims[top]
    else:
        rows     = df.index.to_numpy(dtype=np.int64, copy=False)
        sub_sims = sims[rows]
        k        = min(max(300, int(top_k)), len(rows))
        top      = np.argpartition(-sub_sims, k - 1)[:k]
        top      = top[np.argsort(-sub_sims[top])]
        picked   = df.iloc[top].copy()
        picked["_sim"] = sub_sims[top]

    q_short  = qn[:80]
    songs    = picked["_song_n"].tolist()
    artists  = picked["_artist_n"].tolist()
    labels   = picked["_label_n"].tolist()
    clusters = picked["_cluster_n"].tolist()

    boosts = [
        _score_row(q_short, s, a, l, c, search_fields)
        for s, a, l, c in zip(songs, artists, labels, clusters)
    ]
    picked["_boost"] = boosts
    picked["_rank"]  = picked["_sim"] * 60 + picked["_boost"] * 0.40
    picked = picked.sort_values("_rank", ascending=False)
    return picked.drop(columns=["_sim", "_boost", "_rank"], errors="ignore")


def run_search(df: pd.DataFrame, q: str, search_fields: list[str],
               mode: str, limit: int) -> pd.DataFrame:
    if not q.strip():
        return df
    if "NLP" in mode:
        try:
            return nlp_search(df, q, search_fields, top_k=max(600, limit * 20))
        except Exception:
            return fuzzy_search(df, q, search_fields)
    return fuzzy_search(df, q, search_fields)

MUSIC_SVG = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M9 18V5L21 3V16" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="6" cy="18" r="3" stroke="white" stroke-width="2.2"/>
  <circle cx="18" cy="16" r="3" stroke="white" stroke-width="2.2"/>
</svg>"""

YT_SVG = """<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
  <path d="M23.5 6.2a3 3 0 0 0-2.1-2.1C19.5 3.6 12 3.6 12 3.6s-7.5 0-9.4.5A3 3 0 0 0 .5 6.2 31.2 31.2 0 0 0 0 12a31.2 31.2 0 0 0 .5 5.8 3 3 0 0 0 2.1 2.1c1.9.5 9.4.5 9.4.5s7.5 0 9.4-.5a3 3 0 0 0 2.1-2.1A31.2 31.2 0 0 0 24 12a31.2 31.2 0 0 0-.5-5.8zM9.6 15.6V8.4l6.3 3.6-6.3 3.6z"/>
</svg>"""

def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
  --bg:     #07090d;
  --card:   #0d1420;
  --text:   #e2e8f4;
  --muted:  #8899b4;
  --line:   rgba(136,153,180,0.13);
  --green:  #1DB954;
  --green2: #22c55e;
  --blue:   #38bdf8;
  --shadow: 0 24px 64px rgba(0,0,0,.7);
  --r:      16px;
}

html, body, .stApp { font-family: 'Inter', sans-serif !important; }
.stApp {
  background:
    radial-gradient(ellipse 1400px 700px at 8% 0%,   rgba(29,185,84,.15), transparent 55%),
    radial-gradient(ellipse 900px  500px at 92% 10%,  rgba(56,189,248,.11), transparent 60%),
    radial-gradient(ellipse 600px  400px at 50% 95%,  rgba(167,139,250,.08), transparent 55%),
    var(--bg);
  color: var(--text);
}
[data-testid="stHeader"]  { background: transparent !important; }
.block-container { padding-top: .6rem !important; padding-bottom: 2.5rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(10,16,26,.97), rgba(7,10,16,.95)) !important;
  border-right: 1px solid var(--line) !important;
}
[data-testid="stSidebar"] section { padding-top: 0 !important; }

/* ── Logo ── */
.brand {
  display:flex; align-items:center; gap:12px; padding:6px 0 16px;
  animation: slideDown .5s ease both;
}
.logo-ring {
  width:46px; height:46px; border-radius:14px; flex-shrink:0;
  background: linear-gradient(135deg, #1DB954, #169a42);
  box-shadow: 0 0 0 1px rgba(29,185,84,.45), 0 8px 32px rgba(29,185,84,.28);
  display:flex; align-items:center; justify-content:center;
  position:relative; overflow:hidden;
  animation: logoPulse 3.2s ease-in-out infinite;
}
.logo-ring::after {
  content:""; position:absolute; inset:-50%;
  background: linear-gradient(90deg, transparent 30%, rgba(255,255,255,.28) 50%, transparent 70%);
  animation: logoShine 3.6s ease-in-out infinite;
}
@keyframes logoPulse {
  0%,100% { box-shadow: 0 0 0 1px rgba(29,185,84,.45), 0 8px 32px rgba(29,185,84,.28); }
  50%     { box-shadow: 0 0 0 4px rgba(29,185,84,.22), 0 8px 40px rgba(29,185,84,.48); }
}
@keyframes logoShine {
  0%       { transform: translateX(-120%) rotate(18deg); opacity:0; }
  25%,45%  { opacity:.65; }
  65%,100% { transform: translateX(120%) rotate(18deg); opacity:0; }
}
.brand-name {
  font-size:20px; font-weight:900; margin:0; letter-spacing:-.04em;
  background: linear-gradient(90deg, #fff 45%, #8899b4 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.brand-sub { font-size:11px; color:var(--muted); margin-top:2px; }

/* ── Sidebar stats ── */
.stat-row { display:flex; gap:7px; margin-bottom:4px; }
.stat-box {
  flex:1; padding:10px 6px; border-radius:12px; text-align:center;
  background:rgba(255,255,255,.03); border:1px solid var(--line);
  animation: slideDown .55s ease both;
}
.stat-box .sv { font-size:14px; font-weight:800; color:var(--green2); }
.stat-box .sl { font-size:10px; color:var(--muted); margin-top:1px; }

/* ── Nav radio ── */
[data-testid="stSidebar"] .stRadio > div { gap:6px !important; }
[data-testid="stSidebar"] .stRadio label {
  border:1px solid rgba(136,153,180,.12); border-radius:12px;
  padding:10px 14px; background:rgba(255,255,255,.02);
  transition:all .18s ease; cursor:pointer;
}
[data-testid="stSidebar"] .stRadio label:hover {
  border-color:rgba(29,185,84,.38); background:rgba(29,185,84,.06);
  transform:translateX(3px);
}

/* ── Divider ── */
hr { border-color:var(--line) !important; margin:12px 0 !important; }

/* ── Search Settings Panel ── */
.settings-header {
  font-size:11px; font-weight:700; color:var(--muted);
  letter-spacing:.09em; text-transform:uppercase; margin-bottom:10px;
  display:flex; align-items:center; gap:7px;
}
.settings-header::before {
  content:""; display:block; width:3px; height:14px;
  border-radius:3px; background:var(--green);
  animation: barPulse 2s ease-in-out infinite;
}
@keyframes barPulse {
  0%,100% { opacity:.6; } 50% { opacity:1; }
}

/* ── Pill ── */
.pill {
  display:inline-flex; align-items:center; gap:8px;
  padding:6px 13px; border-radius:999px; font-size:12px;
  background:rgba(255,255,255,.03); border:1px solid var(--line);
  color:var(--muted);
}
.dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; background:var(--muted); }
.dot.on {
  background:var(--green2);
  animation: dotPulse 2.2s ease-in-out infinite;
}
@keyframes dotPulse {
  0%,100% { box-shadow:0 0 0 2px rgba(34,197,94,.14), 0 0 14px rgba(34,197,94,.25); }
  50%     { box-shadow:0 0 0 5px rgba(34,197,94,.22), 0 0 28px rgba(34,197,94,.50); }
}

/* ── Hero ── */
.hero {
  padding:22px 26px; border-radius:var(--r);
  border:1px solid rgba(136,153,180,.13);
  background:
    radial-gradient(ellipse 800px 380px at 0% 0%,   rgba(29,185,84,.22), transparent 55%),
    radial-gradient(ellipse 700px 300px at 100% 50%, rgba(56,189,248,.14), transparent 55%),
    linear-gradient(160deg, rgba(13,20,32,.97), rgba(8,12,20,.94));
  box-shadow:var(--shadow); overflow:hidden; position:relative;
  animation: heroIn .5s cubic-bezier(.22,1,.36,1) both;
  margin-bottom:16px;
}
.hero::before {
  content:""; position:absolute; inset:-60%;
  background:linear-gradient(90deg, transparent 35%, rgba(255,255,255,.055) 50%, transparent 65%);
  animation: heroShine 7s linear infinite; pointer-events:none;
}
@keyframes heroShine { 0%{transform:translateX(-60%) rotate(12deg)} 100%{transform:translateX(60%) rotate(12deg)} }
.heroTitle {
  font-size:27px; font-weight:900; letter-spacing:-.04em; margin:0;
  background:linear-gradient(90deg, #fff 50%, #8899b4 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.heroSub { color:var(--muted); margin-top:5px; font-size:13.5px; line-height:1.5; }

/* ── EQ bars ── */
.eq { display:inline-flex; gap:3px; margin-right:10px; vertical-align:middle; align-items:flex-end; height:20px; }
.eq i { width:3px; border-radius:3px; background:var(--green); animation:eqB 1.3s ease-in-out infinite; display:block; }
.eq i:nth-child(1){height:7px;  animation-delay:0s}
.eq i:nth-child(2){height:15px; animation-delay:.16s}
.eq i:nth-child(3){height:5px;  animation-delay:.32s}
.eq i:nth-child(4){height:17px; animation-delay:.48s}
.eq i:nth-child(5){height:9px;  animation-delay:.64s}
@keyframes eqB {
  0%,100%{transform:scaleY(.4);  opacity:.6}
  50%    {transform:scaleY(1.3); opacity:1}
}

/* ── Inputs ── */
.stTextInput input {
  background:rgba(255,255,255,.045) !important;
  border:1.5px solid rgba(136,153,180,.20) !important;
  border-radius:14px !important; color:var(--text) !important;
  font-size:15px !important; font-family:'Inter',sans-serif !important;
  padding:12px 16px !important;
  transition:border-color .18s, box-shadow .18s !important;
}
.stTextInput input:focus {
  border-color:rgba(29,185,84,.55) !important;
  box-shadow:0 0 0 4px rgba(29,185,84,.11) !important;
}
.stTextArea textarea {
  background:rgba(255,255,255,.04) !important;
  border:1.5px solid rgba(136,153,180,.18) !important;
  border-radius:12px !important; color:var(--text) !important;
  font-family:'Inter',sans-serif !important;
  transition:border-color .18s, box-shadow .18s !important;
}
.stTextArea textarea:focus {
  border-color:rgba(29,185,84,.50) !important;
  box-shadow:0 0 0 4px rgba(29,185,84,.10) !important;
}
.stSelectbox>div>div {
  background:rgba(255,255,255,.04) !important;
  border:1.5px solid rgba(136,153,180,.18) !important;
  border-radius:12px !important;
}

/* ── Search hint ── */
.search-hint {
  margin-bottom:12px; padding:9px 14px; border-radius:11px;
  background:rgba(29,185,84,.07); border:1px solid rgba(29,185,84,.22);
  font-size:12px; color:var(--muted);
  display:flex; align-items:center; gap:8px;
  animation: slideDown .3s ease both;
}
.search-hint b { color:var(--green2); }

/* ── Card shell ── */
.card {
  background:linear-gradient(180deg, rgba(13,20,32,.95), rgba(9,14,22,.90));
  border:1px solid var(--line); border-radius:var(--r);
  box-shadow:var(--shadow);
}

/* ── Song card ── */
.song {
  padding:15px 17px; border-radius:13px;
  border:1px solid rgba(136,153,180,.08);
  background:rgba(255,255,255,.016);
  transition:transform .16s ease, border-color .16s ease, background .16s ease, box-shadow .16s ease;
  animation: cardIn .38s cubic-bezier(.22,1,.36,1) both;
  cursor: pointer;
}
.song:hover {
  transform:translateY(-4px);
  border-color:rgba(29,185,84,.40);
  background:rgba(29,185,84,.042);
  box-shadow:0 10px 38px rgba(29,185,84,.14);
}
.song .t   { font-weight:800; font-size:14.5px; margin:0; color:var(--text); }
.song .a   { color:var(--muted); font-size:12px; margin-top:3px; }
.song .meta{ display:flex; gap:7px; flex-wrap:wrap; margin-top:10px; }
.song .click-hint {
  font-size:11px; color:rgba(136,153,180,.5); margin-top:8px;
  display:flex; align-items:center; gap:5px;
}
@keyframes cardIn {
  from { opacity:0; transform:translateY(10px); }
  to   { opacity:1; transform:translateY(0); }
}

/* ── Tags ── */
.tag {
  font-size:11px; color:var(--muted); padding:3px 10px; border-radius:999px;
  border:1px solid var(--line); background:rgba(255,255,255,.02);
  display:inline-flex; align-items:center; gap:4px;
}
.tag.green  { color:#b7f7cd; border-color:rgba(29,185,84,.32);  background:rgba(29,185,84,.09); }
.tag.blue   { color:#bae6fd; border-color:rgba(56,189,248,.30); background:rgba(56,189,248,.08); }

/* ── YouTube button ── */
a.yt {
  color:var(--text) !important; text-decoration:none !important;
  display:inline-flex; align-items:center; gap:7px;
  padding:7px 15px; border-radius:999px; font-size:12px; font-weight:600;
  background:rgba(29,185,84,.11); border:1px solid rgba(29,185,84,.30);
  transition:transform .14s ease, background .14s ease, box-shadow .14s ease;
}
a.yt:hover {
  transform:translateY(-2px); background:rgba(29,185,84,.22);
  box-shadow:0 4px 22px rgba(29,185,84,.25);
}

/* ── No results ── */
.no-results {
  text-align:center; padding:48px 20px; color:var(--muted); font-size:14px;
  animation: slideDown .4s ease both;
}
.no-results .nr-icon { font-size:44px; margin-bottom:12px; }
.no-results .nr-tip  { font-size:12px; margin-top:7px; opacity:.7; line-height:1.8; }

/* ── Classify result ── */
.classify-result {
  margin-top:16px; padding:18px 22px; border-radius:14px;
  border:1px solid rgba(29,185,84,.32);
  background:linear-gradient(135deg, rgba(29,185,84,.12), rgba(29,185,84,.04));
  animation: heroIn .38s cubic-bezier(.22,1,.36,1) both;
}
.classify-result .cr-label { color:var(--muted); font-size:12px; margin-bottom:5px; }
.classify-result .cr-value { font-size:26px; font-weight:900; color:var(--green2); letter-spacing:-.03em; }

/* ── Lyrics page ── */
.lyrics-page {
  animation: heroIn .45s cubic-bezier(.22,1,.36,1) both;
}
.lyrics-header {
  padding:28px 32px 24px; border-radius:var(--r);
  border:1px solid rgba(136,153,180,.13);
  background:
    radial-gradient(ellipse 900px 400px at 0% 0%, rgba(29,185,84,.20), transparent 55%),
    radial-gradient(ellipse 700px 300px at 100% 100%, rgba(56,189,248,.12), transparent 55%),
    linear-gradient(160deg, rgba(13,20,32,.97), rgba(8,12,20,.94));
  box-shadow: var(--shadow);
  margin-bottom: 20px;
  position: relative;
  overflow: hidden;
}
.lyrics-header::before {
  content:""; position:absolute; inset:-60%;
  background:linear-gradient(90deg, transparent 35%, rgba(255,255,255,.04) 50%, transparent 65%);
  animation: heroShine 9s linear infinite; pointer-events:none;
}
.lyrics-song-title {
  font-size:32px; font-weight:900; letter-spacing:-.04em; margin:0 0 6px;
  background:linear-gradient(90deg, #fff 55%, #8899b4 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.lyrics-artist {
  font-size:16px; color:var(--muted); font-weight:500;
}
.lyrics-body {
  padding:28px 32px; border-radius:var(--r);
  border:1px solid var(--line);
  background:linear-gradient(180deg, rgba(13,20,32,.95), rgba(9,14,22,.90));
  box-shadow: var(--shadow);
}
.lyrics-text {
  font-size:15px; line-height:2; color:var(--text);
  white-space: pre-wrap; font-family:'Inter', sans-serif;
  letter-spacing:.01em;
}
.lyrics-empty {
  text-align:center; padding:48px 20px; color:var(--muted); font-size:14px;
}
.lyrics-empty .le-icon { font-size:44px; margin-bottom:12px; }

/* ── Keyframes ── */
@keyframes slideDown { from{opacity:0;transform:translateY(-8px)} to{opacity:1;transform:translateY(0)} }
@keyframes heroIn    { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:rgba(136,153,180,.18); border-radius:999px; }
::-webkit-scrollbar-thumb:hover { background:rgba(136,153,180,.32); }

/* ── Back button ── */
.stButton > button {
  background: rgba(255,255,255,.045) !important;
  border: 1.5px solid rgba(136,153,180,.22) !important;
  border-radius: 12px !important;
  color: var(--text) !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  padding: 10px 20px !important;
  transition: all .18s ease !important;
}
.stButton > button:hover {
  border-color: rgba(29,185,84,.45) !important;
  background: rgba(29,185,84,.08) !important;
  box-shadow: 0 4px 20px rgba(29,185,84,.15) !important;
  transform: translateY(-2px) !important;
}
</style>
""", unsafe_allow_html=True)


def render_lyrics_page(row: pd.Series):
    """Renders the full lyrics detail page for a selected song."""
    title   = _safe_text(row.get("song"))    or "Unknown Song"
    artist  = _safe_text(row.get("artist"))  or "Unknown Artist"
    label   = _safe_text(row.get("label"))   or "—"
    cluster = _safe_text(row.get("cluster")) or "—"
    views   = _fmt_int(row.get("views"))
    likes   = _fmt_int(row.get("likes"))
    yt      = _safe_text(row.get("youtube_link"))
    lyrics  = _safe_text(row.get("text")).strip()

    st.markdown('<div class="lyrics-page">', unsafe_allow_html=True)

    # Back button
    if st.button("← Back to results", key="back_btn"):
        st.session_state.selected_song_id = None
        st.rerun()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Header card
    yt_btn = (
        f'<a class="yt" href="{yt}" target="_blank" rel="noreferrer">{YT_SVG} Watch on YouTube</a>'
        if yt.startswith("http") else ""
    )
    st.markdown(f"""
<div class="lyrics-header">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:20px;flex-wrap:wrap">
    <div>
      <div class="lyrics-song-title"><span class="eq"><i></i><i></i><i></i><i></i><i></i></span>{title}</div>
      <div class="lyrics-artist">🎤 {artist}</div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:14px">
        <span class="tag green">🏷 {label}</span>
        <span class="tag blue">◎ Cluster {cluster}</span>
        <span class="tag">👁 {views} views</span>
        <span class="tag">❤️ {likes} likes</span>
      </div>
    </div>
    {"<div style='padding-top:4px'>" + yt_btn + "</div>" if yt_btn else ""}
  </div>
</div>
""",  =True)

    # Lyrics body
    st.markdown('<div class="lyrics-body">', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;font-weight:700;color:var(--muted);letter-spacing:.09em;'
        'text-transform:uppercase;margin-bottom:20px;display:flex;align-items:center;gap:7px">'
        '<span style="display:block;width:3px;height:14px;border-radius:3px;background:var(--green)"></span>'
        '📝 Lyrics</div>',
        unsafe_allow_html=True,
    )
    if lyrics:
        # Escape HTML entities to avoid injection, then display
        import html as html_module
        safe_lyrics = html_module.escape(lyrics)
        st.markdown(f'<div class="lyrics-text">{safe_lyrics}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="lyrics-empty">
  <div class="le-icon">🎼</div>
  <div>No lyrics available for this song.</div>
</div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_song_card(row: pd.Series, idx: int = 0):
    title   = _safe_text(row.get("song"))    or "—"
    artist  = _safe_text(row.get("artist"))  or "—"
    label   = _safe_text(row.get("label"))   or "—"
    cluster = _safe_text(row.get("cluster")) or "—"
    views   = _fmt_int(row.get("views"))
    likes   = _fmt_int(row.get("likes"))
    yt      = _safe_text(row.get("youtube_link"))
    has_lyrics = bool(_safe_text(row.get("text")).strip())

    delay  = f"animation-delay:{min(idx * 0.045, 0.7):.2f}s"
    yt_btn = (
        f'<a class="yt" href="{yt}" target="_blank" rel="noreferrer">{YT_SVG} Watch on YouTube</a>'
        if yt.startswith("http") else ""
    )

    lyrics_hint = "📄 Click to view lyrics" if has_lyrics else "📄 No lyrics available"
    lyrics_hint_color = "rgba(29,185,84,.6)" if has_lyrics else "rgba(136,153,180,.35)"

    # Render card HTML
    st.markdown(f"""
<div class="song" style="{delay}">
  <div class="t">🎵 {title}</div>
  <div class="a">🎤 {artist}</div>
  <div class="meta">
    <span class="tag green">🏷 {label}</span>
    <span class="tag blue">◎ Cluster {cluster}</span>
    <span class="tag">👁 {views}</span>
    <span class="tag">❤️ {likes}</span>
  </div>
  {"<div style='margin-top:12px'>" + yt_btn + "</div>" if yt_btn else ""}
  <div class="click-hint" style="color:{lyrics_hint_color}">{lyrics_hint}</div>
</div>""", unsafe_allow_html=True)

    # Invisible Streamlit button overlaid via layout trick
    song_id = _safe_text(row.get("id")) or str(idx)
    if st.button(f"Open · {title[:30]}", key=f"open_{song_id}_{idx}", help=f"View lyrics for {title}"):
        st.session_state.selected_song_id = song_id
        st.rerun()


def render_sidebar(df: pd.DataFrame, labels: list, clusters: list) -> tuple[str, list, str]:
    total_views = _fmt_int(df["views"].sum())
    st.markdown(f"""
<div class="brand">
  <div class="logo-ring">{MUSIC_SVG}</div>
  <div>
    <div class="brand-name">MusicAI</div>
    <div class="brand-sub">Powered by your models</div>
  </div>
</div>
<div class="stat-row">
  <div class="stat-box"><div class="sv">{len(df):,}</div><div class="sl">Songs</div></div>
  <div class="stat-box"><div class="sv">{len(labels)}</div><div class="sl">Labels</div></div>
  <div class="stat-box"><div class="sv">{total_views}</div><div class="sl">Views</div></div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    view = st.radio("🗂 Navigation", ["🔍 Explore", "🤖 Classify"], horizontal=False)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="settings-header">⚙️ Search Settings</div>', unsafe_allow_html=True)

    st.markdown("**Search in these fields:**")
    col_a, col_b = st.columns(2)
    with col_a:
        chk_song    = st.checkbox("🎵 Song name", value=True,  key="chk_song")
        chk_label   = st.checkbox("🏷 Label",      value=True,  key="chk_label")
    with col_b:
        chk_artist  = st.checkbox("🎤 Artist",     value=True,  key="chk_artist")
        chk_cluster = st.checkbox("◎ Cluster",     value=False, key="chk_cluster")

    search_fields = []
    if chk_song:    search_fields.append("song")
    if chk_artist:  search_fields.append("artist")
    if chk_label:   search_fields.append("label")
    if chk_cluster: search_fields.append("cluster")
    if not search_fields:
        search_fields = ["song", "artist", "label"]

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("**Search engine:**")
    search_mode = st.radio(
        "Engine",
        ["🔮 NLP (Semantic)", "⚡ Fuzzy (Fast)"],
        index=0, horizontal=False,
        label_visibility="collapsed",
        key="search_mode",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    active_label = "NLP semantic" if "NLP" in search_mode else "Fuzzy typo-tolerant"
    st.markdown(
        f"<div class='pill'><span class='dot on'></span><span>{active_label} active</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.caption("'Beyonc', 'shackira', 'popy music' — all work fine.")

    return view, search_fields, search_mode


def main():
    st.set_page_config(
        page_title="MusicAI Explorer",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    # Initialize session state for song detail view
    if "selected_song_id" not in st.session_state:
        st.session_state.selected_song_id = None

    with st.spinner("🎵 Loading music library…"):
        df         = load_df()
        clf, embed = load_models()

    labels   = sorted({x for x in df["label"].map(_safe_text).tolist() if x})
    clusters = sorted({x for x in df["cluster"].map(_safe_text).tolist() if x}, key=lambda x: (len(x), x))

    with st.sidebar:
        view, search_fields, search_mode = render_sidebar(df, labels, clusters)

    # ── Lyrics detail page ──────────────────────────────────────────────────
    if st.session_state.selected_song_id is not None:
        song_id = st.session_state.selected_song_id
        match = df[df["id"] == song_id]
        if not match.empty:
            render_lyrics_page(match.iloc[0])
        else:
            st.error("Song not found.")
            if st.button("← Back"):
                st.session_state.selected_song_id = None
                st.rerun()
        return  # Stop rendering the rest of the page

    # ── Normal explore / classify view ─────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns([2.6, 1.1, 1.1, 1.05, 0.8], gap="small")
    with c1:
        q = st.text_input(
            "Search",
            placeholder="🔍  Song name, artist, label… typos are fine!",
            label_visibility="collapsed",
        )
    with c2:
        label_pick = st.selectbox("Label", ["All labels"] + labels, index=0, label_visibility="collapsed")
    with c3:
        cluster_pick = st.selectbox("Cluster", ["All clusters"] + clusters, index=0, label_visibility="collapsed")
    with c4:
        sort_by = st.selectbox("Sort", ["Top (views)", "Top (likes)", "Song A→Z", "Artist A→Z"], label_visibility="collapsed")
    with c5:
        limit = st.selectbox("Show", [20, 40, 60, 80, 120, 200], index=1, label_visibility="collapsed")

    out = df
    if label_pick != "All labels":
        out = out[out["label"].map(_safe_text) == label_pick]
    if cluster_pick != "All clusters":
        out = out[out["cluster"].map(_safe_text) == cluster_pick]

    search_active = bool(q.strip())
    if search_active:
        out = run_search(out, q, search_fields, search_mode, limit)

    if sort_by == "Top (views)":
        if not search_active:
            out = out.sort_values("views", ascending=False)
    elif sort_by == "Top (likes)":
        out = out.sort_values("likes", ascending=False)
    elif sort_by == "Song A→Z":
        out = out.sort_values("song", ascending=True, key=lambda s: s.map(_normalize))
    else:
        out = out.sort_values("artist", ascending=True, key=lambda s: s.map(_normalize))

    field_names = {"song": "Song", "artist": "Artist", "label": "Label", "cluster": "Cluster"}
    active_fields_str = " · ".join(field_names[f] for f in search_fields)
    engine_str = "NLP Semantic" if "NLP" in search_mode else "Fuzzy"

    hero_title = "Search Results" if search_active else "Your Music Explorer"
    hero_sub = (
        f'Searching in <b style="color:var(--green2)">{active_fields_str}</b> '
        f'using <b style="color:var(--blue)">{engine_str}</b> engine'
        if search_active else
        "Search songs by name, artist, or label — with typo tolerance & semantic NLP understanding."
    )

    st.markdown(f"""
<div class="hero">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap">
    <div>
      <div class="heroTitle"><span class="eq"><i></i><i></i><i></i><i></i><i></i></span>{hero_title}</div>
      <div class="heroSub">{hero_sub}</div>
    </div>
    <div class="pill">
      <span class="dot on"></span>
      <span><b>{len(out):,}</b> matches &nbsp;·&nbsp; showing <b>{min(limit, len(out))}</b></span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    if search_active and len(out) > 0:
        q_short = q.strip()[:40]
        st.markdown(
            f'<div class="search-hint">✨ Results for <b>"{q_short}"</b> — '
            f'fields: <b>{active_fields_str}</b> — '
            f'engine: <b>{engine_str}</b> — typos & near-matches included</div>',
            unsafe_allow_html=True,
        )

    if view == "🔍 Explore":
        st.markdown('<div class="card" style="padding:16px 16px 10px">', unsafe_allow_html=True)
        shown = out.head(limit)

        if shown.empty:
            st.markdown(f"""
<div class="no-results">
  <div class="nr-icon">🎵</div>
  <div>No songs matched <b>"{q.strip()}"</b></div>
  <div class="nr-tip">
    Searched in: {active_fields_str}<br>
    Try: a shorter word · check the field checkboxes in the sidebar · switch engine
  </div>
</div>""", unsafe_allow_html=True)
        else:
            for i, (_, row) in enumerate(shown.iterrows()):
                render_song_card(row, idx=i)
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown('<div class="card" style="padding:22px 24px">', unsafe_allow_html=True)
        st.markdown("""
<div style="margin-bottom:18px">
  <div style="font-size:22px;font-weight:900;letter-spacing:-.03em">🤖 Classify a Song</div>
  <div style="color:var(--muted);font-size:13px;margin-top:5px">
    Paste lyrics or describe the vibe — your model predicts the category label.
  </div>
</div>""", unsafe_allow_html=True)

        with st.form("classify_form"):
            col1, col2 = st.columns(2)
            with col1:
                user_song   = st.text_input("🎵 Song title",  placeholder="e.g. Blinding Lights")
                user_artist = st.text_input("🎤 Artist name", placeholder="e.g. The Weeknd")
            with col2:
                seed_mode = st.selectbox(
                    "Embedding input",
                    ["Lyrics / text only", "Title + artist + lyrics"],
                    help="What text to embed before classifying",
                )
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                st.caption("The model embeds the text and predicts the most fitting label.")

            user_text = st.text_area(
                "📝 Lyrics / description",
                height=160,
                placeholder="Paste lyrics here, or describe the mood / genre / style…",
            )
            submitted = st.form_submit_button("✨ Classify Now", use_container_width=True)

        if submitted:
            txt = user_text.strip()
            if seed_mode == "Title + artist + lyrics":
                txt = f"{user_song} {user_artist}\n{txt}".strip()

            if not txt:
                st.error("⚠️ Please enter some lyrics or a description to classify.")
            else:
                with st.spinner("🔮 Embedding + classifying…"):
                    v = np.asarray(embed.encode([txt], batch_size=1, show_progress_bar=False), dtype=np.float32)
                    try:
                        pred = clf.predict(v)[0]
                    except Exception:
                        pred = clf.predict(np.asarray(v).reshape(1, -1))[0]

                st.markdown(f"""
<div class="classify-result">
  <div class="cr-label">🏷 Predicted category (label)</div>
  <div class="cr-value">{_safe_text(pred)}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()