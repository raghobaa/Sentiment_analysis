"""
Sentiment Analysis – Model Comparison Dashboard
================================================
Three-tab Streamlit app:
  Tab 1 – 📊 Metrics Comparison   : radar / bar charts + styled table
  Tab 2 – 🎬 Unseen Reviews       : run all models on a3_IMDb_Unseen_Reviews.csv
  Tab 3 – ✍️  Live Inference       : type any review and compare predictions

Run:
    streamlit run app.py
"""

import os, re, pickle, pathlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR       = pathlib.Path(__file__).parent
IMDB_CSV       = BASE_DIR / "a1_IMDB_Dataset.csv"
UNSEEN_CSV     = BASE_DIR / "a3_IMDb_Unseen_Reviews.csv"
TOKENIZER_PKL  = BASE_DIR / ".tokenizer_cache.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# Model registry
#   maxlen=100  → CNN models + c1_lstm (trained with maxlen=100)
#   maxlen=200  → LSTM models trained with LSTM-specific padding
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = [
    {
        "file":       "c1_cnn_model_acc_flatten()_0.799.h5",
        "name":       "CNN – Flatten",
        "short":      "CNN-Flat",
        "family":     "CNN",
        "maxlen":     100,
        "known_acc":  0.799,
        "desc":       "Conv1D → Flatten → Dense(1)",
    },
    {
        "file":       "c1_cnn_model_acc_flatten()_droput.30.818.h5",
        "name":       "CNN – Flatten + Dropout",
        "short":      "CNN-Flat-Drop",
        "family":     "CNN",
        "maxlen":     100,
        "known_acc":  0.818,
        "desc":       "Conv1D → Flatten → Dropout(0.3) → Dense(1)",
    },
    {
        "file":       "c1_cnn_model_acc_globalmax_droput.30.848.h5",
        "name":       "CNN – GlobalMax + Dropout",
        "short":      "CNN-GMax",
        "family":     "CNN",
        "maxlen":     100,
        "known_acc":  0.848,
        "desc":       "Conv1D → GlobalMaxPool → Dropout(0.3) → Dense(1)",
    },
    {
        "file":       "c1_cnn_model_acc_globalmax_droput.3_earlystop0.864.h5",
        "name":       "CNN – GlobalMax + EarlyStopping",
        "short":      "CNN-GMax-ES",
        "family":     "CNN",
        "maxlen":     100,
        "known_acc":  0.864,
        "desc":       "CNN-GMax with EarlyStopping callback",
    },
    {
        "file":       "c1_lstm_model_acc_0.864.h5",
        "name":       "LSTM – Basic (own emb)",
        "short":      "LSTM-Basic",
        "family":     "LSTM",
        "maxlen":     100,
        "known_acc":  0.864,
        "desc":       "LSTM(128) → Dropout(0.2) → Dense(1), own embeddings",
    },
    {
        "file":       "l1_lstm_model_glove_emb0.864.h5",
        "name":       "LSTM – GloVe Embeddings",
        "short":      "LSTM-GloVe",
        "family":     "LSTM",
        "maxlen":     200,
        "known_acc":  0.864,
        "desc":       "LSTM(128) + frozen GloVe 100d embeddings",
    },
    {
        "file":       "l1_lstm_own_weights.30.865.h5",
        "name":       "LSTM – Own Emb v1",
        "short":      "LSTM-Own-v1",
        "family":     "LSTM",
        "maxlen":     200,
        "known_acc":  0.865,
        "desc":       "Deep LSTM(128→64) + Dropout, own trained embeddings",
    },
    {
        "file":       "l1_lstm_own_weights.30.874.h5",
        "name":       "LSTM – Own Emb v2 (Best)",
        "short":      "LSTM-Own-v2",
        "family":     "LSTM",
        "maxlen":     200,
        "known_acc":  0.874,
        "desc":       "Deep LSTM(128→64) + Dropout, refined training – best model",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Text preprocessing  (mirrors the notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────
TAG_RE = re.compile(r"<[^>]+>")

def remove_tags(text: str) -> str:
    return TAG_RE.sub("", text)

def preprocess_text(sen: str) -> str:
    """Clean raw review text — identical to notebook pipeline."""
    import nltk
    try:
        from nltk.corpus import stopwords
        STOPS = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        STOPS = set(stopwords.words("english"))

    sentence = remove_tags(sen)
    sentence = re.sub(r"[^a-zA-Z]", " ", sentence)   # keep only letters
    sentence = re.sub(r"\s+", " ", sentence)          # collapse spaces
    sentence = sentence.lower().strip()
    tokens   = [w for w in sentence.split() if len(w) >= 2 and w not in STOPS]
    return " ".join(tokens)

# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer — cached to disk after first fit
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔤  Building tokenizer (one-time, ~60s)…")
def load_tokenizer():
    """Fit tokenizer on IMDB training split — same split as notebook (80/20, seed 42)."""
    import nltk
    nltk.download("stopwords", quiet=True)

    if TOKENIZER_PKL.exists():
        with open(TOKENIZER_PKL, "rb") as fh:
            return pickle.load(fh)

    from tensorflow.keras.preprocessing.text import Tokenizer
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(IMDB_CSV)
    reviews = [preprocess_text(r) for r in df["review"]]
    X_train, _ = train_test_split(reviews, test_size=0.20, random_state=42)

    tok = Tokenizer()
    tok.fit_on_texts(X_train)

    with open(TOKENIZER_PKL, "wb") as fh:
        pickle.dump(tok, fh)
    return tok


def tokenize_and_pad(texts: list[str], tokenizer, maxlen: int) -> np.ndarray:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, padding="post", maxlen=maxlen)

# ─────────────────────────────────────────────────────────────────────────────
# Compatibility shims
# ─────────────────────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.layers import LSTM as _KerasLSTM

class CompatLSTM(_KerasLSTM):
    """Drop unsupported kwargs (e.g. `time_major`) added by older TF saves."""
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        config.pop("time_major", None)
        return super().from_config(config)


_COMPAT_OBJECTS = {"LSTM": CompatLSTM}


def _load_with_weight_fallback(path: str):
    """
    Last-resort loader for models whose Conv/Dense weight shapes were serialised
    differently (e.g. Conv1D kernel saved as shape (0,)).

    Strategy:
      1. Load the model graph from the h5 config (no weights).
      2. Open the h5 file with h5py and copy weights layer-by-layer,
         skipping any whose shape doesn't match the built model.
    """
    import h5py
    from tensorflow.keras.models import load_model as _lm

    # Build graph only (ignore weight values)
    model = _lm(path, compile=False, custom_objects=_COMPAT_OBJECTS)

    with h5py.File(path, "r") as f:
        # Keras stores weights under "model_weights" group
        if "model_weights" not in f:
            return model          # nothing more we can do
        mw = f["model_weights"]

        for layer in model.layers:
            lname = layer.name
            if lname not in mw:
                continue
            grp = mw[lname]
            # Collect dataset names two levels deep
            weight_names = []
            for k in grp:
                sub = grp[k]
                if hasattr(sub, "keys"):
                    for kk in sub:
                        weight_names.append(f"{k}/{kk}")
                else:
                    weight_names.append(k)

            saved_weights = [grp[wn][()] for wn in weight_names]
            layer_weights = layer.get_weights()

            if len(saved_weights) != len(layer_weights):
                continue  # skip — count mismatch

            compatible = []
            skip = False
            for sw, lw in zip(saved_weights, layer_weights):
                if sw.shape != lw.shape:
                    skip = True
                    break
                compatible.append(sw)

            if not skip:
                layer.set_weights(compatible)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Model loader — load all 8 models once, cache in session
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🧠  Loading models…")
def load_all_models():
    from tensorflow.keras.models import load_model
    loaded = {}

    for m in MODEL_REGISTRY:
        path = BASE_DIR / m["file"]
        if not path.exists():
            continue

        model = None
        last_err = None

        # ── Strategy 1: plain load ──────────────────────────────────────────
        try:
            model = load_model(str(path), compile=False)
        except Exception as e:
            last_err = e

        # ── Strategy 2: with CompatLSTM (fixes `time_major` error) ─────────
        if model is None:
            try:
                model = load_model(
                    str(path), compile=False, custom_objects=_COMPAT_OBJECTS
                )
            except Exception as e:
                last_err = e

        # ── Strategy 3: weight-shape fallback (fixes Conv1D (0,) error) ────
        if model is None:
            try:
                model = _load_with_weight_fallback(str(path))
            except Exception as e:
                last_err = e

        if model is not None:
            loaded[m["short"]] = model
        else:
            st.warning(f"⚠️ Could not load **{m['name']}**: {last_err}")

    return loaded

# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────
def predict_all(texts: list[str], tokenizer, models: dict) -> dict[str, np.ndarray]:
    """Return raw sigmoid scores (0-1) for each model, shape (n_texts,)."""
    results = {}
    for meta in MODEL_REGISTRY:
        key = meta["short"]
        if key not in models:
            continue
        padded = tokenize_and_pad(texts, tokenizer, meta["maxlen"])
        scores = models[key].predict(padded, verbose=0).flatten()
        results[key] = scores
    return results

# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth derivation for unseen reviews
#   IMDb rating ≥ 6 → positive (1), ≤ 4 → negative (0), else → ambiguous
# ─────────────────────────────────────────────────────────────────────────────
def rating_to_label(rating: int) -> int | None:
    if rating >= 6:
        return 1
    elif rating <= 4:
        return 0
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────
FAMILY_COLORS = {"CNN": "#636EFA", "LSTM": "#EF553B"}

def score_color(val: float, best: float, worst: float) -> str:
    """Return green-to-red background for a metric value."""
    norm = (val - worst) / (best - worst + 1e-9)
    r = int(255 * (1 - norm))
    g = int(200 * norm)
    return f"rgba({r},{g},80,0.25)"

# ─────────────────────────────────────────────────────────────────────────────
# ─── PAGE LAYOUT ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Model Comparison",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg,#1e1e2e,#2d2d44);
        border-radius:12px; padding:18px 22px; margin:6px 0;
        border-left:4px solid #636EFA;
    }
    .stTabs [data-baseweb="tab-list"] { gap:10px; }
    .stTabs [data-baseweb="tab"] {
        font-size:16px; font-weight:600; padding:10px 20px; border-radius:8px;
    }
    div[data-testid="stExpander"] summary { font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.title("🎭 IMDb Sentiment Analysis — Model Comparison Dashboard")
st.caption("Comparing 8 deep learning models (4 CNN · 4 LSTM) trained on the IMDb 50k dataset")

# Load shared resources
tokenizer = load_tokenizer()
models    = load_all_models()

tab1, tab2, tab3 = st.tabs([
    "📊  Metrics Comparison",
    "🎬  Unseen Reviews Benchmark",
    "✍️   Live Inference",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — METRICS COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Model Performance Comparison")
    st.markdown(
        "Accuracy values are taken from model filenames (reported on the 20% held-out IMDB test set). "
        "Precision / Recall / F1 / AUC are computed live on the **6 unseen IMDb reviews** "
        "(ground truth derived from IMDb rating: ≥6 → positive, ≤4 → negative)."
    )

    # ── Compute unseen-review metrics for all models ──────────────────────
    unseen_df = pd.read_csv(UNSEEN_CSV)
    unseen_df["GT_label"] = unseen_df["IMDb Rating"].apply(rating_to_label)
    valid_mask = unseen_df["GT_label"].notna()
    valid_df   = unseen_df[valid_mask].reset_index(drop=True)

    processed_texts = [preprocess_text(r) for r in valid_df["Review Text"]]
    y_true = valid_df["GT_label"].astype(int).values

    raw_preds = predict_all(processed_texts, tokenizer, models)

    metrics_rows = []
    for meta in MODEL_REGISTRY:
        key = meta["short"]
        if key not in models or key not in raw_preds:
            continue
        scores  = raw_preds[key]
        y_pred  = (scores >= 0.5).astype(int)
        try:
            auc = roc_auc_score(y_true, scores)
        except Exception:
            auc = float("nan")
        metrics_rows.append({
            "Model":     meta["name"],
            "Short":     key,
            "Family":    meta["family"],
            "Arch":      meta["desc"],
            "Acc (50k)": meta["known_acc"],
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall":    recall_score(y_true, y_pred, zero_division=0),
            "F1":        f1_score(y_true, y_pred, zero_division=0),
            "AUC":       auc,
            "Acc (6-rev)": accuracy_score(y_true, y_pred),
        })

    metrics_df = pd.DataFrame(metrics_rows)

    # ── KPI summary cards ────────────────────────────────────────────────
    best_model = metrics_df.loc[metrics_df["Acc (50k)"].idxmax()]
    best_f1    = metrics_df.loc[metrics_df["F1"].idxmax()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 Best Accuracy (50k set)", f"{best_model['Acc (50k)']:.1%}", best_model["Short"])
    c2.metric("🎯 Best F1 (unseen set)",    f"{best_f1['F1']:.3f}",           best_f1["Short"])
    c3.metric("📦 Models loaded",           str(len(metrics_df)))
    c4.metric("📋 Unseen reviews",          str(len(valid_df)))

    st.divider()

    # ── Layout: charts left, table right ────────────────────────────────
    col_charts, col_table = st.columns([3, 2], gap="large")

    with col_charts:
        # ── Grouped bar chart ──────────────────────────────────────────
        metric_cols = ["Acc (50k)", "Precision", "Recall", "F1", "AUC"]
        bar_fig = go.Figure()
        colors  = [FAMILY_COLORS[f] for f in metrics_df["Family"]]
        for i, mcol in enumerate(metric_cols):
            bar_fig.add_trace(go.Bar(
                name=mcol,
                x=metrics_df["Short"],
                y=metrics_df[mcol],
                text=[f"{v:.3f}" for v in metrics_df[mcol]],
                textposition="outside",
                visible=(i == 0),
            ))

        # Dropdown to switch metric
        bar_fig.update_layout(
            updatemenus=[dict(
                type="dropdown",
                direction="down",
                x=0.0, y=1.15,
                buttons=[
                    dict(label=m, method="update",
                         args=[{"visible": [j == i for j in range(len(metric_cols))]},
                               {"title": f"Model Comparison — {m}"}])
                    for i, m in enumerate(metric_cols)
                ]
            )],
            title="Model Comparison — Acc (50k)",
            yaxis=dict(range=[0, 1.15], tickformat=".0%"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=False,
            height=400,
        )
        bar_fig.update_traces(marker_color=colors)
        st.plotly_chart(bar_fig, use_container_width=True)

        # ── Radar chart (all metrics together) ────────────────────────
        radar_metrics = ["Acc (50k)", "Precision", "Recall", "F1", "AUC"]
        radar_fig = go.Figure()
        family_palette = {"CNN": "#636EFA", "LSTM": "#EF553B"}
        plotted_families: set[str] = set()

        for _, row in metrics_df.iterrows():
            family  = row["Family"]
            color   = family_palette[family]
            opacity = 0.35
            radar_fig.add_trace(go.Scatterpolar(
                r=[row[m] for m in radar_metrics] + [row[radar_metrics[0]]],
                theta=radar_metrics + [radar_metrics[0]],
                fill="toself",
                fillcolor=color.replace(")", f",{opacity})").replace("rgb", "rgba"),
                line=dict(color=color, width=2),
                name=row["Short"],
            ))

        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1],
                                tickformat=".0%", gridcolor="rgba(255,255,255,0.1)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=True,
            legend=dict(orientation="h", y=-0.15),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=420,
            title="Radar — All Metrics",
        )
        st.plotly_chart(radar_fig, use_container_width=True)

    with col_table:
        st.subheader("📋 Full Metrics Table")
        st.caption("Sorted by Accuracy (50k test set) ↓")

        display_df = (
            metrics_df[["Short", "Family", "Acc (50k)", "Precision", "Recall", "F1", "AUC", "Acc (6-rev)"]]
            .sort_values("Acc (50k)", ascending=False)
            .reset_index(drop=True)
        )
        display_df.index = display_df.index + 1  # 1-based rank

        def highlight_best(col):
            is_numeric = pd.api.types.is_numeric_dtype(col)
            if not is_numeric:
                return [""] * len(col)
            best_idx = col.idxmax()
            return ["background-color: rgba(99,110,250,0.35); font-weight:bold"
                    if i == best_idx else "" for i in col.index]

        numeric_cols = ["Acc (50k)", "Precision", "Recall", "F1", "AUC", "Acc (6-rev)"]
        styled = (
            display_df.style
            .apply(highlight_best, subset=numeric_cols)
            .format({c: "{:.3f}" for c in numeric_cols})
        )
        st.dataframe(styled, use_container_width=True, height=380)

        st.subheader("🏗️ Architecture Details")
        arch_df = metrics_df[["Short", "Family", "Arch"]].rename(columns={"Arch": "Architecture"})
        st.dataframe(arch_df, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — UNSEEN REVIEWS BENCHMARK
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🎬 Unseen IMDb Reviews — Model Predictions")
    st.markdown(
        "All models predict on the **6 held-out movie reviews** (never seen during training). "
        "Ground truth is derived from the IMDb rating: **≥ 6 → Positive**, **≤ 4 → Negative**."
    )

    unseen_df2    = pd.read_csv(UNSEEN_CSV)
    unseen_df2["GT_label"] = unseen_df2["IMDb Rating"].apply(rating_to_label)
    unseen_df2["Ground Truth"] = unseen_df2["GT_label"].map(
        {1: "😊 Positive", 0: "😞 Negative", None: "❓ Ambiguous"}
    )

    # Run predictions (reuse raw_preds already computed in Tab 1)
    processed_all = [preprocess_text(r) for r in unseen_df2["Review Text"]]
    raw_preds_all = predict_all(processed_all, tokenizer, models)

    # ── Per-review prediction table ────────────────────────────────────────
    st.subheader("Per-Review Predictions")

    pred_table = unseen_df2[["Movie", "IMDb Rating", "Ground Truth"]].copy()
    for meta in MODEL_REGISTRY:
        key = meta["short"]
        if key not in raw_preds_all:
            continue
        scores = raw_preds_all[key]
        labels = ["😊 Pos" if s >= 0.5 else "😞 Neg" for s in scores]
        correct_marks = []
        for s, gt in zip(scores, unseen_df2["GT_label"]):
            pred = 1 if s >= 0.5 else 0
            if gt is None:
                correct_marks.append("❓")
            elif pred == int(gt):
                correct_marks.append("✅")
            else:
                correct_marks.append("❌")
        # Show label + tick
        pred_table[key] = [f"{l} {c}" for l, c in zip(labels, correct_marks)]

    # Show review text in expanders below table
    st.dataframe(pred_table, use_container_width=True, hide_index=True)

    st.subheader("📖 Review Texts")
    for i, row in unseen_df2.iterrows():
        with st.expander(f"Review {i+1} — {row['Movie']} (IMDb: {row['IMDb Rating']}/10) — {row['Ground Truth']}"):
            st.write(row["Review Text"])

    st.divider()

    # ── Per-model accuracy on 6 reviews ───────────────────────────────────
    st.subheader("Model Accuracy on Unseen Reviews")

    valid_idx  = unseen_df2["GT_label"].notna()
    y_true_tab2 = unseen_df2.loc[valid_idx, "GT_label"].astype(int).values

    acc_rows = []
    for meta in MODEL_REGISTRY:
        key = meta["short"]
        if key not in raw_preds_all:
            continue
        scores = raw_preds_all[key][valid_idx.values]
        y_pred = (scores >= 0.5).astype(int)
        acc_rows.append({
            "Model": key,
            "Correct": int((y_pred == y_true_tab2).sum()),
            "Total":   len(y_true_tab2),
            "Accuracy": accuracy_score(y_true_tab2, y_pred),
            "Family":  meta["family"],
        })

    acc_df = pd.DataFrame(acc_rows).sort_values("Accuracy", ascending=False)

    acc_fig = px.bar(
        acc_df, x="Model", y="Accuracy", color="Family",
        color_discrete_map=FAMILY_COLORS,
        text=acc_df.apply(lambda r: f"{r['Correct']}/{r['Total']}", axis=1),
        title="Accuracy on 6 Unseen Reviews",
        labels={"Accuracy": "Accuracy"},
    )
    acc_fig.update_layout(
        yaxis=dict(range=[0, 1.15], tickformat=".0%"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=380,
    )
    acc_fig.update_traces(textposition="outside")
    st.plotly_chart(acc_fig, use_container_width=True)

    # ── Confidence heatmap ────────────────────────────────────────────────
    st.subheader("🌡️ Confidence Score Heatmap")
    st.caption("Cell values = model's raw sigmoid output (0 = confident Negative, 1 = confident Positive)")

    model_keys = [m["short"] for m in MODEL_REGISTRY if m["short"] in raw_preds_all]
    heat_data  = np.array([raw_preds_all[k] for k in model_keys])
    review_labels = [
        f"Rev {i+1}: {r['Movie']} ({r['IMDb Rating']}/10)"
        for i, r in unseen_df2.iterrows()
    ]

    heat_fig = go.Figure(data=go.Heatmap(
        z=heat_data,
        x=review_labels,
        y=model_keys,
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        text=np.round(heat_data, 3),
        texttemplate="%{text}",
        hoverongaps=False,
    ))
    heat_fig.update_layout(
        title="Confidence Heatmap (per model × review)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=380,
    )
    st.plotly_chart(heat_fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — LIVE INFERENCE
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("✍️ Live Review Inference")
    st.markdown("Type or paste **any movie review** below. All loaded models will predict its sentiment in real-time.")

    col_input, col_example = st.columns([3, 1])
    with col_example:
        st.subheader("💡 Quick examples")
        EXAMPLES = {
            "Positive review": (
                "An absolute masterpiece! The acting was superb, the direction was flawless, "
                "and the storyline kept me engaged from start to finish. Highly recommended!"
            ),
            "Negative review": (
                "Terrible movie. The plot made no sense, the acting was wooden, "
                "and I fell asleep halfway through. A complete waste of time and money."
            ),
            "Mixed review": (
                "The visuals were stunning and the soundtrack was great, "
                "but the story felt disjointed and the ending was disappointing."
            ),
        }
        chosen = st.radio("Load example:", list(EXAMPLES.keys()), index=None)

    with col_input:
        default_text = EXAMPLES[chosen] if chosen else ""
        user_review = st.text_area(
            "Enter your review here:",
            value=default_text,
            height=180,
            placeholder="e.g. This movie was absolutely brilliant…",
        )

    if st.button("🚀 Predict Sentiment", type="primary", use_container_width=False):
        if not user_review.strip():
            st.warning("Please enter a review first.")
        else:
            clean = preprocess_text(user_review)
            if not clean.strip():
                st.error("Review is empty after preprocessing (try a longer text).")
            else:
                live_preds = predict_all([clean], tokenizer, models)

                st.divider()
                st.subheader("Predictions")

                # Aggregate verdict
                scores_list = [v[0] for v in live_preds.values()]
                avg_score   = float(np.mean(scores_list))
                pos_votes   = sum(1 for s in scores_list if s >= 0.5)
                neg_votes   = len(scores_list) - pos_votes

                verdict_col, detail_col = st.columns([1, 2])
                with verdict_col:
                    if avg_score >= 0.6:
                        st.success(f"### 😊 POSITIVE\nAverage confidence: **{avg_score:.1%}**")
                    elif avg_score <= 0.4:
                        st.error(f"### 😞 NEGATIVE\nAverage confidence: **{1-avg_score:.1%}**")
                    else:
                        st.warning(f"### 🤔 MIXED\nAverage score: **{avg_score:.1%}**")
                    st.metric("Positive votes", f"{pos_votes} / {len(scores_list)}")

                with detail_col:
                    st.markdown("**Per-model confidence (→ Positive)**")
                    sorted_preds = sorted(
                        [(meta["name"], live_preds[meta["short"]][0])
                         for meta in MODEL_REGISTRY if meta["short"] in live_preds],
                        key=lambda x: x[1], reverse=True
                    )
                    for name, score in sorted_preds:
                        label  = "😊 Positive" if score >= 0.5 else "😞 Negative"
                        bar_val = float(score)
                        col_prog, col_lbl = st.columns([3, 1])
                        with col_prog:
                            st.progress(bar_val, text=name)
                        with col_lbl:
                            st.write(f"{score:.3f} — {label}")

                st.divider()

                # Voting chart
                vote_fig = go.Figure(go.Bar(
                    x=[m["short"] for m in MODEL_REGISTRY if m["short"] in live_preds],
                    y=[float(live_preds[m["short"]][0]) for m in MODEL_REGISTRY if m["short"] in live_preds],
                    marker_color=[
                        "#2ecc71" if float(live_preds[m["short"]][0]) >= 0.5 else "#e74c3c"
                        for m in MODEL_REGISTRY if m["short"] in live_preds
                    ],
                    text=[
                        f"{float(live_preds[m['short']][0]):.3f}"
                        for m in MODEL_REGISTRY if m["short"] in live_preds
                    ],
                    textposition="outside",
                ))
                vote_fig.add_hline(y=0.5, line_dash="dash", line_color="white",
                                   annotation_text="Decision boundary (0.5)")
                vote_fig.update_layout(
                    title="Model Confidence Scores",
                    yaxis=dict(range=[0, 1.15], title="P(Positive)", tickformat=".0%"),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    height=380,
                )
                st.plotly_chart(vote_fig, use_container_width=True)

                # Show preprocessed text
                with st.expander("🔍 Preprocessed input sent to models"):
                    st.code(clean)
    else:
        st.info("👆 Enter a review above and click **Predict Sentiment** to see all models in action.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with Streamlit · TensorFlow/Keras · Plotly · "
    "Models trained on IMDb 50k dataset · "
    "GloVe 100d embeddings by Stanford NLP"
)
