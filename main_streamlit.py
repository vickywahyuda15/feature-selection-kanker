from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# DASHBOARD INTERAKTIF - FEATURE SELECTION KANKER PAYUDARA
# Membaca hasil dari validasi_penelitian_final.py
# Tidak menjalankan ulang eksperimen.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "hasil_pengujian"
METHOD_ORDER = ["Baseline", "SFS", "RFE", "GA"]
METHOD_COLORS = {
    "Baseline": "#2563EB",
    "SFS": "#0F766E",
    "RFE": "#7C3AED",
    "GA": "#D97706",
}

st.set_page_config(
    page_title="Dashboard Feature Selection",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- Styling ---------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .stApp {
        background: radial-gradient(circle at 85% 0%, rgba(59,130,246,.08), transparent 25%),
                    linear-gradient(180deg, #f7f9fc 0%, #eef3f9 100%);
      }
      [data-testid="stHeader"] { background: rgba(247,249,252,.88); }
      .block-container { max-width: 1500px; padding-top: 1.4rem; padding-bottom: 2.5rem; }
      [data-testid="stSidebar"] { background: #101a2c; }
      [data-testid="stSidebar"] * { color: #edf3ff !important; }
      [data-testid="stSidebar"] [data-baseweb="select"] div { color: #111827 !important; }
      .hero {
        background: linear-gradient(120deg, #101a2c 0%, #183b66 58%, #087e8b 100%);
        padding: 30px 34px; border-radius: 24px; color: white; margin-bottom: 20px;
        box-shadow: 0 14px 34px rgba(15, 23, 42, .17);
      }
      .hero-kicker { color: #a5f3fc; text-transform: uppercase; letter-spacing: .15em;
        font-size: .76rem; font-weight: 700; margin-bottom: 9px; }
      .hero h1 { color: white; font-family: 'Manrope', sans-serif; font-size: 2.15rem;
        font-weight: 800; line-height: 1.2; margin: 0 0 10px 0; }
      .hero p { color: #dce8f7; margin: 0; font-size: 1rem; max-width: 980px; }
      .hero-chip { display: inline-block; margin-top: 18px; margin-right: 8px; border-radius: 999px;
        border: 1px solid rgba(255,255,255,.25); background: rgba(255,255,255,.10);
        color: #f8fafc; padding: 6px 11px; font-size: .8rem; }
      .section-head { font-family: 'Manrope', sans-serif; font-size: 1.25rem; font-weight: 800;
        color: #14233a; margin: 12px 0 4px 0; }
      .section-sub { color: #64748b; margin: 0 0 14px 0; font-size: .9rem; }
      .metric-card { background: white; border: 1px solid #e2e8f0; border-radius: 18px;
        padding: 18px 20px; min-height: 122px; box-shadow: 0 4px 14px rgba(15,23,42,.045); }
      .metric-label { color: #64748b; font-size: .83rem; font-weight: 600; margin-bottom: 8px; }
      .metric-value { color: #14233a; font-family: 'Manrope', sans-serif; font-size: 1.75rem;
        font-weight: 800; line-height: 1.15; }
      .metric-help { color: #64748b; font-size: .78rem; margin-top: 8px; }
      .panel { background: white; border: 1px solid #e2e8f0; border-radius: 18px;
        padding: 18px 20px; box-shadow: 0 4px 14px rgba(15,23,42,.04); }
      .pill { display: inline-block; border-radius: 999px; padding: 4px 10px; background: #dcfce7;
        color: #166534; font-size: .77rem; font-weight: 700; }
      .note { border-left: 4px solid #0f766e; background: #ecfeff; padding: 12px 15px;
        border-radius: 0 12px 12px 0; color: #164e63; font-size: .91rem; }
      .result-reading { color: #14233a !important; background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 12px 16px; margin: 8px 0; line-height: 1.6; }
      .result-reading, .result-reading p, .result-reading span, .result-reading strong {
        color: #14233a !important; }
      .footer { color: #64748b; font-size: .78rem; text-align: center; padding-top: 24px; }
      div[data-testid="stMetric"] { background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 16px; padding: 12px 16px; box-shadow: 0 4px 14px rgba(15,23,42,.04); }
      div[data-testid="stMetricLabel"] p { color: #64748b; font-weight: 600; }
      div[data-testid="stMetricValue"] { color: #14233a; font-weight: 800; }
      div[data-testid="stTabs"] button { font-weight: 700; }
      .stDataFrame { border-radius: 12px; overflow: hidden; }

      /* Mode presentasi: tata letak lebih ringkas dan bersih */
      [data-testid="stToolbar"],
      [data-testid="stDecoration"],
      #MainMenu,
      footer {
        visibility: hidden !important;
      }
      .block-container {
        max-width: 1320px;
        padding-top: 1rem;
        padding-bottom: 1.5rem;
      }
      .hero {
        padding: 24px 28px;
        border-radius: 20px;
        margin-bottom: 16px;
      }
      .hero h1 {
        font-size: 2rem;
      }
      .section-head {
        margin-top: 8px;
      }
      
    </style>
    """,
    unsafe_allow_html=True,
)


def norm(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def standardize_method(value):
    text = str(value).strip().lower()
    if text in {"baseline", "base line"}:
        return "Baseline"
    if text == "sfs" or "sequentialforward" in text:
        return "SFS"
    if text == "rfe" or "recursivefeature" in text:
        return "RFE"
    if text in {"ga", "genetic algorithm", "geneticalgorithm"} or "genetic" in text:
        return "GA"
    return str(value).strip()


def find_col(df, candidates):
    mapping = {norm(col): col for col in df.columns}
    for candidate in candidates:
        if norm(candidate) in mapping:
            return mapping[norm(candidate)]
    return None


def read_csv(path):
    if not path or not Path(path).exists():
        return None
    for kwargs in ({"encoding": "utf-8-sig"}, {"encoding": "latin1"}):
        try:
            df = pd.read_csv(path, **kwargs)
            if not df.empty:
                return df
        except Exception:
            continue
    return None


def first_existing(folder, names):
    for name in names:
        path = folder / name
        if path.exists():
            return path
    return None


def load_per_fold():
    # Utamakan data final gabungan, agar dashboard tidak secara tidak sengaja
    # membaca file ringkasan lama yang berada di root project.
    path = first_existing(RESULT_DIR, ["seluruh_hasil_per_fold.csv", "hasil_validasi_per_fold.csv"])
    if path:
        return read_csv(path), path

    # Fallback: gabungkan keluaran per seed apabila file gabungan belum ada.
    parts = []
    for seed in [21, 42, 2026]:
        candidate = RESULT_DIR / f"seed_{seed}" / "hasil_per_fold.csv"
        df = read_csv(candidate)
        if df is not None:
            parts.append(df)
    if parts:
        return pd.concat(parts, ignore_index=True), RESULT_DIR

    # Fallback terakhir untuk file lama di direktori project.
    path = first_existing(BASE_DIR, ["seluruh_hasil_per_fold.csv", "hasil_validasi_per_fold.csv", "hasil_perbandingan.csv"])
    return (read_csv(path), path) if path else (None, None)


def load_time_data():
    path = first_existing(RESULT_DIR, ["seluruh_waktu_per_fold.csv", "waktu_validasi_per_fold.csv"])
    if path:
        return read_csv(path), path
    parts = []
    for seed in [21, 42, 2026]:
        candidate = RESULT_DIR / f"seed_{seed}" / "waktu_per_fold.csv"
        df = read_csv(candidate)
        if df is not None:
            parts.append(df)
    if parts:
        return pd.concat(parts, ignore_index=True), RESULT_DIR
    path = first_existing(BASE_DIR, ["seluruh_waktu_per_fold.csv", "waktu_validasi_per_fold.csv", "waktu_seleksi.csv"])
    return (read_csv(path), path) if path else (None, None)


def load_feature_data(fold_df=None):
    path = first_existing(RESULT_DIR, ["seluruh_fitur_per_fold.csv", "fitur_validasi_per_fold.csv"])
    if path:
        return read_csv(path), path
    parts = []
    for seed in [21, 42, 2026]:
        candidate = RESULT_DIR / f"seed_{seed}" / "fitur_per_fold.csv"
        df = read_csv(candidate)
        if df is not None:
            parts.append(df)
    if parts:
        return pd.concat(parts, ignore_index=True), RESULT_DIR
    if fold_df is not None and "Fitur_Terpilih" in fold_df.columns:
        return fold_df[[c for c in ["Seed", "Fold", "Metode", "Fitur_Terpilih"] if c in fold_df.columns]].copy(), None
    path = first_existing(BASE_DIR, ["seluruh_fitur_per_fold.csv", "fitur_validasi_per_fold.csv", "fitur_terpilih.csv"])
    return (read_csv(path), path) if path else (None, None)


def nice_method_order(values):
    values = list(values)
    return [m for m in METHOD_ORDER if m in values] + [m for m in values if m not in METHOD_ORDER]


def styled_bar(metric_df, metric, title, xlabel, percent=False, descending=False):
    """Create one clean horizontal bar chart with a separate figure."""
    if metric not in metric_df.columns:
        st.info(f"Metrik {metric} tidak tersedia pada CSV yang ditemukan.")
        return
    frame = metric_df[[metric]].dropna().copy()
    if frame.empty:
        st.info(f"Belum ada data valid untuk {title.lower()}.")
        return
    if descending:
        frame = frame.sort_values(metric, ascending=False)
    else:
        frame = frame.reindex([m for m in METHOD_ORDER if m in frame.index] + [m for m in frame.index if m not in METHOD_ORDER])
    fig, ax = plt.subplots(figsize=(7.3, 3.25), dpi=140)
    labels = list(frame.index)
    values = frame[metric].astype(float).to_numpy()
    if percent:
        values = values * 100
    colors = [METHOD_COLORS.get(label, "#64748b") for label in labels]
    bars = ax.barh(labels, values, color=colors, height=0.58)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color="#14233a", pad=14)
    ax.set_xlabel(xlabel, color="#64748b", fontsize=9)
    ax.tick_params(axis="both", labelsize=9, colors="#334155", length=0)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    max_val = max(values) if len(values) else 1
    ax.set_xlim(0, max_val * 1.18 if max_val > 0 else 1)
    for bar, value in zip(bars, values):
        label = f"{value:.2f}%" if percent else (f"{value:.2f}" if metric == "Jumlah_Fitur" else f"{value:.3f} detik")
        ax.text(value + max_val * 0.02, bar.get_y() + bar.get_height()/2, label,
                va="center", ha="left", fontsize=9, color="#14233a", fontweight="bold")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.tight_layout(pad=1.25)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def card(label, value, help_text=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-help">{help_text}</div></div>',
        unsafe_allow_html=True,
    )


# --------------------------- Load data ---------------------------
fold_df, fold_path = load_per_fold()
time_df, time_path = load_time_data()
feature_df, feature_path = load_feature_data(fold_df)

if fold_df is None or fold_df.empty:
    st.markdown(
        """
        <div class="hero"><div class="hero-kicker">Machine Learning Research</div>
        <h1>Dashboard Feature Selection</h1>
        <p>Visualisasi evaluasi klasifikasi kanker payudara menggunakan Logistic Regression.</p></div>
        """, unsafe_allow_html=True)
    st.error("File hasil eksperimen belum ditemukan atau tidak dapat dibaca.")
    st.markdown("Pastikan file hasil terbaru berada di folder `hasil_pengujian`.")
    st.code(str(RESULT_DIR))
    st.markdown("Jalankan `python .\\validasi_penelitian_final.py` terlebih dahulu, kemudian muat ulang dashboard.")
    st.stop()

# Normalize names and numeric columns.
fold_df.columns = [str(c).strip() for c in fold_df.columns]
method_col = find_col(fold_df, ["Metode", "Method", "Model"])
if method_col is None:
    st.error("Kolom 'Metode' tidak ditemukan pada file hasil.")
    st.write("Kolom yang tersedia:", list(fold_df.columns))
    st.stop()
fold_df[method_col] = fold_df[method_col].map(standardize_method)

metric_map = {
    "Accuracy": find_col(fold_df, ["Accuracy", "Accuracy_mean", "Akurasi"]),
    "Precision_Malignant": find_col(fold_df, ["Precision_Malignant", "Precision_Malignant_mean"]),
    "Recall_Malignant": find_col(fold_df, ["Recall_Malignant", "Recall_Malignant_mean"]),
    "F1_Malignant": find_col(fold_df, ["F1_Malignant", "F1_Malignant_mean", "F1-score"]),
    "Jumlah_Fitur": find_col(fold_df, ["Jumlah_Fitur", "Jumlah_Fitur_mean", "Fitur"]),
}
if metric_map["Accuracy"] is None:
    st.error("Kolom Accuracy tidak ditemukan pada file hasil.")
    st.write("Kolom yang tersedia:", list(fold_df.columns))
    st.stop()

# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.markdown("## 🧬 Menu Penelitian")
    st.caption("Feature Selection · Logistic Regression")
    st.markdown("---")
    seed_col = find_col(fold_df, ["Seed", "Random_Seed"])
    all_seeds = sorted(pd.to_numeric(fold_df[seed_col], errors="coerce").dropna().unique().tolist()) if seed_col else []
    selected_seeds = st.multiselect("Pilih random seed", all_seeds, default=all_seeds) if all_seeds else []
    all_methods = nice_method_order(fold_df[method_col].dropna().unique().tolist())
    selected_methods = st.multiselect("Pilih metode", all_methods, default=all_methods)
    st.markdown("---")
    st.markdown("**Dataset**")
    st.write("569 sampel · 30 fitur")
    st.markdown("**Validasi**")
    st.write("Stratified 5-Fold CV · 3 seed")
    st.markdown("---")
    st.caption("Dashboard hanya membaca hasil CSV. Eksperimen tidak dijalankan ulang di sini.")

# Filter folds for current view.
view_df = fold_df.copy()
if seed_col and selected_seeds:
    view_df = view_df[pd.to_numeric(view_df[seed_col], errors="coerce").isin(selected_seeds)]
if not selected_methods:
    st.warning("Pilih setidaknya satu metode pada sidebar.")
    st.stop()
view_df = view_df[view_df[method_col].isin(selected_methods)]

# Mean metrics from actual outer-fold rows.
metric_agg_cols = {k: v for k, v in metric_map.items() if v is not None}
for c in metric_agg_cols.values():
    view_df[c] = pd.to_numeric(view_df[c], errors="coerce")
agg = view_df.groupby(method_col).agg({c: ["mean", "std"] for c in metric_agg_cols.values()})
agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
metrics = pd.DataFrame(index=agg.index)
for canonical, raw_col in metric_agg_cols.items():
    metrics[canonical] = agg.get(f"{raw_col}_mean")
    metrics[f"{canonical}_std"] = agg.get(f"{raw_col}_std")
metrics = metrics.reindex(nice_method_order(metrics.index))
metrics.index.name = "Metode"

# Average selection time using fold-level time CSV.
if time_df is not None and not time_df.empty:
    time_method_col = find_col(time_df, ["Metode", "Method", "Model"])
    time_value_col = find_col(time_df, ["Waktu_Detik", "Waktu_Rata_Rata_Detik", "Waktu_Seleksi_Detik"])
    if time_method_col and time_value_col:
        time_df[time_method_col] = time_df[time_method_col].map(standardize_method)
        time_df[time_value_col] = pd.to_numeric(time_df[time_value_col], errors="coerce")
        if seed_col:
            time_seed_col = find_col(time_df, ["Seed", "Random_Seed"])
            if time_seed_col and selected_seeds:
                time_df[time_seed_col] = pd.to_numeric(time_df[time_seed_col], errors="coerce")
                time_view = time_df[time_df[time_seed_col].isin(selected_seeds)]
            else:
                time_view = time_df
        else:
            time_view = time_df
        time_view = time_view[time_view[time_method_col].isin(selected_methods)]
        time_avg = time_view.groupby(time_method_col)[time_value_col].mean()
        metrics["Waktu_Seleksi_Detik"] = time_avg.reindex(metrics.index)

# --------------------------- Header ---------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Presentasi Penelitian · Machine Learning</div>
      <h1>Analisis Feature Selection<br>pada Klasifikasi Kanker Payudara</h1>
             <P>Nama : Afifah Zubdatul Ishlah-072926011 </p>
            <p>Nama : Haykal Saputra Joban-072926031 </p>
            <p>Nama : Muhammad Vicky Wahyuda-072926007 </p>
            <p>Nama : Afrizal Subagja-072926002 </p>
            <p>Dosen : Dr.FAJAR DELLI WIHARTIKO, S.Si, MM, M.Kom </p>
            <p>Perbandingan Baseline, Sequential Forward Selection (SFS), Recursive Feature Elimination (RFE), dan Genetic Algorithm (GA) dengan Logistic Regression.</p>
      <span class="hero-chip">Breast Cancer Wisconsin (Diagnostic)</span>
      <span class="hero-chip">569 sampel · 30 fitur</span>
      <span class="hero-chip">3 seed × 5 fold</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-head">Ringkasan eksekutif</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Ringkasan hasil berdasarkan fold yang dipilih pada panel kiri.</div>', unsafe_allow_html=True)

# Four KPI cards
k1, k2, k3, k4 = st.columns(4)
if "Accuracy" in metrics and metrics["Accuracy"].notna().any():
    best_acc = metrics["Accuracy"].idxmax()
    with k1:
        card("ACCURACY TERTINGGI", f"{metrics.loc[best_acc, 'Accuracy']*100:.2f}%", f"{best_acc} · rata-rata outer fold")
if "Recall_Malignant" in metrics and metrics["Recall_Malignant"].notna().any():
    best_recall = metrics["Recall_Malignant"].idxmax()
    with k2:
        card("RECALL MALIGNANT TERTINGGI", f"{metrics.loc[best_recall, 'Recall_Malignant']*100:.2f}%", f"{best_recall} · kelas positif")
if "Jumlah_Fitur" in metrics and metrics["Jumlah_Fitur"].notna().any():
    fewest_method = metrics["Jumlah_Fitur"].idxmin()
    with k3:
        card("FITUR PALING SEDIKIT", f"{metrics.loc[fewest_method, 'Jumlah_Fitur']:.2f}", f"{fewest_method} · rata-rata fitur")
if "Waktu_Seleksi_Detik" in metrics and metrics["Waktu_Seleksi_Detik"].notna().any():
    times = metrics["Waktu_Seleksi_Detik"].dropna()
    if not times.empty:
        fastest = times.idxmin()
        with k4:
            card("SELEKSI TERCEPAT", f"{times.loc[fastest]:.3f} dtk", fastest)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="note"><b>Catatan interpretasi:</b> hasil ini merupakan evaluasi akademik pada dataset publik. Akurasi tertinggi tidak otomatis berarti metode paling efisien; pertimbangkan juga jumlah fitur, recall kelas malignant, kestabilan fitur, dan waktu komputasi.</div>', unsafe_allow_html=True)

# --------------------------- Tabs ---------------------------
tab_summary, tab_features, tab_folds, tab_about = st.tabs([
    "📊 Ringkasan Model", "🧩 Analisis Fitur", "🔎 Detail Per Fold", "📘 Dataset & Metode"
])

with tab_summary:
    st.markdown('<div class="section-head">Perbandingan performa</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Nilai rata-rata dari outer fold yang dipilih. Simpangan baku ditampilkan sebagai informasi variasi antar-fold.</div>', unsafe_allow_html=True)

    display = pd.DataFrame(index=metrics.index)
    for canonical, label in [
        ("Accuracy", "Accuracy"), ("Precision_Malignant", "Precision malignant"),
        ("Recall_Malignant", "Recall malignant"), ("F1_Malignant", "F1-score malignant")
    ]:
        if canonical in metrics:
            std_col = f"{canonical}_std"
            if std_col in metrics:
                display[label] = [
                    f"{row[canonical]*100:.2f}% ± {row[std_col]*100:.2f}%" if pd.notna(row[canonical]) and pd.notna(row[std_col]) else (f"{row[canonical]*100:.2f}%" if pd.notna(row[canonical]) else "—")
                    for _, row in metrics.iterrows()
                ]
            else:
                display[label] = metrics[canonical].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")
    if "Jumlah_Fitur" in metrics:
        display["Rata-rata fitur"] = metrics["Jumlah_Fitur"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    if "Waktu_Seleksi_Detik" in metrics:
        display["Waktu seleksi (detik)"] = metrics["Waktu_Seleksi_Detik"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    st.dataframe(display, width="stretch")

    left, right = st.columns(2)
    with left:
        styled_bar(metrics, "Accuracy", "Akurasi rata-rata", "Accuracy (%)", percent=True)
    with right:
        styled_bar(metrics, "Recall_Malignant", "Recall kelas malignant", "Recall (%)", percent=True)
    left2, right2 = st.columns(2)
    with left2:
        styled_bar(metrics, "Jumlah_Fitur", "Jumlah fitur rata-rata", "Jumlah fitur")
    with right2:
        styled_bar(metrics, "Waktu_Seleksi_Detik", "Waktu seleksi fitur", "Waktu (detik)")

    st.markdown('<div class="section-head">hasil</div>', unsafe_allow_html=True)
    reading_items = []
    if "Accuracy" in metrics and metrics["Accuracy"].notna().any():
        best = metrics["Accuracy"].idxmax()
        reading_items.append(
            f"<p class='result-reading'>📈 <strong>Accuracy tertinggi:</strong> "
            f"{best} ({metrics.loc[best, 'Accuracy']*100:.2f}%).</p>"
        )
    if "Jumlah_Fitur" in metrics and metrics["Jumlah_Fitur"].notna().any():
        minimum_features = metrics["Jumlah_Fitur"].min()
        fewest_methods = [str(m) for m in metrics.index
                          if pd.notna(metrics.loc[m, "Jumlah_Fitur"])
                          and np.isclose(metrics.loc[m, "Jumlah_Fitur"], minimum_features)]
        fewest_text = " dan ".join(fewest_methods)
        reading_items.append(
            f"<p class='result-reading'>🧩 <strong>Rata-rata jumlah fitur paling sedikit:</strong> "
            f"{fewest_text} ({minimum_features:.2f} fitur).</p>"
        )
    if "Waktu_Seleksi_Detik" in metrics and metrics["Waktu_Seleksi_Detik"].notna().any():
        valid_time = metrics["Waktu_Seleksi_Detik"].dropna()
        if len(valid_time):
            fastest = valid_time.idxmin()
            reading_items.append(
                f"<p class='result-reading'>⚡ <strong>Seleksi fitur tercepat:</strong> "
                f"{fastest} ({valid_time.loc[fastest]:.3f} detik). Baseline tidak memiliki waktu seleksi "
                f"karena tidak memilih subset fitur.</p>"
            )
    for item in reading_items:
        st.markdown(item, unsafe_allow_html=True)

with tab_features:
    st.markdown('<div class="section-head">Kestabilan pemilihan fitur</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Frekuensi kemunculan fitur pada fold yang dipilih. Ini menunjukkan konsistensi algoritma, bukan kepentingan klinis.</div>', unsafe_allow_html=True)
    if feature_df is None or feature_df.empty:
        st.info("Data fitur terpilih belum ditemukan. Pastikan seluruh_fitur_per_fold.csv tersedia di folder hasil final.")
    else:
        f_method_col = find_col(feature_df, ["Metode", "Method", "Model"])
        f_feature_col = find_col(feature_df, ["Fitur_Terpilih", "Selected_Features", "Fitur"]) 
        f_seed_col = find_col(feature_df, ["Seed", "Random_Seed"])
        if f_method_col is None or f_feature_col is None:
            st.info("Format kolom fitur terpilih belum dikenali.")
        else:
            feature_view = feature_df.copy()
            feature_view[f_method_col] = feature_view[f_method_col].map(standardize_method)
            feature_methods = [m for m in selected_methods if m != "Baseline" and m in feature_view[f_method_col].unique()]
            chosen_feature_method = st.selectbox("Pilih metode seleksi", feature_methods or [m for m in ["SFS", "RFE", "GA"] if m in feature_view[f_method_col].unique()])
            if f_seed_col and selected_seeds:
                feature_view[f_seed_col] = pd.to_numeric(feature_view[f_seed_col], errors="coerce")
                feature_view = feature_view[feature_view[f_seed_col].isin(selected_seeds)]
            feature_view = feature_view[feature_view[f_method_col] == chosen_feature_method]
            counts = {}
            for value in feature_view[f_feature_col].dropna().astype(str):
                for feature in value.split(","):
                    feature = feature.strip()
                    if feature:
                        counts[feature] = counts.get(feature, 0) + 1
            if counts:
                total = len(feature_view)
                freq = pd.DataFrame({"Fitur": list(counts.keys()), "Frekuensi": list(counts.values())})
                freq["Persentase fold"] = freq["Frekuensi"] / max(total, 1) * 100
                freq = freq.sort_values(["Frekuensi", "Fitur"], ascending=[False, True]).head(20)
                st.caption(f"Metode {chosen_feature_method}: dihitung dari {total} hasil fold yang terpilih.")
                fig, ax = plt.subplots(figsize=(9, max(4, min(8, len(freq) * 0.34))), dpi=140)
                freq_plot = freq.sort_values("Frekuensi", ascending=True)
                ax.barh(freq_plot["Fitur"], freq_plot["Frekuensi"], color=METHOD_COLORS.get(chosen_feature_method, "#0F766E"), height=0.65)
                ax.set_title(f"20 fitur paling sering dipilih — {chosen_feature_method}", loc="left", fontsize=12, fontweight="bold", color="#14233a", pad=12)
                ax.set_xlabel("Frekuensi pemilihan", color="#64748b", fontsize=9)
                ax.tick_params(axis="both", labelsize=8, colors="#334155", length=0)
                ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.25)
                ax.set_axisbelow(True)
                for spine in ax.spines.values(): spine.set_visible(False)
                ax.set_xlim(0, max(freq_plot["Frekuensi"].max() * 1.18, 1))
                for i, value in enumerate(freq_plot["Frekuensi"]):
                    ax.text(value + 0.08, i, str(int(value)), va="center", fontsize=8, color="#14233a", fontweight="bold")
                fig.patch.set_facecolor("white")
                ax.set_facecolor("white")
                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)
                st.dataframe(freq.sort_values("Frekuensi", ascending=False), width="stretch", hide_index=True)
            else:
                st.info("Tidak ada daftar fitur yang dapat dihitung pada pilihan ini.")

with tab_folds:
    st.markdown('<div class="section-head">Detail evaluasi per fold</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Gunakan tabel ini untuk memeriksa variasi kinerja pada setiap seed dan fold.</div>', unsafe_allow_html=True)
    fold_view = view_df.copy()
    display_cols = []
    for candidates in [["Seed", "Random_Seed"], ["Fold", "Fold_No"], ["Metode", "Method"], ["Accuracy"], ["Precision_Malignant"], ["Recall_Malignant"], ["F1_Malignant"], ["Jumlah_Fitur"], ["Fitur_Terpilih"]]:
        col = find_col(fold_view, candidates)
        if col and col not in display_cols:
            display_cols.append(col)
    if display_cols:
        show_df = fold_view[display_cols].copy()
        for column in show_df.columns:
            canonical = norm(column)
            if canonical in {"accuracy", "precisionmalignant", "recallmalignant", "f1malignant"}:
                show_df[column] = pd.to_numeric(show_df[column], errors="coerce") * 100
        col_config = {}
        for c in show_df.columns:
            if norm(c) in {"accuracy", "precisionmalignant", "recallmalignant", "f1malignant"}:
                col_config[c] = st.column_config.NumberColumn(c + " (%)", format="%.2f%%")
        st.dataframe(show_df, width="stretch", hide_index=True, column_config=col_config, height=460)
        st.download_button("⬇️ Unduh data fold yang sedang ditampilkan", data=fold_view[display_cols].to_csv(index=False).encode("utf-8-sig"), file_name="detail_fold_terfilter.csv", mime="text/csv")

with tab_about:
    st.markdown('<div class="section-head">Dataset dan rancangan eksperimen</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="panel"><h4>Dataset</h4><p><b>Breast Cancer Wisconsin (Diagnostic)</b></p><ul><li>569 sampel</li><li>30 fitur numerik</li><li>212 malignant dan 357 benign</li><li>Target dashboard: malignant sebagai kelas positif</li></ul></div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="panel"><h4>Rancangan pengujian</h4><ul><li>Logistic Regression untuk seluruh metode</li><li>Stratified 5-Fold Cross-Validation</li><li>Random seed: 21, 42, dan 2026</li><li>Baseline, SFS, RFE, dan Genetic Algorithm</li></ul></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="note"><b>Batasan:</b> hasil ini adalah evaluasi akademik pada dataset publik. Angka kinerja tidak dimaksudkan sebagai bukti kesiapan penggunaan klinis. Perbandingan di dashboard bersifat deskriptif; uji statistik tambahan diperlukan untuk klaim perbedaan signifikan.</div>', unsafe_allow_html=True)
    with st.expander("Informasi sumber file yang dibaca"):
        st.write("Folder hasil final:", str(RESULT_DIR))
        st.write("Data per fold:", str(fold_path) if fold_path else "Tidak ditemukan")
        st.write("Data waktu:", str(time_path) if time_path else "Tidak ditemukan")
        st.write("Data fitur:", str(feature_path) if feature_path else "Tidak ditemukan")
        st.write("Kolom data fold:", list(fold_df.columns))
        st.write("Jumlah baris yang sedang dipakai:", len(view_df))

st.markdown('<div class="footer">Dashboard penelitian · Feature Selection dengan Logistic Regression · Hasil dibaca dari CSV eksperimen lokal</div>', unsafe_allow_html=True)
