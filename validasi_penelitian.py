import os
import time
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# =====================================================
# 1. PENGATURAN PENELITIAN
# =====================================================

SEEDS = [21, 42, 2026]
JUMLAH_FOLD = 5
JUMLAH_FITUR = 10

POPULASI_GA = 12
GENERASI_GA = 6
MUTASI_GA = 0.05

OUTPUT_FOLDER = "hasil_pengujian"


# =====================================================
# 2. MEMUAT DATASET
# =====================================================

data = load_breast_cancer(as_frame=True)
X = data.data

# Target: 1 = malignant, 0 = benign
y = (data.target == 0).astype(int)


# =====================================================
# 3. MODEL KLASIFIKASI
# =====================================================

def buat_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=5000,
            random_state=42
        )),
    ])


# =====================================================
# 4. EVALUASI MODEL PADA DATA UJI
# =====================================================

def evaluasi_model(X_train, X_test, y_train, y_test, fitur):
    model = buat_model()
    model.fit(X_train[fitur], y_train)
    prediksi = model.predict(X_test[fitur])

    return {
        "Accuracy": accuracy_score(y_test, prediksi),
        "Precision_Malignant": precision_score(
            y_test, prediksi, zero_division=0
        ),
        "Recall_Malignant": recall_score(
            y_test, prediksi, zero_division=0
        ),
        "F1_Malignant": f1_score(
            y_test, prediksi, zero_division=0
        ),
        "Jumlah_Fitur": len(fitur),
        "Fitur_Terpilih": ", ".join(fitur),
    }


# =====================================================
# 5. SFS - SEQUENTIAL FEATURE SELECTION
# =====================================================

def pilih_sfs(X_train, y_train, seed):
    cv_dalam = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=seed
    )

    selector = SequentialFeatureSelector(
        estimator=buat_model(),
        n_features_to_select=JUMLAH_FITUR,
        direction="forward",
        scoring="accuracy",
        cv=cv_dalam,
        n_jobs=-1,
    )

    selector.fit(X_train, y_train)
    return list(X_train.columns[selector.get_support()])


# =====================================================
# 6. RFE - STANDARDISASI DI DALAM ESTIMATOR
# =====================================================

def pilih_rfe(X_train, y_train):
    estimator = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=5000,
            random_state=42
        )),
    ])

    selector = RFE(
        estimator=estimator,
        n_features_to_select=JUMLAH_FITUR,
        step=1,
        importance_getter="named_steps.model.coef_",
    )

    selector.fit(X_train, y_train)
    return list(X_train.columns[selector.get_support()])


# =====================================================
# 7. GA - GENETIC ALGORITHM
# =====================================================

def pilih_ga(X_train, y_train, seed):
    rng = np.random.default_rng(seed)
    jumlah_total = X_train.shape[1]

    cv_dalam = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=seed
    )

    def nilai_kromosom(kromosom):
        indeks = np.flatnonzero(kromosom)

        if len(indeks) == 0:
            return 0.0

        fitur = X_train.columns[indeks]

        skor = cross_val_score(
            buat_model(),
            X_train[fitur],
            y_train,
            cv=cv_dalam,
            scoring="accuracy",
            n_jobs=-1,
        ).mean()

        penalti = 0.002 * len(indeks) / jumlah_total
        return skor - penalti

    populasi = []

    for _ in range(POPULASI_GA):
        kromosom = rng.integers(
            0, 2, size=jumlah_total
        ).astype(bool)

        if not kromosom.any():
            kromosom[rng.integers(jumlah_total)] = True

        populasi.append(kromosom)

    for _ in range(GENERASI_GA):
        nilai = [nilai_kromosom(k) for k in populasi]
        urutan = np.argsort(nilai)[::-1]

        populasi_baru = [
            populasi[urutan[0]].copy(),
            populasi[urutan[1]].copy(),
        ]

        while len(populasi_baru) < POPULASI_GA:
            kandidat = urutan[:min(4, len(urutan))]
            induk1 = populasi[rng.choice(kandidat)]
            induk2 = populasi[rng.choice(kandidat)]

            titik = rng.integers(1, jumlah_total)
            anak = np.concatenate([
                induk1[:titik],
                induk2[titik:],
            ]).copy()

            mutasi = rng.random(jumlah_total) < MUTASI_GA
            anak[mutasi] = ~anak[mutasi]

            if not anak.any():
                anak[rng.integers(jumlah_total)] = True

            populasi_baru.append(anak)

        populasi = populasi_baru

    nilai_akhir = [nilai_kromosom(k) for k in populasi]
    terbaik = populasi[int(np.argmax(nilai_akhir))]
    return list(X_train.columns[np.flatnonzero(terbaik)])


# =====================================================
# 8. MENJALANKAN PENGUJIAN 3 SEED X 5 FOLD
# =====================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

semua_hasil = []
semua_fitur = []
semua_waktu = []

for seed in SEEDS:
    print("\n" + "=" * 60)
    print(f"MENGUJI SEED: {seed}")
    print("=" * 60)

    folder_seed = os.path.join(OUTPUT_FOLDER, f"seed_{seed}")
    os.makedirs(folder_seed, exist_ok=True)

    cv_luar = StratifiedKFold(
        n_splits=JUMLAH_FOLD,
        shuffle=True,
        random_state=seed,
    )

    for fold, (idx_train, idx_test) in enumerate(
        cv_luar.split(X, y), start=1
    ):
        X_train = X.iloc[idx_train]
        X_test = X.iloc[idx_test]
        y_train = y.iloc[idx_train]
        y_test = y.iloc[idx_test]

        print(f"\nSeed {seed} | Fold {fold}")

        daftar_metode = {"Baseline": list(X.columns)}
        waktu_metode = {"Baseline": np.nan}

        mulai = time.perf_counter()
        daftar_metode["SFS"] = pilih_sfs(
            X_train, y_train, seed * 100 + fold
        )
        waktu_metode["SFS"] = time.perf_counter() - mulai

        mulai = time.perf_counter()
        daftar_metode["RFE"] = pilih_rfe(X_train, y_train)
        waktu_metode["RFE"] = time.perf_counter() - mulai

        mulai = time.perf_counter()
        daftar_metode["GA"] = pilih_ga(
            X_train, y_train, seed * 100 + fold
        )
        waktu_metode["GA"] = time.perf_counter() - mulai

        for metode, fitur in daftar_metode.items():
            metrik = evaluasi_model(
                X_train, X_test, y_train, y_test, fitur
            )

            semua_hasil.append({
                "Seed": seed,
                "Fold": fold,
                "Metode": metode,
                **metrik,
            })

            semua_fitur.append({
                "Seed": seed,
                "Fold": fold,
                "Metode": metode,
                "Fitur_Terpilih": ", ".join(fitur),
            })

            semua_waktu.append({
                "Seed": seed,
                "Fold": fold,
                "Metode": metode,
                "Waktu_Detik": waktu_metode[metode],
            })

            print(
                f"{metode:10} | "
                f"Accuracy: {metrik['Accuracy']:.4f} | "
                f"Recall malignant: {metrik['Recall_Malignant']:.4f} | "
                f"Fitur: {metrik['Jumlah_Fitur']}"
            )

    pd.DataFrame([
        r for r in semua_hasil if r["Seed"] == seed
    ]).to_csv(
        os.path.join(folder_seed, "hasil_per_fold.csv"),
        index=False,
    )

    pd.DataFrame([
        r for r in semua_fitur if r["Seed"] == seed
    ]).to_csv(
        os.path.join(folder_seed, "fitur_per_fold.csv"),
        index=False,
    )

    pd.DataFrame([
        r for r in semua_waktu if r["Seed"] == seed
    ]).to_csv(
        os.path.join(folder_seed, "waktu_per_fold.csv"),
        index=False,
    )


# =====================================================
# 9. MEMBUAT RINGKASAN HASIL
# =====================================================

df_hasil = pd.DataFrame(semua_hasil)
df_waktu = pd.DataFrame(semua_waktu)

kolom_metrik = [
    "Accuracy",
    "Precision_Malignant",
    "Recall_Malignant",
    "F1_Malignant",
    "Jumlah_Fitur",
]

ringkasan = (
    df_hasil.groupby(["Seed", "Metode"])[kolom_metrik]
    .agg(["mean", "std"])
    .round(4)
)

ringkasan.columns = [
    f"{metrik}_{statistik}"
    for metrik, statistik in ringkasan.columns
]
ringkasan = ringkasan.reset_index()

waktu_ringkas = (
    df_waktu.groupby(["Seed", "Metode"])["Waktu_Detik"]
    .mean()
    .reset_index(name="Waktu_Rata_Rata_Detik")
    .round(4)
)

ringkasan = ringkasan.merge(
    waktu_ringkas,
    on=["Seed", "Metode"],
    how="left",
)

ringkasan.to_csv(
    os.path.join(OUTPUT_FOLDER, "ringkasan_semua_seed.csv"),
    index=False,
)

df_hasil.to_csv(
    os.path.join(OUTPUT_FOLDER, "seluruh_hasil_per_fold.csv"),
    index=False,
)

df_waktu.to_csv(
    os.path.join(OUTPUT_FOLDER, "seluruh_waktu_per_fold.csv"),
    index=False,
)

pd.DataFrame(semua_fitur).to_csv(
    os.path.join(OUTPUT_FOLDER, "seluruh_fitur_per_fold.csv"),
    index=False,
)

print("\n" + "=" * 60)
print("PENGUJIAN SELESAI")
print(f"Semua hasil tersimpan di folder: {OUTPUT_FOLDER}")
print("Ringkasan: ringkasan_semua_seed.csv")
print("=" * 60)
