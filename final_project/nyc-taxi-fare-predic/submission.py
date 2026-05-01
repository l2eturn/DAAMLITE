# =============================================================================
#  NYC Taxi Fare Prediction — Phase 5
#  Prediction & Submission
# =============================================================================
#  วิธีใช้:
#   1. รัน phase1 → phase4 ก่อน
#   2. รัน: python phase5_submission.py
#
#  Output: submission.csv  (key, fare_amount) พร้อม submit
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, time

warnings.filterwarnings("ignore")

from sklearn.linear_model    import Ridge, Lasso
from sklearn.preprocessing   import PolynomialFeatures
from sklearn.pipeline        import Pipeline
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.svm             import LinearSVR
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics         import mean_squared_error
import xgboost as xgb
from scipy.stats             import loguniform, uniform, randint

RANDOM_STATE = 42
N_FOLDS      = 5

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: Feature Engineering (ต้องเหมือน Phase 1 ทุกประการ)
# ─────────────────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6_371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2)**2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2)**2)
    return R * 2 * np.arcsin(np.sqrt(a))


def engineer_features(df):
    df = df.copy()
    df["distance_km"]   = haversine_km(df["pickup_latitude"],  df["pickup_longitude"],
                                       df["dropoff_latitude"], df["dropoff_longitude"])
    df["delta_lat"]     = df["dropoff_latitude"]  - df["pickup_latitude"]
    df["delta_lon"]     = df["dropoff_longitude"] - df["pickup_longitude"]
    df["abs_delta_lat"] = df["delta_lat"].abs()
    df["abs_delta_lon"] = df["delta_lon"].abs()
    dt = df["pickup_datetime"]
    df["hour"]        = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["month"]       = dt.dt.month
    df["year"]        = dt.dt.year
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = (df["hour"].between(7,9) | df["hour"].between(16,19)).astype(int)
    df["is_night"]     = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    for name, (alat, alon) in {"jfk":(40.6413,-73.7781),
                                "lga":(40.7769,-73.8740),
                                "ewr":(40.6895,-74.1745)}.items():
        pu = haversine_km(df["pickup_latitude"],  df["pickup_longitude"],  alat, alon) < 2.0
        do = haversine_km(df["dropoff_latitude"], df["dropoff_longitude"], alat, alon) < 2.0
        df[f"is_{name}"] = (pu | do).astype(int)
    return df


NUMERIC_FEATURES = [
    "distance_km","delta_lat","delta_lon","abs_delta_lat","abs_delta_lon",
    "pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude",
    "passenger_count","hour","day_of_week","month","year",
]
BINARY_FEATURES  = ["is_weekend","is_rush_hour","is_night","is_jfk","is_lga","is_ewr"]
ALL_FEATURES     = NUMERIC_FEATURES + BINARY_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
#  1. LOAD ARRAYS + TEST RAW
# ─────────────────────────────────────────────────────────────────────────────
def load_all():
    print("=" * 65)
    print("  PHASE 5 : PREDICTION & SUBMISSION")
    print("=" * 65)

    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    X_test  = np.load("X_test.npy")
    best_name = open("best_tuned_model.txt").read().strip()

    test_raw = pd.read_csv(TEST_PATH, parse_dates=["pickup_datetime"])

    print(f"\n  X_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  test_raw: {test_raw.shape}")
    print(f"  Best model (Phase 4): [{best_name}]")
    return X_train, y_train, X_test, test_raw, best_name


# ─────────────────────────────────────────────────────────────────────────────
#  2. RE-TRAIN BEST MODEL ON FULL TRAIN  (with best params from phase 4)
# ─────────────────────────────────────────────────────────────────────────────
def get_best_tuned_model(best_name):
    """
    สร้างโมเดลพร้อม best params ที่ได้จาก Phase 4
    ในการใช้งานจริง ควรโหลด params จาก tuning_results แต่
    ที่นี่ hardcode ไว้เพื่อความชัดเจน — แก้ได้ตาม output ของ phase 4
    """
    # อ่าน best params จาก tuning CSV ถ้ามี
    try:
        df_tune = pd.read_csv("tuning_results.csv")
        row = df_tune[df_tune["model"] == best_name].iloc[0]
        print(f"\n  โหลด tuning results สำเร็จ — best after RMSE: ${row['after_rmse']:.4f}")
    except Exception:
        pass

    # map best_name → tuned model
    model_map = {
        "LASSO"         : Lasso(alpha=5.67, max_iter=10000),
        "Ridge"         : Ridge(alpha=506.16),
        "Random Forest" : RandomForestRegressor(
                              n_estimators=71, max_depth=5,
                              min_samples_leaf=14, max_features="log2",
                              n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost"       : xgb.XGBRegressor(
                              n_estimators=200, learning_rate=0.022,
                              max_depth=5, subsample=0.787,
                              colsample_bytree=0.990, reg_alpha=0.034,
                              reg_lambda=8.569, verbosity=0,
                              random_state=RANDOM_STATE),
        "Decision Tree" : DecisionTreeRegressor(
                              max_depth=3, max_features="log2",
                              min_samples_leaf=22, min_samples_split=13,
                              random_state=RANDOM_STATE),
        "KNN"           : KNeighborsRegressor(
                              n_neighbors=21, weights="uniform", p=1,
                              n_jobs=-1),
        "SVM (Linear)"  : Pipeline([("svm", LinearSVR(
                              C=0.315, epsilon=1.911,
                              max_iter=3000, random_state=RANDOM_STATE))]),
    }

    model = model_map.get(best_name)
    if model is None:
        print(f"  ⚠️  ไม่พบ '{best_name}' → fallback เป็น LASSO")
        model = Lasso(alpha=5.67, max_iter=10000)

    return model


# ─────────────────────────────────────────────────────────────────────────────
#  3. FINAL TRAIN + PREDICT
# ─────────────────────────────────────────────────────────────────────────────
def final_train_predict(model, X_train, y_train, X_test, best_name):
    print(f"\n{'='*65}")
    print(f"  FINAL TRAINING — {best_name} บน Train ทั้งหมด")
    print(f"{'='*65}")
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    pred_train = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, pred_train))

    pred_test = model.predict(X_test)
    pred_test = np.maximum(pred_test, 2.5)   # floor = base fare NYC

    print(f"  เวลา train   : {elapsed:.2f}s")
    print(f"  Train RMSE   : ${train_rmse:.4f}")
    print(f"\n  Test predictions:")
    print(f"  mean  = ${pred_test.mean():.2f}")
    print(f"  median= ${np.median(pred_test):.2f}")
    print(f"  min   = ${pred_test.min():.2f}")
    print(f"  max   = ${pred_test.max():.2f}")
    print(f"  std   = ${pred_test.std():.2f}")

    return pred_test, train_rmse


# ─────────────────────────────────────────────────────────────────────────────
#  4. BUILD SUBMISSION CSV
# ─────────────────────────────────────────────────────────────────────────────
def build_submission(test_raw, pred_test, save_path="submission.csv"):
    print(f"\n{'='*65}")
    print("  BUILDING SUBMISSION FILE")
    print(f"{'='*65}")

    submission = pd.DataFrame({
        "key"         : test_raw["key"],
        "fare_amount" : np.round(pred_test, 2),
    })

    submission.to_csv(save_path, index=False)
    print(f"  ✅ บันทึก → {save_path}")
    print(f"  จำนวนแถว : {len(submission):,}")
    print(f"\n  ตัวอย่าง 5 แถวแรก:")
    print(submission.head().to_string(index=False))
    return submission


# ─────────────────────────────────────────────────────────────────────────────
#  5. FINAL PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_final(y_train, pred_train, pred_test, best_name,
               save_path="phase5_final.png"):
    print(f"\n  [Plots] → {save_path}")

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"Phase 5 — Final Predictions  [{best_name}]",
                 fontsize=16, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # ── [1] Distribution ของ predictions บน Test ──────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(pred_test, bins=60, color="#4A90D9", edgecolor="white",
             linewidth=0.3, label="Test predictions")
    ax1.hist(y_train,   bins=60, color="#F5A623", edgecolor="white",
             linewidth=0.3, alpha=0.5, label="Train actual")
    ax1.set_xlabel("Fare ($)")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution: Train actual vs\nTest predictions",
                  fontweight="bold")
    ax1.legend(fontsize=9)

    # ── [2] Actual vs Predicted (Train) ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    lim = min(y_train.max(), 100)
    ax2.scatter(y_train[y_train <= lim], pred_train[y_train <= lim],
                alpha=0.2, s=6, color="#4A90D9")
    ax2.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect fit")
    rmse = np.sqrt(mean_squared_error(y_train, pred_train))
    ax2.set_title(f"Actual vs Predicted (Train)\nRMSE = ${rmse:.4f}",
                  fontweight="bold")
    ax2.set_xlabel("Actual fare ($)")
    ax2.set_ylabel("Predicted fare ($)")
    ax2.legend(fontsize=9)

    # ── [3] Residuals distribution ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    residuals = y_train - pred_train
    ax3.hist(residuals, bins=60, color="#9B59B6", edgecolor="white",
             linewidth=0.3)
    ax3.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax3.axvline(residuals.mean(), color="orange", linestyle="-",
                linewidth=1.2, label=f"Mean={residuals.mean():.2f}")
    ax3.set_xlabel("Residual ($)")
    ax3.set_ylabel("Count")
    ax3.set_title("Residual distribution (Train)\n← 0 คือสมบูรณ์แบบ",
                  fontweight="bold")
    ax3.legend(fontsize=9)

    # ── [4] Test prediction distribution (boxplot by fare range) ─────────
    ax4 = fig.add_subplot(gs[1, 0])
    bins   = [0, 10, 20, 35, 500]
    labels = ["<$10", "$10–20", "$20–35", ">$35"]
    pred_s = pd.Series(pred_test)
    groups = [pred_s[(pred_s >= bins[i]) & (pred_s < bins[i+1])].values
              for i in range(len(bins)-1)]
    colors_box = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    bp = ax4.boxplot(groups, patch_artist=True, labels=labels,
                     medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    counts = [len(g) for g in groups]
    for i, (c, n) in enumerate(zip(counts, labels), 1):
        ax4.text(i, ax4.get_ylim()[0], f"n={c:,}", ha="center",
                 va="bottom", fontsize=8, color="gray")
    ax4.set_ylabel("Predicted fare ($)")
    ax4.set_title("Test predictions by fare range",fontweight="bold")

    # ── [5] Cumulative distribution ───────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    sorted_pred = np.sort(pred_test)
    sorted_act  = np.sort(y_train)
    ax5.plot(sorted_pred, np.linspace(0, 1, len(sorted_pred)),
             color="#4A90D9", linewidth=2, label="Test predictions")
    ax5.plot(sorted_act,  np.linspace(0, 1, len(sorted_act)),
             color="#F5A623", linewidth=2, linestyle="--", label="Train actual")
    ax5.set_xlabel("Fare ($)")
    ax5.set_ylabel("Cumulative proportion")
    ax5.set_title("CDF: Train actual vs\nTest predictions",
                  fontweight="bold")
    ax5.set_xlim(0, 80)
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)

    # ── [6] Summary scorecard ─────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    fare_bins = pd.cut(pred_test, bins=[0,10,20,35,500],
                       labels=["<$10","$10-20","$20-35",">$35"])
    fare_dist = fare_bins.value_counts().sort_index()
    lines = [
        "📋  Submission Summary\n\n",
        f"  Model       :  {best_name}\n",
        f"  Train RMSE  :  ${np.sqrt(mean_squared_error(y_train, pred_train)):.4f}\n",
        f"  Test rows   :  {len(pred_test):,}\n\n",
        "  Fare distribution:\n",
    ]
    for label, cnt in fare_dist.items():
        pct = cnt / len(pred_test) * 100
        lines.append(f"  {label:<8}  {cnt:>5,}  ({pct:.1f}%)\n")
    lines += [
        f"\n  Mean   :  ${pred_test.mean():.2f}\n",
        f"  Median :  ${np.median(pred_test):.2f}\n",
        f"  Std    :  ${pred_test.std():.2f}\n",
    ]
    ax6.text(0.05, 0.95, "".join(lines),
             transform=ax6.transAxes, fontsize=10,
             va="top", ha="left", family="monospace",
             bbox=dict(boxstyle="round,pad=0.7", facecolor="#E3F2FD",
                       edgecolor="#1565C0", linewidth=2))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ บันทึก → {save_path}")

TRAIN_PATH = "../raw-data/train_taxi.csv"
TEST_PATH  = "../raw-data/test_taxi.csv"
# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # 1. Load
    X_train, y_train, X_test, test_raw, best_name = load_all()

    # 2. Get best tuned model
    model = get_best_tuned_model(best_name)

    # 3. Final train + predict
    pred_test, train_rmse = final_train_predict(
        model, X_train, y_train, X_test, best_name
    )
    pred_train = model.predict(X_train)

    # 4. Build submission
    submission = build_submission(test_raw, pred_test,
                                  save_path="submission.csv")

    # 5. Plots
    print(f"\n{'='*65}")
    print("  SAVING PLOTS")
    print(f"{'='*65}")
    plot_final(y_train, pred_train, pred_test, best_name,
               save_path="phase5_final.png")

    print(f"\n{'='*65}")
    print("  ✅ PIPELINE เสร็จสมบูรณ์ทุก Phase!")
    print(f"{'='*65}")
    print(f"  Phase 1 → phase1_taxi.py        (cleaning + feature eng)")
    print(f"  Phase 2 → phase2_modeling.py    (8 models + CV)")
    print(f"  Phase 3 → phase3_comparison.py  (comparison + selection)")
    print(f"  Phase 4 → phase4_tuning.py      (hyperparameter tuning)")
    print(f"  Phase 5 → phase5_submission.py  (final predict + submit)")
    print(f"\n  📁 submission.csv  ← ไฟล์นี้พร้อม submit แล้วครับ!")

    return submission


if __name__ == "__main__":
    main()