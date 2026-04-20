# =============================================================================
#  NYC Taxi Fare Prediction — Phase 2
#  Model Training & Comparison (8 Algorithms)
# =============================================================================
#  วิธีใช้:
#   1. รัน phase1_taxi.py ก่อน เพื่อสร้าง X_train.npy, X_test.npy, y_train.npy
#   2. รัน: python phase2_modeling.py
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, time

warnings.filterwarnings("ignore")

from sklearn.linear_model    import LinearRegression, Ridge, Lasso
from sklearn.preprocessing   import PolynomialFeatures
from sklearn.pipeline        import Pipeline
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.svm             import LinearSVR          # เร็วกว่า SVR(RBF) มาก O(n) vs O(n²)
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics         import mean_squared_error
import xgboost as xgb

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
N_FOLDS    = 5       # K-Fold CV
RANDOM_STATE = 42

# ─────────────────────────────────────────────────────────────────────────────
#  1. LOAD DATA (จาก Phase 1)
# ─────────────────────────────────────────────────────────────────────────────
def load_arrays():
    print("=" * 65)
    print("  PHASE 2 : MODEL TRAINING & COMPARISON")
    print("=" * 65)
    X_train = np.load("X_train.npy")
    X_test  = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    print(f"\n  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  y_train : {y_train.shape}  "
          f"(mean=${y_train.mean():.2f}, std=${y_train.std():.2f})")
    return X_train, X_test, y_train


# ─────────────────────────────────────────────────────────────────────────────
#  2. DEFINE MODELS
# ─────────────────────────────────────────────────────────────────────────────
def get_models():
    """
    คืน dict ของ 8 โมเดล พร้อมชื่อย่อสำหรับแสดงผล
    Parameters ที่ใช้เป็น default ที่สมเหตุสมผล — จะ tune ใน Phase 4
    """
    models = {
        # ── Linear family ─────────────────────────────────────────────────────
        "Linear Reg."     : LinearRegression(),

        "Poly Reg. (d=2)" : Pipeline([
            ("poly",  PolynomialFeatures(degree=2, include_bias=False)),
            ("model", LinearRegression()),
        ]),

        "Ridge (α=1)"     : Ridge(alpha=1.0),

        "LASSO (α=0.01)"  : Lasso(alpha=0.01, max_iter=5000),

        # ── Instance-based ────────────────────────────────────────────────────
        "KNN (k=10)"      : KNeighborsRegressor(
                                n_neighbors=10,
                                weights="distance",   # ใกล้ = น้ำหนักมาก
                                n_jobs=-1,
                            ),

        # ── Kernel-based ──────────────────────────────────────────────────────
        # ใช้ LinearSVR แทน SVR(kernel='rbf') — เร็วกว่า 100x บน big data
        # LinearSVR = SVM แบบ linear kernel ซึ่งเหมาะกับ scaled features
        "SVM (Linear)"    : Pipeline([
                                ("svm", LinearSVR(
                                    C=1.0,
                                    epsilon=0.5,
                                    max_iter=2000,
                                    random_state=RANDOM_STATE,
                                )),
                            ]),

        # ── Tree-based ────────────────────────────────────────────────────────
        "Decision Tree"   : DecisionTreeRegressor(
                                max_depth=10,
                                min_samples_leaf=10,
                                random_state=RANDOM_STATE,
                            ),

        # Random Forest: ลด n_estimators=100 + n_jobs=-1 ให้รันคู่ขนาน
        "Random Forest"   : RandomForestRegressor(
                                n_estimators=100,
                                max_depth=12,
                                min_samples_leaf=10,
                                n_jobs=-1,
                                random_state=RANDOM_STATE,
                            ),

        "XGBoost"         : xgb.XGBRegressor(
                                n_estimators=300,
                                learning_rate=0.05,
                                max_depth=6,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                eval_metric="rmse",
                                random_state=RANDOM_STATE,
                                verbosity=0,
                            ),
    }
    return models


# ─────────────────────────────────────────────────────────────────────────────
#  3. CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def run_cv(models, X_train, y_train, n_folds=5):
    """
    ทำ K-Fold CV บนทุกโมเดล → เก็บ RMSE แต่ละ fold + mean + std
    """
    print(f"\n{'='*65}")
    print(f"  {n_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*65}")
    print(f"  {'Model':<20} {'Mean RMSE':>10} {'Std':>8} {'Time':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*8}")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, model in models.items():
        t0 = time.time()
        scores = cross_val_score(
            model, X_train, y_train,
            cv=kf,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1 if name not in ("SVM (RBF)", "Poly Reg. (d=2)") else 1,
        )
        rmse_scores = -scores
        elapsed = time.time() - t0

        results[name] = {
            "fold_rmse" : rmse_scores,
            "mean_rmse" : rmse_scores.mean(),
            "std_rmse"  : rmse_scores.std(),
            "time_s"    : elapsed,
        }
        flag = " ★" if rmse_scores.mean() == min(
            v["mean_rmse"] for v in results.values()) else ""
        print(f"  {name:<20} {rmse_scores.mean():>10.4f} "
              f"{rmse_scores.std():>8.4f} {elapsed:>7.1f}s{flag}")

    return results

# ─────────────────────────────────────────────────────────────────────────────
#  4. TRAIN FULL + PREDICT TEST
# ─────────────────────────────────────────────────────────────────────────────
def train_and_predict(models, X_train, y_train, X_test):
    """
    Train แต่ละโมเดลบน Train ทั้งหมด → Predict บน Test
    """
    print(f"\n{'='*65}")
    print("  TRAIN ON FULL TRAIN SET + PREDICT TEST")
    print(f"{'='*65}")
    predictions = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        pred = np.maximum(pred, 2.5)       # fare ต้องไม่ต่ำกว่า $2.50
        predictions[name] = pred
        print(f"  {name:<20}  predict mean=${pred.mean():.2f}  "
              f"min=${pred.min():.2f}  max=${pred.max():.2f}  "
              f"({time.time()-t0:.1f}s)")
    return predictions


# ─────────────────────────────────────────────────────────────────────────────
#  5. TRAIN RMSE (in-sample) — เพิ่มเติมสำหรับ compare
# ─────────────────────────────────────────────────────────────────────────────
def get_train_rmse(models, X_train, y_train):
    train_rmse = {}
    for name, model in models.items():
        pred = model.predict(X_train)
        rmse = np.sqrt(mean_squared_error(y_train, pred))
        train_rmse[name] = rmse
    return train_rmse


# ─────────────────────────────────────────────────────────────────────────────
#  6. PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(results, train_rmse, models, X_train, y_train,
                 save_path="model_comparison.png"):
    print(f"\n  [Plots] → {save_path}")

    names       = list(results.keys())
    cv_means    = [results[n]["mean_rmse"] for n in names]
    cv_stds     = [results[n]["std_rmse"]  for n in names]
    tr_rmse     = [train_rmse[n]           for n in names]
    times       = [results[n]["time_s"]    for n in names]
    best_idx    = int(np.argmin(cv_means))

    # color: best = gold, others = steelblue
    bar_colors  = ["#F5A623" if i == best_idx else "#4A90D9"
                   for i in range(len(names))]

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("NYC Taxi Fare — Model Comparison", fontsize=16,
                 fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── [1] CV RMSE bar chart ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    bars = ax1.barh(names, cv_means, xerr=cv_stds,
                    color=bar_colors, edgecolor="white",
                    capsize=4, height=0.6)
    ax1.set_xlabel("CV RMSE ($)  ← ยิ่งน้อยยิ่งดี", fontsize=11)
    ax1.set_title(f"{len(names[0].split())}-fold CV RMSE Comparison",
                  fontweight="bold")
    ax1.axvline(cv_means[best_idx], color="#F5A623",
                linestyle="--", linewidth=1.2, alpha=0.7)
    for i, (v, s) in enumerate(zip(cv_means, cv_stds)):
        ax1.text(v + s + 0.05, i, f"${v:.3f}", va="center", fontsize=9)
    ax1.invert_yaxis()

    # ── [2] Train vs CV RMSE (overfitting check) ──────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(names))
    w = 0.38
    ax2.bar(x - w/2, tr_rmse,  w, label="Train RMSE",  color="#5BA85F", alpha=0.85)
    ax2.bar(x + w/2, cv_means, w, label="CV RMSE",     color="#4A90D9", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([n.split("(")[0].strip() for n in names],
                        rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("RMSE ($)")
    ax2.set_title("Train vs CV RMSE\n(gap = overfitting)", fontweight="bold")
    ax2.legend(fontsize=9)

    # ── [3] CV RMSE fold distribution (boxplot) ───────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    fold_data = [results[n]["fold_rmse"] for n in names]
    bp = ax3.boxplot(fold_data, vert=False, patch_artist=True,
                     labels=names,
                     boxprops=dict(facecolor="#B0C4DE", alpha=0.7),
                     medianprops=dict(color="red", linewidth=1.5),
                     whiskerprops=dict(linewidth=0.8),
                     flierprops=dict(marker="o", markersize=4, alpha=0.5))
    ax3.set_xlabel("RMSE ($) per fold")
    ax3.set_title("RMSE Distribution Across Folds", fontweight="bold")
    ax3.axvline(cv_means[best_idx], color="#F5A623",
                linestyle="--", linewidth=1.2, alpha=0.7,
                label=f"Best: {names[best_idx]}")
    ax3.legend(fontsize=9)

    # ── [4] Training time ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    t_colors = ["#F5A623" if i == best_idx else "#90A4AE"
                for i in range(len(names))]
    ax4.barh(names, times, color=t_colors, edgecolor="white", height=0.6)
    ax4.set_xlabel("CV Time (seconds)")
    ax4.set_title("Training Time (CV)", fontweight="bold")
    for i, v in enumerate(times):
        ax4.text(v + 0.1, i, f"{v:.1f}s", va="center", fontsize=8)
    ax4.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ บันทึก → {save_path}")


def plot_feature_importance(models, feature_names,
                            save_path="feature_importance.png"):
    """Feature importance จาก Random Forest และ XGBoost"""
    print(f"  [Feature Importance] → {save_path}")

    fi_models = {n: m for n, m in models.items()
                 if hasattr(m, "feature_importances_")}
    if not fi_models:
        print("  ⚠️  ไม่มีโมเดลที่รองรับ feature importance")
        return

    ncols = len(fi_models)
    fig, axes = plt.subplots(1, ncols, figsize=(9 * ncols, 7))
    if ncols == 1:
        axes = [axes]
    fig.suptitle("Feature Importance", fontsize=14, fontweight="bold")

    for ax, (mname, model) in zip(axes, fi_models.items()):
        imp = model.feature_importances_
        idx = np.argsort(imp)
        top = idx[-15:]
        colors = ["#F5A623" if feature_names[i] == "distance_km"
                  else "#4A90D9" for i in top]
        ax.barh([feature_names[i] for i in top], imp[top],
                color=colors, edgecolor="white")
        ax.set_title(mname, fontweight="bold")
        ax.set_xlabel("Importance score")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ บันทึก → {save_path}")


def plot_pred_vs_actual(models, X_train, y_train,
                        save_path="pred_vs_actual.png"):
    """Scatter: Predicted vs Actual บน Train (สูงสุด 4 โมเดลที่มีอยู่)"""
    print(f"  [Pred vs Actual] → {save_path}")

    focus = list(models.keys())[:4]   # ใช้ 4 โมเดลแรกที่มีอยู่จริง
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Predicted vs Actual Fare (Train set)", fontsize=14,
                 fontweight="bold")

    for ax, name in zip(axes.flat, focus):
        pred = models[name].predict(X_train)
        lim = max(y_train.max(), pred.max()) * 1.05
        ax.scatter(y_train, pred, alpha=0.25, s=8, color="#4A90D9")
        ax.plot([0, lim], [0, lim], "r--", linewidth=1.2, label="Perfect fit")
        rmse = np.sqrt(mean_squared_error(y_train, pred))
        ax.set_title(f"{name}  (Train RMSE=${rmse:.3f})", fontweight="bold")
        ax.set_xlabel("Actual fare ($)")
        ax.set_ylabel("Predicted fare ($)")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.legend(fontsize=9)

    # ปิด axes ที่เหลือถ้าโมเดลน้อยกว่า 4
    for ax in axes.flat[len(focus):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ บันทึก → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  7. SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(results, train_rmse):
    print(f"\n{'='*65}")
    print("  FINAL SUMMARY TABLE")
    print(f"{'='*65}")
    print(f"  {'Rank':<5} {'Model':<20} {'CV RMSE':>10} {'±':>8} "
          f"{'Train RMSE':>12} {'Gap':>8}")
    print(f"  {'-'*5} {'-'*20} {'-'*10} {'-'*8} {'-'*12} {'-'*8}")

    rows = [(n, results[n]["mean_rmse"], results[n]["std_rmse"],
             train_rmse[n]) for n in results]
    rows.sort(key=lambda x: x[1])

    best_cv = rows[0][1]
    for rank, (name, cv_m, cv_s, tr_m) in enumerate(rows, 1):
        gap   = cv_m - tr_m
        medal = " 🥇" if rank == 1 else (" 🥈" if rank == 2 else
                (" 🥉" if rank == 3 else ""))
        print(f"  {rank:<5} {name:<20} {cv_m:>10.4f} {cv_s:>8.4f} "
              f"{tr_m:>12.4f} {gap:>8.4f}{medal}")

    best_name = rows[0][0]
    print(f"\n  ✅ Best model : {best_name}  (CV RMSE = ${best_cv:.4f})")
    print("  → ไป Phase 4: Hyperparameter Tuning บน best model ได้เลย")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "distance_km","delta_lat","delta_lon","abs_delta_lat","abs_delta_lon",
    "pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude",
    "passenger_count","hour","day_of_week","month","year",
    "is_weekend","is_rush_hour","is_night","is_jfk","is_lga","is_ewr",
]

def main():
    # 1. Load
    X_train, X_test, y_train = load_arrays()

    # 2. Define models
    models = get_models()

    # 3. Cross-validation
    results = run_cv(models, X_train, y_train, n_folds=N_FOLDS)

    # 4. Train full + predict test
    predictions = train_and_predict(models, X_train, y_train, X_test)

    # 5. Train RMSE (overfitting check)
    train_rmse = get_train_rmse(models, X_train, y_train)

    # 6. Plots
    print(f"\n{'='*65}")
    print("  SAVING PLOTS")
    print(f"{'='*65}")
    plot_results(results, train_rmse, models, X_train, y_train,
                 save_path="model_comparison.png")
    plot_feature_importance(models, FEATURE_NAMES,
                            save_path="feature_importance.png")
    plot_pred_vs_actual(models, X_train, y_train,
                        save_path="pred_vs_actual.png")

    # 7. Summary
    print_summary(results, train_rmse)

    # 8. Export best model predictions
    names    = list(results.keys())
    cv_means = [results[n]["mean_rmse"] for n in names]
    best_name = names[int(np.argmin(cv_means))]
    np.save("best_predictions.npy", predictions[best_name])
    print(f"\n  💾 Export: best_predictions.npy  (from {best_name})")

    return results, predictions, models, train_rmse


if __name__ == "__main__":
    main()