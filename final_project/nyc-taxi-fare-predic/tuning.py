# =============================================================================
#  NYC Taxi Fare Prediction — Phase 4
#  Hyperparameter Tuning
# =============================================================================
#  วิธีใช้:
#   1. รัน phase1 + phase2 + phase3 ก่อน (สร้าง X_train.npy, y_train.npy)
#   2. รัน: python phase4_tuning.py
#
#  Strategy:
#   - ทุกโมเดล tune ด้วย RandomizedSearchCV (เร็วกว่า GridSearch)
#   - best model จาก Phase 3 จะได้รับ search space กว้างกว่า
#   - เปรียบ Before vs After tuning ด้วย RMSE
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
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from sklearn.metrics         import mean_squared_error
import xgboost as xgb
from scipy.stats             import uniform, randint, loguniform

RANDOM_STATE = 42
N_FOLDS      = 5
N_ITER       = 10     # จำนวน random combinations ต่อโมเดล

# ─────────────────────────────────────────────────────────────────────────────
#  1. LOAD
# ─────────────────────────────────────────────────────────────────────────────
def load_arrays():
    X = np.load("X_train.npy")
    y = np.load("y_train.npy")
    best = open("best_model.txt").read().strip()
    print(f"  X_train: {X.shape}  |  y_train: {y.shape}")
    print(f"  Best model from Phase 3: [{best}]")
    return X, y, best


# ─────────────────────────────────────────────────────────────────────────────
#  2. BASELINE CV RMSE (before tuning)
# ─────────────────────────────────────────────────────────────────────────────
def baseline_rmse(models_default, X, y):
    """วัด CV RMSE ของ default parameters ก่อน tune"""
    print(f"\n{'='*65}")
    print("  BASELINE (default params) — before tuning")
    print(f"{'='*65}")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    baselines = {}
    for name, model in models_default.items():
        scores = -cross_val_score(model, X, y, cv=kf,
                                  scoring="neg_root_mean_squared_error",
                                  n_jobs=1)
        baselines[name] = scores.mean()
        print(f"  {name:<22} RMSE = ${scores.mean():.4f} ± {scores.std():.4f}")
    return baselines


# ─────────────────────────────────────────────────────────────────────────────
#  3. SEARCH SPACES
# ─────────────────────────────────────────────────────────────────────────────
def get_search_configs():
    """
    คืน dict ของ (model, param_distributions) สำหรับแต่ละโมเดล
    ใช้ loguniform สำหรับ alpha/C (span หลาย order of magnitude)
    ใช้ randint สำหรับ integer params
    """
    return {
        "Ridge": (
            Ridge(),
            {"alpha": loguniform(1e-3, 1e3)},   # 0.001 ถึง 1000
        ),
        "LASSO": (
            Lasso(max_iter=5000),
            {"alpha": loguniform(1e-4, 10)},    # 0.0001 ถึง 10
        ),
        "KNN": (
            KNeighborsRegressor(n_jobs=-1),
            {
                "n_neighbors": randint(3, 30),
                "weights"    : ["uniform", "distance"],
                "p"          : [1, 2],           # 1=Manhattan, 2=Euclidean
            },
        ),
        "SVM (Linear)": (
            Pipeline([("svm", LinearSVR(max_iter=3000,
                                        random_state=RANDOM_STATE))]),
            {
                "svm__C"      : loguniform(0.01, 100),
                "svm__epsilon": uniform(0.01, 2.0),
            },
        ),
        "Decision Tree": (
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            {
                "max_depth"       : randint(3, 20),
                "min_samples_leaf": randint(1, 50),
                "min_samples_split": randint(2, 30),
                "max_features"    : ["sqrt", "log2", None],
            },
        ),
        "Random Forest": (
            RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE),
            {
                "n_estimators"    : randint(50, 150),
                "max_depth"       : randint(5, 15),
                "min_samples_leaf": randint(5, 20),
                "max_features"    : ["sqrt", "log2"],
            },
        ),
        "XGBoost": (
            xgb.XGBRegressor(verbosity=0, random_state=RANDOM_STATE,
                             n_estimators=200),
            {
                "learning_rate"    : loguniform(0.01, 0.3),
                "max_depth"        : randint(3, 8),
                "subsample"        : uniform(0.6, 0.4),
                "colsample_bytree" : uniform(0.6, 0.4),
                "reg_alpha"        : loguniform(1e-3, 10),
                "reg_lambda"       : loguniform(1e-3, 10),
            },
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  4. RUN TUNING
# ─────────────────────────────────────────────────────────────────────────────
def run_tuning(search_configs, X, y, n_iter=N_ITER):
    print(f"\n{'='*65}")
    print(f"  RANDOMIZED SEARCH CV  (n_iter={n_iter} per model)")
    print(f"{'='*65}")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    tuned_results = {}

    for name, (model, param_dist) in search_configs.items():
        print(f"\n  ── {name} ──")
        t0 = time.time()

        search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=kf,
            scoring="neg_root_mean_squared_error",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        search.fit(X, y)

        best_rmse = -search.best_score_
        elapsed   = time.time() - t0

        # เก็บผลทุก iteration เพื่อ plot learning curve
        cv_results = search.cv_results_
        iter_rmse  = -cv_results["mean_test_score"]

        tuned_results[name] = {
            "best_model" : search.best_estimator_,
            "best_params": search.best_params_,
            "best_rmse"  : best_rmse,
            "iter_rmse"  : iter_rmse,
            "time_s"     : elapsed,
        }

        print(f"     Best RMSE : ${best_rmse:.4f}  ({elapsed:.1f}s)")
        # แสดง top params
        for k, v in search.best_params_.items():
            short_k = k.replace("svm__", "")
            print(f"     {short_k:<22} = {v}")

    return tuned_results


# ─────────────────────────────────────────────────────────────────────────────
#  5. SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(baselines, tuned_results):
    print(f"\n{'='*75}")
    print("  TUNING RESULTS — Before vs After")
    print(f"{'='*75}")
    print(f"  {'Model':<22} {'Before':>10} {'After':>10} {'Improve':>10} {'%':>7}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*7}")

    rows = []
    for name, res in tuned_results.items():
        before  = baselines.get(name, float("nan"))
        after   = res["best_rmse"]
        improve = before - after
        pct     = improve / before * 100 if before else 0
        rows.append((name, before, after, improve, pct))

    rows.sort(key=lambda x: x[2])   # sort by after RMSE
    medals = {0: "🥇", 1: "🥈", 2: "🥉"}
    for i, (name, before, after, improve, pct) in enumerate(rows):
        m = medals.get(i, "  ")
        sign = "+" if improve < 0 else "-"
        print(f"  {m}{name:<20} ${before:>9.4f} ${after:>9.4f} "
              f"  {'-' if improve >= 0 else '+'}{abs(improve):>8.4f} "
              f"{pct:>6.2f}%")

    best_name  = rows[0][0]
    best_after = rows[0][2]
    print(f"\n  ✅ Best tuned model : {best_name}  (CV RMSE = ${best_after:.4f})")
    return best_name, rows


# ─────────────────────────────────────────────────────────────────────────────
#  6. PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_tuning(baselines, tuned_results, rows, save_path="phase4_tuning.png"):
    print(f"\n  [Plots] → {save_path}")

    names_sorted = [r[0] for r in rows]
    before_vals  = [r[1] for r in rows]
    after_vals   = [r[2] for r in rows]
    improve_pct  = [r[4] for r in rows]

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Phase 4 — Hyperparameter Tuning Results",
                 fontsize=16, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── [1] Before vs After RMSE ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(len(names_sorted))
    w = 0.38
    ax1.bar(x - w/2, before_vals, w, label="Before tuning",
            color="#B0BEC5", edgecolor="white", alpha=0.9)
    ax1.bar(x + w/2, after_vals,  w, label="After tuning",
            color="#4A90D9", edgecolor="white", alpha=0.9)
    # annotate improvement
    for i, (b, a) in enumerate(zip(before_vals, after_vals)):
        diff = b - a
        color = "#27AE60" if diff > 0 else "#E74C3C"
        symbol = f"▼${abs(diff):.3f}" if diff > 0 else f"▲${abs(diff):.3f}"
        ax1.text(i, max(b, a) + 0.05, symbol, ha="center",
                 fontsize=8.5, color=color, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names_sorted, rotation=25, ha="right", fontsize=10)
    ax1.set_ylabel("CV RMSE ($)")
    ax1.set_title("Before vs After Tuning  (▼ = improvement)", fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, max(before_vals) * 1.18)

    # ── [2] Improvement % bar ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    colors_imp = ["#27AE60" if p > 0 else "#E74C3C" for p in improve_pct]
    ax2.barh(names_sorted, improve_pct, color=colors_imp,
             edgecolor="white", height=0.65)
    ax2.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("RMSE improvement (%)")
    ax2.set_title("Improvement %\n(เขียว = ดีขึ้น, แดง = แย่ลง)", fontweight="bold")
    for i, v in enumerate(improve_pct):
        ax2.text(v + (0.05 if v >= 0 else -0.05), i,
                 f"{v:.2f}%", va="center", ha="left" if v >= 0 else "right",
                 fontsize=8.5)
    ax2.invert_yaxis()

    # ── [3–5] Random search convergence ─────────────────────────────────────
    search_colors = ["#4A90D9","#E74C3C","#27AE60","#9B59B6",
                     "#E67E22","#1ABC9C","#F39C12"]
    ax3 = fig.add_subplot(gs[1, :2])
    for i, (name, res) in enumerate(tuned_results.items()):
        iter_rmse = res["iter_rmse"]
        # running minimum
        running_best = np.minimum.accumulate(iter_rmse)
        ax3.plot(range(1, len(running_best) + 1), running_best,
                 marker="o", markersize=3, linewidth=1.8,
                 color=search_colors[i % len(search_colors)],
                 label=name, alpha=0.85)
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Best RMSE so far ($)")
    ax3.set_title("Random Search Convergence\n(running best per iteration)",
                  fontweight="bold")
    ax3.legend(fontsize=8.5, ncol=2)

    # ── [6] Best params scorecard ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    best_name = names_sorted[0]
    best_res  = tuned_results[best_name]
    lines = [f"🏆  Best Tuned Model\n",
             f"  {best_name}\n",
             f"  CV RMSE = ${best_res['best_rmse']:.4f}\n\n",
             "  Optimal params:\n"]
    for k, v in best_res["best_params"].items():
        short_k = k.replace("svm__", "")
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        lines.append(f"  {short_k}: {val_str}\n")
    ax4.text(0.05, 0.95, "".join(lines),
             transform=ax4.transAxes, fontsize=9.5,
             va="top", ha="left", family="monospace",
             bbox=dict(boxstyle="round,pad=0.7", facecolor="#E8F5E9",
                       edgecolor="#27AE60", linewidth=2))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ บันทึก → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  7. EXPORT best tuned model predictions
# ─────────────────────────────────────────────────────────────────────────────
def export_predictions(tuned_results, best_name):
    X_test = np.load("X_test.npy")
    best_model = tuned_results[best_name]["best_model"]
    pred = best_model.predict(X_test)
    pred = np.maximum(pred, 2.5)
    np.save("best_tuned_predictions.npy", pred)
    print(f"\n  💾 Export: best_tuned_predictions.npy")
    print(f"     mean=${pred.mean():.2f}  min=${pred.min():.2f}  max=${pred.max():.2f}")
    return pred


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  PHASE 4 : HYPERPARAMETER TUNING")
    print("=" * 65)

    X, y, phase3_best = load_arrays()

    # default models สำหรับ baseline
    from sklearn.linear_model import Ridge, Lasso
    defaults = {
        "Ridge"         : Ridge(alpha=1.0),
        "LASSO"         : Lasso(alpha=0.01, max_iter=5000),
        "KNN"           : KNeighborsRegressor(n_neighbors=10,
                              weights="distance", n_jobs=-1),
        "SVM (Linear)"  : Pipeline([("svm", LinearSVR(C=1.0, epsilon=0.5,
                              max_iter=2000, random_state=RANDOM_STATE))]),
        "Decision Tree" : DecisionTreeRegressor(max_depth=10,
                              min_samples_leaf=10, random_state=RANDOM_STATE),
        "Random Forest" : RandomForestRegressor(n_estimators=100,
                              max_depth=12, min_samples_leaf=10,
                              n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost"       : xgb.XGBRegressor(n_estimators=300,
                              learning_rate=0.05, max_depth=6,
                              subsample=0.8, colsample_bytree=0.8,
                              verbosity=0, random_state=RANDOM_STATE),
    }

    # baseline
    baselines = baseline_rmse(defaults, X, y)

    # search configs
    search_configs = get_search_configs()

    # run tuning
    tuned_results = run_tuning(search_configs, X, y, n_iter=N_ITER)

    # summary
    best_name, rows = print_summary(baselines, tuned_results)

    # plots
    print(f"\n{'='*65}")
    print("  SAVING PLOTS")
    print(f"{'='*65}")
    plot_tuning(baselines, tuned_results, rows,
                save_path="phase4_tuning.png")

    # export
    pred = export_predictions(tuned_results, best_name)

    # save best model name for phase 5
    with open("best_tuned_model.txt", "w") as f:
        f.write(best_name)

    # save ranking CSV
    df_out = pd.DataFrame(
        [(r[0], r[1], r[2], r[3], r[4]) for r in rows],
        columns=["model", "before_rmse", "after_rmse", "improvement", "pct"]
    )
    df_out.to_csv("tuning_results.csv", index=False, float_format="%.4f")
    print(f"  ✅ บันทึก → tuning_results.csv")

    print(f"\n  ✅ Phase 4 เสร็จสมบูรณ์ — พร้อมไป Phase 5: Prediction & Submission")
    return tuned_results, best_name


if __name__ == "__main__":
    main()