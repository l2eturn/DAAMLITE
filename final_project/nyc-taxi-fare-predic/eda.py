# =============================================================================
#  NYC Taxi Fare Prediction — Phase 1
#  Data Exploration, Cleaning & Feature Engineering
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
try:
    import folium
    from folium.plugins import HeatMap
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False
    print("pip install folium")

warnings.filterwarnings("ignore")

# ── Path config ───────────────────────────────────────────────────────────────
TRAIN_PATH = "../raw-data/train_taxi.csv"
TEST_PATH  = "../raw-data/test_taxi.csv"

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_data(train_path, test_path):
    print("=" * 65)
    print("  SECTION 1 : LOAD DATA")
    print("=" * 65)

    train = pd.read_csv(train_path, parse_dates=["pickup_datetime"])
    test  = pd.read_csv(test_path,  parse_dates=["pickup_datetime"])

    print(f"\n  Train shape : {train.shape}")
    print(f"  Test  shape : {test.shape}")

    print("\n── ตัวอย่าง 5 แถวแรก (Train) ──")
    print(train.head())

    print("\n── Data Types ──")
    print(train.dtypes)

    print("\n── Missing Values (Train) ──")
    mv = train.isnull().sum()
    mv_pct = (mv / len(train) * 100).round(2)
    print(pd.DataFrame({"count": mv, "pct(%)": mv_pct}))

    print("\n── สถิติเบื้องต้น (Train) ──")
    print(train.describe().round(4))

    return train, test


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────────────────────
def plot_eda(df, save_path="eda_plots.png"):
    """วาด 6 กราฟ EDA รวมในไฟล์เดียว"""
    print("\n" + "=" * 65)
    print("  SECTION 2 : EDA PLOTS")
    print("=" * 65)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("NYC Taxi Fare — Exploratory Data Analysis", fontsize=16, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # [1] Distribution ของ fare_amount
    ax1 = fig.add_subplot(gs[0, 0])
    sample = df["fare_amount"].dropna()
    sample = sample[(sample >= 0) & (sample <= 100)]   # ตัด extreme outlier เพื่อ plot
    ax1.hist(sample, bins=60, color="#4A90D9", edgecolor="white", linewidth=0.4)
    ax1.set_title("Distribution of fare_amount", fontweight="bold")
    ax1.set_xlabel("Fare ($)")
    ax1.set_ylabel("Count")
    ax1.axvline(sample.median(), color="red", linestyle="--", linewidth=1.2, label=f"Median ${sample.median():.2f}")
    ax1.legend(fontsize=9)

    # [2] Boxplot fare_amount (แบ่ง passenger_count)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_df = df[(df["fare_amount"].between(0, 80)) & (df["passenger_count"].between(1, 6))].copy()
    ax2.boxplot(
        [plot_df[plot_df["passenger_count"] == p]["fare_amount"] for p in range(1, 7)],
        labels=[str(p) for p in range(1, 7)],
        patch_artist=True,
        boxprops=dict(facecolor="#B0C4DE"),
        medianprops=dict(color="red", linewidth=1.5),
    )
    ax2.set_title("Fare by passenger count", fontweight="bold")
    ax2.set_xlabel("Passenger count")
    ax2.set_ylabel("Fare ($)")

    # [3] Pickup locations (Scatter)
    ax3 = fig.add_subplot(gs[0, 2])
    loc_df = df[
        df["pickup_longitude"].between(-74.1, -73.7) &
        df["pickup_latitude"].between(40.6, 40.9)
    ]
    sample_loc = loc_df.sample(min(5000, len(loc_df)), random_state=42)
    sc = ax3.scatter(
        sample_loc["pickup_longitude"], sample_loc["pickup_latitude"],
        c=sample_loc["fare_amount"].clip(0, 50),
        cmap="YlOrRd", s=1, alpha=0.4
    )
    plt.colorbar(sc, ax=ax3, label="Fare ($)")
    ax3.set_title("Pickup locations (color = fare)", fontweight="bold")
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")

    # [4] Fare by hour of day
    ax4 = fig.add_subplot(gs[1, 0])
    df_hour = df.copy()
    df_hour["hour"] = df_hour["pickup_datetime"].dt.hour
    hourly = df_hour[df_hour["fare_amount"].between(0, 80)].groupby("hour")["fare_amount"].median()
    ax4.bar(hourly.index, hourly.values, color="#5B9BD5", edgecolor="white")
    ax4.set_title("Median fare by hour of day", fontweight="bold")
    ax4.set_xlabel("Hour")
    ax4.set_ylabel("Median fare ($)")
    ax4.axhspan(0, hourly.min(), alpha=0.05, color="green")

    # [5] Fare by day of week
    ax5 = fig.add_subplot(gs[1, 1])
    df_hour["dow"] = df_hour["pickup_datetime"].dt.dayofweek
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow = df_hour[df_hour["fare_amount"].between(0, 80)].groupby("dow")["fare_amount"].median()
    colors = ["#FF7F7F" if d >= 5 else "#5B9BD5" for d in range(7)]
    ax5.bar(dow_labels, dow.values, color=colors, edgecolor="white")
    ax5.set_title("Median fare by day of week", fontweight="bold")
    ax5.set_xlabel("Day")
    ax5.set_ylabel("Median fare ($)")

    # [6] Fare trend by year
    ax6 = fig.add_subplot(gs[1, 2])
    df_hour["year"] = df_hour["pickup_datetime"].dt.year
    yearly = df_hour[df_hour["fare_amount"].between(0, 100)].groupby("year")["fare_amount"].median()
    ax6.plot(yearly.index, yearly.values, marker="o", color="#4A90D9", linewidth=2, markersize=6)
    ax6.fill_between(yearly.index, yearly.values, alpha=0.15, color="#4A90D9")
    ax6.set_title("Median fare by year", fontweight="bold")
    ax6.set_xlabel("Year")
    ax6.set_ylabel("Median fare ($)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ บันทึกกราฟ EDA → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────
def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("  SECTION 3 : DATA CLEANING (Train only)")
    print("=" * 65)
    n0 = len(df)

    # 3.1 ลบ missing values
    df = df.dropna()
    _report("ลบ NaN", n0, len(df))

    # 3.2 fare_amount: $2.50–$500
    #     ขั้นต่ำ NYC Taxi = $2.50 (base fare) / $500 คือ upper bound ที่สมเหตุสมผล
    n = len(df); df = df[(df["fare_amount"] >= 2.5) & (df["fare_amount"] <= 500)]
    _report("fare_amount ∈ [2.50, 500]", n, len(df))

    # 3.3 Coordinates: ครอบคลุม NYC + JFK + EWR + LGA
    LON = (-74.30, -72.90)
    LAT = ( 40.50,  41.80)
    n = len(df)
    df = df[
        df["pickup_longitude"].between(*LON) &
        df["pickup_latitude"].between(*LAT) &
        df["dropoff_longitude"].between(*LON) &
        df["dropoff_latitude"].between(*LAT)
    ]
    _report("กรอง coordinates (NYC bounds)", n, len(df))

    # 3.4 passenger_count: 1–6 (กฎหมาย NYC)
    n = len(df); df = df[df["passenger_count"].between(1, 6)]
    _report("passenger_count ∈ [1, 6]", n, len(df))

    total_removed = n0 - len(df)
    print(f"\n  📊 สรุป: จาก {n0:,} → เหลือ {len(df):,} แถว "
          f"(ลบออก {total_removed:,} = {total_removed/n0*100:.1f}%)")
    return df.reset_index(drop=True)


def _report(step, n_before, n_after):
    removed = n_before - n_after
    print(f"  [{step}] เหลือ {n_after:,} แถว  (ลบ {removed:,})")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Haversine distance (km) — แม่นยำกว่า Euclidean เพราะคำนึงถึงความโค้งโลก"""
    R = 6_371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


def near_airport(lat, lon, alat, alon, radius_km=2.0) -> pd.Series:
    return haversine_km(lat, lon, alat, alon) < radius_km


def engineer_features(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """
    สร้าง Feature ใหม่ — ใช้ฟังก์ชันนี้กับทั้ง Train และ Test
    เพื่อให้ได้ Feature เหมือนกันทุกประการ (ป้องกัน data leakage)
    """
    print(f"\n{'='*65}")
    print(f"  SECTION 4 : FEATURE ENGINEERING  [{label}]")
    print(f"{'='*65}")
    df = df.copy()

    # ── Spatial ────────────────────────────────────────────────────────────────
    df["distance_km"]   = haversine_km(
        df["pickup_latitude"],  df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"]
    )
    # ผลต่างพิกัด (ทิศทางการเดินทาง)
    df["delta_lat"]     = df["dropoff_latitude"]  - df["pickup_latitude"]
    df["delta_lon"]     = df["dropoff_longitude"] - df["pickup_longitude"]
    df["abs_delta_lat"] = df["delta_lat"].abs()
    df["abs_delta_lon"] = df["delta_lon"].abs()

    # ── Datetime ───────────────────────────────────────────────────────────────
    dt = df["pickup_datetime"]
    df["hour"]        = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek        # 0=จันทร์ … 6=อาทิตย์
    df["month"]       = dt.dt.month
    df["year"]        = dt.dt.year

    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    # Rush hour: เช้า 7–9 น. / เย็น 16–19 น.
    df["is_rush_hour"] = (
        df["hour"].between(7, 9) | df["hour"].between(16, 19)
    ).astype(int)
    # กลางคืน: 22–05 น. (NYC คิดค่าโดยสารเพิ่ม)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

    # ── Airport flags ─────────────────────────────────────────────────────────
    airports = {
        "jfk": (40.6413, -73.7781),   # Flat rate $52
        "lga": (40.7769, -73.8740),   # No flat rate
        "ewr": (40.6895, -74.1745),   # Surcharge $17.50
    }
    for name, (alat, alon) in airports.items():
        pu = near_airport(df["pickup_latitude"],  df["pickup_longitude"],  alat, alon)
        do = near_airport(df["dropoff_latitude"], df["dropoff_longitude"], alat, alon)
        df[f"is_{name}"] = (pu | do).astype(int)

    new_cols = [
        "distance_km", "delta_lat", "delta_lon", "abs_delta_lat", "abs_delta_lon",
        "hour", "day_of_week", "month", "year",
        "is_weekend", "is_rush_hour", "is_night",
        "is_jfk", "is_lga", "is_ewr",
    ]
    print(f"  Feature ใหม่ {len(new_cols)} ตัว:")
    for c in new_cols:
        print(f"    + {c}")

    return df


def remove_tiny_trips(df: pd.DataFrame, min_km: float = 0.1) -> pd.DataFrame:
    """ลบ trip ที่ distance < min_km (GPS error / ไม่ขยับ) — Train เท่านั้น"""
    n = len(df)
    df = df[df["distance_km"] >= min_km].reset_index(drop=True)
    print(f"\n  ลบ trip distance < {min_km} km: เหลือ {len(df):,} (ลบ {n-len(df):,})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — PREPARE X, y  + SCALING + ENCODING
# ─────────────────────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "distance_km",
    "delta_lat", "delta_lon", "abs_delta_lat", "abs_delta_lon",
    "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
    "passenger_count",
    "hour", "day_of_week", "month", "year",
]
BINARY_FEATURES = [
    "is_weekend", "is_rush_hour", "is_night",
    "is_jfk", "is_lga", "is_ewr",
]
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES


def prepare_features(train_df, test_df):
    """
    - fit StandardScaler บน Train เท่านั้น
    - transform ทั้ง Train และ Test (ป้องกัน data leakage)
    - Binary features ไม่ต้อง scale (0/1 อยู่แล้ว)
    """
    print("\n" + "=" * 65)
    print("  SECTION 5 : PREPARE X, y + STANDARDIZATION")
    print("=" * 65)

    X_train_raw = train_df[ALL_FEATURES].copy()
    X_test_raw  = test_df[ALL_FEATURES].copy()
    y_train     = train_df["fare_amount"].values

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_raw[NUMERIC_FEATURES])
    X_test_num  = scaler.transform(X_test_raw[NUMERIC_FEATURES])

    X_train = np.hstack([X_train_num, X_train_raw[BINARY_FEATURES].values])
    X_test  = np.hstack([X_test_num,  X_test_raw[BINARY_FEATURES].values])

    print(f"\n  X_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"\n  Features ({len(ALL_FEATURES)} ตัว):")
    for i, name in enumerate(ALL_FEATURES, 1):
        kind = "scaled" if name in NUMERIC_FEATURES else "binary"
        print(f"    {i:2d}. {name:<26} [{kind}]")

    return X_train, X_test, y_train, scaler, ALL_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 — CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
def plot_correlation(train_fe, save_path="correlation.png"):
    print(f"\n  [Correlation heatmap] → {save_path}")
    cols = ALL_FEATURES + ["fare_amount"]
    corr = train_fe[cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, linewidths=0.4, ax=ax,
        annot_kws={"size": 8}
    )
    ax.set_title("Feature Correlation Matrix (Train)", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ บันทึก → {save_path}")

    # top correlations กับ fare_amount
    fare_corr = corr["fare_amount"].drop("fare_amount").abs().sort_values(ascending=False)
    print("\n  Top correlations กับ fare_amount:")
    print(fare_corr.round(4).to_string())


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6B — INTERACTIVE MAP (folium)
# ─────────────────────────────────────────────────────────────────────────────
def plot_map(df: pd.DataFrame, save_path: str = "taxi_map.html"):
    """
    สร้าง Interactive Map ด้วย folium มี 4 layers สลับได้:
      🔥 Heatmap  — ความหนาแน่น pickup
      💰 Fare dots — สีตามช่วงราคา (คลิกดูรายละเอียด)
      🚕 Trip lines — เส้นทาง O→D สีตามราคา
      ✈️  Airports  — JFK / LGA / EWR
    """
    if not HAS_FOLIUM:
        print("  ⚠️  ข้าม map (ไม่มี folium)")
        return

    print(f"\n  [Map] กำลังสร้าง → {save_path}")

    # กรอง coords ให้อยู่ใน NYC ก่อน plot
    df = df[
        df["pickup_longitude"].between(-74.1, -73.7) &
        df["pickup_latitude"].between(40.6, 40.9) &
        df["dropoff_longitude"].between(-74.1, -73.7) &
        df["dropoff_latitude"].between(40.6, 40.9)
    ].copy()
    df["hour"] = df["pickup_datetime"].dt.hour
    df["fare_bin"] = pd.cut(
        df["fare_amount"],
        bins=[0, 10, 20, 35, 500],
        labels=["low (<$10)", "mid ($10-20)", "high ($20-35)", "very high (>$35)"],
    )

    m = folium.Map(
        location=[40.7549, -73.9840], zoom_start=12,
        tiles="CartoDB positron"
    )

    # ── Layer 1: Heatmap ───────────────────────────────────────────────────
    HeatMap(
        df[["pickup_latitude", "pickup_longitude"]].values.tolist(),
        radius=10, blur=12, min_opacity=0.35,
        gradient={0.3: "#313695", 0.5: "#4575b4", 0.7: "#fee090", 1.0: "#d73027"},
        name="🔥 Pickup heatmap",
    ).add_to(m)

    # ── Layer 2: Fare dots (sample 300) ────────────────────────────────────
    fare_layer = folium.FeatureGroup(name="💰 Fare by amount (sample 300)", show=False)
    color_map = {
        "low (<$10)":       "#2196F3",
        "mid ($10-20)":     "#4CAF50",
        "high ($20-35)":    "#FF9800",
        "very high (>$35)": "#F44336",
    }
    for _, row in df.sample(min(300, len(df)), random_state=42).iterrows():
        color = color_map.get(str(row["fare_bin"]), "#999")
        folium.CircleMarker(
            location=[row["pickup_latitude"], row["pickup_longitude"]],
            radius=5, color=color, fill=True,
            fill_color=color, fill_opacity=0.75, weight=0.5,
            popup=folium.Popup(
                f"<b>Fare:</b> ${row['fare_amount']:.2f}<br>"
                f"<b>Passengers:</b> {int(row['passenger_count'])}<br>"
                f"<b>Hour:</b> {int(row['hour'])}:00",
                max_width=200,
            ),
        ).add_to(fare_layer)
    fare_layer.add_to(m)

    # ── Layer 3: Trip O→D lines (sample 80) ────────────────────────────────
    od_layer = folium.FeatureGroup(name="🚕 Trip routes (sample 80)", show=False)
    for _, row in df.sample(min(80, len(df)), random_state=7).iterrows():
        ratio = min(row["fare_amount"] / 60, 1.0)
        r, g, b = int(220 * ratio), int(100 * (1 - ratio)), 80
        folium.PolyLine(
            locations=[
                [row["pickup_latitude"],  row["pickup_longitude"]],
                [row["dropoff_latitude"], row["dropoff_longitude"]],
            ],
            color=f"#{r:02x}{g:02x}{b:02x}",
            weight=2, opacity=0.7,
            tooltip=f"${row['fare_amount']:.2f}",
        ).add_to(od_layer)
    od_layer.add_to(m)

    # ── Layer 4: Airports ──────────────────────────────────────────────────
    airports_layer = folium.FeatureGroup(name="✈️ Airports", show=True)
    for name, lat, lon, desc in [
        ("JFK", 40.6413, -73.7781, "Flat rate $52"),
        ("LGA", 40.7769, -73.8740, "LaGuardia Airport"),
        ("EWR", 40.6895, -74.1745, "Newark — surcharge $17.50"),
    ]:
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(f"<b>{name}</b><br>{desc}", max_width=180),
            tooltip=name,
            icon=folium.Icon(color="darkblue", icon="plane", prefix="fa"),
        ).add_to(airports_layer)
    airports_layer.add_to(m)

    # ── Legend ─────────────────────────────────────────────────────────────
    m.get_root().html.add_child(folium.Element("""
    <div style="position:fixed;bottom:28px;left:28px;z-index:9999;
         background:white;padding:12px 16px;border-radius:10px;
         box-shadow:0 2px 8px rgba(0,0,0,0.2);font-family:Arial;font-size:12px">
      <b style="font-size:13px">💰 Fare range</b><br><br>
      <span style="background:#2196F3;border-radius:50%;display:inline-block;width:12px;height:12px;margin-right:6px"></span>Low &lt;$10<br>
      <span style="background:#4CAF50;border-radius:50%;display:inline-block;width:12px;height:12px;margin-right:6px"></span>Mid $10–20<br>
      <span style="background:#FF9800;border-radius:50%;display:inline-block;width:12px;height:12px;margin-right:6px"></span>High $20–35<br>
      <span style="background:#F44336;border-radius:50%;display:inline-block;width:12px;height:12px;margin-right:6px"></span>Very high &gt;$35<br>
      <hr style="margin:8px 0;border-color:#eee">
      <small style="color:#888">คลิกจุดเพื่อดูรายละเอียด</small>
    </div>"""))

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(save_path)
    print(f"  ✅ บันทึก → {save_path}  (เปิดด้วย browser ได้เลย)")


# ─────────────────────────────────────────────────────────────────────────────
#  SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
def sanity_check(X_train, X_test, y_train, train_fe):
    print("\n" + "=" * 65)
    print("  SANITY CHECK")
    print("=" * 65)
    assert not np.isnan(X_train).any(), "❌ NaN ใน X_train!"
    assert not np.isnan(X_test).any(),  "❌ NaN ใน X_test!"
    assert not np.isnan(y_train).any(), "❌ NaN ใน y_train!"
    print("  ✅ ไม่มี NaN ใน X_train, X_test, y_train")
    print(f"  fare_amount  — mean: ${y_train.mean():.2f}  "
          f"median: ${np.median(y_train):.2f}  std: ${y_train.std():.2f}")
    print(f"  distance_km  — mean: {train_fe['distance_km'].mean():.2f} km")
    corr_dist_fare = np.corrcoef(train_fe["distance_km"], y_train)[0, 1]
    print(f"  corr(distance_km, fare) = {corr_dist_fare:.4f}")
    print(f"\n  ✅ Phase 1 เสร็จสมบูรณ์ — พร้อมเข้า Phase 2 (Modeling)")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # 1. Load
    train_raw, test_raw = load_data(TRAIN_PATH, TEST_PATH)

    # 2. EDA (บันทึกเป็น PNG)
    plot_eda(train_raw, save_path="eda_plots.png")

    # 3. Clean
    train_clean = clean_train(train_raw)

    # 4. Feature Engineering (ใช้ฟังก์ชันเดียวกันกับ Test)
    train_fe = engineer_features(train_clean, label="Train")
    test_fe  = engineer_features(test_raw,   label="Test")

    train_fe = remove_tiny_trips(train_fe, min_km=0.1)

    # 5. Correlation heatmap
    plot_correlation(train_fe, save_path="correlation.png")

    # 5b. Interactive Map
    plot_map(train_fe, save_path="taxi_map.html")

    # 6. Prepare X, y
    X_train, X_test, y_train, scaler, feature_names = prepare_features(train_fe, test_fe)

    # 7. Sanity check
    sanity_check(X_train, X_test, y_train, train_fe)

    # ── Export ─────────────────────────────────────────────────────────────────
    # เซฟเป็น numpy arrays เพื่อใช้ใน Phase 2 ต่อได้เลย
    np.save("X_train.npy", X_train)
    np.save("X_test.npy",  X_test)
    np.save("y_train.npy", y_train)
    print("\n  💾 Export: X_train.npy, X_test.npy, y_train.npy")

    return X_train, X_test, y_train, scaler, feature_names, train_fe, test_fe


if __name__ == "__main__":
    main()
