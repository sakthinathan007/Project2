import pandas as pd
import numpy as np
import random
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import ipywidgets as widgets
from IPython.display import display, clear_output

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────
def load_dataset(csv_path):
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df["event"]   = df["event"].fillna("None")
    df["holiday"] = df["holiday"].fillna("No")
    df["weather"] = df["weather"].fillna("Sunny")
    print(f"✅ Dataset loaded successfully ({len(df)} rows)")
    return df


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    df["timestamp"]  = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"]       = df["timestamp"].dt.hour
    df["month"]      = df["timestamp"].dt.month
    df["is_weekend"] = df["timestamp"].dt.dayofweek.isin([5, 6]).astype(int)

    cat_cols = ["city", "type", "weather", "day_of_week", "holiday", "event", "location_id"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    for i in range(24):
        if f"h{i}" not in df.columns:
            df[f"h{i}"] = 0

    return df, encoders


FEATURE_COLS = [
    "hour", "month", "is_weekend",
    "city_enc", "type_enc", "weather_enc",
    "day_of_week_enc", "holiday_enc", "event_enc",
    "vehicle_count", "avg_speed_kmh", "congestion_pct",
] + [f"h{i}" for i in range(24)]

TARGET_COLS = ["pred_5min", "pred_10min", "pred_15min"]


# ─────────────────────────────────────────────
# TRAIN XGBoost MODELS
# ─────────────────────────────────────────────
def train_xgboost_models(df):
    df_feat, encoders = engineer_features(df)
    X = df_feat[FEATURE_COLS].fillna(0)

    models  = {}
    metrics = {}

    for target in TARGET_COLS:
        y = df_feat[target].fillna(df_feat["vehicle_count"])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = XGBRegressor(
            n_estimators     = 100,
            max_depth        = 4,
            learning_rate    = 0.1,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            random_state     = 42,
            verbosity        = 0,
            n_jobs           = 1
        )
        model.fit(X_train, y_train)
        preds    = model.predict(X_test)
        mae      = mean_absolute_error(y_test, preds)
        rmse     = np.sqrt(mean_squared_error(y_test, preds))
        r2       = r2_score(y_test, preds)
        accuracy = max(0, 100 - (mae / y_test.mean() * 100))

        models[target]  = model
        metrics[target] = {"MAE": mae, "RMSE": rmse, "R2": r2, "Accuracy": accuracy}

    return models, encoders, metrics


# ─────────────────────────────────────────────
# PREDICT FOR A SINGLE ROW
# ─────────────────────────────────────────────
def predict_row(row, models, encoders):
    ts         = pd.to_datetime(row["timestamp"], errors="coerce")
    hour       = ts.hour       if not pd.isna(ts) else 9
    month      = ts.month      if not pd.isna(ts) else 6
    is_weekend = int(ts.dayofweek in [5, 6]) if not pd.isna(ts) else 0

    def safe_encode(encoder, val):
        try:
            return encoder.transform([str(val)])[0]
        except Exception:
            return 0

    feat = {
        "hour"           : hour,
        "month"          : month,
        "is_weekend"     : is_weekend,
        "city_enc"       : safe_encode(encoders["city"],        row["city"]),
        "type_enc"       : safe_encode(encoders["type"],        row["type"]),
        "weather_enc"    : safe_encode(encoders["weather"],     row["weather"]),
        "day_of_week_enc": safe_encode(encoders["day_of_week"], row["day_of_week"]),
        "holiday_enc"    : safe_encode(encoders["holiday"],     row["holiday"]),
        "event_enc"      : safe_encode(encoders["event"],       row["event"]),
        "vehicle_count"  : row["vehicle_count"],
        "avg_speed_kmh"  : row["avg_speed_kmh"],
        "congestion_pct" : row["congestion_pct"],
    }
    for i in range(24):
        feat[f"h{i}"] = row.get(f"h{i}", 0)

    X_pred = pd.DataFrame([feat])[FEATURE_COLS].fillna(0)
    preds  = {}
    for target, label in zip(TARGET_COLS, ["5_min", "10_min", "15_min"]):
        preds[label] = max(0, int(models[target].predict(X_pred)[0]))
    return preds


# ─────────────────────────────────────────────
# TRAFFIC CLASSIFICATION
# ─────────────────────────────────────────────
def classify_traffic(count):
    if count > 180:
        return "🔴 Heavy Traffic"
    elif count >= 100:
        return "🟡 Moderate Traffic"
    else:
        return "🟢 Smooth Traffic"


# ─────────────────────────────────────────────
# AUTO ROUTE SUGGESTION  (data-driven)
# ─────────────────────────────────────────────
def get_auto_route(current_row, df):
    """
    Finds the best alternate route automatically from the dataset by:
    1. Looking at other locations in the SAME CITY
    2. Picking ones with lowest congestion_pct & highest avg_speed_kmh
    3. Excluding the current location itself
    4. Returns top-2 suggestions with their live stats
    """
    city         = current_row["city"]
    current_loc  = current_row["location_id"]
    current_cong = current_row["congestion_pct"]

    # All other locations in same city — use latest reading per location
    city_df = df[df["city"] == city].copy()
    city_df = city_df[city_df["location_id"] != current_loc]

    if city_df.empty:
        return None

    # Get the latest record per location
    city_df = city_df.sort_values("timestamp", ascending=False)
    latest  = city_df.groupby("location_id").first().reset_index()

    # Score: low congestion + high speed = best alternate
    # Normalize both to 0-1 range then combine
    if len(latest) == 0:
        return None

    max_cong  = latest["congestion_pct"].max() or 1
    max_speed = latest["avg_speed_kmh"].max()   or 1

    latest["score"] = (
        (1 - latest["congestion_pct"] / max_cong) * 0.6 +
        (latest["avg_speed_kmh"]      / max_speed) * 0.4
    )

    # Only suggest routes that are actually better than current
    better = latest[latest["congestion_pct"] < current_cong].sort_values("score", ascending=False)

    if better.empty:
        # No better route — still return least congested
        better = latest.sort_values("score", ascending=False)

    return better.head(2)  # top 2 alternates


# ─────────────────────────────────────────────
# PRINT OUTPUT
# ─────────────────────────────────────────────
def show_output(row, models, encoders, df):
    print("\n" + "=" * 55)
    print(f"📍 {row['name']} — {row['location_id']}")
    print(f"   Type      : {row['type']}")
    print(f"   Timestamp : {row['timestamp']}")
    print(f"   Weather   : {row['weather']}")
    print(f"   Day       : {row['day_of_week']}")
    if str(row["holiday"]) == "Yes":
        print(f"   🗓️  Holiday  : Yes")
    if str(row["event"]) not in ["None", "nan", ""]:
        print(f"   🎉 Event    : {row['event']}")
    print("=" * 55)

    # ── XGBoost predictions
    preds = predict_row(row, models, encoders)
    p5, p10, p15 = preds["5_min"], preds["10_min"], preds["15_min"]

    print("\n🔮 XGBoost Predictions:")
    print(f"  +5 min   : {p5:>4} vehicles  →  {classify_traffic(p5)}")
    print(f"  +10 min  : {p10:>4} vehicles  →  {classify_traffic(p10)}")
    print(f"  +15 min  : {p15:>4} vehicles  →  {classify_traffic(p15)}")

    # ── Accuracy
    actual_p5 = int(row["pred_5min"])
    error     = abs(actual_p5 - p5)
    acc       = max(0.0, 100 - (error / max(actual_p5, 1) * 100))
    print(f"\n📐 Prediction Accuracy (vs dataset):")
    print(f"  Actual +5min value  : {actual_p5}")
    print(f"  XGBoost prediction  : {p5}")
    print(f"  Error               : {error}")
    print(f"  Accuracy            : {acc:.2f}%")

    # ── Live status
    live_count = int(row["vehicle_count"])
    live_speed = int(row["avg_speed_kmh"])
    live_cong  = int(row["congestion_pct"])

    print("\n📊 Live Status:")
    print(f"  Vehicles    : {live_count}")
    print(f"  Speed       : {live_speed} km/h")
    print(f"  Congestion  : {live_cong}%")
    print(f"  Status      : {classify_traffic(live_count)}")
    print(f"\n⏰ Peak Hours : {row['peak_hours']}")

    # ── Auto Route Suggestion (data-driven)
    max_pred = max(p5, p10, p15)
    status   = classify_traffic(max_pred)

    print("\n🛣️  Recommendation:")

    if "Heavy" in status or "Moderate" in status:
        emoji = "🚨" if "Heavy" in status else "⚠️ "
        print(f"  {emoji} {status.split(' ',1)[1]} expected — finding alternate routes...\n")

        alternates = get_auto_route(row, df)

        if alternates is not None and len(alternates) > 0:
            print("  📌 Suggested Alternate Routes :")
            print(f"  {'Rank':<5}   {'Location':<28} {'Congestion':>10}        {'Speed':>8}       {'Status'}")
            print("  " + "─" * 85)
            for rank, (_, alt) in enumerate(alternates.iterrows(), 1):
                cong_bar = "█" * int(alt['congestion_pct'] // 10) + "░" * (10 - int(alt['congestion_pct'] // 10))
                alt_status = classify_traffic(int(alt["vehicle_count"]))
                print(f"  #{rank:<4}   {alt['name']:<28} {int(alt['congestion_pct']):>5}% {cong_bar}  {int(alt['avg_speed_kmh']):>4} km/h  {alt_status}")
            print()
        else:
            print("  ℹ️  No significantly better alternate found in the same city.")
    else:
        print(f"  ✅ Traffic is smooth at {row['name']}. No alternate route needed.")

    print("=" * 55)


# ─────────────────────────────────────────────
# JUPYTER SCROLLABLE WIDGET UI
# ─────────────────────────────────────────────
def launch_ui(df, models, encoders):
    location_ids = sorted(df["location_id"].unique().tolist())

    search_box = widgets.Text(
        placeholder="🔍 Type to filter location IDs...",
        layout=widgets.Layout(width="420px")
    )

    list_box = widgets.Select(
        options=location_ids,
        value=location_ids[0],
        rows=15,
        layout=widgets.Layout(width="420px")
    )

    btn = widgets.Button(
        description="Show Traffic Info",
        button_style="primary",
        icon="car",
        layout=widgets.Layout(width="420px", height="36px")
    )

    out = widgets.Output()

    def on_search(change):
        query    = change["new"].lower()
        filtered = [loc for loc in location_ids if query in loc.lower()] if query else location_ids
        list_box.options = filtered if filtered else ["(no match)"]

    search_box.observe(on_search, names="value")

    def on_click(_):
        selected_id = list_box.value
        if not selected_id or selected_id == "(no match)":
            return
        results = df[df["location_id"] == selected_id].sort_values("timestamp", ascending=False)
        if results.empty:
            with out:
                clear_output()
                print(f"❌ No data found for: {selected_id}")
            return
        row = results.iloc[0]
        with out:
            clear_output()
            show_output(row, models, encoders, df)

    btn.on_click(on_click)

    title = widgets.HTML("<h3 style='color:#2c7bb6'>🚦 Indian Traffic Prediction System</h3>")
    label = widgets.HTML("<b>Select Location ID:</b>")
    ui    = widgets.VBox([title, search_box, label, list_box, btn, out])
    display(ui)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    csv_path = "indian_traffic_dataset.csv"

    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return

    df = load_dataset(csv_path)

    print("⏳ Training XGBoost models, please wait...")
    models, encoders, metrics = train_xgboost_models(df)
    print("✅ Models ready! Loading UI...\n")

    launch_ui(df, models, encoders)


random.seed()
main()