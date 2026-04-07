import pandas as pd
import random
import os

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────
def load_dataset(csv_path):
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df["event"] = df["event"].fillna("None")
    df["holiday"] = df["holiday"].fillna("No")
    print(f"\n✅ Dataset loaded successfully ({len(df)} rows)\n")
    return df


# ─────────────────────────────────────────────
# SEARCH LOCATION
# ─────────────────────────────────────────────
def find_location(df, query):
    q = query.lower()
    return df[
        df["name"].str.lower().str.contains(q) |
        df["city"].str.lower().str.contains(q) |
        df["location_id"].str.lower().str.contains(q)
    ]


# ─────────────────────────────────────────────
# TRAFFIC CLASSIFICATION
# ─────────────────────────────────────────────
def classify_traffic(count):
    if count > 180:
        return "Heavy Traffic"
    elif count >= 100:
        return "Moderate Traffic"
    else:
        return "Smooth Traffic"


# ─────────────────────────────────────────────
# PREDICTION SYSTEM
# ─────────────────────────────────────────────
def predict_traffic(row):
    p5 = int(row["pred_5min"]) + random.randint(-5, 5)
    p10 = int(row["pred_10min"]) + random.randint(-5, 5)
    p15 = int(row["pred_15min"]) + random.randint(-5, 5)

    return {
        "5_min": max(0, p5),
        "10_min": max(0, p10),
        "15_min": max(0, p15)
    }


# ─────────────────────────────────────────────
# ROUGHT SELECTION
# ─────────────────────────────────────────────
def get_route(row):
    routes = row.get("routes", "")

    if pd.isna(routes) or not str(routes).strip() or str(routes).strip() == "None":
        fallback_routes = {
            "MG Road": "Brigade Road",
            "Anna Nagar": "Poonamallee High Road",
            "T. Nagar": "Kodambakkam High Road",
            "ECR Highway": "OMR Road",
            "Silk Board Junction": "Hosur Road Flyover",
            "Koyambedu Junction": "Poonamallee High Road",
            "Besant Nagar": "ECR",
            "Whitefield": "Outer Ring Road",
            "Velachery": "Taramani Road",
            "Marathahalli": "Outer Ring Road"
        }
        return fallback_routes.get(row["name"], "Bypass Road")

    routes_list = [r.strip() for r in str(routes).split(",") if r.strip()]
    return routes_list[0] if routes_list else "Bypass Road"


# ─────────────────────────────────────────────
# OUTPUT SYSTEM
# ─────────────────────────────────────────────
def show_output(row):
    print("\n" + "=" * 50)
    print(f"📍 {row['name']} ({row['city']})")
    print("=" * 50)

    preds = predict_traffic(row)

    p5 = preds["5_min"]
    p10 = preds["10_min"]
    p15 = preds["15_min"]

    print("\n🔮 Predictions:")
    print(f"  +5 min  : {p5} → {classify_traffic(p5)}")
    print(f"  +10 min : {p10} → {classify_traffic(p10)}")
    print(f"  +15 min : {p15} → {classify_traffic(p15)}")

    max_pred = max(p5, p10, p15)
    traffic_status = classify_traffic(max_pred)

    route = get_route(row)

    if traffic_status == "Heavy Traffic":
        print("\n🚨 Alert:")
        print(f"  Heavy traffic expected in {row['name']}")

        print("\n🛣️ Recommendation:")
        print(f"  Use {route} to avoid congestion")

    elif traffic_status == "Moderate Traffic":
        print("\n⚠️ Suggestion:")
        print(f"  You may try {route} to save time")

    else:
        print("\n✅ Traffic is smooth. No alternate route needed")

    # fixed live status from dataset
    live_count = int(row["vehicle_count"])
    live_speed = int(row["avg_speed_kmh"])

    print("\n📊 Live Status:")
    print(f"  Vehicles : {live_count}")
    print(f"  Speed    : {live_speed} km/h")
    print(f"  Status   : {classify_traffic(live_count)}")

    if row["event"] != "None":
        print(f"  Event    : {row['event']}")

    print("=" * 50)


# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────
def main():
    csv_path = "traffic_dataset.csv"

    if not os.path.exists(csv_path):
        print("❌ CSV file not found!")
        return

    df = load_dataset(csv_path)

    print("📌 Available Locations:")
    for i, row in df.iterrows():
        print(f"{i+1}. {row['name']} ({row['city']})")

    while True:
        query = input("\nEnter location (or 'q' to quit): ").strip()

        if query.lower() == "q":
            print("👋 Exiting...")
            break

        results = find_location(df, query)

        if results.empty:
            print("❌ Location not found")
            continue

        if len(results) > 1:
            print("⚠️ Multiple matches found:")
            for _, r in results.iterrows():
                print(f" - {r['name']} ({r['city']})")
            continue

        show_output(results.iloc[0])


# ─────────────────────────────────────────────
# RUN PROGRAM
# ─────────────────────────────────────────────
if __name__ == "__main__":
    random.seed()
    main()