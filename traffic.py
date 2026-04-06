# Clean and Interactive Traffic Output System

import pandas as pd
import random
import os

# ---------------- LOAD DATA ----------------

def load_dataset(csv_path):
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df["event"] = df["event"].fillna("None")
    df["holiday"] = df["holiday"].fillna("No")
    return df

# ---------------- LOGIC ----------------

def classify_traffic(count):
    if count > 80:
        return "Heavy Traffic"
    elif count >= 50:
        return "Moderate Traffic"
    else:
        return "Smooth Traffic"


def predict(row):
    noise = random.randint(-5, 5)
    return {
        "5_min": max(5, int(row["pred_5min"]) + noise),
        "10_min": max(5, int(row["pred_10min"]) + noise),
        "15_min": max(5, int(row["pred_15min"]) + noise),
    }


def recommend(count, routes):
    routes = [r.strip() for r in routes.split(",")]
    label = classify_traffic(count)

    if label == "Heavy Traffic":
        return f"Use {routes[0]} to avoid congestion"
    elif label == "Moderate Traffic":
        return f"Consider {routes[0]} to save time"
    else:
        return "No alternate route needed"

# ---------------- CLEAN OUTPUT ----------------

def show_output(row):
    preds = predict(row)
    p5, p10, p15 = preds.values()

    status_now = classify_traffic(row["vehicle_count"])
    status_future = classify_traffic(p10)

    routes_map = {
        "anna_nagar": "Poonamallee High Road, Arcot Road",
        "mg_road": "Residency Road, Richmond Road",
        "t_nagar": "Kodambakkam High Road, Nungambakkam Road",
    }

    routes = routes_map.get(row["location_id"], "Main Road, Bypass")
    suggestion = recommend(p10, routes)

    print("\n" + "="*50)
    print(f"📍 {row['name']} ({row['city']})")
    print("="*50)

    print("\n🔮 Prediction:")
    print(f"  +5 min  : {p5} → {classify_traffic(p5)}")
    print(f"  +10 min : {p10} → {classify_traffic(p10)}")
    print(f"  +15 min : {p15} → {classify_traffic(p15)}")

    print("\n🚨 Alert:")
    print(f"  {status_future}")

    print("\n🛣️ Recommendation:")
    print(f"  {suggestion}")

    print("\n📊 Live Status:")
    print(f"  Vehicles : {row['vehicle_count']}")
    print(f"  Speed    : {row['avg_speed_kmh']} km/h")
    print(f"  Status   : {status_now}")

    print("\n📈 Summary:")
    print(f"  Traffic is currently {status_now} and expected to remain {status_future}.")

    print("="*50)

# ---------------- MAIN (INTERACTIVE SEARCH) ----------------

def main():
    csv_path = "traffic_dataset.csv"
    df = load_dataset(csv_path)

    print("\nAvailable Locations:")
    for i, row in df.iterrows():
        print(f"  - {row['name']} ({row['city']})")

    while True:
        query = input("\nEnter location name (or 'all' / 'q'): ").lower().strip()

        if query == "q":
            print("Goodbye!")
            break

        if query == "all":
            for _, row in df.iterrows():
                show_output(row)
            continue

        results = df[
            df["name"].str.lower().str.contains(query) |
            df["city"].str.lower().str.contains(query)
        ]

        if results.empty:
            print("❌ No location found. Try again.")
        elif len(results) > 1:
            print("\n🔍 Multiple matches found:")
            for _, r in results.iterrows():
                print(f"  - {r['name']} ({r['city']})")
            print("👉 Please be more specific.")
        else:
            show_output(results.iloc[0])

if __name__ == "__main__":
    random.seed(42)
    main()