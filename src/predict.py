"""
predict.py  —  KDU Smart Parking Prediction Module
----------------------------------------------------
1. Generates a realistic synthetic week of occupancy data
   (blended with any real records already in occupancy.json)
2. Trains a Random Forest classifier to predict slot occupancy
   given: hour, minute, day_of_week
3. Saves predictions for the next 60 minutes to
   data/results/predictions.json
4. Prints a human-readable forecast table
"""

import json
import os
import random
import numpy as np
from datetime import datetime, timedelta

# ── Try to import sklearn; guide user if missing ───────────────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score
except ImportError:
    print("ERROR: scikit-learn not installed.")
    print("Run:  pip install scikit-learn")
    raise

OCCUPANCY_PATH   = "data/results/occupancy.json"
PREDICTIONS_PATH = "data/results/predictions.json"
MODEL_REPORT     = "data/results/model_report.json"
NUM_SLOTS        = 7
SLOT_IDS         = [f"S{i}" for i in range(NUM_SLOTS)]
PREDICT_MINUTES  = 60    # forecast window
PREDICT_STEP     = 5     # minutes between each forecast point

os.makedirs("data/results", exist_ok=True)


# ── 1. Synthetic data generator ───────────────────────────────────────────────
def occupancy_probability(hour, minute, day_of_week, slot_id):
    """
    Realistic campus parking probability model.
    Weekdays:  busy 08-18h, peak at 09-11h and 13-15h
    Weekends:  quiet overall, slight activity 10-14h
    Some slots (S0, S1) fill earlier (closer to entrance)
    """
    slot_bias = {"S0": 0.15, "S1": 0.10, "S2": 0.05,
                 "S3": 0.0,  "S4": 0.0,  "S5": -0.05, "S6": -0.10}
    bias = slot_bias.get(slot_id, 0.0)

    t = hour + minute / 60.0
    is_weekend = day_of_week >= 5

    if is_weekend:
        if 10 <= t <= 14:
            base = 0.30
        else:
            base = 0.08
    else:
        # Morning peak
        if 8.5 <= t <= 11.0:
            base = 0.75
        # Lunch dip
        elif 11.0 <= t <= 12.0:
            base = 0.55
        # Afternoon peak
        elif 12.0 <= t <= 17.0:
            base = 0.70
        # Evening wind-down
        elif 17.0 <= t <= 19.0:
            base = 0.35
        # Night / early morning
        else:
            base = 0.05

    return min(1.0, max(0.0, base + bias))


def generate_synthetic_week(base_dt=None):
    """Generate one week of 5-minute interval records."""
    if base_dt is None:
        base_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        base_dt -= timedelta(days=7)   # start 7 days ago

    records = []
    current = base_dt
    end     = base_dt + timedelta(days=7)

    while current < end:
        h   = current.hour
        m   = current.minute
        dow = current.weekday()   # 0=Mon … 6=Sun

        slot_status = {}
        for sid in SLOT_IDS:
            p = occupancy_probability(h, m, dow, sid)
            slot_status[sid] = "OCCUPIED" if random.random() < p else "FREE"

        occupied = sum(1 for v in slot_status.values() if v == "OCCUPIED")
        records.append({
            "timestamp":      current.strftime("%Y-%m-%d %H:%M:%S"),
            "video_time_sec": 0.0,
            "slots":          slot_status,
            "total_free":     NUM_SLOTS - occupied,
            "total_occupied": occupied,
            "total_slots":    NUM_SLOTS,
            "source":         "synthetic"
        })
        current += timedelta(minutes=5)

    return records


# ── 2. Feature extraction ─────────────────────────────────────────────────────
def record_to_features(rec):
    """Convert a record to (features, labels_per_slot)."""
    dt  = datetime.strptime(rec["timestamp"], "%Y-%m-%d %H:%M:%S")
    X   = [dt.hour, dt.minute, dt.weekday(),
           dt.hour * 60 + dt.minute,          # minutes since midnight
           int(dt.weekday() >= 5)]             # is_weekend flag
    y   = {sid: 1 if rec["slots"].get(sid, "FREE") == "OCCUPIED" else 0
           for sid in SLOT_IDS}
    return X, y


# ── 3. Train models ───────────────────────────────────────────────────────────
def train_models(records):
    """Train one RandomForest per slot. Returns dict of models + accuracy."""
    X_all = []
    y_all = {sid: [] for sid in SLOT_IDS}

    for rec in records:
        X, y = record_to_features(rec)
        X_all.append(X)
        for sid in SLOT_IDS:
            y_all[sid].append(y[sid])

    X_all = np.array(X_all)
    models     = {}
    accuracies = {}

    for sid in SLOT_IDS:
        y = np.array(y_all[sid])
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        clf.fit(X_all, y)
        preds = clf.predict(X_all)
        acc   = accuracy_score(y, preds)
        models[sid]     = clf
        accuracies[sid] = round(acc * 100, 1)
        print(f"  Slot {sid}: training accuracy = {acc*100:.1f}%")

    return models, accuracies


# ── 4. Predict next N minutes ─────────────────────────────────────────────────
def predict_next(models, from_dt=None):
    """Generate predictions for the next PREDICT_MINUTES."""
    if from_dt is None:
        from_dt = datetime.now()

    predictions = []
    for step in range(0, PREDICT_MINUTES + 1, PREDICT_STEP):
        target_dt = from_dt + timedelta(minutes=step)
        X = [[target_dt.hour,
              target_dt.minute,
              target_dt.weekday(),
              target_dt.hour * 60 + target_dt.minute,
              int(target_dt.weekday() >= 5)]]

        slot_pred = {}
        for sid in SLOT_IDS:
            prob  = models[sid].predict_proba(X)[0]
            occ_p = prob[1] if len(prob) > 1 else 0.0
            slot_pred[sid] = {
                "prediction":    "OCCUPIED" if occ_p >= 0.5 else "FREE",
                "probability":   round(occ_p * 100, 1)
            }

        occupied = sum(1 for v in slot_pred.values()
                       if v["prediction"] == "OCCUPIED")
        predictions.append({
            "forecast_time":    target_dt.strftime("%Y-%m-%d %H:%M"),
            "minutes_from_now": step,
            "slots":            slot_pred,
            "predicted_free":   NUM_SLOTS - occupied,
            "predicted_occupied": occupied
        })

    return predictions


# ── 5. Print forecast table ───────────────────────────────────────────────────
def print_forecast(predictions):
    print("\n" + "="*70)
    print(f"  PARKING AVAILABILITY FORECAST  (next {PREDICT_MINUTES} min)")
    print("="*70)
    header = f"  {'Time':<18} {'Free':>5} {'Occ':>5}   " + \
             "  ".join(SLOT_IDS)
    print(header)
    print("-"*70)
    for p in predictions:
        slot_icons = []
        for sid in SLOT_IDS:
            icon = "OCC" if p["slots"][sid]["prediction"] == "OCCUPIED" else "---"
            slot_icons.append(f"{icon:>5}")
        row = (f"  {p['forecast_time']:<18} "
               f"{p['predicted_free']:>5} "
               f"{p['predicted_occupied']:>5}   "
               + "  ".join(slot_icons))
        print(row)
    print("="*70)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*55)
    print("  KDU Smart Parking — Prediction Module")
    print("="*55)

    # Load real records
    real_records = []
    if os.path.exists(OCCUPANCY_PATH):
        with open(OCCUPANCY_PATH, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                real_records = data
        print(f"Loaded {len(real_records)} real records from {OCCUPANCY_PATH}")
    else:
        print("No real records found — using synthetic data only.")

    # Generate synthetic week
    print("Generating synthetic training data (1 week × 5-min intervals)...")
    synthetic = generate_synthetic_week()
    print(f"Generated {len(synthetic)} synthetic records")

    # Merge: real records get added on top of synthetic
    all_records = synthetic + real_records
    print(f"Total training records: {len(all_records)}")

    # Train
    print("\nTraining Random Forest models (one per slot)...")
    models, accuracies = train_models(all_records)

    avg_acc = sum(accuracies.values()) / len(accuracies)
    print(f"\nAverage training accuracy: {avg_acc:.1f}%")

    # Predict
    print("\nGenerating forecast for next 60 minutes...")
    predictions = predict_next(models)

    # Save predictions
    output = {
        "generated_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_accuracy": accuracies,
        "avg_accuracy":   round(avg_acc, 1),
        "forecast":       predictions
    }
    with open(PREDICTIONS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved predictions -> {PREDICTIONS_PATH}")

    # Save model report
    report = {
        "trained_at":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_records":  len(all_records),
        "real_records":      len(real_records),
        "synthetic_records": len(synthetic),
        "model_type":        "RandomForestClassifier",
        "features":          ["hour", "minute", "day_of_week",
                              "minutes_since_midnight", "is_weekend"],
        "slot_accuracies":   accuracies,
        "avg_accuracy":      round(avg_acc, 1)
    }
    with open(MODEL_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    # Print forecast table
    print_forecast(predictions)

    print(f"\nDone! Check {PREDICTIONS_PATH} for full forecast data.")


if __name__ == "__main__":
    main()