
import pandas as pd

df = pd.read_csv("training_labels_trip1.csv")

print("LABEL QUALITY REPORT")
print("=" * 40)

for threshold in [10, 20, 30, 50, 100]:
    count = (df["snap_distance_m"] <= threshold).sum()
    percentage = 100 * count / len(df)

    print(
        f"Within {threshold:3d} metres: "
        f"{count:4d}/{len(df)} ({percentage:.2f}%)"
    )

positions = df["route_position_m"]
backward = (positions.diff() < -1).sum()

print("\nBackward route-position jumps:", backward)
print("Duplicate consecutive arc assignments:",
      (df["arc_id"].diff() == 0).sum())

print("Observations farther than 50 metres:",
      (df["snap_distance_m"] > 50).sum())
