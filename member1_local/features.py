
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
TRACK = BASE / "Student folder" / "train" / "00000001" / "00000001.track"

gps = np.loadtxt(TRACK)

df = pd.DataFrame(
    gps, columns=["longitude", "latitude", "timestamp"]
)

lon = np.radians(df["longitude"].to_numpy())
lat = np.radians(df["latitude"].to_numpy())

dlon = np.diff(lon, prepend=lon[0])
dlat = np.diff(lat, prepend=lat[0])

a = (
    np.sin(dlat / 2) ** 2
    + np.cos(lat) * np.cos(np.roll(lat, 1))
    * np.sin(dlon / 2) ** 2
)
a[0] = 0

distance = 6371000 * 2 * np.arcsin(
    np.sqrt(np.clip(a, 0, 1))
)

dt = df["timestamp"].diff().fillna(0).to_numpy()

speed = np.divide(
    distance, dt,
    out=np.zeros_like(distance),
    where=dt > 0
)

y = np.sin(dlon) * np.cos(lat)
x = (
    np.cos(np.roll(lat, 1)) * np.sin(lat)
    - np.sin(np.roll(lat, 1)) * np.cos(lat) * np.cos(dlon)
)

bearing = (np.degrees(np.arctan2(y, x)) + 360) % 360
bearing[0] = 0

df["distance_m"] = distance
df["time_delta_s"] = dt
df["speed_mps"] = speed
df["bearing_deg"] = bearing
df["large_gap"] = (dt > 5).astype(int)

print("\nFEATURE ENGINEERING COMPLETE")
print(df.head(10).to_string(index=False))

print("\nSUMMARY STATISTICS")
print(df.describe().round(2))

print("\nMissing values:")
print(df.isna().sum())

df.to_csv(BASE / "gps_features.csv", index=False)

print("\nSaved gps_features.csv")
