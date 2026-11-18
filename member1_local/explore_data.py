
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
# TRAIN_DIR = BASE / "Student_folder" / "train"
TRAIN_DIR = BASE / "Student folder" / "train"
# TEST_DIR = BASE / "Student_folder" / "student_test"
TEST_DIR = BASE / "Student folder" / "student_test"

def inspect_partition(folder):
    trips = sorted(p for p in folder.iterdir() if p.is_dir())
    print(f"\nPartition: {folder.name}")
    print(f"Number of trips: {len(trips)}")

    for trip in trips[:3]:
        print(f"\nTrip: {trip.name}")

        for extension in ["track", "nodes", "arcs", "route"]:
            file = trip / f"{trip.name}.{extension}"

            if not file.exists():
                print(f"  {extension}: MISSING")
                continue

            data = np.loadtxt(file, ndmin=2)
            print(f"  {extension}: shape={data.shape}")

            if extension == "track":
                print("  First 3 GPS observations:")
                print(data[:3])

                if len(data) > 1:
                    gaps = np.diff(data[:, 2])
                    print(f"  Time gaps > 1 sec: {np.sum(gaps > 1)}")
                    print(f"  Maximum time gap: {np.max(gaps)}")

if __name__ == "__main__":
    inspect_partition(TRAIN_DIR)
    inspect_partition(TEST_DIR)
