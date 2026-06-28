import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
SOURCE_PATH = DATA_DIR / "workzone_traj_motion_agents_map.npz"

VERSIONS = {
    "v1_traj_only": list(range(0, 2)),
    "v2_ego_motion": list(range(0, 9)),
    "v3_agent_context": list(range(0, 18)),
    "v4_map_workzone": list(range(0, 43)),
}

data = np.load(SOURCE_PATH, allow_pickle=True)
X_full = data["X"]
feature_names_full = data["feature_names"]

for version_name, feature_indices in VERSIONS.items():
    X_new = X_full[:, :, feature_indices]
    feature_names_new = feature_names_full[feature_indices]

    out_path = DATA_DIR / f"workzone_{version_name}.npz"

    save_dict = {}

    for key in data.files:
        if key == "X":
            save_dict[key] = X_new
        elif key == "feature_names":
            save_dict[key] = feature_names_new
        else:
            save_dict[key] = data[key]

    np.savez_compressed(out_path, **save_dict)

    print("Saved:", out_path)
    print("X shape:", X_new.shape)
    print("Features:")
    for i, name in enumerate(feature_names_new):
        print(f"  {i}: {name}")
    print()
