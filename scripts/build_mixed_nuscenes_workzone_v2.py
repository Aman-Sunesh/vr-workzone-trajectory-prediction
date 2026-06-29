import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NUSC_PATH = PROJECT_ROOT / "data" / "nuscenes_v2_ego_motion_5hz.npz"
WZ_PATH = PROJECT_ROOT / "data" / "workzone_v2_ego_motion.npz"
OUT_PATH = PROJECT_ROOT / "data" / "mixed_nuscenes_workzone_v2.npz"

CAP_NUSC_TRAIN = 11626
RANDOM_STATE = 43

rng = np.random.default_rng(RANDOM_STATE)

nusc = np.load(NUSC_PATH, allow_pickle=True)
wz = np.load(WZ_PATH, allow_pickle=True)

Xn = nusc["X"].astype(np.float32)
yn = nusc["y"].astype(np.float32)
sn = nusc["split"].astype(str)

Xw = wz["X"].astype(np.float32)
yw = wz["y"].astype(np.float32)
sw = wz["split"].astype(str)

if Xn.shape[-1] != Xw.shape[-1]:
    raise ValueError(f"Feature mismatch: nuScenes {Xn.shape}, workzone {Xw.shape}")

if Xn.shape[1:] != Xw.shape[1:]:
    raise ValueError(f"Shape mismatch: nuScenes {Xn.shape}, workzone {Xw.shape}")

nusc_train_idx = np.where(sn == "train")[0]
wz_train_idx = np.where(sw == "train")[0]
wz_val_idx = np.where(sw == "val")[0]
wz_test_idx = np.where(sw == "test")[0]

if len(nusc_train_idx) > CAP_NUSC_TRAIN:
    nusc_train_idx = rng.choice(nusc_train_idx, size=CAP_NUSC_TRAIN, replace=False)

X_train = np.concatenate([Xn[nusc_train_idx], Xw[wz_train_idx]], axis=0)
y_train = np.concatenate([yn[nusc_train_idx], yw[wz_train_idx]], axis=0)

train_source = np.array(
    ["nuscenes"] * len(nusc_train_idx) + ["workzone"] * len(wz_train_idx),
    dtype=object
)

# Shuffle train only.
perm = rng.permutation(len(X_train))
X_train = X_train[perm]
y_train = y_train[perm]
train_source = train_source[perm]

X_val = Xw[wz_val_idx]
y_val = yw[wz_val_idx]
val_source = np.array(["workzone"] * len(X_val), dtype=object)

X_test = Xw[wz_test_idx]
y_test = yw[wz_test_idx]
test_source = np.array(["workzone"] * len(X_test), dtype=object)

X = np.concatenate([X_train, X_val, X_test], axis=0)
y = np.concatenate([y_train, y_val, y_test], axis=0)

split = np.array(
    ["train"] * len(X_train) +
    ["val"] * len(X_val) +
    ["test"] * len(X_test),
    dtype=object
)

source = np.concatenate([train_source, val_source, test_source], axis=0)

np.savez_compressed(
    OUT_PATH,
    X=X.astype(np.float32),
    y=y.astype(np.float32),
    split=split,
    source=source,
    feature_names=wz["feature_names"],
)

print("Saved:", OUT_PATH)
print("X:", X.shape)
print("y:", y.shape)

print("\nSplit counts:")
for s, c in zip(*np.unique(split, return_counts=True)):
    print(s, c)

print("\nSource counts:")
for s, c in zip(*np.unique(source, return_counts=True)):
    print(s, c)

print("\nTrain source counts:")
for s, c in zip(*np.unique(source[split == 'train'], return_counts=True)):
    print(s, c)
