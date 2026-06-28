import json
import math
from pathlib import Path
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NUSC_ROOT = PROJECT_ROOT / "data" / "nuscenes"
VERSION = "v1.0-mini"
NUSC_META = NUSC_ROOT / VERSION

OUT_PATH = PROJECT_ROOT / "data" / "nuscenes_v2_ego_motion_5hz.npz"

TARGET_HZ = 5.0
DT = 1.0 / TARGET_HZ

OBS_STEPS = 10
FUTURE_STEPS = 25

OBS_TIMES = np.array([-(OBS_STEPS - 1 - i) * DT for i in range(OBS_STEPS)], dtype=np.float64)
FUT_TIMES = np.array([(i + 1) * DT for i in range(FUTURE_STEPS)], dtype=np.float64)

FEATURE_NAMES = np.array([
    "x", "y",
    "vx", "vy",
    "ax", "ay",
    "cos_heading", "sin_heading",
    "speed"
], dtype=object)

VEHICLE_PREFIX = "vehicle."


def load_json(name):
    path = NUSC_META / name
    with open(path, "r") as f:
        return json.load(f)


def yaw_from_quat(q):
    # nuScenes quaternion order: [w, x, y, z]
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def world_to_local(points_world, anchor_xy, anchor_yaw):
    c = np.cos(anchor_yaw)
    s = np.sin(anchor_yaw)

    R = np.array([
        [c, -s],
        [s,  c],
    ], dtype=np.float32)

    return (points_world - anchor_xy.reshape(1, 2)) @ R


def compute_velocity_acceleration(xy):
    velocity = np.zeros_like(xy, dtype=np.float32)
    velocity[1:] = (xy[1:] - xy[:-1]) / DT
    velocity[0] = velocity[1]

    acceleration = np.zeros_like(xy, dtype=np.float32)
    acceleration[1:] = (velocity[1:] - velocity[:-1]) / DT
    acceleration[0] = acceleration[1]

    return velocity, acceleration


def build_features(obs_xy_local, obs_heading_local):
    velocity, acceleration = compute_velocity_acceleration(obs_xy_local)
    speed = np.linalg.norm(velocity, axis=1)

    X = np.concatenate([
        obs_xy_local,
        velocity,
        acceleration,
        np.cos(obs_heading_local).reshape(-1, 1),
        np.sin(obs_heading_local).reshape(-1, 1),
        speed.reshape(-1, 1)
    ], axis=1).astype(np.float32)

    return X


def main():
    print("Using nuScenes metadata:", NUSC_META)

    if not NUSC_META.exists():
        raise FileNotFoundError(f"Missing {NUSC_META}")

    scenes = load_json("scene.json")
    samples = load_json("sample.json")
    anns = load_json("sample_annotation.json")
    instances = load_json("instance.json")
    categories = load_json("category.json")

    sample_by_token = {s["token"]: s for s in samples}
    instance_by_token = {i["token"]: i for i in instances}
    category_by_token = {c["token"]: c for c in categories}
    scene_by_token = {s["token"]: s for s in scenes}

    # Deterministic scene split for mini.
    scene_names = sorted([s["name"] for s in scenes])
    n_scene = len(scene_names)

    train_names = set(scene_names[:max(1, int(0.8 * n_scene))])
    val_names = set(scene_names[max(1, int(0.8 * n_scene)):max(2, int(0.9 * n_scene))])
    test_names = set(scene_names[max(2, int(0.9 * n_scene)):])

    if len(test_names) == 0:
        test_names = set(scene_names[-1:])
        train_names = set(scene_names[:-1])

    def scene_split(scene_name):
        if scene_name in train_names:
            return "train"
        if scene_name in val_names:
            return "val"
        return "test"

    # Group vehicle annotations by instance.
    tracks = {}

    for ann in anns:
        inst = instance_by_token[ann["instance_token"]]
        cat = category_by_token[inst["category_token"]]["name"]

        if not cat.startswith(VEHICLE_PREFIX):
            continue

        sample = sample_by_token[ann["sample_token"]]
        scene = scene_by_token[sample["scene_token"]]

        item = {
            "timestamp": sample["timestamp"] / 1e6,
            "xy": np.array(ann["translation"][:2], dtype=np.float32),
            "yaw": yaw_from_quat(ann["rotation"]),
            "scene_name": scene["name"],
            "instance_token": ann["instance_token"],
            "category": cat,
        }

        tracks.setdefault(ann["instance_token"], []).append(item)

    X_list = []
    y_list = []
    split_list = []
    scene_list = []
    instance_list = []
    category_list = []
    anchor_time_list = []

    skipped_short = 0
    skipped_static = 0

    for instance_token, items in tqdm(tracks.items(), desc="Building nuScenes agent windows"):
        items = sorted(items, key=lambda d: d["timestamp"])

        if len(items) < 16:
            skipped_short += 1
            continue

        times_abs = np.array([d["timestamp"] for d in items], dtype=np.float64)
        times = times_abs - times_abs[0]

        xy = np.stack([d["xy"] for d in items]).astype(np.float32)
        yaw = np.unwrap(np.array([d["yaw"] for d in items], dtype=np.float64))

        # Remove basically parked tracks.
        total_disp = float(np.linalg.norm(xy[-1] - xy[0]))
        duration = max(float(times[-1] - times[0]), 1e-6)
        avg_speed = total_disp / duration

        if avg_speed < 0.5:
            skipped_static += 1
            continue

        for anchor_idx in range(len(items)):
            t_anchor = times[anchor_idx]

            needed_start = t_anchor + OBS_TIMES[0]
            needed_end = t_anchor + FUT_TIMES[-1]

            if needed_start < times[0] or needed_end > times[-1]:
                continue

            target_obs_times = t_anchor + OBS_TIMES
            target_fut_times = t_anchor + FUT_TIMES

            obs_x = np.interp(target_obs_times, times, xy[:, 0])
            obs_y = np.interp(target_obs_times, times, xy[:, 1])
            fut_x = np.interp(target_fut_times, times, xy[:, 0])
            fut_y = np.interp(target_fut_times, times, xy[:, 1])

            obs_yaw = np.interp(target_obs_times, times, yaw)

            obs_world = np.stack([obs_x, obs_y], axis=1).astype(np.float32)
            fut_world = np.stack([fut_x, fut_y], axis=1).astype(np.float32)

            anchor_xy = obs_world[-1].copy()
            anchor_yaw = float(obs_yaw[-1])

            obs_local = world_to_local(obs_world, anchor_xy, anchor_yaw).astype(np.float32)
            fut_local = world_to_local(fut_world, anchor_xy, anchor_yaw).astype(np.float32)

            obs_heading_local = normalize_angle(obs_yaw - anchor_yaw).astype(np.float32)

            X = build_features(obs_local, obs_heading_local)
            y = fut_local.astype(np.float32)

            if not np.isfinite(X).all() or not np.isfinite(y).all():
                continue

            scene_name = items[anchor_idx]["scene_name"]

            X_list.append(X)
            y_list.append(y)
            split_list.append(scene_split(scene_name))
            scene_list.append(scene_name)
            instance_list.append(instance_token)
            category_list.append(items[anchor_idx]["category"])
            anchor_time_list.append(times_abs[anchor_idx])

    if len(X_list) == 0:
        raise RuntimeError("No nuScenes samples were built.")

    X = np.stack(X_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)
    split = np.array(split_list, dtype=object)

    np.savez_compressed(
        OUT_PATH,
        X=X,
        y=y,
        split=split,
        feature_names=FEATURE_NAMES,
        scene_ids=np.array(scene_list, dtype=object),
        instance_tokens=np.array(instance_list, dtype=object),
        categories=np.array(category_list, dtype=object),
        anchor_timestamps=np.array(anchor_time_list, dtype=np.float64),
        source=np.array(["nuscenes"] * len(X), dtype=object),
    )

    print("\nSaved:", OUT_PATH)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("feature_names:", FEATURE_NAMES.tolist())
    print("split counts:")
    for s, c in zip(*np.unique(split, return_counts=True)):
        print(" ", s, c)

    print("skipped_short:", skipped_short)
    print("skipped_static:", skipped_static)


if __name__ == "__main__":
    main()
