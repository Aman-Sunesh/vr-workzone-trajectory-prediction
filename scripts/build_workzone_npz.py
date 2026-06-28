import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

DATA_DIR = Path(PROJECT_ROOT) / "data"
DATA_FINAL_DIR = DATA_DIR / "data_final"

# CSV file that tells us which samples are train/val/test
SPLIT_INDEX_PATH = DATA_DIR / "split_index.csv"

# Output .npz file that this script will create
OUT_PATH = DATA_DIR / "workzone_traj_motion_agents_map.npz"

# Target sampling rate
# 5 Hz means 5 points per second
TARGET_HZ = 5.0

# Time between two consecutive points
# Since 5 Hz = 5 samples per second, DT = 0.2 seconds
DT = 1.0 / TARGET_HZ

OBS_STEPS = 10   # Number of observed/past timesteps
FUTURE_STEPS = 25   # Number of future timesteps to predict

NEARBY_RADIUS = 80.0   # Search radius for nearby agents
MAX_LANE_SEARCH_RADIUS = 50.0   # Maximum distance used when searching for nearest lane



def normalize_angle(angle):
    """
    Normalize an angle to the range [-pi, pi].

    This prevents angles from becoming too large,
    for example converting 4*pi into 0.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def local_to_world(points_local, anchor_xy, anchor_yaw):
    """
    Convert points from local coordinates to world coordinates.

    points_local:
        Points in the ego/local frame.

    anchor_xy:
        World x/y position of the local frame origin.

    anchor_yaw:
        Rotation angle of the local frame in the world.
    """
    c = np.cos(anchor_yaw)
    s = np.sin(anchor_yaw)

    R = np.array(
        [
            [c, -s],
            [s,  c]
        ],
        dtype=np.float32
    )

    return points_local @ R.T + anchor_xy


def find_ego_idx(agent_ids, agent_types):
    """
    Find which agent is the ego vehicle.

    First, we check if agent ID 1 exists.
    If not, we use the first agent whose type is VEHICLE.
    """
    if 1 in agent_ids:
        return agent_ids.index(1)

    for i, agent_type in enumerate(agent_types):
        if str(agent_type).upper() == "VEHICLE":
            return i

    return None


def compute_velocity_and_acceleration(xy):
    """
    Compute velocity and acceleration from x/y positions.

    xy shape:
        [time_steps, 2]

    velocity:
        change in position / time

    acceleration:
        change in velocity / time
    """
    velocity = np.zeros_like(xy, dtype=np.float32)
    velocity[1:] = (xy[1:] - xy[:-1]) / DT
    velocity[0] = velocity[1]

    acceleration = np.zeros_like(xy, dtype=np.float32)
    acceleration[1:] = (velocity[1:] - velocity[:-1]) / DT
    acceleration[0] = acceleration[1]

    return velocity, acceleration


def as_xy_points(obj):
    """
    Convert different point formats into a clean numpy array of x/y points.

    Output shape:
        [num_points, 2]

    If the input is bad or empty, return an empty array.
    """
    if obj is None:
        return np.zeros((0, 2), dtype=np.float32)

    try:
        arr = np.asarray(obj, dtype=np.float32)
    except Exception:
        return np.zeros((0, 2), dtype=np.float32)

    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)

    arr = arr[:, :2]
    arr = arr[np.isfinite(arr).all(axis=1)]

    return arr.astype(np.float32)


def nearest_point_features(points, ego_xy, max_radius):
    """
    Find the nearest point to the ego vehicle.

    This is used for things like:
    - cones
    - workers
    - warning signs

    Returned features:
    [
        nearest_dx,
        nearest_dy,
        nearest_distance,
        number_of_points,
        has_point
    ]
    """
    points = as_xy_points(points)

    if len(points) == 0:
        return np.array(
            [0.0, 0.0, max_radius, 0.0, 0.0],
            dtype=np.float32
        )

    rel = points - ego_xy.reshape(1, 2)
    dists = np.linalg.norm(rel, axis=1)

    idx = int(np.argmin(dists))
    dist = float(min(dists[idx], max_radius))

    return np.array(
        [
            rel[idx, 0],
            rel[idx, 1],
            dist,
            float(len(points)),
            1.0
        ],
        dtype=np.float32
    )


def build_lane_features(map_dict, ego_xy, ego_heading):
    """
    Build lane-related features for the ego vehicle.

    Returned features:
    [
        nearest_lane_dist,
        lane_dir_x,
        lane_dir_y,
        lane_align_cos,
        lane_align_sin,
        has_lane
    ]
    """
    default_features = np.array(
        [MAX_LANE_SEARCH_RADIUS, 1.0, 0.0, 1.0, 0.0, 0.0],
        dtype=np.float32
    )

    lanes = map_dict.get("lanes", [])

    if lanes is None or len(lanes) == 0:
        return default_features

    best_dist = float("inf")
    best_dir = np.array([1.0, 0.0], dtype=np.float32)

    for lane in lanes:
        polyline = lane.get("polyline", [])
        polyline = as_xy_points(polyline)

        if len(polyline) < 2:
            continue

        dists = np.linalg.norm(polyline - ego_xy.reshape(1, 2), axis=1)
        closest_idx = int(np.argmin(dists))
        dist = float(dists[closest_idx])

        if dist < best_dist:
            best_dist = dist

            if closest_idx == 0:
                tangent = polyline[1] - polyline[0]
            elif closest_idx == len(polyline) - 1:
                tangent = polyline[-1] - polyline[-2]
            else:
                tangent = polyline[closest_idx + 1] - polyline[closest_idx - 1]

            norm = np.linalg.norm(tangent)

            if norm > 1e-6:
                best_dir = tangent / norm

    if not np.isfinite(best_dist):
        return default_features

    best_dist = min(best_dist, MAX_LANE_SEARCH_RADIUS)

    ego_dir = np.array(
        [np.cos(ego_heading), np.sin(ego_heading)],
        dtype=np.float32
    )

    lane_align_cos = float(np.dot(best_dir, ego_dir))
    lane_align_cos = float(np.clip(lane_align_cos, -1.0, 1.0))
    lane_align_sin = float(np.sqrt(max(0.0, 1.0 - lane_align_cos ** 2)))

    return np.array(
        [
            best_dist,
            best_dir[0],
            best_dir[1],
            lane_align_cos,
            lane_align_sin,
            1.0
        ],
        dtype=np.float32
    )


def build_wz_corner_features(map_dict, ego_xy):
    """
    Build features for work-zone corners.

    It finds the center of all work-zone corners,
    then measures where that center is relative to the ego vehicle.

    Returned features:
    [
        wz_center_dx,
        wz_center_dy,
        wz_center_dist,
        has_wz_corners
    ]
    """
    corners = as_xy_points(map_dict.get("wz_corners", None))

    if len(corners) == 0:
        return np.array([0.0, 0.0, NEARBY_RADIUS, 0.0], dtype=np.float32)

    center = corners.mean(axis=0)
    rel = center - ego_xy
    dist = float(np.linalg.norm(rel))

    return np.array(
        [rel[0], rel[1], dist, 1.0],
        dtype=np.float32
    )


def build_nearby_agent_features(agents, ego_idx, t):
    """
    Build nearby-agent features for one timestep.

    It counts nearby agents and finds:
    - nearest pedestrian
    - nearest vehicle

    Returned features:
    [
        nearby_agent_count,

        nearest_ped_dx,
        nearest_ped_dy,
        nearest_ped_dist,
        nearest_ped_has,

        nearest_vehicle_dx,
        nearest_vehicle_dy,
        nearest_vehicle_dist,
        nearest_vehicle_has
    ]
    """
    ids = agents["ids"]
    types = agents["types"]
    obs_xy = agents["obs_xy"]
    obs_mask = agents["obs_mask"]

    ego_xy = obs_xy[t, ego_idx]

    nearby_count = 0.0

    nearest_ped_rel = np.array([0.0, 0.0], dtype=np.float32)
    nearest_ped_dist = NEARBY_RADIUS
    nearest_ped_has = 0.0

    nearest_vehicle_rel = np.array([0.0, 0.0], dtype=np.float32)
    nearest_vehicle_dist = NEARBY_RADIUS
    nearest_vehicle_has = 0.0

    for j in range(len(ids)):
        if j == ego_idx:
            continue

        if not obs_mask[t, j]:
            continue

        other_xy = obs_xy[t, j]

        if not np.isfinite(other_xy).all():
            continue

        rel = other_xy - ego_xy
        dist = float(np.linalg.norm(rel))

        if dist > NEARBY_RADIUS:
            continue

        nearby_count += 1.0
        agent_type = str(types[j]).upper()

        if agent_type == "PEDESTRIAN":
            if dist < nearest_ped_dist:
                nearest_ped_dist = dist
                nearest_ped_rel = rel
                nearest_ped_has = 1.0

        elif agent_type == "VEHICLE":
            if dist < nearest_vehicle_dist:
                nearest_vehicle_dist = dist
                nearest_vehicle_rel = rel
                nearest_vehicle_has = 1.0

    return np.array(
        [
            nearby_count,

            nearest_ped_rel[0],
            nearest_ped_rel[1],
            nearest_ped_dist,
            nearest_ped_has,

            nearest_vehicle_rel[0],
            nearest_vehicle_rel[1],
            nearest_vehicle_dist,
            nearest_vehicle_has,
        ],
        dtype=np.float32
    )


def build_map_features(map_dict, ego_xy, ego_heading):
    """
    Build all map-related features.

    This includes:
    - nearest lane
    - work-zone corner center
    - nearest cone
    - nearest worker
    - nearest warning sign
    """
    lane_features = build_lane_features(
        map_dict=map_dict,
        ego_xy=ego_xy,
        ego_heading=ego_heading
    )

    wz_corner_features = build_wz_corner_features(
        map_dict=map_dict,
        ego_xy=ego_xy
    )

    cone_features = nearest_point_features(
        map_dict.get("wz_cones", None),
        ego_xy,
        NEARBY_RADIUS
    )

    worker_features = nearest_point_features(
        map_dict.get("wz_workers", None),
        ego_xy,
        NEARBY_RADIUS
    )

    warning_features = nearest_point_features(
        map_dict.get("wz_warning", None),
        ego_xy,
        NEARBY_RADIUS
    )

    return np.concatenate(
        [
            lane_features,
            wz_corner_features,
            cone_features,
            worker_features,
            warning_features
        ],
        axis=0
    ).astype(np.float32)


def build_feature_names():
    """
    Return the name of every input feature in X.

    These names match the order of features created in build_sample_features().
    """
    names = [
        "x", "y",
        "vx", "vy",
        "ax", "ay",
        "cos_heading", "sin_heading",
        "speed",

        "nearby_agent_count",

        "nearest_ped_dx",
        "nearest_ped_dy",
        "nearest_ped_dist",
        "nearest_ped_has",

        "nearest_vehicle_dx",
        "nearest_vehicle_dy",
        "nearest_vehicle_dist",
        "nearest_vehicle_has",

        "nearest_lane_dist",
        "lane_dir_x",
        "lane_dir_y",
        "lane_align_cos",
        "lane_align_sin",
        "has_lane",

        "wz_center_dx",
        "wz_center_dy",
        "wz_center_dist",
        "has_wz_corners",

        "nearest_cone_dx",
        "nearest_cone_dy",
        "nearest_cone_dist",
        "cone_count",
        "has_cone",

        "nearest_worker_dx",
        "nearest_worker_dy",
        "nearest_worker_dist",
        "worker_count",
        "has_worker",

        "nearest_warning_dx",
        "nearest_warning_dy",
        "nearest_warning_dist",
        "warning_count",
        "has_warning",
    ]

    return names


def build_sample_features(sample):
    """
    Build one training sample.

    Input:
        sample from a .pkl file

    Output:
        A dictionary containing:
        - X: input features
        - y: future trajectory
        - world-coordinate metadata
        - timestamps

    If the sample is invalid, return None.
    """
    agents = sample["agents"]
    map_dict = sample["map"]
    ego_ref = sample["ego_ref"]

    ego_idx = find_ego_idx(
        agent_ids=agents["ids"],
        agent_types=agents["types"]
    )

    if ego_idx is None:
        return None

    obs_xy_all = agents["obs_xy"]
    fut_xy_all = agents["fut_xy"]

    obs_mask = agents["obs_mask"]
    fut_mask = agents["fut_mask"]

    if obs_xy_all.shape[0] != OBS_STEPS:
        return None

    if fut_xy_all.shape[0] != FUTURE_STEPS:
        return None

    if not obs_mask[:, ego_idx].all():
        return None

    if not fut_mask[:, ego_idx].all():
        return None

    ego_obs_xy = obs_xy_all[:, ego_idx, :].astype(np.float32)
    ego_fut_xy = fut_xy_all[:, ego_idx, :].astype(np.float32)

    if not np.isfinite(ego_obs_xy).all():
        return None

    if not np.isfinite(ego_fut_xy).all():
        return None

    velocity, acceleration = compute_velocity_and_acceleration(ego_obs_xy)

    if "obs_heading" in agents:
        heading = agents["obs_heading"][:, ego_idx].astype(np.float32)
        heading = np.nan_to_num(heading, nan=0.0)
    else:
        heading = np.zeros(OBS_STEPS, dtype=np.float32)
        deltas = ego_obs_xy[1:] - ego_obs_xy[:-1]
        heading[1:] = np.arctan2(deltas[:, 1], deltas[:, 0])
        heading[0] = heading[1]

    if "obs_speed" in agents:
        speed = agents["obs_speed"][:, ego_idx].astype(np.float32)
        fallback_speed = np.linalg.norm(velocity, axis=1)
        speed = np.where(np.isfinite(speed), speed, fallback_speed)
    else:
        speed = np.linalg.norm(velocity, axis=1)

    X_steps = []

    for t in range(OBS_STEPS):
        motion_features = np.array(
            [
                ego_obs_xy[t, 0],
                ego_obs_xy[t, 1],
                velocity[t, 0],
                velocity[t, 1],
                acceleration[t, 0],
                acceleration[t, 1],
                np.cos(heading[t]),
                np.sin(heading[t]),
                speed[t],
            ],
            dtype=np.float32
        )

        nearby_features = build_nearby_agent_features(
            agents=agents,
            ego_idx=ego_idx,
            t=t
        )

        map_features = build_map_features(
            map_dict=map_dict,
            ego_xy=ego_obs_xy[t],
            ego_heading=heading[t]
        )

        features = np.concatenate(
            [
                motion_features,
                nearby_features,
                map_features
            ],
            axis=0
        ).astype(np.float32)

        features = np.nan_to_num(
            features,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        X_steps.append(features)

    X = np.stack(X_steps).astype(np.float32)
    y = ego_fut_xy.astype(np.float32)

    anchor_xy = np.asarray(ego_ref["pos_world"], dtype=np.float32)
    anchor_yaw = float(ego_ref["heading_world"])

    obs_world_xy = local_to_world(
        ego_obs_xy,
        anchor_xy=anchor_xy,
        anchor_yaw=anchor_yaw
    )

    future_world_xy = local_to_world(
        ego_fut_xy,
        anchor_xy=anchor_xy,
        anchor_yaw=anchor_yaw
    )

    t_anchor = float(sample["meta"]["t_anchor"])

    obs_timestamps = np.array(
        [
            t_anchor - (OBS_STEPS - 1 - i) * DT
            for i in range(OBS_STEPS)
        ],
        dtype=np.float64
    )

    future_timestamps = np.array(
        [
            t_anchor + (i + 1) * DT
            for i in range(FUTURE_STEPS)
        ],
        dtype=np.float64
    )

    return {
        "X": X,
        "y": y,
        "obs_world_xy": obs_world_xy.astype(np.float32),
        "future_world_xy": future_world_xy.astype(np.float32),
        "obs_timestamps": obs_timestamps,
        "future_timestamps": future_timestamps,
        "anchor_world_xy": anchor_xy.astype(np.float32),
        "anchor_yaw": np.float32(anchor_yaw)
    }


def main():
    if not SPLIT_INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing {SPLIT_INDEX_PATH}")

    split_index = pd.read_csv(SPLIT_INDEX_PATH)

    split_by_sample = dict(
        zip(
            split_index["sample"].astype(str),
            split_index["split"].astype(str)
        )
    )

    pkl_paths = list(DATA_FINAL_DIR.rglob("*.pkl"))
    pkl_by_stem = {p.stem: p for p in pkl_paths}

    feature_names = build_feature_names()

    X_list = []
    y_list = []

    sample_names = []
    scene_ids = []
    participants = []
    scenarios = []
    wz_values = []
    sample_indices = []
    splits = []

    anchor_timestamps = []
    anchor_world_xy = []
    anchor_yaws = []
    obs_timestamps = []
    future_timestamps = []
    obs_world_xy = []
    future_world_xy = []
    meta_list = []

    missing_pkl = 0
    skipped = 0

    for _, row in tqdm(split_index.iterrows(), total=len(split_index), desc="Building work-zone npz"):
        sample_name = str(row["sample"])

        if sample_name not in pkl_by_stem:
            missing_pkl += 1
            continue

        pkl_path = pkl_by_stem[sample_name]

        with open(pkl_path, "rb") as f:
            sample = pickle.load(f)

        built = build_sample_features(sample)

        if built is None:
            skipped += 1
            continue

        meta = sample["meta"]

        X_list.append(built["X"])
        y_list.append(built["y"])

        sample_names.append(sample_name)
        scene_ids.append(str(meta["scene_id"]))
        participants.append(str(meta["participant"]))
        scenarios.append(str(meta["scenario"]))
        wz_values.append(int(meta["wz"]))
        sample_indices.append(int(meta["sample_idx"]))
        splits.append(split_by_sample[sample_name])

        anchor_timestamps.append(float(meta["t_anchor"]))
        anchor_world_xy.append(built["anchor_world_xy"])
        anchor_yaws.append(built["anchor_yaw"])
        obs_timestamps.append(built["obs_timestamps"])
        future_timestamps.append(built["future_timestamps"])
        obs_world_xy.append(built["obs_world_xy"])
        future_world_xy.append(built["future_world_xy"])

        meta_list.append(meta)

    if len(X_list) == 0:
        raise RuntimeError("No valid samples were built.")

    X = np.stack(X_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)

    np.savez_compressed(
        OUT_PATH,
        X=X,
        y=y,
        feature_names=np.array(feature_names, dtype=object),
        meta=np.array(meta_list, dtype=object),

        sample_names=np.array(sample_names, dtype=object),
        scene_ids=np.array(scene_ids, dtype=object),
        participants=np.array(participants, dtype=object),
        scenarios=np.array(scenarios, dtype=object),
        wz=np.array(wz_values, dtype=np.int64),
        sample_indices=np.array(sample_indices, dtype=np.int64),
        split=np.array(splits, dtype=object),

        anchor_timestamps=np.array(anchor_timestamps, dtype=np.float64),
        anchor_world_xy=np.stack(anchor_world_xy).astype(np.float32),
        anchor_yaws=np.array(anchor_yaws, dtype=np.float32),
        obs_timestamps=np.stack(obs_timestamps).astype(np.float64),
        future_timestamps=np.stack(future_timestamps).astype(np.float64),
        obs_world_xy=np.stack(obs_world_xy).astype(np.float32),
        future_world_xy=np.stack(future_world_xy).astype(np.float32),
    )

    print("\nSaved:", OUT_PATH)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Feature count:", len(feature_names))
    print("Feature names:", feature_names)
    print("Missing pkl:", missing_pkl)
    print("Skipped samples:", skipped)

    unique_splits, counts = np.unique(np.array(splits), return_counts=True)
    print("Split counts:")
    for split, count in zip(unique_splits, counts):
        print(split, count)


if __name__ == "__main__":
    main()