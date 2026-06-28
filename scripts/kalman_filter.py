import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from common import (
    OUTPUT_STEPS,
    DT,
    make_output_dirs,
    load_split_arrays,
    ADE,
    FDE,
    horizon_errors,
    save_csv,
    save_summary_results,
    save_horizon_results,
    plot_trajectory_example,
    plot_metric_comparison,
    plot_horizon_comparison
)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "workzone_traj_motion_agents_map.npz")
METHOD_GROUP = "kalman_filter"

RESULTS_DIR, PLOTS_DIR = make_output_dirs(PROJECT_ROOT, METHOD_GROUP)


def kalman_cv_predict(obs_xy, process_var=1.0, measurement_var=2.0):
    """
    Linear Kalman Filter with constant velocity state:
    state = [x, y, vx, vy]
    """
    obs_xy = np.asarray(obs_xy, dtype=np.float32)

    if len(obs_xy) < 2:
        current = obs_xy[-1]
        return np.repeat(current.reshape(1, 2), OUTPUT_STEPS, axis=0)

    initial_velocity = (obs_xy[1] - obs_xy[0]) / DT

    x = np.array(
        [
            obs_xy[0, 0],
            obs_xy[0, 1],
            initial_velocity[0],
            initial_velocity[1]
        ],
        dtype=np.float32
    )

    P = np.eye(4, dtype=np.float32) * 10.0

    F = np.array(
        [
            [1.0, 0.0, DT, 0.0],
            [0.0, 1.0, 0.0, DT],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32
    )

    H = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32
    )

    Q = np.eye(4, dtype=np.float32) * process_var
    R = np.eye(2, dtype=np.float32) * measurement_var

    for z in obs_xy:
        x = F @ x
        P = F @ P @ F.T + Q

        z = np.asarray(z, dtype=np.float32)

        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        x = x + K @ y
        P = (np.eye(4, dtype=np.float32) - K @ H) @ P

    preds = []

    for _ in range(OUTPUT_STEPS):
        x = F @ x
        preds.append(x[:2].copy())

    return np.stack(preds).astype(np.float32)


def ekf_ctrv_predict(X):
    """
    Simple EKF-style CTRV rollout using heading/speed features.
    This is a motion-model baseline, not a learned model.

    Assumes feature columns:
    cos_heading = 6
    sin_heading = 7
    speed       = 8
    """
    X = np.asarray(X, dtype=np.float32)
    obs_xy = X[:, :2]

    current = obs_xy[-1].copy()

    heading = float(np.arctan2(X[-1, 7], X[-1, 6]))
    prev_heading = float(np.arctan2(X[-2, 7], X[-2, 6]))

    yaw_rate = normalize_angle(heading - prev_heading) / DT
    speed = float(X[-1, 8])

    if not np.isfinite(speed):
        speed = float(np.linalg.norm(obs_xy[-1] - obs_xy[-2]) / DT)

    preds = []
    px, py = float(current[0]), float(current[1])
    yaw = heading

    for _ in range(OUTPUT_STEPS):
        if abs(yaw_rate) < 1e-4:
            px = px + speed * DT * np.cos(yaw)
            py = py + speed * DT * np.sin(yaw)
        else:
            px = px + (speed / yaw_rate) * (
                np.sin(yaw + yaw_rate * DT) - np.sin(yaw)
            )
            py = py + (speed / yaw_rate) * (
                -np.cos(yaw + yaw_rate * DT) + np.cos(yaw)
            )
            yaw = yaw + yaw_rate * DT

        preds.append([px, py])

    return np.asarray(preds, dtype=np.float32)


def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def run_filter(method_name, num_plots=20):
    arrays = load_split_arrays(DATA_PATH)

    X_test = arrays["X_test"]
    y_test = arrays["y_test"]
    idx_test = arrays["idx_test"]

    ade_list = []
    fde_list = []
    horizon_error_rows = []
    per_sample_rows = []

    plot_positions = np.linspace(
        0,
        len(X_test) - 1,
        min(num_plots, len(X_test)),
        dtype=int
    )

    for i in range(len(X_test)):
        X_np = X_test[i]
        y_np = y_test[i]

        if method_name == "kalman_cv":
            y_pred_np = kalman_cv_predict(X_np[:, :2])

        elif method_name == "ekf_ctrv":
            y_pred_np = ekf_ctrv_predict(X_np)

        else:
            raise ValueError(f"Unknown method: {method_name}")

        X = torch.tensor(X_np, dtype=torch.float32)
        y_true = torch.tensor(y_np, dtype=torch.float32)
        y_pred = torch.tensor(y_pred_np, dtype=torch.float32)

        ade = ADE(y_pred, y_true)
        fde = FDE(y_pred, y_true)
        h_err = horizon_errors(y_pred, y_true)

        ade_list.append(ade)
        fde_list.append(fde)
        horizon_error_rows.append(h_err)

        original_idx = int(idx_test[i])

        per_sample_rows.append({
            "sample_idx": original_idx,
            "ade": ade,
            "fde": fde
        })

        if i in plot_positions:
            plot_trajectory_example(
                X=X,
                y_true=y_true,
                y_pred=y_pred,
                method_name=method_name,
                idx=original_idx,
                ade=ade,
                fde=fde,
                plots_dir=PLOTS_DIR
            )

    mean_ade = float(np.mean(ade_list))
    mean_fde = float(np.mean(fde_list))
    mean_horizon_errors = np.mean(np.stack(horizon_error_rows), axis=0)

    save_csv(
        os.path.join(RESULTS_DIR, f"{method_name}_per_sample.csv"),
        per_sample_rows,
        fieldnames=["sample_idx", "ade", "fde"]
    )

    print(f"\nMethod: {method_name}")
    print("ADE:", mean_ade)
    print("FDE:", mean_fde)

    return {
        "method": method_name,
        "ade": mean_ade,
        "fde": mean_fde,
        "num_samples": len(X_test),
        "horizon_errors": mean_horizon_errors
    }


def main():
    results = []

    results.append(run_filter("kalman_cv", num_plots=20))
    results.append(run_filter("ekf_ctrv", num_plots=20))

    save_summary_results(
        results,
        RESULTS_DIR,
        METHOD_GROUP,
        fieldnames=["method", "ade", "fde", "num_samples"]
    )

    save_horizon_results(results, RESULTS_DIR, METHOD_GROUP)

    plot_metric_comparison(
        results,
        PLOTS_DIR,
        METHOD_GROUP,
        title="Kalman / EKF Baseline Comparison"
    )

    plot_horizon_comparison(results, PLOTS_DIR, METHOD_GROUP)

    print("\nSaved results to:", RESULTS_DIR)
    print("Saved plots to:", PLOTS_DIR)


if __name__ == "__main__":
    main()