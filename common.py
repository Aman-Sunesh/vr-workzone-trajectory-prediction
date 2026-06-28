import os
import csv
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

# Number of future points we predict
OUTPUT_STEPS = 25

# Number of past points we use as input
INPUT_STEPS = 10

# Time gap between each point
# 0.2 seconds means 5 Hz data
DT = 0.2

# Make a list of prediction times:
# [0.2, 0.4, 0.6, ..., 5.0]
HORIZON_TIMES = []

for i in range(OUTPUT_STEPS):
    time = (i + 1) * DT
    HORIZON_TIMES.append(time)


# Make CSV column names for each horizon error:
# err_0p2s, err_0p4s, ..., err_5p0s
HORIZON_FIELDS = []

for time in HORIZON_TIMES:
    time_string = f"{time:.1f}"          # example: 0.2
    time_string = time_string.replace(".", "p")  # example: 0p2
    field_name = f"err_{time_string}s"   # example: err_0p2s
    HORIZON_FIELDS.append(field_name)


def make_output_dirs(project_root, method_group):
    """
    Create folders where results and plots will be saved.
    """
    results_dir = os.path.join(
        project_root,
        "outputs",
        "results_workzone",
        method_group
    )

    plots_dir = os.path.join(
        project_root,
        "outputs",
        "plots_workzone",
        method_group
    )

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    return results_dir, plots_dir


def load_split_arrays(data_path):
    """
    Load X, y, and split labels from the .npz dataset.

    Then separate the data into train, validation, and test sets.
    """
    data = np.load(data_path, allow_pickle=True)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    splits = data["split"].astype(str)

    all_indices = np.arange(len(X))

    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    return {
        "X_train": X[train_mask],
        "y_train": y[train_mask],
        "idx_train": all_indices[train_mask],

        "X_val": X[val_mask],
        "y_val": y[val_mask],
        "idx_val": all_indices[val_mask],

        "X_test": X[test_mask],
        "y_test": y[test_mask],
        "idx_test": all_indices[test_mask],

        "data": data
    }


def normalize_input_features(X_train, X_val, X_test, eps=1e-6):
    """
    Normalize input features using training-set statistics only.

    Only X is normalized.
    y remains in original x/y coordinate scale so ADE/FDE stay meaningful.
    """
    feature_mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    feature_std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)

    feature_std[feature_std < eps] = 1.0

    X_train_norm = (X_train - feature_mean) / feature_std
    X_val_norm = (X_val - feature_mean) / feature_std
    X_test_norm = (X_test - feature_mean) / feature_std

    return X_train_norm, X_val_norm, X_test_norm, feature_mean, feature_std


def ADE(pred, true):
    errors = torch.norm(pred - true, dim=1)
    return errors.mean().item()

def FDE(pred, true):
    return torch.norm(pred[-1] - true[-1]).item()

def horizon_errors(pred, true):
    errors = torch.norm(pred - true, dim=1)
    return errors.detach().cpu().numpy().astype(float)


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def save_horizon_results(results, results_dir, method_group):
    """
    Save timestep-by-timestep errors for each method.

    Each row is one method.
    Each column is an error at a specific future time.
    """
    csv_path = os.path.join(results_dir, f"{method_group}_horizon_errors.csv")

    rows = []

    for result in results:
        row = {"method": result["method"]}

        for field, value in zip(HORIZON_FIELDS, result["horizon_errors"]):
            row[field] = float(value)

        rows.append(row)

    save_csv(
        csv_path,
        rows,
        fieldnames=["method"] + HORIZON_FIELDS
    )


def save_summary_results(results, results_dir, method_group, fieldnames):
    """
    Save summary metrics like ADE and FDE.

    This removes horizon_errors because that is saved separately.
    """
    csv_path = os.path.join(results_dir, f"{method_group}_summary.csv")
    json_path = os.path.join(results_dir, f"{method_group}_summary.json")

    summary_rows = [
        {k: v for k, v in r.items() if k != "horizon_errors"}
        for r in results
    ]

    save_csv(csv_path, summary_rows, fieldnames=fieldnames)

    with open(json_path, "w") as f:
        json.dump(summary_rows, f, indent=4)


def plot_trajectory_example(X, y_true, y_pred, method_name, idx, ade, fde, plots_dir):
    """
    Plot one trajectory example.

    Shows:
    1. Past trajectory
    2. True future trajectory
    3. Predicted future trajectory
    4. Current position
    """
    X_np = X.detach().cpu().numpy()
    X_xy = X_np[:, :2]

    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    current_np = X_xy[-1:]
    true_future_with_current = np.vstack([current_np, y_true_np])
    pred_future_with_current = np.vstack([current_np, y_pred_np])

    plt.figure(figsize=(7, 6))

    plt.plot(
        X_xy[:, 0],
        X_xy[:, 1],
        marker="o",
        label="Past trajectory"
    )

    plt.plot(
        true_future_with_current[:, 0],
        true_future_with_current[:, 1],
        marker="o",
        label="True future"
    )

    plt.plot(
        pred_future_with_current[:, 0],
        pred_future_with_current[:, 1],
        marker="x",
        label="Predicted future"
    )

    plt.scatter(
        X_xy[-1, 0],
        X_xy[-1, 1],
        marker="s",
        label="Current position"
    )

    plt.title(f"{method_name} | sample {idx}\nADE={ade:.3f}, FDE={fde:.3f}")
    plt.xlabel("x position")
    plt.ylabel("y position")

    all_xy = np.vstack([X_xy, y_true_np, y_pred_np])

    x_min, y_min = all_xy.min(axis=0)
    x_max, y_max = all_xy.max(axis=0)

    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)

    span = max(x_max - x_min, y_max - y_min, 1.0)
    span = span * 1.15

    plt.xlim(x_center - span / 2, x_center + span / 2)
    plt.ylim(y_center - span / 2, y_center + span / 2)

    plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(plots_dir, f"{method_name}_sample_{idx}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_metric_comparison(results, plots_dir, method_group, title):
    """
    Make a bar chart comparing ADE and FDE for all methods.
    """
    methods = [r["method"] for r in results]
    ades = [r["ade"] for r in results]
    fdes = [r["fde"] for r in results]

    x = np.arange(len(methods))
    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(x - width / 2, ades, width, label="ADE")
    plt.bar(x + width / 2, fdes, width, label="FDE")

    plt.xticks(x, methods, rotation=20, ha="right")
    plt.ylabel("Error")
    plt.title(title)
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()

    save_path = os.path.join(plots_dir, f"{method_group}_metrics_comparison.png")
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_horizon_comparison(results, plots_dir, method_group):
    """
    Plot how prediction error changes over time.

    Usually, error increases as prediction horizon gets longer.
    """
    plt.figure(figsize=(9, 6))

    for result in results:
        plt.plot(
            HORIZON_TIMES,
            result["horizon_errors"],
            marker="o",
            label=result["method"]
        )

    plt.xlabel("Prediction horizon (seconds)")
    plt.ylabel("Mean displacement error")
    plt.title(f"{method_group}: Error by Prediction Horizon")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(plots_dir, f"{method_group}_horizon_errors.png")
    plt.savefig(save_path, dpi=200)
    plt.close()