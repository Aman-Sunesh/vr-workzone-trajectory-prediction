import os
import sys
import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from common import (
    OUTPUT_STEPS,
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
METHOD_GROUP = "linear_regression"

RESULTS_DIR, PLOTS_DIR = make_output_dirs(PROJECT_ROOT, METHOD_GROUP)

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def linear_regression(ridge=False, num_plots=20):
    arrays = load_split_arrays(DATA_PATH)

    X_train = arrays["X_train"]
    y_train = arrays["y_train"]

    X_val = arrays["X_val"]
    y_val = arrays["y_val"]

    X_test = arrays["X_test"]
    y_test = arrays["y_test"]
    idx_test = arrays["idx_test"]

    X_train_flat = X_train.reshape(len(X_train), -1)
    y_train_flat = y_train.reshape(len(y_train), -1)

    X_val_flat = X_val.reshape(len(X_val), -1)
    y_val_flat = y_val.reshape(len(y_val), -1)

    X_test_flat = X_test.reshape(len(X_test), -1)

    if not ridge:
        method_name = "linear_regression"
        best_alpha = None

        model = LinearRegression()
        model.fit(X_train_flat, y_train_flat)

    else:
        method_name = "ridge_regression"
        best_alpha = None
        best_val_ade = float("inf")

        for alpha in RIDGE_ALPHAS:
            candidate_model = Ridge(alpha=alpha)
            candidate_model.fit(X_train_flat, y_train_flat)

            val_pred = candidate_model.predict(X_val_flat).reshape(-1, OUTPUT_STEPS, 2)
            val_true = y_val.reshape(-1, OUTPUT_STEPS, 2)

            val_errors = np.linalg.norm(val_pred - val_true, axis=2)
            val_ade = float(val_errors.mean())

            print(f"Ridge alpha={alpha} | validation ADE={val_ade:.4f}")

            if val_ade < best_val_ade:
                best_val_ade = val_ade
                best_alpha = alpha

        print(f"Best Ridge alpha: {best_alpha}")

        X_train_val_flat = np.concatenate([X_train_flat, X_val_flat], axis=0)
        y_train_val_flat = np.concatenate([y_train_flat, y_val_flat], axis=0)

        model = Ridge(alpha=best_alpha)
        model.fit(X_train_val_flat, y_train_val_flat)

    y_pred = model.predict(X_test_flat).reshape(-1, OUTPUT_STEPS, 2)

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
        X_tensor = torch.tensor(X_test[i], dtype=torch.float32)
        y_true_tensor = torch.tensor(y_test[i], dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred[i], dtype=torch.float32)

        ade = ADE(y_pred_tensor, y_true_tensor)
        fde = FDE(y_pred_tensor, y_true_tensor)
        h_err = horizon_errors(y_pred_tensor, y_true_tensor)

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
                X=X_tensor,
                y_true=y_true_tensor,
                y_pred=y_pred_tensor,
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
        "alpha": best_alpha,
        "ade": mean_ade,
        "fde": mean_fde,
        "num_samples": len(X_test),
        "horizon_errors": mean_horizon_errors
    }


def main():
    results = []

    results.append(linear_regression(ridge=False, num_plots=20))
    results.append(linear_regression(ridge=True, num_plots=20))

    save_summary_results(
        results,
        RESULTS_DIR,
        METHOD_GROUP,
        fieldnames=["method", "alpha", "ade", "fde", "num_samples"]
    )

    save_horizon_results(results, RESULTS_DIR, METHOD_GROUP)

    plot_metric_comparison(
        results,
        PLOTS_DIR,
        METHOD_GROUP,
        title="Linear / Ridge Regression Baseline Comparison"
    )

    plot_horizon_comparison(results, PLOTS_DIR, METHOD_GROUP)

    print("\nSaved results to:", RESULTS_DIR)
    print("Saved plots to:", PLOTS_DIR)


if __name__ == "__main__":
    main()