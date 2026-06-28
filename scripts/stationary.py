import os
import sys
import numpy as np
import torch

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
METHOD_GROUP = "stationary"

RESULTS_DIR, PLOTS_DIR = make_output_dirs(PROJECT_ROOT, METHOD_GROUP)


def run_stationary(num_plots=20):
    method_name = "stationary_last_position"

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
        X = torch.tensor(X_test[i], dtype=torch.float32)
        y_true = torch.tensor(y_test[i], dtype=torch.float32)

        current_pos = X[-1, :2]
        y_pred = current_pos.reshape(1, 2).repeat(OUTPUT_STEPS, 1)

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

    results.append(run_stationary(num_plots=20))

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
        title="Stationary Baseline Comparison"
    )

    plot_horizon_comparison(results, PLOTS_DIR, METHOD_GROUP)

    print("\nSaved results to:", RESULTS_DIR)
    print("Saved plots to:", PLOTS_DIR)


if __name__ == "__main__":
    main()