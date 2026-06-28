import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from common import (
    OUTPUT_STEPS,
    make_output_dirs,
    load_split_arrays,
    normalize_input_features,
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
METHOD_GROUP = "mlp"

RESULTS_DIR, PLOTS_DIR = make_output_dirs(PROJECT_ROOT, METHOD_GROUP)

RANDOM_STATE = 43


class MLPBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_steps=25):
        super().__init__()

        self.output_steps = output_steps

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, output_steps * 2)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        out = self.net(x)
        out = out.reshape(x.shape[0], self.output_steps, 2)

        return out


def run_mlp(hidden_dim=128, epochs=100, lr=1e-3, batch_size=64, num_plots=20):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    method_name = f"mlp_hidden_{hidden_dim}"

    arrays = load_split_arrays(DATA_PATH)

    X_train = arrays["X_train"]
    y_train = arrays["y_train"]

    X_val = arrays["X_val"]
    y_val = arrays["y_val"]

    X_test = arrays["X_test"]
    y_test = arrays["y_test"]
    idx_test = arrays["idx_test"]

    X_test_raw = X_test.copy()

    X_train, X_val, X_test, feature_mean, feature_std = normalize_input_features(
        X_train,
        X_val,
        X_test
    )

    input_dim = X_train.shape[1] * X_train.shape[2]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model = MLPBaseline(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_steps=OUTPUT_STEPS
    )

    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.shape[0]

        avg_loss = total_loss / len(train_dataset)

        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_tensor)
                val_loss = loss_fn(val_pred, y_val_tensor).item()
            model.train()

            print(
                f"{method_name} | Epoch {epoch:03d} | "
                f"Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

    model.eval()

    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)

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
        X_plot_tensor = torch.tensor(X_test_raw[i], dtype=torch.float32)
        y_true_tensor = y_test_tensor[i]
        y_pred_single = y_pred_tensor[i]

        ade = ADE(y_pred_single, y_true_tensor)
        fde = FDE(y_pred_single, y_true_tensor)
        h_err = horizon_errors(y_pred_single, y_true_tensor)

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
                X=X_plot_tensor,
                y_true=y_true_tensor,
                y_pred=y_pred_single,
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

    checkpoint_path = os.path.join(RESULTS_DIR, f"{method_name}_checkpoint.pt")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_steps": OUTPUT_STEPS,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "method_name": method_name
        },
        checkpoint_path
    )

    print("Saved checkpoint:", checkpoint_path)

    print(f"\nMethod: {method_name}")
    print("ADE:", mean_ade)
    print("FDE:", mean_fde)

    return {
        "method": method_name,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "lr": lr,
        "ade": mean_ade,
        "fde": mean_fde,
        "num_samples": len(X_test),
        "horizon_errors": mean_horizon_errors
    }


def main():
    results = []

    results.append(run_mlp(hidden_dim=64, epochs=100, lr=1e-3, num_plots=20))
    results.append(run_mlp(hidden_dim=128, epochs=100, lr=1e-3, num_plots=20))
    results.append(run_mlp(hidden_dim=256, epochs=100, lr=1e-3, num_plots=20))

    save_summary_results(
        results,
        RESULTS_DIR,
        METHOD_GROUP,
        fieldnames=[
            "method",
            "hidden_dim",
            "epochs",
            "lr",
            "ade",
            "fde",
            "num_samples"
        ]
    )

    save_horizon_results(results, RESULTS_DIR, METHOD_GROUP)

    plot_metric_comparison(
        results,
        PLOTS_DIR,
        METHOD_GROUP,
        title="MLP Baseline Comparison"
    )

    plot_horizon_comparison(results, PLOTS_DIR, METHOD_GROUP)

    print("\nSaved results to:", RESULTS_DIR)
    print("Saved plots to:", PLOTS_DIR)


if __name__ == "__main__":
    main()