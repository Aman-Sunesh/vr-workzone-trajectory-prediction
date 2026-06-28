import os
import sys
import json
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from common import (
    OUTPUT_STEPS,
    HORIZON_TIMES,
    HORIZON_FIELDS,
    make_output_dirs,
    load_split_arrays,
    normalize_input_features,
    horizon_errors,
    plot_trajectory_example
)

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", required=True)
parser.add_argument("--method-group", required=True)
args = parser.parse_args()

DATA_PATH = args.data_path
METHOD_GROUP = args.method_group

RESULTS_DIR, PLOTS_DIR = make_output_dirs(PROJECT_ROOT, METHOD_GROUP)

RANDOM_STATE = 43


class LSTMBaseline(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=1,
        output_steps=25,
        K=6,
        dropout=0.1
    ):
        super().__init__()

        self.output_steps = output_steps
        self.K = K

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.traj_head = nn.Linear(hidden_dim, self.K * output_steps * 2)
        self.conf_head = nn.Linear(hidden_dim, self.K)

    def forward(self, x):
        _, (h, c) = self.lstm(x)

        last_hidden = h[-1]

        preds = self.traj_head(last_hidden)
        preds = preds.reshape(x.shape[0], self.K, self.output_steps, 2)

        logits = self.conf_head(last_hidden)

        return preds, logits


def topk_wta_conf_loss(preds, logits, true, alpha=0.1):
    """
    preds:  [B, K, T, 2]
    logits: [B, K]
    true:   [B, T, 2]

    Winner-takes-all top-K loss.
    """
    errors = torch.norm(preds - true[:, None, :, :], dim=-1)
    ade_per_k = errors.mean(dim=-1)

    best_k = ade_per_k.argmin(dim=1)
    batch_idx = torch.arange(preds.shape[0], device=preds.device)

    best_preds = preds[batch_idx, best_k]

    loss_reg = nn.functional.smooth_l1_loss(best_preds, true)
    loss_cls = nn.functional.cross_entropy(logits, best_k)

    loss = loss_reg + alpha * loss_cls

    return loss


def topk_metrics_single(preds, logits, true):
    """
    Compute top-K metrics for one sample.
    """
    errors = torch.norm(preds - true[None, :, :], dim=-1)

    ade_per_k = errors.mean(dim=-1)
    fde_per_k = errors[:, -1]

    best_ade_k = int(torch.argmin(ade_per_k).item())
    best_fde_k = int(torch.argmin(fde_per_k).item())

    probs = torch.softmax(logits, dim=0)
    conf_k = int(torch.argmax(probs).item())
    conf_prob = float(probs[conf_k].item())

    min_ade = float(ade_per_k[best_ade_k].item())
    min_fde = float(fde_per_k[best_fde_k].item())

    conf_ade = float(ade_per_k[conf_k].item())
    conf_fde = float(fde_per_k[conf_k].item())

    best_pred_by_ade = preds[best_ade_k]
    conf_pred = preds[conf_k]

    h_err = horizon_errors(best_pred_by_ade, true)

    return {
        "min_ade": min_ade,
        "min_fde": min_fde,
        "best_ade_k": best_ade_k,
        "best_fde_k": best_fde_k,
        "conf_ade": conf_ade,
        "conf_fde": conf_fde,
        "conf_k": conf_k,
        "conf_prob": conf_prob,
        "best_pred_by_ade": best_pred_by_ade,
        "conf_pred": conf_pred,
        "horizon_errors": h_err
    }


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_horizon_results(results):
    csv_path = os.path.join(RESULTS_DIR, f"{METHOD_GROUP}_horizon_errors.csv")

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


def save_per_sample_results(method_name, rows):
    csv_path = os.path.join(RESULTS_DIR, f"{method_name}_per_sample.csv")

    save_csv(
        csv_path,
        rows,
        fieldnames=[
            "sample_idx",
            "min_ade",
            "min_fde",
            "best_ade_k",
            "best_fde_k",
            "conf_ade",
            "conf_fde",
            "conf_k",
            "conf_prob"
        ]
    )


def save_summary_results(results):
    csv_path = os.path.join(RESULTS_DIR, f"{METHOD_GROUP}_summary.csv")
    json_path = os.path.join(RESULTS_DIR, f"{METHOD_GROUP}_summary.json")

    summary_rows = [
        {k: v for k, v in r.items() if k != "horizon_errors"}
        for r in results
    ]

    save_csv(
        csv_path,
        summary_rows,
        fieldnames=[
            "method",
            "K",
            "alpha",
            "hidden_dim",
            "num_layers",
            "epochs",
            "lr",
            "dropout",
            "min_ade",
            "min_fde",
            "conf_ade",
            "conf_fde",
            "num_samples"
        ]
    )

    with open(json_path, "w") as f:
        json.dump(summary_rows, f, indent=4)


def plot_metric_comparison(results):
    methods = [r["method"] for r in results]
    min_ades = [r["min_ade"] for r in results]
    min_fdes = [r["min_fde"] for r in results]
    conf_ades = [r["conf_ade"] for r in results]
    conf_fdes = [r["conf_fde"] for r in results]

    x = np.arange(len(methods))
    width = 0.2

    plt.figure(figsize=(12, 6))

    plt.bar(x - 1.5 * width, min_ades, width, label="minADE@K")
    plt.bar(x - 0.5 * width, min_fdes, width, label="minFDE@K")
    plt.bar(x + 0.5 * width, conf_ades, width, label="confADE@K")
    plt.bar(x + 1.5 * width, conf_fdes, width, label="confFDE@K")

    plt.xticks(x, methods, rotation=20, ha="right")
    plt.ylabel("Error")
    plt.title("Top-K LSTM with Confidence Comparison")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, f"{METHOD_GROUP}_metrics_comparison.png")
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_horizon_comparison(results):
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
    plt.title(f"{METHOD_GROUP}: Error by Prediction Horizon")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, f"{METHOD_GROUP}_horizon_errors.png")
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_lstm(
    hidden_dim=128,
    num_layers=1,
    epochs=100,
    lr=1e-3,
    batch_size=64,
    dropout=0.1,
    K=6,
    alpha=0.1,
    num_plots=20
):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    method_name = f"lstm_top{K}_conf_hidden_{hidden_dim}_layers_{num_layers}"

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

    input_dim = X_train.shape[-1]

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

    model = LSTMBaseline(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_steps=OUTPUT_STEPS,
        K=K,
        dropout=dropout
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            preds, logits = model(X_batch)
            loss = topk_wta_conf_loss(
                preds,
                logits,
                y_batch,
                alpha=alpha
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.shape[0]

        avg_loss = total_loss / len(train_dataset)

        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_preds, val_logits = model(X_val_tensor)
                val_loss = topk_wta_conf_loss(
                    val_preds,
                    val_logits,
                    y_val_tensor,
                    alpha=alpha
                ).item()
            model.train()

            print(
                f"{method_name} | Epoch {epoch:03d} | "
                f"Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

    model.eval()

    with torch.no_grad():
        y_pred_tensor, logits_tensor = model(X_test_tensor)

    ade_list = []
    fde_list = []
    conf_ade_list = []
    conf_fde_list = []
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

        preds_single = y_pred_tensor[i]
        logits_single = logits_tensor[i]

        metrics = topk_metrics_single(
            preds_single,
            logits_single,
            y_true_tensor
        )

        ade_list.append(metrics["min_ade"])
        fde_list.append(metrics["min_fde"])
        conf_ade_list.append(metrics["conf_ade"])
        conf_fde_list.append(metrics["conf_fde"])
        horizon_error_rows.append(metrics["horizon_errors"])

        original_idx = int(idx_test[i])

        per_sample_rows.append({
            "sample_idx": original_idx,
            "min_ade": metrics["min_ade"],
            "min_fde": metrics["min_fde"],
            "best_ade_k": metrics["best_ade_k"],
            "best_fde_k": metrics["best_fde_k"],
            "conf_ade": metrics["conf_ade"],
            "conf_fde": metrics["conf_fde"],
            "conf_k": metrics["conf_k"],
            "conf_prob": metrics["conf_prob"]
        })

        if i in plot_positions:
            plot_trajectory_example(
                X=X_plot_tensor,
                y_true=y_true_tensor,
                y_pred=metrics["best_pred_by_ade"],
                method_name=method_name,
                idx=original_idx,
                ade=metrics["min_ade"],
                fde=metrics["min_fde"],
                plots_dir=PLOTS_DIR
            )

    mean_min_ade = float(np.mean(ade_list))
    mean_min_fde = float(np.mean(fde_list))
    mean_conf_ade = float(np.mean(conf_ade_list))
    mean_conf_fde = float(np.mean(conf_fde_list))
    mean_horizon_errors = np.mean(np.stack(horizon_error_rows), axis=0)

    save_per_sample_results(method_name, per_sample_rows)

    checkpoint_path = os.path.join(RESULTS_DIR, f"{method_name}_checkpoint.pt")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "output_steps": OUTPUT_STEPS,
            "K": K,
            "dropout": dropout,
            "alpha": alpha,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "idx_test": idx_test,
            "method_name": method_name,
        },
        checkpoint_path
    )

    print("Saved checkpoint:", checkpoint_path)

    print(f"\nMethod: {method_name}")
    print(f"minADE@{K}:", mean_min_ade)
    print(f"minFDE@{K}:", mean_min_fde)
    print(f"confADE@{K}:", mean_conf_ade)
    print(f"confFDE@{K}:", mean_conf_fde)

    return {
        "method": method_name,
        "K": K,
        "alpha": alpha,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "epochs": epochs,
        "lr": lr,
        "dropout": dropout,
        "min_ade": mean_min_ade,
        "min_fde": mean_min_fde,
        "conf_ade": mean_conf_ade,
        "conf_fde": mean_conf_fde,
        "num_samples": len(X_test),
        "horizon_errors": mean_horizon_errors
    }


def main():
    results = []

    results.append(
        run_lstm(
            hidden_dim=256,
            num_layers=2,
            epochs=100,
            lr=1e-3,
            dropout=0.1,
            K=6,
            alpha=0.1,
            num_plots=20
        )
    )

    save_summary_results(results)
    save_horizon_results(results)

    plot_metric_comparison(results)
    plot_horizon_comparison(results)

    print("\nSaved results to:", RESULTS_DIR)
    print("Saved plots to:", PLOTS_DIR) 

if __name__ == "__main__":
    main()