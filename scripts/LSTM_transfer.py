import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "results_workzone"

INPUT_STEPS = 10
OUTPUT_STEPS = 25
DT = 0.2

METHOD_NAME = "lstm_top6_conf_hidden_256_layers_2"


class TrajectoryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TopKLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_layers=2,
        output_steps=25,
        K=6,
        dropout=0.1,
    ):
        super().__init__()

        self.output_steps = output_steps
        self.K = K

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.traj_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, K * output_steps * 2),
        )

        self.conf_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, K),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1]

        traj = self.traj_head(h)
        traj = traj.view(x.shape[0], self.K, self.output_steps, 2)

        conf_logits = self.conf_head(h)

        return traj, conf_logits


def load_npz(data_path):
    data = np.load(data_path, allow_pickle=True)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    split = data["split"].astype(str)

    return X, y, split


def normalize_from_train(X_train, X_val, X_test):
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    std[std < 1e-6] = 1.0

    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std
    X_test_n = (X_test - mean) / std

    return X_train_n.astype(np.float32), X_val_n.astype(np.float32), X_test_n.astype(np.float32), mean, std


def topk_loss(pred, conf_logits, y, alpha=0.1):
    # pred: [B, K, T, 2]
    # y:    [B, T, 2]
    diff = pred - y[:, None, :, :]

    # ADE per mode
    per_t_dist = torch.linalg.norm(diff, dim=-1)
    ade_per_mode = per_t_dist.mean(dim=-1)

    best_ade, best_idx = ade_per_mode.min(dim=1)

    # regression loss only for best mode
    batch_idx = torch.arange(y.shape[0], device=y.device)
    best_pred = pred[batch_idx, best_idx]
    reg_loss = torch.mean((best_pred - y) ** 2)

    conf_loss = nn.CrossEntropyLoss()(conf_logits, best_idx)

    loss = reg_loss + alpha * conf_loss

    return loss, reg_loss.detach(), conf_loss.detach()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_rows = []

    total_min_ade = 0.0
    total_min_fde = 0.0
    total_conf_ade = 0.0
    total_conf_fde = 0.0
    total_n = 0

    horizon_sum = np.zeros(OUTPUT_STEPS, dtype=np.float64)

    sample_offset = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        pred, conf_logits = model(X)
        probs = torch.softmax(conf_logits, dim=1)

        diff = pred - y[:, None, :, :]
        dist = torch.linalg.norm(diff, dim=-1)

        ade_per_mode = dist.mean(dim=-1)
        fde_per_mode = dist[:, :, -1]

        min_ade, best_idx = ade_per_mode.min(dim=1)

        batch_idx = torch.arange(y.shape[0], device=device)
        min_fde = fde_per_mode[batch_idx, best_idx]

        conf_idx = probs.argmax(dim=1)
        conf_ade = ade_per_mode[batch_idx, conf_idx]
        conf_fde = fde_per_mode[batch_idx, conf_idx]

        best_horizon = dist[batch_idx, best_idx, :]

        bs = y.shape[0]
        total_n += bs

        total_min_ade += min_ade.sum().item()
        total_min_fde += min_fde.sum().item()
        total_conf_ade += conf_ade.sum().item()
        total_conf_fde += conf_fde.sum().item()

        horizon_sum += best_horizon.sum(dim=0).detach().cpu().numpy()

        for i in range(bs):
            all_rows.append({
                "sample_idx": sample_offset + i,
                "min_ade": float(min_ade[i].cpu()),
                "min_fde": float(min_fde[i].cpu()),
                "conf_ade": float(conf_ade[i].cpu()),
                "conf_fde": float(conf_fde[i].cpu()),
                "best_mode": int(best_idx[i].cpu()),
                "conf_mode": int(conf_idx[i].cpu()),
            })

        sample_offset += bs

    metrics = {
        "min_ade": total_min_ade / total_n,
        "min_fde": total_min_fde / total_n,
        "conf_ade": total_conf_ade / total_n,
        "conf_fde": total_conf_fde / total_n,
        "num_samples": total_n,
    }

    horizon = horizon_sum / total_n

    per_sample = pd.DataFrame(all_rows)

    return metrics, horizon, per_sample


def save_outputs(out_dir, metrics, horizon, per_sample):
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([{
        "method": METHOD_NAME,
        "min_ade": metrics["min_ade"],
        "min_fde": metrics["min_fde"],
        "conf_ade": metrics["conf_ade"],
        "conf_fde": metrics["conf_fde"],
        "num_samples": metrics["num_samples"],
    }])

    summary_path = out_dir / "lstm_topk_conf_summary.csv"
    summary.to_csv(summary_path, index=False)

    horizon_row = {"method": METHOD_NAME}
    for i, err in enumerate(horizon):
        t = (i + 1) * DT
        field = f"err_{t:.1f}s".replace(".", "p")
        horizon_row[field] = err

    horizon_df = pd.DataFrame([horizon_row])
    horizon_path = out_dir / f"{METHOD_NAME}_horizon.csv"
    horizon_df.to_csv(horizon_path, index=False)

    per_sample_path = out_dir / f"{METHOD_NAME}_per_sample.csv"
    per_sample.to_csv(per_sample_path, index=False)

    print("\nSaved:")
    print(summary_path)
    print(horizon_path)
    print(per_sample_path)


def train(args):
    data_path = PROJECT_ROOT / args.data_path
    out_dir = OUTPUT_ROOT / args.method_group
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, split = load_npz(data_path)

    X_train = X[split == "train"]
    y_train = y[split == "train"]

    X_val = X[split == "val"]
    y_val = y[split == "val"]

    X_test = X[split == "test"]
    y_test = y[split == "test"]

    print("Data:", data_path)
    print("X:", X.shape)
    print("y:", y.shape)
    print("train:", X_train.shape, y_train.shape)
    print("val:", X_val.shape, y_val.shape)
    print("test:", X_test.shape, y_test.shape)

    X_train, X_val, X_test, mean, std = normalize_from_train(X_train, X_val, X_test)

    train_ds = TrajectoryDataset(X_train, y_train)
    val_ds = TrajectoryDataset(X_val, y_val)
    test_ds = TrajectoryDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = TopKLSTM(
        input_dim=X.shape[-1],
        hidden_dim=256,
        num_layers=2,
        output_steps=OUTPUT_STEPS,
        K=6,
        dropout=0.1,
    ).to(device)

    if args.init_checkpoint is not None:
        ckpt_path = PROJECT_ROOT / args.init_checkpoint
        print("Loading pretrained checkpoint:", ckpt_path)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

        print("Loaded checkpoint.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    best_ckpt_path = out_dir / f"{METHOD_NAME}_checkpoint.pt"

    for epoch in range(args.epochs):
        model.train()

        total_loss = 0.0
        total_reg = 0.0
        total_conf = 0.0
        total_n = 0

        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred, conf_logits = model(Xb)

            loss, reg_loss, conf_loss = topk_loss(
                pred,
                conf_logits,
                yb,
                alpha=args.alpha,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            bs = Xb.shape[0]
            total_loss += loss.item() * bs
            total_reg += reg_loss.item() * bs
            total_conf += conf_loss.item() * bs
            total_n += bs

        val_metrics, _, _ = evaluate(model, val_loader, device)
        val_score = val_metrics["min_ade"]

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={total_loss / total_n:.4f} | "
            f"reg={total_reg / total_n:.4f} | "
            f"conf={total_conf / total_n:.4f} | "
            f"val_minADE={val_metrics['min_ade']:.4f} | "
            f"val_minFDE={val_metrics['min_fde']:.4f} | "
            f"val_confADE={val_metrics['conf_ade']:.4f} | "
            f"val_confFDE={val_metrics['conf_fde']:.4f}"
        )

        if val_score < best_val:
            best_val = val_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": X.shape[-1],
                    "hidden_dim": 256,
                    "num_layers": 2,
                    "K": 6,
                    "output_steps": OUTPUT_STEPS,
                    "feature_mean": mean,
                    "feature_std": std,
                    "epoch": epoch,
                    "best_val_min_ade": best_val,
                    "method": METHOD_NAME,
                    "data_path": str(data_path),
                    "method_group": args.method_group,
                },
                best_ckpt_path,
            )

    print("\nLoading best checkpoint:", best_ckpt_path)
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics, horizon, per_sample = evaluate(model, test_loader, device)

    print("\nTEST RESULTS")
    for k, v in test_metrics.items():
        print(k, v)

    save_outputs(out_dir, test_metrics, horizon, per_sample)

    print("\nCheckpoint:", best_ckpt_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", required=True)
    parser.add_argument("--method-group", required=True)
    parser.add_argument("--init-checkpoint", default=None)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.1)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
