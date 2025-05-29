#!/usr/bin/env python3
# train_optuna.py
"""
超參數最佳化 + Early-Stopping + tqdm Batch 進度
-------------------------------------------------
• 詳細訓練/驗證列印
• Val_RMSE 改善即儲存 best.pt, config.json, scaler.pkl
"""

import os, json, pickle, argparse, datetime, warnings
import optuna, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm

from data_preprocess import load_and_prepare_data
from dataset import TrainDataset, ValDataset
from model import TimeSeriesTFM

warnings.filterwarnings("ignore", category=UserWarning)

DEVICE, BATCH_SIZE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    16,
)
PATIENCE, IMPROVE_DELTA = 6, 1e-3


# ---------- 工具 ----------
def zscore_fit(df):
    sc = StandardScaler()
    sc.fit(df[["Tx", "RH", "PrecpHour"]])
    return sc


def collate(batch):
    xs, ys = zip(*batch)
    return torch.tensor(xs), torch.tensor(ys)


# ---------- Optuna 目標 ----------
def objective(trial):
    # 1️⃣ Hyper-params
    params = {
        "d_model": trial.suggest_categorical("d_model", [32, 64, 96]),
        "nhead": trial.suggest_categorical("nhead", [4, 8]),
        "num_layers": trial.suggest_int("num_layers", 2, 4),
        "dim_feedforward": trial.suggest_categorical("ff", [64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.2),
        "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
    }

    # 2️⃣ Data
    train_df = load_and_prepare_data(
        ["Data/466920_2020.csv", "Data/466920_2021.csv", "Data/466920_2022.csv"]
    )
    val_df = load_and_prepare_data(
        ["Data/466920_2022.csv", "Data/466920_2023.csv"]
    )

    scaler = zscore_fit(train_df)
    feats = ["Tx", "RH", "PrecpHour"]
    train_df[feats] = scaler.transform(train_df[feats])
    val_df[feats] = scaler.transform(val_df[feats])

    train_loader = DataLoader(
        TrainDataset(train_df),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        ValDataset(val_df, start_date="2023-01-01", end_date="2023-12-31"),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
    )

    # 3️⃣ Model / Opt / Loss
    model = TimeSeriesTFM(**{k: v for k, v in params.items() if k != "lr"}).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
    crit = torch.nn.MSELoss()

    best_rmse, wait = np.inf, 0
    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("models", tag)
    os.makedirs(out_dir, exist_ok=True)

    # Header
    try:
        best_so_far = f"{trial.study.best_value:.4f}"
    except ValueError:
        best_so_far = "—"
    print(
        f"\n=== Trial {trial.number + 1}/{trial.study.user_attrs.get('TOTAL_TRIALS','?')} "
        f"(study-best: {best_so_far}) ==="
    )
    print("Hyper-params:", params)

    # 4️⃣ Epoch loop
    for epoch in range(1, args.max_epochs + 1):
        # ---- train ----
        model.train()
        loss_sum = 0.0
        for X, Y in tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"T{trial.number+1} E{epoch} [train]",
            leave=False,
        ):
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(X, Y), Y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        train_loss = loss_sum / len(train_loader)

        # ---- val ----
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X, Y in tqdm(
                val_loader,
                total=len(val_loader),
                desc=f"T{trial.number+1} E{epoch} [val]  ",
                leave=False,
            ):
                X, Y = X.to(DEVICE), Y.to(DEVICE)
                out = model(X, Y)
                preds.append(out[..., 0].cpu().numpy())
                trues.append(Y[..., 0].cpu().numpy())
        preds = np.concatenate(preds).ravel()
        trues = np.concatenate(trues).ravel()

        # --- RMSE ---
        mse = mean_squared_error(trues, preds)   # <<< FIX
        rmse = np.sqrt(mse)                      # <<< FIX

        trial.report(rmse, epoch)
        if trial.should_prune():
            print("⏩ Trial pruned")
            raise optuna.TrialPruned()

        gpu_mem = (
            torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        )
        try:
            best_global = f"{trial.study.best_value:.4f}"
        except ValueError:
            best_global = f"{rmse:.4f}"

        print(
            f"[T{trial.number:02d} | Ep {epoch:02d}] "
            f"TrainLoss={train_loss:.4f}  Val_RMSE={rmse:.4f}  "
            f"Best(T)={best_rmse:.4f}  StudyBest={best_global}  "
            f"GPU={gpu_mem:,.0f} MB"
        )

        # --- Early-stop / Save ---
        if rmse < best_rmse - IMPROVE_DELTA:
            best_rmse, wait = rmse, 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
            with open(os.path.join(out_dir, "config.json"), "w") as f:
                json.dump(params, f, indent=2)
            with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"❗ Early-stop at epoch {epoch} (no improvement {PATIENCE} ep)")
                break

    return best_rmse


# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--max_epochs", type=int, default=40)
    args = parser.parse_args()

    print(f"Device : {DEVICE}")
    print(f"Trials : {args.trials}")
    print(f"MaxEp  : {args.max_epochs}\n")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.set_user_attr("TOTAL_TRIALS", args.trials)

    study.optimize(objective, n_trials=args.trials)

    print("\n=== Best Trial Overall ===")
    print("Params :", study.best_trial.params)
    print("RMSE_TX:", study.best_value)
