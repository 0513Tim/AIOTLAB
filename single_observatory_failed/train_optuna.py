#!/usr/bin/env python3
# train_optuna.py  (2025-06-03 status bar 版)

"""
• Train: 2010-2020  • Val: 2021-2023
• 7 維輸入/輸出 (Tx, RH, Precp + 4 個時間週期特徵)
• 差分 Loss 抑制過度平滑
• status_bar：永遠佔終端最底行，顯示目前工作狀態
"""

import os, json, pickle, argparse, datetime, warnings, gc
import optuna, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm

from data_preprocess import load_and_prepare_data
from dataset import TrainDataset, ValDataset
from model import TimeSeriesTFM

warnings.filterwarnings("ignore", category=UserWarning)

# --------- 全域常數 ---------
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 16
PATIENCE    = 6
IMPROVE_DEL = 1e-3
FEATS = ["Tx","RH","PrecpHour","h_sin","h_cos","dow_sin","dow_cos"]

# --------- 幫手函式 ---------
def add_time_features(df):
    dt = df["DateTime"]
    hour = dt.dt.hour + dt.dt.minute/60
    dow  = dt.dt.dayofweek
    df["h_sin"]  = np.sin(2*np.pi*hour/24)
    df["h_cos"]  = np.cos(2*np.pi*hour/24)
    df["dow_sin"]= np.sin(2*np.pi*dow/7)
    df["dow_cos"]= np.cos(2*np.pi*dow/7)
    return df

def collate(b): xs,ys = zip(*b); return torch.tensor(xs), torch.tensor(ys)

# --------- Optuna 目標 ---------
def objective(trial):
    # ----- status bar 一行固定在最底 -----
    status_bar = tqdm(total=0, position=999, bar_format='{desc}')
    status_bar.set_description_str("preparing data…")

    # ----- 模型超參 -----
    d_model = trial.suggest_categorical("d_model",[64,96,128,160,192])
    nhead   = trial.suggest_categorical("nhead",[h for h in (4,8,16) if d_model%h==0])
    params  = {
        "input_dim":7, "output_dim":7,
        "d_model":d_model, "nhead":nhead,
        "num_layers":trial.suggest_int("num_layers",3,6),
        "dim_feedforward":trial.suggest_categorical("ff",[d_model*2,d_model*4,d_model*6]),
        "dropout":trial.suggest_float("dropout",0.05,0.30),
    }
    lr   = trial.suggest_float("lr",2e-4,2e-3,log=True)
    wd   = trial.suggest_float("weight_decay",1e-5,1e-3,log=True)
    diff_w = trial.suggest_categorical("diff_w",[0.1,0.2,0.3,0.5])

    # ----- 讀取並轉換資料 -----
    train_df = add_time_features(load_and_prepare_data(
        [f"Data/466920_{y}.csv" for y in range(2010,2021)]))
    val_df   = add_time_features(load_and_prepare_data(
        [f"Data/466920_{y}.csv" for y in (2021,2022,2023)]))

    scaler = StandardScaler().fit(train_df[FEATS])
    train_df[FEATS] = scaler.transform(train_df[FEATS])
    val_df[FEATS]   = scaler.transform(val_df[FEATS])

    train_loader = DataLoader(
        TrainDataset(train_df, feat_cols=FEATS),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=collate
    )
    val_loader = DataLoader(
        ValDataset(val_df,"2021-01-01","2023-12-31", feat_cols=FEATS),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
    )

    tqdm.write(f"[T{trial.number+1:02d}] Train samples: {len(train_loader.dataset)} "
               f"({len(train_loader)} batches) | "
               f"Val samples: {len(val_loader.dataset)} "
               f"({len(val_loader)} batches)")
    status_bar.set_description_str("building model…")

    # ----- 建模 -----
    model = TimeSeriesTFM(**params).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit  = torch.nn.MSELoss()

    best_rmse, wait = np.inf, 0
    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("models", tag); os.makedirs(out_dir, exist_ok=True)

    # ----- Epoch loop -----
    for epoch in range(1, args.max_epochs+1):
        status_bar.set_description_str(f"Epoch {epoch} – training")
        model.train(); loss_sum=0
        for X,Y in tqdm(train_loader,
                        desc=f"T{trial.number+1:02d}E{epoch:02d}[train]",
                        leave=False):
            X,Y = X.to(DEVICE), Y.to(DEVICE)
            opt.zero_grad()
            pred  = model(X,Y,teacher_force=True)
            base  = crit(pred,Y)
            shape = crit(pred[:,1:]-pred[:,:-1], Y[:,1:]-Y[:,:-1])
            loss  = base + diff_w*shape
            loss.backward(); opt.step()
            loss_sum += loss.item()
        train_loss = loss_sum / len(train_loader)

        status_bar.set_description_str(f"Epoch {epoch} – validating")
        model.eval(); preds,trues = [],[]
        with torch.no_grad():
            for X,Y in val_loader:
                X,Y = X.to(DEVICE), Y.to(DEVICE)
                out = model(X,Y,teacher_force=False)
                preds.append(out[...,0].cpu().numpy())
                trues.append(Y[...,0].cpu().numpy())
        rmse = np.sqrt(mean_squared_error(np.concatenate(trues).ravel(),
                                          np.concatenate(preds).ravel()))
        trial.report(rmse, epoch)
        if trial.should_prune(): 
            status_bar.set_description_str("Trial pruned")
            raise optuna.TrialPruned()

        tqdm.write(f"[T{trial.number:02d}|E{epoch:02d}] "
                   f"TrainLoss={train_loss:.4f}  Val_RMSE={rmse:.4f}  "
                   f"Best={best_rmse:.4f}")

        # ----- Early-Stopping / Save -----
        if rmse < best_rmse - IMPROVE_DEL:
            best_rmse, wait = rmse, 0
            status_bar.set_description_str(f"Epoch {epoch} – saving best.pt")
            torch.save(model.state_dict(), os.path.join(out_dir,"best.pt"))
            with open(os.path.join(out_dir,"config.json"),"w") as f:
                json.dump({**params,"lr":lr,"weight_decay":wd,"diff_w":diff_w},f,indent=2)
            with open(os.path.join(out_dir,"scaler.pkl"),"wb") as f:
                pickle.dump(scaler,f)
        else:
            wait += 1
            if wait >= PATIENCE:
                status_bar.set_description_str("⏹ Early-stop (no improve)")
                break

    # -------- Trial 結束清理 --------
    torch.cuda.empty_cache(); gc.collect()
    status_bar.set_description_str("Trial complete")
    return best_rmse

# -------- Main --------
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",type=int,default=20)
    parser.add_argument("--max_epochs",type=int,default=40)
    args = parser.parse_args()

    print("Device:",DEVICE)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective,n_trials=args.trials)

    print("\n=== Best Trial ===")
    print("Params :",study.best_trial.params)
    print("Best RMSE_TX :",study.best_value)
