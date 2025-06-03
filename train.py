#!/usr/bin/env python3
# train.py – Multi-Head Transformer + Temp-Anomaly  (2025-06-06, rev-C)

import os, json, pickle, datetime, warnings, gc, re
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cuda.matmul.allow_tf32 = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────── 超參 ─────────
CSV_PATH, OUT_ROOT = "data.csv", "models"
HISTORY, BATCH = 1, 32
EPOCHS, PATIENCE = 40, 6
LR = 2e-4

D_MODEL, NHEAD, LAYERS, DIM_FF, DROPOUT = 192, 8, 10, 768, 0.1
L_TEMP, L_PREC_C, L_PREC_A, L_PRESS, L_WIND, L_OTHER = 0.35, 0.15, 0.15, 0.20, 0.10, 0.05

# ───────── 公用函式（evaluate 亦可 import）─────────
def inverse_standardize(arr_z, scaler, idxs=None):
    """
    arr_z 可為全欄位或僅 idxs 子集合。
    - 若 `idxs is None` 代表 arr_z 形狀已與 scaler 一致。
    回傳：反標準化但仍為「異常值」的 ndarray。
    """
    if isinstance(arr_z, torch.Tensor):
        arr_z = arr_z.cpu().numpy()

    if idxs is None:
        return arr_z * scaler.scale_ + scaler.mean_

    idxs = np.asarray(idxs)
    if arr_z.shape[1] == len(idxs):          # 輸入僅子集合
        return arr_z * scaler.scale_[idxs] + scaler.mean_[idxs]

    arr = arr_z.copy()
    arr[:, idxs] = arr_z[:, idxs] * scaler.scale_[idxs] + scaler.mean_[idxs]
    return arr

def restore_temperature(temp_anom, months, clim_mu):
    """
    temp_anom: (N, Nt)  ΔT (°C)
    months   : (N,)     1-12
    clim_mu  : (12, Nt) 每月平均 (°C)
    """
    if isinstance(temp_anom, torch.Tensor):
        temp_anom = temp_anom.cpu().numpy()
    return temp_anom + clim_mu[months - 1]          # 月份 1→index 0

# ───────── Dataset ─────────
class SeqDataset(Dataset):
    def __init__(self, arr, months, H):
        self.arr, self.months, self.H = arr.astype(np.float32), months, H
    def __len__(self):  return len(self.arr) - self.H
    def __getitem__(self, idx):
        s, e = idx, idx + self.H
        return (self.arr[s:e], self.months[e-1]), self.arr[e], self.months[e]

def collate(batch):
    xs, ys, ms = zip(*batch)
    x, m_hist  = zip(*xs)
    return (torch.tensor(np.array(x)),
            torch.tensor(np.array(m_hist))), \
           torch.tensor(np.array(ys)), \
           torch.tensor(np.array(ms))

# ───────── PosEnc ─────────
class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pos = torch.arange(max_len)[:, None]
        div = torch.exp(torch.arange(0, d_model, 2)*-(np.log(10000.0)/d_model))
        pe  = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos*div)
        pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

# ───────── Model ─────────
class MultiHeadTFM(nn.Module):
    def __init__(self, input_dim, idxs, *,
                 d_model=D_MODEL, nhead=NHEAD,
                 num_layers=LAYERS, dim_ff=DIM_FF, dropout=DROPOUT):
        super().__init__()
        self.idxs = idxs
        self.inp  = nn.Linear(input_dim, d_model)
        self.pos  = PosEnc(d_model)
        self.enc  = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_ff,
                                       dropout, batch_first=True),
            num_layers)

        self.temp_head   = nn.Linear(d_model, len(idxs["temp"]))
        self.press_head  = nn.Linear(d_model, len(idxs["press"]))
        self.wind_head   = nn.Linear(d_model, len(idxs["wind"]))
        self.other_head  = nn.Linear(d_model, len(idxs["other"]))
        self.pr_cls_head = nn.Linear(d_model, len(idxs["precip"]))
        self.pr_amt_head = nn.Linear(d_model, len(idxs["precip"]))

    def forward(self, x):
        z = self.enc(self.pos(self.inp(x)))[:, -1]
        return {
            "temp" : self.temp_head(z),            # ΔT (z-score)
            "press": self.press_head(z),
            "wind" : self.wind_head(z),
            "other": self.other_head(z),
            "p_cls": torch.sigmoid(self.pr_cls_head(z)),
            "p_amt": torch.relu(self.pr_amt_head(z))
        }

# ───────── 主程式 ─────────
def main():
    # 1. 讀檔 & 基本處理 --------------------------------------------------
    df = pd.read_csv(CSV_PATH)
    df.rename(columns={df.columns[0]: "Datetime"}, inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.sort_values("Datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["month"] = df["Datetime"].dt.month
    df["tod"]   = df["Datetime"].dt.hour + df["Datetime"].dt.minute / 60

    # ----- 取得特徵清單（不含 Datetime）-----
    feats = [c for c in df.columns if c != "Datetime"]

    # ----- 分組（以數值索引）-------------
    pat = {
        "temp" : re.compile("氣溫"),
        "press": re.compile("氣壓"),
        "wind" : re.compile("(風速\\(m/s\\)(?!.*最大))|風向|最大陣風"),
        "prec" : re.compile("降水量\\(mm\\)")
    }
    idxs_num = {k: [] for k in ["temp", "press", "wind", "precip", "other"]}
    for i, col in enumerate(feats):
        if pat["temp"].search(col):       idxs_num["temp"].append(i)
        elif pat["press"].search(col):    idxs_num["press"].append(i)
        elif pat["wind"].search(col):     idxs_num["wind"].append(i)
        elif pat["prec"].search(col):     idxs_num["precip"].append(i)
        else:                             idxs_num["other"].append(i)

    # -------- 方便和 pandas 配合的「欄名版」 ----------
    cols_grp = {k: [feats[i] for i in v] for k, v in idxs_num.items()}

    # 2. 缺值補 -----------------------------------------------------------
    df[cols_grp["precip"]] = (
        df[cols_grp["precip"]].mask(df[cols_grp["precip"]] <= -99, np.nan)
                               .fillna(0)
    )
    df[[c for c in feats if c not in cols_grp["precip"]]] = (
        df[[c for c in feats if c not in cols_grp["precip"]]]
          .mask(df[[c for c in feats if c not in cols_grp["precip"]]] <= -99,
                np.nan)
          .interpolate(limit_direction="both")
    )

    # 3. (站點 × 月份) 平均 climatology (°C) -------------------------------
    clim_mu_df = df.groupby("month")[cols_grp["temp"]].mean()   # (12, Nt)
    clim_mu_arr = clim_mu_df.values                             # ndarray
    # 將所有 temp 欄改成 ΔT
    df[cols_grp["temp"]] = df[cols_grp["temp"]] - \
                           df["month"].map(clim_mu_df.to_dict("index")).apply(
                               lambda d: pd.Series(d)
                           )

    # 4. Split & 標準化 ---------------------------------------------------
    mask_tr = (df["Datetime"] >= "2016-01-01") & (df["Datetime"] < "2023-01-01")
    mask_vl = (df["Datetime"] >= "2023-01-01") & (df["Datetime"] < "2025-01-01")

    scaler = StandardScaler().fit(df.loc[mask_tr, feats])
    data_z = scaler.transform(df[feats])

    train_loader = DataLoader(
        SeqDataset(data_z[mask_tr.values], df.loc[mask_tr, "month"].values,
                   HISTORY),
        BATCH, True, drop_last=True, collate_fn=collate
    )
    val_loader = DataLoader(
        SeqDataset(data_z[mask_vl.values], df.loc[mask_vl, "month"].values,
                   HISTORY),
        BATCH, False, collate_fn=collate
    )

    # ------ 降雨 class-weight ----------
    y_p_tr = df.loc[mask_tr, feats].values[:, idxs_num["precip"]]
    neg, pos = (y_p_tr == 0).sum(0), (y_p_tr > 0).sum(0)
    pos_w = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1., None),
                         dtype=torch.float32, device=DEVICE)

    print(f"Train {len(train_loader.dataset)} | Val {len(val_loader.dataset)}")
    for k in idxs_num:
        print(f"  {k:<6}: {len(idxs_num[k])} features")

    # 5. Model / Optim ----------------------------------------------------
    cfg = dict(input_dim=len(feats), idxs=idxs_num,
               d_model=D_MODEL, nhead=NHEAD,
               num_layers=LAYERS, dim_ff=DIM_FF, dropout=DROPOUT,
               clim_temp=clim_mu_arr.tolist())
    model = MultiHeadTFM(**{k: v for k, v in cfg.items() if k != "clim_temp"}) \
            .to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, tag); os.makedirs(out_dir, exist_ok=True)
    status, best_rmse, wait = tqdm(total=0, position=999, bar_format="{desc}"), np.inf, 0

    # 6. Epoch loop -------------------------------------------------------
    for ep in range(1, EPOCHS + 1):
        # ---------- Train ----------
        status.set_description_str(f"🟢 Epoch {ep} – training")
        model.train(); loss_sum = 0.0

        for (X, _), Y, _ in tqdm(train_loader, desc=f"E{ep:02d}[train]", leave=False):
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            out  = model(X)

            # ---- 反標準化雨量判定 flag ----
            scale_p = torch.tensor(scaler.scale_[idxs_num["precip"]],
                                   dtype=torch.float32, device=DEVICE)
            mean_p  = torch.tensor(scaler.mean_[idxs_num["precip"]],
                                   dtype=torch.float32, device=DEVICE)
            raw_prec = Y[:, idxs_num["precip"]] * scale_p + mean_p
            flag     = (raw_prec > 0).float()
            amt_raw  = raw_prec

            yT, yP, yW, yO = (Y[:, idxs_num["temp"]],
                              Y[:, idxs_num["press"]],
                              Y[:, idxs_num["wind"]],
                              Y[:, idxs_num["other"]])

            l_temp  = F.mse_loss(out["temp"],  yT)
            l_press = F.mse_loss(out["press"], yP)
            l_wind  = F.mse_loss(out["wind"],  yW)
            l_other = F.mse_loss(out["other"], yO)
            l_cls   = F.binary_cross_entropy(out["p_cls"], flag,
                                             weight=flag*pos_w + (1-flag))
            mask = flag.bool()
            l_amt = F.mse_loss(out["p_amt"][mask], amt_raw[mask]) if mask.any() \
                    else torch.tensor(0., device=DEVICE)

            loss = (L_TEMP*l_temp + L_PREC_C*l_cls + L_PREC_A*l_amt +
                    L_PRESS*l_press + L_WIND*l_wind + L_OTHER*l_other)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item()
        train_loss = loss_sum / len(train_loader)

        # ---------- Validate ----------
        status.set_description_str(f"🔵 Epoch {ep} – validating")
        model.eval(); p, v = [], []
        with torch.no_grad():
            for (X, _), Y, _ in val_loader:
                X, Y = X.to(DEVICE), Y.to(DEVICE)
                pred_z = model(X)["temp"]
                true_z = Y[:, idxs_num["temp"]]
                p.append(inverse_standardize(pred_z, scaler, idxs_num["temp"]))
                v.append(inverse_standardize(true_z, scaler, idxs_num["temp"]))
        rmse = np.sqrt(mean_squared_error(np.concatenate(v).ravel(),
                                          np.concatenate(p).ravel()))
        tqdm.write(f"[E{ep:02d}] Train {train_loss:.4f} | "
                   f"Val_RMSE(ΔT°C) {rmse:.4f} | Best {best_rmse:.4f}")

        if rmse < best_rmse - 1e-4:
            best_rmse, wait = rmse, 0
            status.set_description_str(f"💾 Epoch {ep} – saving best")
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
            with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)
            with open(os.path.join(out_dir, "config.json"), "w") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
        else:
            wait += 1
            if wait >= PATIENCE:
                status.set_description_str("⏹ Early stop"); break
        torch.cuda.empty_cache(); gc.collect()

    print(f"\nFinished. Best Val RMSE(ΔT°C) = {best_rmse:.4f}")
    print("Artifacts @", out_dir)

if __name__ == "__main__":
    main()
