#!/usr/bin/env python3
# train.py – 單步預報 + 多站降水雙頭條件迴歸   2025-06-04 fix

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

# ───────── 全域設定 ─────────
CSV_PATH = "data.csv"; OUT_ROOT = "models"
HISTORY  = 1; BATCH = 32; EPOCHS = 20; PATIENCE = 6; LR = 2e-4
D_MODEL  = 128; NHEAD = 8; LAYERS = 3; DIM_FF = 256; DROPOUT = 0.1
LAMBDA_CLS, LAMBDA_AMT, LAMBDA_OTH = 0.3, 0.4, 0.3   # loss 權重

# ───────── Dataset ─────────
class SeqDataset(Dataset):
    def __init__(self, arr: np.ndarray, history: int):
        self.arr, self.H = arr.astype(np.float32), history
    def __len__(self): return len(self.arr) - self.H
    def __getitem__(self, idx):
        return (self.arr[idx:idx+self.H], self.arr[idx+self.H])
def collate(b): xs,ys=zip(*b); return torch.tensor(xs), torch.tensor(ys)

# ───────── Model ──────────
class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pos=torch.arange(max_len)[:,None]
        div=torch.exp(torch.arange(0,d_model,2)*-(np.log(10000.0)/d_model))
        pe=torch.zeros(max_len,d_model)
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("pe",pe.unsqueeze(0))
    def forward(self,x): return x+self.pe[:, :x.size(1)]

class TwoHeadTFM(nn.Module):
    def __init__(self, input_dim, precip_idx,
                 d_model=D_MODEL,nhead=NHEAD,num_layers=LAYERS,
                 dim_ff=DIM_FF,dropout=DROPOUT):
        super().__init__()
        self.idx_p=precip_idx
        self.idx_o=[i for i in range(input_dim) if i not in precip_idx]

        self.inp = nn.Linear(input_dim,d_model)
        self.pos = PosEnc(d_model)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model,nhead,dim_ff,dropout,
                                       batch_first=True),num_layers)
        self.cls_head  = nn.Linear(d_model,len(self.idx_p))   # p_rain
        self.amt_head  = nn.Linear(d_model,len(self.idx_p))   # rain_amt
        self.other_head= nn.Linear(d_model,len(self.idx_o))   # 其餘變量
    def forward(self,x):                       # x:(B,H,F)
        z=self.pos(self.inp(x)); z=self.enc(z)[:,-1]          # (B,D)
        return (torch.sigmoid(self.cls_head(z)),              # p_rain
                torch.relu(self.amt_head(z)),                 # amt
                self.other_head(z))                           # others

# ───────── 主流程 ──────────
def main():
    # 1️⃣ 讀檔＋時間特徵
    df=pd.read_csv(CSV_PATH)
    df.rename(columns={df.columns[0]:"Datetime"},inplace=True)
    df["Datetime"]=pd.to_datetime(df["Datetime"])
    df.sort_values("Datetime",inplace=True); df.reset_index(drop=True,inplace=True)
    df["tod"]=df["Datetime"].dt.hour+df["Datetime"].dt.minute/60
    df["month"]=df["Datetime"].dt.month

    feats=[c for c in df.columns if c!="Datetime"]
    precip_idx=[i for i,c in enumerate(feats) if re.search("降水量\\(mm\\)",c)]
    other_idx=[i for i in range(len(feats)) if i not in precip_idx]

    # 缺失補
    df[feats]=df[feats].mask(df[feats]<=-99,np.nan).interpolate(limit_direction="both")

    # 2️⃣ Split & 標準化
    m_train=(df["Datetime"]>="2016-01-01")&(df["Datetime"]<"2023-01-01")
    m_val  =(df["Datetime"]>="2023-01-01")&(df["Datetime"]<"2025-01-01")

    scaler=StandardScaler().fit(df.loc[m_train,feats])
    data_norm=scaler.transform(df[feats])

    train_loader=DataLoader(SeqDataset(data_norm[m_train],HISTORY),
                            BATCH,shuffle=True,drop_last=True,collate_fn=collate)
    val_loader  =DataLoader(SeqDataset(data_norm[m_val],HISTORY),
                            BATCH,shuffle=False,collate_fn=collate)

    # ─ pos_weight (每個降雨欄位一個值) ─
    train_precip=df.loc[m_train,feats].values[:,precip_idx]
    zeros=(train_precip==0).sum(axis=0); ones=(train_precip>0).sum(axis=0)
    pos_weight=np.clip(zeros/np.maximum(ones,1),1.0,None)     # (Np,)
    pos_w=torch.tensor(pos_weight,dtype=torch.float32,device=DEVICE)

    print(f"Train {len(train_loader.dataset)} | Val {len(val_loader.dataset)} | "
          f"Precips {len(precip_idx)} | Features {len(feats)}")

    # 3️⃣ 模型 / 優化
    cfg=dict(input_dim=len(feats),precip_idx=precip_idx,
             d_model=D_MODEL,nhead=NHEAD,num_layers=LAYERS,
             dim_ff=DIM_FF,dropout=DROPOUT)
    model=TwoHeadTFM(**cfg).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=1e-5)

    # 0 mm 在 Z-score 座標
    thr_scaled=(0 - scaler.mean_[precip_idx]) / scaler.scale_[precip_idx]
    thr_tensor=torch.tensor(thr_scaled,dtype=torch.float32,device=DEVICE)

    tag=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir=os.path.join(OUT_ROOT,tag); os.makedirs(out_dir,exist_ok=True)
    status=tqdm(total=0,position=999,bar_format="{desc}")
    best_rmse,wait=np.inf,0

    # 4️⃣ Epoch loop
    for ep in range(1,EPOCHS+1):
        # ---- Train ----
        status.set_description_str(f"🟢 Epoch {ep} – training")
        model.train(); loss_sum=0
        for X,Y in tqdm(train_loader,desc=f"E{ep:02d}[train]",leave=False):
            X,Y=X.to(DEVICE),Y.to(DEVICE)
            p_hat, amt_hat, oth_hat = model(X)

            flag_true=(Y[:,precip_idx] > thr_tensor).float()   # (B,Np)
            amt_true =Y[:,precip_idx]
            oth_true =Y[:,other_idx]

            bce = F.binary_cross_entropy(
                p_hat, flag_true,
                weight=flag_true*pos_w + (1-flag_true))
            mse_amt = (F.mse_loss(amt_hat[flag_true.bool()],
                                  amt_true[flag_true.bool()])
                        if flag_true.bool().sum()>0 else torch.zeros(1,device=DEVICE))
            mse_oth = F.mse_loss(oth_hat, oth_true)

            loss = (LAMBDA_CLS*bce + LAMBDA_AMT*mse_amt + LAMBDA_OTH*mse_oth)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item()
        train_loss=loss_sum/len(train_loader)

        # ---- Val ----
        status.set_description_str(f"🔵 Epoch {ep} – validating")
        model.eval(); preds,trues=[],[]
        with torch.no_grad():
            for X,Y in val_loader:
                X,Y=X.to(DEVICE),Y.to(DEVICE)
                _, amt_hat, _=model(X)
                preds.append(amt_hat.cpu().numpy())
                trues.append(Y[:,precip_idx].cpu().numpy())
        rmse=np.sqrt(mean_squared_error(np.concatenate(trues).ravel(),
                                        np.concatenate(preds).ravel()))

        tqdm.write(f"[E{ep:02d}] Train {train_loss:.4f} | "
                   f"Val_RMSE(precip) {rmse:.4f} | Best {best_rmse:.4f}")

        if rmse < best_rmse-1e-4:
            best_rmse,wait=rmse,0
            status.set_description_str(f"💾 Epoch {ep} – saving best")
            torch.save(model.state_dict(),os.path.join(out_dir,"best.pt"))
            with open(os.path.join(out_dir,"scaler.pkl"),"wb") as f: pickle.dump(scaler,f)
            with open(os.path.join(out_dir,"config.json"),"w") as f: json.dump(cfg,f,indent=2)
        else:
            wait+=1
            if wait>=PATIENCE:
                status.set_description_str("⏹ Early stop"); break
        torch.cuda.empty_cache(); gc.collect()

    print(f"\nFinished. Best Val_RMSE (precip) = {best_rmse:.4f}")
    print("Artifacts @",out_dir)

if __name__=="__main__":
    main()
