import numpy as np
import pandas as pd
from torch.utils.data import Dataset

HOURS_IN_WEEK  = 168
HISTORY_WEEKS  = 8
HISTORY_HOURS  = HOURS_IN_WEEK * HISTORY_WEEKS      # 1344
PREDICT_HOURS  = HOURS_IN_WEEK                      # 168

class TrainDataset(Dataset):
    def __init__(self, df, feat_cols):
        self.features = df[feat_cols].values.astype(np.float32)
    def __len__(self):
        return len(self.features) - (HISTORY_HOURS + PREDICT_HOURS) + 1
    def __getitem__(self, idx):
        x = self.features[idx : idx + HISTORY_HOURS]
        y = self.features[idx + HISTORY_HOURS : idx + HISTORY_HOURS + PREDICT_HOURS]
        return x, y                               # (1344,F), (168,F)

class ValDataset(Dataset):
    """
    只回傳「歷史充足」的 (x, y)：
      x: 前 8 週   y: 之後 1 週
    """
    def __init__(self, df, start_date, end_date, feat_cols):
        self.features  = df[feat_cols].values.astype(np.float32)
        self.datetimes = df["DateTime"].values
        start = pd.to_datetime(start_date)
        end   = pd.to_datetime(end_date)

        mask = (df["DateTime"] >= start) & (df["DateTime"] < end)
        idx  = np.where(mask)[0]

        # ⬇︎ 剔除歷史不足的目標起點
        self.targets_idx = idx[idx >= HISTORY_HOURS]

    def __len__(self):
        return len(self.targets_idx) - PREDICT_HOURS + 1
    def __getitem__(self, i):
        y_start = self.targets_idx[i]
        y_end   = y_start + PREDICT_HOURS
        x_start = y_start - HISTORY_HOURS
        x = self.features[x_start : y_start]
        y = self.features[y_start : y_end]
        return x, y
