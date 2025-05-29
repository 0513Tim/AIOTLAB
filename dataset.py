import numpy as np
import pandas as pd
from torch.utils.data import Dataset

HOURS_IN_WEEK = 168
HISTORY_WEEKS = 8
HISTORY_HOURS = HOURS_IN_WEEK * HISTORY_WEEKS
PREDICT_HOURS = HOURS_IN_WEEK         # 168

class TrainDataset(Dataset):
    def __init__(self, df):
        self.features = df[["Tx","RH","PrecpHour"]].values.astype(np.float32)
    def __len__(self):
        return len(self.features) - (HISTORY_HOURS + PREDICT_HOURS) + 1
    def __getitem__(self, idx):
        x = self.features[idx : idx+HISTORY_HOURS]
        y = self.features[idx+HISTORY_HOURS : idx+HISTORY_HOURS+PREDICT_HOURS]
        return x, y                       # (1344,3), (168,3)

class ValDataset(Dataset):
    """Y 必須落在 [start_date,end_date)"""
    def __init__(self, df, start_date, end_date):
        self.df = df
        self.start = pd.to_datetime(start_date)
        self.end   = pd.to_datetime(end_date)
        # 過濾出可當 Y 的時間索引
        mask = (df["DateTime"] >= self.start) & (df["DateTime"] < self.end)
        self.targets_idx = np.where(mask)[0]
        self.features = df[["Tx","RH","PrecpHour"]].values.astype(np.float32)
        self.datetimes = df["DateTime"].values
    def __len__(self):
        return len(self.targets_idx) - PREDICT_HOURS + 1
    def __getitem__(self, i):
        y_start = self.targets_idx[i]
        y_end   = y_start + PREDICT_HOURS
        x_start = y_start - HISTORY_HOURS
        x_end   = y_start
        assert x_start >= 0, "歷史不足"
        x = self.features[x_start:x_end]
        y = self.features[y_start:y_end]
        return x, y
