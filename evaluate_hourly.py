"""
e.g. > python evaluate_hourly.py models/20240530_142530
"""
import os, json, pickle, argparse
import numpy as np, matplotlib.pyplot as plt, torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from data_preprocess import load_and_prepare_data
from dataset import ValDataset, HOURS_IN_WEEK, HISTORY_HOURS
from model import TimeSeriesTFM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate(batch):
    xs, ys = zip(*batch)
    return torch.tensor(xs), torch.tensor(ys)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    args = parser.parse_args()

    # -------- load model & scaler --------
    with open(os.path.join(args.model_dir,"config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(args.model_dir,"scaler.pkl"),"rb") as f:
        scaler = pickle.load(f)

    model = TimeSeriesTFM(**cfg).to(DEVICE)
    state = torch.load(os.path.join(args.model_dir,"best.pt"),
                       map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # -------- test data (2024) -----------
    test_df = load_and_prepare_data(["Data/466920_2024.csv"])
    for col in ["Tx","RH","PrecpHour"]:
        test_df[col] = scaler.transform(test_df[[col]])

    test_loader = DataLoader(
        ValDataset(test_df, start_date="2024-01-01", end_date="2025-01-01"),
        batch_size=32, shuffle=False, collate_fn=collate)

    # -------- accumulate per-hour error ---
    sqe_tx = np.zeros(HOURS_IN_WEEK)
    cnt = np.zeros(HOURS_IN_WEEK)

    with torch.no_grad():
        for X,Y in test_loader:
            X,Y = X.to(DEVICE), Y.to(DEVICE)
            p = model(X,Y)                      # Teacher forcing
            err = (p[...,0] - Y[...,0]).cpu().numpy()   # Tx only
            sqe_tx += (err**2).sum(axis=0)              # sum over batch
            cnt += err.shape[0]

    rmse_per_hour = np.sqrt(sqe_tx / cnt)

    # -------- plot ------------------------
    plt.figure(figsize=(6,4))
    plt.plot(range(1,HOURS_IN_WEEK+1), rmse_per_hour, marker="o")
    plt.xlabel("Forecast lead time (hour)")
    plt.ylabel("RMSE - Tx (°C)")
    plt.title("Per-hour RMSE over 1-week horizon")
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_dir,"rmse_per_hour.png"))
    plt.show()
