import pandas as pd
import numpy as np

def load_and_prepare_data(file_paths):
    all_dfs = []
    for fp in file_paths:
        df = pd.read_csv(fp, parse_dates=["DateTime"])
        df = df[["DateTime", "Tx", "RH", "PrecpHour"]]
        all_dfs.append(df)

    full_df = (pd.concat(all_dfs, ignore_index=True)
                 .sort_values("DateTime")
                 .reset_index(drop=True))

    for col in ["Tx", "RH", "PrecpHour"]:
        full_df[col] = full_df[col].mask(full_df[col] < 0, np.nan)

    full_df[["Tx","RH","PrecpHour"]] = (
        full_df[["Tx","RH","PrecpHour"]]
        .interpolate(method="nearest", limit_direction="both")
    )
    return full_df
