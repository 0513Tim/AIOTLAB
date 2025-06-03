import os
import pandas as pd
from collections import Counter

data_folder = "weather_data"
log_file = "Dataset_clean.log"
time_column = "觀測時間(hour)"
cutoff_date = pd.to_datetime("2016-01-01")

deleted_column_counter = Counter()
valid_column_counts = {}

with open(log_file, "w", encoding="utf-8") as log:

    def log_print(msg):
        print(msg)
        log.write(msg + "\n")

    log_print("=== 開始清理資料集（含時間過濾） ===\n")

    for filename in os.listdir(data_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(data_folder, filename)

            try:
                df = pd.read_csv(file_path)

                # 嘗試解析時間欄位
                if time_column not in df.columns:
                    log_print(f"{filename}：❌ 缺少欄位 {time_column}，略過")
                    continue

                df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
                first_valid_time = df[time_column].min()

                if pd.isna(first_valid_time):
                    log_print(f"{filename}：❌ 無法解析任何觀測時間，略過")
                    continue

                # 若整份資料都是 2016 後才開始，直接刪除
                if first_valid_time >= cutoff_date:
                    os.remove(file_path)
                    log_print(f"🗑️ {filename}：最早時間為 {first_valid_time.date()}，晚於 2016，已刪除")
                    continue

                # 篩選時間 ≥ 2016 的資料列
                df = df[df[time_column] >= cutoff_date].copy()

                if df.empty:
                    os.remove(file_path)
                    log_print(f"🗑️ {filename}：2016 後無資料，已刪除")
                    continue

                # 清除欄位空值超過 3 的欄位
                na_counts = df.isna().sum()
                cols_to_drop = na_counts[na_counts > 3].index.tolist()

                if cols_to_drop:
                    log_print(f"{filename}：移除欄位 {cols_to_drop}")
                    deleted_column_counter.update(cols_to_drop)
                    df.drop(columns=cols_to_drop, inplace=True)

                # 如果只剩一欄，刪除
                if df.shape[1] <= 1:
                    os.remove(file_path)
                    log_print(f"❌ {filename}：只剩 1 欄位，已刪除")
                    continue

                # 儲存清理後的 CSV
                df.to_csv(file_path, index=False)
                valid_column_counts[filename] = df.shape[1]
                log_print(f"{filename}：有效欄位數 = {df.shape[1]}，資料筆數 = {len(df)}")

            except Exception as e:
                log_print(f"{filename}：❌ 發生錯誤：{e}")

    # === 統計報告 ===
    log_print("\n=== 清理報告 ===")
    total_dropped_columns = sum(deleted_column_counter.values())
    unique_dropped_columns = len(deleted_column_counter)

    log_print(f"總共刪除了 {total_dropped_columns} 個欄位出現次數（重複計算）")
    log_print(f"共 {unique_dropped_columns} 種不同欄位曾被刪除：")
    for col, count in deleted_column_counter.most_common():
        log_print(f"  - {col}：被刪除了 {count} 次")

    valid_counts = list(valid_column_counts.values())
    if valid_counts:
        min_cols = min(valid_counts)
        max_cols = max(valid_counts)
        avg_cols = sum(valid_counts) / len(valid_counts)
        sum_cols = sum(valid_counts)

        log_print("\n=== 清理後有效欄位統計 ===")
        log_print(f"最小有效欄位數：{min_cols}")
        log_print(f"最大有效欄位數：{max_cols}")
        log_print(f"平均有效欄位數：{avg_cols:.2f}")
        log_print(f"✅ 總有效欄位數加總（全部檔案）：{sum_cols}")
    else:
        log_print("⚠️ 無任何檔案成功清理或符合條件。")
