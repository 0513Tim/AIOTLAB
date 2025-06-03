import os
import pandas as pd

data_folder = "weather_data"
output_file = "data.csv"
time_column = "觀測時間(hour)"

# 合併用列表
merged_df = None
csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

for filename in csv_files:
    file_path = os.path.join(data_folder, filename)
    station_id = os.path.splitext(filename)[0]  # 檔名即為站號

    try:
        df = pd.read_csv(file_path)

        if time_column not in df.columns:
            print(f"❌ {filename}：缺少欄位 {time_column}，略過")
            continue

        # 將時間欄轉換為 datetime
        df[time_column] = pd.to_datetime(df[time_column], errors='coerce')

        # 移除欄位中空白時間
        df = df.dropna(subset=[time_column])

        # 重命名所有欄位，除了觀測時間
        df_renamed = df.rename(columns={
            col: f"{station_id}{col}" for col in df.columns if col != time_column
        })

        # 以觀測時間為主鍵合併
        if merged_df is None:
            merged_df = df_renamed
        else:
            merged_df = pd.merge(merged_df, df_renamed, on=time_column, how="outer")

        print(f"✅ 已合併：{filename}")

    except Exception as e:
        print(f"❌ 讀取錯誤 {filename}：{e}")

# 儲存結果
if merged_df is not None:
    merged_df.sort_values(by=time_column, inplace=True)
    merged_df.to_csv(output_file, index=False)
    print(f"\n✅ 合併完成：{output_file}")
    print(f"總筆數（rows）：{merged_df.shape[0]}")
    print(f"總欄位數（columns）：{merged_df.shape[1]}")
else:
    print("⚠️ 沒有成功合併任何資料")
