# Transformer Climate Predict

此分支包含Transformer模型實驗結構，使用前請先安裝依賴`pip install -r requirements.txt`，並前往 [pytorch](https://pytorch.org/) 下載對應cuda或cpu之torch套件

以下是個別實驗與其結果

---

- ## **驗證Transformer encoder單次輸入最佳週數**  

   於```Find_input_weeks/Find_num_history_weeks.ipynb```完成實驗於該資料夾之```log.json```，可使用```Show_num_history_weeks_graph.ipynb```印出圖表顯示。(注意實驗時須與```Data```資料夾於同一目錄)

   此實驗分別使用 7 * `num_history_weeks`小時 單位完成Transformer模型訓練，每次訓練 5 epoch，各單位時間共測試5次取平均以繪製下圖。

   ![rmse_tx_trial5](https://github.com/user-attachments/assets/d9937f00-d45d-4d1c-9b6a-f9690eadd081)

- ## **驗證模型預測天數之RMSE變化**

   #### 以下為使用範例 (注意實驗時須與```Data```資料夾於同一目錄)
   #### 1. 跑 30 次超參數搜尋，最多每條試驗 40 epochs
   `python train_optuna.py --trials 30 --max_epochs 40`

   期間只要 val_RMSE(Tx) 進步就會把: `best.pt` / `config.json` / `scaler.pkl`  ---> `models/yyyyMMdd_HHmmss/`

   #### 2. 用最佳權重做逐時 RMSE 評估並畫圖
   `python evaluate_hourly.ipynb`
   

   
