# Transformer Climate Predict

此分支包含Transformer模型實驗結構，使用前請先安裝依賴`pip install -r requirements.txt`，並前往 [pytorch](https://pytorch.org/) 下載對應cuda或cpu之torch套件

以下是個別實驗與其結果

---

- ## **驗證Transformer encoder單次輸入最佳週數**  

   於```Find_input_weeks/Find_num_history_weeks.ipynb```完成實驗於該資料夾之```log.json```，可使用```Show_num_history_weeks_graph.ipynb```印出圖表顯示。(注意實驗時須與```Data```資料夾於同一目錄)

   此實驗分別使用 7 * `num_history_weeks`小時 單位完成Transformer模型訓練，每次訓練 5 epoch，各單位時間共測試5次取平均以繪製下圖。(採用teacher forcing所以結果僅供參考)

   ![rmse_tx_trial5](https://github.com/user-attachments/assets/d9937f00-d45d-4d1c-9b6a-f9690eadd081)

- ## **驗證模型預測天數之RMSE變化**

   #### 以下為使用範例 (注意實驗時須與```Data```資料夾於同一目錄)
   #### 1. 跑 30 次超參數搜尋，最多每條試驗 40 epochs
   `python train_optuna.py --trials 30 --max_epochs 40`

   期間只要 val_RMSE(Tx) 進步就會把: `best.pt` / `config.json` / `scaler.pkl`  ---> `models/yyyyMMdd_HHmmss/`

   #### 2. 用最佳權重做逐時 RMSE 評估並畫圖
   `python evaluate_hourly.ipynb`
   
   #### 以下為使用2010 ~ 2023年466920觀測站資料訓練之模型預測20250521 ~ 20250528氣溫之部分實驗結果

   ![timeline_tx](https://github.com/user-attachments/assets/29da9b07-e43f-4a89-bd68-32fd7d4a9da6)
   
   ## 無論怎麼調參擴充歷史資料，模型只學得會這種平滑的曲線，幾乎可以得出以下結論
   - ### 使用單一測站的歷史資料是不會有效的 => 需使用多測站或經緯度級別的資料
   - ### 資料不能只使用溫度、濕度、降雨，需要包含更多的氣象資訊 => 可能還需包含氣壓、風向等其他指標
   - ### 氣象預測不適合長時間續的資料輸入或輸出 => 考慮只預測下一time step(e.g. hour)再套用自回歸

- ## **自回歸單時間部多氣象站資料預測**

  #### 架構將修正為每次預測僅讀取前1小時資料預測下1小時資料，讀取與預測項目包含所有氣象站之所有資料
  ![image](https://github.com/user-attachments/assets/0b9e1268-0256-4ec7-afea-ef1a544f67ed)
  ![image](https://github.com/user-attachments/assets/5e8b849e-6e49-4876-a44c-7b8b124f2456)


  #### 架構將修正為每次預測僅讀取前24小時資料預測下1小時資料，且將降雨獨立head處理是否降雨與降雨量
  ![image](https://github.com/user-attachments/assets/554e8417-b40d-4481-abe3-76b338c0e1d0)
  ![image](https://github.com/user-attachments/assets/7c3162cc-21e5-44ac-a150-b7845b81e41f)
  ![image](https://github.com/user-attachments/assets/54a4ef35-0fb5-4976-aba7-d87b0f78d083)
  ![image](https://github.com/user-attachments/assets/2d861967-4130-4400-9c84-40b12e28bb29)

  #### 修正evaluate bug並將模型架構改為先將溫度、降雨等資料去平均，再學習變化輛特徵
  ![image](https://github.com/user-attachments/assets/2f010a7b-236c-4f2d-b597-49eab2295b9b)
  ![image](https://github.com/user-attachments/assets/621def56-32ed-4f86-be4f-825a6a9a0de8)
  ![image](https://github.com/user-attachments/assets/1f5fc933-654e-4052-94d9-2979fefcf490)
  ![image](https://github.com/user-attachments/assets/54ae44a7-1562-416c-8341-7085be059bd1)


---
## 引用資料
https://github.com/Raingel/historical_weather (臺灣歷史氣象觀測資料庫)

Ou, J.-H., Kuo, C.-H., Wu, Y.-F., Lin, G.-C., Lee, M.-H., Chen, R.-K., Chou, H.-P., Wu, H.-Y., Chu, S.-C., Lai, Q.-J., Tsai, Y.-C., Lin, C.-C., Kuo, C.-C., Liao, C.-T., Chen, Y.-N., Chu, Y.-W., Chen, C.-Y., 2023. Application-oriented deep learning model for early warning of rice blast in Taiwan. Ecological Informatics 73, 101950. https://doi.org/10.1016/j.ecoinf.2022.101950


  


