# 來源與授權範圍

## 本公開版

本版保留本專案開發的 Python 研究模組、離線工具、合成資料測試與說明。核心相依套件透過安裝取得，沒有將套件原始碼、二進位檔或預訓練權重併入儲存庫。

本儲存庫自行開發的程式與文件採用 [MIT License](../LICENSE)，版權署名為 Ethan Liau。第三方套件、競賽資料與外部模型權重不因本儲存庫的 MIT 授權而改變其權利或使用條件。Git 作者身分不代表該作者擁有這些外部資產的權利。

## 外部概念與介面

| 項目 | 參考來源 | 本版處理 |
|---|---|---|
| 競賽任務與輸出格式 | [RSNA Knee 競賽](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) | 保留 12 目標名稱與格式驗證；不散布資料 |
| DICOM 讀取 | [pydicom](https://github.com/pydicom/pydicom) | 套件依賴，不含真實 DICOM |
| 張量與訓練 API | [PyTorch](https://github.com/pytorch/pytorch) | 選用套件依賴，不含權重 |
| 影像骨幹 API | [timm](https://github.com/huggingface/pytorch-image-models) | 選用套件依賴，初始化資產另行核實 |
| Transformer 骨幹 API | [Transformers](https://github.com/huggingface/transformers) | 選用套件依賴；沒有附帶 DINO 或其他資產 |
| Barlow Twins 研究概念 | [原始論文](https://arxiv.org/abs/2103.03230) | 小型研究實作，不附預訓練成果 |

程式中的模型架構名稱、checkpoint 載入介面或資料集路徑範例，不構成對該資產的授權聲明。第三方套件各自適用其上游授權。

## 沒有隨公開版散布的內容

- 競賽 MRI、報告、逐筆標籤、UID 對照與推論 CSV。
- RAPTOR、CoAtNet、DINO、RadImageNet 及其他模型的歷史 checkpoint。
- 匯入的第三方多語報告規則及依賴它們的實驗入口。
- 下載的 Kaggle Notebook、完整來源鏡像、授權往來及原始 API 回覆。
- 僅獲特定競賽使用許可、未取得一般再散布許可的 reader 改寫。

歷史模型曾被使用或取得部分許可，不代表本公開版宣告其完整來源、資料權利或留出折身分已獲驗證。
