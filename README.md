# RSNA Knee：MRI 研究工具與實驗復盤

本專案整理 RSNA Knee Abnormality Detection 競賽期間開發的 Python 研究工具，包含 DICOM 前處理、分組驗證、study-level 模型、OOF 比較及推論產物驗證。

這是精簡後的研究程式庫。**不包含完整歷史提交管線、競賽資料、模型權重或排行榜前十解法。**

## 結果與限制

截至 **2026-10-23 台灣時間**、整理前最後一次正式紀錄核對：

| 項目 | 已確認結果 |
|---|---|
| 歷史最佳正式提交 | Public **0.941** |
| scratch sparse MIL 獨立提交 | Public **0.629** |
| 最後一個 reader 正式重跑 | 執行失敗，沒有分數 |
| Public 前十目標 | **未達成** |
| Private 分數與官方最終選取 | 未核實 |

研究弱標籤 AUC、私有 Notebook 成功和正式競賽評分是不同證據。保留下來的模組不代表可以重現 0.941；該提交依賴的完整模型組合未隨本專案公開。詳細分析見 [研究總結](docs/RESEARCH_SUMMARY.md)。

## 內容

| 位置 | 用途 |
|---|---|
| `rsna_knee/` | 前處理、模型、訓練、切分、研究評估及驗證函式 |
| `tools/` | 離線 Notebook 建置、OOF／產物驗證與公開內容檢查 |
| `tests/` | 使用合成資料、暫存檔及小型 CPU 模型的測試 |
| `examples/` | 不需要競賽資料或權重的離線範例 |
| `docs/` | 研究結果、工具使用及來源限制 |

公開版沒有 Kaggle 認證、自動提交、排程或 agent 協調器；匯入模組不會啟動遠端工作。

## 安裝

需要 **Python 3.10 以上**。請在獨立環境操作：

```sh
python -m venv .venv
```

啟用環境後安裝核心程式與測試工具：

```sh
python -m pip install -e ".[test]"
```

需要執行神經模型及完整 CPU 測試時，再安裝模型依賴：

```sh
python -m pip install -e ".[test,models]"
```

PyTorch 與 torchvision 必須相容；CPU／CUDA 安裝請依自己的環境選擇。安裝依賴可能需要網路；範例與測試不需要連線、GPU、帳號或下載預訓練權重。

## 離線範例與測試

```sh
python examples/offline_demo.py
python -m pytest -q
```

範例使用虛構 study ID 與合成機率，示範固定 80/20 融合及 12 目標輸出驗證。它不是醫療推論、模型成效或提交資格證據。`test` 依賴包含 PyTorch，供小型 CPU 模型測試使用；完整測試不下載實際骨幹或 checkpoint。

公開內容檢查：

```sh
python tools/audit_public_repo.py --history
```

此檢查會掃描已追蹤檔案與可達歷史，回報可疑類型及檔案位置，不列出命中的秘密原文。它是發布檢查的一部分，不是完整資安或醫療去識別化保證。

## 使用自己的資料

競賽資料須由使用者自行依 [官方競賽規則](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules) 取得並保存在 Git 之外。不要提交 DICOM、報告、逐筆標籤、study UID 清單、預測 CSV 或權重。

工具的輸入格式與限制見 [使用說明](docs/USAGE.md)。模型程式不會自動取得歷史 checkpoint；外部初始化權重及其授權由使用者另外核實。

## 來源與安全

- [來源與授權說明](docs/SOURCES.md)：保留程式的範圍與未再散布的第三方資產。
- [安全與隱私說明](SECURITY.md)：資料邊界、可信 checkpoint 與發布前檢查。

本儲存庫自行開發的程式與文件採用 [MIT License](LICENSE)。第三方套件、競賽資料與外部模型權重各自適用其原有條款，本授權不授予這些外部資產的權利。這些研究程式未經臨床驗證，不供醫療診斷。
