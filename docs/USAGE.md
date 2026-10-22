# 離線工具與資料介面

先依 README 安裝套件，從儲存庫根目錄執行以下命令。所有工具都不會提交 Kaggle 或讀取帳號憑證。

## 12 目標輸出驗證

```sh
python tools/validate_submission_output.py /path/to/output
python tools/validate_deployment_artifacts.py /path/to/output
```

第一個工具檢查根目錄的 `submission.csv`，可另傳預期 SHA-256 作為第二個參數。欄位須為 `StudyInstanceUID` 接 12 個目標；此工具要求至少三筆、UID 不重複、機率有限且位於 [0, 1]，並拒絕全表同值。它沒有預期 UID 表，因此不能單獨證明 UID 順序正確。

已知預期順序時，使用 `rsna_knee.artifacts.validate_submission(frame, reference)`；小於三筆的 test-only 輸出需使用 `rsna_knee.inference_execution.validate_probabilities` 並明確設定執行模式。

第二個工具另外要求根目錄的 `candidate_receipt.json` 與 `runtime_receipt.json`。它檢查欄位、CSV 雜湊及既有回執契約；回執聲明仍須由獨立執行證據核實，不等同正式提交資格。

## DICOM 與研究快取

可透過 `rsna_knee.dicom` 的函式選取序列、排序切片、依物理尺寸裁切及建立 study tensor。`rsna_knee.research_cache.build_study_cache` 提供研究視窗與 metadata 建置；函式介面及合成輸入見對應測試。

競賽 DICOM 目錄形式為 `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm`；需另外提供 study／series 表格。輸出影像陣列、UID 索引及 metadata 全部視為私人資料。

壓縮 DICOM 可能另外需要與 transfer syntax 相容的 pydicom 解碼外掛；安裝本套件不代表支援所有醫療影像編碼。

舊版固定 4,345 筆 cohort 的快取 CLI 未納入公開版，避免依賴沒有附帶的歷史 contract。此版提供可重用函式，不提供從原始資料到歷史比賽提交的完整流程。快取建置時間不能直接當作完整模型推論時間。

## OOF 比較

```sh
python tools/compare_research_oof.py --help
```

需要使用者自己的固定 contract，以及 control／candidate 各自的 `run_manifest.json` 與 `oof.csv`。程式檢查 fold、UID 與 checkpoint 回執的一致性後進行比較，不會查詢 Public 榜單。

報告分組 bootstrap 只能反映指定資料與標籤來源；使用報告弱標籤時，結果仍屬 proxy。相同報告分組不保證等同患者分組。

## Notebook 建置與擷取

```sh
python tools/nb.py /path/to/kernel-directory
python tools/ipynb_to_py.py /path/to/input.ipynb /path/to/output.py
```

建置目錄需提供 `kernel-metadata.json` 與 Python 入口，或受支援的基底 Notebook 設定。建置器可將套件嵌入 Notebook，並清除既有執行輸出；它只建置檔案，不執行或上傳。

只處理可信來源。生成 Notebook 預設不納入 Git，公開版亦不附帶歷史 kernel metadata 或第三方基底 Notebook。
