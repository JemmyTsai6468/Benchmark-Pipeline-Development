# MVTec 異常檢測自動化基準測試管道 (Benchmark Pipeline) 架構說明

## 1. 總覽

本文件旨在說明一個為 MVTec Anomaly Detection (AD) 資料集設計的自動化基準測試管道 (Benchmark Pipeline)。此管道的目標是提供一個可重複、可擴充且高度自動化的框架，用以評估和比較不同異常檢測模型的性能。

此設計嚴格遵守 `uv` Python 專案的最佳實踐，並確保與 MVTec 官方提供的原始評估腳本完全相容且不作任何修改。

## 2. 設計理念

此架構基於三大核心設計理念：

*   **配置驅動 (Configuration-Driven)**：所有實驗的變動（如新增模型、更改路徑）都透過修改設定檔來完成，而非修改程式碼。這使得管道極具擴充性與靈活性。

*   **關注點分離 (Separation of Concerns)**：將核心邏輯 (`src`)、設定 (`configs`)、資料 (`data`)、模型輸出 (`models`)、實驗結果 (`results`) 和外部工具 (`tools`) 嚴格分離。這使得專案結構清晰、易於維護。

*   **工具不變性 (Immutability of Tools)**：將官方評估腳本視為不可變的「黑盒子」工具，僅透過標準命令列介面與之互動。這 100% 保證了所有評估的標準一致性與可重複性。

## 3. 目錄結構

```
mvtec_benchmark_pipeline/
├── pyproject.toml            # 專案定義與依賴管理 (uv)
├── README.md                 # 專案說明與使用指南
├── benchmark.md              # (本檔案) 架構說明
├── .gitignore                # 忽略不必要的檔案
│
├── configs/                  # 集中管理所有設定檔
│   └── benchmark_config.yaml # -> 定義要評估的模型、路徑等
│
├── data/                     # 存放資料集
│   └── mvtec_ad/             # -> MVTec AD 資料集應放置於此
│
├── models/                   # 存放模型產生的異常圖
│   ├── model_a/
│   │   └── anomaly_maps/
│   └── model_b/
│       └── anomaly_maps/
│
├── results/                  # 存放所有實驗結果
│   ├── model_a/
│   │   └── metrics.json
│   ├── model_b/
│   │   └── metrics.json
│   └── summary_report.csv    # -> 自動生成的最終比較報告
│
├── src/mvtec_benchmark/      # Pipeline 核心原始碼
│   ├── __init__.py
│   ├── main.py               # -> Pipeline 的主進入點
│   ├── pipeline.py           # -> 執行評估的核心邏輯
│   └── reporting.py          # -> 生成匯總報告的邏輯
│
└── tools/                    # 存放外部工具 (保持原樣)
    └── evaluation_scripts/   # -> 放置原始的評估腳本
        ├── evaluate_experiment.py
        ├── pro_curve_util.py
        ├── roc_curve_util.py
        └── ...
```

## 4. 元件詳解

*   **`pyproject.toml`**:
    專案的核心定義檔，供 `uv` 等工具使用。它鎖定了所有 Python 依賴項，確保任何使用者都能建立完全一致的執行環境，是實現**可重複性**的基石。

*   **`configs/benchmark_config.yaml`**:
    管道的控制中心。使用者在此檔案中定義所有待評估模型的名稱、其異常圖 (`anomaly_maps`) 的路徑以及資料集的位置。新增或移除模型僅需修改此檔案，無需接觸任何程式碼，是實現**可擴充性**的關鍵。

*   **`data/mvtec_ad/`**:
    存放 MVTec AD 官方資料集的標準位置。程式碼與資料分離，使專案管理更清晰。

*   **`models/`**:
    存放各個模型產生的異常圖。此目錄的結構讓 pipeline 能標準化地讀取不同模型的評估輸入。

*   **`results/`**:
    所有實驗輸出的存放地。管道會為每個模型建立子目錄來存放其 `metrics.json` 檔案，並在根目錄下生成一份所有模型的性能匯總報告 `summary_report.csv`。

*   **`src/mvtec_benchmark/`**:
    實現**自動化**流程的核心程式碼。
    *   `main.py`: 整個管道的命令列進入點，負責解析命令並啟動評估流程。
    *   `pipeline.py`: 包含執行單一模型評估的核心邏輯。它會讀取設定檔，為指定的模型建構命令列參數，並呼叫 `tools/` 中的 `evaluate_experiment.py` 腳本。
    *   `reporting.py`: 在所有模型評估完成後執行。它會掃描 `results/` 目錄中的所有 `metrics.json`，將關鍵指標（如 `mean_au_pro`, `mean_classification_au_roc`）提取出來，並整合成一個易於比較的 CSV 報告。

*   **`tools/evaluation_scripts/`**:
    一個獨立的目錄，用於原封不動地存放官方評估腳本。我們的 pipeline 透過 `subprocess` 模組來執行這些腳本，確保評估方法與論文保持**絕對一致**。

## 5. 工作流程

一個典型的基準測試流程如下：

1.  **準備**: 使用者將 MVTec AD 資料集放置在 `data/mvtec_ad/` 目錄下。
2.  **設定**: 使用者將自己模型的異常圖輸出放置在 `models/` 目錄下，並在 `configs/benchmark_config.yaml` 中註冊該模型的名稱與路徑。
3.  **執行**: 使用者在終端執行 `uv run python src/mvtec_benchmark/main.py`。
4.  **自動評估**:
    *   `main.py` 啟動 `pipeline.py`。
    *   `pipeline.py` 讀取設定檔，並為每個註冊的模型依序執行評估。
    *   每個模型的評估結果 `metrics.json` 被儲存到 `results/<model_name>/` 中。
5.  **產出報告**: 所有模型評估完畢後，`reporting.py` 被觸發，自動生成一份 `results/summary_report.csv`，其中包含了所有模型的性能對比。

# Coding Style

- **Single Responsibility Principle (SRP)**: A class or module should have only one reason to change.
- **Keep It Simple, Stupid (KISS)**: Prioritize simplicity and avoid unnecessary complexity.
- **Explicit is Better than Implicit**: Make code clear and direct.
- **Write Meaningful Comments**: Explain the *why* (intent), not the *what*.
- **Consistency**: Adhere to consistent code style (e.g., PEP 8) and formatting.
- **Open-Closed Principle (OCP)**: Software should be open for extension but closed for modification.
- **High Cohesion, Low Coupling**: Modules should be focused and independent.
- **You Aren't Gonna Need It (YAGNI)**: Implement only what is necessary now.
- **Don't Repeat Yourself (DRY)**: Avoid code duplication, but without creating harmful abstractions.
- **Separation of Concerns**: Divide programs into distinct, independent sections.
- **Fail-Fast**: Surface errors as early as possible.
- **Avoid Premature Optimization**: Write clear code first, then optimize only when needed.