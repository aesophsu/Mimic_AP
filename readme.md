## 🚀 重症预测模型：14步标准化研究流 (MIMIC -> eICU)

### 第一阶段：数据工程与基石构建 (MIMIC)

本阶段的核心是将 SQL 原始数据转化为可进入模型的结构化张量，并建立特征元数据字典。

* **`01_mimic_sql_extraction.sql`**：提取 MIMIC 候选特征与结局，输出 `mimic_raw_data.csv`。
* **`02_mimic_cleaning.py`**：执行缺失值审计、异常值处理及衍生变量计算，并**同步初始化** `feature_dictionary.json`。
* **`03_mimic_standardization.py`**：
* 保存 `mimic_raw_scale.csv` (物理单位，用于 Table 1-4)。
* 保存 `mimic_processed.csv` (数值归一化版)。
* **持久化资产**：保存 `mimic_scaler.joblib` 供外部验证使用。



### 第二阶段：统计审计与基线分析

在建模前，确保数据符合临床分布规律。

* **`04_mimic_stat_audit.py`**：执行单因素分析、肾功能亚组对比及三结局发生率审计（产出 Table 2 & 3）。

### 第三阶段：特征精炼与模型竞赛 (Modeling)

* **`05_feature_selection_lasso.py`**：基于 1-SE 准则进行 LASSO 降维。
* **关键产出**：`selected_features.json`（仅含入模特征及其在 eICU 中的映射指令）。


* **`06_model_training_main.py`**：针对三种结局（POF, Mortality, Composite）训练 5 大模型，并将资产分文件夹存入 `artifacts/models/`。

### 第四阶段：截断值计算与效能审计 (Clinical Cutoff)

将概率模型转化为临床诊断工具。

* **`07_optimal_cutoff_analysis.py`**：
* **功能**：基于开发集寻找 Youden Index 最大点，确定各模型的**最佳截断值**。
* **资产**：将 Cutoff 值写入各子目录下的 `thresholds.json`。
* **审计**：输出内部验证的敏感度、特异度、NPV/PPV 及 F1-score。



### 第五阶段：后置外部验证 (eICU Validation)

基于“特征清单”去外部库捞数，实现严格的外部对齐。

* **`08_eicu_sql_extraction.sql`**：**[元数据驱动]** 依据 `selected_features.json` 提取 eICU 对应特征。
* **`09_eicu_alignment_cleaning.py`**：根据字典执行单位换算（如 µmol/L -> mg/dL），产出 `eicu_processed.csv`。
* **`10_cross_cohort_audit.py`**：执行 MIMIC vs eICU 的跨库基线对比（产出 Table 1）。
* **`11_external_validation_perf.py`**：
* **效能评估**：计算 eICU 上的外部 AUC/AUPRC。
* **迁移审计**：直接应用 MIMIC 的阈值，评估模型在外部环境的临床可靠性。



### 第六阶段：临床解释与转化决策 (Translation)

* **`12_model_interpretation_shap.py`**：利用 SHAP 值解释模型决策的临床透明度。
* **`13_clinical_calibration_dca.py`**：绘制校准曲线（可靠性）与决策曲线（临床获益度）。
* **`14_nomogram_odds_ratio.py`**：生成诺莫图并计算变量的比值比 (OR)，完成学术报告。

---

### 📂 项目目录树 (Standardized V3.0)

```text
project_root/
│
├── data/
│   ├── raw/                      # 原始 SQL 提取快照 (只读)
│   │   ├── mimic_raw_data.csv    # 01 步产出
│   │   └── eicu_raw_data.csv     # 08 步产出 (仅含选定特征)
│   ├── cleaned/                  # MIMIC 处理后的数据集
│   │   ├── mimic_raw_scale.csv   # 03 步：物理单位版 (用于统计描述)
│   │   └── mimic_processed.csv   # 03 步：数值归一化版 (用于模型训练)
│   └── external/                 # eICU 处理后的数据集
│       ├── eicu_aligned.csv      # 09 步：完成单位换算与变量对齐
│       └── eicu_processed.csv    # 09 步：使用 MIMIC Scaler 转换后的验证集
│
├── scripts/                      # 14 步标准化工作流核心脚本
│   ├── 01_sql/                   # 01_mimic_extraction.sql, 08_eicu_extraction.sql
│   ├── 02_preprocess/            # 02, 03, 09 步：清洗、标准化与跨库对齐
│   ├── 03_modeling/              # 05, 06, 07 步：筛选、训练与截断值寻优
│   └── 04_audit_eval/            # 04, 10-14 步：基线审计、验证与解释
│
├── artifacts/                    # 全局共享资产与模型大脑
│   ├── models/                   # 按结局(pof/mortality/composite)分类存放
│   │   └── {outcome}/            # 包含 best_models.joblib & thresholds.json
│   ├── scalers/                  # mimic_scaler.joblib (用于 eICU 尺度对齐)
│   └── features/                 # 特征对齐中枢
│       ├── feature_dictionary.json   # 定义字段含义、单位、eICU 映射关系
│       └── selected_features.json    # 05 步生成，含模型入选特征及提取补丁
│
├── results/                      # 最终论文图表产出
│   ├── tables/                   # Table 1-4, 性能对比表 (.csv)
│   └── figures/                  # 按结局分目录存放 (AUC, SHAP, DCA, Nomogram)
│       └── {outcome}/            
│
├── logs/                         # 记录每步运行的时间戳与数据样本量变更
└── requirements.txt              # 运行环境依赖 (Python 3.9+)

```

---

### 🛠️ 流程核心逻辑保障

1. **特征对齐中枢**：通过 `feature_dictionary.json` 解决了不同数据库间“同名不同义”或“同义不同名”的问题，是确保外部验证成功的关键。
2. **资产分层管理**：将 `thresholds.json` 与模型文件绑定，确保从概率输出到临床决策的每一步都有据可查。
3. **结果隔离性**：`results/figures/` 的子文件夹设计，让您在处理三种不同临床终点时，图表输出井然有序，绝不混淆。

---
