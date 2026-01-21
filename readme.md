### 第一阶段：数据工程 (MIMIC)

* **`01_mimic_sql_extraction.sql`**：提取 MIMIC 候选特征与结局。
* **`02_mimic_data_cleaning.py`**：清洗、缺失值处理、衍生变量计算。
* **`03_mimic_data_standardization.py`**：保存 `raw_scale` 和 `processed` 版本及 Scaler 资产。

### 第二阶段：统计审计 (Baseline)

* **`04_mimic_stat_audit.py`**：执行内部单因素、亚组及发生率对比（Table 2 & 3）。

### 第三阶段：模型竞赛与资产固化 (Modeling)

* **`05_feature_selection_lasso.py`**：LASSO 降维，确定最终特征清单。
* **`06_model_training_main.py`**：训练 5 大模型，保存模型资产。

### 第四阶段：截断值计算与效能审计 (Clinical Cutoff) —— **[新增步骤]**

* **`07_optimal_cutoff_analysis.py`**：
* **功能**：基于 MIMIC 开发集，通过 ROC 曲线寻找约登指数最大的点，确定**最佳截断值（Threshold）**。
* **审计**：在截断值下，计算并输出 MIMIC 内部的敏感度（Sensitivity）、特异度（Specificity）、阳性预测值（PPV）、阴性预测值（NPV）及 **F1-score**。
* **资产**：将每个结局对应的截断值保存到 `artifacts/models/thresholds.json`。



### 第五阶段：后置外部验证 (eICU Validation)

* **`08_eicu_sql_extraction.sql`**：根据筛选后的特征从 eICU 提数。
* **`09_eicu_alignment_cleaning.py`**：eICU 特征对齐与预处理。
* **`10_cross_cohort_audit.py`**：生成 MIMIC vs eICU 的跨库基线对比表（Table 1）。
* **`11_external_validation_perf.py`**：
* **概率评估**：计算 eICU 上的 AUC/AUPRC。
* **硬分类审计**：应用第 07 步保存的 `thresholds.json`，在 eICU 上验证该截断值的真实表现（是否依然保持高敏感度/特异度）。



### 第六阶段：临床解释与决策分析 (Translation)

* **`12_model_interpretation_shap.py`**：SHAP 全局与个体特征贡献分析。
* **`13_clinical_calibration_dca.py`**：校准曲线与决策曲线分析。
* **`14_nomogram_odds_ratio.py`**：诺莫图绘制与逻辑回归 OR 值分析。

---
为了完美契合你调整后的 **14步标准研究流程**，我为你重新梳理了详细的目录树。这个结构不仅支持多结局（POF, Mortality, Composite）的资产分类，还专门为**最佳截断值审计**和**后置 eICU 提取**留出了逻辑空间。

---

### 📂 项目目录树

```text
project_root/
│
├── data/
│   ├── raw/                      # 原始 SQL 提取快照 (只读)
│   │   ├── mimic_raw_data.csv    # 01 步产出
│   │   └── eicu_raw_data.csv     # 08 步产出 (仅含对齐后的特征)
│   ├── cleaned/                  # MIMIC 处理后的数据集
│   │   ├── mimic_raw_scale.csv   # 03 步：物理单位版 (用于统计描述)
│   │   └── mimic_processed.csv   # 03 步：数值归一化版 (用于模型训练)
│   └── external/                 # eICU 处理后的数据集
│       ├── eicu_aligned.csv      # 09 步：完成单位换算与变量对齐
│       └── eicu_processed.csv    # 09 步：使用 MIMIC Scaler 转换后的验证集
│
├── scripts/                      # 14 步标准化工作流
│   ├── 01_sql/                   # 数据库提取模块
│   │   ├── 01_mimic_extraction.sql
│   │   └── 08_eicu_extraction.sql
│   ├── 02_preprocess/            # 数据清洗与对齐
│   │   ├── 02_mimic_cleaning.py
│   │   ├── 03_mimic_standardization.py
│   │   └── 09_eicu_alignment_cleaning.py
│   ├── 03_modeling/              # 建模与阈值优化
│   │   ├── 05_feature_selection_lasso.py
│   │   ├── 06_model_training_main.py
│   │   └── 07_optimal_cutoff_analysis.py
│   └── 04_audit_eval/            # 统计、验证与解释
│       ├── 04_mimic_stat_audit.py
│       ├── 10_cross_cohort_audit.py
│       ├── 11_external_validation_perf.py
│       ├── 12_model_interpretation_shap.py
│       ├── 13_clinical_calibration_dca.py
│       └── 14_nomogram_odds_ratio.py
│
├── artifacts/                    # 全局共享资产 (核心)
│   ├── models/                   # 模型与截断值
│   │   ├── pof/                  # 包含 pof_best_models.joblib & thresholds.json
│   │   ├── mortality/
│   │   └── composite/
│   ├── scalers/                  # 预处理转换器
│   │   └── mimic_scaler.joblib   # 确保 eICU 验证时使用相同的 Z-score/MinMax 参数
│   └── features/                 # 特征对齐中枢
│       ├── feature_dictionary.json   # 定义字段含义、单位、eICU 映射关系
│       └── selected_features.json    # 05 步自动生成，存入最终入模特征
│
├── results/                      # 论文图表产出
│   ├── tables/                   # Table 1-4, 性能指标对比 CSV
│   └── figures/                  # 可视化图片 (PDF/PNG)
│       ├── pof/                  # POF 结局的 AUC, Calibration, DCA, SHAP
│       ├── mortality/
│       └── composite/
│
├── logs/                         # 记录每步运行的时间戳与数据行数变更
└── requirements.txt              # 环境依赖

```

---

### 🛠️ 关键目录逻辑说明

1. **`artifacts/models/` 的多结局设计**：
由于你有三个结局，每个结局的最佳截断值（Cutoff）和 5 大模型权重都不同。在 `07_optimal_cutoff_analysis.py` 中，我会帮你把结果分别存入对应的结局子目录下，防止覆盖。
2. **`artifacts/features/selected_features.json` 的中枢作用**：
这个文件是你“后置提取 eICU”的钥匙。`08_eicu_extraction.sql` 的编写将完全参照这个 JSON 中的字段。
3. **`results/figures/` 的隔离性**：
在运行第 11-14 步时，代码会自动根据当前处理的结局，将图表分流到对应的文件夹中。这能让你在最后撰写论文时，非常清晰地对比三个结局的表现。
4. **`data/external/` 的独立性**：
将 eICU 数据与 MIMIC 数据物理隔离。这在学术审计中非常重要，可以证明你没有在训练阶段“偷看”验证集数据。

---
