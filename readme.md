## 🚀 重症预测模型：14步标准化研究流 (MIMIC -> eICU)

### 第一阶段：数据工程与基石构建 (MIMIC)

本阶段的核心是将 SQL 原始数据转化为可进入模型的结构化张量，并建立特征元数据字典。

* **`01_mimic_sql_extraction.sql`**：数据库提取层。输出 `mimic_raw_data.csv`。
* **`02_mimic_cleaning.py`**：**[字典驱动审计]**。加载 `feature_dictionary.json`，执行自动单位换算（如 BUN 校准）、反对数还原（AST/ALT）及生理范围过滤。
* **`03_mimic_standardization.py`**：**[资产化与分流]**。
* **亚组定义**：在此定义“无预存肾损”等临床亚组标记。
* **分流保存**：保存物理值版 `mimic_raw_scale.csv`（用于统计）与 Z-score 归一化版 `mimic_processed.csv`（用于建模）。
* **持久化资产**：保存 `mimic_scaler.joblib`，确保 MIMIC 的缩放参数可被 eICU 严格复用。



### 第二阶段：统计审计与基线分析

在建模前，确保数据符合临床分布规律并产出论文核心表格。

* **`04_mimic_stat_audit.py`**：**[深度审计]**。利用 `TableOne` 产出 Table 1（POF 对比）与 Table 2（亚组对比）。执行单因素分析（P-value 过滤）与缺失率热图审计。

### 第三阶段：特征精炼与模型竞赛 (Modeling)

* **`05_feature_selection_lasso.py`**：**[降维精炼]**。基于 1-SE 准则进行 LASSO 降维，输出 `selected_features.json`（包含最终入模特征及其跨库映射名）。
* **`06_model_training_main.py`**：**[平行训练]**。针对 POF, Mortality 等不同结局平行训练 LR, XGB, LGBM, RF, MLP，资产按结局分类存入 `artifacts/models/`。

### 第四阶段：截断值计算与效能审计 (Clinical Cutoff)

将概率模型转化为临床诊断工具。

* **`07_optimal_cutoff_analysis.py`**：
* **功能**：在开发集寻找 Youden Index 最大点，确定各模型的**最佳截断值**。
* **资产**：将 Cutoff 值写入各模型目录下的 `thresholds.json`，实现模型与阈值绑定。
* **审计**：输出内部验证的敏感度、特异度、NPV/PPV 及 F1-score。



### 第五阶段：后置外部验证 (eICU Validation)

基于“特征清单”去外部库捞数，实现严格的跨中心验证。

* **`08_eicu_sql_extraction.sql`**：**[元数据驱动提取]**。依据 `selected_features.json` 在 eICU 中提取对应列。
* **`09_eicu_alignment_cleaning.py`**：**[严格对齐]**。加载 `feature_dictionary.json` 换算物理单位，并**加载** `mimic_scaler.joblib` 进行尺度变换（不重新 fit），输出 `eicu_processed.csv`。
* **`10_cross_cohort_audit.py`**：**[跨库表 1]**。对比两库基线特征差异，分析人群漂移（Population Drift）。
* **`11_external_validation_perf.py`**：**[盲测评价]**。加载 MIMIC 模型与阈值，在 eICU 上计算外部 AUC/AUPRC 及迁移后的临床效能。

### 第六阶段：临床解释与转化决策 (Translation)

完成从“黑盒模型”到“临床洞察”的转化。

* **`12_model_interpretation_shap.py`**：利用 SHAP 值解释特征对预测结果的贡献度（全局与局部解析）。
* **`13_clinical_calibration_dca.py`**：绘制校准曲线（评估预测概率与真实风险的吻合度）及决策曲线（DCA，评估临床净获益）。
* **`14_nomogram_odds_ratio.py`**：生成可视化诺莫图，计算变量的比值比 (OR) 及 95% 置信区间。


### 📂 项目目录树

```text
project_root/
│
├── data/
│   ├── raw/                           # 原始数据快照 (Immutable)
│   │   ├── mimic_raw_data.csv         # 01 步 SQL 提取产物
│   │   └── eicu_raw_data.csv          # 08 步 SQL 提取产物 (依据 selected_features.json)
│   ├── cleaned/                       # MIMIC 开发集中间产物
│   │   ├── mimic_raw_scale.csv        # 02 步产出：Log 转换前的物理尺度数据 (用于 Table 1)
│   │   └── mimic_processed.csv        # 03 步产出：Log 转换 + MICE 插补 + 标准化后的张量
│   └── external/                      # eICU 验证集中间产物
│       ├── eicu_aligned.csv           # 09 步产出：物理单位对齐后的数据
│       └── eicu_processed.csv         # 09 步产出：应用 MIMIC Scaler 变换后的建模数据
│
├── scripts/                           # 14 步标准化工作流
│   ├── 01_sql/                        # 数据库提取层 (提取 SQL)
│   ├── 02_preprocess/                 # 特征工程层
│   │   ├── 02_mimic_cleaning.py       # 物理清洗、字典对齐
│   │   ├── 03_mimic_standardization.py # 剥离 Scaler、Log 转换、MICE 插补、保存持久化资产
│   │   └── 09_eicu_alignment_cleaning.py
│   ├── 03_modeling/                   # 模型竞赛层
│   │   ├── 05_feature_selection_lasso.py # 执行 1-SE 准则、学术路径图、产出特征清单
│   │   ├── 06_model_training_main.py  # 读取清单、Optuna 寻优、5 大模型竞赛、概率校准
│   │   └── 07_optimal_cutoff_analysis.py # [规划] 计算 Youden Index 最佳截断值
│   └── 04_audit_eval/                 # 验证与统计层
│       ├── 04_mimic_stat_audit.py     # 深度描述统计、缺失值热图
│       ├── 11_external_validation_perf.py 
│       ├── 12_model_interpretation_shap.py # 针对精炼特征的全局/个体 SHAP 解释
│       ├── 13_clinical_calibration_dca.py # 决策曲线分析 (DCA)
│       └── 14_nomogram_odds_ratio.py      # 列线图与 OR 值导出
│
├── artifacts/                         # 项目的大脑：跨脚本调用的中枢资产
│   ├── models/                            # 06步
│   │   ├── performance_report.csv         # 06步：所有结局/算法的汇总性能表
│   │   ├── pof/                           # 06步：POF 结局专属资产
│   │   │   ├── all_models_dict.pkl        # 包含 5 种校准后的模型字典
│   │   │   ├── scaler.pkl                 # 针对 POF 特征子集的标准化器
│   │   │   ├── imputer.pkl                # 针对 POF 特征子集的插补器
│   │   │   ├── selected_features.json     # 该模型实际输入的特征清单
│   │   │   ├── optuna_study.pkl           # XGBoost 参数寻优记录
│   │   │   └── eval_data.pkl              # 测试集张量与亚组 Mask (用于后续统计)
│   │   ├── mortality/                     # 06步：死亡结局专属资产
│   │   │   ├── all_models_dict.pkl        # 包含 5 种校准后的模型字典
│   │   │   ├── scaler.pkl                 # 针对 POF 特征子集的标准化器
│   │   │   ├── imputer.pkl                # 针对 POF 特征子集的插补器
│   │   │   ├── selected_features.json     # 该模型实际输入的特征清单
│   │   │   ├── optuna_study.pkl           # XGBoost 参数寻优记录
│   │   │   └── eval_data.pkl              # 测试集张量与亚组 Mask (用于后续统计)
│   │   ├── composite/                     # 06步：复合结局专属资产
│   │   │   ├── all_models_dict.pkl        # 包含 5 种校准后的模型字典
│   │   │   ├── scaler.pkl                 # 针对 POF 特征子集的标准化器
│   │   │   ├── imputer.pkl                # 针对 POF 特征子集的插补器
│   │   │   ├── selected_features.json     # 该模型实际输入的特征清单
│   │   │   ├── optuna_study.pkl           # XGBoost 参数寻优记录
│   │   │   └── eval_data.pkl              # 测试集张量与亚组 Mask (用于后续统计)
│   ├── scalers/                       # 尺度转换持久化文件 (核心！)
│   │   ├── mimic_scaler.joblib        # 03 步保存的 StandardScaler
│   │   ├── mimic_mice_imputer.joblib  # 03 步保存的 MICE Imputer
│   │   └── skewed_cols_config.pkl     # 记录需要进行 Log1p 转换的列名
│   └── features/                      # 特征中枢配置
│       ├── feature_dictionary.json    # 特征定义全集
│       └── selected_features.json     # 05 步 LASSO 产出的 Top 12 精简清单
│
├── results/                           # 产出层 (直接用于论文)
│   ├── tables/                        # CSV 统计报表 (Table 1-4, OR表, 性能汇总)
│   └── figures/                       # 高清科研插图 (png/pdf/svg)
│       ├── audit/                     # 缺失值热图、亚组分布图
│       ├── lasso/                     # 05 步：Lasso CV 路径图与 1-SE 诊断图
│       ├── pof/                           # 06步
│       │   ├── ROC_Curve.png              # POF 结局多算法对比 ROC 图
│       │   └── Calibration_Curve.png      # POF 结局校准曲线图
│       ├── mortality/                     # 06步
│       │   ├── ROC_Curve.png              # mortality 结局多算法对比 ROC 图
│       │   └── Calibration_Curve.png      # mortality 结局校准曲线图
│       └── composite/                     # 06步
│       │   ├── ROC_Curve.png              # composite 结局多算法对比 ROC 图
│       │   └── Calibration_Curve.png      # composite 结局校准曲线图
│
├── logs/                              # 运行审计与 Optuna 寻优日志
└── requirements.txt                   # 环境依赖 (shap, optuna, xgboost, tableone等)

```

---

### 🛠️ 流程核心逻辑保障

1. **特征对齐中枢**：通过 `feature_dictionary.json` 解决了不同数据库间“同名不同义”或“同义不同名”的问题，是确保外部验证成功的关键。
2. **资产分层管理**：将 `thresholds.json` 与模型文件绑定，确保从概率输出到临床决策的每一步都有据可查。
3. **结果隔离性**：`results/figures/` 的子文件夹设计，让您在处理三种不同临床终点时，图表输出井然有序，绝不混淆。

---
