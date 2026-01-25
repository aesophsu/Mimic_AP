## 重症AP预测模型：标准化研究流

### I. MIMIC 内部开发与建模阶段 (Steps 01-07)

**01. 数据提取**

* **脚本:** `01_mimic_sql_extraction.sql`
* **动作:** 建立临床队列，关联结局。
* **产出:** `data/raw/mimic_raw_data.csv`

**02. 清洗与对齐**

* **脚本:** `02_mimic_cleaning.py`
* **动作:** 逻辑重构、单位换算、盖帽处理 (1%-99%)。
* **产出:** `data/cleaned/mimic_raw_scale.csv`

**03. 标准化与基线**

* **脚本:** `03_mimic_standardization.py`
* **动作:** Log 转换、MICE 插补、Z-Score 标准化。
* **产出:**
* **张量:** `data/cleaned/mimic_processed.csv`
* **模型资产:** `artifacts/scalers/train_assets_bundle.pkl`, `mimic_scaler.joblib`, `mimic_mice_imputer.joblib`
* **统计表:** `results/tables/table1_baseline.csv`, `table2_renal_subgroup.csv`



**04. 数据审计**

* **脚本:** `04_mimic_stat_audit.py`
* **动作:** 审计完整性，绘制缺失模式。
* **产出:** `results/figures/audit/mimic_missing_heatmap_pro.png`

**05. 特征筛选 (LASSO)**

* **脚本:** `05_feature_selection_lasso.py`
* **动作:** 1-SE 准则筛选核心因子，标准化审计。
* **产出:**
* **特征表:** `artifacts/features/selected_features.json`, `artifacts/models/{target}/selected_features.json`
* **影像:** `results/figures/lasso/lasso_diag_{target}.png`, `results/figures/lasso/lasso_importance_{target}.png`



**06. 模型训练与寻优**

* **脚本:** `06_model_training_main.py`
* **动作:** 多模型竞赛 (Optuna)、概率校准。
* **产出:**
* **模型包:** `artifacts/models/{target}/all_models_dict.pkl`, `deploy_bundle.pkl`
* **评估:** `artifacts/models/performance_report.csv`, `feature_importance.csv`, `eval_data.pkl`
* **影像:** `results/figures/{target}/{target}_ROC.pdf`, `results/figures/{target}/{target}_Calibration.pdf`



**07. 阈值寻优与审计**

* **脚本:** `07_optimal_cutoff_analysis.py`
* **动作:** 确定 Youden Index，生成终版效能表。
* **产出:**
* **参数:** `artifacts/models/{target}/thresholds.json` (最佳截断值)
* **表格:** `results/tables/Table3_Final_Performance.csv`, `global_diagnostic_summary.csv`
* **影像:** `results/figures/{target}/07_Diagnostic_{name}.pdf`, `results/figures/sci_forest_plot.pdf`, `results/figures/sci_feature_importance.pdf`



---

### II. eICU 外部验证阶段 (Steps 08-11)

**08. 外部数据提取**

* **脚本:** `08_eicu_external_extraction.sql`
* **动作:** 外部队列打捞，单位与结局逻辑对齐。
* **产出:** `data/raw/eicu_raw_data.csv`, `eicu_cview.ap_external_validation` (视图)

**09. 跨库对齐与清洗**

* **脚本:** `09_eicu_alignment_cleaning.py`
* **动作:** 复用 MIMIC 资产 (Scaler/Imputer) 进行克隆式预处理。
* **产出:** `data/external/eicu_processed_{target}.csv`, `eicu_raw_scale.csv`

**10. 跨队列漂移审计**

* **脚本:** `10_cross_cohort_audit.py`
* **动作:** 计算 KS 统计量，量化特征分布漂移。
* **产出:** `validation/eicu_vs_mimic_drift.json`, `results/figures/comparison/dist_drift_{feat}_{target}.png`

**11. 外部验证性能评估**

* **脚本:** `11_external_validation_perf.py`
* **动作:** 加载模型盲测，Bootstrap 计算置信区间。
* **产出:**
* **表格:** `results/tables/Table4_External_Validation.csv`
* **影像:** `results/figures/comparison/ROC_External_{target}.pdf`, `Table4_Performance_Visualization.png`



---

### III. 临床解释与应用转化 (Steps 12-14)

**12. 模型可解释性 (SHAP)**

* **脚本:** `12_model_interpretation_shap.py`
* **动作:** 量化特征贡献，非线性分析。
* **产出:**
* **数据:** `results/figures/interpretation/shap_values/SHAP_Data_Export_{target}.csv`
* **影像:** `Fig4A_Summary_{target}.pdf`, `Fig4B_Force_{target}.pdf`, `Fig4C_Dep_{target}_{feat}.png`



**13. 临床决策分析 (DCA)**

* **脚本:** `13_clinical_calibration_dca.py`
* **动作:** 计算临床净获益，锚定最优切点。
* **产出:** `results/figures/clinical/DCA_Data_{target}.csv`, `Fig5_DCA_Calibration_{target}.pdf`

**14. 列线图与 OR 分析**

* **脚本:** `14_nomogram_odds_ratio.py`
* **动作:** LR 统计推断，构建临床评分工具。
* **产出:** `results/tables/OR_Statistics_{target}.csv`, `Forest_Plot_{target}_en.pdf`, `Nomogram_{target}_en.pdf`

---


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
│   └── external/                          # eICU 验证产物
│       ├── eicu_aligned.csv               # [09步] 经过字典对齐、单位换算后的物理值
│       └── eicu_processed_{target}.csv    # [09步] 应用对应结局 Scaler 后的标准数据
│
├── scripts/                           # 14 步标准化工作流
│   ├── 01_sql/                        # 数据库提取层 (提取 SQL)
│   │   ├── 01_mimic_sql_extraction.sql
│   │   ├── 08_eicu_sql_extraction.sql
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
│       ├── 10_cross_cohort_audit.py
│       ├── 11_external_validation_perf.py 
│       ├── 12_model_interpretation_shap.py # 针对精炼特征的全局/个体 SHAP 解释
│       ├── 13_clinical_calibration_dca.py # 决策曲线分析 (DCA)
│       └── 14_nomogram_odds_ratio.py      # 列线图与 OR 值导出
│
├── artifacts/                         # 项目的大脑：跨脚本调用的中枢资产
│   ├── models/                            # 06步
│   │   ├── performance_report.csv         # 06步：所有结局/算法的汇总性能表
│   │   ├── global_diagnostic_summary.csv         # 07步：全结局对比汇总表
│   │   ├── pof/
│   │   │   ├── all_models_dict.pkl        # 06步：包含 5 种校准后的模型字典
│   │   │   ├── scaler.pkl                 # 06步：针对 POF 特征子集的标准化器
│   │   │   ├── imputer.pkl                # 06步：针对 POF 特征子集的插补器
│   │   │   ├── selected_features.json     # 06步：该模型实际输入的特征清单
│   │   │   ├── optuna_study.pkl           # 06步：XGBoost 参数寻优记录
│   │   │   ├── eval_data.pkl              # 06 步：存入 X_test, y_test 和 subgroup_flag (No-Renal)
│   │   │   ├── bootstrap_ci_stats.pkl   # 06 步：存入 Dict {'main': (low, high), 'sub': (low, high)}防止 07 步重复跑 Bootstrap，极大节省时间
│   │   │   ├── feature_importance.csv   # 06 步产生：记录该结局下 5 大算法的特征权重排行
│   │   │   ├── thresholds.json                # 07步：POF 最佳截断值资产
│   │   │   └── internal_diagnostic_perf.csv   # 07步：POF 内部验证详细指标
│   │   ├── mortality/
│   │   │   ├── all_models_dict.pkl        # 06步：包含 5 种校准后的模型字典
│   │   │   ├── scaler.pkl                 # 06步：针对 POF 特征子集的标准化器
│   │   │   ├── imputer.pkl                # 06步：针对 POF 特征子集的插补器
│   │   │   ├── selected_features.json     # 06步：该模型实际输入的特征清单
│   │   │   ├── optuna_study.pkl           # 06步：XGBoost 参数寻优记录
│   │   │   ├── eval_data.pkl              # 06步：测试集张量与亚组 Mask (用于后续统计)
│   │   │   ├── bootstrap_ci_stats.pkl   # 06 步产生：存储全人群及“无肾损伤”亚组的 AUC 95% CI (Bootstrap 结果)
│   │   │   ├── feature_importance.csv   # 06 步产生：记录该结局下 5 大算法的特征权重排行
│   │   │   ├── thresholds.json                # 07步：死亡结局最佳截断值资产
│   │   │   └── internal_diagnostic_perf.csv   # 07步：死亡结局内部验证详细指标
│   │   ├── composite/
│   │   │   ├── all_models_dict.pkl        # 06步：包含 5 种校准后的模型字典
│   │   │   ├── scaler.pkl                 # 06步： 针对 POF 特征子集的标准化器
│   │   │   ├── imputer.pkl                # 06步：针对 POF 特征子集的插补器
│   │   │   ├── selected_features.json     # 06步：该模型实际输入的特征清单
│   │   │   ├── optuna_study.pkl           # 06步：XGBoost 参数寻优记录
│   │   │   ├── eval_data.pkl              # 06步：测试集张量与亚组 Mask (用于后续统计)
│   │   │   ├── bootstrap_ci_stats.pkl   # 06 步产生：存储全人群及“无肾损伤”亚组的 AUC 95% CI (Bootstrap 结果)
│   │   │   ├── feature_importance.csv   # 06 步产生：记录该结局下 5 大算法的特征权重排行
│   │   │   ├── thresholds.json                # 07步：复合结局最佳截断值资产
│   │   │   └── internal_diagnostic_perf.csv   # 07步：复合结局内部验证详细指标
│   ├── scalers/                       # 尺度转换持久化文件 (核心！)
│   │   ├── feature_metadata.json
│   │   ├── mimic_scaler.joblib        # 03 步保存的 StandardScaler
│   │   ├── mimic_mice_imputer.joblib  # 03 步保存的 MICE Imputer
│   │   ├── skewed_cols_config.pkl     # 记录需要进行 Log1p 转换的列名
│   │   └── train_assets_bundle.pkl    # 06 步：【枢纽】存储训练集特征列名顺序（Column Order）确保 eICU 输入模型的特征列顺序与训练时 100% 一致
│   │── features/                      # 特征中枢配置
│   │   ├── feature_dictionary.json    # 特征定义全集
│   │   └── selected_features.json     # 05 步 LASSO 产出的 Top 12 精简清单
│   └── validation/                     # 专门存放 11 步外部验证的中间对比资产
│       ├── eicu_vs_mimic_drift.json    # 由 10 步产生：记录人群偏移 (Population Drift) 的统计量
│       └── external_perf_metrics.csv   # 由 11 步产生：eICU 盲测下的 AUC/Brier/Calibration 斜率
│
├── results/                           # 产出层 (直接用于论文)
│   ├── tables/                        # CSV 统计报表 (Table 1-4, OR表, 性能汇总)
│   │   ├── Table3_Internal_Perf_pof.csv                # 07步
│   │   ├── Table3_Internal_Perf_mortality_28d.csv      # 07步
│   │   ├── Table3_Internal_Perf_composite_outcome.csv  # 07步
│   │   ├── Table4_External_Perf_Summary.csv  # 11 步产生：eICU 验证集的效能总表 (直接入论文)
│   │   └── Table_Subgroup_Analysis.csv       # 由 06/11 步产生：MIMIC 与 eICU 在 No-Renal 亚组下的稳健性对比
│   └── figures/                       # 高清科研插图 (png/pdf/svg)
│       ├── audit/                     # 缺失值热图、亚组分布图
│       ├── lasso/                     # 05 步：Lasso CV 路径图与 1-SE 诊断图
│       ├── pof/                           # 06步
│       │   ├── ROC_Curve.png              # POF 结局多算法对比 ROC 图
│       │   ├── Calibration_Curve.png      # POF 结局校准曲线图
│       │   └── 07_Diagnostic_XGBoost.png  # 07步：带 Cutoff 标注的 ROC 与分布图
│       ├── mortality/                     # 06步
│       │   ├── ROC_Curve.png              # mortality 结局多算法对比 ROC 图
│       │   ├── Calibration_Curve.png      # mortality 结局校准曲线图
│       │   └── 07_Diagnostic_Logistic Regression.png  # 07步：带 Cutoff 标注的 ROC 与分布图
│       └── composite/                     # 06步
│       │   ├── ROC_Curve.png              # composite 结局多算法对比 ROC 图
│       │   ├── Calibration_Curve.png      # composite 结局校准曲线图
│       │   └── 07_Diagnostic_Random Forest.png  # 07步：带 Cutoff 标注的 ROC 与分布图
│       ├── comparison/                 # 用于跨库对比的图表
│       │   ├── ROC_MIMIC_vs_eICU_{target}.png # 由 11 步产生：展示模型在两库间的迁移表现
│       │   └── Calibration_External_{target}.png # 由 11 步产生：eICU 验证集的校准度观察图
│       ├── interpretation/             # 模型解释度图表 (12 步)
│       │   ├── SHAP_Summary_{target}.png    # 由 12 步产生：特征贡献全局排名图
│       │   └── SHAP_Force_Plot_Sample.png   # 由 12 步产生：单个高风险病例的解释图
│       └── clinical/                   # 临床应用转化图表 (13/14 步)
│           ├── DCA_Benefit_Curve.png        # 由 13 步产生：决策曲线 (Decision Curve Analysis)
│           └── Nomogram_Visualization.png   # 由 14 步产生：临床医生可用的诺莫图评分板
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
