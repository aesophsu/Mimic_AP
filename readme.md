## 重症AP预测模型：标准化研究流

### 第一阶段：数据工程与资产基石 (Foundation)

* **01_mimic_sql_extraction.sql**: [原始提取] 建立 MIMIC 临床队列并关联结局指标，产出 `data/raw/mimic_raw_data.csv`。
* **02_mimic_cleaning.py**: [审计与对齐] 执行结局指标逻辑重构、单位换算、生理极值清洗与 1%-99% 盖帽处理，产出 data/cleaned/mimic_raw_scale.csv。
* **03_mimic_standardization.py**: [张量化] 执行特征缩放与中位数/MICE插补，产出 `data/cleaned/mimic_processed.csv` 与关键资产 `artifacts/assets/mimic_scaler.joblib`。

### 第二阶段：描述统计与审计 (Audit)

* **04_mimic_stat_audit.py**: [基线分析] 自动化生成 Table 1 基线表与缺失值分布图，产出 `results/tables/table1_baseline.csv`。

### 第三阶段：特征精炼与模型竞赛 (Modeling)

* **05_feature_selection_lasso.py**: [降维] 基于 1-SE 准则通过 LASSO 筛选强预测特征，产出 `features/selected_features.json`。
* **06_model_training_optuna.py**: [进化寻优] 调用 Optuna 引擎进行多模型超参数搜索与 5-Fold 交叉验证，产出 `artifacts/models/best_model_trial.pkl`。
* **07_probability_calibration.py**: [概率校准] 对最优模型进行 Isotonic/Sigmoid 概率对齐，产出校准后的模型资产 `artifacts/models/calibrated_model_bundle.pkl`。
* **08_optimal_cutoff_analysis.py**: [阈值绑定] 基于 Youden Index 计算最优诊断切分点，产出 `artifacts/assets/thresholds.json`。

### 第四阶段：外部验证与人群迁移 (Validation)

* **09_eicu_sql_extraction.sql**: [定向提取] 依据 `selected_features.json` 变量清单在 eICU 数据库进行映射提取，产出 `data/raw/eicu_raw_data.csv`。
* **10_eicu_alignment_standardization.py**: [跨库对齐] 强制加载 `mimic_scaler.joblib` 对外部数据进行同分布转化，产出 `data/external/eicu_processed.csv`。
* **11_cross_cohort_drift_test.py**: [漂移分析] 对比 MIMIC 与 eICU 的特征分布差异（PSI/KL散度），产出 `results/figures/audit/drift_report.png`。
* **12_external_validation_perf.py**: [盲测验证] 使用校准模型对 eICU 进行零样本推理，产出跨库验证对比表 `results/tables/external_validation_metrics.csv`。

### 第五阶段：临床解释与转化决策 (Interpretation)

* **13_model_interpretation_shap.py**: [黑盒拆解] 计算全局 SHAP 贡献度与个体样本解释，产出 `results/figures/interpretation/shap_summary.png`。
* **14_clinical_decision_tool.py**: [转化工具] 绘制 DCA 决策曲线并导出逻辑回归评分板，产出 `results/figures/clinical/DCA_curve.png` 与 `results/tables/nomogram_points.csv`。

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
