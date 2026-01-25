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
│   ├── raw/                           # [Immutable] 原始数据快照
│   │   ├── mimic_raw_data.csv         # [01步] SQL 提取产物
│   │   └── eicu_raw_data.csv          # [08步] SQL 提取产物
│   ├── cleaned/                       # [Internal] MIMIC 中间产物
│   │   ├── mimic_raw_scale.csv        # [02步] 物理清洗后数据 (用于 Table 1)
│   │   └── mimic_processed.csv        # [03步] 建模张量 (Log+MICE+Z-Score)
│   └── external/                      # [External] eICU 验证产物
│       ├── eicu_aligned.csv           # [09步] 逻辑对齐后的物理值
│       └── eicu_processed_{target}.csv# [09步] 结局专属推理张量 (已标准化)
│
├── scripts/                           # 全流程标准化脚本
│   ├── 01_sql/                        # 数据提取
│   │   ├── 01_mimic_sql_extraction.sql
│   │   └── 08_eicu_external_extraction.sql
│   ├── 02_preprocess/                 # 清洗与特征工程
│   │   ├── 02_mimic_cleaning.py       
│   │   ├── 03_mimic_standardization.py
│   │   └── 09_eicu_alignment_cleaning.py
│   ├── 03_modeling/                   # 建模与筛选
│   │   ├── 05_feature_selection_lasso.py
│   │   ├── 06_model_training_main.py  
│   │   └── 07_optimal_cutoff_analysis.py
│   ├── 04_audit_eval/                 # 审计、验证与临床转化
│   │   ├── 04_mimic_stat_audit.py        
│   │   ├── 10_cross_cohort_audit.py
│   │   ├── 11_external_validation_perf.py
│   │   ├── 12_model_interpretation_shap.py
│   │   ├── 13_clinical_calibration_dca.py 
│   │   └── 14_nomogram_odds_ratio.py
│   └── utils/                          # 工具中枢
│       ├── translation_config.py          # 静态配置：特征中英翻译、物理单位映射    
│       ├── feature_manager.py             # 字典维护：自动注入翻译/单位至 JSON  
│       ├── force_update_blacklist.py      # 预处理控制：强制锁定无需 Log 变换的特征
│       ├── feature_utils.py               # 文本渲染：LaTeX 下标美化 (PaO2 -> $PaO_2$)
│       ├── plot_config.py                 # 视觉规范：出版级 DPI、字号、配色标准
│       ├── plot_utils.py                  # 绘图计算：OR 偏移量计算与坐标轴自动缩放
│       ├── skewed_cols_check.py           # 阶段审计：LASSO 特征子集与数据一致性校验
│       └── post_analysis_tools.py         # 终产物审计：Step 07 资产完整性与 Table 3 复核
│
│
├── artifacts/                         # [核心] 资产中枢
│   ├── features/                      
│   │   ├── feature_dictionary.json    # 特征定义全集
│   │   └── selected_features.json     # [05步] 全局特征清单
│   ├── scalers/                       # [03步] 预处理标尺
│   │   ├── mimic_scaler.joblib        # 标准化器
│   │   ├── mimic_mice_imputer.joblib  # 插补模型
│   │   ├── train_assets_bundle.pkl    # [重要] 特征顺序记忆与 Log 策略
│   │   └── skewed_cols_config.pkl     
│   ├── models/                        # [06-07步] 结局专属资产
│   │   ├── performance_report.csv     # 训练集性能汇总
│   │   ├── global_diagnostic_summary.csv # [07步] 诊断指标汇总
│   │   └── {target}/                  # (pof / mortality / composite)
│   │       ├── all_models_dict.pkl    # 所有校准后的模型
│   │       ├── deploy_bundle.pkl      # [06步] 部署包 (特征+Scaler+模型)
│   │       ├── selected_features.json # 该结局专用特征
│   │       ├── thresholds.json        # [07步] 最佳 Youden Index 截断值
│   │       ├── eval_data.pkl          # 固化测试集 (X_test, y_test, mask)
│   │       └── feature_importance.csv # 特征权重表
│   └── validation/                    # [10-11步] 外部验证中间态
│       ├── eicu_vs_mimic_drift.json   # [10步] 漂移审计报告
│       └── external_perf_metrics.csv  # [11步] 外部验证指标缓存
│
├── results/                           # [Paper] 论文最终产出
│   ├── tables/                        
│   │   ├── table1_baseline.csv        # [03步]
│   │   ├── table2_renal_subgroup.csv  # [03步]
│   │   ├── Table3_Final_Performance.csv # [07步] 内部验证终表
│   │   ├── Table4_External_Validation.csv # [11步] 外部验证终表
│   │   └── OR_Statistics_{target}.csv # [14步] 比值比统计表
│   └── figures/                       
│       ├── audit/                     
│       │   └── mimic_missing_heatmap_pro.png # [04步]
│       ├── lasso/                     
│       │   ├── lasso_diag_{target}.png       # [05步]
│       │   └── lasso_importance_{target}.png # [05步]
│       ├── {target}/                  # 内部验证影像
│       │   ├── {target}_ROC.pdf              # [06步]
│       │   ├── {target}_Calibration.pdf      # [06步]
│       │   └── 07_Diagnostic_{name}.pdf      # [07步] 阈值分布图
│       ├── comparison/                # 外部验证影像
│       │   ├── dist_drift_{feat}.png         # [10步] 分布漂移
│       │   ├── ROC_External_{target}.pdf     # [11步] 跨库 ROC 对比
│       │   └── Table4_Performance_Vis.png    # [11步]
│       ├── interpretation/            # 可解释性影像
│       │   ├── Fig4A_Summary_{target}.pdf    # [12步] 蜂群图
│       │   ├── Fig4B_Force_{target}.pdf      # [12步] 个体决策图
│       │   └── Fig4C_Dep_{target}.png        # [12步] 依赖图
│       └── clinical/                  # 临床转化影像
│           ├── Fig5_DCA_Calibration_{target}.pdf # [13步] DCA 决策曲线
│           ├── Forest_Plot_{target}.pdf      # [14步] 森林图
│           └── Nomogram_{target}.pdf         # [14步] 列线图
│
└── logs/                              # 系统运行日志与 Optuna 历史

```

---
