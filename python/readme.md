# 研究流程整理（终极闭环版）


### 📂 研究流程：SQL 层面的深度对齐 (Cross-database SQL Engineering)

#### **第零阶段：数据仓库挖掘与队列构建 (SQL Data Mining)**

### 模块 00-A: 自动化队列提取与结局定义 (SQL Implementation)

#### **核心内容**

* **多维队列精炼 (Multi-stage Cohort Refinement)**：
* **疾病识别**：利用 ICD-9 (5770) 与 ICD-10 (K85.x) 代码精准定位急性胰腺炎 (AP) 患者。
* **偏倚控制**：仅纳入首次 ICU 入院 (`stay_seq = 1`) 且住院时长  小时 (`los >= 1`) 的成年患者，排除因短期观察或复发入院导致的统计偏倚。
* **全局身高回填**：利用 `PERCENTILE_CONT(0.5)` 中位数逻辑打捞患者全病历身高质量数据，有效解决 BMI 计算中原始身高缺失率高的工程难题。


* **POF 结局标签构建与竞争风险审计 (POF & Competitive Risk)**：
* **持续性判定**：基于修正的 Marshall 评分逻辑，提取 ICU 入院 24h 至 7d 内的每日最高 SOFA 评分。通过 SQL 的 `FILTER` 聚合实现“持续性”判定，即呼吸、循环或肾脏分系统评分  且持续至少 2 个采样日（模拟  小时逻辑）。
* **早期死亡审计**：针对 ICU 入院 24-48h 内发生的早期死亡 (`early_death_24_48h`) 进行独立标记。该逻辑作为竞争风险审计，确保无法观测到 48h 持续性评分的极端危重患者被正确纳入 `composite_outcome`。


* **时序趋势特征提取 (Temporal Trend Extraction)**：
* **动态斜率 (Slopes)**：超越传统的静态极值，通过 SQL 计算入院 24h 内乳酸 (`lactate`)、血糖 (`glucose`) 及血氧 (`spo2`) 的变化斜率。
* **质量触发阈值**：严格设定“采样点  且时间间隔  小时”的计算条件，确保斜率特征具备生物学意义，抑制传感器噪声。



#### **产出**

* 生成结构化分析母表 `ap_final_analysis_cohort`，包含结局指标、动态趋势指标、合并症、干预措施及基线生理全景数据。

#### **审计笔记**

* **方法论深度**：在论文 Methods 中可强调：“To capture the dynamic evolution of AP, we integrated temporal trend features (slopes) and implemented a rigorous 48-hour persistent organ failure definition, accounting for early-death competitive risks to ensure the robustness of the primary endpoint.”
* **数据填充率保障**：SQL 脚本中内置了 `COALESCE` 与 `LEFT JOIN` 级联打捞机制（如 LAR、TBAR 等比值计算），确保了机器学习核心预测子的高填充率（Fill-rate），为模块 03 的 MICE 插补提供了高质量的冷启动数据。


### 模块 00-B: eICU 多中心数据特征审计与对齐 (SQL Level)

#### **核心内容**

* **异构数据物理归一化 (Automated Unit & Scale Alignment)**：
* **肌酐 (Creatinine) 智能校准**：通过 `>30` 的阈值探测自动识别 `umol/L`，并应用除以 **88.4** 的转换逻辑，统一为 `mg/dL` 尺度，抹平多中心数据最常见的量纲陷阱。
* **温度 (Temperature) 物理对齐**：通过 `80-115°F` 逻辑区间判定华氏度，并自动化转换为摄氏度，确保生理指标的物理一致性。
* **生理学区间清洗 (Range Scrubbing)**：对身高 (120-250cm)、体重 (30-300kg) 及 BMI (10-60) 实施严格的解剖生理学区间过滤，剔除系统占位符噪声。


* **pH 值四级打捞算法 (Tiered pH Recovery Algorithm)**：
* 针对 eICU 核心预测因子 pH 缺失率高的难题，构建了**“直接提取 -> 血气同步 -> 公式推算 -> 系统兜底”**的闭环打捞链：
1. **直接提取 (Lab)**：从常规生化表中提取直接测量值。
2. **血气对齐 (BG)**：同步打捞 `pivoted_bg` 表中的实时监测数据。
3. **公式推算 (Physiological Derivation)**：利用 **Henderson-Hasselbalch** 公式  进行动态补救。
4. **APACHE 兜底**：利用 `apacheapsvar` 评分组件中的预处理数据完成最终打捞。




* **POF 结局的“临床逻辑模拟” (Outcome Emulation & Logic Alignment)**：
* 在缺乏逐日 SOFA 评分的 eICU 环境下，利用“干预措施 + 暴露时长”构建高可信度结局标签：
* **呼吸衰竭**：`Ventilation`（含 CarePlan 与 Treatment 双重识别）且 ICU 时长  小时（严格排除术后常规插管）。
* **循环衰竭**：多类血管活性药物（含加压素、肾上腺素增强识别）且 ICU 时长  小时。
* **肾脏衰竭**：`CRRT/透析`干预或肌酐最高值 。


* **竞争风险对齐**：同步引入 `early_death_24_48h`（24-48h 内死亡），确保结局定义的统计逻辑与 MIMIC 开发集完全同质。



#### **产出**

* 生成外部验证集母表 `ap_external_validation`，其数据分布与物理量纲已与 MIMIC 开发集完成深度对齐。

#### **审计笔记**

* **创新亮点叙事**：在论文中可将其描述为：“To bridge the granular gap between databases, we developed a multi-tier recovery pipeline for critical features like pH, achieving a [X]% increase in data density through physiological formula-based derivation (Henderson-Hasselbalch).”
* **跨库一致性保障**：该模块通过对结局判定逻辑的“降级模拟”与对特征单位的“智能识别”，从 SQL 底层解决了 Domain Shift 问题，为后续模块的外部验证稳定性提供了数学保障。

## 第一阶段：数据治理与临床场景构建 (Data Engineering)


### **模块 01: 原始数据清洗与跨库单位校准 (Data Cleaning & Unit Harmonization)**

* **核心内容**：
* **结局标签的逻辑重构与早亡修正 (Outcome Logic Enforcement)**：
* **早亡偏倚修正**：代码通过 `early_death_24_48h` 标记，对 `pof`（持续性器官衰竭）进行回填。确保因病情极其危重而在 48 小时内死亡、无法完成持续性判定采样周期的患者，被正确计入阳性结局，避免生存偏倚（Survivor Bias）。
* **复合终点对齐**：通过 `(pof == 1) | (mortality_28d == 1)` 的并集运算构建 `composite_outcome`，实现生存分析与临床预后逻辑的严谨闭环。


* **缺失率过滤与核心白名单保护 (Feature Selection & Safeguarding)**：
* **剔除标准**：自动识别缺失率  的非核心变量进行剔除。
* **白名单（强制保留）**：针对乳酸 (`lactate`)、氧合指数 (`pao2fio2ratio`)、肌酐 (`creatinine`)、尿素氮 (`bun`)、转氨酶 (`ast/alt`) 及核心暴露因素 **BMI** 实施强制保留，即使其缺失率较高，也确保为后续 MICE 多重插补保留核心特征空间。


* **动态跨库单位审计与自动校准 (Database-aware Unit Alignment)**：
* **BUN (尿素氮)**：代码引入 `database` 感知逻辑，若检测到 eICU 库中中位数显著偏低（指示为 `mmol/L`），则自动应用 **2.801** 转换系数，将其统一至 MIMIC 标准的 `mg/dL`。
* **AST/ALT (转氨酶)**：通过中位数探测识别潜在的 Log 尺度，应用 `np.expm1()` 进行反向还原，确保生理指标回归到原始物理意义下的数值。
* **Fibrinogen (纤维蛋白原)**：自动识别 `g/L` 尺度，并校准为 `mg/dL` 量级（乘以 100），消除跨库量纲陷阱。


* **异常值鲁棒性处理 (Robustness Scaling)**：
* 执行 **1%-99% 盖帽处理 (Clipping)**，在保护生理真实性的前提下，抑制因传感器异常或极端录入（如 BMI 达到极端值）产生的离群值干扰。




* **产出**：生成标准物理尺度数据集 `mimic_for_table1.csv`，作为 **Table 1 (基线描述)** 的唯一官方数据源。
* **审计笔记**：
* **物理尺度对齐说明**：在论文 Methods 中可描述为：“To address laboratory heterogeneity across disparate EHR systems, we implemented a database-aware unit auditing mechanism, recalibrating scales (e.g., BUN, AST, Fibrinogen) based on population-level median distributions.”
* **早亡修正的重要性**：强调：“The inclusion of early-death cases (24-48h) into the primary outcome prevents underestimation of severity in patients who died before fulfilling the 48-hour persistent organ failure criteria.”
* **路径闭环**：该模块确保了 MIMIC 与 eICU 的物理量纲完全一致，Table 1 展示将采用此模块产出的原始尺度数据，以符合临床判读习惯。


### **模块 02: 临床场景定义与数据泄露防护 (Clinical Scenarios & Leakage Prevention)**

* **核心内容**：
* **严苛的数据泄露防护 (Data Leakage Safeguard)**：
* **临床评分剔除**：系统性识别并剔除了 SOFA, SAPS II, APS III, OASIS 等重症监护评分。由于这些评分包含未来 24h 的聚合生理信息，纳入预测模型会导致性能虚高。
* **干预措施剔除**：显式移除机械通气 (`vent_flag`)、升压药 (`vaso_flag`) 等后续治疗指标。确保模型仅依赖“基线生理状态”进行早期风险识别，而非识别“医生治疗行为”。
* **非生物特征清理**：移除 ID（Subject/HADM/Stay ID）、时间戳及冗余结局（LOS、死亡时间），消除非生理性噪声。


* **临床亚组定义与分层审计 (Subgroup Stratification)**：
* **“无预存肾损伤”亚组**：基于模块 01 物理校准后的肌酐值，定义 `Creatinine < 1.5 mg/dL` 且无慢性肾病史 (CKD) 的样本集。
* **数据库感知审计 (Database-aware Audit)**：内置跨库分布审查逻辑，自动对比 MIMIC 与 eICU 在该亚组上的比例（如样本量分布、占比差异），预防中心化偏倚。


* **自动化统计报告 (Automated Statistical Profiling)**：
* **出版级 TableOne 产出**：利用 `tableone` 库自动计算中位数 `[IQR]` 与频率分布。
* **Table 1 (POF vs. Non-POF)**：通过非正态分布假设检验 (Kruskal-Wallis)，揭示疾病早期核心驱动因子的显著性差异。
* **Table 2 (Subgroup Comparison)**：对比肾功能亚组基线，为后续敏感性分析提供统计基石。


* **工程优化与物理尺度保持**：
* **Raw Scale 策略**：在统计阶段坚持不做数学转换（如 Log），确保报表数值符合临床直觉（如肌酐展现为 `1.2` 而非归一化后的数值）。
* **显式内存释放**：引入 `gc.collect()` 机制，在生成大型统计表后主动清理内存，确保 Python 管道在处理多中心大数据时的稳定性。




* **产出**：
* 生成 `mimic_for_model.csv`：机器学习管道的最终冷启动输入。
* 生成 `table_1_pof_comparison.csv` 与 `table_2_renal_subgroup.csv`：直接用于论文撰写的统计母表。


* **审计笔记**：
* **学术严谨性**：在 Methods 中强调：“To ensure the model captures intrinsic biological risk rather than therapeutic intervention, we excluded all post-admission treatments and integrated clinical severity scores.”
* **亚组逻辑**：为 Results 章节中的 Subgroup Analysis 埋下伏笔，证明模型在排除慢性干扰后的泛化力。
* **闭环验证**：通过 `assert` 断言机制强制审计终点指标（Target Labels），确保特征精炼过程中的数据完整性。



## 第二阶段：算法开发与极致特征精炼 (Model Development & Feature Refinement)

### **模块 03: 混合算法竞赛与多维评估 (Hybrid Model Training & Multi-dimensional Evaluation)**

* **核心内容**：
* **动态对数转换与物理分布审计 (Skewness Correction)**：
* 针对肌酐、尿素氮、转氨酶等 16 项强偏态指标执行 `Log1p` 转换。
* **双重审计**：代码在转换前后自动打印 Train/Test 组的中位数（Median），确保跨样本集的物理分布一致性，为线性模型（LR/SVM）提供高质量的数学收敛条件。


* **增强型多重插补 (MICE) 与质量预警**：
* 采用 `IterativeImputer` 链式方程补全缺失值。
* **风险监控**：内置缺失率审计机制，对缺失率  的变量（如 Lipase, Bilirubin 等）自动发出“插补噪声”预警，提升数据治理的透明度。


* **极致特征降维：LASSO 1-SE 准则 (Feature Compression)**：
* **学术增强**：超越常规最小 MSE 准则，采用 **1-SE Rule（标准误准则）** 进行特征筛选。该逻辑在性能损失极小的范围内选择最精简的特征子集（Top 12），显著增强了模型的临床解释性与泛化能力。
* **自动化制图**：代码可自动生成出版级 `Academic_Lasso_{target}.png`，展示系数路径与误差条。


* **混合算法竞赛与多指标对齐 (Model Ensemble & Calibration)**：
* **五大算法横跳**：同步训练 XGBoost、Random Forest、SVM、LR 及决策树。
* **贝叶斯寻优**：通过 `Optuna` 对 XGBoost 进行 100 轮超参迭代，最大化 AUC。
* **概率校准**：强制引入 `Isotonic` 校准器，确保预测概率与实际发生率线性对齐，优化临床决策关键指标——**Brier Score**。


* **统计学强度支撑：Bootstrap CI 审计**：
* 内置 1000 次 Bootstrap 抽样函数，为每个模型计算全人群及“无预存肾损伤亚组”的 **AUC 95% 置信区间**。




* **产出**：
* **独立资产包**：为每个研究终点（POF/Mortality/Composite）生成专属的 `all_models_{target}.pkl`、`selected_features_{target}.pkl` 及预处理资产。
* **性能全景图**：自动化生成包含训练集/验证集的 **ROC 曲线** 与 **校准曲线（Calibration Curve）** 矩阵。
* **性能汇总表**：生成 `all_outcomes_performance_summary.csv`，作为论文结果部分的数据源头。


* **审计笔记**：
* **叙事加分项**：在论文 Methods 中强调：“We utilized the LASSO 1-Standard Error (1-SE) rule to maintain model parsimony, ensuring that each outcome-specific model was built on the most predictive yet concise subset of clinical features.”
* **多终点闭环**：该模块通过 Target 循环命名机制，解决了不同结局驱动因子（如年龄对死亡率的影响 vs. 肌酐对 POF 的影响）的异质性难题。
* **亚组泛化**：Bootstrap 计算的亚组 AUC 为敏感性分析提供了严谨的统计学证据。



## 第三阶段：模型评价、可解释性与决策分析 (Evaluation & Interpretability)


### **模块 04: 模型评价、可解释性与决策分析 (Evaluation & Interpretability)**

#### **核心内容**

* **多结局鲁棒性审计 (Multi-target Performance Audit)**：
* **区分度横向对比**：系统性加载模块 03 产出的 `all_models_{target}.pkl`。通过 ROC 曲线簇同步评估模型在 `POF`、`Composite Outcome` 与 `Mortality` 三大终点下的表现。
* **亚组敏感性分析**：代码内置 `No-Renal AUC` 审计逻辑，对比全人群与“无预存肾损伤亚组”的 95% CI。若亚组表现稳定，则有力证明了模型捕捉的是 AP 的急性生理演变，而非受慢性病史驱动。


* **黑盒模型透明化 (SHAP Explainable AI)**：
* **SVM 核心审计**：代码选择非线性表达力最强的 **SVM（校准后版本）** 作为归因对象。通过 SHAP 蒙特卡洛采样，将高维非线性关系转化为直观的特征贡献度。
* **动态缓存机制**：针对 SHAP 计算耗时长的特性，引入了 `svm_shap_values_{target}.pkl` 序列化缓存。当模型或特征更新时，系统可自主决定重算或加载，极大地优化了工程效率。
* **特征交互视图**：生成 `Beeswarm Plot`，揭示核心预测因子（如乳酸、BUN、PaO2/FiO2）对风险概率的正负向影响强度。


* **临床决策价值量化 (Decision Curve Analysis, DCA)**：
* **净获益评价**：超越单纯的准确率指标，利用自定义 `calculate_net_benefit` 函数绘制 DCA 曲线。
* **获益区间界定 (Benefit Window)**：自动化审计模型优于“全干预（Treat All）”或“不干预（Treat None）”的概率截断点。例如，自动输出“获益窗口: 5.0% - 75.0%”，为临床决策路径的构建提供定量支持。


* **出版级图表矩阵 (Automated Visualization)**：
* **Figure 1 (ROC)**：展示区分度与 95% CI。
* **Figure 2 (SHAP)**：展示生物学驱动因子。
* **Figure 3 (DCA)**：展示临床实际应用价值。
* **Table 2 (Summary)**：自动化生成 `Table2_Model_Performance_Summary.csv`，支持 `utf-8-sig` 编码以确保 Excel 兼容。



#### **🎨 临床价值叙事 (Results Section Drafting)**

> **区分度表现**：在针对持续性器官衰竭（POF）的预测中，校准后的 SVM 展现了卓越的区分能力，其验证集 AUC 为 **[X.XX, 95% CI: X.XX-X.XX]**。即便在排除慢性肾病干扰的亚组中，该模型依然保持了稳健的预测效能。
> **驱动因子解析**：SHAP 摘要图（Figure 2）进一步具象化了临床逻辑：**乳酸（Lactate）**的显著升高与**氧合指数（PaO2/FiO2）**的下降是器官受累的最强先兆。
> **决策支持区间**：DCA 获益区间审计（Figure 3）证实，对于大部分中高风险患者（阈值概率 10%-70%），基于本模型引导的早期预警策略能显著提升净获益。

#### **🛠 技术细节修复与审计笔记**

* **校准包装器桥接**：代码通过自定义 `svm_predict` 接口封装了 `predict_proba`，成功解决了 `CalibratedClassifierCV` 失去 `coef_` 属性后无法直接进行 SHAP 归因的行业痛点。
* **数据一致性保障**：通过 `joblib` 循环加载特定结局的测试集 `test_data_main_{target}.pkl`，确保了“模型-特征-数据”的三位一体对齐。
* **工程健壮性**：代码内置了完善的 `try-except` 异常捕获与 `numpy` 强制类型转换，有效规避了版本兼容性及 String-to-Float 的常见工程错误。

---

**[模块状态]**：模块 04 已完成对 MIMIC 数据的深度挖掘与成果可视化。



### 模块 05: 基线特征描述与单因素分析 (Baseline Characteristics & Univariate Analysis)
- **核心内容**：
  - 统计学自动识别 (Statistical Logic)：Shapiro-Wilk 正态性检验，自动切换 Mean ± SD (t-test) 或 Median [IQR] (Mann-Whitney U test)。
  - 分类变量审计：卡方检验，计算频数与构成比。
  - 特征维度对齐：重点分析 Top 12 核心特征在 POF 组与 Non-POF 组间的差异。
  - 自动化表格产出：生成 `Table1_Baseline_Characteristics.csv`。
- **审计笔记**：
  - 统计严谨性：在 Methods 中描述正态性检验。
  - 数据闭环：引用 `selected_features.pkl`，连接建模与临床描述。
  - 结果解读：P 值 < 0.001 增强模型可信度。

### 模块 06: 最佳截断值计算与效能审计 (Optimal Cut-off & Performance Metrics)
- **核心内容**：
  - 约登指数寻优 (Youden’s Index)：确立最佳概率截断值。
  - 多维度效能评估：计算 Sensitivity、Specificity、PPV、NPV、F1 分数。
  - 诊断坐标可视化：绘制带 Cut-off 标记的 ROC 曲线。
- **产出**：生成 `diagnostic_performance_svm.csv` 和 `05_ROC_with_Cutoff.png`。
- **审计笔记**：
  - 临床叙事：在 Results 中描述高 NPV 的 rule-out 价值。
  - NPV 重要性：有助于减少不必要医疗资源占用。
  - 图表闭环：作为论文插图。

## 第四阶段：外部验证与跨库审计 (External Validation & Cross-cohort Auditing)

### 模块 07: 外部数据库对齐与特征审计 (External Data Mapping & Auditing)
- **核心内容**：
  - 多中心列名映射：重构 eICU 命名对齐 MIMIC。
  - 特征缺失审计：对比 Top 12 特征在 eICU 中的缺失率。
  - 跨库尺度对齐：执行 Log1p 转换与 Clipping。
  - 验证矩阵构建：中位数填补处理碎片化数据。
- **产出**：生成 `eicu_for_model.csv`。
- **审计笔记**：
  - 论文亮点：在 Methods 中描述特征审计。
  - 单位校准成功：验证 Log 转换后中位数相似。
  - 缺失值透明度：体现真实世界适应力。

### 模块 08: 强制对齐外部验证与跨库评估 (Forced Alignment & External Validation)
- **核心内容**：
  - 特征空间重构：重建 eICU 维度一致性。
  - 严格预处理复现：复现 Log1p、MICE 和 Scaler。
  - 跨中心性能对标：展示 5 种算法在 MIMIC 与 eICU 的 AUC/Brier。
- **产出**：生成 `external_validation_debug.png`。
- **审计笔记**：
  - 论文核心论点：强调泛化性能。
  - 技术亮点：使用 `feature_names_in_` 确保可重复性。
  - Brier 分数：证明概率精准。

### 模块 09: 跨库基线可比性分析 (Cross-cohort Baseline Comparison)
- **核心内容**：
  - 数据源标签化：合并 MIMIC 与 eICU。
  - 统计描述对齐：统一标签，进行描述性统计。
  - 高级描述性统计：使用 TableOne 区分分布。
  - 泛化偏倚审计：计算 SMD & P-value（SMD < 0.1 为均衡证据）。
- **产出**：生成 `Table1_MIMIC_vs_eICU.csv`。
- **审计笔记**：
  - 科学性加持：回应审稿人关于人群差异的疑问。
  - 偏态分布严谨性：指定 ICU 指标为 nonnormal。
  - 统计工具：TableOne 符合期刊标准。

## 第五阶段：临床价值转化与稳健性评估 (Clinical Utility & Reliability)

### 模块 10: 临床决策曲线分析与外部应用评估 (Decision Curve Analysis, DCA)
- **核心内容**：
  - 资产兼容性审计：修复 `sklearn` 环境兼容。
  - 外部数据强制投影：投影 eICU 数据。
  - 净获益计算引擎：遍历 0% 到 100% 阈值。
  - 临床实用性对标：对比 5 种算法与极端策略。
- **产出**：生成 `dca_final_eicu.png`。
- **审计笔记**：
  - 临床叙事：在 Discussion 中强调净获益优势。
  - 鲁棒性细节：手动实现函数，避免数学报错。
  - 多模型对比：支持最终算法选择。

### 模块 11: 核心特征共线性与聚类审计 (Multicollinearity & Clustering Audit)
- **核心内容**：
  - 多重共线性检测 (VIF Analysis)：VIF < 5 支撑稳定性。
  - 层级聚类热图：识别临床指标簇。
  - 讨论素材自动化：生成临床解释建议。
- **产出**：生成 `feature_collinearity_clustermap.png`。
- **审计笔记**：
  - 证据支撑：特征无严重共线性，模型稳健。
  - 作为补充材料：提升论文深度。

### 模块 12: 概率校准与诺莫图逻辑 (Calibration & Nomogram Interpretation)
- **核心内容**：
  - 校准审计 (Calibration Curve)：评估预测概率与实际一致性。
  - 比值比分析 (Odds Ratio)：提取 OR 值，转化黑盒权重。
  - 床旁工具转化 (Nomogram Foundation)：导出权重，为诺莫图提供基础。
- **审计笔记**：
  - 临床重要性：确保模型不高估/低估风险。
  - 易理解性：OR 值便于医生解读。

## 🏁 研究全流程（终极闭环版）
1. **数据仓库挖掘** (SQL 队列提取、结局定义、跨库对齐)。
2. **数据清洗** (MIMIC/eICU 单位统一)。
3. **特征工程** (Log 转换、MICE 插补)。
4. **算法竞赛** (5 种模型、贝叶斯优化、概率校准)。
5. **内验证** (ROC 曲线、SHAP 归因)。
6. **诊断效能** (约登指数、Cut-off 确定)。
7. **统计描述** (Table 1 基线、单因素分析)。
8. **外验证** (eICU 对齐、泛化性能评估)。
9. **人群对比** (SMD 审计)。
10. **临床价值** (DCA 净获益分析)。
11. **稳健性审计** (VIF 共线性检测)。
12. **可靠性评估** (校准曲线与诺莫图)。
