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



#### **模块 00-B: eICU 多中心数据特征审计与对齐 (SQL Level)**
  - **核心内容**：
    - 精细化单位清洗 (Unit Auditing)：
      - **肌酐 (Creatinine)**：通过 `>30` 的逻辑判定，自动识别并校准 `umol/L` 与 `mg/dL`，抹平了多中心数据最常见的量纲陷阱。
      - **温度 (Temperature)**：内置华氏度与摄氏度的自动识别转换逻辑，确保生理指标的物理一致性。
    - pH 值多维度打捞 (pH Recovery Logic)：
      - 这是本研究的重大亮点。pH 是 AP-POF 预测的核心，但在 eICU 中缺失率极高。
      - 你的 SQL 实现了 **“直接提取 (Lab) -> 血气打捞 (BG) -> 公式计算 (Henderson-Hasselbalch) -> APACHE 兜底”** 的四级打捞机制，极大提升了模型在外部验证集的完整度。
    - POF 结局的“跨库模拟” (Outcome Emulation)：
      - eICU 不像 MIMIC 那样有现成的持续器官衰竭评分。你通过 `CarePlan`（护理计划）、`Treatment`（治疗）结合 `ICU_LOS`（住院时间）巧妙地模拟了 **“持续性（Persistence）”**：
      - **呼吸衰竭**：机械通气且 ICU 时长 （排除术后常规插管）。
      - **循环衰竭**：使用升压药且 ICU 时长 。
  - **产出**：生成对齐后的 eICU 数据集，供后续外部验证使用。
  - **审计笔记**：
    - 方法论亮点：pH 打捞机制可作为论文的创新点描述：“To address high missingness in eICU, we implemented a multi-tier pH recovery algorithm, increasing data completeness by [X]%.”
    - 跨库一致性：这模块直接支撑模块 07/08 的外部验证，确保 eICU 数据在 SQL 层面已与 MIMIC 对齐，减少分布偏移。

## 第一阶段：数据治理与临床场景构建 (Data Engineering)

根据您的要求，我已更新了 **模块 01** 的详细说明。这份说明不仅涵盖了代码逻辑，还从学术论文的角度完善了审计笔记，特别强调了解决 **跨库数据偏移（Domain Shift）** 的核心创新点。

---

---

#### **模块 01: 原始数据清洗与跨库单位校准 (Data Cleaning & Unit Harmonization)**

* **核心内容**：
* **结局指标精准提取 (Outcome Refinement)**：
* 基于时间戳锁定 **28天死亡率 (`mortality_28d`)**，并同步提取多维度器官衰竭指标（POF）。
* 整合 **复合结局 (`composite_outcome`)**，确保生存分析与临床预后逻辑闭环。


* **缺失率过滤与白名单保护 (Feature Selection)**：
* **剔除标准**：自动识别缺失率  的变量进行剔除。
* **白名单（强制保留）**：针对乳酸 (`lactate`)、氧合指数 (`pao2fio2ratio`)、肌酐 (`creatinine`)、尿素氮 (`bun`) 及核心暴露因素 **BMI** 实施强制保留。
* **明确排除**：**淀粉酶 (`amylase`) 不再作为保留特征**，以优化特征空间并减少时间窗偏移产生的噪声。


* **跨库单位审计与自动校准 (Unit Auditing & Alignment)**：
* **BUN**：检测量级并应用 **2.801** 转换系数，将 `mmol/L` 统一为 `mg/dL`。
* **AST/ALT**：利用中位数探测技术自动识别潜在的 Log 尺度，并执行反 Log（`expm1`）还原。
* **Fibrinogen**：自动校准为 `mg/dL` 量级，确保与 eICU 验证集处于同一物理尺度。


* **异常值鲁棒性处理 (Robustness Scaling)**：
* 执行 **1%-99% 盖帽处理 (Clipping)**，在保护生理真实性的前提下，抑制因传感器异常或极端录入产生的离群值干扰。




* **产出**：生成标准物理尺度数据集 `mimic_for_table1.csv`，供后续基线描述和特征工程使用。
* **审计笔记**：
* **物理尺度对齐说明**：在论文 Methods 中可描述为：“To minimize the domain shift between datasets, we implemented an automated unit auditing mechanism based on median distribution detection, ensuring physical alignment of laboratory scales.”
* **路径闭环**：该模块确保了 MIMIC 与 eICU 的物理量纲完全一致。Table 1 展示将采用此模块产出的原始尺度数据，以符合临床判读习惯。



### 模块 02: 临床场景定义与数据泄露防护 (Clinical Scenarios & Leakage Prevention)
- **核心内容**：
  - 数据泄露防护（核心审计）：系统性剔除了 SOFA、SAPS II、APS III 等临床评分系统，以及呼吸机、血管活性药物等后续治疗指标，确保模型仅基于入院 24 小时内的“基线状态”进行预测。
  - 预测因子精炼：移除非生物学特征（ID、时间戳）及冗余结局指标（LOS、死亡时间）。
  - 亚组定义（临床敏感性分析）：基于模块 01 修正后的原始肌酐值，定义了“无预存肾损伤”亚组（Creatinine < 1.5 mg/dL 且无 CKD 史）。
  - 尺度一致性保持：保持原始物理量级，将数学转换（如 Log）后置。
- **产出**：生成 `mimic_for_model.csv`，作为机器学习管道的最终输入。
- **审计笔记**：
  - 数据安全性：在论文 Methods 中强调：“To ensure clinical applicability and avoid data leakage, we excluded dynamic physiological scores and post-admission treatments.”
  - 亚组逻辑：暗示论文中可有 Subgroup Analysis 章节。
  - 变量映射：文件名与后续模块（如 09/10）路径吻合，闭环成功。

## 第二阶段：算法开发与极致特征精炼 (Model Development & Feature Refinement)

### 模块 03: 混合算法竞赛与多维评估 (Hybrid Model Training & Multi-dimensional Evaluation)
- **核心内容**：
  - 动态对数处理 (Skewness Correction)：针对偏态分布指标（如肌酐、淀粉酶、转氨酶）执行 `Log1p` 转换。
  - 先进数据插补 (MICE)：采用多重插补技术，利用变量间链式关系填补缺失值。
  - 特征降维 (LASSO Compression)：应用 LASSO 回归压缩至 Top 12 核心预测因子。
  - 贝叶斯超参优化 (Optuna)：对 XGBoost 进行贝叶斯寻优。
  - 概率校准 (Probability Calibration)：引入 `CalibratedClassifierCV`（Isotonic），优化 Brier Score。
  - 多亚组性能评估：验证在“无预存肾损伤亚组”中的表现。
- **产出**：引入 target 循环命名机制。逻辑更新： 由于 POF、死亡率、复合终点的核心驱动因子不同（如：死亡率受年龄影响大，而 POF 受乳酸和肌酐影响大），必须为每个结局生成独立的特征集和模型包。产出对齐： 保存为 all_models_{target}.pkl 和 selected_features_{target}.pkl。
- **审计笔记**：
  - 方法论亮点：LASSO + MICE + Optuna + Calibration 四组合拳是投稿加分项。
  - 资产保存闭环：`skewed_cols.pkl` 和 `scaler.pkl` 供模块 08 使用。
  - 亚组分析：在论文中可强调：“We developed target-specific predictive pipelines to capture the unique pathophysiological drivers of different clinical outcomes.”

## 第三阶段：模型评价、可解释性与决策分析 (Evaluation & Interpretability)

### 模块 04: 性能可视化与 SHAP 解释 (Visualization & Explainable AI)

#### 核心内容

* **多结局鲁棒性审计 (Multi-target Robustness Validation)**：系统性评估模型在 `POF`、`Composite Outcome` 与 `Mortality` 三大终点下的表现。通过 `Main AUC` 与 `No-Renal Sub-AUC` 的双重对比，证明模型在排除肾功能干扰后依然具备强悍的泛化能力。
* **黑盒模型透明化 (SHAP Explainable AI)**：以 **SVM（校准后版本）** 为核心审计对象。针对每个研究终点，利用 SHAP 蒙特卡洛采样将抽象的非线性向量转化为可直观理解的特征贡献度，解析不同并发症背景下的核心驱动因子。
* **临床应用价值量化 (Decision Curve Analysis, DCA)**：超越单纯的数学指标，从临床获益角度出发，绘制 Net Benefit 曲线。通过动态流行率审计，确立模型在真实临床决策场景中的“优势区间（Benefit Window）”。
* **出版级图表矩阵 (High-Resolution Graphic Matrix)**：自动化产出 300 DPI 高清图表。
* **Figure 1 (ROC)**：展示区分度与 95% CI。
* **Figure 2 (SHAP)**：展示特征对风险的正负向影响。
* **Figure 3 (DCA)**：展示临床净获益。

---

#### 🎨 临床价值叙事 (Results Section Drafting)

基于模块 04 产出的 `Table2_Model_Performance_Summary.csv`，建议论文 Results 段落写作逻辑如下：

> **性能总结 (Performance Summary)**：在针对 POF 及 28 天死亡率的预测中，机器学习模型展现了高度的区分度。以 **XGBoost** 为例，其在复合终点（Composite Outcome）中的测试集 AUC 达到了 **0.887 [95% CI: 0.839-0.932]**。值得注意的是，在非肾源性（No-Renal）亚组验证中，模型依然保持了极高的稳定性。
> **临床决策获益 (Clinical Utility)**：决策曲线分析（DCA）进一步证实了模型的实用价值。对于 POF 预测，在 **3.0% 至 78.0%** 的风险阈值范围内，基于本研究模型的干预策略相比“全干预（Treat All）”展现了显著更高的净获益。
> **特征驱动因子 (Feature Attribution)**：SHAP 摘要图（Figure 2）揭示了模型判定的逻辑：**乳酸（Lactate）**、**血尿素氮（BUN）**及**氧合指数（PaO2/FiO2）**是预测器官衰竭的关键变量。高水平的 BUN 与风险评分呈显著正相关，这与急性胰腺炎累及肾脏代谢的临床病理生理机制高度吻合。

---

#### 🛠 技术细节修复与审计笔记

* **校准包装器兼容性 (Calibration Bridge)**：针对模块 03 采用的 `CalibratedClassifierCV`（Isotonic/Sigmoid 校准），模块 04 采用自定义 `predict_proba` 映射函数，成功解决了校准模型失去 `coef_` 属性后无法进行 SHAP 归因的行业难题。
* ** Table 2 自动化流水线**：代码实现了从 `.pkl` 模型包到 `Table2_Model_Performance_Summary.csv` 的全自动转化，支持自动填入 95% 置信区间与 DCA 获益窗口，确保了从代码到论文数据的数据一致性（Data Integrity）。
* **计算性能优化 (Computational Efficiency)**：针对 SVM SHAP 计算耗时较长的特性，引入了基于 `Target` 命名的序列化缓存机制（Cache Logic）。当 `all_models_{target}.pkl` 更新时，系统可自主选择重算或加载缓存。

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
