# 研究流程整理（终极闭环版）


### 📂 研究流程：SQL 层面的深度对齐 (Cross-database SQL Engineering)

#### **第零阶段：数据仓库挖掘与队列构建 (SQL Data Mining)**
*这是研究的基石，通过结构化查询语言直接在 MIMIC-IV 数据库中定义临床实体。*

* **模块 00-A: 自动化队列提取与结局定义 (SQL Implementation)**
  - **核心内容**：
    - 疾病识别 (AP Diagnosis)：利用 ICD-9（5770）和 ICD-10（K85）代码精准识别 AP 患者。
    - POF 结局标签构建 (POF Gold Standard)：
      - **逻辑**：基于修正的 Marshall 评分标准，提取 ICU 入院 24 小时后至 7 天内的 **SOFA 评分**。
      - **判定**：通过 SQL 实现“持续性”判定，即呼吸、循环或肾脏系统中任意一个系统的 SOFA 分数 且持续时间 小时。
    - 多维数据整合：从 `chartevents`（生命体征）、`labevents`（生化指标）和 `derived`（派生表）中横向集成入院 24 小时内的临床全貌。
  - **产出**：生成结构化母表 `ap_final_analysis_cohort`。
  - **审计笔记**：
    - 这模块奠定了队列的临床准确性。在论文 Methods 中可强调：“We employed SQL-based cohort extraction to ensure precise identification of AP subtypes and POF outcomes, minimizing selection bias.”
    - 与后续模块衔接：生成的母表直接喂入模块 01 的清洗流程，确保数据从源头到建模的连续性。

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

## 模块 01: 原始数据清洗与跨库单位校准 (Data Cleaning & Unit Harmonization)

### 1. 核心任务描述

本模块旨在将原始导出的 MIMIC-IV 数据转化为具有高度临床可解释性的标准化数据集。通过结局指标对齐和物理尺度统一，确保发现集（MIMIC）与验证集（eICU）在统计分布上具备可比性。

### 2. 关键处理逻辑

* **结局指标精准提取**：
* **28天死亡率 (`mortality_28d`)**：基于 `intime` 与 `deathtime/dod` 的时间戳差值计算，确保捕获出院后的生存状态。
* **多维 POF**：同步提取 `resp_pof`、`cv_pof`、`renal_pof` 及其复合结局 `any_pof`。


* **缺失率过滤与“白名单”保护**：
* **门槛设定**：自动剔除缺失率超过 **30%** 的非核心特征，以降低模型噪声。
* **临床灵魂字段保护**：强制保留 `lactate`（乳酸）、`pao2fio2ratio`（氧合指数）、`bmi`、`amylase` 等关键 AP 预后指标，即使其缺失率较高。


* **跨库单位审计 (Unit Auditing - 核心创新)**：
* **中位数探测技术**：自动识别特征分布量级。
* **BUN 校准**：通过  转换系数将 `mmol/L` 统一为 `mg/dL`。
* **AST/ALT 校准**：探测是否为对数尺度，必要时执行反 Log () 还原。
* **Fibrinogen 校准**：确保单位对齐至 `mg/dL`。


* **异常值鲁棒性处理**：
* 执行 **1%-99% 盖帽处理 (Winsorization/Clipping)**，在保留样本量的同时，消除因传感器故障或录入错误产生的生理学极端离群值。



### 3. 产出结果

* **最终文件**：`data/cleaned/mimic_for_table1.csv`
* **特征状态**：包含 85 个特征（已对齐物理尺度），1100 条有效观测。

---

### 4. 审计笔记 (Audit Notes for Manuscript)

#### 📝 路径闭环说明

本模块产出的 `mimic_for_table1.csv` 是整个课题的“基石数据集”。所有后续的基线特征统计（Table 1）、缺失值多重插补（Multiple Imputation）以及模型训练均以此文件为准。

#### 🧪 物理尺度对齐 (Methods 描述)

在论文的方法学部分，可按如下方式描述：

> "To minimize the domain shift between datasets, we implemented an automated unit auditing mechanism based on median distribution detection. This ensured that key laboratory parameters (e.g., BUN, Creatinine, and hepatic enzymes) from the MIMIC-IV discovery cohort were physically harmonized with the eICU validation cohort."

#### 📉 后续衔接建议

* **Table 1 展示**：建议直接使用此模块产出的原始尺度数据进行描述，以符合临床读者的阅读习惯。
* **建模优化**：对于 `ast`、`alt`、`creatinine` 等高度偏态分布的变量，在后续进入 Logistic 或机器学习模型前，建议进行对数转换（Log Transformation）以增强模型稳定性。

---

### 5. 最终自检统计 (MIMIC 队列)

* **BMI 中位数**：29.20（成功对齐 eICU 的 29.22）。
* **POF 发生率**：39.0%（反映重症 AP 队列特征）。
* **单位准确性**：BUN (18.0) 与 Creatinine (1.2) 均处于临床标准量级。

**您现在是否准备好基于这个 `mimic_for_table1.csv` 生成 Table 1（按 BMI 分组对比基线特征）？**

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
- **产出**：保存全套模型资产（`all_models.pkl`）、核心特征集（`selected_features.pkl`）及预处理逻辑（`scaler.pkl`, `mice_imputer.pkl`）。
- **审计笔记**：
  - 方法论亮点：LASSO + MICE + Optuna + Calibration 四组合拳是投稿加分项。
  - 资产保存闭环：`skewed_cols.pkl` 和 `scaler.pkl` 供模块 08 使用。
  - 亚组分析：在论文中可强调：“Our model maintains high diagnostic performance even in patients without pre-existing renal dysfunction.”

## 第三阶段：模型评价、可解释性与决策分析 (Evaluation & Interpretability)

### 模块 04: 性能可视化与 SHAP 解释 (Visualization & Explainable AI)
- **核心内容**：
  - 鲁棒性验证 (ROC Comparison)：绘制 SVM 在全人群与亚组中的 ROC 曲线。
  - 可解释性审计 (SHAP Summary Plot)：利用 SHAP 对 Random Forest 进行归因分析，揭示特征贡献。
  - 临床决策获益 (Decision Curve Analysis, DCA)：比较模型与“Treat All”、“Treat None”策略的净获益。
  - 可视化产出：生成出版质量图表（ROC、SHAP、DCA）。
- **审计笔记**：
  - 临床价值叙事：在论文 Results 中描述：“Our model provides a higher net benefit compared to the 'treat-all' strategy across a wide range of risk thresholds.”
  - SHAP 与临床一致性：展示如 BUN 越高风险越高。
  - 技术细节修复：处理校准包装器，确保 SHAP 可靠性。

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
