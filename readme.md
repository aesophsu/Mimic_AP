### 论文复现部分：论文核心概述（详细版）

#### 目标
开发并验证一个**机器学习预测模型**，用于**酒精性肝硬化（Alcoholic Cirrhosis, AC）患者在 ICU 住院期间的 28 天全因死亡率**。  
模型需在 **MIMIC-IV v3.1** 数据库中完成训练与内部验证，**直接与传统 MELD 评分对比**，证明 ML 模型在预测性能（AUC）上显著优于 MELD（文献报告 MELD AUC ≈ 0.77）。  
同时，**必须使用 SHAP 值进行模型解释**，量化每个特征对个体预测的边际贡献（正/负），为临床提供可解释的决策依据。

#### 关键结果（必须完全复现）
1. **研究队列**  
   - 最终纳入 **2134 名 AC 患者**（用户已有 2136 名，极度接近）。  
   - 基线特征：  
     - 男性占比 **69.5%**  
     - 平均年龄 **56.2 岁**  
     - 28 天死亡率 **29.2%**（624/2134）  

2. **特征选择**  
   - 使用 **LASSO 回归（glmnet 等效）** 从 40+ 个候选变量中筛选出核心变量：  
     ```
     Age, SOFA, APSIII, OASIS, LODS, 
     Temperature (mean), Chloride (min), Lactate (min), 
     Total Bilirubin (Tbil, min), INR (min), APTT (min), 
     Stroke, Malignant tumor, Congenital coagulation defects (CCD)
     ```
   - 这些变量来自首 24 小时的动态数据（vitals + labs）与共病史。

3. **模型性能（AUC 为首要指标）**  
   - **训练集**（70%）：Random Forest AUC **0.908**  
   - **验证集**（30%）：  
     ```
     SVM      0.866（最高）
     XGBoost  0.842
     RF       0.843
     DT       0.739
     LR       ~0.81
     ```
   - **MELD 原版 AUC 0.77**（需作为基准线）。

4. **SHAP 解释**  
   - **正向风险贡献（值越高，死亡风险越大）**：  
     **APSIII > Total Bilirubin > Age > LODS > APTT > SOFA**  
   - **负向保护作用（值越高，风险越低）**：  
     **Temperature, Chloride**  
   - Beeswarm 图需显示：高 APSIII（红点）集中于 SHAP > 0；低 Chloride（蓝点）集中于 SHAP < 0。

5. **其他评估指标（必须出图）**  
   - **ROC 曲线**（验证集，所有模型 + MELD）  
   - **Calibration 曲线**（10 bins，理想对角线）  
   - **Decision Curve Analysis (DCA)**：在阈值概率 0.05–0.50 区间，ML 模型净益（Net Benefit）显著高于 “全治疗” 和 “不治疗” 策略，且高于 MELD。

6. **局限性（需在讨论中提及）**  
   - 单中心（MIMIC-IV），外部验证不足 → 后续需在 **eICU** 数据库验证。  
   - 仅 28 天结局，未评估长期预后。

#### 临床意义（复现后可直接用于投稿/临床）
- 提供**早期、精准、可解释**的 28 天死亡风险分层工具。  
- **高 APSIII + 高 Tbil** 患者应优先考虑肝移植评估或强化监护。  
- **ML 优于 MELD** 的核心在于捕捉**非线性交互**与**动态生理指标**（如 APSIII、LODS），而 MELD 仅依赖 Cr、INR、Tbil 三个实验室指标。  
- SHAP 可生成**个体化解释报告**（e.g., “该患者死亡风险 68%，其中 APSIII 贡献 +0.32，Tbil 贡献 +0.28”），便于临床沟通。

**复现成功标准**：  
1. 队列死亡率
2. LASSO 筛选出 核心变量  
3. 验证集 
4. SHAP beeswarm 图 
5. 四图（ROC + Calibration + SHAP + DCA） 
6. ml_results.png 保存并可直接插入论文

