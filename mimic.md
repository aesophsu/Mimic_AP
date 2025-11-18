### MIMIC-IV 数据库详细结构总结

MIMIC-IV 分为 4 个核心 schema（模式），每个 schema 负责不同层面数据。你的酒精性肝硬化 28 天死亡率复现，主要依赖 **hosp + ic Early Warning Score (EWS) + icu + derived** 四个模块。

| Schema | 作用 | 表格数量 | 数据体积占比 | 你的复现中是否必须 |
|--------|------|----------|--------------|--------------------|
| **mimiciv_hosp** | 整个住院全流程（急诊→普通病房→ICU→出院） | 18 张 | ~35% | 必须（诊断、人口学、共病） |
| **mimiciv_icu** | ICU 内原始高频时间序列（每分钟级） | 9 张 | ~50% | 可选（chartevents 太大，可用 derived 替代） |
| **mimiciv_derived** | 官方预处理表（首24小时汇总 + 所有评分） | 60+ 张 | ~15% | **核心中的核心**（你的 SQL 90% JOIN 这个） |
| **mimiciv_ed** | 急诊室数据 | 10 张 | <5% | 可选 |

#### 1. mimiciv_hosp（住院全景，必须用）
| 表格 | 内容 | 你的复现用途 |
|------|------|--------------|
| patients | 唯一患者主表（subject_id + anchor_age + gender） | 年龄、性别 |
| admissions | 每次住院记录（hadm_id + admittime + deathtime） | 28 天死亡计算 |
| diagnoses_icd | ICD-9/10 诊断 | 筛选酒精性肝硬化（5712 / K7030,K7031） |
| transfers | 病房转科记录 | 确认 ICU 入院时间 |
| labevents | 所有化验原始记录 | derived 已汇总，可不直接用 |
| prescriptions / emar | 药物处方 | 可选（抗生素等） |

#### 2. mimiciv_icu（原始高频数据，可绕过）
| 表格 | 内容 | 建议 |
|------|------|------|
| icustays | 每次 ICU 住院（stay_id + intime/outtime） | 必须（核心枢纽表） |
| chartevents | 最大表（>4亿行），每分钟生命体征 | 太大！用 derived.first_day_vitalsign 替代 |
| inputevents / outputevents | 液体出入量 | 用 derived 汇总表 |
| d_items | itemid 字典 | 查代码含义 |

#### 3. mimiciv_derived（你的 SQL 90% 用这个，官方金矿）
这是 MIMIC-IV 最伟大设计！所有你需要的“首24小时最差值/平均值 + 标准评分”都在这里，已经帮你算好。

| 子类 | 关键表格（你的 SQL 必用） | 内容 |
|------|---------------------------|------|
| 评分系统 | sofa, apsiii, oasis, lods, sapsii, meld | 你的 14 变量里占 5 个（SOFA/APSIII/OASIS/LODS/MELD） |
| 首24小时汇总 | first_day_lab, first_day_vitalsign, first_day_bg | 所有 min/max/mean（Tbil_min, Lactate_min, Temp_mean, Chloride_min 等） |
| 干预 | ventilation, vasoactive_agent, rrt | 机械通气、升压药标志 |
| 合并症 | charlson | Charlson 指数（但你用 ICD 自己算） |
| 其他金表 | icustay_detail（年龄精确计算）, weight_durations, height | 精确年龄、体重 |

#### 4. 你的复现实际只用 12 张表（精简版）

| 来源 | 表格 | 用途 |
|------|------|------|
| hosp | patients, admissions, diagnoses_icd | 基本信息 + 诊断筛选 + 死亡时间 |
| icu | icustays | 首次 ICU + 停留 >24h |
| derived | first_day_sofa → sofa_score | SOFA |
| derived | apsiii, oasis, lods | APSIII/OASIS/LODS |
| derived | first_day_vitalsign | temperature_mean, mbp_mean |
| derived | first_day_lab | bilirubin_total_min, inr_min, ptt_min, chloride_min 等 |
| derived | first_day_bg | lactate_min |
| derived | ventilation, vasoactive_agent | 干预标志（可选） |
| derived | meld | 与 MELD 对比（可选） |

