/* 在执行前，请确保已创建自定义 schema，例如：
CREATE SCHEMA my_custom_schema;
/* ============================================================
    Title: Alcoholic Cirrhosis (AC) 28-day Mortality Cohort Extraction
    Database: MIMIC-IV (v2.x+)
    Author: [Your Name]
    Description:
        构建用于 28 天全因死亡率预测模型的核心分析队列。
        逻辑顺序：
        1. 筛选诊断为酒精性肝硬化的患者；
        2. 获取 ICU 住院记录且停留时间 > 24h；
        3. 保留首次 ICU 入院；
        4. 联接住院与患者基本信息；
        5. 计算 28 天死亡；
        6. 结合 derived 模块中的评分、实验室和干预数据。
   ============================================================ */

-- 如果存在旧表，先删除以防冲突
DROP TABLE IF EXISTS my_custom_schema.ac_final_analysis_cohort;

-- ================================
-- Step 1. 患者诊断筛选
-- ================================
WITH filtered_patients AS (
    SELECT DISTINCT subject_id
    FROM mimiciv_hosp.diagnoses_icd
    WHERE
        (icd_version = 9 AND icd_code = '5712')         -- ICD-9: Alcoholic Cirrhosis
        OR (icd_version = 10 AND icd_code IN ('K7030', 'K7031'))  -- ICD-10: Alcoholic cirrhosis
),

-- ================================
-- Step 2. ICU 入院与时长筛选
-- ================================
first_icustay AS (
    SELECT
        icu.subject_id,
        icu.hadm_id,
        icu.stay_id,
        icu.los,                -- length of stay (days)
        icu.intime,
        ROW_NUMBER() OVER (PARTITION BY icu.subject_id ORDER BY icu.intime) AS rn
    FROM
        filtered_patients p
        INNER JOIN mimiciv_icu.icustays icu
        ON p.subject_id = icu.subject_id
    WHERE
        icu.los > 1             -- ICU stay > 24h
),

-- ================================
-- Step 3. 核心患者队列 + 基本信息
-- ================================
core_cohort AS (
    SELECT
        icu.subject_id,
        icu.hadm_id,
        icu.stay_id,
        icu.los,
        icu.intime,
        adm.admittime,
        adm.dischtime,
        adm.deathtime,
        adm.insurance,
        adm.race,
        adm.hospital_expire_flag,
        pat.dod,                -- date of death (long-term)
        pat.anchor_age,
        pat.gender
    FROM
        first_icustay icu
        INNER JOIN mimiciv_hosp.admissions adm
            ON icu.hadm_id = adm.hadm_id
        INNER JOIN mimiciv_hosp.patients pat
            ON icu.subject_id = pat.subject_id
    WHERE
        icu.rn = 1              -- keep only first ICU admission
        AND pat.anchor_age > 18 -- adult patients only
),

-- ================================
-- Step 4. ICU 干预和并发症数据
-- ================================
vent_status AS (
    -- 24h内是否有机械通气
    SELECT vent.stay_id
    FROM mimiciv_derived.ventilation vent
    INNER JOIN core_cohort c ON vent.stay_id = c.stay_id
    WHERE
        vent.ventilation_status = 'InvasiveVent'
        AND vent.starttime BETWEEN c.intime AND c.intime + INTERVAL '24' HOUR
    GROUP BY vent.stay_id
),
vaso_status AS (
    -- 是否使用血管活性药物
    SELECT stay_id
    FROM mimiciv_derived.vasoactive_agent
    WHERE dopamine IS NOT NULL OR epinephrine IS NOT NULL
       OR norepinephrine IS NOT NULL OR phenylephrine IS NOT NULL
       OR vasopressin IS NOT NULL
    GROUP BY stay_id
),
comorbidities AS (
    -- 提取常见共病（心衰、房颤、COPD、肿瘤等）
    SELECT
        hadm_id,
        MAX(CASE WHEN icd_code LIKE '428%' OR icd_code LIKE 'I50%' THEN 1 ELSE 0 END) AS heart_failure,
        MAX(CASE WHEN icd_code LIKE '4273%' OR icd_code LIKE 'I48%' THEN 1 ELSE 0 END) AS atrial_fibrillation,
        MAX(CASE WHEN icd_code LIKE '585%'  OR icd_code LIKE 'N18%' THEN 1 ELSE 0 END) AS chronic_kidney_disease,
        MAX(CASE WHEN icd_code LIKE '491%' OR icd_code LIKE '492%' OR icd_code LIKE '496%' OR icd_code LIKE 'J44%' OR icd_code LIKE 'J43%' THEN 1 ELSE 0 END) AS copd,
        MAX(CASE WHEN icd_code LIKE '410%' OR icd_code LIKE '411%' OR icd_code LIKE '414%' OR icd_code LIKE 'I20%' OR icd_code LIKE 'I25%' THEN 1 ELSE 0 END) AS coronary_heart_disease,
        MAX(CASE WHEN icd_code LIKE '433%' OR icd_code LIKE '434%' OR icd_code LIKE '436%' OR icd_code LIKE 'I63%' OR icd_code LIKE 'I64%' THEN 1 ELSE 0 END) AS stroke,
        MAX(CASE WHEN icd_code LIKE '14%' OR icd_code LIKE '15%' OR icd_code LIKE '16%' OR icd_code LIKE '17%' OR icd_code LIKE '18%' OR icd_code LIKE '19%' OR icd_code LIKE '20%' OR icd_code LIKE 'C%' THEN 1 ELSE 0 END) AS malignant_tumor,
        MAX(CASE WHEN icd_code LIKE '2860%' OR icd_code LIKE 'D66%' OR icd_code LIKE 'D67%' OR icd_code LIKE 'D68%' THEN 1 ELSE 0 END) AS congenital_coagulation_defects
    FROM mimiciv_hosp.diagnoses_icd
    GROUP BY hadm_id
)

-- ================================
-- Step 5. 最终主查询（整合 derived 特征 + 死亡指标）
-- ================================
SELECT
    c.*,

    -- 计算 ICU 入科后 28 天内全因死亡
    CASE
        WHEN COALESCE(c.deathtime, c.dod) IS NOT NULL
         AND (COALESCE(c.deathtime, c.dod) - c.intime) <= INTERVAL '28 days'
        THEN 1 ELSE 0 END AS mortality_28d,

    -- 临床评分
    sofa.sofa AS sofa_score,
    apsiii.apsiii,
    sapsii.sapsii,
    oasis.oasis,
    lods.lods,

    -- 生命体征（首日均值）
    vitals.mbp_mean,
    vitals.heart_rate_mean,
    vitals.temperature_mean,

    -- 实验室（首日最小值与最大值）
    lab.wbc_min, lab.wbc_max,
    lab.platelets_min, lab.platelets_max,
    lab.sodium_min, lab.sodium_max,
    lab.potassium_min, lab.potassium_max,
    lab.bicarbonate_min, lab.bicarbonate_max,
    lab.chloride_min, lab.chloride_max,
    lab.bun_min, lab.bun_max,
    lab.creatinine_min, lab.creatinine_max,
    lab.bilirubin_total_min, lab.bilirubin_total_max,
    lab.albumin_min, lab.albumin_max,
    lab.inr_min, lab.inr_max,
    lab.ptt_min, lab.ptt_max,
    bg.lactate_min, bg.lactate_max,

    -- 身高与体重
    height.height,
    weight.weight,

    -- 干预标志
    CASE WHEN vent_status.stay_id IS NOT NULL THEN 1 ELSE 0 END AS mechanical_ventilation_flag,
    CASE WHEN vaso_status.stay_id IS NOT NULL THEN 1 ELSE 0 END AS vaso_flag,

    -- 共病标志
    com.heart_failure,
    com.atrial_fibrillation,
    com.chronic_kidney_disease,
    com.copd,
    com.coronary_heart_disease,
    com.stroke,
    com.malignant_tumor,
    com.congenital_coagulation_defects

INTO my_custom_schema.ac_final_analysis_cohort
FROM
    core_cohort c
    LEFT JOIN mimiciv_derived.first_day_sofa sofa      ON c.stay_id = sofa.stay_id
    LEFT JOIN mimiciv_derived.apsiii apsiii            ON c.stay_id = apsiii.stay_id
    LEFT JOIN mimiciv_derived.sapsii sapsii            ON c.stay_id = sapsii.stay_id
    LEFT JOIN mimiciv_derived.oasis oasis              ON c.stay_id = oasis.stay_id
    LEFT JOIN mimiciv_derived.lods lods                ON c.stay_id = lods.stay_id
    LEFT JOIN mimiciv_derived.first_day_vitalsign vitals ON c.stay_id = vitals.stay_id
    LEFT JOIN mimiciv_derived.first_day_lab lab        ON c.stay_id = lab.stay_id
    LEFT JOIN mimiciv_derived.first_day_bg bg          ON c.stay_id = bg.stay_id
    LEFT JOIN mimiciv_derived.first_day_height height  ON c.stay_id = height.stay_id
    LEFT JOIN mimiciv_derived.first_day_weight weight  ON c.stay_id = weight.stay_id
    LEFT JOIN vent_status                              ON c.stay_id = vent_status.stay_id
    LEFT JOIN vaso_status                              ON c.stay_id = vaso_status.stay_id
    LEFT JOIN comorbidities com                        ON c.hadm_id = com.hadm_id;


/*创建后的表
SELECT COUNT(*) FROM my_custom_schema.ac_final_analysis_cohort;
