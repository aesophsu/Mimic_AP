--------------------------------------------------------------------------------
-- 1. First materialize AP diagnoses (ICD-9: 5770, ICD-10: K85%)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_ap_patients;
CREATE TEMP TABLE temp_ap_patients AS
SELECT DISTINCT 
    hadm_id, 
    subject_id
FROM mimiciv_hosp.diagnoses_icd
WHERE (icd_version = 9 AND icd_code = '5770')
   OR (icd_version = 10 AND icd_code LIKE 'K85%');

SELECT 
    COUNT(DISTINCT subject_id) AS total_patients,
    COUNT(DISTINCT hadm_id) AS total_admissions
FROM temp_ap_patients;

--------------------------------------------------------------------------------
-- 2. Core cohort base (Optimized for First AP Admission)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS cohort_base;
CREATE TEMP TABLE cohort_base AS
WITH ap_icu_stays AS (
    SELECT 
        icu.*,
        ROW_NUMBER() OVER (PARTITION BY icu.subject_id ORDER BY icu.intime) AS ap_stay_seq
    FROM mimiciv_icu.icustays icu
    INNER JOIN temp_ap_patients ap ON icu.hadm_id = ap.hadm_id
    WHERE icu.los >= 1 
),
height_global AS (
    SELECT subject_id, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY height) as height_global
    FROM mimiciv_derived.height
    GROUP BY subject_id
),
demographics AS (
    SELECT 
        pat.subject_id, pat.gender, pat.dod,
        pat.anchor_year, pat.anchor_age
    FROM mimiciv_hosp.patients pat
)
SELECT 
    ir.subject_id, ir.hadm_id, ir.stay_id, ir.intime, ir.los,
    adm.admittime, adm.dischtime, adm.deathtime,
    d.dod, d.gender,
    (EXTRACT(YEAR FROM ir.intime) - d.anchor_year + d.anchor_age) AS admission_age,
    w.weight AS weight_admit,
    COALESCE(h.height, hg.height_global) AS height_admit,
    CASE WHEN COALESCE(h.height, hg.height_global) > 0 
         THEN w.weight / power(COALESCE(h.height, hg.height_global) / 100, 2) 
         ELSE NULL END AS bmi
FROM ap_icu_stays ir
INNER JOIN mimiciv_hosp.admissions adm ON ir.hadm_id = adm.hadm_id
INNER JOIN demographics d ON ir.subject_id = d.subject_id
LEFT JOIN mimiciv_derived.first_day_weight w ON ir.stay_id = w.stay_id
LEFT JOIN mimiciv_derived.first_day_height h ON ir.stay_id = h.stay_id
LEFT JOIN height_global hg ON ir.subject_id = hg.subject_id
WHERE ir.ap_stay_seq = 1
  AND (EXTRACT(YEAR FROM ir.intime) - d.anchor_year + d.anchor_age) >= 18;
SELECT 
    COUNT(DISTINCT stay_id) AS final_icu_stay_count
FROM cohort_base;

--------------------------------------------------------------------------------
-- 3. Optimized POF (Persistent Organ Failure)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS pof_intermediate;
CREATE TEMP TABLE pof_intermediate AS
WITH sofa_daily AS (
    SELECT
        s.stay_id,
        DATE_TRUNC('day', s.starttime) AS sofa_day,
        MAX(s.respiration) AS resp,
        MAX(s.cardiovascular) AS cv,
        MAX(s.renal) AS renal
    FROM mimiciv_derived.sofa s
    INNER JOIN cohort_base c ON s.stay_id = c.stay_id
    WHERE s.starttime >= c.intime + INTERVAL '24 hours'
      AND s.starttime < c.intime + INTERVAL '7 days'
    GROUP BY s.stay_id, DATE_TRUNC('day', s.starttime)
),
organ_pof_logic AS (
    SELECT
        stay_id,
        CASE WHEN COUNT(*) FILTER (WHERE resp >= 2) >= 2 THEN 1 ELSE 0 END AS resp_pof,
        CASE WHEN COUNT(*) FILTER (WHERE cv   >= 2) >= 2 THEN 1 ELSE 0 END AS cv_pof,
        CASE WHEN COUNT(*) FILTER (WHERE renal>= 2) >= 2 THEN 1 ELSE 0 END AS renal_pof
    FROM sofa_daily
    GROUP BY stay_id
)
SELECT * FROM organ_pof_logic;

DROP TABLE IF EXISTS pof_results;
CREATE TEMP TABLE pof_results AS
SELECT 
    c.stay_id,
    c.subject_id,
    COALESCE(p.resp_pof, 0) AS resp_pof,
    COALESCE(p.cv_pof, 0) AS cv_pof,
    COALESCE(p.renal_pof, 0) AS renal_pof,
    -- 计算 POF 状态
    CASE WHEN (COALESCE(p.resp_pof, 0) + COALESCE(p.cv_pof, 0) + COALESCE(p.renal_pof, 0)) > 0 
         THEN 1 ELSE 0 END AS any_pof,
    
    -- 【新增字段】判定 24-48小时内死亡
    -- 同时检查 deathtime (医院记录) 和 dod (系统死亡日期)
    CASE WHEN (c.deathtime BETWEEN c.intime + INTERVAL '24 hours' AND c.intime + INTERVAL '48 hours')
           OR (c.dod BETWEEN c.intime + INTERVAL '24 hours' AND c.intime + INTERVAL '48 hours')
         THEN 1 ELSE 0 END AS early_death_24_48h
FROM cohort_base c
LEFT JOIN pof_intermediate p ON c.stay_id = p.stay_id;

SELECT 
    COUNT(*) AS total_rows,
    SUM(any_pof) AS total_pof_cases,
    SUM(early_death_24_48h) AS total_early_death_cases,
    -- 统计：有多少早期死亡患者确实被记为了 POF=0
    COUNT(*) FILTER (WHERE early_death_24_48h = 1 AND any_pof = 0) AS early_death_as_non_pof
FROM pof_results;

--------------------------------------------------------------------------------
-- 4. Batch lab slopes (Optimized: Admission to 24h post-ICU)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_lab_slopes;
CREATE TEMP TABLE temp_lab_slopes AS
WITH lab_agg AS (
    SELECT 
        le.hadm_id,
        AVG(CASE WHEN itemid IN (51256, 52075) THEN valuenum END) AS neutrophils_mean,
        AVG(CASE WHEN itemid IN (51244, 51133) THEN valuenum END) AS lymphocytes_mean,
        MAX(CASE WHEN itemid IN (50915, 51196, 52551) THEN valuenum END) AS d_dimer_max,
        MAX(CASE WHEN itemid = 51214 THEN valuenum END) AS fibrinogen_max,
        MAX(CASE WHEN itemid = 50954 THEN valuenum END) AS ldh_max,
        MAX(CASE WHEN itemid = 50889 THEN valuenum END) AS crp_max,
        MAX(CASE WHEN itemid IN (50867, 53087) THEN valuenum END) AS amylase_max,
        MAX(CASE WHEN itemid = 50956 THEN valuenum END) AS lipase_max,
        MIN(CASE WHEN itemid IN (50931, 50809, 52569) THEN valuenum END) AS glucose_lab_min,
        MAX(CASE WHEN itemid IN (50931, 50809, 52569) THEN valuenum END) AS glucose_lab_max,
        -- 优化后的血糖斜率：只有在采样点 >= 3 且时间差 > 0.5 小时时计算
        CASE WHEN COUNT(CASE WHEN itemid IN (50931, 50809, 52569) THEN 1 END) >= 3 
             AND EXTRACT(EPOCH FROM (MAX(CASE WHEN itemid IN (50931, 50809, 52569) THEN charttime END) - 
                                     MIN(CASE WHEN itemid IN (50931, 50809, 52569) THEN charttime END)))/3600 > 0.5
             THEN (MAX(CASE WHEN itemid IN (50931, 50809, 52569) THEN valuenum END) - 
                   MIN(CASE WHEN itemid IN (50931, 50809, 52569) THEN valuenum END)) / 
                  (EXTRACT(EPOCH FROM (MAX(CASE WHEN itemid IN (50931, 50809, 52569) THEN charttime END) - 
                                       MIN(CASE WHEN itemid IN (50931, 50809, 52569) THEN charttime END)))/3600)
             ELSE NULL END AS glucose_slope,
        MAX(CASE WHEN itemid = 51000 THEN valuenum END) AS triglycerides_max,
        MIN(CASE WHEN itemid = 50907 THEN valuenum END) AS total_cholesterol_min,
        MAX(CASE WHEN itemid IN (51277, 52172) THEN valuenum END) AS rdw_max,
        MIN(CASE WHEN itemid IN (50970) THEN valuenum END) AS phosphate_min,
        MIN(CASE WHEN itemid IN (50893, 50808) THEN valuenum END) AS calcium_min
    FROM mimiciv_hosp.labevents le
    INNER JOIN cohort_base c ON le.hadm_id = c.hadm_id
    WHERE le.charttime >= (c.admittime - INTERVAL '6 hours') 
      AND le.charttime <= (c.intime + INTERVAL '24 hours')
      AND le.valuenum IS NOT NULL
      AND itemid IN (51256, 52075, 51244, 51133, 50915, 51196, 52551, 51214, 50954, 
                     50889, 50867, 53087, 50956, 50931, 50809, 52569, 51000, 50907, 
                     51277, 52172, 50970, 50893, 50808)
    GROUP BY le.hadm_id
),
bg_agg AS (
    SELECT 
        bg.hadm_id,
        MIN(bg.lactate) AS lactate_min,
        MAX(bg.lactate) AS lactate_max,
        -- 优化后的乳酸斜率：增加采样点限制
        CASE WHEN COUNT(bg.lactate) >= 3 
             AND EXTRACT(EPOCH FROM (MAX(bg.charttime) - MIN(bg.charttime)))/3600 > 0.5
             THEN (MAX(bg.lactate) - MIN(bg.lactate)) / 
                  (EXTRACT(EPOCH FROM (MAX(bg.charttime) - MIN(bg.charttime)))/3600)
             ELSE NULL END AS lactate_slope
    FROM mimiciv_derived.bg bg
    INNER JOIN cohort_base c ON bg.hadm_id = c.hadm_id
    WHERE bg.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
      AND bg.lactate IS NOT NULL
    GROUP BY bg.hadm_id
),
vital_agg AS (
    SELECT 
        ce.stay_id,
        MIN(CASE WHEN itemid=220277 THEN valuenum END) AS spo2_min,
        MAX(CASE WHEN itemid=220277 THEN valuenum END) AS spo2_max,
        -- 优化后的血氧斜率
        CASE WHEN COUNT(CASE WHEN itemid=220277 THEN 1 END) >= 3
             AND EXTRACT(EPOCH FROM (MAX(CASE WHEN itemid=220277 THEN charttime END) - 
                                     MIN(CASE WHEN itemid=220277 THEN charttime END)))/3600 > 0.5
             THEN (MAX(CASE WHEN itemid=220277 THEN valuenum END) - 
                   MIN(CASE WHEN itemid=220277 THEN valuenum END)) / 
                  (EXTRACT(EPOCH FROM (MAX(CASE WHEN itemid=220277 THEN charttime END) - 
                                       MIN(CASE WHEN itemid=220277 THEN charttime END)))/3600)
             ELSE NULL END AS spo2_slope
    FROM mimiciv_icu.chartevents ce
    INNER JOIN cohort_base c ON ce.stay_id = c.stay_id
    WHERE ce.itemid = 220277 
      AND ce.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
      AND ce.valuenum IS NOT NULL
    GROUP BY ce.stay_id
)
SELECT 
    c.hadm_id,
    l.neutrophils_mean, l.lymphocytes_mean, l.d_dimer_max, l.fibrinogen_max, l.ldh_max,
    l.crp_max, l.amylase_max, l.lipase_max, l.glucose_lab_min, l.glucose_lab_max, l.glucose_slope,
    l.triglycerides_max, l.total_cholesterol_min, l.rdw_max, l.phosphate_min, l.calcium_min,
    CASE WHEN l.lymphocytes_mean > 0 THEN l.neutrophils_mean / l.lymphocytes_mean ELSE NULL END AS nlr,
    b.lactate_min, b.lactate_max, b.lactate_slope,
    v.spo2_min, v.spo2_max, v.spo2_slope
FROM cohort_base c
LEFT JOIN lab_agg l ON c.hadm_id = l.hadm_id
LEFT JOIN bg_agg b ON c.hadm_id = b.hadm_id
LEFT JOIN vital_agg v ON c.stay_id = v.stay_id;

--------------------------------------------------------------------------------
-- 5. Batch comorbidities
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_comorbidities;
CREATE TEMP TABLE temp_comorbidities AS
SELECT 
    c.hadm_id,
    MAX(CASE WHEN icd_code LIKE '428%' OR icd_code LIKE 'I50%' THEN 1 ELSE 0 END) AS heart_failure,
    MAX(CASE WHEN icd_code LIKE '4273%' OR icd_code LIKE 'I48%' THEN 1 ELSE 0 END) AS atrial_fibrillation,
    MAX(CASE WHEN icd_code LIKE '585%' OR icd_code LIKE 'N18%' THEN 1 ELSE 0 END) AS chronic_kidney_disease,
    MAX(CASE WHEN icd_code LIKE '491%' OR icd_code LIKE '492%' OR icd_code LIKE '496%'
           OR icd_code LIKE 'J44%' OR icd_code LIKE 'J43%' THEN 1 ELSE 0 END) AS copd,
    MAX(CASE WHEN icd_code LIKE '410%' OR icd_code LIKE '411%' OR icd_code LIKE '414%'
           OR icd_code LIKE 'I20%' OR icd_code LIKE 'I25%' THEN 1 ELSE 0 END) AS coronary_heart_disease,
    MAX(CASE WHEN icd_code LIKE '433%' OR icd_code LIKE '434%' OR icd_code LIKE '436%'
           OR icd_code LIKE 'I63%' OR icd_code LIKE 'I64%' THEN 1 ELSE 0 END) AS stroke,
    MAX(CASE WHEN icd_code LIKE '14%' OR icd_code LIKE '15%' OR icd_code LIKE 'C%' THEN 1 ELSE 0 END) AS malignant_tumor
FROM mimiciv_hosp.diagnoses_icd d
INNER JOIN cohort_base c ON d.hadm_id = c.hadm_id
GROUP BY c.hadm_id;

--------------------------------------------------------------------------------
-- 6. Interventions
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_interventions;
CREATE TEMP TABLE temp_interventions AS
SELECT 
    c.stay_id,
    MAX(CASE WHEN vent.ventilation_status IN ('InvasiveVent', 'Tracheostomy') THEN 1 ELSE 0 END) AS mechanical_vent_flag,
    MAX(CASE WHEN vaso.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS vaso_flag
FROM cohort_base c
LEFT JOIN mimiciv_derived.ventilation vent ON c.stay_id = vent.stay_id 
    AND vent.starttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
LEFT JOIN mimiciv_derived.vasoactive_agent vaso ON c.stay_id = vaso.stay_id 
    AND vaso.starttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
GROUP BY c.stay_id;

--------------------------------------------------------------------------------
-- 7. Final Integration
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS my_custom_schema.ap_final_analysis_cohort;
CREATE TABLE my_custom_schema.ap_final_analysis_cohort AS
SELECT
    c.*,
    -- 1. 结局指标 (Outcomes)
    p.any_pof AS pof,
    p.resp_pof, 
    p.cv_pof, 
    p.renal_pof,
    -- 28天死亡标记
    CASE WHEN COALESCE(c.deathtime, c.dod) <= (c.intime + INTERVAL '28 days') THEN 1 ELSE 0 END AS mortality_28d,
    -- 【新增】24-48小时内早期死亡标记 (用于解释竞争风险)
    p.early_death_24_48h,
    -- 【新增】研究共同终点 (POF 或 28天死亡)
    CASE WHEN p.any_pof = 1 OR COALESCE(c.deathtime, c.dod) <= (c.intime + INTERVAL '28 days') THEN 1 ELSE 0 END AS composite_outcome,
    
    -- 2. 临床评分 (Severity Scores)
    sofa.sofa AS sofa_score,
    apsiii.apsiii, sapsii.sapsii, oasis.oasis, lods.lods,
    
    -- 3. 常规实验室检查 (First Day Labs)
    lab.wbc_min, lab.wbc_max, lab.hematocrit_min, lab.hematocrit_max,
    lab.hemoglobin_min, lab.hemoglobin_max, lab.platelets_min, lab.platelets_max,
    lab.albumin_min, lab.albumin_max, lab.aniongap_min, lab.aniongap_max,
    lab.bicarbonate_min, lab.bicarbonate_max, lab.bun_min, lab.bun_max,
    lab.calcium_min as lab_calcium_min, lab.calcium_max, lab.chloride_min, lab.chloride_max,
    lab.creatinine_min, lab.creatinine_max, lab.glucose_min, lab.glucose_max,
    lab.sodium_min, lab.sodium_max, lab.potassium_min, lab.potassium_max,
    lab.bilirubin_total_min, lab.bilirubin_total_max, lab.inr_min, lab.inr_max,
    lab.pt_min, lab.pt_max, lab.ptt_min, lab.ptt_max,
    lab.amylase_min as lab_amylase_min, lab.amylase_max as lab_amylase_max,
    lab.alt_min, lab.alt_max, lab.ast_min, lab.ast_max, lab.alp_min, lab.alp_max,
    
    -- 4. AP 特异性及机器学习特征 (AP Specific & ML Features)
    ls.crp_max, ls.phosphate_min, ls.lipase_max,
    ls.neutrophils_mean, ls.lymphocytes_mean, ls.nlr,
    ls.d_dimer_max, ls.fibrinogen_max, ls.ldh_max,
    ls.glucose_lab_min, ls.glucose_lab_max, ls.glucose_slope,
    ls.rdw_max, ls.triglycerides_max, ls.total_cholesterol_min,
    
    -- 5. 斜率与血气指标 (BG & Vitals Slopes)
    ls.lactate_min, ls.lactate_max, ls.lactate_slope,
    ls.spo2_min, ls.spo2_max, ls.spo2_slope,
    bg.ph_min, bg.ph_max, bg.pao2fio2ratio_min, bg.pao2fio2ratio_max,
    
    -- 6. 衍生比值 (Ratios)
    CASE WHEN lab.albumin_min > 0 THEN ls.lactate_max / lab.albumin_min ELSE NULL END AS lar,
    CASE WHEN lab.albumin_min > 0 THEN lab.bilirubin_total_max / lab.albumin_min ELSE NULL END AS tbar,
    
    -- 7. 干预措施与合并症 (Interventions & Comorbidities)
    COALESCE(intv.mechanical_vent_flag, 0) AS mechanical_vent_flag,
    COALESCE(intv.vaso_flag, 0) AS vaso_flag,
    COALESCE(com.heart_failure, 0) AS heart_failure,
    COALESCE(com.chronic_kidney_disease, 0) AS chronic_kidney_disease,
    COALESCE(com.malignant_tumor, 0) AS malignant_tumor

FROM cohort_base c
LEFT JOIN pof_results p ON c.stay_id = p.stay_id
LEFT JOIN mimiciv_derived.first_day_sofa sofa ON c.stay_id = sofa.stay_id
LEFT JOIN mimiciv_derived.apsiii apsiii ON c.stay_id = apsiii.stay_id
LEFT JOIN mimiciv_derived.sapsii sapsii ON c.stay_id = sapsii.stay_id
LEFT JOIN mimiciv_derived.oasis oasis ON c.stay_id = oasis.stay_id
LEFT JOIN mimiciv_derived.lods lods ON c.stay_id = lods.stay_id
LEFT JOIN mimiciv_derived.first_day_lab lab ON c.stay_id = lab.stay_id
LEFT JOIN mimiciv_derived.first_day_bg bg ON c.stay_id = bg.stay_id
LEFT JOIN temp_lab_slopes ls ON c.hadm_id = ls.hadm_id
LEFT JOIN temp_interventions intv ON c.stay_id = intv.stay_id
LEFT JOIN temp_comorbidities com ON c.hadm_id = com.hadm_id;

-- 统计验证：检查 POF 和 早期死亡的重叠情况
SELECT 
    COUNT(*) AS total_count,
    SUM(pof) AS pof_count,
    SUM(mortality_28d) AS deaths_28d,
    SUM(early_death_24_48h) AS early_deaths,
    SUM(composite_outcome) AS composite_events
FROM my_custom_schema.ap_final_analysis_cohort;

SELECT 
    -- 1. 检查结局指标（应该都是 100%，因为做了 COALESCE）
    COUNT(pof) * 100.0 / COUNT(*) AS pof_fill_rate,
    
    -- 2. 检查关键斜率（Slope）的填充率（反映采样频率是否达标）
    COUNT(glucose_slope) * 100.0 / COUNT(*) AS glucose_slope_fill_rate,
    COUNT(lactate_slope) * 100.0 / COUNT(*) AS lactate_slope_fill_rate,
    COUNT(spo2_slope) * 100.0 / COUNT(*) AS spo2_slope_fill_rate,
    
    -- 3. 检查 AP 特异指标（使用重命名后的正确字段名）
    COUNT(lab_amylase_max) * 100.0 / COUNT(*) AS amylase_fill_rate,
    COUNT(lipase_max) * 100.0 / COUNT(*) AS lipase_fill_rate,
    COUNT(crp_max) * 100.0 / COUNT(*) AS crp_fill_rate,
    COUNT(d_dimer_max) * 100.0 / COUNT(*) AS d_dimer_fill_rate,
    
    -- 4. 检查计算比值
    COUNT(lar) * 100.0 / COUNT(*) AS lar_fill_rate
FROM my_custom_schema.ap_final_analysis_cohort;

SELECT 
    -- 之前的斜率填充率
    COUNT(lactate_slope) * 100.0 / COUNT(*) AS lactate_slope_fill_rate,
    
    -- 退而求其次的单点最大值填充率
    COUNT(lactate_max) * 100.0 / COUNT(*) AS lactate_max_fill_rate,
    
    -- 顺便看看最小值，通常 Max/Min 是一对，填充率应该一致
    COUNT(lactate_min) * 100.0 / COUNT(*) AS lactate_min_fill_rate
FROM my_custom_schema.ap_final_analysis_cohort;
