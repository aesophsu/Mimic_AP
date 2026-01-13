--------------------------------------------------------------------------------
-- 1. AP 诊断提取 (保持不变)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_ap_patients;
CREATE TEMP TABLE temp_ap_patients AS
SELECT hadm_id, subject_id,
    MAX(CASE WHEN icd_code LIKE 'K85.1%' OR icd_code = '5770' THEN 1 ELSE 0 END) AS alcoholic_ap,
    MAX(CASE WHEN icd_code LIKE 'K85.0%' THEN 1 ELSE 0 END) AS biliary_ap,
    MAX(CASE WHEN icd_code LIKE 'K85.2%' OR icd_code LIKE '272.1%' THEN 1 ELSE 0 END) AS hyperlipidemic_ap,
    MAX(CASE WHEN icd_code LIKE 'K85.3%' THEN 1 ELSE 0 END) AS drug_induced_ap
FROM mimiciv_hosp.diagnoses_icd
WHERE (icd_version = 9 AND icd_code = '5770')
   OR (icd_version = 10 AND icd_code LIKE 'K85%')
GROUP BY hadm_id, subject_id;

--------------------------------------------------------------------------------
-- 2. 队列基础表
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS cohort_base;
CREATE TEMP TABLE cohort_base AS
WITH height_global AS (
    SELECT subject_id, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY height) as height_global
    FROM mimiciv_derived.height
    GROUP BY subject_id
),
icu_ranked AS (
    SELECT 
        icu.*,
        ROW_NUMBER() OVER (PARTITION BY icu.hadm_id ORDER BY icu.intime) AS stay_seq
    FROM mimiciv_icu.icustays icu
    INNER JOIN temp_ap_patients ap ON icu.hadm_id = ap.hadm_id
    WHERE icu.los >= 1 
),
demographics AS (
    SELECT 
        pat.subject_id, 
        CASE WHEN pat.gender = 'M' THEN 1 ELSE 0 END AS gender_num, 
        pat.dod, pat.anchor_year, pat.anchor_age
    FROM mimiciv_hosp.patients pat
)
SELECT 
    ir.subject_id, ir.hadm_id, ir.stay_id, ir.intime, ir.los,
    adm.admittime, adm.dischtime, adm.deathtime,
    adm.insurance, adm.race, d.dod, d.gender_num,
    (EXTRACT(YEAR FROM ir.intime) - d.anchor_year + d.anchor_age) AS admission_age,
    ap.alcoholic_ap, ap.biliary_ap, ap.hyperlipidemic_ap, ap.drug_induced_ap,
    w.weight AS weight_admit,
    COALESCE(h.height, hg.height_global) AS height_admit,
    CASE WHEN COALESCE(h.height, hg.height_global) BETWEEN 50 AND 250 
         THEN w.weight / power(COALESCE(h.height, hg.height_global) / 100, 2) 
         ELSE NULL END AS bmi
FROM icu_ranked ir
INNER JOIN temp_ap_patients ap ON ir.hadm_id = ap.hadm_id
INNER JOIN mimiciv_hosp.admissions adm ON ir.hadm_id = adm.hadm_id
INNER JOIN demographics d ON ir.subject_id = d.subject_id
LEFT JOIN mimiciv_derived.first_day_weight w ON ir.stay_id = w.stay_id
LEFT JOIN mimiciv_derived.first_day_height h ON ir.stay_id = h.stay_id
LEFT JOIN height_global hg ON ir.subject_id = hg.subject_id
WHERE ir.stay_seq = 1
  AND (EXTRACT(YEAR FROM ir.intime) - d.anchor_year + d.anchor_age) >= 18;

--------------------------------------------------------------------------------
-- 3. POF 逻辑
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS pof_results;
CREATE TEMP TABLE pof_results AS
WITH sofa_daily AS (
    SELECT
        s.stay_id,
        DATE_TRUNC('day', s.starttime) AS sofa_day,
        MAX(s.respiration)     AS resp,
        MAX(s.cardiovascular)  AS cv,
        MAX(s.renal)           AS renal
    FROM mimiciv_derived.sofa s
    INNER JOIN cohort_base c ON s.stay_id = c.stay_id
    WHERE s.starttime >= c.intime + INTERVAL '24 hours' 
      AND s.starttime <  c.intime + INTERVAL '7 days'
    GROUP BY s.stay_id, DATE_TRUNC('day', s.starttime)
),
organ_pof AS (
    SELECT
        stay_id,
        CASE WHEN COUNT(*) FILTER (WHERE resp  >= 2) >= 2 THEN 1 ELSE 0 END AS resp_pof,
        CASE WHEN COUNT(*) FILTER (WHERE cv    >= 2) >= 2 THEN 1 ELSE 0 END AS cv_pof,
        CASE WHEN COUNT(*) FILTER (WHERE renal >= 2) >= 2 THEN 1 ELSE 0 END AS renal_pof
    FROM sofa_daily
    GROUP BY stay_id
)
SELECT * FROM organ_pof;

--------------------------------------------------------------------------------
-- 4. 核心实验室指标与生命体征 (已修正 hadm_id 重复问题)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_lab_vital_agg;
CREATE TEMP TABLE temp_lab_vital_agg AS
WITH lab_agg AS (
    SELECT 
        le.hadm_id,
        MAX(CASE WHEN itemid = 51214 THEN valuenum END) AS fibrinogen_max,
        MIN(CASE WHEN itemid IN (50820, 50831) AND valuenum BETWEEN 6.5 AND 8.0 THEN valuenum END) AS ph_min,
        MAX(CASE WHEN itemid IN (50820, 50831) AND valuenum BETWEEN 6.5 AND 8.0 THEN valuenum END) AS ph_max,
        MAX(CASE WHEN itemid = 50889 THEN valuenum END) AS crp_max,
        MIN(CASE WHEN itemid IN (50893, 50808) THEN valuenum END) AS calcium_min,
        MIN(CASE WHEN itemid = 50970 THEN valuenum END) AS phosphate_min
    FROM mimiciv_hosp.labevents le
    INNER JOIN cohort_base c ON le.hadm_id = c.hadm_id
    WHERE le.charttime >= (c.admittime - INTERVAL '6 hours') 
      AND le.charttime <= (c.intime + INTERVAL '24 hours')
    GROUP BY le.hadm_id
),
vital_agg AS (
    SELECT 
        ce.stay_id,
        MAX(CASE WHEN itemid IN (220045) THEN valuenum END) AS heart_rate_max,
        MAX(CASE WHEN itemid IN (223761, 223762) THEN (CASE WHEN itemid=223761 THEN (valuenum-32)/1.8 ELSE valuenum END) END) AS temp_max,
        MAX(CASE WHEN itemid IN (220277) THEN valuenum END) AS spo2_max
    FROM mimiciv_icu.chartevents ce
    INNER JOIN cohort_base c ON ce.stay_id = c.stay_id
    WHERE ce.itemid IN (220045, 223761, 223762, 220277)
      AND ce.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
    GROUP BY ce.stay_id
),
bg_agg AS (
    SELECT 
        bg.hadm_id,
        MAX(bg.lactate) AS lactate_max
    FROM mimiciv_derived.bg bg
    INNER JOIN cohort_base c ON bg.hadm_id = c.hadm_id
    WHERE bg.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
    GROUP BY bg.hadm_id
)
SELECT 
    c.stay_id, -- 只保留核心 ID 用于后续 Join
    l.fibrinogen_max, l.ph_min, l.ph_max, l.crp_max, l.calcium_min, l.phosphate_min,
    v.heart_rate_max, v.temp_max, v.spo2_max,
    b.lactate_max
FROM cohort_base c
LEFT JOIN lab_agg l ON c.hadm_id = l.hadm_id
LEFT JOIN vital_agg v ON c.stay_id = v.stay_id
LEFT JOIN bg_agg b ON c.hadm_id = b.hadm_id;

--------------------------------------------------------------------------------
-- 5. 干预措施 (补全缺失的临时表)
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
-- 7. 最终集成
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS my_custom_schema.ap_final_analysis_cohort;
CREATE TABLE my_custom_schema.ap_final_analysis_cohort AS
SELECT
    c.subject_id, c.hadm_id, c.stay_id, c.intime, c.los,
    c.admission_age, c.gender_num, c.weight_admit, c.bmi,
    c.alcoholic_ap, c.biliary_ap, c.hyperlipidemic_ap, c.drug_induced_ap,
    
    CASE WHEN (COALESCE(p.resp_pof,0) + COALESCE(p.cv_pof,0) + COALESCE(p.renal_pof,0)) > 0 THEN 1 ELSE 0 END AS pof,
    
    lab.creatinine AS creatinine_max,
    lab.bun AS bun_max,
    lab.wbc AS wbc_max,
    lab.aniongap AS aniongap_max,
    lab.glucose AS glucose_max,
    lab.ptt AS ptt_max,

    lv.ph_min, lv.fibrinogen_max, lv.lactate_max, lv.heart_rate_max, lv.temp_max, lv.spo2_max,

    COALESCE(intv.mechanical_vent_flag,0) AS mechanical_vent_flag,
    COALESCE(intv.vaso_flag,0) AS vaso_flag

FROM cohort_base c
LEFT JOIN pof_results p ON c.stay_id = p.stay_id
LEFT JOIN mimiciv_derived.first_day_lab lab ON c.stay_id = lab.stay_id
LEFT JOIN temp_lab_vital_agg lv ON c.stay_id = lv.stay_id
LEFT JOIN temp_interventions intv ON c.stay_id = intv.stay_id;

-- 验证最终行数
SELECT COUNT(*) FROM my_custom_schema.ap_final_analysis_cohort;
