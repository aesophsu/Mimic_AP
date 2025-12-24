-- 1. First materialize AP diagnoses
DROP TABLE IF EXISTS temp_ap_patients;
CREATE TEMP TABLE temp_ap_patients AS
SELECT hadm_id, subject_id,
    MAX(CASE WHEN icd_code LIKE 'K85.1%' OR icd_code = '5770' THEN 1 ELSE 0 END) AS alcoholic_ap,
    MAX(CASE WHEN icd_code LIKE 'K85.0%' THEN 1 ELSE 0 END) AS biliary_ap
FROM mimiciv_hosp.diagnoses_icd
WHERE (icd_version = 9 AND icd_code = '5770')
   OR (icd_version = 10 AND icd_code LIKE 'K85%')
GROUP BY hadm_id, subject_id;

-- 2. Core cohort base
DROP TABLE IF EXISTS cohort_base;
CREATE TEMP TABLE cohort_base AS
WITH icu_ranked AS (
    SELECT 
        icu.*,
        ROW_NUMBER() OVER (PARTITION BY icu.hadm_id ORDER BY icu.intime) AS stay_seq
    FROM mimiciv_icu.icustays icu
    INNER JOIN temp_ap_patients ap ON icu.hadm_id = ap.hadm_id
    WHERE icu.los >= 1 
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
    adm.insurance, adm.race, d.dod, d.gender,
    (EXTRACT(YEAR FROM ir.intime) - d.anchor_year + d.anchor_age) AS admission_age,
    ap.alcoholic_ap, ap.biliary_ap,
    w.weight AS weight_admit,
    h.height AS height_admit,
    -- BMI Calc: Weight(kg) / (Height(m)^2)
    CASE WHEN h.height > 0 THEN w.weight / power(h.height / 100, 2) ELSE NULL END AS bmi
FROM icu_ranked ir
INNER JOIN temp_ap_patients ap ON ir.hadm_id = ap.hadm_id
INNER JOIN mimiciv_hosp.admissions adm ON ir.hadm_id = adm.hadm_id
INNER JOIN demographics d ON ir.subject_id = d.subject_id
LEFT JOIN mimiciv_derived.first_day_weight w ON ir.stay_id = w.stay_id
LEFT JOIN mimiciv_derived.first_day_height h ON ir.stay_id = h.stay_id
WHERE ir.stay_seq = 1
  AND (EXTRACT(YEAR FROM ir.intime) - d.anchor_year + d.anchor_age) >= 18;

CREATE INDEX idx_cohort_stay ON cohort_base(stay_id);
CREATE INDEX idx_cohort_hadm ON cohort_base(hadm_id);

-- 3. Optimized POF (Persistent Organ Failure)
DROP TABLE IF EXISTS pof_results;
CREATE TEMP TABLE pof_results AS
WITH sofa_flagged AS (
    SELECT
        s.stay_id, s.starttime, s.endtime,
        CASE WHEN respiration    >= 2 THEN 1 ELSE 0 END AS resp_of,
        CASE WHEN cardiovascular >= 2 THEN 1 ELSE 0 END AS cv_of,
        CASE WHEN renal          >= 2 THEN 1 ELSE 0 END AS renal_of,
        ROW_NUMBER() OVER (PARTITION BY s.stay_id ORDER BY s.starttime) AS rn
    FROM mimiciv_derived.sofa s
    INNER JOIN cohort_base c ON s.stay_id = c.stay_id
    WHERE s.starttime >= c.intime 
),
sofa_streaks AS (
    SELECT *,
        rn - ROW_NUMBER() OVER (PARTITION BY stay_id, resp_of ORDER BY starttime) AS resp_grp,
        rn - ROW_NUMBER() OVER (PARTITION BY stay_id, cv_of ORDER BY starttime) AS cv_grp,
        rn - ROW_NUMBER() OVER (PARTITION BY stay_id, renal_of ORDER BY starttime) AS renal_grp
    FROM sofa_flagged
),
durations AS (
    SELECT stay_id,
        resp_of, SUM(endtime - starttime) OVER (PARTITION BY stay_id, resp_of, resp_grp) AS resp_d,
        cv_of,   SUM(endtime - starttime) OVER (PARTITION BY stay_id, cv_of, cv_grp) AS cv_d,
        renal_of,SUM(endtime - starttime) OVER (PARTITION BY stay_id, renal_of, renal_grp) AS renal_d
    FROM sofa_streaks
)
SELECT stay_id,
    MAX(CASE WHEN resp_of = 1 AND resp_d >= INTERVAL '48 hours' THEN 1 ELSE 0 END) AS resp_pof,
    MAX(CASE WHEN cv_of = 1   AND cv_d   >= INTERVAL '48 hours' THEN 1 ELSE 0 END) AS cv_pof,
    MAX(CASE WHEN renal_of = 1 AND renal_d >= INTERVAL '48 hours' THEN 1 ELSE 0 END) AS renal_pof
FROM durations
GROUP BY stay_id;

-- 4. Batch lab slopes + AP-specific Labs (Fixed Column Ambiguity)
DROP TABLE IF EXISTS temp_lab_slopes;
CREATE TEMP TABLE temp_lab_slopes AS
WITH lab_agg AS (
    SELECT 
        le.hadm_id,
        MIN(CASE WHEN itemid=50889 THEN valuenum END) AS crp_min,
        MAX(CASE WHEN itemid=50889 THEN valuenum END) AS crp_max,
        (MAX(CASE WHEN itemid=50889 THEN valuenum END) - MIN(CASE WHEN itemid=50889 THEN valuenum END)) / 
        NULLIF(EXTRACT(EPOCH FROM (MAX(CASE WHEN itemid=50889 THEN charttime END) - MIN(CASE WHEN itemid=50889 THEN charttime END)))/3600, 0) AS crp_slope,
        MIN(CASE WHEN itemid=50971 THEN valuenum END) AS phosphate_min,
        MAX(CASE WHEN itemid=50971 THEN valuenum END) AS phosphate_max,
        MIN(CASE WHEN itemid IN (50893, 50808) THEN valuenum END) AS calcium_min,
        MAX(CASE WHEN itemid IN (50893, 50808) THEN valuenum END) AS calcium_max,
        MIN(CASE WHEN itemid=50867 THEN valuenum END) AS amylase_min,
        MAX(CASE WHEN itemid=50867 THEN valuenum END) AS amylase_max,
        MIN(CASE WHEN itemid=50956 THEN valuenum END) AS lipase_min,
        MAX(CASE WHEN itemid=50956 THEN valuenum END) AS lipase_max
    FROM mimiciv_hosp.labevents le
    INNER JOIN cohort_base c ON le.hadm_id = c.hadm_id
    WHERE le.itemid IN (50889, 50971, 50893, 50808, 50867, 50956)
      AND le.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
      AND le.valuenum IS NOT NULL
    GROUP BY le.hadm_id
),
bg_agg AS (
    SELECT 
        bg.hadm_id,
        MIN(bg.lactate) AS lactate_min,
        MAX(bg.lactate) AS lactate_max,
        (MAX(bg.lactate) - MIN(bg.lactate)) / 
        NULLIF(EXTRACT(EPOCH FROM (MAX(bg.charttime) - MIN(bg.charttime)))/3600, 0) AS lactate_slope
    FROM mimiciv_derived.bg bg
    INNER JOIN cohort_base c ON bg.hadm_id = c.hadm_id
    WHERE bg.charttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
      AND bg.lactate IS NOT NULL
    GROUP BY bg.hadm_id
)
SELECT 
    c.hadm_id,
    -- 明确列出实验室字段，避免重复 hadm_id
    l.crp_min, l.crp_max, l.crp_slope,
    l.phosphate_min, l.phosphate_max,
    l.calcium_min, l.calcium_max,
    l.amylase_min, l.amylase_max,
    l.lipase_min, l.lipase_max,
    -- 血气字段
    b.lactate_min, b.lactate_max, b.lactate_slope,
    -- 比例计算 (注意：lab 表是按 stay_id 关联的，这里用 MAX 确保聚合一致性)
    CASE WHEN lab.albumin_min > 0 THEN b.lactate_max / lab.albumin_min ELSE NULL END AS lar,
    CASE WHEN lab.albumin_min > 0 THEN lab.bilirubin_total_max / lab.albumin_min ELSE NULL END AS tbar
FROM cohort_base c
LEFT JOIN lab_agg l ON c.hadm_id = l.hadm_id
LEFT JOIN bg_agg b ON c.hadm_id = b.hadm_id
LEFT JOIN mimiciv_derived.first_day_lab lab ON c.stay_id = lab.stay_id;

CREATE INDEX idx_lab_hadm ON temp_lab_slopes(hadm_id);

-- 5. Batch comorbidities
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
    MAX(CASE WHEN icd_code LIKE '14%' OR icd_code LIKE '15%' OR icd_code LIKE 'C%' THEN 1 ELSE 0 END) AS malignant_tumor,
    MAX(CASE WHEN icd_code LIKE '2860%' OR icd_code LIKE 'D66%' THEN 1 ELSE 0 END) AS congenital_coagulation_defects
FROM mimiciv_hosp.diagnoses_icd d
INNER JOIN cohort_base c ON d.hadm_id = c.hadm_id
GROUP BY c.hadm_id;

-- 6. Interventions (Note: Ensure mimiciv_derived tables exist)
DROP TABLE IF EXISTS temp_interventions;
CREATE TEMP TABLE temp_interventions AS
SELECT 
    c.stay_id,
    MAX(CASE WHEN vent.ventilation_status IN ('InvasiveVent', 'Tracheostomy') THEN 1 ELSE 0 END) AS mechanical_vent_flag,
    -- Simplified Vaso check: if any row exists in vasoactive_agent (if table exists)
    -- If this fails, user needs to join norepinephrine/dopamine tables separately
    MAX(CASE WHEN vaso.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS vaso_flag
FROM cohort_base c
LEFT JOIN mimiciv_derived.ventilation vent ON c.stay_id = vent.stay_id 
    AND vent.starttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
LEFT JOIN mimiciv_derived.vasoactive_agent vaso ON c.stay_id = vaso.stay_id 
    AND vaso.starttime BETWEEN c.intime AND c.intime + INTERVAL '24 hours'
GROUP BY c.stay_id;

-- 7. Final integration (FIXED: removed duplicate bmi)
DROP TABLE IF EXISTS my_custom_schema.ap_final_analysis_cohort;
CREATE TABLE my_custom_schema.ap_final_analysis_cohort AS
SELECT
    c.*, -- 这里已经包含了 stay_id, hadm_id, bmi, admission_age 等
    
    -- Outcomes
    CASE WHEN (COALESCE(p.resp_pof,0) + COALESCE(p.cv_pof,0) + COALESCE(p.renal_pof,0)) > 0 THEN 1 ELSE 0 END AS pof,
    COALESCE(p.resp_pof,0) as resp_pof, 
    COALESCE(p.cv_pof,0) as cv_pof, 
    COALESCE(p.renal_pof,0) as renal_pof,
    CASE WHEN COALESCE(c.deathtime, c.dod) <= (c.intime + INTERVAL '28 days') THEN 1 ELSE 0 END AS mortality_28d,
    
    -- Scores
    sofa.sofa AS sofa_score,
    apsiii.apsiii,
    sapsii.sapsii,
    oasis.oasis,
    lods.lods,
    
    -- Standard Labs (From verified first_day_lab)
    lab.wbc_min, lab.wbc_max,
    lab.hematocrit_min, lab.hematocrit_max,
    lab.hemoglobin_min, lab.hemoglobin_max,
    lab.platelets_min, lab.platelets_max,
    lab.albumin_min, lab.albumin_max,
    lab.aniongap_min, lab.aniongap_max,
    lab.bicarbonate_min, lab.bicarbonate_max,
    lab.bun_min, lab.bun_max,
    lab.calcium_min, lab.calcium_max,
    lab.chloride_min, lab.chloride_max,
    lab.creatinine_min, lab.creatinine_max,
    lab.glucose_min, lab.glucose_max,
    lab.sodium_min, lab.sodium_max,
    lab.potassium_min, lab.potassium_max,
    lab.bilirubin_total_min, lab.bilirubin_total_max,
    lab.inr_min, lab.inr_max,
    lab.pt_min, lab.pt_max,
    lab.ptt_min, lab.ptt_max,
    lab.amylase_min, lab.amylase_max,
    lab.alt_min, lab.alt_max,
    lab.ast_min, lab.ast_max,
    lab.alp_min, lab.alp_max,
    
    -- AP Specific & Slopes (From temp_lab_slopes)
    ls.crp_min, ls.crp_max, ls.crp_slope,
    ls.phosphate_min, ls.phosphate_max,
    ls.lipase_min, ls.lipase_max,
    ls.lactate_min, ls.lactate_max, ls.lactate_slope,
    
    -- BG & Ratios
    bg.ph_min, bg.ph_max,
    bg.pao2fio2ratio_min, bg.pao2fio2ratio_max,
    ls.lar, ls.tbar,
    -- BMI 已经包含在 c.* 中，此处删除重复引用
    
    -- Interventions & Comorbidities
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
