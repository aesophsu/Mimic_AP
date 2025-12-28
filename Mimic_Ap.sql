--------------------------------------------------------------------------------
-- 1. First materialize AP diagnoses (ICD-9: 5770, ICD-10: K85%)
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
-- 2. Core cohort base (With Global Height Back-filling)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS cohort_base;
CREATE TEMP TABLE cohort_base AS
WITH height_global AS (
    -- 取该病人所有记录的中位数，显著提升 BMI 覆盖率
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
        pat.subject_id, pat.gender, pat.dod,
        pat.anchor_year, pat.anchor_age
    FROM mimiciv_hosp.patients pat
)
SELECT 
    ir.subject_id, ir.hadm_id, ir.stay_id, ir.intime, ir.los,
    adm.admittime, adm.dischtime, adm.deathtime,
    adm.insurance, adm.race, d.dod, d.gender,
    (EXTRACT(YEAR FROM ir.intime) - d.anchor_year + d.anchor_age) AS admission_age,
    ap.alcoholic_ap, ap.biliary_ap, ap.hyperlipidemic_ap, ap.drug_induced_ap,
    w.weight AS weight_admit,
    COALESCE(h.height, hg.height_global) AS height_admit,
    CASE WHEN COALESCE(h.height, hg.height_global) > 0 
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
-- 3. Optimized POF (Persistent Organ Failure)
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
    WHERE s.starttime >= c.intime + INTERVAL '24 hours'   -- ⭐关键
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
        (MAX(CASE WHEN itemid IN (50931, 50809, 52569) THEN valuenum END) - 
         MIN(CASE WHEN itemid IN (50931, 50809, 52569) THEN valuenum END)) / 
        NULLIF(EXTRACT(EPOCH FROM (MAX(CASE WHEN itemid IN (50931, 50809, 52569) THEN charttime END) - 
                                   MIN(CASE WHEN itemid IN (50931, 50809, 52569) THEN charttime END)))/3600, 0) AS glucose_slope,
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
        (MAX(bg.lactate) - MIN(bg.lactate)) / 
        NULLIF(EXTRACT(EPOCH FROM (MAX(bg.charttime) - MIN(bg.charttime)))/3600, 0) AS lactate_slope
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
        (MAX(CASE WHEN itemid=220277 THEN valuenum END) - MIN(CASE WHEN itemid=220277 THEN valuenum END)) / 
        NULLIF(EXTRACT(EPOCH FROM (MAX(CASE WHEN itemid=220277 THEN charttime END) - MIN(CASE WHEN itemid=220277 THEN charttime END)))/3600, 0) AS spo2_slope
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

    -- Outcomes
    CASE WHEN (COALESCE(p.resp_pof,0)
             + COALESCE(p.cv_pof,0)
             + COALESCE(p.renal_pof,0)) > 0
         THEN 1 ELSE 0 END AS pof,

    p.resp_pof,
    p.cv_pof,
    p.renal_pof,

    CASE
        WHEN COALESCE(c.deathtime, c.dod)
             <= (c.intime + INTERVAL '28 days')
        THEN 1 ELSE 0
    END AS mortality_28d,

    -- ❌ 不再包含 sofa_score

    -- Scores (optional, for comparator only)
    apsiii.apsiii,
    sapsii.sapsii,
    oasis.oasis,
    lods.lods,

    -- First 24h labs
    lab.*,

    -- AP-specific ML features
    ls.*,

    -- Interventions & Comorbidities
    COALESCE(intv.mechanical_vent_flag,0) AS mechanical_vent_flag,
    COALESCE(intv.vaso_flag,0)            AS vaso_flag,
    COALESCE(com.heart_failure,0)         AS heart_failure,
    COALESCE(com.chronic_kidney_disease,0)AS chronic_kidney_disease,
    COALESCE(com.malignant_tumor,0)        AS malignant_tumor

FROM cohort_base c
LEFT JOIN pof_results p      ON c.stay_id = p.stay_id
LEFT JOIN mimiciv_derived.apsiii apsiii ON c.stay_id = apsiii.stay_id
LEFT JOIN mimiciv_derived.sapsii sapsii ON c.stay_id = sapsii.stay_id
LEFT JOIN mimiciv_derived.oasis oasis   ON c.stay_id = oasis.stay_id
LEFT JOIN mimiciv_derived.lods lods     ON c.stay_id = lods.stay_id
LEFT JOIN mimiciv_derived.first_day_lab lab ON c.stay_id = lab.stay_id
LEFT JOIN temp_lab_slopes ls ON c.hadm_id = ls.hadm_id
LEFT JOIN temp_interventions intv ON c.stay_id = intv.stay_id
LEFT JOIN temp_comorbidities com  ON c.hadm_id = com.hadm_id;


-- Final Output
SELECT 'AP Cohort extraction complete. Rows created:' AS status, COUNT(*) FROM my_custom_schema.ap_final_analysis_cohort;
