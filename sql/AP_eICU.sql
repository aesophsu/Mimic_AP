--------------------------------------------------------------------------------
-- 1. 识别 AP 患者
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_ap_patients;
CREATE TEMP TABLE temp_ap_patients AS
SELECT 
    patientunitstayid,
    MAX(CASE WHEN diagnosisstring ILIKE '%alcoholic%' THEN 1 ELSE 0 END) AS alcoholic_ap,
    MAX(CASE WHEN diagnosisstring ILIKE '%biliary%' OR diagnosisstring ILIKE '%gallstone%' THEN 1 ELSE 0 END) AS biliary_ap
FROM eicu_crd.diagnosis
WHERE diagnosisstring ILIKE '%pancreatit%'
GROUP BY patientunitstayid;

--------------------------------------------------------------------------------
-- 2. 构建核心队列 (18岁以上, ICU 停留 >= 24h)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS cohort_base;
CREATE TEMP TABLE cohort_base AS
SELECT 
    i.patientunitstayid,
    i.patienthealthsystemstayid,
    CAST(CASE WHEN i.age = '> 89' THEN '90' 
              WHEN i.age ~ '^[0-9]+$' THEN i.age 
              ELSE '0' END AS INT) AS age,
    i.gender,
    i.admissionheight AS height,
    i.admissionweight AS weight,
    i.icu_los_hours,
    i.hosp_mort,
    ap.alcoholic_ap,
    ap.biliary_ap,
    CASE WHEN i.admissionheight > 0 THEN (i.admissionweight / POWER(i.admissionheight / 100.0, 2)) ELSE NULL END AS bmi
FROM eicu_derived.icustay_detail i
INNER JOIN temp_ap_patients ap ON i.patientunitstayid = ap.patientunitstayid
WHERE i.icu_los_hours >= 24
  AND (CASE WHEN i.age = '> 89' THEN 90 
            WHEN i.age ~ '^[0-9]+$' THEN CAST(i.age AS INT) 
            ELSE 0 END) >= 18;

--------------------------------------------------------------------------------
-- 3. 深度打捞实验室指标 (时间窗: -6h to +24h)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_lab_raw_all;
CREATE TEMP TABLE temp_lab_raw_all AS
WITH lab_filt AS (
    SELECT patientunitstayid, labname, labresult,
           ROW_NUMBER() OVER (PARTITION BY patientunitstayid, labname ORDER BY labresultrevisedoffset DESC) as rn
    FROM eicu_crd.lab
    WHERE labresultoffset BETWEEN -360 AND 1440
      AND labresult IS NOT NULL
)
SELECT 
    patientunitstayid,
    MAX(CASE WHEN labname ILIKE '%BUN%' THEN labresult END) AS bun_max,
    MIN(CASE WHEN labname ILIKE '%BUN%' THEN labresult END) AS bun_min,
    MAX(CASE WHEN labname ILIKE '%creatinine%' THEN labresult END) AS creatinine_max,
    MIN(CASE WHEN labname ILIKE '%calcium%' THEN labresult END) AS lab_calcium_min,
    MIN(CASE WHEN labname ILIKE '%alkaline phos%' OR labname = 'ALP' THEN labresult END) AS alp_min,
    MAX(CASE WHEN labname ILIKE '%alkaline phos%' OR labname = 'ALP' THEN labresult END) AS alp_max,
    MAX(CASE WHEN labname ILIKE '%chloride%' THEN labresult END) AS chloride_max,
    MAX(CASE WHEN labname ILIKE '%WBC%' THEN labresult END) AS wbc_max,
    MIN(CASE WHEN labname ILIKE '%albumin%' THEN labresult END) AS albumin_min,
    MAX(CASE WHEN labname ILIKE '%albumin%' THEN labresult END) AS albumin_max,
    MAX(CASE WHEN labname ILIKE '%fibrinogen%' THEN labresult END) AS fibrinogen_max,
    MAX(CASE WHEN labname ILIKE '%lactate%' THEN labresult END) AS lactate_max,
    MAX(CASE WHEN labname ILIKE '%PTT%' THEN labresult END) AS ptt_max,
    MAX(CASE WHEN labname ILIKE '%anion gap%' THEN labresult END) AS aniongap_max,
    MAX(CASE WHEN labname ILIKE '%glucose%' THEN labresult END) AS glucose_max,
    MIN(CASE WHEN labname ILIKE '%hemoglobin%' OR labname = 'Hgb' THEN labresult END) AS hemoglobin_min,
    MIN(CASE WHEN labname ILIKE '%platelet%' THEN labresult END) AS platelet_min,
    MAX(CASE WHEN labname ILIKE '%sodium%' THEN labresult END) AS sodium_max,
    MAX(CASE WHEN labname ILIKE '%potassium%' THEN labresult END) AS potassium_max
FROM lab_filt
WHERE rn = 1
GROUP BY patientunitstayid;

--------------------------------------------------------------------------------
-- 4. 提取血气/生命体征/GCS/尿量 (-6h to +24h)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_bg;
CREATE TEMP TABLE temp_bg AS
SELECT patientunitstayid, MIN(ph) AS ph_min 
FROM eicu_derived.pivoted_bg WHERE chartoffset BETWEEN -360 AND 1440 GROUP BY patientunitstayid;

DROP TABLE IF EXISTS temp_vital_full;
CREATE TEMP TABLE temp_vital_full AS
SELECT 
    patientunitstayid, MIN(heartrate) AS heart_rate_min, MAX(heartrate) AS heart_rate_max,
    MIN(respiratoryrate) AS resp_rate_min, MAX(respiratoryrate) AS resp_rate_max,
    MIN(nibp_mean) AS mbp_min, MAX(nibp_mean) AS mbp_max,
    MIN(nibp_systolic) AS sbp_min, MAX(nibp_systolic) AS sbp_max,
    MIN(temperature) AS temp_min, MAX(temperature) AS temp_max, MAX(spo2) AS spo2_max
FROM eicu_derived.pivoted_vital WHERE chartoffset BETWEEN -360 AND 1440 GROUP BY patientunitstayid;

DROP TABLE IF EXISTS temp_gcs;
CREATE TEMP TABLE temp_gcs AS
SELECT patientunitstayid, MIN(gcs) AS gcs_min 
FROM eicu_derived.pivoted_gcs WHERE chartoffset BETWEEN -360 AND 1440 GROUP BY patientunitstayid;

DROP TABLE IF EXISTS temp_uo;
CREATE TEMP TABLE temp_uo AS
SELECT patientunitstayid, SUM(urineoutput) AS urineoutput 
FROM eicu_derived.pivoted_uo WHERE chartoffset BETWEEN -360 AND 1440 GROUP BY patientunitstayid;

--------------------------------------------------------------------------------
-- 5. 修正字段名后的干预打捞 (使用 treatmentstring)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_interventions;
CREATE TEMP TABLE temp_interventions AS
SELECT 
    patientunitstayid, 
    MAX(CASE WHEN treatmentstring ILIKE '%vasopressor%' OR treatmentstring ILIKE '%inotropes%' THEN 1 ELSE 0 END) AS vaso_flag,
    MAX(CASE WHEN treatmentstring ILIKE '%dialysis%' OR treatmentstring ILIKE '%CRRT%' THEN 1 ELSE 0 END) AS dialysis_flag
FROM eicu_crd.treatment
WHERE treatmentoffset BETWEEN -360 AND 1440 
GROUP BY patientunitstayid;

DROP TABLE IF EXISTS temp_apache_aps;
CREATE TEMP TABLE temp_apache_aps AS
SELECT patientunitstayid, ph, creatinine, bun, wbc, albumin, vent, dialysis
FROM eicu_crd.apacheapsvar;

--------------------------------------------------------------------------------
-- 6. 最终整合
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS eicu_cview.ap_external_validation;
CREATE TABLE eicu_cview.ap_external_validation AS
SELECT
    c.patientunitstayid, c.age, c.gender, c.height, c.weight, c.bmi,
    c.icu_los_hours, c.hosp_mort, c.alcoholic_ap, c.biliary_ap,

    COALESCE(bg.ph_min, aps.ph) AS ph_min,
    COALESCE(l.creatinine_max, aps.creatinine) AS creatinine_max,
    COALESCE(l.bun_max, aps.bun) AS bun_max,
    COALESCE(l.bun_min, l.bun_max, aps.bun) AS bun_min, 
    COALESCE(l.wbc_max, aps.wbc) AS wbc_max,
    COALESCE(l.albumin_min, aps.albumin) AS albumin_min,
    l.albumin_max, l.lab_calcium_min, l.alp_min, l.alp_max, l.chloride_max, l.aniongap_max, l.glucose_max,
    COALESCE(l.fibrinogen_max, 0) AS fibrinogen_max, l.ptt_max, l.lactate_max,
    l.hemoglobin_min, l.platelet_min, l.sodium_max, l.potassium_max,

    v.heart_rate_min, v.heart_rate_max, v.resp_rate_min, v.resp_rate_max,
    v.mbp_min, v.mbp_max, v.sbp_min, v.sbp_max, v.temp_min, v.temp_max, v.spo2_max,
    g.gcs_min, uo.urineoutput,
    
    COALESCE(intv.vaso_flag, 0) AS vaso_flag,
    COALESCE(aps.vent, 0) AS vent_flag,
    COALESCE(intv.dialysis_flag, aps.dialysis, 0) AS dialysis_flag,

    -- 优化后的 POF 精细化判定
    CASE 
        WHEN c.hosp_mort = 1 THEN 1 
        WHEN COALESCE(intv.vaso_flag, 0) = 1 THEN 1 
        WHEN COALESCE(l.creatinine_max, aps.creatinine) > 1.9 OR COALESCE(intv.dialysis_flag, aps.dialysis) = 1 THEN 1
        WHEN COALESCE(aps.vent, 0) = 1 AND c.icu_los_hours >= 48 THEN 1
        WHEN c.icu_los_hours >= 72 AND (COALESCE(l.lactate_max, 0) > 2.5 OR COALESCE(l.bun_max, 0) > 40) THEN 1
        ELSE 0 
    END AS pof_proxy

FROM cohort_base c
LEFT JOIN temp_lab_raw_all l ON c.patientunitstayid = l.patientunitstayid
LEFT JOIN temp_apache_aps aps ON c.patientunitstayid = aps.patientunitstayid
LEFT JOIN temp_bg bg ON c.patientunitstayid = bg.patientunitstayid
LEFT JOIN temp_vital_full v ON c.patientunitstayid = v.patientunitstayid
LEFT JOIN temp_gcs g ON c.patientunitstayid = g.patientunitstayid
LEFT JOIN temp_uo uo ON c.patientunitstayid = uo.patientunitstayid
LEFT JOIN temp_interventions intv ON c.patientunitstayid = intv.patientunitstayid;

-- 审计输出
SELECT 
    COUNT(*) as total_patients,
    ROUND(COUNT(ph_min)::numeric / COUNT(*), 4) * 100 as ph_fill,
    ROUND(COUNT(bun_min)::numeric / COUNT(*), 4) * 100 as bun_min_fill,
    ROUND(AVG(pof_proxy), 4) * 100 as pof_prevalence
FROM eicu_cview.ap_external_validation;
