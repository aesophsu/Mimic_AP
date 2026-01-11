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
-- 2. 构建核心队列 (18岁以上, ICU >= 24h)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS cohort_base;
CREATE TEMP TABLE cohort_base AS
SELECT 
    i.patientunitstayid,
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
    CASE WHEN i.admissionheight > 100 AND i.admissionweight > 30 
         THEN (i.admissionweight / POWER(i.admissionheight / 100.0, 2)) ELSE NULL END AS bmi
FROM eicu_derived.icustay_detail i
INNER JOIN temp_ap_patients ap ON i.patientunitstayid = ap.patientunitstayid
WHERE i.icu_los_hours >= 24
  AND (CASE WHEN i.age = '> 89' THEN 90 
            WHEN i.age ~ '^[0-9]+$' THEN CAST(i.age AS INT) 
            ELSE 0 END) >= 18;

--------------------------------------------------------------------------------
-- 3. 深度打捞实验室指标 (含 pH 占位符深度剔除)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_lab_raw_all;
CREATE TEMP TABLE temp_lab_raw_all AS
WITH lab_filt AS (
    SELECT 
        patientunitstayid, 
        labname,
        CASE 
            -- pH 深度清洗逻辑：
            -- 1. 范围限定在生理存活极限 6.8 - 7.8
            -- 2. 剔除刚好等于 7.0, 8.0, 7, 8 的整数占位符/无效值
            WHEN labname ILIKE '%pH%' 
                 AND labresult > 6.8 AND labresult < 7.8 
                 AND labresult NOT IN (7.0, 8.0, 7, 8) THEN labresult
            
            -- 其他核心指标阈值过滤
            WHEN labname ILIKE '%creatinine%' AND labresult BETWEEN 0.1 AND 20 THEN labresult
            WHEN labname ILIKE '%creatinine%' AND labresult > 20 THEN labresult / 88.4 
            WHEN labname ILIKE '%BUN%' AND labresult BETWEEN 1 AND 200 THEN labresult
            WHEN labname ILIKE '%WBC%' AND labresult BETWEEN 0.1 AND 500 THEN labresult
            WHEN labname ILIKE '%albumin%' AND labresult BETWEEN 1.0 AND 6.0 THEN labresult
            WHEN labname ILIKE '%lactate%' AND labresult BETWEEN 0.1 AND 30 THEN labresult
            ELSE NULL 
        END AS labresult_clean
    FROM eicu_crd.lab
    WHERE labresultoffset BETWEEN -360 AND 1440 
      AND labresult IS NOT NULL
      AND patientunitstayid IN (SELECT patientunitstayid FROM cohort_base)
)
SELECT 
    patientunitstayid,
    MIN(CASE WHEN labname ILIKE '%pH%' THEN labresult_clean END) AS lab_ph_min,
    MAX(CASE WHEN labname ILIKE '%creatinine%' THEN labresult_clean END) AS creatinine_max,
    MAX(CASE WHEN labname ILIKE '%BUN%' THEN labresult_clean END) AS bun_max,
    MAX(CASE WHEN labname ILIKE '%WBC%' THEN labresult_clean END) AS wbc_max,
    MIN(CASE WHEN labname ILIKE '%albumin%' THEN labresult_clean END) AS albumin_min,
    MAX(CASE WHEN labname ILIKE '%lactate%' THEN labresult_clean END) AS lactate_max
FROM lab_filt
GROUP BY patientunitstayid;

--------------------------------------------------------------------------------
-- 4. 提取辅助指标 (血气表 pH 同步清洗)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_bg;
CREATE TEMP TABLE temp_bg AS
SELECT 
    patientunitstayid, 
    MIN(CASE 
            WHEN ph > 6.8 AND ph < 7.8 
            AND ph NOT IN (7.0, 8.0, 7, 8) 
            THEN ph ELSE NULL 
        END) AS ph_min 
FROM eicu_derived.pivoted_bg 
WHERE chartoffset BETWEEN -360 AND 1440 
  AND patientunitstayid IN (SELECT patientunitstayid FROM cohort_base)
GROUP BY patientunitstayid;

DROP TABLE IF EXISTS temp_vital_full;
CREATE TEMP TABLE temp_vital_full AS
SELECT 
    patientunitstayid, 
    MAX(NULLIF(heartrate, -1)) AS heart_rate_max,
    MAX(NULLIF(respiratoryrate, -1)) AS resp_rate_max,
    MIN(NULLIF(nibp_mean, -1)) AS mbp_min, 
    MAX(NULLIF(spo2, -1)) AS spo2_max
FROM eicu_derived.pivoted_vital 
WHERE chartoffset BETWEEN -360 AND 1440 
  AND patientunitstayid IN (SELECT patientunitstayid FROM cohort_base)
GROUP BY patientunitstayid;

--------------------------------------------------------------------------------
-- 5. 干预与评分系统打捞 (APACHE 评分 pH 清洗)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_interventions;
CREATE TEMP TABLE temp_interventions AS
SELECT 
    patientunitstayid, 
    MAX(CASE WHEN treatmentstring ILIKE '%vasopressor%' OR treatmentstring ILIKE '%inotropes%' THEN 1 ELSE 0 END) AS vaso_flag,
    MAX(CASE WHEN treatmentstring ILIKE '%dialysis%' OR treatmentstring ILIKE '%CRRT%' THEN 1 ELSE 0 END) AS dialysis_flag
FROM eicu_crd.treatment
WHERE treatmentoffset BETWEEN -360 AND 1440 
  AND patientunitstayid IN (SELECT patientunitstayid FROM cohort_base)
GROUP BY patientunitstayid;

DROP TABLE IF EXISTS temp_apache_aps;
CREATE TEMP TABLE temp_apache_aps AS
SELECT 
    patientunitstayid, 
    CASE WHEN ph > 6.8 AND ph < 7.8 AND ph NOT IN (7.0, 8.0) THEN ph ELSE NULL END as ph, 
    CASE WHEN creatinine BETWEEN 0.1 AND 20 THEN creatinine ELSE NULL END as creatinine, 
    vent, dialysis
FROM eicu_crd.apacheapsvar
WHERE patientunitstayid IN (SELECT patientunitstayid FROM cohort_base);

--------------------------------------------------------------------------------
-- 6. 最终整合：级联打捞逻辑 (COALESCE)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS eicu_cview.ap_external_validation;
CREATE TABLE eicu_cview.ap_external_validation AS
SELECT
    c.*,
    -- 级联补全 pH_min (血气 > 化验室 > APACHE)
    COALESCE(bg.ph_min, l.lab_ph_min, aps.ph) AS ph_min,
    
    COALESCE(l.creatinine_max, aps.creatinine) AS creatinine_max,
    COALESCE(l.bun_max, aps.creatinine) AS bun_max, -- 某些情况下可用Cr趋势辅助判定
    l.wbc_max, l.albumin_min, l.lactate_max,
    v.heart_rate_max, v.resp_rate_max, v.mbp_min, v.spo2_max,
    
    COALESCE(intv.vaso_flag, 0) AS vaso_flag,
    COALESCE(aps.vent, 0) AS vent_flag,
    COALESCE(intv.dialysis_flag, aps.dialysis, 0) AS dialysis_flag,

    -- 稳健的 POF 判定
    CASE 
        WHEN c.hosp_mort = 1 THEN 1 
        WHEN COALESCE(intv.vaso_flag, 0) = 1 THEN 1 
        WHEN (COALESCE(l.creatinine_max, aps.creatinine) > 1.9) OR (COALESCE(intv.dialysis_flag, aps.dialysis, 0) = 1) THEN 1
        WHEN COALESCE(aps.vent, 0) = 1 AND c.icu_los_hours >= 48 THEN 1
        ELSE 0 
    END AS pof

FROM cohort_base c
LEFT JOIN temp_lab_raw_all l ON c.patientunitstayid = l.patientunitstayid
LEFT JOIN temp_apache_aps aps ON c.patientunitstayid = aps.patientunitstayid
LEFT JOIN temp_bg bg ON c.patientunitstayid = bg.patientunitstayid
LEFT JOIN temp_vital_full v ON c.patientunitstayid = v.patientunitstayid
LEFT JOIN temp_interventions intv ON c.patientunitstayid = intv.patientunitstayid;

--------------------------------------------------------------------------------
-- 审计输出 (用于检查清洗效果)
--------------------------------------------------------------------------------
SELECT 
    COUNT(*) as total_patients,
    ROUND(AVG(pof)::numeric, 3) as pof_prevalence,
    ROUND(MIN(ph_min)::numeric, 3) as min_ph_check,
    ROUND(MAX(ph_min)::numeric, 3) as max_ph_check,
    ROUND(AVG(ph_min)::numeric, 3) as avg_ph_check,
    ROUND(COUNT(ph_min)::numeric / COUNT(*), 4) * 100 as ph_fill_rate
FROM eicu_cview.ap_external_validation;
