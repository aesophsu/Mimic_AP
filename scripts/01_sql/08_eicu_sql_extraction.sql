--------------------------------------------------------------------------------
-- 1. 识别 AP 患者
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_ap_patients;
CREATE TEMP TABLE temp_ap_patients AS
SELECT patientunitstayid
FROM eicu_crd.diagnosis
WHERE diagnosisstring ILIKE '%pancreatit%'
  AND diagnosisstring NOT ILIKE '%chronic%'
GROUP BY patientunitstayid;

--------------------------------------------------------------------------------
-- 2. 构建核心队列 (18岁以上, ICU >= 24h)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS cohort_base;
CREATE TEMP TABLE cohort_base AS
WITH ranked_stays AS (
    SELECT i.*,
        ROW_NUMBER() OVER (
            PARTITION BY i.uniquepid 
            ORDER BY i.hospitaladmitoffset ASC, i.unitadmitoffset ASC
        ) as stay_rank
    FROM eicu_derived.icustay_detail i
    INNER JOIN temp_ap_patients ap ON i.patientunitstayid = ap.patientunitstayid
)
SELECT 
    r.patientunitstayid,
    r.uniquepid,
    CAST(CASE WHEN r.age = '> 89' THEN '90' 
              WHEN r.age ~ '^[0-9]+$' THEN r.age 
              ELSE '0' END AS INT) AS admission_age,
    r.gender,
    CASE WHEN r.admissionheight BETWEEN 120 AND 250 THEN r.admissionheight ELSE NULL END AS height_admit,
    CASE WHEN r.admissionweight BETWEEN 30 AND 300 THEN r.admissionweight ELSE NULL END AS weight_admit,
    r.icu_los_hours / 24.0 AS los,
    r.hosp_mort,
    CASE 
        WHEN (r.admissionheight BETWEEN 120 AND 250) AND (r.admissionweight BETWEEN 30 AND 300) 
        THEN (r.admissionweight / POWER(r.admissionheight / 100.0, 2)) 
        ELSE NULL 
    END AS bmi
FROM ranked_stays r
WHERE r.stay_rank = 1 
  AND r.icu_los_hours >= 24
  AND (CASE WHEN r.age = '> 89' THEN 90 
            WHEN r.age ~ '^[0-9]+$' THEN CAST(r.age AS INT) 
            ELSE 0 END) >= 18;

--------------------------------------------------------------------------------
-- 3. 深度打捞实验室指标 (补全 RDW, PT, INR, Hct, Bilirubin_Max)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_lab_raw_all;
CREATE TEMP TABLE temp_lab_raw_all AS
WITH lab_filt AS (
    SELECT 
        patientunitstayid, labname, labresult,
        CASE 
            -- 1. BUN 单位 mg/dL (保持原始)
            WHEN labname ILIKE '%BUN%' AND labresult BETWEEN 1 AND 200 
                THEN labresult
            
            -- 2. Creatinine 单位转换补丁 (umol/L -> mg/dL)
            WHEN labname ILIKE '%creatinine%' THEN 
                CASE 
                    WHEN labresult > 30 THEN labresult / 88.4 
                    WHEN labresult BETWEEN 0.1 AND 30 THEN labresult 
                    ELSE NULL 
                END

            -- 3. Hemoglobin & Hematocrit (Hgb: 4-25, Hct: 12-75)
            WHEN labname ILIKE ANY(ARRAY['%hemoglobin%', '%Hgb%', '%total hemoglobin%']) 
                AND labname NOT ILIKE '%A1c%' 
                AND labresult BETWEEN 4 AND 25 THEN labresult 
            WHEN labname ILIKE '%Hct%' AND labresult BETWEEN 12 AND 75 THEN labresult

            -- 4. pH 生理性过滤 (6.7-7.8)
            WHEN labname ILIKE ANY(ARRAY['%pH%', '%arterial pH%']) AND labname NOT ILIKE ANY(ARRAY['%urine%','%fluid%'])
                AND labresult BETWEEN 6.7 AND 7.8 AND labresult NOT IN (7.0, 8.0) THEN labresult
            
            -- 5. 凝血功能 (PTT: 10-150, PT: 5-150, INR: 0.5-20)
            WHEN labname ILIKE ANY(ARRAY['%PTT%', '%Partial Thromboplastin Time%', '%aPTT%']) 
                AND labresult BETWEEN 10 AND 150 THEN labresult
            WHEN labname ILIKE '%PT%' AND labname NOT ILIKE '%PTT%' AND labresult BETWEEN 5 AND 150 THEN labresult
            WHEN labname ILIKE '%INR%' AND labresult BETWEEN 0.5 AND 20 THEN labresult

            -- 6. 乳酸打捞 (0.1-30 mmol/L)
            WHEN labname ILIKE ANY(ARRAY['%lactate%', '%lactic acid%', '%lac%']) 
                AND labresult BETWEEN 0.1 AND 30 THEN labresult

            -- 7. RDW (10-35)
            WHEN labname ILIKE '%RDW%' AND labresult BETWEEN 10 AND 35 THEN labresult

            -- 8. 其他常规指标 (带单位兼容性转换)
            WHEN labname ILIKE '%paCO2%' AND labresult BETWEEN 5 AND 150 THEN labresult
            WHEN (labname ILIKE '%bicarb%' OR labname ILIKE '%HCO3%') AND labname NOT ILIKE '%total co2%'
                AND labresult BETWEEN 2 AND 60 THEN labresult
            WHEN labname ILIKE '%WBC%' AND labresult BETWEEN 0.1 AND 500 THEN labresult
            
            -- 8a. Albumin (兼容 g/L 转 g/dL)
            WHEN labname ILIKE '%albumin%' THEN 
                CASE 
                    WHEN labresult > 10 AND (labresult / 10.0) BETWEEN 1.0 AND 6.0 THEN labresult / 10.0
                    WHEN labresult BETWEEN 1.0 AND 6.0 THEN labresult 
                    ELSE NULL 
                END
            
            WHEN labname ILIKE '%calcium%' AND labname NOT ILIKE '%ion%' AND labresult BETWEEN 4 AND 15 THEN labresult
            WHEN labname ILIKE '%AST%' AND labresult BETWEEN 1 AND 2000 THEN labresult
            WHEN labname ILIKE '%ALT%' AND labresult BETWEEN 1 AND 2000 THEN labresult
            WHEN labname ILIKE '%platelet%' AND labresult BETWEEN 1 AND 1000 THEN labresult
            WHEN labname ILIKE '%anion gap%' AND labresult BETWEEN 2 AND 50 THEN labresult
            
            -- 8b. Total Bilirubin (兼容 umol/L 转 mg/dL)
            WHEN labname ILIKE '%total bilirubin%' THEN 
                CASE 
                    WHEN labresult > 10 AND (labresult / 17.1) BETWEEN 0.1 AND 70 THEN labresult / 17.1
                    WHEN labresult BETWEEN 0.1 AND 70 THEN labresult 
                    ELSE NULL 
                END
                
            WHEN labname ILIKE '%glucose%' AND labresult BETWEEN 10 AND 2000 THEN labresult
            WHEN labname ILIKE '%alkaline phos%' AND labresult BETWEEN 5 AND 2500 THEN labresult
            ELSE NULL 
        END AS labresult_clean
    FROM eicu_crd.lab
    WHERE labresultoffset BETWEEN -1440 AND 1440 
      AND labresult IS NOT NULL
      AND patientunitstayid IN (SELECT patientunitstayid FROM cohort_base)
)
SELECT 
    patientunitstayid,
    -- 基础血气打捞源
    MIN(CASE WHEN labname ILIKE '%pH%' THEN labresult_clean END) AS lab_ph_direct,
    MAX(CASE WHEN labname ILIKE '%pH%' THEN labresult_clean END) AS lab_ph_max,
    AVG(CASE WHEN labname ILIKE '%paCO2%' THEN labresult_clean END) AS lab_paco2,
    AVG(CASE WHEN labname ILIKE '%bicarb%' OR labname ILIKE '%HCO3%' THEN labresult_clean END) AS lab_hco3,
    MIN(CASE WHEN labname ILIKE '%bicarb%' OR labname ILIKE '%HCO3%' THEN labresult_clean END) AS bicarbonate_min,
    MAX(CASE WHEN labname ILIKE '%bicarb%' OR labname ILIKE '%HCO3%' THEN labresult_clean END) AS bicarbonate_max,

    -- 肾功
    MAX(CASE WHEN labname ILIKE '%creatinine%' THEN labresult_clean END) AS creatinine_max,
    MIN(CASE WHEN labname ILIKE '%creatinine%' THEN labresult_clean END) AS creatinine_min,
    MAX(CASE WHEN labname ILIKE '%BUN%' THEN labresult_clean END) AS bun_max,
    MIN(CASE WHEN labname ILIKE '%BUN%' THEN labresult_clean END) AS bun_min,

    -- 血常规
    MAX(CASE WHEN labname ILIKE '%WBC%' THEN labresult_clean END) AS wbc_max,
    MIN(CASE WHEN labname ILIKE '%WBC%' THEN labresult_clean END) AS wbc_min,
    MAX(CASE WHEN labname ILIKE '%RDW%' THEN labresult_clean END) AS rdw_max,
    MAX(CASE WHEN labname ILIKE '%Hct%' THEN labresult_clean END) AS hematocrit_max,
    MIN(CASE WHEN labname ILIKE '%Hct%' THEN labresult_clean END) AS hematocrit_min,
    MIN(CASE WHEN labname ILIKE ANY(ARRAY['%hemoglobin%', '%Hgb%']) THEN labresult_clean END) AS hemoglobin_min,
    MAX(CASE WHEN labname ILIKE ANY(ARRAY['%hemoglobin%', '%Hgb%']) THEN labresult_clean END) AS hemoglobin_max,
    MIN(CASE WHEN labname ILIKE '%platelet%' THEN labresult_clean END) AS platelets_min,
    MAX(CASE WHEN labname ILIKE '%platelet%' THEN labresult_clean END) AS platelets_max,

    -- 肝功
    MIN(CASE WHEN labname ILIKE '%albumin%' THEN labresult_clean END) AS albumin_min,
    MAX(CASE WHEN labname ILIKE '%albumin%' THEN labresult_clean END) AS albumin_max,
    MAX(CASE WHEN labname ILIKE '%total bilirubin%' THEN labresult_clean END) AS bilirubin_total_max,
    MIN(CASE WHEN labname ILIKE '%total bilirubin%' THEN labresult_clean END) AS bilirubin_total_min,
    MAX(CASE WHEN labname ILIKE '%AST%' THEN labresult_clean END) AS ast_max,
    MAX(CASE WHEN labname ILIKE '%ALT%' THEN labresult_clean END) AS alt_max,
    MIN(CASE WHEN labname ILIKE '%alkaline phos%' THEN labresult_clean END) AS alp_min,
    MAX(CASE WHEN labname ILIKE '%alkaline phos%' THEN labresult_clean END) AS alp_max,

    -- 凝血
    MIN(CASE WHEN labname ILIKE ANY(ARRAY['%PTT%', '%aPTT%']) THEN labresult_clean END) AS ptt_min,
    MAX(CASE WHEN labname ILIKE ANY(ARRAY['%PTT%', '%aPTT%']) THEN labresult_clean END) AS ptt_max,
    MAX(CASE WHEN labname ILIKE '%PT%' AND labname NOT ILIKE '%PTT%' THEN labresult_clean END) AS pt_max,
    MIN(CASE WHEN labname ILIKE '%PT%' AND labname NOT ILIKE '%PTT%' THEN labresult_clean END) AS pt_min,
    MAX(CASE WHEN labname ILIKE '%INR%' THEN labresult_clean END) AS inr_max,
    MIN(CASE WHEN labname ILIKE '%INR%' THEN labresult_clean END) AS inr_min,

    -- 血糖与电解质
    MAX(CASE WHEN labname ILIKE '%glucose%' THEN labresult_clean END) AS glucose_max,
    MIN(CASE WHEN labname ILIKE '%glucose%' THEN labresult_clean END) AS glucose_min,
    MAX(CASE WHEN labname ILIKE '%lactate%' THEN labresult_clean END) AS lactate_max,
    MAX(CASE WHEN labname ILIKE '%anion gap%' THEN labresult_clean END) AS aniongap_max,
    MIN(CASE WHEN labname ILIKE '%anion gap%' THEN labresult_clean END) AS aniongap_min
FROM lab_filt
GROUP BY patientunitstayid;

--------------------------------------------------------------------------------
-- 4. 血气增强打捞
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_bg;
CREATE TEMP TABLE temp_bg AS
SELECT patientunitstayid, 
    MIN(CASE WHEN ph BETWEEN 6.7 AND 7.8 AND ph NOT IN (7, 8) THEN ph END) AS bg_ph_min,
    MAX(CASE WHEN ph BETWEEN 6.7 AND 7.8 AND ph NOT IN (7, 8) THEN ph END) AS bg_ph_max,
    AVG(CASE WHEN paco2 BETWEEN 5 AND 150 THEN paco2 END) AS bg_paco2
FROM eicu_derived.pivoted_bg WHERE chartoffset BETWEEN -360 AND 1440 GROUP BY patientunitstayid;


DROP TABLE IF EXISTS temp_respiratory_support;
CREATE TEMP TABLE temp_respiratory_support AS
WITH oxy_data AS (
    SELECT 
        patientunitstayid,
        pao2 / (CASE WHEN fio2 IS NULL THEN 0.21 WHEN fio2 >= 21 THEN fio2/100.0 ELSE 0.21 END) AS pf_val
    FROM eicu_derived.pivoted_bg 
    WHERE pao2 > 0 AND chartoffset BETWEEN -360 AND 1440
),
vital_oxy AS (
    SELECT 
        v.patientunitstayid,
        v.spo2 / (CASE WHEN b.fio2 IS NULL THEN 0.21 WHEN b.fio2 >= 21 THEN b.fio2/100.0 ELSE 0.21 END) AS sf_val
    FROM eicu_derived.pivoted_vital v
    LEFT JOIN eicu_derived.pivoted_bg b ON v.patientunitstayid = b.patientunitstayid AND v.chartoffset = b.chartoffset
    WHERE v.spo2 BETWEEN 50 AND 100 AND v.chartoffset BETWEEN -360 AND 1440
)
SELECT 
    p.patientunitstayid,
    -- 核心：直接将融合结果命名为原来的特征名 pao2fio2ratio_min
    COALESCE(MIN(o.pf_val), MIN(v.sf_val) * 0.9) AS pao2fio2ratio_min
FROM (SELECT DISTINCT patientunitstayid FROM cohort_base) p
LEFT JOIN oxy_data o ON p.patientunitstayid = o.patientunitstayid
LEFT JOIN vital_oxy v ON p.patientunitstayid = v.patientunitstayid
GROUP BY p.patientunitstayid;
--------------------------------------------------------------------------------
-- 补充：计算 SpO2 斜率 (前24小时趋势)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_spo2_trend;
CREATE TEMP TABLE temp_spo2_trend AS
WITH spo2_first_last AS (
    SELECT 
        patientunitstayid,
        FIRST_VALUE(spo2) OVER (PARTITION BY patientunitstayid ORDER BY chartoffset ASC) as spo2_first,
        FIRST_VALUE(spo2) OVER (PARTITION BY patientunitstayid ORDER BY chartoffset DESC) as spo2_last,
        (MAX(chartoffset) OVER (PARTITION BY patientunitstayid) - 
         MIN(chartoffset) OVER (PARTITION BY patientunitstayid)) / 60.0 as time_span_hr
    FROM eicu_derived.pivoted_vital
    WHERE chartoffset BETWEEN -360 AND 1440 
      AND spo2 BETWEEN 50 AND 100
)
SELECT 
    patientunitstayid,
    CASE 
        WHEN time_span_hr > 0 THEN (spo2_last - spo2_first) / time_span_hr 
        ELSE 0 
    END AS spo2_slope
FROM spo2_first_last
GROUP BY patientunitstayid, spo2_first, spo2_last, time_span_hr;

DROP TABLE IF EXISTS temp_vital_full;
CREATE TEMP TABLE temp_vital_full AS
SELECT patientunitstayid, 
    MAX(NULLIF(heartrate, -1)) AS heart_rate_max, MIN(NULLIF(heartrate, -1)) AS heart_rate_min,
    MAX(NULLIF(respiratoryrate, -1)) AS resp_rate_max, MIN(NULLIF(respiratoryrate, -1)) AS resp_rate_min,
    MIN(NULLIF(nibp_mean, -1)) AS mbp_min, MAX(NULLIF(spo2, -1)) AS spo2_max, MIN(NULLIF(spo2, -1)) AS spo2_min,
    MAX(CASE WHEN temperature BETWEEN 80 AND 115 THEN (temperature-32)*5/9 WHEN temperature BETWEEN 30 AND 45 THEN temperature END) AS temp_max,
    MIN(CASE WHEN temperature BETWEEN 80 AND 115 THEN (temperature-32)*5/9 WHEN temperature BETWEEN 30 AND 45 THEN temperature END) AS temp_min
FROM eicu_derived.pivoted_vital WHERE chartoffset BETWEEN -360 AND 1440 GROUP BY patientunitstayid;

DROP TABLE IF EXISTS temp_comorbidity;
CREATE TEMP TABLE temp_comorbidity AS
SELECT patientunitstayid, MAX(CASE WHEN diagnosisstring ILIKE ANY(ARRAY['%malignant%','%cancer%','%metastas%']) THEN 1 ELSE 0 END) AS malignant_tumor
FROM eicu_crd.diagnosis GROUP BY patientunitstayid;

--------------------------------------------------------------------------------
-- 6. POF 定义相关 (Interventions, CarePlan, APACHE, Early Death)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_interventions;
CREATE TEMP TABLE temp_interventions AS
SELECT patientunitstayid, 
    MAX(CASE WHEN treatmentstring ILIKE ANY(ARRAY['%vasopressor%','%dopamine%','%norepinephrine%','%vasopressin%','%epinephrine%']) THEN 1 ELSE 0 END) AS vaso_flag,
    MAX(CASE WHEN treatmentstring ILIKE ANY(ARRAY['%dialysis%','%CRRT%','%hemofiltration%']) THEN 1 ELSE 0 END) AS dialysis_flag,
    MAX(CASE WHEN treatmentstring ILIKE ANY(ARRAY['%ventilation%','%intubat%']) THEN 1 ELSE 0 END) AS vent_treatment_flag
FROM eicu_crd.treatment WHERE treatmentoffset BETWEEN 1440 AND 10080 GROUP BY patientunitstayid;

DROP TABLE IF EXISTS temp_vent_careplan;
CREATE TEMP TABLE temp_vent_careplan AS
SELECT patientunitstayid, 1 AS vent_careplan_flag FROM eicu_crd.careplangeneral
WHERE cplitemoffset BETWEEN 1440 AND 10080 AND cplitemvalue ILIKE ANY(ARRAY['%ventilat%','%intubat%','%ET tube%']) GROUP BY patientunitstayid;

DROP TABLE IF EXISTS temp_apache_aps;
CREATE TEMP TABLE temp_apache_aps AS
SELECT patientunitstayid, 
    CASE WHEN ph BETWEEN 6.8 AND 7.8 AND ph NOT IN (7, 8) THEN ph END as apache_ph, 
    CASE WHEN creatinine BETWEEN 0.1 AND 20 THEN creatinine END as apache_creatinine, 
    vent AS apache_vent_flag, dialysis AS apache_dialysis_flag
FROM eicu_crd.apacheapsvar;

DROP TABLE IF EXISTS temp_early_death;
CREATE TEMP TABLE temp_early_death AS
SELECT patientunitstayid, 1 AS early_death_24_48h FROM eicu_crd.patient 
WHERE unitdischargestatus = 'Expired' AND unitdischargeoffset BETWEEN 1440 AND 2880;

--------------------------------------------------------------------------------
-- 7 最终整合与结局逻辑判定 (修正语法顺序)
--------------------------------------------------------------------------------
-- 第一步：先删除已存在的表
DROP TABLE IF EXISTS eicu_cview.ap_external_validation;
-- 第二步：开始创建表
CREATE TABLE eicu_cview.ap_external_validation AS
WITH tier_calc AS (
    SELECT
        c.patientunitstayid, 
        c.los,
        c.uniquepid, 
        c.admission_age, 
        c.gender, 
        c.bmi,
        c.weight_admit,
        c.height_admit, 
        
        -- 实验室指标
        l.creatinine_max, l.creatinine_min, l.bun_max, l.bun_min,
        l.wbc_max, l.wbc_min, l.rdw_max, l.hematocrit_max, l.hematocrit_min,
        l.hemoglobin_min, l.hemoglobin_max, l.platelets_min, l.platelets_max,
        l.albumin_min, l.albumin_max, l.bilirubin_total_max, l.bilirubin_total_min,
        l.ast_max, l.alt_max, l.alp_min, l.alp_max,
        l.ptt_min, l.ptt_max, l.pt_max, l.pt_min, l.inr_max, l.inr_min,
        l.glucose_max, l.glucose_min, l.lactate_max, l.aniongap_max, l.aniongap_min,
        l.bicarbonate_min, l.bicarbonate_max,
        slp.spo2_slope,
        
        -- pH 打捞链
        bg.bg_ph_min AS ph_t1, 
        l.lab_ph_direct AS ph_t2, 
        CASE 
            WHEN l.lab_hco3 > 0 AND COALESCE(bg.bg_paco2, l.lab_paco2) > 0 
            THEN (6.1 + LOG10(l.lab_hco3 / (0.0301 * COALESCE(bg.bg_paco2, l.lab_paco2)))) 
            ELSE NULL 
        END AS ph_t3, 
        aps.apache_ph AS ph_t4, 
        COALESCE(bg.bg_ph_max, l.lab_ph_max, bg.bg_ph_min, l.lab_ph_direct) AS ph_max_raw,

        -- TBAR 稳定性保护
        (l.bilirubin_total_max / NULLIF(GREATEST(l.albumin_min, 1.0), 0)) AS tbar,

        -- 呼吸指标增强版 (S/F 补全后的指标)
        res.pao2fio2ratio_min,
        
        COALESCE(comb.malignant_tumor, 0) AS malignant_tumor,
        v.heart_rate_max, v.heart_rate_min, v.resp_rate_max, v.resp_rate_min,
        v.mbp_min, v.spo2_max, v.spo2_min, v.temp_max, v.temp_min,
        
        COALESCE(intv.vaso_flag, 0) AS vaso_flag, 
        COALESCE(aps.apache_vent_flag, cp.vent_careplan_flag, intv.vent_treatment_flag, 0) AS mechanical_vent_flag, 
        COALESCE(intv.dialysis_flag, aps.apache_dialysis_flag, 0) AS dialysis_flag,
        
        CASE WHEN c.hosp_mort = 1 AND pat.hospitaldischargeoffset <= 40320 THEN 1 ELSE 0 END AS mortality_28d
    FROM cohort_base c
    INNER JOIN eicu_crd.patient pat ON c.patientunitstayid = pat.patientunitstayid
    LEFT JOIN temp_lab_raw_all l ON c.patientunitstayid = l.patientunitstayid
    LEFT JOIN temp_spo2_trend slp ON c.patientunitstayid = slp.patientunitstayid
    LEFT JOIN temp_apache_aps aps ON c.patientunitstayid = aps.patientunitstayid
    LEFT JOIN temp_bg bg ON c.patientunitstayid = bg.patientunitstayid
    LEFT JOIN temp_vital_full v ON c.patientunitstayid = v.patientunitstayid
    LEFT JOIN temp_interventions intv ON c.patientunitstayid = intv.patientunitstayid
    LEFT JOIN temp_vent_careplan cp ON c.patientunitstayid = cp.patientunitstayid
    LEFT JOIN temp_respiratory_support res ON c.patientunitstayid = res.patientunitstayid
    LEFT JOIN temp_comorbidity comb ON c.patientunitstayid = comb.patientunitstayid
),
base_final AS (
    SELECT *,
        CASE 
            WHEN ph_t1 BETWEEN 6.7 AND 7.8 THEN ph_t1
            WHEN ph_t2 BETWEEN 6.7 AND 7.8 THEN ph_t2
            WHEN ph_t3 BETWEEN 6.7 AND 7.8 THEN CAST(ph_t3 AS NUMERIC)
            WHEN ph_t4 BETWEEN 6.7 AND 7.8 THEN ph_t4
            ELSE NULL 
        END AS ph_min,
        CASE 
            WHEN ph_max_raw BETWEEN 6.7 AND 7.8 THEN ph_max_raw 
            ELSE NULL 
        END AS ph_max
    FROM tier_calc
),
label_calc AS (
    -- 在这一层统一计算结局标签，方便后续 WHERE 过滤
    SELECT *,
        CASE  
            WHEN (vaso_flag = 1 AND los >= 1) 
              OR (mechanical_vent_flag = 1 AND los >= 2) 
              OR (pao2fio2ratio_min < 300 AND los >= 2) 
              OR (dialysis_flag = 1) 
              OR (creatinine_max > 1.9) THEN 1
            WHEN creatinine_max IS NULL AND pao2fio2ratio_min IS NULL THEN NULL
            ELSE 0 
        END AS pof,
        
        CASE 
            WHEN (vaso_flag = 1 AND los >= 1) 
              OR (mechanical_vent_flag = 1 AND los >= 2)
              OR (pao2fio2ratio_min < 300 AND los >= 2) 
              OR (dialysis_flag = 1)
              OR (creatinine_max > 1.9) 
              OR (mortality_28d = 1) THEN 1
            WHEN creatinine_max IS NULL AND pao2fio2ratio_min IS NULL AND mortality_28d IS NULL THEN NULL
            ELSE 0
        END AS composite_outcome
    FROM base_final
)
-- 最终选择：排除 pof 为 NULL 的那 1 例
SELECT * FROM label_calc 
WHERE pof IS NOT NULL;

--------------------------------------------------------------------------------
-- 最终统计信息 (加强版：覆盖率 + 生理分布审计)
--------------------------------------------------------------------------------
SELECT 
    -- 1. 总体样本与结局分布
    COUNT(*) AS total_pts,
    SUM(CASE WHEN pof IS NOT NULL THEN 1 ELSE 0 END) AS pts_with_pof_label,
    SUM(pof) AS pof_cases,
    ROUND(AVG(pof)*100, 2) AS pof_rate_pct,
    SUM(mortality_28d) AS deaths_28d,
    ROUND(AVG(mortality_28d)*100, 2) AS mort_rate_pct,
    SUM(composite_outcome) AS composite_events,
    -- 审计结局缺失情况
    ROUND(SUM(CASE WHEN pof IS NULL THEN 1 ELSE 0 END)::numeric / COUNT(*) * 100, 2) AS pof_null_rate,

    -- 2. 核心特征覆盖率审计 (对应模型 selected_features)
    ROUND(COUNT(ph_min)::numeric / COUNT(*) * 100, 2) AS ph_coverage,
    ROUND(COUNT(pao2fio2ratio_min)::numeric / COUNT(*) * 100, 2) AS pf_coverage,
    ROUND(COUNT(creatinine_max)::numeric / COUNT(*) * 100, 2) AS scr_coverage,
    ROUND(COUNT(inr_max)::numeric / COUNT(*) * 100, 2) AS inr_coverage,
    ROUND(COUNT(lactate_max)::numeric / COUNT(*) * 100, 2) AS lac_coverage,
    ROUND(COUNT(alp_min)::numeric / COUNT(*) * 100, 2) AS alp_coverage,
    ROUND(COUNT(tbar)::numeric / COUNT(*) * 100, 2) AS tbar_coverage,
    ROUND(COUNT(rdw_max)::numeric / COUNT(*) * 100, 2) AS rdw_coverage,
    ROUND(COUNT(ptt_min)::numeric / COUNT(*) * 100, 2) AS ptt_coverage,

    -- 3. 生理值分布审计 (单位与量级校验)
    -- Hgb 应在 8-14 之间
    ROUND(AVG(hemoglobin_min)::numeric, 2) AS avg_hgb, 
    -- BUN 转换后应在 15-40 左右
    ROUND(AVG(bun_max)::numeric, 2) AS avg_bun,
    -- Creatinine 转换后应在 0.5-3.0 左右 (若 >50 说明 umol/L 转换失败)
    ROUND(AVG(creatinine_max)::numeric, 2) AS avg_scr,
    -- Bilirubin 应在 0.5-2.0 左右 (若 >100 说明单位有问题)
    ROUND(AVG(bilirubin_total_min)::numeric, 2) AS avg_bili_min,
    -- pH 生理极限检查
    MIN(ph_min) AS ph_min_limit,
    MAX(ph_max) AS ph_max_limit,
    -- 乳酸中值 (重症 AP 患者通常在 1.5-3.0)
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lactate_max) AS median_lactate,
    -- TBAR 审计 (正常值通常较小)
    ROUND(AVG(tbar)::numeric, 4) AS avg_tbar

FROM eicu_cview.ap_external_validation;
