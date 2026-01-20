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
              ELSE '0' END AS INT) AS age,
    r.gender,
    CASE WHEN r.admissionheight BETWEEN 120 AND 250 THEN r.admissionheight ELSE NULL END AS height,
    CASE WHEN r.admissionweight BETWEEN 30 AND 300 THEN r.admissionweight ELSE NULL END AS weight,
    r.icu_los_hours,
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
-- 3. 深度打捞实验室指标 (更新：修正 Hgb, 增强 Lactate/PTT/pH)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_lab_raw_all;
CREATE TEMP TABLE temp_lab_raw_all AS
WITH lab_filt AS (
    SELECT 
        patientunitstayid, labname, labresult,
        CASE 
            -- 1. BUN 单位转换补丁 (mg/dL -> mmol/L)
            WHEN labname ILIKE '%BUN%' AND labresult BETWEEN 1 AND 200 
                 THEN labresult / 2.801
            
            -- 2. Creatinine 单位转换补丁 (umol/L -> mg/dL)
            WHEN labname ILIKE '%creatinine%' THEN 
                CASE WHEN labresult > 30 THEN labresult / 88.4 
                     WHEN labresult BETWEEN 0.1 AND 30 THEN labresult ELSE NULL END

            -- 3. Hemoglobin 修正与打捞 (g/dL)
            WHEN labname ILIKE ANY(ARRAY['%hemoglobin%', '%Hgb%', '%total hemoglobin%']) 
                 AND labname NOT ILIKE '%A1c%' 
                 AND labresult BETWEEN 4 AND 25 THEN labresult 
            WHEN labname ILIKE '%Hct%' AND labresult BETWEEN 12 AND 75 THEN labresult / 3.0

            -- 4. pH 生理性过滤
            WHEN labname ILIKE ANY(ARRAY['%pH%', '%arterial pH%']) AND labname NOT ILIKE ANY(ARRAY['%urine%','%fluid%'])
                 AND labresult BETWEEN 6.7 AND 7.8 AND labresult NOT IN (7.0, 8.0) THEN labresult
            
            -- 5. PTT 打捞 (放宽命名匹配)
            WHEN labname ILIKE ANY(ARRAY['%PTT%', '%Partial Thromboplastin Time%', '%aPTT%']) 
                 AND labresult BETWEEN 10 AND 150 THEN labresult

            -- 6. 乳酸打捞
            WHEN labname ILIKE ANY(ARRAY['%lactate%', '%lactic acid%', '%lac%']) 
                 AND labresult BETWEEN 0.1 AND 30 THEN labresult

            -- 7. 其他常规指标 (保持原样)
            WHEN labname ILIKE '%paCO2%' AND labresult BETWEEN 5 AND 150 THEN labresult
            WHEN (labname ILIKE '%bicarb%' OR labname ILIKE '%HCO3%') AND labname NOT ILIKE '%total co2%'
                 AND labresult BETWEEN 2 AND 60 THEN labresult
            WHEN labname ILIKE '%WBC%' AND labresult BETWEEN 0.1 AND 500 THEN labresult
            WHEN labname ILIKE '%albumin%' AND labresult BETWEEN 1.0 AND 6.0 THEN labresult
            WHEN labname ILIKE '%calcium%' AND labname NOT ILIKE '%ion%' AND labresult BETWEEN 4 AND 15 THEN labresult
            WHEN labname ILIKE '%AST%' AND labresult BETWEEN 1 AND 2000 THEN labresult
            WHEN labname ILIKE '%ALT%' AND labresult BETWEEN 1 AND 2000 THEN labresult
            WHEN labname ILIKE '%platelet%' AND labresult BETWEEN 1 AND 1000 THEN labresult
            WHEN labname ILIKE '%anion gap%' AND labresult BETWEEN 2 AND 50 THEN labresult
            WHEN labname ILIKE '%total bilirubin%' AND labresult BETWEEN 0.1 AND 70 THEN labresult
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
    MIN(CASE WHEN labname ILIKE '%pH%' THEN labresult_clean END) AS lab_ph_direct,
    MAX(CASE WHEN labname ILIKE '%pH%' THEN labresult_clean END) AS lab_ph_max,
    AVG(CASE WHEN labname ILIKE '%paCO2%' THEN labresult_clean END) AS lab_paco2,
    AVG(CASE WHEN labname ILIKE '%bicarb%' OR labname ILIKE '%HCO3%' THEN labresult_clean END) AS lab_hco3,
    MIN(CASE WHEN labname ILIKE '%bicarb%' OR labname ILIKE '%HCO3%' THEN labresult_clean END) AS bicarbonate_min,
    MAX(CASE WHEN labname ILIKE '%creatinine%' THEN labresult_clean END) AS creatinine_max,
    MAX(CASE WHEN labname ILIKE '%BUN%' THEN labresult_clean END) AS bun_max,
    MIN(CASE WHEN labname ILIKE '%BUN%' THEN labresult_clean END) AS bun_min,
    MAX(CASE WHEN labname ILIKE '%WBC%' THEN labresult_clean END) AS wbc_max,
    MIN(CASE WHEN labname ILIKE '%WBC%' THEN labresult_clean END) AS wbc_min,
    MIN(CASE WHEN labname ILIKE '%albumin%' THEN labresult_clean END) AS albumin_min,
    MAX(CASE WHEN labname ILIKE '%albumin%' THEN labresult_clean END) AS albumin_max,
    MAX(CASE WHEN labname ILIKE '%lactate%' THEN labresult_clean END) AS lactate_max,
    MIN(CASE WHEN labname ILIKE '%calcium%' AND labname NOT ILIKE '%ion%' THEN labresult_clean END) AS lab_calcium_min,
    MAX(CASE WHEN labname ILIKE '%AST%' THEN labresult_clean END) AS ast_max,
    MAX(CASE WHEN labname ILIKE '%ALT%' THEN labresult_clean END) AS alt_max,
    MIN(CASE WHEN labname ILIKE '%platelet%' THEN labresult_clean END) AS platelet_min,
    MAX(CASE WHEN labname ILIKE '%anion gap%' THEN labresult_clean END) AS aniongap_max,
    MIN(CASE WHEN labname ILIKE '%anion gap%' THEN labresult_clean END) AS aniongap_min,
    MAX(CASE WHEN labname ILIKE '%glucose%' THEN labresult_clean END) AS glucose_lab_max,
    MIN(CASE WHEN labname ILIKE '%total bilirubin%' THEN labresult_clean END) AS bilirubin_total_min,
    MAX(CASE WHEN labname ILIKE '%alkaline phos%' THEN labresult_clean END) AS alp_max,
    MIN(CASE WHEN labname ILIKE ANY(ARRAY['%hemoglobin%', '%Hgb%', '%Hct%']) THEN labresult_clean END) AS hemoglobin_min,
    MIN(CASE WHEN labname ILIKE ANY(ARRAY['%PTT%', '%aPTT%']) THEN labresult_clean END) AS ptt_min
FROM lab_filt
GROUP BY patientunitstayid;

--------------------------------------------------------------------------------
-- 4. 血气增强打捞 (增加乳酸)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_bg;
CREATE TEMP TABLE temp_bg AS
SELECT patientunitstayid, 
    MIN(CASE WHEN ph BETWEEN 6.7 AND 7.8 AND ph NOT IN (7, 8) THEN ph END) AS bg_ph_min,
    MAX(CASE WHEN ph BETWEEN 6.7 AND 7.8 AND ph NOT IN (7, 8) THEN ph END) AS bg_ph_max,
    AVG(CASE WHEN paco2 BETWEEN 5 AND 150 THEN paco2 END) AS bg_paco2
FROM eicu_derived.pivoted_bg WHERE chartoffset BETWEEN -360 AND 1440 GROUP BY patientunitstayid;

DROP TABLE IF EXISTS temp_pf_ratio;
CREATE TEMP TABLE temp_pf_ratio AS
SELECT patientunitstayid, MIN(pao2 / (CASE WHEN fio2 IS NULL THEN 0.21 WHEN fio2 >= 21 THEN fio2/100.0 ELSE 0.21 END)) AS pao2fio2ratio_min
FROM eicu_derived.pivoted_bg WHERE pao2 > 0 AND chartoffset BETWEEN -360 AND 1440 GROUP BY patientunitstayid;

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
-- 7. 最终整合：整合补丁
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS eicu_cview.ap_external_validation;
CREATE TABLE eicu_cview.ap_external_validation AS
WITH tier_calc AS (
    SELECT
        c.patientunitstayid, c.age AS admission_age, c.gender, c.bmi, c.icu_los_hours,
        COALESCE(l.creatinine_max, aps.apache_creatinine) AS creatinine_max,
        
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

        -- 实验室指标联合打捞
        l.lactate_max AS lactate_max,
        COALESCE(l.bun_max, aps.apache_creatinine * 20) AS bun_max,
        COALESCE(l.bun_min, l.bun_max) AS bun_min,
        l.wbc_max, l.wbc_min, l.ptt_min, l.albumin_min,
        COALESCE(l.albumin_max, l.albumin_min) AS albumin_max,
        l.bicarbonate_min, l.lab_calcium_min, l.ast_max, l.alt_max, l.platelet_min,
        l.aniongap_max, l.aniongap_min, l.glucose_lab_max, l.bilirubin_total_min, l.alp_max, 
        l.hemoglobin_min, -- 此时已是修正后的 hemoglobin
        pf.pao2fio2ratio_min,
        COALESCE(comb.malignant_tumor, 0) AS malignant_tumor,
        v.heart_rate_max, v.heart_rate_min, v.resp_rate_max, v.resp_rate_min,
        v.mbp_min, v.spo2_max, v.spo2_min, v.temp_max AS temperature_max, v.temp_min AS temperature_min,
        COALESCE(intv.vaso_flag, 0) AS raw_vaso,
        COALESCE(aps.apache_vent_flag, cp.vent_careplan_flag, intv.vent_treatment_flag, 0) AS raw_vent,
        COALESCE(intv.dialysis_flag, aps.apache_dialysis_flag, 0) AS dialysis_flag,
        CASE WHEN c.hosp_mort = 1 AND pat.hospitaldischargeoffset <= 40320 THEN 1 ELSE 0 END AS mortality_28d
    FROM cohort_base c
    INNER JOIN eicu_crd.patient pat ON c.patientunitstayid = pat.patientunitstayid
    LEFT JOIN temp_lab_raw_all l ON c.patientunitstayid = l.patientunitstayid
    LEFT JOIN temp_apache_aps aps ON c.patientunitstayid = aps.patientunitstayid
    LEFT JOIN temp_bg bg ON c.patientunitstayid = bg.patientunitstayid
    LEFT JOIN temp_vital_full v ON c.patientunitstayid = v.patientunitstayid
    LEFT JOIN temp_interventions intv ON c.patientunitstayid = intv.patientunitstayid
    LEFT JOIN temp_vent_careplan cp ON c.patientunitstayid = cp.patientunitstayid
    LEFT JOIN temp_pf_ratio pf ON c.patientunitstayid = pf.patientunitstayid
    LEFT JOIN temp_comorbidity comb ON c.patientunitstayid = comb.patientunitstayid
),
base_final AS (
    SELECT *,
        -- 对 ph_min 执行严格的逐级生理过滤锁 (6.7 - 7.8)
        CASE 
            WHEN ph_t1 BETWEEN 6.7 AND 7.8 THEN ph_t1
            WHEN ph_t2 BETWEEN 6.7 AND 7.8 THEN ph_t2
            WHEN ph_t3 BETWEEN 6.7 AND 7.8 THEN CAST(ph_t3 AS NUMERIC)
            WHEN ph_t4 BETWEEN 6.7 AND 7.8 THEN ph_t4
            ELSE NULL 
        END AS ph_min,
        
        -- 对 ph_max 执行同样的生理过滤锁
        CASE 
            WHEN ph_max_raw BETWEEN 6.7 AND 7.8 THEN ph_max_raw 
            ELSE NULL 
        END AS ph_max
    FROM tier_calc
)
SELECT *,
    CASE 
        WHEN raw_vaso = 1 AND icu_los_hours >= 24 THEN 1
        WHEN (raw_vent = 1 AND icu_los_hours >= 48) OR (pao2fio2ratio_min < 300 AND icu_los_hours >= 48) THEN 1
        WHEN dialysis_flag = 1 OR creatinine_max > 1.9 THEN 1 ELSE 0 
    END AS pof,
    CASE 
        WHEN (raw_vaso = 1 AND icu_los_hours >= 24) OR (raw_vent = 1 AND icu_los_hours >= 48)
          OR (pao2fio2ratio_min < 300 AND icu_los_hours >= 48) OR dialysis_flag = 1
          OR creatinine_max > 1.9 OR mortality_28d = 1 THEN 1 ELSE 0
    END AS composite_outcome
FROM base_final;

--------------------------------------------------------------------------------
-- 最终统计信息 (加强版：覆盖率 + 生理分布审计)
--------------------------------------------------------------------------------
SELECT 
    -- 1. 总体样本与结局分布
    COUNT(*) AS total_pts,
    SUM(pof) AS pof_cases,
    ROUND(AVG(pof)*100, 2) AS pof_rate_pct,
    SUM(mortality_28d) AS deaths_28d,
    ROUND(AVG(mortality_28d)*100, 2) AS mort_rate_pct,
    SUM(composite_outcome) AS composite_events,

    -- 2. 核心打捞特征覆盖率审计 (Missing% 的反面)
    ROUND(COUNT(ph_min)::numeric / COUNT(*) * 100, 2) AS ph_coverage,
    ROUND(COUNT(lactate_max)::numeric / COUNT(*) * 100, 2) AS lac_coverage,
    ROUND(COUNT(hemoglobin_min)::numeric / COUNT(*) * 100, 2) AS hgb_coverage,
    ROUND(COUNT(ptt_min)::numeric / COUNT(*) * 100, 2) AS ptt_coverage,
    ROUND(COUNT(pao2fio2ratio_min)::numeric / COUNT(*) * 100, 2) AS pf_coverage,
    ROUND(COUNT(bun_max)::numeric / COUNT(*) * 100, 2) AS bun_coverage,

    -- 3. 生理值分布审计 (用于快速发现单位错误)
    -- 检查 Hgb 是否回到 10 左右（之前是 2.2）
    ROUND(AVG(hemoglobin_min)::numeric, 2) AS avg_hgb, 
    -- 检查 BUN 是否回到 5-10 左右（单位转换后）
    ROUND(AVG(bun_max)::numeric, 2) AS avg_bun,
    -- 检查 pH 是否在生理范围内
    MIN(ph_min) AS ph_min_limit,
    MAX(ph_min) AS ph_max_limit,
    -- 检查乳酸中值
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lactate_max) AS median_lactate

FROM eicu_cview.ap_external_validation;

--------------------------------------------------------------------------------
-- 附加：各中心数据质量分布 (可选，用于排查是否某中心缺失特别严重)
--------------------------------------------------------------------------------
SELECT 
    p.hospitalid,
    COUNT(*) AS pts_count,
    ROUND(AVG(pof)*100, 2) AS pof_rate,
    ROUND(COUNT(lactate_max)::numeric / COUNT(*) * 100, 2) AS lac_coverage
FROM eicu_cview.ap_external_validation v
JOIN eicu_crd.patient p ON v.patientunitstayid = p.patientunitstayid
GROUP BY p.hospitalid
HAVING COUNT(*) > 10
ORDER BY pts_count DESC
LIMIT 10;
