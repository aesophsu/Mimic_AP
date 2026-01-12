--------------------------------------------------------------------------------
-- 1. 识别 AP 患者 (保持不变)
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
-- 2. 构建核心队列 (18岁以上, ICU >= 24h) (保持不变)
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
-- 3. 深度打捞实验室指标 (含单位换算与 pH 组件提取)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_lab_raw_all;
CREATE TEMP TABLE temp_lab_raw_all AS
WITH lab_filt AS (
    SELECT 
        patientunitstayid, 
        labname,
        labresult,
        -- 清洗逻辑
        CASE 
            -- pH: 6.8-7.8, 排除整数占位符
            WHEN labname ILIKE '%pH%' 
                 AND labresult BETWEEN 6.8 AND 7.8 
                 AND labresult NOT IN (7.0, 8.0, 7, 8) 
                 THEN labresult
            
            -- PaCO2: 用于公式计算 (10-150 mmHg)
            WHEN labname ILIKE '%paCO2%' AND labresult BETWEEN 10 AND 150 THEN labresult
            
            -- HCO3 (Bicarbonate): 用于公式计算 (5-60 mEq/L)
            WHEN (labname ILIKE '%bicarb%' OR labname ILIKE '%HCO3%') AND labresult BETWEEN 5 AND 60 THEN labresult

            -- Creatinine: 自动单位转换 (>30 视为 umol/L -> /88.4)
            WHEN labname ILIKE '%creatinine%' THEN 
                CASE WHEN labresult > 30 THEN labresult / 88.4 
                     WHEN labresult BETWEEN 0.1 AND 30 THEN labresult 
                     ELSE NULL END

            -- BUN: 范围过滤
            WHEN labname ILIKE '%BUN%' AND labresult BETWEEN 1 AND 200 THEN labresult

            -- WBC: 范围过滤
            WHEN labname ILIKE '%WBC%' AND labresult BETWEEN 0.1 AND 500 THEN labresult

            -- Albumin: 范围过滤
            WHEN labname ILIKE '%albumin%' AND labresult BETWEEN 1.0 AND 6.0 THEN labresult
            
            -- Lactate: 范围过滤
            WHEN labname ILIKE '%lactate%' AND labresult BETWEEN 0.1 AND 30 THEN labresult
            
            -- Calcium: 排除游离钙(1.1-1.3), 只取总钙(6-15)
            WHEN labname ILIKE '%calcium%' AND labname NOT ILIKE '%ion%' AND labresult BETWEEN 4 AND 15 THEN labresult
            
            -- AST/ALT: 肝功能
            WHEN labname ILIKE '%AST%' AND labresult BETWEEN 1 AND 2000 THEN labresult
            WHEN labname ILIKE '%ALT%' AND labresult BETWEEN 1 AND 2000 THEN labresult
            
            -- Platelets
            WHEN labname ILIKE '%platelet%' AND labresult BETWEEN 1 AND 1000 THEN labresult

            ELSE NULL 
        END AS labresult_clean
    FROM eicu_crd.lab
    WHERE labresultoffset BETWEEN -360 AND 1440 
      AND labresult IS NOT NULL
      AND patientunitstayid IN (SELECT patientunitstayid FROM cohort_base)
)
SELECT 
    patientunitstayid,
    -- 直接提取的 pH
    MIN(CASE WHEN labname ILIKE '%pH%' THEN labresult_clean END) AS lab_ph_direct,
    -- 提取 PaCO2 和 HCO3 用于后续公式计算
    AVG(CASE WHEN labname ILIKE '%paCO2%' THEN labresult_clean END) AS lab_paco2,
    AVG(CASE WHEN labname ILIKE '%bicarb%' OR labname ILIKE '%HCO3%' THEN labresult_clean END) AS lab_hco3,
    
    -- 核心特征 (注意 MIN/MAX 的区分，对齐模型需求)
    MAX(CASE WHEN labname ILIKE '%creatinine%' THEN labresult_clean END) AS creatinine_max,
    MAX(CASE WHEN labname ILIKE '%BUN%' THEN labresult_clean END) AS bun_max,
    MIN(CASE WHEN labname ILIKE '%BUN%' THEN labresult_clean END) AS bun_min, -- 补全 MIN
    MAX(CASE WHEN labname ILIKE '%WBC%' THEN labresult_clean END) AS wbc_max,
    MIN(CASE WHEN labname ILIKE '%albumin%' THEN labresult_clean END) AS albumin_min,
    MAX(CASE WHEN labname ILIKE '%lactate%' THEN labresult_clean END) AS lactate_max,
    MIN(CASE WHEN labname ILIKE '%calcium%' AND labname NOT ILIKE '%ion%' THEN labresult_clean END) AS lab_calcium_min,
    MAX(CASE WHEN labname ILIKE '%AST%' THEN labresult_clean END) AS ast_max,
    MAX(CASE WHEN labname ILIKE '%ALT%' THEN labresult_clean END) AS alt_max,
    MIN(CASE WHEN labname ILIKE '%platelet%' THEN labresult_clean END) AS platelet_min

FROM lab_filt
GROUP BY patientunitstayid;

--------------------------------------------------------------------------------
-- 4. 提取血气指标 (BG) - 作为 pH 的主要来源
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_bg;
CREATE TEMP TABLE temp_bg AS
SELECT 
    patientunitstayid, 
    -- 血气表中的直接 pH
    MIN(CASE 
            WHEN ph BETWEEN 6.8 AND 7.8 
            AND ph NOT IN (7.0, 8.0, 7, 8) 
            THEN ph ELSE NULL 
        END) AS bg_ph_min,
    -- 血气表中的 PaCO2 (用于公式补救)
    AVG(CASE WHEN paco2 BETWEEN 10 AND 150 THEN paco2 ELSE NULL END) AS bg_paco2
FROM eicu_derived.pivoted_bg 
WHERE chartoffset BETWEEN -360 AND 1440 
  AND patientunitstayid IN (SELECT patientunitstayid FROM cohort_base)
GROUP BY patientunitstayid;

--------------------------------------------------------------------------------
-- 5. 提取生命体征 (Vitals) - 补全 Temperature
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_vital_full;
CREATE TEMP TABLE temp_vital_full AS
SELECT 
    patientunitstayid, 
    MAX(NULLIF(heartrate, -1)) AS heart_rate_max,
    MIN(NULLIF(heartrate, -1)) AS heart_rate_min,
    MAX(NULLIF(respiratoryrate, -1)) AS resp_rate_max,
    MIN(NULLIF(respiratoryrate, -1)) AS resp_rate_min,
    MIN(NULLIF(nibp_mean, -1)) AS mbp_min, 
    MAX(NULLIF(spo2, -1)) AS spo2_max,
    -- 摄氏度转换逻辑：如果 > 50 (Fahrenheit), 转换为 Celsius
    MAX(CASE 
        WHEN temperature > 50 THEN (temperature - 32) * 5/9 
        ELSE temperature 
    END) AS temp_max,
    MIN(CASE 
        WHEN temperature > 50 THEN (temperature - 32) * 5/9 
        ELSE temperature 
    END) AS temp_min
FROM eicu_derived.pivoted_vital 
WHERE chartoffset BETWEEN -360 AND 1440 
  AND patientunitstayid IN (SELECT patientunitstayid FROM cohort_base)
GROUP BY patientunitstayid;

--------------------------------------------------------------------------------
-- 6. 增强版 POF 定义 (引入 CarePlan)
--------------------------------------------------------------------------------
-- 6.1 药物干预
DROP TABLE IF EXISTS temp_interventions;
CREATE TEMP TABLE temp_interventions AS
SELECT 
    patientunitstayid, 
    MAX(CASE WHEN treatmentstring ILIKE '%vasopressor%' OR treatmentstring ILIKE '%dopamine%' OR treatmentstring ILIKE '%norepinephrine%' THEN 1 ELSE 0 END) AS vaso_flag,
    MAX(CASE WHEN treatmentstring ILIKE '%dialysis%' OR treatmentstring ILIKE '%CRRT%' THEN 1 ELSE 0 END) AS dialysis_flag,
    -- 治疗表中的机械通气记录
    MAX(CASE WHEN treatmentstring ILIKE '%ventilation%' OR treatmentstring ILIKE '%intubat%' THEN 1 ELSE 0 END) AS vent_treatment_flag
FROM eicu_crd.treatment
WHERE treatmentoffset BETWEEN -360 AND 1440 
  AND patientunitstayid IN (SELECT patientunitstayid FROM cohort_base)
GROUP BY patientunitstayid;

-- 6.2 护理计划 (CarePlan) - 识别插管/通气的新金标准
DROP TABLE IF EXISTS temp_vent_careplan;
CREATE TEMP TABLE temp_vent_careplan AS
SELECT
    patientunitstayid,
    1 AS vent_careplan_flag
FROM eicu_crd.careplangeneral
WHERE cplitemoffset BETWEEN -360 AND 1440
  AND (
      cplitemvalue ILIKE '%ventilat%' 
      OR cplitemvalue ILIKE '%intubat%' 
      OR cplitemvalue ILIKE '%ET tube%'
  )
  AND patientunitstayid IN (SELECT patientunitstayid FROM cohort_base)
GROUP BY patientunitstayid;

-- 6.3 APACHE 评分 (兜底 pH 和 Cr)
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
-- 7. 最终整合：严谨版 POF 判定与 28天死亡
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS eicu_cview.ap_external_validation;
CREATE TABLE eicu_cview.ap_external_validation AS
WITH base_final AS (
    SELECT
        c.*,
        -- pH 公式补全逻辑 (保持不变)
        CASE 
            WHEN bg.bg_ph_min IS NOT NULL THEN bg.bg_ph_min
            WHEN l.lab_ph_direct IS NOT NULL THEN l.lab_ph_direct
            WHEN l.lab_hco3 > 0 AND COALESCE(bg.bg_paco2, l.lab_paco2) > 0 THEN 
                 CAST(6.1 + LOG10(l.lab_hco3 / (0.0301 * COALESCE(bg.bg_paco2, l.lab_paco2))) AS NUMERIC)
            WHEN aps.ph IS NOT NULL THEN aps.ph
            ELSE NULL
        END AS ph_min,
        
        -- 生化指标
        COALESCE(l.creatinine_max, aps.creatinine) AS creatinine_max,
        COALESCE(l.bun_max, aps.creatinine * 20) AS bun_max,
        COALESCE(l.bun_min, l.bun_max) AS bun_min,
        l.wbc_max, l.albumin_min, l.lactate_max,
        l.lab_calcium_min, l.ast_max, l.alt_max, l.platelet_min,
        
        -- 生命体征
        v.heart_rate_max, v.heart_rate_min, v.resp_rate_max, v.resp_rate_min,
        v.mbp_min, v.spo2_max, v.temp_max, v.temp_min,

        -- 精细化干预逻辑 (排除干扰)
        COALESCE(intv.vaso_flag, 0) AS raw_vaso,
        COALESCE(aps.vent, cp.vent_careplan_flag, intv.vent_treatment_flag, 0) AS raw_vent,
        COALESCE(intv.dialysis_flag, aps.dialysis, 0) AS dialysis_flag,

        -- 判定 28 天死亡
        CASE 
            WHEN c.hosp_mort = 1 AND pat.hospitaldischargeoffset <= 40320 THEN 1 
            ELSE 0 
        END AS death_28d

    FROM cohort_base c
    INNER JOIN eicu_crd.patient pat ON c.patientunitstayid = pat.patientunitstayid
    LEFT JOIN temp_lab_raw_all l ON c.patientunitstayid = l.patientunitstayid
    LEFT JOIN temp_apache_aps aps ON c.patientunitstayid = aps.patientunitstayid
    LEFT JOIN temp_bg bg ON c.patientunitstayid = bg.patientunitstayid
    LEFT JOIN temp_vital_full v ON c.patientunitstayid = v.patientunitstayid
    LEFT JOIN temp_interventions intv ON c.patientunitstayid = intv.patientunitstayid
    LEFT JOIN temp_vent_careplan cp ON c.patientunitstayid = cp.patientunitstayid
)
SELECT
    *,
    -- 【核心】重新定义的 POF (Persistent Organ Failure)
    CASE 
        -- 1. 28天死亡：直接判定为 POF (因为死亡通常是器官衰竭的结局)
        WHEN death_28d = 1 THEN 1 
        
        -- 2. 循环衰竭：必须使用升压药 且 ICU时长 > 24h (排除一过性波动)
        WHEN raw_vaso = 1 AND icu_los_hours >= 24 THEN 1
        
        -- 3. 呼吸衰竭：必须机械通气 且 ICU时长 >= 48h (核心判定，排除麻醉插管)
        WHEN raw_vent = 1 AND icu_los_hours >= 48 THEN 1
        
        -- 4. 肾衰竭：透析 或 肌酐 > 1.9 mg/dL (Marshall评分2分标准)
        WHEN dialysis_flag = 1 OR creatinine_max > 1.9 THEN 1
        
        ELSE 0 
    END AS pof

FROM base_final;

--------------------------------------------------------------------------------
-- 审计输出 3.0
--------------------------------------------------------------------------------
SELECT 
    COUNT(*) as total,
    ROUND(AVG(death_28d)::numeric, 3) as death_28d_rate,
    ROUND(AVG(pof)::numeric, 3) as pof_prevalence_final,
    ROUND(AVG(ph_min)::numeric, 3) as avg_ph,
    ROUND(AVG(creatinine_max)::numeric, 3) as avg_creatinine,
    ROUND(COUNT(ph_min)::numeric / COUNT(*), 4) * 100 as ph_fill_rate
FROM eicu_cview.ap_external_validation;
