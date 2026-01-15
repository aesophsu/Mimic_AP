--------------------------------------------------------------------------------
-- 1. 识别 AP 患者 (保持不变)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_ap_patients;
CREATE TEMP TABLE temp_ap_patients AS
SELECT 
    patientunitstayid
FROM eicu_crd.diagnosis
WHERE diagnosisstring ILIKE '%pancreatit%'
  AND diagnosisstring NOT ILIKE '%chronic%' -- 排除慢性胰腺炎
GROUP BY patientunitstayid;

SELECT 
    COUNT(*) AS total_ap_patients_eicu
FROM temp_ap_patients;

--------------------------------------------------------------------------------
-- 2. 构建核心队列 (18岁以上, ICU >= 24h) (保持不变)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS cohort_base;
CREATE TEMP TABLE cohort_base AS
WITH ranked_stays AS (
    SELECT 
        i.*,
        ROW_NUMBER() OVER (
            PARTITION BY i.uniquepid 
            ORDER BY i.hospitaladmitoffset DESC, i.unitadmitoffset DESC
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
    -- 对原始身高体重进行生理学清洗，不符合条件的记为 NULL
    CASE WHEN r.admissionheight BETWEEN 120 AND 250 THEN r.admissionheight ELSE NULL END AS height,
    CASE WHEN r.admissionweight BETWEEN 30 AND 300 THEN r.admissionweight ELSE NULL END AS weight,
    r.icu_los_hours,
    r.hosp_mort,
    -- 基于清洗后的数据计算 BMI，并增加 BMI 自身的生理限制 (10-60)
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

-- 统计清洗后的有效样本
SELECT 
    COUNT(DISTINCT uniquepid) AS unique_patients,
    COUNT(CASE WHEN bmi IS NOT NULL THEN 1 END) AS patients_with_valid_bmi,
    ROUND(AVG(bmi)::numeric, 2) AS avg_bmi
FROM cohort_base;        
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
            	 AND labname NOT ILIKE '%urine%'
 				 AND labname NOT ILIKE '%fluid%'
                 AND labresult BETWEEN 6.8 AND 7.8 
                 AND labresult NOT IN (7.0, 8.0, 7, 8) 
                 THEN labresult
            
            -- PaCO2: 用于公式计算 (10-150 mmHg)
            WHEN labname ILIKE '%paCO2%' AND labresult BETWEEN 10 AND 150 THEN labresult
            
            -- HCO3 (Bicarbonate): 用于公式计算 (5-60 mEq/L)
            WHEN (labname ILIKE '%bicarb%' OR labname ILIKE '%HCO3%') 
            	 AND labname NOT ILIKE '%total co2%'
            	 AND labresult BETWEEN 5 AND 60 THEN labresult

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

SELECT 
    COUNT(*) AS total_cohort,
    COUNT(creatinine_max) * 100.0 / COUNT(*) AS creatinine_fill_rate,
    COUNT(lactate_max) * 100.0 / COUNT(*) AS lactate_fill_rate,
    COUNT(lab_ph_direct) * 100.0 / COUNT(*) AS ph_fill_rate
FROM temp_lab_raw_all;
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
--------------------------------------------------------------------------------
-- 6.1 药物及替代治疗干预 (增强版)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_interventions;
CREATE TEMP TABLE temp_interventions AS
SELECT 
    patientunitstayid, 
    -- 增加 vasopressin 和 epinephrine 以提高循环系统识别率
    MAX(CASE WHEN treatmentstring ILIKE '%vasopressor%' 
               OR treatmentstring ILIKE '%dopamine%' 
               OR treatmentstring ILIKE '%norepinephrine%' 
               OR treatmentstring ILIKE '%vasopressin%'
               OR treatmentstring ILIKE '%epinephrine%' THEN 1 ELSE 0 END) AS vaso_flag,
    
    -- 肾脏替代治疗
    MAX(CASE WHEN treatmentstring ILIKE '%dialysis%' 
               OR treatmentstring ILIKE '%CRRT%' 
               OR treatmentstring ILIKE '%hemofiltration%' THEN 1 ELSE 0 END) AS dialysis_flag,
    
    -- 呼吸干预
    MAX(CASE WHEN treatmentstring ILIKE '%ventilation%' 
               OR treatmentstring ILIKE '%intubat%' THEN 1 ELSE 0 END) AS vent_treatment_flag
FROM eicu_crd.treatment
WHERE treatmentoffset BETWEEN -360 AND 1440 
  AND patientunitstayid IN (SELECT patientunitstayid FROM cohort_base)
GROUP BY patientunitstayid;

--------------------------------------------------------------------------------
-- 6.2 护理计划 (CarePlan) - 保持双保险识别通气
--------------------------------------------------------------------------------
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

--------------------------------------------------------------------------------
-- 6.3 APACHE 评分 (兜底数据提取)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_apache_aps;
CREATE TEMP TABLE temp_apache_aps AS
SELECT 
    patientunitstayid, 
    CASE WHEN ph > 6.8 AND ph < 7.8 AND ph NOT IN (7.0, 8.0) THEN ph ELSE NULL END as apache_ph, 
    CASE WHEN creatinine BETWEEN 0.1 AND 20 THEN creatinine ELSE NULL END as apache_creatinine, 
    vent AS apache_vent_flag, 
    dialysis AS apache_dialysis_flag
FROM eicu_crd.apacheapsvar
WHERE patientunitstayid IN (SELECT patientunitstayid FROM cohort_base);

--------------------------------------------------------------------------------
-- 6.4 【关键新增】识别 24-48 小时内死亡患者
-- 这是为了落实：死亡患者被记为 POF 为 0 的特殊研究设计
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS temp_early_death;
CREATE TEMP TABLE temp_early_death AS
SELECT 
    p.patientunitstayid,
    1 AS early_death_24_48h
FROM eicu_crd.patient p
WHERE p.unitdischargestatus = 'Expired'
  -- 24小时 (1440 min) 到 48小时 (2880 min) 之间
  AND p.unitdischargeoffset BETWEEN 1440 AND 2880
  AND p.patientunitstayid IN (SELECT patientunitstayid FROM cohort_base);
--------------------------------------------------------------------------------
-- 7. 最终整合：严谨版 POF 判定与 28天死亡 (已修正早期死亡逻辑)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS eicu_cview.ap_external_validation;
CREATE TABLE eicu_cview.ap_external_validation AS
WITH base_final AS (
    SELECT
        c.*,
        -- 修正点：引用 temp_apache_aps 中重命名后的字段
        COALESCE(l.creatinine_max, aps.apache_creatinine) AS creatinine_max,
        
        -- pH 公式补全逻辑
        CASE 
            WHEN bg.bg_ph_min IS NOT NULL THEN bg.bg_ph_min
            WHEN l.lab_ph_direct IS NOT NULL THEN l.lab_ph_direct
            WHEN l.lab_hco3 > 0 AND COALESCE(bg.bg_paco2, l.lab_paco2) > 0 THEN 
                 CAST(6.1 + LOG10(l.lab_hco3 / (0.0301 * COALESCE(bg.bg_paco2, l.lab_paco2))) AS NUMERIC)
            WHEN aps.apache_ph IS NOT NULL THEN aps.apache_ph -- 修正点
            ELSE NULL
        END AS ph_min,
        
        -- 其余生化指标 (BUN 兜底逻辑：若无 BUN，按 Cr*20 估算)
        COALESCE(l.bun_max, aps.apache_creatinine * 20) AS bun_max,
        COALESCE(l.bun_min, l.bun_max) AS bun_min,
        l.wbc_max, l.albumin_min, l.lactate_max,
        l.lab_calcium_min, l.ast_max, l.alt_max, l.platelet_min,
        
        -- 生命体征
        v.heart_rate_max, v.heart_rate_min, v.resp_rate_max, v.resp_rate_min,
        v.mbp_min, v.spo2_max, v.temp_max, v.temp_min,

        -- 干预逻辑汇总
        COALESCE(intv.vaso_flag, 0) AS raw_vaso,
        COALESCE(aps.apache_vent_flag, cp.vent_careplan_flag, intv.vent_treatment_flag, 0) AS raw_vent,
        COALESCE(intv.dialysis_flag, aps.apache_dialysis_flag, 0) AS dialysis_flag,

        -- 判定 28 天死亡
        CASE 
            WHEN c.hosp_mort = 1 AND pat.hospitaldischargeoffset <= 40320 THEN 1 
            ELSE 0 
        END AS death_28d,
        
        -- 引入早期死亡标记
        COALESCE(ed.early_death_24_48h, 0) AS early_death_24_48h

    FROM cohort_base c
    INNER JOIN eicu_crd.patient pat ON c.patientunitstayid = pat.patientunitstayid
    LEFT JOIN temp_lab_raw_all l ON c.patientunitstayid = l.patientunitstayid
    LEFT JOIN temp_apache_aps aps ON c.patientunitstayid = aps.patientunitstayid
    LEFT JOIN temp_bg bg ON c.patientunitstayid = bg.patientunitstayid
    LEFT JOIN temp_vital_full v ON c.patientunitstayid = v.patientunitstayid
    LEFT JOIN temp_interventions intv ON c.patientunitstayid = intv.patientunitstayid
    LEFT JOIN temp_vent_careplan cp ON c.patientunitstayid = cp.patientunitstayid
    LEFT JOIN temp_early_death ed ON c.patientunitstayid = ed.patientunitstayid
)
SELECT
    *,
    -- POF 判定矩阵
    CASE 
        WHEN early_death_24_48h = 1 THEN 0 
        WHEN raw_vaso = 1 AND icu_los_hours >= 24 THEN 1
        WHEN raw_vent = 1 AND icu_los_hours >= 48 THEN 1
        WHEN dialysis_flag = 1 OR creatinine_max > 1.9 THEN 1
        WHEN death_28d = 1 AND icu_los_hours >= 48 THEN 1
        ELSE 0 
    END AS pof
FROM base_final;

--------------------------------------------------------------------------------
-- 最终统计信息
--------------------------------------------------------------------------------
SELECT 
    COUNT(*) as total_cohort,
    SUM(early_death_24_48h) as n_early_death_logic_0,
    SUM(death_28d) as n_death_28d,
    SUM(pof) as n_pof_final,
    ROUND(AVG(pof)::numeric * 100, 2) as pof_prevalence_pct,
    ROUND(COUNT(ph_min)::numeric / COUNT(*) * 100, 2) as ph_data_coverage_pct,
    ROUND(COUNT(creatinine_max)::numeric / COUNT(*) * 100, 2) as creatinine_coverage_pct
FROM eicu_cview.ap_external_validation;

SELECT 
    COUNT(*) AS total_n,
    COUNT(bmi) AS bmi_not_null_n,
    ROUND(COUNT(bmi) * 100.0 / COUNT(*), 2) AS bmi_fill_rate,
    -- 看看 BMI 的分布，排除异常值
    ROUND(AVG(bmi)::numeric, 2) AS avg_bmi,
    MIN(bmi) AS min_bmi,
    MAX(bmi) AS max_bmi
FROM eicu_cview.ap_external_validation;
