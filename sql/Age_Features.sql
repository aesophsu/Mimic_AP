-- 重新设计的年龄提取逻辑：基于 ICU Stay
DROP TABLE IF EXISTS mimiciv_derived.custom_patient_age;
CREATE TABLE mimiciv_derived.custom_patient_age AS
SELECT
    ie.subject_id,
    ie.hadm_id,
    ie.stay_id,
    -- 核心逻辑：锚定年龄 + (ICU入科年份 - 锚定年份)
    -- 使用 DATETIME_DIFF 更加直观且符合 BigQuery/PostgreSQL 通用习惯
    pa.anchor_age + DATETIME_DIFF(ie.intime, DATETIME(pa.anchor_year, 1, 1, 0, 0, 0), SECOND) / 31557600.0 AS age,
    
    -- 额外派生：年龄分层（常用于 Baseline Table 1）
    CASE 
        WHEN pa.anchor_age + DATETIME_DIFF(ie.intime, DATETIME(pa.anchor_year, 1, 1, 0, 0, 0), SECOND) / 31557600.0 < 45 THEN '<45'
        WHEN pa.anchor_age + DATETIME_DIFF(ie.intime, DATETIME(pa.anchor_year, 1, 1, 0, 0, 0), SECOND) / 31557600.0 BETWEEN 45 AND 65 THEN '45-65'
        WHEN pa.anchor_age + DATETIME_DIFF(ie.intime, DATETIME(pa.anchor_year, 1, 1, 0, 0, 0), SECOND) / 31557600.0 BETWEEN 65 AND 80 THEN '65-80'
        ELSE '>80' 
    END AS age_group
FROM `physionet-data.mimiciv_icu.icustays` ie
INNER JOIN `physionet-data.mimiciv_hosp.patients` pa
    ON ie.subject_id = pa.subject_id;
