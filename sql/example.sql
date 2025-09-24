DROP TABLE my_custom_schema.ac_patients_stays;
CREATE TABLE my_custom_schema.ac_patients_stays AS
SELECT DISTINCT
    subject_id
FROM
    mimiciv_hosp.diagnoses_icd
WHERE
    (icd_version = 9 AND icd_code = '5712')
    OR (icd_version = 10 AND icd_code IN ('K7030', 'K7031'))
ORDER BY
    subject_id;

CREATE TABLE my_custom_schema.ac_24hr_plus_icustays AS
SELECT
    t1.subject_id,
    t2.hadm_id,
    t2.stay_id,
    t2.los,
    t2.intime
FROM
    my_custom_schema.ac_patients_stays t1
INNER JOIN
    mimiciv_icu.icustays t2 ON t1.subject_id = t2.subject_id
WHERE
    t2.los > 1; -- 筛选出ICU住院时长大于1天（24小时）的记录

SELECT
    COUNT(DISTINCT subject_id)
FROM
    my_custom_schema.ac_24hr_plus_icustays;
    
CREATE TABLE my_custom_schema.ac_24hr_plus_first_icustay AS
WITH ranked_icu_stays AS (
    -- 1. 为每个患者的ICU住院记录按intime排序
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        los,
        intime,
        ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime) AS rn
    FROM
        my_custom_schema.ac_24hr_plus_icustays
)
-- 2. 只保留每个患者的第一条记录
SELECT
    subject_id,
    hadm_id,
    stay_id,
    los,
    intime
FROM
    ranked_icu_stays
WHERE
    rn = 1;
