-- 在运行此代码前，请确保您已经创建了自定义模式，例如：
-- CREATE SCHEMA my_custom_schema;

DROP TABLE my_custom_schema.ac_24hr_plus_first_icustay;
/*
该SQL查询旨在从MIMIC-IV数据库中创建一个符合特定筛选条件的患者队列。
整个流程分为五个核心步骤：

1.  **患者诊断筛选**: 从 `diagnoses_icd` 表中筛选出所有符合指定诊断代码的患者。
2.  **ICU住院时长筛选**: 将患者与 `icustays` 表连接，并只保留ICU住院时长超过24小时的记录。
3.  **首次ICU入院筛选**: 使用窗口函数 `ROW_NUMBER()` 为每个患者的ICU住院记录按时间排序。
4.  **最终队列选择**: 从排序后的结果中，只保留每个患者的第一条记录。
5.  **患者年龄筛选**: 在上述所有筛选完成后，再排除年龄小于或等于18岁的患者。
*/

CREATE TABLE my_custom_schema.ac_24hr_plus_first_icustay AS
WITH filtered_patients AS (
    -- 步骤1: 筛选出所有符合指定诊断的患者ID
    SELECT DISTINCT
        subject_id
    FROM
        mimiciv_hosp.diagnoses_icd
    WHERE
        (icd_version = 9 AND icd_code = '5712')
        OR (icd_version = 10 AND icd_code IN ('K7030', 'K7031'))
),
filtered_stays AS (
    -- 步骤2: 将符合诊断的患者与ICU住院表连接，并筛选出ICU停留时间大于24小时的记录
    SELECT
        t1.subject_id,
        t2.hadm_id,
        t2.stay_id,
        t2.los,
        t2.intime
    FROM
        filtered_patients t1
    INNER JOIN
        mimiciv_icu.icustays t2 ON t1.subject_id = t2.subject_id
    WHERE
        t2.los > 1
),
ranked_stays AS (
    -- 步骤3: 为每个患者的ICU住院记录按入院时间进行排序
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        los,
        intime,
        ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime) AS rn
    FROM
        filtered_stays
),
final_preliminary_cohort AS (
    -- 步骤4: 从排序后的结果中，只保留每个患者的第一条记录（首次ICU入院）
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        los,
        intime
    FROM
        ranked_stays
    WHERE
        rn = 1
)
-- 步骤5: 将上述结果与患者表连接，并筛选年龄
SELECT
    t1.subject_id,
    t1.hadm_id,
    t1.stay_id,
    t1.los,
    t1.intime,
    t2.anchor_age
FROM
    final_preliminary_cohort t1
INNER JOIN
    mimiciv_hosp.patients t2 ON t1.subject_id = t2.subject_id
WHERE
    t2.anchor_age > 18; -- 筛选年龄大于18岁的患者
    
WITH initial_patients AS (
    -- 步骤1: 计算符合诊断的初始患者总数
    SELECT
        COUNT(DISTINCT subject_id) AS initial_patients_count
    FROM
        mimiciv_hosp.diagnoses_icd
    WHERE
        (icd_version = 9 AND icd_code = '5712')
        OR (icd_version = 10 AND icd_code IN ('K7030', 'K7031'))
)
-- 步骤2: 计算并显示两个数字
SELECT
    '符合诊断的初始患者总数' AS patient_type,
    initial_patients_count AS patient_count
FROM
    initial_patients
UNION ALL
SELECT
    '符合特定筛选条件患者总数' AS patient_type,
    COUNT(DISTINCT subject_id) AS patient_count
FROM
    my_custom_schema.ac_24hr_plus_first_icustay;
