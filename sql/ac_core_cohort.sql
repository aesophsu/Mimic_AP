CREATE TABLE my_custom_schema.ac_core_cohort AS
WITH ranked_weights AS (
    SELECT
        stay_id,
        weight,
        ROW_NUMBER() OVER (
            PARTITION BY stay_id
            ORDER BY
                CASE
                    WHEN weight_type = 'admit' THEN 1
                    ELSE 2
                END,
                starttime
        ) AS rn
    FROM
        mimiciv_derived.weight_durations
)
SELECT
    -- 基础患者及ICU住院信息
    t1.subject_id,
    t1.hadm_id,
    t1.stay_id,
    t1.los,
    t1.intime,
    t1.anchor_age,
    t1.gender,
    -- 从 admissions 表中选择你需要的字段
    t2.admittime,
    t2.dischtime,
    t2.deathtime,
    t2.insurance,
    t2.race,
    t2.hospital_expire_flag,
    -- 从 patients 表中选择你需要的字段
    t3.dod,
    -- 添加筛选后的体重和身高
    t4.weight,
    t5.height
FROM
    my_custom_schema.ac_24hr_plus_first_icustay t1
INNER JOIN
    mimiciv_hosp.admissions t2 ON t1.hadm_id = t2.hadm_id
INNER JOIN
    mimiciv_hosp.patients t3 ON t1.subject_id = t3.subject_id
LEFT JOIN
    ranked_weights t4 ON t1.stay_id = t4.stay_id AND t4.rn = 1
LEFT JOIN
    mimiciv_derived.height t5 ON t1.stay_id = t5.stay_id;
