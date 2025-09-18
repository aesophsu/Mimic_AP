WITH dx_ap_patients AS (
    -- Step 1: 筛选诊断为急性胰腺炎的住院记录
    SELECT DISTINCT
        d.subject_id,
        d.hadm_id
    FROM
        mimiciv_hosp.diagnoses_icd d
    WHERE
        (d.icd_version = 9 AND d.icd_code = '5770')
        OR (d.icd_version = 10 AND d.icd_code LIKE 'K85%')
)
, filtered_patients AS (
    -- Step 2: 联接 ICU 记录和患者信息，筛选住院天数和年龄
    SELECT
        icu.subject_id,
        icu.hadm_id,
        icu.intime
    FROM
        dx_ap_patients dx
    JOIN
        mimiciv_icu.icustays icu ON dx.hadm_id = icu.hadm_id
    JOIN
        mimiciv_hosp.patients p ON icu.subject_id = p.subject_id
    WHERE
        icu.los > 1        -- ICU住院天数大于1天
        AND p.anchor_age >= 18  -- 患者年龄>=18岁
)
, ap_patients AS (
    -- Step 3: 按患者排序住院，判断首次ICU住院
    SELECT
        subject_id,
        hadm_id,
        intime,
        ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime) AS rn
    FROM
        filtered_patients
)

-- 统计各项
SELECT
    '1. 符合急性胰腺炎诊断的患者总数' AS description,
    COUNT(DISTINCT subject_id) AS count_value
FROM dx_ap_patients

UNION ALL

SELECT
    '2. 符合急性胰腺炎诊断且住院筛选的次数',
    COUNT(DISTINCT hadm_id)
FROM ap_patients

UNION ALL

SELECT
    '3. 仅有一次ICU住院记录的患者总数',
    COUNT(subject_id)
FROM (
    SELECT subject_id
    FROM ap_patients
    GROUP BY subject_id
    HAVING COUNT(hadm_id) = 1
) AS single_stay_patients

UNION ALL

SELECT
    '4. 首次ICU住院的患者总数',
    COUNT(subject_id)
FROM (
    SELECT subject_id
    FROM ap_patients
    WHERE rn = 1
) AS first_stay_patients;
