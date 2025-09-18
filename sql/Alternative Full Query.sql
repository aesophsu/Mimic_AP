WITH ac_patients AS (
    -- 查找所有符合酒精性肝硬化诊断的患者的hadm_id
    SELECT DISTINCT hadm_id
    FROM mimiciv_hosp.diagnoses_icd
    WHERE (icd_version = 9 AND icd_code = '5712')
        OR (icd_version = 10 AND icd_code IN ('K7030', 'K7031'))
),
first_icu_stay AS (
    -- 找出每个患者的首次ICU停留
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        intime,
        ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime) AS rn
    FROM mimiciv_icu.icustays
    WHERE los >= 1
),
filtered_stays AS (
    -- 结合首次ICU停留和酒精性肝硬化诊断，并过滤掉不符合条件的
    SELECT
        i.subject_id,
        i.hadm_id,
        i.stay_id,
        i.intime
    FROM first_icu_stay i
    JOIN ac_patients ac ON i.hadm_id = ac.hadm_id
    WHERE i.rn = 1
)
-- 主查询：从filtered_stays中选择患者，并连接所有相关数据
SELECT
    p.subject_id,
    a.hadm_id,
    i.stay_id,
    p.gender,
    p.anchor_age AS age,
    fdv.heart_rate_mean AS heart_rate,
    fdv.temperature_mean AS temperature,
    fdv.mbp_mean AS mean_arterial_pressure,
    fdl.wbc_min AS wbc,
    fdl.platelets_min AS platelet,
    fdl.sodium_min AS sodium,
    fdl.potassium_min AS potassium,
    fdl.bicarbonate_min AS bicarbonate,
    fdl.chloride_min AS chloride,
    fdl.bun_min AS bun,
    fdbg.lactate_min AS lactate,
    fdl.creatinine_min AS creatinine,
    fdl.bilirubin_total_min AS total_bilirubin,
    fdl.albumin_min AS alb,
    co.inr AS inr,
    co.ptt AS aptt,
    s.sofa_24hours AS sofa,
    aps.apsiii AS apsiii,
    o.oasis AS oasis,
    l.lods AS lods,
    charlson.congestive_heart_failure AS heart_failure,
    charlson.renal_disease AS chronic_kidney_disease,
    charlson.chronic_pulmonary_disease AS chronic_obstructive_pulmonary_disease,
    charlson.myocardial_infarct AS coronary_heart_disease,
    charlson.cerebrovascular_disease AS stroke,
    charlson.malignant_cancer AS malignancy,
    charlson.charlson_comorbidity_index,
    CASE WHEN vent.ventilation_status IS NOT NULL THEN 1 ELSE 0 END AS mechanical_ventilation,
    CASE WHEN vaso.stay_id IS NOT NULL THEN 1 ELSE 0 END AS vasopressor,
    CASE WHEN a.deathtime BETWEEN i.intime AND i.intime + INTERVAL '28 days' THEN 1 ELSE 0 END AS mortality_28d,
    meld.meld AS meld_original,
    meld.meld AS meld_2016
FROM mimiciv_hosp.patients p
JOIN mimiciv_hosp.admissions a ON p.subject_id = a.subject_id
JOIN filtered_stays i ON a.hadm_id = i.hadm_id
LEFT JOIN mimiciv_derived.first_day_vitalsign fdv ON i.stay_id = fdv.stay_id
LEFT JOIN mimiciv_derived.first_day_lab fdl ON i.stay_id = fdl.stay_id
LEFT JOIN mimiciv_derived.first_day_bg fdbg ON i.stay_id = fdbg.stay_id
LEFT JOIN mimiciv_derived.coagulation co ON i.hadm_id = co.hadm_id
LEFT JOIN mimiciv_derived.sofa s ON i.stay_id = s.stay_id
LEFT JOIN mimiciv_derived.apsiii aps ON i.stay_id = aps.stay_id
LEFT JOIN mimiciv_derived.oasis o ON i.stay_id = o.stay_id
LEFT JOIN mimiciv_derived.lods l ON i.stay_id = l.stay_id
LEFT JOIN mimiciv_derived.charlson charlson ON a.hadm_id = charlson.hadm_id
LEFT JOIN mimiciv_derived.ventilation vent ON i.stay_id = vent.stay_id
LEFT JOIN mimiciv_derived.vasoactive_agent vaso ON i.stay_id = vaso.stay_id
LEFT JOIN mimiciv_derived.meld meld ON i.stay_id = meld.stay_id
WHERE p.anchor_age >= 18
ORDER BY p.subject_id;
