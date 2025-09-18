CREATE MATERIALIZED VIEW derived_data_summary AS
SELECT
    fdv.stay_id,
    icu.hadm_id, -- 新增: 从icustays表中获取hadm_id
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
    vent.ventilation_status,
    vaso.stay_id AS vasoactive_agent_stay_id,
    meld.meld AS meld_original,
    meld.meld AS meld_2016
FROM mimiciv_derived.first_day_vitalsign fdv
JOIN mimiciv_icu.icustays icu ON fdv.stay_id = icu.stay_id -- 修复：通过 icustays 表联接以获取 hadm_id
LEFT JOIN mimiciv_derived.first_day_lab fdl ON fdv.stay_id = fdl.stay_id
LEFT JOIN mimiciv_derived.first_day_bg fdbg ON fdv.stay_id = fdbg.stay_id
LEFT JOIN mimiciv_derived.coagulation co ON icu.hadm_id = co.hadm_id -- 修复：使用 icu.hadm_id 进行联接
LEFT JOIN mimiciv_derived.sofa s ON fdv.stay_id = s.stay_id
LEFT JOIN mimiciv_derived.apsiii aps ON fdv.stay_id = aps.stay_id
LEFT JOIN mimiciv_derived.oasis o ON fdv.stay_id = o.stay_id
LEFT JOIN mimiciv_derived.lods l ON fdv.stay_id = l.stay_id
LEFT JOIN mimiciv_derived.charlson charlson ON icu.hadm_id = charlson.hadm_id -- 修复：使用 icu.hadm_id 进行联接
LEFT JOIN mimiciv_derived.ventilation vent ON fdv.stay_id = vent.stay_id
LEFT JOIN mimiciv_derived.vasoactive_agent vaso ON fdv.stay_id = vaso.stay_id
LEFT JOIN mimiciv_derived.meld meld ON fdv.stay_id = meld.stay_id;
