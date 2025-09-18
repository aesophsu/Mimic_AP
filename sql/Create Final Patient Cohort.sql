CREATE MATERIALIZED VIEW ac_patients_data AS
SELECT
    p.subject_id,
    a.hadm_id,
    i.stay_id,
    p.gender,
    p.anchor_age AS age,
    s.heart_rate,
    s.temperature,
    s.mean_arterial_pressure,
    s.wbc,
    s.platelet,
    s.sodium,
    s.potassium,
    s.bicarbonate,
    s.chloride,
    s.bun,
    s.lactate,
    s.creatinine,
    s.total_bilirubin,
    s.alb,
    s.inr,
    s.aptt,
    s.sofa,
    s.apsiii,
    s.oasis,
    s.lods,
    s.heart_failure,
    s.chronic_kidney_disease,
    s.chronic_obstructive_pulmonary_disease,
    s.coronary_heart_disease,
    s.stroke,
    s.malignant_cancer,
    s.charlson_comorbidity_index,
    CASE WHEN s.ventilation_status IS NOT NULL THEN 1 ELSE 0 END AS mechanical_ventilation,
    CASE WHEN s.vasoactive_agent_stay_id IS NOT NULL THEN 1 ELSE 0 END AS vasopressor,
    CASE WHEN a.deathtime BETWEEN i.intime AND i.intime + INTERVAL '28 days' THEN 1 ELSE 0 END AS mortality_28d,
    s.meld_original,
    s.meld_2016
FROM mimiciv_hosp.patients p
JOIN mimiciv_hosp.admissions a ON p.subject_id = a.subject_id
JOIN ac_patients_first_icu i ON a.hadm_id = i.hadm_id
LEFT JOIN derived_data_summary s ON i.stay_id = s.stay_id
WHERE p.anchor_age >= 18
ORDER BY p.subject_id;
