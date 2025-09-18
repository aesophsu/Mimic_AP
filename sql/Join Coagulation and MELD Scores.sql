CREATE MATERIALIZED VIEW vls_coag_meld_summary AS
SELECT
    vls.stay_id,
    vls.hadm_id,
    vls.heart_rate,
    vls.temperature,
    vls.mean_arterial_pressure,
    vls.wbc,
    vls.platelet,
    vls.sodium,
    vls.potassium,
    vls.bicarbonate,
    vls.chloride,
    vls.bun,
    vls.lactate,
    vls.creatinine,
    vls.total_bilirubin,
    vls.alb,
    co.inr AS inr,
    co.ptt AS aptt,
    meld.meld AS meld_original,
    meld.meld AS meld_2016
FROM vitalsign_lab_summary vls
LEFT JOIN mimiciv_derived.coagulation co ON vls.hadm_id = co.hadm_id
LEFT JOIN mimiciv_derived.meld meld ON vls.stay_id = meld.stay_id;
