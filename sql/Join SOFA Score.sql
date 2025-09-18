CREATE MATERIALIZED VIEW vls_sofa_summary AS
SELECT
    vcm.stay_id,
    vcm.hadm_id,
    vcm.heart_rate,
    vcm.temperature,
    vcm.mean_arterial_pressure,
    vcm.wbc,
    vcm.platelet,
    vcm.sodium,
    vcm.potassium,
    vcm.bicarbonate,
    vcm.chloride,
    vcm.bun,
    vcm.lactate,
    vcm.creatinine,
    vcm.total_bilirubin,
    vcm.alb,
    vcm.inr,
    vcm.aptt,
    vcm.meld_original,
    vcm.meld_2016,
    s.sofa_24hours AS sofa
FROM vls_coag_meld_summary vcm
LEFT JOIN mimiciv_derived.sofa s ON vcm.stay_id = s.stay_id;
