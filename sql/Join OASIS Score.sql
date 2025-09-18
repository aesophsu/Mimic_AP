DROP MATERIALIZED VIEW IF EXISTS vls_sofa_apsiii_oasis_summary;
CREATE MATERIALIZED VIEW vls_sofa_apsiii_oasis_summary AS
SELECT
    vss.stay_id,
    vss.hadm_id,
    vss.heart_rate,
    vss.temperature,
    vss.mean_arterial_pressure,
    vss.wbc,
    vss.platelet,
    vss.sodium,
    vss.potassium,
    vss.bicarbonate,
    vss.chloride,
    vss.bun,
    vss.lactate,
    vss.creatinine,
    vss.total_bilirubin,
    vss.alb,
    vss.inr,
    vss.aptt,
    vss.meld_original,
    vss.meld_2016,
    vss.sofa,
    vss.apsiii,
    o.oasis AS oasis
FROM vls_sofa_apsiii_summary vss
LEFT JOIN mimiciv_derived.oasis o ON vss.stay_id = o.stay_id;
