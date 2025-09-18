DROP MATERIALIZED VIEW IF EXISTS vls_sofa_apsiii_oasis_sapsii_summary;
CREATE MATERIALIZED VIEW vls_sofa_apsiii_oasis_sapsii_summary AS
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
    vss.oasis,
    s.sapsii AS sapsii
FROM vls_sofa_apsiii_oasis_summary vss
LEFT JOIN mimiciv_derived.sapsii s ON vss.stay_id = s.stay_id;
