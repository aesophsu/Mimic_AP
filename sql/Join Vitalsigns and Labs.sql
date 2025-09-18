DROP MATERIALIZED VIEW IF EXISTS vitalsign_lab_summary;
CREATE MATERIALIZED VIEW vitalsign_lab_summary AS
SELECT
    fdv.stay_id,
    fdl.hadm_id,
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
    meld.meld AS meld_original,
    meld.meld AS meld_2016
FROM mimiciv_derived.first_day_vitalsign fdv
LEFT JOIN mimiciv_derived.first_day_lab fdl ON fdv.stay_id = fdl.stay_id
LEFT JOIN mimiciv_derived.first_day_bg fdbg ON fdv.stay_id = fdbg.stay_id
LEFT JOIN mimiciv_derived.coagulation co ON fdl.hadm_id = co.hadm_id
LEFT JOIN mimiciv_derived.meld meld ON fdv.stay_id = meld.stay_id;
