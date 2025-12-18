/* ============================================================
REVISED: AP Cohort with 100% Comprehensive Lab & BG Mapping
============================================================ */

DROP TABLE IF EXISTS my_custom_schema.ap_final_analysis_cohort;

CREATE TABLE my_custom_schema.ap_final_analysis_cohort AS

WITH filtered_patients AS (
    SELECT DISTINCT subject_id, hadm_id
    FROM mimiciv_hosp.diagnoses_icd
    WHERE (icd_version = 9 AND icd_code = '5770')
       OR (icd_version = 10 AND icd_code LIKE 'K85%')
),

first_icustay AS (
    SELECT 
        icu.subject_id, icu.hadm_id, icu.stay_id, icu.los, icu.intime,
        ROW_NUMBER() OVER (PARTITION BY icu.subject_id ORDER BY icu.intime) AS rn
    FROM mimiciv_icu.icustays icu
    INNER JOIN filtered_patients p ON p.subject_id = icu.subject_id
    WHERE icu.los >= 1 
),

core_cohort AS (
    SELECT 
        icu.subject_id, icu.hadm_id, icu.stay_id, icu.intime, icu.los,
        adm.admittime, adm.dischtime, adm.deathtime,
        adm.insurance, adm.race, pat.dod, pat.gender,
        (EXTRACT(YEAR FROM icu.intime) - pat.anchor_year + pat.anchor_age) AS admission_age
    FROM first_icustay icu
    INNER JOIN mimiciv_hosp.admissions adm ON icu.hadm_id = adm.hadm_id
    INNER JOIN mimiciv_hosp.patients pat ON icu.subject_id = pat.subject_id
    WHERE icu.rn = 1 
      AND (EXTRACT(YEAR FROM icu.intime) - pat.anchor_year + pat.anchor_age) >= 18
),

vent_status AS (
    SELECT DISTINCT vent.stay_id
    FROM mimiciv_derived.ventilation vent
    INNER JOIN core_cohort c ON vent.stay_id = c.stay_id
    WHERE vent.ventilation_status = 'InvasiveVent'
      AND vent.starttime BETWEEN c.intime AND (c.intime + INTERVAL '24 hours')
),

vaso_status AS (
    SELECT DISTINCT v.stay_id
    FROM mimiciv_derived.vasoactive_agent v
    INNER JOIN core_cohort c ON v.stay_id = c.stay_id
    WHERE (dopamine IS NOT NULL OR epinephrine IS NOT NULL 
       OR norepinephrine IS NOT NULL OR phenylephrine IS NOT NULL 
       OR vasopressin IS NOT NULL)
      AND v.starttime BETWEEN c.intime AND (c.intime + INTERVAL '24 hours')
),

comorbidities AS (
    SELECT 
        d.hadm_id,
        MAX(CASE WHEN icd_code LIKE '428%' OR icd_code LIKE 'I50%' THEN 1 ELSE 0 END) AS heart_failure,
        MAX(CASE WHEN icd_code LIKE '4273%' OR icd_code LIKE 'I48%' THEN 1 ELSE 0 END) AS atrial_fibrillation,
        MAX(CASE WHEN icd_code LIKE '585%' OR icd_code LIKE 'N18%' THEN 1 ELSE 0 END) AS chronic_kidney_disease,
        MAX(CASE WHEN icd_code LIKE '491%' OR icd_code LIKE '492%' OR icd_code LIKE '496%' OR icd_code LIKE 'J44%' OR icd_code LIKE 'J43%' THEN 1 ELSE 0 END) AS copd,
        MAX(CASE WHEN icd_code LIKE '410%' OR icd_code LIKE '411%' OR icd_code LIKE '414%' OR icd_code LIKE 'I20%' OR icd_code LIKE 'I25%' THEN 1 ELSE 0 END) AS coronary_heart_disease,
        MAX(CASE WHEN icd_code LIKE '433%' OR icd_code LIKE '434%' OR icd_code LIKE '436%' OR icd_code LIKE 'I63%' OR icd_code LIKE 'I64%' THEN 1 ELSE 0 END) AS stroke,
        MAX(CASE WHEN icd_code LIKE '14%' OR icd_code LIKE '15%' OR icd_code LIKE 'C%' THEN 1 ELSE 0 END) AS malignant_tumor,
        MAX(CASE WHEN icd_code LIKE '2860%' OR icd_code LIKE 'D66%' THEN 1 ELSE 0 END) AS congenital_coagulation_defects
    FROM mimiciv_hosp.diagnoses_icd d
    INNER JOIN core_cohort c ON d.hadm_id = c.hadm_id 
    GROUP BY d.hadm_id
)

SELECT
    c.*,
    -- Outcome: 28-day mortality
    CASE
        WHEN COALESCE(c.deathtime, c.dod) IS NOT NULL
         AND COALESCE(c.deathtime, c.dod) <= (c.intime + INTERVAL '28 days')
        THEN 1 ELSE 0
    END AS mortality_28d,

    -- Severity Scores
    sofa.sofa AS sofa_score, apsiii.apsiii, sapsii.sapsii, oasis.oasis, lods.lods,

    /* =====================================================
       ALL First-day LAB variables (Mapped from your Schema)
       ===================================================== */
    lab.hematocrit_min AS lab_hematocrit_min, lab.hematocrit_max AS lab_hematocrit_max,
    lab.hemoglobin_min AS lab_hemoglobin_min, lab.hemoglobin_max AS lab_hemoglobin_max,
    lab.platelets_min AS lab_platelets_min, lab.platelets_max AS lab_platelets_max,
    lab.wbc_min AS lab_wbc_min, lab.wbc_max AS lab_wbc_max,
    lab.albumin_min AS lab_albumin_min, lab.albumin_max AS lab_albumin_max,
    lab.globulin_min AS lab_globulin_min, lab.globulin_max AS lab_globulin_max,
    lab.total_protein_min AS lab_total_protein_min, lab.total_protein_max AS lab_total_protein_max,
    lab.aniongap_min AS lab_aniongap_min, lab.aniongap_max AS lab_aniongap_max,
    lab.bicarbonate_min AS lab_bicarbonate_min, lab.bicarbonate_max AS lab_bicarbonate_max,
    lab.bun_min AS lab_bun_min, lab.bun_max AS lab_bun_max,
    lab.calcium_min AS lab_calcium_min, lab.calcium_max AS lab_calcium_max,
    lab.chloride_min AS lab_chloride_min, lab.chloride_max AS lab_chloride_max,
    lab.creatinine_min AS lab_creatinine_min, lab.creatinine_max AS lab_creatinine_max,
    lab.glucose_min AS lab_glucose_min, lab.glucose_max AS lab_glucose_max,
    lab.sodium_min AS lab_sodium_min, lab.sodium_max AS lab_sodium_max,
    lab.potassium_min AS lab_potassium_min, lab.potassium_max AS lab_potassium_max,
    lab.abs_basophils_min AS lab_abs_basophils_min, lab.abs_basophils_max AS lab_abs_basophils_max,
    lab.abs_eosinophils_min AS lab_abs_eosinophils_min, lab.abs_eosinophils_max AS lab_abs_eosinophils_max,
    lab.abs_lymphocytes_min AS lab_abs_lymphocytes_min, lab.abs_lymphocytes_max AS lab_abs_lymphocytes_max,
    lab.abs_monocytes_min AS lab_abs_monocytes_min, lab.abs_monocytes_max AS lab_abs_monocytes_max,
    lab.abs_neutrophils_min AS lab_abs_neutrophils_min, lab.abs_neutrophils_max AS lab_abs_neutrophils_max,
    lab.atyps_min AS lab_atyps_min, lab.atyps_max AS lab_atyps_max,
    lab.bands_min AS lab_bands_min, lab.bands_max AS lab_bands_max,
    lab.imm_granulocytes_min AS lab_imm_granulocytes_min, lab.imm_granulocytes_max AS lab_imm_granulocytes_max,
    lab.metas_min AS lab_metas_min, lab.metas_max AS lab_metas_max,
    lab.nrbc_min AS lab_nrbc_min, lab.nrbc_max AS lab_nrbc_max,
    lab.d_dimer_min AS lab_d_dimer_min, lab.d_dimer_max AS lab_d_dimer_max,
    lab.fibrinogen_min AS lab_fibrinogen_min, lab.fibrinogen_max AS lab_fibrinogen_max,
    lab.thrombin_min AS lab_thrombin_min, lab.thrombin_max AS lab_thrombin_max,
    lab.inr_min AS lab_inr_min, lab.inr_max AS lab_inr_max,
    lab.pt_min AS lab_pt_min, lab.pt_max AS lab_pt_max,
    lab.ptt_min AS lab_ptt_min, lab.ptt_max AS lab_ptt_max,
    lab.alt_min AS lab_alt_min, lab.alt_max AS lab_alt_max,
    lab.alp_min AS lab_alp_min, lab.alp_max AS lab_alp_max,
    lab.ast_min AS lab_ast_min, lab.ast_max AS lab_ast_max,
    lab.amylase_min AS lab_amylase_min, lab.amylase_max AS lab_amylase_max,
    lab.bilirubin_total_min AS lab_bilirubin_total_min, lab.bilirubin_total_max AS lab_bilirubin_total_max,
    lab.bilirubin_direct_min AS lab_bilirubin_direct_min, lab.bilirubin_direct_max AS lab_bilirubin_direct_max,
    lab.bilirubin_indirect_min AS lab_bilirubin_indirect_min, lab.bilirubin_indirect_max AS lab_bilirubin_indirect_max,
    lab.ck_cpk_min AS lab_ck_cpk_min, lab.ck_cpk_max AS lab_ck_cpk_max,
    lab.ck_mb_min AS lab_ck_mb_min, lab.ck_mb_max AS lab_ck_mb_max,
    lab.ggt_min AS lab_ggt_min, lab.ggt_max AS lab_ggt_max,
    lab.ld_ldh_min AS lab_ld_ldh_min, lab.ld_ldh_max AS lab_ld_ldh_max,

    /* =====================================================
       ALL First-day BLOOD GAS variables (Mapped from your Schema)
       ===================================================== */
    bg.lactate_min AS bg_lactate_min, bg.lactate_max AS bg_lactate_max,
    bg.ph_min AS bg_ph_min, bg.ph_max AS bg_ph_max,
    bg.so2_min AS bg_so2_min, bg.so2_max AS bg_so2_max,
    bg.po2_min AS bg_po2_min, bg.po2_max AS bg_po2_max,
    bg.pco2_min AS bg_pco2_min, bg.pco2_max AS bg_pco2_max,
    bg.aado2_min AS bg_aado2_min, bg.aado2_max AS bg_aado2_max,
    bg.aado2_calc_min AS bg_aado2_calc_min, bg.aado2_calc_max AS bg_aado2_calc_max,
    bg.pao2fio2ratio_min AS bg_pao2fio2ratio_min, bg.pao2fio2ratio_max AS bg_pao2fio2ratio_max,
    bg.baseexcess_min AS bg_baseexcess_min, bg.baseexcess_max AS bg_baseexcess_max,
    bg.bicarbonate_min AS bg_bicarbonate_min, bg.bicarbonate_max AS bg_bicarbonate_max,
    bg.totalco2_min AS bg_totalco2_min, bg.totalco2_max AS bg_totalco2_max,
    bg.hematocrit_min AS bg_hematocrit_min, bg.hematocrit_max AS bg_hematocrit_max,
    bg.hemoglobin_min AS bg_hemoglobin_min, bg.hemoglobin_max AS bg_hemoglobin_max,
    bg.carboxyhemoglobin_min AS bg_carboxyhemoglobin_min, bg.carboxyhemoglobin_max AS bg_carboxyhemoglobin_max,
    bg.methemoglobin_min AS bg_methemoglobin_min, bg.methemoglobin_max AS bg_methemoglobin_max,
    bg.temperature_min AS bg_temperature_min, bg.temperature_max AS bg_temperature_max,
    bg.chloride_min AS bg_chloride_min, bg.chloride_max AS bg_chloride_max,
    bg.calcium_min AS bg_calcium_min, bg.calcium_max AS bg_calcium_max,
    bg.glucose_min AS bg_glucose_min, bg.glucose_max AS bg_glucose_max,
    bg.potassium_min AS bg_potassium_min, bg.potassium_max AS bg_potassium_max,
    bg.sodium_min AS bg_sodium_min, bg.sodium_max AS bg_sodium_max,

    -- Interventions & Flags
    CASE WHEN vent.stay_id IS NOT NULL THEN 1 ELSE 0 END AS mechanical_ventilation_flag,
    CASE WHEN vaso.stay_id IS NOT NULL THEN 1 ELSE 0 END AS vaso_flag,

    -- Comorbidities (Coalesced to 0 if null)
    COALESCE(com.heart_failure, 0) AS heart_failure,
    COALESCE(com.atrial_fibrillation, 0) AS atrial_fibrillation,
    COALESCE(com.chronic_kidney_disease, 0) AS chronic_kidney_disease,
    COALESCE(com.copd, 0) AS copd,
    COALESCE(com.coronary_heart_disease, 0) AS coronary_heart_disease,
    COALESCE(com.stroke, 0) AS stroke,
    COALESCE(com.malignant_tumor, 0) AS malignant_tumor,
    COALESCE(com.congenital_coagulation_defects, 0) AS congenital_coagulation_defects

FROM core_cohort c
LEFT JOIN mimiciv_derived.first_day_sofa sofa ON c.stay_id = sofa.stay_id
LEFT JOIN mimiciv_derived.apsiii apsiii ON c.stay_id = apsiii.stay_id
LEFT JOIN mimiciv_derived.sapsii sapsii ON c.stay_id = sapsii.stay_id
LEFT JOIN mimiciv_derived.oasis oasis ON c.stay_id = oasis.stay_id
LEFT JOIN mimiciv_derived.lods lods ON c.stay_id = lods.stay_id
LEFT JOIN mimiciv_derived.first_day_lab lab ON c.stay_id = lab.stay_id
LEFT JOIN mimiciv_derived.first_day_bg bg ON c.stay_id = bg.stay_id
LEFT JOIN vent_status vent ON c.stay_id = vent.stay_id
LEFT JOIN vaso_status vaso ON c.stay_id = vaso.stay_id
LEFT JOIN comorbidities com ON c.hadm_id = com.hadm_id;
