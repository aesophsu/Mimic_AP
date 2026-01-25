# translation_config.py

TRANSLATION_MAP = {

    # =====================
    # Identifiers & Time
    # =====================
    'subject_id': {'en': 'Subject ID', 'cn': '患者编号', 'unit': None},
    'hadm_id': {'en': 'Hospital Admission ID', 'cn': '住院号', 'unit': None},
    'stay_id': {'en': 'ICU Stay ID', 'cn': 'ICU留滞ID', 'unit': None},

    'intime': {'en': 'ICU Admission Time', 'cn': '入ICU时间', 'unit': None},
    'admittime': {'en': 'Hospital Admission Time', 'cn': '入院时间', 'unit': None},
    'dischtime': {'en': 'Hospital Discharge Time', 'cn': '出院时间', 'unit': None},
    'deathtime': {'en': 'Time of Death', 'cn': '死亡时间', 'unit': None},
    'dod': {'en': 'Date of Death', 'cn': '死亡日期', 'unit': None},

    'los': {'en': 'Length of Stay', 'cn': '住院天数', 'unit': 'days'},

    # =====================
    # Demographics
    # =====================
    'gender': {'en': 'Gender', 'cn': '性别', 'unit': None},
    'admission_age': {'en': 'Age at Admission', 'cn': '入院年龄', 'unit': 'years'},
    'weight_admit': {'en': 'Admission Weight', 'cn': '入院体重', 'unit': 'kg'},
    'height_admit': {'en': 'Admission Height', 'cn': '入院身高', 'unit': 'cm'},
    'bmi': {'en': 'Body Mass Index (BMI)', 'cn': '体质指数(BMI)', 'unit': 'kg/m²'},

    # =====================
    # Outcomes
    # =====================
    'pof': {'en': 'Persistent Organ Failure', 'cn': '持续性器官功能衰竭', 'unit': None},
    'resp_pof': {'en': 'Respiratory POF', 'cn': '呼吸系统POF', 'unit': None},
    'cv_pof': {'en': 'Cardiovascular POF', 'cn': '心血管系统POF', 'unit': None},
    'renal_pof': {'en': 'Renal POF', 'cn': '肾脏系统POF', 'unit': None},

    'mortality_28d': {'en': '28-Day Mortality', 'cn': '28天死亡率', 'unit': None},
    'early_death_24_48h': {'en': 'Early Death (24–48 h)', 'cn': '早期死亡(24–48小时)', 'unit': None},
    'composite_outcome': {'en': 'Composite Outcome', 'cn': '复合终点', 'unit': None},

    # =====================
    # Severity Scores
    # =====================
    'sofa_score': {'en': 'SOFA Score', 'cn': 'SOFA评分', 'unit': None},
    'apsiii': {'en': 'APS-III Score', 'cn': 'APS-III评分', 'unit': None},
    'sapsii': {'en': 'SAPS-II Score', 'cn': 'SAPS-II评分', 'unit': None},
    'oasis': {'en': 'OASIS Score', 'cn': 'OASIS评分', 'unit': None},
    'lods': {'en': 'LODS Score', 'cn': 'LODS评分', 'unit': None},

    # =====================
    # Hematology
    # =====================
    'wbc_min': {'en': 'WBC Count (Min)', 'cn': '白细胞计数(最小值)', 'unit': '×10⁹/L'},
    'wbc_max': {'en': 'WBC Count (Max)', 'cn': '白细胞计数(最大值)', 'unit': '×10⁹/L'},

    'hemoglobin_min': {'en': 'Hemoglobin (Min)', 'cn': '血红蛋白(最小值)', 'unit': 'g/dL'},
    'hemoglobin_max': {'en': 'Hemoglobin (Max)', 'cn': '血红蛋白(最大值)', 'unit': 'g/dL'},

    'hematocrit_min': {'en': 'Hematocrit (Min)', 'cn': '红细胞压积(最小值)', 'unit': '%'},
    'hematocrit_max': {'en': 'Hematocrit (Max)', 'cn': '红细胞压积(最大值)', 'unit': '%'},

    'platelets_min': {'en': 'Platelets (Min)', 'cn': '血小板计数(最小值)', 'unit': '×10⁹/L'},
    'platelets_max': {'en': 'Platelets (Max)', 'cn': '血小板计数(最大值)', 'unit': '×10⁹/L'},

    'rdw_max': {'en': 'RDW (Max)', 'cn': '红细胞分布宽度(最大值)', 'unit': '%'},

    # =====================
    # Biochemistry
    # =====================
    'albumin_min': {'en': 'Albumin (Min)', 'cn': '白蛋白(最小值)', 'unit': 'g/dL'},
    'albumin_max': {'en': 'Albumin (Max)', 'cn': '白蛋白(最大值)', 'unit': 'g/dL'},

    'bun_min': {'en': 'Blood Urea Nitrogen (Min)', 'cn': '尿素氮(最小值)', 'unit': 'mg/dL'},
    'bun_max': {'en': 'Blood Urea Nitrogen (Max)', 'cn': '尿素氮(最大值)', 'unit': 'mg/dL'},

    'creatinine_min': {'en': 'Creatinine (Min)', 'cn': '肌酐(最小值)', 'unit': 'mg/dL'},
    'creatinine_max': {'en': 'Creatinine (Max)', 'cn': '肌酐(最大值)', 'unit': 'mg/dL'},

    'glucose_min': {'en': 'Glucose (Min)', 'cn': '血糖(最小值)', 'unit': 'mg/dL'},
    'glucose_max': {'en': 'Glucose (Max)', 'cn': '血糖(最大值)', 'unit': 'mg/dL'},
    'glucose_lab_min': {'en': 'Lab Glucose (Min)', 'cn': '实验室血糖(最小值)', 'unit': 'mg/dL'},
    'glucose_lab_max': {'en': 'Lab Glucose (Max)', 'cn': '实验室血糖(最大值)', 'unit': 'mg/dL'},
    'glucose_slope': {'en': 'Glucose Slope', 'cn': '血糖变化斜率', 'unit': 'mg/dL per hour'},

    'sodium_min': {'en': 'Sodium (Min)', 'cn': '血钠(最小值)', 'unit': 'mmol/L'},
    'sodium_max': {'en': 'Sodium (Max)', 'cn': '血钠(最大值)', 'unit': 'mmol/L'},
    'potassium_min': {'en': 'Potassium (Min)', 'cn': '血钾(最小值)', 'unit': 'mmol/L'},
    'potassium_max': {'en': 'Potassium (Max)', 'cn': '血钾(最大值)', 'unit': 'mmol/L'},
    'chloride_min': {'en': 'Chloride (Min)', 'cn': '氯离子(最小值)', 'unit': 'mmol/L'},
    'chloride_max': {'en': 'Chloride (Max)', 'cn': '氯离子(最大值)', 'unit': 'mmol/L'},
    'bicarbonate_min': {'en': 'Bicarbonate (Min)', 'cn': '碳酸氢盐(最小值)', 'unit': 'mmol/L'},
    'bicarbonate_max': {'en': 'Bicarbonate (Max)', 'cn': '碳酸氢盐(最大值)', 'unit': 'mmol/L'},
    'aniongap_min': {'en': 'Anion Gap (Min)', 'cn': '阴离子间隙(最小值)', 'unit': 'mmol/L'},
    'aniongap_max': {'en': 'Anion Gap (Max)', 'cn': '阴离子间隙(最大值)', 'unit': 'mmol/L'},

    'calcium_max': {'en': 'Serum Calcium (Max)', 'cn': '血钙(最大值)', 'unit': 'mg/dL'},
    'lab_calcium_min': {'en': 'Serum Calcium (Min)', 'cn': '血钙(最小值)', 'unit': 'mg/dL'},
    'phosphate_min': {'en': 'Phosphate (Min)', 'cn': '血磷(最小值)', 'unit': 'mg/dL'},

    # =====================
    # Liver & Pancreas
    # =====================
    'alt_min': {'en': 'ALT (Min)', 'cn': '丙氨酸氨基转移酶(最小值)', 'unit': 'U/L'},
    'alt_max': {'en': 'ALT (Max)', 'cn': '丙氨酸氨基转移酶(最大值)', 'unit': 'U/L'},
    'ast_min': {'en': 'AST (Min)', 'cn': '天门冬氨酸氨基转移酶(最小值)', 'unit': 'U/L'},
    'ast_max': {'en': 'AST (Max)', 'cn': '天门冬氨酸氨基转移酶(最大值)', 'unit': 'U/L'},
    'alp_min': {'en': 'ALP (Min)', 'cn': '碱性磷酸酶(最小值)', 'unit': 'U/L'},
    'alp_max': {'en': 'ALP (Max)', 'cn': '碱性磷酸酶(最大值)', 'unit': 'U/L'},

    'bilirubin_total_min': {'en': 'Total Bilirubin (Min)', 'cn': '总胆红素(最小值)', 'unit': 'mg/dL'},
    'bilirubin_total_max': {'en': 'Total Bilirubin (Max)', 'cn': '总胆红素(最大值)', 'unit': 'mg/dL'},
    'lipase_max': {'en': 'Lipase (Max)', 'cn': '脂肪酶(最大值)', 'unit': 'U/L'},

    # =====================
    # Coagulation
    # =====================
    'inr_min': {'en': 'INR (Min)', 'cn': '国际标准化比值(最小值)', 'unit': None},
    'inr_max': {'en': 'INR (Max)', 'cn': '国际标准化比值(最大值)', 'unit': None},
    'pt_min': {'en': 'Prothrombin Time (Min)', 'cn': '凝血酶原时间(最小值)', 'unit': 's'},
    'pt_max': {'en': 'Prothrombin Time (Max)', 'cn': '凝血酶原时间(最大值)', 'unit': 's'},
    'ptt_min': {'en': 'Partial Thromboplastin Time (Min)', 'cn': '部分凝血活酶时间(最小值)', 'unit': 's'},
    'ptt_max': {'en': 'Partial Thromboplastin Time (Max)', 'cn': '部分凝血活酶时间(最大值)', 'unit': 's'},

    # =====================
    # Oxygenation & Blood Gas
    # =====================
    'spo2_min': {'en': 'SpO₂ (Min)', 'cn': '血氧饱和度(最小值)', 'unit': '%'},
    'spo2_max': {'en': 'SpO₂ (Max)', 'cn': '血氧饱和度(最大值)', 'unit': '%'},
    'spo2_slope': {'en': 'SpO₂ Slope', 'cn': '血氧饱和度斜率', 'unit': '% per hour'},

    'ph_min': {'en': 'pH (Min)', 'cn': 'pH值(最小值)', 'unit': None},
    'ph_max': {'en': 'pH (Max)', 'cn': 'pH值(最大值)', 'unit': None},

    'pao2fio2ratio_min': {'en': 'PaO₂/FiO₂ Ratio (Min)', 'cn': '氧合指数(最小值)', 'unit': 'mmHg'},
    'pao2fio2ratio_max': {'en': 'PaO₂/FiO₂ Ratio (Max)', 'cn': '氧合指数(最大值)', 'unit': 'mmHg'},

    'lactate_min': {'en': 'Lactate (Min)', 'cn': '乳酸(最小值)', 'unit': 'mmol/L'},
    'lactate_max': {'en': 'Lactate (Max)', 'cn': '乳酸(最大值)', 'unit': 'mmol/L'},
    'lactate_slope': {'en': 'Lactate Slope', 'cn': '乳酸变化斜率', 'unit': 'mmol/L per hour'},

    # =====================
    # Treatments & Comorbidities
    # =====================
    'mechanical_vent_flag': {'en': 'Mechanical Ventilation', 'cn': '机械通气', 'unit': None},
    'vaso_flag': {'en': 'Vasopressor Use', 'cn': '血管活性药物使用', 'unit': None},

    'heart_failure': {'en': 'Heart Failure', 'cn': '心力衰竭', 'unit': None},
    'chronic_kidney_disease': {'en': 'Chronic Kidney Disease', 'cn': '慢性肾病', 'unit': None},
    'malignant_tumor': {'en': 'Malignant Tumor', 'cn': '恶性肿瘤', 'unit': None},
}
