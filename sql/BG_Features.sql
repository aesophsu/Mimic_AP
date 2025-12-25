DROP TABLE IF EXISTS mimiciv_derived.custom_ap_bg_art;
CREATE TABLE mimiciv_derived.custom_ap_bg_art AS
WITH bg_stg AS (
    SELECT
        ie.stay_id,
        bg.charttime,
        bg.ph,
        bg.pco2,
        bg.baseexcess,
        bg.pao2fio2ratio,
        bg.lactate,
        bg.calcium,
        bg.glucose,
        -- 为计算斜率(Slope)准备：记录入科后的分钟数
        DATETIME_DIFF(bg.charttime, ie.intime, MINUTE) as offset_mins
    FROM `physionet-data.mimiciv_icu.icustays` ie
    LEFT JOIN `physionet-data.mimiciv_derived.bg` bg
        ON ie.subject_id = bg.subject_id
        -- 遵循官方标签：ART. 代表动脉
        AND bg.specimen = 'ART.' 
        -- 遵循官方时间窗：入科前6h 至 入科后24h
        AND bg.charttime >= DATETIME_SUB(ie.intime, INTERVAL '6' HOUR)
        AND bg.charttime <= DATETIME_ADD(ie.intime, INTERVAL '24' HOUR)
)
SELECT
    stay_id,
    -- 1. 酸碱平衡：取极值反映最严重的酸/碱中毒
    MIN(ph) AS ph_min,
    MAX(baseexcess) AS be_max, -- 反映代谢性碱中毒或代偿
    MIN(baseexcess) AS be_min, -- 反映代谢性酸中毒（POF核心指标）
    
    -- 2. 呼吸功能：取最低值反映最严重的换气障碍
    MIN(pao2fio2ratio) AS pao2fio2_min,
    
    -- 3. 组织灌注：取最高值
    MAX(lactate) AS lactate_max,
    
    -- 4. 胰腺炎特异性：最低钙值对 AP 预后极度重要
    MIN(calcium) AS calcium_min,
    
    -- 5. 趋势特征 (Slope) 计算：(最后一次值 - 第一次值) / 时间间隔
    -- 这种写法能比单纯的 MAX/MIN 捕捉到更早的恶化信号
    (MAX(CASE WHEN rn_last = 1 THEN lactate END) - MAX(CASE WHEN rn_first = 1 THEN lactate END)) / 
    NULLIF(MAX(CASE WHEN rn_last = 1 THEN offset_mins END) - MAX(CASE WHEN rn_first = 1 THEN offset_mins END), 0) AS lactate_slope,
    
    (MAX(CASE WHEN rn_last = 1 THEN glucose END) - MAX(CASE WHEN rn_first = 1 THEN glucose END)) / 
    NULLIF(MAX(CASE WHEN rn_last = 1 THEN offset_mins END) - MAX(CASE WHEN rn_first = 1 THEN offset_mins END), 0) AS glucose_slope

FROM (
    SELECT 
        *,
        ROW_NUMBER() OVER(PARTITION BY stay_id ORDER BY charttime ASC) as rn_first,
        ROW_NUMBER() OVER(PARTITION BY stay_id ORDER BY charttime DESC) as rn_last
    FROM bg_stg
) t
GROUP BY stay_id;
