import json
import os
import pandas as pd

class FeatureDictionaryManager:
    def __init__(self, dict_path='../../artifacts/features/feature_dictionary.json'):
        self.dict_path = dict_path
        self.feature_dict = {}
        # 确保目录存在
        os.makedirs(os.path.dirname(self.dict_path), exist_ok=True)
        if os.path.exists(self.dict_path):
            self.load_dict()

    def _get_physiological_presets(self, col):
        """
        核心预设：为常见临床指标自动分配单位和生理极限范围。
        您也可以在生成 JSON 后手动修改这些值。
        """
        presets = {
            # 指标关键字: (单位, 生理最小极限, 生理最大极限)
            'temperature': ("°C", 30.0, 45.0),
            'heart_rate': ("bpm", 20, 250),
            'respiratory_rate': ("bpm", 0, 100),
            'glucose': ("mg/dL", 10, 1500),
            'creatinine': ("mg/dL", 0.1, 25.0),
            'bun': ("mg/dL", 1, 250),
            'lactate': ("mmol/L", 0.1, 35.0),
            'ph': ("units", 6.5, 8.2),
            'amylase': ("IU/L", 0, 15000),
            'lipase': ("IU/L", 0, 15000),
            'spo2': ("%", 40, 100),
            'bilirubin': ("mg/dL", 0, 60),
            'bmi': ("kg/m2", 10, 80),
            'pao2fio2ratio': ("mmHg", 10, 800)
        }
        
        col_lower = col.lower()
        for key, (unit, p_min, p_max) in presets.items():
            if key in col_lower:
                return unit, p_min, p_max
        return "TBD", None, None

    def init_from_dataframe(self, df, overwrite=False):
        if os.path.exists(self.dict_path) and not overwrite:
            print(f"⚠️ 字典已存在，跳过。如需覆盖请设置 overwrite=True")
            return

        for col in df.columns:
            # 1. 自动分类逻辑
            category = 'others'
            if any(trend in col.lower() for trend in ['slope', 'change', 'trend']): category = 'trend'
            elif any(lab in col.lower() for lab in ['max', 'min', 'avg', 'mean']): category = 'lab_test'
            elif col in ['age', 'gender', 'bmi', 'admission_age']: category = 'demographic'
            elif any(out in col.lower() for out in ['pof', 'death', 'mortality', 'outcome']): category = 'outcome'

            # 2. 获取预设的生理阈值与单位
            unit, physio_min, physio_max = self._get_physiological_presets(col)

            # 3. 构造特征元数据
            self.feature_dict[col] = {
                "standard_name": col,
                "mimic_source_col": col,
                "eicu_source_col": "",
                "unit": unit,
                "category": category,
                "is_model_input": True if category not in ['outcome', 'others'] else False,
                "ref_range": {
                    "logical_min": physio_min, # 生理极限最小值
                    "logical_max": physio_max  # 生理极限最大值
                },
                "conversion_factor": 1.0       # 用于跨库对齐时的倍率
            }
        
        self.save_dict()
        print(f"✅ 成功！字典已生成（含生理范围占位符）: {self.dict_path}")

    def load_dict(self):
        with open(self.dict_path, 'r', encoding='utf-8') as f:
            self.feature_dict = json.load(f)

    def save_dict(self):
        with open(self.dict_path, 'w', encoding='utf-8') as f:
            json.dump(self.feature_dict, f, ensure_ascii=False, indent=4)

# --- 独立运行入口 ---
if __name__ == "__main__":
    raw_data_path = '../../data/raw/mimic_raw_data.csv'
    
    if os.path.exists(raw_data_path):
        print(f"读取数据表头: {raw_data_path}")
        df_raw = pd.read_csv(raw_data_path, nrows=5)
        
        manager = FeatureDictionaryManager()
        manager.init_from_dataframe(df_raw, overwrite=True)
        
        print("-" * 30)
        print("💡 下一步操作建议：")
        print("1. 打开 feature_dictionary.json 补全 TBD 的单位。")
        print("2. 针对特殊指标，调整 ref_range 以便在 02_cleaning 脚本中执行自动剔除。")
    else:
        print(f"❌ 错误：找不到文件 {raw_data_path}")
