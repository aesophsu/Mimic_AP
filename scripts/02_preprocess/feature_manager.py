import json
import os
import pandas as pd

class FeatureDictionaryManager:
    def __init__(self, dict_path='../../artifacts/features/feature_dictionary.json'):
        self.dict_path = dict_path
        self.feature_dict = {}
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.dict_path), exist_ok=True)
        
        # 如果文件已存在，直接加载
        if os.path.exists(self.dict_path):
            self.load_dict()

    def init_from_dataframe(self, df, overwrite=False):
        """
        初次运行：扫描 DataFrame 列名并生成初始 JSON 骨架
        """
        if os.path.exists(self.dict_path) and not overwrite:
            print(f"⚠️ 字典已存在于 {self.dict_path}，跳过初始化。如需覆盖请设置 overwrite=True")
            return

        for col in df.columns:
            # 这里的逻辑可以根据你的命名习惯自定义
            category = 'unknown'
            if 'slope' in col: category = 'trend'
            elif 'max' in col or 'min' in col: category = 'lab_test'
            elif col in ['age', 'gender', 'bmi']: category = 'demographic'

            self.feature_dict[col] = {
                "mimic_source_col": col,
                "eicu_source_col": "",  # 留待后续手动或自动映射
                "unit": "check_needed",  # 需要手动核对单位
                "category": category,
                "description": f"Description for {col}",
                "is_model_input": True
            }
        
        self.save_dict()
        print(f"✅ 初始字典已生成并保存至: {self.dict_path}")

    def load_dict(self):
        with open(self.dict_path, 'r', encoding='utf-8') as f:
            self.feature_dict = json.load(f)

    def save_dict(self):
        with open(self.dict_path, 'w', encoding='utf-8') as f:
            json.dump(self.feature_dict, f, ensure_ascii=False, indent=4)

    def apply_standardization(self, df):
        """
        清洗阶段调用：根据字典定义的 mimic_source_col 进行重命名和列过滤
        """
        # 1. 检查数据集中是否存在字典定义的列
        available_cols = [c for c in self.feature_dict.keys() if c in df.columns]
        
        # 2. 可以在这里加入单位换算的占位逻辑
        # for col in available_cols:
        #     if self.feature_dict[col]['unit'] == 'some_unit':
        #         df[col] = df[col] * factor
        
        return df[available_cols]

# --- 使用示例 (在 02_mimic_cleaning.py 中) ---
if __name__ == "__main__":
    # 假设你刚读入 SQL 导出的 csv
    # df = pd.read_csv('../../data/raw/mimic_raw_data.csv')
    
    # 初始化管理器
    # manager = FeatureDictionaryManager()
    
    # 如果是第一次跑，扫描列名生成 JSON
    # manager.init_from_dataframe(df)
    
    pass
