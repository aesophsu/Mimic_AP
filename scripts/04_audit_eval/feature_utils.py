import json
import os
import re  # 导入正则模块
from typing import Dict, Any

class FeatureFormatter:
    """
    Feature metadata formatter for labeling, units, ranges, and categories.
    已增强：自动支持 LaTeX 数学下标渲染 (如 PaO2 -> $PaO_2$)
    """

    def __init__(self, dict_path: str = "../../artifacts/features/feature_dictionary.json"):
        self.dict_path = dict_path
        self.feature_dict: Dict[str, Dict[str, Any]] = self._load_dict()

    # ======================
    # IO
    # ======================
    def _load_dict(self) -> Dict[str, Dict[str, Any]]:
        """加载 JSON 特征字典"""
        if not os.path.exists(self.dict_path):
            print(f"\033[93m[!] 警告: 未找到字典文件 {self.dict_path}\033[0m")
            return {}

        try:
            with open(self.dict_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"\033[91m[!] 字典解析失败: {e}\033[0m")
            return {}

    # ======================
    # Internal Logic (New!)
    # ======================
    def _apply_latex_subscripts(self, text: str) -> str:
        """
        将文本中的医学/化学缩写转换为 LaTeX 下标格式。
        例如: PaO2 -> $PaO_2$, CO2 -> $CO_2$, FiO2 -> $FiO_2$
        """
        if not text:
            return text
        
        # 匹配 字母+数字 的模式，例如 PaO2, FiO2, SpO2, HCO3
        # ([a-zA-Z]+) 捕获字母部分，(\d+) 捕获数字部分
        # \1_\2 转换为 LaTeX 的 下标格式
        # 最后用 $ 符号包裹触发 Matplotlib 的渲染引擎
        pattern = r'([a-zA-Z]+)(\d+)'
        
        # 如果检测到这种模式，且不在现有的 $ 范围内，则转换
        if re.search(pattern, text):
            # 替换逻辑：只针对常见的医学缩写进行替换，避免破坏普通文本
            medical_terms = ['PaO', 'FiO', 'SaO', 'SpO', 'CO', 'HCO', 'PO', 'PCO']
            for term in medical_terms:
                # 匹配 term + 数字，例如 PaO + 2
                text = re.sub(f'({term})(\d+)', r'$\1_\2$', text)
        
        return text

    # ======================
    # Core utilities
    # ======================
    @staticmethod
    def _is_incomplete(value: Any) -> bool:
        """判断字段是否缺失或待补充"""
        if value is None:
            return True
        value = str(value)
        return value == "" or "(待补充)" in value or value == "MISSING"

    @staticmethod
    def _default_name(raw_name: str) -> str:
        """raw_name -> Title Case fallback"""
        return raw_name.replace("_", " ").title()

    def _get_feature(self, raw_name: str) -> Dict[str, Any]:
        """安全获取特征字典"""
        return self.feature_dict.get(raw_name, {})

    # ======================
    # Public API
    # ======================
    def get_label(self, raw_name: str, lang: str = "en", with_unit: bool = False) -> str:
        """
        获取格式化后的特征名称，并自动处理 LaTeX 下标
        """
        feat = self._get_feature(raw_name)

        # --- 1. display name ---
        field = "display_name_en" if lang == "en" else "display_name_cn"
        display_name = feat.get(field)

        if self._is_incomplete(display_name):
            display_name = feat.get(
                "standard_name",
                self._default_name(raw_name)
            )

        # --- 2. apply latex subscript (New!) ---
        # 在拼接单位前或后对名称进行 LaTeX 转换
        display_name = self._apply_latex_subscripts(display_name)

        # --- 3. append unit ---
        if with_unit:
            unit = self.get_unit(raw_name)
            if self._is_valid_unit(unit):
                # 单位也可能需要 LaTeX，如 10^9/L
                unit = self._apply_latex_powers(unit) 
                display_name = f"{display_name} ({unit})"

        return display_name

    def _apply_latex_powers(self, text: str) -> str:
        """处理单位中的上标，如 10^9 -> $10^9$"""
        if '^' in text:
            # 将 10^9 替换为 $10^9$
            return re.sub(r'(\d+)\^(\d+)', r'$\1^\2$', text)
        return text

    def get_unit(self, raw_name: str) -> str:
        """获取特征单位"""
        return self._get_feature(raw_name).get("unit", "")

    # ... 其余 get_ref_range, get_category, _is_valid_unit 保持不变 ...
    def get_ref_range(self, raw_name: str) -> Dict[str, Any]:
        return self._get_feature(raw_name).get("ref_range", {"logical_min": None, "logical_max": None})

    def get_category(self, raw_name: str) -> str:
        return self._get_feature(raw_name).get("category", "others")

    @staticmethod
    def _is_valid_unit(unit: Any) -> bool:
        if unit is None: return False
        unit = str(unit).strip().lower()
        return unit not in {"", "none", "id"}
