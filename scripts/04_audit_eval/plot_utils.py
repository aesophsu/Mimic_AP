import numpy as np
from feature_utils import FeatureFormatter

class PlotUtils:
    def __init__(self, formatter: FeatureFormatter, lang='en'):
        self.formatter = formatter
        self.lang = lang

    def format_feature_labels(self, features, with_unit=True):
        return [
            self.formatter.get_label(f, lang=self.lang, with_unit=with_unit)
            for f in features
        ]

    @staticmethod
    def compute_or_error(or_df):
        left = np.maximum(0, or_df['OR'] - or_df['OR_Lower'])
        right = np.maximum(0, or_df['OR_Upper'] - or_df['OR'])
        return left, right

    @staticmethod
    def compute_or_xlim(or_df):
        x_min = max(0.01, or_df['OR_Lower'].min() * 0.8)
        x_max = min(100, or_df['OR_Upper'].max() * 1.2)
        return x_min, x_max

    @staticmethod
    def format_or_ci(or_val, low, high, digits=2):
        """
        格式化 OR (95% CI) 文本
        """
        fmt = f"{{:.{digits}f}}"
        return f"{fmt.format(or_val)} ({fmt.format(low)}–{fmt.format(high)})"
