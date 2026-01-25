import json
import os
from typing import Dict, Any

# ======================
# Load translation config
# ======================
try:
    from translation_config import TRANSLATION_MAP
except ImportError:
    print("\033[91m[!] 错误: 请确保 translation_config.py 在当前目录下\033[0m")
    TRANSLATION_MAP: Dict[str, Dict[str, Any]] = {}

PATH = "../../artifacts/features/feature_dictionary.json"


class FeatureDictionaryManager:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.data = self._load_json()

    # ---------- IO ----------
    def _load_json(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(self.json_path):
            print(f"\033[91m[!] 错误: 未找到文件 {self.json_path}\033[0m")
            return {}
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_json(self) -> None:
        if not self.data:
            return
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)
        print(f"\n\033[92m[*] 成功保存更新至: {self.json_path}\033[0m")

    # ---------- Utilities ----------
    @staticmethod
    def _needs_completion(cn_name: str) -> bool:
        return (not cn_name) or "(待补充)" in cn_name or cn_name == "MISSING"

    @staticmethod
    def _default_en_name(key: str) -> str:
        return key.replace("_", " ").title()

    # ---------- Display ----------
    def _print_table(self, items, title: str, color_code: str = "94") -> None:
        print(f"\n\033[{color_code}m" + "=" * 130)
        print(f" {title} ".center(130, "="))
        print(f"{'Key':<30} | {'EN Name':<40} | {'CN Name':<25} | {'Unit':<15}")
        print("-" * 130 + "\033[0m")

        for key, info in items:
            en = info.get("display_name_en", "MISSING")
            cn = info.get("display_name_cn", "MISSING")
            unit = info.get("unit", "")
            row_color = "\033[93m" if self._needs_completion(cn) else ""
            print(
                f"{row_color}{key:<30} | {en:<40} | {cn:<25} | {unit:<15}\033[0m"
            )

        print("\033[" + color_code + "m" + "=" * 130 + "\033[0m\n")

    # ---------- Core logic ----------
    def inject_and_check(self, translation_map: Dict[str, Dict[str, Any]]) -> bool:
        """
        Inject EN/CN display names and units from translation_map,
        and perform completeness audit.
        """
        print("\n\033[94m>>> 正在分析特征字典（翻译 + 单位）...\033[0m")

        cnt_update = cnt_pending = cnt_skip = cnt_unit = 0

        for key, feature in self.data.items():
            lower_key = key.lower()
            current_cn = feature.get("display_name_cn", "")

            # ===== Case A: Explicit mapping exists =====
            if lower_key in translation_map:
                mapping = translation_map[lower_key]

                if mapping.get("en"):
                    feature["display_name_en"] = mapping["en"]
                    feature["display_name"] = mapping["en"]

                if mapping.get("cn"):
                    feature["display_name_cn"] = mapping["cn"]

                cnt_update += 1

                # --- unit injection (idempotent) ---
                if "unit" in mapping and feature.get("unit") != mapping["unit"]:
                    feature["unit"] = mapping["unit"]
                    cnt_unit += 1

            # ===== Case B: Missing / placeholder translation =====
            elif self._needs_completion(current_cn):
                def_en = self._default_en_name(key)
                feature.update(
                    {
                        "display_name_en": def_en,
                        "display_name_cn": f"{def_en}(待补充)",
                        "display_name": def_en,
                    }
                )
                cnt_pending += 1

            # ===== Case C: Already complete =====
            else:
                cnt_skip += 1

        print(
            f"[*] 注入摘要: "
            f"\033[92m翻译更新 {cnt_update}\033[0m, "
            f"\033[96m单位注入 {cnt_unit}\033[0m, "
            f"\033[93m待补充 {cnt_pending}\033[0m, "
            f"\033[37m无需改动 {cnt_skip}\033[0m"
        )

        # ===== Final audit =====
        missing_items = [
            (k, v)
            for k, v in self.data.items()
            if self._needs_completion(v.get("display_name_cn", ""))
        ]

        if missing_items:
            self._print_table(
                missing_items,
                f" 剩余 {len(missing_items)} 个特征需要完善翻译或单位 ",
                "93",
            )
            return False

        print("\033[92m✨ 所有特征翻译与单位均已完成！\033[0m")
        return True


# ======================
# Entrypoint
# ======================
if __name__ == "__main__":
    manager = FeatureDictionaryManager(PATH)
    all_finished = manager.inject_and_check(TRANSLATION_MAP)
    manager.save_json()

    if not all_finished:
        print(
            "\033[93m[提示] 请根据上方 Key，"
            "在 translation_config.py 中补充缺失的翻译或单位。\033[0m"
        )
