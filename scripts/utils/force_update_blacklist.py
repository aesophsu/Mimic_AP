import os
import json

# ===================== 配置路径 =====================
BASE_DIR = "../../"
DICT_PATH = os.path.join(BASE_DIR, "artifacts/features/feature_dictionary.json")

# ===================== 黑名单（不需要 log 的特征） =====================
NO_LOG_FEATURES = {
    'aniongap_min', 'aniongap_max',
    'bicarbonate_min', 'bicarbonate_max',
    'chloride_min', 'chloride_max',
    'sodium_min', 'sodium_max',
    'potassium_min', 'potassium_max',
    'calcium_max', 'lab_calcium_min',
    'ph_min', 'ph_max',
    'pao2fio2ratio_min', 'pao2fio2ratio_max',
    'hemoglobin_min', 'hemoglobin_max',
    'albumin_min', 'albumin_max',
}

def main():
    # 检查文件是否存在
    if not os.path.exists(DICT_PATH):
        print(f"❌ 文件不存在：{DICT_PATH}")
        return

    # 读取字典
    with open(DICT_PATH, 'r', encoding='utf-8') as f:
        feat_dict = json.load(f)

    # 1. 强制修正黑名单特征为 false
    updated_count = 0
    for feature_name in NO_LOG_FEATURES:
        if feature_name in feat_dict:
            current_value = feat_dict[feature_name].get('needs_log_transform', True)
            if current_value is True:
                feat_dict[feature_name]['needs_log_transform'] = False
                updated_count += 1
                print(f"强制修正：{feature_name} → false")

    # 如果有更新，则保存
    if updated_count > 0:
        with open(DICT_PATH, 'w', encoding='utf-8') as f:
            json.dump(feat_dict, f, ensure_ascii=False, indent=4)
        print(f"\n已强制修正 {updated_count} 个黑名单特征为 false")
    else:
        print("\n黑名单特征已全部为 false，无需修正")

    print("文件路径：", DICT_PATH)

    # 2. 输出详细列表：所有特征及其设置
    print("\n" + "="*60)
    print("所有特征的 needs_log_transform 设置一览：")
    print("-"*60)
    print(f"{'特征名称':<30} | {'needs_log_transform':<20}")
    print("-"*60)

    for feature_name in sorted(feat_dict.keys()):
        needs = feat_dict[feature_name].get('needs_log_transform', '未设置')
        print(f"{feature_name:<30} | {needs}")

    print("-"*60)

    # 3. 统计总结
    true_count = sum(1 for v in feat_dict.values() if v.get('needs_log_transform') is True)
    false_count = len(feat_dict) - true_count

    print(f"\n最终统计：")
    print(f"  需要 log (True)：{true_count} 个")
    print(f"  不需要 log (False)：{false_count} 个")
    print(f"  总计：{len(feat_dict)} 个")

    print(f"\n文件已更新/检查完成：{DICT_PATH}")
    print("如需只看 True 的特征，可查看上面详细列表或后续运行 03 脚本的 log 输出。")


if __name__ == "__main__":
    main()
