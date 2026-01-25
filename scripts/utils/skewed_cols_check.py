import os
import json
import pandas as pd

# 配置与主脚本保持一致
BASE_DIR = "../../"
INPUT_PATH = os.path.join(BASE_DIR, "data/cleaned/mimic_processed.csv")
ARTIFACTS_JSON = os.path.join(BASE_DIR, "artifacts/features/selected_features.json")
FIG_DIR = os.path.join(BASE_DIR, "results/figures/lasso")

def verify_assets():
    print("🚀 开始资产校验...\n")
    
    # 1. 检查物理文件
    errors = []
    targets = ['pof', 'mortality', 'composite']
    
    # 检查主 JSON
    if not os.path.exists(ARTIFACTS_JSON):
        errors.append(f"❌ 缺失全局特征文件: {ARTIFACTS_JSON}")
    
    # 检查图片
    for t in targets:
        diag_img = os.path.join(FIG_DIR, f"lasso_diag_{t}.png")
        imp_img = os.path.join(FIG_DIR, f"lasso_importance_{t}.png")
        if not os.path.exists(diag_img): errors.append(f"❌ 缺失诊断图: {diag_img}")
        if not os.path.exists(imp_img): errors.append(f"❌ 缺失重要性图: {imp_img}")

    if errors:
        for err in errors: print(err)
        return
    else:
        print("✅ 物理文件校验通过：所有 JSON 和图片均已生成。")

    # 2. 检查内容一致性
    with open(ARTIFACTS_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.read_csv(INPUT_PATH)
    print(f"\n📊 特征一致性分析:")
    print(f"{'Outcome':<12} | {'Selected':<10} | {'Status'}")
    print("-" * 40)
    
    all_selected = []
    for t in targets:
        features = data[t]['features']
        all_selected.extend(features)
        
        # 检查特征是否在 DataFrame 中
        missing_in_df = [f for f in features if f not in df.columns]
        status = "✅ 匹配" if not missing_in_df else f"❌ 缺失 {len(missing_in_df)} 个特征"
        print(f"{t:<12} | {len(features):<10} | {status}")
    
    # 3. 跨结局共性分析 (学术亮点)
    common_features = set(data['pof']['features']) & \
                      set(data['mortality']['features']) & \
                      set(data['composite']['features'])
    
    print(f"\n🔍 跨结局共性特征 (共 {len(common_features)} 个):")
    if common_features:
        print(f"👉 {', '.join(common_features)}")
    else:
        print("👉 无三个结局共有的特征。")

    print("\n🎉 资产校验完成！可以放心进入 06_model_training_main.py")

if __name__ == "__main__":
    verify_assets()
