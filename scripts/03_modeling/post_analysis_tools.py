import os
import pandas as pd
import joblib

def verify_step_07_assets(base_dir="../../", outcomes=['pof', 'mortality', 'composite']):
    model_root = os.path.join(base_dir, "artifacts/models")
    table_root = os.path.join(base_dir, "results/tables")
    fig_root = os.path.join(base_dir, "results/figures")
    
    print("🔍 开始校验第 07 步资源...\n")
    report = []

    # 1. 校验全局汇总表
    summary_path = os.path.join(model_root, "global_diagnostic_summary.csv")
    table3_path = os.path.join(table_root, "Table3_Clinical_Performance.csv")
    
    files_to_check = [
        ("全局审计汇总", summary_path),
        ("论文最终 Table 3", table3_path)
    ]

    for name, path in files_to_check:
        status = "✅ 存在" if os.path.exists(path) else "❌ 缺失"
        size = f"{os.path.getsize(path)/1024:.1f} KB" if os.path.exists(path) else "N/A"
        print(f"[{status}] {name:<15} | 路径: {path} ({size})")

    # 2. 校验每个结局的具体资产
    print(f"\n{'Outcome':<12} | {'Thresholds':<10} | {'Perf CSV':<10} | {'Diagnostic Plot'}")
    print("-" * 70)
    
    for target in outcomes:
        t_dir = os.path.join(model_root, target)
        f_dir = os.path.join(fig_root, target)
        
        # 检查 thresholds.json
        has_json = os.path.exists(os.path.join(t_dir, "thresholds.json"))
        # 检查内部性能表
        has_csv = os.path.exists(os.path.join(t_dir, "internal_diagnostic_perf.csv"))
        # 检查诊断图 (寻找 07 开头的 png)
        has_plot = any("07_Diagnostic" in f for f in os.listdir(f_dir)) if os.path.exists(f_dir) else False
        
        print(f"{target.upper():<12} | {'OK' if has_json else 'MISSING':<10} | "
              f"{'OK' if has_csv else 'MISSING':<10} | {'OK' if has_plot else 'MISSING'}")

    # 3. 逻辑校验：检查 Table 3 是否包含 (95% CI)
    if os.path.exists(table3_path):
        df_t3 = pd.read_csv(table3_path)
        has_ci = df_t3['AUC (95% CI)'].str.contains(r'\(.*\–.*\)', regex=True).any()
        print(f"\n分析 Table 3 完整性:")
        print(f" - 置信区间抓取成功: {'✅ 是 (已对齐第6步数据)' if has_ci else '⚠️ 否 (仅有点估计值)'}")
        print(f" - 包含结局总数: {df_t3['Outcome'].nunique()}")
        print(f" - 包含人群分组: {df_t3['Group'].unique().tolist()}")

if __name__ == "__main__":
    verify_step_07_assets()
