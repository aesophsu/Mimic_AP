import os
import pandas as pd
import numpy as np
import joblib
import optuna

# 机器学习核心库
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve # 将它从 calibration 模块导入# 屏蔽警告
from sklearn.utils import resample # 确保在顶部或此处导入
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =========================================================
# 1. 配置与路径
# =========================================================
BASE_DIR = ".."
INPUT_PATH = os.path.join(BASE_DIR, "data/cleaned/mimic_for_model.csv")
SAVE_DIR = os.path.join(BASE_DIR, "models")
FIG_DIR = os.path.join(BASE_DIR, "figures") # 新增：图片保存目录
for d in [SAVE_DIR, FIG_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

FINAL_SUMMARY_STORAGE = []
def run_module_03_all_outcomes():
    """
    核心控制函数：循环执行不同结局的分析，并汇总最终报表
    """
    global FINAL_SUMMARY_STORAGE
    FINAL_SUMMARY_STORAGE = [] # 清空缓存
    
    # 结局列表：现在包含了死亡率模型
    targets = ['pof', 'composite_outcome', 'mortality_28d']
    
    for current_target in targets:
        print(f"\n\n{'='*30} 正在分析结局: {current_target.upper()} {'='*30}")
        # 获取该结局下的所有模型性能指标
        target_results = train_pipeline(current_target)
        FINAL_SUMMARY_STORAGE.extend(target_results)

    # --- 核心优化：生成全局性能汇总表 ---
    summary_df = pd.DataFrame(FINAL_SUMMARY_STORAGE)
    
    # 按照结局和 AUC 排序，方便查看哪个模型最强
    summary_df = summary_df.sort_values(by=['Outcome', 'Main AUC'], ascending=[True, False])
    
    # 保存汇总表
    summary_save_path = os.path.join(SAVE_DIR, "all_outcomes_performance_summary.csv")
    summary_df.to_csv(summary_save_path, index=False)
    
    print("\n" + "#"*60)
    print("🏆 所有结局分析完成！最终性能汇总表已生成：")
    print(f"📍 路径: {summary_save_path}")
    print("#"*60)
    print(summary_df.to_string(index=False))

def train_pipeline(target):
    print("="*60)
    print("🚀 运行终极重构模块 03: 5 种模型竞赛 + 动态对数处理")
    print("="*60)
    
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 错误: 找不到输入文件 {INPUT_PATH}")
        return
        
    df = pd.read_csv(INPUT_PATH)

    # =========================================================
    # 2. 特征清洗与预处理 (关键：修复文本列报错)
    # =========================================================
    print(f"\n📋 原始数据探测: {df.shape[0]} 行, {df.shape[1]} 列")
    print(f"{'Feature Name':<25} | {'Missing%':<10} | {'Median':<10} | {'Mean':<10} | {'Max':<10}")
    print("-" * 75)
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna() # 排除空值进行计算
            missing = df[col].isnull().mean() * 100
            med = series.median() if not series.empty else 0
            mean = series.mean() if not series.empty else 0
            v_max = series.max() if not series.empty else 0
            print(f"{col:<25} | {missing:>8.2f}% | {med:>10.2f} | {mean:>10.2f} | {v_max:>10.2f}")
            
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'M': 1, 'F': 0})

    outcome_cols = [
        'pof', 'mortality_28d', 'composite_outcome', 
        'renal_pof', 'resp_pof', 'cv_pof'
    ]    
    drop_list = outcome_cols + [
        'subgroup_no_renal', 'hosp_mortality', 'overall_mortality', 'stay_id'
    ]
    

    # 🛡️ 自动剔除非数值特征 (处理 ValueError: could not convert string to float)
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    final_drop = list(set(drop_list + text_cols))
    print(f"🗑️ 自动剔除泄露/非数值特征: {text_cols}")
    
    X = df.drop(columns=[c for c in final_drop if c in df.columns])
    y = df[target]
    subgroup_flag = df['subgroup_no_renal']

    # 处理无穷大并确保数值化
    X = X.replace([np.inf, -np.inf], np.nan).astype(float)
    
    # 划分训练/测试集
    X_train, X_test, y_train, y_test, sub_train, sub_test = train_test_split(
        X, y, subgroup_flag, test_size=0.2, random_state=42, stratify=y
    )
    # --- [修改位置 A]: 划分后的分布审计 ---
    print(f"\n🛡️ 亚组分布平衡审计:")
    print(f"  Train Set: n={len(y_train)}, No-Renal Subgroup={sub_train.sum()} ({sub_train.sum()/len(sub_train):.1%})")
    print(f"  Test Set:  n={len(y_test)}, No-Renal Subgroup={sub_test.sum()} ({sub_test.sum()/len(sub_test):.1%})")
    # =========================================================
    # 3. 🧪 核心修正：动态 Log1p 转换 (救赎线性模型)
    # =========================================================
    skewed_cols = ['creatinine_max', 'creatinine_min', 'bun_max', 'bun_min',
                   'wbc_max', 'wbc_min', 'glucose_max', 'glucose_min',
                   'lipase_max', 'lactate_max',
                   'alt_max', 'ast_max', 'bilirubin_total_max', 
                   'alp_max', 'inr_max', 'rdw_max']
    existing_skewed = [c for c in skewed_cols if c in X_train.columns]
    print(f"\n🔄 正在执行动态 Log1p 转换与量级审计...")
    for col in existing_skewed:
        # 在转换前记录中位数，用于跨库一致性比对
        train_med = X_train[col].median()
        test_med = X_test[col].median()
        print(f"  [Audit] {col:<20}: train_median={train_med:>8.2f}, test_median={test_med:>8.2f}")
        
        # 执行带裁剪的对数转换
        X_train[col] = np.log1p(X_train[col].clip(lower=0))
        X_test[col] = np.log1p(X_test[col].clip(lower=0))

    # =========================================================
    # 4. 增强型多重插补 (MICE) & 标准化
    # =========================================================
    print("🧪 正在执行深度插补 (MICE)...")
    mice_imputer = IterativeImputer(max_iter=20, random_state=42, initial_strategy='median')
    scaler = StandardScaler()

    X_train_imp = mice_imputer.fit_transform(X_train)
    # --- [修改位置 C]: 插补后的质量审计 ---
    # 统计缺失率超过 40% 的特征
    high_missing = X_train.columns[X_train.isnull().mean() > 0.4].tolist()
    if high_missing:
        print(f"⚠️ 插补风险提示: 以下变量缺失率 > 40%，MICE 插补可能引入噪声:\n  {high_missing}")
    X_train_std = scaler.fit_transform(X_train_imp)

    X_test_imp = mice_imputer.transform(X_test)
    X_test_std = scaler.transform(X_test_imp)

    # 保存预处理资产
    joblib.dump(scaler, os.path.join(SAVE_DIR, f"scaler_{target}.pkl"))
    joblib.dump(mice_imputer, os.path.join(SAVE_DIR, f"mice_imputer_{target}.pkl"))
    joblib.dump(existing_skewed, os.path.join(SAVE_DIR, f"skewed_cols_{target}.pkl"))

    # =========================================================
    # 5. LASSO 特征降维 (Top 12) - 学术增强版
    # =========================================================
    print("🧪 正在精选极致核心特征 (Top 12)并生成学术图表...")
    
    # 执行 LassoCV
    lasso = LassoCV(cv=5, random_state=42, max_iter=20000).fit(X_train_std, y_train)
    
    # --- [计算绘图所需指标] ---
    alphas = lasso.alphas_
    log_alphas = np.log10(alphas)
    mse_mean = lasso.mse_path_.mean(axis=1)
    mse_std = lasso.mse_path_.std(axis=1)
    mse_se = mse_std / np.sqrt(lasso.mse_path_.shape[1]) # 标准误
    
    # 找到 Min MSE 和 1-SE 点
    idx_min = np.argmin(mse_mean)
    target_mse = mse_mean[idx_min] + mse_se[idx_min]
    # 1-SE 点：在 idx_min 之后（更简单的模型中）寻找最后一个满足 MSE <= target_mse 的索引
    idx_1se = np.where(mse_mean <= target_mse)[0][0] 

    # 获取特征路径用于顶部计数
    from sklearn.linear_model import lasso_path
    _, coefs_path, _ = lasso_path(X_train_std, y_train, alphas=alphas)
    active_counts = np.sum(coefs_path != 0, axis=0)

    # --- [绘制学术风格 Lasso CV 图] ---
    plt.figure(figsize=(10, 7), dpi=300)
    ax1 = plt.gca()
    
    # 1. 绘制误差棒 (Error Bars)
    ax1.errorbar(log_alphas, mse_mean, yerr=mse_se, fmt='o', color='red', 
                 ecolor='gray', elinewidth=1, capsize=2, mfc='red', ms=5, label='Cross-Validation Error')
    
    # 2. 绘制 Min MSE 线 (蓝) 和 1-SE 线 (黑)
    ax1.axvline(log_alphas[idx_min], color='blue', linestyle='--', label=f'Min Error (n={active_counts[idx_min]})')
    ax1.axvline(log_alphas[idx_1se], color='black', linestyle='--', label=f'1-SE Rule (n={active_counts[idx_1se]})')

    ax1.set_xlabel(r'$\log_{10}(\alpha)$', fontsize=12)
    ax1.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
    ax1.set_title('Lasso Variable Selection with 1-SE Rule', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(alpha=0.3)

    # 3. 添加顶部特征计数轴
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    tick_indices = np.linspace(0, len(log_alphas)-1, 10, dtype=int)
    ax2.set_xticks(log_alphas[tick_indices])
    ax2.set_xticklabels(active_counts[tick_indices])
    ax2.set_xlabel('Number of Non-zero Coefficients', fontsize=12, labelpad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"Academic_Lasso_{target}.png"), dpi=300)
    plt.show()
    plt.close()

    # --- [特征提取保持不变] ---
    coef_abs = np.abs(lasso.coef_)
    indices = np.argsort(coef_abs)[-12:] 
    selected_features = X.columns[indices].tolist()
    
    X_train_final = X_train_std[:, indices]
    X_test_final = X_test_std[:, indices]
    print(f"✅ 特征精简完成: {selected_features}")

    # =========================================================
    # 6. XGBoost Optuna 超参数寻优
    # =========================================================
    print("\n🔬 启动 XGBoost 贝叶斯寻优 (Optuna)...")
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'random_state': 42, 'eval_metric': 'logloss'
        }
        model = XGBClassifier(**param)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        return cross_val_score(model, X_train_final, y_train, cv=cv, scoring='roc_auc').mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)    
    # 持久化 Study 对象
    joblib.dump(study, os.path.join(SAVE_DIR, f"optuna_xgboost_study_{target}.pkl"))
    print(f"✅ Optuna 寻优完成。最佳 AUC: {study.best_value:.4f}")
    
    # 使用最佳参数重新训练
    best_params = study.best_params
    best_xgb = XGBClassifier(**study.best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')

    # =========================================================
    # 7. 🏆 5 种模型算法竞赛 (含概率校准)
    # =========================================================
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, class_weight='balanced'),
        "SVM": SVC(probability=True, kernel='rbf', C=1.0, class_weight='balanced'), 
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, class_weight='balanced'),
        "XGBoost": best_xgb 
    }

    # 准备亚组测试索引
    sub_mask = (sub_test == 1).values
    X_test_sub = X_test_final[sub_mask]
    y_test_sub = y_test.iloc[sub_mask]

    print("\n" + "="*70)
    print(f"{'Algorithm':<20} | {'Main AUC':<10} | {'No-Renal AUC':<10} | {'Brier':<10}")
    print("-" * 70)

    calibrated_results = {}
    for name, model in models.items():
        # 使用概率校准优化 Brier Score
        clf = CalibratedClassifierCV(model, cv=3, method='isotonic')
        clf.fit(X_train_final, y_train)
        
        y_prob = clf.predict_proba(X_test_final)[:, 1]
        auc_main = roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        
        y_prob_sub = clf.predict_proba(X_test_sub)[:, 1]
        auc_sub = roc_auc_score(y_test_sub, y_prob_sub)
        
        calibrated_results[name] = clf
        print(f"{name:<20} | {auc_main:.4f}      | {auc_sub:.4f}          | {brier:.4f}")

    # =========================================================
    # 7.1 统计学增强：Bootstrap 计算 95% CI (全人群 + 亚组)
    # =========================================================
    from sklearn.utils import resample

    def get_auc_ci(model, X_test_data, y_test_data, n_bootstraps=1000):
        """通用的 Bootstrap AUC 置信区间计算函数"""
        bootstrapped_scores = []
        for i in range(n_bootstraps):
            # 使用 i 作为随机种子，确保结果可复现且每次采样不同
            X_res, y_res = resample(X_test_data, y_test_data, random_state=i)
            if len(np.unique(y_res)) < 2: 
                continue
            prob = model.predict_proba(X_res)[:, 1]
            bootstrapped_scores.append(roc_auc_score(y_res, prob))
        
        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()
        # 计算 2.5% 和 97.5% 分位数
        low = sorted_scores[int(0.025 * len(sorted_scores))]
        high = sorted_scores[int(0.975 * len(sorted_scores))]
        return low, high

    print("\n" + "="*110)
    print(f"{'Algorithm':<20} | {'Main AUC (95% CI)':<30} | {'No-Renal AUC (95% CI)':<30} | {'Brier':<8}")
    print("-" * 110)

    for name, clf in calibrated_results.items():
        # --- 1. 全人群指标 ---
        y_prob = clf.predict_proba(X_test_final)[:, 1]
        auc_main = roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        ci_low_m, ci_high_m = get_auc_ci(clf, X_test_final, y_test)
        main_auc_str = f"{auc_main:.3f} ({ci_low_m:.3f}-{ci_high_m:.3f})"
        
        # --- 2. 亚组 (No-Renal) 指标 ---
        # 使用预先准备好的 sub_mask 提取亚组数据
        ci_low_s, ci_high_s = get_auc_ci(clf, X_test_sub, y_test_sub)
        auc_sub = roc_auc_score(y_test_sub, clf.predict_proba(X_test_sub)[:, 1])
        sub_auc_str = f"{auc_sub:.3f} ({ci_low_s:.3f}-{ci_high_s:.3f})"
        
        # --- 3. 打印格式化结果 ---
        print(f"{name:<20} | {main_auc_str:<30} | {sub_auc_str:<30} | {brier:.4f}")
    print("="*110)

    # =========================================================
    # 7.2 性能对比绘图 (单图单文件保存)
    # =========================================================
    def save_final_plots(data_pairs, title_suffix, file_prefix):
        X_data, y_true = data_pairs
        
        # 预先计算所有模型的概率，确保绘图与打印一致
        model_probs = {}
        for name, clf in calibrated_results.items():
            model_probs[name] = clf.predict_proba(X_data)[:, 1]

        # --- 图 A: 纯 ROC 曲线 ---
        plt.figure(figsize=(9, 8))
        for name, y_prob in model_probs.items():
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = roc_auc_score(y_true, y_prob)
            plt.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})', lw=2)
        
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
        plt.title(f'ROC Curves\n({title_suffix})', fontsize=15, fontweight='bold')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.2)
        plt.savefig(os.path.join(FIG_DIR, f"Figure_ROC_{file_prefix}_{target}.png"), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        # --- 图 B: 纯 Calibration 曲线 ---
        plt.figure(figsize=(9, 8))
        for name, y_prob in model_probs.items():
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', label=name, markersize=6, lw=2)
            
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Perfectly Calibrated')
        plt.title(f'Calibration Curves\n({title_suffix})', fontsize=15, fontweight='bold')
        plt.xlabel('Predicted Probability', fontsize=12)
        plt.ylabel('Actual Probability', fontsize=12)
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(alpha=0.2)
        plt.savefig(os.path.join(FIG_DIR, f"Figure_Calib_{file_prefix}_{target}.png"), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        

    # --- 最终执行：生成 4 张独立图片 ---
    print("\n📊 正在生成 4 张独立的论文插图 (ROC & Calibration for Train/Val)...")
    # 验证集图 (对应你终端输出的 0.83 左右)
    save_final_plots((X_test_final, y_test), "Validation Group", "Validation")
    # 训练集图 (对应你看到的 0.90 左右)
    save_final_plots((X_train_final, y_train), "Training Group", "Training")
    # =========================================================
    # 8. 全资产保存 (确保每个 Outcome 独立保存)
    # =========================================================
    # 保存模型字典
    joblib.dump(calibrated_results, os.path.join(SAVE_DIR, f"all_models_{target}.pkl"))
    # --- [新增] 自动保存置信区间 (CI) 审计数据 ---
    ci_audit_data = {}
    sub_ci_audit_data = {}

    for name, clf in calibrated_results.items():
        # 1. 计算全人群 CI
        ci_low_m, ci_high_m = get_auc_ci(clf, X_test_final, y_test)
        auc_main = roc_auc_score(y_test, clf.predict_proba(X_test_final)[:, 1])
        ci_audit_data[name] = f"{auc_main:.3f} ({ci_low_m:.3f}-{ci_high_m:.3f})"
        
        # 2. 计算亚组 CI
        ci_low_s, ci_high_s = get_auc_ci(clf, X_test_sub, y_test_sub)
        auc_sub = roc_auc_score(y_test_sub, clf.predict_proba(X_test_sub)[:, 1])
        sub_ci_audit_data[name] = f"{auc_sub:.3f} ({ci_low_s:.3f}-{ci_high_s:.3f})"

    # 保存 CI 字典，供模块 04 直接调用
    joblib.dump(ci_audit_data, os.path.join(SAVE_DIR, f"ci_main_{target}.pkl"))
    joblib.dump(sub_ci_audit_data, os.path.join(SAVE_DIR, f"ci_sub_{target}.pkl"))
    print(f"📊 {target} 的置信区间数据已自动同步至本地文件。")
    # 保存该结局筛选出的 Top 12 特征名
    joblib.dump(selected_features, os.path.join(SAVE_DIR, f"selected_features_{target}.pkl"))
    
    # 保存测试集数据，方便后续离线做 SHAP 或其他分析
    X_test_final_df = pd.DataFrame(X_test_final, columns=selected_features)
    joblib.dump((X_test_final_df, y_test), os.path.join(SAVE_DIR, f"test_data_main_{target}.pkl"))
    joblib.dump((X_test_sub, y_test_sub), os.path.join(SAVE_DIR, f"test_data_sub_{target}.pkl"))

    # =========================================================
    # 9. 构建最终性能汇总报表
    # =========================================================
    current_outcome_summary = [] # 使用更明确的变量名
    
    for name, clf in calibrated_results.items():
        # 执行 Bootstrap 计算全人群和亚组的 95% CI
        ci_low_m, ci_high_m = get_auc_ci(clf, X_test_final, y_test)
        ci_low_s, ci_high_s = get_auc_ci(clf, X_test_sub, y_test_sub)
        
        # 计算全人群指标
        y_prob = clf.predict_proba(X_test_final)[:, 1]
        auc_main = roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        
        # 计算亚组 (No-Renal) 指标
        y_prob_sub = clf.predict_proba(X_test_sub)[:, 1]
        auc_sub = roc_auc_score(y_test_sub, y_prob_sub)

        # 整理成字典，添加进列表
        current_outcome_summary.append({
            "Outcome": target,
            "Algorithm": name,
            "Main AUC": round(auc_main, 4),
            "Main AUC (95% CI)": f"{auc_main:.3f} ({ci_low_m:.3f}-{ci_high_m:.3f})",
            "No-Renal AUC": round(auc_sub, 4),
            "No-Renal AUC (95% CI)": f"{auc_sub:.3f} ({ci_low_s:.3f}-{ci_high_s:.3f})",
            "Brier Score": round(brier, 4)
        })

    print("-" * 60)
    print(f"✅ 结局 {target.upper()} 分析及资产保存成功！")
    

    train_assets = {
        'medians': X_train.median().to_dict(), # 该结局对应的训练集中位数
        'skewed_cols': existing_skewed,        # 偏态处理列表
        'selected_features': selected_features # 该结局筛选出的 Top 12
    }
    
    # 文件名带上 target 后缀，如 train_assets_pof.pkl
    assets_save_path = os.path.join(SAVE_DIR, f"train_assets_{target}.pkl")
    joblib.dump(train_assets, assets_save_path)

    # 同步保存一份专属特征清单，方便其他模块调用
    joblib.dump(selected_features, os.path.join(SAVE_DIR, f"selected_features_{target}.pkl"))

    print(f"📦 [资产同步] 专属基准已存至: {assets_save_path}")
    print(f"📦 [特征同步] 专属特征清单已存至: selected_features_{target}.pkl")
        
    return current_outcome_summary

if __name__ == "__main__":
    run_module_03_all_outcomes()
