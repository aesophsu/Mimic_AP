import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from scipy.stats import norm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# =========================================================
# 0. 全局配置与路径
# =========================================================
RANDOM_STATE = 42
MAX_LASSO_FEATURES = 20  # 特征数量上限
N_BOOTSTRAPS = 1000      # 计算CI时的重采样次数

BASE_DIR = ".."
DATA_PATH = os.path.join(BASE_DIR, "data/ap_final_analysis_cohort.csv")
SAVE_DIR_FIG = os.path.join(BASE_DIR, "figures/final_robust")
SAVE_DIR_MODEL = os.path.join(BASE_DIR, "models")
SAVE_DIR_FEAT = os.path.join(BASE_DIR, "features")

for d in [SAVE_DIR_FIG, SAVE_DIR_MODEL, SAVE_DIR_FEAT]:
    os.makedirs(d, exist_ok=True)

print(f"✅ Environment Configured. Random State: {RANDOM_STATE}")

# =========================================================
# 1. 核心工具函数 (Metrics & Statistics)
# =========================================================

def get_auc_ci(y_true, y_prob, n_bootstraps=1000, rng_seed=42):
    """
    使用 Bootstrapping 计算 AUC 的 95% 置信区间
    """
    rng = np.random.RandomState(rng_seed)
    bootstrapped_scores = []

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    for i in range(n_bootstraps):
        # 重采样索引
        indices = rng.randint(0, len(y_prob), len(y_prob))
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        score = roc_auc_score(y_true[indices], y_prob[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.sort(bootstrapped_scores)
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    return confidence_lower, confidence_upper

def find_optimal_threshold(y_true, y_prob):
    """
    基于 Youden's Index (J = Sensitivity + Specificity - 1) 寻找最佳阈值
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    
    # 在最佳阈值下的指标
    y_pred = (y_prob >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    
    return best_thresh, sens, spec

def delong_auc_test(y_true, p1, p2):
    """DeLong Test 计算 P-value"""
    def auc_var(y, p):
        pos = p[y == 1]
        neg = p[y == 0]
        v = [(pi > nj) + 0.5*(pi == nj) for pi in pos for nj in neg]
        return np.var(v) / (len(pos)*len(neg))

    auc1 = roc_auc_score(y_true, p1)
    auc2 = roc_auc_score(y_true, p2)
    var = auc_var(y_true, p1) + auc_var(y_true, p2)
    z = (auc1 - auc2) / np.sqrt(var + 1e-8)
    p = 2 * (1 - norm.cdf(abs(z)))
    return auc1 - auc2, p

# =========================================================
# 2. 数据准备
# =========================================================
df = pd.read_csv(DATA_PATH)
TARGET = "pof"

# 排除逻辑
BASE_EXCLUDE = [
    TARGET, "resp_pof", "cv_pof", "renal_pof", "mortality_28d",
    "sofa_score", "apsiii", "sapsii", "oasis", "lods",
    "mechanical_vent_flag", "vaso_flag", "los",
    "stay_id", "hadm_id", "subject_id",
    "intime", "admittime", "dischtime", "deathtime", "dod",
    "race", "insurance", "language" 
]
SENSITIVITY_EXCLUDE = [
    "creatinine_min", "creatinine_max", 
    "bun_min", "bun_max", 
    "chloride_min", "chloride_max"
]

df_filtered = df.drop(columns=[c for c in BASE_EXCLUDE if c in df.columns])
df_numeric = pd.get_dummies(df_filtered, drop_first=True)

all_features = df_numeric.columns.tolist()
sensitivity_features = [c for c in all_features if c not in SENSITIVITY_EXCLUDE]

experiments = {
    "Main_Analysis": all_features,
    "Sensitivity_No_Renal": sensitivity_features
}

# =========================================================
# 3. 实验主循环
# =========================================================
results_store = {}

print(f"{'='*80}")
print(f"{'MODEL TRAINING & EVALUATION REPORT':^80}")
print(f"{'='*80}")
print(f"{'Exp':<15} | {'Model':<12} | {'AUC (95% CI)':<22} | {'Thresh':<6} | {'Sens':<6} | {'Spec':<6} | {'Brier':<6}")
print("-" * 80)

for exp_name, feature_list in experiments.items():
    
    X = df_numeric[feature_list]
    y = df[TARGET]

    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )

    # Preprocessing Pipeline
    preprocess = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    X_train_s = pd.DataFrame(preprocess.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_s = pd.DataFrame(preprocess.transform(X_test), columns=X_test.columns, index=X_test.index)

    # --- LASSO Selection ---
    lasso = LogisticRegressionCV(
        Cs=10, cv=5, penalty="l1", solver="saga", scoring="roc_auc", 
        class_weight="balanced", max_iter=5000, n_jobs=-1, random_state=RANDOM_STATE
    )
    lasso.fit(X_train_s, y_train)
    
    coef = pd.Series(lasso.coef_[0], index=X_train.columns)
    selected_feats = coef[coef != 0].sort_values(key=np.abs, ascending=False).head(MAX_LASSO_FEATURES).index.tolist()
    
    # Save Feature List
    pd.Series(selected_feats).to_csv(os.path.join(SAVE_DIR_FEAT, f"Features_{exp_name}.csv"), index=False)

    X_train_l = X_train_s[selected_feats]
    X_test_l = X_test_s[selected_feats]

    # --- SMOTE ---
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_l, y_train)

    # --- Model Definition (Base Models) ---
    base_models = {
        "XGBoost": XGBClassifier(
            n_estimators=500, max_depth=3, learning_rate=0.02, gamma=1.0, 
            subsample=0.7, colsample_bytree=0.7, min_child_weight=5, 
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5, 
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Logistic": LogisticRegression(
            max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE
        )
    }

    # --- Training Loop with Calibration ---
    for model_name, base_model in base_models.items():
        
        # 1. Calibrated Classifier (Isotonic)
        # 注意：为了避免数据泄露，我们在 SMOTE 数据上拟合，并在内部 CV 中进行校准
        calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        calibrated_model.fit(X_train_res, y_train_res)
        
        # 2. Prediction
        y_prob = calibrated_model.predict_proba(X_test_l)[:, 1]
        
        # 3. Metrics Calculation
        auc = roc_auc_score(y_test, y_prob)
        ci_low, ci_high = get_auc_ci(y_test, y_prob, n_bootstraps=N_BOOTSTRAPS)
        brier = brier_score_loss(y_test, y_prob)
        
        # 4. Optimal Threshold
        best_thresh, sens, spec = find_optimal_threshold(y_test, y_prob)
        
        # 5. Print Row
        print(f"{exp_name[:15]:<15} | {model_name:<12} | {auc:.3f} ({ci_low:.3f}-{ci_high:.3f}) | {best_thresh:.3f}  | {sens:.3f}  | {spec:.3f}  | {brier:.3f}")

        # 6. Store Results
        key = f"{exp_name}_{model_name}"
        results_store[key] = {
            "y_true": y_test,
            "y_prob": y_prob,
            "auc": auc,
            "model": calibrated_model, # Store calibrated model
            "base_model_for_shap": base_model, # Store base model structure
            "X_test": X_test_l,
            "X_train_res": X_train_res, # For SHAP refit
            "y_train_res": y_train_res  # For SHAP refit
        }

        # 7. Save Artifacts (Only Main Analysis XGBoost)
        if exp_name == "Main_Analysis" and model_name == "XGBoost":
            joblib.dump(calibrated_model, os.path.join(SAVE_DIR_MODEL, 'final_xgb_model.pkl'))
            joblib.dump(preprocess, os.path.join(SAVE_DIR_MODEL, 'scaler.pkl'))
            with open(os.path.join(SAVE_DIR_MODEL, 'feature_list_main.txt'), 'w') as f:
                for feat in selected_feats:
                    f.write(f"{feat}\n")

# =========================================================
# 4. 统计与绘图
# =========================================================

# --- A. DeLong Test (XGB Main vs XGB No-Renal) ---
print("\n" + "="*80)
print("STATISTICAL SIGNIFICANCE (DeLong Test)")
print("-" * 80)
delta, p_val = delong_auc_test(
    results_store["Main_Analysis_XGBoost"]["y_true"],
    results_store["Main_Analysis_XGBoost"]["y_prob"],
    results_store["Sensitivity_No_Renal_XGBoost"]["y_prob"]
)
print(f"Main XGB vs. No-Renal XGB: ΔAUC = {delta:.4f}, P-value = {p_val:.4e}")

# --- B. ROC Comparison Plot ---
plt.figure(figsize=(10, 8))

styles = {
    "Main_Analysis_XGBoost": {"c": "#d62728", "ls": "-", "lbl": "XGBoost (Main)"},
    "Main_Analysis_Logistic": {"c": "#ff7f0e", "ls": ":", "lbl": "Logistic (Main)"},
    "Sensitivity_No_Renal_XGBoost": {"c": "#1f77b4", "ls": "--", "lbl": "XGBoost (No-Renal)"}
}

for key, s in styles.items():
    res = results_store[key]
    fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
    plt.plot(fpr, tpr, color=s["c"], ls=s["ls"], lw=2.5, 
             label=f"{s['lbl']} (AUC={res['auc']:.3f})")

# Add SOFA
test_idx = results_store["Main_Analysis_XGBoost"]["y_true"].index
sofa_scores = df.loc[test_idx, "sofa_score"].fillna(0)
fpr_s, tpr_s, _ = roc_curve(results_store["Main_Analysis_XGBoost"]["y_true"], sofa_scores)
sofa_auc = roc_auc_score(results_store["Main_Analysis_XGBoost"]["y_true"], sofa_scores)
plt.plot(fpr_s, tpr_s, 'k-.', alpha=0.6, label=f"SOFA Score (AUC={sofa_auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
plt.title("Calibrated Model Performance Comparison", fontsize=14)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR_FIG, "Final_ROC_Calibrated.png"), dpi=300)

# --- C. Calibration Plot (Reliability Diagram) ---
plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")

for key in ["Main_Analysis_XGBoost", "Main_Analysis_Logistic"]:
    res = results_store[key]
    prob_true, prob_pred = calibration_curve(res["y_true"], res["y_prob"], n_bins=10)
    plt.plot(prob_pred, prob_true, "s-", label=f"{key.replace('Main_Analysis_', '')}")

plt.title("Calibration Curve (Reliability Diagram)")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR_FIG, "Final_Calibration_Curve.png"), dpi=300)

# --- D. SHAP (需重新拟合 Raw Model) ---
print("\n--- Generating SHAP Plot for Main XGBoost ---")
main_res = results_store["Main_Analysis_XGBoost"]
# CalibratedCV 不支持 TreeExplainer，我们需要用其内部的 base_estimator 或重新拟合一个用于解释
# 为了严谨，我们提取 CalibratedClassifierCV 内部拟合得最好的一个基模型，或者重新拟合原始数据
shap_model = main_res["base_model_for_shap"]
shap_model.fit(main_res["X_train_res"], main_res["y_train_res"]) # Quick refit for interpretation

explainer = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(main_res["X_test"])

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, main_res["X_test"], show=False, max_display=15)
plt.title("SHAP Summary (Main XGBoost)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR_FIG, "Final_SHAP_Summary.png"), dpi=300)

print(f"\n✅ Final Pipeline Completed. Results ready in {SAVE_DIR_FIG}")
