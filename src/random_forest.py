"""
RANDOM FOREST CLASSIFICATION FOR EEG (5 CLASSES)
- Input: reports/EEG_features_kmeans.csv (Đặc trưng trích xuất từ K-Means)
- Output:
    + reports/logs/rf_report.txt (Báo cáo các chỉ số phân loại)
    + reports/figures/rf_confusion_matrix.png (Ma trận nhầm lẫn)
    + reports/figures/rf_feature_importance.png (Mức độ quan trọng đặc trưng)
"""

import os
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import GroupShuffleSplit  
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE, RandomOverSampler

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import class_dictionary

plt.style.use('seaborn-v0_8-whitegrid')


def load_data_from_csv(csv_path="reports/EEG_features_kmeans.csv"):
    """
    Tải dữ liệu đặc trưng đã được trích xuất từ file CSV.
    Giữ lại cột subject_id để phục vụ việc chia tập Train/Test độc lập.
    """
    print(f"Loading features from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file: {csv_path}. Vui lòng chạy K-Means pipeline trước.")

    df = pd.read_csv(csv_path)
    
    # Xử lý NaN/Inf 
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    
    cols_to_drop = ["label", "label_name"] 
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    
    X = df.drop(columns=existing_drops)
    y = df["label"]
    
    print("\nPhân bố nhãn toàn bộ tập dữ liệu:")
    for label_idx, count in y.value_counts().items():
        print(f"  {class_dictionary.get(label_idx, label_idx)}: {count}")
        
    return X, y


def train_random_forest(X, y, log_dir="reports/logs", fig_dir="reports/figures"):
    """
    Huấn luyện mô hình Random Forest kết hợp SMOTE.
    Chia dữ liệu theo GroupShuffleSplit để chống Data Leakage.
    """
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("\nĐang chia tập dữ liệu theo bệnh nhân độc lập (Train: 70%, Val: 15%, Test: 15%)...")
    
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(gss1.split(X, y, groups=X['subject_id']))
    
    X_train_full = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    
    X_temp_full = X.iloc[temp_idx]
    y_temp = y.iloc[temp_idx]
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss2.split(X_temp_full, y_temp, groups=X_temp_full['subject_id']))

    X_test_full = X_temp_full.iloc[test_idx]
    y_test  = y_temp.iloc[test_idx]
    
    print(f"  Số bệnh nhân tập Train (70%): {X_train_full['subject_id'].nunique()} -> {X_train_full['subject_id'].unique()}")
    print(f"  Số bệnh nhân tập Test (15%):  {X_test_full['subject_id'].nunique()} -> {X_test_full['subject_id'].unique()}")
    
    X_train = X_train_full.drop(columns=['subject_id'])
    X_test  = X_test_full.drop(columns=['subject_id'])

    print("\nPhân bố nhãn tập Train (Trước khi cân bằng):", dict(Counter(y_train)))

    # NORMALIZE
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) 

    # SMOTE
    min_count = min(Counter(y_train).values())
    if min_count <= 1:
        sampler = RandomOverSampler(random_state=42)
        print("Sử dụng RandomOverSampler do có lớp quá ít mẫu.")
    else:
        k = min(5, min_count - 1)
        sampler = SMOTE(k_neighbors=k, random_state=42)
        print(f"Sử dụng thuật toán SMOTE (k={k}) để cân bằng dữ liệu.")

    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_scaled, y_train)
    print("Phân bố nhãn tập Train (Sau khi cân bằng):", dict(Counter(y_train_resampled)))

    print("\nĐang huấn luyện mô hình Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight='balanced', 
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_resampled, y_train_resampled)
    
    y_pred = rf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "="*45)
    print(f" ĐỘ CHÍNH XÁC TEST (Subject-independent): {acc*100:.2f}%")
    print("="*45 + "\n")

    labels_sorted = sorted(np.unique(np.concatenate([y_test, y_pred])))
    names = [class_dictionary.get(l, f"Class_{l}") for l in labels_sorted]

    report = classification_report(
        y_test, y_pred,
        labels=labels_sorted,
        target_names=names,
        zero_division=0
    )
    print(report)
    
    with open(os.path.join(log_dir, "rf_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"TEST ACCURACY (Subject-independent): {acc*100:.2f}%\n\n")
        f.write(report)

    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names)
    plt.title(f"Random Forest Confusion Matrix (Accuracy: {acc*100:.2f}%)", pad=15)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "rf_confusion_matrix.png"), dpi=150)
    plt.close()

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [X_train.columns[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=sorted_importances, y=sorted_features, hue=sorted_features, palette="viridis", legend=False)
    plt.title("Mức độ quan trọng của các đặc trưng (Feature Importance) - Random Forest")
    plt.xlabel("Mức độ quan trọng (Importance Score)")
    plt.ylabel("Đặc trưng (Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "rf_feature_importance.png"), dpi=150)
    plt.close()

    print(f"Đã lưu báo cáo tại: {log_dir}/ và biểu đồ tại {fig_dir}/")


def run_pipeline(csv_path="reports/EEG_features_kmeans.csv", log_dir="reports/logs", fig_dir="reports/figures"):
    """
    Thực thi luồng công việc cho mô hình Random Forest.
    """
    try:
        X, y = load_data_from_csv(csv_path)
        train_random_forest(X, y, log_dir, fig_dir)
    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình chạy: {e}")


if __name__ == "__main__":
    run_pipeline()