"""
K-Means Clustering cho Dữ liệu EEG Giấc Ngủ.
Trích xuất đặc trưng, giảm chiều dữ liệu, phân cụm và đánh giá dựa trên chuẩn AASM.
"""

import numpy as np
import pandas as pd
import pickle
import os
import glob
import warnings

from joblib import Parallel, delayed
from scipy.stats import skew, kurtosis, mode
from scipy.signal import welch
from scipy.optimize import linear_sum_assignment

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score,
    normalized_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score, confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import class_dictionary

try:
    from scipy.integrate import trapezoid
except ImportError:
    from numpy import trapz as trapezoid

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.figsize': (12, 8), 'font.size': 12})


def bandpower(epoch, fs, fmin, fmax):
    freqs, psd = welch(epoch, fs)
    idx = (freqs >= fmin) & (freqs <= fmax)
    return trapezoid(psd[idx], freqs[idx]) if np.sum(idx) > 0 else 0.0


def calculate_wcss(X, labels, centroids):
    wcss = 0
    for k in range(len(centroids)):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            wcss += np.sum((cluster_points - centroids[k]) ** 2)
    return wcss


def extract_features(epoch, fs):
    epoch = epoch.flatten()
    std_val = np.std(epoch)
    
    if np.isnan(epoch).any() or np.isinf(epoch).any() or std_val < 1e-6:
        return None
    
    bp_delta = bandpower(epoch, fs, 0.5, 4)
    bp_theta = bandpower(epoch, fs, 4, 8)
    bp_alpha = bandpower(epoch, fs, 8, 12)
    bp_beta = bandpower(epoch, fs, 12, 30)
    
    try:
        skew_val = skew(epoch)
        kurt_val = kurtosis(epoch)
        if not np.isfinite(skew_val) or not np.isfinite(kurt_val):
            return None
    except Exception:
        return None
    
    return {
        "mean": np.mean(epoch), "std": std_val, "var": np.var(epoch),
        "min": np.min(epoch), "max": np.max(epoch), "range": np.ptp(epoch),
        "skew": skew_val, "kurt": kurt_val,
        "bp_delta": bp_delta, "bp_theta": bp_theta,
        "bp_alpha": bp_alpha, "bp_beta": bp_beta,
        "ratio_delta_alpha": bp_delta / (bp_alpha + 1e-10),
        "ratio_theta_beta": bp_theta / (bp_beta + 1e-10),
    }


def cluster_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
    return cm[row_ind, col_ind].sum() / cm.sum(), mapping


def save_fig(output_dir, name):
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, name), dpi=150, bbox_inches='tight')
    plt.close()


def smooth_labels(labels, window_size=3):
    smoothed = labels.copy()
    half_w = window_size // 2
    for i in range(half_w, len(labels) - half_w):
        window = labels[i - half_w : i + half_w + 1]
        most_common = mode(window, keepdims=True)[0][0]
        smoothed[i] = most_common
    return smoothed



def process_single_file(pkl_file):
    try:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            
        X_raw, y_idx, fs = data["x"], data["y"], data["fs"]
        subject = os.path.basename(pkl_file).replace('.pkl', '')
        
        local_features, local_skipped = [], 0
        for j, epoch in enumerate(X_raw):
            feats = extract_features(epoch, fs)
            if feats:
                feats.update({
                    "label": y_idx[j],
                    "label_name": class_dictionary.get(y_idx[j], f"Class_{y_idx[j]}"),
                    "subject_id": subject
                })
                local_features.append(feats)
            else:
                local_skipped += 1
        return local_features, fs, local_skipped, subject, len(X_raw)
    except Exception as e:
        print(f"  [ERROR] {pkl_file}: {e}")
        return [], None, 0, None, 0


def load_and_extract(data_dir, output_dir, n_jobs=-3):
    print("=" * 60 + "\nPHẦN 1: TẢI & TRÍCH XUẤT SONG SONG\n" + "=" * 60)
    pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    if not pkl_files:
        raise FileNotFoundError(f"Không tìm thấy file .pkl nào trong {data_dir}!")

    print(f"Tìm thấy {len(pkl_files)} file. Đang xử lý song song...")
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_file)(f) for f in pkl_files
    )

    all_features, fs_global, skipped = [], None, 0
    print("\nKết quả trích xuất:")
    for feats, fs, n_skip, subj, n_total in results:
        if feats:
            all_features.extend(feats)
            skipped += n_skip
            if fs_global is None: fs_global = fs
            print(f"  {subj}: {len(feats)}/{n_total} epochs")

    df = pd.DataFrame(all_features)
    # MỚI: Lưu file tổng trực tiếp vào thư mục reports/
    df.to_csv(os.path.join(output_dir, "EEG_features_kmeans.csv"), index=False)
    
    print(f"\nTổng epochs: {len(df)} | Bỏ qua (NaN/Lỗi): {skipped} | Số Subject: {len(pkl_files)}")
    for l in sorted(df["label"].unique()):
        print(f"  {class_dictionary.get(l)}: {(df['label']==l).sum()} ({100*(df['label']==l).mean():.1f}%)")
    
    return df


def preprocess_features(df):
    print("\n" + "=" * 60 + "\nPHẦN 2: CHUẨN HÓA DỮ LIỆU (Standard Scaling)\n" + "=" * 60)
    feature_cols = [c for c in df.columns if c not in ["label", "label_name", "subject_id"]]
    X = df[feature_cols].copy()
    
    X = X.replace([np.inf, -np.inf], np.nan)
    valid = ~X.isna().any(axis=1)
    
    df_valid = df[valid].reset_index(drop=True)
    X = X[valid].reset_index(drop=True)
    y_true = df_valid["label"].copy()
    
    scaler = StandardScaler()
    X_scaled_original = scaler.fit_transform(X)
    X_scaled_original = np.nan_to_num(X_scaled_original, nan=0.0)
    
    print(f"Kích thước sau Standard Scaling: {X_scaled_original.shape}")
    return X_scaled_original, df_valid, y_true, scaler, feature_cols


def apply_pca(X_scaled_original):
    print("\n" + "=" * 60 + "\nPHẦN 3: GIẢM CHIỀU BẰNG PCA\n" + "=" * 60)
    pca = PCA(n_components=0.95, random_state=42)
    X_scaled_pca = pca.fit_transform(X_scaled_original)
    print(f"Số lượng đặc trưng gốc: {X_scaled_original.shape[1]} -> Giảm xuống còn: {X_scaled_pca.shape[1]}")
    print(f"Phương sai được giữ lại: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
    return X_scaled_pca, pca


def evaluate_optimal_k(X_scaled, y_true, k_range, sample_size, output_dir):
    print("\n" + "=" * 60 + "\nPHẦN 4: ĐÁNH GIÁ K TỐI ƯU (Elbow + Silhouette)\n" + "=" * 60)
    sample_idx = np.random.choice(len(X_scaled), sample_size, replace=False) if len(X_scaled) > sample_size else np.arange(len(X_scaled))
    
    inertias, sil_scores, wcss_values = [], [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        wcss = calculate_wcss(X_scaled, km.labels_, km.cluster_centers_)
        sil = silhouette_score(X_scaled[sample_idx], km.labels_[sample_idx])
        
        wcss_values.append(wcss)
        inertias.append(km.inertia_)
        sil_scores.append(sil)
        print(f"  K={k:2d}: WCSS={wcss:10.2f} | Inertia={km.inertia_:10.2f} | Silhouette={sil:.4f}")

    pd.DataFrame({'K': list(k_range), 'WCSS': wcss_values, 'Inertia': inertias, 'Silhouette': sil_scores}).to_csv(
        os.path.join(output_dir, "logs", "elbow_silhouette_scores.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(k_range, inertias, 'b-o', lw=2, ms=8)
    axes[0].set_title('Phương pháp Elbow (Elbow Method)')
    axes[0].set_xlabel('Số lượng cụm (K)')
    axes[0].set_ylabel('Inertia')
    
    n_classes = y_true.nunique()
    if n_classes in k_range:
        axes[0].axvline(n_classes, color='r', ls='--', label=f'Target K={n_classes}')
        axes[0].legend()
    
    best_k = list(k_range)[np.argmax(sil_scores)]
    axes[1].plot(k_range, sil_scores, 'g-o', lw=2, ms=8)
    axes[1].axvline(best_k, color='r', ls='--', label=f'Best K={best_k}')
    axes[1].set_title('Phân tích Silhouette')
    axes[1].set_xlabel('Số lượng cụm (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].legend()
    
    save_fig(os.path.join(output_dir, "figures"), "elbow_silhouette_analysis.png")
    
    n_classes = y_true.nunique()
    print(f"\nK gợi ý (theo Silhouette): {best_k} | Số nhãn mục tiêu y khoa: {n_classes}")
    return n_classes 


def run_clustering_and_smoothing(X_scaled, df, k, output_dir):
    print("\n" + "=" * 60 + "\nPHẦN 5: PHÂN CỤM K-MEANS & LÀM MƯỢT THỜI GIAN\n" + "=" * 60)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    clusters_raw = kmeans.labels_
    final_wcss = calculate_wcss(X_scaled, clusters_raw, kmeans.cluster_centers_)
    
    clusters = np.zeros_like(clusters_raw)
    start_idx = 0
    for subj in df["subject_id"].unique():
        n_epochs = (df["subject_id"] == subj).sum()
        end_idx = start_idx + n_epochs
        clusters[start_idx:end_idx] = smooth_labels(clusters_raw[start_idx:end_idx])
        start_idx = end_idx

    changed = (clusters != clusters_raw).sum()
    print(f"Quá trình làm mượt đã điều chỉnh {changed} epochs ({100*changed/len(clusters):.2f}% nhiễu).")
    return clusters, kmeans, final_wcss


def evaluate_metrics(X_scaled, y_true, clusters, k, final_wcss, output_dir):
    print("\n" + "=" * 60 + "\nPHẦN 6: ĐÁNH GIÁ METRICS\n" + "=" * 60)
    metrics = {
        'Silhouette': silhouette_score(X_scaled, clusters),
        'ARI (Adjusted Rand Index)': adjusted_rand_score(y_true, clusters),
        'NMI (Normalized Mutual Info)': normalized_mutual_info_score(y_true, clusters),
    }
    
    for k_metric, v in metrics.items():
        print(f"  {k_metric}: {v:.4f}")

    acc, mapping = cluster_accuracy(y_true, clusters)
    print(f"\nAccuracy (Hungarian Mapping): {acc:.4f} ({acc*100:.2f}%)")
    for c, l in sorted(mapping.items()):
        print(f"  Cụm {c} tương ứng với nhãn -> {class_dictionary.get(l, f'Class_{l}')}")

    metrics.update({'K': k, 'WCSS': final_wcss, 'Accuracy': acc})
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, "logs", "clustering_metrics.csv"), index=False)
    
    return acc, mapping, metrics['Silhouette'], metrics['ARI (Adjusted Rand Index)'], metrics['NMI (Normalized Mutual Info)']


def generate_visualizations(X_pca, X_scaled_original, y_true, clusters, kmeans, scaler, X_cols, mapping, acc, output_dir):
    print("\n" + "=" * 60 + "\nPHẦN 7: TRỰC QUAN HÓA (VISUALIZATIONS)\n" + "=" * 60)
    k = len(np.unique(clusters))
    fig_dir = os.path.join(output_dir, "figures")
    
    # Confusion Matrix
    y_pred = np.array([mapping[c] for c in clusters])
    labels_sorted = sorted(np.unique(y_true))
    names = [class_dictionary.get(l, f'Class_{l}') for l in labels_sorted]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_true, y_pred, labels=labels_sorted), 
                annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names)
    ax.set_title(f'K-Means Confusion Matrix (Độ chính xác: {acc*100:.2f}%)')
    save_fig(fig_dir, "kmeans_confusion_matrix.png")

    # entroids Heatmap
    centroids_original = np.zeros((k, X_scaled_original.shape[1]))
    for i in range(k):
        mask = clusters == i
        if mask.sum() > 0: 
            centroids_original[i] = X_scaled_original[mask].mean(axis=0)

    centroids_df = pd.DataFrame(scaler.inverse_transform(centroids_original), columns=X_cols)
    centroids_df.index = [class_dictionary.get(mapping[i], f'Cụm {i}') for i in range(k)]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(centroids_df, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title("Bản đồ nhiệt trung tâm cụm (Đặc trưng gốc)")
    save_fig(fig_dir, "cluster_centroids_heatmap.png")

    # PCA Scatter Plot
    print("  -> Đang vẽ PCA Scatter Plot (Tuyến tính)...")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=[class_dictionary.get(mapping[c]) for c in clusters], 
                    palette='tab10', s=30, alpha=0.5, edgecolor=None, ax=ax)
    ax.set_title('Phân cụm K-Means (PCA 2D Projection)')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save_fig(fig_dir, "pca_clusters_scatter.png")

    # t-SNE Scatter Plot
    print("  -> Đang chạy t-SNE Scatter Plot (Phi tuyến tính - Có thể mất 1-2 phút)...")
    sample_limit = min(3000, len(X_pca))
    idx = np.random.choice(len(X_pca), sample_limit, replace=False)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_pca[idx])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=[class_dictionary.get(mapping[c]) for c in clusters[idx]], 
                    palette='tab10', s=30, alpha=0.7, edgecolor=None, ax=ax)
    ax.set_title(f'Phân cụm K-Means (t-SNE 2D Projection - {sample_limit} samples)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save_fig(fig_dir, "tsne_clusters_scatter.png")


def analyze_subject_level(df, y_true, clusters, X_scaled, output_dir):
    print("\n" + "=" * 60 + "\nPHẦN 8: PHÂN TÍCH TỔNG QUÁT TRÊN TỪNG BỆNH NHÂN\n" + "=" * 60)
    df["cluster"] = clusters
    
    subj_metrics = []
    for subj in df["subject_id"].unique():
        mask = df["subject_id"] == subj
        c, t = clusters[mask], y_true.values[mask]
        acc_s, _ = cluster_accuracy(t, c)
        subj_metrics.append({"subject_id": subj, "n_epochs": mask.sum(), "accuracy": acc_s})

    subj_df = pd.DataFrame(subj_metrics).sort_values("accuracy", ascending=False)
    
    subj_df.to_csv(os.path.join(output_dir, "logs", "subject_metrics.csv"), index=False)

    print(f"Số bệnh nhân: {len(subj_df)} | Accuracy trung bình: {subj_df['accuracy'].mean():.4f}")
    return subj_df



def run_pipeline(data_dir="data/processed", output_dir="reports", sample_size=10000, k_range=range(2, 15)):
    """
    Hàm thực thi toàn bộ luồng công việc K-Means Clustering.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    df_raw = load_and_extract(data_dir, output_dir)
    X_original, df, y_true, scaler, X_cols = preprocess_features(df_raw)
    X_pca, _ = apply_pca(X_original)
    
    target_k = evaluate_optimal_k(X_pca, y_true, k_range, sample_size, output_dir)
    clusters, kmeans, wcss = run_clustering_and_smoothing(X_pca, df, target_k, output_dir)
    
    acc, mapping, sil, ari, nmi = evaluate_metrics(X_pca, y_true, clusters, target_k, wcss, output_dir)
    
    generate_visualizations(X_pca, X_original, y_true, clusters, kmeans, scaler, X_cols, mapping, acc, output_dir)
    subj_df = analyze_subject_level(df, y_true, clusters, X_pca, output_dir)
    
    print("\n" + "=" * 60 + "\nHOÀN THÀNH PIPELINE!\n" + "=" * 60)
    print(f"Dữ liệu xuất ra tại thư mục: {output_dir}/")
    print(f"[KẾT QUẢ TỔNG QUAN]")
    print(f"  WCSS (Inertia):       {wcss:.2f}")
    print(f"  Silhouette Score:     {sil:.4f}")
    print(f"  Adjusted Rand Index:  {ari:.4f}")
    print(f"  Cluster Accuracy:     {acc*100:.2f}%")
    print(f"  Avg Subject Acc:      {subj_df['accuracy'].mean()*100:.2f}%")

if __name__ == "__main__":

    run_pipeline(data_dir="data/processed", output_dir="reports")