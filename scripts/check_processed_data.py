import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Từ điển ánh xạ nhãn
class_dictionary = {
    0: "Sleep stage W",
    1: "Sleep stage N1",
    2: "Sleep stage N2",
    3: "Sleep stage N3",
    4: "Sleep stage R",
}

def check_all_data_and_plot(data_dir="data/processed", output_dir="reports/figures"):
    """
    Quét toàn bộ thư mục, kiểm tra các file .pkl, 
    tổng hợp số liệu và vẽ biểu đồ mất cân bằng lớp tổng thể.
    """
    os.makedirs(output_dir, exist_ok=True)
    pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    
    if not pkl_files:
        print(f"Không tìm thấy file .pkl nào trong {data_dir}")
        return

    print(f"=== ĐANG KIỂM TRA {len(pkl_files)} FILE DỮ LIỆU ===")
    
    total_epochs = 0
    overall_counts = Counter()
    
    for filepath in pkl_files:
        filename = os.path.basename(filepath)
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            
        x_data = data['x']
        y_data = data['y']
        
        # Tính toán nhanh để check lỗi Z-score
        mean_val = x_data.mean()
        std_val = x_data.std()
        n_epochs = len(y_data)
        total_epochs += n_epochs
        
        # Cập nhật số lượng nhãn vào bộ đếm tổng
        labels, counts = np.unique(y_data, return_counts=True)
        for l, c in zip(labels, counts):
            overall_counts[l] += c
            
        # In log rút gọn cho từng file
        print(f"[{filename:^15}] Epochs: {n_epochs:<5} | Mean: {mean_val:>6.4f} | Std: {std_val:>6.4f}")

    # ==========================================
    # TỔNG KẾT VÀ VẼ BIỂU ĐỒ (VISUALIZATION)
    # ==========================================
    print("\n" + "="*45)
    print("TỔNG KẾT TOÀN BỘ DỮ LIỆU (OVERALL SUMMARY)")
    print("="*45)
    print(f"Tổng số files: {len(pkl_files)}")
    print(f"Tổng số epochs: {total_epochs}")
    
    print("\nPhân bố nhãn tổng thể (Total Label Distribution):")
    # Sắp xếp theo thứ tự nhãn từ 0 -> 4
    sorted_labels = sorted(overall_counts.keys())
    counts_array = [overall_counts[l] for l in sorted_labels]
    label_names = [class_dictionary.get(l, f'Class {l}') for l in sorted_labels]
    
    for name, count in zip(label_names, counts_array):
        percentage = (count / total_epochs) * 100
        print(f" - {name}: {count} epochs ({percentage:.2f}%)")

    # Bắt đầu vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    
    # Tự động chọn dải màu đẹp
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(sorted_labels)))
    
    bars = plt.bar(label_names, counts_array, color=colors)
    plt.title(f"Phân bố các giai đoạn giấc ngủ trên toàn bộ {len(pkl_files)} bệnh nhân\n(Tổng: {total_epochs} epochs)", pad=15, fontsize=14)
    plt.ylabel("Số lượng Epochs", fontsize=12)
    plt.xlabel("Giai đoạn ngủ", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ghi số lượng và phần trăm (%) lên đầu mỗi cột
    for bar, count in zip(bars, counts_array):
        yval = bar.get_height()
        percentage = (count / total_epochs) * 100
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(counts_array)*0.02), 
                 f"{count}\n({percentage:.1f}%)", ha='center', va='bottom', fontweight='bold')
                 
    plt.tight_layout()
    
    # Lưu ảnh
    img_path = os.path.join(output_dir, "overall_class_imbalance.png")
    plt.savefig(img_path, dpi=150)
    print(f"\n[HOÀN THÀNH] Đã lưu biểu đồ tổng thể tại: {img_path}")

if __name__ == "__main__":
    check_all_data_and_plot()