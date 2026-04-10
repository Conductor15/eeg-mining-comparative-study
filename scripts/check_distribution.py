import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Định nghĩa dictionary ánh xạ nhãn để hiển thị tên cho đẹp (lấy từ src.config)
class_dictionary = {
    0: "Sleep stage W",
    1: "Sleep stage N1",
    2: "Sleep stage N2",
    3: "Sleep stage N3",
    4: "Sleep stage R",
}

def plot_class_distribution(data_dir="./data/processed", save_dir="./reports/figures"):
    print(f"Đang đọc dữ liệu từ {data_dir}...")
    all_labels = []
    
    # Lấy danh sách các file .pkl
    pkl_files = [f for f in os.listdir(data_dir) if f.endswith(".pkl")]
    
    if not pkl_files:
        print("Không tìm thấy file .pkl nào. Vui lòng chạy prepare_data.py trước!")
        return

    # Duyệt qua từng file và gom nhãn (y) lại
    for f in pkl_files:
        filepath = os.path.join(data_dir, f)
        with open(filepath, "rb") as file:
            data = pickle.load(file)
            all_labels.extend(data['y'])
            
    # Đếm số lượng từng class
    label_counts = Counter(all_labels)
    total_epochs = len(all_labels)
    
    # Sắp xếp theo thứ tự label (0, 1, 2, 3, 4)
    labels_sorted = sorted(label_counts.keys())
    counts = [label_counts[l] for l in labels_sorted]
    names = [class_dictionary.get(l, f"Class {l}") for l in labels_sorted]
    percentages = [(count / total_epochs) * 100 for count in counts]
    
    print(f"Tổng số epochs: {total_epochs}")
    for name, count, pct in zip(names, counts, percentages):
        print(f" - {name}: {count} epochs ({pct:.2f}%)")

    # Vẽ biểu đồ Bar Chart
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(x=names, y=counts, palette="viridis")
    
    # Thêm giá trị cụ thể và % lên trên đỉnh từng cột
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + (max(counts) * 0.01),
                f"{int(height)}\n({percentages[i]:.1f}%)", 
                ha="center", va="bottom", fontsize=11, fontweight='bold')
        
    plt.title("Phân bố nhãn các giai đoạn giấc ngủ (Class Distribution)", fontsize=15, fontweight='bold', pad=20)
    plt.xlabel("Giai đoạn giấc ngủ", fontsize=12)
    plt.ylabel("Số lượng Epochs", fontsize=12)
    plt.ylim(0, max(counts) * 1.15) # Tăng limit trục Y để text không bị cắt
    
    # Tạo thư mục lưu nếu chưa có và lưu hình ảnh
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "class_distribution_bar.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n[HOÀN THÀNH] Đã lưu biểu đồ tại: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    plot_class_distribution(data_dir="./data/processed", save_dir="./reports/figures")