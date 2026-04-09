"""
MÔ HÌNH 1D CONVOLUTIONAL NEURAL NETWORK (CNN 1D) CHO DỮ LIỆU EEG
- Input: data/processed/*.pkl (Mỗi epoch có kích thước 7680x1)
- Output: 
    + reports/models/best_model.keras (Trọng số mô hình tốt nhất)
    + reports/logs/cnn_report.txt (Báo cáo chỉ số)
    + reports/figures/cnn_confusion_matrix.png (Ma trận nhầm lẫn số lượng và % )
    + reports/figures/training_curve.png (Đồ thị Loss và Accuracy)
"""

import os
import pickle
import numpy as np
from collections import Counter

from sklearn.model_selection import GroupShuffleSplit # MỚI: Dùng để chia theo bệnh nhân
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import class_dictionary

plt.style.use('seaborn-v0_8-whitegrid')


def load_all_data(folder="data/processed"):
    """
    Tải toàn bộ dữ liệu EEG thô từ các file .pkl đã được tiền xử lý.
    Tạo thêm mảng groups để đánh dấu ID bệnh nhân cho từng epoch (chống rò rỉ dữ liệu).
    """
    print(f"Đang tải dữ liệu từ {folder}...")
    X_list, y_list, group_list = [], [], []
    files = sorted([f for f in os.listdir(folder) if f.endswith(".pkl")])
    
    for f in files:
        subject_id = f.replace(".pkl", "") # Lấy data_10003 làm ID
        filepath = os.path.join(folder, f)
        
        with open(filepath, "rb") as file:
            data = pickle.load(file)
            X_list.append(data["x"])
            y_list.append(data["y"]) 

            group_list.append(np.array([subject_id] * len(data["y"]))) 

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    groups_all = np.concatenate(group_list)
    
    print(f"Tổng số epochs: {len(X_all)}")
    print("Phân bố nhãn tổng thể:", dict(Counter(y_all)))
    return X_all, y_all, groups_all



def build_model(input_shape, n_classes=5):
    model = Sequential(name="EEG_1D_CNN")

    # Block 1: Low-frequency waves
    model.add(Conv1D(filters=64, kernel_size=50, strides=2, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(0.2))

    # Block 2: Mid-frequency
    model.add(Conv1D(filters=128, kernel_size=10, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    # Block 3: High-frequency
    model.add(Conv1D(filters=256, kernel_size=5, activation='relu'))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling1D()) 

    # Classifier
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model



# VISUALIZATION
def plot_training_curves(history, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', lw=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', lw=2)
    axes[0].set_title('Biểu đồ Độ chính xác (Accuracy)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss', lw=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', lw=2)
    axes[1].set_title('Biểu đồ Độ lỗi (Loss)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "training_curve.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels_sorted, fig_dir):
    names = [class_dictionary.get(l, f"Class_{l}") for l in labels_sorted]
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=names, yticklabels=names, ax=axes[0])
    axes[0].set_title("Ma trận nhầm lẫn (Số lượng)", pad=15)
    axes[0].set_ylabel('Nhãn thực tế (True)')
    axes[0].set_xlabel('Nhãn dự đoán (Predicted)')

    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=names, yticklabels=names, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title("Ma trận nhầm lẫn (Tỷ lệ %)", pad=15)
    axes[1].set_xlabel('Nhãn dự đoán (Predicted)')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "cnn_confusion_matrix.png"), dpi=150)
    plt.close()



def run_pipeline(data_dir="data/processed", log_dir="reports/logs", fig_dir="reports/figures", model_dir="reports/models"):

    for d in [log_dir, fig_dir, model_dir]:
        os.makedirs(d, exist_ok=True)


    X, y, groups = load_all_data(data_dir)

    print("\nĐang chia tập dữ liệu (Train: 70%, Val: 15%, Test: 15%)...")
    
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(gss1.split(X, y, groups))
    
    X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
    X_temp, y_temp, groups_temp = X[temp_idx], y[temp_idx], groups[temp_idx]
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss2.split(X_temp, y_temp, groups_temp))
    
    X_val, y_val, groups_val = X_temp[val_idx], y_temp[val_idx], groups_temp[val_idx]
    X_test, y_test, groups_test = X_temp[test_idx], y_temp[test_idx], groups_temp[test_idx]

    print("\nKiểm tra chia bệnh nhân độc lập (Data Leakage Check):")
    print(f"  Train Subjects ({len(np.unique(groups_train))}):", np.unique(groups_train))
    print(f"  Val Subjects   ({len(np.unique(groups_val))}):", np.unique(groups_val))
    print(f"  Test Subjects  ({len(np.unique(groups_test))}):", np.unique(groups_test))

    print(f"\nSố lượng epochs:")
    print(f"  Train: {X_train.shape[0]}")
    print(f"  Val:   {X_val.shape[0]}")
    print(f"  Test:  {X_test.shape[0]}")

    classes_unique = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes_unique, y=y_train)
    class_weights = dict(zip(classes_unique, weights))
    
    model = build_model(input_shape=(X.shape[1], 1))
    print("\nKiến trúc mô hình:")
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            os.path.join(model_dir, "best_model.keras"), 
            monitor='val_loss', 
            save_best_only=True,
            verbose=1
        )
    ]

    # TRAINING
    print("\nBắt đầu huấn luyện mạng nơ-ron...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50, 
        batch_size=64,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    print("\n" + "="*45)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f" ĐỘ CHÍNH XÁC TEST (Subject-independent): {acc*100:.2f}%")
    print("="*45 + "\n")

    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    labels_sorted = sorted(np.unique(y_test))
    names = [class_dictionary[l] for l in labels_sorted]

    report = classification_report(
        y_test, y_pred,
        labels=labels_sorted,
        target_names=names,
        zero_division=0
    )
    print(report)
    
    with open(os.path.join(log_dir, "cnn_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"TEST ACCURACY (Subject-independent): {acc*100:.2f}%\n\n")
        f.write(report)

    print("Đang tạo các biểu đồ đánh giá...")
    plot_training_curves(history, fig_dir)
    plot_confusion_matrix(y_test, y_pred, labels_sorted, fig_dir)

    print(f"\n[HOÀN THÀNH] Đã lưu mô hình tại: {model_dir}/")
    print(f"             Đã lưu báo cáo tại: {log_dir}/ và {fig_dir}/")


if __name__ == "__main__":
    run_pipeline()