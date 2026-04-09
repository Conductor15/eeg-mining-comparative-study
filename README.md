# Khai phá các đặc trưng tiềm ẩn từ EEG đối với các giai đoạn ngủ

Mục tiêu chính của dự án là áp dụng các kỹ thuật Khai phá Dữ liệu (Unsupervised & Supervised Learning) và Deep Learning để phân loại 5 giai đoạn giấc ngủ (Wake, N1, N2, N3, REM) từ tín hiệu điện não đồ (EEG) kênh đơn (C3-M2) dựa trên chuẩn AASM.

## Cấu trúc thư mục (Project Structure)

Dự án được quy hoạch theo chuẩn **Cookiecutter Data Science**:

```text
Data_mining_btl/
├── data/                   <- Dữ liệu 
│   ├── raw/                <- Dữ liệu gốc (.edf, .tsv) từ NCH Sleep Data Bank
│   └── processed/          <- Dữ liệu đã qua tiền xử lý, cắt epoch 30s (.pkl)
│
├── reports/                <- Nơi chứa kết quả tự động sinh ra
│   ├── figures/            <- Biểu đồ Data Pipeline, t-SNE, Confusion Matrix...
│   ├── logs/               <- File báo cáo Text (Classification Report)
│   └── models/             <- Trọng số mô hình đã huấn luyện 
│
├── src/                    <- Mã nguồn chính 
│   ├── config.py           <- Cấu hình tham số và nhãn y khoa 
│   ├── prepare_data.py     <- Lọc băng thông (Bandpass filter) và cắt Epoch
│   ├── visualize.py        <- Trực quan hóa Data Pipeline
│   ├── kmean_clustering.py <- Trích xuất 14 đặc trưng và phân cụm (K-Means / PCA / t-SNE)
│   ├── random_forest.py    <- Phân loại dựa trên đặc trưng thủ công (Random Forest + SMOTE)
│   └── cnn1d.py            <- Mạng nơ-ron tích chập 1D học trực tiếp từ tín hiệu thô
│
├── requirements.txt        <- Danh sách thư viện Python cần thiết
├── Makefile                <- Tự động hóa các dòng lệnh
└── README.md               <- Tài liệu hướng dẫn dự án