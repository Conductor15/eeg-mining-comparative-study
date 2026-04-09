.PHONY: setup prepare visualize kmeans rf cnn all clean

# Cài đặt môi trường
setup:
	pip install -r requirements.txt

# 1. Tiền xử lý dữ liệu
prepare:
	python -m src.prepare_data

# 2. Trực quan hóa Pipeline
visualize:
	python -m src.visualize

# 3. Chạy mô hình Học không giám sát (Trích xuất đặc trưng & Gom cụm)
kmeans:
	python -m src.kmean_clustering

# 4. Chạy mô hình Học có giám sát (Machine Learning truyền thống)
rf:
	python -m src.random_forest

# 5. Chạy mô hình Deep Learning (Học tự động từ tín hiệu thô)
cnn:
	python -m src.cnn1d

# Chạy toàn bộ luồng từ đầu đến cuối
all: prepare visualize kmeans rf cnn

# Dọn dẹp các file rác và thư mục cache
clean:
	rm -rf reports/ outputs_rf/ outputs_cnn/ outputs/
	find . -type d -name "__pycache__" -exec rm -rf {} +