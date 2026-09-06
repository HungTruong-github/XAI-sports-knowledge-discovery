"""Baseline phi đồ thị — Tầng 3(a) trong chiến lược đối sánh.

Mục đích: kiểm chứng giá trị gia tăng THỰC SỰ của biểu diễn đồ thị,
tránh việc kết quả tốt chỉ do đặc trưng dữ liệu chứ không do cấu trúc đồ thị.

Module dự kiến:
    path_features.py  Trích đặc trưng thống kê từ passing path (số lần chuyền,
                      độ dài chuỗi, góc/khoảng cách trung bình...)
    tabular.py        Logistic Regression, Random Forest, ANN
                      (tiền lệ: "Passing path predicts shooting outcome in football")
"""
