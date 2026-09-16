Dự Án Phân Tích & Xử Lý Overfitting Trong Dự Báo Giá Nhà

Dự án mô phỏng hiện tượng Overfitting (học vẹt) trong mô hình Hồi quy tuyến tính và áp dụng các kỹ thuật khắc phục.

Cấu trúc dự án
 data.csv: Dữ liệu diện tích (m2) và giá nhà (triệu VNĐ).
 main.py: Code Python mô phỏng Overfitting và các giải pháp khắc phục.
 README.md: Tài liệu mô tả dự án và lý thuyết liên quan.

Lý thuyết Overfitting & Kỹ thuật khắc phục
1. Nguyên nhân Overfitting: Sử dụng đa thức bậc cao (`degree=8`) khiến mô hình cố gắng đi qua mọi điểm nhiễu của dữ liệu nhỏ, dẫn đến học thuộc lòng.
2. Kỹ thuật khắc phục:
Train-Test Split: Chia dữ liệu thành tập học (65%) và tập kiểm tra (35%) để phát hiện sai lệch.
Giảm bậc mô hình (Lower Degree): Hạ bậc đa thức về bậc 1 để tìm xu hướng chung.
Ridge Regularization (L2): Phạt các hệ số góc bùng nổ quá lớn để làm mượt đường hồi quy.

