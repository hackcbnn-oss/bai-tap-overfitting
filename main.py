import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Đọc dữ liệu từ file CSV
data = pd.read_csv('data.csv')
X = data[['DienTich']]
y = data['GiaNha']

# 2. Khởi tạo và huấn luyện mô hình Hồi quy tuyến tính
model = LinearRegression()
model.fit(X, y)

# 3. In kết quả phương trình hồi quy: y = ax + b
a = model.coef_[0]
b = model.intercept_
print(f"Phương trình hồi quy: Giá nhà = {a:.2f} * Diện tích + {b:.2f}")

# 4. Dự báo giá nhà cho một diện tích mới (ví dụ: 95 m2)
dien_tich_moi = [[95]]
gia_du_bao = model.predict(dien_tich_moi)[0]
print(f"Dự báo giá nhà diện tích 95 m2 là: {gia_du_bao:.2f} triệu VNĐ")
