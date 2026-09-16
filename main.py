import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline

# 1. Đọc dữ liệu từ file CSV
data = pd.read_csv('data.csv')
X = data[['DienTich']]
y = data['GiaNha']

# 2. Chia tập dữ liệu thành Train (65%) và Test (35%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

print("--- 1. MÔ HÌNH BỊ OVERFITTING (Đa thức bậc 8) ---")
# Đa thức bậc 8 uốn lượn quá mức gây ra hiện tượng học thuộc lòng
model_overfit = make_pipeline(PolynomialFeatures(degree=8), LinearRegression())
model_overfit.fit(X_train, y_train)

print(f"R2 trên tập Train (Học): {model_overfit.score(X_train, y_train):.4f}")
print(f"R2 trên tập Test (Thi):   {model_overfit.score(X_test, y_test):.4f} (Âm -> Overfitting nặng)\n")

print("--- 2. KHẮC PHỤC BẰNG CÁCH GIẢM BẬC MÔ HÌNH (Về Bậc 1) ---")
# Hạ bậc đa thức về bậc 1 (đường thẳng đơn giản)
model_simple = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
model_simple.fit(X_train, y_train)

print(f"R2 trên tập Train: {model_simple.score(X_train, y_train):.4f}")
print(f"R2 trên tập Test:  {model_simple.score(X_test, y_test):.4f} (Tổng quát hóa tốt)\n")

print("--- 3. KHẮC PHỤC BẰNG RIDGE REGULARIZATION (Chuẩn hóa L2) ---")
# Áp dụng Ridge để thu nhỏ các hệ số góc bị bùng nổ
model_ridge = make_pipeline(PolynomialFeatures(degree=8), StandardScaler(), Ridge(alpha=10.0))
model_ridge.fit(X_train, y_train)

print(f"R2 trên tập Train (Ridge): {model_ridge.score(X_train, y_train):.4f}")
print(f"R2 trên tập Test (Ridge):  {model_ridge.score(X_test, y_test):.4f}\n")

# 4. Dự báo thử giá nhà 95m2 với mô hình đã khắc phục
dien_tich_moi = pd.DataFrame({'DienTich': [95]})
gia_du_bao = model_simple.predict(dien_tich_moi)[0]
print(f"Dự báo giá nhà diện tích 95m2: {gia_du_bao:.2f} triệu VNĐ")
