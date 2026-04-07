import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
# 1. Tải dữ liệu
df = pd.read_csv('pima-indians-diabetes.csv')

# 2. Tiền xử lý: Thay giá trị 0 bằng Median
# Các cột không thể bằng 0 trong y khoa
cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in cols_to_fix:
    # Thay 0 bằng NaN để tính Median chính xác, sau đó lấp đầy lại
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# 3. Xử lý Outliers (Ngoại lai)
# Sử dụng phương pháp IQR để giới hạn (clip) các giá trị quá xa trung tâm
def handle_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    return df

all_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
df = handle_outliers(df, all_features)

# 4. Chia đặc trưng (X) và nhãn (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 5. Split: Chia tập Train/Test (80/20) với stratify=y
# stratify=y giúp tỷ lệ người bệnh/không bệnh đồng đều ở cả 2 tập
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Scaling: Dùng StandardScaler
# Đưa dữ liệu về cùng thang đo (trung bình = 0, độ lệch chuẩn = 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Chuyển thành DataFrame để Phong dễ sử dụng
X_train_final = pd.DataFrame(X_train_scaled, columns=X.columns)
X_train_final['Outcome'] = y_train.values

X_test_final = pd.DataFrame(X_test_scaled, columns=X.columns)
X_test_final['Outcome'] = y_test.values

# 7. Output: Xuất file dữ liệu sạch
X_train_final.to_csv('khu_train_cleaned.csv', index=False)
X_test_final.to_csv('khu_test_cleaned.csv', index=False)

print("--- Hoàn tất Tiền xử lý & Feature Engineering ---")
print(f"Số lượng mẫu tập Train: {len(X_train_final)}")
print(f"Số lượng mẫu tập Test: {len(X_test_final)}")
