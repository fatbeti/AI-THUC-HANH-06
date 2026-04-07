import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# ============================================================
# 1. LOAD DATA
# ============================================================
columns = [
    "symboling", "normalized_losses", "make", "fuel_type", "aspiration",
    "num_of_doors", "body_style", "drive_wheels", "engine_location",
    "wheel_base", "length", "width", "height", "curb_weight",
    "engine_type", "num_of_cylinders", "engine_size", "fuel_system",
    "bore", "stroke", "compression_ratio", "horsepower",
    "peak_rpm", "city_mpg", "highway_mpg", "price"
]

df = pd.read_csv("auto_imports.csv", names=columns, skiprows=1)
df.replace("?", np.nan, inplace=True)

print("=== HEAD ===")
print(df.head())
print("\n=== SHAPE ===")
print(df.shape)
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())
 
# ============================================================
# 2. TIỀN XỬ LÝ DỮ LIỆU
# ============================================================
# Chuyển kiểu dữ liệu (phải làm trước khi vẽ biểu đồ)
cols_to_numeric = ["normalized_losses", "bore", "stroke", "horsepower", "peak_rpm", "price"]
for col in cols_to_numeric:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Điền missing values
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

num_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

print("\n=== MISSING AFTER IMPUTATION ===")
print(df.isnull().sum())

# ============================================================
# 3. EDA (EXPLORATORY DATA ANALYSIS)
# ============================================================
sns.set(style="whitegrid")

# Phân phối giá
plt.figure(figsize=(8, 5))
sns.histplot(df["price"], kde=True)
plt.title("Distribution of Price")
plt.tight_layout()
plt.show()

# Scatter plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["engine_size", "horsepower", "curb_weight"]):
    ax.scatter(df[col], df["price"], alpha=0.5)
    ax.set_xlabel(col)
    ax.set_ylabel("price")
    ax.set_title(f"{col} vs price")
plt.tight_layout()
plt.show()

# Boxplot giá
plt.figure(figsize=(8, 4))
sns.boxplot(x=df["price"])
plt.title("Boxplot of Price")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# ============================================================
# 4. CHUẨN BỊ DỮ LIỆU
# ============================================================
X = df.drop("price", axis=1)
y = df["price"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# ============================================================
# 5. HUẤN LUYỆN VÀ SO SÁNH NHIỀU MÔ HÌNH
# ============================================================
models = {
    "Linear Regression": LinearRegression(),
    "Ridge":             Ridge(),
    "Lasso":             Lasso(),
    "Decision Tree":     DecisionTreeRegressor(),
    "Random Forest":     RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "KNN":               KNeighborsRegressor(),
    "SVR":               SVR(),
    "MLP":               MLPRegressor(max_iter=500),
    "AdaBoost":          AdaBoostRegressor(),
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    results.append({"Model": name, "MSE": mse, "RMSE": rmse, "R2": r2})

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
print("\n=== MODEL COMPARISON ===")
print(results_df.to_string(index=False))

# ============================================================
# 6. TINH CHỈNH MÔ HÌNH TỐT NHẤT (GRADIENT BOOSTING)
# ============================================================
param_grid_gb = {
    "n_estimators":  [100, 200],
    "learning_rate": [0.01, 0.1],
    "max_depth":     [3, 5],
}

grid_gb = GridSearchCV(
    GradientBoostingRegressor(),
    param_grid_gb,
    cv=5,
    scoring="r2",
    n_jobs=-1
)
grid_gb.fit(X_train, y_train)

print("\n=== GRADIENT BOOSTING — BEST PARAMS ===")
print("Best Params:", grid_gb.best_params_)
print("CV Best R2: ", grid_gb.best_score_)

y_pred_gb = grid_gb.best_estimator_.predict(X_test)
print("Test R2:    ", r2_score(y_test, y_pred_gb))

# ============================================================
# 7. ĐÁNH GIÁ MÔ HÌNH TỐT NHẤT
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Actual vs Predicted
axes[0].scatter(y_test, y_pred_gb, alpha=0.6)
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], "r--", linewidth=2)
axes[0].set_xlabel("Actual Price")
axes[0].set_ylabel("Predicted Price")
axes[0].set_title("Actual vs Predicted (Gradient Boosting)")

# Residual Plot
residuals = y_test - y_pred_gb
axes[1].scatter(y_pred_gb, residuals, alpha=0.6)
axes[1].axhline(y=0, color="r", linestyle="--", linewidth=2)
axes[1].set_xlabel("Predicted Price")
axes[1].set_ylabel("Residuals")
axes[1].set_title("Residual Plot (Gradient Boosting)")

plt.tight_layout()
plt.show()
