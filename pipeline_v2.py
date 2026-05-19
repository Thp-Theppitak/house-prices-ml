import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv("Data/train.csv")

# แบ่ง train/test ก่อนเลย — ทำอันนี้เป็นอย่างแรกเสมอ
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# เลือก features ที่ดีจาก Day 5
FEATURES = ['OverallQual', 'GrLivArea', 'GarageCars',
            'TotalBsmtSF', 'FullBath', 'YearBuilt',
            'LotArea', 'MasVnrArea']
TARGET = 'SalePrice'
X_train = train_df[FEATURES]
y_train = train_df[TARGET]
X_test = test_df[FEATURES]
y_test = test_df[TARGET]
#สร้าง pipeline 
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    random_state=42))
])


# fit — imputer และ scaler เรียนจาก train เท่านั้น
pipe.fit(X_train,y_train)

# วัดผล
train_preds = pipe.predict(X_train)
test_preds = pipe.predict(X_test)

train_rmse = np.sqrt(((train_preds - y_train) ** 2).mean())
test_rmse  = np.sqrt(((test_preds  - y_test)  ** 2).mean())
print(f"Train RMSE: ${train_rmse:,.0f}")
print(f"Test RMSE : ${test_rmse:,.0f}")

