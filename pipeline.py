import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split , GridSearchCV

class eda_pipeline:
    FEATURES = [ 'OverallQual', 'GrLivArea', 'ExterQual', 'KitchenQual',
                'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF',
                'BsmtQual', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt',
                'FireplaceQu', 'YearRemodAdd'
                    ]
    TARGET = 'SalePrice'



    def __init__(self, model_type: str = 'forest'):
        self.model_type = model_type
        self.model = None 
        self.is_fitted = False
        self.metrics = {}

        if model_type == 'forest':
            self.model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
        elif model_type == 'linear':
            self.model = LinearRegression() 
        else:
            raise ValueError(f"Unknown model_type: {model_type}")


    def _preprocess(self, df):
        df = df.copy()

        # เติมค่าหาย
        none_cols = ['PoolQC', 'MiscFeature', 'Alley',
                     'Fence', 'FireplaceQu', 'GarageType',
                     'GarageFinish', 'GarageQual', 'GarageCond',
                     'BsmtQual', 'BsmtCond', 'BsmtExposure',
                     'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']
        for col in none_cols:
            if col in df.columns:
                df[col] = df[col].fillna("None")

        zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars',
                     'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                     'TotalBsmtSF', 'MasVnrArea']
        for col in zero_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        if 'Electrical' in df.columns:
            df['Electrical'] = df['Electrical'].fillna(
                df['Electrical'].mode()[0])

        if 'LotFrontage' in df.columns:
            df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage']\
                .transform(lambda x: x.fillna(x.median()))
            
        # แปลงคุณภาพเป็นตัวเลข
        quality_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
        for col in ['ExterQual', 'KitchenQual', 'BsmtQual', 'FireplaceQu']:
            if col in df.columns:
                df[col] = df[col].map(quality_map)
        return df

    def fit(self, df):
        df_clean = self._preprocess(df)  # ทำความสะอาดก่อน
        x = df_clean[self.FEATURES]  # เลือกเฉพาะ features ที่เราต้องการใช้ในการ train
        y = df_clean[self.TARGET]  # เลือก target ที่เราต้องการทำนาย
        self.model.fit(x, y)  # train model ด้วยข้อมูลที่เตรียมไว้
        self.is_fitted = True  # เปลี่ยนสถานะว่า model ถูก train แล้ว

        preds = self.model.predict(x)  # ทำนายราคาบ้านด้วยข้อมูล train เพื่อประเมินผล
        rmse = np.sqrt(((preds - y) ** 2).mean())  # คำนวณ RMSE เพื่อดูว่า model fit ขนาดไหน (ยิ่งน้อยยิ่งดี)
        self.metrics['rmse'] = round(rmse, 2)  # เก็บ metric ไว้ใน dictionary เพื่อดูทีหลัง
        print(f"NaN in x: {x.isnull().sum().sum()}")
        print(x.isnull().sum()[x.isnull().sum() > 0])
        return self  # คืนตัวเองเพื่อให้สามารถเรียก method อื่นๆ ต่อได้ เช่น .fit(df).predict(df_test)
    
    def predict(self, df):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        df_clean = self._preprocess(df)  # ทำความสะอาดก่อนเสมอ
        x = df_clean[self.FEATURES]  # เลือก features เดียวกับตอน train
        return self.model.predict(x)  # คืน array ของราคาที่ทำนาย

    def summary(self):
        print(f"Model  : {self.model_type}")
        print(f"Fitted : {self.is_fitted}")
        if self.metrics:
            print(f"RMSE   : ${self.metrics['rmse']:,.0f}")
        # RMSE = 25000 แปลว่า "ทำนายผิดไปเฉลี่ย $25,000 ต่อหลัง"
    
    def __repr__(self):
        state = "fitted" if self.is_fitted else "not fitted"
        return f"HousePricePipeline(model={self.model_type}, {state})"
    def evaluate(self, df):
        """วัด RMSE บน data ที่ไม่เคยเห็น"""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before evaluate()")
    
        df_clean = self._preprocess(df)
        X = df_clean[self.FEATURES]
        y = df_clean[self.TARGET]
    
        preds = self.model.predict(X)
        rmse = np.sqrt(((preds - y) ** 2).mean())
        return round(rmse, 2)
    
    def tune(self, df, cv: int = 10):
        """หา hyperparameters ที่ดีที่สุดด้วย GridSearchCV"""
        df_clean = self._preprocess(df)
        X = df_clean[self.FEATURES]
        y = df_clean[self.TARGET]
       
        param_grid = {
        'n_estimators':     [100, 200, 300, 400, 500],
        'max_depth':        [5, 10, 15, 20, 25, 30, 35, 50, None],
        'min_samples_leaf': [1, 2, 5, 10, 15],
        'min_samples_split':[2, 5, 10, 15],
    }
        
        search = GridSearchCV(
            estimator = RandomForestRegressor(random_state=42),
            param_grid = param_grid,
            cv = cv,
            scoring = 'neg_root_mean_squared_error',
            n_jobs = -1,
            verbose = 1
            )   
        
        print("กำลัง tune...")
        search.fit(X,y)
        print(f"\n✓ Best params: {search.best_params_}")
        print(f"✓ Best RMSE  : ${-search.best_score_:,.0f}")
      # อัพเดท model เป็นตัวที่ดีที่สุด
        self.model = search.best_estimator_
        self.is_fitted = True
    
        return search.best_params_


# ── รันจริง ──────────────────────────────
df = pd.read_csv("Data/train.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print("=== Random Forest ===")
rf = eda_pipeline(model_type="forest")
rf.fit(train_df)
rf.summary()

print("\n=== Linear Regression ===")
lr = eda_pipeline(model_type="linear")
lr.fit(train_df)
lr.summary()

# ทำนายราคา
preds = rf.predict(test_df)
print(f"\nตัวอย่างราคาที่ทำนาย: {preds[:5]}")
print("\n=== Test RMSE (ของจริง) ===")
print(f"Random Forest : ${rf.evaluate(test_df):,.0f}")
print(f"Linear        : ${lr.evaluate(test_df):,.0f}")

# Tune
print("\n=== Tuning Random Forest (GridSearch) ===")
rf2 = eda_pipeline(model_type="forest")
best_params = rf2.tune(train_df, cv=10)

print("\n=== หลัง Tuning ===")
print(f"Test RMSE: ${rf2.evaluate(test_df):,.0f}")
