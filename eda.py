import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Data/train.csv")

# ────────────────────────────────
# ขั้น 1: ทำความเข้าใจ data ก่อน
# ────────────────────────────────
# 1.1 ขนาด data
print("Shape of data")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# 1.2 ค่าหายมีที่ column ไหนบ้าง?
print("\nMissing values in each column:")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing.head(20))

#ดูราคาบ้าน กราฟ
print("\nPrice distribution:")
print(df["SalePrice"].describe())
plt.figure(figsize=(10, 6))
plt.hist(df["SalePrice"], bins=50, color="blue", edgecolor="black")
plt.title("Distribution of SalePrice")
plt.xlabel("SalePrice")
plt.ylabel("Frequency")
plt.show()

def clean_data(df):
    df = df.copy()
    none_cols = ['PoolQC', 'MiscFeature', 'Alley', 
                 'Fence', 'FireplaceQu', 'GarageType',
                 'GarageFinish', 'GarageQual', 'GarageCond',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']
    for col in none_cols:
        df[col] = df[col].fillna("None")

# 2. ตัวเลขที่หายเพราะ "ไม่มี" จริงๆ
    zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars',
                 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                 'TotalBsmtSF', 'MasVnrArea']
    for col in zero_cols:
        df[col] = df[col].fillna(0)

    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    return df
df_clean = clean_data(df)
# columns พวกนี้มีลำดับชัดเจน Poor → Excellent
quality_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
quality_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                'HeatingQC', 'KitchenQual', 'FireplaceQu',
                'GarageQual', 'GarageCond', 'PoolQC']

for col in quality_cols:
    df_clean[col] = df_clean[col].map(quality_map)
# columns ที่เหลือที่เป็น text ไม่มีลำดับ
onehot_cols = ['MSZoning', 'Street', 'Alley', 'LotShape',
               'Neighborhood', 'BldgType', 'HouseStyle',
               'SaleType', 'SaleCondition', 'Foundation',
               'Heating', 'CentralAir', 'GarageType',
               'GarageFinish', 'Fence', 'MiscFeature']

df_encoded = pd.get_dummies(df_clean, columns=onehot_cols, drop_first=True)
numeric_df = df_clean.select_dtypes(include=['int64', 'float64'])
corr = numeric_df.corr()['SalePrice'].sort_values(ascending=False)
good_features = corr[corr > 0.5].drop('SalePrice').index.tolist()




plt.figure(figsize=(10, 8))
corr.drop('SalePrice').head(10).plot(kind='barh', color='steelblue')
plt.title("Top 10 Features vs SalePrice")
plt.xlabel("Correlation Score")
plt.tight_layout()
plt.show()

print(f"Features ที่ดี: {good_features}")
print(f"จำนวน: {len(good_features)} features")
print("\n=== Missing After Cleaning ===")
print(df_clean.isnull().sum().sum())  # ควรได้ 0
print(f"Before encoding: {df_clean.shape}")
print(f"After encoding:  {df_encoded.shape}")
print(df_clean.dtypes)
