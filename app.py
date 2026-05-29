import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ────────────────────────────────
# ตั้งค่าหน้า app
# ────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon= "🏠",
    layout="wide"
)

# ────────────────────────────────
# โหลดและ train model
# ────────────────────────────────
@st.cache_resource # train แค่ครั้งเดียว ไม่ต้อง train ใหม่ทุกครั้งที่ user กด
def load_model():
    df = pd.read_csv("Data/train.csv")

    FEATURES = ['OverallQual', 'GrLivArea', 'GarageCars',
                'TotalBsmtSF', 'FullBath', 'YearBuilt',
                'LotArea', 'MasVnrArea']
    
    train_df, _ = train_test_split(df, test_size=0.2, random_state=42)
    X_train = train_df[FEATURES].fillna(train_df[FEATURES].median())
    y_train = train_df['SalePrice']

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   RandomForestRegressor(
                        n_estimators=200,
                        max_depth=15,
                        random_state=42))
        ])
    pipe.fit(X_train, y_train)
    return pipe, FEATURES
pipe, FEATURES = load_model()

# ────────────────────────────────
# หน้า app
# ────────────────────────────────
st.title("🏠 House Price Predictor")
st.markdown("ใส่ข้อมูลบ้านด้านซ้าย แล้วดูราคาที่ทำนายได้เลย")

# Sidebar — รับ input
st.sidebar.header("📋 ข้อมูลบ้าน")

overall_qual = st.sidebar.slider(
    "คุณภาพโดยรวม (1-10)", 1, 10, 7)

gr_liv_area = st.sidebar.number_input(
    "พื้นที่อยู่อาศัย (ตร.ฟุต)", 500, 6000, 1500)

garage_cars = st.sidebar.slider(
    "จอดรถได้กี่คัน", 0, 4, 2)

total_bsmt = st.sidebar.number_input(
    "พื้นที่ชั้นใต้ดิน (ตร.ฟุต)", 0, 3000, 800)

full_bath = st.sidebar.slider(
    "ห้องน้ำ", 0, 4, 2)

year_built = st.sidebar.number_input(
    "ปีที่สร้าง", 1872, 2010, 2000)

lot_area = st.sidebar.number_input(
    "ขนาดที่ดิน (ตร.ฟุต)", 1000, 50000, 8000)

mas_vnr_area = st.sidebar.number_input(
    "พื้นที่ผนังก่ออิฐ (ตร.ฟุต)", 0, 1500, 100)

# สร้าง input array
input_data = pd.DataFrame([[
    overall_qual, gr_liv_area, garage_cars,
    total_bsmt, full_bath, year_built,
    lot_area, mas_vnr_area
]], columns=FEATURES)

# ────────────────────────────────
# แสดงผล
# ────────────────────────────────
predicted_price = pipe.predict(input_data)[0]

col1, col2, = st.columns(2)

with col1:
    st.metric(
        label="ราคาที่ทำนาย",
        value=f"${predicted_price:,.0f}"
    )

    st.subheader("ข้อมูลที่ใส่")
    st.dataframe(input_data)

with col2:
    # Feature Importance
    st.subheader("Feature Importance")
    importances = pipe.named_steps['model'].feature_importances_
    feat_imp = pd.Series(importances, index=FEATURES).sort_values()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    feat_imp.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title("Feature Importance")
    st.pyplot(fig)