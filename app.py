import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS (Exact Target UI)
# ============================================================
st.markdown("""
<style>
    .stApp {
        background-color: #f1f5f9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .hero-banner {
        background: linear-gradient(120deg, #0f172a 0%, #1e293b 60%, #334155 100%);
        border-radius: 12px;
        padding: 24px 30px;
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .hero-title {
        font-size: 32px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 14px;
        color: #cbd5e1;
        margin-top: 6px;
    }
    .hero-quote {
        font-style: italic;
        font-size: 13px;
        color: #94a3b8;
        border-left: 2px solid #38bdf8;
        padding-left: 10px;
        max-width: 320px;
        text-align: right;
    }
    .ui-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px 22px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        border: 1px solid #e2e8f0;
        margin-bottom: 18px;
    }
    .card-header {
        font-size: 17px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .pred-card {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 14px 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
    }
    .pred-val {
        font-size: 26px;
        font-weight: 800;
        color: #15803d;
        margin: 0;
    }
    .accuracy-badge {
        background-color: #059669;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        text-align: right;
    }
    .kpi-badge {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 8px 4px;
        text-align: center;
    }
    .kpi-val {
        font-size: 15px;
        font-weight: 700;
        color: #1e293b;
    }
    .kpi-lbl {
        font-size: 10.5px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
    }
    .details-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
        margin-top: 6px;
    }
    .details-table td {
        padding: 5px 8px;
        border-bottom: 1px solid #f1f5f9;
    }
    .details-table td.key {
        color: #64748b;
        font-weight: 500;
    }
    .details-table td.val {
        color: #0f172a;
        font-weight: 600;
        text-align: right;
    }
    .info-note {
        background-color: #f8fafc;
        border-left: 3px solid #0284c7;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 11px;
        color: #475569;
        line-height: 1.4;
        margin-top: 10px;
    }
    .footer-bar {
        text-align: center;
        font-size: 12px;
        color: #94a3b8;
        padding: 20px 0 10px 0;
        border-top: 1px solid #e2e8f0;
        margin-top: 25px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("### 🚗 Used Car Price\n**Prediction**  \n`Machine Learning Project`")
    st.write("---")
    selected_nav = st.radio(
        "Navigation",
        ["🏠  Home", "🔮  Predict Price", "📊  Model Performance", "ℹ️  About Project"],
        index=1,
        label_visibility="collapsed"
    )
    st.write("")
    st.markdown("""
        <div style='position: fixed; bottom: 20px; font-size: 11px; color: #94a3b8;'>
            Built with ❤️ using<br>
            Python • Streamlit • Scikit-learn
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# DATA PIPELINE (Reading your exact CSV)
# ============================================================
@st.cache_data
def load_and_preprocess():
    csv_file = "used_car_price_prediction_dataset.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        # Fallback to any matching CSV in folder
        for f in os.listdir("."):
            if f.endswith(".csv"):
                df = pd.read_csv(f)
                break

    df['Car_Age'] = 2026 - df['Manufacturing_Year']
    # Target in Lakhs for readable reporting
    df['Price_Lakh'] = df['Selling_Price'] / 100000.0
    df['Log_Price'] = np.log1p(df['Selling_Price'])
    return df

df = load_and_preprocess()

# ============================================================
# MODEL TRAINING (Random Forest)
# ============================================================
model_df = df.copy()
# Features to encode
cat_features = ['Car_Name', 'Brand', 'Fuel_Type', 'Transmission_Type', 'Number_of_Owners', 'Seller_Type']
existing_cat = [c for c in cat_features if c in model_df.columns]
model_df = pd.get_dummies(model_df, columns=existing_cat, dtype=int)

X = model_df.drop(columns=['Manufacturing_Year', 'Selling_Price', 'Price_Lakh', 'Log_Price'], errors='ignore')
y = model_df['Log_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

@st.cache_resource
def train_rf(X_tr, y_tr):
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    return rf

model = train_rf(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# ============================================================
# TOP HERO BANNER
# ============================================================
st.markdown("""
<div class="hero-banner">
    <div>
        <h1 class="hero-title">Used Car Price Prediction</h1>
        <div class="hero-subtitle">Get the estimated price of a used car using machine learning.<br>Enter the car details below to see the predicted price.</div>
    </div>
    <div style="display: flex; align-items: center; gap: 20px;">
        <img src="https://images.unsplash.com/photo-1617814076367-b759c7d7e738?w=350&auto=format&fit=crop&q=80" style="height: 85px; border-radius: 8px; object-fit: cover;">
        <div class="hero-quote">
            "Good cars don't just take you places,<br>they take you forward."
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MIDDLE ROW: FORM & PREDICTION
# ============================================================
col_input, col_pred = st.columns([1.1, 0.9], gap="medium")

with col_input:
    st.markdown("""
        <div class="ui-card">
            <div class="card-header">
                🚘 Enter Car Details
                <span style="font-size: 11px; font-weight: normal; color: #64748b;">(Fill in the details below to predict the used car price)</span>
            </div>
    """, unsafe_allow_html=True)

    brands = sorted(df["Brand"].unique().tolist())
    names = sorted(df["Car_Name"].unique().tolist())
    fuels = sorted(df["Fuel_Type"].unique().tolist())
    transmissions = sorted(df["Transmission_Type"].unique().tolist())
    owners = sorted(df["Number_of_Owners"].unique().tolist())
    sellers = sorted(df["Seller_Type"].unique().tolist())

    c1, c2 = st.columns(2)
    with c1:
        in_brand = st.selectbox("🏷️ Brand", brands, index=brands.index("Hyundai") if "Hyundai" in brands else 0)
        in_name = st.selectbox("🚘 Car Model", names, index=names.index("Hyundai Creta") if "Hyundai Creta" in names else 0)
        in_trans = st.selectbox("⚙️ Transmission", transmissions)
        in_km = st.number_input("🛣️ Kilometers Driven", min_value=1000.0, value=45000.0, step=2500.0)

    with c2:
        in_year = st.number_input("📅 Manufacturing Year", min_value=2005, max_value=2026, value=2018, step=1)
        in_fuel = st.selectbox("⛽ Fuel Type", fuels)
        in_owner = st.selectbox("👤 Number of Owners", owners)
        in_seller = st.selectbox("🏪 Seller Type", sellers)

    st.write("")
    btn_predict = st.button("🚀 Predict Price", use_container_width=True, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

# Process Inputs for prediction
input_df = pd.DataFrame(0.0, index=[0], columns=X.columns)
input_df.at[0, 'Kilometers_Driven'] = in_km
input_df.at[0, 'Car_Age'] = 2026 - in_year

# Set One-Hot flags
cat_inputs = [
    ('Brand', in_brand),
    ('Car_Name', in_name),
    ('Fuel_Type', in_fuel),
    ('Transmission_Type', in_trans),
    ('Number_of_Owners', in_owner),
    ('Seller_Type', in_seller)
]
for col_name, selected_val in cat_inputs:
    encoded_col = f"{col_name}_{selected_val}"
    if encoded_col in input_df.columns:
        input_df.at[0, encoded_col] = 1.0

log_pred = model.predict(input_df)[0]
predicted_price = np.expm1(log_pred)
price_in_lakhs = predicted_price / 100000.0

with col_pred:
    st.markdown(f"""
        <div class="ui-card">
            <div class="pred-card">
                <div>
                    <div style="font-size: 13px; color: #166534; font-weight: 600;">🚗 Predicted Used Car Price</div>
                    <div class="pred-val">₹ {price_in_lakhs:.2f} Lakhs</div>
                    <div style="font-size: 11px; color: #16a34a;">(Approx. ₹ {predicted_price:,.0f})</div>
                </div>
                <div>
                    <div class="accuracy-badge">
                        R² Score: {r2:.4f}<br>
                        <span style="font-weight: normal; font-size: 9.5px;">Model Accuracy</span>
                    </div>
                </div>
            </div>

            <div style="font-size: 13.5px; font-weight: 700; color: #1e293b; margin-bottom: 8px;">
                📋 Car Details (Input Summary)
            </div>
            
            <div style="display: flex; gap: 16px; align-items: center;">
                <table class="details-table" style="flex: 1.2;">
                    <tr><td class="key">Car Model</td><td class="val">{in_name}</td></tr>
                    <tr><td class="key">Brand</td><td class="val">{in_brand}</td></tr>
                    <tr><td class="key">Manufacturing Year</td><td class="val">{in_year}</td></tr>
                    <tr><td class="key">Fuel / Transmission</td><td class="val">{in_fuel} / {in_trans}</td></tr>
                    <tr><td class="key">Kilometers Driven</td><td class="val">{in_km:,.0f} km</td></tr>
                    <tr><td class="key">Owner & Seller</td><td class="val">{in_owner} ({in_seller})</td></tr>
                </table>
                <div style="flex: 0.8; text-align: center;">
                    <img src="https://images.unsplash.com/photo-1549399542-7e3f8b79c341?w=280&auto=format&fit=crop&q=80" 
                         style="width: 100%; max-width: 170px; border-radius: 8px; border: 1px solid #e2e8f0;">
                </div>
            </div>

            <div class="info-note">
                <strong>💡 Note:</strong> The predicted price is an estimate based on {len(df)} historical market listings and may vary depending on actual physical vehicle condition and local RTO valuation.
            </div>
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# BOTTOM ROW: 3-COLUMN ANALYTICS (Metrics, Feature Importance, About)
# ============================================================
b_c1, b_c2, b_c3 = st.columns([1, 1, 0.9], gap="medium")

with b_c1:
    st.markdown("""
        <div class="ui-card" style="height: 100%;">
            <div class="card-header">📊 Model Performance</div>
    """, unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='kpi-badge'><div class='kpi-lbl'>MAE</div><div class='kpi-val' style='color:#0284c7;'>{mae:.4f}</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='kpi-badge'><div class='kpi-lbl'>MSE</div><div class='kpi-val' style='color:#059669;'>{mse:.4f}</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='kpi-badge'><div class='kpi-lbl'>RMSE</div><div class='kpi-val' style='color:#7c3aed;'>{rmse:.4f}</div></div>", unsafe_allow_html=True)
    with m4:
        st.markdown(f"<div class='kpi-badge'><div class='kpi-lbl'>R² Score</div><div class='kpi-val' style='color:#d97706;'>{r2:.4f}</div></div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<div style='font-size: 11px; font-weight: bold; color: #475569;'>Actual vs Predicted Price (Test Set)</div>", unsafe_allow_html=True)

    fig_eval, ax_eval = plt.subplots(figsize=(4.5, 2.6), dpi=120)
    fig_eval.patch.set_facecolor('#ffffff')
    ax_eval.set_facecolor('#ffffff')

    ax_eval.scatter(y_test, y_pred, alpha=0.6, color='#2563eb', s=14, label='Actual')
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax_eval.plot([min_val, max_val], [min_val, max_val], '--', color='#ef4444', lw=1.2, label='Predicted')

    ax_eval.set_xlabel("Actual Price (Log Scale)", fontsize=8, color="#64748b")
    ax_eval.set_ylabel("Predicted Price (Log Scale)", fontsize=8, color="#64748b")
    ax_eval.tick_params(axis='both', labelsize=7, colors="#64748b")
    ax_eval.grid(True, linestyle=':', alpha=0.5, color="#cbd5e1")
    ax_eval.legend(fontsize=7, loc="upper left", frameon=False)
    plt.tight_layout()
    st.pyplot(fig_eval)
    st.markdown("</div>", unsafe_allow_html=True)

with b_c2:
    st.markdown("""
        <div class="ui-card" style="height: 100%;">
            <div class="card-header">⭐ Feature Importance</div>
    """, unsafe_allow_html=True)

    imp_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).head(10)

    fig_imp, ax_imp = plt.subplots(figsize=(4.5, 2.9), dpi=120)
    fig_imp.patch.set_facecolor('#ffffff')
    ax_imp.set_facecolor('#ffffff')

    palette = ['#38bdf8', '#34d399', '#4ade80', '#a3e635', '#facc15', '#fb923c', '#f87171', '#fb7185', '#e879f9', '#c084fc']
    bars = ax_imp.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1], color=palette[:len(imp_df)], height=0.65)

    for bar in bars:
        ax_imp.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{bar.get_width():.3f}", va='center', ha='left', fontsize=6.5, color='#475569', weight='bold')

    ax_imp.set_xlim(0, max(imp_df["Importance"]) * 1.25)
    ax_imp.tick_params(axis='y', labelsize=7.5, colors="#334155")
    ax_imp.tick_params(axis='x', labelsize=7, colors="#64748b")
    ax_imp.grid(axis='x', linestyle=':', alpha=0.5, color="#cbd5e1")
    plt.tight_layout()
    st.pyplot(fig_imp)

    top_features = ", ".join(imp_df["Feature"].head(5).tolist())
    st.markdown(f"""
        <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 6px 10px; font-size: 10.5px; color: #475569;">
            <strong>🏆 Top 5 Important Features:</strong><br>{top_features}
        </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with b_c3:
    st.markdown("""
        <div class="ui-card" style="height: 100%; font-size: 11.5px; line-height: 1.55;">
            <div class="card-header">ℹ️ About This Project</div>
            
            <div style="font-weight: 700; color: #0f172a; margin-bottom: 2px;">Project Summary</div>
            <p style="color: #64748b; margin-bottom: 10px;">
                This project predicts the price of used cars using machine learning. The model is trained on market features including Brand, Car Model, Manufacturing Year, Kilometers Driven, Fuel Type, Transmission, and Owner History.
            </p>
            
            <div style="font-weight: 700; color: #0f172a; margin-bottom: 4px;">Technologies Used</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px; color: #334155; margin-bottom: 10px;">
                <div>🐍 Python</div>
                <div>🐼 Pandas</div>
                <div>📊 Matplotlib</div>
                <div>🤖 Scikit-learn</div>
                <div>👑 Streamlit</div>
            </div>

            <div style="font-weight: 700; color: #0f172a; margin-bottom: 4px;">Key Steps</div>
            <ul style="padding-left: 16px; margin: 0; color: #64748b;">
                <li>Data Ingestion & Cleaning</li>
                <li>Feature Engineering (<code>Car_Age</code>)</li>
                <li>One-Hot Encoding of Categoricals</li>
                <li>Random Forest Regressor (100 Trees)</li>
                <li>Live R² and Error Analytics</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer-bar">
    Used Car Price Prediction &nbsp;|&nbsp; Powered by Machine Learning &nbsp;|&nbsp; Made with ❤️ using Python & Streamlit
</div>
""", unsafe_allow_html=True)