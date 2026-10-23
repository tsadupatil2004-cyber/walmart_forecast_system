import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Walmart Executive BI Portal",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional Executive CSS Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
    }
    .executive-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 25px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .executive-header h1 {
        color: #ffffff;
        font-size: 28px;
        margin-bottom: 5px;
    }
    .executive-header p {
        color: #94a3b8;
        font-size: 15px;
        margin: 0;
    }
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
        border-left: 4px solid #3b82f6;
    }
    .kpi-card-title {
        color: #64748b;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .kpi-card-value {
        color: #1e293b;
        font-size: 24px;
        font-weight: 700;
        margin: 0;
    }
    .section-container {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
    }
    h3 {
        color: #1e293b;
        font-family: 'Inter', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Model & Data ---
@st.cache_resource
def load_resources():
    if not os.path.exists("models/walmart_model.pkl") or not os.path.exists("data/processed_sample.csv"):
        return None, None
    model = joblib.load("models/walmart_model.pkl")
    df = pd.read_csv("data/processed_sample.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return model, df

model, df = load_resources()

# --- Sidebar Navigation ---
st.sidebar.markdown("### 🛒 Enterprise Navigation")
st.sidebar.markdown("---")

if model is None or df is None:
    st.error("⚠️ Model or data not found! Please run `python src/train.py` first.")
else:
    menu = st.sidebar.radio("Select Module", ["📊 Executive Dashboard", "🔮 AI Demand Forecasting", "💡 Supply Chain Insights"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("**System Status:** Online ✅\n\n**Model:** XGBoost Regressor\n\n**Framework:** Streamlit & Plotly")

    # ==========================================
    # 1. EXECUTIVE DASHBOARD
    # ==========================================
    if menu == "📊 Executive Dashboard":
        st.markdown("""
            <div class="executive-header">
                <h1>📊 Executive Business Intelligence Dashboard</h1>
                <p>Comprehensive exploratory retail analytics tracking revenue performance, store rankings, and seasonal trends.</p>
            </div>
        """, unsafe_allow_html=True)

        # Filter Bar
        col_f1, col_f2 = st.columns([2, 4])
        with col_f1:
            selected_type = st.selectbox("Filter by Store Type", ["All Types", "A", "B", "C"])
        
        filtered_df = df if selected_type == "All Types" else df[df['Type'] == selected_type]

        # Enhanced 4-Column KPI Metrics Grid
        total_sales = filtered_df['Weekly_Sales'].sum()
        avg_weekly = filtered_df['Weekly_Sales'].mean()
        max_weekly = filtered_df['Weekly_Sales'].max()
        store_count = filtered_df['Store'].nunique()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-card-title">Total Revenue</div>
                    <div class="kpi-card-value">${total_sales:,.0f}</div>
                </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
                <div class="kpi-card" style="border-left-color: #10b981;">
                    <div class="kpi-card-title">Mean Weekly Sales</div>
                    <div class="kpi-card-value">${avg_weekly:,.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
                <div class="kpi-card" style="border-left-color: #f59e0b;">
                    <div class="kpi-card-title">Peak Weekly Sales</div>
                    <div class="kpi-card-value">${max_weekly:,.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
                <div class="kpi-card" style="border-left-color: #8b5cf6;">
                    <div class="kpi-card-title">Active Stores</div>
                    <div class="kpi-card-value">{store_count}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Historical Trend Chart Container
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.subheader("📈 Global Weekly Sales Performance Trend")
        trend = filtered_df.groupby('Date')['Weekly_Sales'].sum().reset_index()
        fig = px.line(trend, x='Date', y='Weekly_Sales', template="plotly_white")
        fig.update_traces(line=dict(color='#3b82f6', width=2.5))
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Timeline", yaxis_title="Weekly Sales ($)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Row 2 Charts: Top 10 Stores & Store Type Share
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader("🏆 Top 10 Performing Stores by Revenue")
            top_stores = filtered_df.groupby('Store')['Weekly_Sales'].sum().reset_index().nlargest(10, 'Weekly_Sales')
            top_stores['Store'] = top_stores['Store'].astype(str) + " Store"
            fig_top = px.bar(top_stores, x='Store', y='Weekly_Sales', template="plotly_white", color='Weekly_Sales', color_continuous_scale='Blues')
            fig_top.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="Total Sales ($)")
            st.plotly_chart(fig_top, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_b:
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader("🛍️ Sales Share by Store Type")
            type_df = df.groupby('Type')['Weekly_Sales'].sum().reset_index()
            fig_pie = px.pie(type_df, values='Weekly_Sales', names='Type', hole=0.45, template="plotly_white", color_discrete_sequence=['#3b82f6', '#10b981', '#f59e0b'])
            fig_pie.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Row 3 Charts: Monthly Seasonality & Holiday Impact
        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader("📅 Monthly Sales Distribution (Seasonality)")
            monthly_df = filtered_df.groupby('Month')['Weekly_Sales'].mean().reset_index()
            fig_month = px.bar(monthly_df, x='Month', y='Weekly_Sales', template="plotly_white", color='Weekly_Sales', color_continuous_scale='Purples')
            fig_month.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Month of Year", yaxis_title="Average Sales ($)")
            st.plotly_chart(fig_month, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_d:
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader("🎉 Holiday vs Regular Week Sales")
            hol_df = filtered_df.groupby('IsHoliday')['Weekly_Sales'].mean().reset_index()
            hol_df['IsHoliday'] = hol_df['IsHoliday'].replace({0: 'Regular Week', 1: 'Holiday Week'})
            fig_bar = px.bar(hol_df, x='IsHoliday', y='Weekly_Sales', color='IsHoliday', template="plotly_white", color_discrete_map={'Regular Week': '#3b82f6', 'Holiday Week': '#f59e0b'})
            fig_bar.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="Average Sales ($)")
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 2. AI DEMAND FORECASTING
    # ==========================================
    elif menu == "🔮 AI Demand Forecasting":
        st.markdown("""
            <div class="executive-header">
                <h1>🔮 AI Demand Forecasting Engine</h1>
                <p>Simulate future store conditions, economic indicators, and seasonal metrics to predict weekly retail demand instantly.</p>
            </div>
        """, unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader("⚙️ Parameters")
            store = st.selectbox("Store ID", sorted(df['Store'].unique()))
            dept = st.selectbox("Department ID", sorted(df['Dept'].unique()))
            
            # --- Dynamic Department Category Mapping ---
            if dept in [1, 2, 4, 5]:
                dept_category = "Pharmacy & Health/Wellness"
            elif dept in [3, 6, 7, 8]:
                dept_category = "Electronics & Photo"
            elif dept in [9, 10, 11, 12]:
                dept_category = "Apparel & Accessories"
            elif dept in [38, 40, 41, 42]:
                dept_category = "Home Goods & Hardware"
            elif dept in [90, 91, 92, 93, 94, 95]:
                dept_category = "Grocery & Consumables"
            else:
                dept_category = "General Merchandise & Specialty"

            # --- Dynamic Store & Department Metadata Display ---
            store_row = df[df['Store'] == store].iloc[0]
            store_type = store_row['Type']
            store_size = store_row['Size']
            
            type_desc_map = {
                'A': 'Supercenters / Hypermarkets',
                'B': 'Discount Stores / Mid-sized Supermarkets',
                'C': 'Neighborhood Markets / Smaller Formats'
            }
            type_full_name = type_desc_map.get(store_type, 'Retail Store')
            
            st.markdown(f"""
                <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-bottom: 15px;">
                    <span style="font-size: 12px; font-weight: 700; color: #64748b; text-transform: uppercase;">Selection Profile Info</span><br>
                    <b>Format:</b> Type {store_type} ({type_full_name})<br>
                    <b>Store Size:</b> {store_size:,.0f} sq. ft.<br>
                    <b>Department Category:</b> Dept {dept} ({dept_category})
                </div>
            """, unsafe_allow_html=True)
            
            holiday = st.selectbox("Holiday Week?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            
            temp = st.slider("Temperature (°F)", 10.0, 100.0, float(store_row['Temperature']))
            fuel = st.slider("Fuel Price ($)", 2.0, 5.0, float(store_row['Fuel_Price']))
            cpi = st.slider("CPI Index", 120.0, 230.0, float(store_row['CPI']))
            unemp = st.slider("Unemployment Rate (%)", 3.0, 15.0, float(store_row['Unemployment']))
            
            predict_btn = st.button("🚀 Run Forecast", type="primary", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader("📊 Prediction Results")
            
            if predict_btn:
                input_data = pd.DataFrame([[
                    store, dept, holiday, store_row['Size'], store_row['Type_Encoded'],
                    temp, fuel, 0, 0, 0, 0, 0, cpi, unemp, 2012, 6, 25, 15
                ]], columns=[
                    'Store', 'Dept', 'IsHoliday', 'Size', 'Type_Encoded', 
                    'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 
                    'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 
                    'Unemployment', 'Year', 'Month', 'Week', 'Day'
                ])
                
                pred = model.predict(input_data)[0]
                
                st.markdown(f"""
                    <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 25px; text-align: center; margin-bottom: 20px;">
                        <span style="color: #166534; font-size: 14px; font-weight: 700; text-transform: uppercase;">Predicted Weekly Sales Demand</span>
                        <h1 style="color: #15803d; font-size: 42px; margin-top: 5px; margin-bottom: 0;">${pred:,.2f}</h1>
                    </div>
                """, unsafe_allow_html=True)
                
                hist_avg = df[(df['Store'] == store) & (df['Dept'] == dept)]['Weekly_Sales'].mean()
                if not pd.isna(hist_avg):
                    diff = ((pred - hist_avg) / hist_avg) * 100
                    st.info(f"💡 **Benchmark Comparison:** This prediction is **{abs(diff):.1f}% {'higher' if diff > 0 else 'lower'}** than the historical average for Store {store}, Department {dept} (${hist_avg:,.2f}).")
            else:
                st.info("👈 Configure parameters on the left and click **Run Forecast** to calculate predicted demand.")

            st.markdown("---")
            st.markdown("#### 🔍 Model Architecture Note")
            st.markdown("The underlying **XGBoost regression model** analyzes store scale, department categories, seasonal calendar milestones, and macroeconomic indicators (CPI & Unemployment) to eliminate inventory stockouts.")
            st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 3. SUPPLY CHAIN INSIGHTS
    # ==========================================
    elif menu == "💡 Supply Chain Insights":
        st.markdown("""
            <div class="executive-header">
                <h1>💡 Strategic Recommendations & Sustainability</h1>
                <p>Translating predictive machine learning outputs into actionable supply chain decisions aligned with Sustainable Development Goals (SDG).</p>
            </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader("📌 Operational Action Plan")
            st.markdown("""
            * **Inventory Buffer Management:** Pre-stock high-velocity departments **3 weeks prior** to major Thanksgiving and Christmas retail spikes.
            * **Markdown Timing:** Implement clearance markdowns during lower-demand seasons (January/February) to minimize holding costs.
            * **Automated Restocking:** Integrate this system's backend API directly with warehouse management platforms.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader("🌱 SDG Alignment")
            st.markdown("""
            * **SDG 8 (Decent Work & Economic Growth):** Supports efficient business operations and data-driven economic decision-making.
            * **SDG 9 (Industry, Innovation & Infrastructure):** Leverages advanced Machine Learning and Business Intelligence for retail analytics.
            * **SDG 12 (Responsible Consumption & Production):** Accurate sales forecasting reduces overstocking, unnecessary inventory, and resource wastage.
            """)
            st.markdown('</div>', unsafe_allow_html=True)