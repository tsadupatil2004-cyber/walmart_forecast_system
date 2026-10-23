# 🛒 Walmart Sales Forecasting & Executive Business Intelligence Portal

An end-to-end Machine Learning and Business Intelligence solution designed to analyze historical retail data, forecast weekly sales demand using **XGBoost**, and provide strategic operational insights through an interactive **Streamlit** dashboard.

---

## 👨‍💻 Developer Information

* **Developer:** *Tanvi Sadupatil  BTech-CSE(Data Science)*
* **Project Status:** Running Successfully ✅
* **Domain:** Retail Analytics, AI Demand Forecasting, & Supply Chain Management

---

## 📌 Project Overview

Retail giants like Walmart face significant challenges in optimizing inventory across hundreds of stores and departments. Overstocking leads to high holding costs and waste, while understocking results in lost revenue and dissatisfied customers.

This application solves these challenges by providing:
1. **📊 Executive BI Dashboard:** Interactive visual performance tracking across stores, store types, seasonality trends, and holiday periods using Plotly.
2. **🔮 AI Demand Forecasting Engine:** An XGBoost-powered regression engine capable of real-time demand prediction based on historical patterns, store properties, and macroeconomic indicators (Temperature, Fuel Price, CPI, Unemployment).
3. **💡 Strategic Supply Chain Insights:** Actionable operational recommendations and Sustainable Development Goals (SDGs) alignment.

---

## 🚀 Key Features

* **Interactive Streamlit Interface:** User-friendly navigation with an executive dark-themed header and custom CSS cards.
* **4-Column KPI Metric Cards:** Instant insights into Total Revenue, Mean Weekly Sales, Peak Weekly Sales, and Active Stores.
* **Dynamic Selection Profiles:** Real-time decoding of Store Types (*Supercenters, Discount Stores, Neighborhood Markets*) and Department Categories (*Pharmacy, Electronics, Grocery, etc.*).
* **"What-If" Demand Simulation:** Interactive parameters enabling business users to test economic variations and holiday impacts on demand.
* **Interactive Visualizations:** High-definition Plotly charts detailing sales trends, top store rankings, monthly seasonality, and holiday revenue spikes.

---

## 🛠️ Project Structure

```text
├── data/
│   ├── processed_sample.csv    # Cleaned dataset with engineered features
├── models/
│   ├── walmart_model.pkl       # Trained XGBoost Regression model
├── src/
│   ├── train.py                # Model training script
├── app.py                      # Main Streamlit Web Application
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 💻 Tech Stack & Libraries

* **Language:** Python 3.x
* **Web Framework:** Streamlit
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** XGBoost, Scikit-Learn, Joblib
* **Data Visualization:** Plotly Express

---

## ⚙️ How to Run the Application

### 1. Prerequisites
Ensure you have Python installed on your system. Install all required packages:
```bash
pip install streamlit pandas numpy plotly xgboost scikit-learn joblib
```

### 2. Train the Model (If running for the first time)
Execute the training script to generate the preprocessed dataset and model artifacts:
```bash
python src/train.py
```

### 3. Launch the Streamlit Portal
Run the application using Streamlit:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501` to view the portal.

---

## 🌱 Sustainable Development Goals (SDG) Alignment

This project aligns with key international sustainability standards:
* **SDG 8 (Decent Work & Economic Growth):** Enhances business performance and data-driven economic decision-making.
* **SDG 9 (Industry, Innovation & Infrastructure):** Implements modern Machine Learning and BI infrastructure in retail management.
* **SDG 12 (Responsible Consumption & Production):** Minimizes overstocking, reduces inventory waste, and promotes sustainable resource usage.