# 🚦 Vehicle Collision Analysis Engine

## 📌 Project Overview
The **Vehicle Collision Analysis Engine** is an advanced AI-powered system designed to predict traffic risk and analyze accident patterns across **22 Indian States**. Unlike traditional systems that react to accidents, this engine proactively identifies "Greyspots"—high-risk zones that are likely to become accident hotspots.

## 🚀 Key Features
*   **Hybrid ML Architecture**: Combines **Random Forest** and **XGBoost** for high-precision risk scoring.
*   **Geospatial Intelligence**: Interactive **Mapbox** visualization of accident clusters across India.
*   **Explainable AI (XAI)**: A built-in reasoning engine that explains *why* a risk prediction was made (e.g., "Night time + Rain + Two-wheeler").
*   **MoRTH-Standardized Data**: Trained on data statistically modeled after the **Ministry of Road Transport and Highways (2022-23)** report.
*   **Proactive Recommendations**: Generates automated safety advice for Traffic Authorities and Commuters.

## 🛠️ Technology Stack
*   **Language**: Python 3.11
*   **Analysis**: Pandas, NumPy
*   **Machine Learning**: Scikit-Learn, XGBoost, Joblib
*   **Visualization**: Streamlit, Plotly Express
*   **Forensics**: Synthetic Forensic Data Generation from MoRTH stats.

## 📦 Components
1.  **Dashboard**: The main decision support interface (`dashboard.py`).
2.  **Training Module**: Retrains the hybrid model on new data (`train_model.py`).
3.  **Data Engine**: Generates statistically accurate Indian traffic datasets (`generate_real_morth_data.py`).

## ⚙️ How to Run
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the engine:
    ```bash
    streamlit run dashboard.py
    ```