import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import uuid
import numpy as np

# =========================
# APP CONFIGURATION
# =========================
st.set_page_config(page_title="Smart Loan Recovery System", layout="wide")

st.title("📊 Smart Loan Recovery System with Machine Learning")
st.markdown("An AI-powered dashboard for **Loan Risk Analysis, Borrower Segmentation, and Dynamic Recovery Strategies**.")

# =========================
# SESSION STATE INITIALIZATION
# =========================
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.df_test = None
    st.session_state.features = ['Age', 'Monthly_Income', 'Loan_Amount', 'Loan_Tenure', 'Interest_Rate',
                                'Collateral_Value', 'Outstanding_Loan_Amount', 'Monthly_EMI',
                                'Num_Missed_Payments', 'Days_Past_Due']
    st.session_state.preprocessing_done = False
    st.session_state.error_message = None

# Required columns for the dataset
required_columns = st.session_state.features + ['Borrower_ID', 'Payment_History', 'Recovery_Status',
                                              'Collection_Method', 'Collection_Attempts', 'Legal_Action_Taken']

# =========================
# SAMPLE DATASET GENERATOR
# =========================
def generate_sample_data(n_rows=100):
    np.random.seed(42)
    data = {
        'Borrower_ID': range(1, n_rows + 1),
        'Age': np.random.randint(20, 70, n_rows),
        'Monthly_Income': np.random.uniform(2000, 10000, n_rows).round(2),
        'Loan_Amount': np.random.uniform(5000, 50000, n_rows).round(2),
        'Loan_Tenure': np.random.randint(12, 60, n_rows),
        'Interest_Rate': np.random.uniform(3.0, 10.0, n_rows).round(2),
        'Collateral_Value': np.random.uniform(3000, 40000, n_rows).round(2),
        'Outstanding_Loan_Amount': np.random.uniform(1000, 45000, n_rows).round(2),
        'Monthly_EMI': np.random.uniform(200, 2000, n_rows).round(2),
        'Num_Missed_Payments': np.random.randint(0, 5, n_rows),
        'Days_Past_Due': np.random.randint(0, 90, n_rows),
        'Payment_History': np.random.choice(['On-time', 'Late'], n_rows),
        'Recovery_Status': np.random.choice(['Recovered', 'Not Recovered'], n_rows),
        'Collection_Method': np.random.choice(['Email', 'Phone', 'In-Person'], n_rows),
        'Collection_Attempts': np.random.randint(0, 5, n_rows),
        'Legal_Action_Taken': np.random.choice(['Yes', 'No'], n_rows)
    }
    return pd.DataFrame(data)

# =========================
# PREPROCESSING FUNCTION
# =========================
def preprocess_data(df):
    try:
        features = st.session_state.features
        
        # Ensure numeric columns are numeric
        for col in features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check for missing values
        if df[features].isnull().any().any():
            return None, None, "Missing values detected in numeric columns. Please ensure all feature columns contain valid numbers."
        
        # Borrower Segmentation
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[features])
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df['Borrower_Segment'] = kmeans.fit_predict(df_scaled)
        df['Segment_Name'] = df['Borrower_Segment'].map({
            0: 'Moderate Income, High Loan Burden',
            1: 'High Income, Low Default Risk',
            2: 'Moderate Income, Medium Risk',
            3: 'High Loan, Higher Default Risk'
        })

        # Risk Prediction
        df['High_Risk_Flag'] = df['Segment_Name'].apply(
            lambda x: 1 if x in ['High Loan, Higher Default Risk', 'Moderate Income, High Loan Burden'] else 0)
        X = df[features]
        y = df['High_Risk_Flag']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        risk_scores = rf_model.predict_proba(X_test)[:, 1]
        df_test = X_test.copy()
        df_test['Risk_Score'] = risk_scores
        df_test['Predicted_High_Risk'] = (df_test['Risk_Score'] > 0.5).astype(int)
        df_test = df_test.merge(
            df[['Borrower_ID', 'Segment_Name', 'Recovery_Status', 'Collection_Method', 'Collection_Attempts', 'Legal_Action_Taken']],
            left_index=True, right_index=True)
        
        def assign_recovery_strategy(risk_score):
            if risk_score > 0.75:
                return "⚠️ Immediate Legal Notices & Aggressive Recovery"
            elif 0.50 <= risk_score <= 0.75:
                return "📑 Settlement Offers & Repayment Plans"
            else:
                return "📩 Automated Reminders & Monitoring"
        
        df_test['Recovery_Strategy'] = df_test['Risk_Score'].apply(assign_recovery_strategy)
        
        return df, df_test, None
    except Exception as e:
        return None, None, f"Preprocessing error: {str(e)}"

# =========================
# NAVIGATION
# =========================
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Select a Page", [
    "Upload Dataset",
    "Data Preview",
    "Loan Amount Distribution",
    "Loan Amount vs Monthly Income",
    "Payment History vs Recovery Status",
    "Missed Payments Impact on Recovery",
    "Borrower Segmentation",
    "Risk Prediction",
    "Download Results"
])

# =========================
# FILE UPLOADER
# =========================
if page == "Upload Dataset":
    st.header("📂 Upload Loan Recovery Dataset")
    st.markdown("Upload your `loan_recovery.csv` file or use the sample dataset to test the dashboard.")
    
    use_sample = st.checkbox("Use Sample Dataset (for testing)")
    
    if use_sample:
        st.session_state.df = generate_sample_data()
        st.success("✅ Sample dataset loaded successfully!")
    else:
        uploaded_file = st.file_uploader("Please upload your `loan_recovery.csv` file", type=["csv"], key=f"uploader_{uuid.uuid4()}")
        
        if uploaded_file is not None:
            try:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")
            except Exception as e:
                st.error(f"⚠️ Error reading the file: {str(e)}")
                st.session_state.df = None
                st.session_state.preprocessing_done = False
                st.stop()
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"⚠️ The uploaded file is missing the following required columns: {', '.join(missing_columns)}")
            st.session_state.df = None
            st.session_state.preprocessing_done = False
            st.stop()
        
        # Perform preprocessing
        df, df_test, error = preprocess_data(df)
        if error:
            st.error(f"⚠️ {error}")
            st.session_state.df = None
            st.session_state.preprocessing_done = False
            st.stop()
        
        st.session_state.df = df
        st.session_state.df_test = df_test
        st.session_state.preprocessing_done = True
        
        # Display dataset info
        st.subheader("📄 Dataset Info")
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")
        st.write(f"Columns: {', '.join(df.columns)}")
        st.info("✅ Data processed successfully! Use the sidebar to navigate through the analysis.")
    else:
        st.warning("⚠️ Please upload a `loan_recovery.csv` file or select 'Use Sample Dataset' to proceed.")
        st.stop()

# Check if data is loaded for other pages
if page != "Upload Dataset" and (st.session_state.df is None or not st.session_state.preprocessing_done):
    st.error("⚠️ No dataset uploaded or processed. Please go to the 'Upload Dataset' page and upload a valid `loan_recovery.csv` file or use the sample dataset.")
    st.stop()

df = st.session_state.df
df_test = st.session_state.df_test
features = st.session_state.features

# =========================
# DATA PREVIEW
# =========================
if page == "Data Preview":
    st.header("🔍 Data Preview")
    st.markdown("**Summary**: This section displays the raw loan recovery dataset, allowing you to inspect the first 50 rows to understand the data structure and contents, including borrower details, loan amounts, and recovery status.")
    if st.checkbox("Show Raw Data"):
        st.subheader("📄 Raw Loan Recovery Data")
        st.dataframe(df.head(50))

# =========================
# DATA EXPLORATION
# =========================
if page == "Loan Amount Distribution":
    st.header("1️⃣ Loan Amount Distribution")
    st.markdown("**Summary**: This histogram with a violin plot shows the distribution of loan amounts across borrowers. It highlights the spread and density of loan values, helping identify common loan sizes and outliers.")
    fig1 = px.histogram(df, x='Loan_Amount', nbins=30, marginal="violin", opacity=0.7,
                        title="Loan Amount Distribution",
                        labels={'Loan_Amount': "Loan Amount ($)"},
                        color_discrete_sequence=["royalblue"])
    st.plotly_chart(fig1, use_container_width=True)

elif page == "Loan Amount vs Monthly Income":
    st.header("1️⃣ Loan Amount vs Monthly Income")
    st.markdown("**Summary**: This scatter plot visualizes the relationship between loan amounts and borrowers' monthly income, color-coded by recovery status. Larger points indicate higher loan amounts, revealing patterns in repayment success.")
    fig2 = px.scatter(df, x='Loan_Amount', y='Monthly_Income',
                      color='Recovery_Status', size='Loan_Amount',
                      color_discrete_map={"Recovered": "green", "Not Recovered": "red"},
                      title="Loan Amount vs Monthly Income")
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Payment History vs Recovery Status":
    st.header("1️⃣ Payment History vs Recovery Status")
    st.markdown("**Summary**: This grouped histogram compares payment history (e.g., on-time or late payments) against recovery status, showing how consistent payments correlate with successful loan recovery.")
    fig3 = px.histogram(df, x="Payment_History", color="Recovery_Status", barmode="group",
                        title="Payment History vs Recovery Status",
                        color_discrete_map={"Recovered": "green", "Not Recovered": "red"})
    st.plotly_chart(fig3, use_container_width=True)

elif page == "Missed Payments Impact on Recovery":
    st.header("1️⃣ Missed Payments Impact on Recovery")
    st.markdown("**Summary**: This box plot illustrates the impact of missed payments on loan recovery, comparing the number of missed payments for recovered versus non-recovered loans to highlight risk factors.")
    fig4 = px.box(df, x="Recovery_Status", y="Num_Missed_Payments",
                  color="Recovery_Status",
                  color_discrete_map={"Recovered": "green", "Not Recovered": "red"},
                  title="Missed Payments Impact on Recovery")
    st.plotly_chart(fig4, use_container_width=True)

# =========================
# BORROWER SEGMENTATION
# =========================
elif page == "Borrower Segmentation":
    st.header("2️⃣ Borrower Segmentation with K-Means")
    st.markdown("**Summary**: This scatter plot displays borrower segments identified using K-Means clustering, based on features like income and loan amount. Each segment (e.g., High Income, Low Default Risk) helps tailor recovery strategies.")
    fig5 = px.scatter(df, x='Monthly_Income', y='Loan_Amount',
                      color=df['Segment_Name'],
                      size='Loan_Amount',
                      title="Borrower Segments (K-Means Clustering)",
                      labels={"Monthly_Income": "Monthly Income ($)", "Loan_Amount": "Loan Amount ($)"},
                      color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig5, use_container_width=True)
    st.subheader("📑 Segment Distribution")
    st.dataframe(df[['Borrower_ID', 'Monthly_Income', 'Loan_Amount', 'Segment_Name']].head(20))

# =========================
# RISK PREDICTION MODEL
# =========================
elif page == "Risk Prediction":
    st.header("3️⃣ Risk Prediction with Random Forest")
    st.markdown("**Summary**: This table presents the predicted risk scores and recovery strategies for borrowers, using a Random Forest model to identify high-risk borrowers and recommend actions like legal notices or repayment plans.")
    st.subheader("📌 Predicted Borrower Risk & Recovery Strategies")
    st.dataframe(df_test[['Borrower_ID', 'Monthly_Income', 'Loan_Amount',
                          'Segment_Name', 'Risk_Score', 'Predicted_High_Risk',
                          'Recovery_Strategy']].head(20))

# =========================
# DOWNLOAD REPORT
# =========================
elif page == "Download Results":
    st.header("📥 Download Results")
    st.markdown("**Summary**: Download a CSV file containing the risk analysis results, including borrower details, risk scores, and recommended recovery strategies for further action or review.")
    csv = df_test.to_csv(index=False).encode('utf-8')
    st.download_button("Download Borrower Risk Report (CSV)", data=csv, file_name="borrower_risk_report.csv", mime="text/csv")
    st.success("✅ Dashboard Ready – Explore Insights Above")