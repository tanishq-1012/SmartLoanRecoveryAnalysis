### Smart Loan Recovery System with Machine Learning Dashboard

## Overview

The Smart Loan Recovery System is an AI-powered Streamlit dashboard designed for loan risk analysis, borrower segmentation, and dynamic recovery strategy recommendations. It leverages machine learning models (K-Means clustering and Random Forest classification) to provide actionable insights for financial institutions to optimize loan recovery processes.

### Features

* **Data Upload**: Upload a loan_recovery.csv file or use a built-in sample dataset to analyze loan data.

* **Data Exploration**: Visualize key metrics through interactive Plotly charts, including:

  * Loan Amount Distribution (histogram with violin plot)

  * Loan Amount vs. Monthly Income (scatter plot)

  * Payment History vs. Recovery Status (grouped histogram)

  * Missed Payments Impact on Recovery (box plot)

* **Borrower Segmentation**: Uses K-Means clustering to categorize borrowers into segments (e.g., High Income, Low Default Risk).

* **Risk Prediction**: Employs a Random Forest classifier to predict borrower risk and recommend recovery strategies (e.g., legal notices, repayment plans).

* **Downloadable Reports**: Export analysis results as a CSV file for further action.

* **Multi-Page Interface**: Each visualization and section is displayed on a separate page for a clean user experience.

* **Error Handling**: Robust validation for uploaded datasets, with detailed error messages for missing columns or invalid data.

## Prerequisites

To run the dashboard, ensure you have the following installed:
 * Python 3.8 or higher
 * Required Python packages (install via pip):

 pip install streamlit pandas plotly scikit-learn numpy

## Installation

1. **Clone the Repository** (if hosted on GitHub):

   git clone <repository-url>

   cd smart-loan-recovery-system

2. **Install Dependencies**:

   pip install -r requirements.txt

   Create a requirements.txt with:

   streamlit,
   pandas,
   plotly,
   scikit-learn,
   numpy

## Usage

1. **Launch the Dashboard**:

 * Run streamlit run app.py to start the Streamlit server.

2. **Upload Dataset**:

 * Navigate to the "Upload Dataset" page.
 
 * Upload a loan_recovery.csv file or select the "Use Sample Dataset" checkbox to test with generated data.

3. **Explore Visualizations**:

 * Use the sidebar to navigate to different pages, each containing a specific visualization or analysis:

   * **Data Preview**: View the raw dataset (first 50 rows).

   * **Loan Amount Distribution**: Histogram showing loan amount spread.

   * **Loan Amount vs. Monthly Income**: Scatter plot analyzing income vs. loan repayment.

   * **Payment History vs. Recovery Status**: Histogram comparing payment consistency and recovery.

   * **Missed Payments Impact on Recovery**: Box plot highlighting missed payments' effect.

   * **Borrower Segmentation**: Scatter plot of K-Means clusters with segment details.

   * **Risk Prediction**: Table of predicted risk scores and recovery strategies.

   * **Download Results**: Export the analysis as a CSV file.

4. **Interpret Results**:

  * Each visualization includes a short summary explaining its purpose and insights.

  * The Risk Prediction page provides actionable recovery strategies based on risk scores.

5. **Download Report**:

  * On the "Download Results" page, click the button to download a CSV with risk scores and recovery strategies.
