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
