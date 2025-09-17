# SmartLoanRecoveryAnalysis

## 🚀 Project Overview

The Smart Loan Recovery System leverages Data Analytics, Machine Learning, and Visualization to identify borrower risk levels and suggest dynamic recovery strategies. By analyzing historical loan data, payment behavior, and financial attributes, this project provides actionable insights for banks and financial institutions to improve loan recovery rates while reducing operational costs.
The system combines exploratory data analysis (EDA), clustering for segmentation, and predictive modeling to create a smart loan recovery framework.
Loan defaults pose a significant challenge for financial institutions by affecting profitability and cash flow. Using historical loan repayment data, borrower profiles, and payment behaviours, many financial companies now use a smart loan recovery system to optimize collection efforts, minimize recovery costs, and maximize loan repayments.

## 🛠 Features

### ✅ Data Exploration & Visualization

1.Distribution of loan amounts vs monthly income

2.Effect of payment history on recovery status

3.Impact of missed payments on recovery chances

4.Relationship between income, loan size, and recovery outcomes

### ✅ Borrower Segmentation (K-Means Clustering)

1.Groups borrowers based on financial profiles (income, loan size, tenure, EMI, defaults, etc.)

2.Segments labeled as:
* High Income, Low Default Risk
* Moderate Income, Medium Risk
* Moderate Income, High Loan Burden
* High Loan, Higher Default Risk

### ✅ Risk Prediction (Random Forest Classifier)

1.Flags borrowers as High Risk or Low Risk

2.Generates risk scores (probability of default)

### ✅ Dynamic Recovery Strategies

1.>75% Risk → Immediate legal notices & aggressive recovery

2.50–75% Risk → Settlement offers & structured repayment plans

3.<50% Risk → Automated reminders & monitoring

## 📂Smart Loan Recovery System: Dataset Overview

The project uses a loan dataset (loan recovery.csv) with features such as:

* Demographics: Age

* Financial Attributes: Monthly Income, Loan Amount, Loan Tenure, Interest Rate, Collateral Value

* Behavioral Indicators: Payment History, Missed Payments, Days Past Due

* Loan Attributes: Outstanding Loan Amount, Monthly EMI

* Recovery Indicators: Recovery Status, Collection Attempts, Legal Action Taken

To build a loan recovery system with Machine Learning, we will use a dataset containing borrower profiles, loan details, and repayment histories. This dataset includes critical attributes such as:

* Demographic Information: Age, employment type, income level, and number of dependents.
* Loan Details: Loan amount, tenure, interest rate, and collateral value.
* Repayment History: Number of missed payments, days past due, and monthly EMI payments.
* Collection Efforts: Collection methods used, number of recovery attempts, and legal actions taken.
* Loan Recovery Status: Whether the loan was fully recovered, partially recovered, or remains outstanding.

