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

<img width="1161" height="630" alt="Screenshot 2025-09-17 at 2 11 38 PM" src="https://github.com/user-attachments/assets/6301ec36-93db-4e77-944a-083f025f2027" />

<img width="1161" height="269" alt="Screenshot 2025-09-17 at 2 12 06 PM" src="https://github.com/user-attachments/assets/5326c30c-44d7-4829-9826-d163c42223e6" />

## Analyzing Data Distribution and Relationships

Now, let’s move to analyzing this data in detail. I’ll first have a look at the distribution of the loan amount and its relationship with the monthly income:

<img width="1161" height="522" alt="Screenshot 2025-09-17 at 2 12 27 PM" src="https://github.com/user-attachments/assets/17061755-3e49-4dba-988f-c7aad517d6d0" />

The graph demonstrates a positive relationship between loan amounts and monthly income, indicating that individuals with higher income levels tend to secure larger loans. The density curve at the top shows the distribution of loan amounts, emphasizing that higher loan amounts are more frequent among higher income brackets.

It highlights the proportionality between income and loan size, which shows an income-based approach in loan approvals or customer profiling.

## Analyzing Payment History

Now, let’s have a look at the payment history. I’ll first analyze how payment history affects loan recovery amount:

<img width="1161" height="522" alt="Screenshot 2025-09-17 at 2 12 39 PM" src="https://github.com/user-attachments/assets/c46fa761-52dd-4590-a4d4-ce5246cbee8b" />

Loans with on-time payments are mostly fully recovered. Delayed payments result in a mix of partial and full recoveries, with some written off. Missed payments have a significantly lower recovery rate, with most loans ending up either partially recovered or written off.

Let’s dive into missed payments in detail by analyzing how missed payments affect loan recovery:

<img width="1161" height="522" alt="Screenshot 2025-09-17 at 2 12 48 PM" src="https://github.com/user-attachments/assets/68469625-443a-4a3e-9a85-292701669150" />

Loans with partial recovery typically have up to 4 missed payments. Fully recovered loans tend to have fewer missed payments, mostly between 0 and 2. Written-off loans show a higher range of missed payments, with several exceeding 6. A higher number of missed payments significantly reduces the likelihood of full recovery and increases the chances of loans being written off.

## Analyzing Loan Recovery Based on Monthly Income

Now, let’s dive deep into the relationship between monthly income and loan amount recovery. I’ll first analyze how monthly income and loan amount affect the loan recovery:

<img width="1161" height="522" alt="Screenshot 2025-09-17 at 2 13 02 PM" src="https://github.com/user-attachments/assets/baa111e9-0990-4e93-9c22-e78ea3cfb38d" />

Higher-income individuals are more likely to fully recover their loans, even for larger amounts. Borrowers in lower income brackets face a higher likelihood of loan write-offs or partial recovery. This trend highlights the impact of income on loan recovery, as higher earnings lead to better repayment outcomes and fewer write-offs, even for substantial loans.

Now, using K-Means clustering, I’ll create borrower segments based on monthly income and loan amount:
Let’s visualize the segments to understand them in detail:

<img width="1161" height="496" alt="Screenshot 2025-09-17 at 2 13 21 PM" src="https://github.com/user-attachments/assets/03f38a72-b330-4fbc-8a82-5879b88d6314" />

Segment 1 borrowers take on moderate to high loan amounts, indicating financial stability. The Segment 0 clusters around lower income levels and moderate loan sizes, reflecting potential financial strain. Segment 2 borrowers distribute evenly across the graph, representing a balanced but cautious group. Meanwhile, Segment 3 borrowers concentrate in high-loan areas, especially within specific high-income ranges, highlighting their susceptibility to default despite higher incomes.
