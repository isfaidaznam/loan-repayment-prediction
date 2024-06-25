# Loan Repayment Failure Prediction using AI/ML Models
## 1.0 Introduction

This project aims to develop an AI/ML model that can predict whether existing borrowers will fail to repay their loans. The model will be trained on a historical dataset of borrowers, their features/characteristics, and whether they've failed to repay their loan.

## 1.1 Dataset Analysis
### 1.1.1 Initial Data Inspection

The provided dataset contains a whooping 38,480 total number of unique rows and 36 columns. The target variable is "repay_fail", which indicates whether the borrower failed to repay the loan.

### 1.1.2 Data Preprocessing

Before begin to train the AI models, data preprocessing needs to be done. In this project, 4 main data processing steps had be done.

- Feature Extraction
- Transformation
- Handling Missing Value
- Normalisation

#### Feature Extraction

The list of generated column are as below, along with their decriptions

1. meet_credit_policy
   - Extracted from 'loan_status' column
   - 1 represents 'meet the credit policy'
   - 0.5 represents ambiguous, unknown or not sure. (This includes any unexpected values)
   - 0 represents 'does not meet the credit policy' 
   - Reason of extraction: loan_status column contains additional data about the meeting the credit policy.
2. purpose_asset_type
   - Extracted from 'purpose' column
   - 1 represents Assets (possible investment)
   - 0.5 represents ambiguous, unknown or not sure. (This includes any unexpected values)
   - 0 represents Expenses
   - Reason of extraction: 'purpose' column can be categorised in multiple ways that seems to be important in determining repayment failure
3. purpose_essential
   - Extracted from 'purpose' column
   - 1 represents essential spending
   - 0.5 represents ambiguous, unknown or not sure. (This includes any unexpected values)
   - 0 represents nonessential spending
   - Reason of extraction: 'purpose' column can be categorised in multiple ways that seems to be important in determining repayment failure
4. exist_months_since_last_delinquency
   - Extracted from 'months_since_last_delinquency' column
   - 1 represents existence of last delinquency
   - 0.5 represents unknown, for what ever reason if failed to identify 0 or 1
   - 0 represents no delinquency
   - Reason of extraction: 'months_since_last_delinquency' column does not have any appropriate value for Null values. Setting the Null value to 0 would be misleading. 0 may indicate zero delinquency, instead of unknown.

#### Transformation
#### Handling Missing Value
#### Normalisation

### 1.1.3 Exploratory Data Analysis (EDA)
EDA was conducted to understand the relationships between the features and the target variable. The following insights were gained:

#### Loan Repayment Failure
The target variable "repay_fail" has 2 distinct values, 1 for failure and 0 for success. 
Unfortunately, the distribution are highly imbalanced. 

![Class Distribution](data/analysis/repay_fail_distribution.jpg){: width="500px"}

About 15.15% of the dataset are under failure class. 
These may pose difficulties during AI training. 
However, several steps are made to handle these types of situations. 
Additionaly, low populations of failure may indicate that the company's 
current policy for applying a loan has an effective approval requirements.

#### Failure Rate by Annual Income

![Failure Rate by Annual Income](data/analysis/failure_rate_by_annual_income.jpg)

The figure shows the relationship failure rate and the annual income. 
The data binning of 20 bins had been performed. 

This generally shows that income and repayment failure have a low inverse correlations. 
As income increases, repayment failure rate decreases. 
The failure rate drops to 0% as annual income reached about 1,000,000. 
Unfortunately, there is a sudden spike on failure rate at around 1,250,000 annual income. 
This may be an outlier where loaners might be overly confident with their ability to repay the loan. 
Loaners earning above this seems to have no repayment failure.

### Failure rate by Debt to Income Ratio

![Failure rate by Debt to Income Ratio](data/analysis/failure_rate_by_debt_to_income_ratio.jpg)

The figure shows the relationship failure rate and the debt to income ratio. 
The data binning of 20 bins had been performed. 

Straight of the bat, an outlier is visible at about 95% debt to income ratio. 
Having a debt of 95 times the income with zero failure repayment is not logically sound. 
Ignoring the outlier, the relationship between Failure rate and the debt to income ratio has a positive correlation. 
The higher the debt to income ration, the higher the failure rate. 
In leman terms, the less the debt, the less probable to fail the repayment. 
This aligns to our common sense.

### Failure rate by Employment Length

![Failure rate by Employment Length](data/analysis/failure_rate_by_employment_length.jpg)

The figure shows the relationship failure rate and the Employment rate. 
For this analysis, the data preprocessing function was repurpose for generating line graph. 
The value employment length = 0 was filtered out because value 0 indicate Null value.

Generally, the line graph indicate a low positive relationship between employment length and the failure rate. 
The failure fluctuate between 13% to 16% of failure rate. 
The increases in failure rate may due to increase in debt, loan, and or commitments.

### Failure rate by Home Ownership

![Failure rate by Home Ownership](data/analysis/failure_rate_by_home_ownership.jpg)

This figure shows the failure rates between different home ownerships. 
This may be counter-intuitive, but the data shows that does not own a home hav ethe highest failure rate despite being 1 less monthly commitments compared to other populations. Owning a home, rent, or mortgage it varies in failure rates between 14% to 16%. Other type of ownership (this includes unkown types) has as high as 23% failure rate, just below None ownership.

### Failure rate by Installments

![](data/analysis/failure_rate_by_installment.jpg)

This figure shows the failure rates across the monthly installments.
Unlike any other, the monthly installment may not have any relationship with the failure to repay. 
The failure rate does fluctuate around 16%, ranging from 7.89% to 25.93%. 
Common sense dictates that the higher the installment, the more probable to fail repayments. 
However, The current requirements enforced by the financial institutions to approve a loan may result to a steady repayment ability to installment ratio. 
However, Installments at the higher end may be unpredictable it starts to fluctuate on a very high variance.

## 1.1 Dataset Analysis