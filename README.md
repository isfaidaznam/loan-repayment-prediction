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

1. Transform Categorical Numeric Data into Numerical Data

For this transformation, it converts the categorical data into numerical data where possible. 
The "Categorical Numeric Data" refers to numeric scale that was stored as a categorical values.
For example, the 'term' column is a categorical data that has 2 values, "36 months" and "60 months". 
All letters and symbols will be removed. 
Hence, the "36 months" will transformed into "36".
This method is future-proof where it is ready for any new term, which will somehow be implemented, such as "84 months", will be able to be process without errors.

With some form of variations to handle each column's different needs, this kind of transformation will be applied to the following column:
- term
- employment_length
- revolving_utillization

2. Transform Non-Ordinal Categorical data into Numerical Data

For this transformation, it converts Ordinal data into numerical data using various methods. 
The "Non-Ordinal Categorical data" refers to categorical data that has no clear linear order.
For example, the 'home_ownership' column is a categorical data that has multiple values, such as "own", "rent", "mortgage", "other" and "none". 
Another extreme example is the "purpose" column. the values are vast and seems to be have a multi-dimensional relationship.
Ranging from "car" to "home_improvement", "educational" to "small_business", and "major_purchase" to "debt_consolidation".
These have no clear linear logical order. The "home_ownership" can be ordered in terms of how wealthy the person is.
If we were to sort them from "None" to "Own", the "Mortgage" and "Other" doesn't seems to fit.

Sorting them in terms of risk allows the model to understand the categorical data in a numerical format.
Each category is assigned a numerical value based on its failure rate, which was calculated in the data analysis phase.
These method is much like vectoring the categories into a their own unique vectors, then decreased the dimensionality into 1 dimension that is important to predict the failure repayment, which is the risk dimension. 

With some form of variations to handle each column's different needs, this kind of transformation will be applied to the following column:
- home_ownership 
- verification_status 
- purpose

3. Transform Date Data into Numerical Data

For this transformation, it converts date data into numerical data using the timestamp method.
The purpose of this is to allows the model to understand the relation of a date. 
The model will understood that "12 Jan 2024" is further forward than "6 May 2022" much like the number 500 is further away from 200. 

These transformation were applied on all date type columns such as the following:
- issue_date
- earliest_credit_line
- last_payment_date
- next_payment_date
- last_credit_pull_date

However, all these column were removed at the final stage before training the model. 
These may not be relevant as the question of 'when' did the person pay or 'when' is the issue date and other, may not effect the failure repayment.

Furthermore, including date data in the model may not be relevant for future predictions, as the model would need to account for temporal trends and seasonality, which could add unnecessary complexity. 
By removing these columns, we can focus on the underlying relationships between the borrower's characteristics and loan features, and build a more robust and generalizable model.

4. Transform Zip Code Data into Numerical Data

For this transformation, it converts zip code data into numerical data by extracting the first three digits of the zip code. 
zipcode usually not being used for prediction task.
However, each region consist of different types of populations with variety of portions.
Hence, the data may be relevant in predicting the repayment failure.

Additionally, zipcodes can be sorted as a numerical scale.
Generally, zipcode of 43100 is nearby to zipcode 43100 physically.
Hence, the 'zip_code' column is converted into a numerical value.

5. Transform Address State Data into Numerical Data

For this transformation, it converts address state data into numerical data using a base 26 conversion method. 
Address state such as "AL" will be treated as a base 24 digit.
The symbol A represents the value 1, B represents 2, C represents 3, and so on.
This type of transformation converts these 'base 24' digit into base 10 digit.
Do note that base 10 digits are the ones we are using daily.

The limitation of this method is the lack of relations between the transformed values.
Unlike converting zipcode, the address state does not represent any physical relation to one another.
For example, address code "KJ" does not sit next to "KK".
The 'address_state' column is converted into a numerical value, solely to be able to be process further.
Zipcodes may have more correlations than address state.

#### Handling Missing Value

1. Handling Missing Value in Numeric data

Generally, all numeric values that will not make sense to have a negative number will be assign to -1. This value will represent unkown.
This method applies to columns including, but not limited to:
- loan_amount
- funded_amount
- funded_amount_investors
- interest_rate
- installment
- annual_income
- no_delinquency_2yrs
- no_open_accounts
- public_records

2. Handling Missing Value in Categorical Data

Categorical data that have a defined "other" categories such as "home_ownership" column will be use as an assignment for missing values.
This is due to the unknown nature of the category "other" where the unknown value dimmed fit for the description.

With some form of variations to handle each column's different needs, this kind of handling missing values will be applied to the following column:
- home_ownership : Missing values will be treated as "other"
- verification_status : Missing values will be treated as "Not Verified"
- purpose : Missing values will be treated as "other"

#### Normalisation

After feature extraction, transformation, and handling missing data, all values in all column are now in numeric values.
The variety of range of each columns are vast. 
Some reaches 6,000,000 some are negative numbers. 
Normalizing the values into a (-1,1) range is used on all columns.

### 1.1.3 Exploratory Data Analysis (EDA)
EDA was conducted to understand the relationships between the features and the target variable. The following insights were gained:

#### Loan Repayment Failure
The target variable "repay_fail" has 2 distinct values, 1 for failure and 0 for success. 
Unfortunately, the distribution are highly imbalanced. 

![Class Distribution](data/analysis/repay_fail_distribution.jpg)

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