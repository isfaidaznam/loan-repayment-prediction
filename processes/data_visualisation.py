import pandas as pd
from matplotlib import pyplot as plt


def generate_line_graph(df, column_name):
    try:
        print(f"Generating Line Graph for {column_name}...")
        temp_df = df.copy()
        temp_df[f'{column_name}_binned'] = pd.cut(temp_df[column_name], bins=20)

        # group by the binned column_name and calculate the mean failure rate
        temp_df = temp_df.groupby(f'{column_name}_binned', observed=True)['repay_fail'].mean().reset_index()
        temp_df = temp_df.rename(columns={'repay_fail': 'failure_rate'})
        temp_df['failure_rate'] = temp_df['failure_rate'] * 100

        plt.plot(temp_df[f'{column_name}_binned'].apply(lambda x: x.mid), temp_df['failure_rate'], color='red')
        plt.xlabel(column_name.replace("_"," ").title())
        plt.ylabel('Failure Rate (%)')
        plt.title(f'Failure Rate by {column_name.replace("_"," ").title()}')
        plt.grid(True)
        plt.savefig(f'data/analysis/failure_rate_by_{column_name}.jpg', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error generating line graph: {e}")


def generate_bar_graph(df, column_name):
    try:
        print(f"Generating Bar Graph for {column_name}...")
        temp_df = df.copy()

        # group by the categorical column_name and calculate the mean failure rate
        temp_df = temp_df.groupby(column_name)['repay_fail'].mean().reset_index()
        temp_df = temp_df.rename(columns={'repay_fail': 'failure_rate'})
        temp_df['failure_rate'] = temp_df['failure_rate'] * 100

        # Add count column
        temp_count_df = df.groupby(column_name)['repay_fail'].count().reset_index()
        temp_count_df = temp_count_df.rename(columns={'repay_fail': 'count'})
        temp_df = temp_df.merge(temp_count_df, on=column_name)

        plt.figure(figsize=(10,6))
        plt.bar(temp_df[column_name], temp_df['failure_rate'], color='red')

        # Add data labels
        for i, row in temp_df.iterrows():
            plt.annotate(f"{row['failure_rate']:.2f}% [{row['count']}]",
                         (i, row['failure_rate']),
                         textcoords="offset points",
                         xytext=(0,10),
                         ha='center')

        plt.xlabel(column_name.replace("_"," ").title())
        plt.ylabel('Failure Rate (%)')
        plt.title(f'Failure Rate by {column_name.replace("_"," ").title()}')
        plt.xticks(rotation=90)
        plt.savefig(f'data/analysis/failure_rate_by_{column_name}.jpg', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error generating bar graph: {e}")


def generate_data_visual(df):
    try:
        for column_name in ["loan_amount",
                            "interest_rate",
                            "installment",
                            "annual_income",
                            "debt_to_income_ratio",
                            "no_delinquency_2yrs",
                            "inquiries_last_6mths",
                            "months_since_last_delinquency",
                            "no_open_accounts",
                            "no_total_account"]:
            generate_line_graph(df,column_name)

        for column_name in ["term",
                            "employment_length",
                            "home_ownership",
                            "verification_status",
                            "purpose",
                            "loan_status"]:
            generate_bar_graph(df,column_name)
    except:
        pass