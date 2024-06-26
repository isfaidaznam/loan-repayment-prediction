import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from processes.preprocess_data import transform_employment_length


def generate_line_graph(df, column_name):
    try:
        print(f"Generating Line Graph for {column_name}...")
        temp_df = df.copy()
        temp_df = temp_df.rename(columns={'repay_fail': 'failure_rate'})
        temp_df['failure_rate'] = temp_df['failure_rate'] * 100

        if column_name == "employment_length":
            temp_df[column_name] = [transform_employment_length(s) for s in temp_df[column_name]]
            # Filter out null values (0)
            temp_df[column_name] = [None if s == -1 else s for s in temp_df[column_name]]

        # Bining
        column_for_x_axis = f'{column_name}_binned'
        temp_df[column_for_x_axis] = pd.cut(temp_df[column_name], bins=20)
        temp_df = temp_df.groupby(column_for_x_axis, observed=True)['failure_rate'].mean().reset_index()

        plt.plot(temp_df[column_for_x_axis].apply(lambda x: x.mid), temp_df['failure_rate'], color='red', marker = "o")
        for i, j in zip(temp_df[column_for_x_axis].apply(lambda x: x.mid), temp_df['failure_rate']):
            plt.annotate(f"{j:.2f}", xy=(i, j))

        plt.xlabel(column_name.replace("_"," ").title())
        plt.ylabel('Failure Rate (%)')
        plt.title(f'Failure Rate by {column_name.replace("_"," ").title()}')
        plt.grid(True)
        if not os.path.exists("data/analysis"):
            os.makedirs("data/analysis")
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


def generate_class_distribution(df, column_name):
    try:
        print(f"Generating Pie Chart for {column_name} distribution...")
        column_counts = df[column_name].value_counts()

        plt.figure(figsize=(10, 8))
        plt.pie(column_counts.values, labels=column_counts.index, autopct='%1.1f%%')

        plt.title(f'Distribution of {column_name.replace("_"," ").title()}')
        plt.savefig(f'data/analysis/{column_name}_distribution.jpg', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error generating pie chart for {column_name}: {e}")


def generate_data_visual(df):
    try:
        for column_name in ["repay_fail"]:
            generate_class_distribution(df,column_name)
        for column_name in ["loan_amount",
                            "interest_rate",
                            "installment",
                            "annual_income",
                            "debt_to_income_ratio",
                            "no_delinquency_2yrs",
                            "inquiries_last_6mths",
                            "months_since_last_delinquency",
                            "employment_length",
                            "no_open_accounts",
                            "no_total_account"]:
            generate_line_graph(df,column_name)

        for column_name in ["term",
                            "home_ownership",
                            "verification_status",
                            "purpose",
                            "loan_status"]:
            generate_bar_graph(df,column_name)
    except:
        pass


def export_accuracy_curve(history):
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    plt.plot(accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Curves')

    plt.savefig(f'trained_model/ann_model/Accuracy_Curves.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    print("Training and validation curves saved on 'trained_model/ann_model/Accuracy_Curves.jpg")

def export_loss_curve(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Curves')

    plt.savefig(f'trained_model/ann_model/Loss_Curves.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    print("Training and validation curves saved on 'trained_model/ann_model/Loss_Curves.jpg")
    pass