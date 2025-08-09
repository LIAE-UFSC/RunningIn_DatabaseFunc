import pandas as pd

# Load the dataset
df = pd.read_csv('dataset_completo.csv')

# Get unique values from the 'unit' column
unique_units = df['unit'].unique()

print(f"Unique units found: {unique_units}")

def split_dataframe_by_unit(dataframe, column_name):
    """
    Splits a DataFrame into multiple CSV files based on unique values in a specified column.

    Args:
        dataframe (pd.DataFrame): The DataFrame to split.
        column_name (str): The name of the column to split the DataFrame by.
    """
    for unit_value in dataframe[column_name].unique():
        df_subset = dataframe[dataframe[column_name] == unit_value]
        file_name = f'dataset_{unit_value}.csv'
        df_subset.to_csv(file_name, index=False)
        print(f"Created file: {file_name}")

split_dataframe_by_unit(df, 'unit')