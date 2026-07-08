"""Labeling of the series by time ranges (not run-in × run-in) and grey zone."""

import pandas as pd
import os
from paths import DATASETS_PROC

def label_dataset_by_time(
    input_csv=None,
    df_dados=None,
    time_ranges=[
        (0, 18000, 0),
        (54000, 500000, 1)
    ],
    grey_zone=(18000, 54000),
    exclude_grey=True,
    save_greyzone_csv=False,
    save_csv=True,
    greyzone_csv='greyzone_dataset.csv',
    output_csv=None
):
    """
    Process a dataset: label the data based on time, rename the 'test' column to
    'anomaly' and drop the 'unit' column.

    The function performs the following steps:
    1. Loads the data from a CSV file or an existing DataFrame.
    2. Converts ALL numeric columns (with commas) to float with dots.
    3. Renames the 'test' column to 'anomaly'.
    4. Drops the 'unit' column.
    5. Applies the labels (e.g. 0 for normal, 1 for anomaly) to the 'anomaly' column
       based on the time ranges defined in 'time_ranges'.
    6. Handles a "grey zone" (grey_zone), which can be excluded or saved separately.
    7. Saves the processed DataFrame to a new CSV file.

    Parameters:
    - input_csv (str, optional): Path to the input CSV file.
    - df_dados (DataFrame, optional): DataFrame to process directly.
    - time_ranges (list of tuples): List of tuples (start_time, end_time, label).
    - grey_zone (tuple, optional): Tuple (start_time, end_time) for the grey zone.
    - exclude_grey (bool): If True, removes the grey-zone data.
    - save_greyzone_csv (bool): If True, saves the grey zone separately.
    - save_csv (bool): If True, saves the labeled dataset to a file.
    - greyzone_csv (str): File name for the grey zone.
    - output_csv (str, optional): Output file name.
    """
    df = None
    df_grey = pd.DataFrame()

    # Load the data
    if input_csv:
        df = pd.read_csv(input_csv)
    elif df_dados is not None:
        df = df_dados.copy()
    else:
        raise ValueError("You must provide 'input_csv' or 'df_dados'.")

    # MAIN FIX: convert all numeric columns with commas
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        except (ValueError, AttributeError):
            continue  # Keep non-numeric columns unchanged

    # Standard processing
    df['time'] = df['time'].astype(str).str.replace(',', '.').astype(float)

    if 'test' in df.columns:
        df = df.rename(columns={'test': 'anomaly'})
    else:
        if 'anomaly' not in df.columns:
            df['anomaly'] = None

    if 'unit' in df.columns:
        df = df.drop(columns=['unit'])

    # Generate the output file name if not specified
    if save_csv and output_csv is None:
        if input_csv:
            base_name = os.path.splitext(os.path.basename(input_csv))[0]
            output_csv = f"{base_name.replace('dataset', 'dataset_rotulado')}.csv"
        else:
            output_csv = 'dataset_rotulado_default.csv'

    # Apply the labels
    for start_time, end_time, label in time_ranges:
        mask = (df['time'] >= start_time) & (df['time'] <= end_time)
        df.loc[mask, 'anomaly'] = label

    # Handle the grey zone
    if grey_zone:
        grey_start, grey_end = grey_zone
        grey_mask = (df['time'] >= grey_start) & (df['time'] <= grey_end)

        if save_greyzone_csv:
            df_grey = df[grey_mask].copy()
            df_grey['anomaly'] = 'grey_zone'
            df_grey.to_csv(greyzone_csv, index=False)

        if exclude_grey:
            df = df[~grey_mask]
        else:
            df.loc[grey_mask, 'anomaly'] = 'grey_zone'

    # Save the result
    if save_csv and output_csv:
        df.to_csv(output_csv, index=False)

    return df, df_grey

if __name__ == "__main__":

    print("--- Running Example ---")
    df, grey = label_dataset_by_time(
        input_csv=str(DATASETS_PROC / 'processado_dataset_A5_22_01_NA.csv'),
        output_csv=str(DATASETS_PROC / 'processado_dataset_A5_22_01_NA.csv')
    )
    print("Processed dataset:")
    print(df.head())

    # pasta = "."

    # arquivos_csv = [f for f in os.listdir(pasta) if f.startswith("dataset_A") and f.endswith(".csv")]

    # for arquivo in arquivos_csv:
    #     print(f"--- Running for {arquivo} ---")
    #     df, grey = label_dataset_by_time(
    #     input_csv=arquivo,
    #     time_ranges=[
    #         (0, 18000, 0),
    #         (0, 500000, 1)
    #     ],
    #     grey_zone=None,
    #     )

    #     # save with a prefix
    #     nome_saida = f"processado_{arquivo}"
    #     df.to_csv(nome_saida, index=False)

    #     print(f"Processed dataset saved to: {nome_saida}")
    #     print(df.head())
