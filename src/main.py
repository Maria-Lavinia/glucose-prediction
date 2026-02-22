import pandas as pd
from data_handling import clean_and_summarise_patients_data, parse_and_combine_patients
from parser import parse_dataframe_to_csv, parse_dataframe_to_parquet


data_folder = "./data/raw/test"

processed_patients = []

combined_df = parse_and_combine_patients(data_folder, processed_patients)
cleaned_df = clean_and_summarise_patients_data(combined_df)

output_folder = "./data/processed/test"

df_to_save =cleaned_df.reset_index()

parse_dataframe_to_csv(output_folder, df_to_save)
parse_dataframe_to_parquet(output_folder, df_to_save)