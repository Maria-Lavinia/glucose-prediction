import pandas as pd
from data_handling import clean_and_summarise_patients_data, parse_and_combine_patients, parse_and_combine_patients_bolus
from parser import add_bolus_raw
from validation import check_bolus_dose


data_folder = "./data/raw/test"

processed_patients = []
processed_patients_bolus = []

combined_df = parse_and_combine_patients(data_folder, processed_patients)
cleaned_df = clean_and_summarise_patients_data(combined_df)


bolus_df = parse_and_combine_patients_bolus(data_folder, processed_patients_bolus)
bolus_and_glucose_df = add_bolus_raw(cleaned_df, bolus_df)

check_bolus_dose(bolus_and_glucose_df)
print(bolus_and_glucose_df[bolus_and_glucose_df["bolus_raw"] > 0])
# output_folder = "./data/processed/test"

# df_to_save =cleaned_df.reset_index()

# parse_dataframe_to_csv(output_folder, df_to_save)
# parse_dataframe_to_parquet(output_folder, df_to_save)

# print(cleaned_df.head())
# print(cleaned_df.dtypes)
# print(cleaned_df.index)
