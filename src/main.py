import pandas as pd
from bolus_feature_engineering import add_insulin_activity
from data_handling import clean_and_summarise_patients_data, parse_and_combine_patients, parse_and_combine_patients_bolus, parse_and_combine_patients_basis_steps
from parser import add_bolus_raw, add_basis_steps, parse_dataframe_to_csv
from validation import check_bolus_dose


data_folder = "../data/raw/train"

processed_patients = []
processed_patients_bolus = []

combined_df = parse_and_combine_patients(data_folder, processed_patients)
cleaned_df = clean_and_summarise_patients_data(combined_df)


bolus_df = parse_and_combine_patients_bolus(data_folder, processed_patients_bolus)
bolus_and_glucose_df = add_bolus_raw(cleaned_df, bolus_df)


check_bolus_dose(bolus_and_glucose_df)
bolus = add_insulin_activity(bolus_and_glucose_df)

steps_df = parse_and_combine_patients_basis_steps(data_folder)

print("bolus",bolus)
# print(bolus_and_glucose_df[bolus_and_glucose_df["bolus_raw"] > 0])
output_folder = "../data/processed/steps_and_bolus"

bolus_and_steps_df = add_basis_steps(bolus, steps_df)

df_to_save = bolus_and_steps_df.reset_index()

parse_dataframe_to_csv(output_folder, df_to_save)
# parse_dataframe_to_parquet(output_folder, df_to_save)


print("steps_df head:", steps_df.head())
print("cleaned_df head:", cleaned_df.head())



# Merge bolus_raw into steps_and_glucose_df, preserving steps_raw
parse_dataframe_to_csv(output_folder, df_to_save)

# print(cleaned_df.head())
# print(cleaned_df.dtypes)
# print(cleaned_df.index)
