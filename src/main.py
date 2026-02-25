import os
import pandas as pd
from bolus_feature_engineering import add_insulin_activity
from data_handling import clean_and_summarise_patients_data, parse_and_combine_patients, parse_and_combine_patients_bolus, parse_and_combine_patients_meals
from parser import add_bolus_raw, add_meal_data, parse_dataframe_to_csv, parse_xml_to_meals_dataframe
from validation import check_bolus_dose
from dotenv import load_dotenv

load_dotenv()

data_folder = os.getenv("DATA_PATH")

processed_patients = []
processed_patients_bolus = []

combined_df = parse_and_combine_patients(data_folder, processed_patients)
cleaned_df = clean_and_summarise_patients_data(combined_df)


bolus_df = parse_and_combine_patients_bolus(data_folder, processed_patients_bolus)
bolus_and_glucose_df = add_bolus_raw(cleaned_df, bolus_df)


check_bolus_dose(bolus_and_glucose_df)
bolus = add_insulin_activity(bolus_and_glucose_df)

meal_df =parse_and_combine_patients_meals(data_folder)

print(meal_df.head())

bolus_and_meal_df = add_meal_data(bolus, meal_df)
# print(bolus_and_glucose_df[bolus_and_glucose_df["bolus_raw"] > 0])
output_folder = os.getenv("OUTPUT_PATH")

df_to_save =bolus_and_meal_df.reset_index()

parse_dataframe_to_csv(output_folder, df_to_save)
# parse_dataframe_to_parquet(output_folder, df_to_save)

# print(cleaned_df.head())
# print(cleaned_df.dtypes)
# print(cleaned_df.index)
