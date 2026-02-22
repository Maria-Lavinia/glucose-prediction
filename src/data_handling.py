import os
import pandas as pd

from parser import parse_xml_to_bolus_dataframe, parse_xml_to_dataframe
from preprocessing import preprocess_patient
from validation import check_for_duplicate_boluses, check_for_duplicate_timestamps, check_speed, find_and_remove_extreme_changes, get_summary_by_patient


def parse_and_combine_patients(data_folder, processed_patients):
    for file in os.listdir(data_folder):
        if file.endswith(".xml"):
            patient_id = file.split(".")[0]

            file_path = os.path.join(data_folder, file)

            df = parse_xml_to_dataframe(file_path)
            df_processed = preprocess_patient(df, patient_id)

            processed_patients.append(df_processed)

    # Combine all patients
    combined_df = pd.concat(processed_patients)
    return combined_df

def parse_and_combine_patients_bolus(data_folder, processsed_patinets_bolus):
    for file in os.listdir(data_folder):
        if file.endswith(".xml"):
            patient_id = file.split(".")[0]
            file_path = os.path.join(data_folder, file)

            df = parse_xml_to_bolus_dataframe(file_path, patient_id=patient_id)
            processsed_patinets_bolus.append(df)

    combined_bolus_df = pd.concat(processsed_patinets_bolus)
    check_for_duplicate_boluses(combined_bolus_df)
    return combined_bolus_df

def clean_and_summarise_patients_data(combined_df):
    # Generates a summary of individual patients data 
    summary = get_summary_by_patient(combined_df)
    #print(summary)
    # min: 40 max: 400

    # Check minimum and maaximum values for patients
    speed_summary = check_speed(combined_df)
    #print(speed_summary)

    # Check if there are duplicate timestamps
    #print("Check for duplicate timestamps")
    check_for_duplicate_timestamps(combined_df)

    # There is a case for 2 patients 
    cleaned_combined_df = find_and_remove_extreme_changes(combined_df)
    summary_after_cleaning = get_summary_by_patient(cleaned_combined_df)

    # min: 40 max: 400
    #print("Summary")
    #print(summary_after_cleaning)

    #print("Speed summary")
    speed_summary = check_speed(cleaned_combined_df)
    #print(speed_summary)
    # Speed after the clean max 77 minimum -71
    return cleaned_combined_df
