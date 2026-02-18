import os
import pandas as pd

from parser import parse_xml_to_dataframe
from preprocessing import preprocess_patient

data_folder = "../data/raw/train"

processed_patients = []

for file in os.listdir(data_folder):
    if file.endswith(".xml"):
        patient_id = file.split(".")[0]

        file_path = os.path.join(data_folder, file)

        df = parse_xml_to_dataframe(file_path)
        df_processed = preprocess_patient(df, patient_id)

        processed_patients.append(df_processed)

# Combine all patients
combined_df = pd.concat(processed_patients)

print("Combined shape:", combined_df.shape)
print(combined_df.tail(50))
