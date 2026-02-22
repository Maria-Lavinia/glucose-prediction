import os
import xml.etree.ElementTree as ET
import pandas as pd

def parse_xml_to_dataframe(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    glucose_section = root.find("glucose_level")

    data = []
    for entry in glucose_section:
        timestamp = entry.attrib["ts"]
        value = float(entry.attrib["value"])
        data.append([timestamp, value])

    df = pd.DataFrame(data, columns=["timestamp", "glucose"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")

    return df

def parse_dataframe_to_csv(output_folder, df):
    output_folder_final = output_folder + "/csv"
    os.makedirs(output_folder_final, exist_ok=True)

    for pid, patient_df in df.groupby("patient_id"):
        filename = f"{output_folder_final}/patient_{pid}.csv"
        patient_df.to_csv(filename, index=False)
        print(f"Saved {filename} as csv")
        
def parse_dataframe_to_parquet(output_folder, df):
    output_folder_final = output_folder + "/parquet"
    os.makedirs(output_folder_final, exist_ok=True)

    for pid, patient_df in df.groupby("patient_id"):
        filename = f"{output_folder_final}/patient_{pid}.parquet"
        patient_df.to_parquet(filename, index=False)
        print(f"Saved {filename} as parquet")