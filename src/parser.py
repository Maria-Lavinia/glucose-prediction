import os
import xml.etree.ElementTree as ET
import pandas as pd

def parse_xml_to_dataframe(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    glucose_section = root.find("glucose_level")

    glucose_data = []
    for entry in glucose_section:
        timestamp = entry.attrib["ts"]
        value = float(entry.attrib["value"])
        glucose_data.append([timestamp, value])
    
    df_glucose = pd.DataFrame(glucose_data, columns=["timestamp", "glucose"])
    df_glucose["timestamp"] = pd.to_datetime(df_glucose["timestamp"], dayfirst=True)
    df_glucose = df_glucose.sort_values("timestamp")
    df_glucose = df_glucose.set_index("timestamp")
    print(df_glucose)
    
    return df_glucose

def parse_xml_to_bolus_dataframe(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    bolus_section = root.find("bolus")
    
    bolus_data = []
    for entry in bolus_section:
            insuline_timestamp_begin = entry.attrib["ts_begin"]
            insulin_timestamp_end = entry.attrib["ts_end"]
            insulin_type = entry.attrib["type"]
            dose = float(entry.attrib["dose"])
            carb_input = int(entry.attrib["bwz_carb_input"])
            bolus_data.append([insuline_timestamp_begin, insulin_timestamp_end, insulin_type, dose, carb_input])
            
    
    df_bolus = pd.DataFrame(bolus_data, columns=["insuline_timestamp_begin", "insulin_timestamp_end", "insulin_type", "dose", "carb_input"])
    df_bolus_resampled = prepare_bolus_data(df_bolus)
    return df_bolus_resampled

def add_bolus_raw(df_glucose, df_bolus):

    df_glucose = df_glucose.copy()

    df_glucose["bolus_raw"] = 0.0

    common_index = df_glucose.index.intersection(df_bolus.index)

    df_glucose.loc[common_index, "bolus_raw"] = df_bolus.loc[common_index, "bolus_raw"]

    return df_glucose


def prepare_bolus_data(df_bolus):
    df_bolus = df_bolus.copy()
    
    df_bolus["insuline_timestamp_begin"] = pd.to_datetime(df_bolus["insuline_timestamp_begin"], dayfirst=True)
    df_bolus = df_bolus[df_bolus["dose"] > 0]
    df_bolus = df_bolus.sort_values("insuline_timestamp_begin")
    df_bolus = df_bolus.set_index("insuline_timestamp_begin")
    
    #no missing values
    print(df_bolus["dose"].isna().sum())
    
    df_bolus_resampled = df_bolus["dose"].resample("5min").sum().to_frame(name="bolus_raw")
    
    return df_bolus_resampled

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