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

def parse_xml_to_bolus_dataframe(file_path, patient_id=None):
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
    df_bolus_resampled = df_bolus_resampled.rename_axis("timestamp")

    if patient_id is not None:
        df_bolus_resampled["patient_id"] = patient_id

    return df_bolus_resampled

def parse_xml_to_meals_dataframe(file_path, patient_id=None):
    tree = ET.parse(file_path)
    root = tree.getroot()
    meal_section = root.find("meal")
    
    meal_data = []
    for entry in meal_section:
            meal_timestamp = entry.attrib["ts"]
            meal_type = entry.attrib["type"]
            carbs = int(entry.attrib["carbs"])
            meal_data.append([meal_timestamp, meal_type, carbs])
            
    
    df_meals = pd.DataFrame(meal_data, columns=["meal_timestamp", "meal_type", "carbs"])
    df_meals["meal_timestamp"] = pd.to_datetime(df_meals["meal_timestamp"], dayfirst=True)
    df_meals_resampled = prepare_meal_data(df_meals)

    if patient_id is not None:
        df_meals_resampled["patient_id"] = patient_id

    return df_meals_resampled

def add_bolus_raw(df_glucose, df_bolus):
    df_glucose = df_glucose.copy().reset_index()
    df_bolus = df_bolus.copy().reset_index()

    if "insuline_timestamp_begin" in df_bolus.columns and "timestamp" not in df_bolus.columns:
        df_bolus = df_bolus.rename(columns={"insuline_timestamp_begin": "timestamp"})

    required_glucose_cols = {"timestamp", "patient_id"}
    required_bolus_cols = {"timestamp", "patient_id", "bolus_raw"}

    if not required_glucose_cols.issubset(df_glucose.columns):
        missing = required_glucose_cols.difference(df_glucose.columns)
        raise ValueError(f"df_glucose missing required columns: {sorted(missing)}")

    if not required_bolus_cols.issubset(df_bolus.columns):
        missing = required_bolus_cols.difference(df_bolus.columns)
        raise ValueError(f"df_bolus missing required columns: {sorted(missing)}")

    df_bolus = (
        df_bolus.groupby(["patient_id", "timestamp"], as_index=False)["bolus_raw"]
        .sum()
    )

    merged = df_glucose.merge(
        df_bolus,
        on=["patient_id", "timestamp"],
        how="left"
    )
    merged["bolus_raw"] = merged["bolus_raw"].fillna(0.0)

    return merged.set_index("timestamp").sort_index()

def add_meal_data(df_main, df_meals):
    df_main = df_main.copy().reset_index()
    df_meals = df_meals.copy().reset_index()

    if "meal_timestamp" in df_meals.columns and "timestamp" not in df_meals.columns:
        df_meals = df_meals.rename(columns={"meal_timestamp": "timestamp"})

    required_main_cols = {"timestamp", "patient_id"}
    required_meal_cols = {"timestamp", "patient_id", "carbs", "meal_type"}

    if not required_main_cols.issubset(df_main.columns):
        missing = required_main_cols.difference(df_main.columns)

        raise ValueError(f"df_main missing required columns: {sorted(missing)}")

    if not required_meal_cols.issubset(df_meals.columns):
        missing = required_meal_cols.difference(df_meals.columns)
        raise ValueError(f"df_meals missing required columns: {sorted(missing)}")
    
    dups = df_meals.duplicated(["patient_id", "timestamp"])

    if dups.any():
        raise ValueError("Duplicate patient_id + timestamp rows found.")

    merged = df_main.merge(
        df_meals,
        on=["patient_id", "timestamp"],
        how="left"
    )
    merged["carbs"] = merged["carbs"].fillna(0.0)
    merged["meal_type"] = merged["meal_type"].fillna("none")

    return merged.set_index("timestamp").sort_index()


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

def prepare_meal_data(df_meals):
    df_meals = df_meals.copy()
    
    df_meals["meal_timestamp"] = pd.to_datetime(df_meals["meal_timestamp"], dayfirst=True)
    df_meals = df_meals[df_meals["carbs"] > 0]
    df_meals = df_meals.sort_values("meal_timestamp")
    df_meals = df_meals.set_index("meal_timestamp")
    
    print("carbs missing value:", df_meals["carbs"].isna().sum())
    df_meals_resampled = df_meals.resample("5min").agg({
        "carbs": "sum",
        "meal_type": lambda x: list(x) if len(x) > 0 else "none"
    })
        
    return df_meals_resampled
def prepare_steps_data(df_steps):
    df_steps = df_steps.copy()
    
    df_steps["timestamp"] = pd.to_datetime(df_steps["timestamp"], dayfirst=True)
    df_steps = df_steps[df_steps["steps"] > 0]
    df_steps = df_steps.sort_values("timestamp")
    df_steps = df_steps.set_index("timestamp")
    
    # no missing values
    print(df_steps["steps"].isna().sum())
    df_steps_resampled = df_steps.resample("5min").sum()
    df_steps_resampled.columns = ["steps"]
        
    return df_steps_resampled

def parse_xml_to_basis_steps_dataframe(file_path, patient_id=None):
    tree = ET.parse(file_path)
    root = tree.getroot()
    steps_section = root.find("basis_steps")

    steps_data = []
    for entry in steps_section:
        timestamp = entry.attrib["ts"]
        value = float(entry.attrib["value"])
        steps_data.append([timestamp, value])

    df_steps = pd.DataFrame(steps_data, columns=["timestamp", "steps"])
    df_steps["timestamp"] = pd.to_datetime(df_steps["timestamp"], dayfirst=True)
    df_steps_resampled  = prepare_steps_data(df_steps)

    if patient_id is not None:
        df_steps_resampled["patient_id"] = patient_id

    return df_steps_resampled


def add_basis_steps(df_main, df_steps):
    df_main = df_main.copy().reset_index()
    df_steps = df_steps.copy().reset_index()

    required_main_cols = {"timestamp", "patient_id"}
    required_steps_cols = {"timestamp", "patient_id", "steps"}

    if not required_main_cols.issubset(df_main.columns):
        missing = required_main_cols.difference(df_main.columns)
        raise ValueError(f"df_main missing required columns: {sorted(missing)}")

    if not required_steps_cols.issubset(df_steps.columns):
        missing = required_steps_cols.difference(df_steps.columns)
        raise ValueError(f"df_steps missing required columns: {sorted(missing)}")

    df_steps = (
        df_steps.groupby(["patient_id", "timestamp"], as_index=False)["steps"]
        .sum()
    )

    merged = df_main.merge(
        df_steps,
        on=["patient_id", "timestamp"],
        how="left"
    )
    merged["steps"] = merged["steps"].fillna(0.0)

    return merged.set_index("timestamp").sort_index()

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