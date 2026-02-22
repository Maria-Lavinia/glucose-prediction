import pandas as pd

def check_minimum_and_maximum_values(df):
    df = df.sort_index()
    time_diff = df.index.to_series().diff()
    glucose_diff = df["glucose"].diff()

    valid_changes = glucose_diff[time_diff == pd.Timedelta(minutes=5)]

    return valid_changes.max(), valid_changes.min()

def check_speed(df):
    results = []

    for pid in df["patient_id"].unique():
        patient_df = df[df["patient_id"] == pid]
        max_inc, max_dec = check_minimum_and_maximum_values(patient_df)

        results.append({
            "patient_id": pid,
            "max_5min_increase": max_inc,
            "max_5min_decrease": max_dec
        })

    speed_summary = pd.DataFrame(results)
    return speed_summary

def get_summary_by_patient(df):
    summary = df.groupby("patient_id")["glucose"].agg([
        "min", "max", "mean", "std", "count"
    ])

    summary["missing_%"] = df.groupby(
        "patient_id")["glucose"].apply(lambda x: x.isna().mean() * 100)

    return summary

## function to check for extreme changes in the patients glucose level
def find_and_remove_extreme_changes(df, threshold=90):
    
    df = df.copy()
    
    for pid, patient_df in df.groupby("patient_id"):
        patient_df = patient_df.sort_index()
        time_diff = patient_df.index.to_series().diff()
        glucose_diff = patient_df["glucose"].diff()

        mask = (
            (time_diff == pd.Timedelta(minutes=5)) &
            (glucose_diff.abs() > threshold)
        )

        idx = patient_df[mask].index
        
        if len(idx) >0:
            print(f"\n====== Patient {pid} ======")

        for t in idx:
            print("\n--- Extreme at:", t, "---")
            print(patient_df.loc[t - pd.Timedelta(minutes=15):
                        t + pd.Timedelta(minutes=15)])
    
        df.loc[mask.index[mask], "glucose"] = None

    return df


def check_for_duplicate_timestamps(df):
    duplicates = df.reset_index().duplicated(subset=["patient_id", "timestamp"]).sum()
    print("Duplicate (patient, timestamp) pairs:", duplicates)
    
def check_for_duplicate_boluses(df):
    df_reset = df.reset_index()

    if "timestamp" in df_reset.columns:
        timestamp_col = "timestamp"
    elif "insuline_timestamp_begin" in df_reset.columns:
        timestamp_col = "insuline_timestamp_begin"
    elif "insulin_timestamp_begin" in df_reset.columns:
        timestamp_col = "insulin_timestamp_begin"
    else:
        raise ValueError("No bolus timestamp column found for duplicate check")

    subset_cols = [timestamp_col]
    if "patient_id" in df_reset.columns:
        subset_cols = ["patient_id", timestamp_col]

    duplicates = df_reset.duplicated(subset=subset_cols).sum()
    print(f"Duplicate ({', '.join(subset_cols)}) pairs:", duplicates)
    
def check_bolus_dose(df):
    summary = df["bolus_raw"].describe()
    print(summary)
    
    number_of_dose = (df["bolus_raw"] > 0).sum()
    print("Number of dose: ", number_of_dose)
    return summary