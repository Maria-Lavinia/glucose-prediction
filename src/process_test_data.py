"""
process_test_data.py
====================
Runs the exact same feature engineering pipeline used on training data,
but pointed at data/raw/test/ XMLs.

Output: data/processed/bolus_meal_steps_data/csv/patient_XXX-ws-testing.csv
        (same folder as training files, same columns, ready for model evaluation)

Run from your project root:
    python process_test_data.py
"""

import os
from dotenv import load_dotenv
import pandas as pd

from bolus_feature_engineering import add_insulin_activity
from data_handling import (
    clean_and_summarise_patients_data,
    parse_and_combine_patients,
    parse_and_combine_patients_bolus,
    parse_and_combine_patients_meals,
    parse_and_combine_patients_basis_steps,
)
from meals_feature_engineering import add_meal_activity
from parser import add_bolus_raw, add_meal_data, add_basis_steps
from steps_feature_engineering import add_steps_weighted_avg
from validation import check_bolus_dose

load_dotenv()

# ── Paths ───────────────────────────────────────────────────
TEST_DATA_FOLDER = "data/raw/test"
OUTPUT_FOLDER    = "data/processed/bolus_meal_steps_data"

os.makedirs(os.path.join(OUTPUT_FOLDER, "csv"), exist_ok=True)

print("=" * 60)
print("Processing TEST XML files")
print(f"Input  : {TEST_DATA_FOLDER}")
print(f"Output : {OUTPUT_FOLDER}/csv")
print("=" * 60)

# ── Step 1: Parse glucose + clean ──────────────────────────
print("\nStep 1 - Parsing glucose data...")
processed_patients = []
combined_df = parse_and_combine_patients(TEST_DATA_FOLDER, processed_patients)
cleaned_df  = clean_and_summarise_patients_data(combined_df)
print(f"  Rows after cleaning: {len(cleaned_df)}")

# ── Step 2: Bolus + insulin activity ───────────────────────
print("\nStep 2 - Adding bolus and insulin activity...")
processed_patients_bolus = []
bolus_df             = parse_and_combine_patients_bolus(TEST_DATA_FOLDER, processed_patients_bolus)
bolus_and_glucose_df = add_bolus_raw(cleaned_df, bolus_df)
check_bolus_dose(bolus_and_glucose_df)
bolus = add_insulin_activity(bolus_and_glucose_df)
print(f"  Rows after bolus merge: {len(bolus)}")

# ── Step 3: Meals ───────────────────────────────────────────
print("\nStep 3 - Adding meal data...")
meal_df           = parse_and_combine_patients_meals(TEST_DATA_FOLDER)
bolus_and_meal_df = add_meal_data(bolus, meal_df)
meal              = add_meal_activity(bolus_and_meal_df)
print(f"  Rows after meal merge: {len(meal)}")

# ── Step 4: Steps ───────────────────────────────────────────
print("\nStep 4 - Adding steps data...")
steps_df           = parse_and_combine_patients_basis_steps(TEST_DATA_FOLDER)
bolus_and_steps_df = add_basis_steps(meal, steps_df)
bolus_and_steps_df = add_steps_weighted_avg(bolus_and_steps_df, steps_col='steps')
print(f"  Rows after steps merge: {len(bolus_and_steps_df)}")

# ── Step 5: Sort and save with -ws-testing suffix ──────────
print("\nStep 5 - Saving processed test CSVs...")
df_to_save = bolus_and_steps_df.sort_values(["patient_id", "timestamp"])
df_to_save = df_to_save.reset_index(drop=True)

output_csv_folder = os.path.join(OUTPUT_FOLDER, "csv")
for pid, patient_df in df_to_save.groupby("patient_id"):
    filename = os.path.join(output_csv_folder, f"patient_{pid}-ws-testing.csv")
    patient_df.to_csv(filename, index=False)
    print(f"  Saved {filename}")

# ── Verify ──────────────────────────────────────────────────
saved_files = [f for f in os.listdir(output_csv_folder) if f.endswith('.csv')]
print(f"\nAll files in {output_csv_folder}:")
for f in sorted(saved_files):
    tag = "TRAIN" if "training" in f else "TEST "
    print(f"  [{tag}] {f}")

print("\nDone! You can now run evaluate_test_data.py")