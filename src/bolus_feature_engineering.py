import numpy as np
import pandas as pd

def add_insulin_activity(patient_df, lambda_=0.02, max_duration=300):

    patient_df_with_insulin = patient_df.copy()
    patient_df_with_insulin["insulin_activity"] = 0.0

    # Process each patient separately
    for patient_id in patient_df_with_insulin["patient_id"].unique():
        patient_mask = patient_df_with_insulin["patient_id"] == patient_id
        patient_data = patient_df_with_insulin.loc[patient_mask]
        
        bolus_times = patient_data.index[patient_data["bolus_raw"] > 0]

        print(f"Patient {patient_id}: {len(bolus_times)} doses")

        for t_bolus in bolus_times:
            dose = patient_data.loc[t_bolus, "bolus_raw"]
            
            # Should be scalar now, but keep safety check
            current_dose = dose.iloc[0] if isinstance(dose, pd.Series) else float(dose)

            elapsed_minutes = (
                (patient_data.index - t_bolus) / pd.Timedelta(minutes=1)
            )

            mask = (elapsed_minutes >= 0) & (elapsed_minutes <= max_duration)

            t = elapsed_minutes[mask]
            indices = patient_data.index[mask]

            activity_curve = current_dose * lambda_ * t * np.exp(-lambda_ * t)

            # Update only this patient's rows
            patient_data.loc[indices, "insulin_activity"] += activity_curve.values
            
        patient_df_with_insulin.loc[patient_mask, "insulin_activity"] = patient_data["insulin_activity"].values
                    
    return patient_df_with_insulin