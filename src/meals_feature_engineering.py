import numpy as np
import pandas as pd

def add_meal_activity(patient_df, lambda_meal=1/60, max_duration=240):

    patient_df_with_meal = patient_df.copy()
    patient_df_with_meal["meal_activity"] = 0.0

    # Process each patient separately
    for patient_id in patient_df_with_meal["patient_id"].unique():
        patient_mask = patient_df_with_meal["patient_id"] == patient_id
        patient_data = patient_df_with_meal.loc[patient_mask]
        
        meal_times = patient_data.index[patient_data["carbs"] > 0]

        print(f"Patient {patient_id}: {len(meal_times)} meals")

        for t_meal in meal_times:
            carbs = patient_data.loc[t_meal, "carbs"]
            
            # Should be scalar now, but keep safety check
            current_carbs_intake = carbs if isinstance(carbs, (int, float)) else float(carbs)

            elapsed_minutes = (
                (patient_data.index - t_meal) / pd.Timedelta(minutes=1)
            )

            mask = (elapsed_minutes >= 0) & (elapsed_minutes <= max_duration)

            t = elapsed_minutes[mask]
            indices = patient_data.index[mask]

            activity_curve = current_carbs_intake * lambda_meal * t * np.exp(-lambda_meal * t)

            # Update only this patient's rows
            patient_data.loc[indices, "meal_activity"] += activity_curve.values
            
        patient_df_with_meal.loc[patient_mask, "meal_activity"] = patient_data["meal_activity"].values
                    
    return patient_df_with_meal

