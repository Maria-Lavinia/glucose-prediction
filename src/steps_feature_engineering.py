import numpy as np

def add_steps_weighted_avg(df, steps_col='steps', window=10):
    """
    Adds a weighted average of steps for each timestamp, per patient.
    Args:
        df (pd.DataFrame): DataFrame with 'steps' and 'patient_id' columns.
        steps_col (str): Name of the steps column.
        window (int): Number of previous readings to consider (default 10 for 50 min).
    """
    df_out = df.copy()
    df_out['steps_weighted_avg'] = 0.0
    weights = np.arange(window, 0, -1)  # [10, 9, ..., 1]

    for patient_id in df_out['patient_id'].unique():
        patient_mask = df_out['patient_id'] == patient_id
        patient_data = df_out.loc[patient_mask]
        steps = patient_data[steps_col].values
        weighted_avg = np.zeros(len(steps))
        for i in range(len(steps)):
            start = max(0, i - window + 1)
            steps_window = steps[start:i+1]
            w = weights[-len(steps_window):]
            weighted_avg[i] = np.dot(steps_window, w) / window
        df_out.loc[patient_mask, 'steps_weighted_avg'] = weighted_avg
    return df_out